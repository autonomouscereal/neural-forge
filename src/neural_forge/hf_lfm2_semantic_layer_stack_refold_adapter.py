"""Executable, transactional semantic refold of the complete LFM2 layer stack.

The adapter is deliberately narrower than the model-family-independent planner:
it executes an authenticated LFM2 child transaction whose dynamic layer topology
inputs were already normalized by :mod:`hf_lfm2_semantic_postfold`.  Parent
modules are never constructed or called.  Parent activations come only from
the complete sealed cache and parent weights come only from local safetensors.

Every layer is handled in causal order.  Its operator is compiled from the
actual child residual produced by every previously accepted layer, then the
same macrocell is replayed again before its MLP is compiled.  A third replay
propagates the accepted layer to the next layer.  The last MLP's W2 is shared
with the terminal/lexical writer, so it is deferred and returned as four
authenticated terminal-action evidence banks.  Any failure restores the
entire model to its entry bytes before raising.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.analytic_execution_profile import (
    AnalyticExecutionProfile,
    AnalyticExecutionProfileError,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import run_fold_only_callback
from neural_forge.dynamic_weight_complete_basis import (
    ResidualFacingOperator,
    RoleBalancedOperatorGeometry,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    _tensor_sha256 as _cache_tensor_sha256,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_cpcf_macrocell_replay import (
    Lfm2CpcfMacrocellReplay,
    replay_lfm2_cpcf_macrocell,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_attention_group_qdeim import (
    Lfm2AttentionGroupQdeimSourceLocalOutputFitCase,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256,
)
from neural_forge.hf_lfm2_global_linearization import (
    Lfm2AttentionParameters,
    Lfm2MlpParameters,
    Lfm2ShortConvParameters,
    Lfm2StackPrimalBatch,
    Lfm2TerminalNormParameters,
    capture_lfm2_macrocell_linearization,
    capture_lfm2_terminal_norm_primal,
)
from neural_forge.hf_lfm2_global_residual_closure import (
    Lfm2GlobalResidualClosurePolicy,
)
from neural_forge.lfm2_body_global_coordinate_completion import (
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA,
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES,
    Lfm2BodyGlobalCoordinateCompletion,
    compile_lfm2_body_global_coordinate_completion,
    lfm2_body_global_coordinate_completion_evidence_sha256,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import _rope_from_target
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (
    LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS,
    Lfm2SemanticAttentionReplayEndpoint,
)
from neural_forge.hf_lfm2_semantic_attention_group_qdeim_adapter import (
    Lfm2SemanticAttentionGroupQdeimEvidenceDeficit,
)
from neural_forge.hf_lfm2_semantic_attention_group_qdeim_installer import (
    Lfm2SemanticAttentionGroupQdeimInstallationResult,
    compile_install_lfm2_semantic_attention_group_qdeim,
    seal_lfm2_semantic_attention_group_qdeim_norm_inventory,
)
from neural_forge.hf_lfm2_semantic_attention_source_local_output import (
    LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA,
    Lfm2SemanticAttentionSourceLocalOutputInstallation,
    compile_install_lfm2_semantic_attention_source_local_output,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (
    Lfm2SemanticBodyRecipe,
    is_supported_lfm2_semantic_component_equivalence_inventory,
    resolve_lfm2_semantic_body_recipe,
)
from neural_forge.hf_lfm2_semantic_global_residual_integration import (
    Lfm2SemanticGlobalResidualEvidencePackage,
    Lfm2SemanticGlobalResidualTrajectoryEvidence,
    compile_install_lfm2_semantic_global_residual_closure,
)
from neural_forge.hf_lfm2_producer_aware_semantic_integration import (
    LFM2_PRODUCER_AWARE_SOURCE_LOCAL_BOUNDARY_MODE,
    LFM2_PRODUCER_AWARE_SOURCE_LOCAL_SEMANTIC_INTEGRATION_SCHEMA,
    Lfm2ProducerAwareCorrectionScalePolicy,
    integrate_lfm2_producer_aware_global_closure,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA,
    LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA,
    Lfm2SemanticAttentionTransportEvidence,
    Lfm2SemanticCompilerLayerLineage,
    Lfm2SemanticLayerStackInstallation,
    _ZERO_OPERATIONS,
    _additive_causal_mask,
    _attention_tensor_names,
    _authenticate_lfm2_semantic_attention_operator_receipt,
    _changed_state_keys,
    _fold_parent_qk_weight_priors,
    _fit_normalized_operator_input_decoder,
    _interface_tensor_sha256,
    _lfm2_dense_producer_chain_coverage,
    _local_safetensor_shards,
    _parameter_count,
    _semantic_alignments_from_session,
    _state_tensor_hashes,
    _target_rope,
    _tensor_sha256,
    _validate_attention_installation_receipt,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    Lfm2SemanticSpanCpcfAlignment,
    audit_lfm2_semantic_span_cpcf_refinement,
)
from neural_forge.hf_semantic_attention_lattice import (
    Lfm2SemanticAttentionCacheCapability,
    SemanticAttentionPreRopeHistory,
    assemble_lfm2_explicit_pre_rope_history,
    fold_semantic_attention_parent_pre_rope_priors,
    inspect_lfm2_semantic_attention_cache_capability,
    transport_lfm2_phase_probability_preserving_attention,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
    TiedCausalInterfaceFold,
)
from neural_forge.hf_tied_residual_congruence import (
    TIED_RESIDUAL_CONGRUENCE_SCHEMA,
    TiedResidualAnchorTransportWitness,
    authenticate_tied_residual_congruence_interface,
)
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    fingerprint_hf_tokenizer_semantics,
)


LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA = (
    "neural-forge-lfm2-semantic-layer-stack-refold-v1"
)
_TIED_CAUSAL_INTERFACE_FOLD_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"
# Retained only as the historical LFM2-230M fixture topology used by older tests.
# Runtime topology admission and all transaction counts are derived dynamically.
_EXPECTED_ATTENTION_INDICES = (2, 4, 6, 8, 10, 12)
_EXPECTED_SHORT_CONV_INDICES = (0, 1, 3, 5, 7, 9, 11, 13)
_PARTITIONS = ("fit", "fit_check", "selection", "validation")
_SHORT_CONV_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-installation-v1"
)
_SHORT_CONV_STRUCTURAL_NOOP_STATUS = "structurally-identical-or-invalid-incumbent-noop"
_MLP_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-mlp-installation-v1"
_MLP_STRUCTURAL_NOOP_STATUS = "structurally-identical-or-invalid-incumbent-noop"
_ATTENTION_CANDIDATE_ELIGIBILITY_BASIS = (
    "finite_dense_bfloat16_exact_shape_and_nonincumbent_effect"
)
_COUPLED_ATTENTION_QDEIM_LANE = "intact_gqa_group_qdeim_coupled_qkvo"
_SOURCE_LOCAL_ATTENTION_OUTPUT_LANE = "bounded_source_local_attention_o"
_SOURCE_LOCAL_SHORT_CONV_OUTPUT_LANE = "bounded_source_local_short_conv_writer"
_SOURCE_LOCAL_GLOBAL_RESIDUAL_LANE = "source_local_action_boundaries"
_SOURCE_LOCAL_ATTENTION_COMPOSITE_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-source-local-output-composite-v1"
)
_SOURCE_LOCAL_SHORT_CONV_COMPOSITE_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-source-local-output-composite-v1"
)
_SOURCE_LOCAL_SHORT_CONV_FIT_CASES_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-source-local-fit-cases-v1"
)
_GLOBAL_RESIDUAL_PRODUCER_AWARE_V2A = "producer_aware_v2a"
_GLOBAL_RESIDUAL_PRODUCER_AWARE_SOURCE_LOCAL_V2B = "producer_aware_source_local_v2b"
_GLOBAL_RESIDUAL_OUTPUT_WRITER_V1_DIAGNOSTIC = "output_writer_v1_diagnostic"
_GLOBAL_RESIDUAL_CLOSURE_VERSIONS = frozenset(
    {
        _GLOBAL_RESIDUAL_PRODUCER_AWARE_V2A,
        _GLOBAL_RESIDUAL_PRODUCER_AWARE_SOURCE_LOCAL_V2B,
        _GLOBAL_RESIDUAL_OUTPUT_WRITER_V1_DIAGNOSTIC,
    }
)
_INTERFACE_RUNTIME_NATIVE_TIED = "native_tied"
_INTERFACE_RUNTIME_INDEPENDENT_OUTPUT = "independent_output"
_INTERFACE_RUNTIME_MODES = frozenset(
    {
        _INTERFACE_RUNTIME_NATIVE_TIED,
        _INTERFACE_RUNTIME_INDEPENDENT_OUTPUT,
    }
)
_GLOBAL_RESIDUAL_MLP_LANES = frozenset(
    {
        "nonlinear_qdeim_structural_priority",
        "nonlinear_qdeim_fit_corrected_priority",
    }
)
_MLP_FUNCTIONAL_PROVENANCE_SCHEMA = (
    "neural-forge-lfm2-direct-parent-mlp-functional-provenance-v3"
)
_MAX_FINITE_FLOAT = float.fromhex("0x1.fffffffffffffp+1023")


class Lfm2SemanticLayerStackRefoldError(RuntimeError):
    """The complete causal transaction failed and was rolled back."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic layer-stack refolding requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _sealed(body: Mapping[str, Any]) -> Mapping[str, Any]:
    value = dict(body)
    value["result_sha256"] = canonical_sha256(value)
    return MappingProxyType(value)


def _verify_sealed(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    result = dict(value)
    supplied = result.pop("result_sha256", None)
    if supplied != canonical_sha256(result):
        raise Lfm2SemanticLayerStackRefoldError(f"{label} receipt seal is invalid.")
    return {**result, "result_sha256": supplied}


def _validated_qdeim_compute_execution(
    value: Any,
    *,
    requested_device: str,
    schema: str,
    label: str,
) -> Mapping[str, Any]:
    """Authenticate one CPU/CUDA FP64 Q-DEIM execution without accepting fallback."""

    if not isinstance(value, Mapping):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} omitted its Q-DEIM compute execution receipt."
        )
    if requested_device == "cpu":
        backend = "cpu"
        device_index = None
    else:
        prefix, separator, index_text = requested_device.partition(":")
        if prefix != "cuda" or separator != ":" or not index_text.isdecimal():
            raise Lfm2SemanticLayerStackRefoldError(
                "Fold compute device must be 'cpu' or an explicitly indexed "
                "CUDA device."
            )
        backend = "cuda"
        device_index = int(index_text)
    receipt = dict(value)
    supplied = receipt.pop("receipt_sha256", None)
    _digest(supplied, f"{label} receipt SHA-256")
    if supplied != canonical_sha256(receipt):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} Q-DEIM compute execution seal is invalid."
        )
    expected_cuda = backend == "cuda"
    if (
        receipt.get("schema") != schema
        or receipt.get("requested_device") != requested_device
        or receipt.get("resolved_device") != requested_device
        or receipt.get("backend") != backend
        or receipt.get("compute_dtype") != "float64"
        or receipt.get("authenticated_source_archive_device") != "cpu"
        or receipt.get("active_layer_staged_to_compute_device") is not expected_cuda
        or receipt.get("materialized_candidate_device") != "cpu"
        or receipt.get("system_memory_is_immutable_archive") is not True
        or receipt.get("gpu_is_ephemeral_dense_algebra_workspace") is not expected_cuda
        or receipt.get("silent_device_fallback_performed") is not False
        or receipt.get("training_steps") != 0
        or receipt.get("backward_calls") != 0
        or receipt.get("optimizer_steps") != 0
        or (
            expected_cuda
            and (
                receipt.get("device_index") != device_index
                or not isinstance(receipt.get("device_name"), str)
                or not receipt.get("device_name")
                or not isinstance(receipt.get("compute_capability"), list)
                or len(receipt["compute_capability"]) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, int) or part < 0
                    for part in receipt["compute_capability"]
                )
            )
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} did not prove requested-device FP64 Q-DEIM execution."
        )
    return MappingProxyType({**receipt, "receipt_sha256": supplied})


def _require_body_recipe_binding(
    receipt: Mapping[str, Any],
    recipe: Lfm2SemanticBodyRecipe,
    *,
    component: str,
    label: str,
) -> None:
    expected_lane = recipe.component_lanes.get(component)
    if (
        expected_lane is None
        or receipt.get("body_recipe") != recipe.as_dict()
        or receipt.get("body_recipe_sha256") != recipe.recipe_sha256
        or receipt.get("body_recipe_id") != recipe.recipe_id
        or receipt.get("recipe_component_lane") != expected_lane
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} did not bind the requested semantic body recipe and lane."
        )


def _require_attention_body_recipe_binding(
    receipt: Mapping[str, Any],
    recipe: Lfm2SemanticBodyRecipe,
    *,
    label: str,
) -> None:
    """Bind the nested Q/K and V/O decisions to one complete body recipe."""

    _require_body_recipe_binding(
        receipt,
        recipe,
        component="qk",
        label=label,
    )
    source_local_output = (
        recipe.component_lanes.get("attention_output_residual")
        == _SOURCE_LOCAL_ATTENTION_OUTPUT_LANE
    )
    if (
        recipe.component_lanes["qk"] == _COUPLED_ATTENTION_QDEIM_LANE
        and recipe.component_lanes["value_output"] == _COUPLED_ATTENTION_QDEIM_LANE
    ):
        source_local_contract_valid = bool(
            source_local_output
            and receipt.get("source_local_attention_output") is True
            and receipt.get("source_local_attention_output_lane")
            == _SOURCE_LOCAL_ATTENTION_OUTPUT_LANE
            and receipt.get("structural_coupled_qkvo_candidate_installed") is True
            and receipt.get("source_local_o_only_second_stage") is True
            and receipt.get("split_qk_or_vo_fallback_used") is False
        )
        legacy_contract_valid = bool(
            not source_local_output
            and receipt.get("coupled_qkvo_single_candidate") is True
            and receipt.get("split_qk_or_vo_fallback_used") is False
        )
        if receipt.get(
            "recipe_value_output_lane"
        ) != _COUPLED_ATTENTION_QDEIM_LANE or not (
            source_local_contract_valid or legacy_contract_valid
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"{label} did not bind one coupled Q/K/V/O recipe lane."
            )
        return
    analytic = receipt.get("analytic_refold")
    gate_value = (
        analytic.get("value_output_candidate_gate")
        if isinstance(analytic, Mapping)
        else None
    )
    if not isinstance(analytic, Mapping) or not isinstance(gate_value, Mapping):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} omitted its recipe-bound Q/K or V/O decision."
        )
    if (
        analytic.get("body_recipe") != recipe.as_dict()
        or analytic.get("body_recipe_sha256") != recipe.recipe_sha256
        or analytic.get("body_recipe_id") != recipe.recipe_id
        or analytic.get("recipe_qk_lane") != recipe.component_lanes["qk"]
        or analytic.get("recipe_value_output_lane")
        != recipe.component_lanes["value_output"]
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} analytic receipt changed its Q/K or V/O recipe lane."
        )
    gate = dict(gate_value)
    supplied_gate_sha256 = gate.pop("gate_sha256", None)
    if supplied_gate_sha256 != canonical_sha256(gate):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} V/O decision seal is invalid."
        )
    _require_body_recipe_binding(
        gate,
        recipe,
        component="value_output",
        label=f"{label} V/O decision",
    )
    reports = gate.get("candidate_reports")
    if not isinstance(reports, Mapping):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} V/O decision omitted its candidate frontier."
        )
    analytic_names = ("fixed_v_output_only", "joint_value_output")
    eligible_names = tuple(
        name
        for name in analytic_names
        if isinstance(reports.get(name), Mapping)
        and reports[name].get("eligible") is True
    )
    value_output_lane = recipe.component_lanes["value_output"]
    if value_output_lane == "joint_then_fixed_structural_priority":
        expected = (
            "joint_value_output"
            if "joint_value_output" in eligible_names
            else (
                "fixed_v_output_only"
                if "fixed_v_output_only" in eligible_names
                else "incumbent"
            )
        )
    elif value_output_lane == "fixed_writer_then_joint_residual_priority":
        expected = (
            "fixed_v_output_only"
            if "fixed_v_output_only" in eligible_names
            else (
                "joint_value_output"
                if "joint_value_output" in eligible_names
                else "incumbent"
            )
        )
    else:
        expected = str(gate.get("published_candidate"))
    published = gate.get("published_candidate")
    if (
        set(reports) != {"incumbent", *analytic_names}
        or gate.get("selected_candidate") != published
        or gate.get("structurally_eligible_candidate_count") != len(eligible_names)
        or gate.get("fit_metrics_used_for_candidate_ordering")
        is not recipe.fit_metrics_order_candidates
        or not is_supported_lfm2_semantic_component_equivalence_inventory(
            "value_output",
            value_output_lane,
            gate.get("component_selection_policy_equivalent_recipe_ids"),
            gate.get("component_selection_policy_equivalent_to_other_recipe"),
            materialized_recipe_id=recipe.recipe_id,
        )
        or published != expected
        or (published == "incumbent") is not (not eligible_names)
        or (
            published != "incumbent"
            and (
                not isinstance(reports.get(published), Mapping)
                or reports[published].get("eligible") is not True
            )
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label} V/O candidate is not bound to the declared recipe policy."
        )


def _validate_producer_coverage_decision(
    *,
    layer_index: int,
    producer_kind: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    installation: Mapping[str, Any],
    defer_down_weight: bool = False,
) -> Mapping[str, Any]:
    """Require cold starts to plant a chain and authenticate refinement no-ops."""

    before_receipt = _verify_sealed(before, "producer coverage before")
    after_receipt = _verify_sealed(after, "producer coverage after")
    expected_deferred = ["feed_forward.w2.weight"] if defer_down_weight else []
    common_invalid = bool(
        before_receipt.get("layer_index") != layer_index
        or after_receipt.get("layer_index") != layer_index
        or before_receipt.get("producer_kind") != producer_kind
        or after_receipt.get("producer_kind") != producer_kind
        or before_receipt.get("deferred_roles") != expected_deferred
        or after_receipt.get("deferred_roles") != expected_deferred
        or before_receipt.get("dense_existing_topology_only") is not True
        or after_receipt.get("dense_existing_topology_only") is not True
        or before_receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or after_receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or after_receipt.get("all_required_roles_structurally_nonzero") is not True
        or after_receipt.get("blank_or_partially_blank_required_role_count") != 0
    )
    installed = installation.get("installed") is True
    model_state_changed = bool(
        installation.get("model_state_before_sha256")
        != installation.get("model_state_after_sha256")
    )
    role_state_changed = bool(
        before_receipt.get("required_role_state_sha256")
        != after_receipt.get("required_role_state_sha256")
    )
    cold_start = bool(
        before_receipt.get("all_required_roles_structurally_nonzero") is not True
    )
    retained_nonzero_seed = bool(
        not installed
        and not cold_start
        and not model_state_changed
        and not role_state_changed
    )
    installed_complete_chain = bool(installed and model_state_changed)
    if common_invalid or not (installed_complete_chain or retained_nonzero_seed):
        if cold_start and not installed_complete_chain:
            reason = "blank-or-partial-producer-chain-was-not-densely-planted"
        elif not installed and not retained_nonzero_seed:
            reason = "producer-noop-did-not-retain-an-authenticated-nonzero-seed"
        else:
            reason = "producer-installation-did-not-prove-complete-coverage"
        raise Lfm2SemanticLayerStackRefoldError(
            f"Layer {layer_index} {producer_kind} coverage rejected: {reason}."
        )
    body = {
        "schema": "neural-forge-lfm2-producer-coverage-decision-v1",
        "layer_index": layer_index,
        "producer_kind": producer_kind,
        "deferred_roles": expected_deferred,
        "coverage_before_sha256": before_receipt["result_sha256"],
        "coverage_after_sha256": after_receipt["result_sha256"],
        "coverage_before": before_receipt,
        "coverage_after": after_receipt,
        "cold_start_or_partial_seed_at_entry": cold_start,
        "installed_complete_structurally_nonzero_chain": installed_complete_chain,
        "authenticated_nonzero_folded_seed_retained_by_noop": retained_nonzero_seed,
        "all_required_roles_structurally_nonzero_after_decision": True,
        "blank_or_partially_blank_required_role_count_after_decision": 0,
        "numeric_parent_child_weight_match_objective": False,
        "training_steps": 0,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return _sealed(body)


def _file_size_and_sha256(path: Path) -> tuple[int, str]:
    """Hash one stable ordinary file and bind the digest to that same handle."""

    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticLayerStackRefoldError(
            f"Authenticated direct-parent file is missing or unsafe: {path}."
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while block := handle.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(handle.fileno())
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Could not authenticate direct-parent file {path}."
        ) from exc
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    before_identity = tuple(int(getattr(before, name)) for name in identity_fields)
    after_identity = tuple(int(getattr(after, name)) for name in identity_fields)
    current_identity = tuple(int(getattr(current, name)) for name in identity_fields)
    if before_identity != after_identity or after_identity != current_identity:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Direct-parent file changed during authentication: {path}."
        )
    return int(after.st_size), digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return _file_size_and_sha256(path)[1]


def _direct_parent_runtime_dtype(
    normalized_config: Mapping[str, Any],
    checkpoint_floating_dtypes: set[Any],
) -> Any:
    """Resolve the exact runtime dtype needed to reconstruct parent buffers."""

    torch = _torch()
    if not checkpoint_floating_dtypes or any(
        not isinstance(value, torch.dtype)
        or not value.is_floating_point
        or value.is_complex
        for value in checkpoint_floating_dtypes
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent checkpoint has no single valid real floating runtime dtype."
        )
    checkpoint_names = {
        str(value).removeprefix("torch.") for value in checkpoint_floating_dtypes
    }
    configured = normalized_config.get("dtype")
    if configured is None:
        if len(checkpoint_names) != 1:
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent checkpoint has multiple floating dtypes without "
                "an authenticated runtime dtype."
            )
        dtype_name = next(iter(checkpoint_names))
    else:
        if not isinstance(configured, str):
            raise Lfm2SemanticLayerStackRefoldError(
                "Normalized direct parent config dtype must be text."
            )
        dtype_name = configured.removeprefix("torch.")
        if checkpoint_names != {dtype_name}:
            raise Lfm2SemanticLayerStackRefoldError(
                "Normalized parent config dtype and direct checkpoint floating "
                "dtypes disagree."
            )
    dtype = getattr(torch, dtype_name, None)
    if (
        not isinstance(dtype, torch.dtype)
        or not dtype.is_floating_point
        or dtype.is_complex
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Unsupported direct parent runtime dtype {dtype_name!r}."
        )
    return dtype


def _direct_parent_runtime_buffers(
    config: Any,
    count: int,
    *,
    dtype: Any,
) -> dict[str, Any]:
    """Reconstruct the only runtime-only LFM2 state admitted by the cache."""

    torch = _torch()
    if count == 0:
        return {}
    rope = getattr(config, "rope_parameters", None)
    if (
        count != 2
        or not isinstance(dtype, torch.dtype)
        or not dtype.is_floating_point
        or dtype.is_complex
        or not isinstance(rope, Mapping)
        or rope.get("rope_type") != "default"
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent identity reconstruction supports no runtime buffers "
            "or the authenticated default LFM2 RoPE buffer pair."
        )
    base = rope.get("rope_theta")
    if (
        isinstance(base, bool)
        or not isinstance(base, (int, float))
        or not math.isfinite(float(base))
        or float(base) <= 0.0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent config has no valid RoPE base."
        )
    hidden = int(getattr(config, "hidden_size", 0))
    heads = int(getattr(config, "num_attention_heads", 0))
    head_dim = int(getattr(config, "head_dim", 0) or (hidden // heads if heads else 0))
    if head_dim <= 0 or head_dim % 2 != 0:
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent config cannot reconstruct LFM2 RoPE buffers."
        )
    inv_freq = 1.0 / (
        float(base)
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float32) / head_dim
        )
    )
    inv_freq = inv_freq.to(dtype=dtype)
    return {
        "model.rotary_emb.inv_freq": inv_freq,
        "model.rotary_emb.original_inv_freq": inv_freq.clone(),
    }


def _validated_cache_parent_record(
    value: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate the cache's sealed parent record before local reconstruction."""

    if not isinstance(value, Mapping):
        raise Lfm2SemanticLayerStackRefoldError(
            "The sealed cache omitted its complete parent identity record."
        )
    record = dict(value)
    body = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": record.get("model_class"),
        "config_sha256": record.get("config_sha256"),
        "state_dict_sha256": record.get("state_dict_sha256"),
        "state_tensor_count": record.get("state_tensor_count"),
        "runtime_buffer_sha256": record.get("runtime_buffer_sha256"),
        "runtime_buffer_count": record.get("runtime_buffer_count"),
    }
    digest_fields = (
        "config_sha256",
        "state_dict_sha256",
        "runtime_buffer_sha256",
    )
    if (
        not isinstance(body["model_class"], str)
        or not body["model_class"]
        or any(
            not isinstance(body[name], str)
            or len(body[name]) != 64
            or any(character not in "0123456789abcdef" for character in body[name])
            for name in digest_fields
        )
        or any(
            isinstance(body[name], bool)
            or not isinstance(body[name], int)
            or int(body[name]) < 0
            for name in ("state_tensor_count", "runtime_buffer_count")
        )
        or record.get("identity_sha256") != expected_identity_sha256
        or canonical_sha256(body) != expected_identity_sha256
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "The sealed cache parent record does not reproduce its declared identity."
        )
    return MappingProxyType(record)


def _parameter_topology(model: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            str(name),
            id(value),
            int(value.data_ptr()),
            tuple(int(size) for size in value.shape),
            str(value.dtype),
            str(value.device),
            bool(value.requires_grad),
        )
        for name, value in model.named_parameters(remove_duplicate=False)
    )


@dataclass(slots=True)
class _WholeModelSnapshot:
    state: Mapping[str, Any] = field(repr=False)
    gradients: Mapping[str, Any | None] = field(repr=False)
    fingerprint: str
    parameter_count: int
    topology: tuple[tuple[Any, ...], ...]
    modes: tuple[tuple[str, bool], ...]

    @classmethod
    def capture(cls, model: Any) -> _WholeModelSnapshot:
        state = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in model.state_dict().items()
        }
        gradients = {
            name: (
                None
                if value.grad is None
                else value.grad.detach().to(device="cpu").contiguous().clone()
            )
            for name, value in model.named_parameters()
        }
        return cls(
            state=MappingProxyType(state),
            gradients=MappingProxyType(gradients),
            fingerprint=fingerprint_lfm2_parent(model).identity_sha256,
            parameter_count=_parameter_count(model),
            topology=_parameter_topology(model),
            modes=tuple(
                (str(name), bool(module.training))
                for name, module in model.named_modules()
            ),
        )

    def restore(self, model: Any) -> None:
        torch = _torch()
        current = model.state_dict(keep_vars=True)
        if set(current) != set(self.state):
            raise Lfm2SemanticLayerStackRefoldError(
                "Whole-stage rollback cannot restore a changed state topology."
            )
        with torch.no_grad():
            for name, value in current.items():
                value.copy_(self.state[name].to(device=value.device, dtype=value.dtype))
            parameters = dict(model.named_parameters())
            if set(parameters) != set(self.gradients):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Whole-stage rollback cannot restore changed Parameters."
                )
            for name, parameter in parameters.items():
                gradient = self.gradients[name]
                if gradient is None:
                    parameter.grad = None
                elif parameter.grad is None:
                    parameter.grad = gradient.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ).clone()
                else:
                    parameter.grad.copy_(
                        gradient.to(
                            device=parameter.grad.device,
                            dtype=parameter.grad.dtype,
                        )
                    )
        modules = dict(model.named_modules())
        if set(modules) != {name for name, _mode in self.modes}:
            raise Lfm2SemanticLayerStackRefoldError(
                "Whole-stage rollback cannot restore changed module topology."
            )
        for name, mode in self.modes:
            modules[name].training = mode
        if (
            fingerprint_lfm2_parent(model).identity_sha256 != self.fingerprint
            or _parameter_count(model) != self.parameter_count
            or _parameter_topology(model) != self.topology
            or tuple(
                (str(name), bool(module.training))
                for name, module in model.named_modules()
            )
            != self.modes
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Whole-stage rollback did not restore exact entry state."
            )


def _gradient_state_matches_snapshot(
    model: Any,
    snapshot: _WholeModelSnapshot,
) -> bool:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(snapshot.gradients):
        return False
    for name, parameter in parameters.items():
        expected = snapshot.gradients[name]
        observed = parameter.grad
        if (expected is None) != (observed is None):
            return False
        if expected is not None and _tensor_sha256(expected) != _tensor_sha256(
            observed
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class _LayerReplayBank:
    captured_child_model_state_sha256: str
    layer_index: int
    pre_residual_by_batch: Mapping[int, Any] = field(repr=False, compare=False)
    replay_by_batch: Mapping[int, Lfm2CpcfMacrocellReplay] = field(
        repr=False,
        compare=False,
    )
    bank_sha256: str


def _layer_runtime(layer: Any) -> tuple[Any, Any]:
    torch = _torch()
    parameters = tuple(layer.parameters())
    dtypes = {value.dtype for value in parameters}
    if (
        not parameters
        or any(
            value.device.type == "meta"
            or value.dtype not in {torch.bfloat16, torch.float32}
            or not torch.is_floating_point(value)
            for value in parameters
        )
        or len({value.device for value in parameters}) != 1
        or len(dtypes) != 1
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Every child macrocell must use one materialized BF16 or FP32 dtype "
            "on one device."
        )
    return parameters[0].device, parameters[0].dtype


def _initial_residual_bank(
    target_model: Any,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
) -> Mapping[int, Any]:
    torch = _torch()
    embedding = getattr(target_model.get_input_embeddings(), "weight", None)
    if not isinstance(embedding, torch.nn.Parameter) or embedding.dtype not in {
        torch.bfloat16,
        torch.float32,
    }:
        raise Lfm2SemanticLayerStackRefoldError(
            "Saved child has no materialized BF16 or FP32 input table."
        )
    result = {}
    with torch.no_grad():
        for alignment in alignments:
            ids = alignment.phases[-1].canonical_target_input_ids
            if bool((ids < 0).any().item()) or bool(
                (ids >= int(embedding.shape[0])).any().item()
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Target-native replay IDs exceed the saved vocabulary."
                )
            device_ids = ids.to(device=embedding.device, dtype=torch.long)
            result[alignment.source_batch_index] = (
                embedding.index_select(0, device_ids.reshape(-1))
                .reshape(1, int(ids.shape[1]), int(embedding.shape[1]))
                .detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
    return MappingProxyType(result)


def _replay_current_layer(
    target_model: Any,
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    layer_index: int,
    pre_residual_by_batch: Mapping[int, Any],
) -> _LayerReplayBank:
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index < 0 or layer_index >= len(layers):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Child has no replayable layer {layer_index}."
        )
    state = fingerprint_lfm2_parent(target_model).identity_sha256
    layer = layers[layer_index]
    device, dtype = _layer_runtime(layer)
    replays = {}
    receipts = []
    for alignment in alignments:
        batch = alignment.source_batch_index
        residual = pre_residual_by_batch[batch].to(
            device=device,
            dtype=dtype,
            non_blocking=False,
        )
        final = alignment.phases[-1]
        mask = final.canonical_target_attention_mask
        positions = final.canonical_target_position_ids
        if bool(getattr(layer, "is_attention_layer", False)):
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=_rope_from_target(
                    target_model,
                    residual,
                    positions,
                ),
                additive_causal_mask=_additive_causal_mask(mask, residual),
            )
        else:
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=mask.to(device=device, dtype=dtype),
            )
        replays[batch] = replay
        receipts.append(
            {
                "batch_index": batch,
                "alignment_sha256": alignment.alignment_sha256,
                "replay_sha256": replay.replay_sha256,
            }
        )
    if fingerprint_lfm2_parent(target_model).identity_sha256 != state:
        raise Lfm2SemanticLayerStackRefoldError(
            "Analytical macrocell replay mutated the child."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-layer-replay-bank-v1",
        "layer_index": layer_index,
        "captured_child_model_state_sha256": state,
        "batch_replays": receipts,
        "model_forward_calls": 0,
        "layer_forward_calls": 0,
        "training_steps": 0,
    }
    return _LayerReplayBank(
        captured_child_model_state_sha256=state,
        layer_index=layer_index,
        pre_residual_by_batch=MappingProxyType(dict(pre_residual_by_batch)),
        replay_by_batch=MappingProxyType(replays),
        bank_sha256=canonical_sha256(body),
    )


def _next_residual_bank(bank: _LayerReplayBank) -> Mapping[int, Any]:
    return MappingProxyType(
        {
            batch: replay.post_layer_residual
            for batch, replay in bank.replay_by_batch.items()
        }
    )


@dataclass(slots=True)
class _GlobalResidualTrajectoryAccumulator:
    """Only the frozen rows/primals consumed by the post-stack closure."""

    batch_index: int
    alignment_sha256: str
    selected_target_positions: Any
    source_donor_positions: Any
    compact_donor_positions: Any
    canonical_boundary_row_indices: Any
    macrocells: list[Any] = field(default_factory=list)
    mapped_parent_pre_residuals: list[Any] = field(default_factory=list)
    mapped_parent_post_operator_residuals: list[Any] = field(default_factory=list)
    mapped_parent_post_layer_residuals: list[Any] = field(default_factory=list)
    mapped_parent_terminal_final_norm: Any | None = None


def _source_local_attention_output_required(
    recipe: Lfm2SemanticBodyRecipe,
) -> bool:
    return bool(
        recipe.component_lanes.get("attention_output_residual")
        == _SOURCE_LOCAL_ATTENTION_OUTPUT_LANE
    )


def _source_local_short_conv_output_required(
    recipe: Lfm2SemanticBodyRecipe,
) -> bool:
    return bool(
        recipe.component_lanes.get("short_conv_output_residual")
        == _SOURCE_LOCAL_SHORT_CONV_OUTPUT_LANE
    )


def _source_local_global_residual_required(
    recipe: Lfm2SemanticBodyRecipe,
) -> bool:
    return bool(
        recipe.component_lanes.get("global_residual")
        == _SOURCE_LOCAL_GLOBAL_RESIDUAL_LANE
    )


def _global_residual_closure_required(
    recipe: Lfm2SemanticBodyRecipe,
) -> bool:
    """Select the automatic closure lane without consulting quality metrics."""

    return bool(
        recipe.component_lanes["qk"] == _COUPLED_ATTENTION_QDEIM_LANE
        and recipe.component_lanes["value_output"] == _COUPLED_ATTENTION_QDEIM_LANE
        and recipe.component_lanes["mlp"] in _GLOBAL_RESIDUAL_MLP_LANES
    )


def _prepare_global_residual_accumulators(
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
) -> tuple[
    Mapping[int, _GlobalResidualTrajectoryAccumulator],
    Mapping[str, Any],
]:
    """Freeze exact semantic gather/scatter maps for every cache trajectory."""

    torch = _torch()
    ordered = tuple(sorted(alignments, key=lambda value: value.source_batch_index))
    if tuple(value.source_batch_index for value in ordered) != tuple(
        range(len(ordered))
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Global residual evidence requires contiguous canonical batch indices."
        )
    accumulators: dict[int, _GlobalResidualTrajectoryAccumulator] = {}
    receipt_rows = []
    canonical_offset = 0
    for alignment in ordered:
        final = alignment.phases[-1]
        boundary = tuple(final.target_to_donor_boundary_positions)
        target_positions = tuple(
            index
            for index, donor_position in enumerate(boundary)
            if donor_position is not None
        )
        donor_positions = tuple(
            int(donor_position)
            for donor_position in boundary
            if donor_position is not None
        )
        if not target_positions or donor_positions != tuple(
            range(final.donor_full_prefix_tokens)
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Global residual evidence requires every exact donor boundary once."
            )
        row_count = len(target_positions)
        target_index = torch.tensor(target_positions, dtype=torch.int64)
        source_donor_index = torch.tensor(donor_positions, dtype=torch.int64)
        compact_donor_index = torch.arange(row_count, dtype=torch.int64)
        canonical_index = torch.arange(
            canonical_offset,
            canonical_offset + row_count,
            dtype=torch.int64,
        )
        canonical_offset += row_count
        accumulator = _GlobalResidualTrajectoryAccumulator(
            batch_index=alignment.source_batch_index,
            alignment_sha256=alignment.alignment_sha256,
            selected_target_positions=target_index,
            source_donor_positions=source_donor_index,
            compact_donor_positions=compact_donor_index,
            canonical_boundary_row_indices=canonical_index,
        )
        accumulators[alignment.source_batch_index] = accumulator
        receipt_rows.append(
            {
                "batch_index": alignment.source_batch_index,
                "alignment_sha256": alignment.alignment_sha256,
                "selected_target_positions": list(target_positions),
                "source_donor_positions": list(donor_positions),
                "compact_donor_positions": list(range(row_count)),
                "canonical_boundary_row_indices": list(
                    range(canonical_offset - row_count, canonical_offset)
                ),
            }
        )
    receipt = _sealed(
        {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "kind": "global_residual_semantic_row_maps",
            "trajectory_count": len(ordered),
            "canonical_boundary_row_count": canonical_offset,
            "trajectories": receipt_rows,
            "exact_target_donor_canonical_maps": True,
            "parent_histories_compacted_to_selected_donor_rows": True,
            "numeric_or_manual_gate_used": False,
        }
    )
    return MappingProxyType(accumulators), receipt


def _capture_global_parent_layer_boundaries(
    accumulators: Mapping[int, _GlobalResidualTrajectoryAccumulator],
    *,
    parent: Mapping[int, _ParentLayerTrajectory],
    residual_chart: Any,
    producer_encoder: Any,
    layer_index: int,
    terminal_layer_index: int,
    capture_source_local_pre: bool = False,
) -> None:
    """Retain two selected mapped boundaries, then let raw histories die."""

    torch = _torch()
    chart = residual_chart.detach().to(device="cpu", dtype=torch.float64)
    encoder = producer_encoder.detach().to(device="cpu", dtype=torch.float64)
    if (
        chart.ndim != 2
        or encoder.ndim != 2
        or not bool(torch.isfinite(chart).all().item())
        or not bool(torch.isfinite(encoder).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Global residual parent mappings must be finite matrices."
        )
    for batch_index, accumulator in accumulators.items():
        trajectory = parent.get(batch_index)
        if trajectory is None:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Global residual evidence lacks parent batch {batch_index}."
            )
        donor_index = accumulator.source_donor_positions

        def mapped_boundary(name: str, mapping: Any) -> Any:
            history = trajectory.final_histories.get(name)
            if (
                not isinstance(history, torch.Tensor)
                or history.ndim != 3
                or int(history.shape[0]) != 1
                or int(history.shape[1]) <= int(donor_index.max().item())
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Global residual evidence lacks {name!r} at layer "
                    f"{layer_index}, batch {batch_index}."
                )
            selected = history.detach().to(device="cpu", dtype=torch.float64)
            selected = selected.index_select(1, donor_index)
            result = (selected @ mapping).contiguous()
            if not bool(torch.isfinite(result).all().item()):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Mapped parent boundary {name!r} is nonfinite."
                )
            return result

        if capture_source_local_pre:
            accumulator.mapped_parent_pre_residuals.append(
                mapped_boundary("residual.pre", chart)
            )
        accumulator.mapped_parent_post_operator_residuals.append(
            mapped_boundary("residual.post_operator", chart)
        )
        accumulator.mapped_parent_post_layer_residuals.append(
            mapped_boundary("residual.post_layer", chart)
        )
        if layer_index == terminal_layer_index:
            accumulator.mapped_parent_terminal_final_norm = mapped_boundary(
                "output.final_norm_output",
                encoder,
            )


def _positive_norm_epsilon(module: Any, label: str) -> float:
    value = getattr(module, "variance_epsilon", None)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{label}.variance_epsilon must be finite and positive."
        )
    return float(value)


def _global_macrocell_parameters(
    target_model: Any,
    *,
    layer_index: int,
    residual: Any,
    alignment: Lfm2SemanticSpanCpcfAlignment,
    deferred_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior | None,
) -> tuple[Lfm2AttentionParameters | Lfm2ShortConvParameters, Lfm2MlpParameters]:
    """Translate one installed LFM2 layer into the public analytic FP64 API."""

    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index < 0 or layer_index >= len(layers):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Global residual capture cannot resolve layer {layer_index}."
        )
    layer = layers[layer_index]
    device, dtype = _layer_runtime(layer)
    source = residual.detach().to(device=device, dtype=dtype)
    final = alignment.phases[-1]
    operator_epsilon = _positive_norm_epsilon(
        layer.operator_norm,
        f"model.layers.{layer_index}.operator_norm",
    )
    if bool(getattr(layer, "is_attention_layer", False)):
        attention = layer.self_attn
        head_dim = getattr(attention, "head_dim", None)
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Layer {layer_index} has no valid attention head dimension."
            )
        query_heads = int(attention.q_proj.weight.shape[0]) // head_dim
        key_value_heads = int(attention.k_proj.weight.shape[0]) // head_dim
        query_epsilon = _positive_norm_epsilon(
            attention.q_layernorm,
            f"model.layers.{layer_index}.self_attn.q_layernorm",
        )
        key_epsilon = _positive_norm_epsilon(
            attention.k_layernorm,
            f"model.layers.{layer_index}.self_attn.k_layernorm",
        )
        if query_epsilon != key_epsilon:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Layer {layer_index} Q/K RMSNorm epsilons differ."
            )
        cosine, sine = _rope_from_target(
            target_model,
            source,
            final.canonical_target_position_ids,
        )
        additive_mask = _additive_causal_mask(
            final.canonical_target_attention_mask,
            source,
        )
        operator_parameters: Lfm2AttentionParameters | Lfm2ShortConvParameters = (
            Lfm2AttentionParameters(
                operator_norm_gain=layer.operator_norm.weight,
                operator_norm_epsilon=operator_epsilon,
                query_weight=attention.q_proj.weight,
                key_weight=attention.k_proj.weight,
                value_weight=attention.v_proj.weight,
                output_weight=attention.out_proj.weight,
                query_norm_gain=attention.q_layernorm.weight,
                key_norm_gain=attention.k_layernorm.weight,
                query_key_norm_epsilon=query_epsilon,
                rope_cosine=cosine,
                rope_sine=sine,
                causal_mask=(additive_mask == 0),
                query_heads=query_heads,
                key_value_heads=key_value_heads,
                head_dim=head_dim,
                scaling=float(attention.scaling),
            )
        )
    else:
        short_conv = layer.conv
        operator_parameters = Lfm2ShortConvParameters(
            operator_norm_gain=layer.operator_norm.weight,
            operator_norm_epsilon=operator_epsilon,
            input_weight=short_conv.in_proj.weight,
            input_bias=short_conv.in_proj.bias,
            stored_fir_kernel=short_conv.conv.weight,
            fir_bias=short_conv.conv.bias,
            output_weight=short_conv.out_proj.weight,
            output_bias=short_conv.out_proj.bias,
            padding_mask=final.canonical_target_attention_mask,
        )
    down_weight = layer.feed_forward.w2.weight
    terminal_layer_index = len(layers) - 1
    if layer_index == terminal_layer_index:
        if deferred_final_w2_prior is None:
            raise Lfm2SemanticLayerStackRefoldError(
                "Final global primal requires a real deferred W2 prior."
            )
        down_weight = deferred_final_w2_prior.down_weight
    mlp_parameters = Lfm2MlpParameters(
        norm_gain=layer.ffn_norm.weight,
        norm_epsilon=_positive_norm_epsilon(
            layer.ffn_norm,
            f"model.layers.{layer_index}.ffn_norm",
        ),
        gate_weight=layer.feed_forward.w1.weight,
        up_weight=layer.feed_forward.w3.weight,
        down_weight=down_weight,
    )
    return operator_parameters, mlp_parameters


def _capture_global_child_layer_primals(
    accumulators: Mapping[int, _GlobalResidualTrajectoryAccumulator],
    *,
    target_model: Any,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    replay_bank: _LayerReplayBank,
    layer_index: int,
    deferred_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior | None,
) -> None:
    """Capture final installed child primals without a model/layer forward."""

    by_batch = {value.source_batch_index: value for value in alignments}
    for batch_index, accumulator in accumulators.items():
        alignment = by_batch.get(batch_index)
        residual = replay_bank.pre_residual_by_batch.get(batch_index)
        if alignment is None or residual is None:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Global child primal lacks batch {batch_index}."
            )
        operator_parameters, mlp_parameters = _global_macrocell_parameters(
            target_model,
            layer_index=layer_index,
            residual=residual,
            alignment=alignment,
            deferred_final_w2_prior=deferred_final_w2_prior,
        )
        accumulator.macrocells.append(
            capture_lfm2_macrocell_linearization(
                residual,
                operator_parameters,
                mlp_parameters,
            )
        )


def _global_live_writer_state(
    target_model: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    """Return every live output writer except the deferred terminal W2."""

    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    layer_count = len(layers)
    if layer_count <= 0:
        raise Lfm2SemanticLayerStackRefoldError(
            "Global residual closure requires a nonempty complete child stack."
        )
    terminal_layer_index = layer_count - 1
    parameters = {}
    head_dimensions = set()
    for layer_index, layer in enumerate(layers):
        if bool(getattr(layer, "is_attention_layer", False)):
            site_id = f"model.layers.{layer_index}.self_attn.out_proj.weight"
            parameter = layer.self_attn.out_proj.weight
            head_dimensions.add(int(layer.self_attn.head_dim))
        else:
            site_id = f"model.layers.{layer_index}.conv.out_proj.weight"
            parameter = layer.conv.out_proj.weight
        parameters[site_id] = parameter
        if layer_index < terminal_layer_index:
            parameters[f"model.layers.{layer_index}.feed_forward.w2.weight"] = (
                layer.feed_forward.w2.weight
            )
    if (
        len(parameters) != (2 * layer_count) - 1
        or len({id(value) for value in parameters.values()}) != len(parameters)
        or any(
            not isinstance(value, torch.nn.Parameter) for value in parameters.values()
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Global residual writer topology is not the exact dynamic "
            "live-output/one-deferred contract."
        )
    if not head_dimensions:
        raise Lfm2SemanticLayerStackRefoldError(
            "The v1 global residual kernel requires at least one attention site "
            "to define its shared head dimension; conv-only topology support is "
            "deferred to the per-site-dimension kernel."
        )
    if len(head_dimensions) != 1:
        raise Lfm2SemanticLayerStackRefoldError(
            "The v1 global residual kernel supports one authenticated shared "
            "attention head dimension; heterogeneous per-site head dimensions "
            "require the general topology kernel."
        )
    seeds = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in parameters.items()
    }
    return MappingProxyType(parameters), MappingProxyType(seeds), head_dimensions.pop()


def _build_global_residual_evidence_from_accumulators(
    accumulators: Mapping[int, _GlobalResidualTrajectoryAccumulator],
    *,
    target_model: Any,
    mapped_parent_tied_table: Any,
    deferred_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior,
    source_local: bool = False,
) -> tuple[
    Lfm2SemanticGlobalResidualEvidencePackage,
    Mapping[str, Any],
]:
    """Finalize typed evidence plus its authenticated v1 output-writer seeds."""

    torch = _torch()
    final_norm = getattr(
        getattr(target_model, "model", None),
        "embedding_norm",
        None,
    )
    if final_norm is None:
        raise Lfm2SemanticLayerStackRefoldError(
            "Global residual closure requires the child's final RMSNorm."
        )
    terminal_parameters = Lfm2TerminalNormParameters(
        gain=final_norm.weight,
        epsilon=_positive_norm_epsilon(final_norm, "model.embedding_norm"),
    )
    layer_count = len(
        tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    )
    if layer_count <= 0:
        raise Lfm2SemanticLayerStackRefoldError(
            "Global residual evidence requires a nonempty child stack."
        )
    trajectories = []
    for batch_index in range(len(accumulators)):
        accumulator = accumulators[batch_index]
        if (
            len(accumulator.macrocells) != layer_count
            or (
                source_local
                and len(accumulator.mapped_parent_pre_residuals) != layer_count
            )
            or (not source_local and len(accumulator.mapped_parent_pre_residuals) != 0)
            or len(accumulator.mapped_parent_post_operator_residuals) != layer_count
            or len(accumulator.mapped_parent_post_layer_residuals) != layer_count
            or accumulator.mapped_parent_terminal_final_norm is None
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Global residual trajectory {batch_index} is incomplete."
            )
        terminal = capture_lfm2_terminal_norm_primal(
            accumulator.macrocells[-1].mlp.output,
            terminal_parameters,
        )
        primal = Lfm2StackPrimalBatch(
            macrocells=tuple(accumulator.macrocells),
            selected_target_positions=accumulator.selected_target_positions,
            canonical_boundary_row_indices=(accumulator.canonical_boundary_row_indices),
            terminal=terminal,
        )
        trajectories.append(
            Lfm2SemanticGlobalResidualTrajectoryEvidence(
                batch_index=batch_index,
                primal_batch=primal,
                selected_donor_positions=accumulator.compact_donor_positions,
                mapped_parent_post_operator_residuals=tuple(
                    accumulator.mapped_parent_post_operator_residuals
                ),
                mapped_parent_post_layer_residuals=tuple(
                    accumulator.mapped_parent_post_layer_residuals
                ),
                mapped_parent_terminal_final_norm=(
                    accumulator.mapped_parent_terminal_final_norm
                ),
                mapped_parent_pre_residuals=(
                    tuple(accumulator.mapped_parent_pre_residuals)
                    if source_local
                    else None
                ),
            )
        )
    parameters, seeds, head_dimension = _global_live_writer_state(target_model)
    if not isinstance(mapped_parent_tied_table, torch.Tensor):
        raise Lfm2SemanticLayerStackRefoldError(
            "Global residual closure requires the mapped parent tied table."
        )
    evidence = Lfm2SemanticGlobalResidualEvidencePackage(
        trajectories=tuple(trajectories),
        live_writer_seeds=seeds,
        deferred_final_w2_prior=deferred_final_w2_prior,
        mapped_parent_tied_table=(
            mapped_parent_tied_table.detach().to(device="cpu").contiguous()
        ),
        head_dimension=head_dimension,
    )
    return evidence, parameters


def _global_live_physical_state(target_model: Any) -> Mapping[str, Any]:
    """Return every producer-aware physical matrix except deferred final W2."""

    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if not layers:
        raise Lfm2SemanticLayerStackRefoldError(
            "Producer-aware closure requires a nonempty complete child stack."
        )
    terminal_layer_index = len(layers) - 1
    parameters: dict[str, Any] = {}
    expected_count = 0
    for layer_index, layer in enumerate(layers):
        if bool(getattr(layer, "is_attention_layer", False)):
            prefix = f"model.layers.{layer_index}.self_attn"
            parameters.update(
                {
                    f"{prefix}.q_proj.weight": layer.self_attn.q_proj.weight,
                    f"{prefix}.k_proj.weight": layer.self_attn.k_proj.weight,
                    f"{prefix}.v_proj.weight": layer.self_attn.v_proj.weight,
                    f"{prefix}.out_proj.weight": layer.self_attn.out_proj.weight,
                }
            )
            expected_count += 4
        else:
            prefix = f"model.layers.{layer_index}.conv"
            parameters.update(
                {
                    f"{prefix}.in_proj.weight": layer.conv.in_proj.weight,
                    f"{prefix}.out_proj.weight": layer.conv.out_proj.weight,
                }
            )
            expected_count += 2
        mlp_prefix = f"model.layers.{layer_index}.feed_forward"
        parameters[f"{mlp_prefix}.w1.weight"] = layer.feed_forward.w1.weight
        parameters[f"{mlp_prefix}.w3.weight"] = layer.feed_forward.w3.weight
        expected_count += 2
        if layer_index < terminal_layer_index:
            parameters[f"{mlp_prefix}.w2.weight"] = layer.feed_forward.w2.weight
            expected_count += 1
    if (
        len(parameters) != expected_count
        or len({id(value) for value in parameters.values()}) != len(parameters)
        or any(
            not isinstance(value, torch.nn.Parameter)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.ndim != 2
            for value in parameters.values()
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Producer-aware physical topology is not the exact dynamic Q/K/V or "
            "conv-input, output, gate/up, nonfinal-down contract."
        )
    return MappingProxyType(parameters)


def _compile_install_global_residual_v1_diagnostic_from_accumulators(
    accumulators: Mapping[int, _GlobalResidualTrajectoryAccumulator],
    *,
    target_model: Any,
    mapped_parent_tied_table: Any,
    deferred_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior,
    deferred_final_writer_parameter: Any,
    closure_policy: Lfm2GlobalResidualClosurePolicy | None = None,
) -> tuple[Any, Lfm2SemanticGlobalResidualEvidencePackage]:
    """Run the retained output-writer-only v1 diagnostic baseline explicitly."""

    evidence, parameters = _build_global_residual_evidence_from_accumulators(
        accumulators,
        target_model=target_model,
        mapped_parent_tied_table=mapped_parent_tied_table,
        deferred_final_w2_prior=deferred_final_w2_prior,
    )
    installation = compile_install_lfm2_semantic_global_residual_closure(
        evidence,
        live_writer_parameters=parameters,
        deferred_final_writer_parameter=deferred_final_writer_parameter,
        policy=closure_policy,
    )
    return installation, evidence


def _compile_install_global_residual_from_accumulators(
    accumulators: Mapping[int, _GlobalResidualTrajectoryAccumulator],
    *,
    target_model: Any,
    mapped_parent_tied_table: Any,
    deferred_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior,
    deferred_final_writer_parameter: Any,
    closure_policy: Lfm2GlobalResidualClosurePolicy | None = None,
    correction_scale_frontier: Sequence[float] = (1.0,),
    correction_scale: float = 1.0,
    materialization_compute_device: str = "cpu",
    requested_basis_rank: int | None = None,
    target_dtype_representability_level: str = "first_effectful",
    source_local_boundaries: bool = False,
) -> tuple[Any, Lfm2SemanticGlobalResidualEvidencePackage]:
    """Execute the canonical producer-aware v2 physical residual closure."""

    evidence, _output_writer_parameters = (
        _build_global_residual_evidence_from_accumulators(
            accumulators,
            target_model=target_model,
            mapped_parent_tied_table=mapped_parent_tied_table,
            deferred_final_w2_prior=deferred_final_w2_prior,
            source_local=source_local_boundaries,
        )
    )
    physical_parameters = _global_live_physical_state(target_model)
    scale_policy = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=tuple(
            float(value) for value in correction_scale_frontier
        ),
        materialization_scale=float(correction_scale),
    )
    if requested_basis_rank is not None and (
        isinstance(requested_basis_rank, bool)
        or not isinstance(requested_basis_rank, int)
        or requested_basis_rank <= 0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "requested_basis_rank must be a positive integer or None."
        )
    installation = integrate_lfm2_producer_aware_global_closure(
        evidence,
        live_physical_parameters=physical_parameters,
        deferred_final_writer_parameter=deferred_final_writer_parameter,
        closure_policy=closure_policy,
        correction_scale_policy=scale_policy,
        materialization_compute_device=materialization_compute_device,
        requested_basis_rank=requested_basis_rank,
        target_dtype_representability_level=(target_dtype_representability_level),
        **(
            {"boundary_mode": (LFM2_PRODUCER_AWARE_SOURCE_LOCAL_BOUNDARY_MODE)}
            if source_local_boundaries
            else {}
        ),
    )
    return installation, evidence


class _DirectParentSafetensorsSession:
    """One exact local-parent identity reconstruction and open tensor session."""

    def __init__(
        self,
        model_directory: str | os.PathLike[str],
        *,
        expected_parent_identity_sha256: str,
        expected_parent_record: Mapping[str, Any],
    ) -> None:
        self.directory = Path(model_directory).expanduser().resolve()
        self.parent_identity_sha256 = _digest(
            expected_parent_identity_sha256,
            "parent model identity",
        )
        self.expected_parent_record = _validated_cache_parent_record(
            expected_parent_record,
            expected_identity_sha256=self.parent_identity_sha256,
        )
        self._stack: ExitStack | None = None
        self._handles: tuple[Any, ...] = ()
        self._keys: Mapping[str, int] = MappingProxyType({})
        self._loaded: dict[str, str] = {}
        self._checkpoint_tensor_sha256: Mapping[str, str] = MappingProxyType({})
        self._file_identities: tuple[tuple[str, Path, int, str], ...] = ()
        self._identity_receipt: Mapping[str, Any] = MappingProxyType({})
        self.config_file_sha256 = ""
        self.normalized_config_sha256 = ""
        self.norm_epsilon = 0.0
        self.manifest_sha256 = ""
        self.open_receipt_sha256 = ""
        self.parent_identity_reconstruction_sha256 = ""

    def __enter__(self) -> _DirectParentSafetensorsSession:
        if (
            not self.directory.is_dir()
            or self.directory.is_symlink()
            or self._stack is not None
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Parent model directory is missing, unsafe, or already open."
            )
        try:
            from safetensors import safe_open
            from transformers import Lfm2Config, Lfm2ForCausalLM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Exact direct parent loading requires safetensors and Lfm2Config."
            ) from exc
        config = self.directory / "config.json"
        if not config.is_file() or config.is_symlink():
            raise Lfm2SemanticLayerStackRefoldError(
                "Parent model directory has no safe config.json."
            )
        try:
            config_bytes = config.read_bytes()
            config_value = json.loads(config_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2SemanticLayerStackRefoldError(
                "Parent config.json is invalid."
            ) from exc
        norm_epsilon = (
            config_value.get("norm_eps") if isinstance(config_value, Mapping) else None
        )
        if (
            not isinstance(config_value, Mapping)
            or config_value.get("model_type") != "lfm2"
            or isinstance(norm_epsilon, bool)
            or not isinstance(norm_epsilon, (int, float))
            or not math.isfinite(float(norm_epsilon))
            or float(norm_epsilon) <= 0.0
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent checkpoint is not LFM2 or has no finite norm_eps."
            )
        self.config_file_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self.norm_epsilon = float(norm_epsilon)
        concrete_config = Lfm2Config(**dict(config_value))
        concrete_model_class = (
            f"{Lfm2ForCausalLM.__module__}.{Lfm2ForCausalLM.__qualname__}"
        )
        normalized_config = dict(concrete_config.to_dict())
        normalized_config.pop("_name_or_path", None)
        self.normalized_config_sha256 = canonical_sha256(normalized_config)
        if (
            self.normalized_config_sha256
            != self.expected_parent_record["config_sha256"]
            or concrete_model_class != self.expected_parent_record["model_class"]
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent config or concrete model class differs from the "
                "sealed cache parent identity."
            )
        shards = _local_safetensor_shards(self.directory)
        try:
            resolved_shards = tuple(path.resolve(strict=True) for path in shards)
        except OSError as exc:
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent safetensor shard resolution failed."
            ) from exc
        if any(path.parent != self.directory for path in resolved_shards):
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent safetensor shards must remain inside the "
                "authenticated model directory."
            )
        shards = resolved_shards
        index = self.directory / "model.safetensors.index.json"
        index_record = None
        index_weight_map: Mapping[str, Any] | None = None
        logical_shard_names = tuple(path.name for path in shards)
        if index.exists():
            if not index.is_file() or index.is_symlink():
                raise Lfm2SemanticLayerStackRefoldError(
                    "Parent safetensors index is unsafe."
                )
            try:
                index_bytes = index.read_bytes()
                index_payload = json.loads(index_bytes.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Lfm2SemanticLayerStackRefoldError(
                    "Parent safetensors index is invalid."
                ) from exc
            index_weight_map = (
                index_payload.get("weight_map")
                if isinstance(index_payload, Mapping)
                else None
            )
            if (
                not isinstance(index_weight_map, Mapping)
                or not index_weight_map
                or any(
                    not isinstance(name, str)
                    or not name
                    or not isinstance(shard_name, str)
                    or not shard_name
                    for name, shard_name in index_weight_map.items()
                )
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Parent safetensors index has no safe complete weight map."
                )
            logical_shard_names = tuple(sorted(set(index_weight_map.values())))
            if len(logical_shard_names) != len(shards):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Parent safetensors index and local shard inventory disagree."
                )
            index_record = {
                "name": index.name,
                "size_bytes": len(index_bytes),
                "file_sha256": hashlib.sha256(index_bytes).hexdigest(),
            }
        shard_records = []
        for ordinal, path in enumerate(shards):
            size_bytes, file_sha256 = _file_size_and_sha256(path)
            shard_records.append(
                {
                    "ordinal": ordinal,
                    "logical_name": logical_shard_names[ordinal],
                    "resolved_name": path.name,
                    "size_bytes": size_bytes,
                    "file_sha256": file_sha256,
                }
            )
        manifest = {
            "schema": "neural-forge-direct-parent-checkpoint-manifest-v2",
            "parent_model_identity_sha256": self.parent_identity_sha256,
            "config_file_sha256": self.config_file_sha256,
            "normalized_config_sha256": self.normalized_config_sha256,
            "norm_epsilon": self.norm_epsilon,
            "index": index_record,
            "shards": shard_records,
        }
        self.manifest_sha256 = canonical_sha256(manifest)
        stack = ExitStack()
        handles = []
        keys: dict[str, int] = {}
        checkpoint_tensor_hashes: dict[str, str] = {}
        checkpoint_tensor_records: list[dict[str, Any]] = []
        checkpoint_floating_dtypes: set[Any] = set()
        try:
            for shard_index, path in enumerate(shards):
                handle = stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                handles.append(handle)
                for tensor_ordinal, key in enumerate(sorted(handle.keys())):
                    if key in keys:
                        raise Lfm2SemanticLayerStackRefoldError(
                            f"Parent tensor {key!r} appears in multiple shards."
                        )
                    keys[key] = shard_index
                    value = handle.get_tensor(key)
                    digest = _cache_tensor_sha256(_torch(), value)
                    checkpoint_tensor_hashes[key] = digest
                    if _torch().is_floating_point(value) and not value.is_complex():
                        checkpoint_floating_dtypes.add(value.dtype)
                    checkpoint_tensor_records.append(
                        {
                            "name": key,
                            "shard_ordinal": shard_index,
                            "shard_name": logical_shard_names[shard_index],
                            "tensor_ordinal_in_shard": tensor_ordinal,
                            "dtype": str(value.dtype).removeprefix("torch."),
                            "shape": [int(size) for size in value.shape],
                            "tensor_sha256": digest,
                        }
                    )
            if index_weight_map is not None and (
                set(index_weight_map) != set(keys)
                or any(
                    index_weight_map[name] != logical_shard_names[shard_index]
                    for name, shard_index in keys.items()
                )
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Parent safetensors index does not exactly locate every tensor."
                )

            expected_state_count = int(
                self.expected_parent_record["state_tensor_count"]
            )
            state_tensor_hashes = dict(checkpoint_tensor_hashes)
            if (
                bool(getattr(concrete_config, "tie_word_embeddings", False))
                and len(state_tensor_hashes) + 1 == expected_state_count
            ):
                embedding_key = "model.embed_tokens.weight"
                output_key = "lm_head.weight"
                if (
                    embedding_key in state_tensor_hashes
                    and output_key not in state_tensor_hashes
                ):
                    state_tensor_hashes[output_key] = state_tensor_hashes[embedding_key]
                elif (
                    output_key in state_tensor_hashes
                    and embedding_key not in state_tensor_hashes
                ):
                    state_tensor_hashes[embedding_key] = state_tensor_hashes[output_key]
            if len(state_tensor_hashes) != expected_state_count:
                raise Lfm2SemanticLayerStackRefoldError(
                    "Direct parent safetensors do not reconstruct every cached "
                    "state-dict entry."
                )
            state_digest = hashlib.sha256(b"neural-forge-lfm2-parent-state-v1\0")
            for name in sorted(state_tensor_hashes):
                state_digest.update(name.encode("utf-8"))
                state_digest.update(b"\0")
                state_digest.update(state_tensor_hashes[name].encode("ascii"))
                state_digest.update(b"\0")
            state_sha256 = state_digest.hexdigest()

            expected_buffer_count = int(
                self.expected_parent_record["runtime_buffer_count"]
            )
            runtime_dtype = (
                None
                if expected_buffer_count == 0
                else _direct_parent_runtime_dtype(
                    normalized_config,
                    checkpoint_floating_dtypes,
                )
            )
            runtime_buffers = _direct_parent_runtime_buffers(
                concrete_config,
                expected_buffer_count,
                dtype=runtime_dtype,
            )
            buffer_digest = hashlib.sha256(
                b"neural-forge-lfm2-parent-runtime-buffers-v1\0"
            )
            runtime_buffer_records = []
            for name in sorted(runtime_buffers):
                digest = _cache_tensor_sha256(_torch(), runtime_buffers[name])
                buffer_digest.update(name.encode("utf-8"))
                buffer_digest.update(b"\0")
                buffer_digest.update(digest.encode("ascii"))
                buffer_digest.update(b"\0")
                runtime_buffer_records.append(
                    {
                        "name": name,
                        "dtype": str(runtime_buffers[name].dtype).removeprefix(
                            "torch."
                        ),
                        "shape": [int(size) for size in runtime_buffers[name].shape],
                        "tensor_sha256": digest,
                    }
                )
            runtime_buffer_sha256 = buffer_digest.hexdigest()
            reconstructed_parent = {
                "schema": "neural-forge-lfm2-parent-fingerprint-v2",
                "model_class": concrete_model_class,
                "config_sha256": self.normalized_config_sha256,
                "state_dict_sha256": state_sha256,
                "state_tensor_count": len(state_tensor_hashes),
                "runtime_buffer_sha256": runtime_buffer_sha256,
                "runtime_buffer_count": len(runtime_buffers),
            }
            if (
                any(
                    reconstructed_parent[name] != self.expected_parent_record[name]
                    for name in (
                        "model_class",
                        "config_sha256",
                        "state_dict_sha256",
                        "state_tensor_count",
                        "runtime_buffer_sha256",
                        "runtime_buffer_count",
                    )
                )
                or canonical_sha256(reconstructed_parent) != self.parent_identity_sha256
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Direct parent checkpoint does not reproduce the sealed "
                    "cache's full parent model identity."
                )
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        self._handles = tuple(handles)
        self._keys = MappingProxyType(keys)
        self._checkpoint_tensor_sha256 = MappingProxyType(checkpoint_tensor_hashes)
        file_identities = [
            (
                "config",
                config,
                len(config_bytes),
                self.config_file_sha256,
            )
        ]
        if index_record is not None:
            file_identities.append(
                (
                    "index",
                    index,
                    int(index_record["size_bytes"]),
                    str(index_record["file_sha256"]),
                )
            )
        file_identities.extend(
            (
                f"shard:{record['ordinal']}",
                path,
                int(record["size_bytes"]),
                str(record["file_sha256"]),
            )
            for record, path in zip(shard_records, shards, strict=True)
        )
        self._file_identities = tuple(file_identities)
        identity_body = {
            "schema": "neural-forge-direct-parent-exact-identity-reconstruction-v1",
            "cache_parent_identity_sha256": self.parent_identity_sha256,
            "cache_parent_record": dict(self.expected_parent_record),
            "checkpoint_manifest_sha256": self.manifest_sha256,
            "config_file_sha256": self.config_file_sha256,
            "normalized_config_sha256": self.normalized_config_sha256,
            "checkpoint_tensor_inventory_sha256": canonical_sha256(
                checkpoint_tensor_records
            ),
            "checkpoint_tensor_count": len(checkpoint_tensor_records),
            "checkpoint_tensor_records": checkpoint_tensor_records,
            "reconstructed_state_dict_sha256": state_sha256,
            "reconstructed_state_tensor_count": len(state_tensor_hashes),
            "runtime_buffer_records": runtime_buffer_records,
            "reconstructed_runtime_buffer_sha256": runtime_buffer_sha256,
            "reconstructed_runtime_buffer_count": len(runtime_buffers),
            "reconstructed_parent_identity": reconstructed_parent,
            "reconstructed_parent_identity_sha256": canonical_sha256(
                reconstructed_parent
            ),
            "exact_parent_identity_authenticated": True,
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "network_access_performed": False,
        }
        self.parent_identity_reconstruction_sha256 = canonical_sha256(identity_body)
        self._identity_receipt = MappingProxyType(
            {
                **identity_body,
                "result_sha256": self.parent_identity_reconstruction_sha256,
            }
        )
        self.open_receipt_sha256 = canonical_sha256(
            {
                **manifest,
                "checkpoint_manifest_sha256": self.manifest_sha256,
                "tensor_key_inventory_sha256": canonical_sha256(sorted(keys)),
                "checkpoint_tensor_inventory_sha256": identity_body[
                    "checkpoint_tensor_inventory_sha256"
                ],
                "parent_identity_reconstruction_sha256": (
                    self.parent_identity_reconstruction_sha256
                ),
                "exact_parent_identity_authenticated": True,
                "shards_hashed_once": True,
                "shards_opened_once": True,
                "parent_model_objects_constructed": 0,
            }
        )
        return self

    def tensor(self, key: str) -> Any:
        if self._stack is None or key not in self._keys:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Direct parent checkpoint lacks tensor {key!r}."
            )
        value = self._handles[self._keys[key]].get_tensor(key)
        digest = _cache_tensor_sha256(_torch(), value)
        if digest != self._checkpoint_tensor_sha256[key]:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Parent tensor {key!r} differs from the authenticated inventory."
            )
        previous = self._loaded.setdefault(key, digest)
        if previous != digest:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Parent tensor {key!r} changed during direct loading."
            )
        return value.detach().to(device="cpu").contiguous().clone()

    def optional_tensor(self, key: str) -> Any | None:
        return None if key not in self._keys else self.tensor(key)

    def identity_receipt(self) -> Mapping[str, Any]:
        if self._stack is None or not self._identity_receipt:
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent exact identity session is not open."
            )
        return self._identity_receipt

    def receipt(self) -> Mapping[str, Any]:
        if self._stack is None:
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent session is not open."
            )
        changed_files = [
            label
            for label, path, expected_size, expected_sha256 in self._file_identities
            if (
                not path.is_file()
                or path.stat().st_size != expected_size
                or _file_sha256(path) != expected_sha256
            )
        ]
        if changed_files:
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent checkpoint files changed after exact identity "
                f"reconstruction: {changed_files}."
            )
        body = {
            "schema": "neural-forge-direct-parent-session-v2",
            "parent_model_identity_sha256": self.parent_identity_sha256,
            "checkpoint_manifest_sha256": self.manifest_sha256,
            "open_receipt_sha256": self.open_receipt_sha256,
            "config_file_sha256": self.config_file_sha256,
            "normalized_config_sha256": self.normalized_config_sha256,
            "norm_epsilon": self.norm_epsilon,
            "loaded_tensor_sha256": dict(sorted(self._loaded.items())),
            "parent_identity_reconstruction_sha256": (
                self.parent_identity_reconstruction_sha256
            ),
            "parent_identity_reconstruction": dict(self._identity_receipt),
            "exact_parent_identity_authenticated": True,
            "checkpoint_files_rehashed_after_materialization": True,
            "shards_hashed_once": True,
            "shards_opened_once": True,
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "network_access_performed": False,
        }
        return MappingProxyType({**body, "session_sha256": canonical_sha256(body)})

    def __exit__(self, *_args: Any) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles = ()
        self._keys = MappingProxyType({})


_COMMON_PARENT_FIELDS = (
    "residual.pre",
    "residual.post_operator",
    "mlp.input",
    "mlp.gate",
    "mlp.up",
    "mlp.post_swiglu",
    "residual.post_layer",
)
_SHORT_CONV_HISTORY_FIELDS = (
    "short_conv.b_history",
    "short_conv.c_history",
    "short_conv.x_history",
)


@dataclass(frozen=True, slots=True)
class _ParentLayerTrajectory:
    batch_index: int
    layer_index: int
    alignment_sha256: str
    phase_rows: Mapping[str, Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )
    phase_histories: Mapping[str, Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )
    final_histories: Mapping[str, Any] = field(repr=False, compare=False)
    evidence_sha256: str


def _extend_parent_history(
    previous: Any | None,
    current: Any,
    *,
    expected_full_tokens: int,
    field_name: str,
) -> Any:
    torch = _torch()
    value = current.detach().to(device="cpu").contiguous().clone()
    if value.ndim != 3 or int(value.shape[0]) != 1:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Parent field {field_name!r} is not one rank-3 trajectory."
        )
    if previous is None:
        if int(value.shape[1]) != expected_full_tokens:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Parent prefill field {field_name!r} is not a full prefix."
            )
        return value
    previous_tokens = int(previous.shape[1])
    suffix_tokens = expected_full_tokens - previous_tokens
    if suffix_tokens <= 0:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Parent field {field_name!r} did not advance causally."
        )
    if int(value.shape[1]) == suffix_tokens:
        return torch.cat((previous, value), dim=1).contiguous()
    if int(value.shape[1]) == expected_full_tokens:
        if not torch.equal(value[:, :previous_tokens], previous):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Parent field {field_name!r} changed its causal prefix."
            )
        return value
    raise Lfm2SemanticLayerStackRefoldError(
        f"Parent field {field_name!r} is neither a suffix nor full history."
    )


def _parent_layer_tensor_names(
    reference: Any,
    *,
    alignment: Lfm2SemanticSpanCpcfAlignment,
    layer_index: int,
    operator_kind: str,
    include_final_norm: bool,
) -> tuple[str, ...]:
    fields = list(_COMMON_PARENT_FIELDS)
    if operator_kind == "short_conv":
        fields.extend(_SHORT_CONV_HISTORY_FIELDS)
        fields.extend(
            (
                "short_conv.fir_kernel",
                "short_conv.fir_bias",
                "short_conv.fir_has_bias",
            )
        )
    names_set = {
        f"{phase.phase}.layer.{layer_index:03d}.{field}"
        for phase in alignment.phases
        for field in fields
    }
    if include_final_norm:
        names_set.update(
            f"{phase.phase}.output.final_norm_output" for phase in alignment.phases
        )
    names = tuple(sorted(names_set))
    missing = tuple(sorted(set(names).difference(reference.ordered_tensor_names)))
    if missing:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Complete cache lacks layer {layer_index} fields {list(missing[:6])}."
        )
    return names


def _load_parent_layer_trajectories(
    session: Any,
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    layer_index: int,
    operator_kind: str,
    include_final_norm: bool = False,
) -> Mapping[int, _ParentLayerTrajectory]:
    result = {}
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        if reference.index != alignment.source_batch_index:
            raise Lfm2SemanticLayerStackRefoldError(
                "Cache references and semantic alignments are reordered."
            )
        payload = session.load_selected_tensors(
            reference.index,
            _parent_layer_tensor_names(
                reference,
                alignment=alignment,
                layer_index=layer_index,
                operator_kind=operator_kind,
                include_final_norm=include_final_norm,
            ),
        )
        try:
            raw = dict(payload.tensors)
            fields = list(_COMMON_PARENT_FIELDS)
            if operator_kind == "short_conv":
                fields.extend(_SHORT_CONV_HISTORY_FIELDS)
            histories: dict[str, Any] = {}
            phase_rows = {}
            phase_histories = {}
            fir_reference = None
            for phase in alignment.phases:
                row_values = {}
                history_values = {}
                for field in fields:
                    key = f"{phase.phase}.layer.{layer_index:03d}.{field}"
                    value = raw[key]
                    history = _extend_parent_history(
                        histories.get(field),
                        value,
                        expected_full_tokens=phase.donor_full_prefix_tokens,
                        field_name=key,
                    )
                    histories[field] = history
                    history_values[field] = history
                    row_values[field] = value.contiguous().clone()
                if operator_kind == "short_conv":
                    prefix = f"{phase.phase}.layer.{layer_index:03d}.short_conv."
                    fir = (
                        raw[prefix + "fir_kernel"],
                        raw[prefix + "fir_bias"],
                        raw[prefix + "fir_has_bias"],
                    )
                    if fir_reference is None:
                        fir_reference = tuple(value.clone() for value in fir)
                    elif any(
                        not _torch().equal(left, right)
                        for left, right in zip(fir_reference, fir, strict=True)
                    ):
                        raise Lfm2SemanticLayerStackRefoldError(
                            f"Cached parent FIR changed across layer {layer_index} phases."
                        )
                    row_values["short_conv.fir_kernel"] = fir[0]
                    row_values["short_conv.fir_bias"] = fir[1]
                    row_values["short_conv.fir_has_bias"] = fir[2]
                if include_final_norm:
                    output_key = f"{phase.phase}.output.final_norm_output"
                    final_norm = raw[output_key]
                    final_norm_history = _extend_parent_history(
                        histories.get("output.final_norm_output"),
                        final_norm,
                        expected_full_tokens=phase.donor_full_prefix_tokens,
                        field_name=output_key,
                    )
                    histories["output.final_norm_output"] = final_norm_history
                    history_values["output.final_norm_output"] = final_norm_history
                    row_values["output.final_norm_output"] = (
                        final_norm.contiguous().clone()
                    )
                phase_rows[phase.phase] = MappingProxyType(row_values)
                phase_histories[phase.phase] = MappingProxyType(history_values)
            body = {
                "schema": "neural-forge-lfm2-parent-layer-trajectory-v1",
                "batch_index": reference.index,
                "layer_index": layer_index,
                "alignment_sha256": alignment.alignment_sha256,
                "operator_kind": operator_kind,
                "phase_row_sha256": {
                    phase: {
                        name: _tensor_sha256(value)
                        for name, value in sorted(values.items())
                    }
                    for phase, values in phase_rows.items()
                },
                "final_history_sha256": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(histories.items())
                },
            }
            result[reference.index] = _ParentLayerTrajectory(
                batch_index=reference.index,
                layer_index=layer_index,
                alignment_sha256=alignment.alignment_sha256,
                phase_rows=MappingProxyType(phase_rows),
                phase_histories=MappingProxyType(phase_histories),
                final_histories=MappingProxyType(histories),
                evidence_sha256=canonical_sha256(body),
            )
        finally:
            session.release_selected_tensors(payload)
    return MappingProxyType(result)


def _fit_residual_chart(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    producer_encoder: Any,
    layer_index: int,
    global_gauge_required: bool = False,
) -> tuple[Any, Mapping[str, Any]]:
    torch = _torch()
    ordered = tuple(sorted(alignments, key=lambda value: value.source_batch_index))
    if len(ordered) < 4:
        raise Lfm2SemanticLayerStackRefoldError(
            "Residual chart requires four isolated trajectories."
        )
    encoder = producer_encoder.detach().to(device="cpu", dtype=torch.float64)
    if encoder.ndim != 2 or not bool(torch.isfinite(encoder).all().item()):
        raise Lfm2SemanticLayerStackRefoldError(
            "The base producer encoder is not one finite residual chart."
        )
    count = len(ordered)
    counts = [max(1, count // 2), max(1, (3 * count) // 16), max(1, count // 8), 1]
    while sum(counts) > count:
        for index in (0, 1, 2):
            if counts[index] > 1 and sum(counts) > count:
                counts[index] -= 1
    counts[3] += count - sum(counts)
    partition_batches: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for partition, size in zip(_PARTITIONS, counts, strict=True):
        partition_batches[partition] = tuple(
            value.source_batch_index for value in ordered[cursor : cursor + size]
        )
        cursor += size
    if cursor != count or any(not value for value in partition_batches.values()):
        raise AssertionError("Internal residual-chart partitioning failed.")
    batch_partition = {
        batch: partition
        for partition, batches in partition_batches.items()
        for batch in batches
    }
    banks: dict[str, dict[str, list[Any]]] = {
        partition: {"left": [], "right": []} for partition in _PARTITIONS
    }
    for alignment in ordered:
        batch = alignment.source_batch_index
        source = parent[batch].final_histories["residual.pre"].to(torch.float64)
        target = child_bank.pre_residual_by_batch[batch].to(torch.float64)
        boundary = alignment.phases[-1].target_to_donor_boundary_positions
        target_positions = tuple(
            index for index, donor in enumerate(boundary) if donor is not None
        )
        donor_positions = tuple(int(value) for value in boundary if value is not None)
        if donor_positions != tuple(range(int(source.shape[1]))):
            raise Lfm2SemanticLayerStackRefoldError(
                "Semantic endpoints do not cover every parent residual row."
            )
        selected = target.index_select(
            1,
            torch.tensor(target_positions, dtype=torch.long),
        )[0]
        left = (source[0] @ encoder) / math.sqrt(int(source.shape[1]))
        right = selected / math.sqrt(int(selected.shape[0]))
        partition = batch_partition[batch]
        banks[partition]["left"].append(left)
        banks[partition]["right"].append(right)
    materialized = {
        partition: (
            torch.cat(tuple(banks[partition]["left"]), dim=0),
            torch.cat(tuple(banks[partition]["right"]), dim=0),
        )
        for partition in _PARTITIONS
    }
    left_fit, right_fit = materialized["fit"]
    identity = torch.eye(int(encoder.shape[1]), dtype=torch.float64)
    design_left, singular, design_right_h = torch.linalg.svd(
        left_fit,
        full_matrices=False,
    )
    largest = float(singular.max().item()) if singular.numel() else 0.0
    threshold = max(
        largest * 1e-8,
        largest
        * torch.finfo(torch.float64).eps
        * max(int(left_fit.shape[0]), int(left_fit.shape[1])),
    )
    supported = singular > threshold
    supported_rank = int(supported.sum().item())
    candidates: dict[str, Any] = {"base_identity": identity}
    if not isinstance(global_gauge_required, bool):
        raise TypeError("global_gauge_required must be boolean.")
    if supported_rank and not global_gauge_required:
        supported_right = design_right_h[supported]
        support_projector = supported_right.T @ supported_right
        cross_left, _cross_singular, cross_right_h = torch.linalg.svd(
            left_fit.T @ right_fit,
            full_matrices=False,
        )
        full_procrustes = cross_left @ cross_right_h
        candidates["rank_supported_procrustes"] = (
            identity + support_projector @ (full_procrustes - identity)
        ).contiguous()
        supported_left = design_left[:, supported]
        supported_singular = singular[supported]
        fit_residual = right_fit - left_fit
        projected_residual = supported_left.T @ fit_residual
        scale = max(largest * largest, 1e-30)
        for ratio in (1e-6, 1e-4, 1e-2, 1e-1, 1.0):
            ridge = ratio * scale
            coefficients = supported_singular / (supported_singular.square() + ridge)
            delta = supported_right.T @ (coefficients.unsqueeze(1) * projected_residual)
            candidates[f"rank_supported_ridge_{ratio:g}"] = (
                identity + delta
            ).contiguous()
    candidates = {
        name: value
        for name, value in candidates.items()
        if bool(torch.isfinite(value).all().item())
    }

    def metrics(rotation: Any, partition: str) -> Mapping[str, Any]:
        left, right = materialized[partition]
        prediction = left @ rotation
        difference = prediction - right
        denominator = right.square().mean().sqrt().clamp_min(1e-30)
        prediction_norm = torch.linalg.vector_norm(prediction, dim=1)
        target_norm = torch.linalg.vector_norm(right, dim=1)
        cosine_denominator = prediction_norm * target_norm
        cosine = torch.where(
            cosine_denominator > 0,
            (prediction * right).sum(dim=1) / cosine_denominator.clamp_min(1e-300),
            torch.where(
                torch.linalg.vector_norm(difference, dim=1) == 0,
                torch.ones_like(cosine_denominator),
                torch.zeros_like(cosine_denominator),
            ),
        ).clamp(-1.0, 1.0)
        return {
            "rows": int(prediction.shape[0]),
            "rmse": float(difference.square().mean().sqrt().item()),
            "relative_rmse": float(
                (difference.square().mean().sqrt() / denominator).item()
            ),
            "maximum_absolute_error": float(difference.abs().max().item()),
            "target_maximum_absolute_value": float(right.abs().max().item()),
            "mean_cosine": float(cosine.mean().item()),
            "minimum_cosine": float(cosine.min().item()),
        }

    candidate_metrics = {
        name: {partition: metrics(rotation, partition) for partition in _PARTITIONS}
        for name, rotation in candidates.items()
    }
    baseline = candidate_metrics["base_identity"]

    def nonregressing(
        candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
    ) -> bool:
        rmse_tolerance = max(1e-12, 1e-6 * max(incumbent["rmse"], 1e-12))
        maximum_tolerance = max(
            1e-12,
            1e-6 * max(incumbent["target_maximum_absolute_value"], 1e-12),
        )
        return bool(
            candidate["rmse"] <= incumbent["rmse"] + rmse_tolerance
            and candidate["maximum_absolute_error"]
            <= incumbent["maximum_absolute_error"] + maximum_tolerance
            and candidate["mean_cosine"] + 1e-8 >= incumbent["mean_cosine"]
            and candidate["minimum_cosine"] + 1e-8 >= incumbent["minimum_cosine"]
        )

    fit_check_nonregression_diagnostic = {
        name: nonregressing(values["fit_check"], baseline["fit_check"])
        for name, values in candidate_metrics.items()
    }
    selected = (
        "base_identity"
        if global_gauge_required or supported_rank == 0
        else "rank_supported_ridge_0.0001"
    )
    if selected not in candidates:
        raise Lfm2SemanticLayerStackRefoldError(
            "Deterministic FIT residual chart candidate is unavailable."
        )
    validation_nonregression_diagnostic = nonregressing(
        candidate_metrics[selected]["validation"],
        baseline["validation"],
    )
    rotation = candidates[selected]
    chart = (encoder @ rotation).contiguous()
    if not bool(torch.isfinite(chart).all().item()):
        raise Lfm2SemanticLayerStackRefoldError("Selected residual chart is nonfinite.")
    unsupported_identity_error = 0.0
    if supported_rank < int(design_right_h.shape[0]):
        unsupported = design_right_h[~supported]
        if unsupported.numel():
            unsupported_identity_error = float(
                (unsupported @ (rotation - identity)).abs().max().item()
            )
    if unsupported_identity_error > 1e-10:
        raise Lfm2SemanticLayerStackRefoldError(
            "Selected residual chart changed an unsupported FIT dimension."
        )

    body = {
        "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
        "layer_index": layer_index,
        "captured_child_model_state_sha256": (
            child_bank.captured_child_model_state_sha256
        ),
        "partition_batch_indices": {
            name: list(partition_batches[name]) for name in _PARTITIONS
        },
        "base_producer_encoder_sha256": _tensor_sha256(encoder),
        "rotation_sha256": _tensor_sha256(rotation),
        "chart_sha256": _tensor_sha256(chart),
        "singular_values_sha256": _tensor_sha256(singular),
        "supported_rank": supported_rank,
        "fit_design_width": int(left_fit.shape[1]),
        "support_singular_value_threshold": threshold,
        "unsupported_subspace_identity_maximum_absolute_error": (
            unsupported_identity_error
        ),
        "candidate_rotation_sha256": {
            name: _tensor_sha256(value) for name, value in sorted(candidates.items())
        },
        "candidate_partition_metrics": candidate_metrics,
        "fit_check_nonregression": fit_check_nonregression_diagnostic,
        "selected_before_validation": selected,
        "selected_candidate": selected,
        "global_self_dual_gauge_required": global_gauge_required,
        "independent_layer_rotation_allowed": not global_gauge_required,
        "global_gauge_preserved_without_layer_local_rotation": (
            not global_gauge_required or selected == "base_identity"
        ),
        "validation_nonregression_passed": validation_nonregression_diagnostic,
        "validation_rejection_fell_back_to_base": False,
        "candidate_family_constructed_from_fit_only": True,
        "deterministic_structural_selection_rule": (
            "global-gauge-or-zero-rank-keeps-identity-otherwise-fixed-1e-4-"
            "supported-ridge"
        ),
        "fit_check_used_for_eligibility_only": False,
        "selection_used_for_frozen_choice_only": False,
        "candidate_frozen_before_validation": True,
        "validation_reject_only": False,
        "all_scalar_metrics_diagnostic_only": True,
        "numeric_metric_used_for_candidate_selection": False,
        "numeric_metric_used_for_publication_or_rollback": False,
        "validation_triggered_refit": False,
        "unsupported_fit_subspace_preserved_as_identity": True,
        "trajectory_balanced_rows": True,
        "fit_only": True,
        "validation_rows_used_for_fit": 0,
        "numeric_parent_child_weight_match_objective": False,
        "training_steps": 0,
    }
    report = {**body, "result_sha256": canonical_sha256(body)}
    return chart, MappingProxyType(report)


def _body_global_fit_residual_rows(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    chart_receipt: Mapping[str, Any],
    layer_index: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Materialize the chart-authenticated FIT rows in raw child/parent space."""

    torch = _torch()
    verified_chart = _verify_sealed(chart_receipt, "residual chart")
    partition_value = verified_chart.get("partition_batch_indices")
    fit_batches = (
        partition_value.get("fit") if isinstance(partition_value, Mapping) else None
    )
    if (
        verified_chart.get("schema")
        != "neural-forge-lfm2-layer-stack-residual-chart-v1"
        or verified_chart.get("layer_index") != layer_index
        or not isinstance(fit_batches, list)
        or not fit_batches
        or any(
            isinstance(batch, bool) or not isinstance(batch, int) or batch < 0
            for batch in fit_batches
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global completion requires the sealed residual-chart FIT partition."
        )
    alignment_by_batch = {value.source_batch_index: value for value in alignments}
    if (
        len(alignment_by_batch) != len(alignments)
        or any(batch not in alignment_by_batch for batch in fit_batches)
        or any(batch not in parent for batch in fit_batches)
        or any(batch not in child_bank.pre_residual_by_batch for batch in fit_batches)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global FIT batches do not bind the exact parent/child rows."
        )
    child_rows = []
    parent_rows = []
    row_manifest = []
    for batch in fit_batches:
        alignment = alignment_by_batch[batch]
        boundary = alignment.phases[-1].target_to_donor_boundary_positions
        target_positions = tuple(
            index for index, donor in enumerate(boundary) if donor is not None
        )
        donor_positions = tuple(int(donor) for donor in boundary if donor is not None)
        source = (
            parent[batch]
            .final_histories["residual.pre"]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        target = (
            child_bank.pre_residual_by_batch[batch]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        if (
            source.ndim != 3
            or target.ndim != 3
            or int(source.shape[0]) != 1
            or int(target.shape[0]) != 1
            or donor_positions != tuple(range(int(source.shape[1])))
            or len(target_positions) != int(source.shape[1])
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Body-global completion requires complete one-to-one semantic "
                "residual boundaries."
            )
        source_index = torch.tensor(donor_positions, dtype=torch.long)
        target_index = torch.tensor(target_positions, dtype=torch.long)
        scale = math.sqrt(len(target_positions))
        selected_parent = source[0].index_select(0, source_index) / scale
        selected_child = target[0].index_select(0, target_index) / scale
        parent_rows.append(selected_parent.contiguous())
        child_rows.append(selected_child.contiguous())
        row_manifest.append(
            {
                "batch_index": batch,
                "alignment_sha256": alignment.alignment_sha256,
                "row_count": len(target_positions),
                "child_rows_sha256": _tensor_sha256(selected_child),
                "parent_rows_sha256": _tensor_sha256(selected_parent),
            }
        )
    child_matrix = torch.cat(tuple(child_rows), dim=0).contiguous()
    parent_matrix = torch.cat(tuple(parent_rows), dim=0).contiguous()
    if (
        int(child_matrix.shape[0]) != int(parent_matrix.shape[0])
        or int(child_matrix.shape[0]) <= 0
        or not bool(torch.isfinite(child_matrix).all().item())
        or not bool(torch.isfinite(parent_matrix).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global FIT residual matrices are incompatible or nonfinite."
        )
    body = {
        "schema": "neural-forge-lfm2-body-global-fit-residual-rows-v1",
        "layer_index": layer_index,
        "residual_chart_receipt_sha256": verified_chart["result_sha256"],
        "fit_batch_indices": list(fit_batches),
        "row_manifest": row_manifest,
        "fit_row_count": int(child_matrix.shape[0]),
        "child_width": int(child_matrix.shape[1]),
        "parent_width": int(parent_matrix.shape[1]),
        "fit_child_residual_rows_sha256": _tensor_sha256(child_matrix),
        "fit_parent_residual_rows_sha256": _tensor_sha256(parent_matrix),
        "trajectory_balanced_rows": True,
        "fit_partition_only": True,
        "heldout_rows_used": 0,
        "training_steps": 0,
    }
    return child_matrix, parent_matrix, _sealed(body)


def _fit_layer_normalized_operator_input_decoder(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    parent_operator_norm_weight: Any,
    target_operator_norm_weight: Any,
    parent_norm_epsilon: float,
    target_norm_epsilon: float,
    chart: Any,
    chart_receipt: Mapping[str, Any],
    normalized_input_correction_rho: float = 0.25,
    raw_structural_decoder: Any | None = None,
    raw_structural_decoder_sha256: str | None = None,
    coordinate_completion_receipt: Mapping[str, Any] | None = None,
    coordinate_completion_receipt_sha256: str | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Fit the shared FIT-only operator-input decoder once for one layer."""

    torch = _torch()
    verified_chart_receipt = _verify_sealed(chart_receipt, "residual chart")
    partition_value = verified_chart_receipt.get("partition_batch_indices")
    if (
        verified_chart_receipt.get("schema")
        != "neural-forge-lfm2-layer-stack-residual-chart-v1"
        or verified_chart_receipt.get("chart_sha256") != _tensor_sha256(chart)
        or not isinstance(partition_value, Mapping)
        or set(partition_value) != set(_PARTITIONS)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Normalized operator-input decoder requires the selected sealed "
            "residual-chart partition."
        )
    alignment_by_batch: dict[int, Lfm2SemanticSpanCpcfAlignment] = {}
    for alignment in alignments:
        batch_index = alignment.source_batch_index
        if batch_index in alignment_by_batch:
            raise Lfm2SemanticLayerStackRefoldError(
                "Normalized operator-input decoder received duplicate batches."
            )
        alignment_by_batch[batch_index] = alignment
    partition_batches: dict[str, tuple[int, ...]] = {}
    observed_batches: list[int] = []
    for partition in _PARTITIONS:
        supplied = partition_value.get(partition)
        if (
            not isinstance(supplied, list)
            or not supplied
            or any(
                isinstance(batch_index, bool)
                or not isinstance(batch_index, int)
                or batch_index < 0
                for batch_index in supplied
            )
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Residual-chart decoder partitions must contain batch indices."
            )
        partition_batches[partition] = tuple(supplied)
        observed_batches.extend(supplied)
    if (
        len(observed_batches) != len(set(observed_batches))
        or set(observed_batches) != set(alignment_by_batch)
        or set(parent) != set(alignment_by_batch)
        or set(child_bank.replay_by_batch) != set(alignment_by_batch)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Residual-chart decoder partitions do not cover the exact trajectories."
        )
    trajectory_batch_indices = tuple(observed_batches)
    fit_batch_indices = partition_batches["fit"]
    ordered_alignments = tuple(
        alignment_by_batch[batch_index] for batch_index in trajectory_batch_indices
    )
    parent_operator_rows = []
    child_operator_rows = []
    for alignment in ordered_alignments:
        batch_index = alignment.source_batch_index
        boundary = alignment.phases[-1].target_to_donor_boundary_positions
        target_positions = tuple(
            index for index, value in enumerate(boundary) if value is not None
        )
        donor_positions = tuple(int(value) for value in boundary if value is not None)
        parent_operator_rows.append(
            parent[batch_index]
            .final_histories["residual.pre"][0]
            .index_select(
                0,
                torch.tensor(donor_positions, dtype=torch.long),
            )
        )
        child_operator_rows.append(
            child_bank.replay_by_batch[batch_index]
            .operator_input[0]
            .index_select(
                0,
                torch.tensor(target_positions, dtype=torch.long),
            )
        )
    decoder, decoder_receipt = _fit_normalized_operator_input_decoder(
        parent_raw_rows=parent_operator_rows,
        child_normalized_rows=child_operator_rows,
        parent_operator_norm_weight=parent_operator_norm_weight,
        target_operator_norm_weight=target_operator_norm_weight,
        parent_norm_epsilon=parent_norm_epsilon,
        target_norm_epsilon=target_norm_epsilon,
        residual_chart=chart,
        residual_chart_sha256=verified_chart_receipt["result_sha256"],
        fit_trajectory_count=len(fit_batch_indices),
        normalized_input_correction_rho=normalized_input_correction_rho,
        raw_structural_decoder=raw_structural_decoder,
        raw_structural_decoder_sha256=raw_structural_decoder_sha256,
        coordinate_completion_receipt=coordinate_completion_receipt,
        coordinate_completion_receipt_sha256=(coordinate_completion_receipt_sha256),
    )
    decoder_body = dict(decoder_receipt)
    decoder_body.pop("decoder_sha256")
    decoder_body.update(
        {
            "trajectory_batch_indices": list(trajectory_batch_indices),
            "fit_batch_indices": list(fit_batch_indices),
            "heldout_batch_indices": list(
                trajectory_batch_indices[len(fit_batch_indices) :]
            ),
            "residual_chart_partition_batch_indices_sha256": canonical_sha256(
                {
                    partition: list(partition_batches[partition])
                    for partition in _PARTITIONS
                }
            ),
            "fit_batch_indices_authenticated_from_residual_chart": True,
        }
    )
    return decoder, MappingProxyType(
        {**decoder_body, "decoder_sha256": canonical_sha256(decoder_body)}
    )


def _build_attention_evidence(
    *,
    cache: Lfm2CircuitTraceCache,
    session: Any,
    bridge: StrictTokenizerBridgeArtifact,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    chart: Any,
    chart_receipt: Mapping[str, Any],
    layer_index: int,
    lineage: Lfm2SemanticCompilerLayerLineage,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float,
    parent_qk_tensors: Mapping[str, Any] | None = None,
    target_operator_norm_weight: Any | None = None,
    parent_norm_epsilon: float | None = None,
    target_norm_epsilon: float | None = None,
    normalized_input_correction_rho: float = 0.25,
    raw_structural_decoder: Any | None = None,
    raw_structural_decoder_sha256: str | None = None,
    coordinate_completion_receipt: Mapping[str, Any] | None = None,
    coordinate_completion_receipt_sha256: str | None = None,
) -> tuple[Lfm2SemanticAttentionTransportEvidence, ...]:
    torch = _torch()
    ordered_alignments = tuple(
        sorted(alignments, key=lambda value: value.source_batch_index)
    )
    query_groups = tuple(lineage.query_groups)
    query_group_weights = tuple(lineage.query_group_weights)
    key_value_groups = tuple(getattr(lineage, "key_value_groups", ()))
    key_value_group_weights = tuple(getattr(lineage, "key_value_group_weights", ()))
    complete_parent_prior_contract = (
        parent_qk_tensors is not None
        and target_operator_norm_weight is not None
        and parent_norm_epsilon is not None
        and target_norm_epsilon is not None
    )
    if complete_parent_prior_contract:
        normalized_input_decoder, normalized_input_decoder_receipt = (
            _fit_layer_normalized_operator_input_decoder(
                alignments=ordered_alignments,
                parent=parent,
                child_bank=child_bank,
                parent_operator_norm_weight=parent_qk_tensors["operator_norm"],
                target_operator_norm_weight=target_operator_norm_weight,
                parent_norm_epsilon=parent_norm_epsilon,
                target_norm_epsilon=target_norm_epsilon,
                chart=chart,
                chart_receipt=chart_receipt,
                normalized_input_correction_rho=(normalized_input_correction_rho),
                raw_structural_decoder=raw_structural_decoder,
                raw_structural_decoder_sha256=raw_structural_decoder_sha256,
                coordinate_completion_receipt=coordinate_completion_receipt,
                coordinate_completion_receipt_sha256=(
                    coordinate_completion_receipt_sha256
                ),
            )
        )
        (
            parent_query_weight_prior,
            parent_key_weight_prior,
            parent_value_weight_prior,
            parent_output_weight_prior,
            parent_query_norm_weight,
            parent_key_norm_weight,
            parent_weight_prior_receipt,
        ) = _fold_parent_qk_weight_priors(
            parent_query_weight=parent_qk_tensors["q"],
            parent_key_weight=parent_qk_tensors["k"],
            parent_value_weight=parent_qk_tensors["v"],
            parent_output_weight=parent_qk_tensors["o"],
            parent_query_norm_weight=parent_qk_tensors["q_norm"],
            parent_key_norm_weight=parent_qk_tensors["k_norm"],
            residual_chart=chart,
            query_groups=query_groups,
            query_group_weights=query_group_weights,
            key_value_groups=key_value_groups,
            key_value_group_weights=key_value_group_weights,
            head_dim=int(parent_qk_tensors["q_norm"].numel()),
            source_inventory_sha256=lineage.source_inventory.inventory_sha256,
            residual_chart_sha256=chart_receipt["result_sha256"],
            normalized_input_decoder=normalized_input_decoder,
            normalized_input_decoder_receipt=normalized_input_decoder_receipt,
        )
    else:
        # Narrow compatibility for typed-constructor test doubles.  Production
        # callers always supply the atomic parent Q/K/V/O and norm contract.
        parent_query_weight_prior = None
        parent_key_weight_prior = None
        parent_value_weight_prior = None
        parent_output_weight_prior = None
        parent_query_norm_weight = None
        parent_key_norm_weight = None
        parent_weight_prior_receipt = None
    result = []
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        phases = alignment.phase_order
        payload = session.load_selected_tensors(
            reference.index,
            _attention_tensor_names(
                reference,
                layer_index=layer_index,
                phase_order=phases,
            ),
        )
        try:
            tensors = payload.tensors
            prompt = alignment.phases[0].donor_full_prefix_tokens
            capability = inspect_lfm2_semantic_attention_cache_capability(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            capability.require_exact_history()
            history = assemble_lfm2_explicit_pre_rope_history(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            parent_query = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.q"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            parent_key = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.k"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            replay = child_bank.replay_by_batch[reference.index]
            pre_residual = child_bank.pre_residual_by_batch[reference.index]
            for phase_index, phase_alignment in enumerate(alignment.phases):
                donor_tokens = phase_alignment.donor_full_prefix_tokens
                target_tokens = phase_alignment.target_full_prefix_tokens
                phase_order = phases[: phase_index + 1]
                phase_capability = Lfm2SemanticAttentionCacheCapability(
                    layer_index=layer_index,
                    phase_order=phase_order,
                    donor_prompt_token_count=prompt,
                    selected_route="explicit_pre_rope",
                    prefill_captured_token_count=prompt,
                    assembled_current_row_count=donor_tokens,
                    missing_enhanced_tensor_names=(),
                    missing_legacy_tensor_names=(),
                    missing_donor_prefix_positions=(),
                    blocking_reasons=(),
                    exact_history_supported=True,
                )
                phase_history = SemanticAttentionPreRopeHistory(
                    query_normalized_pre_rope=(
                        history.query_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    key_normalized_pre_rope=(
                        history.key_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    value=history.value[:, :, :donor_tokens],
                    source_capability_sha256=phase_capability.capability_sha256,
                    source_tensor_names=tuple(
                        name
                        for name in history.source_tensor_names
                        if any(name.startswith(item + ".") for item in phase_order)
                    ),
                )
                transport = transport_lfm2_phase_probability_preserving_attention(
                    phase_alignment,
                    parent_query_post_rope=parent_query[:, :, :donor_tokens],
                    parent_key_post_rope=parent_key[:, :, :donor_tokens],
                    parent_value=history.value[:, :, :donor_tokens],
                    target_rope_inv_freq=target_rope_inv_freq,
                    target_rope_attention_scaling=(target_rope_attention_scaling),
                    query_mode="semantic_endpoints",
                )
                if hasattr(
                    transport,
                    "target_to_donor_span_owner_positions",
                ) and hasattr(phase_history, "history_sha256"):
                    (
                        folded_parent_query_pre_rope,
                        folded_parent_key_pre_rope,
                        parent_pre_rope_prior_receipt,
                    ) = fold_semantic_attention_parent_pre_rope_priors(
                        phase_history,
                        transport,
                        query_groups=query_groups,
                        query_group_weights=query_group_weights,
                        key_value_groups=key_value_groups,
                        key_value_group_weights=key_value_group_weights,
                    )
                else:  # Narrow compatibility for typed-constructor test doubles.
                    folded_parent_query_pre_rope = None
                    folded_parent_key_pre_rope = None
                    parent_pre_rope_prior_receipt = None
                query_positions = torch.tensor(
                    transport.query_target_positions,
                    dtype=torch.long,
                )
                parent_history = parent[reference.index].phase_histories[
                    phase_alignment.phase
                ]["residual.post_operator"]
                donor_positions = torch.tensor(
                    tuple(int(value) for value in transport.query_donor_positions),
                    dtype=torch.long,
                )
                if bool((donor_positions < 0).any().item()) or bool(
                    (donor_positions >= int(parent_history.shape[1])).any().item()
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        "Attention query endpoints exceed reconstructed parent history."
                    )
                parent_target = (
                    parent_history.index_select(1, donor_positions).to(torch.float64)
                    @ chart
                ).to(torch.float32)
                child_query = replay.operator_input[:, :target_tokens].index_select(
                    1,
                    query_positions,
                )
                child_key_value = replay.operator_input[:, :target_tokens]
                actual_pre = pre_residual[:, :target_tokens].index_select(
                    1,
                    query_positions,
                )
                weights = torch.full(
                    (1, len(transport.query_target_positions)),
                    1.0 / math.sqrt(len(alignment.phases)),
                    dtype=torch.float64,
                )
                endpoint = Lfm2SemanticAttentionReplayEndpoint(
                    batch_index=reference.index,
                    layer_index=layer_index,
                    alignment_sha256=phase_alignment.alignment_sha256,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    captured_child_model_state_sha256=(
                        child_bank.captured_child_model_state_sha256
                    ),
                    target_query_inputs=child_query,
                    target_key_value_inputs=child_key_value,
                    actual_child_pre_attention_residual=actual_pre,
                    parent_post_attention_residual_target_space=parent_target,
                    query_row_sqrt_weights=weights,
                )
                result.append(
                    Lfm2SemanticAttentionTransportEvidence(
                        batch_index=reference.index,
                        layer_index=layer_index,
                        alignment_sha256=phase_alignment.alignment_sha256,
                        alignment=phase_alignment,
                        capability=phase_capability.to_evidence_dict(),
                        history=phase_history.to_evidence_dict(),
                        actual_child_replay=endpoint,
                        source_inventory=lineage.source_inventory,
                        query_groups=query_groups,
                        query_group_weights=query_group_weights,
                        key_value_groups=key_value_groups,
                        key_value_group_weights=key_value_group_weights,
                        compiler_lineage_sha256=canonical_sha256(
                            {
                                "compiler_lineage_sha256": (
                                    lineage.compiler_lineage_sha256
                                ),
                                "residual_chart_sha256": chart_receipt["result_sha256"],
                                "replay_bank_sha256": child_bank.bank_sha256,
                            }
                        ),
                        transport=transport,
                        folded_parent_query_pre_rope=(folded_parent_query_pre_rope),
                        folded_parent_key_pre_rope=folded_parent_key_pre_rope,
                        parent_pre_rope_prior_receipt=(parent_pre_rope_prior_receipt),
                        parent_query_weight_prior=parent_query_weight_prior,
                        parent_key_weight_prior=parent_key_weight_prior,
                        parent_value_weight_prior=parent_value_weight_prior,
                        parent_output_weight_prior=parent_output_weight_prior,
                        parent_query_norm_weight=parent_query_norm_weight,
                        parent_key_norm_weight=parent_key_norm_weight,
                        parent_weight_prior_receipt=parent_weight_prior_receipt,
                    )
                )
        finally:
            session.release_selected_tensors(payload)
    return tuple(result)


def _is_zero_operation_audit(value: Any) -> bool:
    """Accept adapter-specific zero-operation supersets, never missing/nonzero data."""

    return bool(
        isinstance(value, Mapping)
        and all(
            value.get(name) == expected for name, expected in _ZERO_OPERATIONS.items()
        )
        and all(
            isinstance(name, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count == 0
            for name, count in value.items()
        )
    )


def _fit_ordering_flag_matches_bound_recipe(
    receipt: Mapping[str, Any],
    *,
    component: str,
) -> bool:
    """Authenticate FIT ordering for fit-guided and diagnostics elsewhere."""

    recipe_id = receipt.get("body_recipe_id")
    try:
        recipe = resolve_lfm2_semantic_body_recipe(str(recipe_id))
    except (TypeError, ValueError):
        return False
    return bool(
        receipt.get("body_recipe") == recipe.as_dict()
        and receipt.get("body_recipe_sha256") == recipe.recipe_sha256
        and receipt.get("recipe_component_lane")
        == recipe.component_lanes.get(component)
        and receipt.get("fit_metrics_used_for_candidate_ordering")
        is recipe.fit_metrics_order_candidates
    )


def _authenticated_heldout_incumbent_noop(
    receipt: Mapping[str, Any],
    *,
    component: str,
    before_sha256: str,
    after_sha256: str,
    changed_state_keys: Sequence[str],
) -> bool:
    """Recognize an authenticated incumbent control without requiring mutation."""

    legacy_heldout_noop = bool(
        receipt.get("status") == "heldout-selected-incumbent-noop"
        and receipt.get("incumbent_selected_by_heldout_gates") is True
        and receipt.get("fit_check_nonregression_passed") is True
        and receipt.get("selection_nonregression_passed") is True
        and receipt.get("validation_nonregression_passed") is True
    )
    deterministic_fit_noop = bool(
        receipt.get("status")
        == "deterministic-fit-correction-rounded-to-incumbent-noop"
        and receipt.get("incumbent_selected_by_heldout_gates") is False
        and receipt.get("heldout_metrics_are_diagnostic_only") is True
        and receipt.get("numeric_metric_used_for_publication_or_rollback") is False
    )
    structural_only_noop = bool(
        receipt.get("status") == _SHORT_CONV_STRUCTURAL_NOOP_STATUS
        and receipt.get("selected_candidate") == "incumbent-noop"
        and receipt.get("selected_candidate_state_identical_to_incumbent") is True
        and receipt.get("incumbent_noop_fallback_used") is True
        and receipt.get("incumbent_noop_fallback_reason")
        == "all-analytic-proposals-structurally-invalid-or-identical"
        and receipt.get("incumbent_selected_by_heldout_gates") is False
        and receipt.get("structurally_eligible_candidate_count") == 0
        and receipt.get("candidate_eligibility_basis")
        == "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        and receipt.get("structural_safety_is_only_candidate_veto") is True
        and _fit_ordering_flag_matches_bound_recipe(
            receipt,
            component=component,
        )
        and receipt.get("fit_metrics_used_for_structural_eligibility") is False
        and receipt.get("heldout_metrics_are_diagnostic_only") is True
        and receipt.get("validation_publication_veto_applied") is False
        and receipt.get("rollback_authorized") is False
        and receipt.get("numeric_metrics_have_final_intelligence_authority") is False
        and receipt.get("validation_rejected_candidate") is None
    )
    return bool(
        (legacy_heldout_noop or deterministic_fit_noop or structural_only_noop)
        and receipt.get("installed") is False
        and receipt.get("candidate_frozen_before_validation") is True
        and receipt.get("validation_triggered_refit") is False
        and receipt.get("model_state_before_sha256") == before_sha256
        and receipt.get("model_state_after_sha256") == after_sha256
        and before_sha256 == after_sha256
        and receipt.get(
            "changed_state_keys",
            receipt.get("changed_parameter_roles"),
        )
        == []
        and not changed_state_keys
        and receipt.get("parameter_count_changed") is False
        and receipt.get("topology_changed") is False
        and receipt.get("parameter_identities_preserved") is True
        and receipt.get("parameter_storage_preserved") is True
        and receipt.get("gradient_state_preserved") is True
        and receipt.get("exact_noop_state_proven") is True
        and (
            receipt.get("analytic_cached_evidence_only") is True
            or receipt.get("analytic_linear_and_cluster_solves_only") is True
        )
        and _is_zero_operation_audit(receipt.get("operation_audit"))
    )


def _authenticated_attention_incumbent_noop(
    receipt: Mapping[str, Any],
    *,
    before_sha256: str,
    after_sha256: str,
    changed_state_keys: Sequence[str],
) -> bool:
    """Accept only an attention no-op caused by an empty structural frontier."""

    return bool(
        receipt.get("status") == LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS
        and receipt.get("selected_candidate") == "incumbent-noop"
        and receipt.get("selected_candidate_state_identical_to_incumbent") is True
        and receipt.get("incumbent_noop_fallback_used") is True
        and receipt.get("incumbent_noop_fallback_reason")
        == "all-analytic-proposals-structurally-invalid-or-identical"
        and receipt.get("structurally_eligible_candidate_count") == 0
        and receipt.get("candidate_eligibility_basis")
        == _ATTENTION_CANDIDATE_ELIGIBILITY_BASIS
        and receipt.get("structural_safety_is_only_candidate_veto") is True
        and _fit_ordering_flag_matches_bound_recipe(receipt, component="qk")
        and receipt.get("fit_metrics_used_for_structural_eligibility") is False
        and receipt.get("incumbent_selected_by_heldout_gates") is False
        and receipt.get("heldout_metrics_are_diagnostic_only") is True
        and receipt.get("validation_publication_veto_applied") is False
        and receipt.get("rollback_authorized") is False
        and receipt.get("numeric_metrics_have_final_intelligence_authority") is False
        and receipt.get("validation_rejected_candidate") is None
        and receipt.get("numeric_metric_used_for_publication_or_rollback") is False
        and receipt.get("installed") is False
        and receipt.get("candidate_frozen_before_validation") is True
        and receipt.get("validation_triggered_refit") is False
        and receipt.get("model_state_before_sha256") == before_sha256
        and receipt.get("model_state_after_sha256") == after_sha256
        and before_sha256 == after_sha256
        and receipt.get("changed_state_keys") == []
        and not changed_state_keys
        and receipt.get("parameter_count_changed") is False
        and receipt.get("topology_changed") is False
        and receipt.get("parameter_identities_preserved") is True
        and receipt.get("parameter_storage_preserved") is True
        and receipt.get("gradient_state_preserved") is True
        and receipt.get("exact_noop_state_proven") is True
        and receipt.get("analytic_cached_evidence_only") is True
        and _is_zero_operation_audit(receipt.get("operation_audit"))
    )


def _authenticated_short_conv_incumbent_noop(
    receipt: Mapping[str, Any],
    *,
    before_sha256: str,
    after_sha256: str,
    changed_state_keys: Sequence[str],
) -> bool:
    """Require the structural no-candidate/no-effect reason for short convolution."""

    return bool(
        receipt.get("status") == _SHORT_CONV_STRUCTURAL_NOOP_STATUS
        and _authenticated_heldout_incumbent_noop(
            receipt,
            component="short_conv",
            before_sha256=before_sha256,
            after_sha256=after_sha256,
            changed_state_keys=changed_state_keys,
        )
    )


def _authenticated_mlp_unchanged_state(
    receipt: Mapping[str, Any],
    *,
    before_sha256: str,
    after_sha256: str,
    changed_state_keys: Sequence[str],
) -> bool:
    """Authenticate one MLP receipt that performed no local parameter copy."""

    return bool(
        receipt.get("installed") is False
        and receipt.get("candidate_frozen_before_validation") is True
        and receipt.get("validation_triggered_refit") is False
        and receipt.get("model_state_before_sha256") == before_sha256
        and receipt.get("model_state_after_sha256") == after_sha256
        and before_sha256 == after_sha256
        and receipt.get(
            "changed_state_keys",
            receipt.get("changed_parameter_roles"),
        )
        == []
        and not changed_state_keys
        and receipt.get("parameter_count_changed") is False
        and receipt.get("topology_changed") is False
        and receipt.get("parameter_identities_preserved") is True
        and receipt.get("parameter_storage_preserved") is True
        and receipt.get("gradient_state_preserved") is True
        and receipt.get("exact_noop_state_proven") is True
        and (
            receipt.get("analytic_linear_and_cluster_solves_only") is True
            or receipt.get("analytic_linear_cluster_tangent_and_qdeim_solves_only")
            is True
        )
        and _is_zero_operation_audit(receipt.get("operation_audit"))
    )


def _authenticated_mlp_incumbent_noop(
    receipt: Mapping[str, Any],
    *,
    before_sha256: str,
    after_sha256: str,
    changed_state_keys: Sequence[str],
) -> bool:
    """Require the MLP adapter's structural no-candidate/no-effect reason."""

    return bool(
        receipt.get("status") == _MLP_STRUCTURAL_NOOP_STATUS
        and receipt.get("selected_candidate") == "incumbent-noop"
        and receipt.get("selected_candidate_state_identical_to_incumbent") is True
        and receipt.get("incumbent_noop_fallback_used") is True
        and receipt.get("incumbent_noop_fallback_reason")
        == "all-analytic-proposals-structurally-invalid-or-identical"
        and receipt.get("incumbent_selected_by_heldout_gates") is False
        and receipt.get("structurally_eligible_candidate_count") == 0
        and receipt.get("candidate_eligibility_basis")
        == "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        and receipt.get("structural_safety_is_only_candidate_veto") is True
        and _fit_ordering_flag_matches_bound_recipe(receipt, component="mlp")
        and receipt.get("fit_metrics_used_for_structural_eligibility") is False
        and receipt.get("heldout_metrics_are_diagnostic_only") is True
        and receipt.get("validation_publication_veto_applied") is False
        and receipt.get("rollback_authorized") is False
        and receipt.get("numeric_metrics_have_final_intelligence_authority") is False
        and receipt.get("validation_rejected_candidate") is None
        and _authenticated_mlp_unchanged_state(
            receipt,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
            changed_state_keys=changed_state_keys,
        )
    )


def _authenticated_final_mlp_analytic_deferral(
    receipt: Mapping[str, Any],
    *,
    before_sha256: str,
    after_sha256: str,
    changed_state_keys: Sequence[str],
) -> bool:
    """Allow a safe analytic candidate whose only local change is deferred W2."""

    eligible_count = receipt.get("structurally_eligible_candidate_count")
    return bool(
        receipt.get("status") == "prepared-final-w2-deferred"
        and receipt.get("selected_candidate")
        in {"incumbent-producer-w2", "full-w1-w3-w2"}
        and receipt.get("selected_candidate_state_identical_to_incumbent") is False
        and receipt.get("incumbent_noop_fallback_used") is False
        and receipt.get("incumbent_noop_fallback_reason") is None
        and receipt.get("incumbent_selected_by_heldout_gates") is False
        and type(eligible_count) is int
        and eligible_count > 0
        and receipt.get("candidate_eligibility_basis")
        == "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        and receipt.get("structural_safety_is_only_candidate_veto") is True
        and _fit_ordering_flag_matches_bound_recipe(receipt, component="mlp")
        and receipt.get("fit_metrics_used_for_structural_eligibility") is False
        and receipt.get("heldout_metrics_are_diagnostic_only") is True
        and receipt.get("validation_publication_veto_applied") is False
        and receipt.get("rollback_authorized") is False
        and receipt.get("numeric_metrics_have_final_intelligence_authority") is False
        and receipt.get("validation_rejected_candidate") is None
        and receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is True
        and receipt.get("downstream_terminal_replay_and_validation_required") is True
        and _authenticated_mlp_unchanged_state(
            receipt,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
            changed_state_keys=changed_state_keys,
        )
    )


def _install_attention_operator(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Lfm2SemanticAttentionTransportEvidence],
    cache_key: str,
    bridge_artifact_sha256: str,
    installer: Callable[..., Mapping[str, Any]],
    parameter_count: int,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> Mapping[str, Any]:
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    before_hashes = _state_tensor_hashes(target_model)
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    if any(
        item.actual_child_replay.captured_child_model_state_sha256 != before
        for item in evidence
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Attention evidence is stale before installation."
        )
    candidate = installer(
        target_model=target_model,
        layer_index=layer_index,
        evidence=tuple(evidence),
        cache_key=cache_key,
        bridge_artifact_sha256=bridge_artifact_sha256,
        body_recipe=recipe,
    )
    after_hashes = _state_tensor_hashes(target_model)
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    changed = _changed_state_keys(before_hashes, after_hashes)
    raw_receipt = _verify_sealed(candidate, "semantic attention")
    if raw_receipt.get("status") in {
        "heldout-selected-incumbent-noop",
        "deterministic-fit-correction-rounded-to-incumbent-noop",
    }:
        raise Lfm2SemanticLayerStackRefoldError(
            "Attention installer returned an obsolete metric-based incumbent no-op."
        )
    _require_attention_body_recipe_binding(
        raw_receipt,
        recipe,
        label="Attention installer",
    )
    if _authenticated_attention_incumbent_noop(
        raw_receipt,
        before_sha256=before,
        after_sha256=after,
        changed_state_keys=changed,
    ):
        analytic = raw_receipt.get("analytic_refold", {})
        if (
            raw_receipt.get("schema") != LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA
            or raw_receipt.get("layer_index") != layer_index
            or raw_receipt.get("install_count") != 0
            or raw_receipt.get("transport_evidence_sha256")
            != [item.evidence_sha256 for item in evidence]
            or raw_receipt.get("changed_state_keys") != []
            or raw_receipt.get("parameter_element_count_before") != parameter_count
            or raw_receipt.get("parameter_element_count_after") != parameter_count
            or raw_receipt.get("target_topology_sha256_before")
            != raw_receipt.get("target_topology_sha256_after")
            or analytic.get("cache_key") != cache_key
            or analytic.get("bridge_artifact_sha256") != bridge_artifact_sha256
            or analytic.get("analytically_refit_projection_roles") != []
            or analytic.get("fixed_inherited_projection_roles")
            != ["q_proj", "k_proj", "v_proj", "out_proj"]
            or analytic.get("complete_qkvo_bundle_installed_transactionally")
            is not False
            or analytic.get("complete_qkvo_bundle_transaction_verified") is not True
            or analytic.get(
                "authenticated_incumbent_qkvo_retained_without_installation"
            )
            is not True
            or analytic.get("projection_parameter_copy_calls") != 0
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Attention incumbent no-op receipt is stale or incomplete."
            )
        receipt = raw_receipt
    else:
        receipt = _validate_attention_installation_receipt(
            candidate,
            layer_index=layer_index,
            evidence=evidence,
            before_sha256=before,
            after_sha256=after,
            changed_keys=changed,
            parameter_count=parameter_count,
        )
    if receipt.get("schema") != LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA:
        raise Lfm2SemanticLayerStackRefoldError(
            "Attention operator returned the wrong receipt schema."
        )
    return MappingProxyType(receipt)


def _install_coupled_attention_group_qdeim_operator(
    *,
    target_model: Any,
    session: Any,
    layer_index: int,
    fit_batch_indices: Sequence[int],
    fit_batch_sample_weights: Mapping[int, float],
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_out_weight: Any,
    source_inventory: Any,
    source_query_norm_weight: Any,
    source_key_norm_weight: Any,
    source_norm_inventory: Any,
    normalized_input_decoder: Any,
    normalized_input_decoder_receipt: Mapping[str, Any],
    residual_output_chart: Any,
    residual_chart_receipt: Mapping[str, Any],
    direct_attention_source_receipt: Mapping[str, Any],
    cache_key: str,
    bridge_artifact_sha256: str,
    parameter_count: int,
    fold_compute_device: str = "cpu",
    body_recipe: str | Lfm2SemanticBodyRecipe,
    compiled_candidate_sink: dict[int, Lfm2CompiledAttentionGroupQdeim] | None = None,
    installer: Callable[..., Any] = (
        compile_install_lfm2_semantic_attention_group_qdeim
    ),
) -> Mapping[str, Any]:
    """Install one intact Q/K/V/O/norm candidate and bind it to the recipe."""

    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if (
        recipe.component_lanes["qk"] != _COUPLED_ATTENTION_QDEIM_LANE
        or recipe.component_lanes["value_output"] != _COUPLED_ATTENTION_QDEIM_LANE
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Coupled attention Q-DEIM requires one shared Q/K/V/O recipe lane."
        )
    if not callable(installer):
        raise TypeError("Coupled attention group-QDEIM installer must be callable.")
    if compiled_candidate_sink is not None and (
        not isinstance(compiled_candidate_sink, dict)
        or layer_index in compiled_candidate_sink
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Coupled attention compiled-candidate sink must be one empty mutable "
            "layer slot."
        )
    entry = _WholeModelSnapshot.capture(target_model)
    before_hashes = _state_tensor_hashes(target_model)
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    try:
        result = installer(
            target_model=target_model,
            session=session,
            layer_index=layer_index,
            fit_batch_indices=tuple(fit_batch_indices),
            fit_batch_sample_weights=dict(fit_batch_sample_weights),
            source_query_weight=source_query_weight,
            source_key_weight=source_key_weight,
            source_value_weight=source_value_weight,
            source_out_weight=source_out_weight,
            source_inventory=source_inventory,
            source_query_norm_weight=source_query_norm_weight,
            source_key_norm_weight=source_key_norm_weight,
            source_norm_inventory=source_norm_inventory,
            normalized_input_decoder=normalized_input_decoder,
            normalized_input_decoder_receipt=normalized_input_decoder_receipt,
            residual_output_chart=residual_output_chart,
            residual_chart_receipt=residual_chart_receipt,
            fold_compute_device=fold_compute_device,
        )
        if isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit):
            deficit = dict(result.report)
            raise Lfm2SemanticLayerStackRefoldError(
                "Coupled attention Q-DEIM has no authenticated dense candidate: "
                f"{deficit.get('deficit_kind', 'unknown-evidence-deficit')}."
            )
        if not isinstance(
            result,
            Lfm2SemanticAttentionGroupQdeimInstallationResult,
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Coupled attention Q-DEIM returned no typed installation result."
            )
        raw = _verify_sealed(result.receipt, "coupled attention Q-DEIM installation")
        adapter = _verify_sealed(
            result.prepared.candidate.report,
            "coupled attention Q-DEIM cache candidate",
        )
        compiler = _verify_sealed(
            result.prepared.candidate.compiled.report,
            "coupled attention Q-DEIM standalone compiler",
        )
        qdeim_compute_execution = _validated_qdeim_compute_execution(
            compiler.get("compute_execution"),
            requested_device=fold_compute_device,
            schema="neural-forge-lfm2-attention-qdeim-compute-v1",
            label=f"Layer {layer_index} coupled attention",
        )
        candidate_dtype = compiler.get("materialization", {}).get("candidate_dtype")
        if candidate_dtype not in {"bfloat16", "float32"}:
            raise Lfm2SemanticLayerStackRefoldError(
                "Coupled attention Q-DEIM returned an unsupported target dtype."
            )
        after_hashes = _state_tensor_hashes(target_model)
        after = fingerprint_lfm2_parent(target_model).identity_sha256
        changed = _changed_state_keys(before_hashes, after_hashes)
        prefix = f"model.layers.{layer_index}.self_attn."
        transaction_covered = tuple(
            sorted(
                prefix + role
                for role in (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "out_proj.weight",
                    "q_layernorm.weight",
                    "k_layernorm.weight",
                )
            )
        )
        required_projection_changes = {
            prefix + role
            for role in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
            )
        }
        expected_changed = tuple(raw.get("changed_state_keys", ()))
        if (
            raw.get("installed") is not True
            or raw.get("install_count") != 1
            or raw.get("layer_index") != layer_index
            or raw.get("model_state_before_sha256") != before
            or raw.get("model_state_after_sha256") != after
            or tuple(changed) != expected_changed
            or tuple(raw.get("transaction_covered_state_keys", ()))
            != transaction_covered
            or raw.get("transaction_covered_parameter_role_count") != 6
            or raw.get("all_six_roles_supplied_to_atomic_transaction") is not True
            or raw.get("six_role_transaction_coverage_proven") is not True
            or raw.get("prepared_expected_changed_state_keys") != list(expected_changed)
            or raw.get("prepared_expected_changed_state_keys_matched") is not True
            or raw.get("changed_state_key_count") != len(expected_changed)
            or raw.get("actual_changed_bytes_cover_four_to_six_roles") is not True
            or not 4 <= len(expected_changed) <= 6
            or not required_projection_changes.issubset(expected_changed)
            or any(name not in transaction_covered for name in expected_changed)
            or raw.get("parameter_count_before") != parameter_count
            or raw.get("parameter_count_after") != parameter_count
            or raw.get("parameter_count_changed") is not False
            or raw.get("target_topology_sha256_before")
            != raw.get("target_topology_sha256_after")
            or raw.get("topology_changed") is not False
            or raw.get("complete_six_parameter_bundle_installed_transactionally")
            is not True
            or raw.get("qkvo_installed_from_one_coupled_intact_group_candidate")
            is not True
            or raw.get("query_and_key_norms_installed_from_direct_parent_tensors")
            is not True
            or raw.get("split_qk_or_vo_fallback_used") is not False
            or raw.get("candidate_adapter_result_sha256")
            != adapter.get("result_sha256")
            or raw.get("candidate_compiler_result_sha256")
            != compiler.get("result_sha256")
            or raw.get("fold_compute_device") != fold_compute_device
            or raw.get("qdeim_compute_execution") != dict(qdeim_compute_execution)
            or raw.get("qdeim_compute_execution_sha256")
            != qdeim_compute_execution["receipt_sha256"]
            or adapter.get("fold_compute_device") != fold_compute_device
            or adapter.get("qdeim_compute_execution") != dict(qdeim_compute_execution)
            or raw.get("source_qkvo_inventory_sha256")
            != source_inventory.inventory_sha256
            or raw.get("source_norm_inventory_sha256")
            != source_norm_inventory.inventory_sha256
            or raw.get("cache_selective_session_sha256")
            != adapter.get("cache_selective_session_sha256")
            or adapter.get("fit_batch_indices") != list(fit_batch_indices)
            or adapter.get("heldout_tensor_payload_loads") != 0
            or raw.get("heldout_payload_used_for_compilation") is not False
            or raw.get("numeric_metrics_have_semantic_decision_authority") is not False
            or raw.get("binary_manual_acceptance_gate_required") is not False
            or raw.get("whole_decoded_behavior_is_primary_intelligence_evidence")
            is not True
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Coupled attention Q-DEIM installation lost its cache, recipe, "
                "six-tensor, or no-gating contract."
            )
        evidence_sha256 = [adapter["result_sha256"]]
        body = {
            "schema": (LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA),
            "status": "installed-coupled-intact-group-qdeim-attention",
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_component_lane": recipe.component_lanes["qk"],
            "recipe_qk_lane": recipe.component_lanes["qk"],
            "recipe_value_output_lane": recipe.component_lanes["value_output"],
            "layer_index": layer_index,
            "installed": True,
            "install_count": 1,
            "selected_candidate": "intact-gqa-group-qdeim-qkvo",
            "structurally_eligible_candidate_count": 1,
            "candidate_eligibility_basis": (
                "authenticated_dense_target_dtype_exact_six_parameter_topology"
            ),
            "candidate_materialization_dtype": candidate_dtype,
            "coupled_qkvo_single_candidate": True,
            "split_qk_or_vo_fallback_used": False,
            "legacy_split_attention_installer_called": False,
            "transport_evidence_sha256": evidence_sha256,
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "changed_state_keys": list(changed),
            "transaction_covered_state_keys": list(transaction_covered),
            "transaction_covered_parameter_role_count": 6,
            "all_six_roles_supplied_to_atomic_transaction": True,
            "six_role_transaction_coverage_proven": True,
            "actual_changed_state_key_count": len(changed),
            "actual_changed_bytes_cover_four_to_six_roles": True,
            "parameter_element_count_before": parameter_count,
            "parameter_element_count_after": parameter_count,
            "parameter_count_changed": False,
            "target_topology_sha256_before": raw["target_topology_sha256_before"],
            "target_topology_sha256_after": raw["target_topology_sha256_after"],
            "topology_changed": False,
            "parameter_identities_preserved": True,
            "parameter_storage_preserved": True,
            "gradient_state_preserved": True,
            "fit_batch_indices": list(fit_batch_indices),
            "fit_batch_sample_weights": [
                {
                    "batch_index": int(index),
                    "sample_weight": float(fit_batch_sample_weights[index]),
                }
                for index in fit_batch_indices
            ],
            "cache_key": cache_key,
            "bridge_artifact_sha256": bridge_artifact_sha256,
            "cache_selective_session_sha256": raw["cache_selective_session_sha256"],
            "same_selective_cache_session_as_layer_stack": True,
            "coupled_attention_installation_receipt": raw,
            "coupled_attention_candidate_receipt": adapter,
            "coupled_attention_compiler_receipt": compiler,
            "fold_compute_device": fold_compute_device,
            "qdeim_pod_and_pivot_compute_device": fold_compute_device,
            "qdeim_compute_execution": dict(qdeim_compute_execution),
            "qdeim_compute_execution_sha256": qdeim_compute_execution["receipt_sha256"],
            "requested_fold_compute_device_proven": True,
            "source_norm_inventory": source_norm_inventory.as_dict(),
            "direct_attention_source_receipt": dict(direct_attention_source_receipt),
            "normalized_input_decoder_receipt": dict(normalized_input_decoder_receipt),
            "residual_chart_receipt": dict(residual_chart_receipt),
            "complete_qkvo_bundle_installed_transactionally": True,
            "complete_qkvo_bundle_transaction_verified": True,
            "qk_norm_zero_escape_installed_transactionally": True,
            "analytic_cached_evidence_only": True,
            "heldout_metrics_are_diagnostic_only": True,
            "numeric_metric_used_for_candidate_selection": False,
            "numeric_metric_used_for_publication_or_rollback": False,
            "numeric_metrics_have_final_intelligence_authority": False,
            "binary_manual_threshold_has_iteration_veto_authority": False,
            "binary_manual_acceptance_gate_required": False,
            "intelligence_retention_target_is_objective_not_numeric_cutoff": True,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        receipt = _sealed(body)
        if compiled_candidate_sink is not None:
            compiled = result.prepared.candidate.compiled
            if not isinstance(compiled, Lfm2CompiledAttentionGroupQdeim):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Coupled attention installation lost its structural candidate."
                )
            compiled_candidate_sink[layer_index] = compiled
        return receipt
    except BaseException:
        entry.restore(target_model)
        raise


def _semantic_terminal_target_position(phase: Any) -> int:
    donor_terminal = phase.donor_full_prefix_tokens - 1
    positions = tuple(
        index
        for index, value in enumerate(phase.target_to_donor_boundary_positions)
        if value == donor_terminal
    )
    if len(positions) != 1 or positions[0] != phase.target_full_prefix_tokens - 1:
        raise Lfm2SemanticLayerStackRefoldError(
            "A semantic phase has no unique terminal parent-exact target boundary."
        )
    return positions[0]


def _build_attention_source_local_output_fit_cases(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    structural_replay: _LayerReplayBank,
    fit_batch_indices: Sequence[int],
    fit_batch_sample_weights: Mapping[int, float],
    layer_index: int,
) -> tuple[
    tuple[Lfm2AttentionGroupQdeimSourceLocalOutputFitCase, ...],
    Mapping[str, Any],
]:
    """Assemble exact FIT-only parent-local actions and live pre-O contexts."""

    torch = _torch()
    indices = tuple(fit_batch_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or set(fit_batch_sample_weights) != set(indices)
        or any(
            isinstance(fit_batch_sample_weights[index], bool)
            or not isinstance(fit_batch_sample_weights[index], (int, float))
            or not math.isfinite(float(fit_batch_sample_weights[index]))
            or float(fit_batch_sample_weights[index]) <= 0.0
            for index in indices
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention FIT batches and positive trajectory weights "
            "must be exact."
        )
    if structural_replay.layer_index != layer_index:
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention received a replay from another layer."
        )
    by_batch = {value.source_batch_index: value for value in alignments}
    if any(index not in by_batch for index in indices):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention FIT indices are absent from semantic alignment."
        )
    cases = []
    case_receipts = []
    for batch_index in indices:
        alignment = by_batch[batch_index]
        trajectory = parent.get(batch_index)
        replay = structural_replay.replay_by_batch.get(batch_index)
        if trajectory is None or replay is None:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Source-local attention lacks parent/replay batch {batch_index}."
            )
        final = alignment.phases[-1]
        boundary = tuple(final.target_to_donor_boundary_positions)
        target_positions = tuple(
            index
            for index, donor_position in enumerate(boundary)
            if donor_position is not None
        )
        donor_positions = tuple(
            int(donor_position)
            for donor_position in boundary
            if donor_position is not None
        )
        if (
            not target_positions
            or donor_positions != tuple(range(final.donor_full_prefix_tokens))
            or target_positions[-1] != _semantic_terminal_target_position(final)
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Source-local attention batch {batch_index} does not expose every "
                "exact parent boundary once."
            )
        parent_pre_history = trajectory.final_histories.get("residual.pre")
        parent_post_history = trajectory.final_histories.get("residual.post_operator")
        if (
            not isinstance(parent_pre_history, torch.Tensor)
            or not isinstance(parent_post_history, torch.Tensor)
            or parent_pre_history.ndim != 3
            or tuple(parent_post_history.shape) != tuple(parent_pre_history.shape)
            or int(parent_pre_history.shape[0]) != 1
            or int(parent_pre_history.shape[1]) < len(donor_positions)
            or replay.operator_input.ndim != 3
            or replay.operator_output_input.ndim != 3
            or int(replay.operator_input.shape[0]) != 1
            or int(replay.operator_output_input.shape[0]) != 1
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Source-local attention batch {batch_index} has invalid cached "
                "parent or live child histories."
            )
        donor_index = torch.tensor(donor_positions, dtype=torch.long)
        target_index = torch.tensor(target_positions, dtype=torch.long)
        parent_pre = parent_pre_history.index_select(1, donor_index)[0]
        parent_post = parent_post_history.index_select(1, donor_index)[0]
        child_pre = replay.operator_input.index_select(1, target_index)[0]
        child_context = replay.operator_output_input.index_select(1, target_index)[0]
        row_count = len(target_positions)
        row_weight = float(fit_batch_sample_weights[batch_index]) / row_count
        aligned_row_identity_sha256 = canonical_sha256(
            {
                "schema": (
                    "neural-forge-lfm2-source-local-attention-aligned-fit-rows-v1"
                ),
                "partition": "fit",
                "layer_index": layer_index,
                "batch_index": batch_index,
                "alignment_sha256": alignment.alignment_sha256,
                "target_positions": list(target_positions),
                "donor_positions": list(donor_positions),
                "row_count": row_count,
            }
        )
        case = Lfm2AttentionGroupQdeimSourceLocalOutputFitCase(
            case_id=f"layer-{layer_index:03d}.batch-{batch_index:04d}",
            parent_pre_operator_residual=parent_pre,
            parent_post_operator_residual=parent_post,
            live_child_pre_operator_residual=child_pre,
            live_child_pre_o_context=child_context,
            row_sample_weights=(row_weight,) * row_count,
            parent_trajectory_receipt_sha256=trajectory.evidence_sha256,
            live_child_replay_receipt_sha256=replay.replay_sha256,
            aligned_row_identity_sha256=aligned_row_identity_sha256,
        )
        cases.append(case)
        case_receipts.append(
            {
                "case_id": case.case_id,
                "batch_index": batch_index,
                "row_count": row_count,
                "total_row_mass": row_weight * row_count,
                "parent_trajectory_receipt_sha256": trajectory.evidence_sha256,
                "live_child_replay_receipt_sha256": replay.replay_sha256,
                "aligned_row_identity_sha256": aligned_row_identity_sha256,
            }
        )
    frozen = tuple(cases)
    fit_evidence_sha256 = (
        lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256(frozen)
    )
    receipt = _sealed(
        {
            "schema": (
                "neural-forge-lfm2-semantic-attention-source-local-fit-cases-v1"
            ),
            "layer_index": layer_index,
            "partition": "fit",
            "fit_batch_indices": list(indices),
            "heldout_batch_indices_used": [],
            "fit_case_count": len(frozen),
            "fit_row_count": sum(value["row_count"] for value in case_receipts),
            "fit_cases": case_receipts,
            "fit_evidence_sha256": fit_evidence_sha256,
            "structural_replay_bank_sha256": structural_replay.bank_sha256,
            "structural_replay_model_state_sha256": (
                structural_replay.captured_child_model_state_sha256
            ),
            "parent_forward_calls": 0,
            "target_model_forward_calls": 0,
            "training_steps": 0,
        }
    )
    return frozen, receipt


def _compose_source_local_attention_operator_receipt(
    *,
    structural_operator_receipt: Mapping[str, Any],
    source_local_installation: Lfm2SemanticAttentionSourceLocalOutputInstallation,
    fit_cases_receipt: Mapping[str, Any],
    structural_replay: _LayerReplayBank,
    post_source_local_replay: _LayerReplayBank,
    layer_index: int,
    parameter_count: int,
    body_recipe: str | Lfm2SemanticBodyRecipe,
) -> Mapping[str, Any]:
    """Bind structural QDEIM and its bounded O-only refinement as one operator."""

    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if not _source_local_attention_output_required(recipe):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention composition requires its authenticated lane."
        )
    structural = _verify_sealed(
        structural_operator_receipt,
        f"Layer {layer_index} structural coupled attention",
    )
    structural_auth = _authenticate_lfm2_semantic_attention_operator_receipt(
        structural_operator_receipt,
        layer_index=layer_index,
        body_recipe=recipe,
    )
    low = _verify_sealed(
        source_local_installation.receipt,
        f"Layer {layer_index} source-local O installation",
    )
    compiler = _verify_sealed(
        source_local_installation.compiled.report,
        f"Layer {layer_index} source-local O compiler",
    )
    fit = _verify_sealed(
        fit_cases_receipt,
        f"Layer {layer_index} source-local O FIT cases",
    )
    if (
        low.get("schema")
        != LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA
        or structural_replay.layer_index != layer_index
        or post_source_local_replay.layer_index != layer_index
        or structural_auth.get("model_state_after_sha256")
        != structural_replay.captured_child_model_state_sha256
        or structural_replay.captured_child_model_state_sha256
        != low.get("model_state_before_sha256")
        or low.get("model_state_after_sha256")
        != post_source_local_replay.captured_child_model_state_sha256
        or fit.get("schema")
        != "neural-forge-lfm2-semantic-attention-source-local-fit-cases-v1"
        or fit.get("layer_index") != layer_index
        or fit.get("partition") != "fit"
        or fit.get("heldout_batch_indices_used") != []
        or fit.get("structural_replay_bank_sha256") != structural_replay.bank_sha256
        or fit.get("structural_replay_model_state_sha256")
        != structural_replay.captured_child_model_state_sha256
        or fit.get("fit_evidence_sha256") != low.get("source_local_fit_evidence_sha256")
        or low.get("source_local_compiler_result_sha256")
        != compiler.get("result_sha256")
        or low.get("source_local_corrected_candidate_state_sha256")
        != compiler.get("authentication", {}).get(
            "source_local_corrected_candidate_state_sha256"
        )
        or low.get("only_out_projection_permitted_to_change") is not True
        or low.get("qkv_bit_identical_to_structural_candidate") is not True
        or low.get("qk_norms_unchanged") is not True
        or low.get("parameter_count_before") != parameter_count
        or low.get("parameter_count_after") != parameter_count
        or structural_auth.get("parameter_element_count") != parameter_count
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention structural, FIT, O-only, or replay state "
            "chain is incomplete."
        )
    output_key = f"model.layers.{layer_index}.self_attn.out_proj.weight"
    low_changed_keys = tuple(low.get("changed_state_keys", ()))
    if low_changed_keys not in {(), (output_key,)}:
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention second stage changed more than O."
        )
    structural_changed_keys = tuple(structural_auth["changed_state_keys"])
    live_before = low.get("attention_tensor_sha256_before")
    live_after = low.get("attention_tensor_sha256_after")
    structural_bundle = structural_auth.get("six_tensor_bundle")
    if not all(
        isinstance(value, Mapping)
        for value in (live_before, live_after, structural_bundle)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention omitted structural or final tensor hashes."
        )
    assert isinstance(live_before, Mapping)
    assert isinstance(live_after, Mapping)
    assert isinstance(structural_bundle, Mapping)
    for role in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    ):
        if live_before.get(role) != structural_bundle.get(role) or live_after.get(
            role
        ) != structural_bundle.get(role):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Source-local attention changed structural role {role}."
            )
    if live_before.get("out_proj.weight") != structural_bundle.get("out_proj.weight"):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local attention O prior differs from structural QDEIM."
        )
    final_bundle = {
        **dict(structural_bundle),
        "out_proj.weight": live_after["out_proj.weight"],
    }
    qk_bundle_sha256 = canonical_sha256(
        {
            "q": final_bundle["q_proj.weight"],
            "k": final_bundle["k_proj.weight"],
        }
    )
    qkvo_bundle_sha256 = canonical_sha256(
        {
            "q": final_bundle["q_proj.weight"],
            "k": final_bundle["k_proj.weight"],
            "v": final_bundle["v_proj.weight"],
            "o": final_bundle["out_proj.weight"],
        }
    )
    structural_transport = list(structural_auth["transport_evidence_sha256"])
    transport = [*structural_transport, fit["fit_evidence_sha256"]]
    body = {
        "schema": _SOURCE_LOCAL_ATTENTION_COMPOSITE_SCHEMA,
        "status": "installed-coupled-qdeim-plus-bounded-source-local-o",
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "recipe_component_lane": recipe.component_lanes["qk"],
        "recipe_qk_lane": recipe.component_lanes["qk"],
        "recipe_value_output_lane": recipe.component_lanes["value_output"],
        "source_local_attention_output_lane": recipe.component_lanes[
            "attention_output_residual"
        ],
        "layer_index": layer_index,
        "installed": True,
        "install_count": 1,
        "physical_parameter_transaction_count": 2,
        "selected_candidate": "intact-gqa-qdeim-plus-source-local-o",
        "coupled_attention_group_qdeim": True,
        "coupled_qkvo_single_candidate": False,
        "structural_coupled_qkvo_candidate_installed": True,
        "source_local_attention_output": True,
        "source_local_o_only_second_stage": True,
        "split_qk_or_vo_fallback_used": False,
        "legacy_split_attention_installer_called": False,
        "model_state_before_sha256": structural_auth["model_state_before_sha256"],
        "model_state_intermediate_sha256": (
            structural_replay.captured_child_model_state_sha256
        ),
        "model_state_after_sha256": (
            post_source_local_replay.captured_child_model_state_sha256
        ),
        "changed_state_keys": list(
            dict.fromkeys((*structural_changed_keys, *low_changed_keys))
        ),
        "source_local_second_stage_changed_state_keys": list(low_changed_keys),
        "parameter_element_count_before": parameter_count,
        "parameter_element_count_after": parameter_count,
        "parameter_count_changed": False,
        "target_topology_sha256_before": structural["target_topology_sha256_before"],
        "target_topology_sha256_after": low["target_topology_sha256_after"],
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "transport_evidence_sha256": transport,
        "structural_transport_evidence_sha256": structural_transport,
        "source_local_fit_evidence_sha256": fit["fit_evidence_sha256"],
        "six_tensor_bundle": final_bundle,
        "six_tensor_bundle_sha256": canonical_sha256(final_bundle),
        "qk_bundle_sha256": qk_bundle_sha256,
        "qkvo_bundle_sha256": qkvo_bundle_sha256,
        "structural_attention_operator_receipt": dict(structural_operator_receipt),
        "source_local_output_installation_receipt": dict(
            source_local_installation.receipt
        ),
        "source_local_output_compiler_receipt": dict(
            source_local_installation.compiled.report
        ),
        "source_local_output_fit_cases_receipt": dict(fit_cases_receipt),
        "structural_replay_bank_sha256": structural_replay.bank_sha256,
        "post_source_local_o_replay_bank_sha256": (
            post_source_local_replay.bank_sha256
        ),
        "structural_replay_model_state_sha256": (
            structural_replay.captured_child_model_state_sha256
        ),
        "post_source_local_o_replay_model_state_sha256": (
            post_source_local_replay.captured_child_model_state_sha256
        ),
        "fresh_post_o_replay_required_before_mlp": True,
        "source_local_o_fit_partition_only": True,
        "heldout_payload_used_for_source_local_compilation": False,
        "fold_compute_device": structural["fold_compute_device"],
        "qdeim_pod_and_pivot_compute_device": structural[
            "qdeim_pod_and_pivot_compute_device"
        ],
        "qdeim_compute_execution": structural["qdeim_compute_execution"],
        "qdeim_compute_execution_sha256": structural["qdeim_compute_execution_sha256"],
        "requested_fold_compute_device_proven": True,
        "cache_key": structural["cache_key"],
        "bridge_artifact_sha256": structural["bridge_artifact_sha256"],
        "cache_selective_session_sha256": structural["cache_selective_session_sha256"],
        "source_norm_inventory": structural["source_norm_inventory"],
        "direct_attention_source_receipt": structural[
            "direct_attention_source_receipt"
        ],
        "normalized_input_decoder_receipt": structural[
            "normalized_input_decoder_receipt"
        ],
        "residual_chart_receipt": structural["residual_chart_receipt"],
        "analytic_cached_evidence_only": True,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metric_used_for_candidate_selection": False,
        "numeric_metric_used_for_publication_or_rollback": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "binary_manual_acceptance_gate_required": False,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return _sealed(body)


def _build_short_conv_source_local_writer_fit_cases(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    structural_replay: _LayerReplayBank,
    fit_batch_indices: Sequence[int],
    fit_batch_sample_weights: Mapping[int, float],
    layer_index: int,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Assemble FIT-only short-conv local actions and live writer contexts."""

    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        Lfm2SemanticShortConvSourceLocalWriterFitCase,
        lfm2_semantic_short_conv_source_local_writer_fit_evidence_sha256,
    )

    torch = _torch()
    indices = tuple(fit_batch_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or set(fit_batch_sample_weights) != set(indices)
        or any(
            isinstance(fit_batch_sample_weights[index], bool)
            or not isinstance(fit_batch_sample_weights[index], (int, float))
            or not math.isfinite(float(fit_batch_sample_weights[index]))
            or float(fit_batch_sample_weights[index]) <= 0.0
            for index in indices
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv FIT batches and positive trajectory "
            "weights must be exact."
        )
    if structural_replay.layer_index != layer_index:
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv received a replay from another layer."
        )
    by_batch = {value.source_batch_index: value for value in alignments}
    if any(index not in by_batch for index in indices):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv FIT indices are absent from semantic alignment."
        )
    cases = []
    case_receipts = []
    for batch_index in indices:
        alignment = by_batch[batch_index]
        trajectory = parent.get(batch_index)
        replay = structural_replay.replay_by_batch.get(batch_index)
        if trajectory is None or replay is None:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Source-local short-conv lacks parent/replay batch {batch_index}."
            )
        final = alignment.phases[-1]
        boundary = tuple(final.target_to_donor_boundary_positions)
        target_positions = tuple(
            index
            for index, donor_position in enumerate(boundary)
            if donor_position is not None
        )
        donor_positions = tuple(
            int(donor_position)
            for donor_position in boundary
            if donor_position is not None
        )
        if (
            not target_positions
            or donor_positions != tuple(range(final.donor_full_prefix_tokens))
            or target_positions[-1] != _semantic_terminal_target_position(final)
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Source-local short-conv batch {batch_index} does not expose "
                "every exact parent boundary once."
            )
        parent_pre_history = trajectory.final_histories.get("residual.pre")
        parent_post_history = trajectory.final_histories.get("residual.post_operator")
        if (
            not isinstance(parent_pre_history, torch.Tensor)
            or not isinstance(parent_post_history, torch.Tensor)
            or parent_pre_history.ndim != 3
            or tuple(parent_post_history.shape) != tuple(parent_pre_history.shape)
            or int(parent_pre_history.shape[0]) != 1
            or int(parent_pre_history.shape[1]) < len(donor_positions)
            or replay.operator_input.ndim != 3
            or replay.operator_output_input.ndim != 3
            or int(replay.operator_input.shape[0]) != 1
            or int(replay.operator_output_input.shape[0]) != 1
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Source-local short-conv batch {batch_index} has invalid "
                "cached parent or live child histories."
            )
        donor_index = torch.tensor(donor_positions, dtype=torch.long)
        target_index = torch.tensor(target_positions, dtype=torch.long)
        parent_pre = parent_pre_history.index_select(1, donor_index)[0]
        parent_post = parent_post_history.index_select(1, donor_index)[0]
        child_pre = replay.operator_input.index_select(1, target_index)[0]
        writer_design = replay.operator_output_input.index_select(
            1,
            target_index,
        )[0]
        row_count = len(target_positions)
        row_weight = float(fit_batch_sample_weights[batch_index]) / row_count
        aligned_row_identity_sha256 = canonical_sha256(
            {
                "schema": (
                    "neural-forge-lfm2-source-local-short-conv-aligned-fit-rows-v1"
                ),
                "partition": "fit",
                "layer_index": layer_index,
                "batch_index": batch_index,
                "alignment_sha256": alignment.alignment_sha256,
                "target_positions": list(target_positions),
                "donor_positions": list(donor_positions),
                "row_count": row_count,
            }
        )
        case = Lfm2SemanticShortConvSourceLocalWriterFitCase(
            case_id=f"layer-{layer_index:03d}.batch-{batch_index:04d}",
            parent_pre_operator_residual=parent_pre,
            parent_post_operator_residual=parent_post,
            live_child_pre_operator_residual=child_pre,
            live_child_writer_design=writer_design,
            row_sample_weights=(row_weight,) * row_count,
            parent_trajectory_receipt_sha256=trajectory.evidence_sha256,
            live_child_replay_receipt_sha256=replay.replay_sha256,
            aligned_row_identity_sha256=aligned_row_identity_sha256,
        )
        cases.append(case)
        case_receipts.append(
            {
                "case_id": case.case_id,
                "batch_index": batch_index,
                "row_count": row_count,
                "total_row_mass": row_weight * row_count,
                "parent_trajectory_receipt_sha256": trajectory.evidence_sha256,
                "live_child_replay_receipt_sha256": replay.replay_sha256,
                "aligned_row_identity_sha256": aligned_row_identity_sha256,
            }
        )
    frozen = tuple(cases)
    fit_evidence_sha256 = (
        lfm2_semantic_short_conv_source_local_writer_fit_evidence_sha256(frozen)
    )
    receipt = _sealed(
        {
            "schema": _SOURCE_LOCAL_SHORT_CONV_FIT_CASES_SCHEMA,
            "layer_index": layer_index,
            "partition": "fit",
            "fit_batch_indices": list(indices),
            "heldout_batch_indices_used": [],
            "fit_case_count": len(frozen),
            "fit_row_count": sum(value["row_count"] for value in case_receipts),
            "fit_cases": case_receipts,
            "fit_evidence_sha256": fit_evidence_sha256,
            "structural_replay_bank_sha256": structural_replay.bank_sha256,
            "structural_replay_model_state_sha256": (
                structural_replay.captured_child_model_state_sha256
            ),
            "parent_forward_calls": 0,
            "target_model_forward_calls": 0,
            "training_steps": 0,
        }
    )
    return frozen, receipt


def _compile_install_short_conv_source_local_writer(
    *,
    target_model: Any,
    layer_index: int,
    structural_operator_receipt: Mapping[str, Any],
    structural_replay: _LayerReplayBank,
    fit_cases: Sequence[Any],
    fit_cases_receipt: Mapping[str, Any],
    residual_output_chart: Any,
    parameter_count: int,
    fold_compute_device: str,
    body_recipe: str | Lfm2SemanticBodyRecipe,
) -> tuple[Any, Mapping[str, Any]]:
    """Compile and transactionally publish one bounded short-conv writer delta."""

    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        LFM2_SEMANTIC_SHORT_CONV_SOURCE_LOCAL_WRITER_SCHEMA,
        compile_lfm2_semantic_short_conv_source_local_writer,
        lfm2_semantic_short_conv_source_local_writer_fit_evidence_sha256,
        lfm2_semantic_short_conv_source_local_writer_state_sha256,
    )

    torch = _torch()
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if not _source_local_short_conv_output_required(recipe):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv installation requires its authenticated lane."
        )
    structural = _verify_sealed(
        structural_operator_receipt,
        f"Layer {layer_index} structural short-conv",
    )
    fit = _verify_sealed(
        fit_cases_receipt,
        f"Layer {layer_index} source-local short-conv FIT cases",
    )
    if (
        structural_replay.layer_index != layer_index
        or structural.get("layer_index") != layer_index
        or structural.get("model_state_after_sha256")
        != structural_replay.captured_child_model_state_sha256
        or fit.get("schema") != _SOURCE_LOCAL_SHORT_CONV_FIT_CASES_SCHEMA
        or fit.get("layer_index") != layer_index
        or fit.get("partition") != "fit"
        or fit.get("heldout_batch_indices_used") != []
        or fit.get("structural_replay_bank_sha256") != structural_replay.bank_sha256
        or fit.get("structural_replay_model_state_sha256")
        != structural_replay.captured_child_model_state_sha256
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv structural, FIT, or replay chain is stale."
        )
    backbone = getattr(target_model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or layer_index >= len(layers):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv target layer is absent."
        )
    layer = layers[layer_index]
    if bool(getattr(layer, "is_attention_layer", False)):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv cannot mutate an attention layer."
        )
    out_projection = getattr(getattr(layer, "conv", None), "out_proj", None)
    out_weight = getattr(out_projection, "weight", None)
    out_bias = getattr(out_projection, "bias", None)
    if not isinstance(out_weight, torch.nn.Parameter) or (
        out_bias is not None and not isinstance(out_bias, torch.nn.Parameter)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv requires an ordinary dense output writer."
        )
    structural_out_weight = out_weight.detach().to(device="cpu").contiguous().clone()
    structural_out_bias = (
        None
        if out_bias is None
        else out_bias.detach().to(device="cpu").contiguous().clone()
    )
    structural_writer_sha256 = (
        lfm2_semantic_short_conv_source_local_writer_state_sha256(
            structural_out_weight,
            structural_out_bias,
        )
    )
    fit_evidence_sha256 = (
        lfm2_semantic_short_conv_source_local_writer_fit_evidence_sha256(fit_cases)
    )
    if fit_evidence_sha256 != fit["fit_evidence_sha256"]:
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv FIT tensors changed before compilation."
        )
    compiled = compile_lfm2_semantic_short_conv_source_local_writer(
        structural_out_weight=structural_out_weight,
        structural_out_bias=structural_out_bias,
        expected_structural_writer_state_sha256=structural_writer_sha256,
        residual_output_chart=residual_output_chart,
        expected_residual_output_chart_sha256=_tensor_sha256(residual_output_chart),
        fit_cases=fit_cases,
        expected_fit_evidence_sha256=fit_evidence_sha256,
        fold_compute_device=fold_compute_device,
    )
    compiler = _verify_sealed(
        compiled.report,
        f"Layer {layer_index} source-local short-conv compiler",
    )
    corrected_writer_sha256 = lfm2_semantic_short_conv_source_local_writer_state_sha256(
        compiled.out_weight_source_local_corrected,
        compiled.out_bias_source_local_corrected,
    )
    authentication = compiler.get("authentication")
    if (
        compiler.get("schema") != LFM2_SEMANTIC_SHORT_CONV_SOURCE_LOCAL_WRITER_SCHEMA
        or not isinstance(authentication, Mapping)
        or authentication.get("structural_writer_state_sha256_before")
        != structural_writer_sha256
        or authentication.get("structural_writer_state_sha256_after")
        != structural_writer_sha256
        or authentication.get("fit_evidence_sha256_before") != fit_evidence_sha256
        or authentication.get("fit_evidence_sha256_after") != fit_evidence_sha256
        or authentication.get("corrected_writer_state_sha256")
        != corrected_writer_sha256
        or compiler.get("materialization", {}).get(
            "only_out_projection_writer_permitted_to_change"
        )
        is not True
        or compiler.get("target", {}).get(
            "absolute_or_cumulative_endpoint_target_used_for_correction"
        )
        is not False
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv compiler lost prior, FIT, or local-action "
            "authentication."
        )

    before_model_sha256 = fingerprint_lfm2_parent(target_model).identity_sha256
    before_state_hashes = _state_tensor_hashes(target_model)
    before_parameter_count = _parameter_count(target_model)
    parameters = {"out_proj.weight": out_weight}
    if out_bias is not None:
        parameters["out_proj.bias"] = out_bias
    before_ids = {name: id(value) for name, value in parameters.items()}
    before_storage = {name: int(value.data_ptr()) for name, value in parameters.items()}
    before_gradients = {
        name: None if value.grad is None else _tensor_sha256(value.grad)
        for name, value in parameters.items()
    }
    snapshots = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in parameters.items()
    }
    corrected = {
        "out_proj.weight": compiled.out_weight_source_local_corrected,
    }
    if compiled.out_bias_source_local_corrected is not None:
        corrected["out_proj.bias"] = compiled.out_bias_source_local_corrected
    if set(corrected) != set(parameters):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv compiler changed the structural bias topology."
        )
    key_prefix = f"model.layers.{layer_index}.conv.out_proj."
    expected_changed_keys = tuple(
        key_prefix + name.removeprefix("out_proj.")
        for name, value in parameters.items()
        if _tensor_sha256(value) != _tensor_sha256(corrected[name])
    )

    def transaction() -> None:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(
                    corrected[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

    try:
        _unused, guard = run_fold_only_callback(
            transaction,
            phase="lfm2-semantic-source-local-short-conv-writer-transaction",
            protected_models=(),
            freeze_parameters=False,
        )
        after_state_hashes = _state_tensor_hashes(target_model)
        changed_state_keys = _changed_state_keys(
            before_state_hashes,
            after_state_hashes,
        )
        after_model_sha256 = fingerprint_lfm2_parent(target_model).identity_sha256
        after_parameter_count = _parameter_count(target_model)
        after_gradients = {
            name: None if value.grad is None else _tensor_sha256(value.grad)
            for name, value in parameters.items()
        }
        installed_writer_sha256 = (
            lfm2_semantic_short_conv_source_local_writer_state_sha256(
                out_weight.detach(),
                None if out_bias is None else out_bias.detach(),
            )
        )
        if (
            set(changed_state_keys) != set(expected_changed_keys)
            or len(changed_state_keys) != len(expected_changed_keys)
            or installed_writer_sha256 != corrected_writer_sha256
            or {name: id(value) for name, value in parameters.items()} != before_ids
            or {name: int(value.data_ptr()) for name, value in parameters.items()}
            != before_storage
            or after_gradients != before_gradients
            or before_parameter_count != parameter_count
            or after_parameter_count != parameter_count
            or (
                bool(expected_changed_keys)
                and after_model_sha256 == before_model_sha256
            )
            or (not expected_changed_keys and after_model_sha256 != before_model_sha256)
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Source-local short-conv publication changed stale, unrelated, "
                "topology, identity, storage, or gradient state."
            )
    except BaseException as exc:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(
                    snapshots[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        if _state_tensor_hashes(target_model) != before_state_hashes:
            raise Lfm2SemanticLayerStackRefoldError(
                "Source-local short-conv transaction failed and exact rollback "
                "was not proven."
            ) from exc
        raise
    guard_body = guard.as_dict()
    receipt = _sealed(
        {
            "schema": (
                "neural-forge-lfm2-semantic-short-conv-source-local-writer-"
                "installation-v1"
            ),
            "status": (
                "installed-bounded-source-local-short-conv-writer"
                if expected_changed_keys
                else "authenticated-source-local-short-conv-writer-noop"
            ),
            "layer_index": layer_index,
            "installed": bool(expected_changed_keys),
            "install_count": int(bool(expected_changed_keys)),
            "structural_operator_result_sha256": structural["result_sha256"],
            "source_local_compiler_result_sha256": compiler["result_sha256"],
            "source_local_fit_evidence_sha256": fit_evidence_sha256,
            "structural_writer_state_sha256": structural_writer_sha256,
            "corrected_writer_state_sha256": corrected_writer_sha256,
            "model_state_before_sha256": before_model_sha256,
            "model_state_after_sha256": after_model_sha256,
            "changed_state_keys": list(changed_state_keys),
            "only_out_projection_writer_permitted_to_change": True,
            "parameter_count_before": before_parameter_count,
            "parameter_count_after": after_parameter_count,
            "parameter_count_changed": False,
            "parameter_identities_preserved": True,
            "parameter_storage_preserved": True,
            "gradient_state_preserved": True,
            "fit_partition_only": True,
            "heldout_rows_used": False,
            "fold_compute_device": fold_compute_device,
            "fold_only_transaction_guard": guard_body,
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "target_model_forward_calls": 0,
            "generation_calls": 0,
            "training_steps": 0,
            "backward_calls": guard_body["backward_calls"],
            "autograd_grad_calls": guard_body["autograd_grad_calls"],
            "optimizer_constructions": guard_body["optimizer_constructions"],
            "optimizer_steps": guard_body["optimizer_steps"],
            "pruning_operations": guard_body["pruning_operations"],
            "quantization_operations": guard_body["quantization_operations"],
            "sparsity_used": False,
            "parameter_sharing_used": False,
            "numeric_metrics_have_semantic_decision_authority": False,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        }
    )
    return compiled, receipt


def _compose_source_local_short_conv_operator_receipt(
    *,
    structural_operator_receipt: Mapping[str, Any],
    source_local_compiled: Any,
    source_local_installation_receipt: Mapping[str, Any],
    fit_cases_receipt: Mapping[str, Any],
    structural_replay: _LayerReplayBank,
    post_source_local_replay: _LayerReplayBank,
    layer_index: int,
    parameter_count: int,
    body_recipe: str | Lfm2SemanticBodyRecipe,
) -> Mapping[str, Any]:
    """Bind structural short-conv folding and its local writer as one operator."""

    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        LFM2_SEMANTIC_SHORT_CONV_SOURCE_LOCAL_WRITER_SCHEMA,
    )

    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if not _source_local_short_conv_output_required(recipe):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv composition requires its authenticated lane."
        )
    structural = _verify_sealed(
        structural_operator_receipt,
        f"Layer {layer_index} structural short-conv",
    )
    low = _verify_sealed(
        source_local_installation_receipt,
        f"Layer {layer_index} source-local short-conv installation",
    )
    compiler = _verify_sealed(
        source_local_compiled.report,
        f"Layer {layer_index} source-local short-conv compiler",
    )
    fit = _verify_sealed(
        fit_cases_receipt,
        f"Layer {layer_index} source-local short-conv FIT cases",
    )
    if (
        structural.get("schema") != _SHORT_CONV_INSTALLATION_SCHEMA
        or structural.get("layer_index") != layer_index
        or compiler.get("schema") != LFM2_SEMANTIC_SHORT_CONV_SOURCE_LOCAL_WRITER_SCHEMA
        or low.get("layer_index") != layer_index
        or fit.get("schema") != _SOURCE_LOCAL_SHORT_CONV_FIT_CASES_SCHEMA
        or fit.get("layer_index") != layer_index
        or structural_replay.layer_index != layer_index
        or post_source_local_replay.layer_index != layer_index
        or structural.get("model_state_after_sha256")
        != structural_replay.captured_child_model_state_sha256
        or structural_replay.captured_child_model_state_sha256
        != low.get("model_state_before_sha256")
        or low.get("model_state_after_sha256")
        != post_source_local_replay.captured_child_model_state_sha256
        or fit.get("structural_replay_bank_sha256") != structural_replay.bank_sha256
        or fit.get("structural_replay_model_state_sha256")
        != structural_replay.captured_child_model_state_sha256
        or low.get("source_local_compiler_result_sha256")
        != compiler.get("result_sha256")
        or low.get("source_local_fit_evidence_sha256") != fit.get("fit_evidence_sha256")
        or low.get("corrected_writer_state_sha256")
        != compiler.get("authentication", {}).get("corrected_writer_state_sha256")
        or low.get("only_out_projection_writer_permitted_to_change") is not True
        or low.get("parameter_count_before") != parameter_count
        or low.get("parameter_count_after") != parameter_count
        or structural.get("parameter_element_count") != parameter_count
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv structural, FIT, writer, or replay state "
            "chain is incomplete."
        )
    weight_key = f"model.layers.{layer_index}.conv.out_proj.weight"
    bias_key = f"model.layers.{layer_index}.conv.out_proj.bias"
    low_changed_keys = tuple(low.get("changed_state_keys", ()))
    if any(key not in {weight_key, bias_key} for key in low_changed_keys):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local short-conv second stage changed more than its writer."
        )
    structural_changed_keys = tuple(structural.get("changed_state_keys", ()))
    overall_installed = bool(
        structural["model_state_before_sha256"]
        != post_source_local_replay.captured_child_model_state_sha256
    )
    structural_transport = list(structural["transport_evidence_sha256"])
    body = dict(structural)
    body.pop("result_sha256")
    body.update(
        {
            "schema": _SOURCE_LOCAL_SHORT_CONV_COMPOSITE_SCHEMA,
            "status": (
                "installed-structural-short-conv-plus-bounded-source-local-writer"
            ),
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_component_lane": recipe.component_lanes["short_conv"],
            "source_local_short_conv_output_lane": recipe.component_lanes[
                "short_conv_output_residual"
            ],
            "layer_index": layer_index,
            "installed": overall_installed,
            "install_count": int(overall_installed),
            "physical_parameter_transaction_count": 2,
            "source_local_short_conv_output": True,
            "source_local_writer_only_second_stage": True,
            "model_state_before_sha256": structural["model_state_before_sha256"],
            "model_state_intermediate_sha256": (
                structural_replay.captured_child_model_state_sha256
            ),
            "model_state_after_sha256": (
                post_source_local_replay.captured_child_model_state_sha256
            ),
            "changed_state_keys": list(
                dict.fromkeys((*structural_changed_keys, *low_changed_keys))
            ),
            "source_local_second_stage_changed_state_keys": list(low_changed_keys),
            "parameter_element_count": parameter_count,
            "parameter_count_changed": False,
            "transport_evidence_sha256": [
                *structural_transport,
                fit["fit_evidence_sha256"],
            ],
            "structural_transport_evidence_sha256": structural_transport,
            "source_local_fit_evidence_sha256": fit["fit_evidence_sha256"],
            "structural_writer_state_sha256": low["structural_writer_state_sha256"],
            "corrected_writer_state_sha256": low["corrected_writer_state_sha256"],
            "structural_short_conv_operator_receipt": dict(structural_operator_receipt),
            "source_local_writer_installation_receipt": dict(
                source_local_installation_receipt
            ),
            "source_local_writer_compiler_receipt": dict(source_local_compiled.report),
            "source_local_writer_fit_cases_receipt": dict(fit_cases_receipt),
            "structural_replay_bank_sha256": structural_replay.bank_sha256,
            "post_source_local_writer_replay_bank_sha256": (
                post_source_local_replay.bank_sha256
            ),
            "structural_replay_model_state_sha256": (
                structural_replay.captured_child_model_state_sha256
            ),
            "post_source_local_writer_replay_model_state_sha256": (
                post_source_local_replay.captured_child_model_state_sha256
            ),
            "fresh_post_source_local_writer_replay_required_before_mlp": True,
            "source_local_writer_fit_partition_only": True,
            "heldout_payload_used_for_source_local_compilation": False,
            "source_local_writer_compute_execution": dict(
                compiler["compute_execution"]
            ),
            "analytic_cached_evidence_only": True,
            "numeric_metric_used_for_candidate_selection": False,
            "numeric_metrics_have_final_intelligence_authority": False,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
    )
    return _sealed(body)


def _validate_cached_fir(
    *,
    parent: Mapping[int, _ParentLayerTrajectory],
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    source_weight: Any,
    source_bias: Any | None,
    layer_index: int,
) -> Mapping[str, Any]:
    torch = _torch()
    raw_weight = source_weight.detach().to(device="cpu").contiguous()
    weight = raw_weight[:, 0, :] if raw_weight.ndim == 3 else raw_weight
    if weight.ndim != 2:
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FIR is not depthwise rank-2/3."
        )
    comparisons = []
    for alignment in alignments:
        trajectory = parent[alignment.source_batch_index]
        for phase in alignment.phases:
            rows = trajectory.phase_rows[phase.phase]
            cached_weight = rows["short_conv.fir_kernel"]
            cached_bias = rows["short_conv.fir_bias"]
            has_bias = bool(int(rows["short_conv.fir_has_bias"].reshape(-1)[0].item()))
            expected_bias = (
                torch.zeros_like(cached_bias)
                if source_bias is None
                else source_bias.detach().to(device="cpu").contiguous()
            )
            if (
                not torch.equal(cached_weight, weight)
                or has_bias != (source_bias is not None)
                or not torch.equal(cached_bias, expected_bias)
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Cached and direct-local parent FIR differ at layer {layer_index}."
                )
            comparisons.append(
                canonical_sha256(
                    {
                        "batch_index": alignment.source_batch_index,
                        "phase": phase.phase,
                        "weight_sha256": _tensor_sha256(cached_weight),
                        "bias_sha256": _tensor_sha256(cached_bias),
                        "has_bias": has_bias,
                    }
                )
            )
    body = {
        "schema": "neural-forge-lfm2-direct-cache-fir-equivalence-v1",
        "layer_index": layer_index,
        "direct_weight_sha256": _tensor_sha256(raw_weight),
        "direct_bias_sha256": (
            None if source_bias is None else _tensor_sha256(source_bias)
        ),
        "phase_comparison_sha256": comparisons,
        "all_cached_phases_exact": True,
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _build_short_conv_fhvr_source(
    *,
    parent_session: Any,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    layer_index: int,
    source_fir_weight: Any,
    source_fir_bias: Any | None,
    residual_chart: Any,
    residual_chart_receipt: Mapping[str, Any],
    raw_target_to_parent_decoder: Any | None = None,
    coordinate_completion_candidate_receipt: Mapping[str, Any] | None = None,
) -> Any:
    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        Lfm2SemanticShortConvFhvrSource,
    )

    identity = _verify_sealed(
        parent_session.identity_receipt(),
        "short-conv FHVR direct parent identity",
    )
    if (
        identity.get("exact_parent_identity_authenticated") is not True
        or identity.get("cache_parent_identity_sha256") != cache.parent_fingerprint
        or identity.get("reconstructed_parent_identity_sha256")
        != cache.parent_fingerprint
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "FHVR direct tensors are not bound to the cache parent identity."
        )
    prefix = f"model.layers.{layer_index}.conv."
    source = Lfm2SemanticShortConvFhvrSource(
        layer_index=layer_index,
        cache_key=cache.cache_key,
        bridge_artifact_sha256=bridge.artifact_sha256,
        source_model_identity_sha256=cache.parent_fingerprint,
        parent_session_open_receipt_sha256=parent_session.open_receipt_sha256,
        parent_identity_reconstruction_sha256=identity["result_sha256"],
        residual_chart_sha256=residual_chart_receipt["result_sha256"],
        norm_epsilon=parent_session.norm_epsilon,
        source_operator_norm_weight=parent_session.tensor(
            f"model.layers.{layer_index}.operator_norm.weight"
        ),
        source_in_proj_weight=parent_session.tensor(prefix + "in_proj.weight"),
        source_in_proj_bias=parent_session.optional_tensor(prefix + "in_proj.bias"),
        source_fir_weight=source_fir_weight,
        source_fir_bias=source_fir_bias,
        source_out_proj_weight=parent_session.tensor(prefix + "out_proj.weight"),
        source_out_proj_bias=parent_session.optional_tensor(prefix + "out_proj.bias"),
        residual_output_chart=residual_chart,
        raw_target_to_parent_decoder=raw_target_to_parent_decoder,
        coordinate_completion_candidate_receipt=(
            coordinate_completion_candidate_receipt
        ),
    )
    if not source.verify():
        raise Lfm2SemanticLayerStackRefoldError(
            "FHVR direct source failed final authentication."
        )
    return source


def _resolve_short_conv_adapter() -> tuple[type[Any], Callable[..., Mapping[str, Any]]]:
    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        Lfm2SemanticShortConvReplayEvidence,
        compile_install_lfm2_semantic_short_conv_postfold,
    )

    return (
        Lfm2SemanticShortConvReplayEvidence,
        compile_install_lfm2_semantic_short_conv_postfold,
    )


def _resolve_short_conv_construction(value: Any) -> Any:
    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        resolve_lfm2_semantic_short_conv_construction,
    )

    return resolve_lfm2_semantic_short_conv_construction(value)


def _build_short_conv_evidence(
    *,
    evidence_type: type[Any],
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    chart: Any,
    chart_receipt: Mapping[str, Any],
    layer_index: int,
    fhvr_source: Any | None = None,
) -> tuple[Any, ...]:
    torch = _torch()
    lineage = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-short-conv-semantic-lineage-v1",
            "cache_key": cache.cache_key,
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "layer_index": layer_index,
            "residual_chart_sha256": chart_receipt["result_sha256"],
            "replay_bank_sha256": child_bank.bank_sha256,
        }
    )
    result = []
    for alignment in alignments:
        batch = alignment.source_batch_index
        trajectory = parent[batch]
        replay = child_bank.replay_by_batch[batch]
        pre = child_bank.pre_residual_by_batch[batch]
        for phase_index, phase in enumerate(alignment.phases):
            target_tokens = phase.target_full_prefix_tokens
            terminal = _semantic_terminal_target_position(phase)
            rows = trajectory.phase_rows[phase.phase]
            histories = trajectory.phase_histories[phase.phase]
            target_dtype = replay.operator_input.dtype
            if target_dtype not in {torch.bfloat16, torch.float32}:
                raise Lfm2SemanticLayerStackRefoldError(
                    "Replay-fresh short-conv evidence has an unsupported target dtype."
                )
            parent_post = (
                rows["residual.post_operator"][:, -1:, :].to(torch.float64) @ chart
            ).to(target_dtype)
            history_receipt = canonical_sha256(
                {
                    "batch_index": batch,
                    "phase_index": phase_index,
                    "b_sha256": _tensor_sha256(histories["short_conv.b_history"]),
                    "c_sha256": _tensor_sha256(histories["short_conv.c_history"]),
                    "x_sha256": _tensor_sha256(histories["short_conv.x_history"]),
                    "owner_mapping_sha256": canonical_sha256(
                        list(phase.target_to_donor_span_owner_positions)
                    ),
                    "inserted_rows_parent_exact": False,
                    "semantic_boundary_endpoint_parent_exact": True,
                }
            )
            capability = canonical_sha256(
                {
                    "cache_manifest_sha256": cache.manifest["manifest_sha256"],
                    "parent_trajectory_sha256": trajectory.evidence_sha256,
                    "complete_parent_bcx_history": True,
                    "pure_expansion": (
                        phase.probability_preserving_attention_transport_supported
                    ),
                    "contractions_supported": False,
                }
            )
            future_observable_fields: dict[str, Any] = {}
            if fhvr_source is not None:
                if (
                    not callable(getattr(fhvr_source, "verify", None))
                    or not fhvr_source.verify()
                    or fhvr_source.layer_index != layer_index
                    or fhvr_source.cache_key != cache.cache_key
                    or fhvr_source.bridge_artifact_sha256 != bridge.artifact_sha256
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        "FHVR source changed before parent-input reconstruction."
                    )
                residual = histories["residual.pre"]
                parent_input = _reconstruct_parent_normalized_mlp_input(
                    residual.reshape(-1, int(residual.shape[-1])),
                    ffn_norm_weight=(fhvr_source.source_operator_norm_weight),
                    norm_epsilon=fhvr_source.norm_epsilon,
                    execution_device="cpu",
                ).reshape_as(residual)
                parent_input_provenance = canonical_sha256(
                    {
                        "schema": (
                            "neural-forge-lfm2-short-conv-parent-input-"
                            "reconstruction-v1"
                        ),
                        "fhvr_source_sha256": fhvr_source.source_sha256,
                        "parent_trajectory_sha256": trajectory.evidence_sha256,
                        "batch_index": batch,
                        "phase_index": phase_index,
                        "cached_residual_pre_sha256": _tensor_sha256(residual),
                        "parent_operator_input_sha256": _tensor_sha256(parent_input),
                        "rms_norm_formula": (
                            "weight*(x.float()*rsqrt(mean(x.float()^2)+norm_eps))"
                            ".to(bfloat16)"
                        ),
                        "parent_forward_calls": 0,
                    }
                )
                future_observable_fields = {
                    "parent_operator_input_history": parent_input,
                    "parent_operator_input_provenance_sha256": (
                        parent_input_provenance
                    ),
                }
            result.append(
                evidence_type(
                    batch_index=batch,
                    phase=phase_index,
                    layer_index=layer_index,
                    alignment_sha256=phase.alignment_sha256,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    source_capability_sha256=capability,
                    source_history_sha256=history_receipt,
                    compiler_lineage_sha256=lineage,
                    captured_child_model_state_sha256=(
                        child_bank.captured_child_model_state_sha256
                    ),
                    target_to_parent_span_owner_positions=(
                        phase.target_to_donor_span_owner_positions
                    ),
                    endpoint_target_positions=(terminal,),
                    actual_child_operator_input_history=(
                        replay.operator_input[:, :target_tokens]
                    ),
                    actual_child_pre_operator_residual_history=(pre[:, :target_tokens]),
                    parent_b_history=histories["short_conv.b_history"],
                    parent_c_history=histories["short_conv.c_history"],
                    parent_x_history=histories["short_conv.x_history"],
                    parent_post_operator_residual_target_space=parent_post,
                    endpoint_row_sqrt_weights=torch.full(
                        (1, 1),
                        1.0 / math.sqrt(len(alignment.phases)),
                        dtype=torch.float64,
                    ),
                    **future_observable_fields,
                )
            )
    return tuple(result)


def _install_short_conv_operator(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Any],
    source_fir_weight: Any,
    source_fir_bias: Any | None,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    installer: Callable[..., Mapping[str, Any]],
    parameter_count: int,
    fold_compute_device: str = "cpu",
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
    short_conv_construction: Any = "full-source-structural-v1",
    fhvr_source: Any | None = None,
) -> Mapping[str, Any]:
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    construction = _resolve_short_conv_construction(short_conv_construction)
    before_hashes = _state_tensor_hashes(target_model)
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    if any(item.captured_child_model_state_sha256 != before for item in evidence):
        raise Lfm2SemanticLayerStackRefoldError(
            "Short-conv evidence is stale before installation."
        )
    candidate = installer(
        target_model=target_model,
        layer_index=layer_index,
        evidence=tuple(evidence),
        source_fir_weight=source_fir_weight,
        source_fir_bias=source_fir_bias,
        cache_key=cache.cache_key,
        bridge_artifact_sha256=bridge.artifact_sha256,
        source_model_identity_sha256=cache.parent_fingerprint,
        body_recipe=recipe,
        **(
            {
                "short_conv_construction": construction,
                "fhvr_source": fhvr_source,
            }
            if (
                construction.uses_fhvr_full_coverage
                or construction.uses_body_global_coordinate_completion
            )
            else {}
        ),
        **(
            {"fold_compute_device": fold_compute_device}
            if fold_compute_device != "cpu"
            else {}
        ),
    )
    receipt = _verify_sealed(candidate, "short-conv operator")
    _require_body_recipe_binding(
        receipt,
        recipe,
        component="short_conv",
        label="Short-conv installer",
    )
    if construction.uses_fhvr_full_coverage:
        if (
            receipt.get("short_conv_construction") != construction.as_dict()
            or receipt.get("short_conv_construction_id") != construction.construction_id
            or receipt.get("short_conv_construction_sha256")
            != construction.construction_sha256
            or receipt.get("fhvr_source_sha256")
            != getattr(fhvr_source, "source_sha256", None)
            or receipt.get("full_history_future_observable_full_coverage_hybrid")
            is not True
            or receipt.get("requested_fhvr_compute_route_proved") is not True
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Short-conv installer dropped its authenticated FHVR construction."
            )
    elif construction.uses_body_global_coordinate_completion:
        expected_completion_sha256 = getattr(
            fhvr_source,
            "coordinate_completion_candidate_sha256",
            None,
        )
        if (
            receipt.get("short_conv_construction") != construction.as_dict()
            or receipt.get("short_conv_construction_id") != construction.construction_id
            or receipt.get("short_conv_construction_sha256")
            != construction.construction_sha256
            or receipt.get("coordinate_complete_direct_source_sha256")
            != getattr(fhvr_source, "source_sha256", None)
            or receipt.get("coordinate_completion_candidate_sha256")
            != expected_completion_sha256
            or receipt.get("shared_body_global_completion_sha256")
            != expected_completion_sha256
            or receipt.get("body_global_coordinate_complete_structural_fold")
            is not True
            or receipt.get("coordinate_complete_candidate_selected_by_construction")
            is not True
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Short-conv installer dropped its shared body-global coordinate "
                "completion."
            )
    elif any(
        key in receipt
        for key in (
            "short_conv_construction",
            "fhvr_source_sha256",
            "fhvr_candidate_state_sha256",
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Structural short-conv receipt contains an unrequested FHVR source."
        )
    if fold_compute_device != "cpu":
        structural_execution = receipt.get("structural_cluster_compute_execution")
        writer_execution = receipt.get("writer_compute_execution")
        if (
            receipt.get("fold_compute_device") != fold_compute_device
            or receipt.get("requested_cuda_short_conv_routes_proved") is not True
            or not isinstance(structural_execution, Mapping)
            or not isinstance(writer_execution, Mapping)
            or structural_execution.get("requested_device") != fold_compute_device
            or writer_execution.get("requested_device") != fold_compute_device
            or structural_execution.get("backend") != "cuda"
            or writer_execution.get("backend") != "cuda"
            or structural_execution.get("compute_dtype") != "float64"
            or writer_execution.get("compute_dtype") != "float64"
            or receipt.get("structural_cluster_compute_execution_sha256")
            != structural_execution.get("execution_sha256")
            or receipt.get("writer_compute_execution_sha256")
            != writer_execution.get("execution_sha256")
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Short-conv installer did not prove the requested CUDA fold backend."
            )
    after_hashes = _state_tensor_hashes(target_model)
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    changed = _changed_state_keys(before_hashes, after_hashes)
    prefix = f"model.layers.{layer_index}.conv."
    if _authenticated_short_conv_incumbent_noop(
        receipt,
        before_sha256=before,
        after_sha256=after,
        changed_state_keys=changed,
    ):
        if (
            receipt.get("schema") != _SHORT_CONV_INSTALLATION_SCHEMA
            or receipt.get("layer_index") != layer_index
            or receipt.get("cache_key") != cache.cache_key
            or receipt.get("bridge_artifact_sha256") != bridge.artifact_sha256
            or receipt.get("source_model_identity_sha256") != cache.parent_fingerprint
            or receipt.get("transport_evidence_sha256")
            != [item.evidence_sha256 for item in evidence]
            or receipt.get("changed_state_keys") != []
            or receipt.get("parameter_element_count") != parameter_count
            or _parameter_count(target_model) != parameter_count
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Short-conv incumbent no-op receipt is stale or incomplete."
            )
        return MappingProxyType(receipt)
    if (
        receipt.get("schema") != _SHORT_CONV_INSTALLATION_SCHEMA
        or receipt.get("installed") is not True
        or receipt.get("layer_index") != layer_index
        or receipt.get("cache_key") != cache.cache_key
        or receipt.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or receipt.get("source_model_identity_sha256") != cache.parent_fingerprint
        or receipt.get("model_state_before_sha256") != before
        or receipt.get("model_state_after_sha256") != after
        or before == after
        or receipt.get("parameter_element_count") != parameter_count
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("changed_state_keys") != list(changed)
        or not changed
        or any(
            not str(name).startswith(prefix)
            for name in receipt.get("changed_state_keys", ())
        )
        or receipt.get("parent_bcx_exact_only_at_semantic_span_boundaries") is not True
        or receipt.get("inserted_target_bcx_rows_are_owner_expanded_proxies")
        is not True
        or receipt.get("inserted_target_rows_parent_exact") is not False
        or receipt.get("inserted_target_rows_used_for_endpoint_fit") is not False
        or receipt.get("analytic_cached_evidence_only") is not True
        or _parameter_count(target_model) != parameter_count
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Short-conv installer returned an incomplete or over-broad receipt."
        )
    return MappingProxyType(receipt)


def _resolve_mlp_adapter() -> tuple[type[Any], type[Any], Callable[..., Any]]:
    from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (
        Lfm2DirectLocalMlpSourceInventory,
        Lfm2SemanticMlpEndpointEvidence,
        compile_install_lfm2_semantic_mlp_postfold,
    )

    return (
        Lfm2DirectLocalMlpSourceInventory,
        Lfm2SemanticMlpEndpointEvidence,
        compile_install_lfm2_semantic_mlp_postfold,
    )


def _mlp_lineage_sha256(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    layer_index: int,
    operator_kind: str,
    chart_receipt: Mapping[str, Any],
    attention_lineage: Lfm2SemanticCompilerLayerLineage | None,
) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-semantic-mlp-layer-lineage-v2",
            "cache_key": cache.cache_key,
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "cache_layer_topology_sha256": cache.layer_topology_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "layer_index": layer_index,
            "operator_kind": operator_kind,
            "attention_compiler_lineage_sha256": (
                None
                if attention_lineage is None
                else attention_lineage.compiler_lineage_sha256
            ),
            "fit_only_residual_assignment_sha256": chart_receipt["result_sha256"],
            "selected_residual_chart_sha256": chart_receipt["chart_sha256"],
        }
    )


def _strict_parent_bf16_matrix(value: Any, name: str, *, columns: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) != columns
        or value.dtype != torch.bfloat16
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{name} must be one detached finite BF16 matrix with width {columns}."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _reconstruct_parent_normalized_mlp_input(
    residual: Any,
    *,
    ffn_norm_weight: Any,
    norm_epsilon: float,
    execution_device: Any = "cpu",
) -> Any:
    """Reconstruct parent RMSNorm output from cached residual bytes, no forward."""

    torch = _torch()
    if (
        not isinstance(residual, torch.Tensor)
        or residual.ndim != 2
        or int(residual.shape[0]) <= 0
        or int(residual.shape[1]) <= 0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Cached parent post-operator residual must be one nonempty matrix."
        )
    source_hidden = int(residual.shape[1])
    replay_residual = _strict_parent_bf16_matrix(
        residual,
        "cached parent post-operator residual",
        columns=source_hidden,
    )
    replay_norm = _strict_parent_bf16_matrix(
        ffn_norm_weight.reshape(1, -1),
        "direct parent FFN RMSNorm weight",
        columns=source_hidden,
    )[0]
    if (
        isinstance(norm_epsilon, bool)
        or not isinstance(norm_epsilon, (int, float))
        or not math.isfinite(float(norm_epsilon))
        or float(norm_epsilon) <= 0.0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN normalization has no authenticated positive epsilon."
        )
    device = torch.device(execution_device)
    with torch.no_grad():
        residual_bf16 = replay_residual.to(device=device, dtype=torch.bfloat16)
        norm_bf16 = replay_norm.to(device=device, dtype=torch.bfloat16)
        residual_float = residual_bf16.to(torch.float32)
        variance = residual_float.square().mean(dim=-1, keepdim=True)
        normalized = norm_bf16 * (
            residual_float * torch.rsqrt(variance + float(norm_epsilon))
        ).to(torch.bfloat16)
    return normalized.detach().to(device="cpu").contiguous()


def _bf16_cross_kernel_equivalence_policy() -> Mapping[str, Any]:
    """Return the legacy arithmetic envelope as a diagnostic-only descriptor."""

    torch = _torch()
    machine_epsilon = float(torch.finfo(torch.bfloat16).eps)
    unit_roundoff = machine_epsilon / 2.0
    body = {
        "schema": "neural-forge-bf16-cross-kernel-equivalence-policy-v1",
        "arithmetic_dtype": "bfloat16",
        "machine_epsilon": machine_epsilon,
        "unit_roundoff": unit_roundoff,
        "elementwise_rtol": machine_epsilon,
        "elementwise_atol": unit_roundoff,
        "maximum_relative_rmse": machine_epsilon,
        "relative_rmse_reference": "cached_endpoint_rms",
        "near_zero_ulp_distance_used": False,
        "diagnostic_only": True,
        "numeric_admission_authority": False,
        "fold_selection_or_quality_objective": False,
    }
    return MappingProxyType(
        {
            **body,
            "policy_sha256": canonical_sha256(body),
        }
    )


def _bf16_cross_kernel_equivalence_metrics(
    *,
    replayed: Any,
    cached: Any,
    policy: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    """Measure backend drift without treating numeric equality as folding."""

    torch = _torch()
    if (
        not isinstance(replayed, torch.Tensor)
        or not isinstance(cached, torch.Tensor)
        or replayed.layout != torch.strided
        or cached.layout != torch.strided
        or replayed.device.type != "cpu"
        or cached.device.type != "cpu"
        or replayed.dtype != torch.bfloat16
        or cached.dtype != torch.bfloat16
        or replayed.shape != cached.shape
        or int(replayed.numel()) <= 0
        or not bool(torch.isfinite(cached).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{field_name} cross-kernel provenance tensors must be compatible "
            "finite CPU BF16 tensors."
        )
    replayed_float = replayed.to(torch.float64)
    cached_float = cached.to(torch.float64)
    replayed_finite = torch.isfinite(replayed_float)
    replayed_all_finite = bool(replayed_finite.all().item())
    nonfinite_replayed_element_count = int((~replayed_finite).sum().item())
    finite_replayed = torch.nan_to_num(
        replayed_float,
        nan=0.0,
        posinf=float(torch.finfo(torch.bfloat16).max),
        neginf=-float(torch.finfo(torch.bfloat16).max),
    )
    residual = finite_replayed - cached_float
    absolute = residual.abs()
    error_rms = float(torch.sqrt(residual.square().mean()).item())
    cached_rms = float(torch.sqrt(cached_float.square().mean()).item())
    denominator = max(cached_rms, float(torch.finfo(torch.float64).tiny))
    relative_rmse = min(error_rms / denominator, _MAX_FINITE_FLOAT)
    elementwise_close = bool(
        replayed_all_finite
        and torch.allclose(
            finite_replayed,
            cached_float,
            rtol=float(policy["elementwise_rtol"]),
            atol=float(policy["elementwise_atol"]),
        )
    )
    relative_rmse_close = bool(
        math.isfinite(relative_rmse)
        and relative_rmse <= float(policy["maximum_relative_rmse"])
    )
    exact_element_count = int(torch.eq(replayed, cached).sum().item())
    element_count = int(replayed.numel())
    cached_maximum_absolute_value = float(cached_float.abs().max().item())
    maximum_absolute_error = float(absolute.max().item())
    mean_absolute_error = float(absolute.mean().item())
    maximum_tolerance = min(
        float(policy["elementwise_atol"])
        + float(policy["elementwise_rtol"]) * cached_maximum_absolute_value,
        _MAX_FINITE_FLOAT,
    )
    diagnostics_finite = all(
        math.isfinite(value)
        for value in (
            maximum_absolute_error,
            mean_absolute_error,
            error_rms,
            cached_rms,
            relative_rmse,
            cached_maximum_absolute_value,
            maximum_tolerance,
        )
    )
    if not diagnostics_finite:
        raise Lfm2SemanticLayerStackRefoldError(
            f"{field_name} cross-kernel diagnostics are not finite."
        )
    legacy_envelope_passed = bool(elementwise_close and relative_rmse_close)
    body = {
        "field": field_name,
        "shape": [int(size) for size in replayed.shape],
        "element_count": element_count,
        "replayed_all_finite": replayed_all_finite,
        "nonfinite_replayed_element_count": nonfinite_replayed_element_count,
        "exact_element_count": exact_element_count,
        "mismatched_element_count": element_count - exact_element_count,
        "exact_element_fraction": exact_element_count / element_count,
        "maximum_absolute_error": maximum_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "error_rms": error_rms,
        "cached_endpoint_rms": cached_rms,
        "relative_rmse": relative_rmse,
        "cached_maximum_absolute_value": cached_maximum_absolute_value,
        "maximum_elementwise_tolerance_at_cached_maximum": maximum_tolerance,
        "elementwise_envelope_passed": elementwise_close,
        "relative_rmse_envelope_passed": relative_rmse_close,
        "bit_exact": exact_element_count == element_count,
        "within_legacy_bf16_envelope": legacy_envelope_passed,
        "diagnostics_finite": True,
        "diagnostic_only": True,
        "numeric_admission_authority": False,
    }
    return MappingProxyType(body)


def _replay_parent_mlp_functional_provenance(
    *,
    cache: Lfm2CircuitTraceCache,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    parent_session: _DirectParentSafetensorsSession,
    layer_index: int,
    gate_weight: Any,
    up_weight: Any,
    down_weight: Any,
    ffn_norm_weight: Any,
    execution_device: Any,
) -> Mapping[str, Any]:
    """Seal exact source identity plus diagnostic-only cached endpoint replay."""

    torch = _torch()
    source_intermediate, source_hidden = map(int, gate_weight.shape)
    gate = _strict_parent_bf16_matrix(
        gate_weight,
        "direct parent W1",
        columns=source_hidden,
    )
    up = _strict_parent_bf16_matrix(
        up_weight,
        "direct parent W3",
        columns=source_hidden,
    )
    down = _strict_parent_bf16_matrix(
        down_weight,
        "direct parent W2",
        columns=source_intermediate,
    )
    norm = _strict_parent_bf16_matrix(
        ffn_norm_weight.reshape(1, -1),
        "direct parent FFN RMSNorm weight",
        columns=source_hidden,
    )[0]
    if (
        tuple(up.shape) != tuple(gate.shape)
        or tuple(down.shape) != (source_hidden, source_intermediate)
        or tuple(norm.shape) != (source_hidden,)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN tensors do not form one compatible LFM2 layer."
        )
    epsilon = parent_session.norm_epsilon
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN provenance has no authenticated norm epsilon."
        )
    identity_receipt = _verify_sealed(
        parent_session.identity_receipt(),
        "direct parent exact identity reconstruction",
    )
    reconstructed_parent = identity_receipt.get("reconstructed_parent_identity")
    if (
        identity_receipt.get("exact_parent_identity_authenticated") is not True
        or identity_receipt.get("cache_parent_identity_sha256")
        != cache.parent_fingerprint
        or identity_receipt.get("reconstructed_parent_identity_sha256")
        != cache.parent_fingerprint
        or not isinstance(reconstructed_parent, Mapping)
        or any(
            reconstructed_parent.get(name) != cache.manifest["parent"].get(name)
            for name in (
                "model_class",
                "config_sha256",
                "state_dict_sha256",
                "state_tensor_count",
                "runtime_buffer_sha256",
                "runtime_buffer_count",
            )
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent MLP tensors are not backed by the cache-authenticated "
            "exact local parent identity."
        )

    residual_rows = []
    expected: dict[str, list[Any]] = {
        "mlp.input": [],
        "mlp.gate": [],
        "mlp.up": [],
        "mlp.post_swiglu": [],
        "residual.post_layer": [],
    }
    row_ids = []
    for alignment in sorted(
        alignments,
        key=lambda value: value.source_batch_index,
    ):
        trajectory = parent[alignment.source_batch_index]
        for phase_index, phase in enumerate(alignment.phases):
            rows = trajectory.phase_rows[phase.phase]
            residual = rows["residual.post_operator"][:, -1:, :].reshape(
                1,
                -1,
            )
            residual_rows.append(
                _strict_parent_bf16_matrix(
                    residual,
                    "cached parent post-operator residual endpoint",
                    columns=source_hidden,
                )
            )
            for field_name, columns in (
                ("mlp.input", source_hidden),
                ("mlp.gate", source_intermediate),
                ("mlp.up", source_intermediate),
                ("mlp.post_swiglu", source_intermediate),
                ("residual.post_layer", source_hidden),
            ):
                expected[field_name].append(
                    _strict_parent_bf16_matrix(
                        rows[field_name][:, -1:, :].reshape(1, -1),
                        f"cached parent {field_name} endpoint",
                        columns=columns,
                    )
                )
            row_ids.append(
                {
                    "batch_index": alignment.source_batch_index,
                    "phase_index": phase_index,
                    "phase": phase.phase,
                    "semantic_phase_sha256": phase.semantic_phase_sha256,
                    "parent_trajectory_sha256": trajectory.evidence_sha256,
                }
            )
    if not residual_rows:
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN provenance has no semantic endpoint rows."
        )
    residual = torch.cat(tuple(residual_rows), dim=0)
    cached = {
        field: torch.cat(tuple(values), dim=0) for field, values in expected.items()
    }
    device = torch.device(execution_device)
    normalized_mlp_input = _reconstruct_parent_normalized_mlp_input(
        residual,
        ffn_norm_weight=norm,
        norm_epsilon=float(epsilon),
        execution_device=device,
    )
    with torch.no_grad():
        replay_residual = residual.to(device=device, dtype=torch.bfloat16)
        mlp_input = normalized_mlp_input.to(device=device, dtype=torch.bfloat16)
        replay_gate = torch.nn.functional.linear(
            mlp_input,
            gate.to(device=device, dtype=torch.bfloat16),
        )
        replay_up = torch.nn.functional.linear(
            mlp_input,
            up.to(device=device, dtype=torch.bfloat16),
        )
        replay_post = torch.nn.functional.silu(replay_gate) * replay_up
        replay_down = torch.nn.functional.linear(
            replay_post,
            down.to(device=device, dtype=torch.bfloat16),
        )
        replay_layer = replay_residual + replay_down
    replayed = {
        "mlp.input": normalized_mlp_input,
        "mlp.gate": replay_gate.detach().to(device="cpu").contiguous(),
        "mlp.up": replay_up.detach().to(device="cpu").contiguous(),
        "mlp.post_swiglu": replay_post.detach().to(device="cpu").contiguous(),
        "residual.post_layer": replay_layer.detach().to(device="cpu").contiguous(),
    }
    equivalence_policy = _bf16_cross_kernel_equivalence_policy()
    endpoint_metrics = {
        field: _bf16_cross_kernel_equivalence_metrics(
            replayed=replayed[field],
            cached=cached[field],
            policy=equivalence_policy,
            field_name=field,
        )
        for field in replayed
    }
    outside_legacy_envelope = {
        field: {
            "maximum_absolute_error": metrics["maximum_absolute_error"],
            "relative_rmse": metrics["relative_rmse"],
            "elementwise_envelope_passed": metrics["elementwise_envelope_passed"],
            "relative_rmse_envelope_passed": metrics["relative_rmse_envelope_passed"],
        }
        for field, metrics in endpoint_metrics.items()
        if metrics["within_legacy_bf16_envelope"] is not True
    }
    body = {
        "schema": _MLP_FUNCTIONAL_PROVENANCE_SCHEMA,
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_manifest_sha256": cache.manifest["manifest_sha256"],
        "layer_index": layer_index,
        "parent_checkpoint_manifest_sha256": parent_session.manifest_sha256,
        "parent_session_open_receipt_sha256": parent_session.open_receipt_sha256,
        "parent_config_file_sha256": parent_session.config_file_sha256,
        "parent_normalized_config_sha256": (parent_session.normalized_config_sha256),
        "parent_norm_epsilon": float(epsilon),
        "parent_identity_reconstruction_sha256": identity_receipt["result_sha256"],
        "reconstructed_parent_identity": dict(reconstructed_parent),
        "direct_tensor_sha256": {
            "ffn_norm.weight": _tensor_sha256(norm),
            "feed_forward.w1.weight": _tensor_sha256(gate),
            "feed_forward.w3.weight": _tensor_sha256(up),
            "feed_forward.w2.weight": _tensor_sha256(down),
        },
        "cached_endpoint_sha256": {
            field: _tensor_sha256(value) for field, value in sorted(cached.items())
        },
        "replayed_endpoint_sha256": {
            field: _tensor_sha256(value) for field, value in sorted(replayed.items())
        },
        "reconstructed_normalized_mlp_input_sha256": _tensor_sha256(
            normalized_mlp_input
        ),
        "cached_normalized_mlp_input_sha256": _tensor_sha256(cached["mlp.input"]),
        "semantic_endpoint_rows": len(row_ids),
        "semantic_endpoint_row_ids_sha256": canonical_sha256(row_ids),
        "execution_device": str(device),
        "rms_norm_formula": (
            "weight*(x.float()*rsqrt(mean(x.float()^2)+norm_eps)).to(bfloat16)"
        ),
        "mlp_formula": "W2*(silu(W1*rmsnorm(x))*(W3*rmsnorm(x)))+x",
        "cross_kernel_equivalence_policy": dict(equivalence_policy),
        "endpoint_equivalence_metrics": {
            field: dict(metrics) for field, metrics in sorted(endpoint_metrics.items())
        },
        "outside_legacy_bf16_envelope": outside_legacy_envelope,
        "all_cached_endpoints_bit_exact": all(
            metrics["bit_exact"] is True for metrics in endpoint_metrics.values()
        ),
        "all_cached_endpoints_within_legacy_bf16_envelope": not bool(
            outside_legacy_envelope
        ),
        "all_cross_kernel_replay_diagnostics_finite": all(
            metrics["diagnostics_finite"] is True
            for metrics in endpoint_metrics.values()
        ),
        "cached_mlp_input_is_normalized_evidence_source_of_truth": True,
        "reconstructed_mlp_input_is_diagnostic_only": True,
        "direct_parent_tensors_within_legacy_numeric_envelope": not bool(
            outside_legacy_envelope
        ),
        "semantic_endpoint_scope_is_complete_tensor_identity_proof": False,
        "cryptographic_parent_tensor_identity_proven_by_this_receipt": True,
        "exact_parent_identity_is_source_admission_authority": True,
        "numeric_replay_has_source_admission_authority": False,
        "complete_state_dict_identity_reconstructed": True,
        "runtime_buffer_identity_reconstructed": True,
        "normalized_config_identity_reconstructed": True,
        "parent_checkpoint_files_hashed_by_session": True,
        "direct_tensor_digests_recorded": True,
        "complete_state_dict_digest_compared_to_cache": True,
        "numerical_equivalence_is_diagnostic_only": True,
        "numeric_parent_child_weight_match_objective": False,
        "parent_model_forward_calls": 0,
        "training_steps": 0,
    }
    return _sealed(body)


def _validated_mlp_source_functional_provenance(
    value: Mapping[str, Any],
    *,
    layer_index: int,
) -> Mapping[str, Any]:
    """Authenticate the exact-parent v3 source claim and finite diagnostics."""

    receipt = _verify_sealed(value, "direct parent MLP functional provenance")
    direct_tensor_names = {
        "ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w3.weight",
        "feed_forward.w2.weight",
    }
    endpoint_names = {
        "mlp.input",
        "mlp.gate",
        "mlp.up",
        "mlp.post_swiglu",
        "residual.post_layer",
    }
    direct_digests = receipt.get("direct_tensor_sha256")
    cached_digests = receipt.get("cached_endpoint_sha256")
    replayed_digests = receipt.get("replayed_endpoint_sha256")
    metrics = receipt.get("endpoint_equivalence_metrics")
    policy = receipt.get("cross_kernel_equivalence_policy")
    policy_valid = False
    policy_sha256 = None
    if isinstance(policy, Mapping):
        policy_body = dict(policy)
        policy_sha256 = policy_body.pop("policy_sha256", None)
        policy_valid = bool(
            policy.get("schema")
            == "neural-forge-bf16-cross-kernel-equivalence-policy-v1"
            and policy.get("diagnostic_only") is True
            and policy.get("numeric_admission_authority") is False
            and policy.get("fold_selection_or_quality_objective") is False
            and isinstance(policy_sha256, str)
            and policy_sha256 == canonical_sha256(policy_body)
        )

    def is_digest(candidate: Any) -> bool:
        return bool(
            isinstance(candidate, str)
            and len(candidate) == 64
            and all(character in "0123456789abcdef" for character in candidate)
        )

    def exact_digest_mapping(candidate: Any, names: set[str]) -> bool:
        return bool(
            isinstance(candidate, Mapping)
            and set(candidate) == names
            and all(is_digest(digest) for digest in candidate.values())
        )

    def finite_diagnostic(candidate: Any, name: str) -> bool:
        numeric_names = (
            "maximum_absolute_error",
            "mean_absolute_error",
            "error_rms",
            "cached_endpoint_rms",
            "relative_rmse",
            "cached_maximum_absolute_value",
            "maximum_elementwise_tolerance_at_cached_maximum",
        )
        return bool(
            isinstance(candidate, Mapping)
            and candidate.get("field") == name
            and candidate.get("diagnostics_finite") is True
            and candidate.get("diagnostic_only") is True
            and candidate.get("numeric_admission_authority") is False
            and isinstance(candidate.get("within_legacy_bf16_envelope"), bool)
            and isinstance(candidate.get("replayed_all_finite"), bool)
            and isinstance(candidate.get("nonfinite_replayed_element_count"), int)
            and not isinstance(
                candidate.get("nonfinite_replayed_element_count"),
                bool,
            )
            and int(candidate["nonfinite_replayed_element_count"]) >= 0
            and (
                candidate["replayed_all_finite"]
                is (candidate["nonfinite_replayed_element_count"] == 0)
            )
            and (
                candidate["within_legacy_bf16_envelope"] is not True
                or candidate["replayed_all_finite"] is True
            )
            and all(
                isinstance(candidate.get(field), (int, float))
                and not isinstance(candidate.get(field), bool)
                and math.isfinite(float(candidate[field]))
                for field in numeric_names
            )
        )

    reconstructed_parent = receipt.get("reconstructed_parent_identity")
    reconstructed_parent_valid = bool(
        isinstance(reconstructed_parent, Mapping)
        and reconstructed_parent.get("schema")
        == "neural-forge-lfm2-parent-fingerprint-v2"
        and isinstance(reconstructed_parent.get("model_class"), str)
        and reconstructed_parent.get("model_class")
        and all(
            is_digest(reconstructed_parent.get(name))
            for name in (
                "config_sha256",
                "state_dict_sha256",
                "runtime_buffer_sha256",
            )
        )
        and all(
            isinstance(reconstructed_parent.get(name), int)
            and not isinstance(reconstructed_parent.get(name), bool)
            and int(reconstructed_parent[name]) >= 0
            for name in ("state_tensor_count", "runtime_buffer_count")
        )
        and canonical_sha256(dict(reconstructed_parent))
        == receipt.get("cache_parent_fingerprint")
    )

    if (
        receipt.get("schema") != _MLP_FUNCTIONAL_PROVENANCE_SCHEMA
        or receipt.get("layer_index") != layer_index
        or any(
            not is_digest(receipt.get(name))
            for name in (
                "cache_key",
                "cache_parent_fingerprint",
                "cache_manifest_sha256",
                "parent_checkpoint_manifest_sha256",
                "parent_session_open_receipt_sha256",
                "parent_config_file_sha256",
                "parent_normalized_config_sha256",
                "parent_identity_reconstruction_sha256",
                "reconstructed_normalized_mlp_input_sha256",
                "cached_normalized_mlp_input_sha256",
                "semantic_endpoint_row_ids_sha256",
            )
        )
        or not exact_digest_mapping(direct_digests, direct_tensor_names)
        or not exact_digest_mapping(cached_digests, endpoint_names)
        or not exact_digest_mapping(replayed_digests, endpoint_names)
        or not isinstance(metrics, Mapping)
        or set(metrics) != endpoint_names
        or any(not finite_diagnostic(metrics[name], name) for name in endpoint_names)
        or policy_valid is not True
        or reconstructed_parent_valid is not True
        or not isinstance(receipt.get("execution_device"), str)
        or not receipt["execution_device"]
        or isinstance(receipt.get("parent_norm_epsilon"), bool)
        or not isinstance(receipt.get("parent_norm_epsilon"), (int, float))
        or not math.isfinite(float(receipt["parent_norm_epsilon"]))
        or float(receipt["parent_norm_epsilon"]) <= 0.0
        or not isinstance(receipt.get("all_cached_endpoints_bit_exact"), bool)
        or not isinstance(
            receipt.get("all_cached_endpoints_within_legacy_bf16_envelope"),
            bool,
        )
        or receipt.get("all_cross_kernel_replay_diagnostics_finite") is not True
        or not isinstance(
            receipt.get("direct_parent_tensors_within_legacy_numeric_envelope"),
            bool,
        )
        or receipt.get("cached_mlp_input_is_normalized_evidence_source_of_truth")
        is not True
        or receipt.get("reconstructed_mlp_input_is_diagnostic_only") is not True
        or receipt.get("semantic_endpoint_scope_is_complete_tensor_identity_proof")
        is not False
        or receipt.get("cryptographic_parent_tensor_identity_proven_by_this_receipt")
        is not True
        or receipt.get("exact_parent_identity_is_source_admission_authority")
        is not True
        or receipt.get("numeric_replay_has_source_admission_authority") is not False
        or receipt.get("complete_state_dict_identity_reconstructed") is not True
        or receipt.get("runtime_buffer_identity_reconstructed") is not True
        or receipt.get("normalized_config_identity_reconstructed") is not True
        or receipt.get("parent_checkpoint_files_hashed_by_session") is not True
        or receipt.get("direct_tensor_digests_recorded") is not True
        or receipt.get("complete_state_dict_digest_compared_to_cache") is not True
        or receipt.get("numerical_equivalence_is_diagnostic_only") is not True
        or receipt.get("numeric_parent_child_weight_match_objective") is not False
        or receipt.get("parent_model_forward_calls") != 0
        or receipt.get("training_steps") != 0
        or isinstance(receipt.get("semantic_endpoint_rows"), bool)
        or not isinstance(receipt.get("semantic_endpoint_rows"), int)
        or receipt["semantic_endpoint_rows"] <= 0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent MLP functional provenance does not satisfy the "
            "exact-parent-plus-diagnostic-only v3 contract."
        )
    return MappingProxyType(receipt)


def _direct_mlp_inventory(
    *,
    inventory_type: type[Any],
    parent_session: _DirectParentSafetensorsSession,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    layer_index: int,
    operator_kind: str,
    residual_chart: Any,
    chart_receipt: Mapping[str, Any],
    attention_lineage: Lfm2SemanticCompilerLayerLineage | None,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    target_model: Any,
) -> tuple[Any, Mapping[str, Any]]:
    prefix = f"model.layers.{layer_index}.feed_forward."
    gate = parent_session.tensor(prefix + "w1.weight")
    up = parent_session.tensor(prefix + "w3.weight")
    down = parent_session.tensor(prefix + "w2.weight")
    ffn_norm = parent_session.tensor(f"model.layers.{layer_index}.ffn_norm.weight")
    torch = _torch()
    if (
        not isinstance(residual_chart, torch.Tensor)
        or residual_chart.layout != torch.strided
        or residual_chart.device.type == "meta"
        or residual_chart.ndim != 2
        or residual_chart.numel() == 0
        or not torch.is_floating_point(residual_chart)
        or torch.is_complex(residual_chart)
        or residual_chart.requires_grad
        or not bool(torch.isfinite(residual_chart).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Selected MLP residual chart must be one detached finite dense matrix."
        )
    selected_chart = residual_chart.detach().to(device="cpu").contiguous().clone()
    verified_chart_receipt = _verify_sealed(chart_receipt, "residual chart")
    if (
        verified_chart_receipt.get("schema")
        != "neural-forge-lfm2-layer-stack-residual-chart-v1"
        or verified_chart_receipt.get("layer_index") != layer_index
        or verified_chart_receipt.get("chart_sha256") != _tensor_sha256(selected_chart)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Residual-chart receipt does not authenticate the selected MLP chart."
        )
    lineage = _mlp_lineage_sha256(
        cache=cache,
        bridge=bridge,
        layer_index=layer_index,
        operator_kind=operator_kind,
        chart_receipt=verified_chart_receipt,
        attention_lineage=attention_lineage,
    )
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index >= len(layers):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Child has no layer {layer_index} for parent FFN provenance replay."
        )
    execution_device, _execution_dtype = _layer_runtime(layers[layer_index])
    functional_provenance = _validated_mlp_source_functional_provenance(
        _replay_parent_mlp_functional_provenance(
            cache=cache,
            alignments=alignments,
            parent=parent,
            parent_session=parent_session,
            layer_index=layer_index,
            gate_weight=gate,
            up_weight=up,
            down_weight=down,
            ffn_norm_weight=ffn_norm,
            execution_device=execution_device,
        ),
        layer_index=layer_index,
    )
    expected_direct_digests = {
        "ffn_norm.weight": _tensor_sha256(ffn_norm),
        "feed_forward.w1.weight": _tensor_sha256(gate),
        "feed_forward.w3.weight": _tensor_sha256(up),
        "feed_forward.w2.weight": _tensor_sha256(down),
    }
    if (
        functional_provenance.get("cache_key") != cache.cache_key
        or functional_provenance.get("cache_parent_fingerprint")
        != cache.parent_fingerprint
        or functional_provenance.get("cache_manifest_sha256")
        != cache.manifest["manifest_sha256"]
        or functional_provenance.get("parent_checkpoint_manifest_sha256")
        != parent_session.manifest_sha256
        or functional_provenance.get("parent_session_open_receipt_sha256")
        != parent_session.open_receipt_sha256
        or functional_provenance.get("parent_config_file_sha256")
        != parent_session.config_file_sha256
        or functional_provenance.get("parent_normalized_config_sha256")
        != parent_session.normalized_config_sha256
        or functional_provenance.get("parent_identity_reconstruction_sha256")
        != parent_session.parent_identity_reconstruction_sha256
        or functional_provenance.get("reconstructed_parent_identity")
        != {
            "schema": "neural-forge-lfm2-parent-fingerprint-v2",
            "model_class": cache.manifest["parent"]["model_class"],
            "config_sha256": cache.manifest["parent"]["config_sha256"],
            "state_dict_sha256": cache.manifest["parent"]["state_dict_sha256"],
            "state_tensor_count": cache.manifest["parent"]["state_tensor_count"],
            "runtime_buffer_sha256": cache.manifest["parent"]["runtime_buffer_sha256"],
            "runtime_buffer_count": cache.manifest["parent"]["runtime_buffer_count"],
        }
        or functional_provenance.get("parent_norm_epsilon")
        != parent_session.norm_epsilon
        or functional_provenance.get("direct_tensor_sha256") != expected_direct_digests
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent MLP functional provenance is not bound to the "
            "selected cache and local checkpoint session."
        )
    inventory = inventory_type(
        layer_index=layer_index,
        parent_model_identity_sha256=cache.parent_fingerprint,
        parent_checkpoint_manifest_sha256=parent_session.manifest_sha256,
        direct_local_loader_receipt_sha256=functional_provenance["result_sha256"],
        residual_assignment_sha256=verified_chart_receipt["result_sha256"],
        compiler_lineage_sha256=lineage,
        residual_chart=selected_chart,
        ffn_norm_weight=ffn_norm,
        norm_epsilon=parent_session.norm_epsilon,
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
    )
    inventory_body = inventory.authentication_body()
    if (
        not inventory.verify()
        or inventory.residual_assignment_sha256
        != verified_chart_receipt["result_sha256"]
        or inventory.compiler_lineage_sha256 != lineage
        or inventory_body.get("residual_chart_sha256")
        != verified_chart_receipt["chart_sha256"]
        or inventory_body.get("residual_chart_receipt_sha256")
        != verified_chart_receipt["result_sha256"]
        or inventory_body.get("ffn_norm_weight_sha256") != _tensor_sha256(ffn_norm)
        or inventory_body.get("norm_epsilon") != parent_session.norm_epsilon
        or inventory_body.get("normalization_and_loader_lineage_jointly_hashed")
        is not True
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct-local MLP inventory lost residual-chart or normalization "
            "lineage binding."
        )
    return inventory, functional_provenance


def _build_mlp_evidence(
    *,
    evidence_type: type[Any],
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    interface: TiedCausalInterfaceFold,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    chart: Any,
    source_inventory: Any,
    layer_index: int,
    final_layer: bool,
) -> tuple[Any, ...]:
    torch = _torch()
    manifest_sha = _digest(
        cache.manifest.get("manifest_sha256"),
        "cache manifest SHA-256",
    )
    base_encoder = interface.producer_encoder.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    result = []
    for alignment in alignments:
        batch = alignment.source_batch_index
        replay = child_bank.replay_by_batch[batch]
        trajectory = parent[batch]
        prior_target_tokens = 0
        prior_donor_tokens = 0
        evidence_ordinal = 0
        phase_count = len(alignment.phases)
        if phase_count <= 0:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Semantic MLP trajectory {batch} has no causal phases."
            )
        for source_phase_index, phase in enumerate(alignment.phases):
            target_tokens = int(phase.target_full_prefix_tokens)
            donor_tokens = int(phase.donor_full_prefix_tokens)
            boundary = tuple(phase.target_to_donor_boundary_positions)
            prior_boundary_changed = bool(
                source_phase_index > 0
                and boundary[:prior_target_tokens]
                != tuple(
                    alignment.phases[
                        source_phase_index - 1
                    ].target_to_donor_boundary_positions
                )
            )
            if (
                target_tokens <= prior_target_tokens
                or donor_tokens <= prior_donor_tokens
                or len(boundary) != target_tokens
                or prior_boundary_changed
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} is not one "
                    "append-only causal prefix."
                )
            target_positions = tuple(
                position
                for position in range(prior_target_tokens, target_tokens)
                if boundary[position] is not None
            )
            donor_positions = tuple(
                int(boundary[position]) for position in target_positions
            )
            expected_donor_positions = tuple(range(prior_donor_tokens, donor_tokens))
            if donor_positions != expected_donor_positions:
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} does not expose "
                    "every newly introduced parent-exact causal boundary once."
                )
            # The terminal check independently proves that the newest cached
            # parent state and the newest child prefix end at the same exact
            # byte boundary.  Inserted target rows never enter the evidence.
            if target_positions[-1] != _semantic_terminal_target_position(phase):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} lacks an exact "
                    "terminal row in its newly introduced causal suffix."
                )

            histories = trajectory.phase_histories[phase.phase]
            required_histories = (
                "residual.post_operator",
                "mlp.input",
                "mlp.gate",
                "mlp.up",
                "mlp.post_swiglu",
                "residual.post_layer",
            )
            if any(name not in histories for name in required_histories):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} lacks complete "
                    "cached causal histories."
                )
            donor_index = torch.tensor(donor_positions, dtype=torch.long)
            target_index = torch.tensor(target_positions, dtype=torch.long)
            parent_post_operator = histories["residual.post_operator"].index_select(
                1,
                donor_index,
            )[0]
            cached_parent_normalized_input = (
                histories["mlp.input"]
                .index_select(
                    1,
                    donor_index,
                )[0]
                .detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
            parent_gate = histories["mlp.gate"].index_select(1, donor_index)[0]
            parent_up = histories["mlp.up"].index_select(1, donor_index)[0]
            parent_post = histories["mlp.post_swiglu"].index_select(1, donor_index)[0]
            target_dtype = replay.mlp_input.dtype
            if target_dtype not in {torch.bfloat16, torch.float32}:
                raise Lfm2SemanticLayerStackRefoldError(
                    "Replay-fresh child MLP evidence has an unsupported target dtype."
                )
            parent_post_layer = (
                histories["residual.post_layer"]
                .index_select(1, donor_index)
                .to(torch.float64)
                @ chart
            ).to(target_dtype)[0]
            child_input = replay.mlp_input.index_select(1, target_index)[0]
            child_pre = replay.post_operator_residual.index_select(1, target_index)[0]
            terminal_rows = None
            if final_layer:
                parent_final_norm = histories.get("output.final_norm_output")
                if parent_final_norm is None:
                    raise Lfm2SemanticLayerStackRefoldError(
                        "Final-layer MLP evidence lacks cached parent final-norm "
                        "history."
                    )
                terminal_rows = (
                    parent_final_norm.index_select(1, donor_index).to(torch.float64)
                    @ base_encoder
                ).to(target_dtype)[0]

            row_count = len(target_positions)
            row_weight = 1.0 / (phase_count * row_count)
            for row_offset, (target_position, donor_position) in enumerate(
                zip(target_positions, donor_positions, strict=True)
            ):
                row_phase_sha256 = canonical_sha256(
                    {
                        "schema": (
                            "neural-forge-lfm2-semantic-mlp-causal-boundary-row-v1"
                        ),
                        "source_semantic_phase_sha256": (phase.semantic_phase_sha256),
                        "source_phase_index": source_phase_index,
                        "target_position": target_position,
                        "donor_position": donor_position,
                    }
                )
                row = slice(row_offset, row_offset + 1)
                result.append(
                    evidence_type(
                        batch_index=batch,
                        trajectory_sha256=alignment.alignment_sha256,
                        phase_index=evidence_ordinal,
                        phase_sha256=row_phase_sha256,
                        layer_index=layer_index,
                        alignment_sha256=phase.alignment_sha256,
                        cache_key=cache.cache_key,
                        cache_manifest_sha256=manifest_sha,
                        bridge_artifact_sha256=bridge.artifact_sha256,
                        captured_child_model_state_sha256=(
                            child_bank.captured_child_model_state_sha256
                        ),
                        source_inventory=source_inventory,
                        parent_gate=parent_gate[row],
                        parent_up=parent_up[row],
                        parent_post_swiglu=parent_post[row],
                        parent_pre_mlp_residual=parent_post_operator[row],
                        parent_normalized_mlp_input=(
                            cached_parent_normalized_input[row]
                        ),
                        actual_child_mlp_input=child_input[row],
                        actual_child_pre_mlp_residual=child_pre[row],
                        parent_post_mlp_residual_target_space=(parent_post_layer[row]),
                        parent_final_norm_output_target_space=(
                            None if terminal_rows is None else terminal_rows[row]
                        ),
                        sample_weight=row_weight,
                        parent_exact_semantic_endpoint=True,
                        inserted_target_row=False,
                    )
                )
                evidence_ordinal += 1
            prior_target_tokens = target_tokens
            prior_donor_tokens = donor_tokens
        if evidence_ordinal != prior_donor_tokens:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Semantic MLP trajectory {batch} emitted {evidence_ordinal} exact "
                f"rows for {prior_donor_tokens} cached parent states."
            )
    return tuple(result)


def _install_mlp(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Any],
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    installer: Callable[..., Any],
    final_layer: bool,
    parameter_count: int,
    fold_compute_device: str = "cpu",
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
    normalized_input_correction_rho: float = 0.25,
    normalized_input_request_capture_directory: str | os.PathLike[str] | None = None,
    raw_structural_decoder: Any | None = None,
    raw_structural_decoder_sha256: str | None = None,
    coordinate_completion_receipt: Mapping[str, Any] | None = None,
    coordinate_completion_receipt_sha256: str | None = None,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Lfm2CpcfTerminalActionDownWeightPrior | None,
]:
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    before_hashes = _state_tensor_hashes(target_model)
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    if any(item.captured_child_model_state_sha256 != before for item in evidence):
        raise Lfm2SemanticLayerStackRefoldError(
            "MLP evidence is stale before installation."
        )
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    w2 = getattr(
        getattr(getattr(layers[layer_index], "feed_forward", None), "w2", None),
        "weight",
        None,
    )
    w2_before = _tensor_sha256(w2)
    result = installer(
        target_model=target_model,
        layer_index=layer_index,
        evidence=tuple(evidence),
        cache_key=cache.cache_key,
        cache_manifest_sha256=cache.manifest["manifest_sha256"],
        bridge_artifact_sha256=bridge.artifact_sha256,
        defer_down_weight=final_layer,
        body_recipe=recipe,
        normalized_input_correction_rho=normalized_input_correction_rho,
        **(
            {
                "normalized_input_request_capture_directory": (
                    normalized_input_request_capture_directory
                )
            }
            if normalized_input_request_capture_directory is not None
            else {}
        ),
        raw_structural_decoder=raw_structural_decoder,
        raw_structural_decoder_sha256=raw_structural_decoder_sha256,
        coordinate_completion_receipt=coordinate_completion_receipt,
        coordinate_completion_receipt_sha256=(coordinate_completion_receipt_sha256),
        **(
            {"fold_compute_device": fold_compute_device}
            if fold_compute_device != "cpu"
            else {}
        ),
    )
    receipt_value = getattr(result, "receipt", None)
    terminal_evidence = dict(getattr(result, "final_layer_mlp_evidence", {}))
    deferred_prior = getattr(result, "deferred_selected_w2_prior", None)
    if not isinstance(receipt_value, Mapping):
        raise Lfm2SemanticLayerStackRefoldError(
            "MLP installer returned no authenticated receipt."
        )
    receipt = _verify_sealed(receipt_value, "semantic MLP")
    _require_body_recipe_binding(
        receipt,
        recipe,
        component="mlp",
        label="MLP installer",
    )
    after_hashes = _state_tensor_hashes(target_model)
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    changed = _changed_state_keys(before_hashes, after_hashes)
    w2_after = _tensor_sha256(w2)
    common_invalid = (
        receipt.get("schema") != _MLP_INSTALLATION_SCHEMA
        or receipt.get("layer_index") != layer_index
        or receipt.get("model_state_before_sha256") != before
        or receipt.get("model_state_after_sha256") != after
        or receipt.get("parameter_count_before") != parameter_count
        or receipt.get("parameter_count_after") != parameter_count
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("topology_changed") is not False
        or _parameter_count(target_model) != parameter_count
    )
    if common_invalid:
        raise Lfm2SemanticLayerStackRefoldError(
            "Semantic MLP receipt is stale or changed model topology."
        )
    if raw_structural_decoder is not None:
        planting = receipt.get("source_derived_planting")
        if (
            not isinstance(planting, Mapping)
            or planting.get("normalized_input_correction_rho")
            != normalized_input_correction_rho
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Semantic MLP receipt lost its normalized-input correction rho."
            )
    structural_mlp_noop = _authenticated_mlp_incumbent_noop(
        receipt,
        before_sha256=before,
        after_sha256=after,
        changed_state_keys=changed,
    )
    analytic_final_deferral = _authenticated_final_mlp_analytic_deferral(
        receipt,
        before_sha256=before,
        after_sha256=after,
        changed_state_keys=changed,
    )
    if final_layer:
        evidence_sha = {
            split: value.authentication_sha256
            for split, value in sorted(terminal_evidence.items())
        }
        if (
            receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
            or receipt.get("install_count") not in (0, 1)
            or set(terminal_evidence) != set(_PARTITIONS)
            or any(
                value.split != split or not value.verify_authentication()
                for split, value in terminal_evidence.items()
            )
            or receipt.get("terminal_action_evidence_sha256") != evidence_sha
            or w2_after != w2_before
            or (receipt.get("install_count") == 0 and after != before)
            or (
                receipt.get("install_count") == 0
                and not (structural_mlp_noop or analytic_final_deferral)
            )
            or (
                receipt.get("install_count") == 1
                and (receipt.get("installed") is not True or after == before)
            )
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Final MLP did not defer W2 with four authenticated banks."
            )
        qdeim_lane = recipe.component_lanes["mlp"] in {
            "nonlinear_qdeim_structural_priority",
            "nonlinear_qdeim_fit_corrected_priority",
        }
        if qdeim_lane:
            if (
                not isinstance(
                    deferred_prior,
                    Lfm2CpcfTerminalActionDownWeightPrior,
                )
                or not deferred_prior.verify_authentication()
                or deferred_prior.body_recipe_id != recipe.recipe_id
                or deferred_prior.body_recipe_sha256 != recipe.recipe_sha256
                or deferred_prior.mlp_lane != recipe.component_lanes["mlp"]
                or receipt.get("deferred_selected_w2_prior_authentication_sha256")
                != deferred_prior.authentication_sha256
                or receipt.get("deferred_selected_w2_prior_sha256")
                != _tensor_sha256(deferred_prior.down_weight)
                or receipt.get("deferred_selected_w2_prior_terminal_action_sha256")
                != lfm2_cpcf_terminal_action_tensor_sha256(deferred_prior.down_weight)
                or deferred_prior.live_deferred_w2_sha256
                != lfm2_cpcf_terminal_action_tensor_sha256(w2)
                or receipt.get("deferred_selected_w2_prior_installed_before_terminal")
                is not False
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Final Q-DEIM MLP did not carry its selected W2 as an "
                    "authenticated uninstalled terminal prior."
                )
        elif deferred_prior is not None:
            raise Lfm2SemanticLayerStackRefoldError(
                "A legacy final MLP unexpectedly returned a terminal W2 prior."
            )
    elif structural_mlp_noop:
        if (
            receipt.get("install_count") != 0
            or receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not False
            or receipt.get("cache_key") != cache.cache_key
            or receipt.get("bridge_artifact_sha256") != bridge.artifact_sha256
            or receipt.get("evidence_endpoint_count") != len(evidence)
            or terminal_evidence
            or w2_after != w2_before
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Nonterminal MLP incumbent no-op receipt is stale or incomplete."
            )
    elif (
        receipt.get("installed") is not True
        or receipt.get("install_count") != 1
        or receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not False
        or after == before
        or terminal_evidence
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Nonterminal MLP did not install one accepted complete writer."
        )
    return (
        MappingProxyType(receipt),
        MappingProxyType(terminal_evidence),
        deferred_prior,
    )


def _validated_operator_topology(
    cache: Lfm2CircuitTraceCache,
    target_model: Any,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
) -> tuple[str, ...]:
    topology = cache.manifest.get("layer_topology", {}).get("layers")
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if not isinstance(topology, list) or not topology or len(topology) != len(layers):
        raise Lfm2SemanticLayerStackRefoldError(
            "The executable LFM2 transaction requires one nonempty cache topology "
            "that exactly matches the blank child's layer count."
        )
    kinds = []
    for index, value in enumerate(topology):
        contract = value.get("contract") if isinstance(value, Mapping) else None
        kind = contract.get("kind") if isinstance(contract, Mapping) else None
        if kind not in {"full_attention", "short_conv"}:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Cache layer {index} has no supported LFM2 operator contract."
            )
        is_attention = getattr(layers[index], "is_attention_layer", None)
        if not isinstance(is_attention, bool) or is_attention != (
            kind == "full_attention"
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Child layer {index} differs from the cached operator topology."
            )
        kinds.append(str(kind))
    attention = tuple(
        index for index, kind in enumerate(kinds) if kind == "full_attention"
    )
    if not attention:
        raise Lfm2SemanticLayerStackRefoldError(
            "The authenticated LFM2 topology contains no attention operator."
        )
    child_types = tuple(
        getattr(getattr(target_model, "config", None), "layer_types", ())
    )
    if child_types:
        normalized = tuple(
            "short_conv" if value in {"conv", "short_conv"} else str(value)
            for value in child_types
        )
        if normalized != tuple(kinds):
            raise Lfm2SemanticLayerStackRefoldError(
                "Child config layer types differ from its cached topology."
            )
    if set(compiler_lineage) != set(attention) or any(
        not isinstance(compiler_lineage[index], Lfm2SemanticCompilerLayerLineage)
        or compiler_lineage[index].layer_index != index
        for index in attention
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Attention compiler lineage must cover the exact topology-derived "
            "attention layer indices."
        )
    return tuple(kinds)


def _validated_interface_lineage(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    *,
    mapped_parent_tied_table: Any | None = None,
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
    interface_runtime_mode: str = _INTERFACE_RUNTIME_NATIVE_TIED,
) -> Mapping[str, Any]:
    torch = _torch()
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    input_weight = getattr(target_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(target_model.get_output_embeddings(), "weight", None)
    runtime_mode = str(interface_runtime_mode).strip().lower()
    if runtime_mode not in _INTERFACE_RUNTIME_MODES:
        raise Lfm2SemanticLayerStackRefoldError(
            "interface_runtime_mode must be native_tied or independent_output."
        )
    report = dict(interface.report)
    report_body = dict(report)
    report_sha = report_body.pop("report_sha256", None)
    report_schema = report.get("schema")
    if report_schema in {
        _TIED_CAUSAL_INTERFACE_FOLD_SCHEMA,
        TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
        TIED_RESIDUAL_CONGRUENCE_SCHEMA,
    }:
        tensor_sha256 = _interface_tensor_sha256
        tensor_seal_schema = "dtype-nul-shape-nul-raw-v1"
    elif report_schema == TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA:
        tensor_sha256 = _anchor_interface_tensor_sha256
        tensor_seal_schema = "canonical-descriptor-prefix-raw-v1"
    else:
        raise Lfm2SemanticLayerStackRefoldError(
            "Passed tied-interface coordinates use an unsupported report schema."
        )
    producer_sha = tensor_sha256(interface.producer_encoder)
    target_sha = tensor_sha256(interface.target_weight)
    congruence_authentication = None
    if report_schema == TIED_RESIDUAL_CONGRUENCE_SCHEMA:
        try:
            congruence_authentication = authenticate_tied_residual_congruence_interface(
                interface,
                mapped_parent_tied_table=mapped_parent_tied_table,
                anchor_transport_witness=anchor_transport_witness,
            )
        except (TypeError, ValueError) as exc:
            raise Lfm2SemanticLayerStackRefoldError(
                "Passed tied residual-congruence lineage is invalid."
            ) from exc
    input_matches = bool(
        isinstance(input_weight, torch.nn.Parameter)
        and tuple(interface.target_weight.shape) == tuple(input_weight.shape)
        and torch.equal(
            interface.target_weight.to(device="cpu", dtype=input_weight.dtype),
            input_weight.detach().to(device="cpu"),
        )
    )
    native_tied = bool(
        isinstance(input_weight, torch.nn.Parameter) and output_weight is input_weight
    )
    independent_output = bool(
        isinstance(input_weight, torch.nn.Parameter)
        and isinstance(output_weight, torch.nn.Parameter)
        and output_weight is not input_weight
        and tuple(output_weight.shape) == tuple(input_weight.shape)
        and int(output_weight.untyped_storage().data_ptr())
        != int(input_weight.untyped_storage().data_ptr())
    )
    output_matches_structural_input = bool(
        independent_output
        and torch.equal(
            output_weight.detach().to(device="cpu"),
            input_weight.detach().to(
                device="cpu",
                dtype=output_weight.dtype,
            ),
        )
    )
    configured_tie = getattr(
        getattr(target_model, "config", None),
        "tie_word_embeddings",
        None,
    )
    runtime_topology_valid = bool(
        (
            runtime_mode == _INTERFACE_RUNTIME_NATIVE_TIED
            and native_tied
            and configured_tie is not False
        )
        or (
            runtime_mode == _INTERFACE_RUNTIME_INDEPENDENT_OUTPUT
            and independent_output
            and output_matches_structural_input
            and configured_tie is not True
        )
    )
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or not input_matches
        or not runtime_topology_valid
        or report_sha != canonical_sha256(report_body)
        or report.get("producer_encoder_sha256") != producer_sha
        or report.get("target_weight_sha256") != target_sha
        or (
            report_schema == TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA
            and (
                report.get("source_to_vocabulary_operator_preserved") is not True
                or report.get("full_vocabulary_rows_preserved") is not True
                or report.get("one_global_residual_gauge_available") is not True
            )
        )
        or (
            report_schema == TIED_RESIDUAL_CONGRUENCE_SCHEMA
            and (
                not isinstance(congruence_authentication, Mapping)
                or congruence_authentication.get(
                    "producer_basis_orthogonality_authenticated"
                )
                is not True
                or congruence_authentication.get("interface_report_sha256")
                != report_sha
            )
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Passed tied-interface coordinates lack exact authenticated child lineage."
        )
    return MappingProxyType(
        {
            "interface_report_sha256": report_sha,
            "interface_report_schema": report_schema,
            "interface_tensor_seal_schema": tensor_seal_schema,
            "producer_encoder_sha256": producer_sha,
            "target_weight_sha256": target_sha,
            "interface_runtime_mode": runtime_mode,
            "target_weight_matches_live_input_child": True,
            "target_weight_matches_live_tied_child": native_tied,
            "live_output_parameter_independent": independent_output,
            "output_table_matches_structural_input_at_body_entry": (
                output_matches_structural_input
            ),
            "body_fold_requires_output_parameter_alias": False,
            "producer_encoder_taken_from_authenticated_compiler_lineage": True,
            "producer_encoder_inferred_from_repaired_target_table": False,
            "global_residual_gauge_required": report_schema
            in {
                TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
                TIED_RESIDUAL_CONGRUENCE_SCHEMA,
            },
            "tied_residual_congruence_authenticated": (
                report_schema == TIED_RESIDUAL_CONGRUENCE_SCHEMA
            ),
            "tied_residual_congruence_authentication_sha256": (
                None
                if congruence_authentication is None
                else congruence_authentication["result_sha256"]
            ),
        }
    )


def _anchor_interface_tensor_sha256(value: Any) -> str:
    """Match the tensor seal used by the causal-anchor residual report."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _final_w2_parameter(target_model: Any) -> Any:
    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    parameter = None
    if layers:
        parameter = getattr(
            getattr(getattr(layers[-1], "feed_forward", None), "w2", None),
            "weight",
            None,
        )
    if not isinstance(parameter, torch.nn.Parameter):
        raise Lfm2SemanticLayerStackRefoldError(
            "The final child MLP has no ordinary dense W2 Parameter."
        )
    named = dict(target_model.named_parameters(remove_duplicate=False))
    terminal_layer_index = len(layers) - 1
    if (
        named.get(f"model.layers.{terminal_layer_index}.feed_forward.w2.weight")
        is not parameter
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "The final child W2 is renamed, missing, or not independently addressable."
        )
    return parameter


def _validate_direct_attention_sources(
    parent_session: _DirectParentSafetensorsSession,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
) -> Mapping[int, Mapping[str, Any]]:
    receipts = {}
    for layer_index in sorted(compiler_lineage):
        prefix = f"model.layers.{layer_index}.self_attn."
        tensors = {
            "q": parent_session.tensor(prefix + "q_proj.weight"),
            "k": parent_session.tensor(prefix + "k_proj.weight"),
            "v": parent_session.tensor(prefix + "v_proj.weight"),
            "o": parent_session.tensor(prefix + "out_proj.weight"),
            "q_norm": parent_session.tensor(prefix + "q_layernorm.weight"),
            "k_norm": parent_session.tensor(prefix + "k_layernorm.weight"),
            "operator_norm": parent_session.tensor(
                f"model.layers.{layer_index}.operator_norm.weight"
            ),
        }
        observed = seal_lfm2_dense_qkvo_source_inventory(
            source_query_weight=tensors["q"],
            source_key_weight=tensors["k"],
            source_value_weight=tensors["v"],
            source_output_weight=tensors["o"],
        )
        expected = compiler_lineage[layer_index].source_inventory
        if observed.as_dict() != expected.as_dict():
            raise Lfm2SemanticLayerStackRefoldError(
                f"Direct-local parent Q/K/V/O changed at layer {layer_index}."
            )
        receipts[layer_index] = _sealed(
            {
                "schema": "neural-forge-lfm2-direct-parent-qkvo-session-load-v1",
                "layer_index": layer_index,
                "parent_session_open_receipt_sha256": (
                    parent_session.open_receipt_sha256
                ),
                "source_inventory_sha256": observed.inventory_sha256,
                "query_norm_weight_sha256": _tensor_sha256(tensors["q_norm"]),
                "key_norm_weight_sha256": _tensor_sha256(tensors["k_norm"]),
                "operator_norm_weight_sha256": _tensor_sha256(tensors["operator_norm"]),
                "qk_norms_loaded_as_atomic_attention_source": True,
                "compiler_lineage_sha256": (
                    compiler_lineage[layer_index].compiler_lineage_sha256
                ),
                "direct_local_safetensors": True,
                "parent_model_objects_constructed": 0,
            }
        )
    return MappingProxyType(receipts)


def _layer_receipt(
    *,
    layer_index: int,
    operator_kind: str,
    before_operator_replay: _LayerReplayBank,
    operator_receipt: Mapping[str, Any],
    operator_coverage_decision: Mapping[str, Any],
    after_operator_replay: _LayerReplayBank,
    mlp_receipt: Mapping[str, Any],
    mlp_coverage_decision: Mapping[str, Any],
    mlp_source_provenance: Mapping[str, Any],
    propagation_replay: _LayerReplayBank,
    chart_receipt: Mapping[str, Any],
    parent: Mapping[int, _ParentLayerTrajectory],
    operator_evidence_sha256: Sequence[str],
    mlp_evidence_sha256: Sequence[str],
    fir_receipt: Mapping[str, Any] | None,
    fold_compute_device: str,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
    body_coordinate_completion_receipt: Mapping[str, Any] | None = None,
    body_coordinate_completion_candidate_receipt: Mapping[str, Any] | None = None,
    body_coordinate_completion_source: _BodyCoordinateCompletionSource | None = None,
) -> Mapping[str, Any]:
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if operator_kind == "full_attention":
        _require_attention_body_recipe_binding(
            operator_receipt,
            recipe,
            label=f"Layer {layer_index} operator receipt",
        )
    else:
        _require_body_recipe_binding(
            operator_receipt,
            recipe,
            component="short_conv",
            label=f"Layer {layer_index} operator receipt",
        )
    _require_body_recipe_binding(
        mlp_receipt,
        recipe,
        component="mlp",
        label=f"Layer {layer_index} MLP receipt",
    )
    verified_mlp_source_provenance = _validated_mlp_source_functional_provenance(
        mlp_source_provenance,
        layer_index=layer_index,
    )
    planting = mlp_receipt.get("source_derived_planting")
    completion_values = (
        body_coordinate_completion_receipt,
        body_coordinate_completion_candidate_receipt,
    )
    if any(value is not None for value in completion_values) and not all(
        value is not None for value in completion_values
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Layer {layer_index} received an incomplete body-global completion "
            "binding."
        )
    if body_coordinate_completion_source is not None and not all(
        value is not None for value in completion_values
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Layer {layer_index} dropped its body-global completion source."
        )
    body_completion_fields: dict[str, Any] = {}
    if all(value is not None for value in completion_values):
        assert body_coordinate_completion_receipt is not None
        assert body_coordinate_completion_candidate_receipt is not None
        completion_wrapper = _verify_sealed(
            body_coordinate_completion_receipt,
            "body-global coordinate completion",
        )
        completion_candidate = _verify_sealed(
            body_coordinate_completion_candidate_receipt,
            "body-global coordinate completion candidate",
        )
        candidate_sha256 = completion_candidate["result_sha256"]
        raw_public_sha256 = _digest(
            completion_wrapper.get("public_coordinate_map_sha256"),
            "body-global public coordinate map",
        )
        coordinate_map_sha256 = _digest(
            completion_candidate.get("coordinate_map_sha256"),
            "body-global analytic coordinate map",
        )
        if (
            completion_wrapper.get("selected_candidate_result_sha256")
            != candidate_sha256
            or completion_wrapper.get("coordinate_map_sha256") != coordinate_map_sha256
            or chart_receipt.get("global_self_dual_gauge_required") is not True
            or chart_receipt.get("global_gauge_preserved_without_layer_local_rotation")
            is not True
            or not isinstance(planting, Mapping)
            or planting.get("shared_body_global_completion_sha256") != candidate_sha256
            or planting.get("body_global_raw_decoder_public_sha256")
            != raw_public_sha256
            or planting.get("body_global_coordinate_map_sha256")
            != coordinate_map_sha256
            or not isinstance(
                planting.get("normalized_input_correction_rho"),
                (int, float),
            )
            or isinstance(
                planting.get("normalized_input_correction_rho"),
                bool,
            )
            or not math.isfinite(float(planting["normalized_input_correction_rho"]))
            or float(planting["normalized_input_correction_rho"]) < 0.0
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Layer {layer_index} MLP did not preserve the one shared "
                "body-global coordinate completion."
            )
        if operator_kind == "full_attention":
            normalized_receipt = operator_receipt.get(
                "normalized_input_decoder_receipt"
            )
            normalized_provenance = (
                normalized_receipt.get("raw_structural_decoder_provenance")
                if isinstance(normalized_receipt, Mapping)
                else None
            )
            operator_completion_valid = bool(
                isinstance(normalized_provenance, Mapping)
                and normalized_provenance.get("coordinate_completion_receipt_sha256")
                == candidate_sha256
                and normalized_provenance.get("raw_structural_decoder_sha256")
                == raw_public_sha256
                and normalized_provenance.get(
                    "coordinate_completion_coordinate_map_sha256"
                )
                == coordinate_map_sha256
                and normalized_receipt.get("normalized_input_correction_rho")
                == float(planting["normalized_input_correction_rho"])
                and normalized_receipt.get("structural_prior_relative_trust_bound")
                is True
                and normalized_receipt.get("correction_bound_satisfied") is True
            )
        else:
            operator_completion_valid = bool(
                operator_receipt.get("shared_body_global_completion_sha256")
                == candidate_sha256
                and operator_receipt.get("raw_target_to_parent_decoder_sha256")
                == raw_public_sha256
                and operator_receipt.get("coordinate_completion_coordinate_map_sha256")
                == coordinate_map_sha256
            )
        if not operator_completion_valid:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Layer {layer_index} operator did not preserve the one shared "
                "body-global coordinate completion."
            )
        body_completion_fields = {
            "body_global_coordinate_completion_sha256": completion_wrapper[
                "result_sha256"
            ],
            "shared_body_global_completion_sha256": candidate_sha256,
            "body_global_raw_decoder_public_sha256": raw_public_sha256,
            "body_global_coordinate_map_sha256": coordinate_map_sha256,
            "normalized_input_correction_rho": float(
                planting["normalized_input_correction_rho"]
            ),
            "body_global_completion_shared_by_operator_and_mlp": True,
            "body_global_global_gauge_preserved": True,
        }
        body_completion_fields.update(
            _body_coordinate_geometry_receipt_fields(
                completion_wrapper,
                source=body_coordinate_completion_source,
            )
        )
    execution = (
        planting.get("weight_complete_signature_compute_execution")
        if isinstance(planting, Mapping)
        else None
    )
    mlp_qdeim_lane = recipe.component_lanes["mlp"] in _GLOBAL_RESIDUAL_MLP_LANES
    mlp_qdeim_compute_execution: Mapping[str, Any] | None = None
    if mlp_qdeim_lane:
        if (
            not isinstance(planting, Mapping)
            or planting.get("fold_compute_device") != fold_compute_device
            or planting.get("normalized_decoder_control_compute_device") != "cpu"
            or planting.get("qdeim_pod_and_pivot_compute_device") != fold_compute_device
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Layer {layer_index} MLP did not bind its decoder and Q-DEIM "
                "algebra to the authenticated compute devices."
            )
        mlp_qdeim_compute_execution = _validated_qdeim_compute_execution(
            planting.get("qdeim_compute_execution"),
            requested_device=fold_compute_device,
            schema="neural-forge-lfm2-mlp-qdeim-compute-v1",
            label=f"Layer {layer_index} MLP",
        )
    # Q-DEIM lanes authenticate their own POD/pivot CUDA receipt immediately
    # above.  They intentionally do not compile the legacy weight-complete
    # signature, so requiring that unrelated receipt here rejects valid CUDA
    # Q-DEIM work after it has already executed.
    if (
        fold_compute_device != "cpu"
        and not mlp_qdeim_lane
        and (
            not isinstance(planting, Mapping)
            or planting.get("fold_compute_device") != fold_compute_device
            or not isinstance(execution, Mapping)
            or execution.get("requested_device") != fold_compute_device
            or execution.get("backend") != "cuda"
            or execution.get("compute_dtype") != "float64"
        )
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Layer {layer_index} MLP did not prove the requested CUDA fold backend."
        )
    short_conv_compute_fields: dict[str, Any] = {}
    short_conv_construction_fields: dict[str, Any] = {}
    attention_receipt_fields: dict[str, Any] = {}
    if operator_kind == "full_attention":
        if (
            recipe.fit_metrics_order_candidates
            or recipe.component_lanes["qk"] == _COUPLED_ATTENTION_QDEIM_LANE
        ):
            _authenticate_lfm2_semantic_attention_operator_receipt(
                operator_receipt,
                layer_index=layer_index,
            )
        # The attention adapter's sealed operator result is also its atomic
        # installation receipt.  Persist both bindings explicitly so neither
        # side becomes an opaque hash after the folding process exits.
        attention_receipt_fields = {
            "attention_operator_decision_receipt": dict(operator_receipt),
            "attention_operator_installation_receipt": dict(operator_receipt),
        }
        if recipe.component_lanes["qk"] == _COUPLED_ATTENTION_QDEIM_LANE:
            attention_qdeim_compute_execution = _validated_qdeim_compute_execution(
                operator_receipt.get("qdeim_compute_execution"),
                requested_device=fold_compute_device,
                schema="neural-forge-lfm2-attention-qdeim-compute-v1",
                label=f"Layer {layer_index} coupled attention",
            )
            if (
                operator_receipt.get("fold_compute_device") != fold_compute_device
                or operator_receipt.get("qdeim_pod_and_pivot_compute_device")
                != fold_compute_device
                or operator_receipt.get("qdeim_compute_execution_sha256")
                != attention_qdeim_compute_execution["receipt_sha256"]
                or operator_receipt.get("requested_fold_compute_device_proven")
                is not True
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Layer {layer_index} coupled attention did not preserve its "
                    "requested-device Q-DEIM proof."
                )
            attention_receipt_fields.update(
                {
                    "attention_qdeim_fold_compute_device": fold_compute_device,
                    "attention_qdeim_pod_and_pivot_compute_device": (
                        fold_compute_device
                    ),
                    "attention_qdeim_compute_execution_sha256": (
                        attention_qdeim_compute_execution["receipt_sha256"]
                    ),
                }
            )
    if fold_compute_device != "cpu" and operator_kind == "short_conv":
        structural_execution = operator_receipt.get(
            "structural_cluster_compute_execution"
        )
        writer_execution = operator_receipt.get("writer_compute_execution")
        if (
            operator_receipt.get("fold_compute_device") != fold_compute_device
            or operator_receipt.get("requested_cuda_short_conv_routes_proved")
            is not True
            or not isinstance(structural_execution, Mapping)
            or not isinstance(writer_execution, Mapping)
            or structural_execution.get("requested_device") != fold_compute_device
            or writer_execution.get("requested_device") != fold_compute_device
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Layer {layer_index} short-conv did not prove the requested "
                "CUDA fold backend."
            )
        short_conv_compute_fields = {
            "short_conv_fold_compute_device": fold_compute_device,
            "short_conv_structural_cluster_compute_execution_sha256": (
                structural_execution.get("execution_sha256")
            ),
            "short_conv_writer_compute_execution_sha256": (
                writer_execution.get("execution_sha256")
            ),
        }
    if operator_kind == "short_conv" and "short_conv_construction" in operator_receipt:
        construction = _resolve_short_conv_construction(
            operator_receipt.get("short_conv_construction_id")
        )
        source_field = (
            "coordinate_complete_direct_source"
            if construction.uses_body_global_coordinate_completion
            else "fhvr_source"
        )
        source_sha_field = (
            "coordinate_complete_direct_source_sha256"
            if construction.uses_body_global_coordinate_completion
            else "fhvr_source_sha256"
        )
        source_value = operator_receipt.get(source_field)
        source_body = dict(source_value) if isinstance(source_value, Mapping) else {}
        source_sha256 = source_body.pop("source_sha256", None)
        if (
            not (
                construction.uses_fhvr_full_coverage
                or construction.uses_body_global_coordinate_completion
            )
            or operator_receipt.get("short_conv_construction") != construction.as_dict()
            or operator_receipt.get("short_conv_construction_sha256")
            != construction.construction_sha256
            or source_sha256 != canonical_sha256(source_body)
            or operator_receipt.get(source_sha_field) != source_sha256
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Layer {layer_index} short-conv construction/source binding is "
                "invalid."
            )
        if construction.uses_body_global_coordinate_completion:
            completion_sha256 = operator_receipt.get(
                "coordinate_completion_candidate_sha256"
            )
            if (
                _digest(
                    completion_sha256,
                    "short-conv shared coordinate completion",
                )
                != operator_receipt.get("shared_body_global_completion_sha256")
                or operator_receipt.get(
                    "body_global_coordinate_complete_structural_fold"
                )
                is not True
                or operator_receipt.get(
                    "coordinate_complete_candidate_selected_by_metrics"
                )
                is not False
                or operator_receipt.get(
                    "coordinate_complete_candidate_selected_by_construction"
                )
                is not True
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Layer {layer_index} coordinate-complete short-conv lost "
                    "its shared candidate binding."
                )
            short_conv_construction_fields = {
                "short_conv_construction": construction.as_dict(),
                "short_conv_construction_id": construction.construction_id,
                "short_conv_construction_sha256": (construction.construction_sha256),
                "short_conv_coordinate_source_sha256": source_sha256,
                "short_conv_coordinate_candidate_state_sha256": (
                    operator_receipt["coordinate_complete_candidate_state_sha256"]
                ),
                "short_conv_shared_body_global_completion_sha256": (completion_sha256),
                "short_conv_raw_decoder_public_sha256": operator_receipt[
                    "raw_target_to_parent_decoder_sha256"
                ],
                "short_conv_coordinate_map_sha256": operator_receipt[
                    "coordinate_completion_coordinate_map_sha256"
                ],
            }
        else:
            short_conv_construction_fields = {
                "short_conv_construction": construction.as_dict(),
                "short_conv_construction_id": construction.construction_id,
                "short_conv_construction_sha256": (construction.construction_sha256),
                "short_conv_fhvr_source_sha256": source_sha256,
                "short_conv_fhvr_candidate_state_sha256": operator_receipt[
                    "fhvr_candidate_state_sha256"
                ],
            }
    operator_noop_status = (
        _SHORT_CONV_STRUCTURAL_NOOP_STATUS
        if operator_kind == "short_conv"
        else LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS
    )
    body = {
        "schema": "neural-forge-lfm2-semantic-layer-transaction-v1",
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "layer_index": layer_index,
        "operator_kind": operator_kind,
        "model_state_before_operator_sha256": (
            before_operator_replay.captured_child_model_state_sha256
        ),
        "before_operator_replay_bank_sha256": before_operator_replay.bank_sha256,
        "operator_decision_sha256": operator_receipt["result_sha256"],
        "operator_installation_sha256": operator_receipt["result_sha256"],
        "operator_complete_candidate_frontier_receipt": dict(operator_receipt),
        **attention_receipt_fields,
        "operator_producer_coverage_decision_sha256": (
            operator_coverage_decision["result_sha256"]
        ),
        "operator_producer_coverage_decision": dict(operator_coverage_decision),
        "operator_decision_count": 1,
        "operator_installed": operator_receipt.get("installed") is True,
        "operator_install_count": int(operator_receipt.get("installed") is True),
        "model_state_after_operator_sha256": (
            after_operator_replay.captured_child_model_state_sha256
        ),
        "operator_parameter_bytes_changed": (
            before_operator_replay.captured_child_model_state_sha256
            != after_operator_replay.captured_child_model_state_sha256
        ),
        "operator_incumbent_noop_selected": (
            operator_receipt.get("status") == operator_noop_status
        ),
        "after_operator_replay_bank_sha256": after_operator_replay.bank_sha256,
        "mlp_decision_sha256": mlp_receipt["result_sha256"],
        "mlp_installation_sha256": mlp_receipt["result_sha256"],
        "mlp_complete_candidate_frontier_receipt": dict(mlp_receipt),
        "mlp_producer_coverage_decision_sha256": (
            mlp_coverage_decision["result_sha256"]
        ),
        "mlp_producer_coverage_decision": dict(mlp_coverage_decision),
        "mlp_decision_count": 1,
        "mlp_installed": mlp_receipt.get("installed") is True,
        "mlp_install_count": int(mlp_receipt.get("installed") is True),
        "mlp_source_functional_provenance_sha256": (
            verified_mlp_source_provenance["result_sha256"]
        ),
        "mlp_source_functional_provenance_schema": (
            verified_mlp_source_provenance["schema"]
        ),
        "mlp_source_direct_tensors_within_legacy_numeric_envelope": (
            verified_mlp_source_provenance[
                "direct_parent_tensors_within_legacy_numeric_envelope"
            ]
        ),
        "mlp_source_all_cached_endpoints_bit_exact": (
            verified_mlp_source_provenance["all_cached_endpoints_bit_exact"]
        ),
        "mlp_source_all_cached_endpoints_within_legacy_bf16_envelope": (
            verified_mlp_source_provenance[
                "all_cached_endpoints_within_legacy_bf16_envelope"
            ]
        ),
        "mlp_source_all_cross_kernel_replay_diagnostics_finite": (
            verified_mlp_source_provenance["all_cross_kernel_replay_diagnostics_finite"]
        ),
        "mlp_source_cross_kernel_equivalence_policy_sha256": (
            verified_mlp_source_provenance["cross_kernel_equivalence_policy"][
                "policy_sha256"
            ]
        ),
        "mlp_source_semantic_endpoint_scope_is_complete_tensor_identity_proof": (
            verified_mlp_source_provenance[
                "semantic_endpoint_scope_is_complete_tensor_identity_proof"
            ]
        ),
        "mlp_source_cryptographic_parent_tensor_identity_proven_by_this_receipt": (
            verified_mlp_source_provenance[
                "cryptographic_parent_tensor_identity_proven_by_this_receipt"
            ]
        ),
        "mlp_source_exact_parent_identity_is_source_admission_authority": (
            verified_mlp_source_provenance[
                "exact_parent_identity_is_source_admission_authority"
            ]
        ),
        "mlp_source_numeric_replay_has_source_admission_authority": (
            verified_mlp_source_provenance[
                "numeric_replay_has_source_admission_authority"
            ]
        ),
        "mlp_source_parent_identity_reconstruction_sha256": (
            verified_mlp_source_provenance["parent_identity_reconstruction_sha256"]
        ),
        "mlp_source_parent_checkpoint_files_hashed_by_session": (
            verified_mlp_source_provenance["parent_checkpoint_files_hashed_by_session"]
        ),
        "mlp_source_direct_tensor_digests_recorded": (
            verified_mlp_source_provenance["direct_tensor_digests_recorded"]
        ),
        "mlp_source_complete_state_dict_digest_compared_to_cache": (
            verified_mlp_source_provenance[
                "complete_state_dict_digest_compared_to_cache"
            ]
        ),
        "mlp_source_numerical_equivalence_is_diagnostic_only": (
            verified_mlp_source_provenance["numerical_equivalence_is_diagnostic_only"]
        ),
        "model_state_after_mlp_sha256": (
            propagation_replay.captured_child_model_state_sha256
        ),
        "mlp_parameter_bytes_changed": (
            after_operator_replay.captured_child_model_state_sha256
            != propagation_replay.captured_child_model_state_sha256
        ),
        "mlp_incumbent_noop_selected": (
            mlp_receipt.get("status") == _MLP_STRUCTURAL_NOOP_STATUS
        ),
        "propagation_replay_bank_sha256": propagation_replay.bank_sha256,
        "fit_only_residual_chart_sha256": chart_receipt["result_sha256"],
        "parent_trajectory_sha256": [
            parent[index].evidence_sha256 for index in sorted(parent)
        ],
        "operator_evidence_sha256": list(operator_evidence_sha256),
        "mlp_evidence_sha256": list(mlp_evidence_sha256),
        "mlp_fold_compute_device": (
            planting.get("fold_compute_device", "cpu")
            if isinstance(planting, Mapping)
            else "cpu"
        ),
        "mlp_weight_signature_compute_execution_sha256": (
            execution.get("execution_sha256")
            if isinstance(execution, Mapping)
            else None
        ),
        **(
            {
                "mlp_normalized_decoder_control_compute_device": "cpu",
                "mlp_qdeim_pod_and_pivot_compute_device": fold_compute_device,
                "mlp_qdeim_compute_execution_sha256": (
                    mlp_qdeim_compute_execution["receipt_sha256"]
                ),
                "mlp_requested_fold_compute_device_applies_to_qdeim_pod_and_pivot": (
                    True
                ),
                "mlp_normalized_decoder_control_remains_cpu": True,
            }
            if mlp_qdeim_lane and mlp_qdeim_compute_execution is not None
            else {}
        ),
        **short_conv_compute_fields,
        **short_conv_construction_fields,
        **body_completion_fields,
        "fir_equivalence_sha256": (
            None if fir_receipt is None else fir_receipt["result_sha256"]
        ),
        "operator_then_replay_fresh_mlp_then_propagate": True,
        "operator_and_mlp_producer_chains_structurally_nonzero": True,
        "numeric_parent_child_weight_match_objective": False,
        "numeric_thresholds_used_for_candidate_admission": False,
        "heldout_metrics_have_candidate_or_publication_authority": False,
        "incumbent_fallback_only_when_all_alternatives_invalid_or_identical": True,
        "training_steps": 0,
    }
    return _sealed(body)


@dataclass(frozen=True, slots=True)
class _BodyCoordinateCompletionSource:
    recipe: str
    operators: tuple[ResidualFacingOperator, ...]
    geometry: RoleBalancedOperatorGeometry | None
    least_squares_execution_profile: AnalyticExecutionProfile | None
    ordered_operator_sha256s: tuple[str, ...]
    operator_manifest_sha256: str
    raw_gram_sha256: str | None
    balanced_gram_sha256: str | None

    @property
    def uses_preaggregated_geometry(self) -> bool:
        return self.geometry is not None


def _body_coordinate_completion_source(
    *,
    parent_operators: Sequence[ResidualFacingOperator] | None,
    parent_geometry: RoleBalancedOperatorGeometry | None,
    least_squares_execution_profile: AnalyticExecutionProfile | None,
    recipe: str | None,
) -> _BodyCoordinateCompletionSource | None:
    supplied = (
        parent_operators,
        parent_geometry,
        least_squares_execution_profile,
        recipe,
    )
    if all(value is None for value in supplied):
        return None
    if not isinstance(recipe, str) or not recipe.strip():
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global coordinate completion requires one explicit recipe."
        )
    has_operators = parent_operators is not None
    has_geometry = parent_geometry is not None
    if has_operators == has_geometry:
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global coordinate completion requires exactly one of parent "
            "operators or authenticated preaggregated parent geometry."
        )
    selected_recipe = recipe.strip()
    if selected_recipe not in LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES:
        raise Lfm2SemanticLayerStackRefoldError(
            "Unsupported body-global coordinate completion recipe: " + selected_recipe
        )
    if has_operators:
        if least_squares_execution_profile is not None:
            raise Lfm2SemanticLayerStackRefoldError(
                "The legacy coordinate-completion operator path does not accept "
                "a least-squares execution profile."
            )
        operators = tuple(parent_operators or ())
        if not operators or any(
            not isinstance(value, ResidualFacingOperator) for value in operators
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Body-global coordinate completion requires nonempty authenticated "
                "residual-facing parent operators."
            )
        ordered_operators = tuple(
            sorted(
                operators,
                key=lambda value: (
                    value.role,
                    value.name,
                    value.orientation.value,
                    value.feature_axis,
                    value.operator_sha256,
                ),
            )
        )
        ordered_sha256s = tuple(value.operator_sha256 for value in ordered_operators)
        return _BodyCoordinateCompletionSource(
            recipe=selected_recipe,
            operators=operators,
            geometry=None,
            least_squares_execution_profile=None,
            ordered_operator_sha256s=ordered_sha256s,
            operator_manifest_sha256=canonical_sha256(list(ordered_sha256s)),
            raw_gram_sha256=None,
            balanced_gram_sha256=None,
        )
    if not isinstance(parent_geometry, RoleBalancedOperatorGeometry):
        raise Lfm2SemanticLayerStackRefoldError(
            "coordinate_completion_parent_geometry must be a "
            "RoleBalancedOperatorGeometry."
        )
    try:
        geometry_verified = parent_geometry.verify()
    except (RuntimeError, ValueError) as exc:
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global preaggregated parent geometry authentication failed."
        ) from exc
    if not geometry_verified:
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global preaggregated parent geometry authentication failed."
        )
    if not isinstance(least_squares_execution_profile, AnalyticExecutionProfile):
        raise Lfm2SemanticLayerStackRefoldError(
            "Preaggregated parent geometry requires an authenticated analytic "
            "least-squares execution profile."
        )
    try:
        profile = AnalyticExecutionProfile(least_squares_execution_profile.receipt)
    except AnalyticExecutionProfileError as exc:
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global least-squares execution profile authentication failed."
        ) from exc
    hashes = parent_geometry.report.get("hashes")
    if not isinstance(hashes, Mapping):
        raise Lfm2SemanticLayerStackRefoldError(
            "Preaggregated parent geometry omitted its authenticated Gram hashes."
        )
    raw_gram_sha256 = _digest(
        hashes.get("raw_gram_sha256"),
        "body-global parent raw Gram",
    )
    balanced_gram_sha256 = _digest(
        hashes.get("balanced_gram_sha256"),
        "body-global parent balanced Gram",
    )
    ordered_sha256s = tuple(parent_geometry.ordered_operator_sha256s)
    if not ordered_sha256s:
        raise Lfm2SemanticLayerStackRefoldError(
            "Preaggregated parent geometry has no ordered operator manifest."
        )
    return _BodyCoordinateCompletionSource(
        recipe=selected_recipe,
        operators=(),
        geometry=parent_geometry,
        least_squares_execution_profile=profile,
        ordered_operator_sha256s=ordered_sha256s,
        operator_manifest_sha256=canonical_sha256(list(ordered_sha256s)),
        raw_gram_sha256=raw_gram_sha256,
        balanced_gram_sha256=balanced_gram_sha256,
    )


def _body_coordinate_geometry_receipt_fields(
    wrapper_receipt: Mapping[str, Any],
    *,
    source: _BodyCoordinateCompletionSource | None,
) -> dict[str, Any]:
    geometry_receipt = wrapper_receipt.get("schema") == (
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA
    )
    source_uses_geometry = bool(
        source is not None and source.uses_preaggregated_geometry
    )
    if geometry_receipt != source_uses_geometry:
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global completion substituted or dropped its authenticated "
            "preaggregated geometry source."
        )
    if not geometry_receipt:
        return {}
    assert source is not None
    assert source.geometry is not None
    assert source.least_squares_execution_profile is not None
    profile_receipt = wrapper_receipt.get("least_squares_execution_profile")
    try:
        profile = AnalyticExecutionProfile(profile_receipt)
    except AnalyticExecutionProfileError as exc:
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global completion dropped its authenticated least-squares "
            "execution profile receipt."
        ) from exc
    operator_sha256s = tuple(wrapper_receipt.get("parent_operator_sha256s", ()))
    geometry_sha256 = _digest(
        wrapper_receipt.get("parent_geometry_sha256"),
        "body-global parent geometry",
    )
    raw_gram_sha256 = _digest(
        wrapper_receipt.get("parent_raw_gram_sha256"),
        "body-global parent raw Gram",
    )
    balanced_gram_sha256 = _digest(
        wrapper_receipt.get("parent_balanced_gram_sha256"),
        "body-global parent balanced Gram",
    )
    profile_sha256 = _digest(
        wrapper_receipt.get("least_squares_execution_profile_sha256"),
        "body-global least-squares execution profile",
    )
    if (
        geometry_sha256 != source.geometry.geometry_sha256
        or wrapper_receipt.get("parent_geometry_result_sha256") != geometry_sha256
        or raw_gram_sha256 != source.raw_gram_sha256
        or balanced_gram_sha256 != source.balanced_gram_sha256
        or operator_sha256s != source.ordered_operator_sha256s
        or wrapper_receipt.get("parent_operator_manifest_sha256")
        != source.operator_manifest_sha256
        or wrapper_receipt.get("parent_operator_reconstruction_calls") != 0
        or wrapper_receipt.get("preaggregated_parent_geometry") is not True
        or profile_sha256 != profile.result_sha256
        or profile_sha256 != source.least_squares_execution_profile.result_sha256
        or profile.as_dict() != source.least_squares_execution_profile.as_dict()
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Body-global completion geometry/profile binding changed."
        )
    return {
        "body_global_parent_geometry_sha256": geometry_sha256,
        "body_global_parent_raw_gram_sha256": raw_gram_sha256,
        "body_global_parent_balanced_gram_sha256": balanced_gram_sha256,
        "body_global_least_squares_execution_profile": profile.as_dict(),
        "body_global_least_squares_execution_profile_sha256": profile_sha256,
        "body_global_parent_operator_count": len(operator_sha256s),
        "body_global_parent_operator_sha256s": list(operator_sha256s),
        "body_global_parent_operator_manifest_sha256": (
            source.operator_manifest_sha256
        ),
        "body_global_preaggregated_parent_geometry": True,
        "body_global_parent_operator_reconstruction_calls": 0,
    }


def compile_install_lfm2_semantic_layer_stack_postfold(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
    parent_model_directory: str | os.PathLike[str],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    attention_installer: Callable[..., Mapping[str, Any]] | None,
    short_conv_installer: Callable[..., Mapping[str, Any]] | None = None,
    mlp_installer: Callable[..., Any] | None = None,
    fold_compute_device: str = "cpu",
    global_residual_closure_version: str = _GLOBAL_RESIDUAL_PRODUCER_AWARE_V2A,
    correction_scale_frontier: Sequence[float] = (1.0,),
    correction_scale: float = 1.0,
    global_residual_backend: str = "auto",
    global_residual_basis_rank: int | None = None,
    target_dtype_representability_level: str = "first_effectful",
    normalized_input_correction_rho: float = 0.25,
    normalized_input_request_capture_directory: str | os.PathLike[str] | None = None,
    authenticated_cache_session: Any | None = None,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
    short_conv_construction: Any = "full-source-structural-v1",
    mapped_parent_tied_table: Any | None = None,
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
    interface_runtime_mode: str = _INTERFACE_RUNTIME_NATIVE_TIED,
    coordinate_completion_parent_operators: Sequence[ResidualFacingOperator]
    | None = None,
    coordinate_completion_parent_geometry: RoleBalancedOperatorGeometry | None = None,
    coordinate_completion_least_squares_execution_profile: (
        AnalyticExecutionProfile | None
    ) = None,
    coordinate_completion_recipe: str | None = None,
) -> Lfm2SemanticLayerStackInstallation:
    """Compile and install the complete topology-derived child transaction.

    This is a no-training compiler.  It consumes an immutable parent trace
    cache and direct local parent tensors, then replays only individual child
    macrocells.  It never constructs or forwards the parent model and never
    invokes the child model's public ``forward`` method.
    """

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    if not isinstance(compiler_lineage, Mapping):
        raise TypeError("compiler_lineage must be a mapping.")
    if (
        not isinstance(normalized_input_correction_rho, (int, float))
        or isinstance(normalized_input_correction_rho, bool)
        or not math.isfinite(float(normalized_input_correction_rho))
        or float(normalized_input_correction_rho) < 0.0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "normalized_input_correction_rho must be finite and non-negative."
        )
    normalized_input_correction_rho = float(normalized_input_correction_rho)
    completion_source = _body_coordinate_completion_source(
        parent_operators=coordinate_completion_parent_operators,
        parent_geometry=coordinate_completion_parent_geometry,
        least_squares_execution_profile=(
            coordinate_completion_least_squares_execution_profile
        ),
        recipe=coordinate_completion_recipe,
    )
    coordinate_completion_enabled = completion_source is not None
    completion_operators = (
        () if completion_source is None else completion_source.operators
    )
    completion_geometry = (
        None if completion_source is None else completion_source.geometry
    )
    completion_least_squares_profile = (
        None
        if completion_source is None
        else completion_source.least_squares_execution_profile
    )
    completion_recipe = None if completion_source is None else completion_source.recipe
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    construction = _resolve_short_conv_construction(short_conv_construction)
    if coordinate_completion_enabled != bool(
        construction.uses_body_global_coordinate_completion
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "A body-global coordinate completion must use the coordinate-complete "
            "short-conv construction, and that construction requires completion "
            "evidence."
        )
    coupled_attention_recipe = bool(
        recipe.component_lanes["qk"] == _COUPLED_ATTENTION_QDEIM_LANE
        and recipe.component_lanes["value_output"] == _COUPLED_ATTENTION_QDEIM_LANE
    )
    global_residual_closure_required = _global_residual_closure_required(recipe)
    source_local_global_residual = _source_local_global_residual_required(recipe)
    closure_version = str(global_residual_closure_version).strip().lower()
    if closure_version not in _GLOBAL_RESIDUAL_CLOSURE_VERSIONS:
        raise Lfm2SemanticLayerStackRefoldError(
            "global_residual_closure_version must be producer_aware_v2a, "
            "producer_aware_source_local_v2b, or output_writer_v1_diagnostic."
        )
    if (
        source_local_global_residual
        and closure_version == _GLOBAL_RESIDUAL_PRODUCER_AWARE_V2A
    ):
        closure_version = _GLOBAL_RESIDUAL_PRODUCER_AWARE_SOURCE_LOCAL_V2B
    if (
        source_local_global_residual
        and closure_version != _GLOBAL_RESIDUAL_PRODUCER_AWARE_SOURCE_LOCAL_V2B
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Source-local global residual folding requires the producer-aware "
            "physical closure engine."
        )
    if (
        not source_local_global_residual
        and closure_version == _GLOBAL_RESIDUAL_PRODUCER_AWARE_SOURCE_LOCAL_V2B
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "The source-local v2b closure may only be selected by a body recipe "
            "that supplies source-local global residual actions."
        )
    producer_aware_closure = closure_version in {
        _GLOBAL_RESIDUAL_PRODUCER_AWARE_V2A,
        _GLOBAL_RESIDUAL_PRODUCER_AWARE_SOURCE_LOCAL_V2B,
    }
    closure_policy = Lfm2GlobalResidualClosurePolicy(
        backend=str(global_residual_backend).strip().lower()
    )
    scale_policy = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=tuple(
            float(value) for value in correction_scale_frontier
        ),
        materialization_scale=float(correction_scale),
    )
    if global_residual_basis_rank is not None and (
        isinstance(global_residual_basis_rank, bool)
        or not isinstance(global_residual_basis_rank, int)
        or global_residual_basis_rank <= 0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "global_residual_basis_rank must be a positive integer or None."
        )
    if closure_version == _GLOBAL_RESIDUAL_OUTPUT_WRITER_V1_DIAGNOSTIC and (
        scale_policy.materialization_frontier != (1.0,)
        or scale_policy.materialization_scale != 1.0
        or global_residual_basis_rank is not None
        or str(target_dtype_representability_level).strip().lower() != "first_effectful"
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "The v1 diagnostic baseline does not materialize a correction-scale "
            "frontier or an explicit residual basis rank; use its canonical "
            "scale 1.0 and legacy topology-derived rank."
        )
    if not coupled_attention_recipe and not callable(attention_installer):
        raise Lfm2SemanticLayerStackRefoldError(
            "The complete stack requires the analytic attention installer."
        )
    saved_tokenizer = getattr(
        target_model,
        "_neural_forge_quarantine_tokenizer",
        target_tokenizer,
    )
    if saved_tokenizer is not target_tokenizer and fingerprint_hf_tokenizer_semantics(
        saved_tokenizer
    ) != fingerprint_hf_tokenizer_semantics(target_tokenizer):
        raise Lfm2SemanticLayerStackRefoldError(
            "Saved child tokenizer differs from the semantic target tokenizer."
        )
    kinds = _validated_operator_topology(cache, target_model, compiler_lineage)
    layer_count = len(kinds)
    terminal_layer_index = len(kinds) - 1
    attention_layer_indices = tuple(
        index for index, kind in enumerate(kinds) if kind == "full_attention"
    )
    short_conv_layer_indices = tuple(
        index for index, kind in enumerate(kinds) if kind == "short_conv"
    )
    nonterminal_layer_count = layer_count - 1
    producer_chain_count = 2 * layer_count
    interface_lineage = _validated_interface_lineage(
        target_model,
        interface,
        mapped_parent_tied_table=mapped_parent_tied_table,
        anchor_transport_witness=anchor_transport_witness,
        interface_runtime_mode=interface_runtime_mode,
    )
    effective_global_residual_tied_table = None
    global_residual_tied_table_source = "not_applicable"
    if global_residual_closure_required:
        # The closure's terminal directions live in the child residual space.
        # ``mapped_parent_tied_table`` is deliberately retained above only for
        # source-coordinate congruence authentication; in strong-congruence
        # modes it has shape [target_vocab, parent_hidden] and must never be
        # routed into a child-hidden closure.  ``interface.target_weight`` is
        # byte-authenticated against the live tied child and is exactly the
        # mapped terminal table in child coordinates.
        effective_global_residual_tied_table = interface.target_weight
        global_residual_tied_table_source = (
            "authenticated_live_interface_target_weight_child_coordinates"
        )
    short_evidence_type, default_short_installer = _resolve_short_conv_adapter()
    (
        mlp_inventory_type,
        mlp_evidence_type,
        default_mlp_installer,
    ) = _resolve_mlp_adapter()
    selected_short_installer = (
        default_short_installer
        if short_conv_installer is None
        else short_conv_installer
    )
    selected_mlp_installer = (
        default_mlp_installer if mlp_installer is None else mlp_installer
    )
    if not callable(selected_short_installer) or not callable(selected_mlp_installer):
        raise TypeError("Short-conv and MLP installers must be callable.")
    parameter_count = _parameter_count(target_model)
    if parameter_count <= 0:
        raise Lfm2SemanticLayerStackRefoldError("Child has no dense Parameters.")
    final_w2 = _final_w2_parameter(target_model)
    final_w2_before = _tensor_sha256(final_w2)
    snapshot = _WholeModelSnapshot.capture(target_model)
    entry_state_hashes = {
        name: _tensor_sha256(value) for name, value in snapshot.state.items()
    }
    result: Lfm2SemanticLayerStackInstallation | None = None
    try:
        target_rope_inv_freq, target_rope_attention_scaling = _target_rope(target_model)
        cache_session_context = (
            cache.selective_authenticated_session()
            if authenticated_cache_session is None
            else nullcontext(authenticated_cache_session)
        )
        with (
            cache_session_context as cache_session,
            _DirectParentSafetensorsSession(
                parent_model_directory,
                expected_parent_identity_sha256=cache.parent_fingerprint,
                expected_parent_record=cache.manifest.get("parent"),
            ) as parent_session,
        ):
            alignments, tokenizer_session_receipt = _semantic_alignments_from_session(
                cache=cache,
                session=cache_session,
                bridge=bridge,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
            )
            global_residual_accumulators: Mapping[
                int,
                _GlobalResidualTrajectoryAccumulator,
            ] = MappingProxyType({})
            global_residual_row_map_receipt: Mapping[str, Any] | None = None
            if global_residual_closure_required:
                (
                    global_residual_accumulators,
                    global_residual_row_map_receipt,
                ) = _prepare_global_residual_accumulators(alignments)
            refinement = dict(
                audit_lfm2_semantic_span_cpcf_refinement(
                    alignments,
                    expected_batch_count=cache.batch_count,
                )
            )
            if not refinement.get(
                "probability_preserving_attention_transport_go",
                False,
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Semantic span refinement is not exact: "
                    + ", ".join(refinement.get("blocking_reasons", ()))
                )
            direct_attention_receipts = _validate_direct_attention_sources(
                parent_session,
                compiler_lineage,
            )
            residual_bank = _initial_residual_bank(target_model, alignments)
            layer_receipts = []
            attention_decision_count = 0
            attention_install_count = 0
            attention_incumbent_noop_count = 0
            short_conv_decision_count = 0
            short_conv_install_count = 0
            short_conv_incumbent_noop_count = 0
            operator_parameter_mutation_count = 0
            mlp_evidence_compile_count = 0
            nonterminal_mlp_decision_count = 0
            nonterminal_mlp_install_count = 0
            nonterminal_mlp_incumbent_noop_count = 0
            final_mlp_producer_install_count = 0
            final_mlp_producer_incumbent_noop_count = 0
            mlp_parameter_mutation_count = 0
            attention_producer_coverage_count = 0
            short_conv_producer_coverage_count = 0
            nonterminal_mlp_producer_coverage_count = 0
            final_mlp_producer_coverage_count = 0
            cold_start_producer_chain_count = 0
            installed_complete_producer_chain_count = 0
            retained_nonzero_seed_noop_count = 0
            blank_or_partially_blank_producer_chain_count = 0
            final_terminal_evidence: Mapping[str, Any] = MappingProxyType({})
            final_terminal_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior | None = None
            body_coordinate_completion: Lfm2BodyGlobalCoordinateCompletion | None = None
            body_coordinate_completion_candidate: Any | None = None
            body_coordinate_fit_rows_receipt: Mapping[str, Any] | None = None
            for layer_index, operator_kind in enumerate(kinds):
                after_operator_replay_override: _LayerReplayBank | None = None
                before_operator_replay = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                parent = _load_parent_layer_trajectories(
                    cache_session,
                    alignments=alignments,
                    layer_index=layer_index,
                    operator_kind=operator_kind,
                    include_final_norm=layer_index == terminal_layer_index,
                )
                chart, chart_receipt = _fit_residual_chart(
                    alignments=alignments,
                    parent=parent,
                    child_bank=before_operator_replay,
                    producer_encoder=interface.producer_encoder,
                    layer_index=layer_index,
                    global_gauge_required=(
                        coordinate_completion_enabled
                        or interface_lineage.get("global_residual_gauge_required")
                        is True
                    ),
                )
                if coordinate_completion_enabled and layer_index == 0:
                    (
                        completion_child_rows,
                        completion_parent_rows,
                        body_coordinate_fit_rows_receipt,
                    ) = _body_global_fit_residual_rows(
                        alignments=alignments,
                        parent=parent,
                        child_bank=before_operator_replay,
                        chart_receipt=chart_receipt,
                        layer_index=layer_index,
                    )
                    if completion_geometry is None:
                        completion_evidence_sha256 = (
                            lfm2_body_global_coordinate_completion_evidence_sha256(
                                completion_child_rows,
                                completion_parent_rows,
                                interface.producer_encoder,
                                completion_operators,
                                recipe=completion_recipe,
                            )
                        )
                        body_coordinate_completion = (
                            compile_lfm2_body_global_coordinate_completion(
                                completion_child_rows,
                                completion_parent_rows,
                                interface.producer_encoder,
                                completion_operators,
                                recipe=completion_recipe,
                                expected_evidence_sha256=(completion_evidence_sha256),
                            )
                        )
                    else:
                        if completion_least_squares_profile is None:
                            raise Lfm2SemanticLayerStackRefoldError(
                                "Preaggregated body-global geometry lost its "
                                "least-squares execution profile."
                            )
                        completion_evidence_sha256 = (
                            lfm2_body_global_coordinate_completion_evidence_sha256(
                                completion_child_rows,
                                completion_parent_rows,
                                interface.producer_encoder,
                                parent_geometry=completion_geometry,
                                least_squares_execution_profile=(
                                    completion_least_squares_profile
                                ),
                                recipe=completion_recipe,
                            )
                        )
                        body_coordinate_completion = (
                            compile_lfm2_body_global_coordinate_completion(
                                completion_child_rows,
                                completion_parent_rows,
                                interface.producer_encoder,
                                parent_geometry=completion_geometry,
                                least_squares_execution_profile=(
                                    completion_least_squares_profile
                                ),
                                recipe=completion_recipe,
                                expected_evidence_sha256=(completion_evidence_sha256),
                            )
                        )
                    body_coordinate_completion_candidate = (
                        body_coordinate_completion.frontier.candidate(
                            body_coordinate_completion.receipt[
                                "materialized_candidate_recipe"
                            ]
                        )
                    )
                    if body_coordinate_completion_candidate.report.get(
                        "result_sha256"
                    ) != body_coordinate_completion.receipt[
                        "selected_candidate_result_sha256"
                    ] or not _torch().equal(
                        body_coordinate_completion_candidate.coordinate_map,
                        body_coordinate_completion.raw_decoder,
                    ):
                        raise Lfm2SemanticLayerStackRefoldError(
                            "Body-global wrapper does not bind its selected "
                            "analytic completion candidate."
                        )
                if coordinate_completion_enabled and (
                    body_coordinate_completion is None
                    or body_coordinate_completion_candidate is None
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        "Body-global coordinate completion was not compiled before "
                        "the first residual-facing consumer."
                    )
                completion_kwargs = (
                    {
                        "raw_structural_decoder": (
                            body_coordinate_completion.raw_decoder
                        ),
                        "raw_structural_decoder_sha256": (
                            body_coordinate_completion.receipt[
                                "public_coordinate_map_sha256"
                            ]
                        ),
                        "coordinate_completion_receipt": (
                            body_coordinate_completion_candidate.report
                        ),
                        "coordinate_completion_receipt_sha256": (
                            body_coordinate_completion_candidate.report["result_sha256"]
                        ),
                    }
                    if body_coordinate_completion is not None
                    else {}
                )
                if global_residual_closure_required:
                    _capture_global_parent_layer_boundaries(
                        global_residual_accumulators,
                        parent=parent,
                        residual_chart=chart,
                        producer_encoder=interface.producer_encoder,
                        layer_index=layer_index,
                        terminal_layer_index=terminal_layer_index,
                        capture_source_local_pre=(
                            _source_local_global_residual_required(recipe)
                        ),
                    )
                fir_receipt = None
                operator_coverage_before = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind=operator_kind,
                )
                if operator_kind == "full_attention":
                    lineage = compiler_lineage[layer_index]
                    attention_prefix = f"model.layers.{layer_index}.self_attn."
                    parent_qk_tensors = MappingProxyType(
                        {
                            "q": parent_session.tensor(
                                attention_prefix + "q_proj.weight"
                            ),
                            "k": parent_session.tensor(
                                attention_prefix + "k_proj.weight"
                            ),
                            "v": parent_session.tensor(
                                attention_prefix + "v_proj.weight"
                            ),
                            "o": parent_session.tensor(
                                attention_prefix + "out_proj.weight"
                            ),
                            "q_norm": parent_session.tensor(
                                attention_prefix + "q_layernorm.weight"
                            ),
                            "k_norm": parent_session.tensor(
                                attention_prefix + "k_layernorm.weight"
                            ),
                            "operator_norm": parent_session.tensor(
                                f"model.layers.{layer_index}.operator_norm.weight"
                            ),
                        }
                    )
                    target_backbone = getattr(target_model, "model", None)
                    target_layers = getattr(target_backbone, "layers", None)
                    target_operator_norm_weight = (
                        None
                        if target_layers is None
                        else target_layers[layer_index].operator_norm.weight
                    )
                    target_norm_epsilon = getattr(
                        getattr(target_model, "config", None),
                        "norm_eps",
                        None,
                    )
                    if coupled_attention_recipe:
                        parent_norm_epsilon = getattr(
                            parent_session,
                            "norm_epsilon",
                            None,
                        )
                        if (
                            parent_norm_epsilon is None
                            or target_norm_epsilon is None
                            or target_operator_norm_weight is None
                        ):
                            raise Lfm2SemanticLayerStackRefoldError(
                                "Coupled attention Q-DEIM requires authenticated "
                                "parent and target normalization contracts."
                            )
                        normalized_decoder, normalized_decoder_receipt = (
                            _fit_layer_normalized_operator_input_decoder(
                                alignments=alignments,
                                parent=parent,
                                child_bank=before_operator_replay,
                                parent_operator_norm_weight=(
                                    parent_qk_tensors["operator_norm"]
                                ),
                                target_operator_norm_weight=(
                                    target_operator_norm_weight
                                ),
                                parent_norm_epsilon=parent_norm_epsilon,
                                target_norm_epsilon=target_norm_epsilon,
                                chart=chart,
                                chart_receipt=chart_receipt,
                                normalized_input_correction_rho=(
                                    normalized_input_correction_rho
                                ),
                                **completion_kwargs,
                            )
                        )
                        partition_batches = chart_receipt.get("partition_batch_indices")
                        fit_batch_indices = (
                            partition_batches.get("fit")
                            if isinstance(partition_batches, Mapping)
                            else None
                        )
                        if (
                            not isinstance(fit_batch_indices, list)
                            or not fit_batch_indices
                            or any(
                                isinstance(index, bool)
                                or not isinstance(index, int)
                                or index < 0
                                for index in fit_batch_indices
                            )
                        ):
                            raise Lfm2SemanticLayerStackRefoldError(
                                "Residual chart omitted its authenticated FIT batches."
                            )
                        fit_batch_sample_weights = {
                            index: 1.0 for index in fit_batch_indices
                        }
                        source_norm_inventory = (
                            seal_lfm2_semantic_attention_group_qdeim_norm_inventory(
                                layer_index=layer_index,
                                source_parent_identity_sha256=(
                                    cache.parent_fingerprint
                                ),
                                source_inventory=lineage.source_inventory,
                                source_query_norm_weight=(parent_qk_tensors["q_norm"]),
                                source_key_norm_weight=parent_qk_tensors["k_norm"],
                            )
                        )
                        source_local_attention_output = (
                            _source_local_attention_output_required(recipe)
                        )
                        structural_candidate_sink: (
                            dict[int, Lfm2CompiledAttentionGroupQdeim] | None
                        ) = {} if source_local_attention_output else None
                        structural_operator_receipt = (
                            _install_coupled_attention_group_qdeim_operator(
                                target_model=target_model,
                                session=cache_session,
                                layer_index=layer_index,
                                fit_batch_indices=fit_batch_indices,
                                fit_batch_sample_weights=(fit_batch_sample_weights),
                                source_query_weight=parent_qk_tensors["q"],
                                source_key_weight=parent_qk_tensors["k"],
                                source_value_weight=parent_qk_tensors["v"],
                                source_out_weight=parent_qk_tensors["o"],
                                source_inventory=lineage.source_inventory,
                                source_query_norm_weight=(parent_qk_tensors["q_norm"]),
                                source_key_norm_weight=parent_qk_tensors["k_norm"],
                                source_norm_inventory=source_norm_inventory,
                                normalized_input_decoder=normalized_decoder,
                                normalized_input_decoder_receipt=(
                                    normalized_decoder_receipt
                                ),
                                residual_output_chart=chart,
                                residual_chart_receipt=chart_receipt,
                                direct_attention_source_receipt=(
                                    direct_attention_receipts[layer_index]
                                ),
                                cache_key=cache.cache_key,
                                bridge_artifact_sha256=bridge.artifact_sha256,
                                parameter_count=parameter_count,
                                fold_compute_device=fold_compute_device,
                                body_recipe=recipe,
                                compiled_candidate_sink=(structural_candidate_sink),
                            )
                        )
                        if source_local_attention_output:
                            if structural_candidate_sink is None or set(
                                structural_candidate_sink
                            ) != {layer_index}:
                                raise Lfm2SemanticLayerStackRefoldError(
                                    "Source-local attention lost its pristine "
                                    "structural QDEIM candidate."
                                )
                            structural_replay = _replay_current_layer(
                                target_model,
                                alignments=alignments,
                                layer_index=layer_index,
                                pre_residual_by_batch=residual_bank,
                            )
                            if (
                                structural_replay.captured_child_model_state_sha256
                                != structural_operator_receipt[
                                    "model_state_after_sha256"
                                ]
                            ):
                                raise Lfm2SemanticLayerStackRefoldError(
                                    f"Layer {layer_index} structural attention "
                                    "replay is not fresh."
                                )
                            (
                                source_local_fit_cases,
                                source_local_fit_cases_receipt,
                            ) = _build_attention_source_local_output_fit_cases(
                                alignments=alignments,
                                parent=parent,
                                structural_replay=structural_replay,
                                fit_batch_indices=fit_batch_indices,
                                fit_batch_sample_weights=(fit_batch_sample_weights),
                                layer_index=layer_index,
                            )
                            source_local_installation = compile_install_lfm2_semantic_attention_source_local_output(
                                target_model=target_model,
                                layer_index=layer_index,
                                structural_candidate=(
                                    structural_candidate_sink.pop(layer_index)
                                ),
                                fit_cases=source_local_fit_cases,
                                fold_compute_device=fold_compute_device,
                            )
                            post_source_local_replay = _replay_current_layer(
                                target_model,
                                alignments=alignments,
                                layer_index=layer_index,
                                pre_residual_by_batch=residual_bank,
                            )
                            operator_receipt = (
                                _compose_source_local_attention_operator_receipt(
                                    structural_operator_receipt=(
                                        structural_operator_receipt
                                    ),
                                    source_local_installation=(
                                        source_local_installation
                                    ),
                                    fit_cases_receipt=(source_local_fit_cases_receipt),
                                    structural_replay=structural_replay,
                                    post_source_local_replay=(post_source_local_replay),
                                    layer_index=layer_index,
                                    parameter_count=parameter_count,
                                    body_recipe=recipe,
                                )
                            )
                            after_operator_replay_override = post_source_local_replay
                        else:
                            operator_receipt = structural_operator_receipt
                        operator_evidence_sha256 = tuple(
                            operator_receipt["transport_evidence_sha256"]
                        )
                    else:
                        operator_evidence = _build_attention_evidence(
                            cache=cache,
                            session=cache_session,
                            bridge=bridge,
                            alignments=alignments,
                            parent=parent,
                            child_bank=before_operator_replay,
                            chart=chart,
                            chart_receipt=chart_receipt,
                            layer_index=layer_index,
                            lineage=lineage,
                            target_rope_inv_freq=target_rope_inv_freq,
                            target_rope_attention_scaling=(
                                target_rope_attention_scaling
                            ),
                            parent_qk_tensors=parent_qk_tensors,
                            target_operator_norm_weight=(target_operator_norm_weight),
                            parent_norm_epsilon=getattr(
                                parent_session,
                                "norm_epsilon",
                                None,
                            ),
                            target_norm_epsilon=target_norm_epsilon,
                            normalized_input_correction_rho=(
                                normalized_input_correction_rho
                            ),
                            **completion_kwargs,
                        )
                        operator_receipt = _install_attention_operator(
                            target_model=target_model,
                            layer_index=layer_index,
                            evidence=operator_evidence,
                            cache_key=cache.cache_key,
                            bridge_artifact_sha256=bridge.artifact_sha256,
                            installer=attention_installer,
                            parameter_count=parameter_count,
                            body_recipe=recipe,
                        )
                        operator_evidence_sha256 = tuple(
                            item.evidence_sha256 for item in operator_evidence
                        )
                    attention_decision_count += 1
                    attention_install_count += int(
                        operator_receipt.get("installed") is True
                    )
                    attention_incumbent_noop_count += int(
                        operator_receipt.get("installed") is False
                    )
                    attention_lineage = lineage
                else:
                    attention_lineage = None
                    prefix = f"model.layers.{layer_index}.conv.conv."
                    source_fir_weight = parent_session.tensor(prefix + "weight")
                    source_fir_bias = parent_session.optional_tensor(prefix + "bias")
                    fir_receipt = _validate_cached_fir(
                        parent=parent,
                        alignments=alignments,
                        source_weight=source_fir_weight,
                        source_bias=source_fir_bias,
                        layer_index=layer_index,
                    )
                    fhvr_source = (
                        _build_short_conv_fhvr_source(
                            parent_session=parent_session,
                            cache=cache,
                            bridge=bridge,
                            layer_index=layer_index,
                            source_fir_weight=source_fir_weight,
                            source_fir_bias=source_fir_bias,
                            residual_chart=chart,
                            residual_chart_receipt=chart_receipt,
                            raw_target_to_parent_decoder=(
                                completion_kwargs.get("raw_structural_decoder")
                            ),
                            coordinate_completion_candidate_receipt=(
                                completion_kwargs.get("coordinate_completion_receipt")
                            ),
                        )
                        if (
                            construction.uses_fhvr_full_coverage
                            or construction.uses_body_global_coordinate_completion
                        )
                        else None
                    )
                    operator_evidence = _build_short_conv_evidence(
                        evidence_type=short_evidence_type,
                        cache=cache,
                        bridge=bridge,
                        alignments=alignments,
                        parent=parent,
                        child_bank=before_operator_replay,
                        chart=chart,
                        chart_receipt=chart_receipt,
                        layer_index=layer_index,
                        **(
                            {"fhvr_source": fhvr_source}
                            if fhvr_source is not None
                            else {}
                        ),
                    )
                    operator_evidence_sha256 = tuple(
                        item.evidence_sha256 for item in operator_evidence
                    )
                    structural_operator_receipt = _install_short_conv_operator(
                        target_model=target_model,
                        layer_index=layer_index,
                        evidence=operator_evidence,
                        source_fir_weight=source_fir_weight,
                        source_fir_bias=source_fir_bias,
                        cache=cache,
                        bridge=bridge,
                        installer=selected_short_installer,
                        parameter_count=parameter_count,
                        body_recipe=recipe,
                        short_conv_construction=construction,
                        fhvr_source=fhvr_source,
                        **(
                            {"fold_compute_device": fold_compute_device}
                            if fold_compute_device != "cpu"
                            else {}
                        ),
                    )
                    if _source_local_short_conv_output_required(recipe):
                        partition_batches = chart_receipt.get("partition_batch_indices")
                        fit_batch_indices = (
                            partition_batches.get("fit")
                            if isinstance(partition_batches, Mapping)
                            else None
                        )
                        if (
                            not isinstance(fit_batch_indices, list)
                            or not fit_batch_indices
                            or any(
                                isinstance(index, bool)
                                or not isinstance(index, int)
                                or index < 0
                                for index in fit_batch_indices
                            )
                        ):
                            raise Lfm2SemanticLayerStackRefoldError(
                                "Residual chart omitted its authenticated FIT "
                                "batches for source-local short-conv folding."
                            )
                        fit_batch_sample_weights = {
                            index: 1.0 for index in fit_batch_indices
                        }
                        structural_replay = _replay_current_layer(
                            target_model,
                            alignments=alignments,
                            layer_index=layer_index,
                            pre_residual_by_batch=residual_bank,
                        )
                        if (
                            structural_replay.captured_child_model_state_sha256
                            != structural_operator_receipt["model_state_after_sha256"]
                        ):
                            raise Lfm2SemanticLayerStackRefoldError(
                                f"Layer {layer_index} structural short-conv "
                                "replay is not fresh."
                            )
                        (
                            source_local_fit_cases,
                            source_local_fit_cases_receipt,
                        ) = _build_short_conv_source_local_writer_fit_cases(
                            alignments=alignments,
                            parent=parent,
                            structural_replay=structural_replay,
                            fit_batch_indices=fit_batch_indices,
                            fit_batch_sample_weights=fit_batch_sample_weights,
                            layer_index=layer_index,
                        )
                        (
                            source_local_compiled,
                            source_local_installation_receipt,
                        ) = _compile_install_short_conv_source_local_writer(
                            target_model=target_model,
                            layer_index=layer_index,
                            structural_operator_receipt=(structural_operator_receipt),
                            structural_replay=structural_replay,
                            fit_cases=source_local_fit_cases,
                            fit_cases_receipt=source_local_fit_cases_receipt,
                            residual_output_chart=chart,
                            parameter_count=parameter_count,
                            fold_compute_device=fold_compute_device,
                            body_recipe=recipe,
                        )
                        post_source_local_replay = _replay_current_layer(
                            target_model,
                            alignments=alignments,
                            layer_index=layer_index,
                            pre_residual_by_batch=residual_bank,
                        )
                        operator_receipt = (
                            _compose_source_local_short_conv_operator_receipt(
                                structural_operator_receipt=(
                                    structural_operator_receipt
                                ),
                                source_local_compiled=source_local_compiled,
                                source_local_installation_receipt=(
                                    source_local_installation_receipt
                                ),
                                fit_cases_receipt=(source_local_fit_cases_receipt),
                                structural_replay=structural_replay,
                                post_source_local_replay=(post_source_local_replay),
                                layer_index=layer_index,
                                parameter_count=parameter_count,
                                body_recipe=recipe,
                            )
                        )
                        after_operator_replay_override = post_source_local_replay
                        operator_evidence_sha256 = tuple(
                            operator_receipt["transport_evidence_sha256"]
                        )
                    else:
                        operator_receipt = structural_operator_receipt
                    short_conv_decision_count += 1
                    short_conv_install_count += int(
                        operator_receipt.get("installed") is True
                    )
                    short_conv_incumbent_noop_count += int(
                        operator_receipt.get("installed") is False
                    )
                operator_coverage_after = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind=operator_kind,
                )
                operator_coverage_decision = _validate_producer_coverage_decision(
                    layer_index=layer_index,
                    producer_kind=operator_kind,
                    before=operator_coverage_before,
                    after=operator_coverage_after,
                    installation=operator_receipt,
                )
                attention_producer_coverage_count += int(
                    operator_kind == "full_attention"
                )
                short_conv_producer_coverage_count += int(operator_kind == "short_conv")
                cold_start_producer_chain_count += int(
                    operator_coverage_decision["cold_start_or_partial_seed_at_entry"]
                )
                installed_complete_producer_chain_count += int(
                    operator_coverage_decision[
                        "installed_complete_structurally_nonzero_chain"
                    ]
                )
                retained_nonzero_seed_noop_count += int(
                    operator_coverage_decision[
                        "authenticated_nonzero_folded_seed_retained_by_noop"
                    ]
                )
                blank_or_partially_blank_producer_chain_count += int(
                    operator_coverage_decision[
                        "blank_or_partially_blank_required_role_count_after_decision"
                    ]
                    > 0
                )
                operator_parameter_mutation_count += int(
                    operator_receipt["model_state_before_sha256"]
                    != operator_receipt["model_state_after_sha256"]
                )
                after_operator_replay = (
                    after_operator_replay_override
                    if after_operator_replay_override is not None
                    else _replay_current_layer(
                        target_model,
                        alignments=alignments,
                        layer_index=layer_index,
                        pre_residual_by_batch=residual_bank,
                    )
                )
                if (
                    after_operator_replay.captured_child_model_state_sha256
                    != operator_receipt["model_state_after_sha256"]
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        f"Layer {layer_index} operator replay is not fresh."
                    )
                source_inventory, mlp_source_provenance = _direct_mlp_inventory(
                    inventory_type=mlp_inventory_type,
                    parent_session=parent_session,
                    cache=cache,
                    bridge=bridge,
                    layer_index=layer_index,
                    operator_kind=operator_kind,
                    residual_chart=chart,
                    chart_receipt=chart_receipt,
                    attention_lineage=attention_lineage,
                    alignments=alignments,
                    parent=parent,
                    target_model=target_model,
                )
                mlp_evidence = _build_mlp_evidence(
                    evidence_type=mlp_evidence_type,
                    cache=cache,
                    bridge=bridge,
                    interface=interface,
                    alignments=alignments,
                    parent=parent,
                    child_bank=after_operator_replay,
                    chart=chart,
                    source_inventory=source_inventory,
                    layer_index=layer_index,
                    final_layer=layer_index == terminal_layer_index,
                )
                mlp_evidence_compile_count += 1
                mlp_coverage_before = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind="mlp",
                    defer_down_weight=layer_index == terminal_layer_index,
                )
                mlp_receipt, terminal_evidence, terminal_w2_prior = _install_mlp(
                    target_model=target_model,
                    layer_index=layer_index,
                    evidence=mlp_evidence,
                    cache=cache,
                    bridge=bridge,
                    installer=selected_mlp_installer,
                    final_layer=layer_index == terminal_layer_index,
                    parameter_count=parameter_count,
                    fold_compute_device=fold_compute_device,
                    body_recipe=recipe,
                    normalized_input_correction_rho=(normalized_input_correction_rho),
                    normalized_input_request_capture_directory=(
                        normalized_input_request_capture_directory
                    ),
                    **completion_kwargs,
                )
                mlp_coverage_after = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind="mlp",
                    defer_down_weight=layer_index == terminal_layer_index,
                )
                mlp_coverage_decision = _validate_producer_coverage_decision(
                    layer_index=layer_index,
                    producer_kind="mlp",
                    before=mlp_coverage_before,
                    after=mlp_coverage_after,
                    installation=mlp_receipt,
                    defer_down_weight=layer_index == terminal_layer_index,
                )
                cold_start_producer_chain_count += int(
                    mlp_coverage_decision["cold_start_or_partial_seed_at_entry"]
                )
                installed_complete_producer_chain_count += int(
                    mlp_coverage_decision[
                        "installed_complete_structurally_nonzero_chain"
                    ]
                )
                retained_nonzero_seed_noop_count += int(
                    mlp_coverage_decision[
                        "authenticated_nonzero_folded_seed_retained_by_noop"
                    ]
                )
                blank_or_partially_blank_producer_chain_count += int(
                    mlp_coverage_decision[
                        "blank_or_partially_blank_required_role_count_after_decision"
                    ]
                    > 0
                )
                if layer_index < terminal_layer_index:
                    nonterminal_mlp_producer_coverage_count += 1
                    nonterminal_mlp_decision_count += 1
                    nonterminal_mlp_install_count += int(
                        mlp_receipt.get("installed") is True
                    )
                    nonterminal_mlp_incumbent_noop_count += int(
                        mlp_receipt.get("installed") is False
                        and mlp_receipt.get("status") == _MLP_STRUCTURAL_NOOP_STATUS
                    )
                else:
                    final_mlp_producer_coverage_count = 1
                    final_terminal_evidence = terminal_evidence
                    final_terminal_w2_prior = terminal_w2_prior
                    final_mlp_producer_install_count = int(
                        mlp_receipt.get("installed") is True
                    )
                    final_mlp_producer_incumbent_noop_count = int(
                        mlp_receipt.get("installed") is False
                        and mlp_receipt.get("status")
                        in {
                            _MLP_STRUCTURAL_NOOP_STATUS,
                            "prepared-final-w2-deferred",
                        }
                    )
                mlp_parameter_mutation_count += int(
                    mlp_receipt["model_state_before_sha256"]
                    != mlp_receipt["model_state_after_sha256"]
                )
                propagation_replay = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                if (
                    propagation_replay.captured_child_model_state_sha256
                    != mlp_receipt["model_state_after_sha256"]
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        f"Layer {layer_index} propagation replay is not fresh."
                    )
                if global_residual_closure_required:
                    _capture_global_child_layer_primals(
                        global_residual_accumulators,
                        target_model=target_model,
                        alignments=alignments,
                        replay_bank=propagation_replay,
                        layer_index=layer_index,
                        deferred_final_w2_prior=(
                            final_terminal_w2_prior
                            if layer_index == terminal_layer_index
                            else None
                        ),
                    )
                layer_receipts.append(
                    _layer_receipt(
                        layer_index=layer_index,
                        operator_kind=operator_kind,
                        before_operator_replay=before_operator_replay,
                        operator_receipt=operator_receipt,
                        operator_coverage_decision=operator_coverage_decision,
                        after_operator_replay=after_operator_replay,
                        mlp_receipt=mlp_receipt,
                        mlp_coverage_decision=mlp_coverage_decision,
                        mlp_source_provenance=mlp_source_provenance,
                        propagation_replay=propagation_replay,
                        chart_receipt=chart_receipt,
                        parent=parent,
                        operator_evidence_sha256=operator_evidence_sha256,
                        mlp_evidence_sha256=tuple(
                            item.evidence_sha256 for item in mlp_evidence
                        ),
                        fir_receipt=fir_receipt,
                        fold_compute_device=fold_compute_device,
                        body_recipe=recipe,
                        body_coordinate_completion_receipt=(
                            None
                            if body_coordinate_completion is None
                            else body_coordinate_completion.receipt
                        ),
                        body_coordinate_completion_candidate_receipt=(
                            None
                            if body_coordinate_completion_candidate is None
                            else body_coordinate_completion_candidate.report
                        ),
                        body_coordinate_completion_source=completion_source,
                    )
                )
                residual_bank = _next_residual_bank(propagation_replay)
            causal_stack_state = fingerprint_lfm2_parent(target_model).identity_sha256
            causal_stack_state_hashes = _state_tensor_hashes(target_model)
            causal_stack_changed_keys = _changed_state_keys(
                entry_state_hashes,
                causal_stack_state_hashes,
            )
            global_residual_installation = None
            global_residual_evidence_sha256 = None
            global_residual_closure_state = causal_stack_state
            global_residual_closure_changed_keys: tuple[str, ...] = ()
            if global_residual_closure_required:
                if final_terminal_w2_prior is None:
                    raise Lfm2SemanticLayerStackRefoldError(
                        "Coupled nonlinear Q-DEIM produced no deferred final W2 "
                        "prior for global residual closure."
                    )
                (
                    global_residual_installation,
                    global_residual_evidence,
                ) = (
                    _compile_install_global_residual_from_accumulators(
                        global_residual_accumulators,
                        target_model=target_model,
                        mapped_parent_tied_table=(effective_global_residual_tied_table),
                        deferred_final_w2_prior=final_terminal_w2_prior,
                        deferred_final_writer_parameter=final_w2,
                        closure_policy=closure_policy,
                        correction_scale_frontier=(
                            scale_policy.materialization_frontier
                        ),
                        correction_scale=scale_policy.materialization_scale,
                        materialization_compute_device=fold_compute_device,
                        requested_basis_rank=global_residual_basis_rank,
                        target_dtype_representability_level=(
                            target_dtype_representability_level
                        ),
                        source_local_boundaries=source_local_global_residual,
                    )
                    if producer_aware_closure
                    else _compile_install_global_residual_v1_diagnostic_from_accumulators(
                        global_residual_accumulators,
                        target_model=target_model,
                        mapped_parent_tied_table=(effective_global_residual_tied_table),
                        deferred_final_w2_prior=final_terminal_w2_prior,
                        deferred_final_writer_parameter=final_w2,
                        closure_policy=closure_policy,
                    )
                )
                global_residual_evidence_sha256 = (
                    global_residual_evidence.evidence_sha256
                )
                final_terminal_w2_prior = (
                    global_residual_installation.corrected_final_w2_prior
                )
                global_residual_closure_state = fingerprint_lfm2_parent(
                    target_model
                ).identity_sha256
                global_residual_closure_state_hashes = _state_tensor_hashes(
                    target_model
                )
                global_residual_closure_changed_keys = _changed_state_keys(
                    causal_stack_state_hashes,
                    global_residual_closure_state_hashes,
                )
                del global_residual_evidence
            cache_session.verify()
            cache_session_receipt = cache_session.as_dict()
            parent_session_receipt = dict(parent_session.receipt())
        final_state = fingerprint_lfm2_parent(target_model).identity_sha256
        final_state_hashes = _state_tensor_hashes(target_model)
        changed_keys = _changed_state_keys(entry_state_hashes, final_state_hashes)
        final_w2_after = _tensor_sha256(final_w2)
        terminal_evidence_sha = {
            split: value.authentication_sha256
            for split, value in sorted(final_terminal_evidence.items())
        }
        composed_transition_changed_keys = tuple(
            sorted(
                set(causal_stack_changed_keys).union(
                    global_residual_closure_changed_keys
                )
            )
        )
        global_residual_report: Mapping[str, Any] | None = None
        global_residual_report_changed_keys: tuple[str, ...] = ()
        if global_residual_installation is not None:
            global_residual_report = global_residual_installation.report
            before_by_writer = global_residual_report.get(
                "dense_live_physical_sha256_before"
                if producer_aware_closure
                else "dense_live_writer_sha256_before"
            )
            after_by_writer = global_residual_report.get(
                "dense_live_physical_sha256_after"
                if producer_aware_closure
                else "dense_live_writer_sha256_after"
            )
            if isinstance(before_by_writer, Mapping) and isinstance(
                after_by_writer,
                Mapping,
            ):
                global_residual_report_changed_keys = tuple(
                    sorted(
                        name
                        for name in before_by_writer
                        if before_by_writer.get(name) != after_by_writer.get(name)
                    )
                )
        producer_aware_report_valid = True
        output_writer_diagnostic_report_valid = True
        if global_residual_report is not None:
            if producer_aware_closure:
                physical_ids = tuple(_global_live_physical_state(target_model))
                topology_receipt = global_residual_report.get("topology_receipt")
                topology_body = (
                    dict(topology_receipt)
                    if isinstance(topology_receipt, Mapping)
                    else {}
                )
                topology_sha = topology_body.pop("topology_sha256", None)
                site_specifications = global_residual_report.get("site_specifications")
                correction_policy = global_residual_report.get(
                    "correction_scale_policy"
                )
                materialization_compute_receipt = global_residual_report.get(
                    "materialization_compute_execution_receipt"
                )
                materialization_compute_body = (
                    dict(materialization_compute_receipt)
                    if isinstance(materialization_compute_receipt, Mapping)
                    else {}
                )
                materialization_compute_sha = materialization_compute_body.pop(
                    "report_sha256",
                    None,
                )
                dense_transaction_receipt = global_residual_report.get(
                    "dense_transaction_receipt"
                )
                expected_producer_aware_backend = (
                    "producer_aware_source_local_v2b"
                    if source_local_global_residual
                    else _GLOBAL_RESIDUAL_PRODUCER_AWARE_V2A
                )
                producer_aware_report_valid = bool(
                    global_residual_report.get("closure_backend")
                    == expected_producer_aware_backend
                    and (
                        (
                            global_residual_report.get("schema")
                            == LFM2_PRODUCER_AWARE_SOURCE_LOCAL_SEMANTIC_INTEGRATION_SCHEMA
                            and global_residual_report.get("boundary_mode")
                            == LFM2_PRODUCER_AWARE_SOURCE_LOCAL_BOUNDARY_MODE
                            and global_residual_report.get("source_local_boundary_mode")
                            is True
                            and global_residual_report.get(
                                "absolute_endpoint_boundary_mode"
                            )
                            is False
                            and global_residual_report.get(
                                "all_global_residual_defects_are_source_local_actions"
                            )
                            is True
                            and global_residual_report.get(
                                "absolute_parent_endpoint_defects_used"
                            )
                            is False
                            and global_residual_report.get(
                                "accumulated_residual_origin_catchup_used"
                            )
                            is False
                            and global_residual_report.get(
                                "source_local_terminal_action_solver_objective_active"
                            )
                            is True
                            and global_residual_report.get(
                                "linearization_boundary_mode_lineage_bound"
                            )
                            is True
                        )
                        if source_local_global_residual
                        else global_residual_report.get("boundary_mode") is None
                    )
                    and global_residual_report.get("global_residual_backend")
                    == closure_policy.backend
                    and global_residual_report.get("closure_policy")
                    == closure_policy.as_dict()
                    and global_residual_report.get("closure_policy_sha256")
                    == closure_policy.authentication_sha256
                    and materialization_compute_sha
                    == canonical_sha256(materialization_compute_body)
                    and materialization_compute_sha
                    == global_residual_report.get(
                        "materialization_compute_execution_receipt_sha256"
                    )
                    and materialization_compute_body.get("requested_device")
                    == fold_compute_device
                    and global_residual_report.get("materialization_compute_device")
                    == fold_compute_device
                    and isinstance(dense_transaction_receipt, Mapping)
                    and dense_transaction_receipt.get(
                        "compute_execution_receipt_sha256"
                    )
                    == materialization_compute_sha
                    and dense_transaction_receipt.get("compute_execution_receipt")
                    == materialization_compute_receipt
                    and global_residual_report.get("layer_count") == len(kinds)
                    and global_residual_report.get("writer_site_count")
                    == 4 * len(kinds)
                    and global_residual_report.get("four_sites_per_layer") is True
                    and global_residual_report.get(
                        "dynamic_layer_count_derived_from_authenticated_evidence"
                    )
                    is True
                    and isinstance(site_specifications, list)
                    and len(site_specifications) == 4 * len(kinds)
                    and topology_sha == canonical_sha256(topology_body)
                    and topology_sha == global_residual_report.get("topology_sha256")
                    and global_residual_report.get("installed_parameter_ids")
                    == list(physical_ids)
                    and global_residual_report.get("installed_parameter_count")
                    == len(physical_ids)
                    and global_residual_report.get(
                        "all_nondeferred_physical_parameters_installed"
                    )
                    is True
                    and isinstance(correction_policy, Mapping)
                    and correction_policy.get("authentication_sha256")
                    == scale_policy.authentication_sha256
                    and correction_policy.get("materialization_frontier")
                    == list(scale_policy.materialization_frontier)
                    and correction_policy.get("materialization_scale")
                    == scale_policy.materialization_scale
                    and global_residual_report.get("correction_scale_policy_sha256")
                    == scale_policy.authentication_sha256
                    and global_residual_report.get("materialization_scale")
                    == scale_policy.materialization_scale
                    and global_residual_report.get("materialization_frontier")
                    == list(scale_policy.materialization_frontier)
                    and global_residual_report.get(
                        "target_dtype_representability_level"
                    )
                    == str(target_dtype_representability_level).strip().lower()
                    and isinstance(
                        global_residual_report.get(
                            "target_dtype_representability_frontier"
                        ),
                        list,
                    )
                    and global_residual_report.get(
                        "materialization_scale_is_execution_variant_not_winner"
                    )
                    is True
                    and global_residual_report.get("frontier_member_selected_as_winner")
                    is False
                    and global_residual_report.get(
                        "separate_fold_decode_retain_required_for_each_frontier_scale"
                    )
                    is True
                    and global_residual_report.get("final_w2_parameter_id")
                    == f"model.layers.{terminal_layer_index}.feed_forward.w2.weight"
                    and global_residual_report.get("final_w2_unchanged") is True
                    and global_residual_report.get("numeric_or_manual_quality_gate")
                    is False
                    and global_residual_report.get(
                        "all_frontier_candidates_eligible_for_decoded_behavioral_review"
                    )
                    is True
                    and global_residual_report.get("autograd_grad_calls") == 0
                    and global_residual_report.get("backward_calls") == 0
                    and global_residual_report.get("optimizer_steps") == 0
                    and global_residual_report.get("training_steps") == 0
                )
            else:
                output_writer_diagnostic_report_valid = bool(
                    global_residual_report.get("closure_scope", "").endswith(
                        "output_writers_only_baseline_v1"
                    )
                    and global_residual_report.get(
                        "v2_input_transport_extension_roadmap", {}
                    ).get("implemented_in_this_result")
                    is False
                )
        if (
            attention_decision_count != len(attention_layer_indices)
            or attention_install_count + attention_incumbent_noop_count
            != len(attention_layer_indices)
            or short_conv_decision_count != len(short_conv_layer_indices)
            or short_conv_install_count + short_conv_incumbent_noop_count
            != len(short_conv_layer_indices)
            or mlp_evidence_compile_count != layer_count
            or nonterminal_mlp_decision_count != nonterminal_layer_count
            or (
                nonterminal_mlp_install_count + nonterminal_mlp_incumbent_noop_count
                != nonterminal_layer_count
            )
            or final_mlp_producer_install_count
            + final_mlp_producer_incumbent_noop_count
            != 1
            or attention_producer_coverage_count != len(attention_layer_indices)
            or short_conv_producer_coverage_count != len(short_conv_layer_indices)
            or nonterminal_mlp_producer_coverage_count != nonterminal_layer_count
            or final_mlp_producer_coverage_count != 1
            or blank_or_partially_blank_producer_chain_count != 0
            or installed_complete_producer_chain_count
            + retained_nonzero_seed_noop_count
            != producer_chain_count
            or retained_nonzero_seed_noop_count
            != (
                attention_incumbent_noop_count
                + short_conv_incumbent_noop_count
                + nonterminal_mlp_incumbent_noop_count
                + final_mlp_producer_incumbent_noop_count
            )
            or cold_start_producer_chain_count > installed_complete_producer_chain_count
            or operator_parameter_mutation_count
            != attention_install_count + short_conv_install_count
            or mlp_parameter_mutation_count
            != nonterminal_mlp_install_count + final_mlp_producer_install_count
            or set(final_terminal_evidence) != set(_PARTITIONS)
            or operator_parameter_mutation_count + mlp_parameter_mutation_count <= 0
            or not changed_keys
            or final_state == snapshot.fingerprint
            or _parameter_count(target_model) != parameter_count
            or _parameter_topology(target_model) != snapshot.topology
            or not _gradient_state_matches_snapshot(target_model, snapshot)
            or set(final_state_hashes) != set(snapshot.state)
            or final_w2_after != final_w2_before
            or causal_stack_state != layer_receipts[-1]["model_state_after_mlp_sha256"]
            or (
                not global_residual_closure_required
                and (
                    global_residual_installation is not None
                    or global_residual_row_map_receipt is not None
                    or causal_stack_state != final_state
                    or causal_stack_changed_keys != changed_keys
                    or global_residual_closure_changed_keys
                )
            )
            or (
                global_residual_closure_required
                and (
                    global_residual_installation is None
                    or global_residual_report is None
                    or global_residual_row_map_receipt is None
                    or global_residual_evidence_sha256 is None
                    or not global_residual_installation.verify_authentication()
                    or global_residual_closure_state != final_state
                    or global_residual_report.get("evidence_package_sha256")
                    != global_residual_evidence_sha256
                    or global_residual_report.get(
                        "final_layer_mlp_down_weight_prior_authentication_sha256"
                    )
                    != final_terminal_w2_prior.authentication_sha256
                    or global_residual_report.get(
                        "final_layer_mlp_down_weight_prior_tensor_sha256"
                    )
                    != lfm2_cpcf_terminal_action_tensor_sha256(
                        final_terminal_w2_prior.down_weight
                    )
                    or global_residual_report.get(
                        "final_layer_mlp_down_weight_prior_install_count"
                    )
                    != 0
                    or global_residual_report.get("quality_gate") is not False
                    or global_residual_report.get(
                        "numeric_diagnostics_used_for_auto_veto"
                    )
                    is not False
                    or not producer_aware_report_valid
                    or not output_writer_diagnostic_report_valid
                    or global_residual_report_changed_keys
                    != global_residual_closure_changed_keys
                    or not set(changed_keys).issubset(composed_transition_changed_keys)
                )
            )
            or tuple(
                (str(name), bool(module.training))
                for name, module in target_model.named_modules()
            )
            != snapshot.modes
            or any(
                value.get("body_recipe_id") != recipe.recipe_id
                or value.get("body_recipe_sha256") != recipe.recipe_sha256
                or value.get("body_recipe") != recipe.as_dict()
                or value.get("numeric_thresholds_used_for_candidate_admission")
                is not False
                or value.get("heldout_metrics_have_candidate_or_publication_authority")
                is not False
                for value in layer_receipts
            )
            or any(
                value.get("mlp_fold_compute_device") != fold_compute_device
                for value in layer_receipts
            )
            or (
                recipe.component_lanes["mlp"] in _GLOBAL_RESIDUAL_MLP_LANES
                and any(
                    value.get("mlp_normalized_decoder_control_compute_device") != "cpu"
                    or value.get("mlp_qdeim_pod_and_pivot_compute_device")
                    != fold_compute_device
                    or not isinstance(
                        value.get("mlp_qdeim_compute_execution_sha256"),
                        str,
                    )
                    for value in layer_receipts
                )
            )
            or (
                recipe.component_lanes["qk"] == _COUPLED_ATTENTION_QDEIM_LANE
                and any(
                    value.get("attention_qdeim_fold_compute_device")
                    != fold_compute_device
                    or value.get("attention_qdeim_pod_and_pivot_compute_device")
                    != fold_compute_device
                    or not isinstance(
                        value.get("attention_qdeim_compute_execution_sha256"),
                        str,
                    )
                    for value in layer_receipts
                    if value.get("operator_kind") == "full_attention"
                )
            )
            or (
                fold_compute_device != "cpu"
                and recipe.component_lanes["mlp"] not in _GLOBAL_RESIDUAL_MLP_LANES
                and any(
                    not isinstance(
                        value.get("mlp_weight_signature_compute_execution_sha256"),
                        str,
                    )
                    for value in layer_receipts
                )
            )
            or (
                fold_compute_device != "cpu"
                and any(
                    value.get("short_conv_fold_compute_device") != fold_compute_device
                    or not isinstance(
                        value.get(
                            "short_conv_structural_cluster_compute_execution_sha256"
                        ),
                        str,
                    )
                    or not isinstance(
                        value.get("short_conv_writer_compute_execution_sha256"),
                        str,
                    )
                    for value in layer_receipts
                    if value.get("operator_kind") == "short_conv"
                )
            )
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Complete layer-stack postconditions were not satisfied."
            )
        if (
            construction.uses_fhvr_full_coverage
            or construction.uses_body_global_coordinate_completion
        ) and any(
            value.get("short_conv_construction") != construction.as_dict()
            or value.get("short_conv_construction_sha256")
            != construction.construction_sha256
            or (
                construction.uses_fhvr_full_coverage
                and (
                    not isinstance(value.get("short_conv_fhvr_source_sha256"), str)
                    or not isinstance(
                        value.get("short_conv_fhvr_candidate_state_sha256"),
                        str,
                    )
                )
            )
            or (
                construction.uses_body_global_coordinate_completion
                and (
                    not isinstance(
                        value.get("short_conv_coordinate_source_sha256"), str
                    )
                    or value.get("short_conv_shared_body_global_completion_sha256")
                    != body_coordinate_completion_candidate.report.get("result_sha256")
                )
            )
            for value in layer_receipts
            if value.get("operator_kind") == "short_conv"
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Complete layer stack dropped its requested short-conv "
                "construction root."
            )
        body_coordinate_completion_fields: dict[str, Any] = {}
        if coordinate_completion_enabled:
            if (
                body_coordinate_completion is None
                or body_coordinate_completion_candidate is None
                or body_coordinate_fit_rows_receipt is None
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Complete layer stack did not retain its body-global "
                    "coordinate completion."
                )
            wrapper_receipt = _verify_sealed(
                body_coordinate_completion.receipt,
                "body-global coordinate completion",
            )
            candidate_receipt = _verify_sealed(
                body_coordinate_completion_candidate.report,
                "body-global coordinate completion candidate",
            )
            if completion_source is None:
                raise Lfm2SemanticLayerStackRefoldError(
                    "Complete layer stack dropped its body-global completion source."
                )
            completion_geometry_fields = _body_coordinate_geometry_receipt_fields(
                wrapper_receipt,
                source=completion_source,
            )
            fit_rows_receipt = _verify_sealed(
                body_coordinate_fit_rows_receipt,
                "body-global FIT residual rows",
            )
            candidate_sha256 = candidate_receipt["result_sha256"]
            public_map_sha256 = wrapper_receipt["public_coordinate_map_sha256"]
            analytic_map_sha256 = candidate_receipt["coordinate_map_sha256"]
            layer_candidate_sha256s = {
                value.get("shared_body_global_completion_sha256")
                for value in layer_receipts
            }
            if (
                wrapper_receipt.get("selected_candidate_result_sha256")
                != candidate_sha256
                or wrapper_receipt.get("coordinate_map_sha256") != analytic_map_sha256
                or tuple(wrapper_receipt.get("parent_operator_sha256s", ()))
                != completion_source.ordered_operator_sha256s
                or wrapper_receipt.get("parent_operator_manifest_sha256")
                != completion_source.operator_manifest_sha256
                or wrapper_receipt.get("fit_child_residual_rows_public_sha256")
                != fit_rows_receipt.get("fit_child_residual_rows_sha256")
                or wrapper_receipt.get("fit_parent_residual_rows_public_sha256")
                != fit_rows_receipt.get("fit_parent_residual_rows_sha256")
                or layer_candidate_sha256s != {candidate_sha256}
                or any(
                    value.get("body_global_coordinate_completion_sha256")
                    != wrapper_receipt["result_sha256"]
                    or value.get("body_global_raw_decoder_public_sha256")
                    != public_map_sha256
                    or value.get("body_global_coordinate_map_sha256")
                    != analytic_map_sha256
                    or value.get("normalized_input_correction_rho")
                    != normalized_input_correction_rho
                    or value.get("body_global_global_gauge_preserved") is not True
                    for value in layer_receipts
                )
                or any(
                    any(
                        value.get(key) != expected
                        for key, expected in completion_geometry_fields.items()
                    )
                    for value in layer_receipts
                )
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Complete layer stack did not share one authenticated "
                    "body-global coordinate completion across every layer."
                )
            body_coordinate_completion_fields = {
                "body_global_coordinate_completion_enabled": True,
                "body_global_coordinate_completion_compile_count": 1,
                "body_global_coordinate_completion_compile_layer_index": 0,
                "body_global_coordinate_completion_recipe": completion_recipe,
                "body_global_coordinate_completion_receipt": wrapper_receipt,
                "body_global_coordinate_completion_sha256": wrapper_receipt[
                    "result_sha256"
                ],
                "body_global_coordinate_completion_candidate_receipt": (
                    candidate_receipt
                ),
                "shared_body_global_completion_sha256": candidate_sha256,
                "body_global_raw_decoder_public_sha256": public_map_sha256,
                "body_global_coordinate_map_sha256": analytic_map_sha256,
                "normalized_input_correction_rho": (normalized_input_correction_rho),
                "body_global_fit_residual_rows_receipt": fit_rows_receipt,
                "body_global_fit_residual_rows_sha256": fit_rows_receipt[
                    "result_sha256"
                ],
                "body_global_parent_operator_count": len(
                    completion_source.ordered_operator_sha256s
                ),
                "body_global_parent_operator_sha256s": list(
                    completion_source.ordered_operator_sha256s
                ),
                "body_global_parent_operator_manifest_sha256": wrapper_receipt[
                    "parent_operator_manifest_sha256"
                ],
                "body_global_completion_layer_binding_count": len(layer_receipts),
                "body_global_completion_shared_across_every_layer": True,
                "body_global_completion_shared_across_attention_mlp_and_short_conv": (
                    True
                ),
                "all_layer_residual_charts_preserve_global_gauge": True,
                "body_global_completion_fit_partition_only": True,
                "body_global_completion_training_steps": 0,
                **completion_geometry_fields,
            }
        body = {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            **body_coordinate_completion_fields,
            **(
                {
                    "short_conv_construction": construction.as_dict(),
                    "short_conv_construction_id": construction.construction_id,
                    "short_conv_construction_sha256": (
                        construction.construction_sha256
                    ),
                    "all_short_conv_layers_bound_to_requested_construction": True,
                }
                if (
                    construction.uses_fhvr_full_coverage
                    or construction.uses_body_global_coordinate_completion
                )
                else {}
            ),
            "status": "complete-causal-layer-stack-adjudicated-final-w2-deferred",
            "cache_key": cache.cache_key,
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "cache_layer_topology_sha256": cache.layer_topology_fingerprint,
            "cache_selective_session_sha256": cache_session_receipt["session_sha256"],
            "cache_selective_session": dict(cache_session_receipt),
            "tokenizer_session_receipt_sha256": tokenizer_session_receipt[
                "receipt_sha256"
            ],
            "tokenizer_session": dict(tokenizer_session_receipt),
            "parent_checkpoint_session_sha256": parent_session_receipt[
                "session_sha256"
            ],
            "parent_checkpoint_session": parent_session_receipt,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "tied_interface_lineage": dict(interface_lineage),
            "semantic_refinement_audit_sha256": refinement["report_sha256"],
            "semantic_alignment_sha256": [
                value.alignment_sha256 for value in alignments
            ],
            "direct_attention_source_receipt_sha256": {
                str(index): direct_attention_receipts[index]["result_sha256"]
                for index in attention_layer_indices
            },
            "layer_installation_sha256": [
                value["result_sha256"] for value in layer_receipts
            ],
            "layer_installations": [dict(value) for value in layer_receipts],
            "layer_count": layer_count,
            "operator_layer_kinds": list(kinds),
            "attention_layer_indices": list(attention_layer_indices),
            "short_conv_layer_indices": list(short_conv_layer_indices),
            "terminal_layer_index": terminal_layer_index,
            "dynamic_layer_count_derived_from_authenticated_cache_and_child_topology": (
                True
            ),
            "attention_operator_decision_count": attention_decision_count,
            "attention_operator_install_count": attention_install_count,
            "attention_operator_incumbent_noop_count": (attention_incumbent_noop_count),
            "short_conv_operator_decision_count": short_conv_decision_count,
            "short_conv_operator_install_count": short_conv_install_count,
            "short_conv_operator_incumbent_noop_count": (
                short_conv_incumbent_noop_count
            ),
            "operator_parameter_mutation_count": (operator_parameter_mutation_count),
            "replay_fresh_mlp_evidence_compile_count": (mlp_evidence_compile_count),
            "nonterminal_mlp_writer_decision_count": nonterminal_mlp_decision_count,
            "nonterminal_mlp_writer_install_count": (nonterminal_mlp_install_count),
            "nonterminal_mlp_writer_incumbent_noop_count": (
                nonterminal_mlp_incumbent_noop_count
            ),
            "final_mlp_producer_install_count": final_mlp_producer_install_count,
            "final_mlp_producer_incumbent_noop_count": (
                final_mlp_producer_incumbent_noop_count
            ),
            "attention_producer_structural_coverage_count": (
                attention_producer_coverage_count
            ),
            "short_conv_producer_structural_coverage_count": (
                short_conv_producer_coverage_count
            ),
            "nonterminal_mlp_producer_structural_coverage_count": (
                nonterminal_mlp_producer_coverage_count
            ),
            "final_mlp_producer_structural_coverage_count": (
                final_mlp_producer_coverage_count
            ),
            "cold_start_or_partial_producer_chain_count": (
                cold_start_producer_chain_count
            ),
            "installed_complete_producer_chain_count": (
                installed_complete_producer_chain_count
            ),
            "authenticated_nonzero_folded_seed_retained_noop_count": (
                retained_nonzero_seed_noop_count
            ),
            "blank_or_partially_blank_producer_chain_count": (
                blank_or_partially_blank_producer_chain_count
            ),
            "mlp_parameter_mutation_count": mlp_parameter_mutation_count,
            "heldout_selected_incumbent_noop_count": sum(
                int(value["operator_incumbent_noop_selected"])
                + int(value["mlp_incumbent_noop_selected"])
                for value in layer_receipts
            ),
            "complete_causal_layer_stack_adjudicated": True,
            "complete_nonzero_structural_producer_coverage": True,
            "all_producer_chains_installed_or_retained_nonzero_seed": True,
            "complete_causal_layer_stack_installed": (
                attention_install_count == attention_decision_count
                and short_conv_install_count == short_conv_decision_count
                and nonterminal_mlp_install_count == nonterminal_mlp_decision_count
                and final_mlp_producer_install_count == 1
            ),
            "all_attention_operator_decisions_completed": True,
            "all_attention_operators_installed": (
                attention_install_count == attention_decision_count
            ),
            "all_short_conv_operator_decisions_completed": True,
            "all_short_conv_operators_installed": (
                short_conv_install_count == short_conv_decision_count
            ),
            "all_replay_fresh_mlp_evidence_compiled": True,
            "all_nonterminal_mlp_writer_decisions_completed": True,
            "all_nonterminal_mlp_writers_installed": (
                nonterminal_mlp_install_count == nonterminal_mlp_decision_count
            ),
            "authenticated_incumbent_noop_layer_decisions_permitted": True,
            "whole_stage_contains_meaningful_parameter_changes": True,
            "final_layer_mlp_w2_deferred_to_fused_terminal": True,
            "final_layer_mlp_w2_install_count": 0,
            "final_layer_mlp_evidence_sha256": terminal_evidence_sha,
            "causal_install_order": ("operator_then_replay_fresh_mlp_then_next_layer"),
            "model_state_before_sha256": snapshot.fingerprint,
            "model_state_after_sha256": final_state,
            "changed_state_keys": list(changed_keys),
            "parameter_element_count": parameter_count,
            "parameter_count_changed": False,
            "parameter_topology_changed": False,
            "deferred_final_w2_sha256": final_w2_before,
            **(
                {
                    "final_layer_mlp_down_weight_prior_authentication_sha256": (
                        final_terminal_w2_prior.authentication_sha256
                    ),
                    "final_layer_mlp_down_weight_prior_tensor_sha256": (
                        lfm2_cpcf_terminal_action_tensor_sha256(
                            final_terminal_w2_prior.down_weight
                        )
                    ),
                    "final_layer_mlp_down_weight_prior_live_w2_sha256": (
                        final_terminal_w2_prior.live_deferred_w2_sha256
                    ),
                    "final_layer_mlp_down_weight_prior_installed_before_terminal": (
                        False
                    ),
                    "final_layer_mlp_down_weight_prior_install_count": 0,
                }
                if final_terminal_w2_prior is not None
                else {}
            ),
            **(
                {
                    "global_residual_closure_required_by_body_recipe": True,
                    "global_residual_closure_applied": True,
                    "global_residual_closure_version": closure_version,
                    "global_residual_backend": closure_policy.backend,
                    "global_residual_correction_scale_frontier": list(
                        scale_policy.materialization_frontier
                    ),
                    "global_residual_correction_scale": (
                        scale_policy.materialization_scale
                    ),
                    "global_residual_requested_basis_rank": (
                        global_residual_basis_rank
                    ),
                    **(
                        {
                            "global_residual_target_dtype_representability_level": (
                                str(target_dtype_representability_level).strip().lower()
                            ),
                            "global_residual_target_dtype_representability_ray_multiplier": (  # noqa: E501
                                global_residual_report[
                                    "target_dtype_representability_ray_multiplier"
                                ]
                            ),
                            "global_residual_target_dtype_representability_frontier": list(  # noqa: E501
                                global_residual_report[
                                    "target_dtype_representability_frontier"
                                ]
                            ),
                        }
                        if producer_aware_closure
                        else {}
                    ),
                    "global_residual_correction_scale_policy_sha256": (
                        scale_policy.authentication_sha256
                    ),
                    "global_residual_closure_policy_sha256": (
                        closure_policy.authentication_sha256
                    ),
                    "global_residual_closure_scope": (
                        global_residual_report["closure_scope"]
                    ),
                    "global_residual_closure_evidence_sha256": (
                        global_residual_evidence_sha256
                    ),
                    "global_residual_semantic_row_map_sha256": (
                        global_residual_row_map_receipt["result_sha256"]
                    ),
                    "global_residual_semantic_row_maps": dict(
                        global_residual_row_map_receipt
                    ),
                    "global_residual_closure_receipt_sha256": (
                        global_residual_report["report_sha256"]
                    ),
                    "global_residual_closure_receipt": dict(global_residual_report),
                    "global_residual_closure_result_sha256": (
                        global_residual_report["closure_result_sha256"]
                    ),
                    "global_residual_corrected_final_w2_prior_sha256": (
                        final_terminal_w2_prior.authentication_sha256
                    ),
                    "global_residual_corrected_final_w2_closure_lineage_sha256": (
                        global_residual_report[
                            "corrected_final_w2_closure_lineage_sha256"
                        ]
                    ),
                    "final_layer_mlp_down_weight_prior_closure_lineage_sha256": (
                        global_residual_report[
                            "final_layer_mlp_down_weight_prior_closure_lineage_sha256"
                        ]
                    ),
                    "global_residual_tied_table_source": (
                        global_residual_tied_table_source
                    ),
                    "causal_layer_stack_model_state_after_sha256": (causal_stack_state),
                    "causal_layer_stack_changed_state_keys": list(
                        causal_stack_changed_keys
                    ),
                    "global_residual_closure_model_state_before_sha256": (
                        causal_stack_state
                    ),
                    "global_residual_closure_model_state_after_sha256": (
                        global_residual_closure_state
                    ),
                    "global_residual_closure_changed_state_keys": list(
                        global_residual_closure_changed_keys
                    ),
                    "composed_transition_changed_state_keys": list(
                        composed_transition_changed_keys
                    ),
                    "global_residual_closure_live_final_w2_unchanged": True,
                    "global_residual_closure_parent_histories_compacted": True,
                    **(
                        {
                            "global_residual_producer_aware_v2a": (
                                closure_version == _GLOBAL_RESIDUAL_PRODUCER_AWARE_V2A
                            ),
                            "global_residual_producer_aware_source_local_v2b": (
                                closure_version
                                == _GLOBAL_RESIDUAL_PRODUCER_AWARE_SOURCE_LOCAL_V2B
                            ),
                            "global_residual_boundary_mode": (
                                global_residual_report.get("boundary_mode")
                            ),
                            "global_residual_source_local_action_boundaries": (
                                source_local_global_residual
                            ),
                            "global_residual_absolute_endpoint_boundaries": (
                                not source_local_global_residual
                            ),
                            "global_residual_v1_diagnostic_baseline": False,
                            "global_residual_dynamic_layer_count": len(kinds),
                            "global_residual_producer_aware_site_count": (
                                4 * len(kinds)
                            ),
                            "global_residual_full_physical_parameter_transaction": (
                                True
                            ),
                            "global_residual_input_and_output_producer_roles_folded": (
                                True
                            ),
                            "global_residual_materialization_scale_is_execution_variant_not_winner": (
                                True
                            ),
                            "global_residual_frontier_member_selected_as_winner": False,
                            "global_residual_each_frontier_scale_requires_separate_fold_decode_retain": (
                                True
                            ),
                            "global_residual_materialization_compute_device": (
                                global_residual_report["materialization_compute_device"]
                            ),
                            "global_residual_materialization_compute_execution_receipt_sha256": (
                                global_residual_report[
                                    "materialization_compute_execution_receipt_sha256"
                                ]
                            ),
                            "global_residual_materialization_compute_execution_receipt": dict(
                                global_residual_report[
                                    "materialization_compute_execution_receipt"
                                ]
                            ),
                        }
                        if producer_aware_closure
                        else {
                            "global_residual_producer_aware_v2a": False,
                            "global_residual_producer_aware_source_local_v2b": False,
                            "global_residual_boundary_mode": None,
                            "global_residual_source_local_action_boundaries": False,
                            "global_residual_absolute_endpoint_boundaries": True,
                            "global_residual_v1_diagnostic_baseline": True,
                            "global_residual_v1_shared_attention_head_dimension": True,
                            "global_residual_general_per_site_head_dimensions_claimed": (
                                False
                            ),
                        }
                    ),
                    "global_residual_closure_numeric_quality_gate": False,
                    "global_residual_closure_manual_quality_gate": False,
                    "decoded_behavior_review_is_evidence_for_understanding": True,
                    "decoded_behavior_verdict_required_for_continuation": False,
                }
                if global_residual_installation is not None
                and global_residual_report is not None
                and global_residual_row_map_receipt is not None
                else {}
            ),
            "one_selective_cache_session_for_complete_stack": True,
            "selective_cache_session_owned_by_layer_stack": (
                authenticated_cache_session is None
            ),
            "selective_cache_session_reused_from_pipeline": (
                authenticated_cache_session is not None
            ),
            "one_parent_checkpoint_hash_and_open_session": True,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "child_model_forward_calls": 0,
            "child_macrocell_replay_only": True,
            "stack_compute_device_router_present": True,
            "fold_compute_device": fold_compute_device,
            "all_mlp_layers_proved_requested_fold_compute_device": True,
            **(
                {
                    "all_mlp_qdeim_pod_and_pivot_layers_proved_requested_fold_compute_device": (
                        True
                    ),
                    "requested_fold_compute_device_scope_for_qdeim_mlp": (
                        "qdeim-pod-pivot-and-dense-materialization"
                    ),
                    "mlp_normalized_decoder_control_compute_device": "cpu",
                }
                if recipe.component_lanes["mlp"] in _GLOBAL_RESIDUAL_MLP_LANES
                else {}
            ),
            **(
                {
                    "all_attention_qdeim_pod_and_pivot_layers_proved_requested_fold_compute_device": (
                        True
                    ),
                    "requested_fold_compute_device_scope_for_qdeim_attention": (
                        "group-pod-pivot-interpolation-and-dense-materialization"
                    ),
                }
                if recipe.component_lanes["qk"] == _COUPLED_ATTENTION_QDEIM_LANE
                else {}
            ),
            **(
                {"all_short_conv_layers_proved_requested_fold_compute_device": True}
                if fold_compute_device != "cpu"
                else {}
            ),
            "attention_solver_device_policy_delegated_to_attention_adapter": True,
            "final_norm_target_uses_unrotated_authenticated_interface": True,
            "short_conv_inserted_owner_rows_are_proxies_only": True,
            "numeric_parent_child_weight_match_objective": False,
            "numeric_thresholds_used_for_candidate_admission": False,
            "heldout_metrics_have_candidate_or_publication_authority": False,
            "fit_metrics_order_candidates": recipe.fit_metrics_order_candidates,
            "incumbent_fallback_only_when_all_alternatives_invalid_or_identical": True,
            "complete_candidate_frontiers_embedded_in_layer_receipts": True,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        receipt = _sealed(body)
        result = Lfm2SemanticLayerStackInstallation(
            receipt=receipt,
            final_layer_mlp_evidence=final_terminal_evidence,
            final_layer_mlp_down_weight_prior=final_terminal_w2_prior,
        )
    except BaseException as exc:
        try:
            snapshot.restore(target_model)
        except BaseException as rollback_exc:
            raise Lfm2SemanticLayerStackRefoldError(
                "Layer-stack transaction failed and exact whole-model rollback "
                "could not be proven."
            ) from rollback_exc
        raise Lfm2SemanticLayerStackRefoldError(
            "Layer-stack transaction failed; exact entry bytes were restored."
        ) from exc
    if result is None:  # pragma: no cover - defensive totality guard
        raise AssertionError("Layer-stack transaction produced no typed result.")
    return result


__all__ = [
    "LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA",
    "Lfm2SemanticLayerStackRefoldError",
    "compile_install_lfm2_semantic_layer_stack_postfold",
]
