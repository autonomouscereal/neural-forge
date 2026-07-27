"""Post-target-lattice Q/K output-fold primitives and diagnostic adapter.

The production path is
``hf_lfm2_causal_post_qk_output_fold.compile_install_lfm2_causal_post_qk_output_fold``.
The legacy adapter in this module accepts caller-materialized replay banks and
compiles every O projection against one common pre-install model state.  It is
retained as a diagnostic/authentication fixture and as the shared Q/K seed and
direction-materialization primitive library; it must not publish a production
child because upstream O changes would make downstream replay contexts stale.

The saved-body target-lattice pass deliberately changes only Q/K.  This
adapter consumes that sealed whole-body receipt, authenticates the exact live
treated checkpoint, reuses the existing bounded source-local O compiler, and
publishes every attention ``out_proj.weight`` in one atomic transaction.

No model is executed here.  Parent local-action and child pre-O rows are
already-authenticated fold evidence.  Q/K/V, normalization, topology,
Parameter objects/storage, gradients, and all unrelated state are frozen.
Numeric diagnostics remain advisory; decoded behavior is evaluated later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_state,
    run_fold_only_callback,
)
from neural_forge.hf_lfm2_attention_group_qdeim import (
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    LFM2_ATTENTION_GROUP_QDEIM_SOURCE_LOCAL_OUTPUT_RESIDUAL_SCHEMA,
    Lfm2AttentionGroupQdeimSourceLocalOutputFitCase,
    Lfm2CompiledAttentionGroupQdeim,
    Lfm2CompiledAttentionGroupQdeimSourceLocalOutputResidual,
    compile_lfm2_attention_group_qdeim_source_local_output_residual,
    lfm2_attention_group_qdeim_candidate_state_sha256,
    lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_macrocell_replay import Lfm2CpcfMacrocellReplay
from neural_forge.hf_lfm2_saved_body_target_lattice_qk import (
    LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import _LayerReplayBank
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (
    LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
)
from neural_forge import hf_lfm2_semantic_attention_source_local_output as _local


LFM2_POST_TARGET_LATTICE_QK_OUTPUT_FOLD_SCHEMA = (
    "neural-forge-lfm2-post-target-lattice-qk-output-fold-v1"
)
LFM2_POST_TARGET_LATTICE_QK_OUTPUT_REPLAY_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-post-target-lattice-qk-output-replay-evidence-v1"
)
_QK_TREATED_BODY_PROVENANCE_SCHEMA = (
    "neural-forge-lfm2-exact4x-qk-treated-body-provenance-v2"
)

_QKVO_ROLES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
)
_FIXED_RECEIPT_ROLES = (
    "v_proj.weight",
    "out_proj.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
)
_ZERO_OPERATIONS = MappingProxyType(
    {
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
    }
)
_QK_TREATMENT_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_weight_tensor_loads": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)
_QK_FRESH_RELOAD_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "network_access": False,
    }
)


class Lfm2PostTargetLatticeQkOutputFoldError(RuntimeError):
    """The treated seed, cached evidence, or O-only transaction is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Post-target-lattice attention-output folding requires PyTorch."
        ) from exc
    return torch


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _verified_receipt(
    value: Any,
    *,
    expected_schema: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop("result_sha256", None)
    if body.get("schema") != expected_schema or supplied != canonical_sha256(body):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"{label} is stale, malformed, or unauthenticated."
        )
    return MappingProxyType({**body, "result_sha256": supplied})


def _verified_fresh_reload_provenance(
    value: Any,
    *,
    expected_sha256: str,
) -> Mapping[str, Any]:
    expected = _digest(
        expected_sha256,
        label="expected_qk_fresh_reload_provenance_sha256",
    )
    if not isinstance(value, Mapping):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Q/K fresh-reload provenance must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop("receipt_sha256", None)
    if (
        body.get("schema") != _QK_TREATED_BODY_PROVENANCE_SCHEMA
        or supplied != expected
        or supplied != canonical_sha256(body)
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Q/K fresh-reload provenance is stale, malformed, or not independently "
            "pinned."
        )
    return MappingProxyType({**body, "receipt_sha256": supplied})


def _tensor_sha256(value: Any, *, label: str) -> str:
    try:
        return _local._tensor_sha256(value, label=label)
    except Exception as exc:
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"{label} is not one authenticated dense tensor."
        ) from exc


def _uniform_parameter_dtype(model: Any) -> tuple[Any, str]:
    torch = _torch()
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1 or not dtypes.issubset({torch.bfloat16, torch.float32}):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The treated checkpoint must have one uniform BF16 or FP32 Parameter dtype."
        )
    dtype = next(iter(dtypes))
    return dtype, str(dtype).removeprefix("torch.")


def _model_parameter_runtime_inventory(model: Any) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(name): {
                "object_id": id(parameter),
                "storage_pointer": int(parameter.data_ptr()),
                "requires_grad": bool(parameter.requires_grad),
                "gradient_object_id": (
                    None if parameter.grad is None else id(parameter.grad)
                ),
                "gradient_sha256": (
                    None
                    if parameter.grad is None
                    else _tensor_sha256(
                        parameter.grad,
                        label=f"{name} gradient",
                    )
                ),
            }
            for name, parameter in model.named_parameters()
        }
    )


def _layer_step_by_index(
    treatment: Mapping[str, Any],
) -> Mapping[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for raw in treatment.get("layer_steps", ()):  # sealed by the outer receipt
        if not isinstance(raw, Mapping):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Q/K treatment layer steps are malformed."
            )
        body = _deep_thaw(raw)
        supplied = body.pop("result_sha256", None)
        index = body.get("layer_index")
        if (
            body.get("schema") != "neural-forge-lfm2-saved-body-qk-layer-step-v1"
            or supplied != canonical_sha256(body)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in result
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Q/K treatment layer-step authentication failed."
            )
        result[index] = MappingProxyType({**body, "result_sha256": supplied})
    return MappingProxyType(result)


def _authenticate_treated_qk_seed(
    *,
    target_model: Any,
    treatment_receipt: Mapping[str, Any],
    expected_treatment_receipt_sha256: str,
    layer_indices: tuple[int, ...],
) -> tuple[Mapping[str, Any], Mapping[int, Mapping[str, Any]], str, Any, str]:
    expected = _digest(
        expected_treatment_receipt_sha256,
        label="expected_treatment_receipt_sha256",
    )
    treatment = _verified_receipt(
        treatment_receipt,
        expected_schema=LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
        label="saved-body target-lattice Q/K treatment receipt",
    )
    if treatment["result_sha256"] != expected:
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The supplied Q/K treatment receipt does not match its expected digest."
        )
    dtype, dtype_name = _uniform_parameter_dtype(target_model)
    current_identity = fingerprint_lfm2_parent(target_model).identity_sha256
    receipt_layers = tuple(treatment.get("attention_layer_indices", ()))
    if (
        receipt_layers != layer_indices
        or treatment.get("attention_layer_count") != len(layer_indices)
        or treatment.get("attention_install_count") != len(layer_indices)
        or treatment.get("model_state_after_sha256") != current_identity
        or treatment.get("parameter_dtype") != dtype_name
        or treatment.get("checkpoint_dtype_preserved") is not True
        or treatment.get("actual_changed_state_keys_are_qk_only") is not True
        or treatment.get("all_non_qk_tensors_byte_identical") is not True
        or treatment.get("parameter_count_changed") is not False
        or treatment.get("fixed_child_value_output_used_as_endpoint_teacher")
        is not True
        or treatment.get("causal_layer_order_proven") is not True
        or treatment.get("later_layers_replayed_from_treated_prefix") is not True
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The live model is not the exact dtype-preserving treated Q/K body "
            "authenticated by the supplied receipt."
        )
    actual_count = _local._parameter_count(target_model)
    if (
        treatment.get("parameter_element_count_before") != actual_count
        or treatment.get("parameter_element_count_after") != actual_count
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The Q/K receipt Parameter count differs from the live treated body."
        )
    allowed = {
        f"model.layers.{index}.self_attn.{role}_proj.weight"
        for index in layer_indices
        for role in ("q", "k")
    }
    changed = tuple(treatment.get("changed_state_keys", ()))
    if not changed or not set(changed).issubset(allowed):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The treatment receipt does not authenticate an effectful Q/K-only "
            "transition."
        )

    installations: dict[int, Mapping[str, Any]] = {}
    flattened_changed: list[str] = []
    for raw in treatment.get("attention_installations", ()):
        receipt = _verified_receipt(
            raw,
            expected_schema=LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
            label="per-layer target-lattice Q/K installation receipt",
        )
        index = receipt.get("layer_index")
        prefix = f"model.layers.{index}.self_attn."
        per_changed = tuple(receipt.get("changed_state_keys", ()))
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in layer_indices
            or index in installations
            or receipt.get("installed") is not True
            or receipt.get("qk_only") is not True
            or receipt.get("actual_changed_state_keys_are_qk_only") is not True
            or receipt.get("parameter_dtype") != dtype_name
            or receipt.get("checkpoint_dtype_preserved") is not True
            or not per_changed
            or not set(per_changed).issubset(
                {prefix + "q_proj.weight", prefix + "k_proj.weight"}
            )
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "A per-layer Q/K installation receipt is incomplete or stale."
            )
        installations[index] = receipt
        flattened_changed.extend(per_changed)
    if tuple(sorted(installations)) != layer_indices or tuple(
        sorted(flattened_changed)
    ) != tuple(changed):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Per-layer Q/K receipts do not close the whole-body transition."
        )
    steps = _layer_step_by_index(treatment)
    for index in layer_indices:
        if (
            index not in steps
            or steps[index].get("operator_kind") != "full_attention"
            or steps[index].get("qk_modified") is not True
            or steps[index].get("non_qk_tensor_modified") is not False
            or steps[index].get("qk_installation_sha256")
            != installations[index]["result_sha256"]
            or not isinstance(steps[index].get("propagation_replay_bank_sha256"), str)
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "The Q/K causal replay lineage is incomplete for an attention layer."
            )
    operation_audit = treatment.get("operation_audit")
    if not isinstance(operation_audit, Mapping) or dict(operation_audit) != dict(
        _QK_TREATMENT_ZERO_OPERATION_AUDIT
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The Q/K seed receipt does not prove a fold-only transition."
        )
    return (
        treatment,
        MappingProxyType(installations),
        current_identity,
        dtype,
        dtype_name,
    )


def _bind_fresh_reload_provenance_to_live_seed(
    *,
    provenance: Mapping[str, Any],
    treatment: Mapping[str, Any],
    target_model: Any,
    current_identity: str,
    dtype_name: str,
) -> None:
    digest_fields = (
        "source_checkpoint_inventory_sha256",
        "source_body_model_state_sha256",
        "source_body_lfm2_identity_sha256",
        "source_causal_body_inventory_sha256",
        "treated_checkpoint_inventory_sha256",
        "treated_body_model_state_sha256",
        "treated_body_lfm2_identity_sha256",
        "treated_causal_body_inventory_sha256",
        "target_lattice_qk_treatment_sha256",
    )
    try:
        for name in digest_fields:
            _digest(provenance.get(name), label=f"Q/K fresh-reload {name}")
    except Lfm2PostTargetLatticeQkOutputFoldError:
        raise
    changed = list(treatment["changed_state_keys"])
    if (
        provenance.get("status") != "saved-reloaded-qk-only-treated-body"
        or provenance.get("target_lattice_qk_treatment_sha256")
        != treatment["result_sha256"]
        or provenance.get("treated_body_lfm2_identity_sha256") != current_identity
        or provenance.get("treated_body_model_state_sha256")
        != fingerprint_model_state(target_model)
        or provenance.get("treated_checkpoint_dtype") != dtype_name
        or provenance.get("source_checkpoint_dtype") != dtype_name
        or provenance.get("checkpoint_dtype_preserved") is not True
        or provenance.get("standalone_fresh_reload_verified") is not True
        or provenance.get("causal_output_head_installed") is not False
        or provenance.get("changed_state_keys") != changed
        or provenance.get("changed_checkpoint_relative_paths") != ["model.safetensors"]
        or provenance.get("operation_audit")
        != dict(_QK_FRESH_RELOAD_ZERO_OPERATION_AUDIT)
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The independently pinned Q/K provenance does not authenticate this "
            "freshly reloaded treated checkpoint."
        )


def _adapt_structural_candidate_to_treated_qk(
    *,
    structural_candidate: Lfm2CompiledAttentionGroupQdeim,
    live_parameters: Mapping[str, Any],
    treatment_result_sha256: str,
    qk_installation: Mapping[str, Any],
    model_identity_sha256: str,
    dtype: Any,
    dtype_name: str,
) -> Lfm2CompiledAttentionGroupQdeim:
    if not isinstance(structural_candidate, Lfm2CompiledAttentionGroupQdeim):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Each structural candidate must be one typed intact-group QDEIM candidate."
        )
    original_report = _verified_receipt(
        structural_candidate.report,
        expected_schema=LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        label="pre-Q/K structural attention receipt",
    )
    original_state_sha256 = lfm2_attention_group_qdeim_candidate_state_sha256(
        structural_candidate.query_weight,
        structural_candidate.key_weight,
        structural_candidate.value_weight,
        structural_candidate.out_weight,
    )
    if (
        original_report.get("materialization", {}).get("candidate_state_sha256")
        != original_state_sha256
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The pre-Q/K structural candidate differs from its sealed receipt."
        )
    live = {
        role: live_parameters[role].detach().to(device="cpu").contiguous().clone()
        for role in _QKVO_ROLES
    }
    structural = {
        "q_proj.weight": structural_candidate.query_weight,
        "k_proj.weight": structural_candidate.key_weight,
        "v_proj.weight": structural_candidate.value_weight,
        "out_proj.weight": structural_candidate.out_weight,
    }
    if any(
        live[role].dtype != dtype
        or structural[role].dtype != dtype
        or tuple(live[role].shape) != tuple(structural[role].shape)
        for role in _QKVO_ROLES
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Treated Q/K and structural V/O do not share one exact topology and "
            "checkpoint dtype."
        )
    structural_hashes = {
        role: _tensor_sha256(value, label=f"structural {role}")
        for role, value in structural.items()
    }
    live_hashes = {
        role: _tensor_sha256(value, label=f"live treated {role}")
        for role, value in live.items()
    }
    fixed = qk_installation.get("fixed_tensor_sha256_before_and_after")
    if (
        not isinstance(fixed, Mapping)
        or live_hashes["v_proj.weight"] != structural_hashes["v_proj.weight"]
        or any(
            fixed.get(role) != live_hashes[role] for role in _FIXED_RECEIPT_ROLES[:2]
        )
        or qk_installation.get("frozen_query_weight_sha256")
        != live_hashes["q_proj.weight"]
        or qk_installation.get("frozen_key_weight_sha256")
        != live_hashes["k_proj.weight"]
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Live treated Q/K or unchanged structural V does not match the "
            "per-layer treatment receipt."
        )
    for role in _FIXED_RECEIPT_ROLES[2:]:
        if fixed.get(role) != _tensor_sha256(
            live_parameters[role],
            label=f"live treated {role}",
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "A Q/K normalization tensor differs from the treated seed receipt."
            )

    treated_state_sha256 = lfm2_attention_group_qdeim_candidate_state_sha256(
        live["q_proj.weight"],
        live["k_proj.weight"],
        live["v_proj.weight"],
        live["out_proj.weight"],
    )
    body = _deep_thaw(original_report)
    body.pop("result_sha256", None)
    body["status"] = "authenticated-post-target-lattice-qk-seed-for-o-only-fold"
    body["method"] = "intact-gqa-structural-vo-with-treated-target-lattice-qk"
    materialization = dict(body.get("materialization", {}))
    materialization.update(
        {
            "q_formula": "byte-exact-live-target-lattice-qk-treatment",
            "k_formula": "byte-exact-live-target-lattice-qk-treatment",
            "v_formula": "byte-exact-pre-treatment-structural-v",
            "o_formula": "byte-exact-live-pre-o-post-qk-output-prior",
            "candidate_state_sha256": treated_state_sha256,
            "candidate_dtype": dtype_name,
            "post_target_lattice_qk_seed": True,
        }
    )
    body["materialization"] = materialization
    authentication = dict(body.get("authentication", {}))
    authentication["post_target_lattice_qk_seed"] = {
        "whole_body_qk_treatment_result_sha256": treatment_result_sha256,
        "layer_qk_installation_result_sha256": qk_installation["result_sha256"],
        "live_treated_model_identity_sha256": model_identity_sha256,
        "pre_qk_structural_candidate_state_sha256": original_state_sha256,
        "pre_qk_structural_receipt_sha256": original_report["result_sha256"],
        "live_treated_qkvo_state_sha256": treated_state_sha256,
        "live_treated_tensor_sha256": live_hashes,
        "structural_tensor_sha256": structural_hashes,
        "v_unchanged_from_structural_candidate": True,
        "o_prior_authenticated_from_qk_fixed_tensor_receipt": True,
        "o_prior_may_include_pre_qk_source_local_closure": True,
        "parameter_dtype": dtype_name,
    }
    body["authentication"] = authentication
    report = MappingProxyType({**body, "result_sha256": canonical_sha256(body)})
    return Lfm2CompiledAttentionGroupQdeim(
        selected_source_key_value_groups=(
            structural_candidate.selected_source_key_value_groups
        ),
        selected_source_query_heads=structural_candidate.selected_source_query_heads,
        query_weight=live["q_proj.weight"],
        key_weight=live["k_proj.weight"],
        value_weight=live["v_proj.weight"],
        out_weight=live["out_proj.weight"],
        group_pod_basis=(
            structural_candidate.group_pod_basis.detach().cpu().contiguous().clone()
        ),
        group_interpolation=(
            structural_candidate.group_interpolation.detach().cpu().contiguous().clone()
        ),
        context_reconstruction=(
            structural_candidate.context_reconstruction.detach()
            .cpu()
            .contiguous()
            .clone()
        ),
        normalized_input_decoder=(
            structural_candidate.normalized_input_decoder.detach()
            .cpu()
            .contiguous()
            .clone()
        ),
        residual_output_chart=(
            structural_candidate.residual_output_chart.detach()
            .cpu()
            .contiguous()
            .clone()
        ),
        report=report,
    )


@dataclass(frozen=True, slots=True)
class Lfm2PostTargetLatticeQkOutputReplayCaseBinding:
    """Exact semantic rows selecting one case from a typed causal replay."""

    case_id: str
    batch_index: int
    alignment_sha256: str
    target_positions: tuple[int, ...]
    donor_positions: tuple[int, ...]
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        targets = tuple(self.target_positions)
        donors = tuple(self.donor_positions)
        if (
            not isinstance(self.case_id, str)
            or not self.case_id.strip()
            or isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
            or not targets
            or len(targets) != len(donors)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*targets, *donors)
            )
            or targets != tuple(sorted(set(targets)))
            or donors != tuple(range(len(donors)))
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Post-Q/K replay case binding coordinates are invalid."
            )
        alignment = _digest(
            self.alignment_sha256,
            label="post-Q/K replay case alignment",
        )
        object.__setattr__(self, "target_positions", targets)
        object.__setattr__(self, "donor_positions", donors)
        body = {
            "schema": "neural-forge-lfm2-post-qk-output-replay-case-binding-v1",
            "case_id": self.case_id,
            "batch_index": self.batch_index,
            "alignment_sha256": alignment,
            "target_positions": list(targets),
            "donor_positions": list(donors),
        }
        object.__setattr__(self, "binding_sha256", canonical_sha256(body))

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "batch_index": self.batch_index,
            "alignment_sha256": self.alignment_sha256,
            "target_positions": list(self.target_positions),
            "donor_positions": list(self.donor_positions),
            "binding_sha256": self.binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2PostTargetLatticeQkOutputReplayEvidence:
    """Fit rows proven against an actual typed fresh post-Q/K replay bank."""

    layer_index: int
    qk_treatment_result_sha256: str
    qk_layer_step_result_sha256: str
    qk_layer_propagation_replay_bank_sha256: str
    qk_layer_propagation_model_state_sha256: str
    fresh_reloaded_model_identity_sha256: str
    replay_bank: _LayerReplayBank = field(repr=False, compare=False)
    alignment_sha256_by_batch: Mapping[int, str] = field(repr=False, compare=False)
    case_bindings: tuple[Lfm2PostTargetLatticeQkOutputReplayCaseBinding, ...] = field(
        repr=False, compare=False
    )
    fit_cases: tuple[Lfm2AttentionGroupQdeimSourceLocalOutputFitCase, ...] = field(
        repr=False,
        compare=False,
    )
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Post-Q/K replay evidence layer_index must be nonnegative."
            )
        for name in (
            "qk_treatment_result_sha256",
            "qk_layer_step_result_sha256",
            "qk_layer_propagation_replay_bank_sha256",
            "qk_layer_propagation_model_state_sha256",
            "fresh_reloaded_model_identity_sha256",
        ):
            _digest(getattr(self, name), label=f"post-Q/K replay evidence {name}")
        cases = tuple(self.fit_cases)
        bindings = tuple(self.case_bindings)
        alignments = MappingProxyType(dict(self.alignment_sha256_by_batch))
        object.__setattr__(self, "fit_cases", cases)
        object.__setattr__(self, "case_bindings", bindings)
        object.__setattr__(self, "alignment_sha256_by_batch", alignments)
        body = self._body()
        object.__setattr__(
            self,
            "evidence_sha256",
            canonical_sha256(_deep_thaw(body)),
        )

    def _validated_replay_rows(
        self,
    ) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
        torch = _torch()
        bank = self.replay_bank
        if (
            not isinstance(bank, _LayerReplayBank)
            or bank.layer_index != self.layer_index
            or bank.captured_child_model_state_sha256
            != self.fresh_reloaded_model_identity_sha256
            or not isinstance(bank.replay_by_batch, Mapping)
            or not isinstance(bank.pre_residual_by_batch, Mapping)
            or set(bank.replay_by_batch) != set(self.alignment_sha256_by_batch)
            or set(bank.pre_residual_by_batch) != set(bank.replay_by_batch)
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Post-Q/K replay evidence is not backed by the exact typed fresh "
                "causal replay bank."
            )
        batch_replays = []
        for batch_index, replay in bank.replay_by_batch.items():
            if (
                isinstance(batch_index, bool)
                or not isinstance(batch_index, int)
                or batch_index < 0
                or not isinstance(replay, Lfm2CpcfMacrocellReplay)
            ):
                raise Lfm2PostTargetLatticeQkOutputFoldError(
                    "Post-Q/K replay bank contains an untyped replay."
                )
            alignment = _digest(
                self.alignment_sha256_by_batch[batch_index],
                label=f"post-Q/K replay batch {batch_index} alignment",
            )
            batch_replays.append(
                {
                    "batch_index": batch_index,
                    "alignment_sha256": alignment,
                    "replay_sha256": replay.replay_sha256,
                }
            )
        replay_bank_body = {
            "schema": "neural-forge-lfm2-semantic-layer-replay-bank-v1",
            "layer_index": self.layer_index,
            "captured_child_model_state_sha256": (
                self.fresh_reloaded_model_identity_sha256
            ),
            "batch_replays": batch_replays,
            "model_forward_calls": 0,
            "layer_forward_calls": 0,
            "training_steps": 0,
        }
        if bank.bank_sha256 != canonical_sha256(replay_bank_body):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Typed post-Q/K replay bank differs from its causal bank digest."
            )

        cases_by_id = {case.case_id: case for case in self.fit_cases}
        bindings_by_id = {binding.case_id: binding for binding in self.case_bindings}
        if (
            not cases_by_id
            or len(cases_by_id) != len(self.fit_cases)
            or len(bindings_by_id) != len(self.case_bindings)
            or set(cases_by_id) != set(bindings_by_id)
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Every post-Q/K fit case must have exactly one replay-row binding."
            )
        rows = []
        for case_index, case in enumerate(self.fit_cases):
            if not isinstance(case, Lfm2AttentionGroupQdeimSourceLocalOutputFitCase):
                raise Lfm2PostTargetLatticeQkOutputFoldError(
                    "Post-Q/K replay evidence contains an untyped fit case."
                )
            binding = bindings_by_id[case.case_id]
            if (
                binding.alignment_sha256
                != self.alignment_sha256_by_batch.get(binding.batch_index)
                or binding.batch_index not in bank.replay_by_batch
            ):
                raise Lfm2PostTargetLatticeQkOutputFoldError(
                    "Post-Q/K replay case binding is absent from the typed bank."
                )
            replay = bank.replay_by_batch[binding.batch_index]
            maximum_position = int(replay.operator_input.shape[1])
            if binding.target_positions[-1] >= maximum_position:
                raise Lfm2PostTargetLatticeQkOutputFoldError(
                    "Post-Q/K replay row selection exceeds the typed replay."
                )
            positions = torch.tensor(binding.target_positions, dtype=torch.long)
            child_pre = replay.operator_input.index_select(1, positions)[0]
            child_context = replay.operator_output_input.index_select(1, positions)[0]
            expected_row_identity = canonical_sha256(
                {
                    "schema": (
                        "neural-forge-lfm2-source-local-attention-aligned-fit-rows-v1"
                    ),
                    "partition": "fit",
                    "layer_index": self.layer_index,
                    "batch_index": binding.batch_index,
                    "alignment_sha256": binding.alignment_sha256,
                    "target_positions": list(binding.target_positions),
                    "donor_positions": list(binding.donor_positions),
                    "row_count": len(binding.target_positions),
                }
            )
            if (
                case.live_child_replay_receipt_sha256 != replay.replay_sha256
                or case.aligned_row_identity_sha256 != expected_row_identity
                or _tensor_sha256(
                    case.live_child_pre_operator_residual,
                    label=f"post-Q/K case {case.case_id} supplied child pre-residual",
                )
                != _tensor_sha256(
                    child_pre,
                    label=f"post-Q/K case {case.case_id} replay child pre-residual",
                )
                or _tensor_sha256(
                    case.live_child_pre_o_context,
                    label=f"post-Q/K case {case.case_id} supplied child pre-O context",
                )
                != _tensor_sha256(
                    child_context,
                    label=f"post-Q/K case {case.case_id} replay child pre-O context",
                )
            ):
                raise Lfm2PostTargetLatticeQkOutputFoldError(
                    "Post-Q/K fit tensors, row identity, or replay digest were not "
                    "derived from the typed fresh causal replay."
                )
            tensors = {
                "parent_pre_operator_residual": case.parent_pre_operator_residual,
                "parent_post_operator_residual": case.parent_post_operator_residual,
                "live_child_pre_operator_residual": (
                    case.live_child_pre_operator_residual
                ),
                "live_child_pre_o_context": case.live_child_pre_o_context,
            }
            rows.append(
                MappingProxyType(
                    {
                        "case_index": case_index,
                        **binding.as_dict(),
                        "row_count": len(binding.target_positions),
                        "row_sample_weights": [
                            float(value) for value in case.row_sample_weights
                        ],
                        "parent_trajectory_receipt_sha256": _digest(
                            case.parent_trajectory_receipt_sha256,
                            label=f"post-Q/K case {case.case_id} parent trajectory",
                        ),
                        "live_child_replay_receipt_sha256": replay.replay_sha256,
                        "aligned_row_identity_sha256": expected_row_identity,
                        "tensor_sha256": {
                            name: _tensor_sha256(
                                value,
                                label=f"post-Q/K replay case {case.case_id} {name}",
                            )
                            for name, value in sorted(tensors.items())
                        },
                        "typed_replay_bank_sha256": bank.bank_sha256,
                    }
                )
            )
        return replay_bank_body, tuple(rows)

    def _body(self) -> Mapping[str, Any]:
        replay_bank_body, cases = self._validated_replay_rows()
        bank_body = {
            "schema": "neural-forge-lfm2-post-qk-output-replay-bank-v1",
            "layer_index": self.layer_index,
            "captured_child_model_state_sha256": (
                self.fresh_reloaded_model_identity_sha256
            ),
            "source_qk_layer_step_result_sha256": (self.qk_layer_step_result_sha256),
            "source_qk_propagation_replay_bank_sha256": (
                self.qk_layer_propagation_replay_bank_sha256
            ),
            "typed_layer_replay_bank_sha256": self.replay_bank.bank_sha256,
            "typed_layer_replay_bank_body_sha256": canonical_sha256(replay_bank_body),
            "case_replays": [
                {
                    "case_id": row["case_id"],
                    "batch_index": row["batch_index"],
                    "binding_sha256": row["binding_sha256"],
                    "live_child_replay_receipt_sha256": row[
                        "live_child_replay_receipt_sha256"
                    ],
                    "aligned_row_identity_sha256": row["aligned_row_identity_sha256"],
                    "live_child_pre_operator_residual_sha256": row["tensor_sha256"][
                        "live_child_pre_operator_residual"
                    ],
                    "live_child_pre_o_context_sha256": row["tensor_sha256"][
                        "live_child_pre_o_context"
                    ],
                }
                for row in cases
            ],
        }
        bank_sha256 = canonical_sha256(bank_body)
        return MappingProxyType(
            {
                "schema": (LFM2_POST_TARGET_LATTICE_QK_OUTPUT_REPLAY_EVIDENCE_SCHEMA),
                "status": "authenticated-fresh-reloaded-post-qk-fit-replay",
                "partition": "fit",
                "layer_index": self.layer_index,
                "qk_treatment_result_sha256": self.qk_treatment_result_sha256,
                "qk_layer_step_result_sha256": self.qk_layer_step_result_sha256,
                "qk_layer_propagation_replay_bank_sha256": (
                    self.qk_layer_propagation_replay_bank_sha256
                ),
                "qk_layer_propagation_model_state_sha256": (
                    self.qk_layer_propagation_model_state_sha256
                ),
                "fresh_reloaded_model_identity_sha256": (
                    self.fresh_reloaded_model_identity_sha256
                ),
                "fresh_post_qk_replay_bank": {
                    **bank_body,
                    "bank_sha256": bank_sha256,
                },
                "fresh_post_qk_replay_bank_sha256": bank_sha256,
                "typed_layer_replay_bank_sha256": self.replay_bank.bank_sha256,
                "fit_evidence_sha256": (
                    lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256(
                        self.fit_cases
                    )
                ),
                "cases": list(cases),
                "all_case_tensors_content_addressed": True,
                "all_child_replays_individually_content_addressed": True,
                "all_child_rows_equal_typed_replay_rows": True,
                "fresh_reloaded_child_model_identity_bound": True,
            }
        )

    def verify(self, *, expected_sha256: str | None = None) -> bool:
        try:
            observed = canonical_sha256(_deep_thaw(self._body()))
            if observed != self.evidence_sha256:
                return False
            if expected_sha256 is not None:
                return observed == _digest(
                    expected_sha256,
                    label="expected post-Q/K replay evidence",
                )
        except Exception:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        body = _deep_thaw(self._body())
        return {**body, "evidence_sha256": self.evidence_sha256}


def _direction_materialization_receipt(
    compiled: Lfm2CompiledAttentionGroupQdeimSourceLocalOutputResidual,
) -> Mapping[str, Any]:
    torch = _torch()
    analytic = compiled.analytic_out_correction_fp64.detach().to(torch.float64).T
    published = compiled.out_weight_source_local_corrected.detach().to(
        torch.float64
    ) - compiled.out_weight_structural_prior.detach().to(torch.float64)
    analytic_norm = float(torch.linalg.vector_norm(analytic).item())
    published_norm = float(torch.linalg.vector_norm(published).item())
    dot = float(torch.sum(analytic * published).item())
    cosine = (
        None
        if analytic_norm == 0.0 or published_norm == 0.0
        else dot / (analytic_norm * published_norm)
    )
    changed = int(torch.count_nonzero(published).item())
    if (
        analytic_norm <= 0.0
        or published_norm <= 0.0
        or changed <= 0
        or not bool(torch.isfinite(torch.tensor(dot)).item())
        or dot <= 0.0
        or cosine is None
        or cosine <= 0.0
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The analytic O direction vanished or reversed during target-dtype "
            "materialization; an ineffectual writer is not publishable."
        )
    body = {
        "schema": "neural-forge-lfm2-post-qk-o-direction-materialization-v1",
        "analytic_fp64_direction_sha256": _tensor_sha256(
            analytic,
            label="analytic O direction",
        ),
        "published_fp64_delta_sha256": _tensor_sha256(
            published,
            label="published O delta",
        ),
        "analytic_fp64_l2_norm": analytic_norm,
        "published_fp64_l2_norm": published_norm,
        "analytic_published_dot_product": dot,
        "analytic_published_cosine": cosine,
        "published_changed_element_count": changed,
        "target_dtype": str(
            compiled.out_weight_source_local_corrected.dtype
        ).removeprefix("torch."),
        "effectful_target_dtype_materialization": True,
        "direction_preserved": True,
        "numeric_quality_gate_used": False,
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _authenticate_compiler_tensor_result(
    *,
    layer_index: int,
    compiled: Lfm2CompiledAttentionGroupQdeimSourceLocalOutputResidual,
    compiler: Mapping[str, Any],
    live_parameters: Mapping[str, Any],
    qkv_hashes_before: Mapping[str, str],
    expected_fit_evidence_sha256: str,
    parameter_dtype: Any,
) -> tuple[str, str, Mapping[str, Any]]:
    qkv_result_hashes = {
        role: _tensor_sha256(value, label=f"layer {layer_index} compiled {role}")
        for role, value in (
            ("q_proj.weight", compiled.query_weight),
            ("k_proj.weight", compiled.key_weight),
            ("v_proj.weight", compiled.value_weight),
        )
    }
    if qkv_result_hashes != dict(qkv_hashes_before):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"Layer {layer_index} O compiler changed Q, K, or V."
        )
    prior_hash = _tensor_sha256(
        compiled.out_weight_structural_prior,
        label=f"layer {layer_index} compiler structural O prior",
    )
    live_prior_hash = _tensor_sha256(
        live_parameters["out_proj.weight"],
        label=f"layer {layer_index} live structural O prior",
    )
    analytic_hash = _tensor_sha256(
        compiled.analytic_out_correction_fp64,
        label=f"layer {layer_index} analytic O correction",
    )
    supported_basis_hash = _tensor_sha256(
        compiled.fit_supported_context_basis_fp64,
        label=f"layer {layer_index} supported context basis",
    )
    corrected = compiled.out_weight_source_local_corrected
    corrected_hash = _tensor_sha256(
        corrected,
        label=f"layer {layer_index} corrected out projection",
    )
    corrected_state = lfm2_attention_group_qdeim_candidate_state_sha256(
        compiled.query_weight,
        compiled.key_weight,
        compiled.value_weight,
        corrected,
    )
    authentication = compiler.get("authentication")
    solver = compiler.get("solver")
    materialization = compiler.get("materialization")
    topology = compiler.get("topology")
    if (
        not isinstance(authentication, Mapping)
        or not isinstance(solver, Mapping)
        or not isinstance(materialization, Mapping)
        or not isinstance(topology, Mapping)
        or authentication.get("structural_qkv_tensor_sha256") != dict(qkv_hashes_before)
        or authentication.get("result_qkv_tensor_sha256") != qkv_result_hashes
        or authentication.get("structural_out_weight_sha256") != prior_hash
        or prior_hash != live_prior_hash
        or authentication.get("analytic_out_correction_fp64_sha256") != analytic_hash
        or authentication.get("source_local_corrected_out_weight_sha256")
        != corrected_hash
        or authentication.get("source_local_corrected_candidate_state_sha256")
        != corrected_state
        or authentication.get("fit_evidence_sha256_expected")
        != expected_fit_evidence_sha256
        or authentication.get("fit_evidence_sha256_before")
        != expected_fit_evidence_sha256
        or authentication.get("fit_evidence_sha256_after")
        != expected_fit_evidence_sha256
        or authentication.get("fit_evidence_immutable") is not True
        or solver.get("fit_supported_context_basis_sha256") != supported_basis_hash
        or solver.get("target_materialization_dtype")
        != str(parameter_dtype).removeprefix("torch.")
        or materialization.get("source_local_corrected_candidate_state_sha256")
        != corrected_state
        or materialization.get("qkv_bit_identical_to_structural_candidate") is not True
        or materialization.get("only_out_projection_permitted_to_change") is not True
        or topology.get("structural_selected_source_key_value_groups")
        != list(compiled.selected_source_key_value_groups)
        or topology.get("structural_selected_source_query_heads")
        != list(compiled.selected_source_query_heads)
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"Layer {layer_index} compiler tensors do not match their sealed "
            "compiler receipt."
        )
    if corrected.dtype != parameter_dtype or tuple(corrected.shape) != tuple(
        live_parameters["out_proj.weight"].shape
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"Layer {layer_index} corrected O does not preserve checkpoint dtype "
            "and topology."
        )
    if corrected_hash == live_prior_hash:
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            f"Layer {layer_index} source-local O fold has no target-dtype effect."
        )
    return (
        corrected_hash,
        corrected_state,
        _direction_materialization_receipt(compiled),
    )


def _authenticate_installed_o_state(
    *,
    target_model: Any,
    layer_indices: tuple[int, ...],
    before_state_hashes: Mapping[str, str],
    before_model_identity: str,
    before_topology: str,
    before_parameter_count: int,
    before_runtime: Mapping[str, Any],
    expected_out_hashes: Mapping[int, str],
) -> Mapping[str, Any]:
    after_state_hashes = _local._state_tensor_hashes(target_model)
    changed = _local._changed_state_keys(before_state_hashes, after_state_hashes)
    expected_changed = tuple(
        sorted(
            f"model.layers.{index}.self_attn.out_proj.weight"
            for index in layer_indices
        )
    )
    after_model_identity = fingerprint_lfm2_parent(target_model).identity_sha256
    after_topology = _local._topology_sha256(target_model)
    after_parameter_count = _local._parameter_count(target_model)
    after_runtime = _model_parameter_runtime_inventory(target_model)
    installed_out_hashes = {
        index: _tensor_sha256(
            _local._attention_parameters(target_model, index)["out_proj.weight"],
            label=f"installed layer {index} out projection",
        )
        for index in layer_indices
    }
    if (
        changed != expected_changed
        or installed_out_hashes != dict(expected_out_hashes)
        or after_model_identity == before_model_identity
        or after_topology != before_topology
        or after_parameter_count != before_parameter_count
        or after_runtime != before_runtime
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Post-Q/K O publication changed non-O state, lost an O effect, or "
            "altered topology, Parameter identity/storage, or gradients."
        )
    return MappingProxyType(
        {
            "changed_state_keys": changed,
            "model_state_after_sha256": after_model_identity,
            "target_topology_sha256_after": after_topology,
            "parameter_count_after": after_parameter_count,
            "installed_out_weight_sha256": installed_out_hashes,
            "parameter_runtime_inventory_preserved": True,
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2PostTargetLatticeQkOutputFoldInstallation:
    """All compiled layer writers and their atomic whole-body receipt."""

    compiled_by_layer: Mapping[
        int, Lfm2CompiledAttentionGroupQdeimSourceLocalOutputResidual
    ] = field(repr=False, compare=False)
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        compiled = MappingProxyType(dict(self.compiled_by_layer))
        receipt = MappingProxyType(dict(self.receipt))
        object.__setattr__(self, "compiled_by_layer", compiled)
        object.__setattr__(self, "receipt", receipt)
        if not self.verify():
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Post-Q/K O installation result authentication failed."
            )

    def verify(self) -> bool:
        try:
            receipt = _verified_receipt(
                self.receipt,
                expected_schema=LFM2_POST_TARGET_LATTICE_QK_OUTPUT_FOLD_SCHEMA,
                label="post-Q/K O installation receipt",
            )
            layers = tuple(receipt["attention_layer_indices"])
            if layers != tuple(sorted(self.compiled_by_layer)):
                return False
            for index in layers:
                compiler = _verified_receipt(
                    self.compiled_by_layer[index].report,
                    expected_schema=(
                        LFM2_ATTENTION_GROUP_QDEIM_SOURCE_LOCAL_OUTPUT_RESIDUAL_SCHEMA
                    ),
                    label=f"layer {index} source-local O compiler receipt",
                )
                layer = receipt["layer_results"][str(index)]
                compiled = self.compiled_by_layer[index]
                qkv_before = receipt["qkv_tensor_sha256_before"][str(index)]
                (
                    out_sha256,
                    state_sha256,
                    direction,
                ) = _authenticate_compiler_tensor_result(
                    layer_index=index,
                    compiled=compiled,
                    compiler=compiler,
                    live_parameters={
                        "out_proj.weight": compiled.out_weight_structural_prior,
                    },
                    qkv_hashes_before=qkv_before,
                    expected_fit_evidence_sha256=layer["fit_evidence_sha256"],
                    parameter_dtype=compiled.out_weight_source_local_corrected.dtype,
                )
                if (
                    layer["source_local_compiler_result_sha256"]
                    != compiler["result_sha256"]
                    or layer["corrected_candidate_state_sha256"] != state_sha256
                    or layer["corrected_out_weight_sha256"] != out_sha256
                    or layer["direction_materialization_result_sha256"]
                    != direction["result_sha256"]
                    or receipt["direction_materialization_receipts"][str(index)][
                        "result_sha256"
                    ]
                    != direction["result_sha256"]
                ):
                    return False
        except Exception:
            return False
        return bool(
            receipt.get("actual_changed_state_keys_are_out_proj_only") is True
            and receipt.get("all_attention_out_projections_changed") is True
            and receipt.get("qkv_byte_identical_before_and_after") is True
        )


def compile_install_lfm2_post_target_lattice_qk_output_fold(
    *,
    target_model: Any,
    qk_treatment_receipt: Mapping[str, Any],
    qk_fresh_reload_provenance_receipt: Mapping[str, Any],
    expected_qk_fresh_reload_provenance_sha256: str,
    structural_candidates: Mapping[int, Lfm2CompiledAttentionGroupQdeim],
    replay_evidence_by_layer: Mapping[int, Lfm2PostTargetLatticeQkOutputReplayEvidence],
    expected_replay_evidence_sha256_by_layer: Mapping[int, str],
    ridge: float = 1.0e-6,
    fold_compute_device: str = "cpu",
) -> Lfm2PostTargetLatticeQkOutputFoldInstallation:
    """Compile the retained simultaneous-replay O diagnostic after Q/K.

    The Q/K treatment is trusted only through an independently pinned saved/
    freshly-reloaded checkpoint provenance receipt.  Every layer consumes one
    typed content-addressed post-Q/K replay artifact whose digest is independently
    pinned by ``expected_replay_evidence_sha256_by_layer``.  The function performs
    no replay or model call itself.  This diagnostic deliberately does not
    claim causal downstream replay; production callers must use
    ``compile_install_lfm2_causal_post_qk_output_fold`` instead.
    """

    torch = _torch()
    if (
        not isinstance(structural_candidates, Mapping)
        or not isinstance(replay_evidence_by_layer, Mapping)
        or not isinstance(expected_replay_evidence_sha256_by_layer, Mapping)
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Structural candidates, replay evidence, and pinned replay digests "
            "must be mappings."
        )
    layer_indices = tuple(sorted(structural_candidates))
    if (
        not layer_indices
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in layer_indices
        )
        or set(replay_evidence_by_layer) != set(layer_indices)
        or set(expected_replay_evidence_sha256_by_layer) != set(layer_indices)
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "Structural candidates, replay evidence, and pinned replay digests "
            "must cover the same nonempty attention-layer set."
        )
    provenance = _verified_fresh_reload_provenance(
        qk_fresh_reload_provenance_receipt,
        expected_sha256=expected_qk_fresh_reload_provenance_sha256,
    )
    provenance_treatment_sha256 = _digest(
        provenance.get("target_lattice_qk_treatment_sha256"),
        label="fresh-reload target-lattice Q/K treatment",
    )
    (
        treatment,
        qk_installations,
        treated_model_identity,
        parameter_dtype,
        parameter_dtype_name,
    ) = _authenticate_treated_qk_seed(
        target_model=target_model,
        treatment_receipt=qk_treatment_receipt,
        expected_treatment_receipt_sha256=provenance_treatment_sha256,
        layer_indices=layer_indices,
    )
    _bind_fresh_reload_provenance_to_live_seed(
        provenance=provenance,
        treatment=treatment,
        target_model=target_model,
        current_identity=treated_model_identity,
        dtype_name=parameter_dtype_name,
    )
    steps = _layer_step_by_index(treatment)
    before_state_hashes = _local._state_tensor_hashes(target_model)
    before_topology = _local._topology_sha256(target_model)
    before_parameter_count = _local._parameter_count(target_model)
    before_runtime = _model_parameter_runtime_inventory(target_model)
    live_by_layer = {
        index: _local._attention_parameters(target_model, index)
        for index in layer_indices
    }
    out_snapshots = {
        index: live_by_layer[index]["out_proj.weight"]
        .detach()
        .to(device="cpu")
        .contiguous()
        .clone()
        for index in layer_indices
    }
    qkv_hashes_before = {
        index: {
            role: _tensor_sha256(
                live_by_layer[index][role],
                label=f"layer {index} treated {role}",
            )
            for role in _QKVO_ROLES[:3]
        }
        for index in layer_indices
    }

    treated_candidates: dict[int, Lfm2CompiledAttentionGroupQdeim] = {}
    fit_evidence_sha256: dict[int, str] = {}
    fit_cases_by_layer: dict[
        int, tuple[Lfm2AttentionGroupQdeimSourceLocalOutputFitCase, ...]
    ] = {}
    for index in layer_indices:
        evidence = replay_evidence_by_layer[index]
        pinned_evidence = _digest(
            expected_replay_evidence_sha256_by_layer[index],
            label=f"layer {index} expected post-Q/K replay evidence",
        )
        if (
            not isinstance(evidence, Lfm2PostTargetLatticeQkOutputReplayEvidence)
            or not evidence.verify(expected_sha256=pinned_evidence)
            or evidence.layer_index != index
            or evidence.qk_treatment_result_sha256 != treatment["result_sha256"]
            or evidence.qk_layer_step_result_sha256 != steps[index]["result_sha256"]
            or evidence.qk_layer_propagation_replay_bank_sha256
            != steps[index]["propagation_replay_bank_sha256"]
            or evidence.qk_layer_propagation_model_state_sha256
            != steps[index]["propagation_model_state_sha256"]
            or evidence.fresh_reloaded_model_identity_sha256 != treated_model_identity
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                f"Layer {index} replay evidence is not independently pinned to "
                "the Q/K layer step and fresh-reloaded child state."
            )
        cases = tuple(evidence.fit_cases)
        fit_cases_by_layer[index] = cases
        evidence_receipt = evidence.as_dict()
        fit_evidence_sha256[index] = evidence_receipt["fit_evidence_sha256"]
        treated_candidates[index] = _adapt_structural_candidate_to_treated_qk(
            structural_candidate=structural_candidates[index],
            live_parameters=live_by_layer[index],
            treatment_result_sha256=treatment["result_sha256"],
            qk_installation=qk_installations[index],
            model_identity_sha256=treated_model_identity,
            dtype=parameter_dtype,
            dtype_name=parameter_dtype_name,
        )

    def compile_all() -> dict[
        int, Lfm2CompiledAttentionGroupQdeimSourceLocalOutputResidual
    ]:
        result = {}
        for index in layer_indices:
            candidate = treated_candidates[index]
            chart_sha256 = _tensor_sha256(
                candidate.residual_output_chart,
                label=f"layer {index} residual output chart",
            )
            result[index] = (
                compile_lfm2_attention_group_qdeim_source_local_output_residual(
                    structural_candidate=candidate,
                    expected_structural_candidate_state_sha256=(
                        lfm2_attention_group_qdeim_candidate_state_sha256(
                            candidate.query_weight,
                            candidate.key_weight,
                            candidate.value_weight,
                            candidate.out_weight,
                        )
                    ),
                    expected_structural_receipt_sha256=(
                        candidate.report["result_sha256"]
                    ),
                    expected_residual_output_chart_sha256=chart_sha256,
                    fit_cases=fit_cases_by_layer[index],
                    expected_fit_evidence_sha256=fit_evidence_sha256[index],
                    ridge=ridge,
                    compute_device=fold_compute_device,
                )
            )
        return result

    compiled_by_layer, compile_guard = run_fold_only_callback(
        compile_all,
        phase="lfm2-post-target-lattice-qk-source-local-o-compile",
        protected_models=(),
        freeze_parameters=False,
    )
    if (
        fingerprint_lfm2_parent(target_model).identity_sha256 != treated_model_identity
        or _local._state_tensor_hashes(target_model) != before_state_hashes
        or _model_parameter_runtime_inventory(target_model) != before_runtime
    ):
        raise Lfm2PostTargetLatticeQkOutputFoldError(
            "The live treated seed became stale during O compilation."
        )

    expected_out_hashes: dict[int, str] = {}
    direction_receipts: dict[int, Mapping[str, Any]] = {}
    layer_results: dict[str, Mapping[str, Any]] = {}
    for index in layer_indices:
        compiled = compiled_by_layer[index]
        compiler = _verified_receipt(
            compiled.report,
            expected_schema=(
                LFM2_ATTENTION_GROUP_QDEIM_SOURCE_LOCAL_OUTPUT_RESIDUAL_SCHEMA
            ),
            label=f"layer {index} source-local O compiler receipt",
        )
        (
            expected_out_hashes[index],
            corrected_state,
            direction_receipts[index],
        ) = _authenticate_compiler_tensor_result(
            layer_index=index,
            compiled=compiled,
            compiler=compiler,
            live_parameters=live_by_layer[index],
            qkv_hashes_before=qkv_hashes_before[index],
            expected_fit_evidence_sha256=fit_evidence_sha256[index],
            parameter_dtype=parameter_dtype,
        )
        replay_receipt = replay_evidence_by_layer[index].as_dict()
        layer_results[str(index)] = MappingProxyType(
            {
                "structural_candidate_result_sha256": (
                    structural_candidates[index].report["result_sha256"]
                ),
                "treated_seed_result_sha256": (
                    treated_candidates[index].report["result_sha256"]
                ),
                "qk_installation_result_sha256": (
                    qk_installations[index]["result_sha256"]
                ),
                "qk_layer_propagation_replay_bank_sha256": (
                    steps[index]["propagation_replay_bank_sha256"]
                ),
                "fresh_post_qk_replay_bank_sha256": replay_receipt[
                    "fresh_post_qk_replay_bank_sha256"
                ],
                "post_qk_replay_evidence_sha256": replay_evidence_by_layer[
                    index
                ].evidence_sha256,
                "fit_evidence_sha256": fit_evidence_sha256[index],
                "source_local_compiler_result_sha256": compiler["result_sha256"],
                "direction_materialization_result_sha256": (
                    direction_receipts[index]["result_sha256"]
                ),
                "corrected_out_weight_sha256": expected_out_hashes[index],
                "corrected_candidate_state_sha256": corrected_state,
            }
        )

    def install_all() -> None:
        with torch.no_grad():
            for index in layer_indices:
                parameter = live_by_layer[index]["out_proj.weight"]
                parameter.copy_(
                    compiled_by_layer[index].out_weight_source_local_corrected.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

    try:
        _unused, install_guard = run_fold_only_callback(
            install_all,
            phase="lfm2-post-target-lattice-qk-o-only-transaction",
            protected_models=(),
            freeze_parameters=False,
        )
        installed = _authenticate_installed_o_state(
            target_model=target_model,
            layer_indices=layer_indices,
            before_state_hashes=before_state_hashes,
            before_model_identity=treated_model_identity,
            before_topology=before_topology,
            before_parameter_count=before_parameter_count,
            before_runtime=before_runtime,
            expected_out_hashes=expected_out_hashes,
        )
        qkv_hashes_after = {
            index: {
                role: _tensor_sha256(
                    live_by_layer[index][role],
                    label=f"layer {index} installed {role}",
                )
                for role in _QKVO_ROLES[:3]
            }
            for index in layer_indices
        }
        if qkv_hashes_after != qkv_hashes_before:
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Q/K/V changed during the O-only whole-body transaction."
            )
        compile_guard_body = compile_guard.as_dict()
        install_guard_body = install_guard.as_dict()
        observed_operation_audit = {
            name: int(compile_guard_body[name]) + int(install_guard_body[name])
            for name in _ZERO_OPERATIONS
        }
        if observed_operation_audit != dict(_ZERO_OPERATIONS):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "The fold-only guards observed a forbidden operation."
            )
        body = {
            "schema": LFM2_POST_TARGET_LATTICE_QK_OUTPUT_FOLD_SCHEMA,
            "status": "installed-all-post-qk-source-local-attention-writers",
            "qk_treatment_result_sha256": treatment["result_sha256"],
            "qk_fresh_reload_provenance_sha256": provenance["receipt_sha256"],
            "qk_fresh_reload_provenance_independently_pinned": True,
            "treated_model_state_sha256": treated_model_identity,
            "model_state_before_sha256": treated_model_identity,
            "model_state_after_sha256": installed["model_state_after_sha256"],
            "parameter_dtype": parameter_dtype_name,
            "checkpoint_dtype_preserved": True,
            "attention_layer_indices": list(layer_indices),
            "attention_layer_count": len(layer_indices),
            "out_projection_install_count": len(layer_indices),
            "changed_state_keys": list(installed["changed_state_keys"]),
            "allowed_changed_state_keys": [
                f"model.layers.{index}.self_attn.out_proj.weight"
                for index in layer_indices
            ],
            "actual_changed_state_keys_are_out_proj_only": True,
            "all_attention_out_projections_changed": True,
            "qkv_tensor_sha256_before": qkv_hashes_before,
            "qkv_tensor_sha256_after": qkv_hashes_after,
            "qkv_byte_identical_before_and_after": True,
            "layer_results": layer_results,
            "post_qk_replay_evidence_sha256": {
                str(index): replay_evidence_by_layer[index].evidence_sha256
                for index in layer_indices
            },
            "post_qk_replay_evidence_independently_pinned": True,
            "direction_materialization_receipts": {
                str(index): direction_receipts[index] for index in layer_indices
            },
            "target_topology_sha256_before": before_topology,
            "target_topology_sha256_after": installed["target_topology_sha256_after"],
            "topology_changed": False,
            "parameter_count_before": before_parameter_count,
            "parameter_count_after": installed["parameter_count_after"],
            "parameter_count_changed": False,
            "parameter_identities_preserved": True,
            "parameter_storage_preserved": True,
            "gradient_state_preserved": True,
            "fit_partition_only": True,
            "post_qk_child_replay_reused_without_recapture": True,
            "source_local_parent_action_evidence_reused": True,
            "live_output_prior_authenticated_by_qk_fixed_tensor_receipts": True,
            "pre_qk_source_local_output_prior_permitted": True,
            "pristine_structural_o_equality_required": False,
            "fold_compute_device": fold_compute_device,
            "compile_fold_only_guard": compile_guard_body,
            "installation_fold_only_guard": install_guard_body,
            "operation_audit_scope": (
                "sum-of-process-local-fold-only-guard-observations-for-compile-"
                "and-install; no unobserved model-forward counters claimed"
            ),
            "operation_audit": observed_operation_audit,
            "only_out_projection_permitted_to_change": True,
            "numeric_metrics_have_semantic_decision_authority": False,
            "numeric_threshold_auto_accept_permitted": False,
            "numeric_threshold_auto_veto_permitted": False,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        }
        sealed_body = _deep_thaw(body)
        result = Lfm2PostTargetLatticeQkOutputFoldInstallation(
            compiled_by_layer=compiled_by_layer,
            receipt=MappingProxyType(
                {
                    **sealed_body,
                    "result_sha256": canonical_sha256(sealed_body),
                }
            ),
        )
    except BaseException as exc:
        with torch.no_grad():
            for index in layer_indices:
                parameter = live_by_layer[index]["out_proj.weight"]
                parameter.copy_(
                    out_snapshots[index].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        if (
            _local._state_tensor_hashes(target_model) != before_state_hashes
            or fingerprint_lfm2_parent(target_model).identity_sha256
            != treated_model_identity
            or _local._topology_sha256(target_model) != before_topology
            or _local._parameter_count(target_model) != before_parameter_count
            or _model_parameter_runtime_inventory(target_model) != before_runtime
        ):
            raise Lfm2PostTargetLatticeQkOutputFoldError(
                "Post-Q/K O transaction failed and exact whole-body rollback was "
                "not proven."
            ) from exc
        raise
    return result


__all__ = [
    "LFM2_POST_TARGET_LATTICE_QK_OUTPUT_FOLD_SCHEMA",
    "LFM2_POST_TARGET_LATTICE_QK_OUTPUT_REPLAY_EVIDENCE_SCHEMA",
    "Lfm2PostTargetLatticeQkOutputFoldError",
    "Lfm2PostTargetLatticeQkOutputFoldInstallation",
    "Lfm2PostTargetLatticeQkOutputReplayCaseBinding",
    "Lfm2PostTargetLatticeQkOutputReplayEvidence",
]
