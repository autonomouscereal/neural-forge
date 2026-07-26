"""Transactional source-local residual-attention output folding for LFM2.

The intact-GQA QDEIM compiler first supplies one structural Q/K/V/O block.
This module then consumes cached parent local attention actions and a fresh
analytic child replay at the input of ``out_proj``.  It compiles one bounded
closed-form O-only delta and installs that dense tensor without changing Q, K,
V, topology, parameter identity, storage, gradients, or any unrelated state.

The operation is folding, not training: no model forward, generation,
backward, optimizer, pruning, quantization, sparsity, or parameter sharing is
permitted.  Numeric fit diagnostics have no publication or veto authority;
whole decoded behavior remains the intelligence evidence downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import run_fold_only_callback
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
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)
from neural_forge import hf_lfm2_semantic_attention_refold_adapter as _transaction


LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-source-local-output-installation-v1"
)

_ATTENTION_ROLES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
)
_QKVO_ROLES = _ATTENTION_ROLES[:4]


class Lfm2SemanticAttentionSourceLocalOutputError(RuntimeError):
    """Source-local O compilation, authentication, or transaction failed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Source-local residual-attention output folding requires PyTorch."
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


def _seal(body: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(body)
    return MappingProxyType({**payload, "result_sha256": canonical_sha256(payload)})


def _verified_report(
    report: Mapping[str, Any],
    *,
    expected_schema: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(report)
    supplied = body.pop("result_sha256", None)
    if body.get("schema") != expected_schema or supplied != canonical_sha256(body):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            f"{label} is stale or unauthenticated."
        )
    return MappingProxyType({**body, "result_sha256": supplied})


def _tensor_sha256(value: Any, *, label: str) -> str:
    try:
        return lfm2_authenticated_tensor_sha256(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            f"{label} is not one hashable materialized tensor."
        ) from exc


def _parameter_count(model: Any) -> int:
    return int(_transaction._parameter_count(model))


def _topology_sha256(model: Any) -> str:
    return str(_transaction._topology_sha256(model))


def _state_tensor_hashes(model: Any) -> Mapping[str, str]:
    return MappingProxyType(
        {
            str(name): _tensor_sha256(value, label=f"model state {name}")
            for name, value in model.state_dict().items()
        }
    )


def _changed_state_keys(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[str, ...]:
    if set(before) != set(after):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            "Model state topology changed during the O-only transaction."
        )
    return tuple(sorted(name for name in before if before[name] != after[name]))


def _attention_parameters(target_model: Any, layer_index: int) -> Mapping[str, Any]:
    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index >= len(layers)
    ):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            "layer_index does not identify one live target layer."
        )
    attention = getattr(layers[layer_index], "self_attn", None)
    modules = {
        "q_proj.weight": getattr(getattr(attention, "q_proj", None), "weight", None),
        "k_proj.weight": getattr(getattr(attention, "k_proj", None), "weight", None),
        "v_proj.weight": getattr(getattr(attention, "v_proj", None), "weight", None),
        "out_proj.weight": getattr(
            getattr(attention, "out_proj", None), "weight", None
        ),
        "q_layernorm.weight": getattr(
            getattr(attention, "q_layernorm", None), "weight", None
        ),
        "k_layernorm.weight": getattr(
            getattr(attention, "k_layernorm", None), "weight", None
        ),
    }
    values = tuple(modules.values())
    if (
        any(
            not isinstance(value, torch.nn.Parameter)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or not torch.is_floating_point(value)
            or value.dtype not in {torch.bfloat16, torch.float32}
            for value in values
        )
        or len({value.dtype for value in values}) != 1
        or len({value.device for value in values}) != 1
        or len({id(value) for value in values}) != len(values)
    ):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            "Target attention must expose six independent dense BF16/FP32 "
            "Parameters on one device."
        )
    return MappingProxyType(modules)


def _gradient_inventory(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            role: {
                "object_id": None if value.grad is None else id(value.grad),
                "tensor_sha256": (
                    None
                    if value.grad is None
                    else _tensor_sha256(value.grad, label=f"{role} gradient")
                ),
            }
            for role, value in parameters.items()
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionSourceLocalOutputInstallation:
    """One authenticated O-only publication and its compiled candidate."""

    compiled: Lfm2CompiledAttentionGroupQdeimSourceLocalOutputResidual = field(
        repr=False,
        compare=False,
    )
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(
            self.compiled,
            Lfm2CompiledAttentionGroupQdeimSourceLocalOutputResidual,
        ):
            raise Lfm2SemanticAttentionSourceLocalOutputError(
                "Installation must retain one typed source-local O candidate."
            )
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))
        if not self.verify():
            raise Lfm2SemanticAttentionSourceLocalOutputError(
                "Source-local O installation failed result authentication."
            )

    def verify(self) -> bool:
        try:
            receipt = _verified_report(
                self.receipt,
                expected_schema=(
                    LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA
                ),
                label="source-local O installation receipt",
            )
            compiler = _verified_report(
                self.compiled.report,
                expected_schema=(
                    LFM2_ATTENTION_GROUP_QDEIM_SOURCE_LOCAL_OUTPUT_RESIDUAL_SCHEMA
                ),
                label="source-local O compiler receipt",
            )
            state_sha = lfm2_attention_group_qdeim_candidate_state_sha256(
                self.compiled.query_weight,
                self.compiled.key_weight,
                self.compiled.value_weight,
                self.compiled.out_weight_source_local_corrected,
            )
        except (Lfm2SemanticAttentionSourceLocalOutputError, Exception):
            return False
        return bool(
            receipt.get("source_local_compiler_result_sha256")
            == compiler.get("result_sha256")
            and receipt.get("source_local_corrected_candidate_state_sha256")
            == state_sha
            and receipt.get("only_out_projection_permitted_to_change") is True
        )


def compile_install_lfm2_semantic_attention_source_local_output(
    *,
    target_model: Any,
    layer_index: int,
    structural_candidate: Lfm2CompiledAttentionGroupQdeim,
    fit_cases: Sequence[Lfm2AttentionGroupQdeimSourceLocalOutputFitCase],
    ridge: float = 1.0e-6,
    fold_compute_device: str = "cpu",
) -> Lfm2SemanticAttentionSourceLocalOutputInstallation:
    """Compile and atomically install one bounded source-local O correction."""

    torch = _torch()
    if not isinstance(structural_candidate, Lfm2CompiledAttentionGroupQdeim):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            "structural_candidate must be one pristine intact-GQA QDEIM result."
        )
    structural_report = _verified_report(
        structural_candidate.report,
        expected_schema=LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        label="structural attention compiler receipt",
    )
    fit_cases = tuple(fit_cases)
    expected_fit_sha256 = (
        lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256(fit_cases)
    )
    expected_structural_state_sha256 = (
        lfm2_attention_group_qdeim_candidate_state_sha256(
            structural_candidate.query_weight,
            structural_candidate.key_weight,
            structural_candidate.value_weight,
            structural_candidate.out_weight,
        )
    )
    expected_chart_sha256 = _tensor_sha256(
        structural_candidate.residual_output_chart,
        label="structural residual output chart",
    )
    parameters = _attention_parameters(target_model, layer_index)
    structural_state = {
        "q_proj.weight": structural_candidate.query_weight,
        "k_proj.weight": structural_candidate.key_weight,
        "v_proj.weight": structural_candidate.value_weight,
        "out_proj.weight": structural_candidate.out_weight,
    }
    structural_hashes = {
        role: _tensor_sha256(value, label=f"structural {role}")
        for role, value in structural_state.items()
    }
    live_hashes_before = {
        role: _tensor_sha256(parameters[role], label=f"live {role}")
        for role in _ATTENTION_ROLES
    }
    if any(live_hashes_before[role] != structural_hashes[role] for role in _QKVO_ROLES):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            "Live Q/K/V/O does not equal the authenticated installed structural "
            "candidate."
        )
    if any(
        parameters[role].dtype != structural_state[role].dtype
        or tuple(parameters[role].shape) != tuple(structural_state[role].shape)
        for role in _QKVO_ROLES
    ):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            "Live Q/K/V/O topology differs from the structural candidate."
        )

    compiled = compile_lfm2_attention_group_qdeim_source_local_output_residual(
        structural_candidate=structural_candidate,
        expected_structural_candidate_state_sha256=(expected_structural_state_sha256),
        expected_structural_receipt_sha256=structural_report["result_sha256"],
        expected_residual_output_chart_sha256=expected_chart_sha256,
        fit_cases=fit_cases,
        expected_fit_evidence_sha256=expected_fit_sha256,
        ridge=ridge,
        compute_device=fold_compute_device,
    )
    compiler_report = _verified_report(
        compiled.report,
        expected_schema=(
            LFM2_ATTENTION_GROUP_QDEIM_SOURCE_LOCAL_OUTPUT_RESIDUAL_SCHEMA
        ),
        label="source-local O compiler receipt",
    )
    corrected_out = compiled.out_weight_source_local_corrected
    corrected_state_sha256 = lfm2_attention_group_qdeim_candidate_state_sha256(
        compiled.query_weight,
        compiled.key_weight,
        compiled.value_weight,
        corrected_out,
    )
    compiler_authentication = compiler_report.get("authentication", {})
    if (
        compiler_authentication.get("source_local_corrected_candidate_state_sha256")
        != corrected_state_sha256
        or compiler_authentication.get("fit_evidence_sha256_before")
        != expected_fit_sha256
        or compiler_report.get("materialization", {}).get(
            "qkv_bit_identical_to_structural_candidate"
        )
        is not True
    ):
        raise Lfm2SemanticAttentionSourceLocalOutputError(
            "Source-local O candidate lost structural or fit-evidence lineage."
        )

    before_model_sha256 = fingerprint_lfm2_parent(target_model).identity_sha256
    before_topology_sha256 = _topology_sha256(target_model)
    before_parameter_count = _parameter_count(target_model)
    before_state_hashes = _state_tensor_hashes(target_model)
    before_parameter_ids = {
        role: id(parameter) for role, parameter in parameters.items()
    }
    before_storage_pointers = {
        role: int(parameter.data_ptr()) for role, parameter in parameters.items()
    }
    before_gradients = _gradient_inventory(parameters)
    out_parameter = parameters["out_proj.weight"]
    out_snapshot = out_parameter.detach().to(device="cpu").contiguous().clone()
    expected_out_sha256 = _tensor_sha256(
        corrected_out,
        label="corrected source-local out projection",
    )
    expected_changed = (
        ()
        if live_hashes_before["out_proj.weight"] == expected_out_sha256
        else (f"model.layers.{layer_index}.self_attn.out_proj.weight",)
    )

    def transaction() -> None:
        with torch.no_grad():
            out_parameter.copy_(
                corrected_out.to(
                    device=out_parameter.device,
                    dtype=out_parameter.dtype,
                )
            )

    try:
        _unused, guard = run_fold_only_callback(
            transaction,
            phase="lfm2-semantic-source-local-attention-o-transaction",
            protected_models=(),
            freeze_parameters=False,
        )
        after_state_hashes = _state_tensor_hashes(target_model)
        changed_state_keys = _changed_state_keys(
            before_state_hashes,
            after_state_hashes,
        )
        after_hashes = {
            role: _tensor_sha256(parameters[role], label=f"installed {role}")
            for role in _ATTENTION_ROLES
        }
        after_gradients = _gradient_inventory(parameters)
        after_model_sha256 = fingerprint_lfm2_parent(target_model).identity_sha256
        after_topology_sha256 = _topology_sha256(target_model)
        after_parameter_count = _parameter_count(target_model)
        if (
            changed_state_keys != expected_changed
            or after_hashes["out_proj.weight"] != expected_out_sha256
            or any(
                after_hashes[role] != live_hashes_before[role]
                for role in _ATTENTION_ROLES
                if role != "out_proj.weight"
            )
            or {role: id(value) for role, value in parameters.items()}
            != before_parameter_ids
            or {role: int(value.data_ptr()) for role, value in parameters.items()}
            != before_storage_pointers
            or after_gradients != before_gradients
            or after_topology_sha256 != before_topology_sha256
            or after_parameter_count != before_parameter_count
            or (bool(expected_changed) and after_model_sha256 == before_model_sha256)
            or (not expected_changed and after_model_sha256 != before_model_sha256)
        ):
            raise Lfm2SemanticAttentionSourceLocalOutputError(
                "O-only publication changed stale, unrelated, topology, identity, "
                "storage, or gradient state."
            )
    except BaseException as exc:
        with torch.no_grad():
            out_parameter.copy_(
                out_snapshot.to(
                    device=out_parameter.device,
                    dtype=out_parameter.dtype,
                )
            )
        if _state_tensor_hashes(target_model) != before_state_hashes:
            raise Lfm2SemanticAttentionSourceLocalOutputError(
                "Source-local O transaction failed and exact rollback was not proven."
            ) from exc
        raise

    guard_body = guard.as_dict()
    body = {
        "schema": LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA,
        "status": (
            "installed-bounded-source-local-attention-o"
            if expected_changed
            else "authenticated-source-local-attention-o-noop"
        ),
        "layer_index": layer_index,
        "installed": bool(expected_changed),
        "install_count": int(bool(expected_changed)),
        "structural_candidate_state_sha256": expected_structural_state_sha256,
        "structural_compiler_result_sha256": structural_report["result_sha256"],
        "source_local_compiler_result_sha256": compiler_report["result_sha256"],
        "source_local_fit_evidence_sha256": expected_fit_sha256,
        "source_local_corrected_candidate_state_sha256": corrected_state_sha256,
        "residual_output_chart_sha256": expected_chart_sha256,
        "model_state_before_sha256": before_model_sha256,
        "model_state_after_sha256": after_model_sha256,
        "changed_state_keys": list(changed_state_keys),
        "only_permitted_changed_state_key": (
            f"model.layers.{layer_index}.self_attn.out_proj.weight"
        ),
        "only_out_projection_permitted_to_change": True,
        "qkv_bit_identical_to_structural_candidate": all(
            after_hashes[role] == structural_hashes[role] for role in _QKVO_ROLES[:3]
        ),
        "qk_norms_unchanged": all(
            after_hashes[role] == live_hashes_before[role]
            for role in _ATTENTION_ROLES[-2:]
        ),
        "attention_tensor_sha256_before": live_hashes_before,
        "attention_tensor_sha256_after": after_hashes,
        "parameter_count_before": before_parameter_count,
        "parameter_count_after": after_parameter_count,
        "parameter_count_changed": False,
        "target_topology_sha256_before": before_topology_sha256,
        "target_topology_sha256_after": after_topology_sha256,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "fit_partition_only": True,
        "heldout_rows_used": False,
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
        "fold_only_transaction_guard": guard_body,
        "numeric_metrics_have_semantic_decision_authority": False,
        "numeric_threshold_auto_accept_permitted": False,
        "numeric_threshold_auto_veto_permitted": False,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
    }
    return Lfm2SemanticAttentionSourceLocalOutputInstallation(
        compiled=compiled,
        receipt=_seal(body),
    )


__all__ = [
    "LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA",
    "Lfm2SemanticAttentionSourceLocalOutputError",
    "Lfm2SemanticAttentionSourceLocalOutputInstallation",
    "compile_install_lfm2_semantic_attention_source_local_output",
]
