"""Authenticated interface seams for the exact dense LFM2 composite fold.

The semantic body compiler and the causal token-head compiler intentionally
meet at an *untied* blank interface.  This module owns the two mutations at
that boundary:

* install the standard semantic-foundation interface into both independent
  zero-blank tables; and
* after the body has been folded and replayed, install a causal residual table
  into the output head alone.

Context composition happens in the parent hidden coordinate before the
standard foundation fit.  The structural installer recomputes that contextual
source from the full donor table and sealed sequence plan, then authenticates
the foundation solve against it.  It does not invent another interface type or
re-seal the interface under a competing schema.

Both operations are closed-form tensor copies under ``no_grad``.  They do not
run either model, construct an optimizer, invoke autograd/backward, train,
prune, quantize, sparsify, share runtime weights, or consult a numeric/manual
quality threshold.  Numeric and decoded-behavior observations remain evidence
for human/automatic diagnosis outside these structural integrity checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_state,
    fingerprint_model_state_entries,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetSynthesisResult,
    verify_compiler_authenticated_target,
)
from neural_forge.hf_lfm2_bridge_only_control import (
    _all_named_tensors,
    _storage_identity,
    _tensor_sha256 as _frontier_tensor_sha256,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_interface_frontier import (
    LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA,
    Lfm2CausalResidualTokenBridge,
    Lfm2ChildCausalResidualEvidence,
    compile_lfm2_contextual_input_table,
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_lfm2_semantic_foundation import (
    LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
    Lfm2SemanticFoundationInterfaceSolve,
)
from neural_forge.hf_tied_causal_interface_fold import TiedCausalInterfaceFold


LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA = (
    "neural-forge-lfm2-exact4x-independent-structural-interface-install-v1"
)
LFM2_INDEPENDENT_CAUSAL_OUTPUT_INSTALL_SCHEMA = (
    "neural-forge-lfm2-exact4x-independent-causal-output-install-v1"
)


class Lfm2Exact4xCompositeInterfaceError(RuntimeError):
    """A composite interface transition failed closed."""


@dataclass(frozen=True, slots=True)
class Lfm2IndependentStructuralInterfaceInstall:
    """A compiler blank after its independent structural tables are installed."""

    model: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.model is None:
            raise TypeError("model must be a materialized LFM2 child.")
        value = _verify_seal(
            self.receipt,
            field="result_sha256",
            label="independent structural-interface install",
        )
        if value.get("schema") != LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA:
            raise Lfm2Exact4xCompositeInterfaceError(
                "Structural-interface install receipt uses the wrong schema."
            )
        object.__setattr__(self, "receipt", MappingProxyType(value))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2IndependentCausalOutputInstall:
    """A composite child after its terminal output-only causal mutation."""

    model: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.model is None:
            raise TypeError("model must be a materialized LFM2 child.")
        value = _verify_seal(
            self.receipt,
            field="result_sha256",
            label="independent causal-output install",
        )
        if value.get("schema") != LFM2_INDEPENDENT_CAUSAL_OUTPUT_INSTALL_SCHEMA:
            raise Lfm2Exact4xCompositeInterfaceError(
                "Causal-output install receipt uses the wrong schema."
            )
        object.__setattr__(self, "receipt", MappingProxyType(value))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise RuntimeError("The exact LFM2 composite interface requires PyTorch.") from exc
    return torch


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _verify_seal(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2Exact4xCompositeInterfaceError(f"{label} must be a mapping.")
    result = dict(value)
    body = dict(result)
    supplied = _digest(body.pop(field, None), f"{label} {field}")
    if canonical_sha256(body) != supplied:
        raise Lfm2Exact4xCompositeInterfaceError(f"{label} seal is invalid.")
    return result


def _foundation_tensor_sha256(value: Any) -> str:
    """Match the semantic-foundation tensor authentication convention."""

    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    encoded = json.dumps(
        descriptor,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(encoded)
    digest.update(tensor.reshape(-1).view(_torch().uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_topology(model: Any) -> dict[str, Any]:
    """Return a pointer-free topology/alias description stable across devices."""

    torch = _torch()
    try:
        named_parameters = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        named_parameters = tuple(model.named_parameters())
    try:
        named_buffers = tuple(model.named_buffers(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        named_buffers = tuple(model.named_buffers())
    if not named_parameters:
        raise Lfm2Exact4xCompositeInterfaceError("LFM2 child has no parameters.")

    groups: dict[tuple[int, int, int, int], list[str]] = {}
    parameters = []
    for name, parameter in named_parameters:
        if (
            not isinstance(parameter, torch.nn.Parameter)
            or parameter.device.type == "meta"
            or parameter.layout != torch.strided
            or parameter.is_sparse
        ):
            raise Lfm2Exact4xCompositeInterfaceError(
                f"Parameter {name!r} is not one materialized dense Parameter."
            )
        identity = (
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.storage_offset()),
            int(parameter.numel()),
            int(parameter.element_size()),
        )
        groups.setdefault(identity, []).append(str(name))
        parameters.append(
            {
                "name": str(name),
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    buffers = []
    for name, buffer in named_buffers:
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device.type == "meta"
            or buffer.layout != torch.strided
        ):
            raise Lfm2Exact4xCompositeInterfaceError(
                f"Buffer {name!r} is not one materialized dense tensor."
            )
        buffers.append(
            {
                "name": str(name),
                "shape": [int(size) for size in buffer.shape],
                "dtype": str(buffer.dtype),
            }
        )
    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None)
    if (
        getattr(config, "model_type", None) != "lfm2"
        or not isinstance(layer_types, (list, tuple))
        or not layer_types
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Target must be a materialized LFM2 causal language model."
        )
    hidden_size = int(getattr(config, "hidden_size"))
    query_heads = int(getattr(config, "num_attention_heads"))
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is None:
        if hidden_size % query_heads:
            raise Lfm2Exact4xCompositeInterfaceError(
                "LFM2 hidden width is not divisible by its query-head count."
            )
        head_dim = hidden_size // query_heads
    else:
        head_dim = int(configured_head_dim)
    return {
        "schema": "neural-forge-lfm2-live-parameter-topology-v1",
        "config": {
            "model_type": "lfm2",
            "vocab_size": int(getattr(config, "vocab_size")),
            "hidden_size": hidden_size,
            "intermediate_size": int(getattr(config, "intermediate_size")),
            "num_hidden_layers": int(getattr(config, "num_hidden_layers")),
            "num_attention_heads": int(getattr(config, "num_attention_heads")),
            "num_key_value_heads": int(getattr(config, "num_key_value_heads")),
            "head_dim": head_dim,
            "layer_types": [str(value) for value in layer_types],
            "tie_word_embeddings": bool(
                getattr(config, "tie_word_embeddings", True)
            ),
        },
        "parameters": sorted(parameters, key=lambda item: item["name"]),
        "buffers": sorted(buffers, key=lambda item: item["name"]),
        "parameter_alias_groups": sorted(
            (sorted(names) for names in groups.values()),
            key=lambda names: names,
        ),
    }


def _runtime_layout(model: Any) -> tuple[tuple[Any, ...], ...]:
    """Process-local object/storage signature used only around one copy."""

    try:
        named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        named = tuple(model.named_parameters())
    return tuple(
        sorted(
            (
                str(name),
                id(parameter),
                *_storage_identity(parameter),
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            for name, parameter in named
        )
    )


def _interface(model: Any) -> tuple[Any, Any, set[str], set[str]]:
    torch = _torch()
    input_module = getattr(model, "get_input_embeddings", lambda: None)()
    output_module = getattr(model, "get_output_embeddings", lambda: None)()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or not isinstance(output_weight, torch.nn.Parameter)
        or input_weight.device.type == "meta"
        or output_weight.device.type == "meta"
        or input_weight.layout != torch.strided
        or output_weight.layout != torch.strided
        or input_weight is output_weight
        or _storage_identity(input_weight) == _storage_identity(output_weight)
        or bool(getattr(model.config, "tie_word_embeddings", True))
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Composite folding requires distinct dense untied input/output Parameters."
        )
    input_identity = _storage_identity(input_weight)
    output_identity = _storage_identity(output_weight)
    input_names = {
        name
        for name, value in _all_named_tensors(model)
        if _storage_identity(value) == input_identity
    }
    output_names = {
        name
        for name, value in _all_named_tensors(model)
        if _storage_identity(value) == output_identity
    }
    if not input_names or not output_names or input_names & output_names:
        raise Lfm2Exact4xCompositeInterfaceError(
            "Input/output Parameter paths are empty or aliased."
        )
    return input_weight, output_weight, input_names, output_names


def _verified_parameter_count(
    target: DynamicTargetSynthesisResult,
    model: Any,
) -> tuple[int, str]:
    inventory = dict(independent_parameter_inventory(model))
    observed = int(inventory["unique_parameters"])
    expected = int(target.request.exact_unique_parameters)
    if (
        observed != expected
        or observed != int(target.graph.unique_parameter_count)
        or observed != int(target.resource_profile.unique_parameters)
        or inventory.get("independent_methods_agree") is not True
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Live independent parameter count differs from the exact target plan."
        )
    return observed, _digest(inventory.get("inventory_sha256"), "inventory SHA-256")


def _authenticate_foundation(
    solve: Lfm2SemanticFoundationInterfaceSolve,
    contextual_source: Any,
) -> tuple[TiedCausalInterfaceFold, dict[str, Any]]:
    if not isinstance(solve, Lfm2SemanticFoundationInterfaceSolve):
        raise TypeError(
            "foundation_solve must be Lfm2SemanticFoundationInterfaceSolve."
        )
    interface = solve.interface
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Foundation solve lacks its standard tied causal interface."
        )
    report = _verify_seal(
        solve.report,
        field="result_sha256",
        label="semantic-foundation interface solve",
    )
    active_report = _verify_seal(
        interface.report,
        field="report_sha256",
        label="active standard interface",
    )
    if (
        report.get("schema") != LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA
        or report.get("active_interface_report") != active_report
        or report.get("active_interface_report_sha256")
        != active_report["report_sha256"]
        or report.get("source_tied_table_sha256")
        != _foundation_tensor_sha256(contextual_source)
        or report.get("active_target_weight_sha256")
        != _foundation_tensor_sha256(interface.target_weight)
        or report.get("producer_encoder_sha256")
        != _foundation_tensor_sha256(interface.producer_encoder)
        or report.get("target_hidden_width") != interface.target_width
        or report.get("only_final_norm_payload_materialized") is not True
        or any(
            report.get(name) != 0
            for name in (
                "legacy_layer_compiler_calls",
                "parent_model_forward_calls",
                "child_model_forward_calls",
                "training_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Foundation interface tensors, source coordinate, or operation audit differ."
        )
    for field in (
        "compiler_cache_key",
        "cache_parent_fingerprint",
        "cache_manifest_sha256",
        "bridge_artifact_sha256",
        "trajectory_corpus_sha256",
        "trace_adapter_sha256",
        "selective_catalog_preparation_receipt_sha256",
        "selective_catalog_binding_sha256",
        "selective_catalog_manifest_sha256",
        "selective_catalog_artifact_sha256",
        "final_norm_authentication_sha256",
    ):
        _digest(report.get(field), f"foundation {field}")
    for nested_name, seal_name in (
        ("base_interface_report", "base_interface_report_sha256"),
        ("causal_anchor_report", "causal_anchor_report_sha256"),
    ):
        nested = _verify_seal(
            report.get(nested_name),
            field="report_sha256",
            label=f"foundation {nested_name}",
        )
        if nested["report_sha256"] != report.get(seal_name):
            raise Lfm2Exact4xCompositeInterfaceError(
                f"Foundation {nested_name} ancestry differs."
            )
    contract = _verify_seal(
        report.get("causal_anchor_selection_contract"),
        field="contract_sha256",
        label="foundation anchor selection contract",
    )
    if (
        contract.get("candidate_eligibility_basis")
        != "finite_dense_shape_dtype_device_correct_effectful_residual"
        or contract.get("mapped_parent_behavior_retention_review_goal_is_hard_gate")
        is not False
        or contract.get("relative_table_drift_review_goal_is_hard_gate") is not False
        or contract.get("numeric_metrics_role")
        != "deterministic_diagnostic_ordering_only"
        or contract.get("numeric_metrics_may_restore_base_control") is not False
        or contract.get(
            "base_control_only_if_all_analytic_residuals_invalid_or_identical"
        )
        is not True
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Foundation interface uses behavioral numbers as decision authority."
        )
    return interface, report


def install_lfm2_independent_structural_interface(
    target: DynamicTargetSynthesisResult,
    foundation_solve: Lfm2SemanticFoundationInterfaceSolve,
    *,
    donor_tied_table: Any,
    sequence_plan: Mapping[str, Any],
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
) -> Lfm2IndependentStructuralInterfaceInstall:
    """Install a contextual standard interface into a compiler-created blank.

    ``donor_tied_table`` is the full donor-vocabulary table.  Contextual token
    composition is deliberately performed before the foundation projection;
    the resulting source hash must be exactly the one sealed by
    ``foundation_solve``.
    """

    torch = _torch()
    if not isinstance(target, DynamicTargetSynthesisResult):
        raise TypeError("target must be DynamicTargetSynthesisResult.")
    topology_seal = verify_compiler_authenticated_target(target)
    provenance = target.blank_provenance
    if provenance is None or any(
        getattr(provenance, field) is not True
        for field in (
            "compiler_created_blank",
            "parent_child_alias_free",
            "zeroed_under_no_grad",
            "dense_materialized_parameters",
            "all_parameters_zero",
            "parameter_gradients_absent",
            "runtime_buffers_preserved",
        )
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Target lacks compiler-owned dense zero-blank provenance."
        )
    model = target.model
    if fingerprint_model_state(model) != target.blank_state_fingerprint:
        raise Lfm2Exact4xCompositeInterfaceError(
            "Live target is no longer the compiler-authenticated blank."
        )
    input_weight, output_weight, input_names, output_names = _interface(model)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Compiler blank unexpectedly contains Parameter gradients."
        )
    observed_parameters, inventory_before_sha256 = _verified_parameter_count(
        target,
        model,
    )
    topology = _parameter_topology(model)
    topology_sha256 = canonical_sha256(topology)
    runtime_layout_before = _runtime_layout(model)
    state_before = fingerprint_model_state(model)
    entries_before = dict(fingerprint_model_state_entries(model))
    body_before = fingerprint_lfm2_causal_body_inventory(model)
    donor_before = _frontier_tensor_sha256(donor_tied_table)
    contextual_source = compile_lfm2_contextual_input_table(
        donor_tied_table,
        sequence_plan,
        strict_anchor_ppm=strict_anchor_ppm,
        preserve_anchor_norm=preserve_anchor_norm,
    )
    if _frontier_tensor_sha256(donor_tied_table) != donor_before:
        raise Lfm2Exact4xCompositeInterfaceError(
            "Contextual source compilation mutated the donor table."
        )
    interface, foundation_report = _authenticate_foundation(
        foundation_solve,
        contextual_source,
    )
    if (
        interface.source_width != int(contextual_source.shape[1])
        or interface.vocabulary_size != int(contextual_source.shape[0])
        or tuple(interface.target_weight.shape) != tuple(input_weight.shape)
        or tuple(output_weight.shape) != tuple(input_weight.shape)
        or interface.target_width != int(input_weight.shape[1])
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Contextual foundation interface does not fit the exact blank."
        )
    table = interface.target_weight.detach().to(
        device=input_weight.device,
        dtype=input_weight.dtype,
    ).contiguous()
    if not bool(torch.isfinite(table).all()) or int(table.numel()) <= 0:
        raise Lfm2Exact4xCompositeInterfaceError(
            "Foundation target table is empty or nonfinite after target projection."
        )
    with torch.no_grad():
        input_weight.copy_(table)
        output_weight.copy_(
            table.to(device=output_weight.device, dtype=output_weight.dtype)
        )
    state_after = fingerprint_model_state(model)
    entries_after = dict(fingerprint_model_state_entries(model))
    body_after = fingerprint_lfm2_causal_body_inventory(model)
    runtime_layout_after = _runtime_layout(model)
    topology_after = _parameter_topology(model)
    inventory_after = dict(independent_parameter_inventory(model))
    input_sha256 = _frontier_tensor_sha256(input_weight)
    output_sha256 = _frontier_tensor_sha256(output_weight)
    allowed_names = input_names | output_names
    changed_names = sorted(
        name
        for name in entries_before
        if entries_before[name] != entries_after.get(name)
    )
    if (
        state_after == state_before
        or body_after != body_before
        or runtime_layout_after != runtime_layout_before
        or topology_after != topology
        or int(inventory_after.get("unique_parameters", -1)) != observed_parameters
        or inventory_after.get("inventory_sha256") != inventory_before_sha256
        or not changed_names
        or not set(changed_names).issubset(allowed_names)
        or not input_names.issubset(changed_names)
        or not output_names.issubset(changed_names)
        or input_sha256 != output_sha256
        or not bool(
            torch.equal(
                input_weight.detach().to(device="cpu"),
                output_weight.detach().to(device="cpu"),
            )
        )
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Structural interface mutation changed body/topology/count or missed a table."
        )
    plan_sha256 = _digest(sequence_plan.get("plan_sha256"), "sequence plan SHA-256")
    body = {
        "schema": LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
        "status": "contextual-standard-foundation-interface-installed-into-independent-blank",
        "target_synthesis_fingerprint": target.fingerprint,
        "target_request_fingerprint": target.request.fingerprint,
        "compiler_authenticated_topology_sha256": topology_seal,
        "compiler_blank_provenance_sha256": target.blank_provenance_digest,
        "compiler_blank_state_sha256": target.blank_state_fingerprint,
        "target_topology_sha256": topology_sha256,
        "exact_target_unique_parameters": observed_parameters,
        "parameter_inventory_sha256": inventory_before_sha256,
        "sequence_plan_sha256": plan_sha256,
        "donor_tied_table_sha256": donor_before,
        "contextual_source_table_sha256": _frontier_tensor_sha256(
            contextual_source
        ),
        "contextual_source_foundation_sha256": _foundation_tensor_sha256(
            contextual_source
        ),
        "context_composition_preceded_foundation_projection": True,
        "foundation_interface_receipt_sha256": foundation_report["result_sha256"],
        "foundation_active_interface_report_sha256": foundation_report[
            "active_interface_report_sha256"
        ],
        "foundation_source_tied_table_sha256": foundation_report[
            "source_tied_table_sha256"
        ],
        "foundation_target_weight_sha256": foundation_report[
            "active_target_weight_sha256"
        ],
        "foundation_producer_encoder_sha256": foundation_report[
            "producer_encoder_sha256"
        ],
        "installed_input_table_sha256": input_sha256,
        "installed_output_baseline_sha256": output_sha256,
        "input_and_output_parameters_distinct": True,
        "input_and_output_storage_distinct": True,
        "input_output_values_equal_at_structural_install": True,
        "input_parameter_names": sorted(input_names),
        "output_parameter_names": sorted(output_names),
        "changed_state_paths": changed_names,
        "only_interface_state_changed": True,
        "causal_body_inventory_before_sha256": body_before,
        "causal_body_inventory_after_sha256": body_after,
        "child_state_before_sha256": state_before,
        "child_state_after_sha256": state_after,
        "parameter_topology_preserved": True,
        "independent_parameter_count_preserved": True,
        "diagnostic_metrics_used_as_installation_gate": False,
        "manual_behavior_scores_used_as_installation_gate": False,
        "behavior_retention_goal_is_not_a_structural_cutoff": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "child_generate_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "network_access": False,
        },
    }
    return Lfm2IndependentStructuralInterfaceInstall(
        model=model,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


def _authenticate_causal_ancestry(
    bridge: Lfm2CausalResidualTokenBridge,
    evidence: Lfm2ChildCausalResidualEvidence,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(bridge, Lfm2CausalResidualTokenBridge):
        raise TypeError("causal_bridge must be Lfm2CausalResidualTokenBridge.")
    if not isinstance(evidence, Lfm2ChildCausalResidualEvidence):
        raise TypeError(
            "child_feature_evidence must be Lfm2ChildCausalResidualEvidence."
        )
    bridge_receipt = bridge.as_dict()
    evidence_receipt = evidence.as_dict()
    if (
        bridge_receipt.get("schema") != LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA
        or bridge_receipt.get("child_feature_evidence_sha256")
        != evidence_receipt.get("evidence_sha256")
        or bridge_receipt.get("trie_conditional_evidence_sha256")
        != evidence_receipt.get("trie_evidence_sha256")
        or bridge_receipt.get("child_model_state_sha256")
        != evidence_receipt.get("child_model_state_before_sha256")
        or bridge_receipt.get("context_free_inverse_output_plan_used") is not False
        or bridge_receipt.get("compiled_against_parent_hidden_features") is not False
        or bridge_receipt.get("causal_child_residual_features_used") is not True
        or bridge_receipt.get("materialized_runtime_operator")
        != "ordinary-dense-output-table"
        or bridge_receipt.get("runtime_parent_or_adapter_required") is not False
        or bridge_receipt.get("diagnostic_metrics_are_not_acceptance_gates")
        is not True
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Causal bridge is not bound to the supplied child-coordinate evidence."
        )
    common_fields = (
        "parent_terminal_evidence_sha256",
        "cache_key",
        "parent_identity_sha256",
        "selective_session_sha256",
        "manifest_sha256",
        "completion_sha256",
        "semantic_alignment_sha256",
        "tokenizer_live_session_sha256",
        "causal_root_rows_sha256",
        "contextual_transition_plan_sha256",
        "bridge_artifact_sha256",
        "bridge_profile_sha256",
        "donor_tokenizer_semantics_sha256",
        "target_tokenizer_semantics_sha256",
        "target_prefix_rows_sha256",
        "causal_input_table_sha256",
        "causal_body_inventory_sha256",
    )
    for field in common_fields:
        left = _digest(bridge_receipt.get(field), f"causal bridge {field}")
        right = _digest(evidence_receipt.get(field), f"child evidence {field}")
        if left != right:
            raise Lfm2Exact4xCompositeInterfaceError(
                f"Causal bridge and child evidence disagree on {field}."
            )
    audit = bridge_receipt.get("operation_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("network_access") is not False
        or any(
            audit.get(name) != 0
            for name in (
                "training_steps",
                "optimizer_constructions",
                "optimizer_steps",
                "autograd_grad_calls",
                "backward_calls",
                "parent_forward_calls",
                "parent_generate_calls",
                "parent_trace_recaptures",
                "parameters_pruned",
                "parameters_quantized",
                "parameters_sparsified",
            )
        )
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Causal bridge operation audit permits a forbidden operation."
        )
    return bridge_receipt, evidence_receipt


def install_lfm2_independent_causal_output_head(
    structural_install: Lfm2IndependentStructuralInterfaceInstall,
    causal_bridge: Lfm2CausalResidualTokenBridge,
    child_feature_evidence: Lfm2ChildCausalResidualEvidence,
    *,
    expected_input_table_sha256: str,
    expected_body_inventory_sha256: str,
) -> Lfm2IndependentCausalOutputInstall:
    """Install a child-coordinate causal bridge into the output Parameter only.

    Body and final-W2 mutations are expected between the structural install and
    this call.  Their live fingerprint must match the replay evidence and the
    caller's authenticated expectation; the old full-model state from the
    structural receipt is therefore deliberately *not* required to remain
    unchanged.
    """

    torch = _torch()
    if not isinstance(
        structural_install,
        Lfm2IndependentStructuralInterfaceInstall,
    ):
        raise TypeError(
            "structural_install must be Lfm2IndependentStructuralInterfaceInstall."
        )
    expected_input = _digest(
        expected_input_table_sha256,
        "expected_input_table_sha256",
    )
    expected_body = _digest(
        expected_body_inventory_sha256,
        "expected_body_inventory_sha256",
    )
    structural_receipt = _verify_seal(
        structural_install.receipt,
        field="result_sha256",
        label="independent structural-interface install",
    )
    bridge_receipt, evidence_receipt = _authenticate_causal_ancestry(
        causal_bridge,
        child_feature_evidence,
    )
    model = structural_install.model
    input_weight, output_weight, input_names, output_names = _interface(model)
    input_before = _frontier_tensor_sha256(input_weight)
    output_before = _frontier_tensor_sha256(output_weight)
    body_before = fingerprint_lfm2_causal_body_inventory(model)
    state_before = fingerprint_model_state(model)
    topology = _parameter_topology(model)
    topology_sha256 = canonical_sha256(topology)
    runtime_layout_before = _runtime_layout(model)
    entries_before = dict(fingerprint_model_state_entries(model))
    parameter_inventory_before = dict(independent_parameter_inventory(model))
    if (
        structural_receipt.get("target_topology_sha256") != topology_sha256
        or structural_receipt.get("installed_input_table_sha256") != expected_input
        or structural_receipt.get("installed_output_baseline_sha256")
        != expected_input
        or input_before != expected_input
        or output_before != expected_input
        or not bool(
            torch.equal(
                input_weight.detach().to(device="cpu"),
                output_weight.detach().to(device="cpu"),
            )
        )
        or body_before != expected_body
        or bridge_receipt.get("causal_input_table_sha256") != expected_input
        or bridge_receipt.get("structural_output_table_sha256") != expected_input
        or bridge_receipt.get("causal_body_inventory_sha256") != expected_body
        or evidence_receipt.get("causal_input_table_sha256") != expected_input
        or evidence_receipt.get("causal_body_inventory_sha256") != expected_body
        or bridge_receipt.get("sequence_plan_sha256")
        != structural_receipt.get("sequence_plan_sha256")
        or state_before != evidence_receipt.get("child_model_state_before_sha256")
        or state_before != evidence_receipt.get("child_model_state_after_sha256")
        or state_before != bridge_receipt.get("child_model_state_sha256")
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Live input/body/state is not the child coordinate used by the causal solve."
        )
    if tuple(causal_bridge.table.shape) != tuple(output_weight.shape):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Causal output table does not fit the independent output Parameter."
        )
    compiled = causal_bridge.table.detach().to(
        device=output_weight.device,
        dtype=output_weight.dtype,
    ).contiguous()
    if not bool(torch.isfinite(compiled).all()):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Causal output table is nonfinite after target projection."
        )
    with torch.no_grad():
        output_weight.copy_(compiled)
    state_after = fingerprint_model_state(model)
    entries_after = dict(fingerprint_model_state_entries(model))
    input_after = _frontier_tensor_sha256(input_weight)
    output_after = _frontier_tensor_sha256(output_weight)
    body_after = fingerprint_lfm2_causal_body_inventory(model)
    topology_after = _parameter_topology(model)
    runtime_layout_after = _runtime_layout(model)
    parameter_inventory_after = dict(independent_parameter_inventory(model))
    changed_names = sorted(
        name
        for name in entries_before
        if entries_before[name] != entries_after.get(name)
    )
    if (
        input_after != input_before
        or body_after != body_before
        or topology_after != topology
        or runtime_layout_after != runtime_layout_before
        or parameter_inventory_after.get("unique_parameters")
        != parameter_inventory_before.get("unique_parameters")
        or parameter_inventory_after.get("inventory_sha256")
        != parameter_inventory_before.get("inventory_sha256")
        or not set(changed_names).issubset(output_names)
        or bool(set(changed_names) & input_names)
        or output_after != _frontier_tensor_sha256(compiled)
    ):
        raise Lfm2Exact4xCompositeInterfaceError(
            "Causal install changed input/body/topology/count or missed its output."
        )
    body = {
        "schema": LFM2_INDEPENDENT_CAUSAL_OUTPUT_INSTALL_SCHEMA,
        "status": "child-coordinate-causal-table-installed-output-only",
        "structural_interface_install_sha256": structural_receipt["result_sha256"],
        "target_topology_sha256": topology_sha256,
        "exact_target_unique_parameters": int(
            parameter_inventory_before["unique_parameters"]
        ),
        "parameter_inventory_sha256": parameter_inventory_before[
            "inventory_sha256"
        ],
        "sequence_plan_sha256": structural_receipt["sequence_plan_sha256"],
        "expected_input_table_sha256": expected_input,
        "expected_body_inventory_sha256": expected_body,
        "child_feature_evidence_sha256": evidence_receipt["evidence_sha256"],
        "causal_residual_token_bridge_sha256": bridge_receipt["result_sha256"],
        "trie_conditional_evidence_sha256": bridge_receipt[
            "trie_conditional_evidence_sha256"
        ],
        "input_table_before_sha256": input_before,
        "input_table_after_sha256": input_after,
        "body_inventory_before_sha256": body_before,
        "body_inventory_after_sha256": body_after,
        "output_baseline_sha256": output_before,
        "installed_output_table_sha256": output_after,
        "child_state_before_sha256": state_before,
        "child_state_after_sha256": state_after,
        "output_parameter_names": sorted(output_names),
        "changed_state_paths": changed_names,
        "output_table_values_changed": bool(changed_names),
        "only_distinct_output_parameter_may_change": True,
        "input_and_body_coordinate_preserved": True,
        "parameter_topology_preserved": True,
        "independent_parameter_count_preserved": True,
        "installed_after_authenticated_child_feature_replay": True,
        "future_interface_or_body_mutation_permitted_by_this_receipt": False,
        "diagnostic_metrics_used_as_installation_gate": False,
        "manual_behavior_scores_used_as_installation_gate": False,
        "behavior_retention_goal_is_not_a_structural_cutoff": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "child_generate_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "network_access": False,
        },
    }
    return Lfm2IndependentCausalOutputInstall(
        model=model,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


__all__ = [
    "LFM2_INDEPENDENT_CAUSAL_OUTPUT_INSTALL_SCHEMA",
    "LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA",
    "Lfm2Exact4xCompositeInterfaceError",
    "Lfm2IndependentCausalOutputInstall",
    "Lfm2IndependentStructuralInterfaceInstall",
    "install_lfm2_independent_causal_output_head",
    "install_lfm2_independent_structural_interface",
]
