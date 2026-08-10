"""Component diagnostics inside one complete exact-budget child fold.

The historical LFM2 ablation code installs compact operators in a parent
shell.  That is useful localization evidence, but it is not a standalone
compacted child.  This module describes the stricter experiment: discover
components from the common circuit IR, keep one exact compact-child topology,
fold every component, and substitute an alternate fold strategy only for the
selected focus.  A focus is never permission to leave the rest blank or to
publish a partial/hybrid model.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    JsonScalar,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_json_bytes,
    canonical_sha256,
)
from .dynamic_fold_executor import (
    DynamicFoldRun,
    FoldOperatorRegistry,
)
from .dynamic_fold_planner import (
    PLAN_SCHEMA as DYNAMIC_FOLD_PLAN_SCHEMA,
    DynamicFoldPlan,
    FoldAction,
    validate_fold_method_contract,
)
from .dynamic_fold_registry import (
    ExperimentLedger,
    ExperimentSignature,
    FoldMethodRegistry,
    default_method_registry,
)

SCHEMA = "neural-forge-whole-child-component-diagnostic-plan-v2"
BINDING_SCHEMA = "neural-forge-whole-child-component-diagnostic-binding-v2"
DISPATCH_SCHEMA = "neural-forge-whole-child-component-dispatch-v2"
PLAN_AUTHENTICATION_MANIFEST_SCHEMA = (
    "neural-forge-whole-child-component-plan-authentication-manifest-v2"
)
TRANSACTION_REPORT_BINDING_KEY = "whole_child_component_diagnostic_binding"
DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY = "whole_child_component_diagnostic_dispatch_json"
DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY = (
    "whole_child_component_diagnostic_base_action_id"
)
WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY = (
    "complete_whole_child_component_dispatch_v2"
)
_OPEN_LEDGER_REASON = "No matching closure or prior exact attempt."

_FORBIDDEN_CONFIGURATION_TOKENS = frozenset(
    {
        "adam",
        "adamw",
        "autograd",
        "backprop",
        "backward",
        "finetune",
        "gradient",
        "lora",
        "optimizer",
        "prune",
        "pruning",
        "quantization",
        "quantize",
        "sgd",
        "share",
        "shared",
        "sharing",
        "sidecar",
        "sparse",
        "train",
        "training",
    }
)


class WholeChildComponentDiagnosticError(ValueError):
    """The diagnostic no longer describes a complete compact child fold."""


def _forbidden_words(value: Any) -> tuple[str, ...]:
    words = {token for token in re.split(r"[^a-z0-9]+", str(value).casefold()) if token}
    return tuple(sorted(words & _FORBIDDEN_CONFIGURATION_TOKENS))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WholeChildComponentDiagnosticError(f"{label} must be nonempty.")
    return value.strip()


def _digest(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise WholeChildComponentDiagnosticError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return result


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(sorted(_text(value, label) for value in values))
    if len(result) != len(set(result)):
        raise WholeChildComponentDiagnosticError(f"{label} values must be unique.")
    return result


def _config(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
    *,
    enforce_fold_only: bool = False,
) -> tuple[tuple[str, JsonScalar], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    result = []
    for key, item in items:
        key = _text(key, "configuration key")
        if not isinstance(item, (str, int, float, bool, type(None))) or (
            isinstance(item, float) and not math.isfinite(item)
        ):
            raise WholeChildComponentDiagnosticError(
                f"Configuration {key!r} must be a finite JSON scalar."
            )
        if enforce_fold_only:
            forbidden = _forbidden_words(f"{key} {item}")
            safe_zero_gradient_declaration = (
                key.casefold() == "zero_gradient" and item is True
            )
            if forbidden and not safe_zero_gradient_declaration:
                raise WholeChildComponentDiagnosticError(
                    "Diagnostic strategy configuration requests a forbidden "
                    "non-fold mechanism: " + ", ".join(forbidden) + "."
                )
        result.append((key, item))
    result.sort()
    if len(result) != len({key for key, _item in result}):
        raise WholeChildComponentDiagnosticError("Configuration keys repeat.")
    return tuple(result)


def _is_open_ledger_decision(value: Any) -> bool:
    return (
        getattr(value, "allowed", None) is True
        and getattr(value, "reason", None) == _OPEN_LEDGER_REASON
        and getattr(value, "blocking_record_sha256", None) is None
        and getattr(value, "closure_level", None) is None
    )


def _is_open_ledger_row(value: Mapping[str, Any]) -> bool:
    return (
        value["allowed"] is True
        and value["reason"] == _OPEN_LEDGER_REASON
        and value["blocking_record_sha256"] is None
        and value["closure_level"] is None
    )


def _trusted_method_contract(
    expected_method_id: str,
    expected_method_spec_sha256: str | None,
    expected_evidence_requirements: Sequence[str] | None,
) -> tuple[str, tuple[str, ...]]:
    method_id = _text(expected_method_id, "expected_method_id")
    if (expected_method_spec_sha256 is None) != (
        expected_evidence_requirements is None
    ):
        raise WholeChildComponentDiagnosticError(
            "Trusted method-spec identity and evidence requirements must be "
            "provided together."
        )
    if expected_method_spec_sha256 is not None:
        return (
            _digest(expected_method_spec_sha256, "expected_method_spec_sha256"),
            _names(
                tuple(expected_evidence_requirements or ()),
                "expected evidence requirement",
            ),
        )
    try:
        method = default_method_registry().require_selectable(
            method_id,
            require_zero_gradient=True,
        )
    except (ValueError, RuntimeError) as exc:
        raise WholeChildComponentDiagnosticError(
            "Production dispatch authentication has no trusted live method "
            f"contract for {method_id!r}."
        ) from exc
    return method.fingerprint, method.evidence_channels


def _tokens(role: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", role.casefold()))


def _projection_match(
    role: str,
    words: frozenset[str],
    fragments: tuple[str, ...],
) -> bool:
    lowered = role.casefold()
    return bool(_tokens(role).intersection(words)) or any(
        fragment in lowered for fragment in fragments
    )


def classify_diagnostic_parameter_semantic(
    circuit_kind: CircuitKind | str,
    parameter: ParameterSpec,
) -> str:
    """Classify a parameter from canonical IR kind/role, never module paths.

    Ambiguous fused attention tensors stay coupled; they are not falsely split
    into Q/K/V/O controls.
    """

    if not isinstance(parameter, ParameterSpec):
        raise WholeChildComponentDiagnosticError(
            "parameter must be an inspected ParameterSpec."
        )
    kind = (
        circuit_kind.value
        if isinstance(circuit_kind, CircuitKind)
        else _text(circuit_kind, "circuit_kind")
    )
    role = parameter.role
    lowered = role.casefold()
    if kind == CircuitKind.ATTENTION.value:
        if any(
            part in lowered
            for part in ("q_layernorm", "k_layernorm", "q_norm", "k_norm")
        ):
            return "attention_normalization"
        if any(
            part in lowered
            for part in ("qkv", "query_key_value", "c_attn", "in_proj_weight")
        ):
            return "attention_coupled_projection"
        candidates = []
        if _projection_match(
            role, frozenset({"q", "query", "wq"}), ("q_proj", "query_proj")
        ):
            candidates.append("attention_query")
        if _projection_match(
            role, frozenset({"k", "key", "wk"}), ("k_proj", "key_proj")
        ):
            candidates.append("attention_key")
        if _projection_match(
            role, frozenset({"v", "value", "wv"}), ("v_proj", "value_proj")
        ):
            candidates.append("attention_value")
        if _projection_match(
            role,
            frozenset({"o", "out", "output", "wo"}),
            ("o_proj", "out_proj", "output_proj", "c_proj"),
        ):
            candidates.append("attention_output")
        return candidates[0] if len(candidates) == 1 else "attention_coupled_projection"
    if kind in {CircuitKind.GATED_MLP.value, CircuitKind.DENSE_MLP.value}:
        writer = _projection_match(
            role,
            frozenset({"down", "out", "output", "writer", "w2", "wo", "fc2"}),
            ("down_proj", "out_proj", "output_proj"),
        )
        producer = _projection_match(
            role,
            frozenset({"gate", "up", "in", "input", "w1", "w3", "fc1"}),
            ("gate_proj", "up_proj", "in_proj", "input_proj"),
        )
        if writer != producer:
            return "mlp_writer" if writer else "mlp_producer"
        return "mlp_coupled_projection"
    by_kind = {
        CircuitKind.SHORT_CONV.value: "state_mixer",
        CircuitKind.RECURRENT_MIXER.value: "state_mixer",
        CircuitKind.NORM.value: "normalization",
        CircuitKind.TOKEN_EMBEDDING.value: "token_embedding",
        CircuitKind.OUTPUT_HEAD.value: "output_head",
        CircuitKind.POSITION_ENCODING.value: "position_encoding",
        CircuitKind.MODALITY_INPUT.value: "modality_input",
        CircuitKind.MODALITY_OUTPUT.value: "modality_output",
        CircuitKind.INTERFACE_PROJECTION.value: "interface_projection",
    }
    return by_kind.get(kind, f"other_{kind}")


def classify_diagnostic_measurement_role(
    circuit_kind: CircuitKind | str,
    parameter: ParameterSpec,
) -> str:
    """Return the finest unambiguous role available from canonical IR data.

    A measurement role is deliberately distinct from a physical compression
    bundle.  For example, gate/up/down can be measured independently even
    though reducing an MLP intermediate width changes all three tensors as one
    physical operation.
    """

    semantic = classify_diagnostic_parameter_semantic(circuit_kind, parameter)
    kind = (
        circuit_kind.value
        if isinstance(circuit_kind, CircuitKind)
        else _text(circuit_kind, "circuit_kind")
    )
    role = parameter.role
    if kind in {CircuitKind.GATED_MLP.value, CircuitKind.DENSE_MLP.value}:
        gate = _projection_match(
            role,
            frozenset({"gate", "w1"}),
            ("gate_proj", "gate_projection"),
        )
        up = _projection_match(
            role,
            frozenset({"up", "w3"}),
            ("up_proj", "up_projection"),
        )
        down = _projection_match(
            role,
            frozenset({"down", "out", "output", "writer", "w2", "wo", "fc2"}),
            ("down_proj", "out_proj", "output_proj"),
        )
        matched = [
            value
            for value, selected in (
                ("mlp_gate", gate),
                ("mlp_up", up),
                ("mlp_down", down),
            )
            if selected
        ]
        if len(matched) == 1:
            return matched[0]
        if matched:
            return "mlp_coupled_projection"
    if kind in {
        CircuitKind.SHORT_CONV.value,
        CircuitKind.RECURRENT_MIXER.value,
    }:
        input_projection = _projection_match(
            role,
            frozenset({"in", "input", "reader"}),
            ("in_proj", "input_proj", "proj_in"),
        )
        output_projection = _projection_match(
            role,
            frozenset({"out", "output", "writer"}),
            ("out_proj", "output_proj", "proj_out"),
        )
        kernel = _projection_match(
            role,
            frozenset({"kernel", "filter"}),
            (
                "conv.weight",
                "conv_weight",
                "kernel_weight",
                "filter_weight",
                ".kernel.",
                ".filter.",
            ),
        )
        matched = [
            value
            for value, selected in (
                ("conv_in_proj", input_projection),
                ("conv_kernel", kernel),
                ("conv_out_proj", output_projection),
            )
            if selected
        ]
        if len(matched) == 1:
            return matched[0]
        return "state_mixer_coupled"
    return semantic


def _physical_bundle_id(
    node: CircuitNode,
    parameter: ParameterSpec,
    measurement_role: str,
) -> str:
    """Name the geometry that must contract together for a real child."""

    if node.kind == CircuitKind.ATTENTION.value and measurement_role.startswith(
        "attention_"
    ):
        bundle_kind = "attention_head_geometry"
    elif node.kind in {
        CircuitKind.GATED_MLP.value,
        CircuitKind.DENSE_MLP.value,
    } and measurement_role.startswith("mlp_"):
        bundle_kind = "mlp_intermediate_width"
    elif node.kind in {
        CircuitKind.SHORT_CONV.value,
        CircuitKind.RECURRENT_MIXER.value,
    } and measurement_role in {
        "conv_in_proj",
        "conv_kernel",
        "conv_out_proj",
        "state_mixer_coupled",
    }:
        bundle_kind = "state_mixer_internal_width"
    else:
        # Unknown families and modality adapters are conservatively coupled at
        # the IR-node boundary.  Splitting an unfamiliar node tensor-by-tensor
        # would claim a physical contraction that the generic IR cannot prove.
        # Family inspectors may expose finer canonical circuit kinds later;
        # until then an entire node shares one physical bundle while its
        # parameter roles remain independently measurable.
        bundle_kind = "node_coupled_geometry"
    descriptor = {
        "bundle_kind": bundle_kind,
        "node_id": node.node_id,
        "layer_index": node.layer_index,
        "storage_group": None,
    }
    return f"bundle.{bundle_kind}.{canonical_sha256(descriptor)[:12]}"


def _structural_semantic(kind: str) -> str:
    if kind == CircuitKind.RESIDUAL_ADD.value:
        return "residual_routing"
    if kind == CircuitKind.POSITION_ENCODING.value:
        return "position_encoding"
    return f"structural_{kind}"


@dataclass(frozen=True, slots=True)
class DiagnosticFoldStrategy:
    """Registry- and implementation-bound executable fold strategy."""

    method_id: str
    method_version: int
    family_id: str
    method_spec_sha256: str
    operator_type: str
    operator_source_sha256: str
    operator_prepare_implementation_fingerprint: str
    evidence_requirements: tuple[str, ...] = ()
    operator_resource_fingerprint: str | None = None
    configuration: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        if (
            isinstance(self.method_version, bool)
            or not isinstance(self.method_version, int)
            or self.method_version <= 0
        ):
            raise WholeChildComponentDiagnosticError("method_version must be positive.")
        object.__setattr__(self, "family_id", _text(self.family_id, "family_id"))
        object.__setattr__(
            self,
            "method_spec_sha256",
            _digest(self.method_spec_sha256, "method_spec_sha256"),
        )
        object.__setattr__(
            self, "operator_type", _text(self.operator_type, "operator_type")
        )
        object.__setattr__(
            self,
            "operator_source_sha256",
            _digest(self.operator_source_sha256, "operator_source_sha256"),
        )
        object.__setattr__(
            self,
            "operator_prepare_implementation_fingerprint",
            _digest(
                self.operator_prepare_implementation_fingerprint,
                "operator_prepare_implementation_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "evidence_requirements",
            _names(self.evidence_requirements, "evidence requirement"),
        )
        if self.operator_resource_fingerprint is not None:
            object.__setattr__(
                self,
                "operator_resource_fingerprint",
                _digest(
                    self.operator_resource_fingerprint,
                    "operator_resource_fingerprint",
                ),
            )
        object.__setattr__(
            self,
            "configuration",
            _config(self.configuration, enforce_fold_only=True),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "family_id": self.family_id,
            "method_spec_sha256": self.method_spec_sha256,
            "operator_type": self.operator_type,
            "operator_source_sha256": self.operator_source_sha256,
            "operator_prepare_implementation_fingerprint": (
                self.operator_prepare_implementation_fingerprint
            ),
            "evidence_requirements": list(self.evidence_requirements),
            "operator_resource_fingerprint": self.operator_resource_fingerprint,
            "operator_prepare_static_audit_binding_required_at_compile": True,
            "operator_capability": WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY,
            "configuration": dict(self.configuration),
        }


def _operator_implementation_binding(
    strategy: DiagnosticFoldStrategy,
) -> tuple[str, int, str, str, str, str, str, tuple[str, ...], str | None]:
    """Return the implementation identity shared by diagnostic configurations."""

    return (
        strategy.method_id,
        strategy.method_version,
        strategy.family_id,
        strategy.method_spec_sha256,
        strategy.operator_type,
        strategy.operator_source_sha256,
        strategy.operator_prepare_implementation_fingerprint,
        strategy.evidence_requirements,
        strategy.operator_resource_fingerprint,
    )


def _registered_strategy(
    method_id: str,
    configuration: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
) -> DiagnosticFoldStrategy:
    if not isinstance(methods, FoldMethodRegistry) or not isinstance(
        operators, FoldOperatorRegistry
    ):
        raise WholeChildComponentDiagnosticError(
            "Executable diagnostics require method and operator registries."
        )
    try:
        method = methods.require_selectable(
            _text(method_id, "method_id"), require_zero_gradient=True
        )
        if not method.standalone:
            raise WholeChildComponentDiagnosticError(
                f"Method {method.method_id!r} is not a standalone child fold."
            )
        forbidden_identity = _forbidden_words(f"{method.method_id} {method.family_id}")
        if forbidden_identity:
            raise WholeChildComponentDiagnosticError(
                f"Method {method.method_id!r} has a forbidden non-fold identity: "
                + ", ".join(forbidden_identity)
                + "."
            )
        operator, _prepare, binding = operators.authenticated_prepare(method.method_id)
    except (ValueError, RuntimeError) as exc:
        raise WholeChildComponentDiagnosticError(str(exc)) from exc
    if (
        getattr(operator, "whole_child_component_diagnostic_capability", None)
        != WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY
    ):
        raise WholeChildComponentDiagnosticError(
            f"Operator {method.method_id!r} does not declare the executable "
            "whole-child component-dispatch capability."
        )
    row = next(
        (
            value
            for value in operators.implementation_manifest
            if value["method_id"] == method.method_id
        ),
        None,
    )
    if row is None or not binding.static_zero_training_audit_passed:
        raise WholeChildComponentDiagnosticError(
            f"Operator {method.method_id!r} has no authenticated clean prepare."
        )
    return DiagnosticFoldStrategy(
        method_id=method.method_id,
        method_version=method.version,
        family_id=method.family_id,
        method_spec_sha256=method.fingerprint,
        operator_type=str(row["implementation_type"]),
        operator_source_sha256=str(row["source_sha256"]),
        operator_prepare_implementation_fingerprint=str(
            row["prepare_implementation_fingerprint"]
        ),
        evidence_requirements=method.evidence_channels,
        operator_resource_fingerprint=row["resource_fingerprint"],
        configuration=_config(configuration, enforce_fold_only=True),
    )


@dataclass(frozen=True, slots=True)
class DiagnosticComponent:
    component_id: str
    semantics: tuple[str, ...]
    measurement_roles: tuple[str, ...]
    physical_bundle_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    parameter_paths: tuple[str, ...]
    storage_groups: tuple[str, ...]
    layer_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "semantics": list(self.semantics),
            "measurement_roles": list(self.measurement_roles),
            "physical_bundle_ids": list(self.physical_bundle_ids),
            "node_ids": list(self.node_ids),
            "parameter_paths": list(self.parameter_paths),
            "storage_groups": list(self.storage_groups),
            "layer_indices": list(self.layer_indices),
            "structural_only": not self.parameter_paths,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticScope:
    scope_id: str
    scope_kind: str
    component_ids: tuple[str, ...]
    semantics: tuple[str, ...]
    measurement_roles: tuple[str, ...]
    physical_bundle_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "scope_kind": self.scope_kind,
            "component_ids": list(self.component_ids),
            "component_set_sha256": canonical_sha256(self.component_ids),
            "semantics": list(self.semantics),
            "measurement_roles": list(self.measurement_roles),
            "physical_bundle_ids": list(self.physical_bundle_ids),
        }


@dataclass(frozen=True, slots=True)
class WholeChildDiagnosticVariant:
    variant_id: str
    focus_scope_id: str | None
    component_strategy_fingerprints: tuple[tuple[str, str], ...]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "focus_scope_id": self.focus_scope_id,
            "component_strategy_fingerprints": dict(
                self.component_strategy_fingerprints
            ),
            "complete_whole_child_fold_required": True,
            "focus_is_strategy_substitution_not_partial_fold": True,
        }


@dataclass(frozen=True, slots=True)
class WholeChildComponentDiagnosticPlan:
    parent_graph_fingerprints: tuple[str, ...]
    parent_unique_parameters: int
    target_graph_fingerprint: str
    target_topology_fingerprint: str
    base_fold_plan_fingerprint: str
    blank_child_state_fingerprint: str
    exact_target_unique_parameters: int
    target_parameter_paths_sha256: str
    target_storage_groups_sha256: str
    include_layer_scopes: bool
    baseline_strategy: DiagnosticFoldStrategy
    focus_strategy: DiagnosticFoldStrategy
    components: tuple[DiagnosticComponent, ...]
    scopes: tuple[DiagnosticScope, ...]
    variants: tuple[WholeChildDiagnosticVariant, ...]

    @property
    def compression_ratio(self) -> float:
        return self.parent_unique_parameters / self.exact_target_unique_parameters

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def variant(self, variant_id: str) -> WholeChildDiagnosticVariant:
        for value in self.variants:
            if value.variant_id == variant_id:
                return value
        raise WholeChildComponentDiagnosticError(
            f"Unknown diagnostic variant {variant_id!r}."
        )

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "parent_graph_fingerprints": list(self.parent_graph_fingerprints),
            "parent_unique_parameters": self.parent_unique_parameters,
            "target_graph_fingerprint": self.target_graph_fingerprint,
            "target_topology_fingerprint": self.target_topology_fingerprint,
            "base_fold_plan_fingerprint": self.base_fold_plan_fingerprint,
            "blank_child_state_fingerprint": self.blank_child_state_fingerprint,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "source_to_target_parameter_ratio": self.compression_ratio,
            "target_parameter_paths_sha256": self.target_parameter_paths_sha256,
            "target_storage_groups_sha256": self.target_storage_groups_sha256,
            "include_layer_scopes": self.include_layer_scopes,
            "baseline_strategy": self.baseline_strategy.as_dict(),
            "focus_strategy": self.focus_strategy.as_dict(),
            "components": [value.as_dict() for value in self.components],
            "scopes": [value.as_dict() for value in self.scopes],
            "variants": [value.as_dict() for value in self.variants],
            "complete_whole_model_target_required": True,
            "same_exact_target_for_every_variant": True,
            "component_focus_is_not_partial_compaction": True,
            "parent_shell_hybrid_forbidden": True,
            "fold_only_execution_claim": (
                "unproven_until_a_completed_executor_run_is_validated"
            ),
            "required_execution_attestation": {
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "autograd_grad_calls": 0,
                "pruning_operations": 0,
                "quantization_operations": 0,
                "observed_parent_module_forward_calls": 0,
                "observed_parent_root_forward_calls": 0,
            },
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def _diagnostic_plan_authentication_manifest(
    diagnostic: WholeChildComponentDiagnosticPlan,
) -> dict[str, Any]:
    """Return the linear-size plan projection required by an operator."""

    component_rows = [value.as_dict() for value in diagnostic.components]
    scope_rows = [value.as_dict() for value in diagnostic.scopes]
    return {
        "schema": PLAN_AUTHENTICATION_MANIFEST_SCHEMA,
        "parent_graph_fingerprints": list(diagnostic.parent_graph_fingerprints),
        "parent_unique_parameters": diagnostic.parent_unique_parameters,
        "target_graph_fingerprint": diagnostic.target_graph_fingerprint,
        "target_topology_fingerprint": diagnostic.target_topology_fingerprint,
        "base_fold_plan_fingerprint": diagnostic.base_fold_plan_fingerprint,
        "blank_child_state_fingerprint": diagnostic.blank_child_state_fingerprint,
        "exact_target_unique_parameters": diagnostic.exact_target_unique_parameters,
        "target_parameter_paths_sha256": diagnostic.target_parameter_paths_sha256,
        "target_storage_groups_sha256": diagnostic.target_storage_groups_sha256,
        "include_layer_scopes": diagnostic.include_layer_scopes,
        "baseline_strategy_sha256": diagnostic.baseline_strategy.fingerprint,
        "focus_strategy_sha256": diagnostic.focus_strategy.fingerprint,
        "components_sha256": canonical_sha256(component_rows),
        "scopes_sha256": canonical_sha256(scope_rows),
        "variant_formula_sha256": canonical_sha256(
            {
                "component_ids": [
                    value.component_id for value in diagnostic.components
                ],
                "scope_ids": [value.scope_id for value in diagnostic.scopes],
                "baseline_strategy_sha256": diagnostic.baseline_strategy.fingerprint,
                "focus_strategy_sha256": diagnostic.focus_strategy.fingerprint,
                "baseline_variant_id": "baseline.complete_whole_child",
                "focused_variant_prefix": "focus.",
            }
        ),
        "complete_whole_model_target_required": True,
        "component_focus_is_not_partial_compaction": True,
        "parent_shell_hybrid_forbidden": True,
    }


@dataclass(frozen=True, slots=True)
class WholeChildDiagnosticBinding:
    plan_sha256: str
    variant_id: str
    variant_sha256: str
    target_graph_fingerprint: str
    base_fold_plan_fingerprint: str
    blank_child_state_fingerprint: str
    exact_target_unique_parameters: int
    focus_scope_id: str | None
    assignments_sha256: str

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BINDING_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "variant_id": self.variant_id,
            "variant_sha256": self.variant_sha256,
            "target_graph_fingerprint": self.target_graph_fingerprint,
            "base_fold_plan_fingerprint": self.base_fold_plan_fingerprint,
            "blank_child_state_fingerprint": self.blank_child_state_fingerprint,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "focus_scope_id": self.focus_scope_id,
            "assignments_sha256": self.assignments_sha256,
            "complete_whole_child_fold_required": True,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedWholeChildComponentDispatch:
    """Executor-authenticated per-component strategy assignment.

    This is deliberately narrower than the compiler-side diagnostic plan.  A
    production operator receives only a :class:`FoldAction`, so it replays the
    architecture partition against the live inspected child graph and accepts
    the embedded dispatch only when every component, target path, storage
    group, strategy implementation, and exact-budget binding is intact.
    """

    binding: WholeChildDiagnosticBinding
    dispatch_sha256: str
    selected_strategy_fingerprint: str
    focus_strategy_fingerprint: str
    strategies: tuple[tuple[str, DiagnosticFoldStrategy], ...]
    components: tuple[DiagnosticComponent, ...]
    component_strategy_fingerprints: tuple[tuple[str, str], ...]
    parameter_strategy_fingerprints: tuple[tuple[str, str], ...]

    @property
    def strategy_by_fingerprint(self) -> Mapping[str, DiagnosticFoldStrategy]:
        return dict(self.strategies)

    @property
    def parameter_strategy_by_path(self) -> Mapping[str, str]:
        return dict(self.parameter_strategy_fingerprints)

    @property
    def focused_parameter_paths(self) -> tuple[str, ...]:
        if self.binding.focus_scope_id is None:
            return ()
        return tuple(
            path
            for path, fingerprint in self.parameter_strategy_fingerprints
            if fingerprint == self.focus_strategy_fingerprint
        )

    def strategy_for_parameter(self, path: str) -> DiagnosticFoldStrategy:
        try:
            fingerprint = self.parameter_strategy_by_path[path]
            return self.strategy_by_fingerprint[fingerprint]
        except KeyError as exc:
            raise WholeChildComponentDiagnosticError(
                f"Diagnostic dispatch does not own target parameter {path!r}."
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-authenticated-whole-child-component-dispatch-v2",
            "binding": self.binding.as_dict(),
            "dispatch_sha256": self.dispatch_sha256,
            "selected_strategy_fingerprint": self.selected_strategy_fingerprint,
            "focus_strategy_fingerprint": self.focus_strategy_fingerprint,
            "strategies": {
                fingerprint: strategy.as_dict()
                for fingerprint, strategy in self.strategies
            },
            "component_strategy_fingerprints": dict(
                self.component_strategy_fingerprints
            ),
            "parameter_strategy_fingerprints": dict(
                self.parameter_strategy_fingerprints
            ),
            "focused_parameter_paths": list(self.focused_parameter_paths),
            "complete_whole_child_fold_required": True,
        }


@dataclass(slots=True)
class _Fragment:
    semantics: set[str]
    measurement_roles: set[str]
    physical_bundles: set[str]
    nodes: set[str]
    paths: set[str]
    groups: set[str]
    layers: set[int]


def _discover_components(graph: ModelCircuitGraph) -> tuple[DiagnosticComponent, ...]:
    fragments: list[_Fragment] = []
    for node in graph.nodes:
        if not node.parameters:
            fragments.append(
                _Fragment(
                    {_structural_semantic(node.kind)},
                    {_structural_semantic(node.kind)},
                    set(),
                    {node.node_id},
                    set(),
                    set(),
                    set() if node.layer_index is None else {node.layer_index},
                )
            )
            continue
        by_role: dict[tuple[str, str, str], list[ParameterSpec]] = {}
        for parameter in node.parameters:
            semantic = classify_diagnostic_parameter_semantic(node.kind, parameter)
            measurement_role = classify_diagnostic_measurement_role(
                node.kind, parameter
            )
            physical_bundle = _physical_bundle_id(node, parameter, measurement_role)
            by_role.setdefault(
                (semantic, measurement_role, physical_bundle), []
            ).append(parameter)
        for (
            semantic,
            measurement_role,
            physical_bundle,
        ), parameters in by_role.items():
            fragments.append(
                _Fragment(
                    {semantic},
                    {measurement_role},
                    {physical_bundle},
                    {node.node_id},
                    {value.path for value in parameters},
                    {value.storage_group for value in parameters},
                    set() if node.layer_index is None else {node.layer_index},
                )
            )

    # Storage aliases (notably tied embedding/head weights) are indivisible.
    parents = list(range(len(fragments)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    owner: dict[str, int] = {}
    for index, fragment in enumerate(fragments):
        for group in fragment.groups:
            union(owner.setdefault(group, index), index)
    merged: dict[int, _Fragment] = {}
    for index, fragment in enumerate(fragments):
        current = merged.setdefault(
            find(index), _Fragment(set(), set(), set(), set(), set(), set(), set())
        )
        current.semantics.update(fragment.semantics)
        current.measurement_roles.update(fragment.measurement_roles)
        current.physical_bundles.update(fragment.physical_bundles)
        current.nodes.update(fragment.nodes)
        current.paths.update(fragment.paths)
        current.groups.update(fragment.groups)
        current.layers.update(fragment.layers)

    components = []
    for value in merged.values():
        descriptor = {
            "semantics": sorted(value.semantics),
            "measurement_roles": sorted(value.measurement_roles),
            "physical_bundles": sorted(value.physical_bundles),
            "nodes": sorted(value.nodes),
            "paths": sorted(value.paths),
            "groups": sorted(value.groups),
        }
        label = "_and_".join(sorted(value.semantics))
        components.append(
            DiagnosticComponent(
                component_id=f"component.{label}.{canonical_sha256(descriptor)[:12]}",
                semantics=tuple(sorted(value.semantics)),
                measurement_roles=tuple(sorted(value.measurement_roles)),
                physical_bundle_ids=tuple(sorted(value.physical_bundles)),
                node_ids=tuple(sorted(value.nodes)),
                parameter_paths=tuple(sorted(value.paths)),
                storage_groups=tuple(sorted(value.groups)),
                layer_indices=tuple(sorted(value.layers)),
            )
        )
    return tuple(sorted(components, key=lambda value: value.component_id))


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WholeChildComponentDiagnosticError(
                f"Diagnostic dispatch JSON repeats key {key!r}."
            )
        result[key] = value
    return result


def _exact_object_keys(
    value: Any,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise WholeChildComponentDiagnosticError(
            f"{label} has missing or unexpected fields."
        )
    return value


def _binding_from_dispatch(value: Any) -> WholeChildDiagnosticBinding:
    row = _exact_object_keys(
        value,
        {
            "schema",
            "plan_sha256",
            "variant_id",
            "variant_sha256",
            "target_graph_fingerprint",
            "base_fold_plan_fingerprint",
            "blank_child_state_fingerprint",
            "exact_target_unique_parameters",
            "focus_scope_id",
            "assignments_sha256",
            "complete_whole_child_fold_required",
        },
        "Diagnostic dispatch binding",
    )
    if (
        row["schema"] != BINDING_SCHEMA
        or row["complete_whole_child_fold_required"] is not True
        or (
            row["focus_scope_id"] is not None
            and not isinstance(row["focus_scope_id"], str)
        )
        or isinstance(row["exact_target_unique_parameters"], bool)
        or not isinstance(row["exact_target_unique_parameters"], int)
        or row["exact_target_unique_parameters"] <= 0
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch binding is not a complete exact-budget child."
        )
    binding = WholeChildDiagnosticBinding(
        plan_sha256=_digest(row["plan_sha256"], "plan_sha256"),
        variant_id=_text(row["variant_id"], "variant_id"),
        variant_sha256=_digest(row["variant_sha256"], "variant_sha256"),
        target_graph_fingerprint=_digest(
            row["target_graph_fingerprint"], "target_graph_fingerprint"
        ),
        base_fold_plan_fingerprint=_digest(
            row["base_fold_plan_fingerprint"], "base_fold_plan_fingerprint"
        ),
        blank_child_state_fingerprint=_digest(
            row["blank_child_state_fingerprint"], "blank_child_state_fingerprint"
        ),
        exact_target_unique_parameters=row["exact_target_unique_parameters"],
        focus_scope_id=row["focus_scope_id"],
        assignments_sha256=_digest(row["assignments_sha256"], "assignments_sha256"),
    )
    if binding.as_dict() != dict(row):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch binding is not canonical."
        )
    return binding


def _strategy_from_dispatch(value: Any) -> DiagnosticFoldStrategy:
    row = _exact_object_keys(
        value,
        {
            "method_id",
            "method_version",
            "family_id",
            "method_spec_sha256",
            "operator_type",
            "operator_source_sha256",
            "operator_prepare_implementation_fingerprint",
            "evidence_requirements",
            "operator_resource_fingerprint",
            "operator_prepare_static_audit_binding_required_at_compile",
            "operator_capability",
            "configuration",
        },
        "Diagnostic dispatch strategy",
    )
    if (
        row["operator_prepare_static_audit_binding_required_at_compile"] is not True
        or row["operator_capability"] != WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY
        or not isinstance(row["configuration"], Mapping)
        or not isinstance(row["evidence_requirements"], list)
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch strategy lacks its executable capability binding."
        )
    strategy = DiagnosticFoldStrategy(
        method_id=row["method_id"],
        method_version=row["method_version"],
        family_id=row["family_id"],
        method_spec_sha256=row["method_spec_sha256"],
        operator_type=row["operator_type"],
        operator_source_sha256=row["operator_source_sha256"],
        operator_prepare_implementation_fingerprint=(
            row["operator_prepare_implementation_fingerprint"]
        ),
        evidence_requirements=tuple(row["evidence_requirements"]),
        operator_resource_fingerprint=row["operator_resource_fingerprint"],
        configuration=tuple(row["configuration"].items()),
    )
    if strategy.as_dict() != dict(row):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch strategy is not canonical."
        )
    return strategy


def _fingerprinted_dispatch_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WholeChildComponentDiagnosticError(f"{label} must be an object.")
    row = dict(value)
    fingerprint = _digest(row.pop("fingerprint", None), f"{label} fingerprint")
    if canonical_sha256(row) != fingerprint:
        raise WholeChildComponentDiagnosticError(
            f"{label} fingerprint differs from its canonical contents."
        )
    return dict(value)


def _validate_dispatch_provenance(
    row: Mapping[str, Any],
    binding: WholeChildDiagnosticBinding,
    action: FoldAction,
    child_graph: ModelCircuitGraph,
    strategies: Mapping[str, DiagnosticFoldStrategy],
    components: tuple[DiagnosticComponent, ...],
    selected_strategy_fingerprint: str,
) -> tuple[dict[str, str], str]:
    """Recompute every compiler-owned identity available at operator time."""

    plan = _exact_object_keys(
        row["plan_authentication_manifest"],
        {
            "schema",
            "parent_graph_fingerprints",
            "parent_unique_parameters",
            "target_graph_fingerprint",
            "target_topology_fingerprint",
            "base_fold_plan_fingerprint",
            "blank_child_state_fingerprint",
            "exact_target_unique_parameters",
            "target_parameter_paths_sha256",
            "target_storage_groups_sha256",
            "include_layer_scopes",
            "baseline_strategy_sha256",
            "focus_strategy_sha256",
            "components_sha256",
            "scopes_sha256",
            "variant_formula_sha256",
            "complete_whole_model_target_required",
            "component_focus_is_not_partial_compaction",
            "parent_shell_hybrid_forbidden",
        },
        "Diagnostic plan authentication manifest",
    )
    parent_unique_parameters = plan["parent_unique_parameters"]
    target_groups = sorted(
        {
            parameter.storage_group
            for node in child_graph.nodes
            for parameter in node.parameters
        }
    )
    if (
        plan["schema"] != PLAN_AUTHENTICATION_MANIFEST_SCHEMA
        or canonical_sha256(plan) != binding.plan_sha256
        or plan["parent_graph_fingerprints"]
        != list(action.signature.parent_graph_fingerprints)
        or isinstance(parent_unique_parameters, bool)
        or not isinstance(parent_unique_parameters, int)
        or parent_unique_parameters <= child_graph.unique_parameter_count
        or plan["target_graph_fingerprint"] != child_graph.fingerprint
        or plan["target_topology_fingerprint"]
        != child_graph.structural_topology_fingerprint
        or plan["base_fold_plan_fingerprint"] != binding.base_fold_plan_fingerprint
        or plan["blank_child_state_fingerprint"]
        != action.signature.child_state_fingerprint
        or plan["exact_target_unique_parameters"] != child_graph.unique_parameter_count
        or plan["target_parameter_paths_sha256"]
        != canonical_sha256(sorted(child_graph.parameter_paths))
        or plan["target_storage_groups_sha256"] != canonical_sha256(target_groups)
        or plan["complete_whole_model_target_required"] is not True
        or plan["component_focus_is_not_partial_compaction"] is not True
        or plan["parent_shell_hybrid_forbidden"] is not True
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic plan provenance differs from the live action or child."
        )

    baseline = strategies.get(plan["baseline_strategy_sha256"])
    focus = strategies.get(plan["focus_strategy_sha256"])
    if (
        baseline is None
        or focus is None
        or baseline.fingerprint == focus.fingerprint
        or {baseline.fingerprint, focus.fingerprint} != set(strategies)
        or plan["components_sha256"]
        != canonical_sha256([value.as_dict() for value in components])
        or not isinstance(plan["include_layer_scopes"], bool)
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic plan strategies or components differ from live discovery."
        )
    scopes = _discover_scopes(components, plan["include_layer_scopes"])
    variant_formula_sha256 = canonical_sha256(
        {
            "component_ids": [value.component_id for value in components],
            "scope_ids": [value.scope_id for value in scopes],
            "baseline_strategy_sha256": baseline.fingerprint,
            "focus_strategy_sha256": focus.fingerprint,
            "baseline_variant_id": "baseline.complete_whole_child",
            "focused_variant_prefix": "focus.",
        }
    )
    if (
        plan["scopes_sha256"] != canonical_sha256([value.as_dict() for value in scopes])
        or plan["variant_formula_sha256"] != variant_formula_sha256
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic plan scopes or variants differ from live discovery."
        )
    expected_variant = _expected_variant_by_id(
        components,
        scopes,
        baseline,
        focus,
        binding.variant_id,
    )
    assignments = dict(expected_variant.component_strategy_fingerprints)
    if (
        expected_variant.fingerprint != binding.variant_sha256
        or expected_variant.focus_scope_id != binding.focus_scope_id
        or canonical_sha256(assignments) != binding.assignments_sha256
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic variant or focus scope was not derived from the live plan."
        )

    focus_component_ids = {
        component_id
        for component_id, fingerprint in assignments.items()
        if fingerprint == focus.fingerprint
    }
    focus_paths = {
        path
        for component in components
        if component.component_id in focus_component_ids
        for path in component.parameter_paths
    }
    focus_groups = {
        group
        for component in components
        if component.component_id in focus_component_ids
        for group in component.storage_groups
    }
    expected_selected = (
        focus.fingerprint
        if focus_paths.intersection(action.target_parameter_paths)
        or focus_groups.intersection(action.target_storage_groups)
        else baseline.fingerprint
    )
    if selected_strategy_fingerprint != expected_selected:
        raise WholeChildComponentDiagnosticError(
            "Action-selected strategy differs from its authenticated focus scope."
        )

    base_plan = _exact_object_keys(
        _fingerprinted_dispatch_object(row["base_fold_plan"], "Base fold plan"),
        {
            "schema",
            "parent_graph_fingerprints",
            "parent_state_fingerprints",
            "child_graph_fingerprint",
            "child_state_fingerprint",
            "evidence_fingerprint",
            "exact_target_unique_parameters",
            "actions",
            "gaps",
            "registry_fingerprint",
            "ledger_fingerprint",
            "complete_target_parameter_coverage",
            "known_closed_methods_filtered",
            "ready",
            "expected_logical_compression_delta",
            "fingerprint",
        },
        "Base fold plan",
    )
    base_actions = base_plan["actions"]
    if (
        base_plan["schema"] != DYNAMIC_FOLD_PLAN_SCHEMA
        or base_plan["fingerprint"] != binding.base_fold_plan_fingerprint
        or base_plan["parent_graph_fingerprints"]
        != list(action.signature.parent_graph_fingerprints)
        or base_plan["parent_state_fingerprints"]
        != dict(action.signature.parent_state_fingerprints)
        or base_plan["child_graph_fingerprint"] != child_graph.fingerprint
        or base_plan["child_state_fingerprint"]
        != action.signature.child_state_fingerprint
        or base_plan["evidence_fingerprint"] != action.signature.evidence_fingerprint
        or base_plan["exact_target_unique_parameters"]
        != child_graph.unique_parameter_count
        or base_plan["gaps"] != []
        or base_plan["complete_target_parameter_coverage"] is not True
        or base_plan["known_closed_methods_filtered"] is not True
        or base_plan["ready"] is not True
        or not isinstance(base_actions, list)
        or not base_actions
    ):
        raise WholeChildComponentDiagnosticError(
            "Base fold plan provenance differs from the compiled action."
        )
    _digest(base_plan["registry_fingerprint"], "base registry_fingerprint")
    _digest(base_plan["ledger_fingerprint"], "base ledger_fingerprint")
    base_action_ids: set[str] = set()
    previous_action_ids: set[str] = set()
    compiled_action_ids: dict[str, str] = {}
    base_signatures: dict[str, ExperimentSignature] = {}
    for expected_index, base_action_row in enumerate(base_actions):
        base_action_row = _exact_object_keys(
            base_action_row,
            {
                "action_id",
                "sequence_index",
                "method_id",
                "sources",
                "target_node_ids",
                "target_parameter_paths",
                "target_storage_groups",
                "dependencies",
                "evidence_requirements",
                "source_logical_parameters",
                "target_logical_parameters",
                "compression_delta",
                "configuration",
                "signature",
                "ledger_decision",
            },
            "Embedded base action",
        )
        action_id = _text(base_action_row["action_id"], "base action_id")
        dependencies = base_action_row["dependencies"]
        sources = base_action_row["sources"]
        configuration = base_action_row["configuration"]
        target_node_ids = base_action_row["target_node_ids"]
        target_parameter_paths = base_action_row["target_parameter_paths"]
        target_storage_groups = base_action_row["target_storage_groups"]
        evidence_requirements = base_action_row["evidence_requirements"]
        ledger_decision = _exact_object_keys(
            base_action_row["ledger_decision"],
            {"allowed", "reason", "blocking_record_sha256", "closure_level"},
            "Embedded ledger decision",
        )
        if not isinstance(configuration, Mapping):
            raise WholeChildComponentDiagnosticError(
                "Embedded base action configuration is invalid."
            )
        normalized_configuration = _config(
            configuration,
            enforce_fold_only=True,
        )
        if not isinstance(sources, list) or not sources:
            raise WholeChildComponentDiagnosticError(
                "Embedded base action sources are invalid."
            )
        normalized_sources = []
        for source in sources:
            source = _exact_object_keys(
                source,
                {"parent_name", "node_ids"},
                "Embedded base action source",
            )
            parent_name = _text(source["parent_name"], "source parent_name")
            node_ids = source["node_ids"]
            if (
                parent_name not in base_plan["parent_state_fingerprints"]
                or not isinstance(node_ids, list)
                or not node_ids
                or node_ids
                != sorted({_text(value, "source node_id") for value in node_ids})
            ):
                raise WholeChildComponentDiagnosticError(
                    "Embedded base action source is not canonical."
                )
            normalized_sources.append(
                {"parent_name": parent_name, "node_ids": node_ids}
            )
        list_contracts = (
            ("target_node_ids", target_node_ids, False),
            ("target_parameter_paths", target_parameter_paths, True),
            ("target_storage_groups", target_storage_groups, True),
            ("dependencies", dependencies, True),
            ("evidence_requirements", evidence_requirements, True),
        )
        for label, values, may_be_empty in list_contracts:
            if (
                not isinstance(values, list)
                or (not may_be_empty and not values)
                or values != sorted({_text(value, label) for value in values})
            ):
                raise WholeChildComponentDiagnosticError(
                    f"Embedded base action {label} is not canonical."
                )
        expected_paths, expected_groups = _expected_target_ownership(
            child_graph,
            target_node_ids,
        )
        try:
            signature = ExperimentSignature.from_dict(base_action_row["signature"])
        except (TypeError, ValueError) as exc:
            raise WholeChildComponentDiagnosticError(
                "Embedded base action signature is not canonical."
            ) from exc
        source_unit_ids = sorted(
            f"{source['parent_name']}:{node_id}"
            for source in normalized_sources
            for node_id in source["node_ids"]
        )
        numeric_counts = (
            base_action_row["source_logical_parameters"],
            base_action_row["target_logical_parameters"],
            base_action_row["compression_delta"],
        )
        if (
            action_id in base_action_ids
            or base_action_row["sequence_index"] != expected_index
            or any(value not in previous_action_ids for value in dependencies)
            or base_action_row["method_id"] != signature.method_id
            or tuple(target_parameter_paths) != expected_paths
            or tuple(target_storage_groups) != expected_groups
            or signature.parent_graph_fingerprints
            != tuple(base_plan["parent_graph_fingerprints"])
            or dict(signature.parent_state_fingerprints)
            != base_plan["parent_state_fingerprints"]
            or signature.child_graph_fingerprint != base_plan["child_graph_fingerprint"]
            or signature.child_state_fingerprint != base_plan["child_state_fingerprint"]
            or signature.evidence_fingerprint != base_plan["evidence_fingerprint"]
            or signature.source_unit_ids != tuple(source_unit_ids)
            or signature.target_unit_ids != tuple(sorted(target_node_ids))
            or signature.configuration != normalized_configuration
            or signature.zero_gradient is not True
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in numeric_counts
            )
            or base_action_row["compression_delta"]
            != base_action_row["source_logical_parameters"]
            - base_action_row["target_logical_parameters"]
            or not _is_open_ledger_row(ledger_decision)
        ):
            raise WholeChildComponentDiagnosticError(
                "Embedded base action dependency, signature, or ledger contract is "
                "invalid."
            )
        base_action_ids.add(action_id)
        previous_action_ids.add(action_id)
        base_signatures[action_id] = signature
        row_strategy = (
            focus
            if focus_paths.intersection(target_parameter_paths)
            or focus_groups.intersection(target_storage_groups)
            else baseline
        )
        compiled_action_ids[action_id] = (
            "whole-child-diagnostic."
            + canonical_sha256(
                {
                    "binding": binding.fingerprint,
                    "base_action": action_id,
                    "method_id": row_strategy.method_id,
                }
            )[:20]
        )
    if base_plan["expected_logical_compression_delta"] != sum(
        value["compression_delta"] for value in base_actions
    ):
        raise WholeChildComponentDiagnosticError(
            "Base fold plan compression accounting is invalid."
        )
    if (
        Counter(
            node_id
            for base_action in base_actions
            for node_id in base_action["target_node_ids"]
        )
        != Counter(value.node_id for value in child_graph.nodes)
        or Counter(
            path
            for base_action in base_actions
            for path in base_action["target_parameter_paths"]
        )
        != Counter(child_graph.parameter_paths)
        or Counter(
            group
            for base_action in base_actions
            for group in base_action["target_storage_groups"]
        )
        != Counter({value: 1 for value in target_groups})
    ):
        raise WholeChildComponentDiagnosticError(
            "Embedded base fold plan does not own the complete live child."
        )
    base_action_id = dict(action.configuration).get(
        DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY
    )
    matching_base_actions = [
        value for value in base_actions if value.get("action_id") == base_action_id
    ]
    if len(matching_base_actions) != 1:
        raise WholeChildComponentDiagnosticError(
            "Compiled action does not map to one embedded base action."
        )
    base_action = matching_base_actions[0]
    base_signature = base_signatures[base_action_id]
    compiled_base_configuration = {
        key: value
        for key, value in action.configuration
        if key
        not in {
            DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY,
            DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY,
            "whole_child_component_diagnostic_selected_strategy_sha256",
        }
    }
    if (
        base_action.get("sources") != [value.as_dict() for value in action.sources]
        or action.sequence_index != base_action.get("sequence_index")
        or action.dependencies
        != tuple(
            sorted(
                compiled_action_ids[value]
                for value in base_action.get("dependencies", [])
            )
        )
        or base_action.get("target_node_ids") != list(action.target_node_ids)
        or base_action.get("target_parameter_paths")
        != list(action.target_parameter_paths)
        or base_action.get("target_storage_groups")
        != list(action.target_storage_groups)
        or base_action.get("source_logical_parameters")
        != action.source_logical_parameters
        or base_action.get("target_logical_parameters")
        != action.target_logical_parameters
        or base_action.get("configuration") != compiled_base_configuration
        or action.signature.source_unit_ids != base_signature.source_unit_ids
        or action.signature.target_unit_ids != base_signature.target_unit_ids
        or action.signature.topology_transition != base_signature.topology_transition
        or action.signature.promotion_gate != base_signature.promotion_gate
        or action.signature.selection_policy
        != "architecture_driven_complete_whole_child_component_dispatch"
        or action.signature.fit_objective != "complete_whole_child_component_fold"
    ):
        raise WholeChildComponentDiagnosticError(
            "Embedded base action differs from the live compiled action."
        )
    return assignments, focus.fingerprint


def authenticate_whole_child_component_dispatch(
    action: FoldAction,
    child_graph: ModelCircuitGraph,
    *,
    expected_method_id: str,
    expected_method_spec_sha256: str | None = None,
    expected_evidence_requirements: Sequence[str] | None = None,
) -> AuthenticatedWholeChildComponentDispatch | None:
    """Authenticate an optional compiler dispatch at a production operator.

    Ordinary actions return ``None`` and retain their existing execution path.
    A diagnostic action is accepted only when its authenticated base plan owns
    the complete live graph exactly once and both strategy configurations are
    implementations of the same registered production operator.  Individual
    actions may own disjoint slices of that complete plan.  Other
    families/operators fail closed instead of receiving metadata-only claims.
    """

    if not isinstance(action, FoldAction) or not isinstance(
        child_graph, ModelCircuitGraph
    ):
        raise WholeChildComponentDiagnosticError(
            "Production dispatch authentication requires a FoldAction and graph."
        )
    method_id = _text(expected_method_id, "expected_method_id")
    configuration = dict(action.configuration)
    raw = configuration.get(DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY)
    if raw is None:
        return None
    trusted_method_spec_sha256, trusted_evidence_requirements = (
        _trusted_method_contract(
            method_id,
            expected_method_spec_sha256,
            expected_evidence_requirements,
        )
    )
    if not isinstance(raw, str) or not raw:
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch configuration must be canonical JSON text."
        )
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, TypeError, WholeChildComponentDiagnosticError) as exc:
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch JSON is invalid or ambiguous."
        ) from exc
    if canonical_json_bytes(payload).decode("ascii") != raw:
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch JSON is not in compiler-canonical form."
        )
    row = _exact_object_keys(
        payload,
        {
            "schema",
            "binding",
            "plan_authentication_manifest",
            "base_fold_plan",
            "strategies",
            "complete_whole_child_fold_required",
            "exact_target_unique_parameters",
        },
        "Diagnostic dispatch",
    )
    if (
        row["schema"] != DISPATCH_SCHEMA
        or row["complete_whole_child_fold_required"] is not True
        or row["exact_target_unique_parameters"] != child_graph.unique_parameter_count
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch does not bind the exact complete child budget."
        )
    binding = _binding_from_dispatch(row["binding"])
    if (
        binding.target_graph_fingerprint != child_graph.fingerprint
        or binding.exact_target_unique_parameters != child_graph.unique_parameter_count
        or binding.blank_child_state_fingerprint
        != action.signature.child_state_fingerprint
        or action.signature.child_graph_fingerprint != child_graph.fingerprint
        or action.signature.zero_gradient is not True
        or action.signature.configuration != action.configuration
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch binding differs from the live inspected child or "
            "compiled fold signature."
        )

    selected = _digest(
        configuration.get("whole_child_component_diagnostic_selected_strategy_sha256"),
        "selected diagnostic strategy",
    )
    strategy_rows = row["strategies"]
    if not isinstance(strategy_rows, Mapping) or len(strategy_rows) != 2:
        raise WholeChildComponentDiagnosticError(
            "Diagnostic dispatch must contain exactly two distinct strategies."
        )
    strategies = []
    implementation_bindings = set()
    for fingerprint, value in strategy_rows.items():
        fingerprint = _digest(fingerprint, "strategy fingerprint")
        strategy = _strategy_from_dispatch(value)
        if strategy.fingerprint != fingerprint:
            raise WholeChildComponentDiagnosticError(
                "Diagnostic strategy fingerprint differs from its contents."
            )
        if (
            strategy.method_id != method_id
            or strategy.method_spec_sha256 != trusted_method_spec_sha256
            or strategy.evidence_requirements != trusted_evidence_requirements
        ):
            raise WholeChildComponentDiagnosticError(
                "Production component dispatch differs from its trusted live "
                "method contract."
            )
        implementation_bindings.add(_operator_implementation_binding(strategy))
        strategies.append((fingerprint, strategy))
    if selected not in dict(strategies) or len(implementation_bindings) != 1:
        raise WholeChildComponentDiagnosticError(
            "Diagnostic strategies are not two configurations of one operator."
        )
    if action.method_id != method_id:
        raise WholeChildComponentDiagnosticError(
            "Diagnostic action method differs from its production operator."
        )
    selected_strategy = dict(strategies)[selected]
    base_action_id = configuration.get(DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY)
    expected_action_id = (
        "whole-child-diagnostic."
        + canonical_sha256(
            {
                "binding": binding.fingerprint,
                "base_action": base_action_id,
                "method_id": selected_strategy.method_id,
            }
        )[:20]
    )
    if (
        not isinstance(base_action_id, str)
        or action.action_id != expected_action_id
        or action.signature.method_id != selected_strategy.method_id
        or action.signature.method_version != selected_strategy.method_version
        or action.signature.family_id != selected_strategy.family_id
        or action.evidence_requirements != selected_strategy.evidence_requirements
        or not _is_open_ledger_decision(action.ledger_decision)
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic action identity differs from its selected strategy binding."
        )

    components = _discover_components(child_graph)
    normalized_assignments, focus_strategy_fingerprint = _validate_dispatch_provenance(
        row,
        binding,
        action,
        child_graph,
        dict(strategies),
        components,
        selected,
    )

    parameter_strategies = []
    for component in components:
        assigned = normalized_assignments[component.component_id]
        parameter_strategies.extend(
            (path, assigned) for path in component.parameter_paths
        )
    parameter_strategies.sort()
    if Counter(path for path, _fingerprint in parameter_strategies) != Counter(
        child_graph.parameter_paths
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostic parameter strategies do not partition the whole child."
        )
    selected_paths = tuple(
        path for path, fingerprint in parameter_strategies if fingerprint == selected
    )
    if binding.focus_scope_id is None:
        if len(selected_paths) != len(parameter_strategies):
            raise WholeChildComponentDiagnosticError(
                "Baseline dispatch does not assign one strategy to the whole child."
            )
    elif not selected_paths or len(selected_paths) == len(parameter_strategies):
        raise WholeChildComponentDiagnosticError(
            "Focused dispatch must substitute a nonempty proper child component."
        )

    return AuthenticatedWholeChildComponentDispatch(
        binding=binding,
        dispatch_sha256=canonical_sha256(payload),
        selected_strategy_fingerprint=selected,
        focus_strategy_fingerprint=focus_strategy_fingerprint,
        strategies=tuple(sorted(strategies)),
        components=components,
        component_strategy_fingerprints=tuple(sorted(normalized_assignments.items())),
        parameter_strategy_fingerprints=tuple(parameter_strategies),
    )


_SUBSYSTEMS = {
    "attention": frozenset(
        {
            "attention_query",
            "attention_key",
            "attention_value",
            "attention_output",
            "attention_normalization",
            "attention_coupled_projection",
        }
    ),
    "state_mixer": frozenset({"state_mixer"}),
    "mlp": frozenset({"mlp_producer", "mlp_writer", "mlp_coupled_projection"}),
    "interfaces": frozenset(
        {
            "token_embedding",
            "output_head",
            "modality_input",
            "modality_output",
            "interface_projection",
        }
    ),
    "residual_and_norm": frozenset(
        {"normalization", "residual_routing", "position_encoding"}
    ),
}


def _scope(
    kind: str, label: str, values: Sequence[DiagnosticComponent]
) -> DiagnosticScope:
    component_ids = tuple(sorted(value.component_id for value in values))
    return DiagnosticScope(
        scope_id=f"scope.{kind}.{label}.{canonical_sha256(component_ids)[:12]}",
        scope_kind=kind,
        component_ids=component_ids,
        semantics=tuple(
            sorted({semantic for value in values for semantic in value.semantics})
        ),
        measurement_roles=tuple(
            sorted({role for value in values for role in value.measurement_roles})
        ),
        physical_bundle_ids=tuple(
            sorted({bundle for value in values for bundle in value.physical_bundle_ids})
        ),
    )


def _discover_scopes(
    components: Sequence[DiagnosticComponent],
    include_layer_scopes: bool,
) -> tuple[DiagnosticScope, ...]:
    # Parameterless residual/route nodes stay in the complete graph inventory,
    # but cannot honestly receive a distinct learned-state strategy.  An
    # edge-aware residual method may expose parameterized interface components;
    # until then, no parameterless "residual focus" variant is emitted.
    components = tuple(value for value in components if value.parameter_paths)
    proposed = []
    for component in components:
        proposed.append(_scope("component", component.component_id, [component]))
    for semantics in sorted({value.semantics for value in components}):
        proposed.append(
            _scope(
                "semantic",
                "_and_".join(semantics),
                [value for value in components if value.semantics == semantics],
            )
        )
    for label, semantics in _SUBSYSTEMS.items():
        selected = [value for value in components if set(value.semantics) & semantics]
        if selected:
            proposed.append(_scope("subsystem", label, selected))
    if include_layer_scopes:
        for layer in sorted(
            {layer for value in components for layer in value.layer_indices}
        ):
            selected = [value for value in components if layer in value.layer_indices]
            proposed.append(_scope("layer", str(layer), selected))
            for semantics in sorted({value.semantics for value in selected}):
                subset = [value for value in selected if value.semantics == semantics]
                proposed.append(
                    _scope(
                        "layer_semantic",
                        f"{layer}_{'_and_'.join(semantics)}",
                        subset,
                    )
                )
            for measurement_role in sorted(
                {role for value in selected for role in value.measurement_roles}
            ):
                subset = [
                    value
                    for value in selected
                    if measurement_role in value.measurement_roles
                ]
                proposed.append(
                    _scope(
                        "layer_measurement_role",
                        f"{layer}_{measurement_role}",
                        subset,
                    )
                )
            for label, semantics in _SUBSYSTEMS.items():
                subset = [
                    value for value in selected if set(value.semantics) & semantics
                ]
                if subset:
                    proposed.append(
                        _scope("layer_subsystem", f"{layer}_{label}", subset)
                    )
    # Preserve each diagnostic axis even when a one-layer toy makes two axes
    # select the same physical component set.  The component-set digest lets a
    # runner reuse the underlying solve instead of recomputing it.
    unique: dict[str, DiagnosticScope] = {}
    for value in proposed:
        previous = unique.setdefault(value.scope_id, value)
        if previous != value:
            raise WholeChildComponentDiagnosticError(
                "Architecture-derived diagnostic scope IDs collide."
            )
    return tuple(sorted(unique.values(), key=lambda value: value.scope_id))


def _expected_target_ownership(
    child: ModelCircuitGraph,
    target_node_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    nodes = {value.node_id: value for value in child.nodes}
    try:
        selected = tuple(nodes[value] for value in target_node_ids)
    except KeyError as exc:
        raise WholeChildComponentDiagnosticError(
            "A fold action names a target node outside the inspected child."
        ) from exc
    paths = tuple(
        sorted(parameter.path for node in selected for parameter in node.parameters)
    )
    groups = tuple(
        sorted(
            {
                parameter.storage_group
                for node in selected
                for parameter in node.parameters
            }
        )
    )
    return paths, groups


def _validate_base(
    parent_graphs: Sequence[ModelCircuitGraph],
    child: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
) -> None:
    parents = tuple(parent_graphs)
    if not parents or not isinstance(child, ModelCircuitGraph):
        raise WholeChildComponentDiagnosticError(
            "Diagnostics need inspected parent and child graphs."
        )
    if not isinstance(fold_plan, DynamicFoldPlan) or not fold_plan.ready:
        raise WholeChildComponentDiagnosticError("The base fold plan is not ready.")
    if (
        tuple(sorted(value.fingerprint for value in parents))
        != fold_plan.parent_graph_fingerprints
    ):
        raise WholeChildComponentDiagnosticError(
            "Parent graphs differ from the fold plan."
        )
    if (
        child.fingerprint != fold_plan.child_graph_fingerprint
        or child.unique_parameter_count != fold_plan.exact_target_unique_parameters
    ):
        raise WholeChildComponentDiagnosticError(
            "The child graph differs from the plan's exact target."
        )
    if (
        sum(value.unique_parameter_count for value in parents)
        <= child.unique_parameter_count
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostics require a real positively compacted child."
        )
    planned_nodes = Counter(
        item for action in fold_plan.actions for item in action.target_node_ids
    )
    planned_paths = Counter(
        item for action in fold_plan.actions for item in action.target_parameter_paths
    )
    planned_groups = Counter(
        item for action in fold_plan.actions for item in action.target_storage_groups
    )
    target_groups = {
        parameter.storage_group for node in child.nodes for parameter in node.parameters
    }
    if (
        planned_nodes != Counter(value.node_id for value in child.nodes)
        or planned_paths != Counter(child.parameter_paths)
        or planned_groups != Counter({value: 1 for value in target_groups})
    ):
        raise WholeChildComponentDiagnosticError(
            "The base plan does not own every target node, parameter path, and "
            "learned-storage group exactly once."
        )
    if any(not action.signature.zero_gradient for action in fold_plan.actions):
        raise WholeChildComponentDiagnosticError(
            "The base plan contains a non-fold-only action."
        )
    for action in fold_plan.actions:
        expected_paths, expected_groups = _expected_target_ownership(
            child,
            action.target_node_ids,
        )
        if action.method_id != action.signature.method_id:
            raise WholeChildComponentDiagnosticError(
                "A base action method differs from its experiment signature."
            )
        if (
            action.target_parameter_paths != expected_paths
            or action.target_storage_groups != expected_groups
            or action.signature.target_unit_ids != action.target_node_ids
        ):
            raise WholeChildComponentDiagnosticError(
                "A base action's nodes, parameter paths, learned-storage groups, "
                "and signature target units do not describe one exact child "
                "ownership partition."
            )
        if action.configuration != action.signature.configuration:
            raise WholeChildComponentDiagnosticError(
                "A base action configuration differs from its signature."
            )
        _config(action.configuration, enforce_fold_only=True)


def _expected_variants(
    components: Sequence[DiagnosticComponent],
    scopes: Sequence[DiagnosticScope],
    baseline: DiagnosticFoldStrategy,
    focus: DiagnosticFoldStrategy,
) -> tuple[WholeChildDiagnosticVariant, ...]:
    ids = tuple(value.component_id for value in components)
    result = [
        WholeChildDiagnosticVariant(
            "baseline.complete_whole_child",
            None,
            tuple((value, baseline.fingerprint) for value in ids),
        )
    ]
    for scope in scopes:
        focused = set(scope.component_ids)
        result.append(
            WholeChildDiagnosticVariant(
                f"focus.{scope.scope_id}",
                scope.scope_id,
                tuple(
                    (
                        value,
                        focus.fingerprint if value in focused else baseline.fingerprint,
                    )
                    for value in ids
                ),
            )
        )
    return tuple(sorted(result, key=lambda value: value.variant_id))


def _expected_variant_by_id(
    components: Sequence[DiagnosticComponent],
    scopes: Sequence[DiagnosticScope],
    baseline: DiagnosticFoldStrategy,
    focus: DiagnosticFoldStrategy,
    variant_id: str,
) -> WholeChildDiagnosticVariant:
    component_ids = tuple(value.component_id for value in components)
    if variant_id == "baseline.complete_whole_child":
        return WholeChildDiagnosticVariant(
            variant_id,
            None,
            tuple((value, baseline.fingerprint) for value in component_ids),
        )
    scope = next(
        (value for value in scopes if variant_id == f"focus.{value.scope_id}"), None
    )
    if scope is None:
        raise WholeChildComponentDiagnosticError(
            f"Unknown architecture-derived diagnostic variant {variant_id!r}."
        )
    focused = set(scope.component_ids)
    return WholeChildDiagnosticVariant(
        variant_id,
        scope.scope_id,
        tuple(
            (
                value,
                focus.fingerprint if value in focused else baseline.fingerprint,
            )
            for value in component_ids
        ),
    )


def build_whole_child_component_diagnostic_plan(
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
    *,
    baseline_method_id: str,
    focus_method_id: str,
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
    baseline_configuration: Mapping[str, JsonScalar]
    | Sequence[tuple[str, JsonScalar]] = (),
    focus_configuration: Mapping[str, JsonScalar]
    | Sequence[tuple[str, JsonScalar]] = (),
    include_layer_scopes: bool = False,
) -> WholeChildComponentDiagnosticPlan:
    """Build executable architecture-derived variants for one complete child."""

    _validate_base(parent_graphs, child_graph, fold_plan)
    baseline_strategy = _registered_strategy(
        baseline_method_id,
        baseline_configuration,
        methods,
        operators,
    )
    focus_strategy = _registered_strategy(
        focus_method_id,
        focus_configuration,
        methods,
        operators,
    )
    if baseline_strategy.fingerprint == focus_strategy.fingerprint:
        raise WholeChildComponentDiagnosticError("Strategies must be distinct.")
    if _operator_implementation_binding(
        baseline_strategy
    ) != _operator_implementation_binding(focus_strategy):
        raise WholeChildComponentDiagnosticError(
            "Component diagnostics require two configurations of one production "
            "fold operator; cross-operator plans need an explicit coordinator."
        )
    if not isinstance(include_layer_scopes, bool):
        raise WholeChildComponentDiagnosticError(
            "include_layer_scopes must be boolean."
        )
    components = _discover_components(child_graph)
    paths = [path for value in components for path in value.parameter_paths]
    groups = [group for value in components for group in value.storage_groups]
    if (
        Counter(paths) != Counter(child_graph.parameter_paths)
        or Counter(groups)
        != Counter(
            {
                parameter.storage_group: 1
                for node in child_graph.nodes
                for parameter in node.parameters
            }
        )
        or {node for value in components for node in value.node_ids}
        != {node.node_id for node in child_graph.nodes}
    ):
        raise WholeChildComponentDiagnosticError(
            "Architecture discovery did not partition the complete child."
        )
    scopes = _discover_scopes(components, include_layer_scopes)
    result = WholeChildComponentDiagnosticPlan(
        parent_graph_fingerprints=tuple(
            sorted(value.fingerprint for value in parent_graphs)
        ),
        parent_unique_parameters=sum(
            value.unique_parameter_count for value in parent_graphs
        ),
        target_graph_fingerprint=child_graph.fingerprint,
        target_topology_fingerprint=child_graph.structural_topology_fingerprint,
        base_fold_plan_fingerprint=fold_plan.fingerprint,
        blank_child_state_fingerprint=fold_plan.child_state_fingerprint,
        exact_target_unique_parameters=child_graph.unique_parameter_count,
        target_parameter_paths_sha256=canonical_sha256(
            sorted(child_graph.parameter_paths)
        ),
        target_storage_groups_sha256=canonical_sha256(sorted(groups)),
        include_layer_scopes=include_layer_scopes,
        baseline_strategy=baseline_strategy,
        focus_strategy=focus_strategy,
        components=components,
        scopes=scopes,
        variants=_expected_variants(
            components, scopes, baseline_strategy, focus_strategy
        ),
    )
    return validate_whole_child_component_diagnostic_plan(
        result, parent_graphs, child_graph, fold_plan
    )


def validate_whole_child_component_diagnostic_plan(
    diagnostic: WholeChildComponentDiagnosticPlan,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
) -> WholeChildComponentDiagnosticPlan:
    """Re-derive the component partition and reject stale/tampered plans."""

    if not isinstance(diagnostic, WholeChildComponentDiagnosticPlan):
        raise WholeChildComponentDiagnosticError("diagnostic has the wrong type.")
    _validate_base(parent_graphs, child_graph, fold_plan)
    components = _discover_components(child_graph)
    scopes = _discover_scopes(components, diagnostic.include_layer_scopes)
    groups = [group for value in components for group in value.storage_groups]
    expected = {
        "parent_graph_fingerprints": tuple(
            sorted(value.fingerprint for value in parent_graphs)
        ),
        "parent_unique_parameters": sum(
            value.unique_parameter_count for value in parent_graphs
        ),
        "target_graph_fingerprint": child_graph.fingerprint,
        "target_topology_fingerprint": child_graph.structural_topology_fingerprint,
        "base_fold_plan_fingerprint": fold_plan.fingerprint,
        "blank_child_state_fingerprint": fold_plan.child_state_fingerprint,
        "exact_target_unique_parameters": child_graph.unique_parameter_count,
        "target_parameter_paths_sha256": canonical_sha256(
            sorted(child_graph.parameter_paths)
        ),
        "target_storage_groups_sha256": canonical_sha256(sorted(groups)),
        "components": components,
        "scopes": scopes,
        "variants": _expected_variants(
            components,
            scopes,
            diagnostic.baseline_strategy,
            diagnostic.focus_strategy,
        ),
    }
    for name, value in expected.items():
        if getattr(diagnostic, name) != value:
            raise WholeChildComponentDiagnosticError(
                f"Diagnostic {name} differs from authenticated compiler inputs."
            )
    return diagnostic


def _validate_live_strategy(
    strategy: DiagnosticFoldStrategy,
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
) -> None:
    observed = _registered_strategy(
        strategy.method_id,
        strategy.configuration,
        methods,
        operators,
    )
    if observed != strategy:
        raise WholeChildComponentDiagnosticError(
            f"Strategy {strategy.method_id!r} differs from the live authenticated "
            "method/operator registries."
        )


def _focus_components(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant: WholeChildDiagnosticVariant,
) -> tuple[DiagnosticComponent, ...]:
    if variant.focus_scope_id is None:
        return ()
    scope = next(
        value for value in diagnostic.scopes if value.scope_id == variant.focus_scope_id
    )
    component_ids = set(scope.component_ids)
    return tuple(
        value for value in diagnostic.components if value.component_id in component_ids
    )


def _action_strategy(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant: WholeChildDiagnosticVariant,
    action: FoldAction,
) -> DiagnosticFoldStrategy:
    focused = _focus_components(diagnostic, variant)
    focus_paths = {path for component in focused for path in component.parameter_paths}
    focus_groups = {
        group for component in focused for group in component.storage_groups
    }
    if focus_paths.intersection(action.target_parameter_paths) or (
        focus_groups.intersection(action.target_storage_groups)
    ):
        return diagnostic.focus_strategy
    return diagnostic.baseline_strategy


def _diagnostic_dispatch(
    diagnostic: WholeChildComponentDiagnosticPlan,
    binding: WholeChildDiagnosticBinding,
    base_fold_plan: DynamicFoldPlan,
) -> dict[str, Any]:
    return {
        "schema": DISPATCH_SCHEMA,
        "binding": binding.as_dict(),
        "plan_authentication_manifest": _diagnostic_plan_authentication_manifest(
            diagnostic
        ),
        "base_fold_plan": base_fold_plan.as_dict(),
        "strategies": {
            diagnostic.baseline_strategy.fingerprint: (
                diagnostic.baseline_strategy.as_dict()
            ),
            diagnostic.focus_strategy.fingerprint: diagnostic.focus_strategy.as_dict(),
        },
        "complete_whole_child_fold_required": True,
        "exact_target_unique_parameters": (diagnostic.exact_target_unique_parameters),
    }


def _source_groups_for_action(
    action: FoldAction,
    parent_graphs: Sequence[ModelCircuitGraph],
) -> tuple[tuple[ModelCircuitGraph, tuple[Any, ...]], ...]:
    by_name = {value.model_name: value for value in parent_graphs}
    if len(by_name) != len(tuple(parent_graphs)):
        raise WholeChildComponentDiagnosticError(
            "Parent graph model names must be unique."
        )
    result = []
    for source in action.sources:
        try:
            graph = by_name[source.parent_name]
        except KeyError as exc:
            raise WholeChildComponentDiagnosticError(
                f"Unknown source parent {source.parent_name!r}."
            ) from exc
        nodes = {value.node_id: value for value in graph.nodes}
        try:
            selected = tuple(nodes[value] for value in source.node_ids)
        except KeyError as exc:
            raise WholeChildComponentDiagnosticError(
                "A fold action names an unknown source node."
            ) from exc
        result.append((graph, selected))
    return tuple(result)


def compile_whole_child_component_diagnostic_variant(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    base_fold_plan: DynamicFoldPlan,
    *,
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
    ledger: ExperimentLedger,
) -> DynamicFoldPlan:
    """Compile one diagnostic into the actual executor plan it will run."""

    validate_whole_child_component_diagnostic_plan(
        diagnostic, parent_graphs, child_graph, base_fold_plan
    )
    if not isinstance(ledger, ExperimentLedger):
        raise WholeChildComponentDiagnosticError("ledger must be ExperimentLedger.")
    _validate_live_strategy(diagnostic.baseline_strategy, methods, operators)
    _validate_live_strategy(diagnostic.focus_strategy, methods, operators)
    variant = diagnostic.variant(_text(variant_id, "variant_id"))
    binding = bind_whole_child_component_diagnostic_variant(
        diagnostic,
        variant.variant_id,
        parent_graphs,
        child_graph,
        base_fold_plan,
    )
    dispatch_json = canonical_json_bytes(
        _diagnostic_dispatch(diagnostic, binding, base_fold_plan)
    ).decode("ascii")
    action_ids: dict[str, str] = {}
    action_strategies: dict[str, DiagnosticFoldStrategy] = {}
    for action in base_fold_plan.actions:
        strategy = _action_strategy(diagnostic, variant, action)
        action_strategies[action.action_id] = strategy
        action_ids[action.action_id] = (
            "whole-child-diagnostic."
            + canonical_sha256(
                {
                    "binding": binding.fingerprint,
                    "base_action": action.action_id,
                    "method_id": strategy.method_id,
                }
            )[:20]
        )

    target_nodes = {value.node_id: value for value in child_graph.nodes}
    actions = []
    for action in base_fold_plan.actions:
        strategy = action_strategies[action.action_id]
        method = methods.require_selectable(
            strategy.method_id, require_zero_gradient=True
        )
        if method.fingerprint != strategy.method_spec_sha256:
            raise WholeChildComponentDiagnosticError(
                f"Method contract for {method.method_id!r} changed before compile."
            )
        reserved = {
            DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY,
            DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY,
            "whole_child_component_diagnostic_selected_strategy_sha256",
        }
        base_configuration = dict(action.configuration)
        collision = reserved.intersection(base_configuration)
        if collision:
            raise WholeChildComponentDiagnosticError(
                "Base action uses compiler-reserved diagnostic keys: "
                + ", ".join(sorted(collision))
                + "."
            )
        base_configuration.update(
            {
                DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY: action.action_id,
                DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY: dispatch_json,
                "whole_child_component_diagnostic_selected_strategy_sha256": (
                    strategy.fingerprint
                ),
            }
        )
        configuration = _config(base_configuration)
        signature = replace(
            action.signature,
            method_id=method.method_id,
            method_version=method.version,
            family_id=method.family_id,
            selection_policy=(
                "architecture_driven_complete_whole_child_component_dispatch"
            ),
            fit_objective="complete_whole_child_component_fold",
            zero_gradient=True,
            configuration=configuration,
        )
        compiled = replace(
            action,
            action_id=action_ids[action.action_id],
            method_id=method.method_id,
            dependencies=tuple(action_ids[value] for value in action.dependencies),
            evidence_requirements=method.evidence_channels,
            configuration=configuration,
            signature=signature,
            ledger_decision=ledger.decision(signature),
        )
        try:
            selected_targets = tuple(
                target_nodes[value] for value in compiled.target_node_ids
            )
        except KeyError as exc:
            raise WholeChildComponentDiagnosticError(
                "A compiled action names an unknown target node."
            ) from exc
        validate_fold_method_contract(
            method,
            source_groups=_source_groups_for_action(compiled, parent_graphs),
            target_graph=child_graph,
            target_nodes=selected_targets,
        )
        actions.append(compiled)

    result = DynamicFoldPlan(
        parent_graph_fingerprints=base_fold_plan.parent_graph_fingerprints,
        parent_state_fingerprints=base_fold_plan.parent_state_fingerprints,
        child_graph_fingerprint=base_fold_plan.child_graph_fingerprint,
        child_state_fingerprint=base_fold_plan.child_state_fingerprint,
        evidence_fingerprint=base_fold_plan.evidence_fingerprint,
        exact_target_unique_parameters=base_fold_plan.exact_target_unique_parameters,
        actions=tuple(actions),
        gaps=(),
        registry_fingerprint=methods.fingerprint,
        ledger_fingerprint=ledger.fingerprint,
    )
    if not result.ready:
        raise WholeChildComponentDiagnosticError(
            "The compiled diagnostic is blocked by the experiment ledger."
        )
    if (
        Counter(value for action in result.actions for value in action.target_node_ids)
        != Counter(value.node_id for value in child_graph.nodes)
        or Counter(
            value
            for action in result.actions
            for value in action.target_parameter_paths
        )
        != Counter(child_graph.parameter_paths)
        or Counter(
            value for action in result.actions for value in action.target_storage_groups
        )
        != Counter(
            {
                parameter.storage_group: 1
                for node in child_graph.nodes
                for parameter in node.parameters
            }
        )
    ):
        raise WholeChildComponentDiagnosticError(
            "Compilation changed exact whole-child ownership."
        )
    return result


def bind_whole_child_component_diagnostic_variant(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
) -> WholeChildDiagnosticBinding:
    """Return the exact provenance an operator must seal in its transaction."""

    validate_whole_child_component_diagnostic_plan(
        diagnostic, parent_graphs, child_graph, fold_plan
    )
    variant = diagnostic.variant(_text(variant_id, "variant_id"))
    return WholeChildDiagnosticBinding(
        plan_sha256=canonical_sha256(
            _diagnostic_plan_authentication_manifest(diagnostic)
        ),
        variant_id=variant.variant_id,
        variant_sha256=variant.fingerprint,
        target_graph_fingerprint=child_graph.fingerprint,
        base_fold_plan_fingerprint=fold_plan.fingerprint,
        blank_child_state_fingerprint=fold_plan.child_state_fingerprint,
        exact_target_unique_parameters=child_graph.unique_parameter_count,
        focus_scope_id=variant.focus_scope_id,
        assignments_sha256=canonical_sha256(
            dict(variant.component_strategy_fingerprints)
        ),
    )


def validate_whole_child_diagnostic_transaction_report(
    report: Mapping[str, Any],
    binding: WholeChildDiagnosticBinding,
) -> None:
    """Fail closed unless an operator reports the exact selected binding."""

    if (
        not isinstance(report, Mapping)
        or report.get(TRANSACTION_REPORT_BINDING_KEY) != binding.as_dict()
    ):
        raise WholeChildComponentDiagnosticError(
            "Fold transaction did not seal the whole-child diagnostic binding."
        )


def validate_compiled_whole_child_component_diagnostic_variant(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    compiled_plan: DynamicFoldPlan,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    base_fold_plan: DynamicFoldPlan,
) -> WholeChildDiagnosticBinding:
    """Authenticate compiler-owned dispatch without trusting plan metadata."""

    validate_whole_child_component_diagnostic_plan(
        diagnostic, parent_graphs, child_graph, base_fold_plan
    )
    if not isinstance(compiled_plan, DynamicFoldPlan) or not compiled_plan.ready:
        raise WholeChildComponentDiagnosticError(
            "The compiled diagnostic plan is not executable."
        )
    variant = diagnostic.variant(_text(variant_id, "variant_id"))
    binding = bind_whole_child_component_diagnostic_variant(
        diagnostic,
        variant.variant_id,
        parent_graphs,
        child_graph,
        base_fold_plan,
    )
    dispatch_json = canonical_json_bytes(
        _diagnostic_dispatch(diagnostic, binding, base_fold_plan)
    ).decode("ascii")
    if (
        compiled_plan.parent_graph_fingerprints
        != base_fold_plan.parent_graph_fingerprints
        or compiled_plan.parent_state_fingerprints
        != base_fold_plan.parent_state_fingerprints
        or compiled_plan.child_graph_fingerprint != child_graph.fingerprint
        or compiled_plan.child_state_fingerprint
        != diagnostic.blank_child_state_fingerprint
        or compiled_plan.evidence_fingerprint != base_fold_plan.evidence_fingerprint
        or compiled_plan.exact_target_unique_parameters
        != diagnostic.exact_target_unique_parameters
    ):
        raise WholeChildComponentDiagnosticError(
            "The compiled variant changed the authenticated parents, blank child, "
            "evidence root, topology, or exact budget."
        )
    by_base: dict[str, FoldAction] = {}
    for action in compiled_plan.actions:
        configuration = dict(action.configuration)
        base_id = configuration.get(DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY)
        if not isinstance(base_id, str) or base_id in by_base:
            raise WholeChildComponentDiagnosticError(
                "Compiled actions do not map one-to-one onto base actions."
            )
        by_base[base_id] = action
    if set(by_base) != {value.action_id for value in base_fold_plan.actions}:
        raise WholeChildComponentDiagnosticError(
            "Compiled actions do not cover each base action exactly once."
        )
    for base_action in base_fold_plan.actions:
        action = by_base[base_action.action_id]
        strategy = _action_strategy(diagnostic, variant, base_action)
        authenticated = authenticate_whole_child_component_dispatch(
            action,
            child_graph,
            expected_method_id=strategy.method_id,
            expected_method_spec_sha256=strategy.method_spec_sha256,
            expected_evidence_requirements=strategy.evidence_requirements,
        )
        if authenticated is None or authenticated.binding != binding:
            raise WholeChildComponentDiagnosticError(
                "Compiled action did not replay the authenticated diagnostic dispatch."
            )
        configuration = dict(action.configuration)
        if (
            action.method_id != strategy.method_id
            or action.signature.method_id != strategy.method_id
            or action.signature.method_version != strategy.method_version
            or action.signature.family_id != strategy.family_id
            or action.signature.zero_gradient is not True
            or action.configuration != action.signature.configuration
            or action.evidence_requirements != strategy.evidence_requirements
            or not _is_open_ledger_decision(action.ledger_decision)
            or configuration.get(DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY) != dispatch_json
            or configuration.get(
                "whole_child_component_diagnostic_selected_strategy_sha256"
            )
            != strategy.fingerprint
        ):
            raise WholeChildComponentDiagnosticError(
                "A compiled action does not dispatch its authenticated strategy."
            )
        for key, value in base_action.configuration:
            if configuration.get(key) != value:
                raise WholeChildComponentDiagnosticError(
                    "Compilation changed base fold configuration."
                )
        for name in (
            "sources",
            "target_node_ids",
            "target_parameter_paths",
            "target_storage_groups",
            "source_logical_parameters",
            "target_logical_parameters",
        ):
            if getattr(action, name) != getattr(base_action, name):
                raise WholeChildComponentDiagnosticError(
                    f"Compilation changed base action {name}."
                )
        expected_dependencies = tuple(
            sorted(by_base[value].action_id for value in base_action.dependencies)
        )
        if action.dependencies != expected_dependencies:
            raise WholeChildComponentDiagnosticError(
                "Compilation changed base action dependencies."
            )
    if (
        Counter(
            value
            for action in compiled_plan.actions
            for value in action.target_node_ids
        )
        != Counter(value.node_id for value in child_graph.nodes)
        or Counter(
            value
            for action in compiled_plan.actions
            for value in action.target_parameter_paths
        )
        != Counter(child_graph.parameter_paths)
        or Counter(
            value
            for action in compiled_plan.actions
            for value in action.target_storage_groups
        )
        != Counter(
            {
                parameter.storage_group: 1
                for node in child_graph.nodes
                for parameter in node.parameters
            }
        )
    ):
        raise WholeChildComponentDiagnosticError(
            "Compiled ownership is not exact for the complete child."
        )
    return binding


def validate_executed_whole_child_component_diagnostic_run(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    compiled_plan: DynamicFoldPlan,
    run: DynamicFoldRun,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    base_fold_plan: DynamicFoldPlan,
) -> DynamicFoldRun:
    """Promote fold-only claims only from an observed completed executor run."""

    binding = validate_compiled_whole_child_component_diagnostic_variant(
        diagnostic,
        variant_id,
        compiled_plan,
        parent_graphs,
        child_graph,
        base_fold_plan,
    )
    if not isinstance(run, DynamicFoldRun):
        raise WholeChildComponentDiagnosticError("run must be DynamicFoldRun.")
    if not run.complete or run.plan_fingerprint != compiled_plan.fingerprint:
        raise WholeChildComponentDiagnosticError(
            "The executor did not complete the exact compiled diagnostic plan."
        )
    attempts = {value.action_id: value for value in run.attempts}
    if (
        len(attempts) != len(run.attempts)
        or set(attempts) != {value.action_id for value in compiled_plan.actions}
        or set(run.accepted_action_ids) != set(attempts)
    ):
        raise WholeChildComponentDiagnosticError(
            "The run did not accept each compiled action exactly once."
        )
    for action in compiled_plan.actions:
        attempt = attempts[action.action_id]
        base_id = dict(action.configuration)[DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY]
        base_action = next(
            value for value in base_fold_plan.actions if value.action_id == base_id
        )
        strategy = _action_strategy(
            diagnostic,
            diagnostic.variant(variant_id),
            base_action,
        )
        if attempt.method_id != action.method_id or not attempt.accepted:
            raise WholeChildComponentDiagnosticError(
                "An executor attempt used the wrong registered method."
            )
        if (
            attempt.operator_prepare_implementation_fingerprint
            != strategy.operator_prepare_implementation_fingerprint
        ):
            raise WholeChildComponentDiagnosticError(
                "The executor used a different operator implementation than the "
                "authenticated diagnostic strategy."
            )
        if (
            not attempt.operator_prepare_static_zero_training_audit_passed
            or not attempt.transaction_static_zero_training_audit_passed
        ):
            raise WholeChildComponentDiagnosticError(
                "Executed fold callables failed source-visible static audit."
            )
        validate_whole_child_diagnostic_transaction_report(
            attempt.transaction_report,
            binding,
        )
    counters = {
        "optimizer_constructions": run.optimizer_constructions,
        "optimizer_steps": run.optimizer_steps,
        "backward_calls": run.backward_calls,
        "autograd_grad_calls": run.autograd_grad_calls,
        "pruning_operations": run.pruning_operations,
        "quantization_operations": run.quantization_operations,
        "observed_parent_module_forward_calls": (
            run.observed_parent_module_forward_calls
        ),
        "observed_parent_root_forward_calls": run.observed_parent_root_forward_calls,
    }
    nonzero = {key: value for key, value in counters.items() if value != 0}
    if nonzero:
        raise WholeChildComponentDiagnosticError(
            "Observed execution violated fold-only/cached-parent requirements: "
            + ", ".join(f"{key}={value}" for key, value in sorted(nonzero.items()))
            + "."
        )
    if not run.executed_fold_static_zero_training_audit_passed:
        raise WholeChildComponentDiagnosticError(
            "The executed fold implementation manifest is not statically clean."
        )
    if Counter(run.completed_target_parameter_paths) != Counter(
        child_graph.parameter_paths
    ):
        raise WholeChildComponentDiagnosticError(
            "The run did not complete every exact child parameter path once."
        )
    if run.final_state_fingerprint == run.initial_state_fingerprint:
        raise WholeChildComponentDiagnosticError(
            "The completed diagnostic installed no learned-state delta."
        )
    return run


__all__ = [
    "DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY",
    "DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY",
    "TRANSACTION_REPORT_BINDING_KEY",
    "WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY",
    "AuthenticatedWholeChildComponentDispatch",
    "DiagnosticComponent",
    "DiagnosticFoldStrategy",
    "DiagnosticScope",
    "WholeChildComponentDiagnosticError",
    "WholeChildComponentDiagnosticPlan",
    "WholeChildDiagnosticBinding",
    "WholeChildDiagnosticVariant",
    "bind_whole_child_component_diagnostic_variant",
    "authenticate_whole_child_component_dispatch",
    "build_whole_child_component_diagnostic_plan",
    "classify_diagnostic_measurement_role",
    "classify_diagnostic_parameter_semantic",
    "compile_whole_child_component_diagnostic_variant",
    "validate_compiled_whole_child_component_diagnostic_variant",
    "validate_executed_whole_child_component_diagnostic_run",
    "validate_whole_child_component_diagnostic_plan",
    "validate_whole_child_diagnostic_transaction_report",
]
