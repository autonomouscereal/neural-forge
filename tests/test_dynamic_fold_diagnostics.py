from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_sha256,
)
from neural_forge.dynamic_fold_diagnostics import (
    DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY,
    TRANSACTION_REPORT_BINDING_KEY,
    WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY,
    WholeChildComponentDiagnosticError,
    bind_whole_child_component_diagnostic_variant,
    build_whole_child_component_diagnostic_plan,
    compile_whole_child_component_diagnostic_variant,
    validate_executed_whole_child_component_diagnostic_run,
    validate_whole_child_diagnostic_transaction_report,
)
from neural_forge.dynamic_fold_executor import (
    DynamicFoldExecutor,
    FoldDecision,
    FoldExecutionContext,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import (
    DynamicFoldPlan,
    FoldAction,
    SourceCircuitRef,
)
from neural_forge.dynamic_fold_registry import (
    ExperimentLedger,
    ExperimentSignature,
    FoldMethodRegistry,
    FoldMethodSpec,
    LedgerDecision,
)


def _parameter(
    prefix: str,
    role: str,
    count: int,
    *,
    storage_group: str | None = None,
) -> ParameterSpec:
    slug = role.replace(".", "_")
    return ParameterSpec(
        path=f"{prefix}.{slug}",
        role=role,
        shape=(count,),
        dtype="torch.float32",
        storage_group=storage_group or f"{prefix}.{slug}",
        trainable=False,
    )


def _graph(
    model_name: str,
    *,
    scale: int,
    fused_attention: bool = False,
) -> ModelCircuitGraph:
    prefix = model_name
    embedding_group = f"{prefix}.tied_token_interface"
    definitions = (
        (
            "embedding",
            CircuitKind.TOKEN_EMBEDDING,
            None,
            (
                _parameter(
                    f"{prefix}.embedding",
                    "embedding.weight",
                    4 * scale,
                    storage_group=embedding_group,
                ),
            ),
        ),
        (
            "attention",
            CircuitKind.ATTENTION,
            0,
            (
                (
                    _parameter(
                        f"{prefix}.attention",
                        "operator.c_attn.weight",
                        16 * scale,
                    ),
                )
                if fused_attention
                else (
                    _parameter(
                        f"{prefix}.attention",
                        "operator.q_proj.weight",
                        4 * scale,
                    ),
                    _parameter(
                        f"{prefix}.attention",
                        "operator.k_proj.weight",
                        4 * scale,
                    ),
                    _parameter(
                        f"{prefix}.attention",
                        "operator.v_proj.weight",
                        4 * scale,
                    ),
                    _parameter(
                        f"{prefix}.attention",
                        "operator.out_proj.weight",
                        4 * scale,
                    ),
                )
            ),
        ),
        (
            "state_mixer",
            CircuitKind.SHORT_CONV,
            0,
            (
                _parameter(
                    f"{prefix}.state_mixer",
                    "operator.conv.weight",
                    4 * scale,
                ),
            ),
        ),
        (
            "norm",
            CircuitKind.NORM,
            0,
            (
                _parameter(
                    f"{prefix}.norm",
                    "operator_norm.weight",
                    4 * scale,
                ),
            ),
        ),
        ("residual", CircuitKind.RESIDUAL_ADD, 0, ()),
        (
            "mlp",
            CircuitKind.GATED_MLP,
            0,
            (
                _parameter(
                    f"{prefix}.mlp",
                    "mlp.gate_proj.weight",
                    4 * scale,
                ),
                _parameter(
                    f"{prefix}.mlp",
                    "mlp.up_proj.weight",
                    4 * scale,
                ),
                _parameter(
                    f"{prefix}.mlp",
                    "mlp.down_proj.weight",
                    4 * scale,
                ),
            ),
        ),
        (
            "head",
            CircuitKind.OUTPUT_HEAD,
            None,
            (
                _parameter(
                    f"{prefix}.head",
                    "output_head.weight",
                    4 * scale,
                    storage_group=embedding_group,
                ),
            ),
        ),
    )
    nodes = tuple(
        CircuitNode(
            node_id=f"{prefix}.{label}",
            kind=kind,
            order=index,
            path=f"{prefix}.{label}",
            layer_index=layer_index,
            inputs=(),
            outputs=(),
            parameters=parameters,
            attributes=(("coupled_roles", "whole_child_component_dispatch"),),
        )
        for index, (label, kind, layer_index, parameters) in enumerate(definitions)
    )
    logical = sum(value.logical_parameter_count for value in nodes)
    storage_counts: dict[str, int] = {}
    for node in nodes:
        for parameter in node.parameters:
            storage_counts.setdefault(parameter.storage_group, parameter.numel)
    unique = sum(storage_counts.values())
    return ModelCircuitGraph(
        model_name=model_name,
        family="synthetic",
        model_class="tests.SyntheticWholeChild",
        nodes=nodes,
        edges=(),
        interfaces=(),
        logical_parameter_count=logical,
        unique_parameter_count=unique,
        unique_parameter_bytes=unique * 4,
        config_fingerprint=canonical_sha256(
            {"model_name": model_name, "scale": scale, "fused": fused_attention}
        ),
    )


class _TinyWholeChild(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Parameter(torch.zeros(width))
        self.k_proj = nn.Parameter(torch.zeros(width))
        self.mlp_up = nn.Parameter(torch.zeros(width))
        self.mlp_down = nn.Parameter(torch.zeros(width))
        self.head = nn.Parameter(torch.zeros(width))


def _runtime_graph(model_name: str, width: int) -> ModelCircuitGraph:
    role_rows = (
        (
            "attention",
            CircuitKind.ATTENTION,
            0,
            (
                ("q_proj", "operator.q_proj.weight"),
                ("k_proj", "operator.k_proj.weight"),
            ),
        ),
        (
            "mlp",
            CircuitKind.GATED_MLP,
            0,
            (("mlp_up", "mlp.up_proj.weight"), ("mlp_down", "mlp.down_proj.weight")),
        ),
        (
            "head",
            CircuitKind.OUTPUT_HEAD,
            None,
            (("head", "output_head.weight"),),
        ),
        ("residual", CircuitKind.RESIDUAL_ADD, 0, ()),
    )
    nodes = tuple(
        CircuitNode(
            node_id=f"{model_name}.{label}",
            kind=kind,
            order=index,
            path=f"{model_name}.{label}",
            layer_index=layer_index,
            inputs=(),
            outputs=(),
            parameters=tuple(
                ParameterSpec(
                    path=path,
                    role=role,
                    shape=(width,),
                    dtype="torch.float32",
                    storage_group=f"{model_name}.{path}",
                    trainable=False,
                )
                for path, role in parameters
            ),
            attributes=(("coupled_roles", "whole_child_component_dispatch"),),
        )
        for index, (label, kind, layer_index, parameters) in enumerate(role_rows)
    )
    logical = sum(value.logical_parameter_count for value in nodes)
    return ModelCircuitGraph(
        model_name=model_name,
        family="synthetic_runtime",
        model_class="tests.TinyWholeChild",
        nodes=nodes,
        edges=(),
        interfaces=(),
        logical_parameter_count=logical,
        unique_parameter_count=logical,
        unique_parameter_bytes=logical * 4,
        config_fingerprint=canonical_sha256({"model_name": model_name, "width": width}),
    )


def _fold_plan(
    parent: ModelCircuitGraph,
    child: ModelCircuitGraph,
    *,
    parent_state: str | None = None,
    child_state: str | None = None,
    evidence: str | None = None,
) -> DynamicFoldPlan:
    parent_state = parent_state or canonical_sha256({"state": parent.model_name})
    child_state = child_state or canonical_sha256({"state": "compiler-zero-blank"})
    evidence = evidence or canonical_sha256({"cache": "sealed-parent-once"})
    configuration = (
        ("fit_objective", "complete_coupled_circuit_fold"),
        ("selection_policy", "architecture_driven"),
        ("promotion_gate", "manual_whole_response_review"),
    )
    signature = ExperimentSignature(
        method_id="synthetic_whole_child_fold",
        method_version=1,
        family_id="synthetic_whole_child",
        parent_graph_fingerprints=(parent.fingerprint,),
        parent_state_fingerprints=((parent.model_name, parent_state),),
        child_graph_fingerprint=child.fingerprint,
        child_state_fingerprint=child_state,
        evidence_fingerprint=evidence,
        source_unit_ids=tuple(value.node_id for value in parent.nodes),
        target_unit_ids=tuple(value.node_id for value in child.nodes),
        topology_transition="dense_parent_to_exact_blank_child",
        selection_policy="architecture_driven",
        fit_objective="complete_coupled_circuit_fold",
        promotion_gate="manual_whole_response_review",
        zero_gradient=True,
        configuration=configuration,
    )
    action = FoldAction(
        action_id="whole-child",
        sequence_index=0,
        method_id=signature.method_id,
        sources=(
            SourceCircuitRef(
                parent.model_name,
                tuple(value.node_id for value in parent.nodes),
            ),
        ),
        target_node_ids=tuple(value.node_id for value in child.nodes),
        target_parameter_paths=child.parameter_paths,
        target_storage_groups=tuple(
            sorted(
                {
                    parameter.storage_group
                    for node in child.nodes
                    for parameter in node.parameters
                }
            )
        ),
        dependencies=(),
        evidence_requirements=("structure",),
        source_logical_parameters=parent.logical_parameter_count,
        target_logical_parameters=child.logical_parameter_count,
        configuration=configuration,
        signature=signature,
        ledger_decision=LedgerDecision(True, "new exact candidate"),
    )
    return DynamicFoldPlan(
        parent_graph_fingerprints=(parent.fingerprint,),
        parent_state_fingerprints=((parent.model_name, parent_state),),
        child_graph_fingerprint=child.fingerprint,
        child_state_fingerprint=child_state,
        evidence_fingerprint=evidence,
        exact_target_unique_parameters=child.unique_parameter_count,
        actions=(action,),
        gaps=(),
        registry_fingerprint=canonical_sha256({"registry": "test"}),
        ledger_fingerprint=canonical_sha256({"ledger": "empty"}),
    )


class _DiagnosticTransaction:
    def __init__(self, method_id, model, original, binding, candidate):
        self.method_id = method_id
        self.model = model
        self.original = original
        self.binding = binding
        self.candidate_state_fingerprint = candidate
        self.active = True

    def commit(self):
        self.active = False
        return {TRANSACTION_REPORT_BINDING_KEY: self.binding, "committed": True}

    def rollback(self):
        for name, parameter in self.model.named_parameters():
            parameter.copy_(self.original[name])
        self.active = False
        return {
            TRANSACTION_REPORT_BINDING_KEY: self.binding,
            "committed": False,
            "rollback_exact": True,
        }


class _BaselineDiagnosticOperator:
    method_id = "diagnostic_baseline_fold"
    whole_child_component_diagnostic_capability = (
        WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY
    )

    def prepare(self, action, context, _evidence):
        dispatch = json.loads(
            dict(action.configuration)[DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY]
        )
        original = {
            name: parameter.detach().clone()
            for name, parameter in context.child_model.named_parameters()
        }
        parent_parameters = dict(context.parent_models["parent"].named_parameters())
        for name, parameter in context.child_model.named_parameters():
            rows = parent_parameters[name].detach().reshape(parameter.numel(), -1)
            parameter.copy_(rows.mean(dim=1).reshape_as(parameter))
        return _DiagnosticTransaction(
            self.method_id,
            context.child_model,
            original,
            dispatch["binding"],
            context.child_state_fingerprint,
        )


class _FocusDiagnosticOperator:
    method_id = "diagnostic_attention_focus_fold"
    whole_child_component_diagnostic_capability = (
        WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY
    )

    def prepare(self, action, context, _evidence):
        dispatch = json.loads(
            dict(action.configuration)[DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY]
        )
        original = {
            name: parameter.detach().clone()
            for name, parameter in context.child_model.named_parameters()
        }
        strategies = dispatch["strategies"]
        focus_fingerprint = next(
            fingerprint
            for fingerprint, strategy in strategies.items()
            if strategy["method_id"] == self.method_id
        )
        focused_paths = {
            path
            for component in dispatch["components"]
            if component["assigned_strategy_fingerprint"] == focus_fingerprint
            for path in component["parameter_paths"]
        }
        parent_parameters = dict(context.parent_models["parent"].named_parameters())
        for name, parameter in context.child_model.named_parameters():
            rows = parent_parameters[name].detach().reshape(parameter.numel(), -1)
            folded = rows[:, 0] if name in focused_paths else rows.mean(dim=1)
            parameter.copy_(folded.reshape_as(parameter))
        return _DiagnosticTransaction(
            self.method_id,
            context.child_model,
            original,
            dispatch["binding"],
            context.child_state_fingerprint,
        )


class _BackwardDiagnosticOperator:
    method_id = "diagnostic_malicious_fold"
    whole_child_component_diagnostic_capability = (
        WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY
    )

    def prepare(self, _action, _context, _evidence):
        torch.tensor(1.0, requires_grad=True).backward()
        raise AssertionError("static audit failed to reject backward")


class _NoComponentDispatchOperator:
    method_id = "diagnostic_operator_without_component_dispatch"

    def prepare(self, action, context, evidence):
        return _BaselineDiagnosticOperator().prepare(action, context, evidence)


class _Accept:
    def evaluate(self, _context):
        return FoldDecision.accept("synthetic executor-backed fold accepted")


def _method(method_id: str) -> FoldMethodSpec:
    return FoldMethodSpec(
        method_id=method_id,
        version=1,
        family_id=f"{method_id}_family",
        status="experimental",
        source_kinds=("any",),
        target_kinds=("any",),
        evidence_channels=("structure",),
        coupled_roles=("whole_child_component_dispatch",),
        zero_gradient=True,
        standalone=True,
        reason="Executable component-aware whole-child fold test method.",
    )


def _registries(*, malicious: bool = False):
    operators = [_BaselineDiagnosticOperator(), _FocusDiagnosticOperator()]
    methods = [
        _method(_BaselineDiagnosticOperator.method_id),
        _method(_FocusDiagnosticOperator.method_id),
    ]
    if malicious:
        operators.append(_BackwardDiagnosticOperator())
        methods.append(_method(_BackwardDiagnosticOperator.method_id))
    return FoldMethodRegistry(methods), FoldOperatorRegistry(operators)


def _diagnostic_fixture(*, fused_attention: bool = False):
    parent = _graph("parent", scale=4, fused_attention=fused_attention)
    child = _graph("child", scale=1, fused_attention=fused_attention)
    fold_plan = _fold_plan(parent, child)
    methods, operators = _registries()
    diagnostic = build_whole_child_component_diagnostic_plan(
        (parent,),
        child,
        fold_plan,
        baseline_method_id=_BaselineDiagnosticOperator.method_id,
        focus_method_id=_FocusDiagnosticOperator.method_id,
        methods=methods,
        operators=operators,
        baseline_configuration={"residual_basis": "balanced"},
        focus_configuration={"residual_basis": "attention_aware"},
        include_layer_scopes=True,
    )
    return parent, child, fold_plan, diagnostic, methods, operators


def test_architecture_discovery_builds_full_child_component_variants() -> None:
    parent, child, fold_plan, diagnostic, _methods, _operators = _diagnostic_fixture()

    assert diagnostic.compression_ratio == pytest.approx(4.0)
    assert diagnostic.exact_target_unique_parameters == child.unique_parameter_count
    assert len(diagnostic.variants) == len(diagnostic.scopes) + 1
    semantics = {
        semantic
        for component in diagnostic.components
        for semantic in component.semantics
    }
    assert {
        "attention_query",
        "attention_key",
        "attention_value",
        "attention_output",
        "state_mixer",
        "mlp_producer",
        "mlp_writer",
        "normalization",
        "residual_routing",
    } <= semantics

    tied = [
        value
        for value in diagnostic.components
        if set(value.semantics) == {"token_embedding", "output_head"}
    ]
    assert len(tied) == 1
    assert len(tied[0].storage_groups) == 1

    query_scope = next(
        value for value in diagnostic.scopes if value.semantics == ("attention_query",)
    )
    query_variant = diagnostic.variant(f"focus.{query_scope.scope_id}")
    assignments = dict(query_variant.component_strategy_fingerprints)
    assert set(assignments) == {value.component_id for value in diagnostic.components}
    assert {
        component_id
        for component_id, strategy in assignments.items()
        if strategy == diagnostic.focus_strategy.fingerprint
    } == set(query_scope.component_ids)
    assert {
        component_id
        for component_id, strategy in assignments.items()
        if strategy == diagnostic.baseline_strategy.fingerprint
    } == set(assignments) - set(query_scope.component_ids)

    binding = bind_whole_child_component_diagnostic_variant(
        diagnostic,
        query_variant.variant_id,
        (parent,),
        child,
        fold_plan,
    )
    assert binding.exact_target_unique_parameters == child.unique_parameter_count
    assert binding.base_fold_plan_fingerprint == fold_plan.fingerprint
    assert binding.as_dict()["complete_whole_child_fold_required"] is True
    assert diagnostic.as_dict()["parent_shell_hybrid_forbidden"] is True
    assert diagnostic.as_dict()["fold_only_execution_claim"].startswith("unproven")
    residual_component_ids = {
        value.component_id
        for value in diagnostic.components
        if value.semantics == ("residual_routing",)
    }
    assert residual_component_ids
    assert all(
        not residual_component_ids.intersection(scope.component_ids)
        for scope in diagnostic.scopes
    )
    validate_whole_child_diagnostic_transaction_report(
        {TRANSACTION_REPORT_BINDING_KEY: binding.as_dict()},
        binding,
    )
    with pytest.raises(WholeChildComponentDiagnosticError, match="seal"):
        validate_whole_child_diagnostic_transaction_report({}, binding)


def test_fused_attention_is_not_falsely_split_into_qkvo() -> None:
    _parent, _child, _fold_plan_value, diagnostic, _methods, _operators = (
        _diagnostic_fixture(fused_attention=True)
    )
    semantics = {
        semantic
        for component in diagnostic.components
        for semantic in component.semantics
    }
    assert "attention_coupled_projection" in semantics
    assert not {
        "attention_query",
        "attention_key",
        "attention_value",
    }.intersection(semantics)


def test_partial_plan_and_non_compacting_control_fail_closed() -> None:
    parent = _graph("parent", scale=4)
    child = _graph("child", scale=1)
    fold_plan = _fold_plan(parent, child)
    methods, operators = _registries()
    action = fold_plan.actions[0]
    partial_action = replace(
        action,
        target_node_ids=action.target_node_ids[:-1],
        target_parameter_paths=action.target_parameter_paths[:-1],
    )
    partial = replace(fold_plan, actions=(partial_action,))
    with pytest.raises(WholeChildComponentDiagnosticError, match="exactly once"):
        build_whole_child_component_diagnostic_plan(
            (parent,),
            child,
            partial,
            baseline_method_id=_BaselineDiagnosticOperator.method_id,
            focus_method_id=_FocusDiagnosticOperator.method_id,
            methods=methods,
            operators=operators,
        )

    same_size_parent = _graph("same_size_parent", scale=1)
    same_size_plan = _fold_plan(same_size_parent, child)
    with pytest.raises(
        WholeChildComponentDiagnosticError, match="positively compacted"
    ):
        build_whole_child_component_diagnostic_plan(
            (same_size_parent,),
            child,
            same_size_plan,
            baseline_method_id=_BaselineDiagnosticOperator.method_id,
            focus_method_id=_FocusDiagnosticOperator.method_id,
            methods=methods,
            operators=operators,
        )


def test_duplicate_node_owner_fails_exact_multiplicity_validation() -> None:
    parent = _graph("parent", scale=4)
    child = _graph("child", scale=1)
    fold_plan = _fold_plan(parent, child)
    methods, operators = _registries()
    base_action = fold_plan.actions[0]
    residual_id = next(
        value.node_id
        for value in child.nodes
        if value.kind == CircuitKind.RESIDUAL_ADD.value
    )
    duplicate = replace(
        base_action,
        action_id="duplicate-residual-owner",
        sequence_index=1,
        target_node_ids=(residual_id,),
        target_parameter_paths=(),
        target_storage_groups=(),
        target_logical_parameters=0,
        signature=replace(
            base_action.signature,
            target_unit_ids=(residual_id,),
        ),
    )
    duplicate_plan = replace(fold_plan, actions=(base_action, duplicate))

    with pytest.raises(WholeChildComponentDiagnosticError, match="exactly once"):
        build_whole_child_component_diagnostic_plan(
            (parent,),
            child,
            duplicate_plan,
            baseline_method_id=_BaselineDiagnosticOperator.method_id,
            focus_method_id=_FocusDiagnosticOperator.method_id,
            methods=methods,
            operators=operators,
        )


@pytest.mark.parametrize(
    "configuration",
    ({"adam_training": True}, {"solver": "backward"}),
)
def test_non_fold_strategy_configuration_is_rejected(configuration) -> None:
    parent = _graph("parent", scale=4)
    child = _graph("child", scale=1)
    methods, operators = _registries()
    with pytest.raises(WholeChildComponentDiagnosticError, match="forbidden non-fold"):
        build_whole_child_component_diagnostic_plan(
            (parent,),
            child,
            _fold_plan(parent, child),
            baseline_method_id=_BaselineDiagnosticOperator.method_id,
            focus_method_id=_FocusDiagnosticOperator.method_id,
            methods=methods,
            operators=operators,
            focus_configuration=configuration,
        )


def test_source_visible_backward_operator_is_rejected_before_compile() -> None:
    parent = _graph("parent", scale=4)
    child = _graph("child", scale=1)
    methods, operators = _registries(malicious=True)
    with pytest.raises(
        WholeChildComponentDiagnosticError, match="static zero-training audit"
    ):
        build_whole_child_component_diagnostic_plan(
            (parent,),
            child,
            _fold_plan(parent, child),
            baseline_method_id=_BaselineDiagnosticOperator.method_id,
            focus_method_id=_BackwardDiagnosticOperator.method_id,
            methods=methods,
            operators=operators,
        )


def test_operator_without_production_dispatch_capability_fails_closed() -> None:
    parent = _graph("parent", scale=4)
    child = _graph("child", scale=1)
    methods = FoldMethodRegistry(
        (
            _method(_BaselineDiagnosticOperator.method_id),
            _method(_NoComponentDispatchOperator.method_id),
        )
    )
    operators = FoldOperatorRegistry(
        (_BaselineDiagnosticOperator(), _NoComponentDispatchOperator())
    )
    with pytest.raises(
        WholeChildComponentDiagnosticError,
        match="does not declare the executable whole-child component-dispatch",
    ):
        build_whole_child_component_diagnostic_plan(
            (parent,),
            child,
            _fold_plan(parent, child),
            baseline_method_id=_BaselineDiagnosticOperator.method_id,
            focus_method_id=_NoComponentDispatchOperator.method_id,
            methods=methods,
            operators=operators,
        )


def test_variants_execute_distinct_authenticated_whole_child_folds() -> None:
    parent_model = _TinyWholeChild(8)
    with torch.no_grad():
        for index, parameter in enumerate(parent_model.parameters(), start=1):
            parameter.copy_(
                torch.arange(parameter.numel(), dtype=parameter.dtype)
                + float(index * 10)
            )
    blank = _TinyWholeChild(2)
    parent_graph = _runtime_graph("parent", 8)
    child_graph = _runtime_graph("child", 2)
    evidence = canonical_sha256({"cache": "one-sealed-parent-trace-cache"})
    base_plan = _fold_plan(
        parent_graph,
        child_graph,
        parent_state=fingerprint_model_state(parent_model),
        child_state=fingerprint_model_state(blank),
        evidence=evidence,
    )
    methods, operators = _registries()
    diagnostic = build_whole_child_component_diagnostic_plan(
        (parent_graph,),
        child_graph,
        base_plan,
        baseline_method_id=_BaselineDiagnosticOperator.method_id,
        focus_method_id=_FocusDiagnosticOperator.method_id,
        methods=methods,
        operators=operators,
    )
    query_scope = next(
        value for value in diagnostic.scopes if value.semantics == ("attention_query",)
    )
    baseline_variant = diagnostic.variant("baseline.complete_whole_child")
    focus_variant = diagnostic.variant(f"focus.{query_scope.scope_id}")
    baseline_ledger = ExperimentLedger()
    focus_ledger = ExperimentLedger()
    baseline_plan = compile_whole_child_component_diagnostic_variant(
        diagnostic,
        baseline_variant.variant_id,
        (parent_graph,),
        child_graph,
        base_plan,
        methods=methods,
        operators=operators,
        ledger=baseline_ledger,
    )
    focus_plan = compile_whole_child_component_diagnostic_variant(
        diagnostic,
        focus_variant.variant_id,
        (parent_graph,),
        child_graph,
        base_plan,
        methods=methods,
        operators=operators,
        ledger=focus_ledger,
    )
    assert {value.method_id for value in baseline_plan.actions} == {
        _BaselineDiagnosticOperator.method_id
    }
    assert {value.method_id for value in focus_plan.actions} == {
        _FocusDiagnosticOperator.method_id
    }
    assert baseline_plan.child_graph_fingerprint == focus_plan.child_graph_fingerprint
    assert (
        baseline_plan.exact_target_unique_parameters
        == focus_plan.exact_target_unique_parameters
        == child_graph.unique_parameter_count
    )

    baseline_child = _TinyWholeChild(2)
    focus_child = _TinyWholeChild(2)
    baseline_run = DynamicFoldExecutor(
        operators=operators,
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        final_evaluator=_Accept(),
        ledger=baseline_ledger,
    ).execute(
        baseline_plan,
        FoldExecutionContext(
            parent_models={"parent": parent_model},
            child_model=baseline_child,
            parent_graphs={"parent": parent_graph},
            child_graph=child_graph,
        ),
    )
    focus_run = DynamicFoldExecutor(
        operators=operators,
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        final_evaluator=_Accept(),
        ledger=focus_ledger,
    ).execute(
        focus_plan,
        FoldExecutionContext(
            parent_models={"parent": parent_model},
            child_model=focus_child,
            parent_graphs={"parent": parent_graph},
            child_graph=child_graph,
        ),
    )
    validate_executed_whole_child_component_diagnostic_run(
        diagnostic,
        baseline_variant.variant_id,
        baseline_plan,
        baseline_run,
        (parent_graph,),
        child_graph,
        base_plan,
    )
    validate_executed_whole_child_component_diagnostic_run(
        diagnostic,
        focus_variant.variant_id,
        focus_plan,
        focus_run,
        (parent_graph,),
        child_graph,
        base_plan,
    )
    assert baseline_run.final_state_fingerprint != focus_run.final_state_fingerprint
    assert torch.equal(baseline_child.k_proj, focus_child.k_proj)
    assert not torch.equal(baseline_child.q_proj, focus_child.q_proj)
    parent_q_rows = parent_model.q_proj.detach().reshape(2, -1)
    assert torch.equal(baseline_child.q_proj, parent_q_rows.mean(dim=1))
    assert torch.equal(focus_child.q_proj, parent_q_rows[:, 0])
