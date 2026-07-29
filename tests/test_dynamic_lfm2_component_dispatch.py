from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_diagnostics import (
    TRANSACTION_REPORT_BINDING_KEY,
    build_whole_child_component_diagnostic_plan,
    compile_whole_child_component_diagnostic_variant,
    validate_executed_whole_child_component_diagnostic_run,
)
from neural_forge.dynamic_fold_executor import (
    DynamicFoldExecutor,
    FoldDecision,
    FoldExecutionContext,
    FoldOperatorRegistry,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_fold_registry import (
    ExperimentLedger,
    default_method_registry,
)
from neural_forge.dynamic_lfm2_adapter import (
    Lfm2TargetSynthesisAdapter,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (
    DENSE_BUDGET_MODEL_METHOD,
    LFM2_COMPONENT_CLOSED_LOOP_STAGE,
    LFM2_COMPONENT_DISPATCH_EXECUTION_SCHEMA,
    LFM2_COMPONENT_FOLD_STAGE_CONFIGURATION_KEY,
    LFM2_COMPONENT_STRUCTURAL_STAGE,
    Lfm2DenseBudgetAdmissionEvaluator,
    Lfm2DenseBudgetEvidenceProvider,
    Lfm2DenseBudgetFoldConfig,
    Lfm2DenseBudgetFoldOnlyInitializer,
    Lfm2DenseBudgetModelOperator,
)
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (
    Lfm2DynamicClosedLoopAdapterConfig,
)
from neural_forge.dynamic_closed_loop_fold import ClosedLoopCompilerConfig


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")


def _tokenizer_fingerprint() -> str:
    return canonical_sha256(
        {"kind": "lfm2-exact4x-component-dispatch-test", "vocabulary_size": 37}
    )


def _model(*, hidden: int, intermediate: int, heads: int, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=37,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _batch(prompt: tuple[int, ...], forced: tuple[int, ...]):
    input_ids = torch.tensor([prompt], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([forced], dtype=torch.long),
    }


class _Accept:
    def evaluate(self, _context):
        return FoldDecision.accept("manual whole-response review remains downstream")


def _exact4x_target(parent, cache):
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    request = plan_budgeted_dynamic_target(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=460,
            match_policy="exact",
            maximum_candidates=64,
        ),
        child_name="exact4x-component-child",
        tokenizer_fingerprint=tokenizer,
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        blank_seed=7421,
        catalogs=DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    minimum_hidden_size=4,
                    hidden_size_multiple=4,
                    minimum_head_dim=4,
                ),
            )
        ),
    ).target_request
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (
                Lfm2TargetSynthesisAdapter(
                    terminal_topology=Lfm2Topology(1, 1, 6),
                    evidence_fingerprint=cache.cache_key,
                    device="cpu",
                    dtype=torch.float32,
                ),
            )
        ),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    return parent_graph, target


def _execute(
    plan,
    *,
    ledger,
    operators,
    cache,
    target,
    parent,
    parent_graph,
):
    return DynamicFoldExecutor(
        operators=operators,
        evidence_provider=Lfm2DenseBudgetEvidenceProvider(
            cache,
            cache.cache_key,
        ),
        evaluator=Lfm2DenseBudgetAdmissionEvaluator(),
        final_evaluator=_Accept(),
        ledger=ledger,
        initializer=Lfm2DenseBudgetFoldOnlyInitializer(
            cache.cache_key,
            target.adapter_payload,
        ),
    ).execute(
        plan,
        FoldExecutionContext(
            parent_models={"parent": parent},
            child_model=target.model,
            parent_graphs={"parent": parent_graph},
            child_graph=target.graph,
        ),
    )


def test_real_lfm2_operator_executes_exact4x_component_dispatch(tmp_path) -> None:
    """One real inspected LFM2 child stays complete while attention is isolated."""

    parent = _model(hidden=8, intermediate=20, heads=2, seed=7400)
    parent_state = fingerprint_model_state(parent)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "sealed-parent-cache",
        parent,
        (
            _batch((1, 2, 3, 4), (5,)),
            _batch((6, 7, 8, 9), (10,)),
            _batch((11, 12, 13, 14), (15,)),
            _batch((16, 17, 18, 19), (20,)),
        ),
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=1,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    parent_graph, target = _exact4x_target(parent, cache)
    assert parent_graph.family == target.graph.family == "lfm2"
    assert parent_graph.unique_parameter_count == 1_840
    assert target.graph.unique_parameter_count == 460
    assert (
        parent_graph.unique_parameter_count / target.graph.unique_parameter_count == 4
    )
    assert set(target.graph.parameter_paths) == {
        name
        for name, _parameter in target.model.named_parameters(
            recurse=True, remove_duplicate=False
        )
    }

    methods = default_method_registry()
    base_plan = DynamicFoldPlanner(
        methods=methods,
        ledger=ExperimentLedger(),
    ).plan(
        (parent_graph,),
        target.graph,
        child_state_fingerprint=fingerprint_model_state(target.model),
        evidence_fingerprint=cache.cache_key,
        parent_state_fingerprints={"parent": parent_state},
        exact_target_unique_parameters=target.graph.unique_parameter_count,
    )
    assert base_plan.ready
    assert len(base_plan.actions) == 1
    assert base_plan.actions[0].method_id == DENSE_BUDGET_MODEL_METHOD
    assert set(base_plan.actions[0].target_parameter_paths) == set(
        target.graph.parameter_paths
    )

    operator = Lfm2DenseBudgetModelOperator(
        target.adapter_payload,
        cache,
        config=Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="weight_complete",
            run_closed_loop=True,
            closed_loop_adapter_config=Lfm2DynamicClosedLoopAdapterConfig(
                ridge=1e-5,
                norm_ridge=1e-5,
                max_prior_delta_ratio=4.0,
                cluster_max_iterations=4,
                cluster_distance_chunk_size=16,
                minimum_layer_relative_rmse_improvement=-1_000_000.0,
                minimum_final_norm_relative_rmse_improvement=-1_000_000.0,
                minimum_output_kl_improvement=-1_000_000.0,
                minimum_output_known_sequence_nll_improvement=-1_000_000.0,
                output_validation_tolerance=1_000_000.0,
            ),
            closed_loop_compiler_config=ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-component-dispatch-exact4x-v1",
            ),
        ),
    )
    operators = FoldOperatorRegistry((operator,))
    diagnostic = build_whole_child_component_diagnostic_plan(
        (parent_graph,),
        target.graph,
        base_plan,
        baseline_method_id=DENSE_BUDGET_MODEL_METHOD,
        focus_method_id=DENSE_BUDGET_MODEL_METHOD,
        methods=methods,
        operators=operators,
        baseline_configuration={
            LFM2_COMPONENT_FOLD_STAGE_CONFIGURATION_KEY: (
                LFM2_COMPONENT_STRUCTURAL_STAGE
            )
        },
        focus_configuration={
            LFM2_COMPONENT_FOLD_STAGE_CONFIGURATION_KEY: (
                LFM2_COMPONENT_CLOSED_LOOP_STAGE
            )
        },
        include_layer_scopes=True,
    )
    attention_scope = next(
        scope
        for scope in diagnostic.scopes
        if scope.scope_id.startswith("scope.subsystem.attention.")
    )
    baseline_variant = diagnostic.variant("baseline.complete_whole_child")
    focus_variant = diagnostic.variant(f"focus.{attention_scope.scope_id}")
    baseline_ledger = ExperimentLedger()
    focus_ledger = ExperimentLedger()
    baseline_plan = compile_whole_child_component_diagnostic_variant(
        diagnostic,
        baseline_variant.variant_id,
        (parent_graph,),
        target.graph,
        base_plan,
        methods=methods,
        operators=operators,
        ledger=baseline_ledger,
    )
    focus_plan = compile_whole_child_component_diagnostic_variant(
        diagnostic,
        focus_variant.variant_id,
        (parent_graph,),
        target.graph,
        base_plan,
        methods=methods,
        operators=operators,
        ledger=focus_ledger,
    )

    baseline_target = SimpleNamespace(
        model=copy.deepcopy(target.model),
        adapter_payload=target.adapter_payload,
        graph=target.graph,
    )
    focus_target = SimpleNamespace(
        model=copy.deepcopy(target.model),
        adapter_payload=target.adapter_payload,
        graph=target.graph,
    )
    baseline_run = _execute(
        baseline_plan,
        ledger=baseline_ledger,
        operators=operators,
        cache=cache,
        target=baseline_target,
        parent=parent,
        parent_graph=parent_graph,
    )
    focus_run = _execute(
        focus_plan,
        ledger=focus_ledger,
        operators=operators,
        cache=cache,
        target=focus_target,
        parent=parent,
        parent_graph=parent_graph,
    )
    validate_executed_whole_child_component_diagnostic_run(
        diagnostic,
        baseline_variant.variant_id,
        baseline_plan,
        baseline_run,
        (parent_graph,),
        target.graph,
        base_plan,
    )
    validate_executed_whole_child_component_diagnostic_run(
        diagnostic,
        focus_variant.variant_id,
        focus_plan,
        focus_run,
        (parent_graph,),
        target.graph,
        base_plan,
    )

    focused_paths = {
        path
        for component in diagnostic.components
        if component.component_id in set(attention_scope.component_ids)
        for path in component.parameter_paths
    }
    baseline_parameters = dict(
        baseline_target.model.named_parameters(recurse=True, remove_duplicate=False)
    )
    focus_parameters = dict(
        focus_target.model.named_parameters(recurse=True, remove_duplicate=False)
    )
    assert (
        set(baseline_parameters)
        == set(focus_parameters)
        == set(target.graph.parameter_paths)
    )
    assert any(
        not torch.equal(baseline_parameters[path], focus_parameters[path])
        for path in focused_paths
    )
    assert all(
        torch.equal(baseline_parameters[path], focus_parameters[path])
        for path in set(baseline_parameters) - focused_paths
    )

    baseline_report = baseline_run.attempts[0].transaction_report
    focus_report = focus_run.attempts[0].transaction_report
    execution = focus_report["whole_child_component_diagnostic_execution"]
    assert execution["schema"] == LFM2_COMPONENT_DISPATCH_EXECUTION_SCHEMA
    assert execution["exact_target_unique_parameters"] == 460
    assert execution["complete_parameter_path_coverage"] is True
    assert execution["actual_executor_action_count"] > 0
    assert all(
        action["executor_action_performed"] and action["production_fold_operators"]
        for action in execution["actual_executor_actions"]
    )
    assert execution["focus_parameter_path_count"] == len(focused_paths)
    assert execution["focus_changed_parameter_path_count"] > 0
    assert baseline_report["closed_loop_enabled"] is False
    assert focus_report["closed_loop_enabled"] is True
    assert focus_report[TRANSACTION_REPORT_BINDING_KEY] == (execution["binding"])
    assert focus_run.optimizer_constructions == 0
    assert focus_run.optimizer_steps == 0
    assert focus_run.backward_calls == 0
    assert focus_run.autograd_grad_calls == 0
    assert focus_run.observed_parent_module_forward_calls == 0
    assert focus_run.observed_parent_root_forward_calls == 0
    assert fingerprint_model_state(parent) == parent_state
