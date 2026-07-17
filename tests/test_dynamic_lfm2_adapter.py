from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import FoldDecision, fingerprint_model_state
from neural_forge.dynamic_fold_registry import (
    ClosureLevel,
    ExperimentLedger,
    LedgerRecord,
    RecordStatus,
)
from neural_forge.dynamic_lfm2_adapter import (
    DynamicLfm2AdapterError,
    Lfm2PreparedEvidence,
    Lfm2SelectedTarget,
    Lfm2TargetSynthesisAdapter,
    assert_lfm2_fold_only_runtime,
    build_lfm2_dynamic_compiler_bridge,
    derive_lfm2_target_ladder,
    execute_selected_lfm2_target,
    materialize_and_execute_selected_lfm2_target,
    materialize_hybrid_lfm2_child,
    materialize_selected_lfm2_child,
    select_first_open_lfm2_target,
    select_exact_lfm2_target,
    select_lfm2_hybrid_target,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_lfm2_closed_loop_fold import (
    Lfm2AttentionFoldEvidence,
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2MLPFoldEvidence,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(seed: int, intermediate_size: int):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["full_attention"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    return transformers.Lfm2ForCausalLM(config)


def _evidence(seed: int = 100) -> Lfm2ClosedLoopFoldEvidence:
    generator = torch.Generator().manual_seed(seed)
    hidden_size = 8
    source_query_heads = 4
    source_key_value_heads = 2
    head_dim = 2
    source_neurons = 6
    target_neurons = 5
    selection_rows = 24
    fit_rows = 32

    query = torch.randn(
        selection_rows,
        source_query_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    key = torch.randn(
        selection_rows,
        source_key_value_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    value = torch.randn(
        selection_rows,
        source_key_value_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    context = torch.randn(
        selection_rows,
        source_query_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    attention_contribution = torch.randn(
        selection_rows,
        source_query_heads,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    child_context = torch.randn(
        fit_rows,
        source_query_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    desired_output = torch.randn(
        hidden_size,
        source_query_heads * head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    attention_residual = torch.randn(
        fit_rows,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )

    gate = torch.randn(
        selection_rows,
        source_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    up = torch.randn(
        selection_rows,
        source_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    post = torch.nn.functional.silu(gate) * up
    mlp_contribution = torch.randn(
        selection_rows,
        source_neurons,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    child_post = torch.randn(
        fit_rows,
        target_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    desired_down = torch.randn(
        hidden_size,
        target_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    mlp_residual = torch.randn(
        fit_rows,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    return Lfm2ClosedLoopFoldEvidence(
        attention=Lfm2AttentionFoldEvidence(
            parent_query_rows=query,
            parent_key_rows=key,
            parent_value_rows=value,
            parent_context_rows=context,
            parent_contribution_rows=attention_contribution,
            child_context_rows=child_context,
            parent_branch_output_rows=(
                attention_residual
                + child_context.reshape(fit_rows, -1) @ desired_output.T
            ),
            child_pre_branch_residual_rows=attention_residual,
        ),
        mlp=Lfm2MLPFoldEvidence(
            parent_gate_rows=gate,
            parent_up_rows=up,
            parent_post_swiglu_rows=post,
            parent_contribution_rows=mlp_contribution,
            child_post_swiglu_rows=child_post,
            parent_branch_output_rows=mlp_residual + child_post @ desired_down.T,
            child_pre_branch_residual_rows=mlp_residual,
        ),
    )


class _FinalAccept:
    def evaluate(self, _context):
        return FoldDecision.accept("external whole-output gate accepted in test")


class _FinalReject:
    def evaluate(self, _context):
        return FoldDecision.reject("external whole-output gate rejected in test")


class _RejectLfm:
    def evaluate(self, context):
        if context.action.method_id == "lfm2_closed_loop_layer":
            return FoldDecision.reject("intentional adapter rollback test")
        return FoldDecision.accept("exact-copy control accepted")


def _cache():
    payload = {"cached_parent_rows": True}
    return payload, canonical_sha256(payload)


def _tokenizer_fingerprint():
    return canonical_sha256({"tokenizer": "shared-local-test"})


def _bridge(evidence, selected_target, *, final_evaluator):
    parent_cache_payload, parent_cache_fingerprint = _cache()

    def builder(request):
        assert request.parent_cache_payload == parent_cache_payload
        return Lfm2PreparedEvidence(
            evidence=evidence,
            child_forward_count=0,
        )

    return build_lfm2_dynamic_compiler_bridge(
        parent_cache_fingerprint=parent_cache_fingerprint,
        parent_cache_payload=parent_cache_payload,
        evidence_builder=builder,
        selected_target=selected_target,
        final_evaluator=final_evaluator,
        fold_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=0.0,
            max_endpoint_relative_rmse=1e-9,
            min_endpoint_cosine_similarity=0.999999999,
        ),
    )


def test_target_adapter_derives_ladder_and_skips_closed_exact_budget() -> None:
    parent = _model(1, 6)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    evidence_fingerprint = canonical_sha256({"cache": "once"})

    ladder = derive_lfm2_target_ladder(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 3),
    )
    first = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 3),
        evidence_fingerprint=evidence_fingerprint,
    )

    assert ladder.source == Lfm2Topology(4, 2, 6)
    assert [stage.topology.intermediate_size for stage in ladder.stages] == [
        5,
        4,
        3,
    ]
    assert first.stage == ladder.stages[0]
    assert first.exact_target_unique_parameters == graph.unique_parameter_count - 24
    assert (
        dict(first.signature.configuration)["exact_target_unique_parameters"]
        == first.exact_target_unique_parameters
    )

    record = LedgerRecord(
        signature=first.signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.REJECTED,
        reason_class="behavior",
        reason="The first exact budget was already tested.",
    )
    second = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 3),
        evidence_fingerprint=evidence_fingerprint,
        ledger=ExperimentLedger((record,)),
    )

    assert second.stage == ladder.stages[1]
    assert second.exact_target_unique_parameters == graph.unique_parameter_count - 48


def test_materializer_builds_exact_local_child_without_a_caller_shell() -> None:
    parent = _model(2, 6)
    tokenizer_fingerprint = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=_cache()[1],
    )

    first = materialize_selected_lfm2_child(
        parent,
        selected,
        child_name="child-a",
        seed=90,
    )
    second = materialize_selected_lfm2_child(
        parent,
        selected,
        child_name="child-b",
        seed=91,
    )

    assert first.exact_unique_parameters == selected.exact_target_unique_parameters
    assert second.exact_unique_parameters == selected.exact_target_unique_parameters
    assert first.blank_state_fingerprint != second.blank_state_fingerprint
    assert first.as_dict()["hub_loader_used"] is False


def test_hybrid_materializer_preserves_parent_head_dim_when_heads_shrink() -> None:
    torch.manual_seed(200)
    parent_config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    parent = transformers.Lfm2ForCausalLM(parent_config)
    tokenizer = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    # Exact count for: tied 32x8 vocabulary, final norm, one 8-wide
    # convolution block, one 2Q/1KV attention block at the parent's 2D head
    # width, and 5-neuron SwiGLU branches in both blocks.
    exact_target_parameters = 916
    selected = select_lfm2_hybrid_target(
        parent,
        graph,
        target_layer_types=("conv", "full_attention"),
        ordered_source_spans=((0, 1), (2,)),
        topology=Lfm2Topology(2, 1, 5),
        exact_target_unique_parameters=exact_target_parameters,
        evidence_fingerprint=_cache()[1],
    )

    child = materialize_hybrid_lfm2_child(
        parent,
        selected,
        child_name="hybrid-child",
        seed=201,
    )

    assert child.exact_unique_parameters == exact_target_parameters
    assert child.model.config.head_dim == 2
    attention = child.model.model.layers[1].self_attn
    assert attention.head_dim == 2
    assert attention.q_proj.weight.shape == (4, 8)


def test_exact_lfm_target_selection_uses_generic_budget_solver() -> None:
    parent = _model(3, 6)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    exact_budget = graph.unique_parameter_count - 48

    selected = select_exact_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 3),
        exact_target_unique_parameters=exact_budget,
        evidence_fingerprint=_cache()[1],
    )

    assert selected.exact_target_unique_parameters == exact_budget
    assert selected.stage.topology.intermediate_size == 4


def test_generic_target_registry_materializes_real_exact_lfm_child() -> None:
    parent = _model(4, 6)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    exact_budget = parent_graph.unique_parameter_count - 48
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(4, 2, 3),
        evidence_fingerprint=_cache()[1],
    )

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="automatic-lfm-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=exact_budget,
            tokenizer_fingerprint=tokenizer,
            blank_seed=5,
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.adapter_id == adapter.adapter_id
    assert result.graph.unique_parameter_count == exact_budget
    assert isinstance(result.adapter_payload, Lfm2SelectedTarget)
    assert result.adapter_payload.stage.topology.intermediate_size == 4
    assert result.as_dict()["pre_fold_state_independently_fingerprinted"] is True
    assert result.blank_state_fingerprint == fingerprint_model_state(result.model)


def test_generic_compiler_executes_lfm_bridge_with_cached_parent_and_seed_erasure() -> (
    None
):
    parent = _model(10, 6)
    parent_before = fingerprint_model_state(parent)
    evidence = _evidence()

    def compile_child(seed: int):
        child = _model(seed, 5)
        tokenizer_fingerprint = _tokenizer_fingerprint()
        graph = inspect_model_circuits(
            parent,
            model_name="parent",
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        ledger = ExperimentLedger()
        _payload, cache_fingerprint = _cache()
        selected = select_first_open_lfm2_target(
            parent,
            graph,
            terminal_topology=Lfm2Topology(4, 2, 5),
            evidence_fingerprint=cache_fingerprint,
            ledger=ledger,
        )
        bridge = _bridge(
            evidence,
            selected,
            final_evaluator=_FinalAccept(),
        )
        target_run = execute_selected_lfm2_target(
            parent_model=parent,
            parent_name="parent",
            child_model=child,
            child_name="child",
            bridge=bridge,
            ledger=ledger,
            parent_tokenizer_fingerprint=tokenizer_fingerprint,
            child_tokenizer_fingerprint=tokenizer_fingerprint,
        )
        return child, bridge, target_run, ledger

    child_a, bridge_a, target_run_a, ledger_a = compile_child(20)
    child_b, bridge_b, target_run_b, _ledger_b = compile_child(30)
    result_a = target_run_a.compilation
    result_b = target_run_b.compilation

    assert result_a.complete
    assert result_b.complete
    assert any(
        action.method_id == "lfm2_closed_loop_layer" for action in result_a.plan.actions
    )
    assert bridge_a.evidence_provider.build_count == 1
    assert bridge_b.evidence_provider.build_count == 1
    assert bridge_a.as_dict()["parent_trace_cache_reused"] is True
    assert bridge_a.as_dict()["requires_external_semantic_final_gate"] is False
    assert target_run_a.selection_record.status == RecordStatus.ACCEPTED.value
    assert ledger_a.decision(bridge_a.selected_target.signature).allowed is False
    lfm_attempt = next(
        attempt
        for attempt in result_a.run.attempts
        if attempt.method_id == "lfm2_closed_loop_layer"
    )
    assert lfm_attempt.transaction_report["parent_traces_regenerated"] is False
    assert lfm_attempt.transaction_report["zero_gradient"] is True
    assert fingerprint_model_state(child_a) == fingerprint_model_state(child_b)
    assert fingerprint_model_state(parent) == parent_before
    assert assert_lfm2_fold_only_runtime(child_a)["gradient_tensors_absent"] is True
    assert all(parameter.grad is None for parameter in child_a.parameters())
    assert result_a.run.as_dict()["optimizer_steps"] == 0
    assert result_a.run.as_dict()["backward_calls"] == 0


def test_high_level_materialize_and_execute_path_needs_no_manual_child() -> None:
    parent = _model(31, 6)
    tokenizer_fingerprint = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    ledger = ExperimentLedger()
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=_cache()[1],
        ledger=ledger,
    )
    bridge = _bridge(
        _evidence(310),
        selected,
        final_evaluator=_FinalAccept(),
    )

    materialized, target_run = materialize_and_execute_selected_lfm2_target(
        parent_model=parent,
        parent_name="parent",
        child_name="automatic-child",
        bridge=bridge,
        ledger=ledger,
        tokenizer_fingerprint=tokenizer_fingerprint,
        blank_seed=32,
    )

    assert target_run.complete
    assert (
        materialized.exact_unique_parameters == selected.exact_target_unique_parameters
    )
    assert target_run.selection_record.status == RecordStatus.ACCEPTED.value


def test_generic_executor_rejection_restores_lfm_layer_exactly() -> None:
    parent = _model(40, 6)
    child = _model(41, 5)
    layer_before = {
        name: value.detach().clone()
        for name, value in child.model.layers[0].state_dict().items()
    }
    tokenizer_fingerprint = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    ledger = ExperimentLedger()
    _payload, cache_fingerprint = _cache()
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=cache_fingerprint,
        ledger=ledger,
    )
    bridge = _bridge(
        _evidence(400),
        selected,
        final_evaluator=_FinalAccept(),
    )
    bridge = type(bridge)(
        operators=bridge.operators,
        evidence_provider=bridge.evidence_provider,
        initializer=bridge.initializer,
        evaluator=_RejectLfm(),
        final_evaluator=bridge.final_evaluator,
        selected_target=bridge.selected_target,
    )
    target_run = execute_selected_lfm2_target(
        parent_model=parent,
        parent_name="parent",
        child_model=child,
        child_name="child",
        bridge=bridge,
        ledger=ledger,
        parent_tokenizer_fingerprint=tokenizer_fingerprint,
        child_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    result = target_run.compilation

    assert result.run.status == "rejected"
    assert target_run.selection_record.status == RecordStatus.REJECTED.value
    rejected = result.run.attempts[-1]
    assert rejected.method_id == "lfm2_closed_loop_layer"
    assert rejected.rollback_exact is True
    for name, value in child.model.layers[0].state_dict().items():
        assert torch.equal(value, layer_before[name]), name
    assert all(parameter.grad is None for parameter in child.parameters())


def test_selected_stage_rejects_a_different_materialized_child_budget() -> None:
    parent = _model(50, 6)
    wrong_child = _model(51, 4)
    tokenizer_fingerprint = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    ledger = ExperimentLedger()
    _payload, cache_fingerprint = _cache()
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=cache_fingerprint,
        ledger=ledger,
    )
    bridge = _bridge(
        _evidence(500),
        selected,
        final_evaluator=_FinalAccept(),
    )
    before = fingerprint_model_state(wrong_child)

    with pytest.raises(DynamicLfm2AdapterError, match="selected LFM2 target"):
        execute_selected_lfm2_target(
            parent_model=parent,
            parent_name="parent",
            child_model=wrong_child,
            child_name="child",
            bridge=bridge,
            ledger=ledger,
            parent_tokenizer_fingerprint=tokenizer_fingerprint,
            child_tokenizer_fingerprint=tokenizer_fingerprint,
        )

    assert fingerprint_model_state(wrong_child) == before


def test_final_behavior_rejection_restores_child_and_evidence_state() -> None:
    parent = _model(60, 6)
    child = _model(61, 5)
    initial_child = fingerprint_model_state(child)
    tokenizer_fingerprint = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    ledger = ExperimentLedger()
    _payload, cache_fingerprint = _cache()
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=cache_fingerprint,
        ledger=ledger,
    )
    bridge = _bridge(
        _evidence(600),
        selected,
        final_evaluator=_FinalReject(),
    )

    target_run = execute_selected_lfm2_target(
        parent_model=parent,
        parent_name="parent",
        child_model=child,
        child_name="child",
        bridge=bridge,
        ledger=ledger,
        parent_tokenizer_fingerprint=tokenizer_fingerprint,
        child_tokenizer_fingerprint=tokenizer_fingerprint,
    )

    assert target_run.compilation.run.status == "final_gate_rejected"
    assert target_run.selection_record.status == RecordStatus.REJECTED.value
    assert fingerprint_model_state(child) == initial_child
    assert bridge.evidence_provider.restore_count == 1
    assert target_run.compilation.run.accepted_action_ids == ()


def test_selected_target_rejects_stale_parent_checkpoint_before_child_mutation() -> (
    None
):
    parent = _model(70, 6)
    child = _model(71, 5)
    child_before = fingerprint_model_state(child)
    tokenizer_fingerprint = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    ledger = ExperimentLedger()
    _payload, cache_fingerprint = _cache()
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=cache_fingerprint,
        ledger=ledger,
    )
    bridge = _bridge(
        _evidence(700),
        selected,
        final_evaluator=_FinalAccept(),
    )
    with torch.no_grad():
        next(parent.parameters()).add_(1.0)

    with pytest.raises(DynamicLfm2AdapterError, match="checkpoint differs"):
        execute_selected_lfm2_target(
            parent_model=parent,
            parent_name="parent",
            child_model=child,
            child_name="child",
            bridge=bridge,
            ledger=ledger,
            parent_tokenizer_fingerprint=tokenizer_fingerprint,
            child_tokenizer_fingerprint=tokenizer_fingerprint,
        )

    assert fingerprint_model_state(child) == child_before


def test_selected_target_dataclass_rejects_cross_stage_signature_forgery() -> None:
    parent = _model(80, 6)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 3),
        evidence_fingerprint=_cache()[1],
    )
    assert len(selected.plan.stages) > 1

    with pytest.raises(DynamicLfm2AdapterError, match="does not match its stage"):
        Lfm2SelectedTarget(
            plan=selected.plan,
            stage=selected.plan.stages[1],
            signature=selected.signature,
            ledger_decision=replace(selected.ledger_decision, allowed=True),
        )


def test_execution_rejects_forged_selected_source_identity() -> None:
    parent = _model(81, 6)
    child = _model(82, 5)
    child_before = fingerprint_model_state(child)
    tokenizer_fingerprint = _tokenizer_fingerprint()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    selected = select_first_open_lfm2_target(
        parent,
        graph,
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=_cache()[1],
    )
    forged = Lfm2SelectedTarget(
        plan=selected.plan,
        stage=selected.stage,
        signature=replace(
            selected.signature,
            source_unit_ids=("fake.source",),
        ),
        ledger_decision=selected.ledger_decision,
    )
    bridge = _bridge(
        _evidence(810),
        forged,
        final_evaluator=_FinalAccept(),
    )

    with pytest.raises(DynamicLfm2AdapterError, match="inspected parent source"):
        execute_selected_lfm2_target(
            parent_model=parent,
            parent_name="parent",
            child_model=child,
            child_name="child",
            bridge=bridge,
            ledger=ExperimentLedger(),
            parent_tokenizer_fingerprint=tokenizer_fingerprint,
            child_tokenizer_fingerprint=tokenizer_fingerprint,
        )

    assert fingerprint_model_state(child) == child_before
