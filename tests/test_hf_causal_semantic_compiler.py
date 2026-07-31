from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_fold_executor import (  # noqa: E402
    FoldExecutionContext,
    fingerprint_model_state,
)
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (  # noqa: E402
    CAP_ATTENTION_TOPOLOGY_PROJECTION,
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_FFN_WIDTH_PROJECTION,
    CAP_RESIDUAL_WIDTH_PROJECTION,
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY,
    HfLlamaDenseTargetTopologyCatalog,
    HfLlamaDenseTopology,
    component_allocation_topology_fingerprint,
)
from neural_forge.dynamic_semantic_fold import (  # noqa: E402
    SemanticWholeModelCompileRequest,
    SemanticWholeModelFoldEvidence,
)
from neural_forge.dynamic_target_synthesis import (  # noqa: E402
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from neural_forge.hf_causal_blank_target_adapter import (  # noqa: E402
    HFCausalBlankTargetSynthesisAdapter,
)
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    HFCausalCircuitEvidence,
    HFCausalCircuitEvidenceError,
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_causal_semantic_compiler import (  # noqa: E402
    HF_CAUSAL_SAME_DEPTH_SEMANTIC_METHOD,
    HFCausalSameDepthSemanticCompiler,
    HFCausalSemanticCompilerError,
)


TOKENIZER = "a" * 64
CAPABILITIES = (
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_RESIDUAL_WIDTH_PROJECTION,
    CAP_FFN_WIDTH_PROJECTION,
    CAP_ATTENTION_TOPOLOGY_PROJECTION,
)


def _parent(seed: int = 9):
    torch.manual_seed(seed)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=32,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            pad_token_id=0,
        )
    )
    model.eval()
    return model


def _target_topology() -> HfLlamaDenseTopology:
    return HfLlamaDenseTopology(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )


def _catalog() -> HfLlamaDenseTargetTopologyCatalog:
    return HfLlamaDenseTargetTopologyCatalog(
        minimum_hidden_size=16,
        hidden_size_multiple=8,
        minimum_intermediate_size=16,
        intermediate_size_multiple=8,
        minimum_head_dim=8,
        head_dim_multiple=8,
        compiler_capabilities=CAPABILITIES,
    )


def _plan(parent_graph, *, blank_seed: int):
    topology = _target_topology()
    return plan_budgeted_dynamic_target(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="llama",
            target_modalities=("text",),
            target_unique_parameters=topology.unique_parameters,
            target_vocabulary_size=topology.vocab_size,
            match_policy="exact",
            maximum_candidates=8,
        ),
        child_name=f"same-depth-child-{blank_seed}",
        tokenizer_fingerprint=TOKENIZER,
        adapter_id=HFCausalBlankTargetSynthesisAdapter.adapter_id,
        fold_method_id=HF_CAUSAL_SAME_DEPTH_SEMANTIC_METHOD,
        blank_seed=blank_seed,
        catalogs=DenseTargetTopologyCatalogRegistry((_catalog(),)),
    )


def _target(parent, parent_graph, *, blank_seed: int):
    plan = _plan(parent_graph, blank_seed=blank_seed)
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (HFCausalBlankTargetSynthesisAdapter(dtype=torch.float32),)
        ),
    ).synthesize(
        request=plan.target_request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": TOKENIZER},
    )
    return plan, target


def _compile_request(parent, parent_graph, target, evidence):
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=target.model,
        parent_graphs={"parent": parent_graph},
        child_graph=target.graph,
    )
    semantic_evidence = SemanticWholeModelFoldEvidence(
        root_fingerprint=evidence.cache_key,
        source=evidence,
    )
    return SemanticWholeModelCompileRequest(
        target=target,
        action=None,  # The compiler does not own planner orchestration.
        context=context,
        evidence=semantic_evidence,
    )


def _capture(parent, parent_graph):
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        dtype=torch.long,
    )
    return input_ids, capture_hf_causal_circuit_evidence(
        parent,
        {"input_ids": input_ids},
        parent_graph_fingerprint=parent_graph.fingerprint,
        metadata={"test": "high-ratio-same-depth"},
    )


def test_one_forward_full_circuit_evidence_is_sealed_and_round_trips(tmp_path) -> None:
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    forwards = 0

    def counted(_module, _args, _output):
        nonlocal forwards
        forwards += 1

    handle = parent.register_forward_hook(counted)
    try:
        _input_ids, evidence = _capture(parent, graph)
    finally:
        handle.remove()

    assert forwards == evidence.parent_forward_calls == 1
    assert len(evidence.layers) == 2
    assert evidence.fit_token_mask.logical_and(
        evidence.validation_token_mask
    ).sum().item() == 0
    reopened = HFCausalCircuitEvidence.open(evidence.save(tmp_path / "cache"))
    assert reopened.cache_key == evidence.cache_key
    assert reopened.tensor_manifest == evidence.tensor_manifest
    reopened.layers[0].block_input.add_(1.0)
    with pytest.raises(HFCausalCircuitEvidenceError, match="changed after sealing"):
        reopened.verify()


def test_high_ratio_full_model_fold_is_complete_fold_only_and_seed_invariant() -> None:
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    input_ids, evidence = _capture(parent, graph)
    parent_forwards_after_capture = 0

    def counted(_module, _args, _output):
        nonlocal parent_forwards_after_capture
        parent_forwards_after_capture += 1

    handle = parent.register_forward_hook(counted)
    folded_states = []
    try:
        for seed in (1, 987654):
            plan, target = _target(parent, graph, blank_seed=seed)
            assert plan.target_certificate.planned_graph.ordered_source_spans == (
                (0,),
                (1,),
            )
            ratio = graph.unique_parameter_count / target.graph.unique_parameter_count
            assert ratio >= 10.0
            blank = target.blank_state_fingerprint
            result = HFCausalSameDepthSemanticCompiler().compile(
                _compile_request(parent, graph, target, evidence)
            )
            assert set(result.covered_target_parameter_paths) == set(
                target.graph.parameter_paths
            )
            assert len(result.component_receipts) == 2
            assert result.whole_model_receipt[
                "every_target_parameter_installed_exactly_once"
            ] is True
            assert result.whole_model_receipt["parameter_compaction_ratio"] >= 10.0
            assert result.whole_model_receipt["source_parameter_count"] == (
                graph.unique_parameter_count
            )
            assert result.whole_model_receipt["target_parameter_count"] == (
                target.graph.unique_parameter_count
            )
            assert target.graph.unique_parameter_count == (
                _target_topology().unique_parameters
            )
            packet = result.whole_model_receipt["behavior_review_packet"]
            assert packet["valid_token_count"] == int(evidence.valid_token_mask.sum())
            assert packet["manual_review_required"] is True
            assert packet["intelligence_claim"] is False
            assert fingerprint_model_state(target.model) != blank
            for name in (
                "training_steps",
                "optimizer_constructions",
                "optimizer_steps",
                "backward_calls",
                "autograd_grad_calls",
                "pruning_operations",
                "quantization_operations",
                "parent_forward_calls",
                "parent_trace_recaptures",
            ):
                assert result.operation_audit[name] == 0
            assert result.operation_audit["cached_parent_evidence_only"] is True
            folded_states.append(
                {
                    name: value.detach().clone()
                    for name, value in target.model.state_dict().items()
                }
            )
            with torch.no_grad():
                logits = target.model(input_ids=input_ids, use_cache=False).logits
            assert logits.shape == (2, 6, 64)
            assert torch.isfinite(logits).all()
    finally:
        handle.remove()

    assert parent_forwards_after_capture == 0
    assert folded_states[0].keys() == folded_states[1].keys()
    assert all(
        torch.equal(folded_states[0][name], folded_states[1][name])
        for name in folded_states[0]
    )


def _resealed_depth_span_tamper(target, spans: str):
    """Re-authenticate a tampered span configuration the way synthesis does.

    The compiler's own ordered-depth guard can only be reached through a
    request whose component-allocation receipt genuinely matches its
    materializable topology, so the tampered configuration is re-sealed with a
    receipt bound to the new topology fingerprint.
    """

    configuration = dict(target.request.configuration)
    configuration[HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY] = spans
    topology_sha256 = component_allocation_topology_fingerprint(configuration)
    receipt = replace(
        target.request.component_allocation_receipt,
        topology_configuration_sha256=topology_sha256,
    )
    configuration[COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY] = topology_sha256
    configuration[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY] = receipt.fingerprint
    return replace(
        target,
        request=replace(
            target.request,
            configuration=tuple(configuration.items()),
            component_allocation_receipt=receipt,
        ),
    )


def test_compiler_fails_closed_before_mutation_on_non_singleton_depth_span() -> None:
    parent = _parent(seed=11)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    _input_ids, evidence = _capture(parent, graph)
    _plan_value, target = _target(parent, graph, blank_seed=3)
    bad_target = _resealed_depth_span_tamper(target, "0,1;0")
    before = fingerprint_model_state(target.model)

    assert dict(bad_target.request.configuration)[
        HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY
    ] == "0,1;0"
    with pytest.raises(
        HFCausalSemanticCompilerError,
        match="Ordered depth composition",
    ):
        HFCausalSameDepthSemanticCompiler().compile(
            _compile_request(parent, graph, bad_target, evidence)
        )

    assert fingerprint_model_state(target.model) == before


def test_unsealed_depth_span_tamper_fails_component_receipt_authentication() -> None:
    parent = _parent(seed=13)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    _plan_value, target = _target(parent, graph, blank_seed=5)
    configuration = dict(target.request.configuration)
    configuration[HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY] = "0,1;0"
    before = fingerprint_model_state(target.model)

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="component-allocation receipt",
    ):
        replace(target.request, configuration=tuple(configuration.items()))

    assert fingerprint_model_state(target.model) == before


def test_compiler_fails_closed_before_mutation_on_qk_norm_crossing() -> None:
    parent = _parent(seed=12)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    _input_ids, evidence = _capture(parent, graph)
    crossed = HFCausalCircuitEvidence(
        parent_graph_fingerprint=evidence.parent_graph_fingerprint,
        architecture=replace(evidence.architecture, query_key_norm="rms_norm"),
        input_ids=evidence.input_ids,
        attention_mask=evidence.attention_mask,
        position_ids=evidence.position_ids,
        valid_token_mask=evidence.valid_token_mask,
        fit_token_mask=evidence.fit_token_mask,
        validation_token_mask=evidence.validation_token_mask,
        token_embedding_weight=evidence.token_embedding_weight,
        lm_head_weight=evidence.lm_head_weight,
        layers=evidence.layers,
        final_norm_output=evidence.final_norm_output,
        logits=evidence.logits,
        metadata=evidence.metadata,
    )
    _plan_value, target = _target(parent, graph, blank_seed=4)
    before = fingerprint_model_state(target.model)

    with pytest.raises(
        HFCausalSemanticCompilerError,
        match="Q/K normalization translation",
    ):
        HFCausalSameDepthSemanticCompiler().compile(
            _compile_request(parent, graph, target, crossed)
        )

    assert fingerprint_model_state(target.model) == before
