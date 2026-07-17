from __future__ import annotations

import copy

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_compiler import DynamicFoldingCompiler
from neural_forge.dynamic_fold_executor import (
    FoldDecision,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import plan_dynamic_fold
from neural_forge.dynamic_fold_registry import default_method_registry
from neural_forge.dynamic_lfm2_adapter import (
    HYBRID_MODEL_METHOD,
    Lfm2HybridSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    build_lfm2_dynamic_compiler_bridge,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    DynamicModelInspectionError,
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tokenizer_fingerprint() -> str:
    return canonical_sha256({"tokenizer": "hybrid-catalog-test"})


def _model(
    seed: int,
    *,
    layer_types: tuple[str, ...],
    intermediate_size: int,
):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=list(layer_types),
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    return transformers.Lfm2ForCausalLM(config).eval()


def _unique_parameters(model) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def _hybrid_target_count(parent) -> int:
    config = copy.deepcopy(parent.config)
    config.num_hidden_layers = 2
    config.layer_types = ["full_attention", "conv"]
    config.intermediate_size = 5
    config.block_ff_dim = 5
    config.block_auto_adjust_ff_dim = False
    return _unique_parameters(type(parent)(config))


def _synthesize_hybrid(parent):
    tokenizer = _tokenizer_fingerprint()
    exact_count = _hybrid_target_count(parent)
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=canonical_sha256({"cache": "hybrid"}),
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="hybrid-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=exact_count,
            tokenizer_fingerprint=tokenizer,
            adapter_id=adapter.adapter_id,
            fold_method_id=HYBRID_MODEL_METHOD,
            blank_seed=7,
            configuration=(
                ("source_parent", "parent"),
                ("target_depth", 2),
                ("target_layer_types", "full_attention,conv"),
                ("ordered_source_spans", "0,1;2"),
            ),
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    return target, adapter


def test_hybrid_target_materializes_and_plans_one_whole_model_action() -> None:
    parent = _model(
        1,
        layer_types=("conv", "full_attention", "conv"),
        intermediate_size=6,
    )
    target, _adapter = _synthesize_hybrid(parent)
    selected = target.adapter_payload

    assert isinstance(selected, Lfm2HybridSelectedTarget)
    assert selected.ordered_source_spans == ((0, 1), (2,))
    assert selected.target_layer_types == ("full_attention", "conv")
    assert target.resource_profile.layer_count == 2
    assert dict(target.graph.metadata)["fold_method_id"] == HYBRID_MODEL_METHOD

    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    plan = plan_dynamic_fold(
        (parent_graph,),
        target.graph,
        child_state_fingerprint=target.blank_state_fingerprint,
        evidence_fingerprint=selected.signature.evidence_fingerprint,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
        exact_target_unique_parameters=target.resource_profile.unique_parameters,
    )

    assert plan.ready
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.method_id == HYBRID_MODEL_METHOD
    assert dict(action.configuration)["ordered_source_spans"] == "0,1;2"
    assert set(action.target_node_ids) == {
        node.node_id for node in target.graph.nodes
    }
    assert default_method_registry().get(HYBRID_MODEL_METHOD).standalone is True


def test_hybrid_child_marker_bundle_is_structurally_fail_closed() -> None:
    child = _model(
        2,
        layer_types=("full_attention", "conv"),
        intermediate_size=5,
    )
    child.config._neural_forge_fold_method_id = HYBRID_MODEL_METHOD

    with pytest.raises(
        DynamicModelInspectionError,
        match="one complete bundle",
    ):
        inspect_model_circuits(
            child,
            model_name="tampered-child",
            tokenizer_fingerprint=_tokenizer_fingerprint(),
        )


def test_hybrid_compiler_markers_survive_local_config_round_trip() -> None:
    parent = _model(
        21,
        layer_types=("conv", "full_attention", "conv"),
        intermediate_size=6,
    )
    target, _adapter = _synthesize_hybrid(parent)
    reloaded_config = transformers.Lfm2Config.from_dict(
        target.model.config.to_dict()
    )
    reloaded = transformers.Lfm2ForCausalLM(reloaded_config).eval()
    graph = inspect_model_circuits(
        reloaded,
        model_name="reloaded-hybrid",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )

    metadata = dict(graph.metadata)
    assert metadata["fold_method_id"] == HYBRID_MODEL_METHOD
    assert metadata["target_layer_types"] == "full_attention,conv"
    assert metadata["ordered_source_spans"] == "0,1;2"


class _Accept:
    def evaluate(self, _context):
        return FoldDecision.accept("test final gate")


def test_hybrid_provider_binds_parent_only_builder_from_blank_without_mutation() -> None:
    parent = _model(
        3,
        layer_types=("conv", "full_attention", "conv"),
        intermediate_size=6,
    )
    target, _adapter = _synthesize_hybrid(parent)
    selected = target.adapter_payload
    cache_payload = {"cache": "hybrid"}
    cache_fingerprint = canonical_sha256(cache_payload)
    called = 0
    blank_before = fingerprint_model_state(target.model)

    def model_builder(request):
        nonlocal called
        called += 1
        assert request.child_state_fingerprint == blank_before
        assert fingerprint_model_state(request.child_model) == blank_before
        raise RuntimeError("parent-only hybrid builder reached")

    selected = type(selected)(
        topology=selected.topology,
        source_layer_types=selected.source_layer_types,
        target_layer_types=selected.target_layer_types,
        ordered_source_spans=selected.ordered_source_spans,
        exact_target_unique_parameters=selected.exact_target_unique_parameters,
        signature=selected.signature,
        ledger_decision=selected.ledger_decision,
    )
    bridge = build_lfm2_dynamic_compiler_bridge(
        parent_cache_fingerprint=cache_fingerprint,
        parent_cache_payload=cache_payload,
        evidence_builder=lambda _request: None,
        model_evidence_builder=model_builder,
        selected_target=selected,
        final_evaluator=_Accept(),
        blank_state_fingerprint=target.blank_state_fingerprint,
    )

    with pytest.raises(
        RuntimeError,
        match="parent-only hybrid builder reached",
    ):
        DynamicFoldingCompiler(operators=bridge.operators).compile(
            parent_models={"parent": parent},
            child_model=target.model,
            child_name="hybrid-child",
            evidence_provider=bridge.evidence_provider,
            evaluator=bridge.evaluator,
            final_evaluator=bridge.final_evaluator,
            evidence_fingerprint=cache_fingerprint,
            parent_tokenizer_fingerprints={
                "parent": _tokenizer_fingerprint()
            },
            child_tokenizer_fingerprint=_tokenizer_fingerprint(),
            exact_target_unique_parameters=target.resource_profile.unique_parameters,
            initializer=bridge.initializer,
        )

    assert called == 1
    assert fingerprint_model_state(target.model) == blank_before
