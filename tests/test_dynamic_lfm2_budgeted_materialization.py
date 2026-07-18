from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import DenseParameterBudgetRequest
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (
    DENSE_BUDGET_MODEL_METHOD,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tokenizer_fingerprint() -> str:
    return canonical_sha256(
        {
            "kind": "test-local-tokenizer",
            "vocabulary_size": 64,
        }
    )


def _parent(seed: int):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=[
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
        ],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _adapter() -> Lfm2TargetSynthesisAdapter:
    # The terminal topology remains available for legacy progressive requests;
    # budgeted requests must use their own catalog-selected topology instead.
    return Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=canonical_sha256({"kind": "sealed-test-cache"}),
        device="cpu",
        dtype=torch.float32,
    )


@pytest.mark.parametrize("compression_ratio", (2, 4, 10))
def test_automatic_budget_plan_materializes_exact_dense_lfm2_blank(
    compression_ratio: int,
) -> None:
    parent = _parent(100 + compression_ratio)
    parent_state = fingerprint_model_state(parent)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    adapter = _adapter()
    adapter_registry = DynamicTargetAdapterRegistry((adapter,))
    system = AutomaticDynamicFoldingSystem(
        target_adapters=adapter_registry,
        stack_adapters=DynamicFoldStackAdapterRegistry(),
    )
    plan = system.plan_budgeted_target(
        parent_graph=parent_graph,
        budget_request=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=compression_ratio,
            maximum_candidates=3,
        ),
        child_name=f"automatic-{compression_ratio}x-blank",
        selected_candidate_rank=0,
        tokenizer_fingerprint=tokenizer,
        adapter_id=adapter.adapter_id,
        blank_seed=200 + compression_ratio,
    )

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=adapter_registry,
    ).synthesize_planned(
        plan=plan,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    candidate = plan.frontier.candidates[0]
    selected = result.adapter_payload
    assert isinstance(selected, Lfm2DenseBudgetedSelectedTarget)
    assert result.model is not parent
    assert type(result.model) is type(parent)
    assert fingerprint_model_state(parent) == parent_state
    assert result.graph.unique_parameter_count == candidate.unique_parameters
    assert result.graph.logical_parameter_count == candidate.logical_parameters
    assert result.resource_profile.unique_parameters == candidate.unique_parameters
    assert result.resource_profile.unique_parameters <= (
        parent_graph.unique_parameter_count // compression_ratio
    )
    assert result.graph.unique_parameter_count < parent_graph.unique_parameter_count
    assert len(result.graph.layer_indices) == len(candidate.ordered_source_spans)
    assert tuple(result.model.config.layer_types) == selected.layer_types
    assert int(result.model.config.hidden_size) == selected.hidden_size
    assert int(result.model.config.intermediate_size) == (
        selected.intermediate_size
    )
    assert int(result.model.config.num_attention_heads) == selected.query_heads
    assert int(result.model.config.num_key_value_heads) == (
        selected.key_value_heads
    )
    assert int(result.model.config.head_dim) == selected.head_dim
    assert result.model.get_input_embeddings().weight is (
        result.model.get_output_embeddings().weight
    )
    assert all(
        parameter.layout == torch.strided
        and not parameter.is_sparse
        and int(torch.count_nonzero(parameter).item()) == 0
        for parameter in result.model.parameters()
    )
    report = dict(result.adapter_report)
    assert report["dense_budget_catalog"] == plan.frontier.catalog_id
    assert report["dense_parameters"] is True
    assert report["quantization_used"] is False
    assert report["pruning_used"] is False
    assert report["training_used"] is False
    assert result.blank_provenance.compiler_created_blank is True
    assert result.blank_provenance.all_parameters_zero is True
    assert result.blank_provenance.dense_materialized_parameters is True


def test_planned_materialization_rejects_a_different_source_graph() -> None:
    tokenizer = _tokenizer_fingerprint()
    planned_parent = _parent(301)
    parent_graph = inspect_model_circuits(
        planned_parent,
        model_name="planned-parent",
        tokenizer_fingerprint=tokenizer,
    )
    adapter = _adapter()
    registry = DynamicTargetAdapterRegistry((adapter,))
    plan = AutomaticDynamicFoldingSystem(
        target_adapters=registry,
        stack_adapters=DynamicFoldStackAdapterRegistry(),
    ).plan_budgeted_target(
        parent_graph=parent_graph,
        budget_request=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
        ),
        child_name="source-bound-blank",
        selected_candidate_rank=0,
        tokenizer_fingerprint=tokenizer,
        adapter_id=adapter.adapter_id,
    )
    different_parent = _parent(302)

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="selected source parent differs",
    ):
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=registry,
        ).synthesize_planned(
            plan=plan,
            parent_models={"different-parent": different_parent},
            parent_tokenizer_fingerprints={"different-parent": tokenizer},
        )


def test_dense_budget_target_adapter_accepts_pinned_production_method() -> None:
    parent = _parent(401)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    adapter = _adapter()
    plan = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry(),
    ).plan_budgeted_target(
        parent_graph=parent_graph,
        budget_request=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
        ),
        child_name="production-method-blank",
        selected_candidate_rank=0,
        tokenizer_fingerprint=tokenizer,
        adapter_id=adapter.adapter_id,
    )

    pinned_request = replace(
        plan.target_request,
        fold_method_id=DENSE_BUDGET_MODEL_METHOD,
    )
    assert adapter.supports(pinned_request, {"parent": parent_graph})

    synthesized = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=pinned_request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert synthesized.request.fold_method_id == DENSE_BUDGET_MODEL_METHOD
    assert synthesized.graph.unique_parameter_count == (
        pinned_request.exact_unique_parameters
    )
