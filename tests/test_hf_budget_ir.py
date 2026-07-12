from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_arch_layout import build_decoder_layout  # noqa: E402
from neural_forge.hf_budget_ir import (  # noqa: E402
    HFBudgetIRError,
    HFBudgetUnattainableError,
    compile_hf_budget_ir,
    materialize_hf_budget_ir,
    plan_hf_budget_ir,
)
from neural_forge.hf_factorized_vocab import (  # noqa: E402
    FactorizedEmbedding,
    FactorizedLMHead,
    exact_parameter_report,
)
from neural_forge.hf_parameter_sharing import (  # noqa: E402
    parameter_sharing_report,
)


def _tiny_gpt2():
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=31,
            n_positions=16,
            n_embd=8,
            n_layer=3,
            n_head=2,
            n_inner=12,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
        )
    )


def _tiny_llama():
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=31,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
        )
    )


@pytest.fixture(params=[_tiny_gpt2, _tiny_llama], ids=["gpt2", "llama"])
def tiny_model(request):
    torch.manual_seed(41)
    return request.param()


def _minimum_budget(model) -> int:
    source_unique = int(exact_parameter_report(model)["unique_parameter_count"])
    probe = plan_hf_budget_ir(model, max_unique_parameters=source_unique)
    return int(probe.diagnostics["minimum_representable_unique_parameters"])


def test_plan_is_deterministic_auditable_and_exact(tiny_model) -> None:
    source = exact_parameter_report(tiny_model)
    budget = _minimum_budget(tiny_model)

    first = plan_hf_budget_ir(tiny_model, exact_unique_parameters=budget)
    second = plan_hf_budget_ir(tiny_model, exact_budget=budget)

    assert first == second
    assert first.exact_unique_parameters == budget
    assert first.budget_headroom == 0
    assert first.logical_dense_parameters == source["logical_parameter_count"]
    assert first.source_unique_parameters == source["unique_parameter_count"]
    assert first.execution_depth == 3
    assert first.mlp_groups == ((0, 1, 2),)
    assert first.shared_groups == ((0, 1, 2),)
    assert first.embedding_rank >= 1
    assert first.lm_head_rank >= 1
    assert (
        first.fixed_unique_parameters
        + first.vocabulary_unique_parameters
        + first.mlp_unique_parameters
        == budget
    )
    report = first.as_dict()
    assert report["factor_ranks"] == {
        "embedding": first.embedding_rank,
        "lm_head": first.lm_head_rank,
    }
    assert report["all_logical_layers_executed"] is True
    assert report["diagnostics"]["selected_exactly_matches_budget"] is True


def test_compile_executes_every_layer_and_materializes_dense_independent_mlps(
    tiny_model,
) -> None:
    tiny_model.eval()
    source_unique = int(exact_parameter_report(tiny_model)["unique_parameter_count"])
    budget = _minimum_budget(tiny_model)
    compiled = compile_hf_budget_ir(tiny_model, exact_budget=budget)

    assert compiled.exact_unique_parameters == budget
    assert isinstance(tiny_model.get_input_embeddings(), FactorizedEmbedding)
    assert isinstance(tiny_model.get_output_embeddings(), FactorizedLMHead)
    shared = parameter_sharing_report(compiled.mlp_modules, compiled.mlp_layout)
    assert shared.execution_depth == 3
    assert shared.fully_shared is True
    assert shared.unique_parameter_sets == 1

    calls = [0, 0, 0]
    handles = []
    for index, module in enumerate(compiled.mlp_modules):
        def count_call(_module, _inputs, _output, *, layer=index):
            calls[layer] += 1

        handles.append(module.register_forward_hook(count_call))
    input_ids = torch.tensor([[1, 5, 7, 2]], dtype=torch.long)
    with torch.no_grad():
        compiled_logits = tiny_model(input_ids).logits
    for handle in handles:
        handle.remove()
    assert calls == [1, 1, 1]

    materialized = materialize_hf_budget_ir(compiled)
    with torch.no_grad():
        dense_logits = materialized(input_ids).logits
    torch.testing.assert_close(dense_logits, compiled_logits, rtol=1e-5, atol=1e-6)

    assert isinstance(materialized.get_input_embeddings(), torch.nn.Embedding)
    assert isinstance(materialized.get_output_embeddings(), torch.nn.Linear)
    expanded = parameter_sharing_report(compiled.mlp_modules, compiled.mlp_layout)
    assert expanded.fully_expanded is True
    assert expanded.unique_parameter_sets == 3
    assert int(exact_parameter_report(materialized)["unique_parameter_count"]) == (
        source_unique
    )
    assert build_decoder_layout(materialized).layer_count == 3


def test_unrepresentable_exact_and_too_small_max_budgets_fail_closed(
    tiny_model,
) -> None:
    source_before = exact_parameter_report(tiny_model)
    probe = plan_hf_budget_ir(
        tiny_model,
        max_unique_parameters=int(source_before["unique_parameter_count"]),
    )
    minimum = int(probe.diagnostics["minimum_representable_unique_parameters"])
    maximum = int(probe.diagnostics["maximum_representable_unique_parameters"])

    with pytest.raises(HFBudgetUnattainableError) as exact_error:
        plan_hf_budget_ir(tiny_model, exact_unique_parameters=maximum + 1)
    assert exact_error.value.diagnostics["reason"] == (
        "exact_budget_not_representable"
    )
    assert exact_error.value.diagnostics["exact_budget_representable"] is False
    assert exact_error.value.diagnostics["nearest_representable_above"] is None
    assert exact_error.value.diagnostics["nearest_representable_below"][
        "unique_parameters"
    ] == maximum

    with pytest.raises(HFBudgetUnattainableError) as max_error:
        compile_hf_budget_ir(tiny_model, max_unique_parameters=minimum - 1)
    assert max_error.value.diagnostics["reason"] == (
        "no_plan_at_or_below_max_budget"
    )
    assert exact_parameter_report(tiny_model) == source_before
    assert isinstance(tiny_model.get_input_embeddings(), torch.nn.Embedding)


def test_budget_selector_rejects_ambiguity_and_non_integer_values(tiny_model) -> None:
    with pytest.raises(HFBudgetIRError, match="exactly one"):
        plan_hf_budget_ir(tiny_model)
    with pytest.raises(HFBudgetIRError, match="exactly one"):
        plan_hf_budget_ir(
            tiny_model,
            exact_unique_parameters=100,
            max_unique_parameters=200,
        )
    with pytest.raises(HFBudgetIRError, match="positive integer"):
        plan_hf_budget_ir(tiny_model, max_unique_parameters=True)
