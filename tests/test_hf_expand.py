from __future__ import annotations

import pytest

from neural_forge.hf_arch_layout import build_decoder_layout
from neural_forge.hf_expand import (
    ExpansionTopologyError,
    NoExactExpansionPlanError,
    count_unique_parameters,
    expand_hf_model,
    plan_hf_expansion,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_llama():
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=48,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _tiny_gpt2():
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=48,
            n_positions=32,
            n_embd=12,
            n_layer=2,
            n_head=3,
            n_inner=20,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _inputs():
    return {
        "input_ids": torch.tensor([[1, 3, 5, 7], [1, 4, 6, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }


def test_llama_gated_net2wider_is_exact_and_deterministic() -> None:
    torch.manual_seed(101)
    model = _tiny_llama()
    source_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    source_count = count_unique_parameters(model)
    per_channel = 3 * model.config.hidden_size
    target = source_count + model.config.num_hidden_layers * per_channel

    first_plan = plan_hf_expansion(model, target)
    second_plan = plan_hf_expansion(model, target)
    assert first_plan == second_plan
    assert first_plan.added_channels == 1
    assert first_plan.added_layers == 0

    result = expand_hf_model(
        model,
        target,
        validation_inputs=_inputs(),
        max_equivalence_error=2e-6,
    )

    assert count_unique_parameters(result.model) == target
    assert result.report.actual_parameter_delta == 2 * per_channel
    assert result.report.width_parameter_delta == 2 * per_channel
    assert result.report.depth_parameter_delta == 0
    assert result.report.function_equivalence_max_error is not None
    assert result.report.function_equivalence_max_error <= 2e-6
    assert result.report.source_coverage.replicated_source_channels == (0,)
    assert result.model.config.intermediate_size == 21
    assert result.model.model.layers[0].mlp.gate_proj.weight.shape == (21, 12)
    assert result.model.model.layers[0].mlp.down_proj.weight.shape == (12, 21)
    repeated = expand_hf_model(model, target)
    for name, value in result.model.state_dict().items():
        assert torch.equal(value, repeated.model.state_dict()[name])
    for name, value in model.state_dict().items():
        assert torch.equal(value, source_state[name])


@pytest.mark.parametrize("factory", [_tiny_llama, _tiny_gpt2])
def test_identity_depth_expansion_preserves_logits_and_reports_provenance(factory) -> None:
    torch.manual_seed(202)
    model = factory()
    layout = build_decoder_layout(model)
    blocks = model.transformer.h if layout.family == "gpt2" else model.model.layers
    block_parameters = sum(parameter.numel() for parameter in blocks[0].parameters())
    target = count_unique_parameters(model) + block_parameters

    result = expand_hf_model(
        model,
        target,
        validation_inputs=_inputs(),
        max_equivalence_error=2e-6,
    )

    assert result.report.plan.added_layers == 1
    assert result.report.plan.added_channels == 0
    assert result.report.actual_target_parameter_count == target
    assert result.report.depth_parameter_delta == block_parameters
    assert result.report.function_equivalence_max_error is not None
    assert result.report.function_equivalence_max_error <= 2e-6
    assert len(result.report.inserted_layer_provenance) == 1
    provenance = result.report.inserted_layer_provenance[0]
    assert provenance.inserted_index == 2
    assert provenance.donor_index == 0
    assert len(provenance.zeroed_operator_paths) == 2
    assert all(
        torch.count_nonzero(dict(result.model.named_parameters())[name]).item() == 0
        for name in provenance.zeroed_parameter_names
    )
    expanded_layout = build_decoder_layout(result.model)
    assert expanded_layout.layer_count == 3
    assert len(blocks) == 2


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3Config"), reason="Qwen3 unavailable"
)
def test_qwen3_combined_width_and_depth_hits_exact_count() -> None:
    torch.manual_seed(303)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=48,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )
    source = count_unique_parameters(model)
    block = sum(parameter.numel() for parameter in model.model.layers[0].parameters())
    channel = 3 * model.config.hidden_size
    target = source + 2 * channel + block + channel

    plan = plan_hf_expansion(model, target)
    assert (plan.added_layers, plan.added_channels) == (1, 1)
    result = expand_hf_model(
        model,
        target,
        validation_inputs=_inputs(),
        max_equivalence_error=2e-6,
    )

    assert count_unique_parameters(result.model) == target
    assert result.model.config.num_hidden_layers == 3
    assert result.model.config.intermediate_size == 21
    assert result.model.config.layer_types == ["full_attention"] * 3
    assert result.model.model.layers[2].self_attn.layer_idx == 2
    assert result.report.function_equivalence_max_error is not None
    assert result.report.function_equivalence_max_error <= 2e-6


def test_dense_gpt2_net2wider_preserves_logits() -> None:
    torch.manual_seed(404)
    model = _tiny_gpt2()
    per_channel = 2 * model.config.n_embd + 1
    target = count_unique_parameters(model) + model.config.n_layer * per_channel

    result = expand_hf_model(
        model,
        target,
        validation_inputs=_inputs(),
        max_equivalence_error=2e-6,
    )

    assert result.report.plan.added_channels == 1
    assert result.model.config.n_inner == 21
    assert result.model.transformer.h[0].mlp.c_fc.weight.shape == (12, 21)
    assert result.model.transformer.h[0].mlp.c_proj.weight.shape == (21, 12)
    assert result.report.function_equivalence_max_error is not None
    assert result.report.function_equivalence_max_error <= 2e-6


@pytest.mark.parametrize("family", ["gemma2", "gemma3"])
def test_gemma_family_width_expansion_preserves_logits_and_schedule(family) -> None:
    required = (
        ("Gemma2Config", "Gemma2ForCausalLM")
        if family == "gemma2"
        else ("Gemma3TextConfig", "Gemma3ForCausalLM")
    )
    if not all(hasattr(transformers, name) for name in required):
        pytest.skip(f"{family} unavailable")
    values = {
        "vocab_size": 48,
        "hidden_size": 12,
        "intermediate_size": 20,
        "num_hidden_layers": 2 if family == "gemma2" else 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "sliding_window": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_word_embeddings": True,
    }
    if family == "gemma2":
        model = transformers.Gemma2ForCausalLM(
            transformers.Gemma2Config(**values)
        )
    else:
        model = transformers.Gemma3ForCausalLM(
            transformers.Gemma3TextConfig(**values)
        )
    original_schedule = tuple(model.config.layer_types)
    target = count_unique_parameters(model) + (
        model.config.num_hidden_layers * 3 * model.config.hidden_size
    )

    result = expand_hf_model(
        model,
        target,
        validation_inputs=_inputs(),
        max_equivalence_error=2e-6,
    )

    assert count_unique_parameters(result.model) == target
    assert result.model.config.intermediate_size == 21
    assert tuple(result.model.config.layer_types) == original_schedule
    assert result.report.function_equivalence_max_error is not None
    assert result.report.function_equivalence_max_error <= 2e-6


def test_unreachable_budget_and_invalid_topology_fail_closed() -> None:
    model = _tiny_llama()
    with pytest.raises(NoExactExpansionPlanError, match="unreachable"):
        plan_hf_expansion(model, count_unique_parameters(model) + 1)

    model.model.layers[1].mlp.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ExpansionTopologyError, match="equal unique parameter counts"):
        plan_hf_expansion(model, count_unique_parameters(model) + 1000)
