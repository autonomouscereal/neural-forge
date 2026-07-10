from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_blank_bootstrap import (  # noqa: E402
    BlankBootstrapError,
    bootstrap_blank_decoder_models,
)
from neural_forge.hf_transport_manifest import (  # noqa: E402
    build_decoder_transport_manifest,
)


def _gpt2_source():
    torch.manual_seed(101)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=12,
            n_layer=3,
            n_head=3,
            n_inner=20,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _llama_source():
    torch.manual_seed(202)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=3,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _llama_target(*, hidden_size: int = 8):
    torch.manual_seed(303)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=hidden_size,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _batch(seed: int = 404):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(3, 64, (8, 8), generator=generator)
    input_ids[:, 0] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@pytest.mark.parametrize("source_factory", [_gpt2_source, _llama_source])
def test_blank_bootstrap_improves_source_anchors_and_covers_every_target_role(
    source_factory,
) -> None:
    source = source_factory()
    target = _llama_target()
    calibration = _batch()
    validation = _batch(405)

    result = bootstrap_blank_decoder_models(
        source,
        target,
        calibration,
        validation_inputs=validation,
    )

    assert result.transaction.accepted is True
    assert result.transaction.report["coverage"]["complete"] is True
    manifest = build_decoder_transport_manifest(target)
    assert result.transaction.report["coverage"]["evidenced_role_count"] == len(
        manifest.plan.required_roles
    )
    assert result.donor_layer_spans == ((0,), (1, 2))
    assert result.candidate_gate_metrics.final_state.relative_rmse <= (
        result.baseline_gate_metrics.final_state.relative_rmse
    )
    assert result.candidate_gate_metrics.logits.relative_rmse <= (
        result.baseline_gate_metrics.logits.relative_rmse
    )
    assert result.candidate_gate_metrics.top1_agreement >= (
        result.baseline_gate_metrics.top1_agreement
    )
    assert all(torch.isfinite(parameter).all() for parameter in target.parameters())
    assert all(parameter.grad is None for parameter in source.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())


def test_rejected_bootstrap_restores_the_exact_blank_target() -> None:
    source = _gpt2_source()
    target = _llama_target()
    before = _state(target)
    batch = _batch()

    result = bootstrap_blank_decoder_models(
        source,
        target,
        batch,
        validation_inputs=batch,
        min_final_state_relative_improvement=10.0,
        min_logit_relative_improvement=10.0,
        min_top1_agreement_gain=1.0,
    )

    assert result.transaction.accepted is False
    assert result.transaction.report["rollback"] == {
        "performed": True,
        "exact": True,
    }
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_blank_bootstrap_rejects_a_wider_target_before_mutation() -> None:
    source = _gpt2_source()
    target = _llama_target(hidden_size=16)
    before = _state(target)

    with pytest.raises(BlankBootstrapError, match="hidden size"):
        bootstrap_blank_decoder_models(source, target, _batch())

    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name
