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


def _qwen3_source():
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(252)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
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


def _batch(seed: int = 404, *, batch_size: int = 8, sequence_length: int = 8):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        3,
        64,
        (batch_size, sequence_length),
        generator=generator,
    )
    input_ids[:, 0] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _right_pad(batch, padded_length: int):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    padding = padded_length - int(input_ids.shape[1])
    assert padding > 0
    return {
        "input_ids": torch.cat(
            [input_ids, torch.zeros((input_ids.shape[0], padding), dtype=torch.long)],
            dim=1,
        ),
        "attention_mask": torch.cat(
            [
                attention_mask,
                torch.zeros((input_ids.shape[0], padding), dtype=torch.long),
            ],
            dim=1,
        ),
    }


@pytest.mark.parametrize(
    "source_factory",
    [_gpt2_source, _llama_source, _qwen3_source],
)
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
        max_coordinate_rows=24,
        vocab_projection_chunk_size=7,
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
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
    if source_factory is _qwen3_source:
        fold_report = result.fit_reports["block.0.attention.feature_folds"]
        assert fold_report["source_query_head_norms"] == [
            "model.layers.0.self_attn.q_norm"
        ]
        assert fold_report["source_key_head_norms"] == [
            "model.layers.0.self_attn.k_norm"
        ]


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


def test_prompt_pooled_bootstrap_accepts_different_token_sequence_lengths() -> None:
    source = _gpt2_source()
    target = _llama_target()
    target_calibration = _batch(501, batch_size=16, sequence_length=5)
    target_validation = _batch(502, batch_size=16, sequence_length=4)

    result = bootstrap_blank_decoder_models(
        source,
        target,
        _right_pad(target_calibration, 9),
        validation_inputs=_right_pad(target_validation, 7),
        target_calibration_inputs=target_calibration,
        target_validation_inputs=target_validation,
        trace_pooling="mean",
        fit_full_lm_head=False,
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
    assert result.transaction.report["coverage"]["complete"] is True
    assert result.candidate_gate_metrics.final_state.relative_rmse <= (
        result.baseline_gate_metrics.final_state.relative_rmse
    )
    assert result.candidate_gate_metrics.logits.relative_rmse <= (
        result.baseline_gate_metrics.logits.relative_rmse
    )


def test_blank_bootstrap_rejects_a_wider_target_before_mutation() -> None:
    source = _gpt2_source()
    target = _llama_target(hidden_size=16)
    before = _state(target)

    with pytest.raises(BlankBootstrapError, match="hidden size"):
        bootstrap_blank_decoder_models(source, target, _batch())

    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_blank_bootstrap_rejects_an_undersized_coordinate_sample() -> None:
    with pytest.raises(BlankBootstrapError, match="max_coordinate_rows"):
        bootstrap_blank_decoder_models(
            _gpt2_source(),
            _llama_target(),
            _batch(),
            max_coordinate_rows=7,
        )
