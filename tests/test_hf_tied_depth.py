from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_tied_depth import (  # noqa: E402
    ATTENTION_PROJECTIONS,
    MLP_PROJECTIONS,
    TiedDepthError,
    exact_token_row_mapping,
    prepare_qwen3_tied_depth_model,
)


class _Tokenizer:
    def __init__(self, tokens, *, bos=1, eos=2, pad=0):
        self._tokens = tuple(tokens)
        self._vocab = {token: index for index, token in enumerate(self._tokens)}
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad

    def __len__(self):
        return len(self._tokens)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        if 0 <= token_id < len(self._tokens):
            return self._tokens[token_id]
        return None


def _tiny_qwen3():
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(2421)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=31,
            hidden_size=12,
            intermediate_size=16,
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
            tie_word_embeddings=True,
        )
    ).eval()


def _tokenizers():
    donor_tokens = [f"token-{index}" for index in range(31)]
    selected = (0, 3, 7, 11, 15, 19, 23, 27)
    target_tokens = [donor_tokens[index] for index in selected]
    donor = _Tokenizer(donor_tokens)
    target = _Tokenizer(target_tokens, bos=1, eos=2, pad=0)
    return donor, target, selected


def test_exact_token_row_mapping_uses_raw_vocab_identity() -> None:
    donor, target, selected = _tokenizers()

    assert exact_token_row_mapping(donor, target) == selected

    missing = _Tokenizer(["token-0", "not-in-donor"])
    with pytest.raises(TiedDepthError, match="no exact donor row"):
        exact_token_row_mapping(donor, missing)


def test_tied_depth_preserves_full_execution_and_real_parent_parameters() -> None:
    model = _tiny_qwen3()
    donor, target, selected = _tokenizers()
    original_embedding = model.get_input_embeddings().weight.detach().clone()
    layers = model.model.layers
    source_attention = {
        role: getattr(layers[1].self_attn, role).weight
        for role in ATTENTION_PROJECTIONS
    }
    source_mlp = {role: getattr(layers[2].mlp, role).weight for role in MLP_PROJECTIONS}
    original_norms = [layer.input_layernorm.weight for layer in layers]
    source_unique = sum(parameter.numel() for parameter in model.parameters())

    report = prepare_qwen3_tied_depth_model(
        model,
        donor,
        target,
        attention_source_layer=1,
        mlp_source_layer=2,
    )

    assert report.logical_layers == 3
    assert report.source_unique_parameter_count == source_unique
    assert report.child_unique_parameter_count < source_unique
    assert report.full_logical_depth is True
    assert report.zero_gradient is True
    assert report.exact_parent_token_rows is True
    assert report.target_vocab_size == len(selected)
    assert model.config.vocab_size == len(selected)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    torch.testing.assert_close(
        model.get_input_embeddings().weight,
        original_embedding[list(selected)],
    )
    for layer in layers:
        for role, parameter in source_attention.items():
            assert getattr(layer.self_attn, role).weight is parameter
        for role, parameter in source_mlp.items():
            assert getattr(layer.mlp, role).weight is parameter
    assert [layer.input_layernorm.weight for layer in layers] == original_norms
    assert len({id(value) for value in original_norms}) == len(layers)

    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert tuple(logits.shape) == (1, 4, len(selected))
    assert bool(torch.isfinite(logits).all())


def test_tied_depth_rejects_repeat_and_out_of_range_sources() -> None:
    model = _tiny_qwen3()
    donor, target, _ = _tokenizers()
    with pytest.raises(TiedDepthError, match="must be in"):
        prepare_qwen3_tied_depth_model(
            model,
            donor,
            target,
            attention_source_layer=3,
            mlp_source_layer=0,
        )

    prepare_qwen3_tied_depth_model(
        model,
        donor,
        target,
        attention_source_layer=0,
        mlp_source_layer=0,
    )
    with pytest.raises(TiedDepthError, match="already prepared"):
        prepare_qwen3_tied_depth_model(
            model,
            donor,
            target,
            attention_source_layer=0,
            mlp_source_layer=0,
        )
