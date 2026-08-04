from __future__ import annotations

from types import SimpleNamespace

import pytest

from neural_forge.dynamic_model_inspector import (
    DynamicModelInspectionError,
    inspect_model_circuits,
    wrap_bare_tied_causal_body,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Model"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)


def _config(*, tie: bool = True) -> "transformers.Qwen3Config":
    return transformers.Qwen3Config(
        vocab_size=24,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        max_position_embeddings=32,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        tie_word_embeddings=tie,
    )


def _bare_body(*, tie: bool = True) -> torch.nn.Module:
    torch.manual_seed(4108)
    return transformers.Qwen3Model(_config(tie=tie)).eval()


def test_wrapped_bare_body_ties_the_head_to_the_embedding() -> None:
    body = _bare_body()
    wrapper = wrap_bare_tied_causal_body(body)

    assert type(wrapper).__name__ == "Qwen3ForCausalLM"
    assert wrapper.lm_head.weight is body.embed_tokens.weight
    assert wrapper.lm_head.bias is None
    assert wrapper.model is body
    assert not [name for name, value in wrapper.named_parameters() if value.is_meta]
    # Wrapping shares storage; it must not clone the embedding.
    assert wrapper.lm_head.weight.data_ptr() == body.embed_tokens.weight.data_ptr()


def test_wrapped_logits_equal_hidden_states_times_embedding_transpose() -> None:
    body = _bare_body()
    wrapper = wrap_bare_tied_causal_body(body)
    ids = torch.tensor([[1, 5, 17, 23, 2]], dtype=torch.long)

    with torch.no_grad():
        hidden = body(input_ids=ids).last_hidden_state
        manual = hidden @ body.embed_tokens.weight.T
        logits = wrapper(input_ids=ids).logits

    assert torch.equal(manual, logits)


def test_wrapped_bare_body_becomes_inspectable_with_shared_head_storage() -> None:
    body = _bare_body()
    wrapper = wrap_bare_tied_causal_body(body)

    graph = inspect_model_circuits(wrapper, model_name="tiny-bare-qwen3")

    assert graph.family == "qwen3"
    assert graph.layer_indices == (0, 1)
    embedding = graph.node("input.token_embedding").parameters[0]
    head = graph.node("output.lm_head").parameters[0]
    assert embedding.path == "model.embed_tokens.weight"
    assert head.path == "lm_head.weight"
    assert embedding.storage_group == head.storage_group
    assert graph.unique_parameter_count == sum(
        int(value.numel()) for value in wrapper.parameters()
    )
    # The tie is accounted, not hidden: the head repeats the embedding rows.
    assert graph.logical_parameter_count - graph.unique_parameter_count == int(
        body.embed_tokens.weight.numel()
    )


def test_bare_body_itself_still_fails_the_decoder_layout_contract() -> None:
    body = _bare_body()

    with pytest.raises(
        DynamicModelInspectionError,
        match="failed its complete topology contract",
    ):
        inspect_model_circuits(body, model_name="tiny-bare-qwen3")


def test_untied_bare_body_fails_closed_rather_than_inventing_a_head() -> None:
    body = _bare_body(tie=False)

    with pytest.raises(
        DynamicModelInspectionError,
        match="Only a tied bare body determines its own head",
    ):
        wrap_bare_tied_causal_body(body)


def test_already_wrapped_causal_lm_is_refused() -> None:
    causal = transformers.Qwen3ForCausalLM(_config()).eval()

    with pytest.raises(
        DynamicModelInspectionError,
        match="Expected a bare Qwen3Model body to wrap",
    ):
        wrap_bare_tied_causal_body(causal)


def test_unsupported_model_type_is_refused() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="mamba"))

    with pytest.raises(
        DynamicModelInspectionError,
        match="Bare tied causal body wrapping supports model_type",
    ):
        wrap_bare_tied_causal_body(model)


def test_missing_configuration_is_refused() -> None:
    with pytest.raises(
        DynamicModelInspectionError,
        match="must carry a Hugging Face configuration",
    ):
        wrap_bare_tied_causal_body(SimpleNamespace(config=None))


def test_mismatched_embedding_shape_is_refused() -> None:
    body = _bare_body()
    body.embed_tokens = torch.nn.Embedding(24, 4)

    with pytest.raises(
        DynamicModelInspectionError,
        match=r"embed_tokens.weight must be \(vocab_size, hidden_size\)",
    ):
        wrap_bare_tied_causal_body(body)


def test_body_without_embedding_weight_is_refused() -> None:
    body = _bare_body()
    body.embed_tokens = torch.nn.Identity()

    with pytest.raises(
        DynamicModelInspectionError,
        match="exposes no embed_tokens.weight",
    ):
        wrap_bare_tied_causal_body(body)
