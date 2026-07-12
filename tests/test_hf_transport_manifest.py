from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_transport import (  # noqa: E402
    Disposition,
    PlanValidationError,
    TransportCoverage,
)
from neural_forge.hf_transport_manifest import (  # noqa: E402
    build_decoder_transport_manifest,
)


def _tiny_gpt2(*, tied: bool = True):
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=16,
            n_layer=2,
            n_head=4,
            n_inner=24,
            activation_function="gelu_new",
            tie_word_embeddings=tied,
        )
    )


def _tiny_llama(*, tied: bool = False):
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=tied,
        )
    )


@pytest.mark.parametrize("model", [_tiny_gpt2(), _tiny_llama()])
def test_manifest_claims_every_supported_decoder_parameter(model) -> None:
    manifest = build_decoder_transport_manifest(model)

    assert manifest.unclaimed_parameter_tensors == ()
    assert manifest.unclaimed_buffer_tensors == ()
    assert set(manifest.parameter_tensors) <= set(manifest.claimed_tensors)
    assert set(manifest.buffer_tensors) <= set(manifest.claimed_tensors)
    assert manifest.plan.required_roles
    coverage = TransportCoverage(manifest.plan, manifest.claimed_tensors)
    assert coverage.report()["required_role_count"] == len(manifest.plan.required_roles)


def test_manifest_preserves_tied_embedding_and_head_alias_roles() -> None:
    manifest = build_decoder_transport_manifest(_tiny_gpt2(tied=True))
    roles = manifest.plan.roles_by_name

    assert roles["embeddings.token"].target_tensors == ("model.transformer.wte.weight",)
    assert roles["lm_head"].target_tensors == ("model.lm_head.weight",)
    assert roles["embeddings.tie_policy"].allowed_dispositions == (
        Disposition.TARGET_NATIVE,
    )


def test_tensorless_roles_are_target_native_only() -> None:
    manifest = build_decoder_transport_manifest(_tiny_llama())
    tensorless = [
        role for role in manifest.plan.required_roles if not role.target_tensors
    ]

    assert tensorless
    assert all(
        role.allowed_dispositions == (Disposition.TARGET_NATIVE,) for role in tensorless
    )
    assert manifest.plan.roles_by_name["position.rope"].allowed_dispositions == (
        Disposition.TARGET_NATIVE,
    )


def test_unexpected_parameter_fails_closed() -> None:
    model = _tiny_llama()
    model.register_parameter(
        "surprise_parameter",
        torch.nn.Parameter(torch.ones(3)),
    )

    with pytest.raises(
        PlanValidationError,
        match=r"unclaimed.*model\.surprise_parameter",
    ):
        build_decoder_transport_manifest(model)
