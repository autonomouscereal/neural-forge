from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import neural_forge.hf_dense_lexical_residual_adapter as lexical
from neural_forge.hf_dense_lexical_residual_adapter import (
    DenseLexicalResidualAccounting,
    DenseLexicalResidualAccountingError,
    DenseLexicalResidualEmbedding,
    DenseLexicalResidualError,
    DenseLexicalResidualLMHead,
    DenseLexicalResidualRegistry,
    DenseLexicalResidualTopologyError,
    install_dense_lexical_residual_adapter,
)


def _parameter_count(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters())


class _ToyBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.mixer = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, input_ids: Tensor) -> Tensor:
        return torch.tanh(self.mixer(self.embed_tokens(input_ids)))


class _ToyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 13,
        hidden_size: int = 6,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = _ToyBackbone(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        hidden = self.backbone(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _install(model: _ToyCausalLM, *, rank: int = 2):
    return install_dense_lexical_residual_adapter(
        model,
        rank=rank,
        initialization_seed=17,
    )


def test_accounting_is_closed_form_and_publication_native() -> None:
    report = DenseLexicalResidualAccounting.build(
        vocab_size=11,
        hidden_size=5,
        rank=3,
    )

    assert report.embedding_left_parameters == 33
    assert report.embedding_right_parameters == 15
    assert report.lm_head_left_parameters == 33
    assert report.lm_head_right_parameters == 15
    assert report.embedding_adapter_parameters == 48
    assert report.lm_head_adapter_parameters == 48
    assert report.adapter_parameters == 96
    assert report.installed_parameter_delta == 96
    assert report.installed_factor_tensor_count == 4
    assert report.installed_factor_payload_bytes == 384
    assert report.baked_publication_parameter_delta == 0
    assert report.baked_publication_adapter_tensor_count == 0
    assert report.baked_publication_adapter_payload_bytes == 0
    assert report.embedding_residual_multiply_accumulates_per_token == 15
    assert report.lm_head_residual_multiply_accumulates_per_token == 48
    assert report.total_residual_multiply_accumulates_per_token == 63
    payload = report.as_dict()
    assert payload["factor_dtype"] == "torch.float32"
    assert payload["published_adapter_parameter_count"] == 0
    assert payload["parent_runtime_reference_count"] == 0
    assert payload["source_sized_basis_parameters"] == 0
    assert payload["source_sized_sidecar_retained"] is False


def test_registry_roundtrips_with_fp32_exact_zero_factors() -> None:
    first = DenseLexicalResidualRegistry(
        vocab_size=17,
        hidden_size=7,
        rank=3,
        embedding_device="cpu",
        lm_head_device="cpu",
        initialization_seed=8,
    )
    second = DenseLexicalResidualRegistry(
        vocab_size=17,
        hidden_size=7,
        rank=3,
        embedding_device="cpu",
        lm_head_device="cpu",
        initialization_seed=99,
    )
    second.load_state_dict(first.state_dict(), strict=True)

    assert first.starts_as_exact_zero_correction
    assert first.input_factors_are_nonzero
    assert first.learned_parameter_count == 2 * 3 * (17 + 7)
    assert all(value.dtype == torch.float32 for value in first.parameters())
    assert torch.count_nonzero(first.dense_delta("embedding")) == 0
    assert torch.count_nonzero(first.dense_delta("lm_head")) == 0
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    payload = first.as_dict()
    assert payload["source_tensor_reference_count"] == 0
    assert payload["activation_cache_reference_count"] == 0


def test_fp16_install_is_exact_zero_and_rollback_restores_objects_and_keys() -> None:
    torch.manual_seed(3)
    model = _ToyCausalLM().half().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    original_embedding = model.backbone.embed_tokens
    original_head = model.lm_head
    original_keys = tuple(model.state_dict())
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    before_count = _parameter_count(model)
    baseline = model(input_ids).logits.detach().clone()

    handle = _install(model, rank=3)
    installed = model(input_ids).logits

    assert torch.equal(installed, baseline)
    assert isinstance(model.backbone.embed_tokens, DenseLexicalResidualEmbedding)
    assert isinstance(model.lm_head, DenseLexicalResidualLMHead)
    assert all(
        value.dtype == torch.float32 for value in handle.adapter_registry.parameters()
    )
    expected_adapter = 2 * 3 * (13 + 6)
    assert handle.report.adapter_parameter_count == expected_adapter
    assert _parameter_count(model) == before_count + expected_adapter
    payload = handle.report.as_dict()
    assert payload["zero_step_identity"] is True
    assert payload["factor_only_output_training"] is True
    assert payload["parent_runtime_reference_count"] == 0
    assert payload["source_tensor_reference_count"] == 0
    assert payload["sidecar_reference_count"] == 0

    handle.rollback()
    assert model.backbone.embed_tokens is original_embedding
    assert model.lm_head is original_head
    assert tuple(model.state_dict()) == original_keys
    assert _parameter_count(model) == before_count
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in original_state.items()
    )


def test_factor_only_output_training_freezes_every_native_parameter() -> None:
    torch.manual_seed(5)
    model = _ToyCausalLM().train()
    handle = _install(model)
    factor_ids = {id(value) for value in handle.adapter_registry.parameters()}
    base = tuple(value for value in model.parameters() if id(value) not in factor_ids)
    base_versions = tuple(int(value._version) for value in base)
    base_modes = tuple((module, module.training) for module in model.modules())
    scope = handle.output_training_scope()

    with scope as trainable:
        assert len(trainable) == 4
        assert {id(value) for value in trainable} == factor_ids
        assert all(value.dtype == torch.float32 for value in trainable)
        assert all(value.requires_grad for value in trainable)
        assert all(not value.requires_grad for value in base)
        assert model.training is False

        logits = model(torch.tensor([[1, 2, 3, 4]])).logits
        gradients = torch.autograd.grad(logits.square().sum(), trainable)
        named_gradients = dict(
            zip(
                (name for name, _value in handle.adapter_registry.named_parameters()),
                gradients,
                strict=True,
            )
        )
        assert all(
            torch.count_nonzero(value) > 0
            for name, value in named_gradients.items()
            if name.endswith("_left")
        )
        assert all(
            torch.count_nonzero(value) == 0
            for name, value in named_gradients.items()
            if name.endswith("_right")
        )
        with torch.no_grad():
            for parameter, gradient in zip(trainable, gradients, strict=True):
                parameter.add_(gradient, alpha=-1e-4)
        second_logits = model(torch.tensor([[1, 2, 3, 4]])).logits
        second_gradients = torch.autograd.grad(second_logits.square().sum(), trainable)
        assert all(
            torch.count_nonzero(value) > 0
            for (name, _parameter), value in zip(
                handle.adapter_registry.named_parameters(),
                second_gradients,
                strict=True,
            )
            if name.endswith("_right")
        )

    assert tuple(int(value._version) for value in base) == base_versions
    assert all(value.requires_grad for value in base)
    assert all(module.training is training for module, training in base_modes)
    scope_payload = scope.as_dict()
    assert scope_payload["factor_parameters_only"] is True
    assert scope_payload["accepted_target_space"].endswith("actions-or-logits")
    assert scope_payload["internal_similarity_target_api_exposed"] is False
    assert scope_payload["parent_model_api_exposed"] is False
    handle.rollback()


def test_nonfinite_factor_fails_forward_but_cannot_block_exact_rollback() -> None:
    model = _ToyCausalLM()
    original_embedding = model.backbone.embed_tokens
    original_head = model.lm_head
    original_keys = tuple(model.state_dict())
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    before_count = _parameter_count(model)
    handle = _install(model)
    with torch.no_grad():
        handle.adapter_registry.embedding_left[1, 0] = float("nan")

    with pytest.raises(DenseLexicalResidualError, match="NaN or infinity"):
        model(torch.tensor([[1, 2]])).logits

    handle.rollback()
    assert model.backbone.embed_tokens is original_embedding
    assert model.lm_head is original_head
    assert tuple(model.state_dict()) == original_keys
    assert _parameter_count(model) == before_count
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in original_state.items()
    )


def test_bake_restores_plain_dense_topology_count_keys_and_strict_reload() -> None:
    torch.manual_seed(7)
    model = _ToyCausalLM().double().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    original_keys = tuple(model.state_dict())
    before_count = _parameter_count(model)
    original_embedding_weight = model.backbone.embed_tokens.weight.detach().clone()
    original_head_weight = model.lm_head.weight.detach().clone()
    assert model.lm_head.bias is not None
    original_head_bias = model.lm_head.bias.detach().clone()
    handle = _install(model, rank=2)
    with torch.no_grad():
        handle.adapter_registry.embedding_left.fill_(0.0125)
        handle.adapter_registry.embedding_right.fill_(0.025)
        handle.adapter_registry.lm_head_left.fill_(0.0375)
        handle.adapter_registry.lm_head_right.fill_(0.05)
    embedding_delta = handle.adapter_registry.dense_delta("embedding").detach().double()
    head_delta = handle.adapter_registry.dense_delta("lm_head").detach().double()
    wrapped = model(input_ids).logits.detach().clone()

    report = handle.bake_for_publication()
    baked = model(input_ids).logits.detach()

    torch.testing.assert_close(baked, wrapped, rtol=2e-6, atol=2e-6)
    assert type(model.backbone.embed_tokens) is nn.Embedding
    assert type(model.lm_head) is nn.Linear
    assert tuple(model.state_dict()) == original_keys
    assert _parameter_count(model) == before_count
    assert not hasattr(model, handle.report.owner_name)
    assert not any("base_embedding" in key for key in model.state_dict())
    assert not any("base_lm_head" in key for key in model.state_dict())
    torch.testing.assert_close(
        model.backbone.embed_tokens.weight,
        original_embedding_weight + embedding_delta,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.lm_head.weight,
        original_head_weight + head_delta,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(model.lm_head.bias, original_head_bias)
    assert report.adapter_parameters_removed == 2 * 2 * (13 + 6)
    assert report.model_parameter_count_after_bake == before_count
    assert report.affected_dense_matrix_count == 2
    payload = report.as_dict()
    assert payload["published_state_key_topology_exact"] is True
    assert payload["published_adapter_parameter_count"] == 0
    assert payload["source_sized_sidecar_retained"] is False
    assert payload["parent_runtime_reference_count"] == 0

    fresh = _ToyCausalLM().double().eval()
    fresh.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(fresh(input_ids).logits, baked)
    with pytest.raises(DenseLexicalResidualTopologyError, match="restore"):
        handle.rollback()


def test_prepublication_restore_returns_original_native_modules_exactly() -> None:
    model = _ToyCausalLM().double()
    original_embedding = model.backbone.embed_tokens
    original_head = model.lm_head
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    handle = _install(model)
    with torch.no_grad():
        handle.adapter_registry.embedding_left.fill_(0.1)
        handle.adapter_registry.lm_head_left.fill_(0.1)
    handle.bake_for_publication()
    handle.restore_prepublication_native()

    assert model.backbone.embed_tokens is original_embedding
    assert model.lm_head is original_head
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in original_state.items()
    )


def test_failed_bake_restores_live_wrappers_and_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyCausalLM().double()
    handle = _install(model)
    original_state = {
        name: snapshot.sha256 for name, snapshot in handle._base_state_before.items()
    }
    real_count = lexical._unique_parameter_count
    calls = 0

    def wrong_after_install(module: nn.Module) -> int:
        nonlocal calls
        calls += 1
        value = real_count(module)
        return value + (1 if calls >= 2 else 0)

    monkeypatch.setattr(lexical, "_unique_parameter_count", wrong_after_install)
    with pytest.raises(DenseLexicalResidualAccountingError):
        handle.bake_for_publication()

    assert getattr(model, handle.report.owner_name) is handle.adapter_registry
    assert model.backbone.embed_tokens is handle._installed_embedding
    assert model.lm_head is handle._installed_lm_head
    assert {
        name: lexical._tensor_sha256(snapshot.tensor)
        for name, snapshot in handle._base_state_before.items()
    } == original_state
    monkeypatch.setattr(lexical, "_unique_parameter_count", real_count)
    handle.assert_live()
    handle.rollback()


def test_base_mutation_and_live_wrapper_replacement_fail_closed() -> None:
    model = _ToyCausalLM()
    handle = _install(model)
    original = handle._original_embedding.weight.detach().clone()
    with torch.no_grad():
        handle._original_embedding.weight[0, 0].add_(1.0)
    with pytest.raises(DenseLexicalResidualTopologyError, match="base tensor"):
        handle.rollback()
    with torch.no_grad():
        handle._original_embedding.weight.copy_(original)
    # Byte restoration cannot undo the tensor version, so this transaction
    # remains intentionally unclaimable as an exact rollback.
    with pytest.raises(DenseLexicalResidualTopologyError):
        handle.rollback()

    model = _ToyCausalLM()
    handle = _install(model)
    installed = model.lm_head
    model.lm_head = nn.Identity()
    with pytest.raises(DenseLexicalResidualTopologyError, match="wrapper"):
        handle.rollback()
    model.lm_head = installed
    handle.rollback()


def test_tied_mismatched_aliased_and_mutating_native_modules_are_rejected() -> None:
    tied = _ToyCausalLM(head_bias=False)
    tied.lm_head.weight = tied.backbone.embed_tokens.weight
    with pytest.raises(DenseLexicalResidualTopologyError, match="untied"):
        _install(tied)

    mismatch = _ToyCausalLM()
    mismatch.lm_head = nn.Linear(6, 12)
    with pytest.raises(DenseLexicalResidualTopologyError, match="vocab and hidden"):
        _install(mismatch)

    aliased = _ToyCausalLM()
    aliased.output_alias = aliased.lm_head
    with pytest.raises(DenseLexicalResidualTopologyError, match="exactly one"):
        _install(aliased)

    max_norm = _ToyCausalLM()
    max_norm.backbone.embed_tokens = nn.Embedding(13, 6, max_norm=1.0)
    with pytest.raises(DenseLexicalResidualTopologyError, match="max_norm"):
        _install(max_norm)


def test_explicit_modules_work_without_hugging_face_getters() -> None:
    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = nn.Embedding(9, 4)
            self.output = nn.Linear(4, 9, bias=False)

        def forward(self, input_ids: Tensor) -> Tensor:
            return self.output(self.tokens(input_ids))

    model = Plain()
    baseline = model(torch.tensor([[1, 2]])).detach().clone()
    handle = install_dense_lexical_residual_adapter(
        model,
        model.tokens,
        model.output,
        rank=1,
    )
    assert torch.equal(model(torch.tensor([[1, 2]])), baseline)
    handle.bake_for_publication()
    assert type(model.tokens) is nn.Embedding
    assert type(model.output) is nn.Linear


def test_real_qwen_untied_lexical_boundary_bakes_and_strictly_reloads() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_bias=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(31)
    model = transformers.Qwen3ForCausalLM(config).eval()
    assert (
        model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    original_keys = tuple(model.state_dict())
    before_count = _parameter_count(model)
    baseline = model(input_ids, use_cache=False).logits.detach().clone()

    handle = install_dense_lexical_residual_adapter(
        model,
        rank=2,
        initialization_seed=4,
    )
    zero = model(input_ids, use_cache=False).logits
    assert torch.equal(zero, baseline)
    with torch.no_grad():
        handle.adapter_registry.embedding_left.fill_(0.002)
        handle.adapter_registry.embedding_right.fill_(0.003)
        handle.adapter_registry.lm_head_left.fill_(0.004)
        handle.adapter_registry.lm_head_right.fill_(0.005)
    active = model(input_ids, use_cache=False).logits.detach().clone()
    handle.bake_for_publication()
    baked = model(input_ids, use_cache=False).logits.detach()

    torch.testing.assert_close(baked, active, rtol=2e-5, atol=2e-5)
    assert type(model.get_input_embeddings()) is nn.Embedding
    assert type(model.get_output_embeddings()) is nn.Linear
    assert tuple(model.state_dict()) == original_keys
    assert _parameter_count(model) == before_count
    fresh = transformers.Qwen3ForCausalLM(config).eval()
    fresh.load_state_dict(model.state_dict(), strict=True)
    reloaded = fresh(input_ids, use_cache=False).logits
    torch.testing.assert_close(reloaded, baked, rtol=0.0, atol=0.0)
