from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import neural_forge.hf_dense_gqa_qkvo_residual_adapter as dense_qkvo
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAGeometry,
    DenseGQAQKVOResidualAccounting,
    DenseGQAQKVOResidualAccountingError,
    DenseGQAQKVOResidualError,
    DenseGQAQKVOResidualProjection,
    DenseGQAQKVOResidualTopologyError,
    DenseGQASite,
    SparseDenseGQAQKVOResidualRegistry,
    install_dense_gqa_qkvo_residual_adapter,
)


GEOMETRY = DenseGQAGeometry(
    logical_layers=3,
    hidden_size=8,
    query_heads=4,
    key_value_heads=2,
    head_dim=2,
)


def test_arbitrary_dense_geometry_accounting_is_exact_before_and_after_bake() -> None:
    report = DenseGQAQKVOResidualAccounting.build(
        logical_layers=3,
        hidden_size=8,
        query_heads=4,
        key_value_heads=2,
        head_dim=2,
        rank=3,
        site_count=2,
        checkpoint_element_size=4,
    )

    assert report.query_group_width == 4
    assert report.q_adapter_parameters == 72
    assert report.k_adapter_parameters == 60
    assert report.v_adapter_parameters == 60
    assert report.o_adapter_parameters == 72
    assert report.adapter_parameters == 264
    assert report.installed_parameter_delta == 264
    assert report.baked_publication_parameter_delta == 0
    assert report.installed_learned_parameter_bytes == 1_056
    assert report.adapter_site_index_bytes == 16
    assert report.adapter_schema_buffer_bytes == 8
    assert report.installed_tensor_payload_bytes == 1_080
    assert report.baked_publication_tensor_payload_bytes == 0
    assert report.flops_per_site_token_while_installed == 264
    assert report.baked_publication_adapter_flops_per_token == 0
    payload = report.as_dict()
    assert payload["source_sized_basis_parameters"] == 0
    assert payload["published_dense_parameter_topology_unchanged"] is True


def test_geometry_and_site_contracts_fail_closed() -> None:
    with pytest.raises(DenseGQAQKVOResidualTopologyError):
        DenseGQAGeometry(2, 8, 3, 2, 2)
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="kv-major"):
        DenseGQAGeometry(2, 8, 4, 2, 2, query_group_layout="interleaved")

    for sites in (
        (DenseGQASite(1, 0), DenseGQASite(0, 0)),
        (DenseGQASite(0, 0), DenseGQASite(0, 0)),
        (DenseGQASite(3, 0),),
        (DenseGQASite(0, 2),),
    ):
        with pytest.raises(DenseGQAQKVOResidualTopologyError):
            SparseDenseGQAQKVOResidualRegistry(
                sites,
                geometry=GEOMETRY,
                rank=1,
                dtype=torch.float32,
                device="cpu",
            )


def test_sparse_target_registry_roundtrips_and_starts_as_exact_zero() -> None:
    sites = (DenseGQASite(0, 1), DenseGQASite(2, 0))
    first = SparseDenseGQAQKVOResidualRegistry(
        sites,
        geometry=GEOMETRY,
        rank=2,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=17,
    )
    second = SparseDenseGQAQKVOResidualRegistry(
        sites,
        geometry=GEOMETRY,
        rank=2,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=999,
    )
    second.load_state_dict(first.state_dict(), strict=True)

    assert first.site_count == 2
    assert first.starts_as_exact_zero_correction
    assert first.input_factors_are_nonzero
    assert first.learned_parameter_count == 2 * 2 * (4 * 8 + 2 * 4 + 2 * 2)
    assert first.index_buffer_bytes == 16
    assert first.schema_buffer_bytes == 8
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    assert first.as_dict()["parent_runtime_references"] == 0


def _active_registry() -> SparseDenseGQAQKVOResidualRegistry:
    registry = SparseDenseGQAQKVOResidualRegistry(
        (DenseGQASite(1, 1),),
        geometry=GEOMETRY,
        rank=1,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=3,
    )
    with torch.no_grad():
        for role in ("q", "k", "v", "o"):
            getattr(registry, f"{role}_left").fill_(1.0)
            getattr(registry, f"{role}_right").fill_(1.0)
    return registry


def test_direct_qkvo_corrections_touch_only_the_target_physical_group() -> None:
    registry = _active_registry()
    hidden = torch.arange(1.0, 9.0).reshape(1, 8)
    q_base = torch.zeros(1, 8)
    k_base = torch.zeros(1, 4)
    v_base = torch.zeros(1, 4)
    contexts = torch.tensor([[50.0, 50.0, 50.0, 50.0, 1.0, 2.0, 3.0, 4.0]])
    o_base = torch.zeros(1, 8)

    q = registry.apply_to_projection("q", hidden, q_base, layer_index=1)
    k = registry.apply_to_projection("k", hidden, k_base, layer_index=1)
    v = registry.apply_to_projection("v", hidden, v_base, layer_index=1)
    o = registry.apply_to_output(contexts, o_base, layer_index=1)

    assert torch.equal(q[..., :4], torch.zeros(1, 4))
    assert torch.equal(q[..., 4:], torch.full((1, 4), 36.0))
    assert torch.equal(k[..., :2], torch.zeros(1, 2))
    assert torch.equal(k[..., 2:], torch.full((1, 2), 36.0))
    assert torch.equal(v[..., :2], torch.zeros(1, 2))
    assert torch.equal(v[..., 2:], torch.full((1, 2), 36.0))
    assert torch.equal(o, torch.full((1, 8), 10.0))
    assert torch.equal(
        registry.apply_to_projection("q", hidden, q_base, layer_index=0),
        q_base,
    )


def test_registry_is_strictly_pointwise_over_sequence_positions() -> None:
    registry = _active_registry()
    hidden = torch.randn(2, 5, 8)
    q_base = torch.randn(2, 5, 8)
    contexts = torch.randn(2, 5, 8)
    o_base = torch.randn(2, 5, 8)

    q_full = registry.apply_to_projection("q", hidden, q_base, layer_index=1)
    q_prefix = registry.apply_to_projection(
        "q", hidden[:, :3], q_base[:, :3], layer_index=1
    )
    o_full = registry.apply_to_output(contexts, o_base, layer_index=1)
    o_prefix = registry.apply_to_output(contexts[:, :3], o_base[:, :3], layer_index=1)

    assert torch.equal(q_full[:, :3], q_prefix)
    assert torch.equal(o_full[:, :3], o_prefix)
    assert registry.as_dict()["cross_token_mixing_inside_adapter"] is False


class _RecordingNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: Tensor | None = None

    def forward(self, value: Tensor) -> Tensor:
        self.last_input = value.detach().clone()
        return value


class _RecordingCache:
    def __init__(self) -> None:
        self.key: Tensor | None = None
        self.value: Tensor | None = None

    def update(self, key: Tensor, value: Tensor, layer_index: int):
        del layer_index
        self.key = key.detach().clone()
        self.value = value.detach().clone()
        return key, value


class _DenseToyAttention(nn.Module):
    def __init__(
        self, layer_index: int, *, names: dict[str, str] | None = None
    ) -> None:
        super().__init__()
        projection_names = names or {
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "o_proj",
        }
        setattr(self, projection_names["q"], nn.Linear(8, 8, bias=True))
        setattr(self, projection_names["k"], nn.Linear(8, 4, bias=True))
        setattr(self, projection_names["v"], nn.Linear(8, 4, bias=True))
        setattr(self, projection_names["o"], nn.Linear(8, 8, bias=True))
        self.projection_names = projection_names
        self.q_norm = _RecordingNorm()
        self.k_norm = _RecordingNorm()
        self.layer_idx = layer_index

    def projection(self, role: str) -> nn.Module:
        return getattr(self, self.projection_names[role])

    def forward(self, hidden: Tensor, cache: _RecordingCache | None = None) -> Tensor:
        q = self.q_norm(self.projection("q")(hidden))
        k = self.k_norm(self.projection("k")(hidden))
        v = self.projection("v")(hidden)
        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
        context = q + k.repeat_interleave(2, dim=-1) + v.repeat_interleave(2, dim=-1)
        return self.projection("o")(context)


class _DenseToyModel(nn.Module):
    def __init__(self, *, names: dict[str, str] | None = None) -> None:
        super().__init__()
        self.attentions = nn.ModuleList(
            _DenseToyAttention(index, names=names) for index in range(3)
        )
        self.embedding = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 7, bias=False)

    def forward(self, input_ids: Tensor, **kwargs) -> SimpleNamespace:
        del kwargs
        hidden = self.embedding(input_ids)
        hidden = self.attentions[1](hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _install(
    model: _DenseToyModel,
    *,
    sites: tuple[DenseGQASite, ...] = (DenseGQASite(1, 1),),
    rank: int = 1,
    projection_names: dict[str, str] | None = None,
):
    return install_dense_gqa_qkvo_residual_adapter(
        model,
        model.attentions,
        geometry=GEOMETRY,
        sites=sites,
        rank=rank,
        projection_names=projection_names,
        initialization_seed=5,
    )


def test_install_is_zero_exact_and_qkv_execute_before_norm_and_cache() -> None:
    torch.manual_seed(11)
    model = _DenseToyModel()
    attention = model.attentions[1]
    originals = {role: attention.projection(role) for role in ("q", "k", "v", "o")}
    hidden = torch.randn(2, 3, 8)
    baseline_cache = _RecordingCache()
    baseline = attention(hidden, baseline_cache).detach().clone()
    before_count = sum(value.numel() for value in model.parameters())

    handle = _install(model)
    installed_cache = _RecordingCache()
    installed = attention(hidden, installed_cache)

    assert torch.equal(installed, baseline)
    assert torch.equal(installed_cache.key, baseline_cache.key)
    assert torch.equal(installed_cache.value, baseline_cache.value)
    assert handle.report.adapter_parameter_count == 44
    assert sum(value.numel() for value in model.parameters()) == before_count + 44
    assert all(
        isinstance(attention.projection(role), DenseGQAQKVOResidualProjection)
        for role in ("q", "k", "v", "o")
    )

    with torch.no_grad():
        handle.adapter_registry.q_left.fill_(0.25)
        handle.adapter_registry.k_left.fill_(0.5)
        handle.adapter_registry.v_left.fill_(0.75)
    q_delta = attention.projection("q")(hidden) - originals["q"](hidden)
    k_delta = attention.projection("k")(hidden) - originals["k"](hidden)
    v_delta = attention.projection("v")(hidden) - originals["v"](hidden)
    assert torch.count_nonzero(q_delta[..., :4]) == 0
    assert torch.count_nonzero(q_delta[..., 4:]) > 0
    assert torch.count_nonzero(k_delta[..., :2]) == 0
    assert torch.count_nonzero(k_delta[..., 2:]) > 0
    assert torch.count_nonzero(v_delta[..., :2]) == 0
    assert torch.count_nonzero(v_delta[..., 2:]) > 0

    changed_cache = _RecordingCache()
    attention(hidden, changed_cache)
    assert torch.equal(attention.q_norm.last_input, attention.projection("q")(hidden))
    assert torch.equal(attention.k_norm.last_input, attention.projection("k")(hidden))
    assert torch.equal(changed_cache.key, attention.projection("k")(hidden))
    assert torch.equal(changed_cache.value, attention.projection("v")(hidden))
    payload = handle.report.as_dict()
    assert payload["q_correction_boundary"].endswith("before-q-norm-and-rope")
    assert payload["k_correction_boundary"].endswith(
        "before-k-norm-rope-and-cache-append"
    )
    assert payload["parent_runtime_reference_count"] == 0

    handle.rollback()
    assert sum(value.numel() for value in model.parameters()) == before_count
    assert all(attention.projection(role) is originals[role] for role in originals)


def test_output_training_scope_freezes_dense_base_and_exposes_only_factors() -> None:
    model = _DenseToyModel()
    handle = _install(model)
    adapter_ids = {id(value) for value in handle.adapter_registry.parameters()}
    base = tuple(value for value in model.parameters() if id(value) not in adapter_ids)
    versions = tuple(int(value._version) for value in base)

    with handle.output_training_scope() as trainable:
        assert {id(value) for value in trainable} == adapter_ids
        assert all(value.requires_grad for value in trainable)
        assert all(not value.requires_grad for value in base)
        logits = model(torch.tensor([[1, 2]])).logits
        gradients = torch.autograd.grad(logits.sum(), trainable)
        assert len(gradients) == 8
        first_gradients = dict(
            zip(
                (name for name, _value in handle.adapter_registry.named_parameters()),
                gradients,
                strict=True,
            )
        )
        assert all(
            int(torch.count_nonzero(value).item()) == 0
            for name, value in first_gradients.items()
            if name.endswith("_right")
        )
        assert all(
            int(torch.count_nonzero(value).item()) > 0
            for name, value in first_gradients.items()
            if name.endswith("_left")
        )
        with torch.no_grad():
            for parameter, gradient in zip(trainable, gradients, strict=True):
                parameter.add_(gradient, alpha=-1e-3)
        second_logits = model(torch.tensor([[1, 2]])).logits
        second_gradients = torch.autograd.grad(second_logits.sum(), trainable)
        assert all(
            int(torch.count_nonzero(value).item()) > 0
            for (name, _parameter), value in zip(
                handle.adapter_registry.named_parameters(),
                second_gradients,
                strict=True,
            )
            if name.endswith("_right")
        )

    assert tuple(int(value._version) for value in base) == versions
    assert all(value.requires_grad for value in base)
    scope = handle.output_training_scope().as_dict()
    assert scope["accepted_target_space"] == "causal-model-output-actions-or-logits"
    assert scope["internal_similarity_target_api_exposed"] is False
    handle.rollback()


def test_base_mutation_and_projection_replacement_fail_closed() -> None:
    model = _DenseToyModel()
    handle = _install(model)
    attention = model.attentions[1]
    base = handle._original_projections[(1, "q")]
    original_weight = base.weight.detach().clone()
    with torch.no_grad():
        base.weight[0, 0].add_(1.0)
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="base projection"):
        handle.assert_live()
    with torch.no_grad():
        base.weight.copy_(original_weight)
    # Copying the bytes back cannot restore the tensor version; the fail-closed
    # handle must not pretend this candidate can be exactly rolled back.
    with pytest.raises(DenseGQAQKVOResidualTopologyError):
        handle.rollback()

    model = _DenseToyModel()
    handle = _install(model)
    attention = model.attentions[1]
    installed = attention.k_proj
    attention.k_proj = nn.Identity()
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="wrapper was replaced"):
        handle.rollback()
    attention.k_proj = installed
    handle.rollback()


def test_nonfinite_adapter_fails_forward_but_remains_exactly_removable() -> None:
    model = _DenseToyModel()
    attention = model.attentions[1]
    originals = {role: attention.projection(role) for role in ("q", "k", "v", "o")}
    handle = _install(model)
    with torch.no_grad():
        handle.adapter_registry.k_left[0, 0, 0] = float("nan")
    with pytest.raises(DenseGQAQKVOResidualError):
        attention(torch.randn(1, 2, 8))
    handle.rollback()
    assert all(attention.projection(role) is originals[role] for role in originals)


def test_bake_materializes_exact_group_slices_and_removes_all_scaffolding() -> None:
    torch.manual_seed(23)
    model = _DenseToyModel().to(torch.float64)
    attention = model.attentions[1]
    before_count = sum(value.numel() for value in model.parameters())
    original_weights = {
        role: attention.projection(role).weight.detach().clone()
        for role in ("q", "k", "v", "o")
    }
    original_biases = {
        role: attention.projection(role).bias.detach().clone()
        for role in ("q", "k", "v", "o")
    }
    handle = _install(model, rank=2)
    with torch.no_grad():
        for role_index, role in enumerate(("q", "k", "v", "o"), start=1):
            getattr(handle.adapter_registry, f"{role}_left").fill_(role_index / 10)
            getattr(handle.adapter_registry, f"{role}_right").fill_(role_index / 20)
    hidden = torch.randn(2, 4, 8, dtype=torch.float64)
    wrapped_output = attention(hidden).detach().clone()
    expected_deltas = {
        role: handle.adapter_registry.dense_delta(role, 0).detach().clone()
        for role in ("q", "k", "v", "o")
    }

    report = handle.bake_for_publication()
    baked_output = attention(hidden).detach()

    torch.testing.assert_close(baked_output, wrapped_output, rtol=1e-12, atol=1e-12)
    assert report.model_parameter_count_before_installation == before_count
    assert report.model_parameter_count_after_bake == before_count
    assert report.adapter_parameters_removed == 88
    assert report.affected_slice_count == 4
    assert handle.is_baked
    assert not hasattr(model, handle.report.owner_name)
    assert all(
        type(attention.projection(role)) is nn.Linear for role in original_weights
    )
    assert all(
        torch.equal(attention.projection(role).bias, original_biases[role])
        for role in original_biases
    )
    state_keys = set(model.state_dict())
    assert not any("base_projection" in key for key in state_keys)
    assert not any(handle.report.owner_name in key for key in state_keys)

    torch.testing.assert_close(
        attention.q_proj.weight[4:8],
        original_weights["q"][4:8] + expected_deltas["q"],
    )
    assert torch.equal(attention.q_proj.weight[:4], original_weights["q"][:4])
    torch.testing.assert_close(
        attention.k_proj.weight[2:4],
        original_weights["k"][2:4] + expected_deltas["k"],
    )
    assert torch.equal(attention.k_proj.weight[:2], original_weights["k"][:2])
    torch.testing.assert_close(
        attention.v_proj.weight[2:4],
        original_weights["v"][2:4] + expected_deltas["v"],
    )
    assert torch.equal(attention.v_proj.weight[:2], original_weights["v"][:2])
    torch.testing.assert_close(
        attention.o_proj.weight[:, 4:8],
        original_weights["o"][:, 4:8] + expected_deltas["o"],
    )
    assert torch.equal(attention.o_proj.weight[:, :4], original_weights["o"][:, :4])
    payload = report.as_dict()
    assert payload["published_adapter_parameter_count"] == 0
    assert payload["source_sized_sidecar_retained"] is False
    assert payload["functional_publication_gate_still_required"] is True
    with pytest.raises(DenseGQAQKVOResidualTopologyError):
        handle.rollback()


def test_multi_site_bake_handles_multiple_layers_without_cross_group_damage() -> None:
    model = _DenseToyModel().to(torch.float64)
    sites = (
        DenseGQASite(0, 0),
        DenseGQASite(0, 1),
        DenseGQASite(2, 1),
    )
    untouched_layer_one = model.attentions[1].q_proj.weight.detach().clone()
    originals = {
        (layer, role): model.attentions[layer].projection(role).weight.detach().clone()
        for layer in (0, 2)
        for role in ("q", "k", "v", "o")
    }
    handle = _install(model, sites=sites)
    with torch.no_grad():
        for role in ("q", "k", "v", "o"):
            getattr(handle.adapter_registry, f"{role}_left").fill_(0.5)
            getattr(handle.adapter_registry, f"{role}_right").fill_(0.25)
    handle.bake_for_publication()

    assert not torch.equal(
        model.attentions[0].q_proj.weight[:4], originals[(0, "q")][:4]
    )
    assert not torch.equal(
        model.attentions[0].q_proj.weight[4:], originals[(0, "q")][4:]
    )
    assert torch.equal(model.attentions[2].q_proj.weight[:4], originals[(2, "q")][:4])
    assert not torch.equal(
        model.attentions[2].q_proj.weight[4:], originals[(2, "q")][4:]
    )
    assert torch.equal(model.attentions[1].q_proj.weight, untouched_layer_one)


def test_failed_bake_restores_live_wrappers_and_exact_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DenseToyModel().to(torch.float64)
    handle = _install(model)
    originals = {
        key: value.weight.detach().clone()
        for key, value in handle._original_projections.items()
    }
    real_count = dense_qkvo._unique_parameter_count
    calls = 0

    def wrong_after_install(module: nn.Module) -> int:
        nonlocal calls
        calls += 1
        value = real_count(module)
        return value + (1 if calls >= 2 else 0)

    monkeypatch.setattr(dense_qkvo, "_unique_parameter_count", wrong_after_install)
    with pytest.raises(DenseGQAQKVOResidualAccountingError):
        handle.bake_for_publication()

    assert getattr(model, handle.report.owner_name) is handle.adapter_registry
    assert all(
        isinstance(
            model.attentions[layer].projection(role), DenseGQAQKVOResidualProjection
        )
        for layer, role in handle._original_projections
    )
    assert all(
        torch.equal(projection.weight, originals[key])
        for key, projection in handle._original_projections.items()
    )
    monkeypatch.setattr(dense_qkvo, "_unique_parameter_count", real_count)
    handle.assert_live()
    handle.rollback()


def test_custom_projection_names_are_supported_and_bake_to_plain_linears() -> None:
    names = {"q": "query", "k": "key", "v": "value", "o": "output"}
    model = _DenseToyModel(names=names).to(torch.float64)
    handle = _install(model, projection_names=names)
    handle.bake_for_publication()
    assert all(
        type(getattr(model.attentions[1], name)) is nn.Linear for name in names.values()
    )


def test_shape_mismatch_and_aliased_bake_weights_are_rejected() -> None:
    model = _DenseToyModel()
    model.attentions[1].q_proj = nn.Linear(8, 6)
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="shape"):
        _install(model)

    model = _DenseToyModel()
    model.attentions[1].k_proj.weight = model.attentions[1].v_proj.weight
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="aliased"):
        _install(model)


@pytest.mark.parametrize("attn_implementation", ("eager", "sdpa"))
def test_real_qwen_dense_boundary_cache_bake_and_strict_reload(
    attn_implementation: str,
) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    config._attn_implementation = attn_implementation
    torch.manual_seed(61)
    model = transformers.Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    baseline = model(input_ids, use_cache=True)
    baseline_keys = set(model.state_dict())
    before_count = sum(value.numel() for value in model.parameters())
    geometry = DenseGQAGeometry(
        logical_layers=2,
        hidden_size=12,
        query_heads=4,
        key_value_heads=2,
        head_dim=4,
    )
    attentions = [layer.self_attn for layer in model.model.layers]
    handle = install_dense_gqa_qkvo_residual_adapter(
        model,
        attentions,
        geometry=geometry,
        sites=(DenseGQASite(1, 1),),
        rank=2,
        initialization_seed=9,
    )
    zero = model(input_ids, use_cache=True)
    assert torch.equal(zero.logits, baseline.logits)
    assert torch.equal(
        zero.past_key_values.layers[1].keys,
        baseline.past_key_values.layers[1].keys,
    )
    assert torch.equal(
        zero.past_key_values.layers[1].values,
        baseline.past_key_values.layers[1].values,
    )

    records: dict[str, Tensor] = {}
    attention = attentions[1]

    def record_output(name: str):
        def hook(_module: nn.Module, _inputs, output: Tensor) -> None:
            records[name] = output.detach().clone()

        return hook

    def record_input(name: str):
        def hook(_module: nn.Module, inputs) -> None:
            records[name] = inputs[0].detach().clone()

        return hook

    hooks = (
        attention.q_proj.register_forward_hook(record_output("q_projection")),
        attention.k_proj.register_forward_hook(record_output("k_projection")),
        attention.q_norm.register_forward_pre_hook(record_input("q_norm")),
        attention.k_norm.register_forward_pre_hook(record_input("k_norm")),
        attention.o_proj.register_forward_pre_hook(record_input("o_context")),
    )
    try:
        with torch.no_grad():
            for role_index, role in enumerate(("q", "k", "v", "o"), start=1):
                getattr(handle.adapter_registry, f"{role}_left").fill_(
                    role_index * 0.02
                )
                getattr(handle.adapter_registry, f"{role}_right").fill_(
                    role_index * 0.03
                )
        active = model(input_ids, use_cache=True)
    finally:
        for hook in hooks:
            hook.remove()

    assert torch.equal(
        records["q_norm"],
        records["q_projection"].reshape(1, 4, 4, 4),
    )
    assert torch.equal(
        records["k_norm"],
        records["k_projection"].reshape(1, 4, 2, 4),
    )
    assert tuple(records["o_context"].shape) == (1, 4, 16)
    assert not torch.equal(
        active.past_key_values.layers[1].keys,
        baseline.past_key_values.layers[1].keys,
    )
    assert not torch.equal(
        active.past_key_values.layers[1].values,
        baseline.past_key_values.layers[1].values,
    )

    bake = handle.bake_for_publication()
    baked = model(input_ids, use_cache=True)
    torch.testing.assert_close(baked.logits, active.logits, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        baked.past_key_values.layers[1].keys,
        active.past_key_values.layers[1].keys,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        baked.past_key_values.layers[1].values,
        active.past_key_values.layers[1].values,
        rtol=2e-5,
        atol=2e-5,
    )
    assert bake.model_parameter_count_after_bake == before_count
    assert set(model.state_dict()) == baseline_keys
    assert all(
        type(getattr(attention, f"{role}_proj")) is nn.Linear
        for role in ("q", "k", "v", "o")
    )

    incremental_cache = None
    incremental_rows = []
    for token_index in range(input_ids.shape[1]):
        step = model(
            input_ids[:, token_index : token_index + 1],
            past_key_values=incremental_cache,
            use_cache=True,
        )
        incremental_cache = step.past_key_values
        incremental_rows.append(step.logits)
    torch.testing.assert_close(
        torch.cat(incremental_rows, dim=1),
        baked.logits,
        rtol=2e-5,
        atol=2e-5,
    )

    fresh = transformers.Qwen3ForCausalLM(config).eval()
    fresh.load_state_dict(model.state_dict(), strict=True)
    reloaded = fresh(input_ids, use_cache=True)
    torch.testing.assert_close(reloaded.logits, baked.logits, rtol=0.0, atol=0.0)
