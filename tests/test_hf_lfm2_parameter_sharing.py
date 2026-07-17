from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_parameter_sharing as sharing  # noqa: E402
from neural_forge.hf_lfm2_parameter_sharing import (  # noqa: E402
    Lfm2ParameterSharingEvidenceError,
    Lfm2ParameterSharingTopologyError,
    Lfm2ParameterSharingTransactionError,
    install_lfm2_parameter_sharing_plan,
    lfm2_ffn_parameter_sharing_savings,
    plan_lfm2_activation_aligned_parameter_sharing,
    seal_lfm2_activation_evidence,
)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"
_CACHE_KEY = "a" * 64
_CORPUS_KEY = "b" * 64


class _Config(SimpleNamespace):
    def to_dict(self) -> dict[str, object]:
        return dict(vars(self))


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float64))

    def forward(self, inputs):
        variance = inputs.square().mean(dim=-1, keepdim=True)
        return inputs * torch.rsqrt(variance + 1e-6) * self.weight


class Lfm2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=torch.float64,
        )
        self.w3 = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=torch.float64,
        )
        self.w2 = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=torch.float64,
        )

    def forward(self, inputs):
        return self.w2(torch.nn.functional.silu(self.w1(inputs)) * self.w3(inputs))


class Lfm2ShortConv(nn.Module):
    def __init__(self, config: _Config, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            config.conv_L_cache,
            groups=hidden_size,
            bias=config.conv_bias,
            padding=config.conv_L_cache - 1,
            dtype=torch.float64,
        )
        self.in_proj = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=config.conv_bias,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=config.conv_bias,
            dtype=torch.float64,
        )
        self.register_buffer(
            "recurrent_states",
            torch.zeros(hidden_size, config.conv_L_cache, dtype=torch.float64),
        )

    def forward(self, inputs):
        projected = self.in_proj(inputs).transpose(1, 2)
        b_values, c_values, x_values = projected.chunk(3, dim=1)
        filtered = self.conv(b_values * x_values)[..., : inputs.shape[1]]
        return self.out_proj((c_values * filtered).transpose(1, 2))


class Lfm2Attention(nn.Module):
    def __init__(self, config: _Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.float64,
        )
        self.q_layernorm = Lfm2RMSNorm(self.head_dim)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim)

    def forward(self, inputs):
        # The actual q/k/v Parameters execute.  This small test module avoids
        # building a cache while retaining the concrete validated topology.
        query = self.q_proj(inputs)
        key_scale = self.k_proj(inputs).square().mean(dim=-1, keepdim=True)
        value_scale = self.v_proj(inputs).mean(dim=-1, keepdim=True)
        return self.out_proj(query) + 0.01 * (key_scale + value_scale)


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: _Config, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)

    def forward(self, inputs):
        operator = self.self_attn if self.is_attention_layer else self.conv
        hidden = inputs + operator(self.operator_norm(inputs))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


for _class in (
    Lfm2RMSNorm,
    Lfm2MLP,
    Lfm2ShortConv,
    Lfm2Attention,
    Lfm2DecoderLayer,
):
    _class.__module__ = _LFM2_MODULE


class _Backbone(nn.Module):
    def __init__(self, config: _Config):
        super().__init__()
        self.layers = nn.ModuleList(
            Lfm2DecoderLayer(config, index) for index in range(len(config.layer_types))
        )

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _TinyLfm(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 4,
        intermediate_size: int = 6,
        layer_types: tuple[str, ...] = (
            "conv",
            "full_attention",
            "conv",
            "full_attention",
        ),
    ):
        super().__init__()
        self.config = _Config(
            model_type="lfm2",
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            conv_L_cache=3,
            conv_bias=False,
            layer_types=list(layer_types),
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=len(layer_types),
        )
        self.model = _Backbone(self.config)
        self.forward_calls = 0

    def forward(self, hidden):
        self.forward_calls += 1
        return self.model(hidden)


def _model(seed: int = 100) -> _TinyLfm:
    torch.manual_seed(seed)
    return _TinyLfm().eval()


def _heldout(seed: int = 200):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(3, 5, 4, generator=generator, dtype=torch.float64)


def _capture_post_swiglu(model: _TinyLfm, hidden) -> tuple[object, dict[int, object]]:
    captured: dict[int, list[object]] = {
        index: [] for index in range(len(model.model.layers))
    }
    handles = []
    for index, layer in enumerate(model.model.layers):

        def capture(module, arguments, *, layer_index=index):
            del module
            captured[layer_index].append(
                arguments[0]
                .detach()
                .reshape(
                    -1,
                    model.config.intermediate_size,
                )
            )

        handles.append(layer.feed_forward.w2.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            output = model(hidden)
    finally:
        for handle in handles:
            handle.remove()
    rows = {index: torch.cat(values, dim=0) for index, values in captured.items()}
    return output, rows


def _evidence(model: _TinyLfm, hidden=None):
    if hidden is None:
        hidden = _heldout()
    output, rows = _capture_post_swiglu(model, hidden)
    evidence = seal_lfm2_activation_evidence(
        model,
        rows,
        cache_key=_CACHE_KEY,
        heldout_corpus_sha256=_CORPUS_KEY,
    )
    return output, rows, evidence


def _parameter_objects(model: _TinyLfm) -> dict[str, object]:
    return dict(model.named_parameters(remove_duplicate=False))


def test_exact_230m_three_pair_savings_is_23592960() -> None:
    assert lfm2_ffn_parameter_sharing_savings(1024, 2560, 3) == 23_592_960


def test_deterministic_similarity_selection_uses_disjoint_pairs() -> None:
    model = _model(301)
    rows_a = torch.randn(24, 6, generator=torch.Generator().manual_seed(302))
    rows_b = torch.randn(24, 6, generator=torch.Generator().manual_seed(303))
    with torch.no_grad():
        model.model.layers[1].feed_forward.w2.weight.copy_(
            model.model.layers[0].feed_forward.w2.weight
        )
        model.model.layers[3].feed_forward.w2.weight.copy_(
            model.model.layers[2].feed_forward.w2.weight
        )
    rows = {
        0: rows_a,
        1: rows_a.clone(),
        2: rows_b,
        3: rows_b.clone(),
    }
    evidence = seal_lfm2_activation_evidence(
        model,
        rows,
        cache_key=_CACHE_KEY,
        heldout_corpus_sha256=_CORPUS_KEY,
    )

    first = plan_lfm2_activation_aligned_parameter_sharing(
        model,
        evidence,
        pair_count=2,
        exact_assignment_limit=6,
    )
    second = plan_lfm2_activation_aligned_parameter_sharing(
        model,
        evidence,
        pair_count=2,
        exact_assignment_limit=6,
    )

    assert first.selected_pairs == ((0, 1), (2, 3))
    assert second.selected_pairs == first.selected_pairs
    assert len(first.plan_sha256) == 64
    assert second.plan_sha256 == first.plan_sha256
    assert first.as_dict()["plan_sha256"] == first.plan_sha256
    assert all(
        pair.similarity.combined_similarity == pytest.approx(1.0)
        for pair in first.pair_plans
    )
    assert all(
        torch.equal(left.permutation, right.permutation)
        for left, right in zip(first.pair_plans, second.pair_plans)
    )
    assert len({index for pair in first.selected_pairs for index in pair}) == 4


def test_real_forward_install_counts_aliases_and_exact_rollback() -> None:
    model = _model(401)
    inputs = _heldout(402)
    before_output, _, evidence = _evidence(model, inputs)
    forward_calls_after_capture = model.forward_calls
    parameters_before = _parameter_objects(model)
    layer_ids = tuple(id(layer) for layer in model.model.layers)
    ffn_ids = tuple(id(layer.feed_forward) for layer in model.model.layers)
    operator_ids = tuple(
        id(layer.self_attn if layer.is_attention_layer else layer.conv)
        for layer in model.model.layers
    )
    buffer_snapshot = {
        name: (id(value), value.detach().clone())
        for name, value in model.named_buffers(remove_duplicate=False)
    }

    plan = plan_lfm2_activation_aligned_parameter_sharing(
        model,
        evidence,
        pair_count=2,
        layer_pairs=((0, 1), (2, 3)),
        merge_weights=(1.0, 3.0),
        exact_assignment_limit=6,
    )
    assert model.forward_calls == forward_calls_after_capture
    assert plan.merge_weights == pytest.approx((0.25, 0.75))
    expected_savings = 2 * 3 * 4 * 6
    assert plan.projected_parameters_saved == expected_savings

    installed = install_lfm2_parameter_sharing_plan(model, evidence, plan)
    assert installed.report.parameters_saved == expected_savings
    assert installed.report.logical_parameters == plan.logical_parameters
    assert (
        installed.report.unique_parameters_before
        - installed.report.unique_parameters_after
        == expected_savings
    )
    assert installed.report.tied_parameter_objects == 6
    assert installed.report.aligned_neurons_per_pair == 6
    assert installed.report.training_steps == 0
    assert installed.report.gradients_used is False
    assert model.forward_calls == forward_calls_after_capture

    for pair_plan in plan.pair_plans:
        left, right = pair_plan.layers
        for role, bundle_role in (
            ("w1", "gate_weight"),
            ("w3", "up_weight"),
            ("w2", "down_weight"),
        ):
            left_parameter = getattr(
                model.model.layers[left].feed_forward,
                role,
            ).weight
            right_parameter = getattr(
                model.model.layers[right].feed_forward,
                role,
            ).weight
            assert left_parameter is right_parameter
            torch.testing.assert_close(
                left_parameter.detach(),
                getattr(pair_plan.merged_bundle, bundle_role).to(
                    dtype=left_parameter.dtype
                ),
            )
    assert (
        model.model.layers[0].feed_forward.w1.weight
        is not model.model.layers[0].feed_forward.w3.weight
    )
    assert (
        model.model.layers[0].feed_forward.w1.weight
        is not model.model.layers[2].feed_forward.w1.weight
    )
    assert tuple(id(layer) for layer in model.model.layers) == layer_ids
    assert tuple(id(layer.feed_forward) for layer in model.model.layers) == ffn_ids
    assert (
        tuple(
            id(layer.self_attn if layer.is_attention_layer else layer.conv)
            for layer in model.model.layers
        )
        == operator_ids
    )
    for name, value in model.named_buffers(remove_duplicate=False):
        original_id, original_value = buffer_snapshot[name]
        assert id(value) == original_id
        torch.testing.assert_close(value, original_value)

    with torch.no_grad():
        after_output = model(inputs)
    assert bool(torch.isfinite(after_output).all())
    assert not torch.equal(after_output, before_output)

    assert installed.rollback() is True
    assert installed.active is False
    assert installed.rollback() is False
    parameters_restored = _parameter_objects(model)
    assert parameters_restored.keys() == parameters_before.keys()
    assert all(
        parameters_restored[name] is parameter
        for name, parameter in parameters_before.items()
    )
    with torch.no_grad():
        rolled_back_output = model(inputs)
    torch.testing.assert_close(
        rolled_back_output,
        before_output,
        atol=0.0,
        rtol=0.0,
    )


def test_missing_modified_and_stale_cache_evidence_fail_closed() -> None:
    model = _model(501)
    _, rows, evidence = _evidence(model, _heldout(502))
    missing = dict(rows)
    missing.pop(3)
    with pytest.raises(
        Lfm2ParameterSharingEvidenceError,
        match="cover every layer",
    ):
        seal_lfm2_activation_evidence(
            model,
            missing,
            cache_key=_CACHE_KEY,
            heldout_corpus_sha256=_CORPUS_KEY,
        )
    with pytest.raises(
        Lfm2ParameterSharingEvidenceError,
        match="sealed",
    ):
        plan_lfm2_activation_aligned_parameter_sharing(
            model,
            None,
            pair_count=1,
        )

    evidence.layer_rows[0][0, 0].add_(1.0)
    with pytest.raises(
        Lfm2ParameterSharingEvidenceError,
        match="changed after evidence sealing",
    ):
        plan_lfm2_activation_aligned_parameter_sharing(
            model,
            evidence,
            pair_count=1,
        )

    _, _, fresh = _evidence(model, _heldout(503))
    with torch.no_grad():
        model.model.layers[0].feed_forward.w1.weight.add_(0.01)
    with pytest.raises(Lfm2ParameterSharingEvidenceError, match="stale"):
        plan_lfm2_activation_aligned_parameter_sharing(
            model,
            fresh,
            pair_count=1,
        )


def test_overlapping_pairs_and_partial_aliasing_are_rejected() -> None:
    model = _model(601)
    _, _, evidence = _evidence(model, _heldout(602))
    with pytest.raises(
        Lfm2ParameterSharingTopologyError,
        match="Overlapping",
    ):
        plan_lfm2_activation_aligned_parameter_sharing(
            model,
            evidence,
            pair_count=2,
            layer_pairs=((0, 1), (1, 2)),
        )

    partial = _model(603)
    partial.model.layers[1].feed_forward.w1.weight = partial.model.layers[
        0
    ].feed_forward.w1.weight
    _, partial_rows = _capture_post_swiglu(partial, _heldout(604))
    with pytest.raises(
        Lfm2ParameterSharingTopologyError,
        match="partial aliasing|shared FFN storage",
    ):
        seal_lfm2_activation_evidence(
            partial,
            partial_rows,
            cache_key=_CACHE_KEY,
            heldout_corpus_sha256=_CORPUS_KEY,
        )


def test_non_lfm_topology_is_rejected() -> None:
    model = _model(701)
    _, rows = _capture_post_swiglu(model, _heldout(702))
    model.config.model_type = "not_lfm2"
    with pytest.raises(Lfm2ParameterSharingTopologyError, match="model_type"):
        seal_lfm2_activation_evidence(
            model,
            rows,
            cache_key=_CACHE_KEY,
            heldout_corpus_sha256=_CORPUS_KEY,
        )


def test_multi_pair_install_failure_rolls_back_the_first_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(801)
    inputs = _heldout(802)
    before_output, _, evidence = _evidence(model, inputs)
    parameters_before = _parameter_objects(model)
    plan = plan_lfm2_activation_aligned_parameter_sharing(
        model,
        evidence,
        pair_count=2,
        layer_pairs=((0, 1), (2, 3)),
        exact_assignment_limit=6,
    )
    real_tie = sharing.tie_mlp_modules
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-pair failure")
        return real_tie(*args, **kwargs)

    monkeypatch.setattr(sharing, "tie_mlp_modules", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        install_lfm2_parameter_sharing_plan(model, evidence, plan)

    parameters_after = _parameter_objects(model)
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )
    with torch.no_grad():
        after_output = model(inputs)
    torch.testing.assert_close(after_output, before_output, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "interruption",
    (KeyboardInterrupt, SystemExit),
)
def test_base_exception_during_multi_pair_install_rolls_back_exact_objects(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    model = _model(901)
    inputs = _heldout(902)
    before_output, _, evidence = _evidence(model, inputs)
    parameters_before = _parameter_objects(model)
    plan = plan_lfm2_activation_aligned_parameter_sharing(
        model,
        evidence,
        pair_count=2,
        layer_pairs=((0, 1), (2, 3)),
        exact_assignment_limit=6,
    )
    real_tie = sharing.tie_mlp_modules
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption("injected base exception")
        return real_tie(*args, **kwargs)

    monkeypatch.setattr(sharing, "tie_mlp_modules", interrupt_second)
    with pytest.raises(interruption, match="injected base exception"):
        install_lfm2_parameter_sharing_plan(model, evidence, plan)

    parameters_after = _parameter_objects(model)
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )
    with torch.no_grad():
        after_output = model(inputs)
    torch.testing.assert_close(
        after_output,
        before_output,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("tamper", ("merged_tensor", "permutation"))
def test_finite_plan_tensor_tamper_fails_before_any_model_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    model = _model(1001)
    inputs = _heldout(1002)
    before_output, _, evidence = _evidence(model, inputs)
    parameters_before = _parameter_objects(model)
    plan = plan_lfm2_activation_aligned_parameter_sharing(
        model,
        evidence,
        pair_count=1,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )
    with torch.no_grad():
        if tamper == "merged_tensor":
            plan.pair_plans[0].merged_bundle.gate_weight[0, 0].add_(0.125)
        else:
            permutation = plan.pair_plans[0].permutation
            first = permutation[0].clone()
            permutation[0].copy_(permutation[1])
            permutation[1].copy_(first)
    tie_calls = 0

    def forbidden_tie(*_args, **_kwargs):
        nonlocal tie_calls
        tie_calls += 1
        raise AssertionError("mutation began before plan seal validation")

    monkeypatch.setattr(sharing, "tie_mlp_modules", forbidden_tie)
    with pytest.raises(
        Lfm2ParameterSharingTransactionError,
        match="sealed sharing plan changed",
    ):
        install_lfm2_parameter_sharing_plan(model, evidence, plan)

    assert tie_calls == 0
    parameters_after = _parameter_objects(model)
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )
    with torch.no_grad():
        after_output = model(inputs)
    torch.testing.assert_close(
        after_output,
        before_output,
        atol=0.0,
        rtol=0.0,
    )
