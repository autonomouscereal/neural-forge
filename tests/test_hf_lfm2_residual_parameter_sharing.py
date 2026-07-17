from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_residual_parameter_sharing as residual  # noqa: E402
from neural_forge.hf_lfm2_parameter_sharing import (  # noqa: E402
    Lfm2ParameterSharingEvidenceError,
    Lfm2ParameterSharingTopologyError,
    seal_lfm2_activation_evidence,
)
from neural_forge.hf_lfm2_residual_parameter_sharing import (  # noqa: E402
    Lfm2ResidualParameterSharingTopologyError,
    Lfm2ResidualParameterSharingTransactionError,
    Lfm2SharedBaseResidualLinear,
    install_lfm2_residual_parameter_sharing_plan,
    lfm2_residual_parameter_sharing_savings,
    plan_lfm2_residual_parameter_sharing,
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
        self.feed_forward = Lfm2MLP(config.hidden_size, config.intermediate_size)
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
    def __init__(self):
        super().__init__()
        self.config = _Config(
            model_type="lfm2",
            hidden_size=4,
            intermediate_size=6,
            conv_L_cache=3,
            conv_bias=False,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=4,
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


def _capture_post_swiglu(model: _TinyLfm, hidden):
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
                .reshape(-1, model.config.intermediate_size)
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


def _module_snapshot(model: _TinyLfm) -> dict[tuple[int, str], object]:
    return {
        (index, role): getattr(layer.feed_forward, role)
        for index, layer in enumerate(model.model.layers)
        for role in ("w1", "w3", "w2")
    }


def test_exact_lfm2_230m_savings_for_large_ranked_targets() -> None:
    assert (
        lfm2_residual_parameter_sharing_savings(1024, 2560, 4, 64)
        == 28_704_768
    )
    assert (
        lfm2_residual_parameter_sharing_savings(1024, 2560, 7, 96)
        == 47_824_896
    )


@pytest.mark.parametrize("preserve", ["earlier", "later"])
def test_full_rank_is_functionally_equivalent_and_rolls_back_exactly(
    preserve: str,
) -> None:
    model = _model(301)
    inputs = _heldout(302)
    before_output, _, evidence = _evidence(model, inputs)
    modules_before = _module_snapshot(model)
    parameters_before = _parameter_objects(model)
    forward_calls = model.forward_calls

    plan = plan_lfm2_residual_parameter_sharing(
        model,
        evidence,
        pair_count=1,
        layer_pairs=((0, 1),),
        rank=4,
        preserve=preserve,
        exact_assignment_limit=6,
    )
    assert model.forward_calls == forward_calls
    installed = install_lfm2_residual_parameter_sharing_plan(
        model,
        evidence,
        plan,
    )
    assert model.forward_calls == forward_calls
    pair = plan.pair_plans[0]
    assert pair.preserve == preserve
    assert all(
        role.relative_frobenius_error < 1e-12
        for role in pair.role_plans
    )
    for role in ("w1", "w3", "w2"):
        preserved = getattr(
            model.model.layers[pair.preserved_layer].feed_forward,
            role,
        )
        approximated = getattr(
            model.model.layers[pair.approximated_layer].feed_forward,
            role,
        )
        assert preserved is modules_before[(pair.preserved_layer, role)]
        assert isinstance(approximated, Lfm2SharedBaseResidualLinear)
        assert approximated.weight is preserved.weight

    with torch.no_grad():
        after_output = model(inputs)
    torch.testing.assert_close(after_output, before_output, atol=1e-11, rtol=1e-11)
    assert installed.report.training_steps == 0
    assert installed.report.gradients_used is False
    assert all(parameter.grad is None for parameter in model.parameters())

    assert installed.rollback() is True
    assert installed.rollback() is False
    assert _module_snapshot(model) == modules_before
    parameters_after = _parameter_objects(model)
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )
    with torch.no_grad():
        rolled_back = model(inputs)
    torch.testing.assert_close(rolled_back, before_output, atol=0.0, rtol=0.0)


def test_low_rank_later_orientation_counts_and_preserves_all_operators() -> None:
    model = _model(401)
    inputs = _heldout(402)
    _, _, evidence = _evidence(model, inputs)
    modules_before = _module_snapshot(model)
    operators_before = tuple(
        layer.self_attn if layer.is_attention_layer else layer.conv
        for layer in model.model.layers
    )
    operator_parameters_before = {
        id(parameter): parameter.detach().clone()
        for operator in operators_before
        for parameter in operator.parameters()
    }
    buffers_before = {
        name: (value, value.detach().clone())
        for name, value in model.named_buffers(remove_duplicate=False)
    }

    plan = plan_lfm2_residual_parameter_sharing(
        model,
        evidence,
        pair_count=2,
        layer_pairs=((0, 1), (2, 3)),
        rank=1,
        preserve=("later", "later"),
        exact_assignment_limit=6,
    )
    expected_savings = 2 * (3 * 4 * 6 - 3 * (4 + 6))
    assert plan.projected_parameters_saved == expected_savings
    installed = install_lfm2_residual_parameter_sharing_plan(
        model,
        evidence,
        plan,
    )
    assert installed.report.parameters_saved == expected_savings
    assert (
        installed.report.unique_parameters_before
        - installed.report.unique_parameters_after
        == expected_savings
    )
    assert installed.report.logical_parameters_after == (
        installed.report.logical_parameters_before + 2 * 3 * (4 + 6)
    )
    assert installed.report.preserved_layers == (1, 3)
    assert installed.report.approximated_layers == (0, 2)
    assert installed.report.shared_base_aliases == 6

    for index in (1, 3):
        for role in ("w1", "w3", "w2"):
            assert (
                getattr(model.model.layers[index].feed_forward, role)
                is modules_before[(index, role)]
            )
    for index in (0, 2):
        for role in ("w1", "w3", "w2"):
            module = getattr(model.model.layers[index].feed_forward, role)
            assert isinstance(module, Lfm2SharedBaseResidualLinear)
            assert module.residual_rank == 1
    assert tuple(
        layer.self_attn if layer.is_attention_layer else layer.conv
        for layer in model.model.layers
    ) == operators_before
    for operator in operators_before:
        for parameter in operator.parameters():
            torch.testing.assert_close(
                parameter,
                operator_parameters_before[id(parameter)],
                atol=0.0,
                rtol=0.0,
            )
    for name, value in model.named_buffers(remove_duplicate=False):
        original, contents = buffers_before[name]
        assert value is original
        torch.testing.assert_close(value, contents, atol=0.0, rtol=0.0)
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        output = model(inputs)
    assert bool(torch.isfinite(output).all())


@pytest.mark.parametrize("rank", [0, -1, True, 5])
def test_invalid_ranks_fail_closed(rank: object) -> None:
    model = _model(501)
    _, _, evidence = _evidence(model, _heldout(502))
    with pytest.raises(Lfm2ResidualParameterSharingTopologyError, match="rank"):
        plan_lfm2_residual_parameter_sharing(
            model,
            evidence,
            pair_count=1,
            layer_pairs=((0, 1),),
            rank=rank,  # type: ignore[arg-type]
            exact_assignment_limit=6,
        )


def test_stale_evidence_and_overlapping_or_partial_aliases_are_rejected() -> None:
    model = _model(601)
    _, _, evidence = _evidence(model, _heldout(602))
    plan = plan_lfm2_residual_parameter_sharing(
        model,
        evidence,
        pair_count=1,
        layer_pairs=((0, 1),),
        rank=1,
        exact_assignment_limit=6,
    )
    modules_before = _module_snapshot(model)
    with torch.no_grad():
        model.model.layers[0].feed_forward.w1.weight.add_(0.01)
    with pytest.raises(Lfm2ParameterSharingEvidenceError, match="stale"):
        install_lfm2_residual_parameter_sharing_plan(model, evidence, plan)
    assert _module_snapshot(model) == modules_before

    overlap = _model(603)
    _, _, overlap_evidence = _evidence(overlap, _heldout(604))
    with pytest.raises(Lfm2ParameterSharingTopologyError, match="Overlapping"):
        plan_lfm2_residual_parameter_sharing(
            overlap,
            overlap_evidence,
            pair_count=2,
            layer_pairs=((0, 1), (1, 2)),
            rank=1,
            exact_assignment_limit=6,
        )

    partial = _model(605)
    partial.model.layers[1].feed_forward.w1.weight = partial.model.layers[
        0
    ].feed_forward.w1.weight
    _, rows = _capture_post_swiglu(partial, _heldout(606))
    with pytest.raises(
        Lfm2ParameterSharingTopologyError,
        match="partial aliasing|shared FFN storage",
    ):
        seal_lfm2_activation_evidence(
            partial,
            rows,
            cache_key=_CACHE_KEY,
            heldout_corpus_sha256=_CORPUS_KEY,
        )


def test_nonfinite_svd_fails_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(701)
    _, _, evidence = _evidence(model, _heldout(702))
    modules_before = _module_snapshot(model)
    real_svd = torch.linalg.svd

    def nonfinite_svd(*args, **kwargs):
        left, values, right = real_svd(*args, **kwargs)
        values[0] = float("nan")
        return left, values, right

    monkeypatch.setattr(torch.linalg, "svd", nonfinite_svd)
    with pytest.raises(
        Lfm2ResidualParameterSharingTransactionError,
        match="non-finite",
    ):
        plan_lfm2_residual_parameter_sharing(
            model,
            evidence,
            pair_count=1,
            layer_pairs=((0, 1),),
            rank=1,
            exact_assignment_limit=6,
        )
    assert _module_snapshot(model) == modules_before


def test_injected_multi_pair_install_failure_restores_exact_modules_and_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(801)
    inputs = _heldout(802)
    before_output, _, evidence = _evidence(model, inputs)
    modules_before = _module_snapshot(model)
    parameters_before = _parameter_objects(model)
    plan = plan_lfm2_residual_parameter_sharing(
        model,
        evidence,
        pair_count=2,
        layer_pairs=((0, 1), (2, 3)),
        rank=1,
        exact_assignment_limit=6,
    )
    real_replace = residual._replace_projection
    calls = 0

    def fail_on_fourth(owner, role, module):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("injected second-pair failure")
        real_replace(owner, role, module)

    monkeypatch.setattr(residual, "_replace_projection", fail_on_fourth)
    with pytest.raises(RuntimeError, match="injected"):
        install_lfm2_residual_parameter_sharing_plan(model, evidence, plan)

    assert _module_snapshot(model) == modules_before
    parameters_after = _parameter_objects(model)
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )
    with torch.no_grad():
        output = model(inputs)
    torch.testing.assert_close(output, before_output, atol=0.0, rtol=0.0)


def test_modified_residual_factor_is_rejected_without_mutation() -> None:
    model = _model(901)
    _, _, evidence = _evidence(model, _heldout(902))
    plan = plan_lfm2_residual_parameter_sharing(
        model,
        evidence,
        pair_count=1,
        layer_pairs=((0, 1),),
        rank=1,
        exact_assignment_limit=6,
    )
    modules_before = _module_snapshot(model)
    plan.pair_plans[0].role_plans[0].residual_left[0, 0].add_(1.0)
    with pytest.raises(
        Lfm2ResidualParameterSharingTransactionError,
        match="changed after planning",
    ):
        install_lfm2_residual_parameter_sharing_plan(model, evidence, plan)
    assert _module_snapshot(model) == modules_before
