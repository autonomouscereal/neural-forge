from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn
pytest.importorskip("transformers")

from neural_forge.hf_lfm2_variable_width_fold import (  # noqa: E402
    Lfm2LayerWidthCurve,
    Lfm2VariableWidthFoldError,
    Lfm2WidthCostPoint,
    allocate_lfm2_variable_widths,
    install_lfm2_variable_width_fold_plan,
    plan_lfm2_variable_width_fold,
    seal_lfm2_variable_width_evidence,
)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"
_LAYERS = 14
_HIDDEN = 4
_INTERMEDIATE = 4
_SAVINGS_PER_NEURON = 3 * _HIDDEN


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float64))


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

    def forward(self, hidden_states):
        return self.w2(torch.nn.functional.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class Lfm2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.float64,
        )
        self.q_layernorm = Lfm2RMSNorm(config.head_dim)
        self.k_layernorm = Lfm2RMSNorm(config.head_dim)


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.is_attention_layer = True
        self.self_attn = Lfm2Attention(config, layer_idx)
        self.feed_forward = Lfm2MLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)


for _class in (Lfm2RMSNorm, Lfm2MLP, Lfm2Attention, Lfm2DecoderLayer):
    _class.__module__ = _LFM2_MODULE


class _Body(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=torch.float64,
        )
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(config, index) for index in range(_LAYERS)]
        )
        self.norm = Lfm2RMSNorm(config.hidden_size)


class _TinyLfmCausal(nn.Module):
    def __init__(self, config: SimpleNamespace, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = _Body(config)
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=torch.float64,
            )
        self.lm_head.weight = self.model.embed_tokens.weight
        self.config = config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _Config(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


def _config():
    return _Config(
        model_type="lfm2",
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        block_ff_dim=_INTERMEDIATE,
        block_auto_adjust_ff_dim=False,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_heads=2,
        head_dim=2,
        layer_types=["full_attention"] * _LAYERS,
        conv_L_cache=3,
        conv_bias=False,
        vocab_size=11,
    )


def _state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_state(model, expected):
    observed = model.state_dict()
    assert set(observed) == set(expected)
    for name, value in expected.items():
        assert torch.equal(observed[name].detach().cpu(), value), name


def _activation_rows(parent):
    generator = torch.Generator().manual_seed(901)
    hidden = torch.randn(
        12,
        _HIDDEN,
        generator=generator,
        dtype=torch.float64,
    )
    result = {}
    with torch.no_grad():
        for index, layer in enumerate(parent.model.layers):
            gate = layer.feed_forward.w1(hidden)
            up = layer.feed_forward.w3(hidden)
            result[index] = {
                "gate": gate,
                "up": up,
                "post_swiglu": torch.nn.functional.silu(gate) * up,
            }
    return result


def _curves():
    result = []
    for index in range(_LAYERS):
        if index == 0:
            costs = ((0.0, 0.0), (0.04, 0.06), (0.08, 0.12))
        elif index == 1:
            costs = ((0.0, 0.0), (0.06, 0.09), (0.12, 0.18))
        else:
            costs = ((0.0, 0.0), (5.0 + index, 5.0), (10.0 + index, 10.0))
        result.append(
            Lfm2LayerWidthCurve(
                layer_index=index,
                points=tuple(
                    Lfm2WidthCostPoint(
                        width=width,
                        whole_trajectory_cost=cost[0],
                        residual_cost=cost[1],
                    )
                    for width, cost in zip((4, 3, 2), costs, strict=True)
                ),
            )
        )
    return tuple(result)


def _evidence(parent):
    return seal_lfm2_variable_width_evidence(
        parent,
        _activation_rows(parent),
        _curves(),
        cache_key="a" * 64,
        heldout_corpus_sha256="b" * 64,
        evidence_builder_id="test.variable-width.curves.v1",
    )


def _plan(parent, evidence, *, saved=2 * _SAVINGS_PER_NEURON, ridge=1e-8):
    return plan_lfm2_variable_width_fold(
        parent,
        evidence,
        target_parameters_saved=saved,
        minimum_width=2,
        maximum_width_step=1,
        cluster_ridge=ridge,
    )


def test_allocator_spends_the_budget_on_low_cost_layers() -> None:
    parent = _TinyLfmCausal(_config(), seed=10)
    evidence = _evidence(parent)

    allocation = allocate_lfm2_variable_widths(
        evidence,
        target_parameters_saved=2 * _SAVINGS_PER_NEURON,
        minimum_width=2,
        maximum_width_step=1,
    )

    assert allocation.target_widths == (2,) + (4,) * 13
    assert allocation.parameters_saved == 2 * _SAVINGS_PER_NEURON
    assert allocation.weighted_validation_cost == pytest.approx(0.2)
    assert len(allocation.allocation_sha256) == 64


def test_nonuniform_shapes_exact_accounting_and_parent_tensor_copy() -> None:
    parent = _TinyLfmCausal(_config(), seed=20)
    child = _TinyLfmCausal(_config(), seed=21)
    evidence = _evidence(parent)
    plan = _plan(
        parent,
        evidence,
        saved=3 * _SAVINGS_PER_NEURON,
    )
    blank = _state(child)
    original_ffns = tuple(
        layer.feed_forward for layer in child.model.layers
    )
    correction_order = []

    def correct(context):
        correction_order.append(context.layer_index)
        return None

    transaction = install_lfm2_variable_width_fold_plan(
        parent,
        child,
        evidence,
        plan,
        residual_corrector=correct,
    )

    assert plan.target_widths == (2, 3) + (4,) * 12
    assert len(plan.target_widths) == 14
    assert child.model.layers[0].feed_forward.w1.weight.shape == (2, 4)
    assert child.model.layers[0].feed_forward.w3.weight.shape == (2, 4)
    assert child.model.layers[0].feed_forward.w2.weight.shape == (4, 2)
    assert child.model.layers[1].feed_forward.w1.weight.shape == (3, 4)
    assert child.model.layers[2].feed_forward.w1.weight.shape == (4, 4)
    assert transaction.report.parameters_saved == 3 * _SAVINGS_PER_NEURON
    assert (
        transaction.report.unique_parameters_before
        - transaction.report.unique_parameters_after
        == 3 * _SAVINGS_PER_NEURON
    )
    assert correction_order == list(range(14))
    assert torch.equal(
        child.model.layers[7].self_attn.q_proj.weight,
        parent.model.layers[7].self_attn.q_proj.weight,
    )
    assert transaction.rollback() is True
    _assert_state(child, blank)
    assert tuple(
        layer.feed_forward for layer in child.model.layers
    ) == original_ffns


def test_plan_is_deterministic_and_tensor_tamper_fails_before_mutation() -> None:
    parent = _TinyLfmCausal(_config(), seed=30)
    child = _TinyLfmCausal(_config(), seed=31)
    evidence = _evidence(parent)
    first = _plan(parent, evidence)
    second = _plan(parent, evidence)

    assert first.plan_sha256 == second.plan_sha256
    assert tuple(
        layer.medoid_source_neurons for layer in first.layer_plans
    ) == tuple(
        layer.medoid_source_neurons for layer in second.layer_plans
    )

    blank = _state(child)
    with torch.no_grad():
        first.layer_plans[0].gate_weight[0, 0].add_(1.0)
    with pytest.raises(Lfm2VariableWidthFoldError):
        install_lfm2_variable_width_fold_plan(
            parent,
            child,
            evidence,
            first,
        )
    _assert_state(child, blank)
    assert not hasattr(
        child,
        "_neural_forge_variable_width_fold_transaction",
    )


def test_activation_evidence_tamper_is_rejected() -> None:
    parent = _TinyLfmCausal(_config(), seed=40)
    evidence = _evidence(parent)
    with torch.no_grad():
        evidence.layer_activations[0].gate[0, 0].add_(1.0)

    with pytest.raises(Lfm2VariableWidthFoldError):
        _plan(parent, evidence)


def test_redundant_coupled_neurons_fold_to_matching_toy_behavior() -> None:
    parent = _TinyLfmCausal(_config(), seed=50)
    child = _TinyLfmCausal(_config(), seed=51)
    layer = parent.model.layers[0].feed_forward
    with torch.no_grad():
        gate_a = torch.tensor([0.4, -0.2, 0.7, 0.1], dtype=torch.float64)
        gate_b = torch.tensor([-0.3, 0.8, 0.2, -0.5], dtype=torch.float64)
        up_a = torch.tensor([0.6, 0.1, -0.4, 0.9], dtype=torch.float64)
        up_b = torch.tensor([-0.7, 0.5, 0.3, 0.2], dtype=torch.float64)
        down_a = torch.tensor([0.3, -0.6, 0.8, 0.1], dtype=torch.float64)
        down_b = torch.tensor([-0.2, 0.4, 0.5, -0.9], dtype=torch.float64)
        layer.w1.weight.copy_(torch.stack((gate_a, gate_a, gate_b, gate_b)))
        layer.w3.weight.copy_(torch.stack((up_a, up_a, up_b, up_b)))
        layer.w2.weight.copy_(
            torch.stack((down_a, down_a, down_b, down_b), dim=1)
        )
    evidence = _evidence(parent)
    plan = _plan(parent, evidence, ridge=0.0)
    generator = torch.Generator().manual_seed(777)
    inputs = torch.randn(
        17,
        _HIDDEN,
        generator=generator,
        dtype=torch.float64,
    )
    with torch.no_grad():
        expected = parent.model.layers[0].feed_forward(inputs)

    transaction = install_lfm2_variable_width_fold_plan(
        parent,
        child,
        evidence,
        plan,
    )
    with torch.no_grad():
        observed = child.model.layers[0].feed_forward(inputs)

    assert torch.allclose(observed, expected, rtol=1e-10, atol=1e-10)
    assert transaction.rollback() is True


def test_keyboard_interrupt_rolls_back_exact_modules_and_bytes() -> None:
    parent = _TinyLfmCausal(_config(), seed=60)
    child = _TinyLfmCausal(_config(), seed=61)
    evidence = _evidence(parent)
    plan = _plan(parent, evidence)
    blank = _state(child)
    original_ffns = tuple(
        layer.feed_forward for layer in child.model.layers
    )
    calls = []

    def interrupt(context):
        calls.append(context.layer_index)
        if context.layer_index == 2:
            raise KeyboardInterrupt("synthetic interrupt")
        return None

    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        install_lfm2_variable_width_fold_plan(
            parent,
            child,
            evidence,
            plan,
            residual_corrector=interrupt,
        )

    assert calls == [0, 1, 2]
    _assert_state(child, blank)
    assert tuple(
        layer.feed_forward for layer in child.model.layers
    ) == original_ffns
    assert not hasattr(
        child,
        "_neural_forge_variable_width_fold_transaction",
    )
