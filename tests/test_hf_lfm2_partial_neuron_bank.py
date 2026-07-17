from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_partial_neuron_bank as partial  # noqa: E402
from neural_forge.hf_lfm2_partial_neuron_bank import (  # noqa: E402
    Lfm2PartialNeuronBankEvidenceError,
    Lfm2PartialNeuronBankMLP,
    Lfm2PartialNeuronBankTransactionError,
    install_lfm2_partial_neuron_bank_plan,
    partial_neuron_bank_pair_savings,
    plan_lfm2_partial_neuron_bank,
    seal_lfm2_partial_neuron_bank_evidence,
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
        return self.w2(
            torch.nn.functional.silu(self.w1(inputs)) * self.w3(inputs)
        )


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
        self.is_attention_layer = True
        self.self_attn = Lfm2Attention(config, layer_idx)
        self.feed_forward = Lfm2MLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)

    def forward(self, inputs):
        hidden = inputs + self.self_attn(self.operator_norm(inputs))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


for _class in (
    Lfm2RMSNorm,
    Lfm2MLP,
    Lfm2Attention,
    Lfm2DecoderLayer,
):
    _class.__module__ = _LFM2_MODULE


class _Backbone(nn.Module):
    def __init__(self, config: _Config):
        super().__init__()
        self.layers = nn.ModuleList(
            Lfm2DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _TinyLfm(nn.Module):
    def __init__(self, seed: int = 100):
        super().__init__()
        self.config = _Config(
            model_type="lfm2",
            hidden_size=4,
            intermediate_size=6,
            conv_L_cache=3,
            conv_bias=False,
            layer_types=["full_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=2,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = _Backbone(self.config)

    def forward(self, hidden):
        return self.model(hidden)


def _parameter_objects(model: nn.Module) -> dict[str, object]:
    return dict(model.named_parameters(remove_duplicate=False))


def _unique_parameters(model: nn.Module) -> int:
    unique = {id(parameter): parameter for parameter in model.parameters()}
    return sum(int(parameter.numel()) for parameter in unique.values())


def _make_all_units_match_by_permutation(
    model: _TinyLfm,
) -> tuple[int, ...]:
    left = model.model.layers[0].feed_forward
    right = model.model.layers[1].feed_forward
    permutation = (4, 2, 5, 1, 3, 0)
    with torch.no_grad():
        for left_unit, right_unit in enumerate(permutation):
            right.w1.weight[right_unit].copy_(left.w1.weight[left_unit])
            right.w3.weight[right_unit].copy_(left.w3.weight[left_unit])
            right.w2.weight[:, right_unit].copy_(
                left.w2.weight[:, left_unit]
            )
    return permutation


def _activation_rows(
    permutation: tuple[int, ...],
    seed: int = 200,
) -> dict[int, object]:
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(32, 6, generator=generator, dtype=torch.float64)
    right = torch.empty_like(left)
    for left_unit, right_unit in enumerate(permutation):
        right[:, right_unit] = left[:, left_unit]
    return {0: left, 1: right}


def _evidence(model: _TinyLfm, seed: int = 200):
    permutation = _make_all_units_match_by_permutation(model)
    rows = _activation_rows(permutation, seed)
    evidence = seal_lfm2_partial_neuron_bank_evidence(
        model,
        rows,
        cache_key=_CACHE_KEY,
        heldout_corpus_sha256=_CORPUS_KEY,
    )
    return permutation, rows, evidence


def test_pair_accounting_formula_is_exact() -> None:
    assert partial_neuron_bank_pair_savings(1024, 1) == 3_072
    assert partial_neuron_bank_pair_savings(1024, 256) == 786_432


def test_matching_coupled_unit_is_numerically_equivalent_and_reduces_storage() -> (
    None
):
    model = _TinyLfm(301).eval()
    inputs = torch.randn(
        3,
        5,
        4,
        generator=torch.Generator().manual_seed(302),
        dtype=torch.float64,
    )
    permutation, _rows, evidence = _evidence(model, 303)
    with torch.no_grad():
        expected = model(inputs)
    parent_unique = _unique_parameters(model)
    originals = tuple(
        layer.feed_forward for layer in model.model.layers
    )
    plan = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        target_parameters_saved=3 * model.config.hidden_size,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )

    assert plan.target_parameters_saved == 12
    assert plan.unit_pair_count == 1
    assert plan.projected_unique_parameters_after == parent_unique - 12
    assert plan.selected_pairs[0].right_unit == permutation[
        plan.selected_pairs[0].left_unit
    ]
    assert plan.selected_pairs[0].activation_similarity == pytest.approx(1.0)
    assert plan.selected_pairs[0].weight_similarity == pytest.approx(1.0)
    assert len(plan.plan_sha256) == 64

    installed = install_lfm2_partial_neuron_bank_plan(
        model,
        evidence,
        plan,
    )
    with torch.no_grad():
        observed = model(inputs)

    assert all(
        isinstance(layer.feed_forward, Lfm2PartialNeuronBankMLP)
        for layer in model.model.layers
    )
    assert (
        model.model.layers[0].feed_forward.bank
        is model.model.layers[1].feed_forward.bank
    )
    assert _unique_parameters(model) == parent_unique - 12
    assert installed.report.parameters_saved == 12
    assert installed.report.bank_parameter_count == 12
    assert installed.report.private_parameter_count == 120
    assert installed.report.integer_index_buffer_count == 2
    assert installed.report.integer_index_buffer_bytes == 16
    assert installed.report.compensation_parameter_count == 0
    assert installed.report.unshared_units_preserved_exactly is True
    torch.testing.assert_close(
        observed,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )

    assert installed.rollback() is True
    assert tuple(
        layer.feed_forward for layer in model.model.layers
    ) == originals
    with torch.no_grad():
        restored = model(inputs)
    torch.testing.assert_close(restored, expected, atol=0.0, rtol=0.0)


def test_selection_and_budget_rounding_are_deterministic() -> None:
    model = _TinyLfm(401).eval()
    _permutation, _rows, evidence = _evidence(model, 402)
    unique_before = _unique_parameters(model)

    first = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        target_unique_parameter_budget=unique_before - 1,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )
    second = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        target_unique_parameter_budget=unique_before - 1,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )

    assert first.target_parameters_saved == 12
    assert first.projected_unique_parameters_after == unique_before - 12
    assert first.projected_unique_parameters_after <= unique_before - 1
    assert first.selected_pairs == second.selected_pairs
    assert first.plan_sha256 == second.plan_sha256


def test_unshared_private_payloads_are_byte_exact() -> None:
    model = _TinyLfm(501).eval()
    _permutation, _rows, evidence = _evidence(model, 502)
    plan = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        target_parameters_saved=12,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )

    for layer_plan in plan.layer_plans:
        layer = model.model.layers[layer_plan.layer_index]
        index = torch.tensor(
            layer_plan.private_unit_indices,
            dtype=torch.long,
        )
        torch.testing.assert_close(
            layer_plan.private_gate,
            layer.feed_forward.w1.weight.detach().index_select(0, index),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            layer_plan.private_up,
            layer.feed_forward.w3.weight.detach().index_select(0, index),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            layer_plan.private_down,
            layer.feed_forward.w2.weight.detach()
            .transpose(0, 1)
            .index_select(0, index),
            atol=0.0,
            rtol=0.0,
        )


def test_finite_plan_payload_tamper_is_rejected_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyLfm(601).eval()
    _permutation, _rows, evidence = _evidence(model, 602)
    plan = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        target_parameters_saved=12,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )
    originals = tuple(
        layer.feed_forward for layer in model.model.layers
    )
    parameters_before = _parameter_objects(model)
    plan.bank_gate[0, 0].add_(0.125)
    assignments = 0

    def forbidden_assignment(*_args, **_kwargs):
        nonlocal assignments
        assignments += 1
        raise AssertionError("model mutation began before seal validation")

    monkeypatch.setattr(partial, "_assign_feed_forward", forbidden_assignment)
    with pytest.raises(
        Lfm2PartialNeuronBankTransactionError,
        match="sealed partial-neuron bank plan changed",
    ):
        install_lfm2_partial_neuron_bank_plan(model, evidence, plan)

    assert assignments == 0
    assert tuple(
        layer.feed_forward for layer in model.model.layers
    ) == originals
    parameters_after = _parameter_objects(model)
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_base_exception_after_first_layer_mutation_rolls_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    model = _TinyLfm(701).eval()
    inputs = torch.randn(
        2,
        4,
        4,
        generator=torch.Generator().manual_seed(702),
        dtype=torch.float64,
    )
    _permutation, _rows, evidence = _evidence(model, 703)
    plan = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        target_parameters_saved=12,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )
    originals = tuple(
        layer.feed_forward for layer in model.model.layers
    )
    parameters_before = _parameter_objects(model)
    with torch.no_grad():
        output_before = model(inputs)
    real_assign = partial._assign_feed_forward
    calls = 0

    def interrupt_second(layer, module):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption("injected partial-bank interruption")
        real_assign(layer, module)

    monkeypatch.setattr(partial, "_assign_feed_forward", interrupt_second)
    with pytest.raises(
        interruption,
        match="injected partial-bank interruption",
    ):
        install_lfm2_partial_neuron_bank_plan(model, evidence, plan)

    assert tuple(
        layer.feed_forward for layer in model.model.layers
    ) == originals
    parameters_after = _parameter_objects(model)
    assert parameters_after.keys() == parameters_before.keys()
    assert all(
        parameters_after[name] is parameter
        for name, parameter in parameters_before.items()
    )
    with torch.no_grad():
        output_after = model(inputs)
    torch.testing.assert_close(
        output_after,
        output_before,
        atol=0.0,
        rtol=0.0,
    )


def test_nonfinite_activation_evidence_is_rejected() -> None:
    model = _TinyLfm(801).eval()
    permutation = _make_all_units_match_by_permutation(model)
    rows = _activation_rows(permutation, 802)
    rows[0][0, 0] = torch.nan

    with pytest.raises(
        Lfm2PartialNeuronBankEvidenceError,
        match="non-finite",
    ):
        seal_lfm2_partial_neuron_bank_evidence(
            model,
            rows,
            cache_key=_CACHE_KEY,
            heldout_corpus_sha256=_CORPUS_KEY,
        )


@pytest.mark.parametrize("tamper", ("activation", "weight"))
def test_sealed_activation_and_weight_evidence_bind_the_exact_parent(
    tamper: str,
) -> None:
    model = _TinyLfm(851).eval()
    _permutation, _rows, evidence = _evidence(model, 852)
    with torch.no_grad():
        if tamper == "activation":
            evidence.layer_rows[0][0, 0].add_(0.125)
        else:
            model.model.layers[0].feed_forward.w1.weight[0, 0].add_(0.125)

    with pytest.raises(
        Lfm2PartialNeuronBankEvidenceError,
        match="changed|stale",
    ):
        plan_lfm2_partial_neuron_bank(
            model,
            evidence,
            target_parameters_saved=12,
            layer_pairs=((0, 1),),
            exact_assignment_limit=6,
        )


def test_commit_keeps_physical_bank_and_releases_rollback_state() -> None:
    model = _TinyLfm(901).eval()
    _permutation, _rows, evidence = _evidence(model, 902)
    unique_before = _unique_parameters(model)
    plan = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        target_parameters_saved=12,
        layer_pairs=((0, 1),),
        exact_assignment_limit=6,
    )
    installed = install_lfm2_partial_neuron_bank_plan(
        model,
        evidence,
        plan,
    )

    report = installed.commit()

    assert installed.active is False
    assert installed.committed is True
    assert report.unique_parameters_after == unique_before - 12
    assert _unique_parameters(model) == unique_before - 12
    assert installed.rollback() is False


def test_source_has_no_training_or_fake_dense_sharing_path() -> None:
    source = inspect.getsource(partial)

    assert ".backward(" not in source
    assert ".train(" not in source
    assert "torch.optim" not in source
    assert "tie_mlp_modules" not in source
    assert "optimizer_constructed" not in source
    assert "compensation_parameter_count=0" in source
