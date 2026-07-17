from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_activation_factorization as actlr  # noqa: E402
from neural_forge.hf_lfm2_activation_factorization import (  # noqa: E402
    Lfm2ActivationFactorizationEvidenceError,
    Lfm2ActivationFactorizationTransactionError,
    Lfm2ActivationFactorizedLinear,
    factorize_activation_weighted_linear,
    install_lfm2_activation_factorization_plan,
    plan_lfm2_activation_factorization,
    seal_lfm2_activation_factorization_evidence,
)


_CACHE_KEY = "a" * 64
_CORPUS_KEY = "b" * 64


class _Config(SimpleNamespace):
    def to_dict(self) -> dict[str, object]:
        return dict(vars(self))


class _TinyMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.w1 = nn.Linear(
            hidden,
            intermediate,
            bias=False,
            dtype=torch.float64,
        )
        self.w3 = nn.Linear(
            hidden,
            intermediate,
            bias=False,
            dtype=torch.float64,
        )
        self.w2 = nn.Linear(
            intermediate,
            hidden,
            bias=False,
            dtype=torch.float64,
        )

    def forward(self, inputs):
        post = torch.nn.functional.silu(self.w1(inputs)) * self.w3(inputs)
        return self.w2(post)


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.operator = nn.Linear(
            hidden,
            hidden,
            bias=False,
            dtype=torch.float64,
        )
        self.feed_forward = _TinyMlp(hidden, intermediate)

    def forward(self, inputs):
        hidden = inputs + self.operator(inputs)
        return hidden + self.feed_forward(hidden)


class _Backbone(nn.Module):
    def __init__(self, hidden: int, intermediate: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            _TinyLayer(hidden, intermediate) for _ in range(layers)
        )

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class _TinyLfm(nn.Module):
    def __init__(self, seed: int = 100):
        super().__init__()
        self.config = _Config(
            model_type="lfm2",
            hidden_size=4,
            intermediate_size=6,
            num_hidden_layers=2,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = _Backbone(4, 6, 2)

    def forward(self, inputs):
        return self.model(inputs)


def _unique_parameters(model: nn.Module) -> int:
    values = {id(parameter): parameter for parameter in model.parameters()}
    return sum(int(parameter.numel()) for parameter in values.values())


def _parameter_objects(model: nn.Module) -> dict[str, object]:
    return dict(model.named_parameters(remove_duplicate=False))


def _evidence(model: _TinyLfm, seed: int = 200):
    generator = torch.Generator().manual_seed(seed)
    fit_ids = tuple(f"fit-{index}" for index in range(24))
    validation_ids = tuple(f"validation-{index}" for index in range(12))
    fit_inputs = {}
    validation_inputs = {}
    fit_post = {}
    validation_post = {}
    for layer_index, layer in enumerate(model.model.layers):
        fit = torch.randn(
            len(fit_ids),
            4,
            generator=generator,
            dtype=torch.float64,
        )
        validation = torch.randn(
            len(validation_ids),
            4,
            generator=generator,
            dtype=torch.float64,
        )
        with torch.no_grad():
            fit_hidden = fit + layer.operator(fit)
            validation_hidden = validation + layer.operator(validation)
            fit_swiglu = (
                torch.nn.functional.silu(
                    layer.feed_forward.w1(fit_hidden)
                )
                * layer.feed_forward.w3(fit_hidden)
            )
            validation_swiglu = (
                torch.nn.functional.silu(
                    layer.feed_forward.w1(validation_hidden)
                )
                * layer.feed_forward.w3(validation_hidden)
            )
        fit_inputs[layer_index] = fit_hidden
        validation_inputs[layer_index] = validation_hidden
        fit_post[layer_index] = fit_swiglu
        validation_post[layer_index] = validation_swiglu
    return seal_lfm2_activation_factorization_evidence(
        model,
        fit_inputs,
        validation_inputs,
        fit_post,
        validation_post,
        fit_row_ids=fit_ids,
        validation_row_ids=validation_ids,
        cache_key=_CACHE_KEY,
        heldout_corpus_sha256=_CORPUS_KEY,
    )


def test_truly_rank_one_weight_is_functionally_exact() -> None:
    model = _TinyLfm(301).eval()
    layer = model.model.layers[0]
    left = torch.tensor(
        [[1.0], [2.0], [-1.0], [0.5], [3.0], [-2.0]],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[0.25, -0.5, 1.5, 2.0]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.feed_forward.w1.weight.copy_(left @ right)
    evidence = _evidence(model, 302)
    inputs = torch.randn(
        3,
        5,
        4,
        generator=torch.Generator().manual_seed(303),
        dtype=torch.float64,
    )
    with torch.no_grad():
        expected = model(inputs)
    parent_unique = _unique_parameters(model)
    dense = 6 * 4
    factor = 1 * (6 + 4)
    plan = plan_lfm2_activation_factorization(
        model,
        evidence,
        target_unique_parameter_budget=parent_unique - dense + factor,
        layer_indices=(0,),
        roles=("w1",),
    )
    installed = install_lfm2_activation_factorization_plan(
        model,
        evidence,
        plan,
    )
    with torch.no_grad():
        observed = model(inputs)

    assert isinstance(
        model.model.layers[0].feed_forward.w1,
        Lfm2ActivationFactorizedLinear,
    )
    assert plan.role_plans[0].rank == 1
    assert installed.report.parameters_saved == dense - factor
    torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)


def test_activation_weighting_beats_raw_svd_on_heldout_geometry() -> None:
    weight = torch.diag(
        torch.tensor([10.0, 1.0], dtype=torch.float64)
    )
    fit = torch.tensor(
        [
            [0.001, -10.0],
            [-0.002, -5.0],
            [0.0015, 4.0],
            [-0.001, 8.0],
        ],
        dtype=torch.float64,
    )
    validation = torch.tensor(
        [
            [0.001, -7.0],
            [-0.001, 3.0],
            [0.002, 12.0],
        ],
        dtype=torch.float64,
    )

    result = factorize_activation_weighted_linear(
        weight,
        fit,
        validation,
        rank=1,
    )

    assert result.method == "activation_weighted"
    assert result.validation_relative_rmse < 0.01
    assert (
        result.validation_relative_rmse
        <= result.raw_svd_validation_relative_rmse
    )


def test_physical_savings_and_unselected_parameters_are_exact() -> None:
    model = _TinyLfm(401).eval()
    evidence = _evidence(model, 402)
    parent_unique = _unique_parameters(model)
    unselected = {
        name: parameter
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if "layers.0.feed_forward.w1.weight" not in name
    }
    target = parent_unique - (6 * 4) + (6 + 4)
    plan = plan_lfm2_activation_factorization(
        model,
        evidence,
        target_unique_parameter_budget=target,
        layer_indices=(0,),
        roles=("w1",),
    )
    installed = install_lfm2_activation_factorization_plan(
        model,
        evidence,
        plan,
    )

    assert _unique_parameters(model) == target
    assert installed.report.dense_parameters_removed == 24
    assert installed.report.factor_parameters_added == 10
    assert installed.report.parameters_saved == 14
    assert installed.report.integer_metadata_values == 3
    assert installed.report.integer_tensor_count == 0
    assert installed.report.integer_tensor_bytes == 0
    after = dict(model.named_parameters(remove_duplicate=False))
    assert all(after[name] is parameter for name, parameter in unselected.items())


def test_global_rank_allocator_is_deterministic_and_budget_bounded() -> None:
    first_model = _TinyLfm(501).eval()
    second_model = _TinyLfm(501).eval()
    first_evidence = _evidence(first_model, 502)
    second_evidence = _evidence(second_model, 502)
    parent_unique = _unique_parameters(first_model)
    # Two [6,4] matrices: rank-1 costs 10 each and rank-2 costs 20 each.
    target = parent_unique - 48 + 30

    first = plan_lfm2_activation_factorization(
        first_model,
        first_evidence,
        target_unique_parameter_budget=target,
        layer_indices=(0,),
        roles=("w1", "w3"),
    )
    second = plan_lfm2_activation_factorization(
        second_model,
        second_evidence,
        target_unique_parameter_budget=target,
        layer_indices=(0,),
        roles=("w1", "w3"),
    )

    assert first.projected_unique_parameters_after == target
    assert tuple(value.rank for value in first.role_plans) in ((2, 1), (1, 2))
    assert first.as_dict() == second.as_dict()
    assert len(first.allocation_steps) == 1
    step = first.allocation_steps[0]
    assert step.parameter_cost == 10
    assert step.rank_before == 1
    assert step.rank_after == 2


def test_tampered_factor_is_rejected_before_model_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyLfm(601).eval()
    evidence = _evidence(model, 602)
    parent_unique = _unique_parameters(model)
    plan = plan_lfm2_activation_factorization(
        model,
        evidence,
        target_unique_parameter_budget=parent_unique - 14,
        layer_indices=(0,),
        roles=("w1",),
    )
    plan.role_plans[0].factorization.input_factor[0, 0].add_(0.125)
    assignments = 0

    def forbidden_assignment(*_args, **_kwargs):
        nonlocal assignments
        assignments += 1
        raise AssertionError("mutation began before seal validation")

    monkeypatch.setattr(actlr, "_assign_projection", forbidden_assignment)
    with pytest.raises(
        Lfm2ActivationFactorizationTransactionError,
        match="sealed activation factor payload changed",
    ):
        install_lfm2_activation_factorization_plan(model, evidence, plan)

    assert assignments == 0


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_base_exception_after_first_mutation_rolls_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    model = _TinyLfm(701).eval()
    evidence = _evidence(model, 702)
    parent_unique = _unique_parameters(model)
    plan = plan_lfm2_activation_factorization(
        model,
        evidence,
        target_unique_parameter_budget=parent_unique - 28,
        layer_indices=(0,),
        roles=("w1", "w3"),
    )
    before = _parameter_objects(model)
    originals = (
        model.model.layers[0].feed_forward.w1,
        model.model.layers[0].feed_forward.w3,
    )
    real_assign = actlr._assign_projection
    calls = 0

    def interrupt_second(owner, role, module):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption("injected activation-factor interruption")
        real_assign(owner, role, module)

    monkeypatch.setattr(actlr, "_assign_projection", interrupt_second)
    with pytest.raises(
        interruption,
        match="injected activation-factor interruption",
    ):
        install_lfm2_activation_factorization_plan(model, evidence, plan)

    assert (
        model.model.layers[0].feed_forward.w1,
        model.model.layers[0].feed_forward.w3,
    ) == originals
    after = _parameter_objects(model)
    assert after.keys() == before.keys()
    assert all(after[name] is parameter for name, parameter in before.items())


def test_evidence_and_parent_tampering_are_rejected() -> None:
    model = _TinyLfm(801).eval()
    evidence = _evidence(model, 802)
    evidence.layers[0].fit_layer_inputs[0, 0].add_(0.25)
    with pytest.raises(
        Lfm2ActivationFactorizationEvidenceError,
        match="changed",
    ):
        plan_lfm2_activation_factorization(
            model,
            evidence,
            target_unique_parameter_budget=_unique_parameters(model) - 14,
            layer_indices=(0,),
            roles=("w1",),
        )

    model = _TinyLfm(803).eval()
    evidence = _evidence(model, 804)
    with torch.no_grad():
        model.model.layers[0].feed_forward.w1.weight[0, 0].add_(0.25)
    with pytest.raises(
        Lfm2ActivationFactorizationEvidenceError,
        match="stale",
    ):
        plan_lfm2_activation_factorization(
            model,
            evidence,
            target_unique_parameter_budget=_unique_parameters(model) - 14,
            layer_indices=(0,),
            roles=("w1",),
        )


def test_explicit_rollback_restores_exact_module_and_parameters() -> None:
    model = _TinyLfm(901).eval()
    evidence = _evidence(model, 902)
    parent_unique = _unique_parameters(model)
    before = _parameter_objects(model)
    original = model.model.layers[0].feed_forward.w1
    plan = plan_lfm2_activation_factorization(
        model,
        evidence,
        target_unique_parameter_budget=parent_unique - 14,
        layer_indices=(0,),
        roles=("w1",),
    )
    installed = install_lfm2_activation_factorization_plan(
        model,
        evidence,
        plan,
    )

    assert installed.rollback() is True
    assert model.model.layers[0].feed_forward.w1 is original
    after = _parameter_objects(model)
    assert after.keys() == before.keys()
    assert all(after[name] is parameter for name, parameter in before.items())
    assert _unique_parameters(model) == parent_unique


def test_source_contains_no_training_or_dense_fallback_runtime() -> None:
    source = inspect.getsource(actlr)

    assert ".backward(" not in source
    assert ".train(" not in source
    assert "torch.optim" not in source
    assert "self.weight =" not in source
    assert "except BaseException as exc:" in source
