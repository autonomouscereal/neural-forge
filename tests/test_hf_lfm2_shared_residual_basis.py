from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_shared_residual_basis as shared  # noqa: E402
from neural_forge.hf_lfm2_activation_factorization import (  # noqa: E402
    Lfm2ActivationFactorizationEvidenceError,
    seal_lfm2_activation_factorization_evidence,
)
from neural_forge.hf_lfm2_shared_residual_basis import (  # noqa: E402
    BASIS_ATTRIBUTE,
    Lfm2SharedResidualBasisTransactionError,
    Lfm2SharedResidualFactorizedLinear,
    factorize_shared_residual_output_basis,
    install_lfm2_shared_residual_basis_plan,
    plan_lfm2_shared_residual_basis,
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


def test_shared_budget_buys_higher_rank_and_one_physical_basis() -> None:
    model = _TinyLfm(301).eval()
    evidence = _evidence(model, 302)
    parent_unique = _unique_parameters(model)
    # Remove two dense [4,6] matrices and permit 32 factor scalars:
    # shared rank cost = 4 + 2*6 = 16, independent cost = 2*(4+6) = 20.
    target = parent_unique - 48 + 32
    operators = tuple(layer.operator for layer in model.model.layers)
    unselected = {
        name: parameter
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if ".feed_forward.w2.weight" not in name
    }

    plan = plan_lfm2_shared_residual_basis(
        model,
        evidence,
        target_unique_parameter_budget=target,
    )
    installation = install_lfm2_shared_residual_basis_plan(
        model,
        evidence,
        plan,
    )

    assert plan.rank == 2
    assert plan.independent_uniform_rank_at_same_factor_budget == 1
    assert plan.projected_unique_parameters_after == target
    assert _unique_parameters(model) == target
    assert installation.report.shared_basis_parameter_occurrences == 1
    assert installation.report.shared_basis_unique_parameter_objects == 1
    assert installation.report.private_factor_unique_parameter_objects == 2
    assert installation.report.dense_w2_weights_retained is False
    assert tuple(layer.operator for layer in model.model.layers) == operators
    after = dict(model.named_parameters(remove_duplicate=False))
    assert all(after[name] is parameter for name, parameter in unselected.items())

    runtimes = tuple(
        layer.feed_forward.w2 for layer in model.model.layers
    )
    assert all(
        isinstance(value, Lfm2SharedResidualFactorizedLinear)
        for value in runtimes
    )
    assert runtimes[0].basis_owner is runtimes[1].basis_owner
    basis = getattr(model, BASIS_ATTRIBUTE).output_basis
    occurrences = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter is basis
    ]
    assert len(occurrences) == 1
    assert all(not hasattr(value, "weight") for value in runtimes)


def test_full_residual_rank_reconstructs_distinct_operators_exactly() -> None:
    generator = torch.Generator().manual_seed(401)
    weights = tuple(
        torch.randn(4, 6, generator=generator, dtype=torch.float64)
        for _ in range(3)
    )
    fit = tuple(
        torch.randn(20, 6, generator=generator, dtype=torch.float64)
        for _ in weights
    )
    validation = tuple(
        torch.randn(11, 6, generator=generator, dtype=torch.float64)
        for _ in weights
    )

    factorization = factorize_shared_residual_output_basis(
        weights,
        fit,
        validation,
        rank=4,
    )

    assert factorization.rank == 4
    assert factorization.validation_relative_rmse < 1e-10
    for weight, layer in zip(weights, factorization.layers):
        reconstructed = factorization.output_basis @ layer.input_factor
        torch.testing.assert_close(
            reconstructed,
            weight,
            atol=1e-10,
            rtol=1e-10,
        )
    # The private layer operators remain distinct even though B is shared.
    assert not torch.equal(
        factorization.layers[0].input_factor,
        factorization.layers[1].input_factor,
    )


def test_basis_depends_only_on_fit_outputs_not_validation_rows() -> None:
    generator = torch.Generator().manual_seed(501)
    weights = tuple(
        torch.randn(4, 6, generator=generator, dtype=torch.float64)
        for _ in range(2)
    )
    fit = tuple(
        torch.randn(18, 6, generator=generator, dtype=torch.float64)
        for _ in weights
    )
    first_validation = tuple(
        torch.randn(9, 6, generator=generator, dtype=torch.float64)
        for _ in weights
    )
    second_validation = tuple(value * 100.0 for value in first_validation)

    first = factorize_shared_residual_output_basis(
        weights,
        fit,
        first_validation,
        rank=2,
    )
    second = factorize_shared_residual_output_basis(
        weights,
        fit,
        second_validation,
        rank=2,
    )

    assert first.output_basis_sha256 == second.output_basis_sha256
    torch.testing.assert_close(first.output_basis, second.output_basis)
    assert first.fit_stack_sha256 == second.fit_stack_sha256


def test_layer_balancing_prevents_loud_layer_from_erasing_quiet_direction() -> None:
    weights = (
        torch.diag(
            torch.tensor([1000.0, 900.0, 0.0], dtype=torch.float64)
        ),
        torch.diag(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        ),
    )
    generator = torch.Generator().manual_seed(551)
    shared_fit = torch.randn(
        128,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    shared_validation = torch.randn(
        64,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    fit = (shared_fit, shared_fit.clone())
    validation = (shared_validation, shared_validation.clone())

    balanced = factorize_shared_residual_output_basis(
        weights,
        fit,
        validation,
        rank=2,
        basis_weighting="layer_balanced",
    )
    raw = factorize_shared_residual_output_basis(
        weights,
        fit,
        validation,
        rank=2,
        basis_weighting="raw_stack",
    )

    quiet_balanced = balanced.layers[1]
    quiet_raw = raw.layers[1]
    assert balanced.basis_weighting == "layer_balanced"
    assert balanced.fit_layer_output_scales[0] > (
        100.0 * balanced.fit_layer_output_scales[1]
    )
    assert quiet_balanced.validation_relative_rmse < 1e-10
    assert quiet_balanced.validation_cosine > 0.999999999
    assert quiet_raw.validation_relative_rmse > 0.99
    assert quiet_raw.validation_cosine == pytest.approx(0.0)
    assert (
        balanced.validation_worst_layer_relative_rmse
        < raw.validation_worst_layer_relative_rmse
    )


def test_tampered_basis_is_rejected_before_any_model_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyLfm(601).eval()
    evidence = _evidence(model, 602)
    plan = plan_lfm2_shared_residual_basis(
        model,
        evidence,
        target_unique_parameter_budget=_unique_parameters(model) - 16,
    )
    plan.factorization.output_basis[0, 0].add_(0.125)
    attachments = 0

    def forbidden_attach(*_args, **_kwargs):
        nonlocal attachments
        attachments += 1
        raise AssertionError("mutation began before factor seal validation")

    monkeypatch.setattr(shared, "_attach_basis", forbidden_attach)
    with pytest.raises(
        Lfm2SharedResidualBasisTransactionError,
        match="sealed shared residual factor payload changed",
    ):
        install_lfm2_shared_residual_basis_plan(model, evidence, plan)

    assert attachments == 0
    assert not hasattr(model, BASIS_ATTRIBUTE)


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_base_exception_after_first_layer_mutation_rolls_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    model = _TinyLfm(701).eval()
    evidence = _evidence(model, 702)
    plan = plan_lfm2_shared_residual_basis(
        model,
        evidence,
        target_unique_parameter_budget=_unique_parameters(model) - 16,
    )
    before = _parameter_objects(model)
    originals = tuple(
        layer.feed_forward.w2 for layer in model.model.layers
    )
    real_assign = shared._assign_projection
    calls = 0

    def interrupt_second(owner, module):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption("injected shared-basis interruption")
        real_assign(owner, module)

    monkeypatch.setattr(shared, "_assign_projection", interrupt_second)
    with pytest.raises(
        interruption,
        match="injected shared-basis interruption",
    ):
        install_lfm2_shared_residual_basis_plan(model, evidence, plan)

    assert tuple(
        layer.feed_forward.w2 for layer in model.model.layers
    ) == originals
    assert not hasattr(model, BASIS_ATTRIBUTE)
    after = _parameter_objects(model)
    assert after.keys() == before.keys()
    assert all(after[name] is parameter for name, parameter in before.items())


def test_explicit_rollback_restores_exact_modules_parameters_and_identity() -> None:
    model = _TinyLfm(801).eval()
    evidence = _evidence(model, 802)
    parent_unique = _unique_parameters(model)
    before = _parameter_objects(model)
    originals = tuple(
        layer.feed_forward.w2 for layer in model.model.layers
    )
    plan = plan_lfm2_shared_residual_basis(
        model,
        evidence,
        target_unique_parameter_budget=parent_unique - 16,
    )
    installed = install_lfm2_shared_residual_basis_plan(
        model,
        evidence,
        plan,
    )

    assert installed.rollback() is True
    assert tuple(
        layer.feed_forward.w2 for layer in model.model.layers
    ) == originals
    assert not hasattr(model, BASIS_ATTRIBUTE)
    after = _parameter_objects(model)
    assert after.keys() == before.keys()
    assert all(after[name] is parameter for name, parameter in before.items())
    assert _unique_parameters(model) == parent_unique


def test_tampered_or_overlapping_evidence_is_rejected() -> None:
    model = _TinyLfm(901).eval()
    evidence = _evidence(model, 902)
    evidence.layers[0].fit_post_swiglu[0, 0].add_(0.25)
    with pytest.raises(
        Lfm2ActivationFactorizationEvidenceError,
        match="changed",
    ):
        plan_lfm2_shared_residual_basis(
            model,
            evidence,
            target_unique_parameter_budget=_unique_parameters(model) - 16,
        )

    model = _TinyLfm(903).eval()
    evidence = _evidence(model, 904)
    object.__setattr__(
        evidence,
        "validation_row_ids",
        evidence.fit_row_ids[: len(evidence.validation_row_ids)],
    )
    with pytest.raises(
        Lfm2ActivationFactorizationEvidenceError,
        match="disjoint",
    ):
        plan_lfm2_shared_residual_basis(
            model,
            evidence,
            target_unique_parameter_budget=_unique_parameters(model) - 16,
        )


def test_source_contains_no_training_or_dense_fallback_runtime() -> None:
    source = inspect.getsource(shared)

    assert ".backward(" not in source
    assert ".train(" not in source
    assert "torch.optim" not in source
    assert "self.weight =" not in source
    assert "except BaseException as exc:" in source
    assert "dense_w2_weights_retained=False" in source
