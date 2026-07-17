from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_seed_erasure import (  # noqa: E402
    DishonestSeedExceptionError,
    SeedCompileOutcome,
    SeedErasureContractError,
    SeedLeakError,
    SeedOwnedExceptionScope,
    SeedOwnedField,
    audit_seed_erasure,
)


class _Config:
    def to_dict(self) -> dict[str, object]:
        return {"family": "unit-test", "hidden_size": 2}


class _TinyModel(torch.nn.Module):
    def __init__(
        self,
        *,
        seed: int,
        compiled_weight: float,
        module_name: str = "projection",
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layer = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            layer.weight.fill_(compiled_weight)
        setattr(self, module_name, layer)
        self.module_name = module_name
        self.register_buffer("scale", torch.tensor([1.0, 2.0]))
        self.config = _Config()
        self.eval()

    def forward(self, values):
        return getattr(self, self.module_name)(values) * self.scale


def _outcome(
    seed: int,
    *,
    weight: float = 1.0,
    module_name: str = "projection",
    budget: int = 4,
    report: dict[str, object] | None = None,
) -> SeedCompileOutcome:
    model = _TinyModel(
        seed=seed,
        compiled_weight=weight,
        module_name=module_name,
    )
    return SeedCompileOutcome(
        seed=seed,
        model=model,
        parameter_budget=budget,
        compiler_report=(
            report
            if report is not None
            else {
                "schema": "unit-compiler-v1",
                "seed": seed,
                "accepted": True,
                "target_parameter_budget": 4,
            }
        ),
        label=f"compile-{seed}",
        structure_metadata={"carrier": "tiny"},
    )


def _sha_tensor(value) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _behavior(outcome: SeedCompileOutcome) -> dict[str, str]:
    inputs = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    with torch.no_grad():
        outputs = outcome.model(inputs)
    return {"fixed-probe": _sha_tensor(outputs)}


def _seed_report_exception() -> SeedOwnedField:
    return SeedOwnedField(
        name="/seed",
        reason="The compiler records the requested initialization seed as provenance.",
    )


def test_exactly_erased_outcomes_produce_immutable_proof() -> None:
    report = audit_seed_erasure(
        [_outcome(11), _outcome(29), _outcome(47)],
        _behavior,
        exceptions=SeedOwnedExceptionScope(
            compiler_report_paths=(_seed_report_exception(),),
        ),
    )

    assert report.seed_erasure_proven is True
    assert report.seed_invariant is True
    assert report.seed_leak_detected is False
    assert report.named_tensors_seed_invariant is True
    assert report.behavioral_outputs_seed_invariant is True
    assert report.differing_tensor_names == ()
    assert report.differing_output_names == ()
    assert report.differing_compiler_report_paths == ("/seed",)
    assert report.exact_parameter_budget_match is True
    assert {outcome.unique_parameter_count for outcome in report.outcomes} == {4}
    json.dumps(report.as_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        report.seed_erasure_proven = False


def test_standard_torch_dtype_in_model_config_is_canonicalized() -> None:
    outcomes = [_outcome(11), _outcome(29)]
    for outcome in outcomes:
        outcome.model.config = type(
            "_DtypeConfig",
            (),
            {
                "to_dict": lambda self: {
                    "family": "unit-test",
                    "dtype": torch.bfloat16,
                    "kernel": {"accumulator_dtype": torch.float32},
                }
            },
        )()

    report = audit_seed_erasure(
        outcomes,
        _behavior,
        exceptions=SeedOwnedExceptionScope(
            compiler_report_paths=(_seed_report_exception(),),
        ),
    )

    assert report.seed_erasure_proven is True
    assert report.exact_structure_match is True


def test_declared_tensor_and_output_differences_report_a_seed_leak() -> None:
    report = audit_seed_erasure(
        [_outcome(11, weight=1.0), _outcome(29, weight=2.0)],
        _behavior,
        exceptions=SeedOwnedExceptionScope(
            tensors=(
                SeedOwnedField(
                    name="projection.weight",
                    reason="This experimental carrier intentionally retains its seed.",
                ),
            ),
            outputs=(
                SeedOwnedField(
                    name="fixed-probe",
                    reason="The retained carrier weight changes this fixed probe.",
                ),
            ),
            compiler_report_paths=(_seed_report_exception(),),
        ),
    )

    assert report.seed_erasure_proven is False
    assert report.seed_invariant is False
    assert report.seed_leak_detected is True
    assert report.initial_randomness_remains_in_named_tensors is True
    assert report.initial_randomness_observable_in_outputs is True
    assert report.differing_tensor_names == ("projection.weight",)
    assert report.differing_output_names == ("fixed-probe",)


def test_undeclared_tensor_difference_fails_closed() -> None:
    with pytest.raises(SeedLeakError, match="projection.weight"):
        audit_seed_erasure(
            [_outcome(11, weight=1.0), _outcome(29, weight=2.0)],
            _behavior,
            exceptions=SeedOwnedExceptionScope(
                compiler_report_paths=(_seed_report_exception(),),
            ),
        )


def test_missing_or_renamed_tensor_fails_before_exceptions() -> None:
    with pytest.raises(SeedErasureContractError, match="Exact model structure differs"):
        audit_seed_erasure(
            [_outcome(11), _outcome(29, module_name="renamed_projection")],
            _behavior,
            exceptions=SeedOwnedExceptionScope(
                tensors=(
                    SeedOwnedField(
                        name="projection.weight",
                        reason="An exception cannot hide a renamed tensor.",
                    ),
                ),
                compiler_report_paths=(_seed_report_exception(),),
            ),
        )


@pytest.mark.parametrize(
    "outcomes, message",
    [
        ([_outcome(11, budget=5), _outcome(29, budget=5)], "model has 4"),
        ([_outcome(11, budget=4), _outcome(29, budget=5)], "budgets differ"),
    ],
)
def test_budget_mismatch_or_lie_fails_closed(outcomes, message: str) -> None:
    with pytest.raises(SeedErasureContractError, match=message):
        audit_seed_erasure(
            outcomes,
            _behavior,
            exceptions=SeedOwnedExceptionScope(
                compiler_report_paths=(_seed_report_exception(),),
            ),
        )


def test_missing_or_renamed_compiler_report_field_fails_closed() -> None:
    first = _outcome(11)
    second = _outcome(
        29,
        report={
            "schema": "unit-compiler-v1",
            "compile_seed": 29,
            "accepted": True,
            "target_parameter_budget": 4,
        },
    )
    with pytest.raises(SeedErasureContractError, match="report structure differs"):
        audit_seed_erasure(
            [first, second],
            _behavior,
            exceptions=SeedOwnedExceptionScope(
                compiler_report_paths=(_seed_report_exception(),),
            ),
        )


def test_undeclared_compiler_report_value_fails_closed() -> None:
    with pytest.raises(SeedLeakError, match="/seed"):
        audit_seed_erasure([_outcome(11), _outcome(29)], _behavior)


def test_unused_or_container_exception_is_rejected_as_dishonest() -> None:
    with pytest.raises(
        DishonestSeedExceptionError,
        match="do not actually differ",
    ):
        audit_seed_erasure(
            [_outcome(11), _outcome(29)],
            _behavior,
            exceptions=SeedOwnedExceptionScope(
                tensors=(
                    SeedOwnedField(
                        name="projection.weight",
                        reason="This declaration is intentionally overbroad.",
                    ),
                ),
                compiler_report_paths=(_seed_report_exception(),),
            ),
        )

    container_outcomes = [
        _outcome(
            seed,
            report={
                "schema": "unit-compiler-v1",
                "accepted": True,
                "target_parameter_budget": 4,
                "provenance": {"seed": seed},
            },
        )
        for seed in (11, 29)
    ]
    with pytest.raises(DishonestSeedExceptionError, match="only leaves"):
        audit_seed_erasure(
            container_outcomes,
            _behavior,
            exceptions=SeedOwnedExceptionScope(
                compiler_report_paths=(
                    SeedOwnedField(
                        name="/provenance",
                        reason="A whole report cannot be hidden by one declaration.",
                    ),
                ),
            ),
        )


def test_output_only_seed_difference_cannot_be_declared_as_model_state() -> None:
    def seed_derived_output(outcome: SeedCompileOutcome) -> dict[str, str]:
        return {
            "fixed-probe": hashlib.sha256(str(outcome.seed).encode("ascii")).hexdigest()
        }

    with pytest.raises(DishonestSeedExceptionError, match="every named model tensor"):
        audit_seed_erasure(
            [_outcome(11), _outcome(29)],
            seed_derived_output,
            exceptions=SeedOwnedExceptionScope(
                outputs=(
                    SeedOwnedField(
                        name="fixed-probe",
                        reason="The callback improperly fingerprints the seed itself.",
                    ),
                ),
                compiler_report_paths=(_seed_report_exception(),),
            ),
        )


def test_callback_mutation_is_detected() -> None:
    def mutating_callback(outcome: SeedCompileOutcome) -> dict[str, str]:
        with torch.no_grad():
            outcome.model.projection.weight.add_(1.0)
        return _behavior(outcome)

    with pytest.raises(SeedErasureContractError, match="mutated model"):
        audit_seed_erasure(
            [_outcome(11), _outcome(29)],
            mutating_callback,
            exceptions=SeedOwnedExceptionScope(
                compiler_report_paths=(_seed_report_exception(),),
            ),
        )


def test_reusing_one_model_object_is_not_an_independent_compile() -> None:
    first = _outcome(11)
    second = SeedCompileOutcome(
        seed=29,
        model=first.model,
        parameter_budget=4,
        compiler_report={
            "schema": "unit-compiler-v1",
            "seed": 29,
            "accepted": True,
            "target_parameter_budget": 4,
        },
    )
    with pytest.raises(SeedErasureContractError, match="distinct compiled model"):
        audit_seed_erasure([first, second], _behavior)
