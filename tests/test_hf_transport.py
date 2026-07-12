from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_transport import (  # noqa: E402
    Disposition,
    DuplicateRoleEvidenceError,
    EvidenceValidationError,
    GateDecision,
    IncompleteCoverageError,
    PlanValidationError,
    RequiredTargetRole,
    RejectedCoverageError,
    RoleEvidence,
    TransactionInvariantError,
    TransportPlan,
    UnknownRoleError,
    run_transport_transaction,
)


class _ObservedLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(3, 2, bias=True)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return super().forward(inputs)


def _plan(*roles: RequiredTargetRole) -> TransportPlan:
    return TransportPlan(name="tiny-cross-family", required_roles=tuple(roles))


def _transported(role: str, *targets: str, metadata=None) -> RoleEvidence:
    return RoleEvidence(
        role=role,
        disposition=Disposition.TRANSPORTED,
        method="orthogonal-coordinate-transport",
        target_tensors=tuple(targets),
        source_artifacts=(f"donor:{role}",),
        metadata={} if metadata is None else metadata,
    )


def test_complete_coverage_accepts_every_explicit_role() -> None:
    model = _ObservedLinear()
    plan = _plan(
        RequiredTargetRole(
            "projection",
            ("model.weight",),
            (Disposition.TRANSPORTED,),
        ),
        RequiredTargetRole(
            "bias",
            ("model.bias",),
            (Disposition.TARGET_NATIVE,),
        ),
    )
    original_bias = model.bias.detach().clone()

    def mutate(coverage) -> None:
        model.weight.fill_(2.5)
        coverage.record(_transported("projection", "model.weight"))
        coverage.record(
            RoleEvidence(
                role="bias",
                disposition=Disposition.TARGET_NATIVE,
                method="verified-existing-target-bias",
                target_tensors=("model.bias",),
                metadata={"finite": True},
            )
        )

    result = run_transport_transaction(
        plan=plan,
        targets={"model": model},
        mutation=mutate,
        gate=lambda report: GateDecision(
            accepted=report["complete"],
            reason="coverage-complete",
            evidence={"heldout_error": 0.25},
        ),
    )

    assert result.accepted is True
    assert torch.equal(model.weight, torch.full_like(model.weight, 2.5))
    assert torch.equal(model.bias, original_bias)
    assert result.report["coverage"]["evidenced_role_count"] == 2
    assert result.report["coverage"]["complete"] is True
    assert result.report["rollback"] == {"performed": False, "exact": None}
    assert len(result.report["report_sha256"]) == 64


def test_missing_role_rejects_and_restores_mutated_tensor() -> None:
    model = _ObservedLinear()
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    plan = _plan(
        RequiredTargetRole("projection", ("model.weight",)),
        RequiredTargetRole("bias", ("model.bias",)),
    )

    def mutate(coverage) -> None:
        model.weight.zero_()
        model.bias.fill_(9.0)
        coverage.record(_transported("projection", "model.weight"))

    with pytest.raises(IncompleteCoverageError) as caught:
        run_transport_transaction(plan=plan, targets={"model": model}, mutation=mutate)

    assert caught.value.report["missing_roles"] == ["bias"]
    assert caught.value.report["complete"] is False
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_random_initialization_cannot_satisfy_a_required_role() -> None:
    model = _ObservedLinear()
    original = model.weight.detach().clone()
    plan = _plan(
        RequiredTargetRole(
            "projection",
            ("model.weight",),
            (Disposition.TARGET_NATIVE,),
        )
    )

    def mutate(coverage) -> None:
        model.weight.normal_()
        coverage.record(
            RoleEvidence(
                role="projection",
                disposition=Disposition.TARGET_NATIVE,
                method="random-seed-zero",
                target_tensors=("model.weight",),
                random_initialization=True,
            )
        )

    with pytest.raises(EvidenceValidationError, match="Random initialization"):
        run_transport_transaction(plan=plan, targets={"model": model}, mutation=mutate)
    assert torch.equal(model.weight, original)


def test_target_native_requires_explicit_role_permission() -> None:
    model = _ObservedLinear()
    plan = _plan(RequiredTargetRole("projection", ("model.weight",)))

    def mutate(coverage) -> None:
        coverage.record(
            RoleEvidence(
                role="projection",
                disposition=Disposition.TARGET_NATIVE,
                method="leave-existing-value",
                target_tensors=("model.weight",),
            )
        )

    with pytest.raises(EvidenceValidationError, match="not allowed"):
        run_transport_transaction(plan=plan, targets={"model": model}, mutation=mutate)


def test_tensorless_operator_role_is_explicitly_target_native_only() -> None:
    model = _ObservedLinear()
    plan = _plan(
        RequiredTargetRole(
            "activation",
            (),
            (Disposition.TARGET_NATIVE,),
        ),
        RequiredTargetRole("projection", ("model.weight",)),
    )

    def mutate(coverage) -> None:
        model.weight.fill_(3.0)
        coverage.record(
            RoleEvidence(
                role="activation",
                disposition=Disposition.TARGET_NATIVE,
                method="validated-configured-activation",
                target_tensors=(),
                metadata={"name": "silu"},
            )
        )
        coverage.record(_transported("projection", "model.weight"))

    result = run_transport_transaction(
        plan=plan,
        targets={"model": model},
        mutation=mutate,
    )

    assert result.accepted is True
    activation = result.report["coverage"]["roles"][0]
    assert activation["role"] == "activation"
    assert activation["required_target_tensors"] == []
    assert activation["evidence"]["disposition"] == "target-native"

    with pytest.raises(PlanValidationError, match="Tensorless role"):
        RequiredTargetRole(
            "missing-weight",
            (),
            (Disposition.TRANSPORTED,),
        )


def test_unknown_and_duplicate_evidence_are_fail_closed() -> None:
    model = _ObservedLinear()
    original = model.weight.detach().clone()
    plan = _plan(RequiredTargetRole("projection", ("model.weight",)))

    def unknown(coverage) -> None:
        model.weight.add_(1.0)
        coverage.record(_transported("undeclared", "model.weight"))

    with pytest.raises(UnknownRoleError):
        run_transport_transaction(plan=plan, targets={"model": model}, mutation=unknown)
    assert torch.equal(model.weight, original)

    def duplicate(coverage) -> None:
        model.weight.add_(1.0)
        coverage.record(_transported("projection", "model.weight"))
        coverage.record(_transported("projection", "model.weight"))

    with pytest.raises(DuplicateRoleEvidenceError):
        run_transport_transaction(
            plan=plan, targets={"model": model}, mutation=duplicate
        )
    assert torch.equal(model.weight, original)

    def swallowed_duplicate(coverage) -> None:
        model.weight.add_(1.0)
        coverage.record(_transported("projection", "model.weight"))
        try:
            coverage.record(_transported("projection", "model.weight"))
        except DuplicateRoleEvidenceError:
            pass

    with pytest.raises(RejectedCoverageError) as caught:
        run_transport_transaction(
            plan=plan,
            targets={"model": model},
            mutation=swallowed_duplicate,
        )
    assert caught.value.report["duplicate_roles"] == ["projection"]
    assert caught.value.report["complete"] is False
    assert torch.equal(model.weight, original)


def test_exception_and_gate_failure_restore_values_gradients_and_modes() -> None:
    model = _ObservedLinear()
    model.train(True)
    model.weight.grad = torch.full_like(model.weight, 0.75)
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    original_grad = model.weight.grad.detach().clone()
    plan = _plan(RequiredTargetRole("projection", ("model.weight",)))

    def explode(_coverage) -> None:
        model.weight.fill_(11.0)
        model.eval()
        model.weight.grad.zero_()
        raise RuntimeError("operator failed")

    with pytest.raises(RuntimeError, match="operator failed"):
        run_transport_transaction(plan=plan, targets={"model": model}, mutation=explode)
    assert model.training is True
    assert torch.equal(model.weight.grad, original_grad)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name])

    def mutate(coverage) -> None:
        model.weight.fill_(7.0)
        coverage.record(_transported("projection", "model.weight"))

    result = run_transport_transaction(
        plan=plan,
        targets={"model": model},
        mutation=mutate,
        gate=lambda _report: GateDecision(False, "heldout-regressed", {"delta": 1.5}),
    )
    assert result.accepted is False
    assert result.report["rollback"] == {"performed": True, "exact": True}
    assert model.training is True
    assert torch.equal(model.weight.grad, original_grad)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name])


def test_equivalent_transactions_emit_identical_deterministic_reports() -> None:
    torch.manual_seed(17)
    left = _ObservedLinear()
    right = _ObservedLinear()
    right.load_state_dict(left.state_dict())
    plan = _plan(RequiredTargetRole("projection", ("model.weight",)))

    def run(model, reverse_metadata: bool):
        metadata = (
            {"z_score": 0.5, "samples": 16}
            if reverse_metadata
            else {"samples": 16, "z_score": 0.5}
        )

        def mutate(coverage) -> None:
            model.weight.add_(0.125)
            coverage.record(
                _transported("projection", "model.weight", metadata=metadata)
            )

        return run_transport_transaction(
            plan=plan,
            targets={"model": model},
            mutation=mutate,
            gate=lambda _report: GateDecision(
                True,
                "exact-contract",
                {"b": [2, 1], "a": True},
            ),
        ).report

    left_report = run(left, reverse_metadata=False)
    right_report = run(right, reverse_metadata=True)
    assert left_report == right_report
    assert json.dumps(left_report, sort_keys=True) == json.dumps(
        right_report, sort_keys=True
    )


def test_core_never_runs_forward_backward_optimizer_or_changes_training_state() -> None:
    model = _ObservedLinear()
    model.train(True)
    assert model.weight.grad is None
    plan = _plan(RequiredTargetRole("projection", ("model.weight",)))

    def mutate(coverage) -> None:
        assert torch.is_grad_enabled() is False
        model.weight.copy_(torch.ones_like(model.weight))
        coverage.record(_transported("projection", "model.weight"))

    def gate(_report):
        assert torch.is_grad_enabled() is False
        return True

    result = run_transport_transaction(
        plan=plan,
        targets={"model": model},
        mutation=mutate,
        gate=gate,
    )

    assert model.forward_calls == 0
    assert model.training is True
    assert model.weight.grad is None
    assert result.report["execution_bounds"] == {
        "core_optimizer_steps": 0,
        "core_forward_calls": 0,
        "core_backward_calls": 0,
        "mutation_grad_mode_disabled": True,
        "gate_grad_mode_disabled": True,
        "gradients_unchanged": True,
        "module_training_modes_unchanged": True,
    }

    accepted_weight = model.weight.detach().clone()

    def violate_bounds(coverage) -> None:
        model.eval()
        model.weight.grad = torch.ones_like(model.weight)
        coverage.record(_transported("projection", "model.weight"))

    with pytest.raises(TransactionInvariantError, match="escaped its mutation bounds"):
        run_transport_transaction(
            plan=plan,
            targets={"model": model},
            mutation=violate_bounds,
        )
    assert model.training is True
    assert model.weight.grad is None
    assert torch.equal(model.weight, accepted_weight)
