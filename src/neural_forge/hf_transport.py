"""Fail-closed transaction and coverage primitives for checkpoint transport.

This module deliberately knows nothing about a particular model architecture or
tokenizer.  Architecture adapters declare the target roles they own, mutate only
the named target tensors, and attach one evidence record to every required role.
The transaction is accepted only after complete coverage and an optional external
gate both pass.  It does not instantiate an optimizer or invoke a model, forward,
backward, or training method.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


REPORT_SCHEMA = "neural-forge-transport-transaction-v1"


class Disposition(str, Enum):
    """The only admissible ways to satisfy a required target role."""

    TRANSPORTED = "transported"
    CLOSED_FORM_FITTED = "closed-form-fitted"
    TARGET_NATIVE = "target-native"


class TransportError(RuntimeError):
    """Base error for transport planning and transactions."""


class PlanValidationError(TransportError, ValueError):
    """A transport plan or target inventory is invalid."""


class EvidenceValidationError(TransportError, ValueError):
    """A role evidence record is invalid or inadmissible."""


class UnknownRoleError(EvidenceValidationError):
    """Evidence was supplied for a role the plan did not declare."""


class DuplicateRoleEvidenceError(EvidenceValidationError):
    """A required role received more than one evidence record."""


class CoverageValidationError(EvidenceValidationError):
    """The final evidence ledger is incomplete or contains a rejected record."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


class IncompleteCoverageError(CoverageValidationError):
    """At least one required role has no evidence."""


class RejectedCoverageError(CoverageValidationError):
    """A caught evidence error permanently invalidated the transaction ledger."""


class TransactionInvariantError(TransportError):
    """Mutation escaped the zero-training or fixed-target transaction bounds."""


class RollbackError(TransportError):
    """A rejected or failed transaction could not be restored exactly."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without the HF extra
        raise RuntimeError(
            "PyTorch is required for neural_forge.hf_transport tensor transactions."
        ) from exc
    return torch


def _validate_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PlanValidationError(f"{label} must be a non-empty, trimmed string.")
    return value


def _normalize_names(
    values: Sequence[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PlanValidationError(f"{label} must be a sequence of names, not a string.")
    normalized = tuple(_validate_name(value, label) for value in values)
    if not normalized and not allow_empty:
        raise PlanValidationError(f"{label} cannot be empty.")
    if len(set(normalized)) != len(normalized):
        raise PlanValidationError(f"{label} contains duplicate names.")
    return tuple(sorted(normalized))


def _normalize_json(value: Any, path: str = "metadata") -> Any:
    """Return a deterministic JSON value or reject ambiguous/non-finite data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceValidationError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Enum):
        return _normalize_json(value.value, path)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EvidenceValidationError(f"{path} keys must be strings.")
        result: dict[str, Any] = {}
        for key in sorted(value):
            result[key] = _normalize_json(value[key], f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise EvidenceValidationError(
        f"{path} contains unsupported non-JSON value {type(value).__name__}."
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _with_report_hash(report: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(report)
    result["report_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


@dataclass(frozen=True)
class RequiredTargetRole:
    """One target operator role whose disposition must be proven exactly once."""

    name: str
    target_tensors: tuple[str, ...]
    allowed_dispositions: tuple[Disposition, ...] = (
        Disposition.TRANSPORTED,
        Disposition.CLOSED_FORM_FITTED,
    )
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, "role name"))
        if not isinstance(self.description, str):
            raise PlanValidationError("role description must be a string.")

        if isinstance(self.allowed_dispositions, (str, bytes)):
            raise PlanValidationError(
                f"Role {self.name!r} dispositions must be a sequence, not a string."
            )
        dispositions: list[Disposition] = []
        for value in self.allowed_dispositions:
            try:
                disposition = Disposition(value)
            except (TypeError, ValueError) as exc:
                raise PlanValidationError(
                    f"Role {self.name!r} has unknown disposition {value!r}."
                ) from exc
            if disposition in dispositions:
                raise PlanValidationError(
                    f"Role {self.name!r} repeats disposition {disposition.value!r}."
                )
            dispositions.append(disposition)
        if not dispositions:
            raise PlanValidationError(
                f"Role {self.name!r} needs at least one allowed disposition."
            )
        object.__setattr__(self, "allowed_dispositions", tuple(dispositions))
        target_tensors = _normalize_names(
            self.target_tensors,
            f"target tensors for {self.name!r}",
            allow_empty=True,
        )
        if not target_tensors and tuple(dispositions) != (Disposition.TARGET_NATIVE,):
            raise PlanValidationError(
                f"Tensorless role {self.name!r} must allow only target-native evidence."
            )
        object.__setattr__(self, "target_tensors", target_tensors)


@dataclass(frozen=True)
class RoleEvidence:
    """Auditable evidence that a required role was materialized."""

    role: str
    disposition: Disposition
    method: str
    target_tensors: tuple[str, ...]
    source_artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    random_initialization: bool = False


@dataclass(frozen=True)
class TransportPlan:
    """Immutable manifest of all roles a transport transaction must satisfy."""

    name: str
    required_roles: tuple[RequiredTargetRole, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, "plan name"))
        roles = tuple(self.required_roles)
        if not roles:
            raise PlanValidationError(
                "A transport plan needs at least one required role."
            )
        if any(not isinstance(role, RequiredTargetRole) for role in roles):
            raise PlanValidationError(
                "TransportPlan.required_roles must contain RequiredTargetRole values."
            )
        names = [role.name for role in roles]
        if len(set(names)) != len(names):
            raise PlanValidationError("A transport plan contains duplicate role names.")
        object.__setattr__(
            self, "required_roles", tuple(sorted(roles, key=lambda role: role.name))
        )

    @property
    def roles_by_name(self) -> dict[str, RequiredTargetRole]:
        return {role.name: role for role in self.required_roles}


@dataclass(frozen=True)
class GateDecision:
    """Deterministic external acceptance decision for a complete transaction."""

    accepted: bool
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportTransactionResult:
    """Accepted or gate-rejected transaction result."""

    accepted: bool
    report: Mapping[str, Any]


@dataclass
class _TensorState:
    name: str
    tensor: Any
    value: Any
    requires_grad: bool
    grad: Any | None
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any
    device: Any
    layout: Any


@dataclass
class _ModuleState:
    name: str
    module: Any
    training: bool
    modules: tuple[tuple[str, Any], ...]
    parameters: tuple[tuple[str, Any], ...]
    buffers: tuple[tuple[str, Any], ...]
    non_persistent_buffers: frozenset[str]


def _clone_tensor(torch, tensor):
    return tensor.detach().clone()


def _tensor_bits_equal(torch, left, right) -> bool:
    if (
        tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
        or left.layout != right.layout
        or tuple(left.stride()) != tuple(right.stride())
    ):
        return False
    if left.layout == torch.strided:
        try:
            left_bytes = left.detach().contiguous().view(torch.uint8)
            right_bytes = right.detach().contiguous().view(torch.uint8)
            return bool(torch.equal(left_bytes, right_bytes))
        except (RuntimeError, TypeError):
            pass
    return bool(torch.equal(left.detach(), right.detach()))


class TargetSnapshot:
    """Exact restorable snapshot of named tensors and registered module state."""

    def __init__(
        self,
        *,
        tensor_states: Sequence[_TensorState],
        module_states: Sequence[_ModuleState],
    ) -> None:
        self._tensor_states = tuple(tensor_states)
        self._module_states = tuple(module_states)

    @classmethod
    def capture(cls, targets: Mapping[str, Any]) -> TargetSnapshot:
        torch = _require_torch()
        if not isinstance(targets, Mapping) or not targets:
            raise PlanValidationError(
                "Transaction targets must be a non-empty mapping."
            )

        tensor_states: list[_TensorState] = []
        module_states: list[_ModuleState] = []
        inventory_names: set[str] = set()

        def add_tensor(name: str, tensor: Any) -> None:
            if name in inventory_names:
                raise PlanValidationError(
                    f"Target inventory contains duplicate tensor name {name!r}."
                )
            if not isinstance(tensor, torch.Tensor):
                raise PlanValidationError(f"Target {name!r} is not a torch tensor.")
            if not tensor.is_leaf:
                raise PlanValidationError(
                    f"Target tensor {name!r} must be a leaf tensor for exact restoration."
                )
            inventory_names.add(name)
            grad = tensor.grad
            tensor_states.append(
                _TensorState(
                    name=name,
                    tensor=tensor,
                    value=_clone_tensor(torch, tensor),
                    requires_grad=bool(tensor.requires_grad),
                    grad=None if grad is None else _clone_tensor(torch, grad),
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    layout=tensor.layout,
                )
            )

        def walk_module(name: str, module: Any, ancestors: frozenset[int]) -> None:
            if id(module) in ancestors:
                raise PlanValidationError(
                    f"Target module graph contains a cycle at {name!r}."
                )
            module_states.append(
                _ModuleState(
                    name=name,
                    module=module,
                    training=bool(module.training),
                    modules=tuple(module._modules.items()),
                    parameters=tuple(module._parameters.items()),
                    buffers=tuple(module._buffers.items()),
                    non_persistent_buffers=frozenset(
                        module._non_persistent_buffers_set
                    ),
                )
            )
            for local_name, parameter in module._parameters.items():
                if parameter is not None:
                    add_tensor(f"{name}.{local_name}", parameter)
            for local_name, buffer in module._buffers.items():
                if buffer is not None:
                    add_tensor(f"{name}.{local_name}", buffer)
            next_ancestors = ancestors | {id(module)}
            for local_name, child in module._modules.items():
                if child is not None:
                    walk_module(f"{name}.{local_name}", child, next_ancestors)

        raw_names = tuple(targets)
        if any(not isinstance(name, str) for name in raw_names):
            raise PlanValidationError("Transaction target names must be strings.")
        seen_target_names: set[str] = set()
        for raw_name in sorted(raw_names):
            name = _validate_name(raw_name, "target name")
            if name in seen_target_names:
                raise PlanValidationError(f"Duplicate target name {name!r}.")
            seen_target_names.add(name)
            target = targets[raw_name]
            if isinstance(target, torch.nn.Module):
                walk_module(name, target, frozenset())
            elif isinstance(target, torch.Tensor):
                add_tensor(name, target)
            else:
                raise PlanValidationError(
                    f"Target {name!r} must be a torch Tensor or nn.Module."
                )
        if not tensor_states:
            raise PlanValidationError("Transaction target modules contain no tensors.")
        return cls(tensor_states=tensor_states, module_states=module_states)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(state.name for state in self._tensor_states))

    def inventory_report(self) -> list[dict[str, Any]]:
        return [
            {
                "name": state.name,
                "shape": list(state.shape),
                "dtype": str(state.dtype),
                "requires_grad": state.requires_grad,
            }
            for state in sorted(self._tensor_states, key=lambda state: state.name)
        ]

    def restore(self) -> None:
        torch = _require_torch()
        with torch.no_grad():
            for state in sorted(
                self._module_states,
                key=lambda state: state.name.count("."),
            ):
                state.module._modules.clear()
                state.module._modules.update(state.modules)
                state.module._parameters.clear()
                state.module._parameters.update(state.parameters)
                state.module._buffers.clear()
                state.module._buffers.update(state.buffers)
                state.module._non_persistent_buffers_set = set(
                    state.non_persistent_buffers
                )
                state.module.training = state.training

            restored_tensors: set[int] = set()
            for state in self._tensor_states:
                tensor = state.tensor
                if id(tensor) not in restored_tensors:
                    tensor.data = _clone_tensor(torch, state.value)
                    if bool(tensor.requires_grad) != state.requires_grad:
                        tensor.requires_grad_(state.requires_grad)
                    tensor.grad = (
                        None if state.grad is None else _clone_tensor(torch, state.grad)
                    )
                    restored_tensors.add(id(tensor))

    def assert_exact(self) -> None:
        problems = self._invariant_problems(include_values=True)
        if problems:
            raise RollbackError(
                "Target rollback was not exact: " + "; ".join(sorted(problems))
            )

    def assert_unchanged(self, context: str) -> None:
        """Require values and transaction invariants to match this snapshot."""

        problems = self._invariant_problems(include_values=True)
        if problems:
            raise TransactionInvariantError(
                f"{context} mutated transaction targets: " + "; ".join(sorted(problems))
            )

    def assert_non_training_invariants(self) -> None:
        problems = self._invariant_problems(include_values=False)
        if problems:
            raise TransactionInvariantError(
                "Transport transaction escaped its mutation bounds: "
                + "; ".join(sorted(problems))
            )

    def _invariant_problems(self, *, include_values: bool) -> list[str]:
        torch = _require_torch()
        problems: list[str] = []
        for state in self._module_states:
            module = state.module
            if bool(module.training) != state.training:
                problems.append(f"module mode changed at {state.name}")
            for label, current, expected in (
                ("submodules", tuple(module._modules.items()), state.modules),
                ("parameters", tuple(module._parameters.items()), state.parameters),
                ("buffers", tuple(module._buffers.items()), state.buffers),
            ):
                if len(current) != len(expected) or any(
                    left_name != right_name or left_value is not right_value
                    for (left_name, left_value), (right_name, right_value) in zip(
                        current, expected, strict=True
                    )
                ):
                    problems.append(f"{label} changed at {state.name}")
            if (
                frozenset(module._non_persistent_buffers_set)
                != state.non_persistent_buffers
            ):
                problems.append(f"buffer persistence changed at {state.name}")

        checked_tensors: set[int] = set()
        for state in self._tensor_states:
            tensor = state.tensor
            if id(tensor) in checked_tensors:
                continue
            checked_tensors.add(id(tensor))
            if (
                tuple(tensor.shape) != state.shape
                or tuple(tensor.stride()) != state.stride
                or tensor.dtype != state.dtype
                or tensor.device != state.device
                or tensor.layout != state.layout
            ):
                problems.append(f"tensor metadata changed at {state.name}")
                continue
            if bool(tensor.requires_grad) != state.requires_grad:
                problems.append(f"requires_grad changed at {state.name}")
            if (tensor.grad is None) != (state.grad is None) or (
                tensor.grad is not None
                and state.grad is not None
                and not _tensor_bits_equal(torch, tensor.grad, state.grad)
            ):
                problems.append(f"gradient changed at {state.name}")
            if include_values and not _tensor_bits_equal(torch, tensor, state.value):
                problems.append(f"tensor value changed at {state.name}")
        return problems


class TransportCoverage:
    """Single-use evidence ledger for one transport transaction."""

    def __init__(
        self,
        plan: TransportPlan,
        available_target_tensors: Sequence[str],
    ) -> None:
        self._plan = plan
        self._available_targets = frozenset(available_target_tensors)
        self._evidence: dict[str, dict[str, Any]] = {}
        self._violations: list[dict[str, str]] = []
        missing_targets = sorted(
            {
                tensor_name
                for role in plan.required_roles
                for tensor_name in role.target_tensors
                if tensor_name not in self._available_targets
            }
        )
        if missing_targets:
            raise PlanValidationError(
                "Required role tensors are absent from the transaction target inventory: "
                + ", ".join(missing_targets)
            )

    def record(self, evidence: RoleEvidence) -> None:
        """Record one role, permanently latching any rejected evidence attempt."""

        try:
            self._record_once(evidence)
        except EvidenceValidationError as exc:
            if isinstance(exc, UnknownRoleError):
                kind = "unknown-role"
            elif isinstance(exc, DuplicateRoleEvidenceError):
                kind = "duplicate-role"
            else:
                kind = "invalid-evidence"
            role = (
                evidence.role
                if isinstance(evidence, RoleEvidence)
                and isinstance(evidence.role, str)
                and evidence.role
                else "<invalid>"
            )
            self._violations.append({"kind": kind, "role": role, "reason": str(exc)})
            raise
        except PlanValidationError as exc:
            wrapped = EvidenceValidationError(str(exc))
            role = (
                evidence.role
                if isinstance(evidence, RoleEvidence)
                and isinstance(evidence.role, str)
                and evidence.role
                else "<invalid>"
            )
            self._violations.append(
                {"kind": "invalid-evidence", "role": role, "reason": str(exc)}
            )
            raise wrapped from exc

    def _record_once(self, evidence: RoleEvidence) -> None:
        if not isinstance(evidence, RoleEvidence):
            raise EvidenceValidationError("record() requires a RoleEvidence instance.")
        role_name = _validate_name(evidence.role, "evidence role")
        requirement = self._plan.roles_by_name.get(role_name)
        if requirement is None:
            raise UnknownRoleError(f"Evidence references unknown role {role_name!r}.")
        if role_name in self._evidence:
            raise DuplicateRoleEvidenceError(
                f"Role {role_name!r} already has an evidence record."
            )
        try:
            disposition = Disposition(evidence.disposition)
        except (TypeError, ValueError) as exc:
            raise EvidenceValidationError(
                f"Role {role_name!r} uses unknown disposition {evidence.disposition!r}."
            ) from exc
        if disposition not in requirement.allowed_dispositions:
            raise EvidenceValidationError(
                f"Disposition {disposition.value!r} is not allowed for role {role_name!r}."
            )
        if evidence.random_initialization:
            raise EvidenceValidationError(
                f"Random initialization cannot satisfy required role {role_name!r}."
            )
        method = _validate_name(evidence.method, f"method for role {role_name!r}")
        target_tensors = _normalize_names(
            evidence.target_tensors,
            f"evidence target tensors for role {role_name!r}",
            allow_empty=True,
        )
        if target_tensors != requirement.target_tensors:
            raise EvidenceValidationError(
                f"Evidence for role {role_name!r} must cover exactly "
                f"{list(requirement.target_tensors)!r}; got {list(target_tensors)!r}."
            )
        unknown_targets = sorted(set(target_tensors) - self._available_targets)
        if unknown_targets:
            raise EvidenceValidationError(
                f"Evidence for role {role_name!r} references unknown target tensors: "
                + ", ".join(unknown_targets)
            )
        if isinstance(evidence.source_artifacts, (str, bytes)):
            raise EvidenceValidationError(
                f"Source artifacts for role {role_name!r} must be a sequence, not a string."
            )
        source_artifacts = tuple(
            sorted(
                _validate_name(value, f"source artifact for role {role_name!r}")
                for value in evidence.source_artifacts
            )
        )
        if len(set(source_artifacts)) != len(source_artifacts):
            raise EvidenceValidationError(
                f"Evidence for role {role_name!r} repeats a source artifact."
            )
        if disposition is not Disposition.TARGET_NATIVE and not source_artifacts:
            raise EvidenceValidationError(
                f"Evidence for {disposition.value} role {role_name!r} needs a source artifact."
            )
        metadata = _normalize_json(evidence.metadata)
        self._evidence[role_name] = {
            "role": role_name,
            "disposition": disposition.value,
            "method": method,
            "target_tensors": list(target_tensors),
            "source_artifacts": list(source_artifacts),
            "random_initialization": False,
            "metadata": metadata,
        }

    def validate_complete(self) -> dict[str, Any]:
        report = self.report()
        if report["violations"]:
            raise RejectedCoverageError(
                "The coverage ledger contains rejected evidence attempts.",
                report,
            )
        if not report["complete"]:
            missing = report["missing_roles"]
            raise IncompleteCoverageError(
                "Required target roles lack evidence: " + ", ".join(missing),
                report,
            )
        return report

    def report(self) -> dict[str, Any]:
        missing = sorted(set(self._plan.roles_by_name) - set(self._evidence))
        violations = sorted(
            {(item["kind"], item["role"], item["reason"]) for item in self._violations}
        )
        violation_reports = [
            {"kind": kind, "role": role, "reason": reason}
            for kind, role, reason in violations
        ]
        roles = []
        for requirement in self._plan.required_roles:
            roles.append(
                {
                    "role": requirement.name,
                    "description": requirement.description,
                    "required_target_tensors": list(requirement.target_tensors),
                    "allowed_dispositions": [
                        disposition.value
                        for disposition in requirement.allowed_dispositions
                    ],
                    "evidence": self._evidence.get(requirement.name),
                }
            )
        return {
            "plan": self._plan.name,
            "required_role_count": len(self._plan.required_roles),
            "evidenced_role_count": len(self._evidence),
            "complete": not missing and not violations,
            "missing_roles": missing,
            "unknown_roles": sorted(
                {role for kind, role, _reason in violations if kind == "unknown-role"}
            ),
            "duplicate_roles": sorted(
                {role for kind, role, _reason in violations if kind == "duplicate-role"}
            ),
            "invalid_roles": sorted(
                {
                    role
                    for kind, role, _reason in violations
                    if kind == "invalid-evidence"
                }
            ),
            "violations": violation_reports,
            "roles": roles,
        }


def _normalize_gate_decision(value: bool | GateDecision) -> dict[str, Any]:
    if isinstance(value, bool):
        return {
            "accepted": value,
            "reason": "accepted" if value else "rejected",
            "evidence": {},
        }
    if not isinstance(value, GateDecision):
        raise TransactionInvariantError(
            "A transaction gate must return bool or GateDecision."
        )
    if not isinstance(value.accepted, bool):
        raise TransactionInvariantError("GateDecision.accepted must be bool.")
    reason = _validate_name(value.reason, "gate reason")
    return {
        "accepted": value.accepted,
        "reason": reason,
        "evidence": _normalize_json(value.evidence, "gate evidence"),
    }


def _rollback_after_failure(snapshot: TargetSnapshot) -> None:
    try:
        snapshot.restore()
        snapshot.assert_exact()
    except BaseException as exc:
        if isinstance(exc, RollbackError):
            raise
        raise RollbackError("Transport rollback raised an exception.") from exc


def run_transport_transaction(
    *,
    plan: TransportPlan,
    targets: Mapping[str, Any],
    mutation: Callable[[TransportCoverage], None],
    gate: Callable[[Mapping[str, Any]], bool | GateDecision] | None = None,
) -> TransportTransactionResult:
    """Apply a zero-gradient transport mutation as an all-or-nothing transaction.

    ``mutation`` receives the transaction's evidence ledger.  It must record every
    required role exactly once.  The optional gate receives only the deterministic
    coverage report.  A false gate restores the snapshot and returns a rejected
    result; exceptions and validation failures restore the snapshot and re-raise.
    """

    torch = _require_torch()
    if not isinstance(plan, TransportPlan):
        raise PlanValidationError("plan must be a TransportPlan.")
    if not callable(mutation):
        raise PlanValidationError("mutation must be callable.")
    if gate is not None and not callable(gate):
        raise PlanValidationError("gate must be callable when supplied.")

    snapshot = TargetSnapshot.capture(targets)
    coverage = TransportCoverage(plan, snapshot.tensor_names)

    try:
        with torch.no_grad():
            mutation_result = mutation(coverage)
        if mutation_result is not None:
            raise TransactionInvariantError(
                "Transport mutation callbacks must return None; evidence belongs in the ledger."
            )
        snapshot.assert_non_training_invariants()
        coverage_report = coverage.validate_complete()
        candidate_snapshot = TargetSnapshot.capture(targets)
        with torch.no_grad():
            gate_report = _normalize_gate_decision(
                True if gate is None else gate(coverage_report)
            )
        candidate_snapshot.assert_unchanged("Transaction gate")
    except BaseException:
        _rollback_after_failure(snapshot)
        raise

    accepted = bool(gate_report["accepted"])
    rollback_report = {"performed": False, "exact": None}
    if not accepted:
        _rollback_after_failure(snapshot)
        rollback_report = {"performed": True, "exact": True}

    report = _with_report_hash(
        {
            "schema": REPORT_SCHEMA,
            "plan": plan.name,
            "accepted": accepted,
            "coverage": coverage_report,
            "gate": gate_report,
            "rollback": rollback_report,
            "target_inventory": snapshot.inventory_report(),
            "execution_bounds": {
                "core_optimizer_steps": 0,
                "core_forward_calls": 0,
                "core_backward_calls": 0,
                "mutation_grad_mode_disabled": True,
                "gate_grad_mode_disabled": True,
                "gradients_unchanged": True,
                "module_training_modes_unchanged": True,
            },
        }
    )
    return TransportTransactionResult(accepted=accepted, report=report)


__all__ = [
    "REPORT_SCHEMA",
    "CoverageValidationError",
    "Disposition",
    "DuplicateRoleEvidenceError",
    "EvidenceValidationError",
    "GateDecision",
    "IncompleteCoverageError",
    "PlanValidationError",
    "RequiredTargetRole",
    "RejectedCoverageError",
    "RoleEvidence",
    "RollbackError",
    "TargetSnapshot",
    "TransactionInvariantError",
    "TransportCoverage",
    "TransportError",
    "TransportPlan",
    "TransportTransactionResult",
    "UnknownRoleError",
    "run_transport_transaction",
]
