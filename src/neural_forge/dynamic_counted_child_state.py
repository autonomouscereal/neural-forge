"""Exact accounting for standalone child state outside dense model weights.

The dense topology planner counts registered model parameters.  Some valid
folds also need fixed continuous state at runtime (for example independent
``C_in`` and ``C_out`` semantic-route coefficients).  Those values are part of
the child even when they are not registered ``torch.nn.Parameter`` objects.

This module reserves them *inside* the requested target count and derives the
smaller dense-model budget.  It deliberately gives no credit for quantization,
sparse storage, shared storage, a parent sidecar, or an unpackaged artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_parameter_frontier import DenseParameterBudgetRequest


SCHEMA = "neural-forge-counted-child-state-plan-v1"
ALLOCATION_SCHEMA = "neural-forge-continuous-child-state-allocation-v1"
VERIFICATION_SCHEMA = "neural-forge-counted-child-state-verification-v1"
_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


class CountedChildStateError(ValueError):
    """A child-state allocation is invalid or would exceed its target."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CountedChildStateError(f"{name} must be nonempty.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CountedChildStateError(f"{name} must be a positive integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CountedChildStateError(f"{name} must be a lowercase SHA-256.")
    return value


def _dtype_bytes(dtype: str) -> int:
    return 2 if dtype in {"float16", "bfloat16"} else 4 if dtype == "float32" else 8


@dataclass(frozen=True, slots=True)
class ContinuousChildStateAllocation:
    """One independently counted continuous tensor packaged with the child."""

    state_id: str
    parameter_elements: int
    tensor_sha256: str
    artifact_sha256: str
    dtype: str = "float32"
    packaged_with_child: bool = True
    storage_shared_with_dense_model: bool = False
    storage_shared_with_other_allocation: bool = False
    parent_runtime_dependency: bool = False
    quantized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _text(self.state_id, "state_id"))
        object.__setattr__(
            self,
            "parameter_elements",
            _positive_int(self.parameter_elements, "parameter_elements"),
        )
        object.__setattr__(
            self,
            "tensor_sha256",
            _digest(self.tensor_sha256, "tensor_sha256"),
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest(self.artifact_sha256, "artifact_sha256"),
        )
        dtype = _text(self.dtype, "dtype").removeprefix("torch.").lower()
        if dtype not in _FLOAT_DTYPES:
            raise CountedChildStateError(
                "dtype must be float16, bfloat16, float32, or float64."
            )
        object.__setattr__(self, "dtype", dtype)
        boolean_fields = (
            "packaged_with_child",
            "storage_shared_with_dense_model",
            "storage_shared_with_other_allocation",
            "parent_runtime_dependency",
            "quantized",
        )
        if any(not isinstance(getattr(self, name), bool) for name in boolean_fields):
            raise CountedChildStateError("Allocation policy fields must be boolean.")
        if not self.packaged_with_child:
            raise CountedChildStateError(
                "Continuous runtime state must be packaged with the standalone child."
            )
        if self.storage_shared_with_dense_model or self.storage_shared_with_other_allocation:
            raise CountedChildStateError(
                "Continuous state cannot receive artificial shared-storage credit."
            )
        if self.parent_runtime_dependency:
            raise CountedChildStateError(
                "Continuous child state cannot depend on a parent runtime sidecar."
            )
        if self.quantized:
            raise CountedChildStateError(
                "Quantization cannot satisfy the fold-only parameter target."
            )

    @property
    def storage_bytes(self) -> int:
        return self.parameter_elements * _dtype_bytes(self.dtype)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ALLOCATION_SCHEMA,
            "state_id": self.state_id,
            "parameter_elements": self.parameter_elements,
            "tensor_sha256": self.tensor_sha256,
            "artifact_sha256": self.artifact_sha256,
            "dtype": self.dtype,
            "storage_bytes": self.storage_bytes,
            "packaged_with_child": self.packaged_with_child,
            "storage_shared_with_dense_model": self.storage_shared_with_dense_model,
            "storage_shared_with_other_allocation": (
                self.storage_shared_with_other_allocation
            ),
            "parent_runtime_dependency": self.parent_runtime_dependency,
            "quantized": self.quantized,
            "counts_as_child_state": True,
        }


@dataclass(frozen=True, slots=True)
class CountedChildStatePlan:
    """Split one exact child target between dense weights and fixed state."""

    parent_unique_parameters: int
    target_total_parameter_elements: int
    dense_model_parameter_elements: int
    continuous_allocations: tuple[ContinuousChildStateAllocation, ...]

    def __post_init__(self) -> None:
        parent = _positive_int(
            self.parent_unique_parameters,
            "parent_unique_parameters",
        )
        target = _positive_int(
            self.target_total_parameter_elements,
            "target_total_parameter_elements",
        )
        dense = _positive_int(
            self.dense_model_parameter_elements,
            "dense_model_parameter_elements",
        )
        allocations = tuple(self.continuous_allocations)
        if not allocations or any(
            not isinstance(value, ContinuousChildStateAllocation)
            for value in allocations
        ):
            raise CountedChildStateError(
                "continuous_allocations must contain typed allocations."
            )
        allocations = tuple(sorted(allocations, key=lambda value: value.state_id))
        if len({value.state_id for value in allocations}) != len(allocations):
            raise CountedChildStateError("Continuous state IDs must be unique.")
        continuous = sum(value.parameter_elements for value in allocations)
        if dense + continuous != target:
            raise CountedChildStateError(
                "Dense weights plus every continuous allocation must equal the exact "
                "child target."
            )
        if target >= parent:
            raise CountedChildStateError(
                "The counted child target must be smaller than the parent."
            )
        object.__setattr__(self, "parent_unique_parameters", parent)
        object.__setattr__(self, "target_total_parameter_elements", target)
        object.__setattr__(self, "dense_model_parameter_elements", dense)
        object.__setattr__(self, "continuous_allocations", allocations)

    @property
    def continuous_parameter_elements(self) -> int:
        return sum(value.parameter_elements for value in self.continuous_allocations)

    @property
    def realized_parameter_compaction_ratio(self) -> float:
        return self.parent_unique_parameters / self.target_total_parameter_elements

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": SCHEMA,
            "parent_unique_parameters": self.parent_unique_parameters,
            "target_total_parameter_elements": self.target_total_parameter_elements,
            "dense_model_parameter_elements": self.dense_model_parameter_elements,
            "continuous_parameter_elements": self.continuous_parameter_elements,
            "continuous_allocations": [
                value.as_dict() for value in self.continuous_allocations
            ],
            "realized_parameter_compaction_ratio": (
                self.realized_parameter_compaction_ratio
            ),
            "all_continuous_state_counted_once": True,
            "all_continuous_state_packaged_with_child": True,
            "artificial_parameter_sharing_credit": False,
            "parent_runtime_sidecar": False,
            "quantization_credit": False,
            "pruning_credit": False,
        }
        if include_fingerprint:
            body["fingerprint"] = canonical_sha256(body)
        return body

    def dense_budget(self, request: DenseParameterBudgetRequest) -> DenseParameterBudgetRequest:
        """Return the physical dense budget after reserving fixed child state."""

        if not isinstance(request, DenseParameterBudgetRequest):
            raise CountedChildStateError(
                "request must be a DenseParameterBudgetRequest."
            )
        if (
            request.target_unique_parameters
            != self.target_total_parameter_elements
            or request.compression_ratio is not None
        ):
            raise CountedChildStateError(
                "The neutral request must name the plan's exact total child count."
            )
        return replace(
            request,
            target_unique_parameters=self.dense_model_parameter_elements,
        )


def build_counted_child_state_plan(
    *,
    parent_unique_parameters: int,
    target_total_parameter_elements: int,
    continuous_allocations: Sequence[ContinuousChildStateAllocation],
) -> CountedChildStatePlan:
    """Reserve every supplied allocation and derive the dense physical target."""

    parent = _positive_int(parent_unique_parameters, "parent_unique_parameters")
    target = _positive_int(
        target_total_parameter_elements,
        "target_total_parameter_elements",
    )
    allocations = tuple(continuous_allocations)
    if not allocations:
        raise CountedChildStateError(
            "At least one continuous child-state allocation is required."
        )
    continuous = sum(
        _positive_int(value.parameter_elements, "allocation parameter_elements")
        if isinstance(value, ContinuousChildStateAllocation)
        else 0
        for value in allocations
    )
    if any(
        not isinstance(value, ContinuousChildStateAllocation)
        for value in allocations
    ):
        raise CountedChildStateError(
            "continuous_allocations must contain typed allocations."
        )
    dense = target - continuous
    if dense <= 0:
        raise CountedChildStateError(
            "Continuous child state consumes the entire requested target."
        )
    return CountedChildStatePlan(parent, target, dense, allocations)


def counted_child_state_plan_from_dict(value: Mapping[str, Any]) -> CountedChildStatePlan:
    """Restore and verify a serialized counted-state plan."""

    if not isinstance(value, Mapping):
        raise CountedChildStateError("Counted child-state plan must be a mapping.")
    payload = dict(value)
    allowed = {
        "schema",
        "parent_unique_parameters",
        "target_total_parameter_elements",
        "dense_model_parameter_elements",
        "continuous_parameter_elements",
        "continuous_allocations",
        "realized_parameter_compaction_ratio",
        "all_continuous_state_counted_once",
        "all_continuous_state_packaged_with_child",
        "artificial_parameter_sharing_credit",
        "parent_runtime_sidecar",
        "quantization_credit",
        "pruning_credit",
        "fingerprint",
    }
    if set(payload) - allowed or payload.get("schema") != SCHEMA:
        raise CountedChildStateError("Counted child-state plan schema is invalid.")
    allocation_values = payload.get("continuous_allocations")
    if not isinstance(allocation_values, list) or not allocation_values:
        raise CountedChildStateError(
            "Counted child-state allocations must be a nonempty JSON list."
        )
    allocations: list[ContinuousChildStateAllocation] = []
    for index, raw in enumerate(allocation_values):
        if not isinstance(raw, Mapping):
            raise CountedChildStateError(f"Allocation {index} must be a mapping.")
        row = dict(raw)
        if row.pop("schema", None) != ALLOCATION_SCHEMA:
            raise CountedChildStateError(f"Allocation {index} schema is invalid.")
        row.pop("storage_bytes", None)
        row.pop("counts_as_child_state", None)
        allocations.append(ContinuousChildStateAllocation(**row))
    plan = CountedChildStatePlan(
        parent_unique_parameters=payload.get("parent_unique_parameters"),
        target_total_parameter_elements=payload.get(
            "target_total_parameter_elements"
        ),
        dense_model_parameter_elements=payload.get("dense_model_parameter_elements"),
        continuous_allocations=tuple(allocations),
    )
    expected = payload.get("fingerprint")
    if expected is not None and _digest(expected, "fingerprint") != plan.fingerprint:
        raise CountedChildStateError("Counted child-state plan fingerprint changed.")
    canonical = plan.as_dict(include_fingerprint=False)
    for key, observed in canonical.items():
        if key in payload and payload[key] != observed:
            raise CountedChildStateError(
                f"Counted child-state derived field {key!r} changed."
            )
    return plan


def verify_counted_child_state(
    plan: CountedChildStatePlan,
    *,
    physical_dense_parameter_elements: int,
    observed_allocations: Sequence[ContinuousChildStateAllocation],
) -> dict[str, Any]:
    """Seal proof that one published standalone child meets the exact target."""

    if not isinstance(plan, CountedChildStatePlan):
        raise CountedChildStateError("plan must be CountedChildStatePlan.")
    physical = _positive_int(
        physical_dense_parameter_elements,
        "physical_dense_parameter_elements",
    )
    observed = tuple(sorted(observed_allocations, key=lambda value: value.state_id))
    if physical != plan.dense_model_parameter_elements:
        raise CountedChildStateError(
            "Published dense model count differs from the reserved physical budget."
        )
    if observed != plan.continuous_allocations:
        raise CountedChildStateError(
            "Published continuous child state differs from the sealed allocation."
        )
    total = physical + sum(value.parameter_elements for value in observed)
    if total != plan.target_total_parameter_elements:
        raise CountedChildStateError("Published child does not meet the exact target.")
    body = {
        "schema": VERIFICATION_SCHEMA,
        "status": "standalone-counted-child-state-exact",
        "plan_fingerprint": plan.fingerprint,
        "parent_unique_parameters": plan.parent_unique_parameters,
        "physical_dense_parameter_elements": physical,
        "continuous_parameter_elements": plan.continuous_parameter_elements,
        "target_total_parameter_elements": total,
        "realized_parameter_compaction_ratio": (
            plan.realized_parameter_compaction_ratio
        ),
        "allocation_artifact_sha256s": {
            value.state_id: value.artifact_sha256 for value in observed
        },
        "allocation_tensor_sha256s": {
            value.state_id: value.tensor_sha256 for value in observed
        },
        "all_continuous_state_counted_once": True,
        "all_continuous_state_packaged_with_child": True,
        "parent_runtime_sidecar": False,
        "quantization_credit": False,
        "pruning_credit": False,
        "artificial_parameter_sharing_credit": False,
    }
    return {**body, "verification_sha256": canonical_sha256(body)}


__all__ = [
    "ALLOCATION_SCHEMA",
    "SCHEMA",
    "VERIFICATION_SCHEMA",
    "ContinuousChildStateAllocation",
    "CountedChildStateError",
    "CountedChildStatePlan",
    "build_counted_child_state_plan",
    "counted_child_state_plan_from_dict",
    "verify_counted_child_state",
]
