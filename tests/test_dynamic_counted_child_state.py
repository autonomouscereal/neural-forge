from __future__ import annotations

import pytest

from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    CountedChildStateError,
    build_counted_child_state_plan,
    counted_child_state_plan_from_dict,
    verify_counted_child_state,
)
from neural_forge.dynamic_parameter_frontier import DenseParameterBudgetRequest


PARENT = 229_693_184
EXACT_TOTAL = 57_420_144
CHANNEL_COEFFICIENTS = 1_933_888
EXPECTED_DENSE = 53_552_368


def _allocation(channel: str) -> ContinuousChildStateAllocation:
    digit = "1" if channel == "c_in" else "2"
    return ContinuousChildStateAllocation(
        state_id=f"lfm2.semantic.{channel}",
        parameter_elements=CHANNEL_COEFFICIENTS,
        tensor_sha256=digit * 64,
        artifact_sha256="a" * 64,
        dtype="float32",
    )


def _plan():
    return build_counted_child_state_plan(
        parent_unique_parameters=PARENT,
        target_total_parameter_elements=EXACT_TOTAL,
        continuous_allocations=(_allocation("c_in"), _allocation("c_out")),
    )


def test_exact4x_reserves_both_semantic_channels_inside_total() -> None:
    plan = _plan()

    assert plan.continuous_parameter_elements == 3_867_776
    assert plan.dense_model_parameter_elements == EXPECTED_DENSE
    assert (
        plan.dense_model_parameter_elements + plan.continuous_parameter_elements
        == EXACT_TOTAL
    )
    assert plan.realized_parameter_compaction_ratio == pytest.approx(
        PARENT / EXACT_TOTAL
    )
    assert [
        value.state_id for value in plan.continuous_allocations
    ] == ["lfm2.semantic.c_in", "lfm2.semantic.c_out"]
    assert plan.as_dict()["all_continuous_state_counted_once"] is True
    assert plan.as_dict()["artificial_parameter_sharing_credit"] is False


def test_dense_planner_receives_remainder_not_total_child_budget() -> None:
    plan = _plan()
    neutral = DenseParameterBudgetRequest(
        target_family="lfm2",
        target_modalities=("text",),
        target_unique_parameters=EXACT_TOTAL,
        target_vocabulary_size=4_096,
        match_policy="exact",
        maximum_candidates=32,
    )

    physical = plan.dense_budget(neutral)

    assert neutral.target_unique_parameters == EXACT_TOTAL
    assert physical.target_unique_parameters == EXPECTED_DENSE
    assert physical.target_family == neutral.target_family
    assert physical.target_modalities == neutral.target_modalities
    assert physical.target_vocabulary_size == neutral.target_vocabulary_size
    assert physical.match_policy == "exact"


def test_current_57420144_dense_child_plus_routes_is_rejected_as_over_budget() -> None:
    plan = _plan()

    with pytest.raises(CountedChildStateError, match="physical budget"):
        verify_counted_child_state(
            plan,
            physical_dense_parameter_elements=EXACT_TOTAL,
            observed_allocations=plan.continuous_allocations,
        )


def test_verification_seals_packaged_exact_standalone_state() -> None:
    plan = _plan()

    receipt = verify_counted_child_state(
        plan,
        physical_dense_parameter_elements=EXPECTED_DENSE,
        observed_allocations=tuple(reversed(plan.continuous_allocations)),
    )

    assert receipt["status"] == "standalone-counted-child-state-exact"
    assert receipt["target_total_parameter_elements"] == EXACT_TOTAL
    assert receipt["continuous_parameter_elements"] == 3_867_776
    assert receipt["parent_runtime_sidecar"] is False
    assert receipt["quantization_credit"] is False
    assert len(receipt["verification_sha256"]) == 64


def test_round_trip_rejects_derived_count_or_fingerprint_tamper() -> None:
    plan = _plan()
    payload = plan.as_dict()
    assert counted_child_state_plan_from_dict(payload) == plan

    changed_count = dict(payload)
    changed_count["continuous_parameter_elements"] += 1
    with pytest.raises(CountedChildStateError, match="derived field"):
        counted_child_state_plan_from_dict(changed_count)

    changed_fingerprint = dict(payload)
    changed_fingerprint["fingerprint"] = "f" * 64
    with pytest.raises(CountedChildStateError, match="fingerprint changed"):
        counted_child_state_plan_from_dict(changed_fingerprint)


@pytest.mark.parametrize(
    "changes, message",
    (
        ({"packaged_with_child": False}, "packaged"),
        ({"storage_shared_with_dense_model": True}, "shared-storage"),
        ({"storage_shared_with_other_allocation": True}, "shared-storage"),
        ({"parent_runtime_dependency": True}, "sidecar"),
        ({"quantized": True}, "Quantization"),
    ),
)
def test_allocation_refuses_free_or_external_continuous_state(
    changes,
    message,
) -> None:
    values = {
        "state_id": "lfm2.semantic.c_in",
        "parameter_elements": CHANNEL_COEFFICIENTS,
        "tensor_sha256": "1" * 64,
        "artifact_sha256": "a" * 64,
        "dtype": "float32",
    }
    values.update(changes)

    with pytest.raises(CountedChildStateError, match=message):
        ContinuousChildStateAllocation(**values)


def test_plan_refuses_total_that_cannot_hold_continuous_state() -> None:
    with pytest.raises(CountedChildStateError, match="entire requested target"):
        build_counted_child_state_plan(
            parent_unique_parameters=PARENT,
            target_total_parameter_elements=3_000_000,
            continuous_allocations=(_allocation("c_in"), _allocation("c_out")),
        )
