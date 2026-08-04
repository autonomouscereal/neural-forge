from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest
import torch

import neural_forge.hf_lfm2_moving_gauge_causal_targets as causal_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_moving_gauge_causal_targets import (
    Lfm2MovingGaugeCausalBoundaryPlan,
    Lfm2MovingGaugeCausalTargetAuthenticationError,
    Lfm2MovingGaugeCausalTargetError,
    authenticate_lfm2_moving_gauge_causal_state_receipt,
    build_lfm2_joint_writer_target,
    build_lfm2_moving_gauge_causal_state_receipt,
    build_lfm2_sequential_writer_target,
    project_lfm2_boundary_state,
    verify_lfm2_moving_gauge_causal_target_report,
)
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBoundaryEvidence,
    fit_lfm2_cycle_coupled_moving_gauge_boundary,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reseal(report: object, **changes: object) -> dict[str, object]:
    body = dict(report)  # type: ignore[arg-type]
    body.pop("result_sha256")
    body.update(changes)
    return {**body, "result_sha256": canonical_sha256(body)}


def _rotation(angle: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=torch.float64,
    )


def _base_encoder(*, foreign: bool = False) -> torch.Tensor:
    tail = [0.75, 0.125] if foreign else [0.25, -0.5]
    return torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], tail],
        dtype=torch.float64,
    )


def _operator_kind(layer_index: int) -> str:
    return "full_attention" if layer_index in {2, 4, 6, 8, 10, 12} else "short_conv"


def _topology(layer_count: int = 14) -> dict[str, object]:
    body = {
        "schema": "neural-forge-lfm2-layer-topology-v1",
        "layer_count": layer_count,
        "layers": [
            {
                "container_index": index,
                "layer_class": "tests.Lfm2DecoderLayer",
                "operator_class": "tests.Lfm2Operator",
                "feed_forward_class": "tests.Lfm2MLP",
                "contract": {
                    "schema": "neural-forge-lfm2-layer-contract-v1",
                    "layer_index": index,
                    "kind": _operator_kind(index),
                },
            }
            for index in range(layer_count)
        ],
    }
    return {**body, "layer_topology_sha256": canonical_sha256(body)}


def _fit_chart_pair(
    boundary_id: str,
    parent_rows: torch.Tensor,
    child_rows: torch.Tensor,
    *,
    role: str,
    locked: bool,
    child_recapture_sha256: str = _sha("child-recapture"),
    foreign_base: bool = False,
) -> tuple[Lfm2MovingGaugeBoundaryEvidence, object]:
    evidence = Lfm2MovingGaugeBoundaryEvidence(
        boundary_id=boundary_id,
        interface_role=role,
        interface_locked=locked,
        base_encoder=_base_encoder(foreign=foreign_base),
        fit_parent_rows=parent_rows,
        fit_child_rows=child_rows,
        fit_weights=torch.ones(len(parent_rows), dtype=torch.float64),
        fit_provenance=f"fit:{boundary_id}",
        row_alignment_sha256=_sha("alignment"),
        cached_parent_evidence_sha256=_sha("parent-cache"),
        live_child_recapture_sha256=child_recapture_sha256,
    )
    return evidence, fit_lfm2_cycle_coupled_moving_gauge_boundary(evidence)


def _plan(
    *,
    layer_index: int = 0,
    layer_count: int = 14,
    include_post_operator: bool = True,
) -> Lfm2MovingGaugeCausalBoundaryPlan:
    generator = torch.Generator().manual_seed(105 + layer_index)
    parent = torch.randn(48, 3, generator=generator, dtype=torch.float64)
    encoded = parent @ _base_encoder()
    pre_role = "input" if layer_index == 0 else "intermediate"
    pre_locked = layer_index == 0
    pre_evidence, pre = _fit_chart_pair(
        f"layer.{layer_index:03d}.residual.pre",
        parent,
        encoded if pre_locked else encoded @ _rotation(0.19),
        role=pre_role,
        locked=pre_locked,
    )
    post_operator_evidence = None
    post_operator = None
    if include_post_operator:
        post_operator_evidence, post_operator = _fit_chart_pair(
            f"layer.{layer_index:03d}.residual.post_operator",
            parent,
            encoded @ _rotation(0.63),
            role="intermediate",
            locked=False,
        )
    terminal_evidence, terminal = _fit_chart_pair(
        "stack.terminal.residual",
        parent,
        encoded,
        role="terminal",
        locked=True,
    )
    if layer_index == layer_count - 1:
        post_layer_evidence, post_layer = terminal_evidence, terminal
    else:
        post_layer_evidence, post_layer = _fit_chart_pair(
            f"layer.{layer_index:03d}.residual.post_layer",
            parent,
            encoded @ _rotation(-0.41),
            role="intermediate",
            locked=False,
        )
    return Lfm2MovingGaugeCausalBoundaryPlan(
        layer_index=layer_index,
        layer_count=layer_count,
        operator_kind=_operator_kind(layer_index),
        pre_chart=pre,
        post_layer_chart=post_layer,
        terminal_chart=terminal,
        pre_evidence=pre_evidence,
        post_layer_evidence=post_layer_evidence,
        terminal_evidence=terminal_evidence,
        parent_cache_sha256=_sha("parent-cache"),
        child_state_sha256=_sha("child-state"),
        child_recapture_sha256=_sha("child-recapture"),
        row_alignment_sha256=_sha("alignment"),
        split_identity_sha256=_sha("split"),
        layer_topology=_topology(layer_count),
        post_operator_chart=post_operator,
        post_operator_evidence=post_operator_evidence,
    )


def _joint_inputs():
    parent_post_layer = torch.tensor(
        [[[0.5, -0.2, 0.4], [1.2, 0.7, -0.3]]], dtype=torch.float64
    )
    actual_pre = torch.tensor([[[0.1, 0.3], [-0.4, 0.9]]], dtype=torch.float64)
    existing_group = torch.tensor([[[0.4, -0.6], [0.2, 0.1]]], dtype=torch.float64)
    fixed_other = torch.tensor([[[0.03, -0.02], [0.07, 0.04]]], dtype=torch.float64)
    return (
        parent_post_layer,
        actual_pre,
        actual_pre + existing_group + fixed_other,
        existing_group,
        fixed_other,
    )


def _joint_target(plan: Lfm2MovingGaugeCausalBoundaryPlan):
    parent, pre, post, existing, fixed = _joint_inputs()
    receipt = build_lfm2_moving_gauge_causal_state_receipt(
        plan,
        kind="baseline_current_child",
        tensors={
            "actual_child_post_layer": post,
            "actual_child_pre": pre,
            "existing_combined_writer_output": existing,
            "parent_post_layer": parent,
        },
    )
    target = build_lfm2_joint_writer_target(
        plan,
        parent_post_layer=parent,
        actual_child_pre=pre,
        actual_child_post_layer=post,
        existing_combined_writer_output=existing,
        state_receipt=receipt,
    )
    return target, parent, pre, existing, fixed


def _sequential_target(plan: Lfm2MovingGaugeCausalBoundaryPlan):
    parent_pre = torch.tensor(
        [[[0.3, -0.7, 0.2], [1.1, 0.2, -0.5]]], dtype=torch.float64
    )
    parent_post_operator = parent_pre + torch.tensor(
        [[[0.8, -0.1, 0.3], [-0.4, 0.5, 0.2]]], dtype=torch.float64
    )
    parent_post_layer = parent_post_operator + torch.tensor(
        [[[0.2, 0.6, -0.4], [0.3, -0.9, 0.1]]], dtype=torch.float64
    )
    actual_child_pre = plan.pre_chart.chart.encode_shaped(parent_pre) + 0.125
    existing_operator = torch.tensor([[[0.4, -0.2], [0.1, 0.3]]], dtype=torch.float64)
    operator_fixed = torch.full_like(actual_child_pre, 0.015)
    actual_post_operator = actual_child_pre + existing_operator + operator_fixed
    expected_operator_endpoint = plan.post_operator_chart.chart.encode_shaped(
        parent_post_operator
    )
    fresh_post_operator = expected_operator_endpoint + torch.tensor(
        [[[0.02, -0.04], [-0.03, 0.01]]], dtype=torch.float64
    )
    existing_mlp = torch.tensor([[[0.2, 0.1], [-0.5, 0.4]]], dtype=torch.float64)
    mlp_fixed = torch.full_like(actual_child_pre, -0.01)
    fresh_post_layer = fresh_post_operator + existing_mlp + mlp_fixed
    operator_receipt = build_lfm2_moving_gauge_causal_state_receipt(
        plan,
        kind="baseline_current_child",
        tensors={
            "actual_child_post_operator": actual_post_operator,
            "actual_child_pre": actual_child_pre,
            "existing_operator_writer_output": existing_operator,
            "parent_post_operator": parent_post_operator,
        },
    )
    mlp_receipt = build_lfm2_moving_gauge_causal_state_receipt(
        plan,
        kind="operator_conditioned_temporary_intervention",
        tensors={
            "existing_mlp_writer_output": existing_mlp,
            "fresh_child_post_layer": fresh_post_layer,
            "fresh_child_post_operator": fresh_post_operator,
            "parent_post_layer": parent_post_layer,
        },
        child_recapture_sha256=_sha("fresh-recapture"),
        operator_solution_sha256=_sha("operator-solution"),
    )
    target = build_lfm2_sequential_writer_target(
        plan,
        parent_post_operator=parent_post_operator,
        actual_child_pre=actual_child_pre,
        actual_child_post_operator=actual_post_operator,
        existing_operator_writer_output=existing_operator,
        parent_post_layer=parent_post_layer,
        fresh_child_post_operator=fresh_post_operator,
        fresh_child_post_layer=fresh_post_layer,
        existing_mlp_writer_output=existing_mlp,
        operator_source_receipt=operator_receipt,
        mlp_source_receipt=mlp_receipt,
    )
    return target, actual_child_pre, operator_fixed, fresh_post_operator, mlp_fixed


@pytest.mark.parametrize("layer_index", (0, 10))
def test_sequential_targets_close_observed_state_conditioned_endpoints(
    layer_index: int,
) -> None:
    plan = _plan(layer_index=layer_index)
    target, actual_child_pre, operator_fixed, fresh_post_operator, mlp_fixed = (
        _sequential_target(plan)
    )

    torch.testing.assert_close(target.operator_fixed_other_write, operator_fixed)
    torch.testing.assert_close(target.mlp_fixed_other_write, mlp_fixed)
    torch.testing.assert_close(
        actual_child_pre + operator_fixed + target.operator_write,
        target.operator_endpoint,
        rtol=0.0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        fresh_post_operator + mlp_fixed + target.mlp_write,
        target.post_layer_endpoint,
        rtol=0.0,
        atol=1e-15,
    )
    assert target.verify_authentication()
    assert target.report["operator_kind"] == _operator_kind(layer_index)
    assert target.report["evidence_stage"] == ("temporary_operator_mlp_target_ready")
    assert target.report["full_post_layer_reclosure_verified"] is False
    assert target.report["raw_parent_branch_delta_projection_used"] is False


def test_joint_target_matches_r12_combined_writer_semantics() -> None:
    plan = _plan()
    target, parent_post_layer, actual_pre, _, fixed_other = _joint_target(plan)
    expected_endpoint = plan.post_layer_chart.chart.encode_shaped(parent_post_layer)
    expected_group = expected_endpoint - actual_pre - fixed_other

    torch.testing.assert_close(target.desired_endpoint, expected_endpoint)
    torch.testing.assert_close(target.fixed_other_write, fixed_other)
    torch.testing.assert_close(target.target_combined_writer_output, expected_group)
    assert target.verify_authentication()


def test_joint_r12_plan_does_not_require_invented_post_operator_chart() -> None:
    plan = _plan(include_post_operator=False)
    target, *_ = _joint_target(plan)

    assert plan.report["available_target_modes"] == ["joint"]
    assert plan.post_operator_chart is None
    assert target.verify_authentication()


def test_final_layer_post_layer_is_exact_terminal_chart() -> None:
    plan = _plan(layer_index=13)

    assert plan.report["is_final_layer"] is True
    assert plan.post_layer_chart is plan.terminal_chart
    assert plan.report["chart_ids"]["post_layer"] == "stack.terminal.residual"
    assert plan.verify_authentication()


def test_locked_input_and_terminal_projection_are_chart_bound() -> None:
    plan = _plan()
    parent = torch.tensor([[0.2, -0.8, 0.1], [1.0, 0.5, -0.4]], dtype=torch.float64)

    torch.testing.assert_close(
        project_lfm2_boundary_state(plan, boundary="pre", parent_state=parent),
        parent @ _base_encoder(),
    )
    torch.testing.assert_close(
        project_lfm2_boundary_state(plan, boundary="terminal", parent_state=parent),
        parent @ _base_encoder(),
    )


def test_layer_relabel_and_wrong_operator_kind_are_rejected() -> None:
    valid = _plan(layer_index=5)
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="operator_kind differs|boundary ids",
    ):
        replace(
            valid,
            layer_index=10,
            operator_kind="full_attention",
            report={},
        )


def test_raw_global_map_and_locked_intermediate_are_rejected() -> None:
    valid = _plan()

    class RawGlobalMap:
        source_width = 3
        target_width = 2

    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="raw global coordinate maps are forbidden",
    ):
        replace(valid, post_operator_chart=RawGlobalMap(), report={})  # type: ignore[arg-type]

    parent = torch.randn(
        24, 3, generator=torch.Generator().manual_seed(17), dtype=torch.float64
    )
    locked_evidence, locked = _fit_chart_pair(
        "layer.000.residual.post_operator",
        parent,
        parent @ _base_encoder(),
        role="intermediate",
        locked=True,
    )
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="unlocked internal boundary",
    ):
        replace(
            valid,
            post_operator_chart=locked,
            post_operator_evidence=locked_evidence,
            report={},
        )


def test_mixed_recapture_and_base_encoder_provenance_are_rejected() -> None:
    valid = _plan()
    parent = torch.randn(
        24, 3, generator=torch.Generator().manual_seed(19), dtype=torch.float64
    )
    foreign_evidence, foreign_chart = _fit_chart_pair(
        "layer.000.residual.post_operator",
        parent,
        parent @ _base_encoder(),
        role="intermediate",
        locked=False,
        child_recapture_sha256=_sha("foreign-child-recapture"),
    )
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="mixes parent cache, child recapture, or row-alignment",
    ):
        replace(
            valid,
            post_operator_chart=foreign_chart,
            post_operator_evidence=foreign_evidence,
            report={},
        )

    other_evidence, other_chart = _fit_chart_pair(
        "layer.000.residual.post_operator",
        parent,
        parent @ _base_encoder(foreign=True),
        role="intermediate",
        locked=False,
        foreign_base=True,
    )
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="share one base encoder",
    ):
        replace(
            valid,
            post_operator_chart=other_chart,
            post_operator_evidence=other_evidence,
            report={},
        )


def test_tampered_state_receipt_is_rejected() -> None:
    plan = _plan()
    parent, pre, post, existing, _ = _joint_inputs()
    receipt = dict(
        build_lfm2_moving_gauge_causal_state_receipt(
            plan,
            kind="baseline_current_child",
            tensors={
                "actual_child_post_layer": post,
                "actual_child_pre": pre,
                "existing_combined_writer_output": existing,
                "parent_post_layer": parent,
            },
        )
    )
    receipt["tensor_sha256s"]["actual_child_pre"] = _sha("fabricated")
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="seal or plan binding|does not bind",
    ):
        build_lfm2_joint_writer_target(
            plan,
            parent_post_layer=parent,
            actual_child_pre=pre,
            actual_child_post_layer=post,
            existing_combined_writer_output=existing,
            state_receipt=receipt,
        )


def test_fp32_and_bfloat16_inputs_are_canonicalized_to_fp64() -> None:
    plan = _plan()
    parent = torch.randn(4, 3, dtype=torch.float32)
    pre = torch.randn(4, 2, dtype=torch.bfloat16)
    post = torch.randn(4, 2, dtype=torch.float32)
    existing = torch.randn(4, 2, dtype=torch.bfloat16)
    receipt = build_lfm2_moving_gauge_causal_state_receipt(
        plan,
        kind="baseline_current_child",
        tensors={
            "actual_child_post_layer": post,
            "actual_child_pre": pre,
            "existing_combined_writer_output": existing,
            "parent_post_layer": parent,
        },
    )
    target = build_lfm2_joint_writer_target(
        plan,
        parent_post_layer=parent,
        actual_child_pre=pre,
        actual_child_post_layer=post,
        existing_combined_writer_output=existing,
        state_receipt=receipt,
    )

    assert target.parent_post_layer.dtype == torch.float64
    assert target.actual_child_pre.dtype == torch.float64
    assert target.target_combined_writer_output.dtype == torch.float64
    assert target.verify_authentication()


def test_conditioned_receipt_requires_fresh_capture_and_solution() -> None:
    plan = _plan()
    tensors = {
        "existing_mlp_writer_output": torch.zeros(2, 2),
        "fresh_child_post_layer": torch.zeros(2, 2),
        "fresh_child_post_operator": torch.zeros(2, 2),
        "parent_post_layer": torch.zeros(2, 3),
    }
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="operator_solution_sha256",
    ):
        build_lfm2_moving_gauge_causal_state_receipt(
            plan,
            kind="operator_conditioned_temporary_intervention",
            tensors=tensors,
            child_recapture_sha256=_sha("fresh"),
        )
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="fresh recapture",
    ):
        build_lfm2_moving_gauge_causal_state_receipt(
            plan,
            kind="operator_conditioned_temporary_intervention",
            tensors=tensors,
            child_recapture_sha256=_sha("child-recapture"),
            operator_solution_sha256=_sha("solution"),
        )


def test_self_resealed_target_overclaims_are_rejected() -> None:
    plan = _plan()
    joint, *_ = _joint_target(plan)
    forged_joint = replace(
        joint,
        report=_reseal(joint.report, solution_installed=True),
    )
    assert not forged_joint.verify_authentication()

    sequential, *_ = _sequential_target(plan)
    forged_sequential = replace(
        sequential,
        report=_reseal(
            sequential.report,
            full_post_layer_reclosure_verified=True,
        ),
    )
    assert not forged_sequential.verify_authentication()


def test_generic_report_and_receipt_verifiers_reject_resealed_overclaims() -> None:
    plan = _plan()
    joint, parent, pre, existing, fixed = _joint_target(plan)
    assert not verify_lfm2_moving_gauge_causal_target_report(
        _reseal(
            joint.report,
            unexpected_completion_claim=True,
        )
    )
    assert not verify_lfm2_moving_gauge_causal_target_report(
        _reseal(
            joint.report,
            solution_installed=True,
        )
    )
    assert not verify_lfm2_moving_gauge_causal_target_report(
        _reseal(
            joint.report,
            desired_total_formula="fabricated",
            plan_sha256="not-a-digest",
            tensor_sha256s={},
        )
    )

    forged_receipt = _reseal(
        joint.state_receipt,
        execution={"training_steps": 99, "model_mutations": 123},
        operator_solution_installed=True,
    )
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="seal or plan binding",
    ):
        build_lfm2_joint_writer_target(
            plan,
            parent_post_layer=parent,
            actual_child_pre=pre,
            actual_child_post_layer=pre + existing + fixed,
            existing_combined_writer_output=existing,
            state_receipt=forged_receipt,
        )


def test_receipt_authenticator_rejects_stale_plan_and_foreign_kind() -> None:
    plan = _plan()
    parent, pre, post, existing, _ = _joint_inputs()
    tensors = {
        "actual_child_post_layer": post,
        "actual_child_pre": pre,
        "existing_combined_writer_output": existing,
        "parent_post_layer": parent,
    }
    receipt = build_lfm2_moving_gauge_causal_state_receipt(
        plan,
        kind="baseline_current_child",
        tensors=tensors,
    )
    plan.layer_topology["layers"][0]["contract"]["kind"] = "full_attention"
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="authenticated causal plan",
    ):
        authenticate_lfm2_moving_gauge_causal_state_receipt(
            receipt,
            plan=plan,
            tensors=tensors,
            allowed_kinds={"baseline_current_child"},
        )

    valid_plan = _plan()
    foreign = _reseal(
        build_lfm2_moving_gauge_causal_state_receipt(
            valid_plan,
            kind="baseline_current_child",
            tensors=tensors,
        ),
        kind="invented_conditioned_kind",
        child_recapture_sha256=_sha("fresh-foreign"),
        operator_solution_sha256=_sha("foreign-solution"),
    )
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="closed receipt-kind inventory",
    ):
        authenticate_lfm2_moving_gauge_causal_state_receipt(
            foreign,
            plan=valid_plan,
            tensors=tensors,
            allowed_kinds={"invented_conditioned_kind"},
        )


def test_direct_target_construction_cannot_broadcast_misaligned_states() -> None:
    plan = _plan()
    parent = torch.tensor(
        [[0.5, -0.2, 0.4], [1.2, 0.7, -0.3]],
        dtype=torch.float64,
    )
    pre = torch.tensor([[0.1, 0.3]], dtype=torch.float64)
    existing = torch.tensor([[0.4, -0.6], [0.2, 0.1]], dtype=torch.float64)
    post = pre + existing + 0.03
    receipt = build_lfm2_moving_gauge_causal_state_receipt(
        plan,
        kind="baseline_current_child",
        tensors={
            "actual_child_post_layer": post,
            "actual_child_pre": pre,
            "existing_combined_writer_output": existing,
            "parent_post_layer": parent,
        },
    )
    endpoint = plan.post_layer_chart.chart.encode_shaped(parent)
    desired = endpoint - pre
    fixed = post - pre - existing
    output = desired - fixed
    tensors = {
        "parent_post_layer": parent,
        "actual_child_pre": pre,
        "actual_child_post_layer": post,
        "existing_combined_writer_output": existing,
        "desired_endpoint": endpoint,
        "desired_total_write": desired,
        "fixed_other_write": fixed,
        "target_combined_writer_output": output,
    }
    target = causal_module.Lfm2MovingGaugeJointWriterTarget(
        plan=plan,
        **tensors,
        state_receipt=receipt,
        report=causal_module._joint_writer_report(plan, receipt, tensors),
    )
    assert not target.verify_authentication()


def test_nested_topology_mutation_and_resealed_chart_ids_fail_closed() -> None:
    topology_plan = _plan()
    topology_plan.layer_topology["layers"][0]["contract"]["kind"] = "full_attention"
    assert not topology_plan.verify_authentication()

    chart_plan = _plan()
    forged_ids = dict(chart_plan.report["chart_ids"])
    forged_ids["post_layer"] = "layer.999.residual.post_layer"
    with pytest.raises(
        Lfm2MovingGaugeCausalTargetAuthenticationError,
        match="stale or malformed",
    ):
        replace(
            chart_plan,
            report=_reseal(chart_plan.report, chart_ids=forged_ids),
        )


def test_removed_postoperator_and_relabelled_recapture_fail_closed() -> None:
    plan = _plan()
    object.__setattr__(plan, "post_operator_chart", None)
    assert not plan.verify_authentication()

    sequential, *_ = _sequential_target(_plan())
    forged = replace(
        sequential,
        fresh_recapture_kind="baseline_current_child",
        operator_solution_sha256=None,
        report=_reseal(
            sequential.report,
            fresh_recapture_kind="baseline_current_child",
            operator_solution_sha256=None,
            evidence_stage="baseline_diagnostic",
        ),
    )
    assert not forged.verify_authentication()


def test_nonfinite_receipt_tensor_and_misaligned_states_fail_closed() -> None:
    plan = _plan()
    with pytest.raises(Lfm2MovingGaugeCausalTargetError, match="NaN or infinity"):
        build_lfm2_moving_gauge_causal_state_receipt(
            plan,
            kind="baseline_current_child",
            tensors={"bad": torch.full((2, 2), float("nan"))},
        )

    parent, pre, post, existing, _ = _joint_inputs()
    receipt = build_lfm2_moving_gauge_causal_state_receipt(
        plan,
        kind="baseline_current_child",
        tensors={
            "actual_child_post_layer": post,
            "actual_child_pre": pre,
            "existing_combined_writer_output": existing,
            "parent_post_layer": parent,
        },
    )
    with pytest.raises(Lfm2MovingGaugeCausalTargetError, match="aligned leading"):
        build_lfm2_joint_writer_target(
            plan,
            parent_post_layer=parent,
            actual_child_pre=torch.zeros(3, 2),
            actual_child_post_layer=post,
            existing_combined_writer_output=existing,
            state_receipt=receipt,
        )
