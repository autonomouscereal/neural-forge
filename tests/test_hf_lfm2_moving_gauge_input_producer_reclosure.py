from __future__ import annotations

import ast
import hashlib
import inspect

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_moving_gauge_input_producer_reclosure as module  # noqa: E402
from neural_forge.hf_lfm2_moving_gauge_input_producer_reclosure import (  # noqa: E402
    Lfm2MovingGaugeInputProducerAuthenticationError,
    Lfm2MovingGaugeInputProducerReclosureError,
    Lfm2MovingGaugeInputProducerReclosurePolicy,
    Lfm2MovingGaugeInputProducerSiteEvidence,
    compile_lfm2_moving_gauge_input_producer_reclosure,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _site(
    *,
    parameter_id: str,
    group_id: str,
    role: str,
    seed: torch.Tensor,
    old: torch.Tensor,
    new: torch.Tensor,
    alignment: str = "aligned-rows",
    old_state: str = "old-child-state",
    new_state: str = "new-child-state",
) -> Lfm2MovingGaugeInputProducerSiteEvidence:
    return Lfm2MovingGaugeInputProducerSiteEvidence(
        parameter_id=parameter_id,
        producer_group_id=group_id,
        producer_role=role,
        seed_weight=seed,
        old_actual_normalized_input_rows=old,
        new_actual_normalized_input_rows=new,
        row_alignment_sha256=_sha(alignment),
        old_child_state_sha256=_sha(old_state),
        new_child_state_sha256=_sha(new_state),
    )


def test_grouped_fp32_qkv_reclosure_preserves_exactly_transformable_outputs() -> None:
    generator = torch.Generator().manual_seed(20260728)
    row_count = 31
    input_width = 5
    old = torch.randn(
        row_count,
        input_width,
        dtype=torch.float64,
        generator=generator,
    )
    transform = torch.tensor(
        [
            [1.25, 0.10, 0.00, 0.00, 0.00],
            [0.00, 0.80, 0.05, 0.00, 0.00],
            [0.00, 0.00, 1.10, 0.08, 0.00],
            [0.00, 0.00, 0.00, 0.70, 0.04],
            [0.03, 0.00, 0.00, 0.00, 1.35],
        ],
        dtype=torch.float64,
    )
    new = old @ transform
    sites = tuple(
        _site(
            parameter_id=f"layers.9.attention.{role}_proj.weight",
            group_id="layers.9.attention.input_norm",
            role=f"attention_{role}",
            seed=torch.randn(
                output_width,
                input_width,
                dtype=torch.float32,
                generator=generator,
            ),
            old=old,
            new=new,
        )
        for role, output_width in (("q", 7), ("k", 3), ("v", 4))
    )
    state = compile_lfm2_moving_gauge_input_producer_reclosure(
        tuple(reversed(sites)),
        policy=Lfm2MovingGaugeInputProducerReclosurePolicy(
            ridge_ratio=0.0,
            absolute_ridge_floor=1e-16,
        ),
    )

    assert state.verify_authentication()
    assert tuple(state.replacements_by_parameter_id) == tuple(
        sorted(site.parameter_id for site in sites)
    )
    assert state.report["changed_parameter_count"] == 3
    assert state.report["changed_coordinate_count"] > 0
    assert state.report["fold_only_guard"]["clean"] is True
    assert state.report["automatic_quality_accept_reject"] is False
    assert state.report["manual_whole_response_review_required"] is True
    for site in state.evidence:
        replacement = state.replacements_by_parameter_id[site.parameter_id]
        target = (
            site.old_actual_normalized_input_rows.to(torch.float64)
            @ site.seed_weight.to(torch.float64).T
        )
        realized = (
            site.new_actual_normalized_input_rows.to(torch.float64)
            @ replacement.to(torch.float64).T
        )
        assert replacement.dtype is torch.float32
        assert replacement.shape == site.seed_weight.shape
        assert not torch.equal(replacement, site.seed_weight)
        assert torch.allclose(realized, target, rtol=2e-6, atol=2e-6)
        receipt = next(
            value.report
            for value in state.replacements
            if value.parameter_id == site.parameter_id
        )
        assert receipt["producer_family"] == "attention"
        assert receipt["target_dtype_effectful"] is True
        assert receipt["group_solve_receipt"]["solve_dtype"] == "float64"
        assert (
            receipt["group_solve_receipt"]["closed_form_cholesky_solve_performed"]
            is True
        )
        assert receipt["arbitrary_residual_map_commuted_through_rmsnorm"] is False
        assert receipt["post_reclosure_relative_rmse"] < 2e-6
        assert receipt["all_numeric_metrics_diagnostic_only"] is True


def test_bfloat16_conv_and_grouped_mlp_publish_effectful_dense_weights() -> None:
    generator = torch.Generator().manual_seed(17)
    old_conv = torch.randn(23, 6, generator=generator)
    new_conv = old_conv * torch.tensor([0.55, 1.70, 0.80, 1.25, 0.65, 1.45])
    old_mlp = torch.randn(23, 6, generator=generator)
    new_mlp = old_mlp * torch.tensor([1.60, 0.70, 1.30, 0.60, 1.80, 0.75])
    sites = (
        _site(
            parameter_id="layers.4.short_conv.in_proj.weight",
            group_id="layers.4.short_conv.input_norm",
            role="short_conv_in_proj",
            seed=torch.randn(11, 6, generator=generator).to(torch.bfloat16),
            old=old_conv,
            new=new_conv,
            alignment="conv-alignment",
            old_state="conv-old-state",
            new_state="conv-new-state",
        ),
        _site(
            parameter_id="layers.4.mlp.w1.weight",
            group_id="layers.4.mlp.input_norm",
            role="mlp_w1",
            seed=torch.randn(13, 6, generator=generator).to(torch.bfloat16),
            old=old_mlp,
            new=new_mlp,
            alignment="mlp-alignment",
            old_state="mlp-old-state",
            new_state="mlp-new-state",
        ),
        _site(
            parameter_id="layers.4.mlp.w3.weight",
            group_id="layers.4.mlp.input_norm",
            role="mlp_w3",
            seed=torch.randn(9, 6, generator=generator).to(torch.bfloat16),
            old=old_mlp,
            new=new_mlp,
            alignment="mlp-alignment",
            old_state="mlp-old-state",
            new_state="mlp-new-state",
        ),
    )
    state = compile_lfm2_moving_gauge_input_producer_reclosure(sites)

    assert state.verify_authentication()
    assert state.report["changed_parameter_count"] == 3
    assert state.report["execution"]["parameters_quantized"] == 0
    assert state.report["execution"]["parameters_pruned"] == 0
    assert state.report["execution"]["training_steps"] == 0
    for replacement in state.replacements:
        assert replacement.replacement_weight.dtype is torch.bfloat16
        assert replacement.report["target_dtype_effectful"] is True
        assert replacement.report["ordinary_bfloat16_cast_is_quantization"] is False
        assert (
            replacement.report["post_reclosure_rmse"]
            < (replacement.report["pre_reclosure_rmse"])
        )


def test_exact_normalized_input_noop_is_byte_exact_and_not_rejected() -> None:
    generator = torch.Generator().manual_seed(91)
    rows = torch.randn(7, 4, generator=generator)
    site = _site(
        parameter_id="layers.0.attention.q_proj.weight",
        group_id="layers.0.attention.input_norm",
        role="attention_q",
        seed=torch.randn(6, 4, generator=generator),
        old=rows,
        new=rows.clone(),
        old_state="unchanged-state",
        new_state="unchanged-state",
    )
    state = compile_lfm2_moving_gauge_input_producer_reclosure((site,))
    replacement = state.replacements[0]

    assert torch.equal(replacement.replacement_weight, state.evidence[0].seed_weight)
    assert replacement.report["exact_normalized_input_noop"] is True
    assert replacement.report["changed_coordinate_count"] == 0
    assert replacement.report["target_dtype_effectful"] is False
    assert replacement.report["post_reclosure_rmse"] == 0.0
    assert state.report["changed_parameter_count"] == 0
    assert state.report["exact_noop_parameter_count"] == 1
    assert state.verify_authentication()


def test_evidence_and_replacement_tampering_fail_authentication() -> None:
    generator = torch.Generator().manual_seed(3)
    old = torch.randn(9, 3, generator=generator)
    new = old * torch.tensor([1.5, 0.6, 1.2])
    site = _site(
        parameter_id="layers.2.mlp.w1.weight",
        group_id="layers.2.mlp.input_norm",
        role="mlp_w1",
        seed=torch.randn(8, 3, generator=generator),
        old=old,
        new=new,
    )
    state = compile_lfm2_moving_gauge_input_producer_reclosure((site,))
    state.replacements[0].replacement_weight[0, 0].add_(1.0)
    assert state.replacements[0].verify_authentication() is False
    assert state.verify_authentication() is False

    fresh = _site(
        parameter_id="layers.2.mlp.w3.weight",
        group_id="layers.2.mlp.input_norm.second",
        role="mlp_w3",
        seed=torch.randn(8, 3, generator=generator),
        old=old,
        new=new,
    )
    fresh.new_actual_normalized_input_rows[0, 0].add_(0.25)
    assert fresh.verify_authentication() is False
    with pytest.raises(
        Lfm2MovingGaugeInputProducerAuthenticationError,
        match="site evidence failed authentication",
    ):
        compile_lfm2_moving_gauge_input_producer_reclosure((fresh,))


def test_shape_hash_and_group_contracts_fail_closed() -> None:
    generator = torch.Generator().manual_seed(6)
    old = torch.randn(8, 4, generator=generator)
    with pytest.raises(
        Lfm2MovingGaugeInputProducerReclosureError,
        match="identical shape",
    ):
        _site(
            parameter_id="bad.shape",
            group_id="bad.group",
            role="attention_q",
            seed=torch.randn(5, 4, generator=generator),
            old=old,
            new=torch.randn(8, 5, generator=generator),
        )
    with pytest.raises(
        Lfm2MovingGaugeInputProducerAuthenticationError,
        match="distinct old/new child-state hashes",
    ):
        _site(
            parameter_id="bad.state.hash",
            group_id="bad.state.group",
            role="attention_q",
            seed=torch.randn(5, 4, generator=generator),
            old=old,
            new=old * 1.1,
            old_state="same-state",
            new_state="same-state",
        )

    first = _site(
        parameter_id="layers.7.attention.q_proj.weight",
        group_id="layers.7.attention.input_norm",
        role="attention_q",
        seed=torch.randn(5, 4, generator=generator),
        old=old,
        new=old * 1.2,
        alignment="first-alignment",
    )
    second = _site(
        parameter_id="layers.7.attention.k_proj.weight",
        group_id="layers.7.attention.input_norm",
        role="attention_k",
        seed=torch.randn(2, 4, generator=generator),
        old=old,
        new=old * 1.2,
        alignment="different-alignment",
    )
    with pytest.raises(
        Lfm2MovingGaugeInputProducerReclosureError,
        match="does not share one authenticated",
    ):
        compile_lfm2_moving_gauge_input_producer_reclosure((first, second))

    mlp = _site(
        parameter_id="layers.7.mlp.w1.weight",
        group_id="layers.7.attention.input_norm",
        role="mlp_w1",
        seed=torch.randn(9, 4, generator=generator),
        old=old,
        new=old * 1.2,
        alignment="first-alignment",
    )
    with pytest.raises(
        Lfm2MovingGaugeInputProducerReclosureError,
        match="mixes semantic families",
    ):
        compile_lfm2_moving_gauge_input_producer_reclosure((first, mlp))


def test_public_api_has_no_model_execution_or_training_surface() -> None:
    signature = inspect.signature(compile_lfm2_moving_gauge_input_producer_reclosure)
    assert tuple(signature.parameters) == ("site_evidence", "policy")
    tree = ast.parse(inspect.getsource(module))
    called_attribute_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_attribute_names.isdisjoint(
        {
            "backward",
            "grad",
            "step",
            "prune",
            "quantize",
            "forward",
            "generate",
        }
    )

    rows = torch.eye(3)
    state = compile_lfm2_moving_gauge_input_producer_reclosure(
        (
            _site(
                parameter_id="audit.short_conv.in_proj.weight",
                group_id="audit.short_conv.input_norm",
                role="short_conv_in_proj",
                seed=torch.eye(3),
                old=rows,
                new=rows.clone(),
                old_state="audit-state",
                new_state="audit-state",
            ),
        )
    )
    execution = state.report["execution"]
    assert execution["parent_model_objects_accepted"] == 0
    assert execution["parent_model_forward_calls"] == 0
    assert execution["target_model_forward_calls"] == 0
    assert execution["backward_calls"] == 0
    assert execution["autograd_grad_calls"] == 0
    assert execution["optimizer_constructions"] == 0
    assert execution["optimizer_steps"] == 0
    assert execution["parameters_pruned"] == 0
    assert execution["parameters_quantized"] == 0
    assert execution["guard_clean"] is True
