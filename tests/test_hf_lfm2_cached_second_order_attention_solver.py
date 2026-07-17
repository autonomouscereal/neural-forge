from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import neural_forge.hf_lfm2_cached_second_order_attention_solver as solver_module
from neural_forge.hf_lfm2_cached_second_order_attention_solver import (
    Lfm2CachedSecondOrderError,
    compute_lfm2_second_order_statistics,
    seal_lfm2_attention_bridge,
    seal_lfm2_cached_second_order_evidence,
    solve_lfm2_crevf_second_order_grid,
    solve_lfm2_joint_attention_vo,
    verify_lfm2_attention_bridge,
    verify_lfm2_cached_second_order_evidence,
)
from neural_forge.hf_lfm2_causal_residual_entropy_value_fold import (
    CREVF_RECORD_ENVELOPE_BYTES,
    decode_crevf_tensor,
    inspect_crevf_tensor,
)


def _single_site(
    *,
    seed: int = 7,
    input_features: int = 32,
    output_features: int = 16,
    sample_count: int = 96,
    with_sensitivity: bool = False,
    weight_scale: float = 0.1,
    input_mismatch_scale: float = 0.15,
    parent_delta_scale: float = 0.03,
):
    generator = torch.Generator().manual_seed(seed)
    weight = (
        torch.randn(
            output_features,
            input_features,
            generator=generator,
        )
        * weight_scale
    ).to(torch.bfloat16)
    child_inputs = torch.randn(
        sample_count,
        input_features,
        generator=generator,
    )
    parent_inputs = child_inputs + input_mismatch_scale * torch.randn(
        sample_count,
        input_features,
        generator=generator,
    )
    parent_delta = parent_delta_scale * torch.randn(
        output_features,
        input_features,
        generator=generator,
    )
    child_outputs = child_inputs @ weight.float().T
    parent_outputs = parent_inputs @ (weight.float() + parent_delta).T
    sensitivity = (
        torch.randn(
            sample_count,
            output_features,
            generator=generator,
        )
        if with_sensitivity
        else None
    )
    evidence = seal_lfm2_cached_second_order_evidence(
        site_id="model.layers.0.self_attn.v_proj",
        tensor_role="attention.v",
        parent_inputs=parent_inputs.clone(),
        child_inputs=child_inputs.clone(),
        parent_outputs=parent_outputs.clone(),
        child_outputs=child_outputs.clone(),
        output_sensitivities=(None if sensitivity is None else sensitivity.clone()),
    )
    return weight, evidence


def _one_point_grid(weight, evidence, **overrides):
    arguments = {
        "name": "model.layers.0.self_attn.v_proj.weight",
        "role": "attention.v",
        "bit_widths": (4,),
        "group_sizes": (32,),
        "clipping_ratios": (1.0,),
        "rounding_offsets": (0.0,),
        "outliers_per_group": (0,),
        "mismatch_coefficients": (0.0,),
        "row_block_size": 8,
    }
    arguments.update(overrides)
    return solve_lfm2_crevf_second_order_grid(
        weight,
        evidence,
        **arguments,
    )


def _assert_no_parameter(value: Any, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    assert not isinstance(value, nn.Parameter)
    if isinstance(value, torch.Tensor):
        return
    if dataclasses.is_dataclass(value):
        for item in dataclasses.fields(value):
            _assert_no_parameter(getattr(value, item.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_no_parameter(key, seen)
            _assert_no_parameter(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_parameter(item, seen)


def test_sealed_evidence_builds_authenticated_damped_statistics() -> None:
    _weight, evidence = _single_site(with_sensitivity=True)
    verify_lfm2_cached_second_order_evidence(evidence)
    statistics = compute_lfm2_second_order_statistics(
        evidence,
        damping_ratio=1e-4,
        parent_input_blend=0.35,
        output_damping_ratio=2e-4,
    )
    assert evidence.parent_inputs.dtype == torch.float64
    assert evidence.parent_inputs.device.type == "cpu"
    assert evidence.parent_inputs.is_contiguous()
    assert len(evidence.evidence_sha256) == 64
    assert len(statistics.statistics_sha256) == 64
    assert statistics.h_out is not None
    assert torch.linalg.cholesky_ex(statistics.h_in).info.item() == 0
    assert torch.linalg.cholesky_ex(statistics.damped_child_hessian).info.item() == 0
    assert torch.linalg.cholesky_ex(statistics.h_out).info.item() == 0
    expected_mismatch = (
        evidence.child_inputs.T
        @ (evidence.parent_inputs - evidence.child_inputs)
        / evidence.sample_count
    )
    assert torch.equal(
        statistics.mismatch_cross_covariance,
        expected_mismatch,
    )


def test_successive_solver_strictly_improves_synthetic_cached_objective() -> None:
    weight, evidence = _single_site()
    original = weight.clone()
    result = _one_point_grid(weight, evidence)
    option = result.best_option
    assert option.full_precision_objective < result.current_objective
    assert option.raw_successive_objective < option.rtn_objective
    assert option.selected_method == "successive"
    assert option.selected_objective == option.raw_successive_objective
    assert option.selected_objective < result.current_objective
    assert option.improvement_vs_rtn > 0
    assert torch.equal(weight, original)
    assert option.ops_audit.cached_tensors_only
    assert option.ops_audit.source_rollback_verified
    assert option.ops_audit.parent_model_calls == 0
    assert option.ops_audit.child_model_calls == 0
    assert option.ops_audit.forward_calls == 0
    assert option.ops_audit.gradient_calls == 0
    assert option.ops_audit.optimizer_steps == 0
    _assert_no_parameter(result)


def test_emitted_codes_scales_outliers_and_reconstruction_are_real_crevf() -> None:
    weight, evidence = _single_site()
    result = _one_point_grid(
        weight,
        evidence,
        outliers_per_group=(1,),
    )
    payload = result.best_option.payload
    metadata = inspect_crevf_tensor(payload.tensor_blob)
    decoded = decode_crevf_tensor(payload.tensor_blob)
    assert tuple(decoded.parameters()) == ()
    assert torch.equal(
        decoded.dequantize(),
        payload.reconstructed_tensor,
    )
    assert payload.group_codes.dtype == torch.uint8
    assert payload.scales.dtype == torch.float16
    assert payload.group_codes.shape == (weight.shape[0], 32)
    assert len(payload.exact_outliers) == weight.shape[0]
    assert payload.tensor_blob_bytes == len(payload.tensor_blob)
    assert payload.physical_record_bytes == (
        len(payload.tensor_blob) + CREVF_RECORD_ENVELOPE_BYTES
    )
    assert payload.dense_bf16_bytes == weight.numel() * 2
    assert payload.persistent_encoded_bytes == metadata.persistent_encoded_bytes
    assert payload.decoded_full_state_bytes == metadata.decoded_full_state_bytes
    assert payload.decoded_code_transient_bytes == metadata.decoded_code_transient_bytes
    assert len(payload.source_sha256) == 64
    assert len(payload.reconstruction_sha256) == 64
    assert len(payload.tensor_blob_sha256) == 64
    assert len(payload.physical_payload_sha256) == 64


def test_grid_is_byte_and_objective_deterministic_and_exposes_mismatch_axis() -> None:
    weight, evidence = _single_site()
    arguments = {
        "bit_widths": (4, 5),
        "clipping_ratios": (0.9, 1.0),
        "mismatch_coefficients": (0.0, 1.0),
    }
    first = _one_point_grid(weight, evidence, **arguments)
    second = _one_point_grid(weight, evidence, **arguments)
    assert first.grid_sha256 == second.grid_sha256
    assert first.best_option.option_id == second.best_option.option_id
    assert len(first.options) == 8
    assert tuple(item.option_id for item in first.options) == tuple(
        item.option_id for item in second.options
    )
    assert tuple(item.option_sha256 for item in first.options) == tuple(
        item.option_sha256 for item in second.options
    )
    assert tuple(item.payload.tensor_blob for item in first.options) == tuple(
        item.payload.tensor_blob for item in second.options
    )
    assert tuple(item.selected_objective for item in first.options) == tuple(
        item.selected_objective for item in second.options
    )
    assert {item.config.mismatch_coefficient_ppm for item in first.options} == {
        0,
        1_000_000,
    }
    mismatch_reconstructions = {
        item.payload.reconstruction_sha256
        for item in first.options
        if item.config.bit_width == 4 and item.config.clipping_ratio_ppm == 1_000_000
    }
    assert len(mismatch_reconstructions) == 2


def test_rtn_is_an_atomic_rollback_when_successive_rounding_is_worse() -> None:
    weight, evidence = _single_site(
        seed=13,
        output_features=8,
        sample_count=48,
        with_sensitivity=True,
        weight_scale=0.15,
        input_mismatch_scale=0.3,
        parent_delta_scale=0.08,
    )
    result = _one_point_grid(
        weight,
        evidence,
        clipping_ratios=(0.75,),
        rounding_offsets=(0.2,),
        mismatch_coefficients=(1.0,),
        row_block_size=4,
    )
    option = result.best_option
    assert option.raw_successive_objective > option.rtn_objective
    assert option.selected_method == "rtn_rollback"
    assert option.selected_objective == option.rtn_objective
    assert option.improvement_vs_rtn == 0


def test_off_diagonal_h_out_changes_row_compensation_but_not_true_rtn() -> None:
    weight, evidence = _single_site(
        seed=21,
        output_features=16,
        sample_count=96,
        with_sensitivity=True,
    )
    statistics = compute_lfm2_second_order_statistics(evidence)
    assert statistics.h_out is not None
    off_diagonal = statistics.h_out - torch.diag(torch.diag(statistics.h_out))
    assert float(off_diagonal.abs().amax()) > 0.1
    without_output_hessian = seal_lfm2_cached_second_order_evidence(
        site_id=evidence.site_id,
        tensor_role=evidence.tensor_role,
        parent_inputs=evidence.parent_inputs.clone(),
        child_inputs=evidence.child_inputs.clone(),
        parent_outputs=evidence.parent_outputs.clone(),
        child_outputs=evidence.child_outputs.clone(),
    )
    row_coupled = _one_point_grid(weight, evidence).best_option
    input_only = _one_point_grid(
        weight,
        without_output_hessian,
    ).best_option
    assert row_coupled.ops_audit.output_hessian_row_compensation
    assert not input_only.ops_audit.output_hessian_row_compensation
    assert (
        row_coupled.rtn_physical_payload_sha256
        == input_only.rtn_physical_payload_sha256
    )
    assert (
        row_coupled.raw_successive_physical_payload_sha256
        != input_only.raw_successive_physical_payload_sha256
    )
    assert row_coupled.selected_objective <= row_coupled.rtn_objective


def _joint_attention_site():
    generator = torch.Generator().manual_seed(11)
    batch_size = 48
    query_tokens = 2
    key_tokens = 2
    query_heads = 16
    key_value_heads = 8
    head_dim = 2
    input_features = query_heads * head_dim
    value_features = key_value_heads * head_dim
    context_features = query_heads * head_dim
    output_features = 16
    value_weight = (
        torch.randn(
            value_features,
            input_features,
            generator=generator,
        )
        * 0.09
    ).to(torch.bfloat16)
    output_weight = (
        torch.randn(
            output_features,
            context_features,
            generator=generator,
        )
        * 0.08
    ).to(torch.bfloat16)
    parent_inputs = torch.randn(
        batch_size * key_tokens,
        input_features,
        generator=generator,
    )
    child_inputs = parent_inputs + 0.12 * torch.randn(
        batch_size * key_tokens,
        input_features,
        generator=generator,
    )
    parent_value_weight = value_weight.float() + 0.025 * torch.randn(
        value_features,
        input_features,
        generator=generator,
    )
    parent_output_weight = output_weight.float() + 0.02 * torch.randn(
        output_features,
        context_features,
        generator=generator,
    )
    logits = 0.6 * torch.randn(
        batch_size,
        query_heads,
        query_tokens,
        key_tokens,
        generator=generator,
    )
    logits[..., 0, 1] += 0.7
    logits[..., 1, 0] += 0.5
    child_probabilities = torch.softmax(logits, dim=-1)
    attention_mask = torch.ones(
        batch_size,
        1,
        query_tokens,
        key_tokens,
        dtype=torch.bool,
    )

    def mix_values(values):
        heads = values.reshape(
            batch_size,
            key_tokens,
            key_value_heads,
            head_dim,
        ).permute(0, 2, 1, 3)
        repeated = heads.repeat_interleave(
            query_heads // key_value_heads,
            dim=1,
        )
        context_heads = torch.einsum(
            "bhqk,bhkd->bhqd",
            child_probabilities,
            repeated,
        )
        return context_heads.permute(0, 2, 1, 3).reshape(
            batch_size * query_tokens,
            context_features,
        )

    parent_values = parent_inputs @ parent_value_weight.T
    child_values = child_inputs @ value_weight.float().T
    parent_context = mix_values(parent_values)
    child_context = mix_values(child_values)
    parent_outputs = parent_context @ parent_output_weight.T
    child_outputs = child_context @ output_weight.float().T
    value_evidence = seal_lfm2_cached_second_order_evidence(
        site_id="model.layers.0.self_attn.v_proj",
        tensor_role="attention.v",
        parent_inputs=parent_inputs.clone(),
        child_inputs=child_inputs.clone(),
        parent_outputs=parent_values.clone(),
        child_outputs=child_values.clone(),
    )
    output_evidence = seal_lfm2_cached_second_order_evidence(
        site_id="model.layers.0.self_attn.o_proj",
        tensor_role="attention.o",
        parent_inputs=parent_context.clone(),
        child_inputs=child_context.clone(),
        parent_outputs=parent_outputs.clone(),
        child_outputs=child_outputs.clone(),
    )
    attention_bridge = seal_lfm2_attention_bridge(
        site_id="model.layers.0.self_attn",
        child_attention_probabilities=child_probabilities.clone(),
        batch_size=batch_size,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        trace_attention_config_sha256=hashlib.sha256(
            b"lfm2-qh16-kvh8-hd2-softmax-fp32"
        ).hexdigest(),
        attention_mask=attention_mask.clone(),
    )
    return (
        value_weight,
        output_weight,
        value_evidence,
        output_evidence,
        attention_bridge,
    )


def test_joint_attention_vo_correction_is_coupled_sealed_and_better_than_rtn() -> None:
    (
        value_weight,
        output_weight,
        value_evidence,
        output_evidence,
        attention_bridge,
    ) = _joint_attention_site()
    original_value = value_weight.clone()
    original_output = output_weight.clone()
    result = solve_lfm2_joint_attention_vo(
        value_weight,
        output_weight,
        value_evidence,
        output_evidence,
        attention_bridge=attention_bridge,
        value_name="model.layers.0.self_attn.v_proj.weight",
        output_name="model.layers.0.self_attn.o_proj.weight",
        bit_widths=(4,),
        group_sizes=(32,),
        clipping_ratios=(1.0,),
        rounding_offsets=(0.0,),
        outliers_per_group=(0,),
        mismatch_coefficients=(0.0,),
        propagation_coefficients=(1.0,),
        value_objective_weight=0.1,
        row_block_size=8,
    )
    best = result.best_option
    assert best.raw_successive_joint_objective < best.rtn_joint_objective
    assert best.selected_method == "successive"
    assert best.selected_joint_objective < result.current_joint_objective
    assert best.total_physical_record_bytes == (
        best.value_payload.physical_record_bytes
        + best.output_payload.physical_record_bytes
    )
    assert (
        best.total_dense_bf16_bytes
        == (value_weight.numel() + output_weight.numel()) * 2
    )
    assert len(best.evidence_bundle_sha256) == 64
    assert len(best.physical_bundle_sha256) == 64
    assert len(best.option_sha256) == 64
    assert attention_bridge.query_heads == 16
    assert attention_bridge.key_value_heads == 8
    assert attention_bridge.query_tokens == 2
    assert attention_bridge.key_tokens == 2
    assert value_evidence.output_features != output_evidence.input_features
    off_diagonal = attention_bridge.child_attention_probabilities[
        ...,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    ]
    assert bool(off_diagonal.gt(0.05).all())
    assert best.ops_audit.fixed_attention_probability_bridge
    assert not best.ops_audit.nonlinear_query_key_coupling
    assert torch.equal(
        decode_crevf_tensor(best.value_payload.tensor_blob).dequantize(),
        best.value_payload.reconstructed_tensor,
    )
    assert torch.equal(
        decode_crevf_tensor(best.output_payload.tensor_blob).dequantize(),
        best.output_payload.reconstructed_tensor,
    )
    assert torch.equal(value_weight, original_value)
    assert torch.equal(output_weight, original_output)
    _assert_no_parameter(result)


def test_identity_attention_shortcut_is_rejected_for_nonidentity_gqa_trace() -> None:
    (
        value_weight,
        output_weight,
        value_evidence,
        output_evidence,
        attention_bridge,
    ) = _joint_attention_site()
    verify_lfm2_attention_bridge(attention_bridge)
    identity_probabilities = (
        torch.eye(
            attention_bridge.query_tokens,
            attention_bridge.key_tokens,
        )
        .reshape(
            1,
            1,
            attention_bridge.query_tokens,
            attention_bridge.key_tokens,
        )
        .repeat(
            attention_bridge.batch_size,
            attention_bridge.query_heads,
            1,
            1,
        )
    )
    identity_bridge = seal_lfm2_attention_bridge(
        site_id=attention_bridge.site_id,
        child_attention_probabilities=identity_probabilities,
        batch_size=attention_bridge.batch_size,
        query_tokens=attention_bridge.query_tokens,
        key_tokens=attention_bridge.key_tokens,
        query_heads=attention_bridge.query_heads,
        key_value_heads=attention_bridge.key_value_heads,
        head_dim=attention_bridge.head_dim,
        trace_attention_config_sha256=(attention_bridge.trace_attention_config_sha256),
        attention_mask=attention_bridge.attention_mask.clone(),
    )
    with pytest.raises(
        Lfm2CachedSecondOrderError,
        match="do not reproduce",
    ):
        solve_lfm2_joint_attention_vo(
            value_weight,
            output_weight,
            value_evidence,
            output_evidence,
            attention_bridge=identity_bridge,
            value_name="model.layers.0.self_attn.v_proj.weight",
            output_name="model.layers.0.self_attn.o_proj.weight",
            bit_widths=(4,),
            group_sizes=(32,),
            clipping_ratios=(1.0,),
            rounding_offsets=(0.0,),
            outliers_per_group=(0,),
            mismatch_coefficients=(0.0,),
            propagation_coefficients=(1.0,),
        )


def test_evidence_and_geometry_fail_closed_without_mutating_source() -> None:
    generator = torch.Generator().manual_seed(99)
    aliased = torch.randn(12, 32, generator=generator)
    parent_y = torch.randn(12, 8, generator=generator)
    child_y = torch.randn(12, 8, generator=generator)
    with pytest.raises(Lfm2CachedSecondOrderError, match="alias"):
        seal_lfm2_cached_second_order_evidence(
            site_id="bad",
            tensor_role="attention.v",
            parent_inputs=aliased,
            child_inputs=aliased,
            parent_outputs=parent_y,
            child_outputs=child_y,
        )
    with pytest.raises(Lfm2CachedSecondOrderError, match="must be a Tensor"):
        seal_lfm2_cached_second_order_evidence(
            site_id="bad",
            tensor_role="attention.v",
            parent_inputs=None,
            child_inputs=aliased.clone(),
            parent_outputs=parent_y.clone(),
            child_outputs=child_y.clone(),
        )

    weight, evidence = _single_site()
    evidence.parent_inputs[0, 0] += 1
    with pytest.raises(Lfm2CachedSecondOrderError, match="authentication"):
        verify_lfm2_cached_second_order_evidence(evidence)

    wide_weight, wide_evidence = _single_site(input_features=48)
    original = wide_weight.clone()
    with pytest.raises(Lfm2CachedSecondOrderError, match="No requested"):
        _one_point_grid(
            wide_weight,
            wide_evidence,
            group_sizes=(32,),
        )
    assert torch.equal(wide_weight, original)

    parameter, clean_evidence = _single_site()
    with pytest.raises(Lfm2CachedSecondOrderError, match="nn.Parameter"):
        _one_point_grid(nn.Parameter(parameter), clean_evidence)


def test_module_has_no_model_training_or_generation_surface() -> None:
    source = Path(solver_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "from_pretrained(",
        ".generate(",
        ".backward(",
        "torch.optim",
        "optimizer.step(",
        "nn.Parameter(",
    ):
        assert forbidden not in source
    single_parameters = inspect.signature(solve_lfm2_crevf_second_order_grid).parameters
    joint_parameters = inspect.signature(solve_lfm2_joint_attention_vo).parameters
    assert "model" not in single_parameters
    assert "parent" not in single_parameters
    assert "model" not in joint_parameters
    assert "parent" not in joint_parameters
