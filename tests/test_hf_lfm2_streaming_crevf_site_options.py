from __future__ import annotations

import dataclasses
import hashlib
import inspect
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import neural_forge.hf_lfm2_streaming_crevf_site_options as stream_module
from neural_forge.hf_lfm2_cached_second_order_attention_solver import (
    seal_lfm2_attention_bridge,
    seal_lfm2_cached_second_order_evidence,
)
from neural_forge.hf_lfm2_causal_residual_entropy_value_fold import (
    decode_crevf_tensor,
)
from neural_forge.hf_lfm2_crevf_budget import plan_crevf_sites
from neural_forge.hf_lfm2_streaming_crevf_site_options import (
    Lfm2StreamingCrevfConfig,
    Lfm2StreamingCrevfError,
    Lfm2StreamingJointAttentionRequest,
    Lfm2StreamingTensorSiteRequest,
    build_lfm2_streaming_crevf_site,
    build_lfm2_streaming_joint_attention_site,
)


def _linear_site(*, seed: int = 71):
    generator = torch.Generator().manual_seed(seed)
    samples, input_features, output_features = 64, 32, 8
    weight = (
        torch.randn(
            output_features,
            input_features,
            generator=generator,
        )
        * 0.12
    ).to(torch.bfloat16)
    child_inputs = torch.randn(
        samples,
        input_features,
        generator=generator,
    )
    parent_inputs = child_inputs + 0.2 * torch.randn(
        samples,
        input_features,
        generator=generator,
    )
    parent_weight = weight.float() + 0.04 * torch.randn(
        output_features,
        input_features,
        generator=generator,
    )
    child_outputs = child_inputs @ weight.float().T
    parent_outputs = parent_inputs @ parent_weight.T
    evidence = seal_lfm2_cached_second_order_evidence(
        site_id="model.layers.0.feed_forward.w2",
        tensor_role="ffn_w2",
        parent_inputs=parent_inputs.clone(),
        child_inputs=child_inputs.clone(),
        parent_outputs=parent_outputs.clone(),
        child_outputs=child_outputs.clone(),
    )
    return weight, evidence


def _request(
    weight,
    evidence,
    *,
    name: str = "model.layers.0.feed_forward.w2.weight",
    role: str = "ffn_w2",
    semantic_kind: str = "other",
    importance_ppm: int = 1_000_000,
):
    return Lfm2StreamingTensorSiteRequest(
        name=name,
        role=role,
        weight=weight,
        evidence=evidence,
        semantic_kind=semantic_kind,
        importance_ppm=importance_ppm,
    )


def _config(**overrides):
    values = {
        "bit_widths": (4,),
        "group_sizes": (32,),
        "clipping_ratios": (1.0,),
        "rounding_offsets": (0.0,),
        "outliers_per_group": (0,),
        "mismatch_coefficients": (0.0,),
        "propagation_coefficients": (1.0,),
        "top_k_per_physical_band": 2,
        "physical_byte_band_width": 10_000,
        "max_grid_points": 64,
        "row_block_size": 4,
    }
    values.update(overrides)
    return Lfm2StreamingCrevfConfig(**values)


def _assert_contains_no_tensor(
    value: Any,
    seen: set[int] | None = None,
) -> None:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    assert not isinstance(value, (torch.Tensor, nn.Parameter))
    if dataclasses.is_dataclass(value):
        for item in dataclasses.fields(value):
            _assert_contains_no_tensor(getattr(value, item.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_contains_no_tensor(key, seen)
            _assert_contains_no_tensor(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_contains_no_tensor(item, seen)


def _option_rank(option):
    return (
        option.loss_units,
        option.encoded_record_bytes,
        option.persistent_encoded_bytes,
        option.option_id,
        option.bundle_sha256,
    )


def test_streaming_top_k_matches_full_frontier_in_each_physical_band() -> None:
    weight, evidence = _linear_site()
    original = weight.clone()
    full_config = _config(
        bit_widths=(4, 8),
        clipping_ratios=(0.75, 1.0),
        top_k_per_physical_band=10,
        physical_byte_band_width=100,
    )
    pruned_config = _config(
        bit_widths=(4, 8),
        clipping_ratios=(0.75, 1.0),
        top_k_per_physical_band=1,
        physical_byte_band_width=100,
    )
    full = build_lfm2_streaming_crevf_site(
        _request(weight, evidence),
        config=full_config,
    )
    pruned = build_lfm2_streaming_crevf_site(
        _request(weight, evidence),
        config=pruned_config,
    )
    expected = {}
    for option in full.site.options:
        band = (option.encoded_record_bytes - 1) // 100
        current = expected.get(band)
        if current is None or _option_rank(option) < _option_rank(current):
            expected[band] = option
    observed = {
        (option.encoded_record_bytes - 1) // 100: option
        for option in pruned.site.options
    }
    assert {index: option.bundle_sha256 for index, option in observed.items()} == {
        index: option.bundle_sha256 for index, option in expected.items()
    }
    assert pruned.ops_audit.solver_invocations == 4
    assert pruned.ops_audit.generated_encoded_options == 4
    assert pruned.ops_audit.retained_encoded_options == len(expected)
    assert pruned.ops_audit.discarded_encoded_options == 4 - len(expected)
    assert pruned.ops_audit.returned_dense_tensor_count == 0
    assert all(len(report.retained_option_ids) <= 1 for report in pruned.physical_bands)
    assert torch.equal(weight, original)
    _assert_contains_no_tensor(pruned)

    for option in pruned.site.options:
        restored = decode_crevf_tensor(option.tensor_blob).dequantize()
        assert restored.shape == weight.shape
        assert bool(torch.isfinite(restored).all())
        assert option.encoded_record_bytes > len(option.tensor_blob)
    plan = plan_crevf_sites(
        [pruned.site],
        dense_reference_bytes=100_000,
        target_reduction=0.50,
    )
    assert plan.build_artifact()


def test_streaming_build_and_integer_loss_are_deterministic() -> None:
    weight, evidence = _linear_site(seed=72)
    config = _config(objective_loss_scale=10**12)
    request = _request(
        weight,
        evidence,
        importance_ppm=1_750_000,
    )
    first = build_lfm2_streaming_crevf_site(request, config=config)
    second = build_lfm2_streaming_crevf_site(request, config=config)
    assert first.build_sha256 == second.build_sha256
    assert first.source_bundle_sha256 == second.source_bundle_sha256
    assert tuple(option.tensor_blob for option in first.site.options) == tuple(
        option.tensor_blob for option in second.site.options
    )
    assert first.retained_options == second.retained_options
    retained = first.retained_options[0]
    expected = int(
        (
            Decimal(retained.selected_objective)
            * Decimal(config.objective_loss_scale)
            * Decimal(request.importance_ppm)
            / Decimal(1_000_000)
        ).to_integral_value(rounding=ROUND_HALF_UP)
    )
    assert retained.loss_units == expected
    assert first.site.options[0].loss_units == expected


def test_qk_and_exact_norm_control_floors_are_applied_before_solving() -> None:
    weight, evidence = _linear_site(seed=73)
    config = _config(
        bit_widths=(4, 5, 6, 7, 8),
        top_k_per_physical_band=10,
    )
    query = build_lfm2_streaming_crevf_site(
        _request(
            weight,
            evidence,
            name="model.layers.0.self_attn.q_proj.weight",
            role="attention.q",
            semantic_kind="attention_query",
        ),
        config=config,
    )
    assert query.ops_audit.solver_invocations == 2
    assert query.site.options
    assert all(option.minimum_bit_width >= 7 for option in query.site.options)
    assert query.site.eligible_options == tuple(
        sorted(query.site.options, key=lambda option: option.option_id)
    )

    normalization = build_lfm2_streaming_crevf_site(
        _request(
            weight,
            evidence,
            name="model.layers.0.operator_norm.weight",
            role="rms_norm",
            semantic_kind="normalization",
        ),
        config=config,
    )
    control = build_lfm2_streaming_crevf_site(
        _request(
            weight,
            evidence,
            name="model.layers.0.opaque_control.weight",
            role="control_anchor",
            semantic_kind="control_anchor",
        ),
        config=config,
    )
    for exact in (normalization, control):
        assert exact.ops_audit.solver_invocations == 0
        assert exact.ops_audit.generated_encoded_options == 1
        assert len(exact.site.options) == 1
        option = exact.site.options[0]
        assert option.exact_bf16
        assert option.minimum_bit_width == 16
        assert torch.equal(
            decode_crevf_tensor(option.tensor_blob).dequantize(),
            weight,
        )

    with pytest.raises(Lfm2StreamingCrevfError, match="understates"):
        build_lfm2_streaming_crevf_site(
            _request(
                weight,
                evidence,
                name="model.layers.0.self_attn.k_proj.weight",
                role="ffn_w2",
            ),
            config=config,
        )


def _joint_site():
    generator = torch.Generator().manual_seed(81)
    batch_size, tokens = 12, 2
    query_heads, key_value_heads, head_dim = 16, 8, 2
    hidden = query_heads * head_dim
    value_hidden = key_value_heads * head_dim
    output_features = 8
    value_weight = (
        torch.randn(
            value_hidden,
            hidden,
            generator=generator,
        )
        * 0.08
    ).to(torch.bfloat16)
    output_weight = (
        torch.randn(
            output_features,
            hidden,
            generator=generator,
        )
        * 0.07
    ).to(torch.bfloat16)
    child_inputs = torch.randn(
        batch_size * tokens,
        hidden,
        generator=generator,
    )
    parent_inputs = child_inputs + 0.1 * torch.randn(
        batch_size * tokens,
        hidden,
        generator=generator,
    )
    parent_value_weight = value_weight.float() + 0.025 * torch.randn(
        value_hidden,
        hidden,
        generator=generator,
    )
    parent_output_weight = output_weight.float() + 0.02 * torch.randn(
        output_features,
        hidden,
        generator=generator,
    )
    logits = torch.randn(
        batch_size,
        query_heads,
        tokens,
        tokens,
        generator=generator,
    )
    logits[..., 0, 1] += 0.8
    logits[..., 1, 0] += 0.6
    probabilities = torch.softmax(logits, dim=-1)

    def context(values):
        value_heads = values.reshape(
            batch_size,
            tokens,
            key_value_heads,
            head_dim,
        ).permute(0, 2, 1, 3)
        repeated = value_heads.repeat_interleave(
            query_heads // key_value_heads,
            dim=1,
        )
        mixed = torch.einsum(
            "bhqk,bhkd->bhqd",
            probabilities,
            repeated,
        )
        return mixed.permute(0, 2, 1, 3).reshape(
            batch_size * tokens,
            hidden,
        )

    child_values = child_inputs @ value_weight.float().T
    parent_values = parent_inputs @ parent_value_weight.T
    child_context = context(child_values)
    parent_context = context(parent_values)
    child_outputs = child_context @ output_weight.float().T
    parent_outputs = parent_context @ parent_output_weight.T
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
    bridge = seal_lfm2_attention_bridge(
        site_id="model.layers.0.self_attn",
        child_attention_probabilities=probabilities.clone(),
        batch_size=batch_size,
        query_tokens=tokens,
        key_tokens=tokens,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        trace_attention_config_sha256=hashlib.sha256(
            b"stream-test-qh16-kvh8"
        ).hexdigest(),
        attention_mask=torch.ones(
            batch_size,
            1,
            tokens,
            tokens,
            dtype=torch.bool,
        ),
    )
    return Lfm2StreamingJointAttentionRequest(
        value_name="model.layers.0.self_attn.v_proj.weight",
        output_name="model.layers.0.self_attn.o_proj.weight",
        value_weight=value_weight,
        output_weight=output_weight,
        value_evidence=value_evidence,
        output_evidence=output_evidence,
        attention_bridge=bridge,
    )


def test_joint_attention_is_retained_as_one_atomic_v_o_bundle() -> None:
    request = _joint_site()
    config = _config(
        bit_widths=(4, 5),
        top_k_per_physical_band=1,
    )
    result = build_lfm2_streaming_joint_attention_site(
        request,
        config=config,
    )
    assert result.ops_audit.solver_invocations == 2
    assert result.ops_audit.generated_encoded_options == 2
    assert result.ops_audit.retained_encoded_options == 1
    assert result.ops_audit.returned_dense_tensor_count == 0
    option = result.site.options[0]
    assert len(option.correction_tensor_blobs) == 1
    assert option.record_names == (
        request.output_name,
        request.value_name,
    )
    assert option.record_identity_floors_satisfied
    assert option in result.site.eligible_options
    value = decode_crevf_tensor(option.tensor_blob).dequantize()
    output = decode_crevf_tensor(option.correction_tensor_blobs[0]).dequantize()
    assert value.shape == request.value_weight.shape
    assert output.shape == request.output_weight.shape
    assert option.encoded_record_bytes == (
        result.retained_options[0].encoded_record_bytes
    )
    assert len(option.bundle_sha256) == 64
    _assert_contains_no_tensor(result)


def test_grid_guard_fails_before_any_solver_invocation(monkeypatch) -> None:
    weight, evidence = _linear_site(seed=74)
    called = 0

    def forbidden_solver(*_args, **_kwargs):
        nonlocal called
        called += 1
        raise AssertionError("solver should not be called")

    monkeypatch.setattr(
        stream_module,
        "prepare_lfm2_second_order_site",
        forbidden_solver,
    )
    with pytest.raises(Lfm2StreamingCrevfError, match="max_grid_points"):
        build_lfm2_streaming_crevf_site(
            _request(weight, evidence),
            config=_config(bit_widths=(4, 5), max_grid_points=1),
        )
    assert called == 0
    with pytest.raises(Lfm2StreamingCrevfError, match="bit_widths"):
        _config(bit_widths=(4, 9))
    with pytest.raises(Lfm2StreamingCrevfError, match="parts per million"):
        _config(clipping_ratios=(0.9000001,))
    with pytest.raises(Lfm2StreamingCrevfError, match="outlier count"):
        _config(outliers_per_group=(32,))


def test_streaming_boundary_rejects_parameters_and_has_no_execution_surface() -> None:
    weight, evidence = _linear_site(seed=75)
    with pytest.raises(Lfm2StreamingCrevfError, match="nn.Parameter"):
        build_lfm2_streaming_crevf_site(
            _request(nn.Parameter(weight), evidence),
            config=_config(),
        )
    source = Path(stream_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "from_pretrained(",
        ".generate(",
        ".backward(",
        "torch.optim",
        "optimizer.step(",
        "nn.Parameter(",
    ):
        assert forbidden not in source
    for function in (
        build_lfm2_streaming_crevf_site,
        build_lfm2_streaming_joint_attention_site,
    ):
        parameters = inspect.signature(function).parameters
        assert "model" not in parameters
        assert "parent" not in parameters
