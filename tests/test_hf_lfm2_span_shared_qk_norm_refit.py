from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_span_shared_qk_norm_refit import (
    Lfm2SpanSharedQkNormRefitError,
    compile_lfm2_span_shared_qk_norm_refit,
    lfm2_span_shared_qk_norm_refit_tensor_sha256,
    verify_lfm2_span_shared_qk_norm_refit_receipt,
)


torch = pytest.importorskip("torch")


def _normalized(raw, norms, epsilon):
    direction = raw.to(torch.float64)
    direction = direction / torch.sqrt(
        direction.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return direction * norms.to(torch.float64).unsqueeze(0)


def _case(*, dtype=torch.bfloat16, diagnostics=False, bad_diagnostics=False):
    generator = torch.Generator().manual_seed(7601)
    rows = 5
    hidden = 7
    head_dim = 2
    query_heads = 4
    key_heads = 2
    epsilon = 1e-5
    x = torch.randn(rows, hidden, generator=generator, dtype=torch.float64) * 0.5
    query_prior = torch.randn(
        query_heads * head_dim,
        hidden,
        generator=generator,
        dtype=torch.float64,
    ).to(dtype)
    key_prior = torch.randn(
        key_heads * head_dim,
        hidden,
        generator=generator,
        dtype=torch.float64,
    ).to(dtype)
    query_norms = torch.tensor(
        [[1.5, 0.5], [0.75, 2.0], [2.5, 0.4], [0.6, 3.0]],
        dtype=dtype,
    )
    key_norms = torch.tensor([[1.25, 2.5], [2.0, 0.75]], dtype=dtype)
    query_raw = (x @ query_prior.to(torch.float64).T).reshape(
        rows, query_heads, head_dim
    )
    key_raw = (x @ key_prior.to(torch.float64).T).reshape(rows, key_heads, head_dim)
    query_targets = _normalized(query_raw, query_norms, epsilon)
    key_targets = _normalized(key_raw, key_norms, epsilon)
    value = torch.randn(
        key_heads * head_dim,
        hidden,
        generator=generator,
        dtype=torch.float64,
    ).to(dtype)
    out = torch.randn(
        hidden,
        query_heads * head_dim,
        generator=generator,
        dtype=torch.float64,
    ).to(dtype)
    arguments = {
        "transported_query_weight": query_prior,
        "transported_key_weight": key_prior,
        "unchanged_value_weight": value,
        "unchanged_out_weight": out,
        "selected_query_source_norm_rows": query_norms,
        "selected_key_source_norm_rows": key_norms,
        "cached_child_normalized_input_rows": x,
        "cached_query_normalized_pre_rope_rows": query_targets,
        "cached_key_normalized_pre_rope_rows": key_targets,
        "target_query_norm_weight_template": torch.ones(head_dim, dtype=dtype),
        "target_key_norm_weight_template": torch.ones(head_dim, dtype=dtype),
        "child_norm_epsilon": epsilon,
        "ridge": 1e-8,
        "selected_source_query_heads": ((0, 0), (0, 1), (1, 2), (1, 3)),
        "selected_source_key_value_groups": ((0, 0), (1, 1)),
        "parent_attention_compiler_result_sha256": "a" * 64,
    }
    if diagnostics:
        diagnostic_x = torch.randn(3, hidden, generator=generator, dtype=torch.float64)
        factor = 1e6 if bad_diagnostics else 1.0
        arguments.update(
            {
                "diagnostic_child_normalized_input_rows": diagnostic_x,
                "diagnostic_query_normalized_pre_rope_rows": factor
                * torch.randn(
                    3,
                    query_heads,
                    head_dim,
                    generator=generator,
                    dtype=torch.float64,
                ),
                "diagnostic_key_normalized_pre_rope_rows": factor
                * torch.randn(
                    3,
                    key_heads,
                    head_dim,
                    generator=generator,
                    dtype=torch.float64,
                ),
            }
        )
    return arguments


def _coordinate_dominant_bfloat16_case(seed: int):
    generator = torch.Generator().manual_seed(seed)
    rows = 24
    hidden = 32
    head_dim = 64
    query_heads = 8
    key_heads = 4
    epsilon = 1e-5
    x = torch.randn(rows, hidden, generator=generator).to(torch.bfloat16)
    query_prior = torch.randn(
        query_heads * head_dim,
        hidden,
        generator=generator,
    ).to(torch.bfloat16)
    key_prior = torch.randn(
        key_heads * head_dim,
        hidden,
        generator=generator,
    ).to(torch.bfloat16)
    dominant = (0.75 + 0.5 * torch.rand(head_dim, generator=generator)).to(
        torch.bfloat16
    )
    query_norms = torch.stack(
        [dominant]
        + [
            (dominant * (0.70 + 0.05 * index)).to(torch.bfloat16)
            for index in range(1, query_heads)
        ]
    )
    key_norms = torch.stack(
        [dominant]
        + [
            (dominant * (0.70 + 0.05 * index)).to(torch.bfloat16)
            for index in range(1, key_heads)
        ]
    )
    query_linear = (x @ query_prior.T).reshape(rows, query_heads, head_dim)
    key_linear = (x @ key_prior.T).reshape(rows, key_heads, head_dim)
    query_targets = _runtime_replay(
        x,
        query_prior,
        query_norms,
        heads=query_heads,
        epsilon=epsilon,
    )
    key_targets = _runtime_replay(
        x,
        key_prior,
        key_norms,
        heads=key_heads,
        epsilon=epsilon,
    )
    assert query_linear.shape == query_targets.shape
    assert key_linear.shape == key_targets.shape
    return {
        "transported_query_weight": query_prior,
        "transported_key_weight": key_prior,
        "unchanged_value_weight": torch.randn(
            key_heads * head_dim,
            hidden,
            generator=generator,
        ).to(torch.bfloat16),
        "unchanged_out_weight": torch.randn(
            hidden,
            query_heads * head_dim,
            generator=generator,
        ).to(torch.bfloat16),
        "selected_query_source_norm_rows": query_norms,
        "selected_key_source_norm_rows": key_norms,
        "cached_child_normalized_input_rows": x,
        "cached_query_normalized_pre_rope_rows": query_targets,
        "cached_key_normalized_pre_rope_rows": key_targets,
        "target_query_norm_weight_template": torch.ones(head_dim, dtype=torch.bfloat16),
        "target_key_norm_weight_template": torch.ones(head_dim, dtype=torch.bfloat16),
        "child_norm_epsilon": epsilon,
        "ridge": 1e-8,
        "selected_source_query_heads": tuple(
            (index // 2, index) for index in range(query_heads)
        ),
        "selected_source_key_value_groups": tuple(
            (index, index) for index in range(key_heads)
        ),
        "parent_attention_compiler_result_sha256": "b" * 64,
    }


def _runtime_replay(x, weight, norm, *, heads, epsilon):
    native_x = x.to(weight.dtype)
    linear = (native_x @ weight.T).reshape(int(x.shape[0]), heads, -1)
    direction = linear.to(torch.float32)
    direction = direction * torch.rsqrt(
        direction.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return norm.to(linear.dtype) * direction.to(linear.dtype)


def test_nonproportional_envelope_publishes_finite_native_dense_state() -> None:
    case = _case()
    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    repeated = compile_lfm2_span_shared_qk_norm_refit(**case)

    state = result.state_dict()
    assert set(state) == {
        "q_proj.weight",
        "k_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    }
    assert all(value.dtype == torch.bfloat16 for value in state.values())
    assert all(bool(torch.isfinite(value).all().item()) for value in state.values())
    expected_query = case["selected_query_source_norm_rows"].abs().amax(dim=0)
    expected_key = case["selected_key_source_norm_rows"].abs().amax(dim=0)
    assert torch.equal(result.query_norm_weight, expected_query)
    assert torch.equal(result.key_norm_weight, expected_key)
    assert not any(
        torch.equal(result.query_norm_weight, row)
        for row in case["selected_query_source_norm_rows"]
    )
    assert not torch.equal(
        result.query_norm_weight,
        case["selected_query_source_norm_rows"].mean(dim=0),
    )
    assert not any(
        torch.equal(result.key_norm_weight, row)
        for row in case["selected_key_source_norm_rows"]
    )
    assert not torch.equal(
        result.key_norm_weight,
        case["selected_key_source_norm_rows"].mean(dim=0),
    )
    repair = result.report["representability_repair"]
    assert repair["applied"] is False
    assert repair["common_scale_target_dtype_value"] == 1.0
    assert repair["common_scale_ulp_distance_above_one"] == 0
    assert result.report["result_sha256"] == repeated.report["result_sha256"]
    for name, value in result.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[name]), name
    assert result.verify_authentication() == result.report["result_sha256"]


def test_exact_equality_boundary_gets_one_common_native_ulp_of_safety() -> None:
    case = _case()
    case["selected_query_source_norm_rows"] = torch.ones(4, 2, dtype=torch.bfloat16)
    case["selected_key_source_norm_rows"] = torch.ones(2, 2, dtype=torch.bfloat16)
    case["cached_query_normalized_pre_rope_rows"] = torch.ones(
        5, 4, 2, dtype=torch.bfloat16
    )
    case["cached_key_normalized_pre_rope_rows"] = torch.full(
        (5, 2, 2), 0.5, dtype=torch.bfloat16
    )

    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    repair = result.report["representability_repair"]
    one = torch.ones((), dtype=torch.bfloat16)
    expected_scale = torch.nextafter(one, torch.full_like(one, float("inf")))
    expected_norm = torch.ones(2, dtype=torch.bfloat16) * expected_scale

    assert repair["applied"] is True
    assert repair["common_scale_target_dtype_value"] == float(expected_scale.item())
    assert repair["common_scale_ulp_distance_above_one"] == 1
    assert repair["worst_pre_repair_maximum_direction_mean_square"] == 1.0
    assert repair["previous_scale_worst_direction_mean_square"] == 1.0
    assert repair["one_ulp_safety_above_equality_boundary"] is True
    assert repair["worst_post_repair_maximum_direction_mean_square"] < 1.0
    assert torch.equal(result.query_norm_weight, expected_norm)
    assert torch.equal(result.key_norm_weight, expected_norm)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bfloat16_coordinate_dominance_uses_deterministic_minimal_repair(
    seed: int,
) -> None:
    case = _coordinate_dominant_bfloat16_case(seed)
    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    repeated = compile_lfm2_span_shared_qk_norm_refit(**case)
    repair = result.report["representability_repair"]
    query_envelope = case["selected_query_source_norm_rows"].abs().amax(dim=0)
    key_envelope = case["selected_key_source_norm_rows"].abs().amax(dim=0)
    native_scale = torch.as_tensor(
        repair["common_scale_target_dtype_value"], dtype=torch.bfloat16
    )

    assert repair["worst_pre_repair_maximum_direction_mean_square"] >= 1.0
    assert repair["applied"] is True
    assert repair["common_scale_ulp_distance_above_one"] == 1
    assert repair["worst_post_repair_maximum_direction_mean_square"] < 1.0
    assert repair["previous_scale_worst_direction_mean_square"] >= 1.0
    assert repair["representability_repair_has_quality_gate_authority"] is False
    assert torch.equal(result.query_norm_weight, query_envelope * native_scale)
    assert torch.equal(result.key_norm_weight, key_envelope * native_scale)
    assert result.report["result_sha256"] == repeated.report["result_sha256"]
    for name, value in result.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[name]), name


def test_bfloat16_runtime_replay_and_unchanged_value_output_are_authenticated() -> None:
    case = _case()
    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    query = _runtime_replay(
        case["cached_child_normalized_input_rows"],
        result.query_weight,
        result.query_norm_weight,
        heads=4,
        epsilon=case["child_norm_epsilon"],
    )
    key = _runtime_replay(
        case["cached_child_normalized_input_rows"],
        result.key_weight,
        result.key_norm_weight,
        heads=2,
        epsilon=case["child_norm_epsilon"],
    )
    assert torch.allclose(
        query.to(torch.float64),
        case["cached_query_normalized_pre_rope_rows"],
        rtol=4e-2,
        atol=4e-2,
    )
    assert torch.allclose(
        key.to(torch.float64),
        case["cached_key_normalized_pre_rope_rows"],
        rtol=4e-2,
        atol=4e-2,
    )
    unchanged = result.report["unchanged_state_sha256"]
    assert unchanged == {
        "out_proj.weight": lfm2_span_shared_qk_norm_refit_tensor_sha256(
            case["unchanged_out_weight"]
        ),
        "v_proj.weight": lfm2_span_shared_qk_norm_refit_tensor_sha256(
            case["unchanged_value_weight"]
        ),
    }


def test_float64_analytic_precast_rowspace_property_is_not_overclaimed_for_final() -> (
    None
):
    case = _case(dtype=torch.float64)
    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    x = case["cached_child_normalized_input_rows"]
    _u, _singular, vh = torch.linalg.svd(x, full_matrices=True)
    nullspace = vh[int(x.shape[0]) :, :]
    target_norm = result.query_norm_weight.repeat(4)
    source_norm = case["selected_query_source_norm_rows"].reshape(-1)
    compensated = case["transported_query_weight"] * (
        source_norm / target_norm
    ).reshape(-1, 1)
    scale = result.report["query_refit"]["nonnegative_prior_scale"]
    delta = result.query_weight - scale * compensated
    assert float((delta @ nullspace.T).abs().max().item()) < 1e-10
    assert (
        result.report["query_refit"][
            "analytic_precast_prior_nullspace_maximum_absolute_leakage_advisory"
        ]
        < 1e-10
    )
    assert (
        result.report["query_refit"][
            "analytic_precast_correction_constructed_in_cached_input_rowspace"
        ]
        is True
    )
    assert (
        result.report["query_refit"][
            "final_target_dtype_prior_nullspace_preservation_claimed"
        ]
        is False
    )


def test_bfloat16_final_nullspace_leakage_is_authenticated_advisory() -> None:
    case = _case()
    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    x = case["cached_child_normalized_input_rows"].to(torch.float64)
    _u, singular, vh = torch.linalg.svd(x, full_matrices=True)
    tolerance = (
        max(int(x.shape[0]), int(x.shape[1]))
        * torch.finfo(torch.float64).eps
        * float(singular.max().item())
    )
    nullspace = vh[int((singular > tolerance).sum().item()) :, :]
    target_norm = result.query_norm_weight.to(torch.float64).repeat(4)
    source_norm = case["selected_query_source_norm_rows"].to(torch.float64).reshape(-1)
    compensated = case["transported_query_weight"].to(torch.float64) * (
        source_norm / target_norm
    ).reshape(-1, 1)
    scale = result.report["query_refit"]["nonnegative_prior_scale"]
    final_delta = result.query_weight.to(torch.float64) - scale * compensated
    measured = float((final_delta @ nullspace.T).abs().max().item())
    query_report = result.report["query_refit"]

    assert query_report[
        "final_target_dtype_prior_nullspace_maximum_absolute_leakage_advisory"
    ] == pytest.approx(measured, rel=0.0, abs=1e-15)
    assert (
        query_report["final_target_dtype_prior_nullspace_preservation_claimed"] is False
    )
    assert query_report["nullspace_leakage_diagnostics_have_install_authority"] is False


def test_float64_source_envelope_rounds_up_minimally_into_bfloat16() -> None:
    case = _case()
    smallest = float(
        torch.nextafter(
            torch.zeros((), dtype=torch.bfloat16),
            torch.full((), float("inf"), dtype=torch.bfloat16),
        ).item()
    )
    query_norms = torch.tensor(
        [
            [1.003, smallest / 2.0],
            [0.9, smallest / 4.0],
            [0.8, smallest / 8.0],
            [0.7, smallest / 16.0],
        ],
        dtype=torch.float64,
    )
    key_norms = torch.tensor(
        [[1.003, smallest / 2.0], [0.75, smallest / 4.0]],
        dtype=torch.float64,
    )
    case["selected_query_source_norm_rows"] = query_norms
    case["selected_key_source_norm_rows"] = key_norms
    case["cached_query_normalized_pre_rope_rows"] = torch.zeros(
        5, 4, 2, dtype=torch.float64
    )
    case["cached_key_normalized_pre_rope_rows"] = torch.zeros(
        5, 2, 2, dtype=torch.float64
    )

    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    exact_query = query_norms.abs().amax(dim=0)
    installed = result.query_norm_weight
    previous = torch.nextafter(installed, torch.zeros_like(installed))
    envelope = result.report["query_refit"]["coordinatewise_envelope"]

    assert torch.all(installed.to(torch.float64) >= exact_query)
    assert torch.all(previous.to(torch.float64) < exact_query)
    assert float(installed[0].item()) == 1.0078125
    assert float(installed[1].item()) == smallest
    assert envelope["source_absolute_max_float64_sha256"] == (
        lfm2_span_shared_qk_norm_refit_tensor_sha256(exact_query)
    )
    assert envelope["positive_sub_tiny_source_coordinate_count"] == 1
    assert envelope["upward_nextafter_coordinate_count"] == 2
    assert envelope["minimal_per_coordinate_target_dtype_ceiling_verified"] is True
    assert envelope["rounded_envelope_is_not_below_float64_lower_bound"] is True


def test_zero_ridge_uses_deterministic_minimum_norm_pseudoinverse() -> None:
    case = _case()
    rank_deficient = case["cached_child_normalized_input_rows"].clone()
    rank_deficient[1] = rank_deficient[0]
    rank_deficient[3] = rank_deficient[2]
    case["cached_child_normalized_input_rows"] = rank_deficient
    case["ridge"] = 0.0

    result = compile_lfm2_span_shared_qk_norm_refit(**case)
    repeated = compile_lfm2_span_shared_qk_norm_refit(**case)

    assert result.report["rowspace_ridge"] == 0.0
    assert result.report["zero_ridge_uses_minimum_frobenius_norm_pseudoinverse"] is True
    for branch_name in ("query_refit", "key_refit"):
        branch = result.report[branch_name]
        for solver_name in (
            "analytic_precast_rowspace_solver",
            "fixed_native_residual_rowspace_solver",
        ):
            solver = branch[solver_name]
            assert solver["method"] == "zero-ridge-moore-penrose-minimum-frobenius-norm"
            assert solver["ridge"] == 0.0
            assert solver["minimum_frobenius_norm_solution"] is True
    assert result.report["result_sha256"] == repeated.report["result_sha256"]
    for name, value in result.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[name]), name


def test_diagnostics_cannot_change_installed_state_and_bad_metrics_do_not_block() -> (
    None
):
    plain = compile_lfm2_span_shared_qk_norm_refit(**_case())
    diagnostic = compile_lfm2_span_shared_qk_norm_refit(
        **_case(diagnostics=True, bad_diagnostics=True)
    )
    for name, value in plain.state_dict().items():
        assert torch.equal(value, diagnostic.state_dict()[name]), name
    assert (
        diagnostic.report["diagnostic_evidence"][
            "numeric_diagnostics_have_install_authority"
        ]
        is False
    )
    assert (
        diagnostic.report["diagnostic_evidence"]["query_relative_rmse_diagnostic"] > 0.9
    )


def test_receipt_tampering_and_outer_manifest_disagreement_fail_closed() -> None:
    result = compile_lfm2_span_shared_qk_norm_refit(**_case())
    corrupt = result.as_dict()
    corrupt["rowspace_ridge"] = 7.0
    with pytest.raises(Lfm2SpanSharedQkNormRefitError, match="self-hash"):
        replace(result, report=MappingProxyType(corrupt)).verify_authentication()

    resealed = result.as_dict()
    resealed["installed_attention_state_sha256"]["q_proj.weight"] = "0" * 64
    resealed["refit_state_sha256"]["q_proj.weight"] = "0" * 64
    resealed.pop("result_sha256")
    resealed["result_sha256"] = canonical_sha256(resealed)
    expected = dict(result.report["installed_attention_state_sha256"])
    with pytest.raises(Lfm2SpanSharedQkNormRefitError, match="outer installed"):
        verify_lfm2_span_shared_qk_norm_refit_receipt(
            resealed,
            installed_attention_state_sha256=expected,
        )

    resealed = result.as_dict()
    resealed["representability_repair"][
        "representability_repair_has_quality_gate_authority"
    ] = True
    resealed.pop("result_sha256")
    resealed["result_sha256"] = canonical_sha256(resealed)
    with pytest.raises(Lfm2SpanSharedQkNormRefitError, match="representability repair"):
        verify_lfm2_span_shared_qk_norm_refit_receipt(resealed)


def test_operation_audit_records_zero_prohibited_operations() -> None:
    result = compile_lfm2_span_shared_qk_norm_refit(**_case())
    audit = result.report["operation_audit"]
    assert audit == {
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "parameter_sharing_sites": 0,
        "parent_sidecars": 0,
    }
