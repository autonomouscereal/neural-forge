from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_gated_mlp_weight_signature import (
    DOWN_OUTPUT_ROLE,
    GATED_MLP_WEIGHT_SIGNATURE_ROLES,
    GATE_INPUT_ROLE,
    UP_INPUT_ROLE,
    GatedMlpWeightSignatureAuthenticationError,
    GatedMlpWeightSignatureFiniteError,
    GatedMlpWeightSignatureInvariantError,
    GatedMlpWeightSignatureShapeError,
    build_gated_mlp_weight_complete_signature,
    gated_mlp_weight_signature_tensor_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


def _case(
    *,
    source_neurons: int = 7,
    source_hidden: int = 6,
    target_hidden: int = 4,
) -> dict[str, torch.Tensor]:
    basis = torch.eye(source_hidden, dtype=torch.float64)[:, :target_hidden]
    gate = (
        torch.arange(
            1,
            source_neurons * source_hidden + 1,
            dtype=torch.float64,
        ).reshape(source_neurons, source_hidden)
        / 17.0
    )
    up = torch.flip(gate, dims=(0, 1)).contiguous() * 0.7 + 0.25
    down = (
        torch.arange(
            1,
            source_hidden * source_neurons + 1,
            dtype=torch.float64,
        ).reshape(source_hidden, source_neurons)
        / 23.0
    )
    return {
        "gate_weight": gate,
        "up_weight": up,
        "down_weight": down,
        "residual_basis": basis,
    }


def _pins(values: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        "w1.weight": gated_mlp_weight_signature_tensor_sha256(values["gate_weight"]),
        "w3.weight": gated_mlp_weight_signature_tensor_sha256(values["up_weight"]),
        "w2.weight": gated_mlp_weight_signature_tensor_sha256(values["down_weight"]),
        "residual_basis": gated_mlp_weight_signature_tensor_sha256(
            values["residual_basis"]
        ),
    }


def _build(values: dict[str, torch.Tensor], **policy):
    return build_gated_mlp_weight_complete_signature(
        **values,
        expected_input_sha256=_pins(values),
        **policy,
    )


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.square().mean())


def test_signature_matches_all_three_weight_projections_and_rms_balances() -> None:
    values = _case()
    originals = {name: value.clone() for name, value in values.items()}
    values["gate_weight"].requires_grad_(True)
    values["up_weight"].requires_grad_(True)
    values["down_weight"].requires_grad_(True)
    pins = _pins(values)

    result = build_gated_mlp_weight_complete_signature(
        **values,
        expected_input_sha256=pins,
    )

    basis = values["residual_basis"]
    raw = {
        GATE_INPUT_ROLE: values["gate_weight"].detach() @ basis,
        UP_INPUT_ROLE: values["up_weight"].detach() @ basis,
        DOWN_OUTPUT_ROLE: (basis.T @ values["down_weight"].detach()).T,
    }
    expected = torch.cat(
        tuple(raw[role] / _rms(raw[role]) for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES),
        dim=1,
    )
    torch.testing.assert_close(result.signature, expected, rtol=0, atol=0)
    for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES:
        torch.testing.assert_close(
            _rms(result.role_block(role)),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=0,
            atol=1e-15,
        )
    assert result.signature.shape == (7, 12)
    assert result.feature_width == 12
    assert result.signature.requires_grad is False
    assert result.signature.grad_fn is None
    assert result.report["signature_orientation"] == "source-neuron-major"
    assert result.report["invariants"]["prompt_independent"] is True
    assert result.report["invariants"]["no_trace_rows_used"] is True
    assert result.report["trace_rows_used"] == 0
    assert result.report["operation_audit"]["training_steps"] == 0
    assert result.report["operation_audit"]["parent_model_forward_calls"] == 0
    assert result.verify()["accepted"] is True
    for name, original in originals.items():
        torch.testing.assert_close(values[name].detach(), original, rtol=0, atol=0)


def test_real_source_neuron_count_has_complete_three_role_coverage() -> None:
    values = _case(
        source_neurons=2560,
        source_hidden=8,
        target_hidden=4,
    )

    result = _build(values)

    coverage = result.report["coverage"]
    assert result.signature.shape == (2560, 12)
    assert coverage["source_neuron_count"] == 2560
    assert coverage["covered_source_neuron_count"] == 2560
    assert coverage["role_count"] == 3
    assert coverage["all_source_neurons_have_all_three_roles"] is True
    assert coverage["source_rows_sampled"] == 0
    assert coverage["source_rows_dropped"] == 0
    assert all(
        role["source_neuron_rows_addressed"] == 2560
        and role["source_neuron_coverage_fraction"] == 1.0
        for role in coverage["roles"].values()
    )
    assert result.report["work"]["projection_multiply_accumulate_count"] == (
        3 * 2560 * 8 * 4
    )


def test_signature_and_report_are_exact_across_caller_thread_counts() -> None:
    values = _case(source_neurons=31, source_hidden=12, target_hidden=7)
    original_threads = torch.get_num_threads()
    available_threads = max(2, min(4, original_threads))
    try:
        torch.set_num_threads(1)
        first = _build(values)
        torch.set_num_threads(available_threads)
        second = _build(values)
        assert torch.get_num_threads() == available_threads
    finally:
        torch.set_num_threads(original_threads)

    assert torch.equal(first.signature, second.signature)
    assert first.signature_sha256 == second.signature_sha256
    assert first.as_dict() == second.as_dict()


def test_explicit_cpu_float64_policy_preserves_default_bytes_and_report() -> None:
    values = _case(source_neurons=31, source_hidden=12, target_hidden=7)

    implicit = _build(values)
    explicit = _build(
        values,
        compute_device="cpu",
        compute_dtype="float64",
    )

    assert torch.equal(explicit.signature, implicit.signature)
    assert explicit.signature_sha256 == implicit.signature_sha256
    assert explicit.as_dict() == implicit.as_dict()
    assert "compute_execution" not in explicit.report
    assert explicit.report["operation_audit"] == implicit.report["operation_audit"]


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({"compute_device": "cuda"}, "explicitly indexed"),
        ({"compute_device": "cuda:-1"}, "explicitly indexed"),
        ({"compute_device": "cpu:0"}, "explicitly indexed"),
        ({"compute_dtype": "float32"}, "exactly 'float64'"),
    ],
)
def test_invalid_compute_policy_fails_closed(policy, message: str) -> None:
    with pytest.raises(GatedMlpWeightSignatureShapeError, match=message):
        _build(_case(), **policy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_cuda_float64_gemms_are_deterministic_and_transactional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    values = _case(source_neurons=31, source_hidden=12, target_hidden=7)
    originals = {name: value.clone() for name, value in values.items()}
    backend_before = {
        "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }

    first = _build(
        values,
        compute_device="cuda:0",
        compute_dtype="float64",
    )
    second = _build(
        values,
        compute_device="cuda:0",
        compute_dtype="float64",
    )

    assert torch.equal(first.signature, second.signature)
    assert first.signature_sha256 == second.signature_sha256
    assert first.as_dict() == second.as_dict()
    raw = {
        GATE_INPUT_ROLE: values["gate_weight"] @ values["residual_basis"],
        UP_INPUT_ROLE: values["up_weight"] @ values["residual_basis"],
        DOWN_OUTPUT_ROLE: (values["residual_basis"].T @ values["down_weight"]).T,
    }
    expected = torch.cat(
        tuple(raw[role] / _rms(raw[role]) for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES),
        dim=1,
    )
    torch.testing.assert_close(first.signature, expected, rtol=1e-14, atol=1e-14)
    execution = first.report["compute_execution"]
    execution_body = dict(execution)
    execution_sha256 = execution_body.pop("execution_sha256")
    assert canonical_sha256(execution_body) == execution_sha256
    assert execution["backend"] == "cuda"
    assert execution["requested_device"] == "cuda:0"
    assert execution["resolved_device"] == "cuda:0"
    assert execution["compute_dtype"] == "float64"
    assert execution["projection_gemm_calls"] == 3
    assert execution["parent_trace_rows_staged"] == 0
    assert execution["canonical_input_authentication_device"] == "cpu"
    assert execution["canonical_result_materialization_device"] == "cpu"
    assert first.report["operation_audit"]["cuda_signature_projection_gemm_calls"] == 3
    assert (
        first.report["invariants"]["deterministic_cuda_float64_explicit_device"] is True
    )
    assert first.report["invariants"]["all_hold"] is True
    for name, original in originals.items():
        assert values[name].device.type == "cpu"
        torch.testing.assert_close(values[name], original, rtol=0, atol=0)
    backend_after = {
        "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }
    assert backend_after == backend_before


def test_one_global_orthogonal_gauge_preserves_structural_selection_geometry() -> None:
    values = _case()
    baseline = _build(values)
    rotation = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    rotated_values = {
        **values,
        "residual_basis": values["residual_basis"] @ rotation,
    }

    rotated = _build(rotated_values)

    for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES:
        torch.testing.assert_close(
            rotated.role_block(role),
            baseline.role_block(role) @ rotation,
            rtol=0,
            atol=1e-15,
        )
    baseline_distances = torch.cdist(
        baseline.selection_signature,
        baseline.selection_signature,
    )
    rotated_distances = torch.cdist(
        rotated.selection_signature,
        rotated.selection_signature,
    )
    torch.testing.assert_close(
        rotated_distances,
        baseline_distances,
        rtol=1e-14,
        atol=1e-14,
    )


def test_input_pins_are_compatible_with_semantic_lfm2_tensor_digests() -> None:
    values = _case()
    assert all(
        gated_mlp_weight_signature_tensor_sha256(value)
        == lfm2_authenticated_tensor_sha256(value)
        for value in values.values()
    )

    pins = _pins(values)
    changed = {**values, "gate_weight": values["gate_weight"].clone()}
    changed["gate_weight"][0, 0] += 1.0
    with pytest.raises(
        GatedMlpWeightSignatureAuthenticationError,
        match="input authentication failed",
    ):
        build_gated_mlp_weight_complete_signature(
            **changed,
            expected_input_sha256=pins,
        )

    missing = dict(pins)
    missing.pop("w2.weight")
    with pytest.raises(
        GatedMlpWeightSignatureAuthenticationError,
        match="must pin exactly",
    ):
        build_gated_mlp_weight_complete_signature(
            **values,
            expected_input_sha256=missing,
        )


def test_each_weight_role_is_cryptographically_bound_to_selection_data() -> None:
    values = _case()
    baseline = _build(values)
    role_for_input = {
        "gate_weight": GATE_INPUT_ROLE,
        "up_weight": UP_INPUT_ROLE,
        "down_weight": DOWN_OUTPUT_ROLE,
    }

    for name, role in role_for_input.items():
        changed = {key: value.clone() for key, value in values.items()}
        if name == "down_weight":
            changed[name][0, -1] += 0.125
        else:
            changed[name][-1, 0] += 0.125
        candidate = _build(changed)
        assert candidate.signature_sha256 != baseline.signature_sha256
        assert (
            candidate.report["raw_role_sha256"][role]
            != (baseline.report["raw_role_sha256"][role])
        )
        for unchanged_role in set(GATED_MLP_WEIGHT_SIGNATURE_ROLES) - {role}:
            assert (
                candidate.report["raw_role_sha256"][unchanged_role]
                == (baseline.report["raw_role_sha256"][unchanged_role])
            )


def test_invalid_basis_shapes_nonorthogonality_and_zero_roles_fail_closed() -> None:
    values = _case()
    wrong_down = {**values, "down_weight": values["down_weight"][:, :-1]}
    with pytest.raises(GatedMlpWeightSignatureShapeError, match="W2 must"):
        _build(wrong_down)

    nonorthogonal = {
        **values,
        "residual_basis": values["residual_basis"].clone(),
    }
    nonorthogonal["residual_basis"][:, 0] *= 2.0
    with pytest.raises(
        GatedMlpWeightSignatureInvariantError,
        match="orthonormal columns",
    ):
        _build(nonorthogonal)

    zero_gate = {**values, "gate_weight": torch.zeros_like(values["gate_weight"])}
    with pytest.raises(
        GatedMlpWeightSignatureFiniteError,
        match="finite nonzero RMS",
    ):
        _build(zero_gate)

    nonfinite = {**values, "up_weight": values["up_weight"].clone()}
    nonfinite["up_weight"][0, 0] = float("nan")
    with pytest.raises(GatedMlpWeightSignatureFiniteError, match="NaN or infinity"):
        build_gated_mlp_weight_complete_signature(
            **nonfinite,
            expected_input_sha256=_pins(values),
        )


def test_materialized_signature_tampering_is_detected() -> None:
    result = _build(_case())
    result.signature[0, 0] += 1.0

    with pytest.raises(
        GatedMlpWeightSignatureAuthenticationError,
        match="changed after construction",
    ):
        result.verify()


def test_module_ast_has_no_trace_model_execution_or_training_surface() -> None:
    module = Path("src/neural_forge/hf_gated_mlp_weight_signature.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(module)
    forbidden_names = {
        "backward",
        "forward",
        "generate",
        "grad",
        "optimizer",
        "step",
    }
    called_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert forbidden_names.isdisjoint(called_names)
