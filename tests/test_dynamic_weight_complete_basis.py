from __future__ import annotations

import ast
import gc
import json
import weakref
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

import neural_forge.dynamic_weight_complete_basis as weight_basis  # noqa: E402
from neural_forge.analytic_execution_profile import (  # noqa: E402
    resolve_analytic_execution_profile,
)
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
    ROLE_BALANCED_OPERATOR_GEOMETRY_ARITHMETIC_RECIPE,
    ResidualFacingOperator,
    RoleBalancedOperatorGeometry,
    WeightCompleteBasisError,
    build_weight_complete_basis,
    compile_role_balanced_operator_geometry,
    conv_state_operators,
    embedding_head_operators,
    norm_operators,
    output_mlp_span_operators,
    qkv_input_sensitivity_operators,
    residual_facing_operator,
)


def _role_balanced_geometry_operators() -> tuple[ResidualFacingOperator, ...]:
    return (
        ResidualFacingOperator(
            role="alpha",
            name="a_consumer",
            tensor=torch.tensor(
                [[1.0, 2.0, 0.5], [3.0, 0.25, 4.0]],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
            gram_weight=0.75,
        ),
        ResidualFacingOperator(
            role="alpha",
            name="z_norm",
            tensor=torch.tensor([1.5, 0.5, 2.0], dtype=torch.float64),
            orientation=FeatureOrientation.OUTPUT,
            diagonal=True,
        ),
        ResidualFacingOperator(
            role="beta",
            name="producer",
            tensor=torch.tensor(
                [[2.0, 0.0], [1.0, 3.0], [0.5, 2.0]],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.OUTPUT,
        ),
    )


def test_role_balanced_geometry_is_exact_analytic_division_formula() -> None:
    operators = _role_balanced_geometry_operators()
    geometry = compile_role_balanced_operator_geometry(tuple(reversed(operators)))
    consumer, norm, producer = operators
    alpha = consumer.tensor.T @ consumer.tensor * consumer.gram_weight + torch.diag(
        norm.tensor.square()
    )
    beta = producer.tensor @ producer.tensor.T
    expected_raw = alpha + beta
    expected_balanced = alpha / (2 * float(torch.trace(alpha).item())) + beta / (
        2 * float(torch.trace(beta).item())
    )

    assert torch.equal(geometry.raw_gram, expected_raw)
    assert torch.equal(geometry.balanced_gram, expected_balanced)
    assert geometry.raw_gram.device.type == "cpu"
    assert geometry.balanced_gram.device.type == "cpu"
    assert geometry.raw_gram.dtype == torch.float64
    assert geometry.balanced_gram.dtype == torch.float64
    assert geometry.raw_gram.is_contiguous()
    assert geometry.balanced_gram.is_contiguous()
    assert RoleBalancedOperatorGeometry.__slots__ == (
        "raw_gram",
        "balanced_gram",
        "report",
    )
    assert geometry.report["arithmetic_recipe"] == (
        ROLE_BALANCED_OPERATOR_GEOMETRY_ARITHMETIC_RECIPE
    )
    assert geometry.ordered_operator_sha256s == tuple(
        value.operator_sha256
        for value in sorted(
            operators,
            key=lambda value: (
                value.role,
                value.name,
                value.orientation.value,
                value.feature_axis,
                value.operator_sha256,
            ),
        )
    )
    assert len(geometry.operator_reports) == len(operators)
    assert set(geometry.role_reports) == {"alpha", "beta"}
    assert (
        geometry.report["compute_execution"]["deterministic_execution_settings_applied"]
        is True
    )
    assert (
        geometry.report["compute_execution"][
            "deterministic_execution_settings_restoration_required"
        ]
        is True
    )
    assert (
        geometry.report["compute_execution"][
            "deterministic_execution_settings_restored_before_return"
        ]
        is True
    )
    assert geometry.verify() is True
    json.dumps(geometry.as_dict(), allow_nan=False, sort_keys=True)

    basis = build_weight_complete_basis(
        operators,
        target_width=2,
        require_all_source_axis_loading=False,
    )
    assert torch.equal(basis.full_weight_gram, geometry.raw_gram)
    assert torch.equal(
        basis.balanced_structural_gram,
        geometry.balanced_gram,
    )
    reciprocal_multiply = alpha * (
        1.0 / (2 * float(torch.trace(alpha).item()))
    ) + beta * (1.0 / (2 * float(torch.trace(beta).item())))
    torch.testing.assert_close(
        geometry.balanced_gram,
        reciprocal_multiply,
        rtol=2e-15,
        atol=2e-16,
    )


def test_operator_cached_tensor_hash_preserves_identity_and_detects_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    operator = ResidualFacingOperator(
        role="r",
        name="n",
        tensor=source,
        orientation=FeatureOrientation.INPUT,
    )
    assert operator.tensor_sha256 == (
        "5722f90a406e2c73c1863eb60f9cd1c60c8f6d352af851660ebf1c57278f1859"
    )
    assert operator.operator_sha256 == (
        "e3c87bd56bbb9f1401f13ff3e75a48ddcda1ea69c827343d1be33443811013ed"
    )
    source.fill_(999.0)
    assert operator.verify() is True
    original_tensor_sha256 = weight_basis._tensor_sha256
    reauthentication_calls = 0

    def counted_tensor_sha256(value: torch.Tensor) -> str:
        nonlocal reauthentication_calls
        reauthentication_calls += 1
        return original_tensor_sha256(value)

    monkeypatch.setattr(
        weight_basis,
        "_tensor_sha256",
        counted_tensor_sha256,
    )
    assert operator.as_dict()["tensor_sha256"] == operator.tensor_sha256
    assert reauthentication_calls == 0
    assert operator.verify() is True
    assert reauthentication_calls == 1

    operator.tensor[0, 0] += 1.0
    assert operator.verify() is False
    assert reauthentication_calls == 2
    with pytest.raises(WeightCompleteBasisError, match="changed after authentication"):
        compile_role_balanced_operator_geometry((operator,))
    assert reauthentication_calls == 3


def test_geometry_rejects_mutation_after_descriptor_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _role_balanced_geometry_operators()[0]
    original_order = weight_basis._ordered_operator_descriptors
    gram_calls = 0

    def mutate_at_boundary(values):
        ordered = original_order(values)
        ordered[0].tensor[0, 0] += 0.5
        return ordered

    def forbidden_gram(*_args, **_kwargs):
        nonlocal gram_calls
        gram_calls += 1
        raise AssertionError("A stale operator SHA authorized a changed Gram.")

    monkeypatch.setattr(
        weight_basis,
        "_ordered_operator_descriptors",
        mutate_at_boundary,
    )
    monkeypatch.setattr(weight_basis, "_operator_gram_on_device", forbidden_gram)

    with pytest.raises(WeightCompleteBasisError, match="changed after authentication"):
        compile_role_balanced_operator_geometry((operator,))

    assert gram_calls == 0


def test_geometry_owns_only_one_authenticated_operator_snapshot_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operators = _role_balanced_geometry_operators()
    original_snapshot = weight_basis._owning_operator_snapshot
    snapshot_tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    snapshot_calls = 0

    def observed_snapshot(descriptor: ResidualFacingOperator):
        nonlocal snapshot_calls
        if snapshot_tensor_refs:
            assert snapshot_tensor_refs[-1]() is None
        snapshot = original_snapshot(descriptor)
        assert snapshot.tensor.data_ptr() != descriptor.tensor.data_ptr()
        snapshot_tensor_refs.append(weakref.ref(snapshot.tensor))
        snapshot_calls += 1
        return snapshot

    monkeypatch.setattr(
        weight_basis,
        "_owning_operator_snapshot",
        observed_snapshot,
    )

    geometry = compile_role_balanced_operator_geometry(operators)
    gc.collect()

    assert geometry.verify() is True
    assert snapshot_calls == len(operators)
    assert all(reference() is None for reference in snapshot_tensor_refs)


def test_role_balanced_geometry_rejects_tensor_and_resealed_metadata_tampering() -> (
    None
):
    geometry = compile_role_balanced_operator_geometry(
        _role_balanced_geometry_operators()
    )
    tampered_raw = geometry.raw_gram.clone()
    tampered_raw[0, 0] += 1.0
    with pytest.raises(WeightCompleteBasisError, match="aggregate|authentication"):
        RoleBalancedOperatorGeometry(
            raw_gram=tampered_raw,
            balanced_gram=geometry.balanced_gram,
            report=geometry.as_dict(),
        )

    tampered_report = geometry.as_dict()
    tampered_report["roles"]["alpha"]["operator_count"] += 1
    tampered_body = {
        key: value for key, value in tampered_report.items() if key != "result_sha256"
    }
    tampered_report["result_sha256"] = weight_basis._canonical_json_sha256(
        tampered_body
    )
    with pytest.raises(WeightCompleteBasisError, match="aggregate metadata"):
        RoleBalancedOperatorGeometry(
            raw_gram=geometry.raw_gram,
            balanced_gram=geometry.balanced_gram,
            report=tampered_report,
        )
    with pytest.raises(TypeError):
        geometry.report["roles"]["alpha"]["operator_count"] = 99

    geometry.raw_gram[0, 0] += 1.0
    assert geometry.verify() is False


@pytest.mark.parametrize(
    "malformation",
    (
        "drop_top_level_field",
        "unknown_top_level_field",
        "drop_operator_name",
        "substitute_operator_name",
        "uppercase_operator_digest",
        "boolean_operator_source_width",
        "boolean_feature_axis",
        "integer_gram_weight",
        "mismatched_tensor_shape",
        "inconsistent_diagonal_flag",
        "noncanonical_operator_order",
        "substituted_role_membership",
        "substituted_role_count",
        "drop_compute_execution_field",
        "substitute_compute_device",
        "false_settings_application",
        "false_settings_restoration",
    ),
)
def test_role_balanced_geometry_rejects_resealed_malformed_schema(
    malformation: str,
) -> None:
    geometry = compile_role_balanced_operator_geometry(
        _role_balanced_geometry_operators()
    )
    report = geometry.as_dict()
    report.pop("result_sha256")

    if malformation == "drop_top_level_field":
        report.pop("model_forward_calls")
    elif malformation == "unknown_top_level_field":
        report["unsealed_extension"] = True
    elif malformation == "drop_operator_name":
        # This malformed artifact previously authenticated and later raised a
        # raw KeyError in the analytic preaggregated-geometry consumer.
        report["operators"][0].pop("name")
    elif malformation == "substitute_operator_name":
        report["operators"][0]["name"] = "forged_name"
    elif malformation == "uppercase_operator_digest":
        report["operators"][0]["operator_sha256"] = report["operators"][0][
            "operator_sha256"
        ].upper()
        report["ordered_operator_sha256s"][0] = report["operators"][0][
            "operator_sha256"
        ]
    elif malformation == "boolean_operator_source_width":
        report["operators"][0]["source_width"] = True
    elif malformation == "boolean_feature_axis":
        report["operators"][0]["feature_axis"] = True
    elif malformation == "integer_gram_weight":
        report["operators"][0]["gram_weight"] = 1
    elif malformation == "mismatched_tensor_shape":
        report["operators"][0]["tensor_shape"][
            report["operators"][0]["feature_axis"]
        ] += 1
    elif malformation == "inconsistent_diagonal_flag":
        report["operators"][0]["diagonal"] = True
    elif malformation == "noncanonical_operator_order":
        report["operators"][0], report["operators"][1] = (
            report["operators"][1],
            report["operators"][0],
        )
    elif malformation == "substituted_role_membership":
        report["roles"]["forged_role"] = report["roles"].pop("alpha")
    elif malformation == "substituted_role_count":
        report["roles"]["alpha"]["operator_count"] += 1
    elif malformation == "drop_compute_execution_field":
        report["compute_execution"].pop("backend")
    elif malformation == "substitute_compute_device":
        report["compute_execution"]["requested_device"] = "cuda:7"
    elif malformation == "false_settings_application":
        report["compute_execution"]["deterministic_execution_settings_applied"] = False
    elif malformation == "false_settings_restoration":
        report["compute_execution"][
            "deterministic_execution_settings_restored_before_return"
        ] = False
    else:  # pragma: no cover - exhaustive parametrization guard
        raise AssertionError(malformation)

    report["result_sha256"] = weight_basis._canonical_json_sha256(report)
    with pytest.raises(WeightCompleteBasisError):
        RoleBalancedOperatorGeometry(
            raw_gram=geometry.raw_gram,
            balanced_gram=geometry.balanced_gram,
            report=report,
        )


def test_basis_compiles_each_operator_gram_once_through_shared_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operators = _role_balanced_geometry_operators()
    operator_calls: list[str] = []
    compile_calls = 0
    original_operator_gram = weight_basis._operator_gram
    original_compile = weight_basis._compile_role_balanced_operator_geometry

    def counted_operator_gram(operator: ResidualFacingOperator) -> torch.Tensor:
        operator_calls.append(operator.operator_sha256)
        return original_operator_gram(operator)

    def counted_compile(*args: object, **kwargs: object):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(weight_basis, "_operator_gram", counted_operator_gram)
    monkeypatch.setattr(
        weight_basis,
        "_compile_role_balanced_operator_geometry",
        counted_compile,
    )

    build_weight_complete_basis(
        operators,
        target_width=2,
        require_all_source_axis_loading=False,
    )

    assert compile_calls == 1
    assert len(operator_calls) == len(operators)
    assert set(operator_calls) == {operator.operator_sha256 for operator in operators}


def test_geometry_accepts_explicit_authenticated_cpu_fallback_profile() -> None:
    profile = resolve_analytic_execution_profile(
        "cuda:999999",
        allow_cpu_fallback=True,
    )
    assert profile.cpu_fallback_used is True

    geometry = compile_role_balanced_operator_geometry(
        _role_balanced_geometry_operators(),
        compute_device=profile,
    )

    assert geometry.report["execution_profile_sha256"] == profile.result_sha256
    assert geometry.report["execution_profile"]["requested_device"] == (
        profile.requested_device
    )
    assert geometry.report["execution_profile"]["resolved_device"] == "cpu"
    assert geometry.report["execution_profile"]["cpu_fallback_used"] is True


@pytest.mark.parametrize("fail", (False, True), ids=("success", "exception"))
def test_geometry_applies_and_restores_sealed_cpu_execution_settings(
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    initial = {
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
    }
    configured_threads = max(2, min(4, initial["threads"]))
    original_gram = weight_basis._operator_gram_on_device
    observations: list[dict[str, object]] = []

    def observed_gram(operator, device):
        observations.append(
            {
                "threads": torch.get_num_threads(),
                "deterministic": (torch.are_deterministic_algorithms_enabled()),
                "warn_only": (torch.is_deterministic_algorithms_warn_only_enabled()),
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "benchmark": torch.backends.cudnn.benchmark,
            }
        )
        if fail:
            raise RuntimeError("injected Gram failure")
        return original_gram(operator, device)

    try:
        torch.set_num_threads(configured_threads)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        monkeypatch.setattr(
            weight_basis,
            "_operator_gram_on_device",
            observed_gram,
        )
        if fail:
            with pytest.raises(RuntimeError, match="injected Gram failure"):
                compile_role_balanced_operator_geometry(
                    _role_balanced_geometry_operators()
                )
        else:
            compile_role_balanced_operator_geometry(_role_balanced_geometry_operators())

        assert observations
        assert all(value["threads"] == 1 for value in observations)
        assert all(value["deterministic"] is True for value in observations)
        assert all(value["warn_only"] is False for value in observations)
        assert all(value["tf32"] is False for value in observations)
        assert all(value["benchmark"] is False for value in observations)
        assert torch.get_num_threads() == configured_threads
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.benchmark = initial["benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = initial["tf32"]
        torch.use_deterministic_algorithms(
            initial["deterministic"],
            warn_only=initial["warn_only"],
        )
        torch.set_num_threads(initial["threads"])


def test_cuda_execution_profile_context_never_mutates_cpu_thread_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
    }

    class _CudaProfile:
        torch_device = torch.device("cuda:0")
        receipt = {"cpu_thread_count": 1}

    def forbidden_thread_mutation(_threads: int) -> None:
        raise AssertionError("CUDA geometry changed the CPU thread count")

    monkeypatch.setattr(torch, "set_num_threads", forbidden_thread_mutation)
    with weight_basis._execution_profile_settings(_CudaProfile()):
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False

    assert torch.are_deterministic_algorithms_enabled() == initial["deterministic"]
    assert torch.is_deterministic_algorithms_warn_only_enabled() == initial["warn_only"]
    assert torch.backends.cuda.matmul.allow_tf32 == initial["tf32"]
    assert torch.backends.cudnn.benchmark == initial["benchmark"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_geometry_matches_cpu_with_tight_fp64_tolerance() -> None:
    operators = _role_balanced_geometry_operators()
    cpu = compile_role_balanced_operator_geometry(operators)
    cuda = compile_role_balanced_operator_geometry(
        operators,
        compute_device="cuda:0",
    )

    torch.testing.assert_close(cuda.raw_gram, cpu.raw_gram, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(
        cuda.balanced_gram,
        cpu.balanced_gram,
        rtol=1e-12,
        atol=1e-13,
    )
    assert cuda.raw_gram.device.type == "cpu"
    assert cuda.balanced_gram.device.type == "cpu"
    assert (
        cuda.report["compute_execution"]["cpu_thread_count_forced_by_this_compiler"]
        is None
    )


def test_orientation_helpers_accumulate_complete_source_width_gram() -> None:
    consumer = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]],
        dtype=torch.float64,
    )
    producer = torch.tensor(
        [[1.0, 0.0], [2.0, 1.0], [0.0, 2.0]],
        dtype=torch.float64,
    )
    norm = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    operators = (
        residual_facing_operator(
            consumer,
            role="consumer",
            name="consumer",
            orientation=FeatureOrientation.INPUT,
        ),
        residual_facing_operator(
            producer,
            role="producer",
            name="producer",
            orientation=FeatureOrientation.OUTPUT,
        ),
        *norm_operators(norm, role="norm"),
    )

    result = build_weight_complete_basis(operators, target_width=3)
    expected = consumer.T @ consumer + producer @ producer.T + torch.diag(norm.square())

    assert torch.equal(result.full_weight_gram, expected)
    assert result.report["source_width"] == 3
    assert result.report["full_weight_rank"] == 3
    assert result.report["basis_rank"] == 3
    assert (
        result.report["source_axis_loading"][
            "all_source_axes_structurally_covered_and_loaded"
        ]
        is True
    )
    assert result.report["coverage_gate"]["passed"] is True
    assert result.report["concatenated_operator_rows_materialized"] is False
    assert result.report["model_forward_calls"] == 0
    assert result.report["trace_capture_calls"] == 0
    assert result.report["training_steps"] == 0
    assert result.report["autograd_used"] is False
    assert result.report["filesystem_writes"] == 0


def test_role_frobenius_balancing_blocks_large_embedding_domination() -> None:
    def build(scale: float):
        return build_weight_complete_basis(
            (
                ResidualFacingOperator(
                    role="embedding",
                    name="embedding",
                    tensor=torch.tensor([[scale, 0.0]], dtype=torch.float64),
                    orientation=FeatureOrientation.INPUT,
                ),
                ResidualFacingOperator(
                    role="state",
                    name="state",
                    tensor=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                    orientation=FeatureOrientation.OUTPUT,
                ),
            ),
            target_width=1,
            require_all_source_axis_loading=False,
        )

    moderate = build(1e3)
    enormous = build(1e12)

    assert torch.equal(
        moderate.balanced_structural_gram,
        enormous.balanced_structural_gram,
    )
    assert torch.equal(moderate.basis, enormous.basis)
    for role in ("embedding", "state"):
        assert moderate.report["roles"][role][
            "balanced_trace_contribution"
        ] == pytest.approx(0.5)
    assert (
        moderate.report["role_frobenius_balancing"][
            "large_row_count_roles_cannot_dominate_by_size"
        ]
        is True
    )
    assert (
        moderate.report["full_weight_raw_trace_energy"]
        != enormous.report["full_weight_raw_trace_energy"]
    )


def test_degenerate_basis_is_stable_across_thread_counts_and_input_order() -> None:
    operators = (
        ResidualFacingOperator(
            role="input",
            name="identity_input",
            tensor=torch.eye(4, dtype=torch.float64),
            orientation=FeatureOrientation.INPUT,
        ),
        ResidualFacingOperator(
            role="output",
            name="identity_output",
            tensor=torch.eye(4, dtype=torch.float64),
            orientation=FeatureOrientation.OUTPUT,
        ),
    )
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = build_weight_complete_basis(
            operators,
            target_width=2,
            require_all_source_axis_loading=False,
        )
        torch.set_num_threads(2)
        second = build_weight_complete_basis(
            tuple(reversed(operators)),
            target_width=2,
            require_all_source_axis_loading=False,
        )
    finally:
        torch.set_num_threads(original_threads)

    expected = torch.eye(4, dtype=torch.float64)[:, :2]
    assert torch.equal(first.basis, expected)
    assert torch.equal(second.basis, expected)
    assert first.report["hashes"] == second.report["hashes"]
    assert first.report["run_sha256"] == second.report["run_sha256"]
    assert first.report["degenerate_eigenspaces"][0]["multiplicity"] == 4
    assert (
        first.report["determinism"][
            "degenerate_subspaces_canonicalized_from_projectors"
        ]
        is True
    )
    assert first.report["determinism"]["caller_thread_count_independent"] is True


def test_bounded_activation_gram_cannot_erase_structural_direction() -> None:
    weight = torch.tensor(
        [
            [3.0, 1.0, 0.5],
            [1.0, 2.0, 0.7],
            [0.5, 0.7, 1.5],
        ],
        dtype=torch.float64,
    )
    structural_operator = ResidualFacingOperator(
        role="structural",
        name="dense_structural",
        tensor=weight,
        orientation=FeatureOrientation.INPUT,
    )
    activation = torch.diag(torch.tensor([1e12, 0.0, 0.0], dtype=torch.float64))
    structural = build_weight_complete_basis(
        (structural_operator,),
        target_width=2,
    )
    hybrid = build_weight_complete_basis(
        (structural_operator,),
        target_width=2,
        activation_gram=activation,
        activation_weight=0.2,
        maximum_activation_weight=0.25,
    )
    direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    structural_energy = float(direction @ hybrid.balanced_structural_gram @ direction)
    combined_energy = float(direction @ hybrid.combined_gram @ direction)

    assert combined_energy + 1e-15 >= 0.8 * structural_energy
    assert combined_energy > 0.0
    assert float(hybrid.basis[2].square().sum().item()) > 0.0
    assert (
        hybrid.report["source_axis_loading"][
            "all_source_axes_structurally_covered_and_loaded"
        ]
        is True
    )
    assert (
        hybrid.report["activation_supplement"][
            "activation_cannot_replace_structural_objective"
        ]
        is True
    )
    assert hybrid.report["activation_supplement"][
        "structural_direction_energy_lower_bound_factor"
    ] == pytest.approx(0.8)
    assert (
        hybrid.report["hashes"]["full_weight_gram_sha256"]
        == (structural.report["hashes"]["full_weight_gram_sha256"])
    )
    with pytest.raises(Exception, match="exceeds"):
        build_weight_complete_basis(
            (structural_operator,),
            target_width=2,
            activation_gram=activation,
            activation_weight=0.3,
            maximum_activation_weight=0.25,
        )


def test_composition_helpers_cover_all_structural_operator_families() -> None:
    generator = torch.Generator().manual_seed(44)
    width = 5
    operators = (
        *qkv_input_sensitivity_operators(
            torch.randn(4, width, generator=generator),
            torch.randn(2, width, generator=generator),
            torch.randn(2, width, generator=generator),
        ),
        *output_mlp_span_operators(
            torch.randn(width, 4, generator=generator),
            torch.randn(width, 7, generator=generator),
        ),
        *embedding_head_operators(
            torch.randn(40, width, generator=generator),
            torch.randn(40, width, generator=generator),
        ),
        *norm_operators(
            (
                torch.randn(width, generator=generator),
                torch.randn(width, generator=generator),
            )
        ),
        *conv_state_operators(
            torch.randn(width, 1, 3, generator=generator),
            torch.randn(width, width, generator=generator),
        ),
    )

    result = build_weight_complete_basis(
        operators,
        target_width=4,
        minimum_full_weight_retained_energy_fraction=0.25,
    )

    assert set(result.report["roles"]) == {
        "convolution_output_span",
        "embedding_interface",
        "mlp_output_span",
        "normalization",
        "operator_output_span",
        "output_head_interface",
        "qkv_input_sensitivity",
        "state_operator_span",
    }
    assert result.report["operator_count"] == 11
    assert result.report["coverage_gate"]["passed"] is True
    assert all(
        value > 0.0 for value in result.report["per_role_retained_energy"].values()
    )

    source = Path("src/neural_forge/dynamic_weight_complete_basis.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "forward" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert "save" not in called_attributes
    assert "write_text" not in called_attributes
    assert "write_bytes" not in called_attributes
    assert "transformers" not in source.lower()


def test_result_implements_dense_coordinate_fold_protocol() -> None:
    generator = torch.Generator().manual_seed(990)
    operators = (
        residual_facing_operator(
            torch.randn(7, 5, generator=generator, dtype=torch.float64),
            role="consumer",
            name="consumer",
            orientation=FeatureOrientation.INPUT,
        ),
        residual_facing_operator(
            torch.randn(5, 9, generator=generator, dtype=torch.float64),
            role="producer",
            name="producer",
            orientation=FeatureOrientation.OUTPUT,
        ),
    )
    result = build_weight_complete_basis(
        operators,
        3,
        enforce_coverage_gate=False,
    )
    consumer = torch.randn(4, 5, generator=generator)
    producer = torch.randn(5, 6, generator=generator)
    rows = torch.randn(8, 5, generator=generator)
    encoded = result.encode(rows)
    assert torch.equal(encoded, result(rows))
    assert torch.equal(encoded, result.apply(rows))
    assert torch.allclose(
        result.fold_consumer_weight(consumer),
        consumer.to(torch.float64) @ result.basis,
    )
    assert torch.allclose(
        result.fold_producer_weight(producer),
        result.basis.T @ producer.to(torch.float64),
    )
    assert torch.equal(
        result.materialize_dense(
            max_elements=15,
            dtype=torch.float32,
        ),
        result.basis.to(torch.float32),
    )
    assert result.map_sha256 == result.report["hashes"]["basis_sha256"]
    with pytest.raises(WeightCompleteBasisError, match="materialization bound"):
        result.materialize_dense(max_elements=14)
