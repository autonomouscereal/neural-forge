from __future__ import annotations

import ast
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
    ResidualFacingOperator,
    WeightCompleteBasisError,
    build_weight_complete_basis,
    conv_state_operators,
    embedding_head_operators,
    norm_operators,
    output_mlp_span_operators,
    qkv_input_sensitivity_operators,
    residual_facing_operator,
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
    expected = (
        consumer.T @ consumer
        + producer @ producer.T
        + torch.diag(norm.square())
    )

    assert torch.equal(result.full_weight_gram, expected)
    assert result.report["source_width"] == 3
    assert result.report["full_weight_rank"] == 3
    assert result.report["basis_rank"] == 3
    assert result.report["source_axis_loading"][
        "all_source_axes_structurally_covered_and_loaded"
    ] is True
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
                    tensor=torch.tensor(
                        [[scale, 0.0]], dtype=torch.float64
                    ),
                    orientation=FeatureOrientation.INPUT,
                ),
                ResidualFacingOperator(
                    role="state",
                    name="state",
                    tensor=torch.tensor(
                        [[0.0], [1.0]], dtype=torch.float64
                    ),
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
    assert moderate.report["role_frobenius_balancing"][
        "large_row_count_roles_cannot_dominate_by_size"
    ] is True
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
    assert first.report["determinism"][
        "degenerate_subspaces_canonicalized_from_projectors"
    ] is True
    assert first.report["determinism"][
        "caller_thread_count_independent"
    ] is True


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
    activation = torch.diag(
        torch.tensor([1e12, 0.0, 0.0], dtype=torch.float64)
    )
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
    structural_energy = float(
        direction @ hybrid.balanced_structural_gram @ direction
    )
    combined_energy = float(direction @ hybrid.combined_gram @ direction)

    assert combined_energy + 1e-15 >= 0.8 * structural_energy
    assert combined_energy > 0.0
    assert float(hybrid.basis[2].square().sum().item()) > 0.0
    assert hybrid.report["source_axis_loading"][
        "all_source_axes_structurally_covered_and_loaded"
    ] is True
    assert hybrid.report["activation_supplement"][
        "activation_cannot_replace_structural_objective"
    ] is True
    assert hybrid.report["activation_supplement"][
        "structural_direction_energy_lower_bound_factor"
    ] == pytest.approx(0.8)
    assert hybrid.report["hashes"]["full_weight_gram_sha256"] == (
        structural.report["hashes"]["full_weight_gram_sha256"]
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
        value > 0.0
        for value in result.report["per_role_retained_energy"].values()
    )

    source = Path(
        "src/neural_forge/dynamic_weight_complete_basis.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
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
