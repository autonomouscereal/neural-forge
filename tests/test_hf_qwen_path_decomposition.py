from __future__ import annotations

import copy

import pytest

from neural_forge.hf_qwen_path_decomposition import (
    compress_post_nonlinearity_paths,
)


torch = pytest.importorskip("torch")


def _duplicate_response_problem(*, dtype=torch.float64):
    rows = 16
    position = torch.linspace(-1.25, 1.5, rows, dtype=dtype)
    first = position + 0.25 * position.square()
    second = torch.cos(position * 1.7) + 0.1 * position
    features = torch.stack(
        [first, second, 0.5 * first, 0.25 * second], dim=1
    )
    directions = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=dtype,
    )
    prompts = [f"prompt-{index}" for index in range(rows)]
    domains = ["math" if index % 2 == 0 else "code" for index in range(rows)]
    layers = [0, 1, 0, 1]
    return features, directions, prompts, domains, layers


def _compress_duplicate_problem(**overrides):
    features, directions, prompts, domains, layers = _duplicate_response_problem()
    arguments = {
        "prompt_ids": prompts,
        "domain_labels": domains,
        "donor_layer_ids": layers,
        "target_path_count": 2,
        "ridge": 1e-12,
        "split_seed": 19,
        "selection_sketch_size": 4096,
    }
    arguments.update(overrides)
    return compress_post_nonlinearity_paths(features, directions, **arguments)


@pytest.mark.parametrize("interpolation_mode", ["dense", "sparse-signature"])
def test_dense_and_sparse_fold_discarded_duplicate_paths(
    interpolation_mode: str,
) -> None:
    rows = 14
    activation = torch.linspace(-1.0, 2.0, rows, dtype=torch.float64)
    features = torch.stack([activation, activation], dim=1)
    directions = torch.tensor([[2.0], [2.0]], dtype=torch.float64)
    result = compress_post_nonlinearity_paths(
        features,
        directions,
        prompt_ids=[f"duplicate-{index}" for index in range(rows)],
        domain_labels=["math" if index % 2 else "code" for index in range(rows)],
        donor_layer_ids=[0, 0],
        target_path_count=1,
        ridge=1e-12,
        split_seed=7,
        selection_sketch_size=1024,
        interpolation_mode=interpolation_mode,
    )

    assert result["source_path_to_selected_path"].tolist() == [0, 0]
    assert result["selected_native_output_directions"].item() == pytest.approx(2.0)
    assert result["aggregate_output_coefficients"].item() == pytest.approx(
        4.0, rel=1e-9, abs=1e-9
    )
    assert result["report"]["diagnostics"]["held_out"]["fitted"][
        "absolute_rmse"
    ] < 1e-9
    assert result["report"]["coefficient_fit"][
        "source_paths_with_nonzero_fitted_output_contribution"
    ] == 2


def test_sparse_signature_is_deterministic_compact_and_bounded() -> None:
    common = {
        "interpolation_mode": "sparse-signature",
        "sparse_chunk_size": 2,
        "diagnostic_sketch_size": 1,
    }
    first = _compress_duplicate_problem(**common)
    second = _compress_duplicate_problem(**common)

    assert torch.equal(first["selected_path_indices"], second["selected_path_indices"])
    assert torch.equal(
        first["source_path_to_selected_path"],
        second["source_path_to_selected_path"],
    )
    assert torch.equal(
        first["source_scalar_coefficients"],
        second["source_scalar_coefficients"],
    )
    assert torch.equal(
        first["aggregate_output_coefficients"],
        second["aggregate_output_coefficients"],
    )
    assert first["report"]["hashes"] == second["report"]["hashes"]

    coefficients = first["source_to_selected_coefficients"]
    assert coefficients.shape == (4,)
    assert first["source_scalar_coefficients"].data_ptr() == coefficients.data_ptr()
    assert torch.isfinite(coefficients).all()
    assert torch.isfinite(first["aggregate_output_coefficients"]).all()
    selected = set(first["selected_path_indices"].tolist())
    assert set(first["source_path_to_selected_path"].tolist()) <= selected

    report = first["report"]
    assert report["interpolation_mode"] == "sparse-signature"
    assert report["shape"]["source_to_selected_coefficients"] == [4]
    assert "interpolation_sha256" not in report["hashes"]
    assert report["coverage"]["all_source_paths_mapped"] is True
    assert report["coverage"]["all_path_coverage_fraction"] == 1.0
    assert report["coefficient_fit"][
        "all_source_paths_have_finite_coefficients"
    ] is True
    assert report["coefficient_fit"]["sparse_coefficient_statistics"][
        "finite_count"
    ] == 4

    materialization = report["materialization"]
    assert materialization[
        "dense_target_by_source_coefficients_materialized"
    ] is False
    assert materialization["assignment_similarity_matrix_shape_bound"][0] < 4
    forbidden_shapes = ([2, 4], [4, 2], [4, 1, 2, 1])
    assert all(
        shape not in forbidden_shapes
        for shape in materialization["matrix_shape_bounds"].values()
    )
    held_out = report["diagnostics"]["held_out"]
    assert held_out["approximate"] is True
    scalable = report["diagnostics"]["scalable_evaluation"]
    assert scalable["approximation_explicit"] is True
    assert scalable["output_space"]["evaluated_output_features"] == 1
    assert len(held_out["by_prompt"]) > 0
    assert {item["domain"] for item in held_out["by_domain"]} == {
        "math",
        "code",
    }


def test_sparse_signature_fits_signed_source_scalars() -> None:
    rows = 12
    activation = torch.linspace(-1.0, 1.0, rows, dtype=torch.float64)
    features = torch.stack([activation, -activation], dim=1)
    directions = torch.ones((2, 1), dtype=torch.float64)
    result = compress_post_nonlinearity_paths(
        features,
        directions,
        prompt_ids=[f"signed-{index}" for index in range(rows)],
        domain_labels=["a" if index % 2 else "b" for index in range(rows)],
        donor_layer_ids=[0, 0],
        target_path_count=1,
        interpolation_mode="sparse-signature",
        ridge=1e-12,
        split_seed=5,
        selection_sketch_size=1024,
    )

    coefficients = result["source_scalar_coefficients"]
    assert coefficients[0].item() == 1.0
    assert coefficients[1].item() == pytest.approx(-1.0, rel=1e-9, abs=1e-9)
    assert result["aggregate_output_coefficients"].item() == pytest.approx(
        0.0, abs=1e-9
    )
    assert result["report"]["coefficient_fit"][
        "sparse_coefficient_statistics"
    ]["negative_count"] == 1


def test_sparse_selector_bounds_candidates_and_preserves_exact_layer_quotas() -> None:
    rows = 16
    source_paths = 40
    activation = torch.linspace(-1.0, 1.0, rows, dtype=torch.float64)
    features = activation[:, None].repeat(1, source_paths)
    directions = torch.ones((source_paths, 2), dtype=torch.float64)
    result = compress_post_nonlinearity_paths(
        features,
        directions,
        prompt_ids=[f"quota-{index}" for index in range(rows)],
        domain_labels=["math" if index % 2 else "code" for index in range(rows)],
        donor_layer_ids=[0] * 20 + [1] * 20,
        target_path_count=6,
        minimum_paths_per_layer={0: 3, 1: 3},
        interpolation_mode="sparse-signature",
        split_seed=11,
        selection_sketch_size=1024,
    )

    assert result["selected_layer_ids"].count(0) == 3
    assert result["selected_layer_ids"].count(1) == 3
    scaling = result["report"]["selection"]["sparse_scalability"]
    assert scaling["strategy"] == "independent-per-layer-vectorized-pivoted-qr"
    assert scaling["candidate_pool_size"] == 12
    assert scaling["candidate_pool_size"] < source_paths
    assert scaling["complexity_bounds"]["python_pivot_iterations"] == 6
    assert scaling["complexity_bounds"]["python_source_iterations_per_pivot"] == 0
    assert scaling["complexity_bounds"]["pivot_score_elements"] < 6 * source_paths
    assert scaling["fallback"]["used"] is False
    assert scaling["rank_exhaustion"]["occurred"] is True
    assert scaling["rank_exhaustion"]["fallback_pivot_count"] > 0
    assert result["report"]["coverage"]["all_source_paths_mapped"] is True
    assert torch.isfinite(result["aggregate_output_coefficients"]).all()


def test_discarded_paths_change_aggregate_output_coefficients() -> None:
    rows = 12
    activation = torch.linspace(0.25, 2.0, rows, dtype=torch.float64)
    features = torch.stack([activation, 0.5 * activation], dim=1)
    directions = torch.tensor([[1.0], [4.0]], dtype=torch.float64)
    common = {
        "prompt_ids": [f"p-{index}" for index in range(rows)],
        "domain_labels": ["a" if index % 2 == 0 else "b" for index in range(rows)],
        "donor_layer_ids": [3, 3],
        "target_path_count": 1,
        "ridge": 1e-12,
        "split_seed": 4,
        "selection_sketch_size": 1024,
    }

    folded = compress_post_nonlinearity_paths(features, directions, **common)
    without_discarded_direction = compress_post_nonlinearity_paths(
        features,
        torch.tensor([[0.0], [4.0]], dtype=torch.float64),
        **common,
    )

    assert folded["selected_path_indices"].tolist() == [1]
    assert without_discarded_direction["selected_path_indices"].tolist() == [1]
    assert folded["selected_native_output_directions"].item() == pytest.approx(4.0)
    assert folded["aggregate_output_coefficients"].item() == pytest.approx(
        6.0, rel=1e-9, abs=1e-9
    )
    assert without_discarded_direction[
        "aggregate_output_coefficients"
    ].item() == pytest.approx(4.0, rel=1e-9, abs=1e-9)
    assert folded["source_to_selected_coefficients"][0, 0].item() == pytest.approx(
        2.0, rel=1e-9, abs=1e-9
    )
    assert folded["source_path_to_selected_path"].tolist() == [1, 1]
    fit_report = folded["report"]["coefficient_fit"]
    assert fit_report["every_source_path_entered_response_target"] is True
    assert fit_report["source_paths_with_nonzero_fitted_output_contribution"] == 2


@pytest.mark.parametrize("interpolation_mode", ["dense", "sparse-signature"])
def test_layer_quotas_and_domain_diversity_are_enforced(
    interpolation_mode: str,
) -> None:
    rows = 16
    features = torch.zeros((rows, 4), dtype=torch.float64)
    features[:8, 0] = torch.linspace(1.0, 2.0, 8)
    features[:8, 2] = torch.linspace(0.2, 0.4, 8)
    features[8:, 1] = torch.linspace(0.2, 0.4, 8)
    features[8:, 3] = torch.linspace(1.0, 2.0, 8)
    directions = torch.ones((4, 1), dtype=torch.float64)
    domains = ["red"] * 8 + ["blue"] * 8

    result = compress_post_nonlinearity_paths(
        features,
        directions,
        prompt_ids=[f"prompt-{index}" for index in range(rows)],
        domain_labels=domains,
        donor_layer_ids=[0, 0, 1, 1],
        target_path_count=2,
        minimum_paths_per_layer={0: 1, 1: 1},
        minimum_paths_per_domain={"red": 1, "blue": 1},
        split_seed=9,
        selection_sketch_size=1024,
        interpolation_mode=interpolation_mode,
    )

    assert set(result["selected_layer_ids"]) == {0, 1}
    assert set(result["selected_dominant_domains"]) == {"red", "blue"}
    assert result["report"]["diversity"]["selected_dominant_domain_count"] == 2
    assert all(
        item["quota_satisfied"] for item in result["report"]["coverage"]["layers"]
    )
    assert all(
        item["quota_satisfied"] for item in result["report"]["coverage"]["domains"]
    )


def test_deterministic_fold_improves_prompt_held_out_response() -> None:
    first = _compress_duplicate_problem()
    second = _compress_duplicate_problem()

    assert torch.equal(
        first["selected_path_indices"], second["selected_path_indices"]
    )
    assert torch.equal(
        first["aggregate_output_coefficients"],
        second["aggregate_output_coefficients"],
    )
    assert torch.equal(
        first["source_to_selected_coefficients"],
        second["source_to_selected_coefficients"],
    )
    assert first["report"]["hashes"] == second["report"]["hashes"]

    held_out = first["report"]["diagnostics"]["held_out"]
    assert held_out["unit"] == "prompt"
    assert held_out["improved_absolute_rmse"] is True
    assert held_out["fitted"]["absolute_rmse"] < 1e-9
    assert held_out["baseline"]["absolute_rmse"] > 0.1
    assert held_out["fitted"]["cosine_similarity_mean"] >= held_out["baseline"][
        "cosine_similarity_mean"
    ]
    assert len(held_out["by_prompt"]) > 0
    assert {item["domain"] for item in held_out["by_domain"]} == {"math", "code"}


def test_vector_attention_paths_keep_post_softmax_channels() -> None:
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(14, 3, 2, generator=generator, dtype=torch.float64)
    directions = torch.randn(3, 2, 4, generator=generator, dtype=torch.float64)

    result = compress_post_nonlinearity_paths(
        features,
        directions,
        prompt_ids=[f"attention-{index}" for index in range(14)],
        domain_labels=["vision" if index % 2 else "text" for index in range(14)],
        donor_layer_ids=[2, 4, 6],
        target_path_count=2,
        minimum_paths_per_layer={2: 1, 6: 1},
        path_kind="attention",
        split_seed=22,
        selection_sketch_size=8,
    )

    assert result["aggregate_output_coefficients"].shape == (2, 2, 4)
    assert result["source_to_selected_coefficients"].shape == (3, 2, 2, 2)
    assert result["selected_native_output_directions"].shape == (2, 2, 4)
    provenance = result["report"]["provenance"]
    assert provenance["input_stage"] == "post-softmax-attention-head-response"
    assert provenance["qk_averaged_before_softmax"] is False
    assert provenance["gate_up_averaged_before_silu"] is False
    assert result["report"]["selection"]["sketch"]["exact"] is False

    sparse = compress_post_nonlinearity_paths(
        features,
        directions,
        prompt_ids=[f"attention-{index}" for index in range(14)],
        domain_labels=["vision" if index % 2 else "text" for index in range(14)],
        donor_layer_ids=[2, 4, 6],
        target_path_count=2,
        minimum_paths_per_layer={2: 1, 6: 1},
        path_kind="attention",
        split_seed=22,
        selection_sketch_size=8,
        interpolation_mode="sparse-signature",
    )
    assert sparse["aggregate_output_coefficients"].shape == (2, 2, 4)
    assert sparse["source_to_selected_coefficients"].shape == (3,)
    assert sparse["source_path_to_selected_path"].shape == (3,)
    assert torch.isfinite(sparse["aggregate_output_coefficients"]).all()
    assert sparse["report"]["coverage"]["all_source_paths_mapped"] is True


@pytest.mark.parametrize("interpolation_mode", ["dense", "sparse-signature"])
def test_preserves_inputs_metadata_and_existing_gradients(
    interpolation_mode: str,
) -> None:
    features, directions, prompts, domains, layers = _duplicate_response_problem(
        dtype=torch.float32
    )
    features = features.requires_grad_(True)
    directions = directions.requires_grad_(True)
    path_weights = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, requires_grad=True
    )
    features.grad = torch.full_like(features, 3.0)
    directions.grad = torch.full_like(directions, 5.0)
    path_weights.grad = torch.full_like(path_weights, 7.0)
    tensors = [features, directions, path_weights]
    values_before = [tensor.detach().clone() for tensor in tensors]
    gradients_before = [tensor.grad.detach().clone() for tensor in tensors]
    metadata_before = copy.deepcopy([prompts, domains, layers])

    result = compress_post_nonlinearity_paths(
        features,
        directions,
        prompt_ids=prompts,
        domain_labels=domains,
        donor_layer_ids=layers,
        target_path_count=2,
        path_weights=path_weights,
        split_seed=31,
        selection_sketch_size=4096,
        interpolation_mode=interpolation_mode,
    )

    for tensor, value, gradient in zip(
        tensors, values_before, gradients_before, strict=True
    ):
        assert torch.equal(tensor.detach(), value)
        assert torch.equal(tensor.grad, gradient)
    assert [prompts, domains, layers] == metadata_before
    assert result["aggregate_output_coefficients"].requires_grad is False
    assert result["source_to_selected_coefficients"].requires_grad is False
    report = result["report"]
    assert report["zero_gradient"] is True
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["autograd_enabled_during_fit"] is False
    assert report["inputs_mutated"] is False
    assert report["input_gradients_mutated"] is False
    assert report["hashes"]["inputs_before"] == report["hashes"]["inputs_after"]
    assert report["hashes"]["input_gradients_before"] == report["hashes"][
        "input_gradients_after"
    ]
    assert len(report["hashes"]["aggregate_output_coefficients_sha256"]) == 64


def test_rejects_invalid_inputs_and_infeasible_constraints() -> None:
    features, directions, prompts, domains, layers = _duplicate_response_problem()
    common = {
        "prompt_ids": prompts,
        "domain_labels": domains,
        "donor_layer_ids": layers,
        "target_path_count": 2,
    }

    with pytest.raises(ValueError, match="path counts must match"):
        compress_post_nonlinearity_paths(features, directions[:-1], **common)
    with pytest.raises(ValueError, match="cannot exceed"):
        compress_post_nonlinearity_paths(
            features, directions, **{**common, "target_path_count": 5}
        )
    with pytest.raises(ValueError, match="Layer minimum quotas exceed"):
        compress_post_nonlinearity_paths(
            features,
            directions,
            **{
                **common,
                "minimum_paths_per_layer": {0: 2, 1: 2},
            },
        )

    nonfinite = features.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compress_post_nonlinearity_paths(nonfinite, directions, **common)

    with pytest.raises(ValueError, match="interpolation_mode"):
        compress_post_nonlinearity_paths(
            features,
            directions,
            **{**common, "interpolation_mode": "unsupported"},
        )

    repeated_prompts = prompts.copy()
    repeated_prompts[1] = repeated_prompts[0]
    mixed_domains = domains.copy()
    mixed_domains[0] = "one"
    mixed_domains[1] = "two"
    with pytest.raises(ValueError, match="same domain"):
        compress_post_nonlinearity_paths(
            features,
            directions,
            **{
                **common,
                "prompt_ids": repeated_prompts,
                "domain_labels": mixed_domains,
            },
        )
