from __future__ import annotations

import json

import pytest

from neural_forge.hf_qwen_observable import apply_chart_transition
from neural_forge.hf_qwen_observable import compose_chart_transitions
from neural_forge.hf_qwen_observable import encode_observable_chart
from neural_forge.hf_qwen_observable import fit_logit_observable_charts
from neural_forge.hf_qwen_observable import predict_future_observable


torch = pytest.importorskip("torch")


def _latent_problem(*, dtype=torch.float64):
    generator = torch.Generator().manual_seed(811)
    prompt_count = 80
    rows_per_prompt = 2
    row_count = prompt_count * rows_per_prompt
    latent = torch.randn(row_count, 3, generator=generator, dtype=dtype)
    observable_head = torch.randn(3, 7, generator=generator, dtype=dtype)
    observable = latent.matmul(observable_head)

    nuisance_0 = 30.0 * torch.randn(
        row_count, 9, generator=generator, dtype=dtype
    )
    nuisance_1 = 20.0 * torch.randn(
        row_count, 6, generator=generator, dtype=dtype
    )
    mix_1 = torch.tensor(
        [[1.2, -0.3, 0.5], [0.1, 0.9, -0.4], [-0.2, 0.6, 1.1]],
        dtype=dtype,
    )
    mix_2 = torch.tensor(
        [[0.8, 0.2, -0.1], [-0.4, 1.3, 0.3], [0.5, -0.2, 0.9]],
        dtype=dtype,
    )
    boundary_0 = torch.cat([nuisance_0, 0.1 * latent], dim=1)
    boundary_1 = torch.cat([nuisance_1, 0.2 * latent.matmul(mix_1)], dim=1)
    boundary_2 = 0.3 * latent.matmul(mix_2)
    prompt_ids = [f"prompt-{index // rows_per_prompt}" for index in range(row_count)]
    domains = [
        "math" if (index // rows_per_prompt) % 2 == 0 else "code"
        for index in range(row_count)
    ]
    return {
        "boundaries": {
            "embed": boundary_0,
            "middle": boundary_1,
            "final": boundary_2,
        },
        "observable": observable,
        "prompt_ids": prompt_ids,
        "domains": domains,
    }


def _activation_pca_prediction(
    train_x,
    train_z,
    validation_x,
    *,
    rank: int,
):
    x_mean = train_x.mean(dim=0)
    z_mean = train_z.mean(dim=0)
    _, _, right = torch.linalg.svd(train_x - x_mean, full_matrices=False)
    basis = right[:rank].t()
    train_coordinates = (train_x - x_mean).matmul(basis)
    decoder = torch.linalg.lstsq(
        train_coordinates, train_z - z_mean
    ).solution
    return (validation_x - x_mean).matmul(basis).matmul(decoder) + z_mean


def test_future_observable_chart_beats_activation_pca_on_low_variance_signal() -> None:
    problem = _latent_problem()
    result = fit_logit_observable_charts(
        problem["boundaries"],
        problem["observable"],
        rank=3,
        prompt_ids=problem["prompt_ids"],
        domains=problem["domains"],
        domain_weights={"math": 2.0, "code": 0.5},
        ridge=1e-10,
        split_seed=19,
    )

    train = result["split"]["train_indices"]
    validation = result["split"]["validation_indices"]
    boundary = problem["boundaries"]["embed"]
    observable = problem["observable"]
    prediction = predict_future_observable(
        boundary.index_select(0, validation), result["charts"][0]
    )
    pca_prediction = _activation_pca_prediction(
        boundary.index_select(0, train),
        observable.index_select(0, train),
        boundary.index_select(0, validation),
        rank=3,
    )
    actual = observable.index_select(0, validation)
    observable_rmse = (prediction - actual).square().mean().sqrt().item()
    pca_rmse = (pca_prediction - actual).square().mean().sqrt().item()

    assert observable_rmse < 0.01 * pca_rmse
    held_out = result["report"]["charts"][0]["held_out"]["overall"]
    assert held_out["future_observable_cosine"] > 0.999
    assert held_out["future_observable_variance_explained"] > 0.999
    assert result["report"]["weighting"]["domain_weights"] == {
        "code": 0.5,
        "math": 2.0,
    }
    train_prompts = set(result["report"]["split"]["train_prompt_ids"])
    validation_prompts = set(
        result["report"]["split"]["validation_prompt_ids"]
    )
    assert train_prompts.isdisjoint(validation_prompts)


def test_rank_truncation_and_cca_fit_are_deterministic() -> None:
    generator = torch.Generator().manual_seed(91)
    latent = torch.randn(96, 2, generator=generator, dtype=torch.float64)
    x = latent.matmul(
        torch.randn(2, 8, generator=generator, dtype=torch.float64)
    )
    z = latent.matmul(
        torch.randn(2, 6, generator=generator, dtype=torch.float64)
    )
    prompt_ids = [f"p-{index // 2}" for index in range(96)]

    first = fit_logit_observable_charts(
        [x],
        z,
        rank=7,
        prompt_ids=prompt_ids,
        method="cca",
        ridge=1e-10,
        rank_tolerance=1e-9,
        split_seed=5,
    )
    second = fit_logit_observable_charts(
        [x],
        z,
        rank=7,
        prompt_ids=prompt_ids,
        method="cca",
        ridge=1e-10,
        rank_tolerance=1e-9,
        split_seed=5,
    )

    chart = first["charts"][0]
    assert chart["retained_rank"] == 2
    assert torch.equal(chart["encoder"], second["charts"][0]["encoder"])
    assert torch.equal(chart["decoder"], second["charts"][0]["decoder"])
    assert first["report"]["split"] == second["report"]["split"]
    assert chart["report"]["retained_correlation_energy_fraction"] == pytest.approx(
        1.0
    )
    assert chart["report"]["held_out"]["overall"][
        "future_observable_cosine"
    ] > 0.999


def test_materialized_transition_composition_matches_sequential_application() -> None:
    problem = _latent_problem()
    result = fit_logit_observable_charts(
        problem["boundaries"],
        problem["observable"],
        rank=3,
        prompt_ids=problem["prompt_ids"],
        ridge=1e-10,
        transition_ridge=1e-12,
        split_seed=31,
    )
    validation = result["split"]["validation_indices"]
    first_rows = problem["boundaries"]["embed"].index_select(0, validation)
    first_coordinates = encode_observable_chart(first_rows, result["charts"][0])
    sequential = first_coordinates
    for transition in result["transitions"]:
        sequential = apply_chart_transition(sequential, transition)
    composed = compose_chart_transitions(result["transitions"])
    direct = apply_chart_transition(first_coordinates, composed)

    assert len(result["transitions"]) == 2
    assert composed["source"] == "embed"
    assert composed["target"] == "final"
    assert composed["composed_transition_count"] == 2
    assert torch.allclose(direct, sequential, atol=1e-12, rtol=1e-12)
    assert result["report"]["transitions"][0]["held_out"][
        "future_observable_cosine"
    ] > 0.99


def test_fit_preserves_values_and_existing_gradients() -> None:
    problem = _latent_problem(dtype=torch.float32)
    inputs = [
        value.detach().clone().requires_grad_(True)
        for value in problem["boundaries"].values()
    ]
    observable = problem["observable"].detach().clone().requires_grad_(True)
    all_inputs = [*inputs, observable]
    for index, tensor in enumerate(all_inputs):
        tensor.grad = torch.full_like(tensor, float(index + 1))
    values_before = [tensor.detach().clone() for tensor in all_inputs]
    gradients_before = [tensor.grad.detach().clone() for tensor in all_inputs]

    result = fit_logit_observable_charts(
        inputs,
        observable,
        rank=3,
        boundary_names=["a", "b", "c"],
        prompt_ids=problem["prompt_ids"],
        ridge=1e-7,
    )

    for tensor, value_before, gradient_before in zip(
        all_inputs, values_before, gradients_before, strict=True
    ):
        assert torch.equal(tensor.detach(), value_before)
        assert torch.equal(tensor.grad, gradient_before)
    for chart in result["charts"]:
        assert chart["encoder"].requires_grad is False
        assert chart["decoder"].requires_grad is False
    for transition in result["transitions"]:
        assert transition["matrix"].requires_grad is False
        assert transition["bias"].requires_grad is False
    report = result["report"]
    assert report["zero_gradient"] is True
    assert report["closed_form"] is True
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["autograd_enabled_during_fit"] is False
    assert report["inputs_mutated"] is False
    assert report["input_gradients_mutated"] is False
    assert report["hashes"]["inputs_before"] == report["hashes"][
        "inputs_after"
    ]
    assert report["hashes"]["input_gradients_before"] == report["hashes"][
        "input_gradients_after"
    ]
    json.dumps(report, allow_nan=False)


def test_rejects_invalid_shapes_nonfinite_values_and_split_metadata() -> None:
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(24, 5, generator=generator)
    z = torch.randn(24, 4, generator=generator)
    prompt_ids = [f"p-{index}" for index in range(24)]

    with pytest.raises(ValueError, match="same row count"):
        fit_logit_observable_charts([x, x[:-1]], z, rank=2)
    with pytest.raises(ValueError, match="same row count"):
        fit_logit_observable_charts([x], z[:-1], rank=2)
    bad = x.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_logit_observable_charts([bad], z, rank=2)
    with pytest.raises(ValueError, match="prompt_ids length"):
        fit_logit_observable_charts(
            [x], z, rank=2, prompt_ids=prompt_ids[:-1]
        )
    with pytest.raises(ValueError, match="domains length"):
        fit_logit_observable_charts(
            [x], z, rank=2, prompt_ids=prompt_ids, domains=["math"]
        )
    with pytest.raises(ValueError, match="floating-point"):
        fit_logit_observable_charts([torch.ones(24, 5, dtype=torch.long)], z, rank=2)
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        fit_logit_observable_charts([x], z, rank=0)
    with pytest.raises(ValueError, match="method must be rrr or cca"):
        fit_logit_observable_charts([x], z, rank=2, method="pca")
