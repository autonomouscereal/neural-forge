from __future__ import annotations

import json
import math

import pytest

from neural_forge.hf_qwen_span_costs import (
    estimate_qwen_span_costs,
    plan_qwen_activation_spans,
)


def _aligned_profile(torch, initial, deltas, *, requires_grad: bool = False):
    inputs = {}
    outputs = {}
    state = initial.detach().clone()
    for layer, delta in enumerate(deltas):
        input_rows = state.detach().clone()
        output_rows = (state + delta).detach().clone()
        if requires_grad:
            input_rows.requires_grad_(True)
            output_rows.requires_grad_(True)
            input_rows.grad = torch.full_like(input_rows, float(layer + 1))
            output_rows.grad = torch.full_like(output_rows, float(layer + 11))
        inputs[layer] = input_rows
        outputs[layer] = output_rows
        state = output_rows.detach()
    return {
        "hidden_basis_input_rows": inputs,
        "hidden_basis_output_rows": outputs,
    }


def _assert_finite(value) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        assert math.isfinite(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite(item)
        return
    raise AssertionError(f"Unexpected diagnostic type: {type(value)!r}")


def test_span_costs_are_complete_finite_and_deterministic() -> None:
    torch = pytest.importorskip("torch")
    initial = torch.tensor(
        [
            [4.0, 3.0, 2.0],
            [3.0, 4.0, 2.0],
            [4.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )
    redundant = torch.tensor(
        [
            [0.10, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [-0.10, 0.0, 0.0],
            [0.05, 0.0, 0.0],
        ]
    )
    profile = _aligned_profile(
        torch,
        initial,
        [
            torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                    [2.0, 2.0, 0.0],
                ]
            ),
            redundant,
            redundant * 0.5,
            torch.tensor(
                [
                    [0.0, 0.30, 0.0],
                    [0.0, -0.20, 0.0],
                    [0.0, 0.10, 0.0],
                    [0.0, 0.20, 0.0],
                ]
            ),
        ],
        requires_grad=True,
    )
    values_before = {
        side: {layer: tensor.detach().clone() for layer, tensor in rows.items()}
        for side, rows in profile.items()
    }
    gradients_before = {
        side: {layer: tensor.grad.clone() for layer, tensor in rows.items()}
        for side, rows in profile.items()
    }

    first = estimate_qwen_span_costs(
        profile,
        target_hidden_width=1,
        max_span_length=2,
        critical_quantile=0.75,
    )
    second = estimate_qwen_span_costs(
        profile,
        target_hidden_width=1,
        max_span_length=2,
        critical_quantile=0.75,
    )

    assert set(first["span_costs"]) == {
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
    }
    assert all(first["span_costs"][(layer, layer + 1)] == 0.0 for layer in range(4))
    redundant_pair = next(
        item for item in first["span_diagnostics"] if item["half_open_span"] == [1, 3]
    )
    assert redundant_pair["consecutive_delta_cosine"] == pytest.approx(1.0)
    assert redundant_pair["consecutive_delta_incompatibility"] == pytest.approx(0.0)
    assert redundant_pair["rank_reconstruction"][
        "retained_energy_fraction"
    ] == pytest.approx(1.0)
    assert redundant_pair["rank_reconstruction"][
        "reconstruction_loss_fraction"
    ] == pytest.approx(0.0, abs=1e-12)
    assert first["hashes"] == second["hashes"]
    assert first["report"]["diagnostics"]["autograd_graph_constructed"] is False
    assert first["report"]["diagnostics"]["input_tensors_required_grad"] is True
    json.dumps(first["report"], allow_nan=False, sort_keys=True)
    _assert_finite(first["report"])

    for side, rows in profile.items():
        for layer, tensor in rows.items():
            assert torch.equal(tensor.detach(), values_before[side][layer])
            assert torch.equal(tensor.grad, gradients_before[side][layer])


def test_planner_keeps_high_critical_layer_single_and_pairs_redundant_layers() -> None:
    torch = pytest.importorskip("torch")
    initial = torch.tensor(
        [
            [8.0, 7.0, 6.0],
            [7.0, 8.0, 6.0],
            [8.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
        ]
    )
    redundant = torch.tensor(
        [
            [0.08, 0.0, 0.0],
            [0.16, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
            [0.04, 0.0, 0.0],
        ]
    )
    profile = _aligned_profile(
        torch,
        initial,
        [
            torch.tensor(
                [
                    [6.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 0.0, 6.0],
                    [3.0, 3.0, 0.0],
                ]
            ),
            redundant,
            redundant * 1.05,
            torch.tensor(
                [
                    [0.0, 0.45, 0.0],
                    [0.0, -0.30, 0.0],
                    [0.0, 0.20, 0.0],
                    [0.0, 0.35, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, 0.50],
                    [0.0, 0.0, 0.35],
                    [0.0, 0.0, -0.25],
                    [0.0, 0.0, 0.20],
                ]
            ),
        ],
    )

    result = plan_qwen_activation_spans(
        profile,
        target_hidden_width=1,
        target_layer_count=4,
        max_span_length=2,
        critical_quantile=0.8,
    )

    assert result["critical_layers"] == [0]
    assert result["span_ranges"] == [[0, 1], [1, 3], [3, 4], [4, 5]]
    assert result["donor_layer_spans"] == [[0], [1, 2], [3], [4]]
    assert (
        next(span for span in result["spans"] if 0 in span["donor_layers"])["length"]
        == 1
    )
    costs = result["span_cost_estimate"]["span_costs"]
    assert costs[(1, 3)] < costs[(0, 2)]
    assert costs[(1, 3)] < costs[(2, 4)]
    assert costs[(1, 3)] < costs[(3, 5)]
    assert result["report"]["cost_provenance"]["resolved_layout"] == "half-open"
    assert result["report"]["constraints"]["maximum_span_length"] == 2
    assert (
        result["planning_provenance"]["planner_plan_sha256"]
        == result["report"]["hashes"]["plan_sha256"]
    )


def test_zero_residual_rows_keep_all_diagnostics_finite() -> None:
    torch = pytest.importorskip("torch")
    zeros = torch.zeros((3, 4), dtype=torch.float32)
    profile = _aligned_profile(torch, zeros, [zeros.clone(), zeros.clone()])

    result = estimate_qwen_span_costs(
        profile,
        target_hidden_width=2,
        critical_quantile=0.5,
    )

    pair = next(
        item for item in result["span_diagnostics"] if item["half_open_span"] == [0, 2]
    )
    assert pair["consecutive_delta_cosine"] == 1.0
    assert pair["consecutive_delta_cosine_defined"] is False
    assert pair["rank_reconstruction"]["retained_energy_fraction"] == 1.0
    assert pair["rank_reconstruction"]["reconstruction_loss_fraction"] == 0.0
    assert result["span_costs"][(0, 2)] == 0.0
    _assert_finite(result["report"])
