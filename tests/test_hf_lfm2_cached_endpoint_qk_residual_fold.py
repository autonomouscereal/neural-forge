from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold as endpoint_fold  # noqa: E402
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (  # noqa: E402
    Lfm2EndpointQkResidualAuthenticationError,
    Lfm2EndpointQkResidualCase,
    Lfm2EndpointQkResidualFoldPolicy,
    build_lfm2_endpoint_qkvo_seed,
    compile_lfm2_cached_endpoint_qk_fit_proposal,
    compile_lfm2_cached_endpoint_qk_residual_fold,
    replay_lfm2_endpoint_qkvo,
    seal_lfm2_endpoint_qk_fit_bank,
    seal_lfm2_endpoint_qk_fit_check_bank,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rmsnorm(value: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    direction = value.float()
    normalized = direction * torch.rsqrt(
        direction.square().mean(dim=-1, keepdim=True) + 1e-5
    )
    return normalized.to(torch.bfloat16) * gain


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _teacher_outputs(
    query_inputs: torch.Tensor,
    key_inputs: torch.Tensor,
    *,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    query_cosine: torch.Tensor,
    query_sine: torch.Tensor,
    key_cosine: torch.Tensor,
    key_sine: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, query_length, _hidden = query_inputs.shape
    key_length = key_inputs.shape[1]
    query = torch.nn.functional.linear(query_inputs, query_weight).view(
        batch,
        query_length,
        2,
        4,
    )
    key = torch.nn.functional.linear(key_inputs, key_weight).view(
        batch,
        key_length,
        1,
        4,
    )
    value = torch.nn.functional.linear(key_inputs, value_weight).view(
        batch,
        key_length,
        1,
        4,
    )
    query = _rmsnorm(query, norm_weight).transpose(1, 2)
    key = _rmsnorm(key, norm_weight).transpose(1, 2)
    value = value.transpose(1, 2).repeat_interleave(2, dim=1)
    query = query * query_cosine.unsqueeze(1) + _rotate_half(
        query
    ) * query_sine.unsqueeze(1)
    key = key * key_cosine.unsqueeze(1) + _rotate_half(key) * key_sine.unsqueeze(1)
    key = key.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores * float(4**-0.5)
    additive_mask = torch.zeros_like(scores).masked_fill(
        ~mask,
        torch.finfo(scores.dtype).min,
    )
    probabilities = torch.nn.functional.softmax(
        scores + additive_mask,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    context = probabilities @ value
    concatenated = (
        context.transpose(1, 2)
        .contiguous()
        .reshape(
            batch,
            query_length,
            8,
        )
        .contiguous()
    )
    endpoint = torch.nn.functional.linear(
        concatenated,
        output_weight,
    )
    return probabilities, endpoint


def _paired_parent_probabilities(
    target: torch.Tensor,
) -> torch.Tensor:
    """Create distinct parent routes whose arithmetic barycenter is `target`."""

    source_heads = []
    for head in range(target.shape[1]):
        positive = target[:, head].clone()
        negative = target[:, head].clone()
        for query_index in range(1, target.shape[2]):
            left = target[0, head, query_index, 0]
            right = target[0, head, query_index, query_index]
            shift = 0.2 * torch.minimum(left, right)
            positive[0, query_index, 0] += shift
            positive[0, query_index, query_index] -= shift
            negative[0, query_index, 0] -= shift
            negative[0, query_index, query_index] += shift
        source_heads.extend((positive, negative))
    return torch.stack(source_heads, dim=1)


def _make_fixture() -> dict[str, object]:
    generator = torch.Generator().manual_seed(20260720)
    query_teacher = (torch.randn(8, 8, generator=generator) * 0.45).to(torch.bfloat16)
    key_teacher = (torch.randn(4, 8, generator=generator) * 0.45).to(torch.bfloat16)
    value = (torch.randn(4, 8, generator=generator) * 0.55).to(torch.bfloat16)
    output = (torch.randn(8, 8, generator=generator) * 0.55).to(torch.bfloat16)
    norm = torch.ones(4, dtype=torch.bfloat16)
    query_seed = (
        query_teacher.float()
        + 0.14 * torch.randn(query_teacher.shape, generator=generator)
    ).to(torch.bfloat16)
    key_seed = (
        key_teacher.float() + 0.14 * torch.randn(key_teacher.shape, generator=generator)
    ).to(torch.bfloat16)
    source_query = torch.randn(
        (16, 16),
        generator=generator,
        dtype=torch.bfloat16,
    )
    source_key = torch.randn(
        (8, 16),
        generator=generator,
        dtype=torch.bfloat16,
    )
    source_value = torch.randn(
        (8, 16),
        generator=generator,
        dtype=torch.bfloat16,
    )
    source_output = torch.randn(
        (16, 16),
        generator=generator,
        dtype=torch.bfloat16,
    )
    source_inventory = endpoint_fold.seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=source_query,
        source_key_weight=source_key,
        source_value_weight=source_value,
        source_output_weight=source_output,
    )

    def make_case(index: int) -> Lfm2EndpointQkResidualCase:
        query_length = 6
        key_length = 6
        query_inputs = (torch.randn(1, query_length, 8, generator=generator) * 0.8).to(
            torch.bfloat16
        )
        key_inputs = (
            query_inputs.float()
            + 0.03 * torch.randn(query_inputs.shape, generator=generator)
        ).to(torch.bfloat16)
        position = torch.arange(query_length, dtype=torch.float32)
        frequency = torch.tensor([0.07, 0.19], dtype=torch.float32)
        phase = position[:, None] * frequency[None, :]
        cosine = torch.cat((phase.cos(), phase.cos()), dim=-1)[None].to(torch.bfloat16)
        sine = torch.cat((phase.sin(), phase.sin()), dim=-1)[None].to(torch.bfloat16)
        causal = (
            torch.arange(key_length)[None, None, None, :]
            <= torch.arange(query_length)[None, None, :, None]
        ).expand(1, 2, query_length, key_length)
        target_probabilities, target_endpoint = _teacher_outputs(
            query_inputs,
            key_inputs,
            query_weight=query_teacher,
            key_weight=key_teacher,
            value_weight=value,
            output_weight=output,
            norm_weight=norm,
            query_cosine=cosine,
            query_sine=sine,
            key_cosine=cosine,
            key_sine=sine,
            mask=causal,
        )
        target_probabilities = target_probabilities.to(torch.float64)
        target_probabilities = target_probabilities / target_probabilities.sum(
            dim=-1,
            keepdim=True,
        )
        parent_probabilities = _paired_parent_probabilities(target_probabilities)
        parent_mask = causal.repeat_interleave(2, dim=1)
        return Lfm2EndpointQkResidualCase(
            target_query_inputs=query_inputs,
            target_key_value_inputs=key_inputs,
            parent_probabilities=parent_probabilities,
            parent_causal_mask=parent_mask,
            parent_endpoint_target=target_endpoint,
            query_rope_cos=cosine,
            query_rope_sin=sine,
            key_rope_cos=cosine,
            key_rope_sin=sine,
            query_row_sqrt_weights=torch.ones(
                1,
                query_length,
                dtype=torch.float64,
            ),
            query_groups=((0, 1), (2, 3)),
            query_group_weights=((0.5, 0.5), (0.5, 0.5)),
            trajectory_sha256=_sha(f"trajectory-{index}"),
        )

    source_artifact = source_inventory.inventory_sha256
    split = _sha("synthetic-source-split")
    cases = tuple(make_case(index) for index in range(10))
    fit_bank = seal_lfm2_endpoint_qk_fit_bank(
        cases[:7],
        source_split_sha256=split,
        source_artifact_sha256=source_artifact,
    )
    fit_check_bank = seal_lfm2_endpoint_qk_fit_check_bank(
        cases[7:],
        source_split_sha256=split,
        source_artifact_sha256=source_artifact,
    )

    def make_seed(blank_seed: int):
        blank_generator = torch.Generator().manual_seed(blank_seed)
        return build_lfm2_endpoint_qkvo_seed(
            query_weight=query_seed,
            key_weight=key_seed,
            value_weight=value,
            output_weight=output,
            query_norm_weight=norm,
            key_norm_weight=norm,
            norm_epsilon=1e-5,
            source_query_weight=source_query,
            source_key_weight=source_key,
            source_value_weight=source_value,
            source_output_weight=source_output,
            source_artifact_sha256=source_artifact,
            target_query_weight_template=torch.randn(
                query_seed.shape,
                generator=blank_generator,
                dtype=torch.bfloat16,
            ),
            target_key_weight_template=torch.randn(
                key_seed.shape,
                generator=blank_generator,
                dtype=torch.bfloat16,
            ),
            target_value_weight_template=torch.randn(
                value.shape,
                generator=blank_generator,
                dtype=torch.bfloat16,
            ),
            target_output_weight_template=torch.randn(
                output.shape,
                generator=blank_generator,
                dtype=torch.bfloat16,
            ),
        )

    return {
        "fit_bank": fit_bank,
        "fit_check_bank": fit_check_bank,
        "seed_a": make_seed(11),
        "seed_b": make_seed(991),
        "first_case": cases[0],
        "source_query": source_query,
        "source_key": source_key,
        "source_value": source_value,
        "source_output": source_output,
    }


def _policy() -> Lfm2EndpointQkResidualFoldPolicy:
    return Lfm2EndpointQkResidualFoldPolicy(
        ridge=1e-4,
        route_objective_weight=0.2,
        trust_radius=1.0,
        maximum_sweeps=4,
        maximum_cg_iterations=512,
        minimum_cg_iterations=2,
        cg_restart_iterations=64,
        cg_relative_tolerance=5e-4,
        metric_regression_tolerance=1e-7,
        minimum_fit_endpoint_improvement=1e-7,
    )


def _one_iteration_policy() -> Lfm2EndpointQkResidualFoldPolicy:
    return replace(
        _policy(),
        maximum_cg_iterations=1,
        minimum_cg_iterations=1,
        cg_restart_iterations=1,
        cg_relative_tolerance=1e-12,
        authentication_sha256="",
    )


def _adversarial_routing_metric_tensors(
    mask_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, queries, keys = 2, 3, 5, 7
    mask = torch.zeros(batch, heads, queries, keys, dtype=torch.bool)
    if mask_kind == "causal":
        for batch_index in range(batch):
            for head in range(heads):
                for query_index in range(queries):
                    valid_count = max(
                        1,
                        query_index + 1 - ((batch_index + head) % 2),
                    )
                    mask[batch_index, head, query_index, :valid_count] = True
    elif mask_kind == "ragged":
        patterns = (
            (),
            (3,),
            (0, 6),
            (1, 3, 5),
            (0, 2, 4, 6),
            (0, 1, 3, 4, 6),
            (0, 1, 2, 3, 4, 5, 6),
        )
        for flat_index, row in enumerate(mask.reshape(-1, keys)):
            row[list(patterns[flat_index % len(patterns)])] = True
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unknown mask kind: {mask_kind}")

    element_count = batch * heads * queries * keys
    ordinal = torch.arange(element_count, dtype=torch.float64).reshape(
        batch,
        heads,
        queries,
        keys,
    )
    target_raw = torch.remainder(ordinal * 7.0 + 3.0, 19.0) + 1.0
    prediction_raw = torch.remainder(ordinal.flip(-1) * 11.0 + 5.0, 23.0) + 1.0
    target_raw = torch.where(mask, target_raw, torch.zeros_like(target_raw))
    prediction_raw = torch.where(
        mask,
        prediction_raw,
        torch.zeros_like(prediction_raw),
    )
    for flat_index, (target_row, prediction_row, mask_row) in enumerate(
        zip(
            target_raw.reshape(-1, keys),
            prediction_raw.reshape(-1, keys),
            mask.reshape(-1, keys),
            strict=True,
        )
    ):
        valid_indices = torch.nonzero(mask_row, as_tuple=False).flatten()
        if int(valid_indices.numel()) < 2:
            continue
        if flat_index % 2 == 0:
            tied_value = target_row.index_select(0, valid_indices).max() + 7.0
            target_row[valid_indices[0]] = tied_value
            target_row[valid_indices[1]] = tied_value
            prediction_row[valid_indices[1]] = (
                prediction_row.index_select(0, valid_indices).max() + 9.0
            )
        else:
            target_row[valid_indices[0]] = (
                target_row.index_select(0, valid_indices).max() + 7.0
            )
            prediction_row[valid_indices[-1]] = (
                prediction_row.index_select(0, valid_indices).max() + 9.0
            )
    target = target_raw / target_raw.sum(dim=-1, keepdim=True).clamp_min(1.0)
    prediction = prediction_raw / prediction_raw.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)
    row_weights = torch.tensor(
        (
            (0.125, 0.5, 1.75, 2.0, 3.25),
            (4.5, 0.375, 2.25, 1.125, 5.0),
        ),
        dtype=torch.float64,
    )
    return prediction, target, mask, row_weights


def _scalar_routing_metric_rows(
    prediction_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    mask: torch.Tensor,
    query_row_weights: torch.Tensor,
) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[float, ...]]:
    js_values = []
    top1_matches = []
    weights = []
    tiny = torch.finfo(torch.float64).tiny
    batch, heads, query_length, _keys = map(int, prediction_probabilities.shape)
    for batch_index in range(batch):
        for head in range(heads):
            for query_index in range(query_length):
                valid = mask[batch_index, head, query_index]
                if int(valid.sum().item()) < 2:
                    continue
                prediction = prediction_probabilities[
                    batch_index,
                    head,
                    query_index,
                    valid,
                ]
                target = target_probabilities[
                    batch_index,
                    head,
                    query_index,
                    valid,
                ]
                midpoint = 0.5 * (prediction + target)
                js = (
                    0.5
                    * (
                        (
                            target
                            * (
                                torch.log(target.clamp_min(tiny))
                                - torch.log(midpoint.clamp_min(tiny))
                            )
                        ).sum()
                        + (
                            prediction
                            * (
                                torch.log(prediction.clamp_min(tiny))
                                - torch.log(midpoint.clamp_min(tiny))
                            )
                        ).sum()
                    )
                    / math.log(2.0)
                )
                target_winners = torch.isclose(
                    target,
                    target.max(),
                    rtol=1e-10,
                    atol=1e-12,
                )
                js_values.append(float(js.clamp_min(0.0).item()))
                top1_matches.append(
                    bool(target_winners[int(prediction.argmax().item())].item())
                )
                weights.append(float(query_row_weights[batch_index, query_index]))
    return tuple(js_values), tuple(top1_matches), tuple(weights)


@pytest.mark.parametrize("mask_kind", ("causal", "ragged"))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ),
)
def test_grouped_routing_metrics_exactly_match_scalar_reference(
    mask_kind: str,
    device: str,
) -> None:
    prediction, target, mask, row_weights = _adversarial_routing_metric_tensors(
        mask_kind
    )
    prediction = prediction.to(device)
    target = target.to(device)
    mask = mask.to(device)
    row_weights = row_weights.to(device)
    expected_rows = _scalar_routing_metric_rows(
        prediction,
        target,
        mask,
        row_weights,
    )
    actual_rows = endpoint_fold._MetricAccumulator._row_routing_metrics(
        prediction,
        target,
        mask,
        row_weights,
    )
    assert actual_rows == expected_rows
    assert any(actual_rows[1])
    assert not all(actual_rows[1])

    target_endpoint = torch.linspace(
        -0.75,
        0.9,
        prediction.shape[0] * prediction.shape[2] * 4,
        dtype=torch.float64,
        device=device,
    ).reshape(prediction.shape[0], prediction.shape[2], 4)
    proposed_endpoint = target_endpoint + 0.03125
    accumulator = endpoint_fold._MetricAccumulator()
    accumulator.add(
        SimpleNamespace(
            query_row_weights=row_weights,
            target_endpoint=target_endpoint,
            target_probabilities=target,
            mask=mask,
        ),
        SimpleNamespace(
            endpoint=proposed_endpoint,
            probabilities=prediction,
        ),
    )
    actual_metrics = accumulator.finish()
    endpoint_error = float(
        (row_weights.unsqueeze(-1) * (proposed_endpoint - target_endpoint).square())
        .sum()
        .item()
    )
    endpoint_energy = float(
        (row_weights.unsqueeze(-1) * target_endpoint.square()).sum().item()
    )
    probability_weights = row_weights[:, None, :, None]
    probability_error = float(
        (probability_weights * (prediction - target).square()).sum().item()
    )
    probability_energy = float((probability_weights * target.square()).sum().item())
    expected_js_sum = 0.0
    expected_js_weight = 0.0
    expected_top1 = 0.0
    for js_value, matched, weight in zip(*expected_rows, strict=True):
        expected_js_sum += weight * js_value
        expected_js_weight += weight
        expected_top1 += weight * float(matched)
    expected_metrics = endpoint_fold.Lfm2EndpointQkMetrics(
        case_count=1,
        eligible_route_rows=len(expected_rows[0]),
        endpoint_relative_rmse=math.sqrt(endpoint_error / endpoint_energy),
        probability_relative_rmse=math.sqrt(probability_error / probability_energy),
        js_divergence_mean=expected_js_sum / expected_js_weight,
        js_divergence_worst=max(expected_rows[0]),
        top1_route_agreement_fraction=expected_top1 / expected_js_weight,
        maximum_causal_probability_leakage=0.0,
    )
    assert actual_metrics == expected_metrics


@pytest.fixture(scope="module")
def fit_proposal_fixture() -> tuple[dict[str, object], object]:
    fixture = _make_fixture()
    proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        policy=_policy(),
    )
    return fixture, proposal


def test_endpoint_jvp_and_vjp_are_adjoint() -> None:
    fixture = _make_fixture()
    seed = fixture["seed_a"]
    case = fixture["first_case"]
    assert isinstance(seed, endpoint_fold.Lfm2EndpointQkvoSeed)
    assert isinstance(case, Lfm2EndpointQkResidualCase)
    prepared = endpoint_fold._prepare_case(case, seed, device=torch.device("cpu"))
    endpoint_fold._attach_replay(
        (prepared,),
        seed,
        seed.query_weight,
        seed.key_weight,
    )
    operator = endpoint_fold.Lfm2EndpointQkLinearOperator(
        seed=seed,
        prepared_cases=(prepared,),
        current_query_weight=seed.query_weight.float(),
        current_key_weight=seed.key_weight.float(),
        policy=_policy(),
    )
    generator = torch.Generator().manual_seed(101)
    vector = torch.randn(
        seed.query_weight.numel() + seed.key_weight.numel(),
        generator=generator,
    )
    delta_probability, delta_endpoint = operator.jvp(vector)[0]
    probability_cotangent = torch.randn(
        delta_probability.shape,
        generator=generator,
        dtype=torch.float64,
    )
    endpoint_cotangent = torch.randn(
        delta_endpoint.shape,
        generator=generator,
        dtype=torch.float64,
    )
    left = (delta_probability * probability_cotangent).sum() + (
        delta_endpoint * endpoint_cotangent
    ).sum()
    right = (
        vector.double()
        * operator.vjp(((endpoint_cotangent, probability_cotangent),)).double()
    ).sum()
    assert torch.allclose(left, right, rtol=2e-4, atol=2e-4)


def test_target_is_direct_parent_probability_barycenter_not_logit_average() -> None:
    case = _make_fixture()["first_case"]
    assert isinstance(case, Lfm2EndpointQkResidualCase)
    parent_pair = case.parent_probabilities[0, :2, -1]
    arithmetic = parent_pair.mean(dim=0)
    valid = case.target_causal_mask[0, 0, -1]
    geometric = torch.sqrt(parent_pair[0, valid] * parent_pair[1, valid])
    geometric = geometric / geometric.sum()

    assert torch.allclose(
        case.target_probability_barycenter[0, 0, -1],
        arithmetic,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not torch.allclose(
        case.target_probability_barycenter[0, 0, -1, valid],
        geometric,
        rtol=1e-8,
        atol=1e-10,
    )
    assert case.target_receipt["probability_target_is_mean_of_parent_probabilities"]
    assert not case.target_receipt[
        "probability_target_is_softmax_of_mean_parent_logits"
    ]


def test_finite_one_iteration_cg_direction_enters_exact_structural_frontier() -> None:
    fixture = _make_fixture()
    proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        policy=_one_iteration_policy(),
    )

    assert proposal.changed
    assert proposal.proposed_fit_metrics.endpoint_relative_rmse < (
        proposal.initial_fit_metrics.endpoint_relative_rmse
    )
    assert endpoint_fold._route_nonregressing(
        proposal.proposed_fit_metrics,
        proposal.initial_fit_metrics,
        proposal.policy.metric_regression_tolerance,
    )
    assert proposal.report["fit_candidate_eligibility_rule"] == (
        "finite_shape_correct_effectful_exact_bfloat16_state"
    )
    assert proposal.report["fit_numeric_metrics_can_veto_effectful_candidate"] is False
    assert (
        proposal.report["fit_numeric_metrics_have_final_intelligence_authority"]
        is False
    )
    for sweep in proposal.report["sweeps"]:
        solve = sweep["solve_report"]
        assert solve["converged"] is False
        assert solve["termination"] == "maximum_iterations"
        assert solve["usable_direction"] is True
        assert solve["direction_is_finite"] is True
        assert solve["direction_has_nonzero_entry"] is True
        selected = next(
            row for row in sweep["frontier"] if row["alpha"] == sweep["selected_alpha"]
        )
        assert sweep["selection_reason"] == (
            "best_effectful_structural_candidate_by_fit_metrics"
        )
        assert selected["candidate_finite"] is True
        assert selected["candidate_shape_correct"] is True
        assert selected["candidate_effectful"] is True
        assert selected["eligible"] is True
        assert isinstance(selected["fit_endpoint_improvement_proxy_passed"], bool)
        assert isinstance(selected["route_nonregression_proxy_passed"], bool)
        assert isinstance(selected["legacy_numeric_proxy_gate_passed"], bool)
        alpha_zero = sweep["frontier"][0]
        assert alpha_zero["alpha"] == 0.0
        assert alpha_zero["candidate_effectful"] is False
        assert alpha_zero["eligible"] is False
    assert proposal.verify() == proposal.report_sha256

    changed_semantics = json.loads(json.dumps(dict(proposal.report)))
    changed_semantics.pop("report_sha256")
    changed_semantics["sweeps"][0]["solve_report"]["usable_direction"] = False
    changed_semantics["report_sha256"] = endpoint_fold._canonical_sha256(
        changed_semantics
    )
    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="usable-direction semantics",
    ):
        replace(proposal, report=changed_semantics)


def test_exact_frontier_retains_a_finite_truncated_regressing_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture()

    def reversed_truncated_direction(
        operator: object,
        right_hand_side: torch.Tensor,
        policy: Lfm2EndpointQkResidualFoldPolicy,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        direction = -right_hand_side
        return direction, endpoint_fold._cg_solve_report(
            direction,
            converged=False,
            iterations=policy.maximum_cg_iterations,
            relative_normal_residual=1.0,
            termination="maximum_iterations",
            normal_operator_calls=operator.normal_operator_calls,
        )

    monkeypatch.setattr(endpoint_fold, "_ridge_cg", reversed_truncated_direction)
    proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        policy=replace(
            _one_iteration_policy(),
            maximum_sweeps=2,
            authentication_sha256="",
        ),
    )

    assert proposal.report["sweeps"][0]["solve_report"]["usable_direction"] is True
    assert proposal.report["sweeps"][0]["selected_alpha"] > 0.0
    assert proposal.report["sweeps"][0]["selection_reason"] == (
        "best_effectful_structural_candidate_by_fit_metrics"
    )
    selected = next(
        row
        for row in proposal.report["sweeps"][0]["frontier"]
        if row["alpha"] == proposal.report["sweeps"][0]["selected_alpha"]
    )
    assert selected["candidate_finite"] is True
    assert selected["candidate_shape_correct"] is True
    assert selected["candidate_effectful"] is True
    assert selected["eligible"] is True
    assert selected["legacy_numeric_proxy_gate_passed"] is False
    assert proposal.changed is True
    assert not (
        torch.equal(proposal.query_weight, proposal.seed.query_weight)
        and torch.equal(proposal.key_weight, proposal.seed.key_weight)
    )
    assert proposal.verify() == proposal.report_sha256


def test_alpha_zero_is_only_used_when_every_nonzero_bf16_state_is_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture()

    def zero_trust_direction(
        delta_query: torch.Tensor,
        delta_key: torch.Tensor,
        current_query: torch.Tensor,
        current_key: torch.Tensor,
        trust_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
        del current_query, current_key, trust_radius
        return (
            torch.zeros_like(delta_query),
            torch.zeros_like(delta_key),
            0.0,
            0.0,
            1.0,
        )

    monkeypatch.setattr(endpoint_fold, "_trust_scale", zero_trust_direction)
    proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        policy=replace(
            _one_iteration_policy(),
            maximum_sweeps=2,
            authentication_sha256="",
        ),
    )

    sweep = proposal.report["sweeps"][0]
    assert sweep["selected_alpha"] == 0.0
    assert sweep["selection_reason"] == "all_nonzero_candidates_identical_or_invalid"
    assert all(row["candidate_finite"] for row in sweep["frontier"])
    assert all(row["candidate_shape_correct"] for row in sweep["frontier"])
    assert all(not row["candidate_effectful"] for row in sweep["frontier"])
    assert all(not row["eligible"] for row in sweep["frontier"])
    assert proposal.changed is False
    assert torch.equal(proposal.query_weight, proposal.seed.query_weight)
    assert torch.equal(proposal.key_weight, proposal.seed.key_weight)
    assert proposal.verify() == proposal.report_sha256


def test_alpha_zero_is_used_when_every_nonzero_bf16_state_is_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture()

    def overflowing_trust_direction(
        delta_query: torch.Tensor,
        delta_key: torch.Tensor,
        current_query: torch.Tensor,
        current_key: torch.Tensor,
        trust_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
        del current_query, current_key, trust_radius
        maximum = torch.finfo(torch.float32).max
        return (
            torch.full_like(delta_query, maximum),
            torch.full_like(delta_key, maximum),
            1.0,
            1.0,
            1.0,
        )

    monkeypatch.setattr(endpoint_fold, "_trust_scale", overflowing_trust_direction)
    policy = replace(
        _one_iteration_policy(),
        maximum_sweeps=2,
        alpha_frontier=(0.0, 1.0),
        authentication_sha256="",
    )
    proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        policy=policy,
    )

    sweep = proposal.report["sweeps"][0]
    assert sweep["selected_alpha"] == 0.0
    assert sweep["selection_reason"] == "all_nonzero_candidates_identical_or_invalid"
    invalid = sweep["frontier"][1]
    assert invalid["candidate_finite"] is False
    assert invalid["candidate_shape_correct"] is True
    assert invalid["candidate_effectful"] is True
    assert invalid["eligible"] is False
    assert invalid["metrics"] is None
    assert invalid["legacy_numeric_proxy_gate_passed"] is None
    assert proposal.changed is False
    assert proposal.verify() == proposal.report_sha256


def test_resealed_fit_frontier_cannot_relabel_effectful_candidate_ineligible(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    effectful = next(
        row
        for sweep in report["sweeps"]
        for row in sweep["frontier"]
        if row["candidate_effectful"]
    )
    assert effectful["eligible"] is True
    effectful["eligible"] = False
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="structural eligibility",
    ):
        replace(proposal, report=report)


def test_resealed_fit_frontier_cannot_forge_numeric_proxy_diagnostics(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    candidate = next(
        row
        for sweep in report["sweeps"]
        for row in sweep["frontier"]
        if row["candidate_finite"] and row["candidate_shape_correct"]
    )
    candidate["legacy_numeric_proxy_gate_passed"] = not candidate[
        "legacy_numeric_proxy_gate_passed"
    ]
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="numeric proxy diagnostics",
    ):
        replace(proposal, report=report)


def test_private_lineage_rejects_nonselected_state_effect_and_eligibility_relabel(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    sweep = next(value for value in report["sweeps"] if "frontier" in value)
    alpha_zero = next(row for row in sweep["frontier"] if row["alpha"] == 0.0)
    candidate = next(
        row
        for row in sweep["frontier"]
        if row["alpha"] not in (0.0, sweep["selected_alpha"])
        and row["candidate_effectful"]
    )
    candidate["state_sha256"] = alpha_zero["state_sha256"]
    candidate["candidate_effectful"] = False
    candidate["eligible"] = False
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="frontier lineage structural eligibility",
    ):
        replace(proposal, report=report)


def test_private_lineage_rejects_selected_state_and_alpha_provenance_swap(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    sweep = next(value for value in report["sweeps"] if "frontier" in value)
    selected = next(
        row for row in sweep["frontier"] if row["alpha"] == sweep["selected_alpha"]
    )
    nonselected = next(
        row
        for row in sweep["frontier"]
        if row["alpha"] not in (0.0, sweep["selected_alpha"])
        and row["candidate_effectful"]
    )
    selected_alpha = selected["alpha"]
    nonselected_alpha = nonselected["alpha"]
    selected_payload = {key: value for key, value in selected.items() if key != "alpha"}
    nonselected_payload = {
        key: value for key, value in nonselected.items() if key != "alpha"
    }
    selected.clear()
    selected.update({"alpha": selected_alpha, **nonselected_payload})
    nonselected.clear()
    nonselected.update({"alpha": nonselected_alpha, **selected_payload})
    sweep["selected_alpha"] = nonselected_alpha
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="frontier lineage structural eligibility|frontier ranking",
    ):
        replace(proposal, report=report)


@pytest.mark.parametrize("tamper", ("metric-count", "coherent-proxy"))
def test_private_lineage_rejects_resealed_nonselected_diagnostics(
    fit_proposal_fixture: tuple[dict[str, object], object],
    tamper: str,
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    sweep = next(value for value in report["sweeps"] if "frontier" in value)
    candidate = next(
        row
        for row in sweep["frontier"]
        if row["alpha"] not in (0.0, sweep["selected_alpha"])
        and row["metrics"] is not None
    )
    if tamper == "metric-count":
        candidate["metrics"]["case_count"] += 1
    else:
        candidate["metrics"]["js_divergence_mean"] = 1e6
        candidate["route_nonregression_proxy_passed"] = False
        candidate["legacy_numeric_proxy_gate_passed"] = False
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="row diagnostics or counts",
    ):
        replace(proposal, report=report)


@pytest.mark.parametrize("replacement", (None, object()))
def test_compiled_fit_proposal_requires_original_private_frontier_lineage(
    fit_proposal_fixture: tuple[dict[str, object], object],
    replacement: object,
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="lineage record is absent, replaced, or invalid",
    ):
        replace(proposal, _frontier_lineage_binding=replacement)


def test_resealed_public_lineage_digest_must_match_private_compiler_record(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    report["fit_frontier_lineage_binding_sha256"] = "0" * 64
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="failed authentication",
    ):
        replace(proposal, report=report)


def test_fit_proposal_deep_copied_private_record_loses_opaque_proof(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    copied = copy.copy(proposal)
    copied_record = copy.deepcopy(proposal._frontier_lineage_binding)
    object.__setattr__(copied, "_frontier_lineage_binding", copied_record)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="proof no longer names its compiler record",
    ):
        copied.verify()


def test_fit_proposal_rejects_deterministic_record_substitution(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    second = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        policy=proposal.policy,
    )
    assert second._frontier_lineage_binding.binding_sha256 == (
        proposal._frontier_lineage_binding.binding_sha256
    )
    assert second._frontier_lineage_binding is not proposal._frontier_lineage_binding
    substituted = copy.copy(proposal)
    object.__setattr__(
        substituted,
        "_frontier_lineage_binding",
        second._frontier_lineage_binding,
    )

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="proof no longer names its compiler record",
    ):
        substituted.verify()


def test_fit_proposal_rejects_coherently_resealed_report_and_lineage_copy(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    original_binding = proposal._frontier_lineage_binding
    assert isinstance(original_binding, endpoint_fold._FitFrontierLineageBinding)

    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    sweep_index = next(
        index for index, sweep in enumerate(report["sweeps"]) if "frontier" in sweep
    )
    sweep = report["sweeps"][sweep_index]
    row_index = next(
        index
        for index, row in enumerate(sweep["frontier"])
        if row["alpha"] not in (0.0, sweep["selected_alpha"])
        and row["metrics"] is not None
    )
    row = sweep["frontier"][row_index]
    row["metrics"]["case_count"] += 1

    lineage_sweep = original_binding.sweeps[sweep_index]
    changed_rows = list(lineage_sweep.rows)
    changed_rows[row_index] = replace(
        changed_rows[row_index],
        report_row_sha256=endpoint_fold._canonical_sha256(row),
    )
    changed_lineage_sweep = replace(
        lineage_sweep,
        rows=tuple(changed_rows),
        report_sweep_sha256=endpoint_fold._canonical_sha256(sweep),
    )
    changed_sweeps = list(original_binding.sweeps)
    changed_sweeps[sweep_index] = changed_lineage_sweep
    unsigned_binding = replace(
        original_binding,
        sweeps=tuple(changed_sweeps),
        binding_sha256="",
    )
    changed_binding = replace(
        unsigned_binding,
        binding_sha256=endpoint_fold._canonical_sha256(unsigned_binding.as_dict()),
    )
    report["fit_frontier_lineage_binding_sha256"] = changed_binding.binding_sha256
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    forged = copy.copy(proposal)
    object.__setattr__(forged, "report", report)
    object.__setattr__(forged, "_frontier_lineage_binding", changed_binding)
    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="proof no longer names its compiler record",
    ):
        forged.verify()


@pytest.mark.parametrize("replacement", (None, object()))
def test_fit_proposal_rejects_postconstruction_opaque_proof_replacement(
    fit_proposal_fixture: tuple[dict[str, object], object],
    replacement: object,
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    replaced = copy.copy(proposal)
    object.__setattr__(replaced, "_frontier_lineage_proof", replacement)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="lineage proof is missing or replaced",
    ):
        replaced.verify()


def test_normal_dataclass_reconstruction_cannot_supply_an_opaque_lineage_proof(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    proof_field = proposal.__dataclass_fields__["_frontier_lineage_proof"]
    assert proof_field.init is False

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="not compiler-issued with an opaque lineage proof",
    ):
        replace(proposal)


def test_exact_zero_trust_reference_preserves_a_real_cold_direction() -> None:
    delta_query = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32)
    delta_key = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    current_query = torch.zeros_like(delta_query)
    current_key = torch.zeros_like(delta_key)

    query, key, query_relative, key_relative, scale = endpoint_fold._trust_scale(
        delta_query,
        delta_key,
        current_query,
        current_key,
        0.25,
    )

    assert query_relative == 1.0
    assert key_relative == 1.0
    assert scale == 0.25
    assert torch.equal(query, delta_query * 0.25)
    assert torch.equal(key, delta_key * 0.25)
    assert int(torch.count_nonzero(query.to(torch.bfloat16)).item()) > 0
    assert int(torch.count_nonzero(key.to(torch.bfloat16)).item()) > 0


def test_nonzero_trust_reference_is_the_original_relative_norm_formula() -> None:
    delta_query = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32)
    delta_key = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    current_query = torch.tensor([[0.5, 1.0], [-1.5, 2.0]], dtype=torch.float32)
    current_key = torch.tensor([[1.5, -0.25]], dtype=torch.float32)
    trust_radius = 0.375
    tiny = torch.finfo(torch.float64).tiny
    expected_query_relative = float(
        torch.linalg.vector_norm(delta_query.to(torch.float64)).item()
    ) / max(
        float(torch.linalg.vector_norm(current_query.to(torch.float64)).item()),
        tiny,
    )
    expected_key_relative = float(
        torch.linalg.vector_norm(delta_key.to(torch.float64)).item()
    ) / max(
        float(torch.linalg.vector_norm(current_key.to(torch.float64)).item()),
        tiny,
    )
    expected_scale = min(
        1.0,
        trust_radius / max(expected_query_relative, tiny),
        trust_radius / max(expected_key_relative, tiny),
    )

    query, key, query_relative, key_relative, scale = endpoint_fold._trust_scale(
        delta_query,
        delta_key,
        current_query,
        current_key,
        trust_radius,
    )

    assert query_relative == expected_query_relative
    assert key_relative == expected_key_relative
    assert scale == expected_scale
    assert torch.equal(query, delta_query * expected_scale)
    assert torch.equal(key, delta_key * expected_scale)


def test_nonpositive_curvature_direction_is_never_usable() -> None:
    class NonpositiveOperator:
        normal_operator_calls = 0

        def normal(self, vector: torch.Tensor) -> torch.Tensor:
            self.normal_operator_calls += 1
            return -vector

    operator = NonpositiveOperator()
    direction, report = endpoint_fold._ridge_cg(
        operator,
        torch.ones(8, dtype=torch.float32),
        _one_iteration_policy(),
    )
    assert torch.equal(direction, torch.zeros_like(direction))
    assert report["termination"] == "nonpositive_or_nonfinite_curvature"
    assert report["converged"] is False
    assert report["usable_direction"] is False
    nonfinite = endpoint_fold._cg_solve_report(
        torch.tensor([float("nan")]),
        converged=False,
        iterations=1,
        relative_normal_residual=1.0,
        termination="maximum_iterations",
        normal_operator_calls=1,
    )
    assert nonfinite["direction_is_finite"] is False
    assert nonfinite["usable_direction"] is False


def test_exact_bfloat16_endpoint_fold_proves_blank_independent_4x_qkvo() -> None:
    fixture = _make_fixture()
    first = compile_lfm2_cached_endpoint_qk_residual_fold(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        fit_check_bank=fixture["fit_check_bank"],
        policy=_policy(),
    )
    second = compile_lfm2_cached_endpoint_qk_residual_fold(
        seed=fixture["seed_b"],
        fit_bank=fixture["fit_bank"],
        fit_check_bank=fixture["fit_check_bank"],
        policy=_policy(),
    )

    assert first.accepted and first.changed
    assert first.status == "deterministic_fit_proposal_published"
    assert first.seed.seed_sha256 == second.seed.seed_sha256
    assert torch.equal(first.query_weight, second.query_weight)
    assert torch.equal(first.key_weight, second.key_weight)
    assert first.verify() == second.verify()
    assert (
        first.fit_metrics.endpoint_relative_rmse
        < first.report["initial_fit_metrics"]["endpoint_relative_rmse"]
    )
    assert (
        first.fit_metrics.probability_relative_rmse
        <= first.report["initial_fit_metrics"]["probability_relative_rmse"]
    )
    assert first.report["fit_check_endpoint_and_route_nonregressing"]
    assert first.report["heldout_metric_name_is_compatibility_alias_only"]
    assert first.report["fit_check_role"] == (
        "diagnostic_only_after_deterministic_fit_publication"
    )
    assert first.report["fit_check_can_accept_reject_or_rollback"] is False
    assert first.report["numeric_metric_used_for_publication"] is False
    assert first.report["publication_rule"] == (
        "publish_authenticated_fit_only_proposal_if_bfloat16_state_changed"
    )
    assert first.report["unbiased_generalization_claimed"] is False
    assert first.report["fresh_whole_child_intelligence_validation_required"]
    assert first.report["parameter_compaction_ratio"] == 4.0
    assert first.report["target_logical_parameters"] == 192
    assert first.report["source_logical_parameters"] == 768
    assert first.seed.source_inventory.logical_parameters == 768
    assert first.seed.source_inventory.source_state_sha256 == (
        endpoint_fold._state_sha256(
            {
                "q_proj.weight": fixture["source_query"],
                "k_proj.weight": fixture["source_key"],
                "v_proj.weight": fixture["source_value"],
                "out_proj.weight": fixture["source_output"],
            }
        )
    )
    assert first.report["blank_target_values_read"] is False
    assert all(
        value == 0
        for name, value in first.report["operation_audit"].items()
        if name
        in {
            "model_forward_calls",
            "model_generate_calls",
            "backward_calls",
            "optimizer_steps",
            "training_steps",
            "parameters_pruned",
            "parameters_quantized",
            "trace_recaptures",
            "blank_target_value_reads",
        }
    )
    state = first.state_dict()
    assert sum(tensor.numel() for tensor in state.values()) == 192
    assert torch.equal(state["v_proj.weight"], first.seed.value_weight)
    assert torch.equal(state["out_proj.weight"], first.seed.output_weight)

    probabilities, endpoint = replay_lfm2_endpoint_qkvo(
        fixture["first_case"],
        first.seed,
        query_weight=first.query_weight,
        key_weight=first.key_weight,
    )
    assert probabilities.shape == (1, 2, 6, 6)
    assert endpoint.shape == (1, 6, 8)


def test_nonblank_fit_check_regression_cannot_rollback_fit_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture()
    monkeypatch.setattr(endpoint_fold, "_all_nonregressing", lambda *_args: False)

    result = compile_lfm2_cached_endpoint_qk_residual_fold(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        fit_check_bank=fixture["fit_check_bank"],
        policy=_policy(),
    )

    assert result.fit_proposal.changed is True
    assert result.accepted is True
    assert result.changed is True
    assert result.status == "deterministic_fit_proposal_published"
    assert result.report["fit_check_endpoint_and_route_nonregressing"] is False
    assert result.report["fit_check_can_accept_reject_or_rollback"] is False
    assert result.report["numeric_metric_used_for_publication"] is False
    assert torch.equal(result.query_weight, result.fit_proposal.query_weight)
    assert torch.equal(result.key_weight, result.fit_proposal.key_weight)
    assert (
        result.fit_check_metrics.as_dict()
        == result.report["proposed_fit_check_metrics"]
    )
    assert result.verify() == result.report["report_sha256"]


def test_cached_exact_replay_matches_live_lfm2_bfloat16_attention_helper() -> None:
    transformers = pytest.importorskip("transformers")
    import neural_forge.hf_lfm2_cpcf_macrocell_replay as macrocell_replay

    config = transformers.Lfm2Config(
        vocab_size=37,
        hidden_size=8,
        intermediate_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention"],
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260721)
        model = transformers.Lfm2ForCausalLM(config)
    model.to(dtype=torch.bfloat16)
    model.eval()
    layer = model.model.layers[0]
    with torch.no_grad():
        layer.self_attn.q_layernorm.weight.copy_(
            torch.tensor((0.75, 0.9375, 1.125, 1.25), dtype=torch.bfloat16)
        )
        layer.self_attn.k_layernorm.weight.copy_(
            torch.tensor((1.25, 1.0625, 0.875, 0.75), dtype=torch.bfloat16)
        )
        residual = model.model.embed_tokens(
            torch.tensor(((1, 2, 3, 4, 5),), dtype=torch.long)
        ).detach()
        positions = torch.arange(5, dtype=torch.long).unsqueeze(0)
        cosine, sine = model.model.rotary_emb(residual, position_ids=positions)
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    additive_mask = torch.zeros(
        (1, 1, 5, 5),
        dtype=torch.bfloat16,
    ).masked_fill(future, torch.finfo(torch.bfloat16).min)
    live = macrocell_replay.replay_lfm2_cpcf_macrocell(
        layer,
        residual,
        position_embeddings=(cosine, sine),
        additive_causal_mask=additive_mask,
    )

    attention = layer.self_attn
    query_weight = attention.q_proj.weight.detach().clone()
    key_weight = attention.k_proj.weight.detach().clone()
    value_weight = attention.v_proj.weight.detach().clone()
    output_weight = attention.out_proj.weight.detach().clone()
    generator = torch.Generator().manual_seed(20260722)
    source_query = torch.randn((16, 16), generator=generator).to(torch.bfloat16)
    source_key = torch.randn((8, 16), generator=generator).to(torch.bfloat16)
    source_value = torch.randn((8, 16), generator=generator).to(torch.bfloat16)
    source_output = torch.randn((16, 16), generator=generator).to(torch.bfloat16)
    source_inventory = endpoint_fold.seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=source_query,
        source_key_weight=source_key,
        source_value_weight=source_value,
        source_output_weight=source_output,
    )
    seed = build_lfm2_endpoint_qkvo_seed(
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        output_weight=output_weight,
        query_norm_weight=attention.q_layernorm.weight.detach().clone(),
        key_norm_weight=attention.k_layernorm.weight.detach().clone(),
        norm_epsilon=float(attention.q_layernorm.variance_epsilon),
        source_inventory=source_inventory,
        source_artifact_sha256=source_inventory.inventory_sha256,
        target_query_weight_template=torch.empty_like(query_weight),
        target_key_weight_template=torch.empty_like(key_weight),
        target_value_weight_template=torch.empty_like(value_weight),
        target_output_weight_template=torch.empty_like(output_weight),
    )
    causal_mask = (~future).reshape(1, 1, 5, 5).expand(1, 2, 5, 5)
    uniform_probabilities = causal_mask.to(torch.float64)
    uniform_probabilities = uniform_probabilities / uniform_probabilities.sum(
        dim=-1,
        keepdim=True,
    )
    case = Lfm2EndpointQkResidualCase(
        target_query_inputs=live.operator_input,
        target_key_value_inputs=live.operator_input,
        parent_probabilities=uniform_probabilities,
        parent_causal_mask=causal_mask,
        parent_endpoint_target=torch.zeros_like(live.operator_delta),
        query_rope_cos=cosine,
        query_rope_sin=sine,
        key_rope_cos=cosine,
        key_rope_sin=sine,
        query_row_sqrt_weights=torch.ones((1, 5), dtype=torch.float64),
        query_groups=((0,), (1,)),
        query_group_weights=((1.0,), (1.0,)),
        trajectory_sha256=_sha("live-bfloat16-attention-parity"),
    )

    probabilities, endpoint = replay_lfm2_endpoint_qkvo(case, seed)
    assert probabilities.dtype == torch.bfloat16
    assert endpoint.dtype == torch.bfloat16
    assert torch.equal(endpoint, live.operator_delta)
    assert (
        macrocell_replay._tensor_sha256(probabilities)
        == live.report["operator_detail_sha256"]["probabilities"]
    )

    value = torch.nn.functional.linear(live.operator_input, value_weight).view(
        1,
        5,
        1,
        4,
    )
    repeated_value = value.transpose(1, 2).repeat_interleave(2, dim=1)
    context = torch.matmul(probabilities, repeated_value)
    concatenated = context.transpose(1, 2).contiguous().reshape(1, 5, 8).contiguous()
    assert torch.equal(concatenated, live.operator_output_input)


def test_fit_and_fit_check_trajectory_overlap_is_rejected() -> None:
    fixture = _make_fixture()
    fit_bank = fixture["fit_bank"]
    overlap = seal_lfm2_endpoint_qk_fit_check_bank(
        (fit_bank.cases[0],),
        source_split_sha256=fit_bank.source_split_sha256,
        source_artifact_sha256=fit_bank.source_artifact_sha256,
    )
    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="FIT/FIT-CHECK",
    ):
        compile_lfm2_cached_endpoint_qk_residual_fold(
            seed=fixture["seed_a"],
            fit_bank=fit_bank,
            fit_check_bank=overlap,
            policy=_policy(),
        )


def test_execution_receipt_authenticates_real_semantic_work_and_resources(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    receipt = proposal.report["observed_execution_receipt"]
    assert receipt["schema"] == endpoint_fold.LFM2_ENDPOINT_QK_EXECUTION_SCHEMA
    assert receipt["total_exact_replay_calls"] > 0
    assert receipt["total_endpoint_jvp_calls"] > 0
    assert receipt["total_endpoint_vjp_calls"] > 0
    assert receipt["total_normal_operator_calls"] > 0
    assert receipt["total_semantic_probe_calls"] == len(proposal.fit_bank.cases)
    resources = receipt["resource_observation"]
    assert resources["within_all_resource_ceilings"]
    assert resources["every_compute_device_within_byte_ceiling"]
    assert resources["prepared_persistent_tensor_bytes"] > 0
    assert resources["attached_replay_tensor_bytes"] > 0
    assert resources["fixed_tensor_cache_count"] == 1
    assert resources["shared_fixed_tensor_cache_bytes"] > 0
    summary = receipt["compute_device_summaries"][0]
    assert summary["logical_device"] == "cpu"
    assert summary["case_assignment_indices"] == list(
        range(len(proposal.fit_bank.cases))
    )
    assert summary["fixed_cache_receipt"]["fixed_cache_builds"] == 1
    assert (
        summary["replay_qk_shard_broadcasts"]
        == (receipt["compute_observations"][0]["exact_replay_calls"])
    )
    assert (
        summary["replay_qk_shard_broadcasts"] * summary["case_count"]
        == (summary["exact_replay_calls"])
    )
    assert (
        summary["normal_qk_direction_shard_broadcasts"]
        == (receipt["numerical_normal_operator_calls"])
    )
    assert (
        summary["normal_gradient_shard_reductions"]
        == (receipt["numerical_normal_operator_calls"])
    )
    assert (
        summary["rhs_gradient_shard_reductions"] == (receipt["numerical_sweep_count"])
    )
    assert summary["semantic_direction_shard_builds"] == 1
    assert summary["semantic_gradient_shard_reductions"] == 1
    assert summary["concurrent_rhs_shard_launches"] == 0
    assert summary["concurrent_normal_shard_launches"] == 0
    assert receipt["compute_device_concurrency"] == "serial_static_device_shards"
    static = receipt["static_device_shard_receipt"]
    assert static["shards"][0]["all_cases_share_fixed_cache_by_identity"]
    assert summary["within_compute_device_byte_ceiling"]
    assert proposal.verify() == proposal.report_sha256


def test_prepared_cases_share_fixed_cache_and_normal_transfers_once_per_shard() -> None:
    fixture = _make_fixture()
    seed = fixture["seed_a"]
    fit_bank = fixture["fit_bank"]
    cases = endpoint_fold._prepare_cases_round_robin(
        fit_bank.cases,
        seed,
        compute_devices=(torch.device("cpu"),),
    )
    assert len({id(case.fixed_cache) for case in cases}) == 1
    cache = cases[0].fixed_cache
    assert all(case.fixed_cache is cache for case in cases)
    assert cache.output_weight_bfloat16.dtype == torch.bfloat16
    fixed_receipt = endpoint_fold._fixed_device_cache_receipt(cache)
    assert fixed_receipt["schema"].endswith("-v2")
    assert "fixed_output_bfloat16_sha256" in fixed_receipt
    assert "fixed_output_float64_sha256" not in fixed_receipt
    assert fixed_receipt["shared_fixed_tensor_bytes"] == (
        endpoint_fold._fixed_device_tensor_cache_bytes(cache)
    )

    endpoint_fold._attach_replay(
        cases,
        seed,
        seed.query_weight,
        seed.key_weight,
    )
    assert cache.replay_qk_shard_broadcasts == 1
    assert all(case.probabilities.dtype == torch.bfloat16 for case in cases)
    assert all(case.repeated_value.dtype == torch.bfloat16 for case in cases)
    assert all(case.endpoint.dtype == torch.bfloat16 for case in cases)
    operator = endpoint_fold.Lfm2EndpointQkLinearOperator(
        seed=seed,
        prepared_cases=cases,
        current_query_weight=seed.query_weight.to(torch.float32),
        current_key_weight=seed.key_weight.to(torch.float32),
        policy=_policy(),
    )
    direction = operator.pack(
        torch.ones_like(seed.query_weight, dtype=torch.float32),
        torch.ones_like(seed.key_weight, dtype=torch.float32),
    )
    first = operator.normal(direction)
    assert torch.isfinite(first).all()
    assert cache.normal_qk_direction_shard_broadcasts == 1
    assert cache.normal_gradient_shard_reductions == 1
    assert sum(case.endpoint_jvp_calls for case in cases) == len(cases)
    assert sum(case.endpoint_vjp_calls for case in cases) == len(cases)


def test_concurrent_shard_launcher_enters_both_workers_and_returns_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = (
        SimpleNamespace(device=torch.device("cpu"), marker="first"),
        SimpleNamespace(device=torch.device("cpu"), marker="second"),
    )
    both_workers_inside_operation = threading.Barrier(2)
    monkeypatch.setattr(
        endpoint_fold,
        "_uses_concurrent_static_cuda_shards",
        lambda _shards: True,
    )

    def operation(shard: SimpleNamespace) -> str:
        both_workers_inside_operation.wait(timeout=5.0)
        return shard.marker

    assert endpoint_fold._execute_static_device_shards(shards, operation) == (
        "first",
        "second",
    )


def test_two_phase_rhs_and_normal_match_serial_static_shard_reduction_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture()
    seed = fixture["seed_a"]
    cases = endpoint_fold._prepare_cases_round_robin(
        fixture["fit_bank"].cases,
        seed,
        compute_devices=(torch.device("cpu"),),
    )
    endpoint_fold._attach_replay(
        cases,
        seed,
        seed.query_weight,
        seed.key_weight,
    )
    operator = endpoint_fold.Lfm2EndpointQkLinearOperator(
        seed=seed,
        prepared_cases=cases,
        current_query_weight=seed.query_weight.to(torch.float32),
        current_key_weight=seed.key_weight.to(torch.float32),
        policy=_policy(),
    )
    midpoint = (len(cases) + 1) // 2
    operator.shards = (
        endpoint_fold._StaticDeviceShard(
            device=torch.device("cpu"),
            fixed_cache=cases[0].fixed_cache,
            cases=tuple(cases[:midpoint]),
        ),
        endpoint_fold._StaticDeviceShard(
            device=torch.device("cpu"),
            fixed_cache=cases[0].fixed_cache,
            cases=tuple(cases[midpoint:]),
        ),
    )
    direction = operator.pack(
        torch.linspace(-0.25, 0.25, seed.query_weight.numel()).reshape(
            seed.query_weight.shape
        ),
        torch.linspace(0.25, -0.25, seed.key_weight.numel()).reshape(
            seed.key_weight.shape
        ),
    )

    monkeypatch.setattr(
        endpoint_fold,
        "_uses_concurrent_static_cuda_shards",
        lambda _shards: False,
    )
    serial_rhs = operator.right_hand_side()
    serial_normal = operator.normal(direction)

    monkeypatch.setattr(
        endpoint_fold,
        "_uses_concurrent_static_cuda_shards",
        lambda _shards: True,
    )
    concurrent_rhs = operator.right_hand_side()
    concurrent_normal = operator.normal(direction)

    assert torch.equal(concurrent_rhs, serial_rhs)
    assert torch.equal(concurrent_normal, serial_normal)


def test_resealed_forged_static_shard_transfer_count_is_rejected(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    receipt = report["observed_execution_receipt"]
    receipt["compute_device_summaries"][0]["replay_qk_shard_broadcasts"] += 1
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = endpoint_fold._canonical_sha256(receipt)
    report.pop("report_sha256")
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)
    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="aggregate receipt",
    ):
        replace(proposal, report=report)


def test_resealed_forged_concurrent_execution_mode_is_rejected(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    receipt = report["observed_execution_receipt"]
    receipt["compute_device_concurrency"] = (
        "thread_barrier_concurrent_static_cuda_shards"
    )
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = endpoint_fold._canonical_sha256(receipt)
    report.pop("report_sha256")
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)
    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="requested devices/cases",
    ):
        replace(proposal, report=report)


def test_resealed_forged_semantic_witness_is_rejected(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    observation = report["observed_execution_receipt"]["compute_observations"][0]
    observation["semantic_delta_endpoint_sha256"] = "0" * 64
    observation_body = dict(observation)
    observation_body.pop("case_execution_receipt_sha256")
    observation["case_execution_receipt_sha256"] = endpoint_fold._canonical_sha256(
        observation_body
    )
    receipt = report["observed_execution_receipt"]
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256")
    receipt["receipt_sha256"] = endpoint_fold._canonical_sha256(receipt_body)
    report_body = dict(report)
    report_body.pop("report_sha256")
    report["report_sha256"] = endpoint_fold._canonical_sha256(report_body)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="does not replay",
    ):
        replace(proposal, report=report)


def test_refreshed_proposal_report_cannot_change_no_training_audit(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    report["operation_audit"]["training_steps"] = 1
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="failed authentication",
    ):
        replace(proposal, report=report)


def test_refreshed_nested_execution_audit_is_rejected(
    fit_proposal_fixture: tuple[dict[str, object], object],
) -> None:
    _fixture, proposal = fit_proposal_fixture
    assert isinstance(proposal, endpoint_fold.Lfm2CompiledEndpointQkFitProposal)
    report = json.loads(json.dumps(dict(proposal.report)))
    report.pop("report_sha256")
    receipt = report["observed_execution_receipt"]
    receipt.pop("receipt_sha256")
    receipt["operation_audit"]["training_steps"] = 1
    receipt["receipt_sha256"] = endpoint_fold._canonical_sha256(receipt)
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)

    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="requested devices/cases",
    ):
        replace(proposal, report=report)


def test_resealed_forged_result_metrics_are_rejected() -> None:
    fixture = _make_fixture()
    result = compile_lfm2_cached_endpoint_qk_residual_fold(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        fit_check_bank=fixture["fit_check_bank"],
        policy=_policy(),
    )
    forged_metrics = replace(
        result.fit_metrics,
        endpoint_relative_rmse=0.0,
    )
    report = dict(result.report)
    report.pop("report_sha256")
    report["fit_metrics"] = forged_metrics.as_dict()
    report["report_sha256"] = endpoint_fold._canonical_sha256(report)
    with pytest.raises(
        Lfm2EndpointQkResidualAuthenticationError,
        match="failed authentication",
    ):
        replace(
            result,
            fit_metrics=forged_metrics,
            report=report,
        )


def test_refreshed_result_report_cannot_change_blank_or_training_audit() -> None:
    fixture = _make_fixture()
    result = compile_lfm2_cached_endpoint_qk_residual_fold(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        fit_check_bank=fixture["fit_check_bank"],
        policy=_policy(),
    )
    for field, value in (
        ("blank_target_values_read", True),
        ("operation_audit.training_steps", 1),
    ):
        report = json.loads(json.dumps(dict(result.report)))
        report.pop("report_sha256")
        if field == "blank_target_values_read":
            report[field] = value
        else:
            report["operation_audit"]["training_steps"] = value
        report["report_sha256"] = endpoint_fold._canonical_sha256(report)
        with pytest.raises(
            Lfm2EndpointQkResidualAuthenticationError,
            match="failed authentication",
        ):
            replace(result, report=report)


def test_two_cuda_workers_receive_real_round_robin_semantic_work() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    fixture = _make_fixture()
    policy = replace(
        _policy(),
        solver_device="cuda:0",
        compute_devices=("cuda:0", "cuda:1"),
        authentication_sha256="",
    )
    proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=fixture["seed_a"],
        fit_bank=fixture["fit_bank"],
        policy=policy,
    )
    receipt = proposal.report["observed_execution_receipt"]
    assert receipt["distinct_cuda_uuid_count"] == 2
    assert (
        receipt["compute_device_concurrency"]
        == "thread_barrier_concurrent_static_cuda_shards"
    )
    assert receipt["qk_transfer_execution"] == (
        "one_local_accumulation_per_concurrent_static_cuda_shard_then_"
        "canonical_requested_device_reduction"
    )
    assert [
        summary["case_count"] for summary in receipt["compute_device_summaries"]
    ] == [
        4,
        3,
    ]
    assert all(
        summary["semantic_probe_calls"] == summary["case_count"]
        for summary in receipt["compute_device_summaries"]
    )
    assert all(
        summary["fixed_cache_receipt"]["fixed_cache_builds"] == 1
        and summary["replay_qk_shard_broadcasts"]
        == receipt["compute_observations"][0]["exact_replay_calls"]
        and summary["normal_qk_direction_shard_broadcasts"]
        == receipt["numerical_normal_operator_calls"]
        and summary["normal_gradient_shard_reductions"]
        == receipt["numerical_normal_operator_calls"]
        and summary["concurrent_rhs_shard_launches"] == receipt["numerical_sweep_count"]
        and summary["concurrent_normal_shard_launches"]
        == receipt["numerical_normal_operator_calls"]
        for summary in receipt["compute_device_summaries"]
    )
