from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

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
    query = (
        (query_inputs @ query_weight.T)
        .reshape(batch, query_length, 2, 4)
        .permute(0, 2, 1, 3)
    )
    key = (
        (key_inputs @ key_weight.T).reshape(batch, key_length, 1, 4).permute(0, 2, 1, 3)
    )
    value = (
        (key_inputs @ value_weight.T)
        .reshape(batch, key_length, 1, 4)
        .permute(0, 2, 1, 3)
        .repeat_interleave(2, dim=1)
        .to(torch.float64)
    )
    query = _rmsnorm(query, norm_weight)
    key = _rmsnorm(key, norm_weight)
    query = query * query_cosine.unsqueeze(1) + _rotate_half(
        query
    ) * query_sine.unsqueeze(1)
    key = key * key_cosine.unsqueeze(1) + _rotate_half(key) * key_sine.unsqueeze(1)
    key = key.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, key.transpose(-2, -1)).float() / math.sqrt(4)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    scores = torch.where(
        mask,
        scores - (scores * mask).sum(dim=-1, keepdim=True) / count,
        torch.zeros_like(scores),
    )
    probabilities = torch.softmax(
        scores.double().masked_fill(~mask, -torch.inf),
        dim=-1,
    )
    probabilities = torch.where(
        mask,
        probabilities,
        torch.zeros_like(probabilities),
    )
    context = probabilities @ value
    endpoint = (
        context.permute(0, 2, 1, 3).reshape(batch, query_length, 8)
        @ output_weight.double().T
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


def test_finite_one_iteration_cg_direction_enters_exact_nonregressing_frontier() -> None:
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
    for sweep in proposal.report["sweeps"]:
        solve = sweep["solve_report"]
        assert solve["converged"] is False
        assert solve["termination"] == "maximum_iterations"
        assert solve["usable_direction"] is True
        assert solve["direction_is_finite"] is True
        assert solve["direction_has_nonzero_entry"] is True
        selected = next(
            row
            for row in sweep["frontier"]
            if row["alpha"] == sweep["selected_alpha"]
        )
        assert selected["eligible"] is True
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


def test_exact_frontier_rejects_a_finite_truncated_regressing_direction(
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
    assert proposal.report["sweeps"][0]["selected_alpha"] == 0.0
    assert proposal.changed is False
    assert (
        proposal.proposed_fit_metrics.as_dict()
        == proposal.initial_fit_metrics.as_dict()
    )
    assert torch.equal(proposal.query_weight, proposal.seed.query_weight)
    assert torch.equal(proposal.key_weight, proposal.seed.key_weight)


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
    assert first.status == "fit_check_accepted"
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
        "reused_development_acceptance_gate_not_unbiased_validation"
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
    assert summary["replay_qk_shard_broadcasts"] == (
        receipt["compute_observations"][0]["exact_replay_calls"]
    )
    assert summary["replay_qk_shard_broadcasts"] * summary["case_count"] == (
        summary["exact_replay_calls"]
    )
    assert summary["normal_qk_direction_shard_broadcasts"] == (
        receipt["numerical_normal_operator_calls"]
    )
    assert summary["normal_gradient_shard_reductions"] == (
        receipt["numerical_normal_operator_calls"]
    )
    assert summary["rhs_gradient_shard_reductions"] == (
        receipt["numerical_sweep_count"]
    )
    assert summary["semantic_direction_shard_builds"] == 1
    assert summary["semantic_gradient_shard_reductions"] == 1
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

    endpoint_fold._attach_replay(
        cases,
        seed,
        seed.query_weight,
        seed.key_weight,
    )
    assert cache.replay_qk_shard_broadcasts == 1
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
        for summary in receipt["compute_device_summaries"]
    )
