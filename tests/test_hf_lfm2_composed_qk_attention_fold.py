from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_composed_qk_attention_fold as composed_fold  # noqa: E402
from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (  # noqa: E402
    lfm2_runtime_rms_norm,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedAttentionFoldCase,
    compile_lfm2_structural_attention_value_output_fold,
)


SOURCE_HIDDEN = 8
TARGET_HIDDEN = 4
SOURCE_QUERY_HEADS = 4
SOURCE_KEY_VALUE_HEADS = 2
TARGET_QUERY_HEADS = 2
TARGET_KEY_VALUE_HEADS = 1
HEAD_DIM = 2
SOURCE_SPLIT_SHA256 = "a" * 64


def _assert_zero_training_counters(value: object) -> None:
    counter_names = {
        "model_forward_calls",
        "model_generate_calls",
        "backward_calls",
        "optimizer_steps",
        "gradient_updates",
        "training_steps",
        "training_examples",
        "pruned_parameters",
        "quantized_parameters",
        "parameters_pruned",
        "parameters_quantized",
        "trace_recaptures",
    }
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name in counter_names:
                assert item == 0
            _assert_zero_training_counters(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_zero_training_counters(item)


def _target_state() -> SimpleNamespace:
    query = torch.tensor(
        [
            [0.75, 0.10, -0.20, 0.30],
            [0.05, 0.80, 0.25, -0.15],
            [0.40, -0.30, 0.70, 0.10],
            [-0.20, 0.45, 0.15, 0.65],
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [0.65, -0.10, 0.30, 0.20],
            [0.15, 0.70, -0.20, 0.25],
        ],
        dtype=torch.bfloat16,
    )
    value = torch.tensor(
        [
            [0.50, 0.10, -0.25, 0.35],
            [-0.15, 0.60, 0.30, 0.20],
        ],
        dtype=torch.bfloat16,
    )
    output = torch.tensor(
        [
            [0.50, -0.10, 0.20, 0.05],
            [0.15, 0.45, -0.05, 0.20],
            [-0.20, 0.10, 0.55, 0.15],
            [0.05, 0.25, 0.10, 0.50],
        ],
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        query=query,
        key=key,
        value=value,
        output=output,
    )


def _source_state(target: SimpleNamespace) -> SimpleNamespace:
    encoder = torch.zeros(
        (TARGET_HIDDEN, SOURCE_HIDDEN),
        dtype=torch.bfloat16,
    )
    encoder[:, :TARGET_HIDDEN] = torch.eye(
        TARGET_HIDDEN,
        dtype=torch.bfloat16,
    )
    query_heads = target.query.reshape(
        TARGET_QUERY_HEADS,
        HEAD_DIM,
        TARGET_HIDDEN,
    )
    query = (
        torch.cat(
            (
                query_heads[0],
                query_heads[1],
                query_heads[0],
                query_heads[1],
            ),
            dim=0,
        )
        @ encoder
    )
    key = torch.cat((target.key, target.key), dim=0) @ encoder
    value = torch.cat((target.value, target.value), dim=0) @ encoder
    output = torch.zeros(
        (SOURCE_HIDDEN, SOURCE_QUERY_HEADS * HEAD_DIM),
        dtype=torch.bfloat16,
    )
    target_output_blocks = target.output.reshape(
        TARGET_HIDDEN,
        TARGET_QUERY_HEADS,
        HEAD_DIM,
    )
    source_blocks = (
        target_output_blocks[:, 0] / 2,
        target_output_blocks[:, 1] / 2,
        target_output_blocks[:, 0] / 2,
        target_output_blocks[:, 1] / 2,
    )
    output[:TARGET_HIDDEN] = torch.cat(source_blocks, dim=1)
    return SimpleNamespace(
        query=query.contiguous(),
        key=key.contiguous(),
        value=value.contiguous(),
        output=output.contiguous(),
    )


def _case(
    source: SimpleNamespace,
    *,
    seed: int,
    length: int,
) -> Lfm2CachedAttentionFoldCase:
    generator = torch.Generator().manual_seed(seed)
    target_inputs = (
        torch.randn(
            (1, length, TARGET_HIDDEN),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.45
        + torch.linspace(-0.2, 0.25, length).reshape(1, length, 1)
    ).to(torch.bfloat16)
    source_inputs = torch.zeros(
        (1, length, SOURCE_HIDDEN),
        dtype=torch.bfloat16,
    )
    source_inputs[..., :TARGET_HIDDEN] = target_inputs
    query_linear = (
        (source_inputs @ source.query.T)
        .reshape(
            1,
            length,
            SOURCE_QUERY_HEADS,
            HEAD_DIM,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_linear = (
        (source_inputs @ source.key.T)
        .reshape(
            1,
            length,
            SOURCE_KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    # Preserve exact duplicated source-head structure while introducing a
    # small trajectory-local gauge mismatch.  This prevents the synchronized
    # BF16 seed from being an almost-zero-residual PCG corner case, while the
    # bounded common correction remains well inside the production gates.
    variation = 0.055
    query_first = query_linear[..., 0].clone()
    query_second = query_linear[..., 1].clone()
    query_linear[..., 0] = query_first + variation * query_second
    query_linear[..., 1] = query_second - 0.5 * variation * query_first
    key_first = key_linear[..., 0].clone()
    key_second = key_linear[..., 1].clone()
    key_linear[..., 0] = key_first - 0.35 * variation * key_second
    key_linear[..., 1] = key_second + 0.65 * variation * key_first
    value = (
        (source_inputs @ source.value.T)
        .reshape(
            1,
            length,
            SOURCE_KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    norm = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    query = lfm2_runtime_rms_norm(query_linear, norm, 1e-5)
    key = lfm2_runtime_rms_norm(key_linear, norm, 1e-5)
    mask = torch.tril(torch.ones((1, length, length), dtype=torch.bool)).contiguous()
    repeated_key = key.repeat_interleave(
        SOURCE_QUERY_HEADS // SOURCE_KEY_VALUE_HEADS,
        dim=1,
    )
    scores = query.float() @ repeated_key.float().transpose(-2, -1)
    scores = scores / math.sqrt(HEAD_DIM)
    probabilities = torch.softmax(
        scores.masked_fill(~mask.unsqueeze(1), -torch.inf),
        dim=-1,
    )
    context = (
        probabilities
        @ value.float().repeat_interleave(
            SOURCE_QUERY_HEADS // SOURCE_KEY_VALUE_HEADS,
            dim=1,
        )
    ).to(torch.bfloat16)
    flattened = context.permute(0, 2, 1, 3).reshape(
        1,
        length,
        SOURCE_QUERY_HEADS * HEAD_DIM,
    )
    parent_delta = (flattened @ source.output.T)[..., :TARGET_HIDDEN]
    endpoint = (target_inputs + parent_delta).contiguous()
    contributions = []
    for head in range(SOURCE_QUERY_HEADS):
        block = source.output[
            :TARGET_HIDDEN,
            head * HEAD_DIM : (head + 1) * HEAD_DIM,
        ]
        contributions.append(context[:, head] @ block.T)
    ones = torch.ones((1, length), dtype=torch.bfloat16)
    cosine = torch.ones((1, length, HEAD_DIM), dtype=torch.bfloat16)
    sine = torch.zeros_like(cosine)
    return Lfm2CachedAttentionFoldCase(
        target_query_input=target_inputs,
        target_key_value_input=target_inputs,
        parent_query_linear=query_linear,
        parent_key_linear=key_linear,
        parent_value=value,
        parent_query=query,
        parent_key=key,
        parent_context=context,
        parent_probabilities=probabilities.contiguous(),
        parent_post_attention_residual_target_space=endpoint,
        target_pre_attention_residual=target_inputs,
        query_cosine=cosine,
        query_sine=sine,
        key_cosine=cosine,
        key_sine=sine,
        key_mask=mask,
        parent_contribution=torch.stack(contributions, dim=1),
        query_row_sqrt_weights=ones,
        key_value_row_sqrt_weights=ones,
    )


def _child_context(
    producer: object,
    case: Lfm2CachedAttentionFoldCase,
) -> torch.Tensor:
    query_input = case.target_query_input
    key_input = case.target_key_value_input
    batch, query_length, _hidden = query_input.shape
    key_length = key_input.shape[1]
    query_linear = (
        (query_input @ producer.query_weight.T)
        .reshape(
            batch,
            query_length,
            TARGET_QUERY_HEADS,
            HEAD_DIM,
        )
        .permute(0, 2, 1, 3)
    )
    key_linear = (
        (key_input @ producer.key_weight.T)
        .reshape(
            batch,
            key_length,
            TARGET_KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .permute(0, 2, 1, 3)
    )
    value = (
        (key_input @ producer.value_weight.T)
        .reshape(
            batch,
            key_length,
            TARGET_KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .permute(0, 2, 1, 3)
    )
    norm = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    query = lfm2_runtime_rms_norm(query_linear, norm, 1e-5)
    key = lfm2_runtime_rms_norm(key_linear, norm, 1e-5)
    probabilities = torch.softmax(
        (
            query.float()
            @ key.float()
            .repeat_interleave(
                TARGET_QUERY_HEADS // TARGET_KEY_VALUE_HEADS,
                dim=1,
            )
            .transpose(-2, -1)
            / math.sqrt(HEAD_DIM)
        ).masked_fill(~case.key_mask.unsqueeze(1), -torch.inf),
        dim=-1,
    )
    return (
        probabilities
        @ value.float().repeat_interleave(
            TARGET_QUERY_HEADS // TARGET_KEY_VALUE_HEADS,
            dim=1,
        )
    ).to(torch.bfloat16)


@pytest.fixture(scope="module")
def composed_artifacts() -> SimpleNamespace:
    target = _target_state()
    source = _source_state(target)
    cases = tuple(
        _case(source, seed=8100 + index, length=4 + index) for index in range(4)
    )
    trajectories = tuple(f"{100 + index:064x}" for index in range(len(cases)))
    query_template = torch.zeros_like(target.query)
    key_template = torch.zeros_like(target.key)
    value_template = torch.zeros_like(target.value)
    output_template = torch.zeros_like(target.output)
    norm = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    policy = composed_fold.Lfm2ComposedQkAttentionFoldPolicy(
        fit_check_count=1,
        global_policy=composed_fold.Lfm2QkGlobalGaugeSyncPolicy(
            source_anchor_prior_ridge=1.0,
            global_prior_ridge=1.0,
        ),
        score_policy=composed_fold.Lfm2QkScoreResidualFoldPolicy(
            ridge=2e-3,
            trust_radius=2.0,
            maximum_sweeps=2,
            maximum_pcg_iterations=96,
            minimum_pcg_iterations=2,
            pcg_restart_iterations=48,
            pcg_relative_tolerance=2e-4,
            metric_regression_tolerance=5e-3,
            maximum_fit_centered_score_relative_rmse=0.10,
            maximum_fit_check_centered_score_relative_rmse=0.20,
        ),
    )
    composed = composed_fold.compile_lfm2_composed_qk_attention_producer_fold(
        source_query_weight=source.query,
        source_key_weight=source.key,
        source_value_weight=source.value,
        source_output_weight=source.output,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        target_value_weight_template=value_template,
        target_output_weight_template=output_template,
        upstream_fit_cases=cases,
        trajectory_sha256s=trajectories,
        source_split_sha256=SOURCE_SPLIT_SHA256,
        source_query_norm_weight=norm,
        source_key_norm_weight=norm,
        target_query_norm_weight=norm,
        target_key_norm_weight=norm,
        source_head_dim=HEAD_DIM,
        target_head_dim=HEAD_DIM,
        norm_epsilon=1e-5,
        residual_assignment_sha256="b" * 64,
        policy=policy,
    )
    actual_contexts = tuple(
        _child_context(composed.producer_artifact, case) for case in cases
    )
    value_output = compile_lfm2_structural_attention_value_output_fold(
        producer=composed.producer_artifact,
        target_value_weight_template=value_template,
        target_output_weight_template=output_template,
        actual_child_contexts=actual_contexts,
        cases=cases,
        live_value_weight=composed.producer_artifact.value_weight,
        value_weight_prior=composed.producer_artifact.value_weight,
        output_weight_prior=output_template,
        query_norm_weight=norm,
        key_norm_weight=norm,
        norm_epsilon=1e-5,
        ridge=1e-8,
        context_ridge=1e-6,
        max_prior_delta_ratio=None,
        iterations=2,
        context_equivalence_tolerance=0.10,
    )
    outer = composed_fold.compile_lfm2_composed_qk_sequence_effective_attention_fold(
        composed_qk=composed,
        fit_value_output_artifact=value_output,
        upstream_fit_cases=cases,
        actual_child_contexts=actual_contexts,
        target_value_weight_template=value_template,
        target_output_weight_template=output_template,
        query_norm_weight=norm,
        key_norm_weight=norm,
        norm_epsilon=1e-5,
    )
    return SimpleNamespace(
        target=target,
        source=source,
        cases=cases,
        actual_contexts=actual_contexts,
        trajectories=trajectories,
        composed=composed,
        value_output=value_output,
        outer=outer,
        norm=norm,
        value_template=value_template,
        output_template=output_template,
    )


@pytest.fixture(scope="module")
def endpoint_residual_artifacts(
    composed_artifacts: SimpleNamespace,
) -> SimpleNamespace:
    composed = composed_artifacts.composed
    fit_cases = tuple(composed.qk_fit_bank.cases)
    fit_contexts = tuple(
        composed_artifacts.actual_contexts[ordinal]
        for ordinal in composed.qk_fit_bank.source_case_ordinals
    )
    # Keep a deliberately imperfect but authenticated closed-form V/O seed so
    # the endpoint-QK block has a material correction to prove. The fifth arm
    # must then refit V/O analytically and generalize to the untouched case.
    output_prior = (composed_artifacts.target.output.float() * 1.5).to(torch.bfloat16)
    fit_value_output = compile_lfm2_structural_attention_value_output_fold(
        producer=composed.producer_artifact,
        target_value_weight_template=composed_artifacts.value_template,
        target_output_weight_template=composed_artifacts.output_template,
        actual_child_contexts=fit_contexts,
        cases=fit_cases,
        live_value_weight=composed.producer_artifact.value_weight,
        value_weight_prior=composed.producer_artifact.value_weight,
        output_weight_prior=output_prior,
        query_norm_weight=composed_artifacts.norm,
        key_norm_weight=composed_artifacts.norm,
        norm_epsilon=1e-5,
        ridge=1e-8,
        context_ridge=1e-6,
        max_prior_delta_ratio=0.0,
        iterations=2,
        context_equivalence_tolerance=0.10,
    )
    endpoint_policy = composed_fold.Lfm2EndpointQkResidualFoldPolicy(
        ridge=1e-4,
        route_objective_weight=0.2,
        trust_radius=2.0,
        maximum_sweeps=4,
        maximum_cg_iterations=512,
        minimum_cg_iterations=2,
        cg_restart_iterations=64,
        cg_relative_tolerance=1e-3,
        metric_regression_tolerance=5e-5,
        minimum_fit_endpoint_improvement=1e-10,
        solver_device="cpu",
        compute_devices=("cpu",),
    )
    fifth = composed_fold.compile_lfm2_composed_endpoint_residual_attention_fold(
        composed_qk=composed,
        fit_value_output_artifact=fit_value_output,
        source_query_weight=composed_artifacts.source.query,
        source_key_weight=composed_artifacts.source.key,
        source_value_weight=composed_artifacts.source.value,
        source_output_weight=composed_artifacts.source.output,
        target_query_weight_template=torch.zeros_like(composed_artifacts.target.query),
        target_key_weight_template=torch.zeros_like(composed_artifacts.target.key),
        target_value_weight_template=composed_artifacts.value_template,
        target_output_weight_template=composed_artifacts.output_template,
        query_norm_weight=composed_artifacts.norm,
        key_norm_weight=composed_artifacts.norm,
        norm_epsilon=1e-5,
        policy=(
            composed_fold.Lfm2ComposedEndpointResidualAttentionFoldPolicy(
                maximum_outer_blocks=2,
                endpoint_policy=endpoint_policy,
            )
        ),
    )
    return SimpleNamespace(
        fifth=fifth,
        fit_value_output=fit_value_output,
        fit_cases=fit_cases,
        fit_contexts=fit_contexts,
    )


def test_composed_producer_uses_distinct_fit_check_and_plants_exact_state(
    composed_artifacts: SimpleNamespace,
) -> None:
    artifact = composed_artifacts.composed
    assert artifact.verify_authentication() == artifact.artifact_sha256
    assert artifact.upstream_trajectory_sha256s == (composed_artifacts.trajectories)
    assert artifact.upstream_trajectory_bundle_sha256 == (
        composed_fold._canonical_sha256(list(composed_artifacts.trajectories))
    )
    assert artifact.report["upstream_trajectory_sha256s"] == list(
        composed_artifacts.trajectories
    )
    assert artifact.report["upstream_trajectory_bundle_sha256"] == (
        artifact.upstream_trajectory_bundle_sha256
    )
    assert set(artifact.qk_fit_bank.source_case_ordinals).isdisjoint(
        artifact.qk_fit_check_bank.source_case_ordinals
    )
    assert set(artifact.qk_fit_bank.trajectory_sha256s).isdisjoint(
        artifact.qk_fit_check_bank.trajectory_sha256s
    )
    assert len(artifact.qk_fit_bank.cases) == 3
    assert len(artifact.qk_fit_check_bank.cases) == 1
    assert torch.equal(
        artifact.score_result.query_weight,
        artifact.corrected_producer.query_weight,
    )
    assert torch.equal(
        artifact.score_result.key_weight,
        artifact.corrected_producer.key_weight,
    )
    assert torch.equal(
        artifact.qk_fit_structural_producer.value_weight,
        artifact.corrected_producer.value_weight,
    )
    assert (
        artifact.qk_fit_structural_producer.query_groups
        == artifact.corrected_producer.query_groups
    )
    assert (
        artifact.qk_fit_structural_producer.key_value_groups
        == artifact.corrected_producer.key_value_groups
    )
    assert artifact.report["fit_check_used_for_structural_groups_or_priors"] is False
    assert artifact.report["primitive_local_qk_published"] is False
    assert artifact.score_result.seed.upstream_fold is artifact.primitive_fold
    assert artifact.score_result.seed.global_sync is artifact.global_sync
    assert (
        artifact.global_sync.report["provenance"]["fit_receipt_sha256"]
        == artifact.primitive_fold.report["report_sha256"]
    )
    assert all(value == 0 for value in artifact.report["operation_audit"].values())
    _assert_zero_training_counters(artifact.primitive_fold.report)
    _assert_zero_training_counters(artifact.global_sync.as_dict())
    _assert_zero_training_counters(dict(artifact.score_result.report))
    _assert_zero_training_counters(artifact.corrected_producer.as_dict())
    source_qkvo = sum(
        int(value.numel())
        for value in (
            composed_artifacts.source.query,
            composed_artifacts.source.key,
            composed_artifacts.source.value,
            composed_artifacts.source.output,
        )
    )
    target_qkvo = sum(
        int(value.numel()) for value in composed_artifacts.outer.state_dict().values()
    )
    assert source_qkvo == 192
    assert target_qkvo == 48
    assert source_qkvo == 4 * target_qkvo
    assert artifact.producer_artifact.source_logical_parameters == 128
    assert artifact.producer_artifact.target_logical_parameters == 32


def test_composed_producer_is_independent_of_blank_qk_initialization(
    composed_artifacts: SimpleNamespace,
) -> None:
    baseline = composed_artifacts.composed
    generator = torch.Generator().manual_seed(9981)
    query_template = torch.randn(
        composed_artifacts.target.query.shape,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key_template = torch.randn(
        composed_artifacts.target.key.shape,
        dtype=torch.bfloat16,
        generator=generator,
    )
    changed_blank = composed_fold.compile_lfm2_composed_qk_attention_producer_fold(
        source_query_weight=composed_artifacts.source.query,
        source_key_weight=composed_artifacts.source.key,
        source_value_weight=composed_artifacts.source.value,
        source_output_weight=composed_artifacts.source.output,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        target_value_weight_template=composed_artifacts.value_template,
        target_output_weight_template=composed_artifacts.output_template,
        upstream_fit_cases=composed_artifacts.cases,
        trajectory_sha256s=composed_artifacts.trajectories,
        source_split_sha256=SOURCE_SPLIT_SHA256,
        source_query_norm_weight=composed_artifacts.norm,
        source_key_norm_weight=composed_artifacts.norm,
        target_query_norm_weight=composed_artifacts.norm,
        target_key_norm_weight=composed_artifacts.norm,
        source_head_dim=HEAD_DIM,
        target_head_dim=HEAD_DIM,
        norm_epsilon=1e-5,
        residual_assignment_sha256="b" * 64,
        policy=baseline.policy,
    )
    assert changed_blank.verify_authentication() == changed_blank.artifact_sha256
    assert changed_blank.artifact_sha256 != baseline.artifact_sha256
    assert torch.equal(
        changed_blank.producer_artifact.query_weight,
        baseline.producer_artifact.query_weight,
    )
    assert torch.equal(
        changed_blank.producer_artifact.key_weight,
        baseline.producer_artifact.key_weight,
    )
    assert torch.equal(
        changed_blank.producer_artifact.value_weight,
        baseline.producer_artifact.value_weight,
    )
    assert (
        changed_blank.producer_artifact.query_groups
        == baseline.producer_artifact.query_groups
    )
    assert (
        changed_blank.producer_artifact.key_value_groups
        == baseline.producer_artifact.key_value_groups
    )


def test_composed_producer_rejects_alien_full_fit_bundle_receipt(
    composed_artifacts: SimpleNamespace,
) -> None:
    artifact = composed_artifacts.composed
    alien_bundle = "f" * 64
    body = dict(artifact.report)
    body.pop("artifact_sha256")
    body["upstream_fit_case_bundle_sha256"] = alien_bundle
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="partition union",
    ):
        replace(
            artifact,
            upstream_fit_case_bundle_sha256=alien_bundle,
            report={
                **body,
                "artifact_sha256": composed_fold._canonical_sha256(body),
            },
        )


def _resealed_producer_report_with_trajectories(
    artifact: composed_fold.Lfm2CompiledComposedQkAttentionProducerFold,
    trajectories: tuple[str, ...],
) -> dict[str, object]:
    body = dict(artifact.report)
    body.pop("artifact_sha256")
    body["upstream_trajectory_sha256s"] = list(trajectories)
    body["upstream_trajectory_bundle_sha256"] = composed_fold._canonical_sha256(
        list(trajectories)
    )
    return {
        **body,
        "artifact_sha256": composed_fold._canonical_sha256(body),
    }


def test_composed_producer_rejects_missing_upstream_trajectory_receipt(
    composed_artifacts: SimpleNamespace,
) -> None:
    artifact = composed_artifacts.composed
    trajectories = artifact.upstream_trajectory_sha256s[:-1]
    bundle = composed_fold._canonical_sha256(list(trajectories))

    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="exactly one distinct receipt",
    ):
        replace(
            artifact,
            upstream_trajectory_sha256s=trajectories,
            upstream_trajectory_bundle_sha256=bundle,
            report=_resealed_producer_report_with_trajectories(
                artifact,
                trajectories,
            ),
        )


def test_composed_producer_rejects_reordered_resealed_upstream_trajectories(
    composed_artifacts: SimpleNamespace,
) -> None:
    artifact = composed_artifacts.composed
    trajectories = (
        artifact.upstream_trajectory_sha256s[1],
        artifact.upstream_trajectory_sha256s[0],
        *artifact.upstream_trajectory_sha256s[2:],
    )
    bundle = composed_fold._canonical_sha256(list(trajectories))

    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="reconstruct the exact ordered",
    ):
        replace(
            artifact,
            upstream_trajectory_sha256s=trajectories,
            upstream_trajectory_bundle_sha256=bundle,
            report=_resealed_producer_report_with_trajectories(
                artifact,
                trajectories,
            ),
        )


def test_composed_producer_rejects_substituted_resealed_upstream_trajectory(
    composed_artifacts: SimpleNamespace,
) -> None:
    artifact = composed_artifacts.composed
    trajectories = (
        "f" * 64,
        *artifact.upstream_trajectory_sha256s[1:],
    )
    bundle = composed_fold._canonical_sha256(list(trajectories))

    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="reconstruct the exact ordered",
    ):
        replace(
            artifact,
            upstream_trajectory_sha256s=trajectories,
            upstream_trajectory_bundle_sha256=bundle,
            report=_resealed_producer_report_with_trajectories(
                artifact,
                trajectories,
            ),
        )


def test_residual_vo_seed_and_outer_qk_are_cross_authenticated_exactly(
    composed_artifacts: SimpleNamespace,
) -> None:
    outer = composed_artifacts.outer
    composed = composed_artifacts.composed
    assert outer.verify_authentication() == outer.artifact_sha256
    assert outer.report["source_seed_fit_provenance_cross_authenticated"] is True
    assert (
        outer.residual_vo_seed.provenance.source_compiled_state_sha256
        == composed_artifacts.value_output.compiled_state_sha256
    )
    assert torch.equal(
        outer.residual_vo_seed.value_weight,
        composed_artifacts.value_output.value_weight,
    )
    assert torch.equal(
        outer.residual_vo_seed.output_weight,
        composed_artifacts.value_output.output_weight,
    )
    assert torch.equal(
        outer.state_dict()["q_proj.weight"],
        composed.score_result.query_weight,
    )
    assert torch.equal(
        outer.state_dict()["k_proj.weight"],
        composed.score_result.key_weight,
    )
    assert outer.producer_artifact is composed.producer_artifact
    assert outer.report["upstream_trajectory_sha256s"] == list(
        composed.upstream_trajectory_sha256s
    )
    assert outer.report["upstream_trajectory_bundle_sha256"] == (
        composed.upstream_trajectory_bundle_sha256
    )
    assert all(value == 0 for value in outer.report["operation_audit"].values())
    _assert_zero_training_counters(outer.inner_artifact.as_dict())
    _assert_zero_training_counters(outer.as_dict())


def test_outer_rejects_resealed_upstream_trajectory_bundle_drift(
    composed_artifacts: SimpleNamespace,
) -> None:
    outer = composed_artifacts.outer
    body = dict(outer.report)
    body.pop("artifact_sha256")
    trajectories = list(reversed(body["upstream_trajectory_sha256s"]))
    body["upstream_trajectory_sha256s"] = trajectories
    body["upstream_trajectory_bundle_sha256"] = composed_fold._canonical_sha256(
        trajectories
    )

    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="outer composed sequence-effective receipt",
    ):
        replace(
            outer,
            report={
                **body,
                "artifact_sha256": composed_fold._canonical_sha256(body),
            },
        )


def test_outer_rejects_value_output_fit_on_alien_cases(
    composed_artifacts: SimpleNamespace,
) -> None:
    altered_cases = list(composed_artifacts.cases)
    altered_cases[0] = replace(
        altered_cases[0],
        target_pre_attention_residual=(
            altered_cases[0].target_pre_attention_residual + 0.125
        ),
    )
    alien_value_output = compile_lfm2_structural_attention_value_output_fold(
        producer=composed_artifacts.composed.producer_artifact,
        target_value_weight_template=composed_artifacts.value_template,
        target_output_weight_template=composed_artifacts.output_template,
        actual_child_contexts=composed_artifacts.actual_contexts,
        cases=tuple(altered_cases),
        live_value_weight=(composed_artifacts.composed.producer_artifact.value_weight),
        value_weight_prior=(composed_artifacts.composed.producer_artifact.value_weight),
        output_weight_prior=composed_artifacts.output_template,
        query_norm_weight=composed_artifacts.norm,
        key_norm_weight=composed_artifacts.norm,
        norm_epsilon=1e-5,
        ridge=1e-8,
        context_ridge=1e-6,
        max_prior_delta_ratio=None,
        iterations=2,
        context_equivalence_tolerance=0.10,
    )
    assert (
        alien_value_output.endpoint_target_sha256
        != composed_artifacts.value_output.endpoint_target_sha256
    )
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="exact cases/contexts",
    ):
        composed_fold.compile_lfm2_composed_qk_sequence_effective_attention_fold(
            composed_qk=composed_artifacts.composed,
            fit_value_output_artifact=alien_value_output,
            upstream_fit_cases=composed_artifacts.cases,
            actual_child_contexts=composed_artifacts.actual_contexts,
            target_value_weight_template=composed_artifacts.value_template,
            target_output_weight_template=composed_artifacts.output_template,
            query_norm_weight=composed_artifacts.norm,
            key_norm_weight=composed_artifacts.norm,
            norm_epsilon=1e-5,
        )


@pytest.mark.parametrize("contract_change", ("query_norm", "epsilon"))
def test_outer_rejects_stage_two_runtime_contract_drift(
    composed_artifacts: SimpleNamespace,
    contract_change: str,
) -> None:
    query_norm = composed_artifacts.norm
    epsilon = 1e-5
    if contract_change == "query_norm":
        query_norm = query_norm.clone()
        query_norm[0] = 1.0078125
    else:
        epsilon = 2e-5
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="exactly match",
    ):
        composed_fold.compile_lfm2_composed_qk_sequence_effective_attention_fold(
            composed_qk=composed_artifacts.composed,
            fit_value_output_artifact=composed_artifacts.value_output,
            upstream_fit_cases=composed_artifacts.cases,
            actual_child_contexts=composed_artifacts.actual_contexts,
            target_value_weight_template=composed_artifacts.value_template,
            target_output_weight_template=composed_artifacts.output_template,
            query_norm_weight=query_norm,
            key_norm_weight=composed_artifacts.norm,
            norm_epsilon=epsilon,
        )


def test_outer_rejects_a_different_upstream_fit_bundle(
    composed_artifacts: SimpleNamespace,
) -> None:
    altered_cases = list(composed_artifacts.cases)
    altered_cases[0] = replace(
        altered_cases[0],
        target_pre_attention_residual=(
            altered_cases[0].target_pre_attention_residual + 0.125
        ),
    )
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="exact full upstream FIT case bundle",
    ):
        composed_fold.compile_lfm2_composed_qk_sequence_effective_attention_fold(
            composed_qk=composed_artifacts.composed,
            fit_value_output_artifact=composed_artifacts.value_output,
            upstream_fit_cases=tuple(altered_cases),
            actual_child_contexts=composed_artifacts.actual_contexts,
            target_value_weight_template=composed_artifacts.value_template,
            target_output_weight_template=composed_artifacts.output_template,
            query_norm_weight=composed_artifacts.norm,
            key_norm_weight=composed_artifacts.norm,
            norm_epsilon=1e-5,
        )


def test_endpoint_residual_fifth_arm_is_an_authenticated_4x_qkvo_chain(
    composed_artifacts: SimpleNamespace,
    endpoint_residual_artifacts: SimpleNamespace,
) -> None:
    fifth = endpoint_residual_artifacts.fifth
    assert fifth.accepted
    assert fifth.candidate_id == (
        composed_fold.ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
    )
    assert fifth.verify_authentication() == fifth.artifact_sha256
    assert len(fifth.blocks) == 2
    assert fifth.report["outer_block_count"] == 2
    assert fifth.report["fit_check_endpoint_and_route_nonregressing"]
    assert fifth.report["fit_check_first_metric_evaluation_after_all_outer_blocks"]
    assert fifth.report["fit_check_triggered_refit"] is False
    assert fifth.report["fit_check_role"] == (
        "reused_development_acceptance_gate_not_unbiased_validation"
    )
    assert fifth.report["unbiased_generalization_claimed"] is False
    assert fifth.report["fresh_whole_child_intelligence_validation_required"]
    assert fifth.report["proof_scope"] == (
        "source_bound_local_attention_operator_fold_not_whole_child_intelligence"
    )
    assert fifth.report["all_parent_probability_targets_from_authenticated_cache"]
    assert fifth.report["legacy_recomputed_parent_probability_case_count"] == 0
    assert (
        fifth.final_fit_metrics.endpoint_relative_rmse
        < fifth.baseline_fit_metrics.endpoint_relative_rmse
    )
    tolerance = fifth.policy.endpoint_policy.metric_regression_tolerance
    assert (
        fifth.final_fit_check_metrics.endpoint_relative_rmse
        <= fifth.baseline_fit_check_metrics.endpoint_relative_rmse + tolerance
    )
    assert (
        fifth.final_fit_check_metrics.probability_relative_rmse
        <= fifth.baseline_fit_check_metrics.probability_relative_rmse + tolerance
    )
    assert (
        fifth.final_fit_check_metrics.js_divergence_mean
        <= fifth.baseline_fit_check_metrics.js_divergence_mean + tolerance
    )
    assert (
        fifth.final_fit_check_metrics.js_divergence_worst
        <= fifth.baseline_fit_check_metrics.js_divergence_worst + tolerance
    )
    assert (
        fifth.final_fit_check_metrics.top1_route_agreement_fraction + tolerance
        >= fifth.baseline_fit_check_metrics.top1_route_agreement_fraction
    )
    source_parameters = sum(
        tensor.numel()
        for tensor in (
            composed_artifacts.source.query,
            composed_artifacts.source.key,
            composed_artifacts.source.value,
            composed_artifacts.source.output,
        )
    )
    target_parameters = sum(tensor.numel() for tensor in fifth.state_dict().values())
    assert source_parameters == 192
    assert target_parameters == 48
    assert fifth.source_inventory.logical_parameters == source_parameters
    assert fifth.report["source_logical_parameters"] == source_parameters
    assert fifth.report["target_logical_parameters"] == target_parameters
    assert fifth.report["parameter_compaction_ratio"] == 4.0
    assert fifth.report["source_inventory"] == fifth.source_inventory.as_dict()
    assert all(value == 0 for value in fifth.report["operation_audit"].values())
    _assert_zero_training_counters(fifth.as_dict())

    previous_state_sha = fifth.report["initial_state_sha256"]
    for block in fifth.blocks:
        assert block.verify() == block.block_sha256
        assert block.input_state_sha256 == previous_state_sha
        assert block.fit_bank.partition == "fit"
        assert block.qk_proposal.report["fit_check_accessible"] is False
        assert block.qk_proposal.report["fit_check_cases_consumed"] == 0
        assert block.report["fit_check_accessible"] is False
        assert block.report["fit_check_cases_consumed"] == 0
        assert torch.equal(
            block.endpoint_seed.value_weight,
            block.residual_vo_seed.value_weight,
        )
        assert torch.equal(
            block.endpoint_seed.output_weight,
            block.residual_vo_seed.output_weight,
        )
        assert torch.equal(
            block.qk_proposal.query_weight,
            block.vo_artifact.producer_artifact.query_weight,
        )
        assert torch.equal(
            block.qk_proposal.key_weight,
            block.vo_artifact.producer_artifact.key_weight,
        )
        execution = block.qk_proposal.report["observed_execution_receipt"]
        assert execution["requested_solver_device"] == "cpu"
        assert execution["requested_compute_devices"] == ["cpu"]
        assert execution["case_assignment"] == ("authenticated_bank_order_round_robin")
        assert execution["every_requested_compute_device_observed"]
        assert execution["compute_observations"][0]["case_execution_receipt_sha256"]
        assert execution["solver_observation"]["solver_execution_receipt_sha256"]
        previous_state_sha = block.output_state_sha256
    assert previous_state_sha == fifth.report["final_state_sha256"]


def test_endpoint_cases_use_arithmetic_parent_barycenters_and_direct_endpoints(
    endpoint_residual_artifacts: SimpleNamespace,
) -> None:
    fifth = endpoint_residual_artifacts.fifth
    case = fifth.evaluation_fit_bank.cases[0]
    parent_probabilities = case.parent_probabilities
    expected_heads = []
    for members, weights in zip(
        case.query_groups,
        case.query_group_weights,
        strict=True,
    ):
        expected_heads.append(
            sum(
                parent_probabilities[:, member] * weight
                for member, weight in zip(members, weights, strict=True)
            )
        )
    expected = torch.stack(expected_heads, dim=1)
    expected = torch.where(
        case.target_causal_mask,
        expected,
        torch.zeros_like(expected),
    )
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(
        case.target_probability_barycenter,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    cached = endpoint_residual_artifacts.fit_cases[0]
    direct_endpoint = (
        cached.parent_post_attention_residual_target_space.float()
        - cached.target_pre_attention_residual.float()
    )
    assert torch.equal(case.parent_endpoint_target, direct_endpoint)
    assert case.target_receipt["probability_target_is_mean_of_parent_probabilities"]
    assert not case.target_receipt[
        "probability_target_is_softmax_of_mean_parent_logits"
    ]
    assert case.parent_probability_provenance == (
        "authenticated_cached_parent_attention_probabilities"
    )
    assert torch.equal(
        case.parent_probabilities,
        cached.parent_probabilities.double(),
    )
    check_receipts = set(fifth.evaluation_fit_check_bank.target_receipt_sha256s)
    assert check_receipts
    assert all(
        check_receipts.isdisjoint(block.fit_bank.target_receipt_sha256s)
        for block in fifth.blocks
    )


def test_endpoint_fifth_arm_rejects_unsealed_source_tensor_substitution(
    composed_artifacts: SimpleNamespace,
    endpoint_residual_artifacts: SimpleNamespace,
) -> None:
    altered_query = composed_artifacts.source.query.clone()
    altered_query[0, 0] += torch.tensor(0.25, dtype=torch.bfloat16)
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="Actual source Q/K/V/O tensors differ",
    ):
        composed_fold.compile_lfm2_composed_endpoint_residual_attention_fold(
            composed_qk=composed_artifacts.composed,
            fit_value_output_artifact=(endpoint_residual_artifacts.fit_value_output),
            source_query_weight=altered_query,
            source_key_weight=composed_artifacts.source.key,
            source_value_weight=composed_artifacts.source.value,
            source_output_weight=composed_artifacts.source.output,
            target_query_weight_template=torch.zeros_like(
                composed_artifacts.target.query
            ),
            target_key_weight_template=torch.zeros_like(composed_artifacts.target.key),
            target_value_weight_template=composed_artifacts.value_template,
            target_output_weight_template=composed_artifacts.output_template,
            query_norm_weight=composed_artifacts.norm,
            key_norm_weight=composed_artifacts.norm,
            norm_epsilon=1e-5,
            policy=endpoint_residual_artifacts.fifth.policy,
        )


def test_endpoint_fifth_arm_is_independent_of_blank_target_values(
    composed_artifacts: SimpleNamespace,
    endpoint_residual_artifacts: SimpleNamespace,
) -> None:
    generator = torch.Generator().manual_seed(20260721)
    changed_blank = (
        composed_fold.compile_lfm2_composed_endpoint_residual_attention_fold(
            composed_qk=composed_artifacts.composed,
            fit_value_output_artifact=(endpoint_residual_artifacts.fit_value_output),
            source_query_weight=composed_artifacts.source.query,
            source_key_weight=composed_artifacts.source.key,
            source_value_weight=composed_artifacts.source.value,
            source_output_weight=composed_artifacts.source.output,
            target_query_weight_template=torch.randn(
                composed_artifacts.target.query.shape,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            target_key_weight_template=torch.randn(
                composed_artifacts.target.key.shape,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            target_value_weight_template=torch.randn(
                composed_artifacts.target.value.shape,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            target_output_weight_template=torch.randn(
                composed_artifacts.target.output.shape,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            query_norm_weight=composed_artifacts.norm,
            key_norm_weight=composed_artifacts.norm,
            norm_epsilon=1e-5,
            policy=endpoint_residual_artifacts.fifth.policy,
        )
    )
    baseline = endpoint_residual_artifacts.fifth
    assert changed_blank.verify_authentication() == changed_blank.artifact_sha256
    assert changed_blank.compiled_state_sha256 == baseline.compiled_state_sha256
    assert (
        changed_blank.report["final_state_sha256"]
        == baseline.report["final_state_sha256"]
    )
    assert tuple(changed_blank.state_dict()) == tuple(baseline.state_dict())
    assert all(
        torch.equal(
            changed_blank.state_dict()[name],
            baseline.state_dict()[name],
        )
        for name in baseline.state_dict()
    )
    assert all(
        left.endpoint_seed.target_template_spec_sha256
        == right.endpoint_seed.target_template_spec_sha256
        for left, right in zip(
            changed_blank.blocks,
            baseline.blocks,
            strict=True,
        )
    )


def test_endpoint_fifth_arm_rejects_resealed_forged_final_metrics(
    endpoint_residual_artifacts: SimpleNamespace,
) -> None:
    fifth = endpoint_residual_artifacts.fifth
    forged_metrics = replace(
        fifth.final_fit_metrics,
        endpoint_relative_rmse=0.0,
    )
    report = dict(fifth.report)
    report.pop("artifact_sha256")
    report["final_fit_metrics"] = forged_metrics.as_dict()
    report["artifact_sha256"] = composed_fold._canonical_sha256(report)
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="metrics do not replay",
    ):
        replace(
            fifth,
            final_fit_metrics=forged_metrics,
            report=report,
        )


def test_endpoint_direct_cached_probability_context_mismatch_is_rejected(
    composed_artifacts: SimpleNamespace,
) -> None:
    case = composed_artifacts.cases[0]
    altered_context = case.parent_context.clone()
    altered_context[0, 0, 0, 0] += torch.tensor(
        1.0,
        dtype=altered_context.dtype,
    )
    alien = replace(case, parent_context=altered_context)
    producer = composed_artifacts.composed.producer_artifact
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionAuthenticationError,
        match="trace-cache tolerance",
    ):
        composed_fold._endpoint_case_from_cached_attention(
            alien,
            trajectory_sha256=composed_artifacts.trajectories[0],
            query_groups=producer.query_groups,
            query_group_weights=producer.query_barycentric_weights,
        )


def test_policy_cannot_relax_authoritative_score_gate() -> None:
    with pytest.raises(
        composed_fold.Lfm2ComposedQkAttentionFoldError,
        match="cannot relax",
    ):
        composed_fold.Lfm2ComposedQkAttentionFoldPolicy(
            score_policy=composed_fold.Lfm2QkScoreResidualFoldPolicy(
                maximum_fit_centered_score_relative_rmse=0.20,
            )
        )
