from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields

import pytest
import torch

import neural_forge.hf_lfm2_cached_qk_score_residual_fold as score_fold
from neural_forge.hf_lfm2_cached_qk_score_residual_fold import (
    Lfm2AuthenticatedDenseQkSeed,
    Lfm2CachedQkScoreResidualAuthenticationError,
    Lfm2CachedQkScoreResidualFoldError,
    Lfm2QkScoreResidualCase,
    Lfm2QkScoreResidualFoldPolicy,
    apply_lfm2_finite_radius_rope,
    apply_lfm2_finite_radius_rope_transpose,
    build_lfm2_authenticated_dense_qk_seed,
    build_lfm2_qk_score_residual_case,
    center_lfm2_causal_scores,
    compile_lfm2_cached_qk_score_residual_fold,
    lfm2_continuous_rmsnorm_jvp,
    lfm2_continuous_rmsnorm_vjp,
    seal_lfm2_qk_score_residual_fit_bank,
    seal_lfm2_qk_score_residual_fit_check_bank,
)
from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (
    Lfm2CompiledCpcfAttentionDimensionFold,
    Lfm2CpcfAttentionCaseProvenance,
    Lfm2CpcfAttentionFactorSufficientState,
    Lfm2CpcfAttentionDimensionFoldError,
    Lfm2CpcfAttentionDimensionPolicy,
    Lfm2CpcfAttentionDimensionProductionCase,
    Lfm2CpcfAttentionDimensionRuntime,
    Lfm2CpcfAttentionDimensionTopology,
    Lfm2CpcfBarycentricHeadGroup,
    compile_lfm2_cpcf_attention_dimension_fold,
)
from neural_forge.hf_lfm2_qk_global_gauge_sync import (
    Lfm2CompiledQkGlobalGaugeSync,
    Lfm2QkGlobalGaugeSyncPolicy,
    compile_lfm2_qk_global_gauge_sync,
    seal_lfm2_qk_global_gauge_sync_provenance,
)


@dataclass(slots=True)
class _ScalarRoutingMetricReference:
    """Test-only copy of the former per-row routing metric implementation."""

    case_count: int = 0
    score_error: float = 0.0
    score_target_energy: float = 0.0
    probability_error: float = 0.0
    probability_target_energy: float = 0.0
    js_sum: float = 0.0
    js_worst: float = 0.0
    top1_matches: int = 0
    eligible_rows: int = 0
    leakage: float = 0.0

    def add(
        self,
        case: score_fold._PreparedCase,
        prediction_scores: torch.Tensor,
    ) -> None:
        self.case_count += 1
        prediction_probabilities = score_fold._masked_probabilities(
            prediction_scores,
            case.mask,
        )
        weights64 = case.cell_weights.to(torch.float64)
        score_delta = prediction_scores.to(torch.float64) - case.target_scores.to(
            torch.float64
        )
        self.score_error += float((weights64 * score_delta.square()).sum().item())
        self.score_target_energy += float(
            (weights64 * case.target_scores.to(torch.float64).square()).sum().item()
        )
        probability_delta = prediction_probabilities - case.target_probabilities
        self.probability_error += float(
            (weights64 * probability_delta.square()).sum().item()
        )
        self.probability_target_energy += float(
            (weights64 * case.target_probabilities.square()).sum().item()
        )
        if bool((~case.mask).any().item()):
            self.leakage = max(
                self.leakage,
                float(prediction_probabilities[~case.mask].abs().max().item()),
            )
        flat_prediction = prediction_probabilities.reshape(
            -1,
            int(prediction_probabilities.shape[-1]),
        )
        flat_target = case.target_probabilities.reshape_as(flat_prediction)
        flat_mask = case.mask.reshape_as(flat_prediction)
        for row in range(int(flat_prediction.shape[0])):
            valid = flat_mask[row]
            if int(valid.sum().item()) <= 1:
                continue
            prediction = flat_prediction[row, valid]
            target = flat_target[row, valid]
            midpoint = 0.5 * (prediction + target)
            tiny = torch.finfo(torch.float64).tiny
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
            js_value = float(js.clamp_min(0.0).item())
            self.js_sum += js_value
            self.js_worst = max(self.js_worst, js_value)
            target_max = target.max()
            target_winners = torch.isclose(
                target,
                target_max,
                rtol=1e-10,
                atol=1e-12,
            )
            self.top1_matches += int(
                bool(target_winners[int(prediction.argmax().item())].item())
            )
            self.eligible_rows += 1

    def finish(self) -> score_fold.Lfm2QkRoutingMetrics:
        if self.eligible_rows == 0:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Q/K evidence contains no eligible causal query row."
            )
        tiny = torch.finfo(torch.float64).tiny
        return score_fold.Lfm2QkRoutingMetrics(
            case_count=self.case_count,
            eligible_query_rows=self.eligible_rows,
            centered_score_relative_rmse=math.sqrt(
                self.score_error / max(self.score_target_energy, tiny)
            ),
            probability_relative_rmse=math.sqrt(
                self.probability_error / max(self.probability_target_energy, tiny)
            ),
            js_divergence_mean=self.js_sum / self.eligible_rows,
            js_divergence_worst=self.js_worst,
            top1_route_agreement_fraction=(self.top1_matches / self.eligible_rows),
            maximum_causal_probability_leakage=self.leakage,
        )


def _random_metric_case(
    seed: int,
    *,
    only_one_key_rows: bool = False,
) -> tuple[score_fold._PreparedCase, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch, heads, query_length, key_length = 2, 3, 4, 7
    shape = (batch, heads, query_length, key_length)
    mask = torch.zeros(shape, dtype=torch.bool)
    flat_mask = mask.reshape(-1, key_length)
    for row in range(int(flat_mask.shape[0])):
        valid_count = 1 if only_one_key_rows else 1 + (row % key_length)
        positions = torch.randperm(key_length, generator=generator)[:valid_count]
        flat_mask[row, positions] = True

    target_scores = torch.randn(shape, generator=generator, dtype=torch.float32)
    prediction_scores = torch.randn(shape, generator=generator, dtype=torch.float32)
    target_scores.masked_fill_(~mask, 0.0)
    prediction_scores.masked_fill_(~mask, 0.0)
    if not only_one_key_rows:
        for row in range(0, int(flat_mask.shape[0]), 4):
            valid_positions = torch.nonzero(
                flat_mask[row],
                as_tuple=False,
            ).flatten()
            if int(valid_positions.shape[0]) <= 1:
                continue
            flat_target = target_scores.reshape(-1, key_length)[row]
            flat_prediction = prediction_scores.reshape(-1, key_length)[row]
            flat_target[valid_positions] = -2.0
            flat_prediction[valid_positions] = -1.0
            flat_target[valid_positions[:2]] = 3.0
            flat_prediction[valid_positions[:2]] = 2.0
    target_scores = center_lfm2_causal_scores(target_scores, mask)
    prediction_scores = center_lfm2_causal_scores(prediction_scores, mask)
    target_probabilities = score_fold._masked_probabilities(target_scores, mask)
    informative_query_rows = mask.sum(dim=-1) > 1
    informative_cells = mask & informative_query_rows.unsqueeze(-1)
    cell_weights = torch.where(
        informative_cells,
        torch.rand(shape, generator=generator, dtype=torch.float32) + 0.125,
        torch.zeros(shape, dtype=torch.float32),
    )
    case = score_fold._PreparedCase(
        source=None,
        query_inputs=None,
        key_inputs=None,
        target_scores=target_scores,
        target_probabilities=target_probabilities,
        mask=mask,
        informative_query_rows=informative_query_rows,
        query_cosine=None,
        query_sine=None,
        key_cosine=None,
        key_sine=None,
        cell_weights=cell_weights,
    )
    return case, prediction_scores


def _metric_case_on_device(
    case: score_fold._PreparedCase,
    prediction_scores: torch.Tensor,
    device: torch.device,
) -> tuple[score_fold._PreparedCase, torch.Tensor]:
    return (
        score_fold._PreparedCase(
            source=case.source,
            query_inputs=case.query_inputs,
            key_inputs=case.key_inputs,
            target_scores=case.target_scores.to(device),
            target_probabilities=case.target_probabilities.to(device),
            mask=case.mask.to(device),
            informative_query_rows=case.informative_query_rows.to(device),
            query_cosine=case.query_cosine,
            query_sine=case.query_sine,
            key_cosine=case.key_cosine,
            key_sine=case.key_sine,
            cell_weights=case.cell_weights.to(device),
        ),
        prediction_scores.to(device),
    )


@pytest.mark.parametrize("seed", range(8))
def test_vectorized_routing_metrics_are_bit_exact_to_scalar_reference(
    seed: int,
) -> None:
    vectorized = score_fold._RoutingMetricAccumulator()
    scalar = _ScalarRoutingMetricReference()
    for case_offset in range(3):
        case, prediction_scores = _random_metric_case(
            seed * 17 + case_offset,
        )
        vectorized.add(case, prediction_scores)
        scalar.add(case, prediction_scores)
        for metric_field in fields(_ScalarRoutingMetricReference):
            assert getattr(vectorized, metric_field.name) == getattr(
                scalar,
                metric_field.name,
            )
    assert vectorized.finish() == scalar.finish()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("seed", range(2))
def test_cuda_vectorized_routing_metrics_are_bit_exact_to_scalar_reference(
    seed: int,
) -> None:
    case, prediction_scores = _random_metric_case(seed + 1000)
    case, prediction_scores = _metric_case_on_device(
        case,
        prediction_scores,
        torch.device("cuda:0"),
    )
    vectorized = score_fold._RoutingMetricAccumulator()
    scalar = _ScalarRoutingMetricReference()
    vectorized.add(case, prediction_scores)
    scalar.add(case, prediction_scores)
    for metric_field in fields(_ScalarRoutingMetricReference):
        assert getattr(vectorized, metric_field.name) == getattr(
            scalar,
            metric_field.name,
        )
    assert vectorized.finish() == scalar.finish()


def test_vectorized_routing_metrics_preserve_zero_informative_row_failure() -> None:
    case, prediction_scores = _random_metric_case(
        101,
        only_one_key_rows=True,
    )
    vectorized = score_fold._RoutingMetricAccumulator()
    scalar = _ScalarRoutingMetricReference()
    vectorized.add(case, prediction_scores)
    scalar.add(case, prediction_scores)
    for metric_field in fields(_ScalarRoutingMetricReference):
        assert getattr(vectorized, metric_field.name) == getattr(
            scalar,
            metric_field.name,
        )
    with pytest.raises(
        Lfm2CachedQkScoreResidualFoldError,
        match="no eligible causal query row",
    ):
        vectorized.finish()
    with pytest.raises(
        Lfm2CachedQkScoreResidualFoldError,
        match="no eligible causal query row",
    ):
        scalar.finish()


_SOURCE_SPLIT_SHA = "1" * 64
_SOURCE_RUNTIME_SHA = Lfm2CpcfAttentionDimensionRuntime(
    source_query_norm_weight=torch.tensor(
        [1.0, 0.75],
        dtype=torch.bfloat16,
    ),
    source_key_norm_weight=torch.tensor(
        [0.875, 1.125],
        dtype=torch.bfloat16,
    ),
    target_query_norm_weight=torch.tensor(
        [1.0, 0.75],
        dtype=torch.bfloat16,
    ),
    target_key_norm_weight=torch.tensor(
        [0.875, 1.125],
        dtype=torch.bfloat16,
    ),
    target_query_weight_template=torch.zeros((4, 4), dtype=torch.bfloat16),
    target_key_weight_template=torch.zeros((2, 4), dtype=torch.bfloat16),
).source_runtime_sha256


@dataclass(frozen=True)
class _SeedTemplate:
    query_weight: torch.Tensor
    key_weight: torch.Tensor
    query_norm_weight: torch.Tensor
    key_norm_weight: torch.Tensor
    head_dim: int = 2
    norm_epsilon: float = 1e-5

    @property
    def hidden_size(self) -> int:
        return int(self.query_weight.shape[1])

    @property
    def query_heads(self) -> int:
        return int(self.query_weight.shape[0]) // self.head_dim

    @property
    def key_value_heads(self) -> int:
        return int(self.key_weight.shape[0]) // self.head_dim

    @property
    def queries_per_key_value(self) -> int:
        return self.query_heads // self.key_value_heads


def _seed(*, weight_scale: float = 1.0) -> _SeedTemplate:
    return _SeedTemplate(
        query_weight=(
            torch.tensor(
                [
                    [0.80, 0.10, 0.20, -0.10],
                    [0.05, 0.75, -0.10, 0.20],
                    [0.65, 0.25, 0.15, 0.05],
                    [0.15, 0.55, -0.05, 0.25],
                ],
                dtype=torch.float32,
            )
            * weight_scale
        ).to(torch.bfloat16),
        key_weight=(
            torch.tensor(
                [
                    [0.70, 0.10, 0.25, -0.05],
                    [0.10, 0.65, -0.10, 0.20],
                ],
                dtype=torch.float32,
            )
            * weight_scale
        ).to(torch.bfloat16),
        query_norm_weight=torch.tensor(
            [1.0, 0.75],
            dtype=torch.bfloat16,
        ),
        key_norm_weight=torch.tensor(
            [0.875, 1.125],
            dtype=torch.bfloat16,
        ),
    )


def _planted_weights(
    seed: Lfm2AuthenticatedDenseQkSeed,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = seed.query_weight.float().clone()
    key = seed.key_weight.float().clone()
    query += scale * torch.tensor(
        [
            [0.04, -0.02, 0.02, 0.00],
            [-0.03, 0.02, 0.01, -0.02],
            [0.02, 0.03, -0.02, 0.01],
            [-0.02, 0.01, 0.03, -0.01],
        ]
    )
    key += scale * torch.tensor(
        [
            [-0.03, 0.02, 0.01, 0.02],
            [0.02, -0.02, 0.03, -0.01],
        ]
    )
    return query.to(torch.bfloat16), key.to(torch.bfloat16)


def _rope(length: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(offset, offset + length, dtype=torch.float32)
    angle = positions * 0.37
    cosine = torch.cos(angle).reshape(1, length, 1).repeat(1, 1, 2)
    sine = torch.sin(angle).reshape(1, length, 1).repeat(1, 1, 2)
    return cosine.to(torch.bfloat16), sine.to(torch.bfloat16)


def _runtime_rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    direction = value.float()
    direction = direction * torch.rsqrt(
        direction.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return direction.to(value.dtype) * weight.to(value.dtype)


def _replay(
    seed: Lfm2AuthenticatedDenseQkSeed,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    query_cosine: torch.Tensor,
    query_sine: torch.Tensor,
    key_cosine: torch.Tensor,
    key_sine: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_length, _hidden = query_input.shape
    key_length = key_input.shape[1]
    query_linear = (
        (query_input.to(torch.bfloat16) @ query_weight.to(torch.bfloat16).T)
        .reshape(batch, query_length, seed.query_heads, seed.head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_linear = (
        (key_input.to(torch.bfloat16) @ key_weight.to(torch.bfloat16).T)
        .reshape(batch, key_length, seed.key_value_heads, seed.head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    query = apply_lfm2_finite_radius_rope(
        _runtime_rmsnorm(
            query_linear,
            seed.query_norm_weight,
            seed.norm_epsilon,
        ),
        query_cosine,
        query_sine,
    )
    key = apply_lfm2_finite_radius_rope(
        _runtime_rmsnorm(
            key_linear,
            seed.key_norm_weight,
            seed.norm_epsilon,
        ),
        key_cosine,
        key_sine,
    )
    scores = torch.matmul(
        query,
        key.repeat_interleave(seed.queries_per_key_value, dim=1).transpose(
            -2,
            -1,
        ),
    ) / math.sqrt(seed.head_dim)
    centered = center_lfm2_causal_scores(scores, mask)
    probabilities = torch.softmax(
        centered.double().masked_fill(~mask, -torch.inf),
        dim=-1,
    )
    probabilities = torch.where(
        mask,
        probabilities,
        torch.zeros_like(probabilities),
    )
    return query_linear, key_linear, query, key, centered


def _production_case(
    *,
    seed: Lfm2AuthenticatedDenseQkSeed,
    partition: str,
    ordinal: int,
    planted_query: torch.Tensor,
    planted_key: torch.Tensor,
    input_offset: float,
    target_uses_seed: bool = False,
    trajectory_sha256: str | None = None,
    sequence_length: int = 3,
) -> Lfm2CpcfAttentionDimensionProductionCase:
    if sequence_length not in {1, 3}:
        raise ValueError("The test fixture supports sequence lengths one and three.")
    query_input = torch.tensor(
        [
            [
                [1.0, 0.2, -0.1, 0.4],
                [0.1, 1.1, 0.3, -0.2],
                [0.5, -0.3, 0.9, 0.2],
            ]
        ],
        dtype=torch.float32,
    )[:, :sequence_length]
    query_input = (query_input + input_offset).to(torch.bfloat16)
    key_input = torch.tensor(
        [
            [
                [0.9, 0.1, 0.2, -0.3],
                [0.2, 0.8, -0.2, 0.5],
                [0.4, -0.1, 1.0, 0.1],
            ]
        ],
        dtype=torch.float32,
    )[:, :sequence_length]
    key_input = (key_input - input_offset * 0.5).to(torch.bfloat16)
    query_cosine, query_sine = _rope(sequence_length)
    key_cosine, key_sine = _rope(sequence_length)
    mask = torch.tril(
        torch.ones(
            (1, 1, sequence_length, sequence_length),
            dtype=torch.bool,
        )
    ).expand(
        -1,
        seed.query_heads,
        -1,
        -1,
    )
    target_query = seed.query_weight if target_uses_seed else planted_query
    target_key = seed.key_weight if target_uses_seed else planted_key
    (
        query_linear,
        key_linear,
        query,
        key,
        target_scores,
    ) = _replay(
        seed,
        target_query,
        target_key,
        query_input,
        key_input,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        mask,
    )
    target_probabilities = torch.softmax(
        target_scores.double().masked_fill(~mask, -torch.inf),
        dim=-1,
    )
    target_probabilities = torch.where(
        mask,
        target_probabilities,
        torch.zeros_like(target_probabilities),
    )
    provenance = Lfm2CpcfAttentionCaseProvenance(
        partition=partition,
        trajectory_sha256=trajectory_sha256 or f"{ordinal + 10:064x}",
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=_SOURCE_RUNTIME_SHA,
        case_ordinal=ordinal,
    )
    production = Lfm2CpcfAttentionDimensionProductionCase(
        target_query_residual_inputs=query_input,
        target_key_value_residual_inputs=key_input,
        parent_query_linear=query_linear,
        parent_key_linear=key_linear,
        parent_query_post_rope=query,
        parent_key_post_rope=key,
        parent_centered_scores=target_scores,
        parent_probabilities=target_probabilities,
        query_rope_cos=query_cosine,
        query_rope_sin=query_sine,
        key_rope_cos=key_cosine,
        key_rope_sin=key_sine,
        causal_mask=mask,
        query_row_sqrt_weights=torch.ones(
            (1, sequence_length),
            dtype=torch.float32,
        ),
        key_value_row_sqrt_weights=torch.ones(
            (1, sequence_length),
            dtype=torch.float32,
        ),
        provenance=provenance,
    )
    return production


def _upstream_runtime(
    seed: Lfm2AuthenticatedDenseQkSeed,
) -> Lfm2CpcfAttentionDimensionRuntime:
    return Lfm2CpcfAttentionDimensionRuntime(
        source_query_norm_weight=seed.query_norm_weight,
        source_key_norm_weight=seed.key_norm_weight,
        target_query_norm_weight=seed.query_norm_weight,
        target_key_norm_weight=seed.key_norm_weight,
        target_query_weight_template=torch.zeros_like(seed.query_weight),
        target_key_weight_template=torch.zeros_like(seed.key_weight),
        norm_epsilon=seed.norm_epsilon,
    )


def _upstream_topology(
    seed: Lfm2AuthenticatedDenseQkSeed,
) -> Lfm2CpcfAttentionDimensionTopology:
    return Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=seed.query_heads,
        source_key_value_heads=seed.key_value_heads,
        target_query_heads=seed.query_heads,
        target_key_value_heads=seed.key_value_heads,
        source_head_dim=seed.head_dim,
        target_head_dim=seed.head_dim,
    )


def _upstream_policy() -> Lfm2CpcfAttentionDimensionPolicy:
    return Lfm2CpcfAttentionDimensionPolicy(
        minimum_score_spectral_energy_fraction=0.0,
        maximum_fit_centered_score_relative_rmse=None,
        maximum_validation_centered_score_relative_rmse=None,
        maximum_validation_js_divergence=None,
        maximum_validation_route_probability_relative_rmse=None,
        minimum_validation_top1_route_agreement=None,
        anchor_exact_relative_rmse=0.01,
    )


def _compile_upstream(
    seed: _SeedTemplate,
    fit_cases: tuple[Lfm2CpcfAttentionDimensionProductionCase, ...],
    fit_check_cases: tuple[Lfm2CpcfAttentionDimensionProductionCase, ...],
) -> Lfm2CompiledCpcfAttentionDimensionFold:
    topology = _upstream_topology(seed)
    return compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=fit_cases,
        fit_check_cases=fit_check_cases,
        validation_cases=(),
        query_groups=tuple(
            Lfm2CpcfBarycentricHeadGroup((index,), (1.0,))
            for index in range(seed.query_heads)
        ),
        key_value_groups=tuple(
            Lfm2CpcfBarycentricHeadGroup((index,), (1.0,))
            for index in range(seed.key_value_heads)
        ),
        runtime=_upstream_runtime(seed),
        policy=_upstream_policy(),
        ridge=1e-5,
    )


def _compile_global_seed(
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
    source_prior: _SeedTemplate,
) -> Lfm2AuthenticatedDenseQkSeed:
    runtime = upstream_fold.runtime
    assert runtime is not None
    fit_states = tuple(
        state
        for state in upstream_fold.factor_sufficient_states
        if state.partition == "fit"
    )
    provenance = seal_lfm2_qk_global_gauge_sync_provenance(
        fit_receipt_sha256=upstream_fold.verify_authentication(),
        source_runtime_receipt_sha256=runtime.source_runtime_sha256,
        factor_states=fit_states,
        source_query_weight_prior=source_prior.query_weight,
        source_key_weight_prior=source_prior.key_weight,
        target_query_norm_weight=runtime.target_query_norm_weight,
        target_key_norm_weight=runtime.target_key_norm_weight,
        target_query_weight_template=runtime.target_query_weight_template,
        target_key_weight_template=runtime.target_key_weight_template,
        norm_epsilon=runtime.norm_epsilon,
    )
    global_sync = compile_lfm2_qk_global_gauge_sync(
        factor_states=fit_states,
        target_query_norm_weight=runtime.target_query_norm_weight,
        target_key_norm_weight=runtime.target_key_norm_weight,
        target_query_weight_template=runtime.target_query_weight_template,
        target_key_weight_template=runtime.target_key_weight_template,
        source_query_weight_prior=source_prior.query_weight,
        source_key_weight_prior=source_prior.key_weight,
        norm_epsilon=runtime.norm_epsilon,
        provenance=provenance,
        policy=Lfm2QkGlobalGaugeSyncPolicy(
            sweeps=2,
            source_anchor_prior_ridge=1e6,
            global_prior_ridge=1e6,
            radius_prior_strength=1e6,
        ),
    )
    return build_lfm2_authenticated_dense_qk_seed(
        upstream_fold=upstream_fold,
        global_sync=global_sync,
    )


def _build_residual_cases(
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
    production_cases: tuple[Lfm2CpcfAttentionDimensionProductionCase, ...],
) -> tuple[Lfm2QkScoreResidualCase, ...]:
    states = {
        (value.partition, value.case_ordinal, value.evidence_sha256): value
        for value in upstream_fold.factor_sufficient_states
    }
    return tuple(
        build_lfm2_qk_score_residual_case(
            production_case=production,
            upstream_fold=upstream_fold,
            factor_state=states[
                (
                    production.provenance.partition,
                    production.provenance.case_ordinal,
                    production.evidence_sha256,
                )
            ],
        )
        for production in production_cases
    )


def _build_partition_pair(
    seed: _SeedTemplate,
    fit_production: tuple[Lfm2CpcfAttentionDimensionProductionCase, ...],
    fit_check_production: tuple[
        Lfm2CpcfAttentionDimensionProductionCase,
        ...,
    ],
) -> tuple[
    Lfm2CompiledCpcfAttentionDimensionFold,
    Lfm2AuthenticatedDenseQkSeed,
    tuple[Lfm2QkScoreResidualCase, ...],
    tuple[Lfm2QkScoreResidualCase, ...],
]:
    upstream_fold = _compile_upstream(
        seed,
        fit_production,
        fit_check_production,
    )
    authenticated_seed = _compile_global_seed(upstream_fold, seed)
    return (
        upstream_fold,
        authenticated_seed,
        _build_residual_cases(upstream_fold, fit_production),
        _build_residual_cases(upstream_fold, fit_check_production),
    )


def _banks(
    *,
    target_uses_seed_on_fit_check: bool = False,
    plant_scale: float = 1.0,
    seed_value: _SeedTemplate | None = None,
) -> tuple[
    Lfm2AuthenticatedDenseQkSeed,
    object,
    object,
]:
    seed_template = seed_value or _seed()
    planted_query, planted_key = _planted_weights(
        seed_template,
        scale=plant_scale,
    )
    fit_production = (
        _production_case(
            seed=seed_template,
            partition="fit",
            ordinal=0,
            planted_query=planted_query,
            planted_key=planted_key,
            input_offset=0.0,
        ),
        _production_case(
            seed=seed_template,
            partition="fit",
            ordinal=1,
            planted_query=planted_query,
            planted_key=planted_key,
            input_offset=0.17,
        ),
    )
    fit_check_production = (
        _production_case(
            seed=seed_template,
            partition="fit_check",
            ordinal=2,
            planted_query=planted_query,
            planted_key=planted_key,
            input_offset=-0.13,
            target_uses_seed=target_uses_seed_on_fit_check,
        ),
    )
    upstream_fold = _compile_upstream(
        seed_template,
        fit_production,
        fit_check_production,
    )
    seed = _compile_global_seed(upstream_fold, seed_template)
    fit = _build_residual_cases(upstream_fold, fit_production)
    fit_check = _build_residual_cases(upstream_fold, fit_check_production)
    return (
        seed,
        seal_lfm2_qk_score_residual_fit_bank(
            fit,
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=seed.source_runtime_sha256,
            source_artifact_sha256=seed.source_artifact_sha256,
        ),
        seal_lfm2_qk_score_residual_fit_check_bank(
            fit_check,
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=seed.source_runtime_sha256,
            source_artifact_sha256=seed.source_artifact_sha256,
        ),
    )


def test_continuous_rmsnorm_jvp_matches_finite_difference_and_vjp() -> None:
    raw = torch.tensor(
        [[[[-0.7, 1.3, 0.4, 0.8]]]],
        dtype=torch.float64,
    )
    tangent = torch.tensor(
        [[[[0.2, -0.1, 0.3, -0.4]]]],
        dtype=torch.float64,
    )
    gain = torch.tensor([0.7, 1.2, 0.9, 1.1], dtype=torch.float64)
    epsilon = 1e-4

    def continuous(value: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
        return gain * value / radius

    step = 1e-4
    finite_difference = (
        continuous(raw + step * tangent) - continuous(raw - step * tangent)
    ) / (2.0 * step)
    jvp = lfm2_continuous_rmsnorm_jvp(raw, tangent, gain, epsilon)
    assert torch.allclose(jvp.double(), finite_difference, rtol=2e-4, atol=2e-5)

    cotangent = torch.tensor(
        [[[[0.5, -0.3, 0.7, 0.2]]]],
        dtype=torch.float64,
    )
    vjp = lfm2_continuous_rmsnorm_vjp(raw, cotangent, gain, epsilon)
    assert torch.allclose(
        (jvp.double() * cotangent).sum(),
        (tangent * vjp.double()).sum(),
        rtol=2e-5,
        atol=2e-6,
    )


def test_finite_radius_rope_transpose_is_adjoint_not_inverse() -> None:
    value = torch.tensor(
        [[[[0.3, -0.7, 1.1, 0.2], [0.5, 0.4, -0.2, 0.8]]]],
        dtype=torch.float32,
    )
    cotangent = torch.tensor(
        [[[[0.2, 0.6, -0.5, 0.9], [-0.3, 0.7, 0.4, -0.1]]]],
        dtype=torch.float32,
    )
    cosine = torch.tensor([[[0.75, 0.50, 0.75, 0.50], [0.60, 0.80, 0.60, 0.80]]])
    sine = torch.tensor([[[0.55, 0.45, 0.55, 0.45], [0.70, 0.35, 0.70, 0.35]]])
    assert not torch.allclose(cosine.square() + sine.square(), torch.ones_like(cosine))
    forward = apply_lfm2_finite_radius_rope(value, cosine, sine)
    transpose = apply_lfm2_finite_radius_rope_transpose(
        cotangent,
        cosine,
        sine,
    )
    assert torch.allclose(
        (forward * cotangent).sum(),
        (value * transpose).sum(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(
        apply_lfm2_finite_radius_rope(transpose, cosine, sine),
        cotangent,
    )


def test_weight_then_center_adjoint_and_gqa_shared_key_vjp() -> None:
    seed, fit_bank, _fit_check = _banks()
    policy = Lfm2QkScoreResidualFoldPolicy(
        maximum_sweeps=1,
        maximum_pcg_iterations=4,
        minimum_pcg_iterations=1,
        pcg_restart_iterations=4,
    )
    prepared = tuple(
        score_fold._prepare_evidence_case(
            value,
            seed,
            policy,
            device=torch.device("cpu"),
        )
        for value in fit_bank.cases[:1]
    )
    score_fold._attach_linearization(
        prepared,
        seed,
        seed.query_weight,
        seed.key_weight,
    )
    operator = score_fold.Lfm2QkScoreResidualLinearOperator(
        seed=seed,
        prepared_cases=prepared,
        current_query_weight=seed.query_weight.float(),
        current_key_weight=seed.key_weight.float(),
        ridge=1e-3,
    )
    vector = torch.linspace(
        -0.02,
        0.03,
        seed.query_weight.numel() + seed.key_weight.numel(),
    )
    jvp = operator.jvp(vector)
    weighted = tuple(
        case.cell_weights
        * torch.linspace(
            -0.4,
            0.6,
            case.cell_weights.numel(),
        ).reshape_as(case.cell_weights)
        for case in prepared
    )
    vjp = operator.vjp(weighted)
    left = sum(
        float((value * cotangent).sum().item())
        for value, cotangent in zip(jvp, weighted, strict=True)
    )
    right = float((vector * vjp).sum().item())
    assert left == pytest.approx(right, rel=2e-4, abs=2e-5)

    # Both Q heads share the one KV head; a cotangent on either head must
    # contribute to that same K projection block.
    first_head = torch.zeros_like(prepared[0].cell_weights)
    second_head = torch.zeros_like(prepared[0].cell_weights)
    first_head[:, 0] = 1.0
    second_head[:, 1] = 1.0
    _, key_from_first = score_fold._score_vjp(prepared[0], seed, first_head)
    _, key_from_second = score_fold._score_vjp(prepared[0], seed, second_head)
    _, key_from_both = score_fold._score_vjp(
        prepared[0],
        seed,
        first_head + second_head,
    )
    assert torch.allclose(
        key_from_both,
        key_from_first + key_from_second,
        rtol=1e-5,
        atol=1e-5,
    )


def test_matrix_free_normal_is_symmetric_positive_with_ridge() -> None:
    seed, fit_bank, _fit_check = _banks()
    policy = Lfm2QkScoreResidualFoldPolicy(
        maximum_sweeps=1,
        maximum_pcg_iterations=4,
        minimum_pcg_iterations=1,
        pcg_restart_iterations=4,
    )
    prepared = tuple(
        score_fold._prepare_evidence_case(
            value,
            seed,
            policy,
            device=torch.device("cpu"),
        )
        for value in fit_bank.cases
    )
    score_fold._attach_linearization(
        prepared,
        seed,
        seed.query_weight,
        seed.key_weight,
    )
    operator = score_fold.Lfm2QkScoreResidualLinearOperator(
        seed=seed,
        prepared_cases=prepared,
        current_query_weight=seed.query_weight.float(),
        current_key_weight=seed.key_weight.float(),
        ridge=1e-2,
    )
    size = seed.query_weight.numel() + seed.key_weight.numel()
    left = torch.linspace(-0.3, 0.5, size)
    right = torch.linspace(0.6, -0.2, size)
    normal_left = operator.normal(left)
    normal_right = operator.normal(right)
    assert float((left * normal_right).sum()) == pytest.approx(
        float((normal_left * right).sum()),
        rel=3e-4,
        abs=3e-5,
    )
    assert float((left * normal_left).sum()) > 0.0
    streaming = score_fold._StreamingQkScoreResidualLinearOperator(
        seed=seed,
        evidence_cases=fit_bank.cases,
        current_query_weight=seed.query_weight.float(),
        current_key_weight=seed.key_weight.float(),
        ridge=1e-2,
        policy=policy,
        solver_device=torch.device("cpu"),
        compute_devices=(torch.device("cpu"),),
    )
    assert streaming.residual_energy == pytest.approx(
        operator.residual_energy,
        rel=1e-12,
        abs=1e-12,
    )
    assert torch.allclose(
        streaming.right_hand_side(),
        operator.right_hand_side(),
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.allclose(
        streaming.normal(left),
        normal_left,
        rtol=1e-6,
        atol=1e-7,
    )


def test_compile_improves_fit_and_accepts_frozen_fit_check_without_training() -> None:
    seed, fit_bank, fit_check_bank = _banks()
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        policy=Lfm2QkScoreResidualFoldPolicy(
            ridge=2e-3,
            trust_radius=2.0,
            maximum_sweeps=2,
            maximum_pcg_iterations=96,
            minimum_pcg_iterations=2,
            pcg_restart_iterations=48,
            pcg_relative_tolerance=2e-4,
            metric_regression_tolerance=5e-3,
        ),
    )
    assert result.verify() == result.report["report_sha256"]
    assert result.accepted
    assert result.changed
    assert result.status == "fit_check_accepted"
    assert result.report["fit_check_evaluated"]
    assert not result.report["fit_check_triggered_refit"]
    assert result.report["source_runtime_sha256"] == _SOURCE_RUNTIME_SHA
    execution = result.report["execution"]
    assert {
        name: execution[name]
        for name in (
            "solver_device",
            "compute_devices",
            "case_assignment",
            "compute_device_concurrency",
            "streamed_evidence",
            "maximum_logically_resident_prepared_cases",
            "jvp_vjp_dtype",
            "pcg_vector_dtype",
            "scalar_reduction_dtype",
        )
    } == {
        "solver_device": "cpu",
        "compute_devices": ["cpu"],
        "case_assignment": "authenticated_bank_order_round_robin",
        "compute_device_concurrency": "serial_round_robin",
        "streamed_evidence": True,
        "maximum_logically_resident_prepared_cases": 1,
        "jvp_vjp_dtype": "torch.float32",
        "pcg_vector_dtype": "torch.float32",
        "scalar_reduction_dtype": "torch.float64",
    }
    observed = execution["observed_execution_receipt"]
    assert observed["schema"] == score_fold.LFM2_QK_OBSERVED_EXECUTION_RECEIPT_SCHEMA
    assert observed["requested_solver_device"] == "cpu"
    assert observed["requested_compute_devices"] == ["cpu"]
    assert observed["every_requested_compute_device_observed"]
    assert observed["solver_device_observed"]
    assert observed["distinct_case_receipts"] == [fit_bank.cases[0].evidence_sha256]
    assert observed["distinct_case_receipt_count"] == 1
    assert observed["cached_evidence_analytical_tensor_operation_count"] == 1
    assert observed["solver_tensor_operation_count"] == 1
    observed_body = dict(observed)
    observed_seal = observed_body.pop("receipt_sha256")
    assert score_fold._canonical_sha256(observed_body) == observed_seal
    compute_observation = observed["compute_observations"][0]
    compute_body = dict(compute_observation)
    compute_seal = compute_body.pop("case_execution_receipt_sha256")
    assert score_fold._canonical_sha256(compute_body) == compute_seal
    assert compute_observation["intended_logical_device"] == "cpu"
    assert compute_observation["observed_logical_device"] == "cpu"
    assert compute_observation["cuda_uuid"] is None
    assert compute_observation["cuda_name"] is None
    assert compute_observation["analytical_scalar"] > 0.0
    solver_observation = observed["solver_observation"]
    solver_body = dict(solver_observation)
    solver_seal = solver_body.pop("solver_execution_receipt_sha256")
    assert score_fold._canonical_sha256(solver_body) == solver_seal
    assert solver_observation["intended_logical_device"] == "cpu"
    assert solver_observation["observed_logical_device"] == "cpu"
    assert solver_observation["analytical_scalar"] > 0.0
    sealed_report_body = copy.deepcopy(dict(result.report))
    sealed_report_sha256 = sealed_report_body.pop("report_sha256")
    assert score_fold._canonical_sha256(sealed_report_body) == sealed_report_sha256
    sealed_report_body["execution"]["observed_execution_receipt"][
        "compute_observations"
    ][0]["analytical_scalar"] += 1.0
    assert score_fold._canonical_sha256(sealed_report_body) != sealed_report_sha256
    assert (
        result.report["fit_metrics"]["centered_score_relative_rmse"]
        < result.report["initial_fit_metrics"]["centered_score_relative_rmse"]
    )
    audit = result.report["operation_audit"]
    assert audit == {
        "fit_cases_consumed": 2,
        "fit_check_cases_consumed": 1,
        "selection_cases_consumed": 0,
        "true_validation_cases_consumed": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "target_mutations_during_compile": 0,
        "published_state_cast_before_fit_check": True,
    }


def test_observed_execution_receipt_forces_synthetic_cpu_tensor_work() -> None:
    seed, fit_bank, _fit_check_bank = _banks()
    receipt = score_fold._build_observed_execution_receipt(
        seed=seed,
        fit_cases=fit_bank.cases,
        solver_device=torch.device("cpu"),
        compute_devices=(torch.device("cpu"),),
    )
    assert receipt["requested_compute_device_count"] == 1
    assert receipt["observed_compute_device_count"] == 1
    assert receipt["per_device_case_counts"] == {"cpu": 1}
    assert receipt["distinct_case_execution_receipt_count"] == 1
    assert receipt["operation_audit"] == {
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    compute = receipt["compute_observations"][0]
    assert compute["analytical_operation"] == (
        "masked_centered_score_energy_plus_valid_cell_count"
    )
    assert compute["analytical_tensor_element_count"] == int(
        fit_bank.cases[0].target_centered_scores.numel()
    )
    assert compute["forced_host_scalar_read"]
    solver = receipt["solver_observation"]
    assert solver["analytical_operation"] == (
        "dense_seed_qk_energy_plus_parameter_count"
    )
    assert solver["analytical_tensor_element_count"] == int(
        seed.query_weight.numel() + seed.key_weight.numel()
    )
    assert solver["forced_host_scalar_read"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA execution identity requires a live CUDA runtime.",
)
def test_observed_execution_receipt_records_live_cuda_uuid_and_name() -> None:
    seed, fit_bank, _fit_check_bank = _banks()
    device_count = min(int(torch.cuda.device_count()), len(fit_bank.cases))
    devices = tuple(torch.device(f"cuda:{index}") for index in range(device_count))
    receipt = score_fold._build_observed_execution_receipt(
        seed=seed,
        fit_cases=fit_bank.cases,
        solver_device=devices[-1],
        compute_devices=devices,
    )
    assert receipt["requested_compute_devices"] == [str(device) for device in devices]
    assert receipt["distinct_case_receipt_count"] == device_count
    assert receipt["distinct_case_execution_receipt_count"] == device_count
    for index, observation in enumerate(receipt["compute_observations"]):
        properties = torch.cuda.get_device_properties(index)
        assert observation["intended_logical_device"] == f"cuda:{index}"
        assert observation["observed_logical_device"] == f"cuda:{index}"
        assert observation["cuda_uuid"] == str(properties.uuid)
        assert observation["cuda_name"] == str(properties.name)
        assert observation["analytical_scalar"] > 0.0
    solver = receipt["solver_observation"]
    solver_properties = torch.cuda.get_device_properties(devices[-1].index)
    assert solver["observed_logical_device"] == str(devices[-1])
    assert solver["cuda_uuid"] == str(solver_properties.uuid)
    assert solver["cuda_name"] == str(solver_properties.name)
    assert solver["analytical_scalar"] > 0.0


@pytest.mark.parametrize("uninformative_partition", ["fit", "fit_check", "both"])
def test_one_key_only_partition_is_rejected_independently_before_solve(
    uninformative_partition: str,
) -> None:
    seed_template = _seed()
    planted_query, planted_key = _planted_weights(seed_template)
    fit_length = 1 if uninformative_partition in {"fit", "both"} else 3
    fit_check_length = 1 if uninformative_partition in {"fit_check", "both"} else 3
    upstream, seed, fit_cases, fit_check_cases = _build_partition_pair(
        seed_template,
        (
            _production_case(
                seed=seed_template,
                partition="fit",
                ordinal=0,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.0,
                sequence_length=fit_length,
            ),
            _production_case(
                seed=seed_template,
                partition="fit",
                ordinal=1,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.17,
                sequence_length=fit_length,
            ),
        ),
        (
            _production_case(
                seed=seed_template,
                partition="fit_check",
                ordinal=2,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=-0.13,
                sequence_length=fit_check_length,
            ),
        ),
    )
    assert upstream is seed.upstream_fold
    fit_bank = seal_lfm2_qk_score_residual_fit_bank(
        fit_cases,
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    fit_check_bank = seal_lfm2_qk_score_residual_fit_check_bank(
        fit_check_cases,
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="independently contain at least one causal query row",
    ):
        compile_lfm2_cached_qk_score_residual_fold(
            seed=seed,
            fit_bank=fit_bank,
            fit_check_bank=fit_check_bank,
        )


def test_mixed_causal_banks_exclude_one_key_rows_but_retain_provenance() -> None:
    seed, fit_bank, fit_check_bank = _banks()
    policy = Lfm2QkScoreResidualFoldPolicy()
    first = fit_bank.cases[0]
    prepared = score_fold._prepare_evidence_case(
        first,
        seed,
        policy,
        device=torch.device("cpu"),
    )
    assert first.target_causal_mask.sum(dim=-1).tolist() == [[[1, 2, 3], [1, 2, 3]]]
    assert not bool(prepared.cell_weights[:, :, 0, :].any().item())
    assert bool(prepared.cell_weights[:, :, 1:, :].any().item())
    assert len(first.solver_row_sha256s) == 5
    assert len(first.trajectory_row_sha256s) == 6

    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
    )
    assert result.fit_metrics.eligible_query_rows == 8
    assert result.fit_check_metrics is not None
    assert result.fit_check_metrics.eligible_query_rows == 4


def test_exact_seed_target_is_authenticated_no_op_without_fit_check_access() -> None:
    seed_template = _seed()
    planted_query, planted_key = _planted_weights(seed_template)
    _upstream, seed, (fit_case, _fit_extra), (fit_check_case,) = _build_partition_pair(
        seed_template,
        (
            _production_case(
                seed=seed_template,
                partition="fit",
                ordinal=0,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.0,
                target_uses_seed=True,
            ),
            _production_case(
                seed=seed_template,
                partition="fit",
                ordinal=1,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=-0.11,
                target_uses_seed=True,
            ),
        ),
        (
            _production_case(
                seed=seed_template,
                partition="fit_check",
                ordinal=2,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.2,
                target_uses_seed=True,
            ),
        ),
    )
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=seal_lfm2_qk_score_residual_fit_bank(
            (fit_case,),
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=seed.source_runtime_sha256,
            source_artifact_sha256=seed.source_artifact_sha256,
        ),
        fit_check_bank=seal_lfm2_qk_score_residual_fit_check_bank(
            (fit_check_case,),
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=seed.source_runtime_sha256,
            source_artifact_sha256=seed.source_artifact_sha256,
        ),
        policy=Lfm2QkScoreResidualFoldPolicy(
            maximum_sweeps=1,
            maximum_pcg_iterations=8,
            minimum_pcg_iterations=1,
            pcg_restart_iterations=8,
        ),
    )
    assert result.accepted
    assert not result.changed
    assert torch.equal(result.query_weight, seed.query_weight)
    assert torch.equal(result.key_weight, seed.key_weight)
    assert result.status == "quality_gated_seed_no_op"
    assert result.report["fit_check_evaluated"]
    assert result.report["operation_audit"]["fit_check_cases_consumed"] == 1


def test_pcg_nonconvergence_returns_exact_seed_and_never_opens_fit_check() -> None:
    seed, fit_bank, fit_check_bank = _banks()
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        policy=Lfm2QkScoreResidualFoldPolicy(
            maximum_sweeps=1,
            maximum_pcg_iterations=1,
            minimum_pcg_iterations=1,
            pcg_restart_iterations=1,
            pcg_relative_tolerance=1e-15,
        ),
    )
    assert not result.accepted
    assert not result.changed
    assert result.status == "pcg_nonconverged_seed_fallback"
    assert torch.equal(result.query_weight, seed.query_weight)
    assert torch.equal(result.key_weight, seed.key_weight)
    assert result.report["operation_audit"]["fit_check_cases_consumed"] == 0
    assert not result.report["fit_check_evaluated"]


def test_fit_check_rejection_falls_back_without_refit() -> None:
    seed, fit_bank, fit_check_bank = _banks(
        target_uses_seed_on_fit_check=True,
    )
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        policy=Lfm2QkScoreResidualFoldPolicy(
            ridge=2e-3,
            maximum_sweeps=1,
            maximum_pcg_iterations=96,
            minimum_pcg_iterations=2,
            pcg_restart_iterations=48,
            pcg_relative_tolerance=2e-4,
            metric_regression_tolerance=5e-3,
        ),
    )
    assert not result.accepted
    assert not result.changed
    assert result.status == "fit_check_rejected_seed_fallback"
    assert torch.equal(result.query_weight, seed.query_weight)
    assert torch.equal(result.key_weight, seed.key_weight)
    assert result.report["fit_check_evaluated"]
    assert not result.report["fit_check_triggered_refit"]
    assert len(result.report["sweeps"]) == 1


def test_overlapping_fit_trajectory_is_rejected_before_solve() -> None:
    seed, fit_bank, fit_check_bank = _banks()
    production = fit_check_bank.cases[0].production_case
    overlapping_provenance = Lfm2CpcfAttentionCaseProvenance(
        partition="fit_check",
        trajectory_sha256=fit_bank.trajectory_sha256s[0],
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=_SOURCE_RUNTIME_SHA,
        case_ordinal=production.provenance.case_ordinal,
    )
    overlapping_production = Lfm2CpcfAttentionDimensionProductionCase(
        **{
            name: (
                overlapping_provenance
                if name == "provenance"
                else getattr(production, name)
            )
            for name in production.__dataclass_fields__
            if name != "evidence_sha256"
        }
    )
    with pytest.raises(
        Lfm2CpcfAttentionDimensionFoldError,
        match="trajectory",
    ):
        _compile_upstream(
            seed,
            tuple(case.production_case for case in fit_bank.cases),
            (overlapping_production,),
        )


def test_repeated_trajectory_phases_are_valid_but_duplicate_phase_is_not() -> None:
    seed = _seed()
    planted_query, planted_key = _planted_weights(seed)
    trajectory_sha = "a" * 64
    _upstream, authenticated_seed, (first, second), _fit_check = _build_partition_pair(
        seed,
        (
            _production_case(
                seed=seed,
                partition="fit",
                ordinal=0,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.0,
                trajectory_sha256=trajectory_sha,
            ),
            _production_case(
                seed=seed,
                partition="fit",
                ordinal=1,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.2,
                trajectory_sha256=trajectory_sha,
            ),
        ),
        (
            _production_case(
                seed=seed,
                partition="fit_check",
                ordinal=2,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=-0.2,
            ),
        ),
    )
    bank = seal_lfm2_qk_score_residual_fit_bank(
        (first, second),
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=authenticated_seed.source_runtime_sha256,
        source_artifact_sha256=authenticated_seed.source_artifact_sha256,
    )
    assert bank.trajectory_sha256s == (trajectory_sha, trajectory_sha)
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="trajectory-phase",
    ):
        seal_lfm2_qk_score_residual_fit_bank(
            (first, first),
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=authenticated_seed.source_runtime_sha256,
            source_artifact_sha256=authenticated_seed.source_artifact_sha256,
        )


def test_forged_scaled_global_seed_is_rejected_before_solve() -> None:
    seed, _fit_bank, _fit_check_bank = _banks()
    forged_global = object.__new__(Lfm2CompiledQkGlobalGaugeSync)
    object.__setattr__(
        forged_global,
        "query_weight",
        (seed.query_weight.float() * 1.05).to(torch.bfloat16),
    )
    object.__setattr__(
        forged_global,
        "key_weight",
        (seed.key_weight.float() * 0.95).to(torch.bfloat16),
    )
    object.__setattr__(forged_global, "report", seed.global_sync.report)
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="global Q/K synchronization artifact",
    ):
        build_lfm2_authenticated_dense_qk_seed(
            upstream_fold=seed.upstream_fold,
            global_sync=forged_global,
        )


def test_seed_is_cross_bound_to_score_banks_exact_upstream_fold() -> None:
    seed, _fit_bank, _fit_check_bank = _banks()
    _foreign_seed, foreign_fit, foreign_fit_check = _banks(plant_scale=2.0)
    relabelled_fit = seal_lfm2_qk_score_residual_fit_bank(
        foreign_fit.cases,
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    relabelled_fit_check = seal_lfm2_qk_score_residual_fit_check_bank(
        foreign_fit_check.cases,
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="different authenticated upstream CPCF folds",
    ):
        compile_lfm2_cached_qk_score_residual_fold(
            seed=seed,
            fit_bank=relabelled_fit,
            fit_check_bank=relabelled_fit_check,
        )


def test_raw_authored_seed_weights_are_not_a_public_constructor_path() -> None:
    template = _seed()
    with pytest.raises(TypeError):
        Lfm2AuthenticatedDenseQkSeed(  # type: ignore[call-arg]
            query_weight=template.query_weight,
            key_weight=template.key_weight,
            query_norm_weight=template.query_norm_weight,
            key_norm_weight=template.key_norm_weight,
            head_dim=template.head_dim,
            norm_epsilon=template.norm_epsilon,
            source_artifact_sha256="5" * 64,
            source_runtime_sha256=_SOURCE_RUNTIME_SHA,
        )


def test_raw_authored_seed_seal_is_not_a_constructor_path() -> None:
    seed, _fit_bank, _fit_check_bank = _banks()
    with pytest.raises(TypeError):
        Lfm2AuthenticatedDenseQkSeed(  # type: ignore[call-arg]
            upstream_fold=seed.upstream_fold,
            global_sync=seed.global_sync,
            seed_sha256="5" * 64,
        )


def test_raw_caller_target_scores_cannot_become_production_evidence() -> None:
    _seed_value, fit_bank, _fit_check_bank = _banks()
    source = fit_bank.cases[0]
    tampered_scores = source.target_centered_scores.clone()
    tampered_scores[0, 0, 1, 0] += 0.25
    constructor = {
        field.name: getattr(source, field.name)
        for field in fields(source)
        if field.name not in {"content_sha256", "evidence_sha256"}
    }
    constructor["target_centered_scores"] = tampered_scores
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="Raw or altered target-score evidence",
    ):
        Lfm2QkScoreResidualCase(**constructor)


def test_builder_rejects_wrong_authenticated_factor_member_for_case() -> None:
    _seed_value, fit_bank, _fit_check_bank = _banks()
    first, second = fit_bank.cases
    assert first.upstream_fold is second.upstream_fold
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="partition/ordinal/evidence identity",
    ):
        build_lfm2_qk_score_residual_case(
            production_case=first.production_case,
            upstream_fold=first.upstream_fold,
            factor_state=second.factor_state,
        )


def test_source_head_dim_four_uses_authenticated_target_head_dim_two_rope() -> None:
    source_query_norm = torch.tensor(
        [0.875, 1.0, 1.125, 0.75],
        dtype=torch.bfloat16,
    )
    source_key_norm = torch.tensor(
        [1.125, 0.875, 0.75, 1.0],
        dtype=torch.bfloat16,
    )
    target_seed_template = _seed()
    runtime = Lfm2CpcfAttentionDimensionRuntime(
        source_query_norm_weight=source_query_norm,
        source_key_norm_weight=source_key_norm,
        target_query_norm_weight=target_seed_template.query_norm_weight,
        target_key_norm_weight=target_seed_template.key_norm_weight,
        target_query_weight_template=torch.zeros_like(
            target_seed_template.query_weight
        ),
        target_key_weight_template=torch.zeros_like(target_seed_template.key_weight),
        norm_epsilon=target_seed_template.norm_epsilon,
    )
    topology = Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=2,
        source_key_value_heads=1,
        target_query_heads=2,
        target_key_value_heads=1,
        source_head_dim=4,
        target_head_dim=2,
    )
    source_query_weight = torch.tensor(
        [
            [0.80, 0.10, 0.20, -0.10],
            [0.05, 0.75, -0.10, 0.20],
            [0.35, -0.20, 0.15, 0.30],
            [-0.15, 0.25, 0.40, 0.10],
            [0.65, 0.25, 0.15, 0.05],
            [0.15, 0.55, -0.05, 0.25],
            [0.30, 0.10, -0.25, 0.45],
            [-0.10, 0.35, 0.20, 0.15],
        ],
        dtype=torch.bfloat16,
    )
    source_key_weight = torch.tensor(
        [
            [0.70, 0.10, 0.25, -0.05],
            [0.10, 0.65, -0.10, 0.20],
            [0.25, -0.15, 0.35, 0.10],
            [-0.20, 0.30, 0.05, 0.40],
        ],
        dtype=torch.bfloat16,
    )

    def source_rope(length: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, dtype=torch.float32).reshape(length, 1)
        frequencies = torch.tensor([[0.19, 0.43]], dtype=torch.float32)
        angles = positions * frequencies
        cosine_half = angles.cos().to(torch.bfloat16)
        sine_half = angles.sin().to(torch.bfloat16)
        return (
            torch.cat((cosine_half, cosine_half), dim=-1).unsqueeze(0),
            torch.cat((sine_half, sine_half), dim=-1).unsqueeze(0),
        )

    def source_production(
        *,
        partition: str,
        ordinal: int,
        offset: float,
    ) -> Lfm2CpcfAttentionDimensionProductionCase:
        query_input = (
            torch.tensor(
                [
                    [
                        [1.0, 0.2, -0.1, 0.4],
                        [0.1, 1.1, 0.3, -0.2],
                        [0.5, -0.3, 0.9, 0.2],
                    ]
                ],
                dtype=torch.float32,
            )
            + offset
        ).to(torch.bfloat16)
        key_input = (
            torch.tensor(
                [
                    [
                        [0.9, 0.1, 0.2, -0.3],
                        [0.2, 0.8, -0.2, 0.5],
                        [0.4, -0.1, 1.0, 0.1],
                    ]
                ],
                dtype=torch.float32,
            )
            - offset * 0.5
        ).to(torch.bfloat16)
        query_linear = (
            (query_input @ source_query_weight.T)
            .reshape(1, 3, 2, 4)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        key_linear = (
            (key_input @ source_key_weight.T)
            .reshape(1, 3, 1, 4)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        query_cosine, query_sine = source_rope(3)
        key_cosine, key_sine = source_rope(3)
        query_post = apply_lfm2_finite_radius_rope(
            _runtime_rmsnorm(
                query_linear,
                source_query_norm,
                runtime.norm_epsilon,
            ),
            query_cosine,
            query_sine,
        )
        key_post = apply_lfm2_finite_radius_rope(
            _runtime_rmsnorm(
                key_linear,
                source_key_norm,
                runtime.norm_epsilon,
            ),
            key_cosine,
            key_sine,
        )
        mask = torch.tril(torch.ones((1, 1, 3, 3), dtype=torch.bool)).expand(
            -1, 2, -1, -1
        )
        scores = center_lfm2_causal_scores(
            torch.matmul(
                query_post,
                key_post.repeat_interleave(2, dim=1).transpose(-2, -1),
            )
            / math.sqrt(4),
            mask,
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
        return Lfm2CpcfAttentionDimensionProductionCase(
            target_query_residual_inputs=query_input,
            target_key_value_residual_inputs=key_input,
            parent_query_linear=query_linear,
            parent_key_linear=key_linear,
            parent_query_post_rope=query_post,
            parent_key_post_rope=key_post,
            parent_centered_scores=scores,
            parent_probabilities=probabilities,
            query_rope_cos=query_cosine,
            query_rope_sin=query_sine,
            key_rope_cos=key_cosine,
            key_rope_sin=key_sine,
            causal_mask=mask,
            query_row_sqrt_weights=torch.ones((1, 3), dtype=torch.bfloat16),
            key_value_row_sqrt_weights=torch.ones(
                (1, 3),
                dtype=torch.bfloat16,
            ),
            provenance=Lfm2CpcfAttentionCaseProvenance(
                partition=partition,
                trajectory_sha256=f"{ordinal + 40:064x}",
                source_split_sha256=_SOURCE_SPLIT_SHA,
                source_runtime_sha256=runtime.source_runtime_sha256,
                case_ordinal=ordinal,
            ),
        )

    fit_production = (
        source_production(partition="fit", ordinal=0, offset=0.0),
        source_production(partition="fit", ordinal=1, offset=-0.11),
    )
    fit_check_production = (
        source_production(partition="fit_check", ordinal=2, offset=0.17),
    )
    upstream = compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=fit_production,
        fit_check_cases=fit_check_production,
        validation_cases=(),
        query_groups=(
            Lfm2CpcfBarycentricHeadGroup((0,), (1.0,)),
            Lfm2CpcfBarycentricHeadGroup((1,), (1.0,)),
        ),
        key_value_groups=(Lfm2CpcfBarycentricHeadGroup((0,), (1.0,)),),
        runtime=runtime,
        policy=_upstream_policy(),
        ridge=1e-5,
    )
    seed = _compile_global_seed(upstream, target_seed_template)
    fit_case = _build_residual_cases(upstream, fit_production)[0]
    fit_check_case = _build_residual_cases(
        upstream,
        fit_check_production,
    )[0]
    assert fit_case.production_case.query_rope_cos.shape[-1] == 4
    assert fit_case.factor_state.target_query_rope_cos.shape[-1] == 2
    assert fit_case.query_rope_cos.shape[-1] == 2
    prepared = score_fold._prepare_evidence_case(
        fit_case,
        seed,
        Lfm2QkScoreResidualFoldPolicy(),
        device=torch.device("cpu"),
    )
    assert prepared.query_cosine.shape[-1] == seed.head_dim == 2
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=seal_lfm2_qk_score_residual_fit_bank(
            (fit_case,),
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=seed.source_runtime_sha256,
            source_artifact_sha256=seed.source_artifact_sha256,
        ),
        fit_check_bank=seal_lfm2_qk_score_residual_fit_check_bank(
            (fit_check_case,),
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=seed.source_runtime_sha256,
            source_artifact_sha256=seed.source_artifact_sha256,
        ),
        policy=Lfm2QkScoreResidualFoldPolicy(
            maximum_sweeps=1,
            maximum_pcg_iterations=32,
            minimum_pcg_iterations=1,
            pcg_restart_iterations=32,
        ),
    )
    assert result.verify() == result.report["report_sha256"]


def test_policy_rejects_training_like_unbounded_sweeps() -> None:
    with pytest.raises(
        Lfm2CachedQkScoreResidualFoldError,
        match="at most two sweeps",
    ):
        Lfm2QkScoreResidualFoldPolicy(maximum_sweeps=3)


def test_hard_plant_frontier_recovers_scale_eight_and_rejects_scale_twelve() -> None:
    seed, fit_bank, fit_check_bank = _banks(plant_scale=8.0)
    accepted = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
    )
    assert accepted.status == "fit_check_accepted"
    assert accepted.accepted and accepted.changed
    assert accepted.report["initial_fit_metrics"]["centered_score_relative_rmse"] > 1.0
    assert accepted.fit_metrics.centered_score_relative_rmse < 0.35
    assert accepted.fit_check_metrics is not None
    assert accepted.fit_check_metrics.centered_score_relative_rmse < 0.35

    seed, fit_bank, fit_check_bank = _banks(plant_scale=12.0)
    rejected = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
    )
    assert rejected.status == "poor_quality_seed_fallback"
    assert not rejected.accepted and not rejected.changed
    assert torch.equal(rejected.query_weight, seed.query_weight)
    assert torch.equal(rejected.key_weight, seed.key_weight)
    assert (
        rejected.report["proposed_fit_metrics"]["centered_score_relative_rmse"] > 0.35
    )
    assert not rejected.report["fit_check_evaluated"]


def test_zero_and_near_zero_global_priors_fail_closed() -> None:
    zero_template = _seed(weight_scale=0.0)
    zero, fit_bank, fit_check_bank = _banks(seed_value=zero_template)
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=zero,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
    )
    # An authenticated global synchronization may move an inconsistent all-zero
    # prior slightly toward its nonzero FIT factors, but the residual fold must
    # still reject the unusable state without opening FIT-CHECK.
    assert result.status == "poor_quality_seed_fallback"
    assert not result.accepted and not result.changed
    assert result.fit_metrics.centered_score_relative_rmse > 0.999
    assert not result.report["fit_check_evaluated"]

    near_zero_template = _seed(weight_scale=1e-4)
    near_zero, fit_bank, fit_check_bank = _banks(seed_value=near_zero_template)
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=near_zero,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
    )
    assert result.status == "poor_quality_seed_fallback"
    assert not result.accepted and not result.changed
    assert not result.report["fit_check_evaluated"]


def test_exact_authenticated_zero_seed_target_preserves_valid_no_op() -> None:
    zero_template = _seed(weight_scale=0.0)
    planted_query, planted_key = _planted_weights(zero_template)
    _upstream, zero, (fit_case, _fit_extra), (fit_check_case,) = _build_partition_pair(
        zero_template,
        (
            _production_case(
                seed=zero_template,
                partition="fit",
                ordinal=0,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.0,
                target_uses_seed=True,
            ),
            _production_case(
                seed=zero_template,
                partition="fit",
                ordinal=1,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=-0.11,
                target_uses_seed=True,
            ),
        ),
        (
            _production_case(
                seed=zero_template,
                partition="fit_check",
                ordinal=2,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.2,
                target_uses_seed=True,
            ),
        ),
    )
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=zero,
        fit_bank=seal_lfm2_qk_score_residual_fit_bank(
            (fit_case,),
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=zero.source_runtime_sha256,
            source_artifact_sha256=zero.source_artifact_sha256,
        ),
        fit_check_bank=seal_lfm2_qk_score_residual_fit_check_bank(
            (fit_check_case,),
            source_split_sha256=_SOURCE_SPLIT_SHA,
            source_runtime_sha256=zero.source_runtime_sha256,
            source_artifact_sha256=zero.source_artifact_sha256,
        ),
    )
    assert result.accepted and not result.changed
    assert result.status == "quality_gated_seed_no_op"
    assert result.report["fit_exact_seed_target_proof"]
    assert result.report["fit_check_exact_seed_target_proof"]
    assert result.report["sweeps"] == []


def test_pcg_accepts_an_exact_solution_before_minimum_iterations() -> None:
    class IdentityNormal:
        normal_operator_calls = 0

        def normal(self, vector: torch.Tensor) -> torch.Tensor:
            self.normal_operator_calls += 1
            return vector.clone()

    operator = IdentityNormal()
    right_hand_side = torch.tensor([1.0, 2.0, -3.0])
    solution, report = score_fold._pcg(
        operator,
        right_hand_side,
        Lfm2QkScoreResidualFoldPolicy(
            maximum_sweeps=1,
            maximum_pcg_iterations=8,
            minimum_pcg_iterations=2,
            pcg_restart_iterations=8,
        ),
    )
    assert report.converged
    assert report.iterations == 1
    assert report.termination == "exact_normal_residual"
    assert torch.equal(solution, right_hand_side)


def test_result_authentication_binds_exposed_metrics() -> None:
    seed, fit_bank, fit_check_bank = _banks()
    result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
    )
    fake = type(result.fit_metrics)(
        case_count=999,
        eligible_query_rows=999,
        centered_score_relative_rmse=0.0,
        probability_relative_rmse=0.0,
        js_divergence_mean=0.0,
        js_divergence_worst=0.0,
        top1_route_agreement_fraction=1.0,
        maximum_causal_probability_leakage=0.0,
    )
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="failed authentication",
    ):
        type(result)(
            query_weight=result.query_weight,
            key_weight=result.key_weight,
            seed=result.seed,
            accepted=result.accepted,
            changed=result.changed,
            status=result.status,
            fit_metrics=fake,
            fit_check_metrics=result.fit_check_metrics,
            report=result.report,
        )


def test_relabelled_identical_fit_check_content_is_rejected() -> None:
    seed_template = _seed()
    planted_query, planted_key = _planted_weights(seed_template)
    _upstream, seed, (fit_case, _fit_extra), (fit_check_case,) = _build_partition_pair(
        seed_template,
        (
            _production_case(
                seed=seed_template,
                partition="fit",
                ordinal=0,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.0,
                trajectory_sha256="a" * 64,
            ),
            _production_case(
                seed=seed_template,
                partition="fit",
                ordinal=2,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.19,
                trajectory_sha256="e" * 64,
            ),
        ),
        (
            _production_case(
                seed=seed_template,
                partition="fit_check",
                ordinal=1,
                planted_query=planted_query,
                planted_key=planted_key,
                input_offset=0.0,
                trajectory_sha256="b" * 64,
            ),
        ),
    )
    fit_bank = seal_lfm2_qk_score_residual_fit_bank(
        (fit_case,),
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    fit_check_bank = seal_lfm2_qk_score_residual_fit_check_bank(
        (fit_check_case,),
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="numerical content overlaps",
    ):
        compile_lfm2_cached_qk_score_residual_fold(
            seed=seed,
            fit_bank=fit_bank,
            fit_check_bank=fit_check_bank,
        )


def test_solver_fingerprints_ignore_unused_parent_packaging_and_reject_overlap() -> (
    None
):
    seed_template = _seed()
    planted_query, planted_key = _planted_weights(seed_template)
    fit_production = _production_case(
        seed=seed_template,
        partition="fit",
        ordinal=0,
        planted_query=planted_query,
        planted_key=planted_key,
        input_offset=0.0,
        trajectory_sha256="c" * 64,
    )
    perturbed_probabilities = fit_production.parent_probabilities.clone()
    perturbation = torch.tensor(
        1e-8,
        dtype=perturbed_probabilities.dtype,
    )
    perturbed_probabilities[0, 0, 1, 0] += perturbation
    perturbed_probabilities[0, 0, 1, 1] -= perturbation
    fit_check_provenance = Lfm2CpcfAttentionCaseProvenance(
        partition="fit_check",
        trajectory_sha256="d" * 64,
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=_SOURCE_RUNTIME_SHA,
        case_ordinal=1,
    )
    fit_check_production = Lfm2CpcfAttentionDimensionProductionCase(
        **{
            field.name: (
                fit_check_provenance
                if field.name == "provenance"
                else perturbed_probabilities
                if field.name == "parent_probabilities"
                else getattr(fit_production, field.name)
            )
            for field in fields(fit_production)
            if field.name != "evidence_sha256"
        }
    )
    extra_fit_production = _production_case(
        seed=seed_template,
        partition="fit",
        ordinal=2,
        planted_query=planted_query,
        planted_key=planted_key,
        input_offset=0.21,
        trajectory_sha256="e" * 64,
    )
    _upstream, seed, (fit_case, _fit_extra), (fit_check_case,) = _build_partition_pair(
        seed_template,
        (fit_production, extra_fit_production),
        (fit_check_production,),
    )
    assert fit_case.production_case.evidence_sha256 != (
        fit_check_case.production_case.evidence_sha256
    )
    assert fit_case.transform_receipt_sha256 != (
        fit_check_case.transform_receipt_sha256
    )
    assert fit_case.content_sha256 == fit_check_case.content_sha256
    assert set(fit_case.solver_row_sha256s) & set(fit_check_case.solver_row_sha256s)
    fit_bank = seal_lfm2_qk_score_residual_fit_bank(
        (fit_case,),
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    fit_check_bank = seal_lfm2_qk_score_residual_fit_check_bank(
        (fit_check_case,),
        source_split_sha256=_SOURCE_SPLIT_SHA,
        source_runtime_sha256=seed.source_runtime_sha256,
        source_artifact_sha256=seed.source_artifact_sha256,
    )
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="numerical content overlaps",
    ):
        compile_lfm2_cached_qk_score_residual_fold(
            seed=seed,
            fit_bank=fit_bank,
            fit_check_bank=fit_check_bank,
        )


def test_caller_cloned_or_tampered_factor_state_is_not_production_evidence() -> None:
    seed, fit_bank, _fit_check = _banks()
    source = fit_bank.cases[0]
    production = source.production_case
    factor_state = source.factor_state
    bad_sine = factor_state.target_query_rope_sin.clone()
    bad_sine[..., seed.head_dim // 2 :] += torch.tensor(
        0.125,
        dtype=bad_sine.dtype,
    )
    bad_factor_state = Lfm2CpcfAttentionFactorSufficientState(
        **{
            name: (
                bad_sine
                if name == "target_query_rope_sin"
                else getattr(factor_state, name)
            )
            for name in factor_state.__dataclass_fields__
        }
    )
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="exact member",
    ):
        build_lfm2_qk_score_residual_case(
            production_case=production,
            upstream_fold=source.upstream_fold,
            factor_state=bad_factor_state,
        )

    zero_cosine = torch.zeros_like(factor_state.target_query_rope_cos)
    zero_sine = torch.zeros_like(factor_state.target_query_rope_sin)
    zero_factor_state = Lfm2CpcfAttentionFactorSufficientState(
        **{
            name: (
                zero_cosine
                if name == "target_query_rope_cos"
                else zero_sine
                if name == "target_query_rope_sin"
                else getattr(factor_state, name)
            )
            for name in factor_state.__dataclass_fields__
        }
    )
    with pytest.raises(
        Lfm2CachedQkScoreResidualAuthenticationError,
        match="exact member",
    ):
        build_lfm2_qk_score_residual_case(
            production_case=production,
            upstream_fold=source.upstream_fold,
            factor_state=zero_factor_state,
        )


@pytest.mark.parametrize(
    "override",
    (
        {"trust_radius": 2.01},
        {"maximum_pcg_iterations": 4_097},
        {"maximum_fit_cases": 4_097},
        {"alpha_frontier": tuple(float(index) / 9.0 for index in range(10))},
        {"maximum_fit_centered_score_relative_rmse": 2.0},
        {"maximum_fit_check_centered_score_relative_rmse": 2.0},
        {"maximum_fit_probability_relative_rmse": 2.0},
        {"maximum_fit_check_probability_relative_rmse": 2.0},
        {"maximum_fit_js_divergence_mean": 2.0},
        {"maximum_fit_check_js_divergence_mean": 2.0},
        {"maximum_fit_js_divergence_worst": 2.0},
        {"maximum_fit_check_js_divergence_worst": 2.0},
        {"minimum_fit_top1_route_agreement": 0.0},
        {"minimum_fit_check_top1_route_agreement": 0.0},
    ),
)
def test_policy_enforces_hard_analytical_ceilings(
    override: dict[str, object],
) -> None:
    with pytest.raises(
        Lfm2CachedQkScoreResidualFoldError,
        match="hard|at most",
    ):
        Lfm2QkScoreResidualFoldPolicy(**override)


def test_compile_enforces_aggregate_evidence_ceiling() -> None:
    seed, fit_bank, fit_check_bank = _banks()
    with pytest.raises(
        Lfm2CachedQkScoreResidualFoldError,
        match="aggregate analytical evidence ceilings",
    ):
        compile_lfm2_cached_qk_score_residual_fold(
            seed=seed,
            fit_bank=fit_bank,
            fit_check_bank=fit_check_bank,
            policy=Lfm2QkScoreResidualFoldPolicy(
                maximum_aggregate_score_elements=1,
            ),
        )
