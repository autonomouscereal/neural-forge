from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest
import torch

import neural_forge.hf_lfm2_qk_global_gauge_sync as global_sync
from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (
    Lfm2CpcfAttentionFactorSufficientState,
    lfm2_projection_target_from_runtime,
    lfm2_runtime_rms_norm,
)
from neural_forge.hf_lfm2_qk_global_gauge_sync import (
    Lfm2QkGlobalGaugeSyncError,
    Lfm2QkGlobalGaugeSyncPolicy,
    _canonical_sha256,
    _divided_rmsnorm_adjoint_rows,
    _divided_rmsnorm_rows,
    _positive_radial_preimage,
    compile_lfm2_qk_global_gauge_sync,
    seal_lfm2_qk_global_gauge_sync_provenance,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return (
        value * cosine.unsqueeze(1) + _rotate_half(value) * sine.unsqueeze(1)
    ).contiguous()


def _rope(
    *,
    length: int,
    offset: float,
    constant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = (
        torch.full((length,), offset, dtype=torch.float64)
        if constant
        else torch.arange(length, dtype=torch.float64) + offset
    )
    first = positions.reshape(-1, 1) * torch.tensor(
        [0.173, 0.037],
        dtype=torch.float64,
    ).reshape(1, -1)
    cosine_half = torch.cos(first)
    sine_half = torch.sin(first)
    cosine = torch.cat((cosine_half, cosine_half), dim=-1)
    sine = torch.cat((sine_half, sine_half), dim=-1)
    return (
        cosine.unsqueeze(0).to(torch.bfloat16),
        sine.unsqueeze(0).to(torch.bfloat16),
    )


def _runtime(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    norm: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, _hidden = inputs.shape
    raw = (
        (inputs @ weight.T)
        .reshape(batch, length, 1, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    normalized = lfm2_runtime_rms_norm(raw, norm, 1e-5)
    return raw, _apply_rope(normalized, cosine, sine)


def _orthogonal(angle: float, *, swap: bool = False) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    first = torch.tensor(
        [
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, cosine, sine],
            [0.0, 0.0, -sine, cosine],
        ],
        dtype=torch.float64,
    )
    if not swap:
        return first
    permutation = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    return first @ permutation


def _fixture(
    gauges: tuple[torch.Tensor, ...],
) -> tuple[
    tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    query_norm = torch.tensor(
        [0.75, 1.25, 0.75, 1.25],
        dtype=torch.bfloat16,
    )
    key_norm = query_norm.clone()
    true_query_weight = torch.eye(4, dtype=torch.bfloat16)
    true_key_weight = torch.eye(4, dtype=torch.bfloat16)
    states: list[Lfm2CpcfAttentionFactorSufficientState] = []
    for case_index, gauge in enumerate(gauges):
        if case_index == 0:
            query_inputs = (
                torch.eye(4, dtype=torch.float64)
                * torch.tensor(
                    [0.5, 2.0, 8.0, 32.0],
                    dtype=torch.float64,
                ).reshape(-1, 1)
            ).unsqueeze(0)
            key_inputs = (
                torch.eye(4, dtype=torch.float64)
                * torch.tensor(
                    [16.0, 1.0, 4.0, 0.75],
                    dtype=torch.float64,
                ).reshape(-1, 1)
            ).unsqueeze(0)
            q_cos, q_sin = _rope(length=4, offset=1.5, constant=True)
            k_cos, k_sin = _rope(length=4, offset=1.5, constant=True)
        else:
            generator = torch.Generator().manual_seed(3100 + case_index)
            query_base = torch.randn(
                (7, 4),
                generator=generator,
                dtype=torch.float64,
            )
            key_base = torch.randn(
                (6, 4),
                generator=generator,
                dtype=torch.float64,
            )
            query_radii = torch.tensor(
                [0.375, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0],
                dtype=torch.float64,
            )
            key_radii = torch.tensor(
                [20.0, 10.0, 5.0, 2.5, 1.25, 0.625],
                dtype=torch.float64,
            )
            query_inputs = (
                query_base
                / torch.sqrt(query_base.square().mean(dim=-1, keepdim=True))
                * query_radii.reshape(-1, 1)
            ).unsqueeze(0)
            key_inputs = (
                key_base
                / torch.sqrt(key_base.square().mean(dim=-1, keepdim=True))
                * key_radii.reshape(-1, 1)
            ).unsqueeze(0)
            q_cos, q_sin = _rope(
                length=int(query_inputs.shape[1]),
                offset=2.0 + case_index,
            )
            k_cos, k_sin = _rope(
                length=int(key_inputs.shape[1]),
                offset=1.0 + case_index,
            )
        query_inputs = query_inputs.to(torch.bfloat16)
        key_inputs = key_inputs.to(torch.bfloat16)
        query_raw, query_post = _runtime(
            query_inputs,
            true_query_weight,
            query_norm,
            q_cos,
            q_sin,
        )
        key_raw, key_post = _runtime(
            key_inputs,
            true_key_weight,
            key_norm,
            k_cos,
            k_sin,
        )
        local_query = query_post.to(torch.float64) @ gauge
        local_key = key_post.to(torch.float64) @ gauge
        # The legacy one-shot route receives a different local post-RoPE
        # anchor in every case and therefore preserves contradictory gauges.
        local_anchor_gauge = _orthogonal(
            0.31 + 0.47 * case_index,
            swap=case_index % 2 == 0,
        )
        states.append(
            Lfm2CpcfAttentionFactorSufficientState(
                partition="fit",
                case_ordinal=case_index,
                evidence_sha256=_sha(f"case-{case_index}"),
                local_query_post_rope_factor=local_query,
                local_key_post_rope_factor=local_key,
                query_post_rope_anchor=(
                    query_post.to(torch.float64) @ local_anchor_gauge
                ),
                key_post_rope_anchor=(key_post.to(torch.float64) @ local_anchor_gauge),
                query_raw_linear_anchor=query_raw,
                key_raw_linear_anchor=key_raw,
                target_query_inputs=query_inputs,
                target_key_inputs=key_inputs,
                query_row_sqrt_weights=torch.linspace(
                    0.7,
                    1.3,
                    int(query_inputs.shape[1]),
                    dtype=torch.float64,
                ).reshape(1, -1),
                key_row_sqrt_weights=torch.linspace(
                    1.2,
                    0.8,
                    int(key_inputs.shape[1]),
                    dtype=torch.float64,
                ).reshape(1, -1),
                target_query_rope_cos=q_cos,
                target_query_rope_sin=q_sin,
                target_key_rope_cos=k_cos,
                target_key_rope_sin=k_sin,
            )
        )
    return (
        tuple(states),
        query_norm,
        key_norm,
        true_query_weight,
        true_key_weight,
    )


def _score(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.matmul(
            query.to(torch.float64),
            key.to(torch.float64).transpose(-2, -1),
        )
        / 2.0
    )


def _score_rrmse(
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    query_norm: torch.Tensor,
    key_norm: torch.Tensor,
) -> float:
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for state in states:
        _query_raw, query_post = _runtime(
            state.target_query_inputs,
            query_weight,
            query_norm,
            state.target_query_rope_cos,
            state.target_query_rope_sin,
        )
        _key_raw, key_post = _runtime(
            state.target_key_inputs,
            key_weight,
            key_norm,
            state.target_key_rope_cos,
            state.target_key_rope_sin,
        )
        prediction = _score(query_post, key_post)
        target = _score(
            state.local_query_post_rope_factor,
            state.local_key_post_rope_factor,
        )
        weights = state.query_row_sqrt_weights.square().reshape(
            int(target.shape[0]),
            1,
            int(target.shape[2]),
            1,
        ) * state.key_row_sqrt_weights.square().reshape(
            int(target.shape[0]),
            1,
            1,
            int(target.shape[3]),
        )
        numerator += (weights * (prediction - target).square()).sum()
        denominator += (weights * target.square()).sum()
    return float(torch.sqrt(numerator / denominator).item())


def _legacy_local_gauge_one_shot(
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    query_norm: torch.Tensor,
    key_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_inputs: list[torch.Tensor] = []
    query_targets: list[torch.Tensor] = []
    key_inputs: list[torch.Tensor] = []
    key_targets: list[torch.Tensor] = []
    for state in states:
        local_query = state.local_query_post_rope_factor.to(torch.float64)
        local_key = state.local_key_post_rope_factor.to(torch.float64)
        anchor_query = state.query_post_rope_anchor.to(torch.float64)
        anchor_key = state.key_post_rope_anchor.to(torch.float64)
        query_sqrt = state.query_row_sqrt_weights.to(torch.float64).reshape(-1, 1)
        key_sqrt = state.key_row_sqrt_weights.to(torch.float64).reshape(-1, 1)
        cross = (local_query.reshape(-1, 4) * query_sqrt).T @ (
            anchor_query.reshape(-1, 4) * query_sqrt
        ) + (local_key.reshape(-1, 4) * key_sqrt).T @ (
            anchor_key.reshape(-1, 4) * key_sqrt
        )
        left, _singular, right_h = torch.linalg.svd(cross, full_matrices=False)
        local_rotation = left @ right_h
        locally_oriented_query = local_query @ local_rotation
        locally_oriented_key = local_key @ local_rotation
        query_target = lfm2_projection_target_from_runtime(
            locally_oriented_query,
            state.query_raw_linear_anchor,
            norm_weight=query_norm,
            epsilon=1e-5,
            cosine=state.target_query_rope_cos,
            sine=state.target_query_rope_sin,
        )
        key_target = lfm2_projection_target_from_runtime(
            locally_oriented_key,
            state.key_raw_linear_anchor,
            norm_weight=key_norm,
            epsilon=1e-5,
            cosine=state.target_key_rope_cos,
            sine=state.target_key_rope_sin,
        )
        query_inputs.append(state.target_query_inputs.to(torch.float64).reshape(-1, 4))
        query_targets.append(query_target.permute(0, 2, 1, 3).reshape(-1, 4))
        key_inputs.append(state.target_key_inputs.to(torch.float64).reshape(-1, 4))
        key_targets.append(key_target.permute(0, 2, 1, 3).reshape(-1, 4))

    def solve(inputs: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, dim=0)
        y = torch.cat(targets, dim=0)
        return torch.linalg.lstsq(x, y).solution.T.to(torch.bfloat16)

    return solve(query_inputs, query_targets), solve(key_inputs, key_targets)


def _compile(
    gauges: tuple[torch.Tensor, ...],
    *,
    source_query_prior: torch.Tensor | None = None,
    source_key_prior: torch.Tensor | None = None,
) -> tuple[
    tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    torch.Tensor,
    torch.Tensor,
    object,
]:
    states, query_norm, key_norm, true_query, true_key = _fixture(gauges)
    query_prior = true_query if source_query_prior is None else source_query_prior
    key_prior = true_key if source_key_prior is None else source_key_prior
    query_template = torch.zeros_like(true_query)
    key_template = torch.zeros_like(true_key)
    provenance = seal_lfm2_qk_global_gauge_sync_provenance(
        fit_receipt_sha256=_sha("fit-receipt"),
        source_runtime_receipt_sha256=_sha("source-runtime-receipt"),
        factor_states=states,
        source_query_weight_prior=query_prior,
        source_key_weight_prior=key_prior,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        norm_epsilon=1e-5,
    )
    compiled = compile_lfm2_qk_global_gauge_sync(
        factor_states=states,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        source_query_weight_prior=query_prior,
        source_key_weight_prior=key_prior,
        norm_epsilon=1e-5,
        provenance=provenance,
        policy=Lfm2QkGlobalGaugeSyncPolicy(
            sweeps=3,
            global_prior_ridge=1e-12,
            radius_prior_strength=1e-10,
        ),
    )
    return states, query_norm, key_norm, compiled


def _gqa_compile() -> object:
    query_norm = torch.tensor(
        [0.75, 1.25, 0.75, 1.25],
        dtype=torch.bfloat16,
    )
    key_norm = query_norm.clone()
    identity = torch.eye(4, dtype=torch.bfloat16)
    second_query = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    query_weight = torch.cat((identity, second_query), dim=0)
    key_weight = identity
    states: list[Lfm2CpcfAttentionFactorSufficientState] = []
    for case_index, gauge in enumerate(
        (_orthogonal(0.63), _orthogonal(-1.07, swap=True))
    ):
        generator = torch.Generator().manual_seed(8110 + case_index)
        query_inputs = torch.randn(
            (1, 8, 4),
            generator=generator,
            dtype=torch.float64,
        )
        key_inputs = torch.randn(
            (1, 7, 4),
            generator=generator,
            dtype=torch.float64,
        )
        query_inputs *= torch.linspace(
            0.5,
            9.0,
            8,
            dtype=torch.float64,
        ).reshape(1, -1, 1)
        key_inputs *= torch.linspace(
            7.0,
            0.75,
            7,
            dtype=torch.float64,
        ).reshape(1, -1, 1)
        query_inputs = query_inputs.to(torch.bfloat16)
        key_inputs = key_inputs.to(torch.bfloat16)
        q_cos, q_sin = _rope(length=8, offset=1.0 + case_index)
        k_cos, k_sin = _rope(length=7, offset=2.0 + case_index)
        query_raw = (
            (query_inputs @ query_weight.T)
            .reshape(1, 8, 2, 4)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        key_raw = (
            (key_inputs @ key_weight.T)
            .reshape(1, 7, 1, 4)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        query_post = _apply_rope(
            lfm2_runtime_rms_norm(query_raw, query_norm, 1e-5),
            q_cos,
            q_sin,
        )
        key_post = _apply_rope(
            lfm2_runtime_rms_norm(key_raw, key_norm, 1e-5),
            k_cos,
            k_sin,
        )
        states.append(
            Lfm2CpcfAttentionFactorSufficientState(
                partition="fit",
                case_ordinal=case_index,
                evidence_sha256=_sha(f"gqa-case-{case_index}"),
                local_query_post_rope_factor=(query_post.to(torch.float64) @ gauge),
                local_key_post_rope_factor=key_post.to(torch.float64) @ gauge,
                query_post_rope_anchor=query_post,
                key_post_rope_anchor=key_post,
                query_raw_linear_anchor=query_raw,
                key_raw_linear_anchor=key_raw,
                target_query_inputs=query_inputs,
                target_key_inputs=key_inputs,
                query_row_sqrt_weights=torch.ones((1, 8), dtype=torch.float64),
                key_row_sqrt_weights=torch.ones((1, 7), dtype=torch.float64),
                target_query_rope_cos=q_cos,
                target_query_rope_sin=q_sin,
                target_key_rope_cos=k_cos,
                target_key_rope_sin=k_sin,
            )
        )
    state_values = tuple(states)
    query_template = torch.zeros_like(query_weight)
    key_template = torch.zeros_like(key_weight)
    provenance = seal_lfm2_qk_global_gauge_sync_provenance(
        fit_receipt_sha256=_sha("gqa-fit-receipt"),
        source_runtime_receipt_sha256=_sha("gqa-source-runtime-receipt"),
        factor_states=state_values,
        source_query_weight_prior=query_weight,
        source_key_weight_prior=key_weight,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        norm_epsilon=1e-5,
    )
    return compile_lfm2_qk_global_gauge_sync(
        factor_states=state_values,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        source_query_weight_prior=query_weight,
        source_key_weight_prior=key_weight,
        norm_epsilon=1e-5,
        provenance=provenance,
        policy=Lfm2QkGlobalGaugeSyncPolicy(global_prior_ridge=1e-12),
    )


def _compile_explicit_states(
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    query_norm: torch.Tensor,
    key_norm: torch.Tensor,
    query_prior: torch.Tensor,
    key_prior: torch.Tensor,
    *,
    receipt_label: str,
    policy: Lfm2QkGlobalGaugeSyncPolicy | None = None,
) -> object:
    query_template = torch.zeros_like(query_prior)
    key_template = torch.zeros_like(key_prior)
    provenance = seal_lfm2_qk_global_gauge_sync_provenance(
        fit_receipt_sha256=_sha(f"{receipt_label}-fit"),
        source_runtime_receipt_sha256=_sha(f"{receipt_label}-runtime"),
        factor_states=states,
        source_query_weight_prior=query_prior,
        source_key_weight_prior=key_prior,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        norm_epsilon=1e-5,
    )
    return compile_lfm2_qk_global_gauge_sync(
        factor_states=states,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=query_template,
        target_key_weight_template=key_template,
        source_query_weight_prior=query_prior,
        source_key_weight_prior=key_prior,
        norm_epsilon=1e-5,
        provenance=provenance,
        policy=policy,
    )


def _balanced_nonoracle_fixture() -> tuple[
    tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    query_norm = torch.tensor(
        [0.75, 1.25, 0.9, 1.1],
        dtype=torch.bfloat16,
    )
    key_norm = torch.tensor(
        [1.15, 0.8, 1.3, 0.7],
        dtype=torch.bfloat16,
    )
    true_query = torch.tensor(
        [
            [1.4, 0.3, -0.2, 0.1],
            [-0.1, 0.7, 0.5, -0.2],
            [0.2, -0.4, 1.1, 0.3],
            [0.35, 0.05, -0.25, 0.8],
        ],
        dtype=torch.bfloat16,
    )
    true_key = torch.tensor(
        [
            [0.6, -0.4, 0.3, 0.2],
            [0.25, 1.3, -0.2, 0.1],
            [-0.15, 0.35, 0.75, -0.45],
            [0.4, -0.1, 0.2, 1.15],
        ],
        dtype=torch.bfloat16,
    )
    anchor_query = torch.eye(4, dtype=torch.bfloat16)
    anchor_key = torch.eye(4, dtype=torch.bfloat16)
    states: list[Lfm2CpcfAttentionFactorSufficientState] = []
    for case_index in range(3):
        generator = torch.Generator().manual_seed(5700 + case_index)
        query_inputs = torch.randn(
            (1, 9, 4),
            generator=generator,
            dtype=torch.float64,
        )
        key_inputs = torch.randn(
            (1, 8, 4),
            generator=generator,
            dtype=torch.float64,
        )
        query_inputs *= torch.linspace(
            0.5,
            7.0,
            9,
            dtype=torch.float64,
        ).reshape(1, -1, 1)
        key_inputs *= torch.linspace(
            6.0,
            0.75,
            8,
            dtype=torch.float64,
        ).reshape(1, -1, 1)
        query_inputs = query_inputs.to(torch.bfloat16)
        key_inputs = key_inputs.to(torch.bfloat16)
        query_cos = torch.ones((1, 9, 4), dtype=torch.bfloat16)
        query_sin = torch.zeros_like(query_cos)
        key_cos = torch.ones((1, 8, 4), dtype=torch.bfloat16)
        key_sin = torch.zeros_like(key_cos)
        true_query_raw, true_query_post = _runtime(
            query_inputs,
            true_query,
            query_norm,
            query_cos,
            query_sin,
        )
        true_key_raw, true_key_post = _runtime(
            key_inputs,
            true_key,
            key_norm,
            key_cos,
            key_sin,
        )
        anchor_query_raw, anchor_query_post = _runtime(
            query_inputs,
            anchor_query,
            query_norm,
            query_cos,
            query_sin,
        )
        anchor_key_raw, anchor_key_post = _runtime(
            key_inputs,
            anchor_key,
            key_norm,
            key_cos,
            key_sin,
        )
        desired_scores = (
            true_query_post[0, 0].to(torch.float64)
            @ true_key_post[0, 0].to(torch.float64).T
        )
        left, singular, right_h = torch.linalg.svd(
            desired_scores,
            full_matrices=False,
        )
        root = torch.sqrt(singular[:4])
        balanced_query = left[:, :4] * root.reshape(1, -1)
        balanced_key = right_h[:4].T * root.reshape(1, -1)
        cross = balanced_query.T @ anchor_query_post[0, 0].to(
            torch.float64
        ) + balanced_key.T @ anchor_key_post[0, 0].to(torch.float64)
        orient_left, _orient_singular, orient_right_h = torch.linalg.svd(
            cross,
            full_matrices=False,
        )
        rotation = orient_left @ orient_right_h
        states.append(
            Lfm2CpcfAttentionFactorSufficientState(
                partition="fit",
                case_ordinal=case_index,
                evidence_sha256=_sha(f"balanced-nonoracle-{case_index}"),
                local_query_post_rope_factor=(balanced_query @ rotation).reshape(
                    1, 1, 9, 4
                ),
                local_key_post_rope_factor=(balanced_key @ rotation).reshape(
                    1, 1, 8, 4
                ),
                query_post_rope_anchor=anchor_query_post,
                key_post_rope_anchor=anchor_key_post,
                query_raw_linear_anchor=anchor_query_raw,
                key_raw_linear_anchor=anchor_key_raw,
                target_query_inputs=query_inputs,
                target_key_inputs=key_inputs,
                query_row_sqrt_weights=torch.ones((1, 9), dtype=torch.float64),
                key_row_sqrt_weights=torch.ones((1, 8), dtype=torch.float64),
                target_query_rope_cos=query_cos,
                target_query_rope_sin=query_sin,
                target_key_rope_cos=key_cos,
                target_key_rope_sin=key_sin,
            )
        )
        assert true_query_raw.shape == anchor_query_raw.shape
        assert true_key_raw.shape == anchor_key_raw.shape
    return (
        tuple(states),
        query_norm,
        key_norm,
        anchor_query,
        anchor_key,
        true_query,
        true_key,
    )


def _balanced_nonoracle_zero_first_score_fixture() -> tuple[
    tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        true_query,
        true_key,
    ) = _balanced_nonoracle_fixture()
    zero_first_states: list[Lfm2CpcfAttentionFactorSufficientState] = []
    for state in states:
        _query_raw, query_post = _runtime(
            state.target_query_inputs,
            true_query,
            query_norm,
            state.target_query_rope_cos,
            state.target_query_rope_sin,
        )
        _key_raw, key_post = _runtime(
            state.target_key_inputs,
            true_key,
            key_norm,
            state.target_key_rope_cos,
            state.target_key_rope_sin,
        )
        desired_scores = (
            query_post[0, 0].to(torch.float64) @ key_post[0, 0].to(torch.float64).T
        )
        # Production order: causal centering creates the mathematical zero
        # score row first, then the balanced SVD emits machine-precision dust
        # in its query factor.
        desired_scores[0] = 0.0
        left, singular, right_h = torch.linalg.svd(
            desired_scores,
            full_matrices=False,
        )
        root = torch.sqrt(singular[:4])
        balanced_query = left[:, :4] * root.reshape(1, -1)
        balanced_key = right_h[:4].T * root.reshape(1, -1)
        cross = balanced_query.T @ state.query_post_rope_anchor[0, 0].to(
            torch.float64
        ) + balanced_key.T @ state.key_post_rope_anchor[0, 0].to(torch.float64)
        orient_left, _orient_singular, orient_right_h = torch.linalg.svd(
            cross,
            full_matrices=False,
        )
        rotation = orient_left @ orient_right_h
        zero_first_states.append(
            replace(
                state,
                evidence_sha256=_sha(f"balanced-pre-svd-zero-{state.case_ordinal}"),
                local_query_post_rope_factor=(balanced_query @ rotation).reshape_as(
                    state.local_query_post_rope_factor
                ),
                local_key_post_rope_factor=(balanced_key @ rotation).reshape_as(
                    state.local_key_post_rope_factor
                ),
            )
        )
    return (
        tuple(zero_first_states),
        query_norm,
        key_norm,
        query_prior,
        key_prior,
    )


def test_reachable_rmsnorm_gain_uses_exact_positive_radius_and_bf16_frontier() -> None:
    norm = torch.ones(4, dtype=torch.bfloat16)
    cosine = torch.ones((1, 1, 4), dtype=torch.bfloat16)
    sine = torch.zeros_like(cosine)
    desired = torch.full((1, 1, 1, 4), 0.5, dtype=torch.float64)
    current_raw = torch.ones((1, 1, 1, 4), dtype=torch.bfloat16)

    raw, post, record = _positive_radial_preimage(
        desired,
        current_raw,
        norm,
        cosine,
        sine,
        epsilon=1e-5,
        policy=Lfm2QkGlobalGaugeSyncPolicy(),
    )

    expected_radius = math.sqrt(1e-5) * 0.5 / math.sqrt(1.0 - 0.5**2)
    solved_radius = float(torch.sqrt(raw.square().mean()).item())
    assert solved_radius == pytest.approx(expected_radius, rel=0.01)
    assert torch.allclose(
        post.to(torch.float64),
        desired,
        rtol=0.0,
        atol=0.004,
    )
    assert record["continuous_reachable_row_count"] == 1
    assert record["maximum_reachable_continuous_gain_absolute_error"] < 1e-14
    assert record["exact_shared_preimage_rows_retained"] == 0


def test_divided_rmsnorm_analytic_adjoint_matches_inner_product_identity() -> None:
    generator = torch.Generator().manual_seed(8227)
    row_count = 257
    dimension = 64
    primal = torch.randn(
        (row_count, dimension),
        generator=generator,
        dtype=torch.float64,
    )
    cotangent = torch.randn(
        (row_count, dimension),
        generator=generator,
        dtype=torch.float64,
    )
    norm = (
        torch.randn(
            (dimension,),
            generator=generator,
            dtype=torch.float64,
        ).abs()
        + 0.25
    )
    angles = torch.randn(
        (row_count, dimension // 2),
        generator=generator,
        dtype=torch.float64,
    )
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)

    forward = _divided_rmsnorm_rows(
        primal,
        norm,
        cosine,
        sine,
    )
    adjoint = _divided_rmsnorm_adjoint_rows(
        cotangent,
        norm,
        cosine,
        sine,
    )

    assert tuple(adjoint.shape) == (row_count, dimension)
    assert torch.isfinite(adjoint).all()
    assert float((forward * cotangent).sum()) == pytest.approx(
        float((primal * adjoint).sum()),
        rel=0.0,
        abs=2e-11,
    )


def test_score_activity_is_gauge_invariant_and_has_no_absolute_small_floor() -> None:
    query = torch.eye(4, dtype=torch.float64) * 1e-120
    key = torch.eye(4, dtype=torch.float64) * 1e-120
    gauge = torch.diag(torch.tensor([2.0, 0.5, 1.5, 0.75], dtype=torch.float64))
    transformed_query = query @ gauge
    transformed_key = key @ torch.linalg.inv(gauge).T

    query_activity, key_activity, diagnostic = (
        global_sync._gauge_invariant_score_activity(query, key)
    )
    transformed_query_activity, transformed_key_activity, transformed_diagnostic = (
        global_sync._gauge_invariant_score_activity(
            transformed_query,
            transformed_key,
        )
    )

    assert query_activity.all()
    assert key_activity.all()
    assert torch.equal(transformed_query_activity, query_activity)
    assert torch.equal(transformed_key_activity, key_activity)
    assert diagnostic["score_maximum_absolute_value"] == pytest.approx(1e-240)
    assert 0.0 < diagnostic["score_activity_absolute_tolerance"] < 1e-240
    assert diagnostic["fixed_absolute_score_floor_used"] is False
    assert transformed_diagnostic["score_activity_rule"] == (
        "gauge_invariant_score_max_exceeds_scaled_fp64_roundoff"
    )


def test_symmetric_gain_design_streaming_matches_materialized_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(8231)
    query = torch.randn((12, 4), generator=generator, dtype=torch.float64)
    key = torch.randn((11, 4), generator=generator, dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)
    norm = torch.tensor([0.75, 1.25, 0.9, 1.1], dtype=torch.float64)
    query_cosine = torch.ones_like(query)
    query_sine = torch.zeros_like(query)
    key_cosine = torch.ones_like(key)
    key_sine = torch.zeros_like(key)
    query_weights = torch.linspace(0.7, 1.3, 12, dtype=torch.float64)
    key_weights = torch.linspace(1.2, 0.8, 11, dtype=torch.float64)

    materialized, _materialized_frontier = global_sync._rmsnorm_image_balanced_gauge(
        query,
        key,
        identity,
        norm,
        norm,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        query_weights,
        key_weights,
        maximum_condition_number=4096.0,
    )
    monkeypatch.setattr(global_sync, "_GAIN_DESIGN_CHUNK_ELEMENTS", 32)
    streamed, streamed_frontier = global_sync._rmsnorm_image_balanced_gauge(
        query,
        key,
        identity,
        norm,
        norm,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        query_weights,
        key_weights,
        maximum_condition_number=4096.0,
    )

    assert torch.allclose(streamed, materialized, rtol=0.0, atol=2e-12)
    assert any(step["gain_design_streamed"] for step in streamed_frontier)
    assert all(
        step["gain_design_peak_elements"] <= 32
        for step in streamed_frontier
        if step["gain_design_streamed"]
    )


def test_zero_radial_rows_are_exact_and_excluded_from_positive_inverse() -> None:
    norm = torch.ones(4, dtype=torch.bfloat16)
    cosine = torch.ones((1, 2, 4), dtype=torch.bfloat16)
    sine = torch.zeros_like(cosine)
    desired = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]]],
        dtype=torch.float64,
    )
    current_raw = torch.ones((1, 1, 2, 4), dtype=torch.bfloat16)

    raw, post, record = _positive_radial_preimage(
        desired,
        current_raw,
        norm,
        cosine,
        sine,
        epsilon=1e-5,
        policy=Lfm2QkGlobalGaugeSyncPolicy(),
    )

    assert torch.equal(raw[0, 0, 0], torch.zeros(4, dtype=torch.float64))
    assert torch.equal(post[0, 0, 0], torch.zeros(4, dtype=torch.bfloat16))
    assert record["zero_target_row_count"] == 1
    assert record["positive_target_row_count"] == 1
    assert record["exact_zero_preimage_rows"] == 1
    assert record["continuous_reachable_row_count"] == 1
    assert record["continuous_unreachable_row_count"] == 0
    assert record["positive_radius_enforced_for_nonzero_rows"] is True
    assert record["exact_zero_radius_permitted"] is True


def test_full_bilinear_gauge_beats_production_balanced_svd_nonoracle_anchor() -> None:
    (
        states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        true_query,
        true_key,
    ) = _balanced_nonoracle_fixture()
    compiled = _compile_explicit_states(
        states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        receipt_label="balanced-nonoracle",
        policy=Lfm2QkGlobalGaugeSyncPolicy(
            sweeps=3,
            source_anchor_prior_ridge=1e-12,
            global_prior_ridge=1e-12,
            gauge_map_ridge=1e-10,
            radius_prior_strength=1e-10,
        ),
    )
    prior_error = _score_rrmse(
        states,
        query_prior,
        key_prior,
        query_norm,
        key_norm,
    )
    true_error = _score_rrmse(
        states,
        true_query,
        true_key,
        query_norm,
        key_norm,
    )
    final_error = _score_rrmse(
        states,
        compiled.query_weight,
        compiled.key_weight,
        query_norm,
        key_norm,
    )

    assert prior_error > 0.8
    assert true_error < 1e-12
    assert final_error < 0.05
    assert final_error < 0.239869 * 0.20
    assert final_error < prior_error * 0.25
    assert compiled.report["error_ladder"]["E_runtime"] == pytest.approx(
        final_error,
        abs=1e-15,
    )
    assert all(
        record["gauge_family"] == "full_gl_bilinear_qg_k_inverse_transpose"
        and record["score_invariance_relative_rmse"] < 1e-10
        for frontier in compiled.report["deterministic_frontier"]
        for record in frontier["gauge_records"]
    )


def test_pre_svd_zero_score_rows_ignore_nonzero_factor_roundoff_dust() -> None:
    states, query_norm, key_norm, query_prior, key_prior = (
        _balanced_nonoracle_zero_first_score_fixture()
    )
    dust = [
        float(state.local_query_post_rope_factor[0, 0, 0].abs().max())
        for state in states
    ]
    assert all(0.0 < value < 1e-12 for value in dust)

    compiled = _compile_explicit_states(
        states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        receipt_label="balanced-pre-svd-zero",
        policy=Lfm2QkGlobalGaugeSyncPolicy(
            sweeps=3,
            source_anchor_prior_ridge=1e-12,
            global_prior_ridge=1e-12,
            gauge_map_ridge=1e-10,
            radius_prior_strength=1e-10,
        ),
    )

    activity = compiled.report["factor_activity"]
    assert activity["exact_zero_query_factor_row_count"] == 0
    assert activity["score_uninformative_query_factor_row_count"] == len(states)
    assert activity["informative_query_factor_row_count"] == 24
    assert activity["informative_key_factor_row_count"] == 24
    assert activity["informative_score_cell_count"] == 192
    assert activity["fixed_absolute_score_floor_used"] is False
    assert activity["maximum_score_activity_absolute_tolerance"] < 1e-12
    assert compiled.report["error_ladder"]["E_runtime"] < 0.06
    assert compiled.verify_authentication() == compiled.report["report_sha256"]
    assert compiled.report["operation_audit"]["exact_zero_radial_preimages"] == 0
    assert compiled.report["operation_audit"]["nonzero_svd_dust_radial_preimages"] == 9
    assert all(
        radius["query"]["input_exact_zero_target_row_count"] == 0
        and radius["query"]["score_uninformative_target_row_count"] == 1
        and radius["query"]["nonzero_svd_dust_rows_forced_to_zero"] == 1
        and radius["query"]["zero_target_row_count"] == 1
        and radius["key"]["score_uninformative_target_row_count"] == 0
        for sweep in compiled.report["deterministic_frontier"]
        for radius in sweep["radius_records"]
    )
    assert all(
        record["rmsnorm_image_balance_frontier"][0][
            "informative_query_factor_row_count"
        ]
        == 8
        and record["rmsnorm_image_balance_frontier"][0][
            "informative_key_factor_row_count"
        ]
        == 8
        and record["rmsnorm_image_balance_frontier"][0][
            "score_activity_is_full_gl_gauge_invariant"
        ]
        for sweep in compiled.report["deterministic_frontier"]
        for record in sweep["gauge_records"]
    )


def test_rank_deficient_short_cases_use_deterministic_nullspace_completion() -> None:
    states, query_norm, key_norm, query_prior, key_prior = _fixture(
        (
            _orthogonal(0.2),
            _orthogonal(-0.4, swap=True),
        )
    )
    short_states = tuple(
        replace(
            state,
            evidence_sha256=_sha(f"short-{state.case_ordinal}"),
            local_query_post_rope_factor=(
                state.local_query_post_rope_factor[:, :, :1].clone()
            ),
            local_key_post_rope_factor=(
                state.local_key_post_rope_factor[:, :, :1].clone()
            ),
            query_post_rope_anchor=state.query_post_rope_anchor[:, :, :1].clone(),
            key_post_rope_anchor=state.key_post_rope_anchor[:, :, :1].clone(),
            query_raw_linear_anchor=state.query_raw_linear_anchor[:, :, :1].clone(),
            key_raw_linear_anchor=state.key_raw_linear_anchor[:, :, :1].clone(),
            target_query_inputs=state.target_query_inputs[:, :1].clone(),
            target_key_inputs=state.target_key_inputs[:, :1].clone(),
            query_row_sqrt_weights=state.query_row_sqrt_weights[:, :1].clone(),
            key_row_sqrt_weights=state.key_row_sqrt_weights[:, :1].clone(),
            target_query_rope_cos=state.target_query_rope_cos[:, :1].clone(),
            target_query_rope_sin=state.target_query_rope_sin[:, :1].clone(),
            target_key_rope_cos=state.target_key_rope_cos[:, :1].clone(),
            target_key_rope_sin=state.target_key_rope_sin[:, :1].clone(),
        )
        for state in states
    )
    first = _compile_explicit_states(
        short_states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        receipt_label="short-rank",
    )
    second = _compile_explicit_states(
        short_states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        receipt_label="short-rank",
    )

    assert torch.equal(first.query_weight, second.query_weight)
    assert torch.equal(first.key_weight, second.key_weight)
    assert math.isfinite(float(first.report["error_ladder"]["E_runtime"]))
    records = first.report["deterministic_frontier"][0]["gauge_records"]
    assert all(record["identifiable_rank"] < 4 for record in records)
    assert all(record["nullspace_dimension"] > 0 for record in records)
    assert all(
        record["nullspace_completion"] == "weighted_ridge_to_identity"
        for record in records
    )
    assert all(record["score_invariance_relative_rmse"] < 1e-10 for record in records)


def test_mixed_zero_factor_rows_are_ignored_without_hiding_nonzero_fit() -> None:
    states, query_norm, key_norm, query_prior, key_prior = _fixture(
        (
            _orthogonal(0.2),
            _orthogonal(-0.4, swap=True),
        )
    )
    mixed_states: list[Lfm2CpcfAttentionFactorSufficientState] = []
    expected_informative_cells = 0
    for state in states:
        query = state.local_query_post_rope_factor.clone()
        key = state.local_key_post_rope_factor.clone()
        query[:, :, 0] = 0.0
        key[:, :, -1] = 0.0
        query_activity, key_activity, _diagnostic = (
            global_sync._gauge_invariant_score_activity(
                query[0, 0],
                key[0, 0],
            )
        )
        expected_informative_cells += int(query_activity.sum()) * int(
            key_activity.sum()
        )
        mixed_states.append(
            replace(
                state,
                evidence_sha256=_sha(f"mixed-zero-{state.case_ordinal}"),
                local_query_post_rope_factor=query,
                local_key_post_rope_factor=key,
            )
        )
    compiled = _compile_explicit_states(
        tuple(mixed_states),
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        receipt_label="mixed-zero",
    )

    activity = compiled.report["factor_activity"]
    assert activity["exact_zero_query_factor_row_count"] == len(states)
    assert activity["exact_zero_key_factor_row_count"] == len(states)
    assert activity["informative_score_cell_count"] == expected_informative_cells
    assert activity["counts_authenticated_by_factor_state_and_report_sha256"] is True
    assert compiled.report["fit_score_evidence"]["has_informative_factor_score_cells"]
    assert math.isfinite(float(compiled.report["error_ladder"]["E_runtime"]))
    assert compiled.report["error_ladder"]["E_runtime"] < 1e-12
    assert compiled.verify_authentication() == compiled.report["report_sha256"]
    assert all(
        record["query"]["zero_target_row_count"] >= 1
        and record["key"]["zero_target_row_count"] >= 1
        and record["query"]["exact_zero_preimage_rows"] >= 1
        and record["key"]["exact_zero_preimage_rows"] >= 1
        for sweep in compiled.report["deterministic_frontier"]
        for record in sweep["radius_records"]
    )
    assert (
        compiled.report["operation_audit"][
            "zero_or_score_uninformative_factor_rows_used_in_global_ridge"
        ]
        is False
    )


def test_all_zero_one_key_states_retain_prior_and_report_no_fit_evidence() -> None:
    states, query_norm, key_norm, query_prior, key_prior = _fixture(
        (
            _orthogonal(0.2),
            _orthogonal(-0.4, swap=True),
        )
    )
    zero_states: list[Lfm2CpcfAttentionFactorSufficientState] = []
    for state in states:
        query_length = min(3, int(state.target_query_inputs.shape[1]))
        zero_states.append(
            replace(
                state,
                evidence_sha256=_sha(f"all-zero-one-key-{state.case_ordinal}"),
                local_query_post_rope_factor=torch.zeros_like(
                    state.local_query_post_rope_factor[:, :, :query_length]
                ),
                local_key_post_rope_factor=torch.zeros_like(
                    state.local_key_post_rope_factor[:, :, :1]
                ),
                query_post_rope_anchor=(
                    state.query_post_rope_anchor[:, :, :query_length].clone()
                ),
                key_post_rope_anchor=state.key_post_rope_anchor[:, :, :1].clone(),
                query_raw_linear_anchor=(
                    state.query_raw_linear_anchor[:, :, :query_length].clone()
                ),
                key_raw_linear_anchor=(state.key_raw_linear_anchor[:, :, :1].clone()),
                target_query_inputs=(
                    state.target_query_inputs[:, :query_length].clone()
                ),
                target_key_inputs=state.target_key_inputs[:, :1].clone(),
                query_row_sqrt_weights=(
                    state.query_row_sqrt_weights[:, :query_length].clone()
                ),
                key_row_sqrt_weights=state.key_row_sqrt_weights[:, :1].clone(),
                target_query_rope_cos=(
                    state.target_query_rope_cos[:, :query_length].clone()
                ),
                target_query_rope_sin=(
                    state.target_query_rope_sin[:, :query_length].clone()
                ),
                target_key_rope_cos=state.target_key_rope_cos[:, :1].clone(),
                target_key_rope_sin=state.target_key_rope_sin[:, :1].clone(),
            )
        )
    compiled = _compile_explicit_states(
        tuple(zero_states),
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        receipt_label="all-zero-one-key",
    )

    assert (
        compiled.report["query_weight_sha256"]
        == compiled.report["source_anchor_initialization"]["query_weight_sha256"]
    )
    assert (
        compiled.report["key_weight_sha256"]
        == compiled.report["source_anchor_initialization"]["key_weight_sha256"]
    )
    assert compiled.report["published_sweep"] == 0
    assert compiled.report["error_ladder"]["E_runtime"] == 0.0
    assert compiled.report["factor_activity"]["informative_score_cell_count"] == 0
    assert compiled.report["factor_activity"]["score_uninformative_cell_count"] == 6
    assert compiled.report["fit_score_evidence"] == {
        "has_informative_factor_score_cells": False,
        "zero_information_error_ladder": True,
        "zero_information_state_can_retain_source_initialization": True,
    }
    assert all(
        not sweep["accepted_into_monotone_frontier"]
        and sweep["selected_damping_fraction"] == 0.0
        and sweep["projection_fit"]["query_head_ridge_solve_count"] == 0
        and sweep["projection_fit"]["key_head_ridge_solve_count"] == 0
        for sweep in compiled.report["deterministic_frontier"]
    )
    audit = compiled.report["operation_audit"]
    assert audit["symmetric_rmsnorm_gain_gauge_solves"] == 0
    assert audit["positive_radius_closed_form_solves"] == 0
    assert audit["exact_zero_radial_preimages"] == 24
    assert audit["global_prior_ridge_solves"] == 0
    assert audit["dense_rmsnorm_linear_maps_materialized"] == 0
    assert audit["dense_rmsnorm_linear_map_elements_materialized"] == 0
    assert compiled.verify_authentication() == compiled.report["report_sha256"]


def test_global_sync_recovers_shared_runtime_that_local_gauges_contradict() -> None:
    gauges = (
        _orthogonal(0.73),
        _orthogonal(-0.91, swap=True),
        _orthogonal(1.27),
    )
    states, query_norm, key_norm, compiled = _compile(gauges)
    local_query, local_key = _legacy_local_gauge_one_shot(
        states,
        query_norm,
        key_norm,
    )
    local_error = _score_rrmse(
        states,
        local_query,
        local_key,
        query_norm,
        key_norm,
    )
    synchronized_error = _score_rrmse(
        states,
        compiled.query_weight,
        compiled.key_weight,
        query_norm,
        key_norm,
    )
    assert local_error > 0.25
    assert synchronized_error < 1e-12
    assert compiled.report["error_ladder"]["E_runtime"] == pytest.approx(
        synchronized_error,
        rel=0.0,
        abs=1e-15,
    )
    assert compiled.report["error_ladder"]["E_rank"] is None
    assert compiled.report["error_ladder_meaning"]["E_rank"].startswith("unavailable")
    assert compiled.query_weight.dtype == torch.bfloat16
    assert compiled.key_weight.dtype == torch.bfloat16
    assert torch.allclose(
        compiled.query_weight,
        torch.eye(4, dtype=torch.bfloat16),
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.allclose(
        compiled.key_weight,
        torch.eye(4, dtype=torch.bfloat16),
        rtol=0.0,
        atol=1e-12,
    )

    # The first constant nonzero RoPE case has two repeated score singular pairs.
    singular = torch.linalg.svdvals(
        _score(
            states[0].local_query_post_rope_factor,
            states[0].local_key_post_rope_factor,
        )[0, 0]
    )
    assert singular[0] == pytest.approx(float(singular[1]), rel=0.02)
    assert singular[2] == pytest.approx(float(singular[3]), rel=0.02)


def test_source_anchor_initialization_escapes_materially_bad_qk_priors() -> None:
    gauges = (
        _orthogonal(0.73),
        _orthogonal(-0.91, swap=True),
        _orthogonal(1.27),
    )
    query_mixing = torch.tensor(
        [
            [1.35, 0.31, -0.18, 0.09],
            [-0.22, 0.72, 0.27, -0.16],
            [0.14, -0.35, 1.18, 0.21],
            [0.19, 0.08, -0.29, 0.81],
        ],
        dtype=torch.float64,
    )
    key_mixing = torch.tensor(
        [
            [0.68, -0.34, 0.24, 0.17],
            [0.29, 1.42, -0.31, 0.12],
            [-0.11, 0.26, 0.77, -0.38],
            [0.33, -0.19, 0.28, 1.21],
        ],
        dtype=torch.float64,
    )
    bad_query_prior = (query_mixing @ _orthogonal(0.86, swap=True)).to(torch.bfloat16)
    bad_key_prior = (key_mixing @ _orthogonal(-0.67)).to(torch.bfloat16)
    states, query_norm, key_norm, compiled = _compile(
        gauges,
        source_query_prior=bad_query_prior,
        source_key_prior=bad_key_prior,
    )
    prior_error = _score_rrmse(
        states,
        bad_query_prior,
        bad_key_prior,
        query_norm,
        key_norm,
    )
    final_error = _score_rrmse(
        states,
        compiled.query_weight,
        compiled.key_weight,
        query_norm,
        key_norm,
    )
    initialization = compiled.report["source_anchor_initialization"]
    assert prior_error > 1.0
    assert initialization["source_prior_runtime_score_relative_rmse"] == (
        pytest.approx(prior_error, abs=1e-15)
    )
    assert initialization["source_anchor_runtime_score_relative_rmse"] < 1e-12
    assert final_error < 1e-12
    assert compiled.report["error_ladder"]["E_runtime"] < 1e-12
    assert all(
        value["error_ladder"]["E_runtime"] < 1e-12
        for value in compiled.report["deterministic_frontier"]
    )
    assert torch.allclose(
        compiled.query_weight,
        torch.eye(4, dtype=torch.bfloat16),
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.allclose(
        compiled.key_weight,
        torch.eye(4, dtype=torch.bfloat16),
        rtol=0.0,
        atol=1e-12,
    )


def test_global_sync_replays_two_query_heads_through_one_shared_gqa_key() -> None:
    compiled = _gqa_compile()
    assert tuple(compiled.query_weight.shape) == (8, 4)
    assert tuple(compiled.key_weight.shape) == (4, 4)
    assert compiled.report["query_heads"] == 2
    assert compiled.report["key_value_heads"] == 1
    assert compiled.report["queries_per_key_value"] == 2
    assert compiled.report["error_ladder"]["E_runtime"] < 1e-12
    assert all(
        len(frontier["gauge_records"]) == 2
        for frontier in compiled.report["deterministic_frontier"]
    )


def test_global_sync_is_invariant_to_case_gauges_and_raw_row_radii() -> None:
    first = _compile(
        (
            _orthogonal(0.21),
            _orthogonal(0.88, swap=True),
            _orthogonal(-1.11),
        )
    )
    second = _compile(
        (
            _orthogonal(-1.33, swap=True),
            _orthogonal(1.49),
            _orthogonal(0.51, swap=True),
        )
    )
    first_compiled = first[3]
    second_compiled = second[3]
    assert torch.allclose(
        first_compiled.query_weight,
        second_compiled.query_weight,
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.allclose(
        first_compiled.key_weight,
        second_compiled.key_weight,
        rtol=0.0,
        atol=1e-12,
    )
    assert first_compiled.report["error_ladder"]["E_runtime"] < 1e-12
    assert second_compiled.report["error_ladder"]["E_runtime"] < 1e-12
    assert all(
        record["query"]["minimum_solved_radius"] > 0.0
        and record["key"]["minimum_solved_radius"] > 0.0
        for record in first_compiled.report["deterministic_frontier"][-1][
            "radius_records"
        ]
    )
    assert (
        first_compiled.report["deterministic_frontier"][-1]["radius_records"][0][
            "query"
        ]["maximum_solved_radius"]
        / first_compiled.report["deterministic_frontier"][-1]["radius_records"][0][
            "query"
        ]["minimum_solved_radius"]
        > 20.0
    )


def test_receipt_proves_fit_only_zero_training_and_sealed_dense_state() -> None:
    gauges = (
        _orthogonal(0.2),
        _orthogonal(-0.4, swap=True),
    )
    _states, _query_norm, _key_norm, compiled = _compile(gauges)
    repeated = _compile(gauges)[3]
    assert compiled.verify_authentication() == compiled.report["report_sha256"]
    assert repeated.report["report_sha256"] == compiled.report["report_sha256"]
    assert torch.equal(repeated.query_weight, compiled.query_weight)
    assert torch.equal(repeated.key_weight, compiled.key_weight)
    assert set(compiled.state_dict()) == {"q_proj.weight", "k_proj.weight"}
    assert compiled.report["scientific_scope"] == "fit_only_qk_analytical_compiler"
    assert compiled.report["frontier_selection"] == (
        "minimum_fit_runtime_error_on_fixed_analytical_sweep_and_damping_schedule"
    )
    assert len(compiled.report["deterministic_frontier"]) == 3
    audit = compiled.report["operation_audit"]
    assert audit["model_forward_calls"] == 0
    assert audit["backward_calls"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["gradient_updates"] == 0
    assert audit["training_examples"] == 0
    assert audit["selection_case_count"] == 0
    assert audit["validation_case_count"] == 0
    assert audit["fit_check_case_count"] == 0


def test_monotone_fixed_fit_frontier_never_publishes_a_regressed_sweep() -> None:
    (
        states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        _true_query,
        _true_key,
    ) = _balanced_nonoracle_fixture()
    compiled = _compile_explicit_states(
        states,
        query_norm,
        key_norm,
        query_prior,
        key_prior,
        receipt_label="monotone-frontier",
        policy=Lfm2QkGlobalGaugeSyncPolicy(
            sweeps=3,
            source_anchor_prior_ridge=1e-12,
            global_prior_ridge=1e-12,
            gauge_map_ridge=1e-10,
            radius_prior_strength=1e-10,
        ),
    )
    published = [
        float(
            compiled.report["source_anchor_initialization"][
                "source_anchor_runtime_score_relative_rmse"
            ]
        ),
        *[
            float(value["published_runtime_score_relative_rmse"])
            for value in compiled.report["deterministic_frontier"]
        ],
    ]

    assert all(
        later <= earlier
        for earlier, later in zip(
            published[:-1],
            published[1:],
            strict=True,
        )
    )
    assert published[-1] == pytest.approx(
        float(compiled.report["error_ladder"]["E_runtime"]),
        abs=1e-15,
    )


def test_nested_receipt_is_frozen_and_external_authority_detects_resealing() -> None:
    states, _query_norm, _key_norm, compiled = _compile(
        (
            _orthogonal(0.2),
            _orthogonal(-0.4, swap=True),
        )
    )
    provenance = compiled.report["provenance"]
    fit_receipt = str(provenance["fit_receipt_sha256"])
    runtime_receipt = str(provenance["source_runtime_receipt_sha256"])
    factor_receipt = str(provenance["factor_state_bundle_sha256"])
    assert factor_receipt == _canonical_sha256(
        [state.as_digest_dict() for state in states]
    )
    assert (
        compiled.verify_authentication(
            expected_fit_receipt_sha256=fit_receipt,
            expected_source_runtime_receipt_sha256=runtime_receipt,
            expected_factor_state_bundle_sha256=factor_receipt,
        )
        == compiled.report["report_sha256"]
    )
    with pytest.raises(TypeError):
        compiled.report["provenance"]["fit_receipt_sha256"] = _sha("mutated")

    payload = compiled.as_dict()
    payload["provenance"]["source_runtime_receipt_sha256"] = _sha("resealed-runtime")
    provenance_body = dict(payload["provenance"])
    provenance_body.pop("provenance_sha256")
    payload["provenance"]["provenance_sha256"] = _canonical_sha256(provenance_body)
    report_body = dict(payload)
    report_body.pop("report_sha256")
    payload["report_sha256"] = _canonical_sha256(report_body)
    resealed = replace(compiled, report=payload)
    with pytest.raises(
        Lfm2QkGlobalGaugeSyncError,
        match="authoritative receipt",
    ):
        resealed.verify_authentication(
            expected_source_runtime_receipt_sha256=runtime_receipt
        )

    changed_query = compiled.query_weight.clone()
    changed_query[0, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
    with pytest.raises(
        Lfm2QkGlobalGaugeSyncError,
        match="receipt is invalid",
    ):
        replace(compiled, query_weight=changed_query)


def test_duplicate_case_ordinal_is_rejected_even_with_distinct_evidence() -> None:
    states, query_norm, key_norm, query_prior, key_prior = _fixture(
        (
            _orthogonal(0.2),
            _orthogonal(-0.4, swap=True),
        )
    )
    duplicated = (
        states[0],
        replace(states[1], case_ordinal=states[0].case_ordinal),
    )
    with pytest.raises(
        Lfm2QkGlobalGaugeSyncError,
        match="case ordinals must be unique",
    ):
        _compile_explicit_states(
            duplicated,
            query_norm,
            key_norm,
            query_prior,
            key_prior,
            receipt_label="duplicate-ordinal",
        )


@pytest.mark.parametrize(
    ("policy_field", "message"),
    (
        ("maximum_total_rows", "rows exceed"),
        ("maximum_total_score_elements", "score elements exceed"),
        ("maximum_total_factor_elements", "factor elements exceed"),
        ("maximum_total_output_elements", "output elements exceed"),
        ("maximum_total_svd_workspace_elements", "SVD workspace exceeds"),
        ("maximum_normal_equation_elements", "normal equation exceeds"),
    ),
)
def test_independent_analytical_resource_ceilings_fail_closed(
    policy_field: str,
    message: str,
) -> None:
    states, query_norm, key_norm, query_prior, key_prior = _fixture(
        (
            _orthogonal(0.2),
            _orthogonal(-0.4, swap=True),
        )
    )
    policy = replace(
        Lfm2QkGlobalGaugeSyncPolicy(),
        **{policy_field: 1},
    )
    with pytest.raises(Lfm2QkGlobalGaugeSyncError, match=message):
        _compile_explicit_states(
            states,
            query_norm,
            key_norm,
            query_prior,
            key_prior,
            receipt_label=f"ceiling-{policy_field}",
            policy=policy,
        )


def test_provenance_and_fit_partition_fail_closed() -> None:
    states, query_norm, key_norm, query_prior, key_prior = _fixture(
        (
            _orthogonal(0.2),
            _orthogonal(-0.4, swap=True),
        )
    )
    templates = torch.zeros_like(query_prior)
    provenance = seal_lfm2_qk_global_gauge_sync_provenance(
        fit_receipt_sha256=_sha("fit-receipt"),
        source_runtime_receipt_sha256=_sha("source-runtime-receipt"),
        factor_states=states,
        source_query_weight_prior=query_prior,
        source_key_weight_prior=key_prior,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=templates,
        target_key_weight_template=templates,
        norm_epsilon=1e-5,
    )
    changed_prior = query_prior.clone()
    changed_prior[0, 0] = torch.tensor(0.5, dtype=torch.bfloat16)
    with pytest.raises(
        Lfm2QkGlobalGaugeSyncError,
        match="sealed provenance",
    ):
        compile_lfm2_qk_global_gauge_sync(
            factor_states=states,
            target_query_norm_weight=query_norm,
            target_key_norm_weight=key_norm,
            target_query_weight_template=templates,
            target_key_weight_template=templates,
            source_query_weight_prior=changed_prior,
            source_key_weight_prior=key_prior,
            norm_epsilon=1e-5,
            provenance=provenance,
        )

    heldout = tuple(
        Lfm2CpcfAttentionFactorSufficientState(
            **{
                name: ("fit_check" if name == "partition" else getattr(state, name))
                for name in state.__dataclass_fields__
            }
        )
        for state in states
    )
    heldout_provenance = seal_lfm2_qk_global_gauge_sync_provenance(
        fit_receipt_sha256=_sha("fit-receipt-heldout"),
        source_runtime_receipt_sha256=_sha("source-runtime-receipt-heldout"),
        factor_states=heldout,
        source_query_weight_prior=query_prior,
        source_key_weight_prior=key_prior,
        target_query_norm_weight=query_norm,
        target_key_norm_weight=key_norm,
        target_query_weight_template=templates,
        target_key_weight_template=templates,
        norm_epsilon=1e-5,
    )
    with pytest.raises(
        Lfm2QkGlobalGaugeSyncError,
        match="literal FIT states only",
    ):
        compile_lfm2_qk_global_gauge_sync(
            factor_states=heldout,
            target_query_norm_weight=query_norm,
            target_key_norm_weight=key_norm,
            target_query_weight_template=templates,
            target_key_weight_template=templates,
            source_query_weight_prior=query_prior,
            source_key_weight_prior=key_prior,
            norm_epsilon=1e-5,
            provenance=heldout_provenance,
        )
