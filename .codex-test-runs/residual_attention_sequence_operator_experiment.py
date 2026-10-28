"""Ephemeral zero-training residual-attention folding experiment.

This intentionally lives under .codex-test-runs and imports the production
attention compilers without modifying tracked code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedAttentionFoldCase,
    compile_lfm2_structural_attention_producer_fold,
    compile_lfm2_structural_attention_value_output_fold,
    compile_lfm2_structural_residual_attention_fold,
    evaluate_lfm2_compiled_attention_producer_score_preservation,
)


DTYPE = torch.float64
SOURCE_HIDDEN = 32
TARGET_HIDDEN = 8
SOURCE_QUERY_HEADS = 8
SOURCE_KV_HEADS = 4
TARGET_QUERY_HEADS = 2
TARGET_KV_HEADS = 1
HEAD_DIM = 4
PREFILL_LENGTH = 6
DECODE_STEPS = 6


@dataclass(frozen=True)
class ParentOperator:
    decoder: torch.Tensor
    encoder: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    transition: torch.Tensor


@dataclass(frozen=True)
class SequenceDataset:
    cases: tuple[Lfm2CachedAttentionFoldCase, ...]
    phase_names: tuple[str, ...]
    initial_history: torch.Tensor
    forcing: torch.Tensor
    decode_inputs: tuple[torch.Tensor, ...]
    decode_endpoints: tuple[torch.Tensor, ...]


def _rrmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.mean((prediction - target).square()).sqrt()
    denominator = torch.mean(target.square()).sqrt().clamp_min(1e-15)
    return float((numerator / denominator).item())


def _cosine(prediction: torch.Tensor, target: torch.Tensor) -> float:
    left = prediction.reshape(-1)
    right = target.reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float((torch.dot(left, right) / denominator.clamp_min(1e-15)).item())


def _sha_tensor_bundle(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _orthonormal_decoder(generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(
        SOURCE_HIDDEN,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    return torch.linalg.qr(raw, mode="reduced").Q.contiguous()


def _make_parent(seed: int) -> ParentOperator:
    generator = torch.Generator().manual_seed(seed)
    decoder = _orthonormal_decoder(generator)
    encoder = decoder.T.contiguous()

    base_query = (
        torch.randn(
            TARGET_QUERY_HEADS,
            HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        )
        / math.sqrt(TARGET_HIDDEN)
    )
    base_key = (
        torch.randn(
            HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        )
        / math.sqrt(TARGET_HIDDEN)
    )
    base_value = (
        torch.randn(
            HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        )
        / math.sqrt(TARGET_HIDDEN)
    )
    base_output = (
        torch.randn(
            TARGET_QUERY_HEADS,
            TARGET_HIDDEN,
            HEAD_DIM,
            generator=generator,
            dtype=DTYPE,
        )
        / math.sqrt(HEAD_DIM)
    )

    query_blocks = []
    key_blocks = []
    value_blocks = []
    output_blocks = []
    for group in range(SOURCE_KV_HEADS):
        key_latent = base_key + 0.18 * torch.randn(
            base_key.shape,
            generator=generator,
            dtype=DTYPE,
        )
        value_latent = base_value + 0.32 * torch.randn(
            base_value.shape,
            generator=generator,
            dtype=DTYPE,
        )
        key_blocks.append(key_latent @ encoder)
        value_blocks.append(value_latent @ encoder)
        for offset in range(2):
            query_latent = base_query[offset] + 0.16 * torch.randn(
                base_query[offset].shape,
                generator=generator,
                dtype=DTYPE,
            )
            query_blocks.append(query_latent @ encoder)
            target_output_block = (
                base_output[offset]
                + 0.24
                * torch.randn(
                    base_output[offset].shape,
                    generator=generator,
                    dtype=DTYPE,
                )
            ) / SOURCE_KV_HEADS
            output_blocks.append(decoder @ target_output_block)

    transition = torch.randn(
        TARGET_HIDDEN,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    spectral = torch.linalg.matrix_norm(transition, ord=2).clamp_min(1e-12)
    transition = 0.75 * transition / spectral
    return ParentOperator(
        decoder=decoder,
        encoder=encoder,
        query=torch.cat(query_blocks, dim=0).contiguous(),
        key=torch.cat(key_blocks, dim=0).contiguous(),
        value=torch.cat(value_blocks, dim=0).contiguous(),
        output=torch.cat(output_blocks, dim=1).contiguous(),
        transition=transition.contiguous(),
    )


def _parent_case(
    parent: ParentOperator,
    history: torch.Tensor,
    *,
    query_start: int,
) -> tuple[Lfm2CachedAttentionFoldCase, torch.Tensor]:
    batch, key_length, _hidden = history.shape
    query_input = history[:, query_start:]
    query_length = int(query_input.shape[1])
    parent_hidden = history @ parent.decoder.T
    parent_query_hidden = parent_hidden[:, query_start:]
    query = (
        (parent_query_hidden @ parent.query.T)
        .reshape(batch, query_length, SOURCE_QUERY_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key = (
        (parent_hidden @ parent.key.T)
        .reshape(batch, key_length, SOURCE_KV_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    value = (
        (parent_hidden @ parent.value.T)
        .reshape(batch, key_length, SOURCE_KV_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    query_positions = torch.arange(
        query_start,
        key_length,
        dtype=torch.long,
    )
    key_positions = torch.arange(key_length, dtype=torch.long)
    mask_2d = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    key_mask = mask_2d.unsqueeze(0).expand(batch, -1, -1).contiguous()
    repeated_key = key.repeat_interleave(
        SOURCE_QUERY_HEADS // SOURCE_KV_HEADS,
        dim=1,
    )
    scores = query @ repeated_key.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    scores = scores.masked_fill(~key_mask.unsqueeze(1), -1e30)
    probability = torch.softmax(scores, dim=-1, dtype=DTYPE)
    repeated_value = value.repeat_interleave(
        SOURCE_QUERY_HEADS // SOURCE_KV_HEADS,
        dim=1,
    )
    context = probability @ repeated_value
    flattened = context.permute(0, 2, 1, 3).reshape(
        batch,
        query_length,
        SOURCE_QUERY_HEADS * HEAD_DIM,
    )
    parent_delta = flattened @ parent.output.T
    target_delta = parent_delta @ parent.encoder.T
    endpoint = query_input + target_delta

    contributions = []
    for head in range(SOURCE_QUERY_HEADS):
        output_block = parent.output[
            :,
            head * HEAD_DIM : (head + 1) * HEAD_DIM,
        ]
        contributions.append(context[:, head] @ output_block.T @ parent.encoder.T)
    contribution = torch.stack(contributions, dim=1)
    case = Lfm2CachedAttentionFoldCase(
        target_query_input=query_input,
        target_key_value_input=history,
        parent_query_linear=query,
        parent_key_linear=key,
        parent_value=value,
        parent_query=query,
        parent_key=key,
        parent_context=context,
        parent_post_attention_residual_target_space=endpoint,
        target_pre_attention_residual=query_input,
        key_mask=key_mask,
        parent_contribution=contribution,
    )
    return case, endpoint


def _next_latent(
    last_input: torch.Tensor,
    last_endpoint: torch.Tensor,
    forcing: torch.Tensor,
    transition: torch.Tensor,
) -> torch.Tensor:
    return torch.tanh(
        0.58 * last_endpoint
        + 0.27 * (last_input @ transition.T)
        + 0.15 * forcing
    )


def _equal_phase_weights(
    cases: tuple[Lfm2CachedAttentionFoldCase, ...],
) -> tuple[Lfm2CachedAttentionFoldCase, ...]:
    query_rows = [
        int(case.target_query_input.shape[0] * case.target_query_input.shape[1])
        for case in cases
    ]
    key_rows = [
        int(
            case.target_key_value_input.shape[0]
            * case.target_key_value_input.shape[1]
        )
        for case in cases
    ]
    total_query_rows = sum(query_rows)
    total_key_rows = sum(key_rows)
    phase_count = len(cases)
    weighted = []
    for case, query_count, key_count in zip(
        cases,
        query_rows,
        key_rows,
        strict=True,
    ):
        query_weight = math.sqrt(total_query_rows / (phase_count * query_count))
        key_weight = math.sqrt(total_key_rows / (phase_count * key_count))
        weighted.append(
            replace(
                case,
                query_row_sqrt_weights=torch.full(
                    case.target_query_input.shape[:2],
                    query_weight,
                    dtype=DTYPE,
                ),
                key_value_row_sqrt_weights=torch.full(
                    case.target_key_value_input.shape[:2],
                    key_weight,
                    dtype=DTYPE,
                ),
            )
        )
    return tuple(weighted)


def _make_dataset(
    parent: ParentOperator,
    *,
    seed: int,
    batch_size: int,
) -> SequenceDataset:
    generator = torch.Generator().manual_seed(seed)
    initial = torch.empty(
        batch_size,
        PREFILL_LENGTH,
        TARGET_HIDDEN,
        dtype=DTYPE,
    )
    initial[:, 0] = torch.randn(
        batch_size,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    prefill_noise = torch.randn(
        batch_size,
        PREFILL_LENGTH - 1,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    for index in range(1, PREFILL_LENGTH):
        initial[:, index] = torch.tanh(
            initial[:, index - 1] @ parent.transition.T
            + 0.35 * prefill_noise[:, index - 1]
        )
    forcing = torch.randn(
        batch_size,
        DECODE_STEPS,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )

    prefill_case, prefill_endpoint = _parent_case(
        parent,
        initial,
        query_start=0,
    )
    cases = [prefill_case]
    phase_names = ["prefill"]
    history = initial.clone()
    last_endpoint = prefill_endpoint[:, -1]
    decode_inputs = []
    decode_endpoints = []
    for step in range(DECODE_STEPS):
        next_input = _next_latent(
            history[:, -1],
            last_endpoint,
            forcing[:, step],
            parent.transition,
        )
        history = torch.cat((history, next_input.unsqueeze(1)), dim=1)
        case, endpoint = _parent_case(
            parent,
            history,
            query_start=int(history.shape[1]) - 1,
        )
        cases.append(case)
        phase_names.append(f"decode.{step:03d}")
        decode_inputs.append(next_input)
        decode_endpoints.append(endpoint[:, 0])
        last_endpoint = endpoint[:, 0]
    return SequenceDataset(
        cases=_equal_phase_weights(tuple(cases)),
        phase_names=tuple(phase_names),
        initial_history=initial,
        forcing=forcing,
        decode_inputs=tuple(decode_inputs),
        decode_endpoints=tuple(decode_endpoints),
    )


def _child_case_forward(
    state: dict[str, torch.Tensor],
    case: Lfm2CachedAttentionFoldCase,
) -> dict[str, torch.Tensor]:
    batch, query_length, _hidden = case.target_query_input.shape
    key_length = int(case.target_key_value_input.shape[1])
    query = (
        (case.target_query_input @ state["query"].T)
        .reshape(batch, query_length, TARGET_QUERY_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    key = (
        (case.target_key_value_input @ state["key"].T)
        .reshape(batch, key_length, TARGET_KV_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    value = (
        (case.target_key_value_input @ state["value"].T)
        .reshape(batch, key_length, TARGET_KV_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    repeated_key = key.repeat_interleave(
        TARGET_QUERY_HEADS // TARGET_KV_HEADS,
        dim=1,
    )
    scores = query @ repeated_key.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    if case.key_mask is not None:
        scores = scores.masked_fill(~case.key_mask.unsqueeze(1), -1e30)
    probability = torch.softmax(scores, dim=-1, dtype=DTYPE)
    repeated_value = value.repeat_interleave(
        TARGET_QUERY_HEADS // TARGET_KV_HEADS,
        dim=1,
    )
    context_heads = probability @ repeated_value
    context = context_heads.permute(0, 2, 1, 3).reshape(
        batch,
        query_length,
        TARGET_QUERY_HEADS * HEAD_DIM,
    )
    delta = context @ state["output"].T
    endpoint = case.target_pre_attention_residual + delta
    return {
        "scores": scores,
        "probability": probability,
        "context_heads": context_heads,
        "context": context,
        "delta": delta,
        "endpoint": endpoint,
    }


def _state_from_direct(artifact: Any) -> dict[str, torch.Tensor]:
    return {
        "query": artifact.query_weight.to(DTYPE),
        "key": artifact.key_weight.to(DTYPE),
        "value": artifact.value_weight.to(DTYPE),
        "output": artifact.output_weight.to(DTYPE),
    }


def _state_from_split(producer: Any, consumer: Any) -> dict[str, torch.Tensor]:
    return {
        "query": producer.query_weight.to(DTYPE),
        "key": producer.key_weight.to(DTYPE),
        "value": consumer.value_weight.to(DTYPE),
        "output": consumer.output_weight.to(DTYPE),
    }


def _actual_contexts(
    state: dict[str, torch.Tensor],
    cases: tuple[Lfm2CachedAttentionFoldCase, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(_child_case_forward(state, case)["context"] for case in cases)


def _weighted_fixed_score_design(
    qk_state: dict[str, torch.Tensor],
    cases: tuple[Lfm2CachedAttentionFoldCase, ...],
    *,
    horizon_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    design_rows = []
    target_rows = []
    for phase_index, case in enumerate(cases):
        forward = _child_case_forward(
            {
                **qk_state,
                "value": torch.zeros(
                    TARGET_KV_HEADS * HEAD_DIM,
                    TARGET_HIDDEN,
                    dtype=DTYPE,
                ),
                "output": torch.zeros(
                    TARGET_HIDDEN,
                    TARGET_QUERY_HEADS * HEAD_DIM,
                    dtype=DTYPE,
                ),
            },
            case,
        )
        attended_inputs = torch.einsum(
            "bhqk,bki->bqhi",
            forward["probability"],
            case.target_key_value_input,
        )
        target_delta = (
            case.parent_post_attention_residual_target_space
            - case.target_pre_attention_residual
        )
        base_weight = case.query_row_sqrt_weights
        if base_weight is None:
            base_weight = torch.ones(
                case.target_query_input.shape[:2],
                dtype=DTYPE,
            )
        horizon_multiplier = (phase_index + 1) ** (0.5 * horizon_power)
        row_weight = (base_weight * horizon_multiplier).reshape(-1, 1)
        design_rows.append(
            attended_inputs.reshape(
                -1,
                TARGET_QUERY_HEADS * TARGET_HIDDEN,
            )
            * row_weight
        )
        target_rows.append(target_delta.reshape(-1, TARGET_HIDDEN) * row_weight)
    return torch.cat(design_rows, dim=0), torch.cat(target_rows, dim=0)


def _ridge_solve(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge_scale: float,
) -> torch.Tensor:
    gram = design.T @ design
    scale = float(torch.trace(gram).item()) / max(int(gram.shape[0]), 1)
    ridge = ridge_scale * max(scale, 1e-12)
    return torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=DTYPE),
        design.T @ target,
    )


def _fit_sequence_effective_operator(
    qk_state: dict[str, torch.Tensor],
    cases: tuple[Lfm2CachedAttentionFoldCase, ...],
    *,
    ridge_scale: float,
    horizon_power: float,
    value_subspace_anchor: torch.Tensor,
    value_anchor_fraction: float,
    iterations: int = 12,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    design, target = _weighted_fixed_score_design(
        qk_state,
        cases,
        horizon_power=horizon_power,
    )
    effective = _ridge_solve(design, target, ridge_scale=ridge_scale).reshape(
        TARGET_QUERY_HEADS,
        TARGET_HIDDEN,
        TARGET_HIDDEN,
    )
    concatenated = torch.cat(tuple(effective), dim=1)
    left, singular, right = torch.linalg.svd(concatenated, full_matrices=False)
    root = singular[:HEAD_DIM].sqrt()
    value_factor = left[:, :HEAD_DIM] * root.unsqueeze(0)
    output_factor = root.unsqueeze(1) * right[:HEAD_DIM]
    output_blocks = output_factor.reshape(
        HEAD_DIM,
        TARGET_QUERY_HEADS,
        TARGET_HIDDEN,
    ).permute(1, 0, 2).contiguous()

    attended = design.reshape(
        -1,
        TARGET_QUERY_HEADS,
        TARGET_HIDDEN,
    )
    history = []
    accepted = 0
    prior_error = math.inf
    for _iteration in range(iterations):
        features = torch.einsum("nhi,id->nhd", attended, value_factor).reshape(
            -1,
            TARGET_QUERY_HEADS * HEAD_DIM,
        )
        output_transpose = _ridge_solve(
            features,
            target,
            ridge_scale=ridge_scale,
        )
        output_blocks = output_transpose.reshape(
            TARGET_QUERY_HEADS,
            HEAD_DIM,
            TARGET_HIDDEN,
        )
        coefficients = torch.einsum(
            "nhi,hdo->noid",
            attended,
            output_blocks,
        ).reshape(
            -1,
            TARGET_HIDDEN * HEAD_DIM,
        )
        value_vector = _ridge_solve(
            coefficients,
            target.reshape(-1, 1),
            ridge_scale=ridge_scale,
        ).reshape(TARGET_HIDDEN, HEAD_DIM)
        candidate_features = torch.einsum(
            "nhi,id->nhd",
            attended,
            value_vector,
        ).reshape(-1, TARGET_QUERY_HEADS * HEAD_DIM)
        candidate_output_transpose = _ridge_solve(
            candidate_features,
            target,
            ridge_scale=ridge_scale,
        )
        prediction = candidate_features @ candidate_output_transpose
        error = _rrmse(prediction, target)
        if error > prior_error + 1e-12:
            break
        value_factor = value_vector
        output_blocks = candidate_output_transpose.reshape(
            TARGET_QUERY_HEADS,
            HEAD_DIM,
            TARGET_HIDDEN,
        )
        prior_error = error
        history.append(error)
        accepted += 1
    if value_anchor_fraction:
        anchor = value_subspace_anchor.T.to(DTYPE)
        alignment = torch.linalg.lstsq(anchor, value_factor).solution
        aligned_anchor = anchor @ alignment
        value_factor = (
            (1.0 - value_anchor_fraction) * value_factor
            + value_anchor_fraction * aligned_anchor
        )
        anchored_features = torch.einsum(
            "nhi,id->nhd",
            attended,
            value_factor,
        ).reshape(-1, TARGET_QUERY_HEADS * HEAD_DIM)
        anchored_output_transpose = _ridge_solve(
            anchored_features,
            target,
            ridge_scale=ridge_scale,
        )
        anchored_error = _rrmse(
            anchored_features @ anchored_output_transpose,
            target,
        )
        output_blocks = anchored_output_transpose.reshape(
            TARGET_QUERY_HEADS,
            HEAD_DIM,
            TARGET_HIDDEN,
        )
        history.append(anchored_error)
    state = {
        "query": qk_state["query"].clone(),
        "key": qk_state["key"].clone(),
        "value": value_factor.T.contiguous(),
        "output": output_blocks.reshape(
            TARGET_QUERY_HEADS * HEAD_DIM,
            TARGET_HIDDEN,
        )
        .T.contiguous(),
    }
    return state, {
        "algorithm": (
            "fixed-QK causal-attention effective-operator ridge plus shared-value-"
            "subspace SVD and monotone closed-form V/O ALS"
        ),
        "ridge_scale": ridge_scale,
        "horizon_power": horizon_power,
        "value_anchor_fraction": value_anchor_fraction,
        "iterations_requested": iterations,
        "iterations_accepted": accepted,
        "weighted_fit_rrmse_history": history,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }


def _teacher_forced_metrics(
    state: dict[str, torch.Tensor],
    dataset: SequenceDataset,
) -> dict[str, Any]:
    endpoint_predictions = []
    endpoint_targets = []
    delta_predictions = []
    delta_targets = []
    phase_metrics = {}
    for phase, case in zip(dataset.phase_names, dataset.cases, strict=True):
        forward = _child_case_forward(state, case)
        endpoint_target = case.parent_post_attention_residual_target_space
        delta_target = endpoint_target - case.target_pre_attention_residual
        endpoint_predictions.append(forward["endpoint"].reshape(-1, TARGET_HIDDEN))
        endpoint_targets.append(endpoint_target.reshape(-1, TARGET_HIDDEN))
        delta_predictions.append(forward["delta"].reshape(-1, TARGET_HIDDEN))
        delta_targets.append(delta_target.reshape(-1, TARGET_HIDDEN))
        phase_metrics[phase] = {
            "endpoint_rrmse": _rrmse(forward["endpoint"], endpoint_target),
            "endpoint_cosine": _cosine(forward["endpoint"], endpoint_target),
            "delta_rrmse": _rrmse(forward["delta"], delta_target),
            "delta_cosine": _cosine(forward["delta"], delta_target),
        }
    endpoint_prediction = torch.cat(endpoint_predictions, dim=0)
    endpoint_target = torch.cat(endpoint_targets, dim=0)
    delta_prediction = torch.cat(delta_predictions, dim=0)
    delta_target = torch.cat(delta_targets, dim=0)
    return {
        "endpoint_rrmse": _rrmse(endpoint_prediction, endpoint_target),
        "endpoint_cosine": _cosine(endpoint_prediction, endpoint_target),
        "delta_rrmse": _rrmse(delta_prediction, delta_target),
        "delta_cosine": _cosine(delta_prediction, delta_target),
        "phase_metrics": phase_metrics,
    }


def _closed_loop_metrics(
    state: dict[str, torch.Tensor],
    dataset: SequenceDataset,
    parent: ParentOperator,
) -> dict[str, Any]:
    history = dataset.initial_history.clone()
    prefill_case = replace(
        dataset.cases[0],
        target_query_input=history,
        target_key_value_input=history,
        target_pre_attention_residual=history,
    )
    last_endpoint = _child_case_forward(state, prefill_case)["endpoint"][:, -1]
    input_curve = []
    endpoint_curve = []
    endpoint_cosine_curve = []
    for step in range(DECODE_STEPS):
        next_input = _next_latent(
            history[:, -1],
            last_endpoint,
            dataset.forcing[:, step],
            parent.transition,
        )
        input_curve.append(_rrmse(next_input, dataset.decode_inputs[step]))
        history = torch.cat((history, next_input.unsqueeze(1)), dim=1)
        reference_case = dataset.cases[step + 1]
        child_case = replace(
            reference_case,
            target_query_input=next_input.unsqueeze(1),
            target_key_value_input=history,
            target_pre_attention_residual=next_input.unsqueeze(1),
            key_mask=torch.ones(
                int(history.shape[0]),
                1,
                int(history.shape[1]),
                dtype=torch.bool,
            ),
        )
        last_endpoint = _child_case_forward(state, child_case)["endpoint"][:, 0]
        endpoint_curve.append(
            _rrmse(last_endpoint, dataset.decode_endpoints[step])
        )
        endpoint_cosine_curve.append(
            _cosine(last_endpoint, dataset.decode_endpoints[step])
        )
    return {
        "decode_input_rrmse_curve": input_curve,
        "decode_endpoint_rrmse_curve": endpoint_curve,
        "decode_endpoint_cosine_curve": endpoint_cosine_curve,
        "mean_decode_endpoint_rrmse": sum(endpoint_curve) / len(endpoint_curve),
        "final_decode_endpoint_rrmse": endpoint_curve[-1],
        "final_decode_endpoint_cosine": endpoint_cosine_curve[-1],
        "final_decode_input_rrmse": input_curve[-1],
    }


def _score_metrics(producer: Any, dataset: SequenceDataset) -> dict[str, Any]:
    report = evaluate_lfm2_compiled_attention_producer_score_preservation(
        producer,
        dataset.cases,
    )
    return {
        "softmax_fisher_error_mean": report.softmax_fisher_error_mean,
        "softmax_fisher_error_worst": report.softmax_fisher_error_worst,
        "js_divergence_mean": report.js_divergence_mean,
        "js_divergence_worst": report.js_divergence_worst,
        "top1_key_agreement_fraction": report.top1_key_agreement_fraction,
        "winner_margin_sign_agreement_fraction": (
            report.winner_margin_sign_agreement_fraction
        ),
    }


def _compressed_parent_context(
    producer: Any,
    case: Lfm2CachedAttentionFoldCase,
) -> torch.Tensor:
    grouped = []
    for source_heads, weights in zip(
        producer.query_groups,
        producer.query_barycentric_weights,
        strict=True,
    ):
        weight = torch.tensor(weights, dtype=DTYPE).reshape(1, -1, 1, 1)
        grouped.append(
            (
                case.parent_context[:, tuple(source_heads)]
                * weight
            ).sum(dim=1)
        )
    return torch.stack(grouped, dim=1).permute(0, 2, 1, 3).reshape(
        int(case.target_query_input.shape[0]),
        int(case.target_query_input.shape[1]),
        TARGET_QUERY_HEADS * HEAD_DIM,
    )


def _context_readout_metrics(
    state: dict[str, torch.Tensor],
    *,
    producer: Any,
    fit: SequenceDataset,
    holdout: SequenceDataset,
) -> dict[str, float]:
    fit_child = []
    fit_parent = []
    for case in fit.cases:
        child = _child_case_forward(state, case)["context"]
        parent = _compressed_parent_context(producer, case)
        weights = case.query_row_sqrt_weights
        if weights is None:
            weights = torch.ones(case.target_query_input.shape[:2], dtype=DTYPE)
        fit_child.append(child.reshape(-1, child.shape[-1]) * weights.reshape(-1, 1))
        fit_parent.append(
            parent.reshape(-1, parent.shape[-1]) * weights.reshape(-1, 1)
        )
    bridge = _ridge_solve(
        torch.cat(fit_child, dim=0),
        torch.cat(fit_parent, dim=0),
        ridge_scale=1e-8,
    )
    predictions = []
    targets = []
    for case in holdout.cases:
        child = _child_case_forward(state, case)["context"]
        parent = _compressed_parent_context(producer, case)
        predictions.append(child.reshape(-1, child.shape[-1]) @ bridge)
        targets.append(parent.reshape(-1, parent.shape[-1]))
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    return {
        "linear_readout_rrmse": _rrmse(prediction, target),
        "linear_readout_cosine": _cosine(prediction, target),
    }


def _compile_current_frontier(
    parent: ParentOperator,
    fit: SequenceDataset,
    *,
    seed: int,
) -> tuple[Any, Any, Any, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    templates = {
        "query": torch.randn(
            TARGET_QUERY_HEADS * HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        ),
        "key": torch.randn(
            TARGET_KV_HEADS * HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        ),
        "value": torch.randn(
            TARGET_KV_HEADS * HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        ),
        "output": torch.randn(
            TARGET_HIDDEN,
            TARGET_QUERY_HEADS * HEAD_DIM,
            generator=generator,
            dtype=DTYPE,
        ),
    }
    common = {
        "source_query_weight": parent.query,
        "source_key_weight": parent.key,
        "source_value_weight": parent.value,
        "source_output_weight": parent.output,
        "target_query_weight_template": templates["query"],
        "target_key_weight_template": templates["key"],
        "target_value_weight_template": templates["value"],
        "target_output_weight_template": templates["output"],
        "cases": fit.cases,
        "head_dim": HEAD_DIM,
        "residual_assignment_sha256": "a" * 64,
        "ridge": 1e-8,
        "cluster_ridge": 1e-8,
        "max_prior_delta_ratio": None,
    }
    direct = compile_lfm2_structural_residual_attention_fold(**common)
    producer = compile_lfm2_structural_attention_producer_fold(**common)
    producer_state = {
        "query": producer.query_weight.to(DTYPE),
        "key": producer.key_weight.to(DTYPE),
        "value": producer.value_weight.to(DTYPE),
        "output": templates["output"],
    }
    actual_contexts = _actual_contexts(producer_state, fit.cases)
    split_consumer = compile_lfm2_structural_attention_value_output_fold(
        producer=producer,
        target_value_weight_template=templates["value"],
        target_output_weight_template=templates["output"],
        actual_child_contexts=actual_contexts,
        cases=fit.cases,
        live_value_weight=producer.value_weight,
        value_weight_prior=producer.value_weight,
        output_weight_prior=templates["output"],
        ridge=1e-8,
        context_ridge=1e-6,
        max_prior_delta_ratio=None,
        iterations=2,
    )
    direct_state = _state_from_direct(direct)
    split_state = _state_from_split(producer, split_consumer)
    return direct, producer, split_consumer, direct_state, split_state


def _candidate_report(
    state: dict[str, torch.Tensor],
    *,
    producer: Any,
    fit: SequenceDataset,
    selection: SequenceDataset,
    holdout: SequenceDataset,
    parent: ParentOperator,
    score_metrics: dict[str, Any],
    fit_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_sha256": _sha_tensor_bundle(state),
        "logical_parameters": sum(int(value.numel()) for value in state.values()),
        "selection": _teacher_forced_metrics(state, selection),
        "holdout": _teacher_forced_metrics(state, holdout),
        "holdout_attention_map": score_metrics,
        "holdout_context_preservation": _context_readout_metrics(
            state,
            producer=producer,
            fit=fit,
            holdout=holdout,
        ),
        "holdout_closed_loop": _closed_loop_metrics(state, holdout, parent),
        "fit_receipt": fit_receipt,
    }


def _run_seed(seed: int) -> dict[str, Any]:
    parent = _make_parent(seed)
    fit = _make_dataset(parent, seed=seed + 10_000, batch_size=28)
    selection = _make_dataset(parent, seed=seed + 20_000, batch_size=14)
    holdout = _make_dataset(parent, seed=seed + 30_000, batch_size=28)
    (
        direct,
        producer,
        split_consumer,
        direct_state,
        split_state,
    ) = _compile_current_frontier(parent, fit, seed=seed + 40_000)
    direct_qk_sha = _sha_tensor_bundle(
        {"query": direct_state["query"], "key": direct_state["key"]}
    )
    producer_qk_sha = _sha_tensor_bundle(
        {"query": producer.query_weight, "key": producer.key_weight}
    )
    if direct_qk_sha != producer_qk_sha:
        raise RuntimeError("Current direct and split candidates did not share Q/K.")
    attention_map = _score_metrics(producer, holdout)

    direct_report = _candidate_report(
        direct_state,
        producer=producer,
        fit=fit,
        selection=selection,
        holdout=holdout,
        parent=parent,
        score_metrics=attention_map,
        fit_receipt={
            "algorithm": "production direct_monolithic",
            "fit_residual_endpoint_rrmse": direct.residual_endpoint_relative_rmse,
            "fit_centered_score_rrmse": direct.centered_score_relative_rmse,
        },
    )
    split_report = _candidate_report(
        split_state,
        producer=producer,
        fit=fit,
        selection=selection,
        holdout=holdout,
        parent=parent,
        score_metrics=attention_map,
        fit_receipt={
            "algorithm": "production residual_attention_split",
            "fit_residual_endpoint_rrmse": (
                split_consumer.residual_endpoint_relative_rmse
            ),
            "fit_history": list(split_consumer.fit_relative_rmse_history),
            "iterations_accepted": split_consumer.iterations_accepted,
        },
    )

    qk_state = {
        "query": producer.query_weight.to(DTYPE),
        "key": producer.key_weight.to(DTYPE),
    }
    alternatives = []
    for ridge_scale in (1e-10, 1e-8, 1e-6, 1e-4):
        for horizon_power in (0.0, 0.5, 1.0):
            for value_anchor_fraction in (0.0, 0.1, 0.25, 0.5):
                state, receipt = _fit_sequence_effective_operator(
                    qk_state,
                    fit.cases,
                    ridge_scale=ridge_scale,
                    horizon_power=horizon_power,
                    value_subspace_anchor=producer.value_weight,
                    value_anchor_fraction=value_anchor_fraction,
                )
                selection_metrics = _teacher_forced_metrics(state, selection)
                selection_rollout = _closed_loop_metrics(state, selection, parent)
                selection_context = _context_readout_metrics(
                    state,
                    producer=producer,
                    fit=fit,
                    holdout=selection,
                )
                composite = (
                    selection_metrics["endpoint_rrmse"]
                    + 0.20 * selection_rollout["mean_decode_endpoint_rrmse"]
                    + 0.10 * selection_rollout["final_decode_endpoint_rrmse"]
                    + 0.20 * selection_context["linear_readout_rrmse"]
                )
                alternatives.append((composite, state, receipt))
    alternatives.sort(
        key=lambda value: (
            value[0],
            value[2]["ridge_scale"],
            value[2]["horizon_power"],
            value[2]["value_anchor_fraction"],
        )
    )
    alternative_composite, alternative_state, alternative_receipt = alternatives[0]
    alternative_receipt = {
        **alternative_receipt,
        "bounded_frontier_size": len(alternatives),
        "selection_composite": alternative_composite,
        "selection_policy": (
            "endpoint_rrmse + 0.20*mean_closed_loop_endpoint_rrmse + "
            "0.10*final_closed_loop_endpoint_rrmse + "
            "0.20*context_linear_readout_rrmse"
        ),
    }
    alternative_report = _candidate_report(
        alternative_state,
        producer=producer,
        fit=fit,
        selection=selection,
        holdout=holdout,
        parent=parent,
        score_metrics=attention_map,
        fit_receipt=alternative_receipt,
    )

    current = {
        "direct_monolithic": direct_report,
        "residual_attention_split": split_report,
    }
    current_winner = min(
        current,
        key=lambda name: current[name]["selection"]["endpoint_rrmse"],
    )
    return {
        "seed": seed,
        "source_state_sha256": _sha_tensor_bundle(
            {
                "query": parent.query,
                "key": parent.key,
                "value": parent.value,
                "output": parent.output,
            }
        ),
        "current_frontier_winner": current_winner,
        "current_frontier": current,
        "sequence_effective_operator": alternative_report,
        "attention_qk_identical_across_all_candidates": True,
    }


def _aggregate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "direct_monolithic",
        "residual_attention_split",
        "current_frontier_winner",
        "sequence_effective_operator",
    )
    aggregate = {}
    for name in names:
        reports = []
        for seed_report in seed_reports:
            if name == "current_frontier_winner":
                report = seed_report["current_frontier"][
                    seed_report["current_frontier_winner"]
                ]
            elif name == "sequence_effective_operator":
                report = seed_report[name]
            else:
                report = seed_report["current_frontier"][name]
            reports.append(report)
        aggregate[name] = {
            "mean_holdout_endpoint_rrmse": sum(
                value["holdout"]["endpoint_rrmse"] for value in reports
            )
            / len(reports),
            "mean_holdout_endpoint_cosine": sum(
                value["holdout"]["endpoint_cosine"] for value in reports
            )
            / len(reports),
            "mean_holdout_delta_rrmse": sum(
                value["holdout"]["delta_rrmse"] for value in reports
            )
            / len(reports),
            "mean_holdout_delta_cosine": sum(
                value["holdout"]["delta_cosine"] for value in reports
            )
            / len(reports),
            "mean_attention_js_divergence": sum(
                value["holdout_attention_map"]["js_divergence_mean"]
                for value in reports
            )
            / len(reports),
            "mean_context_readout_rrmse": sum(
                value["holdout_context_preservation"]["linear_readout_rrmse"]
                for value in reports
            )
            / len(reports),
            "mean_context_readout_cosine": sum(
                value["holdout_context_preservation"]["linear_readout_cosine"]
                for value in reports
            )
            / len(reports),
            "mean_closed_loop_endpoint_rrmse": sum(
                value["holdout_closed_loop"]["mean_decode_endpoint_rrmse"]
                for value in reports
            )
            / len(reports),
            "mean_final_closed_loop_endpoint_rrmse": sum(
                value["holdout_closed_loop"]["final_decode_endpoint_rrmse"]
                for value in reports
            )
            / len(reports),
            "mean_final_closed_loop_endpoint_cosine": sum(
                value["holdout_closed_loop"]["final_decode_endpoint_cosine"]
                for value in reports
            )
            / len(reports),
        }
    baseline = aggregate["current_frontier_winner"]
    candidate = aggregate["sequence_effective_operator"]
    improvements = {
        "holdout_endpoint_rrmse_fraction": (
            baseline["mean_holdout_endpoint_rrmse"]
            - candidate["mean_holdout_endpoint_rrmse"]
        )
        / baseline["mean_holdout_endpoint_rrmse"],
        "holdout_delta_rrmse_fraction": (
            baseline["mean_holdout_delta_rrmse"]
            - candidate["mean_holdout_delta_rrmse"]
        )
        / baseline["mean_holdout_delta_rrmse"],
        "closed_loop_mean_rrmse_fraction": (
            baseline["mean_closed_loop_endpoint_rrmse"]
            - candidate["mean_closed_loop_endpoint_rrmse"]
        )
        / baseline["mean_closed_loop_endpoint_rrmse"],
        "closed_loop_final_rrmse_fraction": (
            baseline["mean_final_closed_loop_endpoint_rrmse"]
            - candidate["mean_final_closed_loop_endpoint_rrmse"]
        )
        / baseline["mean_final_closed_loop_endpoint_rrmse"],
        "context_readout_rrmse_fraction": (
            baseline["mean_context_readout_rrmse"]
            - candidate["mean_context_readout_rrmse"]
        )
        / baseline["mean_context_readout_rrmse"],
        "attention_js_divergence_delta": (
            candidate["mean_attention_js_divergence"]
            - baseline["mean_attention_js_divergence"]
        ),
    }
    strict_win = all(
        (
            seed_report["sequence_effective_operator"]["holdout"][
                "endpoint_rrmse"
            ]
            < seed_report["current_frontier"][
                seed_report["current_frontier_winner"]
            ]["holdout"]["endpoint_rrmse"]
            and seed_report["sequence_effective_operator"]["holdout_closed_loop"][
                "mean_decode_endpoint_rrmse"
            ]
            < seed_report["current_frontier"][
                seed_report["current_frontier_winner"]
            ]["holdout_closed_loop"]["mean_decode_endpoint_rrmse"]
        )
        for seed_report in seed_reports
    )
    strict_frontier_win = all(
        all(
            (
                seed_report["sequence_effective_operator"]["holdout"][
                    "endpoint_rrmse"
                ]
                < seed_report["current_frontier"][name]["holdout"][
                    "endpoint_rrmse"
                ]
                and seed_report["sequence_effective_operator"][
                    "holdout_closed_loop"
                ]["mean_decode_endpoint_rrmse"]
                < seed_report["current_frontier"][name][
                    "holdout_closed_loop"
                ]["mean_decode_endpoint_rrmse"]
            )
            for name in ("direct_monolithic", "residual_attention_split")
        )
        for seed_report in seed_reports
    )
    return {
        "candidate_metrics": aggregate,
        "alternative_improvement_over_current_winner": improvements,
        "alternative_strictly_won_endpoint_and_mean_drift_every_seed": strict_win,
        "alternative_strictly_won_against_both_frontier_arms_every_seed": (
            strict_frontier_win
        ),
        "integration_candidate_supported": bool(
            strict_frontier_win
            and improvements["holdout_endpoint_rrmse_fraction"] > 0.0
            and improvements["closed_loop_mean_rrmse_fraction"] > 0.0
            and improvements["context_readout_rrmse_fraction"] >= 0.0
            and abs(improvements["attention_js_divergence_delta"]) <= 1e-15
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seeds = [1101, 2202, 3303]
    seed_reports = [_run_seed(seed) for seed in seeds]
    report = {
        "schema": "neural-forge-ephemeral-residual-attention-sequence-experiment-v1",
        "setup": {
            "seeds": seeds,
            "source_hidden": SOURCE_HIDDEN,
            "target_hidden": TARGET_HIDDEN,
            "hidden_reduction": SOURCE_HIDDEN / TARGET_HIDDEN,
            "source_query_heads": SOURCE_QUERY_HEADS,
            "target_query_heads": TARGET_QUERY_HEADS,
            "query_head_reduction": SOURCE_QUERY_HEADS / TARGET_QUERY_HEADS,
            "source_kv_heads": SOURCE_KV_HEADS,
            "target_kv_heads": TARGET_KV_HEADS,
            "kv_head_reduction": SOURCE_KV_HEADS / TARGET_KV_HEADS,
            "head_dim": HEAD_DIM,
            "prefill_tokens": PREFILL_LENGTH,
            "decode_steps": DECODE_STEPS,
            "fit_sequences_per_seed": 28,
            "selection_sequences_per_seed": 14,
            "heldout_sequences_per_seed": 28,
            "source_attention_parameters": (
                SOURCE_HIDDEN * SOURCE_HIDDEN
                + 2 * SOURCE_KV_HEADS * HEAD_DIM * SOURCE_HIDDEN
                + SOURCE_HIDDEN * SOURCE_QUERY_HEADS * HEAD_DIM
            ),
            "target_attention_parameters": (
                TARGET_QUERY_HEADS * HEAD_DIM * TARGET_HIDDEN
                + 2 * TARGET_KV_HEADS * HEAD_DIM * TARGET_HIDDEN
                + TARGET_HIDDEN * TARGET_QUERY_HEADS * HEAD_DIM
            ),
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "heldout_used_for_selection": False,
        },
        "seed_reports": seed_reports,
        "aggregate": _aggregate(seed_reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
