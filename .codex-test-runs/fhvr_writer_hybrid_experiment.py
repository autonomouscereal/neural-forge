from __future__ import annotations

import json
import itertools
import math
from dataclasses import dataclass

import torch

from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
    causal_depthwise_context,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    Lfm2CpcfShortConvNumericalFrontierRequest,
    Lfm2CpcfShortConvV4FoldCase,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_fhvr_mf import (
    FHVR_MF_CANDIDATE_KIND,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)


SOURCE_HIDDEN = 8
TARGET_HIDDEN = 4
SOURCE_FIR_LENGTH = 6
TARGET_FIR_LENGTH = 3
FIT_TRAJECTORIES = 6
FIT_TOKENS = 24
SELECTION_TRAJECTORIES = 8
SELECTION_TOKENS = 48
HELDOUT_TRAJECTORIES = 16
HELDOUT_TOKENS = 96
BLEND_FRACTIONS = (0.25, 0.5, 0.75)
SEEDS = (1401, 2502, 3603, 4704, 5805)


@dataclass(frozen=True)
class Teacher:
    source_in_weight: torch.Tensor
    source_in_bias: torch.Tensor
    source_fir: torch.Tensor
    source_fir_bias: torch.Tensor
    source_writer: torch.Tensor
    source_writer_bias: torch.Tensor
    output_encoder: torch.Tensor
    input_lift: torch.Tensor
    mlp_gate: torch.Tensor
    mlp_up: torch.Tensor
    mlp_down: torch.Tensor
    mlp_gate_bias: torch.Tensor
    mlp_up_bias: torch.Tensor
    mlp_down_bias: torch.Tensor
    proxy_gate: torch.Tensor
    proxy_up: torch.Tensor


@dataclass(frozen=True)
class Trajectory:
    child: torch.Tensor
    parent_input: torch.Tensor
    parent_b: torch.Tensor
    parent_c: torch.Tensor
    parent_x: torch.Tensor
    parent_context: torch.Tensor
    parent_write: torch.Tensor
    parent_mlp_write: torch.Tensor
    valid_mask: torch.Tensor


def _randn(
    generator: torch.Generator,
    shape: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float64) * scale


def _teacher(seed: int) -> Teacher:
    generator = torch.Generator().manual_seed(seed)
    base_fir = torch.tensor(
        [0.07, -0.11, 0.18, 0.29, 0.43, 0.67],
        dtype=torch.float64,
    )
    intermediate = 12
    return Teacher(
        source_in_weight=_randn(
            generator,
            (3 * SOURCE_HIDDEN, SOURCE_HIDDEN),
            0.23,
        ),
        source_in_bias=_randn(generator, (3 * SOURCE_HIDDEN,), 0.025),
        source_fir=base_fir.repeat(SOURCE_HIDDEN, 1),
        source_fir_bias=_randn(generator, (SOURCE_HIDDEN,), 0.015),
        source_writer=_randn(
            generator,
            (SOURCE_HIDDEN, SOURCE_HIDDEN),
            0.26,
        ),
        source_writer_bias=_randn(generator, (SOURCE_HIDDEN,), 0.015),
        output_encoder=_randn(
            generator,
            (SOURCE_HIDDEN, TARGET_HIDDEN),
            0.38,
        ),
        input_lift=_randn(
            generator,
            (TARGET_HIDDEN, SOURCE_HIDDEN),
            0.52,
        ),
        mlp_gate=_randn(generator, (SOURCE_HIDDEN, intermediate), 0.24),
        mlp_up=_randn(generator, (SOURCE_HIDDEN, intermediate), 0.24),
        mlp_down=_randn(generator, (intermediate, SOURCE_HIDDEN), 0.20),
        mlp_gate_bias=_randn(generator, (intermediate,), 0.02),
        mlp_up_bias=_randn(generator, (intermediate,), 0.02),
        mlp_down_bias=_randn(generator, (SOURCE_HIDDEN,), 0.01),
        proxy_gate=_randn(generator, (TARGET_HIDDEN, 12), 0.45),
        proxy_up=_randn(generator, (TARGET_HIDDEN, 12), 0.45),
    )


def _child_trajectories(
    teacher: Teacher,
    *,
    seed: int,
    count: int,
    tokens: int,
) -> tuple[Trajectory, ...]:
    generator = torch.Generator().manual_seed(seed)
    result = []
    for trajectory_index in range(count):
        innovations = _randn(generator, (1, tokens, TARGET_HIDDEN), 0.75)
        child = torch.empty_like(innovations)
        previous = _randn(generator, (1, TARGET_HIDDEN), 0.25)
        for time_index in range(tokens):
            previous = (
                0.82 * previous
                + innovations[:, time_index]
                + 0.08
                * torch.sin(
                    innovations[:, time_index]
                    * float(trajectory_index + 1)
                )
            )
            child[:, time_index] = previous
        parent_input = child @ teacher.input_lift
        projection = (
            parent_input @ teacher.source_in_weight.T
            + teacher.source_in_bias
        )
        parent_b, parent_c, parent_x = projection.chunk(3, dim=-1)
        parent_context = causal_depthwise_context(
            parent_b,
            parent_c,
            parent_x,
            teacher.source_fir,
            teacher.source_fir_bias,
        )
        source_write = (
            parent_context @ teacher.source_writer.T
            + teacher.source_writer_bias
        )
        parent_write = source_write @ teacher.output_encoder
        parent_post = parent_input + source_write
        gate = parent_post @ teacher.mlp_gate + teacher.mlp_gate_bias
        up = parent_post @ teacher.mlp_up + teacher.mlp_up_bias
        source_mlp = (
            torch.nn.functional.silu(gate) * up
        ) @ teacher.mlp_down + teacher.mlp_down_bias
        parent_mlp_write = source_mlp @ teacher.output_encoder
        valid_mask = torch.ones((1, tokens), dtype=torch.bool)
        if trajectory_index % 3 == 1:
            valid_mask[:, -1] = False
        result.append(
            Trajectory(
                child=child,
                parent_input=parent_input,
                parent_b=parent_b,
                parent_c=parent_c,
                parent_x=parent_x,
                parent_context=parent_context,
                parent_write=parent_write,
                parent_mlp_write=parent_mlp_write,
                valid_mask=valid_mask,
            )
        )
    return tuple(result)


def _request(
    teacher: Teacher,
    fit: tuple[Trajectory, ...],
) -> Lfm2CpcfShortConvNumericalFrontierRequest:
    cases = []
    contexts = {}
    suffix_start = SOURCE_FIR_LENGTH + 2
    for index, trajectory in enumerate(fit):
        case_id = f"fit-{index:02d}"
        cases.append(
            Lfm2CpcfShortConvV4FoldCase(
                case_id=case_id,
                split="fit",
                suffix_start=suffix_start,
                actual_child_operator_input_history=trajectory.child,
                parent_operator_input_history=trajectory.parent_input,
                parent_b_history=trajectory.parent_b,
                parent_c_history=trajectory.parent_c,
                parent_x_history=trajectory.parent_x,
                parent_post_operator_residual_target_space_suffix=(
                    trajectory.child[:, suffix_start:]
                    + trajectory.parent_write[:, suffix_start:]
                ),
                actual_child_pre_operator_residual_suffix=(
                    trajectory.child[:, suffix_start:]
                ),
                valid_prefix_mask=trajectory.valid_mask,
            )
        )
        contexts[case_id] = trajectory.parent_context
    return Lfm2CpcfShortConvNumericalFrontierRequest(
        source_tensors={
            "source_in_proj_weight": teacher.source_in_weight,
            "source_in_proj_bias": teacher.source_in_bias,
            "source_fir_kernel": teacher.source_fir,
            "source_fir_bias": teacher.source_fir_bias,
            "source_out_proj_weight": teacher.source_writer,
            "source_out_proj_bias": teacher.source_writer_bias,
            "output_encoder": teacher.output_encoder,
        },
        cases=tuple(cases),
        parent_contexts=contexts,
        target_templates={
            "in_proj_weight": torch.zeros(
                (3 * TARGET_HIDDEN, TARGET_HIDDEN),
                dtype=torch.float64,
            ),
            "in_proj_bias": torch.zeros(
                (3 * TARGET_HIDDEN,),
                dtype=torch.float64,
            ),
            "fir_kernel": torch.zeros(
                (TARGET_HIDDEN, TARGET_FIR_LENGTH),
                dtype=torch.float64,
            ),
            "fir_bias": torch.zeros(
                (TARGET_HIDDEN,),
                dtype=torch.float64,
            ),
            "writer_weight": torch.zeros(
                (TARGET_HIDDEN, TARGET_HIDDEN),
                dtype=torch.float64,
            ),
            "writer_bias": torch.zeros(
                (TARGET_HIDDEN,),
                dtype=torch.float64,
            ),
        },
        policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-6,),
            variable_projection_passes=2,
            target_dtype="float64",
        ),
    )


def _candidate_state(candidate) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in candidate.state_dict().items()
        if value is not None
    }


def _blend(
    writer_state: dict[str, torch.Tensor],
    fhvr_state: dict[str, torch.Tensor],
    fraction: float,
) -> dict[str, torch.Tensor]:
    return {
        name: (
            writer_state[name]
            + fraction * (fhvr_state[name] - writer_state[name])
        ).contiguous()
        for name in writer_state
    }


def _candidate_context(
    state: dict[str, torch.Tensor],
    child: torch.Tensor,
) -> torch.Tensor:
    projection = child @ state["in_proj_weight"].T + state["in_proj_bias"]
    b_value, c_value, x_value = projection.chunk(3, dim=-1)
    return causal_depthwise_context(
        b_value,
        c_value,
        x_value,
        state["fir_kernel"],
        state["fir_bias"],
    )


def _operator(
    state: dict[str, torch.Tensor],
    child: torch.Tensor,
) -> torch.Tensor:
    context = _candidate_context(state, child)
    return context @ state["writer_weight"].T + state["writer_bias"]


def _gauge_align_fhvr_to_writer(
    writer_state: dict[str, torch.Tensor],
    fhvr_state: dict[str, torch.Tensor],
    fit: tuple[Trajectory, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    writer_context = _stack_valid(
        fit,
        lambda trajectory: _candidate_context(
            writer_state,
            trajectory.child,
        ),
    )
    fhvr_context = _stack_valid(
        fit,
        lambda trajectory: _candidate_context(
            fhvr_state,
            trajectory.child,
        ),
    )
    normalized_writer = writer_context / writer_context.square().sum(
        dim=0,
        keepdim=True,
    ).sqrt().clamp_min(torch.finfo(writer_context.dtype).eps)
    normalized_fhvr = fhvr_context / fhvr_context.square().sum(
        dim=0,
        keepdim=True,
    ).sqrt().clamp_min(torch.finfo(fhvr_context.dtype).eps)
    correlation = normalized_writer.T @ normalized_fhvr
    permutation = max(
        itertools.permutations(range(TARGET_HIDDEN)),
        key=lambda candidate: sum(
            abs(float(correlation[writer_channel, fhvr_channel].item()))
            for writer_channel, fhvr_channel in enumerate(candidate)
        ),
    )
    ordered_context = fhvr_context[:, permutation]
    signs = torch.sign(
        (writer_context * ordered_context).sum(dim=0)
    )
    signs[signs == 0.0] = 1.0
    writer_rms = writer_context.square().mean(dim=0).sqrt()
    fhvr_rms = ordered_context.square().mean(dim=0).sqrt().clamp_min(
        torch.finfo(ordered_context.dtype).eps
    )
    scales = (writer_rms / fhvr_rms).clamp(0.125, 8.0)
    context_gauge = signs * scales
    aligned = {name: value.clone() for name, value in fhvr_state.items()}
    hidden = TARGET_HIDDEN
    in_weight = fhvr_state["in_proj_weight"].reshape(3, hidden, hidden)
    in_bias = fhvr_state["in_proj_bias"].reshape(3, hidden)
    aligned_weight = in_weight[:, permutation].clone()
    aligned_bias = in_bias[:, permutation].clone()
    aligned_weight[1] = aligned_weight[1] * context_gauge.unsqueeze(-1)
    aligned_bias[1] = aligned_bias[1] * context_gauge
    aligned["in_proj_weight"] = aligned_weight.reshape(3 * hidden, hidden)
    aligned["in_proj_bias"] = aligned_bias.reshape(3 * hidden)
    aligned["fir_kernel"] = fhvr_state["fir_kernel"][list(permutation)].clone()
    aligned["fir_bias"] = fhvr_state["fir_bias"][list(permutation)].clone()
    aligned["writer_weight"] = (
        fhvr_state["writer_weight"][:, list(permutation)]
        / context_gauge.unsqueeze(0)
    ).clone()
    before = _stack_valid(
        fit,
        lambda trajectory: _operator(fhvr_state, trajectory.child),
    )
    after = _stack_valid(
        fit,
        lambda trajectory: _operator(aligned, trajectory.child),
    )
    maximum_invariance_error = float((before - after).abs().max().item())
    return aligned, {
        "permutation": list(permutation),
        "context_gauge": [float(value) for value in context_gauge],
        "maximum_fit_operator_invariance_error": maximum_invariance_error,
    }


def _response_refactor(
    *,
    base_state: dict[str, torch.Tensor],
    writer_state: dict[str, torch.Tensor],
    fhvr_state: dict[str, torch.Tensor],
    fhvr_fraction: float,
    fit: tuple[Trajectory, ...],
) -> dict[str, torch.Tensor]:
    design_rows = []
    target_rows = []
    for trajectory in fit:
        context = _candidate_context(base_state, trajectory.child)
        ones = torch.ones(
            (*context.shape[:-1], 1),
            dtype=context.dtype,
            device=context.device,
        )
        design = torch.cat((context, ones), dim=-1)
        writer_response = _operator(writer_state, trajectory.child)
        fhvr_response = _operator(fhvr_state, trajectory.child)
        desired = writer_response + fhvr_fraction * (
            fhvr_response - writer_response
        )
        design_rows.append(design[trajectory.valid_mask])
        target_rows.append(desired[trajectory.valid_mask])
    design = torch.cat(design_rows, dim=0)
    target = torch.cat(target_rows, dim=0)
    prior = torch.cat(
        (
            base_state["writer_weight"].T,
            base_state["writer_bias"].unsqueeze(0),
        ),
        dim=0,
    )
    gram = design.T @ design
    scale = float(gram.diagonal().mean().item())
    ridge = max(scale * 1e-6, 1e-12)
    solved = torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=gram.dtype),
        design.T @ target + ridge * prior,
    )
    result = {name: value.clone() for name, value in base_state.items()}
    result["writer_weight"] = solved[:-1].T.contiguous()
    result["writer_bias"] = solved[-1].contiguous()
    return result


def _channel_dictionary_hybrid(
    *,
    writer_state: dict[str, torch.Tensor],
    fhvr_state: dict[str, torch.Tensor],
    fhvr_fraction: float,
    fit: tuple[Trajectory, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    writer_context = _stack_valid(
        fit,
        lambda trajectory: _candidate_context(
            writer_state,
            trajectory.child,
        ),
    )
    fhvr_context = _stack_valid(
        fit,
        lambda trajectory: _candidate_context(
            fhvr_state,
            trajectory.child,
        ),
    )
    target = _stack_valid(fit, lambda trajectory: trajectory.parent_write)
    fhvr_quota = int(round(TARGET_HIDDEN * fhvr_fraction))
    fhvr_quota = max(1, min(TARGET_HIDDEN - 1, fhvr_quota))
    writer_quota = TARGET_HIDDEN - fhvr_quota
    source_features = {
        "writer": writer_context,
        "fhvr": fhvr_context,
    }
    remaining = {
        "writer": writer_quota,
        "fhvr": fhvr_quota,
    }
    selected: list[tuple[str, int]] = []
    residual = target - target.mean(dim=0, keepdim=True)
    for _ in range(TARGET_HIDDEN):
        best = None
        for source_name in ("writer", "fhvr"):
            if remaining[source_name] == 0:
                continue
            for channel in range(TARGET_HIDDEN):
                identity = (source_name, channel)
                if identity in selected:
                    continue
                feature = source_features[source_name][:, channel]
                centered = feature - feature.mean()
                energy = float(centered.square().sum().item())
                score = float(
                    (centered.unsqueeze(-1) * residual).sum(dim=0).square().sum().item()
                    / max(energy, torch.finfo(feature.dtype).eps)
                )
                key = (-score, source_name, channel)
                if best is None or key < best[0]:
                    best = (key, identity)
        assert best is not None
        selected.append(best[1])
        remaining[best[1][0]] -= 1
        design = torch.stack(
            [
                source_features[source_name][:, channel]
                for source_name, channel in selected
            ],
            dim=-1,
        )
        design = torch.cat(
            (
                design,
                torch.ones(
                    (int(design.shape[0]), 1),
                    dtype=design.dtype,
                ),
            ),
            dim=-1,
        )
        solved = torch.linalg.lstsq(design, target).solution
        residual = target - design @ solved
    roles = {}
    for role in ("in_proj_weight", "in_proj_bias"):
        value_shape = (
            (3, TARGET_HIDDEN, TARGET_HIDDEN)
            if role == "in_proj_weight"
            else (3, TARGET_HIDDEN)
        )
        writer_value = writer_state[role].reshape(value_shape)
        fhvr_value = fhvr_state[role].reshape(value_shape)
        source_values = {
            "writer": writer_value,
            "fhvr": fhvr_value,
        }
        role_rows = [
            source_values[source_name][:, channel]
            for source_name, channel in selected
        ]
        stacked = torch.stack(role_rows, dim=1)
        roles[role] = stacked.reshape(writer_state[role].shape).contiguous()
    fir_kernel = torch.stack(
        [
            (
                writer_state["fir_kernel"]
                if source_name == "writer"
                else fhvr_state["fir_kernel"]
            )[channel]
            for source_name, channel in selected
        ],
        dim=0,
    )
    fir_bias = torch.stack(
        [
            (
                writer_state["fir_bias"]
                if source_name == "writer"
                else fhvr_state["fir_bias"]
            )[channel]
            for source_name, channel in selected
        ],
        dim=0,
    )
    selected_context = torch.stack(
        [
            source_features[source_name][:, channel]
            for source_name, channel in selected
        ],
        dim=-1,
    )
    design = torch.cat(
        (
            selected_context,
            torch.ones(
                (int(selected_context.shape[0]), 1),
                dtype=selected_context.dtype,
            ),
        ),
        dim=-1,
    )
    prior_columns = torch.stack(
        [
            (
                writer_state["writer_weight"]
                if source_name == "writer"
                else fhvr_state["writer_weight"]
            )[:, channel]
            for source_name, channel in selected
        ],
        dim=1,
    )
    prior_bias = (
        writer_state["writer_bias"] + fhvr_state["writer_bias"]
    ) / 2.0
    prior = torch.cat((prior_columns.T, prior_bias.unsqueeze(0)), dim=0)
    gram = design.T @ design
    scale = float(gram.diagonal().mean().item())
    ridge = max(scale * 1e-6, 1e-12)
    solved = torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=gram.dtype),
        design.T @ target + ridge * prior,
    )
    return {
        **roles,
        "fir_kernel": fir_kernel.contiguous(),
        "fir_bias": fir_bias.contiguous(),
        "writer_weight": solved[:-1].T.contiguous(),
        "writer_bias": solved[-1].contiguous(),
    }, {
        "fhvr_fraction": fhvr_fraction,
        "writer_channel_quota": writer_quota,
        "fhvr_channel_quota": fhvr_quota,
        "selected_channels": [
            {"source": source_name, "channel": channel}
            for source_name, channel in selected
        ],
        "fit_only": True,
    }


def _solve_role_c(
    state: dict[str, torch.Tensor],
    fit: tuple[Trajectory, ...],
    desired_contexts: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    hidden = TARGET_HIDDEN
    roles_weight = state["in_proj_weight"].reshape(3, hidden, hidden).clone()
    roles_bias = state["in_proj_bias"].reshape(3, hidden).clone()
    for channel in range(hidden):
        design_rows = []
        target_rows = []
        for trajectory, desired in zip(fit, desired_contexts, strict=True):
            projection = (
                trajectory.child @ state["in_proj_weight"].T
                + state["in_proj_bias"]
            )
            b_value, c_value, x_value = projection.chunk(3, dim=-1)
            filtered = causal_depthwise_context(
                b_value,
                torch.ones_like(c_value),
                x_value,
                state["fir_kernel"],
                state["fir_bias"],
            )
            augmented = torch.cat(
                (
                    trajectory.child,
                    torch.ones(
                        (*trajectory.child.shape[:-1], 1),
                        dtype=trajectory.child.dtype,
                    ),
                ),
                dim=-1,
            )
            design = filtered[..., channel].unsqueeze(-1) * augmented
            design_rows.append(design[trajectory.valid_mask])
            target_rows.append(desired[..., channel][trajectory.valid_mask])
        design = torch.cat(design_rows, dim=0)
        target = torch.cat(target_rows, dim=0)
        prior = torch.cat(
            (
                roles_weight[1, channel],
                roles_bias[1, channel].reshape(1),
            )
        )
        gram = design.T @ design
        scale = float(gram.diagonal().mean().item())
        ridge = max(scale * 1e-6, 1e-12)
        solved = torch.linalg.solve(
            gram + ridge * torch.eye(int(gram.shape[0]), dtype=gram.dtype),
            design.T @ target + ridge * prior,
        )
        roles_weight[1, channel] = solved[:-1]
        roles_bias[1, channel] = solved[-1]
    result = {name: value.clone() for name, value in state.items()}
    result["in_proj_weight"] = roles_weight.reshape(3 * hidden, hidden).contiguous()
    result["in_proj_bias"] = roles_bias.reshape(3 * hidden).contiguous()
    return result


def _solve_role_fir(
    state: dict[str, torch.Tensor],
    fit: tuple[Trajectory, ...],
    desired_contexts: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    hidden = TARGET_HIDDEN
    fir_length = TARGET_FIR_LENGTH
    kernel = state["fir_kernel"].clone()
    bias = state["fir_bias"].clone()
    for channel in range(hidden):
        design_rows = []
        target_rows = []
        for trajectory, desired in zip(fit, desired_contexts, strict=True):
            projection = (
                trajectory.child @ state["in_proj_weight"].T
                + state["in_proj_bias"]
            )
            b_value, c_value, x_value = projection.chunk(3, dim=-1)
            gated = (b_value * x_value)[..., channel]
            lagged = torch.nn.functional.pad(
                gated,
                (fir_length - 1, 0),
            ).unfold(
                dimension=1,
                size=fir_length,
                step=1,
            )
            c_channel = c_value[..., channel]
            design = torch.cat(
                (
                    c_channel.unsqueeze(-1) * lagged,
                    c_channel.unsqueeze(-1),
                ),
                dim=-1,
            )
            design_rows.append(design[trajectory.valid_mask])
            target_rows.append(desired[..., channel][trajectory.valid_mask])
        design = torch.cat(design_rows, dim=0)
        target = torch.cat(target_rows, dim=0)
        prior = torch.cat((kernel[channel], bias[channel].reshape(1)))
        gram = design.T @ design
        scale = float(gram.diagonal().mean().item())
        ridge = max(scale * 1e-6, 1e-12)
        solved = torch.linalg.solve(
            gram + ridge * torch.eye(int(gram.shape[0]), dtype=gram.dtype),
            design.T @ target + ridge * prior,
        )
        kernel[channel] = solved[:-1]
        bias[channel] = solved[-1]
    result = {name: value.clone() for name, value in state.items()}
    result["fir_kernel"] = kernel.contiguous()
    result["fir_bias"] = bias.contiguous()
    return result


def _full_coverage_context_merge(
    *,
    writer_state: dict[str, torch.Tensor],
    fhvr_state: dict[str, torch.Tensor],
    fhvr_fraction: float,
    fit: tuple[Trajectory, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    aligned_fhvr, alignment = _gauge_align_fhvr_to_writer(
        writer_state,
        fhvr_state,
        fit,
    )
    desired_contexts = tuple(
        _candidate_context(writer_state, trajectory.child)
        + fhvr_fraction
        * (
            _candidate_context(aligned_fhvr, trajectory.child)
            - _candidate_context(writer_state, trajectory.child)
        )
        for trajectory in fit
    )
    state = _blend(writer_state, aligned_fhvr, fhvr_fraction)
    for _ in range(2):
        state = _solve_role_c(state, fit, desired_contexts)
        state = _solve_role_fir(state, fit, desired_contexts)
    context_rows = _stack_valid(
        fit,
        lambda trajectory: _candidate_context(state, trajectory.child),
    )
    target_rows = _stack_valid(fit, lambda trajectory: trajectory.parent_write)
    design = torch.cat(
        (
            context_rows,
            torch.ones(
                (int(context_rows.shape[0]), 1),
                dtype=context_rows.dtype,
            ),
        ),
        dim=-1,
    )
    prior = torch.cat(
        (
            state["writer_weight"].T,
            state["writer_bias"].unsqueeze(0),
        ),
        dim=0,
    )
    gram = design.T @ design
    scale = float(gram.diagonal().mean().item())
    ridge = max(scale * 1e-6, 1e-12)
    solved = torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=gram.dtype),
        design.T @ target_rows + ridge * prior,
    )
    state["writer_weight"] = solved[:-1].T.contiguous()
    state["writer_bias"] = solved[-1].contiguous()
    desired_rows = torch.cat(
        [
            desired[trajectory.valid_mask]
            for trajectory, desired in zip(fit, desired_contexts, strict=True)
        ],
        dim=0,
    )
    return state, {
        "fhvr_fraction": fhvr_fraction,
        "writer_fraction": 1.0 - fhvr_fraction,
        "source_channel_count": 2 * TARGET_HIDDEN,
        "target_channel_count": TARGET_HIDDEN,
        "writer_channels_covered": TARGET_HIDDEN,
        "fhvr_channels_covered": TARGET_HIDDEN,
        "source_channels_dropped": 0,
        "source_channels_pruned": 0,
        "one_to_one_pairing_after_authenticated_gauge_alignment": True,
        "complete_native_tuple_initial_merge": True,
        "fixed_closed_form_c_fir_sweeps": 2,
        "fit_context_refactor_rrmse": _relative_rmse(
            context_rows,
            desired_rows,
        ),
        "alignment": alignment,
        "fit_only": True,
    }


def _relative_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(
        torch.finfo(target.dtype).eps
    )
    return float((numerator / denominator).item())


def _cosine(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            prediction.reshape(1, -1),
            target.reshape(1, -1),
            dim=-1,
        ).item()
    )


def _stack_valid(
    values: tuple[Trajectory, ...],
    producer,
) -> torch.Tensor:
    rows = []
    for trajectory in values:
        produced = producer(trajectory)
        rows.append(produced[trajectory.valid_mask])
    return torch.cat(rows, dim=0)


def _operator_metrics(
    state: dict[str, torch.Tensor],
    values: tuple[Trajectory, ...],
) -> dict[str, float]:
    prediction = _stack_valid(
        values,
        lambda trajectory: _operator(state, trajectory.child),
    )
    target = _stack_valid(values, lambda trajectory: trajectory.parent_write)
    return {
        "rrmse": _relative_rmse(prediction, target),
        "cosine": _cosine(prediction, target),
    }


def _proxy_features(
    teacher: Teacher,
    post_operator: torch.Tensor,
) -> torch.Tensor:
    gate = post_operator @ teacher.proxy_gate
    up = post_operator @ teacher.proxy_up
    hidden = torch.nn.functional.silu(gate) * up
    ones = torch.ones(
        (*hidden.shape[:-1], 1),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return torch.cat((hidden, post_operator, ones), dim=-1)


def _fit_proxy_mlp(
    teacher: Teacher,
    state: dict[str, torch.Tensor],
    fit: tuple[Trajectory, ...],
) -> torch.Tensor:
    design = _stack_valid(
        fit,
        lambda trajectory: _proxy_features(
            teacher,
            trajectory.child + _operator(state, trajectory.child),
        ),
    )
    target = _stack_valid(fit, lambda trajectory: trajectory.parent_mlp_write)
    gram = design.T @ design
    scale = float(gram.diagonal().mean().item())
    ridge = max(scale * 1e-5, 1e-10)
    return torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=gram.dtype),
        design.T @ target,
    )


def _whole_macrocell_metrics(
    teacher: Teacher,
    state: dict[str, torch.Tensor],
    mlp_weight: torch.Tensor,
    values: tuple[Trajectory, ...],
) -> dict[str, float]:
    predicted_rows = []
    target_rows = []
    for trajectory in values:
        operator_write = _operator(state, trajectory.child)
        post_operator = trajectory.child + operator_write
        mlp_write = _proxy_features(teacher, post_operator) @ mlp_weight
        prediction = post_operator + mlp_write
        target = (
            trajectory.child
            + trajectory.parent_write
            + trajectory.parent_mlp_write
        )
        predicted_rows.append(prediction[trajectory.valid_mask])
        target_rows.append(target[trajectory.valid_mask])
    prediction = torch.cat(predicted_rows, dim=0)
    target = torch.cat(target_rows, dim=0)
    return {
        "rrmse": _relative_rmse(prediction, target),
        "cosine": _cosine(prediction, target),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    all_runs = []
    for seed in SEEDS:
        teacher = _teacher(seed)
        fit = _child_trajectories(
            teacher,
            seed=seed + 101,
            count=FIT_TRAJECTORIES,
            tokens=FIT_TOKENS,
        )
        selection = _child_trajectories(
            teacher,
            seed=seed + 202,
            count=SELECTION_TRAJECTORIES,
            tokens=SELECTION_TOKENS,
        )
        heldout = _child_trajectories(
            teacher,
            seed=seed + 303,
            count=HELDOUT_TRAJECTORIES,
            tokens=HELDOUT_TOKENS,
        )
        frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(
            _request(teacher, fit)
        )
        writer_candidates = [
            candidate
            for candidate in frontier.raw_candidates
            if candidate.candidate_kind == "accepted_fit_projection_path_anchor"
            and candidate.candidate_id.endswith("-writer")
        ]
        writer = min(
            writer_candidates,
            key=lambda candidate: (
                _operator_metrics(_candidate_state(candidate), selection)["rrmse"],
                candidate.candidate_id,
            ),
        )
        writer_state = _candidate_state(writer)
        fhvr_candidates = [
            candidate
            for candidate in frontier.raw_candidates
            if candidate.candidate_kind == FHVR_MF_CANDIDATE_KIND
        ]
        states = {
            "writer": writer_state,
            **{
                candidate.candidate_id: _candidate_state(candidate)
                for candidate in fhvr_candidates
            },
        }
        alignments = {}
        dictionary_selections = {}
        full_coverage_merges = {}
        for candidate in fhvr_candidates:
            fhvr_state = _candidate_state(candidate)
            aligned_state, alignment = _gauge_align_fhvr_to_writer(
                writer_state,
                fhvr_state,
                fit,
            )
            alignments[candidate.candidate_id] = alignment
            for fraction in BLEND_FRACTIONS:
                states[
                    f"{candidate.candidate_id}::gauge-state-fhvr-{fraction:g}"
                ] = _blend(writer_state, aligned_state, fraction)
                states[
                    f"{candidate.candidate_id}::response-fhvr-base-fhvr-{fraction:g}"
                ] = _response_refactor(
                    base_state=fhvr_state,
                    writer_state=writer_state,
                    fhvr_state=fhvr_state,
                    fhvr_fraction=fraction,
                    fit=fit,
                )
                states[
                    f"{candidate.candidate_id}::response-writer-base-fhvr-{fraction:g}"
                ] = _response_refactor(
                    base_state=writer_state,
                    writer_state=writer_state,
                    fhvr_state=fhvr_state,
                    fhvr_fraction=fraction,
                    fit=fit,
                )
                dictionary_state, dictionary_selection = (
                    _channel_dictionary_hybrid(
                        writer_state=writer_state,
                        fhvr_state=fhvr_state,
                        fhvr_fraction=fraction,
                        fit=fit,
                    )
                )
                states[
                    f"{candidate.candidate_id}::dictionary-fhvr-{fraction:g}"
                ] = dictionary_state
                dictionary_selections[
                    f"{candidate.candidate_id}::dictionary-fhvr-{fraction:g}"
                ] = dictionary_selection
                full_coverage_state, full_coverage_report = (
                    _full_coverage_context_merge(
                        writer_state=writer_state,
                        fhvr_state=fhvr_state,
                        fhvr_fraction=fraction,
                        fit=fit,
                    )
                )
                states[
                    f"{candidate.candidate_id}::full-coverage-fhvr-{fraction:g}"
                ] = full_coverage_state
                full_coverage_merges[
                    f"{candidate.candidate_id}::full-coverage-fhvr-{fraction:g}"
                ] = full_coverage_report
        metrics = {}
        for name, state in states.items():
            mlp_weight = _fit_proxy_mlp(teacher, state, fit)
            metrics[name] = {
                "selection_operator": _operator_metrics(state, selection),
                "heldout_operator": _operator_metrics(state, heldout),
                "selection_final": _whole_macrocell_metrics(
                    teacher,
                    state,
                    mlp_weight,
                    selection,
                ),
                "heldout_final": _whole_macrocell_metrics(
                    teacher,
                    state,
                    mlp_weight,
                    heldout,
                ),
            }
        all_runs.append(
            {
                "seed": seed,
                "writer_candidate_id": writer.candidate_id,
                "gauge_alignments": alignments,
                "dictionary_selections": dictionary_selections,
                "full_coverage_merges": full_coverage_merges,
                "metrics": metrics,
            }
        )

    arms = sorted(all_runs[0]["metrics"])
    aggregate = {}
    for arm in arms:
        aggregate[arm] = {}
        for metric_group in (
            "heldout_operator",
            "heldout_final",
        ):
            aggregate[arm][metric_group] = {
                metric: _mean(
                    [
                        run["metrics"][arm][metric_group][metric]
                        for run in all_runs
                    ]
                )
                for metric in ("rrmse", "cosine")
            }

    per_fhvr = {}
    for fhvr in [
        arm
        for arm in arms
        if arm != "writer" and "::" not in arm
    ]:
        baseline = aggregate[fhvr]["heldout_final"]["rrmse"]
        candidates = [
            arm for arm in arms if arm.startswith(f"{fhvr}::")
        ]
        best = min(
            candidates,
            key=lambda arm: aggregate[arm]["heldout_final"]["rrmse"],
        )
        per_fhvr[fhvr] = {
            "best_fixed_hybrid": best,
            "mean_heldout_final_rrmse_improvement_fraction": (
                baseline - aggregate[best]["heldout_final"]["rrmse"]
            )
            / baseline,
            "seed_win_count_vs_pure_fhvr": sum(
                run["metrics"][best]["heldout_final"]["rrmse"]
                < run["metrics"][fhvr]["heldout_final"]["rrmse"]
                for run in all_runs
            ),
            "seed_win_count_vs_writer": sum(
                run["metrics"][best]["heldout_final"]["rrmse"]
                < run["metrics"]["writer"]["heldout_final"]["rrmse"]
                for run in all_runs
            ),
        }
    result = {
        "schema": "fhvr-writer-fixed-blend-synthetic-v1",
        "fit_only_candidate_and_proxy_mlp_fitting": True,
        "selection_used_only_to_identify_writer_control": True,
        "heldout_used_for_reporting_only": True,
        "no_training_or_optimizer": True,
        "source_hidden": SOURCE_HIDDEN,
        "target_hidden": TARGET_HIDDEN,
        "parameter_width_compaction": SOURCE_HIDDEN / TARGET_HIDDEN,
        "source_fir_length": SOURCE_FIR_LENGTH,
        "target_fir_length": TARGET_FIR_LENGTH,
        "blend_fraction_is_fhvr_mass": True,
        "blend_fractions": list(BLEND_FRACTIONS),
        "seeds": list(SEEDS),
        "aggregate": aggregate,
        "per_fhvr": per_fhvr,
        "runs": all_runs,
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
