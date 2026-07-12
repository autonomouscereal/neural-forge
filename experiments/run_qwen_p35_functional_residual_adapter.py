#!/usr/bin/env python3
"""Fit one rank-8 p35 residual-attention adapter by output behavior only.

The structural child is fixed to the exact ``L19/G4 -> L21/G6`` redirect.
Only a sparse ``U @ (R @ z_group)`` correction is learned, where ``z_group``
is the real post-softmax group context immediately before the O projection and
residual add.  Fit gradients use fixed parent actions/logits on paired parent-
greedy and baseline-child-greedy prefixes.  Calibration alone locks the
in-memory winner.  Heldout model outputs are first touched after that lock.

This is a local, rollback-only experiment.  It cannot save or commit a child.
Weights and internal activations are never similarity targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_p35_behavior_vo_tuner as behavior_runner
import run_qwen_p35_vo_role_scale_search as role_search
import run_qwen_residual_attention_probe as base
from neural_forge.hf_functional_decode_trace import (
    FunctionalDecodeLimits,
    FunctionalDecodeTrace,
    collect_functional_decode_trace,
)
from neural_forge.hf_functional_residual_adapter import (
    P35_L19G4_TO_L21G6,
    FunctionalActionBatch,
    FunctionalActionEvaluation,
    InstalledFunctionalResidualAdapter,
    RedirectedAttentionSite,
    evaluate_functional_action_batches,
    install_functional_residual_adapter,
    train_functional_residual_adapter_disjoint,
)
from neural_forge.hf_qwen_attention_surgery import (
    Qwen3AttentionRuntimeSnapshot,
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report


RUN_SCHEMA = "neural-forge-qwen-p35-functional-residual-adapter-run-v1"
EXACT_REDIRECT = behavior_runner.EXACT_REDIRECT
EXACT_AFFECTED_SITES = behavior_runner.EXACT_AFFECTED_SITES
EXACT_PARENT_PROTOTYPE_COUNT = 280
EXACT_CHILD_PROTOTYPE_COUNT = 279
EXACT_RANK = 8
EXACT_STEPS = 32
EXACT_MICRO_BATCH_SIZE = 2
EXACT_CALIBRATION_STEPS = (0, 8, 16, 32)
EXACT_INCUMBENT_V_SCALE = -0.0012493133544921875
EXACT_INCUMBENT_O_SCALE = -0.0005393028259277344

DEFAULT_FIT_PROMPTS = behavior_runner.DEFAULT_FIT_PROMPTS
DEFAULT_CALIBRATION_PROMPTS = behavior_runner.DEFAULT_CALIBRATION_PROMPTS
DEFAULT_HELDOUT_PROMPTS = behavior_runner.DEFAULT_HELDOUT_PROMPTS
DEFAULT_FIT_MANIFEST = behavior_runner.DEFAULT_FIT_MANIFEST
DEFAULT_AUDITED_MANIFEST = behavior_runner.DEFAULT_AUDITED_MANIFEST
DEFAULT_TRANSITION_INVENTORY = behavior_runner.DEFAULT_TRANSITION_INVENTORY
EXPECTED_SPLIT_COUNTS = behavior_runner.EXPECTED_SPLIT_COUNTS

ActionSplit = Literal["fit", "calibration-selection", "heldout-evaluation"]
PrefixSource = Literal["parent-greedy", "student-greedy"]


class QwenP35FunctionalResidualAdapterError(RuntimeError):
    """A functional target, selection, accounting, or rollback proof failed."""


def _positive_int(value: Any, name: str) -> int:
    return role_search._positive_int(value, name)


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QwenP35FunctionalResidualAdapterError(
            f"{name} must be finite and positive."
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise QwenP35FunctionalResidualAdapterError(
            f"{name} must be finite and positive."
        )
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_state_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in sorted(module.state_dict().items())
    }


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    return base._sha256(
        {
            name: _tensor_sha256(value)
            for name, value in sorted(state.items())
        }
    )


def _load_exact_state(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    current = module.state_dict()
    if set(current) != set(state):
        raise QwenP35FunctionalResidualAdapterError(
            "Locked adapter state keys disagree with the reinstalled adapter."
        )
    with torch.no_grad():
        for name, destination in current.items():
            source = state[name]
            if tuple(destination.shape) != tuple(source.shape):
                raise QwenP35FunctionalResidualAdapterError(
                    f"Locked adapter tensor {name!r} changed shape."
                )
            destination.copy_(source.to(destination.device, destination.dtype))
    reloaded = _module_state_snapshot(module)
    if _state_sha256(reloaded) != _state_sha256(state):
        raise QwenP35FunctionalResidualAdapterError(
            "Reinstalled adapter does not exactly reproduce the locked state."
        )


def _redirected_site(
    registry: Any,
    redirect: base.AttentionRedirect,
) -> RedirectedAttentionSite:
    groups = int(registry.key_value_heads)
    return RedirectedAttentionSite(
        layer_index=redirect.source_layer,
        physical_group=redirect.source_group,
        removed_source_flat_index=redirect.source_layer * groups
        + redirect.source_group,
        survivor_layer_index=redirect.target_layer,
        survivor_physical_group=redirect.target_group,
        survivor_source_flat_index=redirect.target_layer * groups
        + redirect.target_group,
    )


def _incumbent_role_scale_audit(
    registry: Any,
    *,
    redirect: base.AttentionRedirect,
    require_exact_p35: bool,
) -> dict[str, Any]:
    canonical = role_search._canonical_assignment(
        registry,
        layer_index=redirect.source_layer,
        physical_group=redirect.source_group,
    )
    scales = registry.role_scales
    if scales is None:
        raise QwenP35FunctionalResidualAdapterError(
            "Tentative redirect exposes no per-assignment role scales."
        )
    row = tuple(
        float(value)
        for value in scales[
            redirect.source_layer, canonical
        ].detach().to(device="cpu", dtype=torch.float64).tolist()
    )
    if require_exact_p35 and row != (
        1.0,
        1.0,
        EXACT_INCUMBENT_V_SCALE,
        EXACT_INCUMBENT_O_SCALE,
    ):
        raise QwenP35FunctionalResidualAdapterError(
            "Exact run must retain the untouched p35 redirect Q/K/V/O scales; "
            f"got {row!r}."
        )
    return {
        "layer_index": redirect.source_layer,
        "physical_group": redirect.source_group,
        "canonical_assignment_axis": canonical,
        "q_scale": row[0],
        "k_scale": row[1],
        "v_scale": row[2],
        "o_scale": row[3],
        "exact_incumbent_v_scale": EXACT_INCUMBENT_V_SCALE,
        "exact_incumbent_o_scale": EXACT_INCUMBENT_O_SCALE,
        "fitted_scalar_warm_start_used": False,
        "role_scales_trainable": False,
    }


def _collect_greedy_split(
    runtime: nn.Module,
    prompts: Sequence[Tensor],
    *,
    split: str,
    horizon: int,
    limits: FunctionalDecodeLimits,
) -> tuple[tuple[FunctionalDecodeTrace, ...], dict[int, tuple[int, ...]]]:
    traces: list[FunctionalDecodeTrace] = []
    drivers: dict[int, tuple[int, ...]] = {}
    for prompt_index, prompt in enumerate(prompts):
        trace = collect_functional_decode_trace(
            runtime,
            prompt,
            horizon=horizon,
            sequence_id=f"{split}-prompt-{prompt_index}",
            lane="greedy",
            limits=limits,
        )
        traces.append(trace)
        drivers[prompt_index] = trace.selected_token_ids
    return tuple(traces), drivers


def _collect_forced_split(
    runtime: nn.Module,
    prompts: Sequence[Tensor],
    drivers: Mapping[int, tuple[int, ...]],
    *,
    split: str,
    horizon: int,
    limits: FunctionalDecodeLimits,
) -> tuple[FunctionalDecodeTrace, ...]:
    if set(drivers) != set(range(len(prompts))):
        raise QwenP35FunctionalResidualAdapterError(
            f"{split} student drivers do not match its prompt rows."
        )
    return tuple(
        collect_functional_decode_trace(
            runtime,
            prompt,
            horizon=horizon,
            sequence_id=f"{split}-prompt-{prompt_index}",
            lane="forced",
            forced_token_ids=drivers[prompt_index],
            limits=limits,
        )
        for prompt_index, prompt in enumerate(prompts)
    )


def _trace_manifest(traces: Sequence[FunctionalDecodeTrace]) -> dict[str, Any]:
    return {
        trace.sequence_id: trace.as_dict()
        for trace in traces
    }


def _trace_prefix_rows(trace: FunctionalDecodeTrace) -> list[tuple[list[int], int, Tensor]]:
    rows: list[tuple[list[int], int, Tensor]] = []
    for step in trace.steps:
        if step.step_index != len(rows):
            raise QwenP35FunctionalResidualAdapterError(
                "Functional trace step indices are not contiguous."
            )
        prefix = [
            *trace.prompt_token_ids,
            *trace.selected_token_ids[: step.step_index],
        ]
        if len(prefix) != step.prefix_length:
            raise QwenP35FunctionalResidualAdapterError(
                "Functional trace prefix provenance is inconsistent."
            )
        logits = step.action_logits.detach().cpu()
        if tuple(logits.shape) != (1, trace.vocabulary_size):
            raise QwenP35FunctionalResidualAdapterError(
                "Parent action logits have an unexpected shape."
            )
        rows.append((prefix, step.greedy_token_id, logits.squeeze(0)))
    return rows


def build_functional_action_batches(
    traces: Sequence[FunctionalDecodeTrace],
    *,
    split: ActionSplit,
    prefix_source: PrefixSource,
    micro_batch_size: int = EXACT_MICRO_BATCH_SIZE,
    padding_token_id: int = 0,
) -> tuple[FunctionalActionBatch, ...]:
    """Materialize fixed parent output targets in deterministic micro-batches."""

    batch_size = _positive_int(micro_batch_size, "micro_batch_size")
    if isinstance(padding_token_id, bool) or int(padding_token_id) < 0:
        raise QwenP35FunctionalResidualAdapterError(
            "padding_token_id must be a nonnegative integer."
        )
    expected_lane = "greedy" if prefix_source == "parent-greedy" else "forced"
    values = tuple(traces)
    if not values or any(trace.lane != expected_lane for trace in values):
        raise QwenP35FunctionalResidualAdapterError(
            f"{prefix_source} targets require only {expected_lane} parent traces."
        )
    result: list[FunctionalActionBatch] = []
    for trace in values:
        rows = _trace_prefix_rows(trace)
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset : offset + batch_size]
            width = max(len(prefix) for prefix, _action, _logits in chunk)
            input_ids = torch.full(
                (len(chunk), width), int(padding_token_id), dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            action_ids = torch.empty((len(chunk),), dtype=torch.long)
            logits: list[Tensor] = []
            for row_index, (prefix, action, action_logits) in enumerate(chunk):
                length = len(prefix)
                input_ids[row_index, :length] = torch.tensor(prefix, dtype=torch.long)
                attention_mask[row_index, :length] = 1
                action_ids[row_index] = int(action)
                logits.append(action_logits)
            result.append(
                FunctionalActionBatch(
                    sequence_id=(
                        f"{split}:{prefix_source}:{trace.sequence_id}:"
                        f"steps-{offset}-{offset + len(chunk) - 1}"
                    ),
                    split=split,
                    prefix_source=prefix_source,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    parent_action_ids=action_ids,
                    parent_action_logits=torch.stack(logits),
                )
            )
    return tuple(result)


def _paired_batches(
    parent: Sequence[FunctionalActionBatch],
    student: Sequence[FunctionalActionBatch],
    *,
    split: ActionSplit,
) -> tuple[FunctionalActionBatch, ...]:
    parent_values = tuple(parent)
    student_values = tuple(student)
    if not parent_values or len(parent_values) != len(student_values):
        raise QwenP35FunctionalResidualAdapterError(
            f"{split} parent/student prefix micro-batches are not paired."
        )
    result: list[FunctionalActionBatch] = []
    for parent_batch, student_batch in zip(
        parent_values, student_values, strict=True
    ):
        if (
            parent_batch.split != split
            or student_batch.split != split
            or parent_batch.prefix_source != "parent-greedy"
            or student_batch.prefix_source != "student-greedy"
            or parent_batch.batch_size != student_batch.batch_size
        ):
            raise QwenP35FunctionalResidualAdapterError(
                f"{split} parent/student prefix micro-batch alignment failed."
            )
        result.extend((parent_batch, student_batch))
    return tuple(result)


def _batch_manifest(batches: Sequence[FunctionalActionBatch]) -> dict[str, Any]:
    values = tuple(batches)
    return {
        "batch_count": len(values),
        "example_count": sum(value.batch_size for value in values),
        "batch_sizes": [value.batch_size for value in values],
        "prefix_sources_in_update_order": [
            value.prefix_source for value in values
        ],
        "batches": [value.as_dict() for value in values],
        "manifest_sha256": base._sha256([value.as_dict() for value in values]),
    }


def _greedy_rollout_comparison(
    parent: Sequence[FunctionalDecodeTrace],
    child: Sequence[FunctionalDecodeTrace],
) -> dict[str, Any]:
    parent_values = tuple(parent)
    child_values = tuple(child)
    if not parent_values or len(parent_values) != len(child_values):
        raise QwenP35FunctionalResidualAdapterError(
            "Parent and child greedy rollout sets do not align."
        )
    exact = 0
    token_matches = 0
    token_count = 0
    lcp_total = 0
    rows: list[dict[str, Any]] = []
    for parent_trace, child_trace in zip(
        parent_values, child_values, strict=True
    ):
        if (
            parent_trace.sequence_id != child_trace.sequence_id
            or parent_trace.prompt_token_ids != child_trace.prompt_token_ids
            or len(parent_trace.selected_token_ids)
            != len(child_trace.selected_token_ids)
        ):
            raise QwenP35FunctionalResidualAdapterError(
                "Parent and child greedy trace provenance does not align."
            )
        parent_tokens = parent_trace.selected_token_ids
        child_tokens = child_trace.selected_token_ids
        matches = sum(
            int(left == right)
            for left, right in zip(parent_tokens, child_tokens, strict=True)
        )
        lcp = 0
        for left, right in zip(parent_tokens, child_tokens, strict=True):
            if left != right:
                break
            lcp += 1
        is_exact = parent_tokens == child_tokens
        exact += int(is_exact)
        token_matches += matches
        token_count += len(parent_tokens)
        lcp_total += lcp
        rows.append(
            {
                "sequence_id": parent_trace.sequence_id,
                "exact_rollout_match": is_exact,
                "token_matches": matches,
                "token_count": len(parent_tokens),
                "longest_common_prefix": lcp,
            }
        )
    payload = {
        "prompt_count": len(rows),
        "exact_rollout_matches": exact,
        "exact_rollout_rate": exact / len(rows),
        "token_matches": token_matches,
        "token_count": token_count,
        "token_match_rate": token_matches / token_count,
        "mean_longest_common_prefix": lcp_total / len(rows),
        "rows": rows,
        "output_tokens_only": True,
    }
    payload["comparison_sha256"] = base._sha256(payload)
    return payload


def _evaluation_delta(
    baseline: FunctionalActionEvaluation,
    locked: FunctionalActionEvaluation,
    *,
    student_prefix_domains_identical: bool,
) -> dict[str, Any]:
    return {
        "parent_action_top1_rate": (
            locked.parent_action_top1_rate - baseline.parent_action_top1_rate
        ),
        "mean_parent_action_rank": (
            locked.mean_parent_action_rank - baseline.mean_parent_action_rank
        ),
        "mean_parent_action_log_probability": (
            locked.mean_parent_action_log_probability
            - baseline.mean_parent_action_log_probability
        ),
        "total_output_loss": locked.total_loss - baseline.total_loss,
        "discrete_top1_improved": (
            locked.parent_action_top1_rate > baseline.parent_action_top1_rate
        ),
        "discrete_rank_improved": (
            locked.mean_parent_action_rank < baseline.mean_parent_action_rank
        ),
        "comparison_domain": (
            "common-parent-greedy-plus-each-child-own-student-greedy"
        ),
        "student_prefix_domains_identical": student_prefix_domains_identical,
        "common_prefix_causal_delta_claimed": student_prefix_domains_identical,
        "own_rollout_domain_comparison_when_student_prefixes_differ": True,
        "used_for_selection": False,
    }


def _event(events: list[dict[str, Any]], name: str, **payload: Any) -> None:
    events.append({"index": len(events), "event": name, **payload})


def _restore_parent_exact(
    runtime: nn.Module,
    installation: Any,
    snapshot: Qwen3AttentionRuntimeSnapshot,
    identity_sha256: str,
) -> bool:
    return bool(
        attention_surgery._restore_installed(runtime, installation)
        and qwen3_attention_runtime_snapshot(runtime) == snapshot
        and role_search._runtime_state_identity_sha256(runtime) == identity_sha256
    )


def run_qwen_p35_functional_residual_adapter(
    runtime: nn.Module,
    fit_prompt_input_ids: Sequence[Tensor],
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    *,
    split_provenance: Mapping[str, Any],
    redirect: base.AttentionRedirect = EXACT_REDIRECT,
    require_exact_p35: bool = True,
    rank: int = EXACT_RANK,
    horizon: int = 4,
    steps: int = EXACT_STEPS,
    micro_batch_size: int = EXACT_MICRO_BATCH_SIZE,
    learning_rate: float = 1.5e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    hard_action_weight: float = 1.0,
    distribution_weight: float = 0.1,
    temperature: float = 1.0,
    max_gradient_norm: float = 1.0,
    calibration_steps: Sequence[int] = EXACT_CALIBRATION_STEPS,
    initialization_seed: int = 0,
    trace_limits: FunctionalDecodeLimits | None = None,
) -> dict[str, Any]:
    """Run one exact output-trained adapter transaction and restore p35."""

    rank_value = _positive_int(rank, "rank")
    horizon_value = _positive_int(horizon, "horizon")
    step_count = _positive_int(steps, "steps")
    batch_size = _positive_int(micro_batch_size, "micro_batch_size")
    learning_rate_value = _finite_positive(learning_rate, "learning_rate")
    max_gradient_norm_value = _finite_positive(
        max_gradient_norm, "max_gradient_norm"
    )
    selection_steps = tuple(int(value) for value in calibration_steps)
    if require_exact_p35:
        if redirect != EXACT_REDIRECT:
            raise QwenP35FunctionalResidualAdapterError(
                "Exact p35 run is fixed to L19/G4 -> L21/G6."
            )
        if (
            rank_value != EXACT_RANK
            or step_count != EXACT_STEPS
            or batch_size != EXACT_MICRO_BATCH_SIZE
            or selection_steps != EXACT_CALIBRATION_STEPS
            or not 1e-3 <= learning_rate_value <= 2e-3
            or max_gradient_norm_value != 1.0
        ):
            raise QwenP35FunctionalResidualAdapterError(
                "The first exact p35 run is fixed to rank 8, 32 batch-2 updates, "
                "calibration at 0/8/16/32, learning rate 1e-3..2e-3, and clip 1."
            )
    if not isinstance(split_provenance, Mapping) or (
        require_exact_p35
        and split_provenance.get("verified_exact_fixture_provenance") is not True
    ):
        raise QwenP35FunctionalResidualAdapterError(
            "Exact p35 mode requires verified three-split fixture provenance."
        )
    limits = trace_limits or FunctionalDecodeLimits(max_horizon=horizon_value)
    if horizon_value > limits.max_horizon:
        raise QwenP35FunctionalResidualAdapterError(
            "horizon exceeds the functional decoder bound."
        )
    fit_prompts, calibration_prompts, heldout_prompts, token_audit = (
        behavior_runner._validate_token_splits(
            fit_prompt_input_ids,
            calibration_prompt_input_ids,
            heldout_prompt_input_ids,
            max_prompts_per_split=max(EXPECTED_SPLIT_COUNTS.values()),
            require_exact_p35=require_exact_p35,
        )
    )

    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_identity = role_search._runtime_state_identity_sha256(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    site = _redirected_site(parent_registry, redirect)
    if require_exact_p35 and site != P35_L19G4_TO_L21G6:
        raise QwenP35FunctionalResidualAdapterError(
            "Exact redirect provenance is not L19/G4 source 156 to L21/G6 source 174."
        )
    _bridge, _proposal, proposal_candidates, tentative = base._tentative_codebook(
        runtime, (redirect,)
    )
    codebook, used_projections = tentative
    proposed_registry = codebook.registry
    affected_sites, sites_by_source = base._affected_sites(
        parent_registry, proposed_registry, (redirect,)
    )
    role_scale_audit = _incumbent_role_scale_audit(
        proposed_registry,
        redirect=redirect,
        require_exact_p35=require_exact_p35,
    )
    proposed_registry_state_before = {
        name: _tensor_sha256(value)
        for name, value in sorted(proposed_registry.state_dict().items())
    }
    events: list[dict[str, Any]] = []
    _event(events, "fixtures-validated", heldout_model_forward=False)

    adapter_handle: InstalledFunctionalResidualAdapter | None = None
    second_handle: InstalledFunctionalResidualAdapter | None = None
    proposed_snapshot: Qwen3AttentionRuntimeSnapshot | None = None
    lineage: dict[str, Any] | None = None
    training_report: Any = None
    installation_report: dict[str, Any] | None = None
    locked_state: dict[str, Tensor] | None = None
    locked_state_sha256: str | None = None
    heldout_baseline: FunctionalActionEvaluation | None = None
    heldout_locked: FunctionalActionEvaluation | None = None
    parent_heldout_greedy: tuple[FunctionalDecodeTrace, ...] = ()
    baseline_heldout_greedy: tuple[FunctionalDecodeTrace, ...] = ()
    locked_heldout_greedy: tuple[FunctionalDecodeTrace, ...] = ()
    parent_rollback_exact = False

    try:
        _event(events, "parent-fit-greedy-targets", heldout_model_forward=False)
        parent_fit_greedy, _parent_fit_drivers = _collect_greedy_split(
            runtime,
            fit_prompts,
            split="fit",
            horizon=horizon_value,
            limits=limits,
        )
        _event(
            events,
            "parent-calibration-greedy-targets",
            heldout_model_forward=False,
        )
        parent_calibration_greedy, _parent_calibration_drivers = (
            _collect_greedy_split(
                runtime,
                calibration_prompts,
                split="calibration-selection",
                horizon=horizon_value,
                limits=limits,
            )
        )
        if (
            qwen3_attention_runtime_snapshot(runtime) != parent_snapshot
            or role_search._runtime_state_identity_sha256(runtime) != parent_identity
        ):
            raise QwenP35FunctionalResidualAdapterError(
                "Parent fit/calibration output collection changed p35."
            )

        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        proposed_snapshot = qwen3_attention_runtime_snapshot(runtime)
        lineage = role_search._audit_lineage(
            parent_snapshot,
            proposed_snapshot,
            redirect=redirect,
            affected_sites=affected_sites,
            require_exact_p35=require_exact_p35,
        )
        _event(events, "tentative-child-installed", heldout_model_forward=False)
        _event(events, "child-fit-greedy-drivers", heldout_model_forward=False)
        child_fit_greedy, child_fit_drivers = _collect_greedy_split(
            runtime,
            fit_prompts,
            split="fit",
            horizon=horizon_value,
            limits=limits,
        )
        _event(
            events,
            "child-calibration-greedy-drivers",
            heldout_model_forward=False,
        )
        child_calibration_greedy, child_calibration_drivers = (
            _collect_greedy_split(
                runtime,
                calibration_prompts,
                split="calibration-selection",
                horizon=horizon_value,
                limits=limits,
            )
        )
        if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
            raise QwenP35FunctionalResidualAdapterError(
                "Baseline child driver collection changed the tentative child."
            )
        if not _restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        ):
            raise QwenP35FunctionalResidualAdapterError(
                "Could not restore p35 before collecting student-prefix targets."
            )
        _event(
            events,
            "parent-restored-before-student-prefix-targets",
            heldout_model_forward=False,
        )

        _event(
            events,
            "parent-fit-student-prefix-targets",
            heldout_model_forward=False,
        )
        parent_fit_on_student = _collect_forced_split(
            runtime,
            fit_prompts,
            child_fit_drivers,
            split="fit",
            horizon=horizon_value,
            limits=limits,
        )
        _event(
            events,
            "parent-calibration-student-prefix-targets",
            heldout_model_forward=False,
        )
        parent_calibration_on_student = _collect_forced_split(
            runtime,
            calibration_prompts,
            child_calibration_drivers,
            split="calibration-selection",
            horizon=horizon_value,
            limits=limits,
        )

        fit_parent_batches = build_functional_action_batches(
            parent_fit_greedy,
            split="fit",
            prefix_source="parent-greedy",
            micro_batch_size=batch_size,
        )
        fit_student_batches = build_functional_action_batches(
            parent_fit_on_student,
            split="fit",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        fit_batches = _paired_batches(
            fit_parent_batches, fit_student_batches, split="fit"
        )
        calibration_parent_batches = build_functional_action_batches(
            parent_calibration_greedy,
            split="calibration-selection",
            prefix_source="parent-greedy",
            micro_batch_size=batch_size,
        )
        calibration_student_batches = build_functional_action_batches(
            parent_calibration_on_student,
            split="calibration-selection",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        calibration_batches = _paired_batches(
            calibration_parent_batches,
            calibration_student_batches,
            split="calibration-selection",
        )
        if require_exact_p35 and (
            len(fit_batches) != EXACT_STEPS
            or any(value.batch_size != EXACT_MICRO_BATCH_SIZE for value in fit_batches)
        ):
            raise QwenP35FunctionalResidualAdapterError(
                "Exact 8-prompt horizon-4 fit must yield 32 batch-2 updates."
            )

        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
            raise QwenP35FunctionalResidualAdapterError(
                "Reinstalled tentative child differs before adapter training."
            )
        _event(
            events,
            "tentative-child-installed-for-training",
            heldout_model_forward=False,
        )
        adapter_handle = install_functional_residual_adapter(
            runtime,
            proposed_registry,
            site=site,
            rank=rank_value,
            initialization_seed=initialization_seed,
            require_qwen3_4b_dimensions=require_exact_p35,
        )
        installation_report = adapter_handle.report.as_dict()
        if not adapter_handle.adapter_registry.starts_as_exact_zero_correction:
            raise QwenP35FunctionalResidualAdapterError(
                "Adapter U must start at exact zero."
            )
        if not bool(torch.count_nonzero(adapter_handle.adapter_registry.right).item()):
            raise QwenP35FunctionalResidualAdapterError(
                "Adapter R must use a deterministic nonzero random initialization."
            )
        _event(
            events,
            "adapter-installed-zero-functional-correction",
            heldout_model_forward=False,
        )
        _event(events, "training-start", heldout_model_forward=False)
        training_report = train_functional_residual_adapter_disjoint(
            adapter_handle,
            fit_batches,
            calibration_batches,
            steps=step_count,
            learning_rate=learning_rate_value,
            beta1=beta1,
            beta2=beta2,
            adam_epsilon=adam_epsilon,
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            max_gradient_norm=max_gradient_norm_value,
            calibration_steps=selection_steps,
        )
        adapter_handle.assert_live()
        locked_state = _module_state_snapshot(adapter_handle.adapter_registry)
        locked_state_sha256 = _state_sha256(locked_state)
        if locked_state_sha256 != training_report.adapter_state_sha256:
            raise QwenP35FunctionalResidualAdapterError(
                "Runner lock hash disagrees with the trainer-selected adapter state."
            )
        _event(
            events,
            "winner-locked",
            heldout_model_forward=False,
            best_step=training_report.best_step,
            adapter_state_sha256=locked_state_sha256,
        )

        # The first heldout model forward is deliberately below winner-lock.
        _event(
            events,
            "heldout-locked-child-greedy-drivers",
            heldout_model_forward=True,
        )
        locked_heldout_greedy, locked_heldout_drivers = _collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        adapter_handle.rollback()
        adapter_handle = None
        if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
            raise QwenP35FunctionalResidualAdapterError(
                "Adapter rollback did not recover the tentative child."
            )
        _event(
            events,
            "heldout-baseline-child-greedy-drivers",
            heldout_model_forward=True,
        )
        baseline_heldout_greedy, baseline_heldout_drivers = _collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        if not _restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        ):
            raise QwenP35FunctionalResidualAdapterError(
                "Could not restore p35 for post-lock heldout targets."
            )
        _event(
            events,
            "heldout-parent-greedy-targets",
            heldout_model_forward=True,
        )
        parent_heldout_greedy, _parent_heldout_drivers = _collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        _event(
            events,
            "heldout-parent-student-prefix-targets",
            heldout_model_forward=True,
        )
        parent_heldout_on_locked_student = _collect_forced_split(
            runtime,
            heldout_prompts,
            locked_heldout_drivers,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        parent_heldout_on_baseline_student = _collect_forced_split(
            runtime,
            heldout_prompts,
            baseline_heldout_drivers,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        heldout_parent_batches = build_functional_action_batches(
            parent_heldout_greedy,
            split="heldout-evaluation",
            prefix_source="parent-greedy",
            micro_batch_size=batch_size,
        )
        heldout_locked_student_batches = build_functional_action_batches(
            parent_heldout_on_locked_student,
            split="heldout-evaluation",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        heldout_baseline_student_batches = build_functional_action_batches(
            parent_heldout_on_baseline_student,
            split="heldout-evaluation",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        heldout_locked_batches = _paired_batches(
            heldout_parent_batches,
            heldout_locked_student_batches,
            split="heldout-evaluation",
        )
        heldout_baseline_batches = _paired_batches(
            heldout_parent_batches,
            heldout_baseline_student_batches,
            split="heldout-evaluation",
        )

        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        _event(
            events,
            "heldout-baseline-action-evaluation",
            heldout_model_forward=True,
        )
        heldout_baseline = evaluate_functional_action_batches(
            runtime,
            heldout_baseline_batches,
            expected_split="heldout-evaluation",
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=False,
        )
        second_handle = install_functional_residual_adapter(
            runtime,
            proposed_registry,
            site=site,
            rank=rank_value,
            initialization_seed=initialization_seed,
            require_qwen3_4b_dimensions=require_exact_p35,
        )
        _load_exact_state(second_handle.adapter_registry, locked_state)
        _event(
            events,
            "heldout-locked-action-evaluation",
            heldout_model_forward=True,
        )
        heldout_locked = evaluate_functional_action_batches(
            runtime,
            heldout_locked_batches,
            expected_split="heldout-evaluation",
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=False,
        )
        second_handle.rollback()
        second_handle = None
        parent_rollback_exact = _restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        )
        if not parent_rollback_exact:
            raise QwenP35FunctionalResidualAdapterError(
                "Final transaction did not restore incumbent p35 exactly."
            )
        _event(events, "incumbent-p35-restored", heldout_model_forward=False)
    finally:
        failure_in_flight = sys.exc_info()[0] is not None
        cleanup_errors: list[Exception] = []
        for handle in (second_handle, adapter_handle):
            if handle is not None and handle.is_applied:
                try:
                    handle.rollback()
                except Exception as error:  # pragma: no cover - defensive cleanup
                    cleanup_errors.append(error)
        try:
            parent_rollback_exact = _restore_parent_exact(
                runtime, parent_installation, parent_snapshot, parent_identity
            )
            if not parent_rollback_exact:
                cleanup_errors.append(
                    QwenP35FunctionalResidualAdapterError(
                        "Cleanup did not restore incumbent p35 exactly."
                    )
                )
        except Exception as error:  # pragma: no cover - defensive cleanup
            cleanup_errors.append(error)
        if cleanup_errors and not failure_in_flight:
            raise QwenP35FunctionalResidualAdapterError(
                "Functional-adapter cleanup failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            ) from cleanup_errors[0]

    if any(
        value is None
        for value in (
            proposed_snapshot,
            lineage,
            training_report,
            installation_report,
            locked_state,
            locked_state_sha256,
            heldout_baseline,
            heldout_locked,
        )
    ) or not parent_rollback_exact:
        raise QwenP35FunctionalResidualAdapterError(
            "Functional-adapter run completed without a full transaction proof."
        )
    assert proposed_snapshot is not None
    assert lineage is not None
    assert training_report is not None
    assert installation_report is not None
    assert locked_state_sha256 is not None
    assert heldout_baseline is not None
    assert heldout_locked is not None

    proposed_registry_state_after = {
        name: _tensor_sha256(value)
        for name, value in sorted(proposed_registry.state_dict().items())
    }
    if proposed_registry_state_after != proposed_registry_state_before:
        raise QwenP35FunctionalResidualAdapterError(
            "Adapter training changed the tentative child base registry."
        )
    first_heldout = next(
        value for value in events if value["heldout_model_forward"]
    )
    winner_lock = next(value for value in events if value["event"] == "winner-locked")
    if first_heldout["index"] <= winner_lock["index"]:
        raise QwenP35FunctionalResidualAdapterError(
            "Heldout model output was accessed before the calibration winner lock."
        )

    accounting = installation_report["accounting"]
    if require_exact_p35 and (
        accounting["adapter_parameters"] != 24_576
        or accounting["removed_site_parameters"] != 3_276_800
        or accounting["redirect_role_scale_parameters"] != 1_152
        or accounting["net_parameters_saved_vs_incumbent"] != 3_251_072
        or accounting["learned_payload_bytes_saved"] != 6_502_144
    ):
        raise QwenP35FunctionalResidualAdapterError(
            "Exact rank-8 adapter accounting differs from the closed form."
        )

    parent_bindings = base._materialized_site_bindings(parent_registry)
    proposed_bindings = base._materialized_site_bindings(proposed_registry)
    affected_payload = [
        {
            "layer_index": layer,
            "physical_group": group,
            "canonical_assignment_axis": proposed_bindings[(layer, group)].assignment_axis,
            "parent_source_flat_index": parent_bindings[(layer, group)].source_flat_index,
            "proposed_source_flat_index": proposed_bindings[(layer, group)].source_flat_index,
            "proposed_active_slot": proposed_bindings[(layer, group)].active_slot,
        }
        for layer, group in affected_sites
    ]
    baseline_rollout = _greedy_rollout_comparison(
        parent_heldout_greedy, baseline_heldout_greedy
    )
    locked_rollout = _greedy_rollout_comparison(
        parent_heldout_greedy, locked_heldout_greedy
    )
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "redirect": redirect.as_dict(),
        "exact_p35_mode": require_exact_p35,
        "rank": rank_value,
        "horizon": horizon_value,
        "prompt_fixture_provenance": dict(split_provenance),
        "prompt_split_audit": token_audit,
        "parent_attention_snapshot": parent_snapshot.as_dict(),
        "proposed_attention_snapshot": proposed_snapshot.as_dict(),
        "lineage_audit": lineage,
        "incumbent_role_scale_audit": role_scale_audit,
        "affected_assignments": affected_payload,
        "affected_assignment_count": len(affected_payload),
        "affected_assignment_sha256": base._sha256(affected_payload),
        "survivor_assignment_sets": {
            str(source): [list(site_value) for site_value in sites]
            for source, sites in sites_by_source.items()
        },
        "used_redirect_projections": [
            value.as_dict() for value in used_projections
        ],
        "proposal_candidate_count": len(proposal_candidates),
        "fit_action_batches": _batch_manifest(fit_batches),
        "calibration_action_batches": _batch_manifest(calibration_batches),
        "fixed_parent_trace_provenance": {
            "fit_parent_greedy": _trace_manifest(parent_fit_greedy),
            "fit_parent_on_student_greedy": _trace_manifest(parent_fit_on_student),
            "calibration_parent_greedy": _trace_manifest(
                parent_calibration_greedy
            ),
            "calibration_parent_on_student_greedy": _trace_manifest(
                parent_calibration_on_student
            ),
            "baseline_child_fit_drivers": _trace_manifest(child_fit_greedy),
            "baseline_child_calibration_drivers": _trace_manifest(
                child_calibration_greedy
            ),
        },
        "adapter_installation": installation_report,
        "training": training_report.as_dict(),
        "calibration_winner_lock": {
            "best_step": training_report.best_step,
            "adapter_state_sha256": locked_state_sha256,
            "selected_by": "calibration-discrete-parent-action-ordering-only",
            "continuous_log_probability_or_loss_can_promote": False,
            "checkpoint_location": None,
            "in_memory_only": True,
        },
        "heldout_evaluation": {
            "baseline_action_function": heldout_baseline.as_dict(),
            "locked_adapter_action_function": heldout_locked.as_dict(),
            "locked_minus_baseline": _evaluation_delta(
                heldout_baseline,
                heldout_locked,
                student_prefix_domains_identical=(
                    baseline_heldout_drivers == locked_heldout_drivers
                ),
            ),
            "baseline_greedy_rollout": baseline_rollout,
            "locked_adapter_greedy_rollout": locked_rollout,
            "winner_named": False,
            "used_for_fitting": False,
            "used_for_selection": False,
            "first_model_forward_after_winner_lock": True,
        },
        "heldout_access_audit": {
            "events": events,
            "winner_lock_event_index": winner_lock["index"],
            "first_heldout_model_forward_event_index": first_heldout["index"],
            "heldout_model_forward_only_after_lock": True,
        },
        "objective_contract": {
            "parent_top1_actions_and_output_logits_only": True,
            "paired_parent_and_student_greedy_prefixes": True,
            "post_softmax_context_used_by_child_operator": True,
            "weight_values_required_to_match_numerically": False,
            "weight_similarity_used": False,
            "qkv_similarity_used": False,
            "attention_similarity_used": False,
            "activation_similarity_used": False,
            "residual_similarity_used": False,
            "cache_value_similarity_used": False,
        },
        "accounting": {
            "parent_prototype_count": parent_snapshot.prototype_count,
            "tentative_child_prototype_count": proposed_snapshot.prototype_count,
            "prototype_count_removed": 1,
            "exact_280_to_279": bool(
                parent_snapshot.prototype_count == EXACT_PARENT_PROTOTYPE_COUNT
                and proposed_snapshot.prototype_count == EXACT_CHILD_PROTOTYPE_COUNT
            ),
            "parent_attention_tensor_bytes": parent_snapshot.tensor_bytes,
            "tentative_child_attention_tensor_bytes": proposed_snapshot.tensor_bytes,
            "attention_tensor_bytes_saved_before_adapter": (
                parent_snapshot.tensor_bytes - proposed_snapshot.tensor_bytes
            ),
            "adapter": accounting,
            "one_redirect_only": True,
            "full_p35_to_p34_requires_redirect_count": 8,
            "optimizer_state_saved_or_counted": False,
            "structural_compaction_measured": True,
            "functional_compaction_certified": False,
        },
        "transaction": {
            "tentative_child_base_registry_unchanged": True,
            "adapter_parameters_only_trained": True,
            "locked_adapter_reinstalled_exactly_for_heldout": True,
            "incumbent_p35_runtime_restored_exactly": parent_rollback_exact,
            "candidate_installed_after_return": False,
        },
        "checkpoint_saved": False,
        "fold_committed": False,
        "compaction_certified": False,
        "training_used": True,
        "training_scope": "rank-8-one-site-functional-residual-adapter-only",
    }
    report["report_sha256"] = base._sha256(report)
    return report


def load_verified_prompt_splits(
    fit_path: Path,
    calibration_path: Path,
    heldout_path: Path,
    *,
    fit_manifest_path: Path = DEFAULT_FIT_MANIFEST,
    audited_manifest_path: Path = DEFAULT_AUDITED_MANIFEST,
    transition_inventory_path: Path = DEFAULT_TRANSITION_INVENTORY,
) -> behavior_runner.VerifiedPromptSplits:
    return behavior_runner.load_verified_prompt_splits(
        fit_path,
        calibration_path,
        heldout_path,
        fit_manifest_path=fit_manifest_path,
        audited_manifest_path=audited_manifest_path,
        transition_inventory_path=transition_inventory_path,
    )


def _encode_prompts(
    tokenizer: Any,
    values: Sequence[str],
    *,
    max_prompt_tokens: int,
) -> tuple[Tensor, ...]:
    return behavior_runner._encode_prompts(
        tokenizer, values, max_prompt_tokens=max_prompt_tokens
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--fit-prompts-json", type=Path, default=DEFAULT_FIT_PROMPTS
    )
    parser.add_argument(
        "--calibration-selection-prompts-json",
        type=Path,
        default=DEFAULT_CALIBRATION_PROMPTS,
    )
    parser.add_argument(
        "--heldout-evaluation-prompts-json",
        type=Path,
        default=DEFAULT_HELDOUT_PROMPTS,
    )
    parser.add_argument(
        "--fit-prompt-manifest", type=Path, default=DEFAULT_FIT_MANIFEST
    )
    parser.add_argument(
        "--audited-prompt-manifest", type=Path, default=DEFAULT_AUDITED_MANIFEST
    )
    parser.add_argument(
        "--transition-inventory", type=Path, default=DEFAULT_TRANSITION_INVENTORY
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--rank", type=int, default=EXACT_RANK)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--steps", type=int, default=EXACT_STEPS)
    parser.add_argument("--micro-batch-size", type=int, default=EXACT_MICRO_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--hard-action-weight", type=float, default=1.0)
    parser.add_argument("--distribution-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-vocabulary", type=int, default=1_000_000)
    parser.add_argument("--max-logit-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(model=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixtures = load_verified_prompt_splits(
        args.fit_prompts_json,
        args.calibration_selection_prompts_json,
        args.heldout_evaluation_prompts_json,
        fit_manifest_path=args.fit_prompt_manifest,
        audited_manifest_path=args.audited_prompt_manifest,
        transition_inventory_path=args.transition_inventory,
    )
    runtime, tokenizer, source = base._load_runtime_and_tokenizer(args)
    fit_prompts = _encode_prompts(
        tokenizer, fixtures.fit, max_prompt_tokens=args.max_prompt_tokens
    )
    calibration_prompts = _encode_prompts(
        tokenizer, fixtures.calibration, max_prompt_tokens=args.max_prompt_tokens
    )
    heldout_prompts = _encode_prompts(
        tokenizer, fixtures.heldout, max_prompt_tokens=args.max_prompt_tokens
    )
    limits = FunctionalDecodeLimits(
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_vocabulary=args.max_vocabulary,
        max_logit_bytes=args.max_logit_bytes,
        max_model_devices=1,
    )
    report = run_qwen_p35_functional_residual_adapter(
        runtime,
        fit_prompts,
        calibration_prompts,
        heldout_prompts,
        split_provenance=fixtures.provenance,
        rank=args.rank,
        horizon=args.horizon,
        steps=args.steps,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_epsilon=args.adam_epsilon,
        hard_action_weight=args.hard_action_weight,
        distribution_weight=args.distribution_weight,
        temperature=args.temperature,
        max_gradient_norm=args.max_gradient_norm,
        calibration_steps=EXACT_CALIBRATION_STEPS,
        initialization_seed=args.initialization_seed,
        trace_limits=limits,
    )
    report["source"] = source
    report["parameter_report"] = qwen3_codebook_parameter_report(runtime).as_dict()
    report["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "dtype": args.dtype,
        "collector": "functional-output-only-cached-decode",
        "update_rule": "manual-adam-adapter-parameters-only",
    }
    report["report_sha256"] = base._sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    base._atomic_write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "best_step": report["calibration_winner_lock"]["best_step"],
                "heldout_top1_delta": report["heldout_evaluation"][
                    "locked_minus_baseline"
                ]["parent_action_top1_rate"],
                "runtime_restored": report["transaction"][
                    "incumbent_p35_runtime_restored_exactly"
                ],
                "checkpoint_saved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
