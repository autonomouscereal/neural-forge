#!/usr/bin/env python3
"""Train the exact p35 assignment-local rank-8 Q/K/V/O residual fallback.

The structural child remains the single ``L19/G4 -> L21/G6`` 280-to-279
redirect.  Sparse Q/K/V corrections execute before normalization, RoPE, and
cache append; sparse O consumes the real post-softmax physical-group context.
Only the eight low-rank factor tensors train, using fixed parent actions and
output logits on disjoint fit prefixes. Calibration locks by discrete output
behavior, heldout model forwards begin only after lock, and p35 is restored.

This runner writes JSON only. It cannot save or commit a child checkpoint.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_p35_functional_residual_adapter as o_runner
import run_qwen_p35_vo_role_scale_search as role_search
import run_qwen_residual_attention_probe as base
from neural_forge.hf_assignment_local_qkvo_residual_adapter import (
    InstalledAssignmentLocalQKVOResidualAdapter,
    install_assignment_local_qkvo_residual_adapter,
    train_assignment_local_qkvo_residual_disjoint,
)
from neural_forge.hf_functional_decode_trace import FunctionalDecodeLimits
from neural_forge.hf_functional_residual_adapter import (
    FunctionalActionEvaluation,
    _action_logits,
    evaluate_functional_action_batches,
)
from neural_forge.hf_qwen_attention_surgery import (
    Qwen3AttentionRuntimeSnapshot,
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report


RUN_SCHEMA = "neural-forge-qwen-p35-assignment-local-qkvo-residual-run-v1"
EXACT_REDIRECT = o_runner.EXACT_REDIRECT
EXACT_PARENT_PROTOTYPE_COUNT = 280
EXACT_CHILD_PROTOTYPE_COUNT = 279
EXACT_RANK = 8
EXACT_HORIZON = 4
EXACT_STEPS = 32
EXACT_MICRO_BATCH_SIZE = 2
EXACT_CALIBRATION_STEPS = (0, 8, 16, 32)
EXACT_SELECTED_REPLAY_STEPS = 16
EXACT_SELECTED_REPLAY_CALIBRATION_STEPS = (0, 8, 16)
EXACT_Q_LEARNING_RATE = 1e-3
EXACT_K_LEARNING_RATE = 1e-3
EXACT_V_LEARNING_RATE = 2e-3
EXACT_O_LEARNING_RATE = 2e-3

DEFAULT_FIT_PROMPTS = o_runner.DEFAULT_FIT_PROMPTS
DEFAULT_CALIBRATION_PROMPTS = o_runner.DEFAULT_CALIBRATION_PROMPTS
DEFAULT_HELDOUT_PROMPTS = o_runner.DEFAULT_HELDOUT_PROMPTS
DEFAULT_FIT_MANIFEST = o_runner.DEFAULT_FIT_MANIFEST
DEFAULT_AUDITED_MANIFEST = o_runner.DEFAULT_AUDITED_MANIFEST
DEFAULT_TRANSITION_INVENTORY = o_runner.DEFAULT_TRANSITION_INVENTORY
EXPECTED_SPLIT_COUNTS = o_runner.EXPECTED_SPLIT_COUNTS

O_ONLY_NEGATIVE_PROVENANCE = {
    "report_file": (
        "p35_L19G4_to_L21G6_functional_residual_adapter_rank8_h4_s32_fp16_p40_gpu3.json"
    ),
    "file_sha256": ("531942412df7a1b9900014ad5481a03aba0e331609e6d5bb27ae447dbbeff243"),
    "report_sha256": (
        "d84670b556b2df03e289085d0e16c3670df60bfaa010f4317d20b0aa0d17cac4"
    ),
    "best_step": 0,
    "calibration_parent_action_top1_rate": 0.7916666666666666,
    "calibration_mean_parent_action_rank": 1.3541666666666667,
    "heldout_baseline_parent_action_top1_rate": 0.875,
    "heldout_locked_parent_action_top1_rate": 0.875,
    "heldout_parent_action_top1_delta": 0.0,
    "heldout_baseline_mean_parent_action_rank": 1.140625,
    "heldout_locked_mean_parent_action_rank": 1.140625,
    "heldout_exact_rollout_matches": "5/8 baseline and locked",
    "heldout_token_matches": "20/32 baseline and locked",
    "conclusion": "rank-8 O-only residual produced no discrete heldout gain",
    "used_as_training_input": False,
    "used_as_selection_input": False,
}


class QwenP35AssignmentLocalQKVOResidualError(RuntimeError):
    """The exact QKVO behavior, lineage, or rollback contract failed."""


def _action_output_manifest(
    runtime: nn.Module,
    batches: Sequence[Any],
) -> dict[str, Any]:
    """Hash exact candidate action logits without retaining/saving them."""

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in batches:
            logits = _action_logits(runtime, batch).detach().cpu().contiguous()
            rows.append(
                {
                    "sequence_id": batch.sequence_id,
                    "batch_size": batch.batch_size,
                    "shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                    "action_logits_sha256": o_runner._tensor_sha256(logits),
                    "greedy_action_ids": logits.argmax(dim=-1).tolist(),
                }
            )
    return {
        "batch_count": len(rows),
        "example_count": sum(value["batch_size"] for value in rows),
        "batches": rows,
        "manifest_sha256": base._sha256(rows),
        "full_logits_retained": False,
    }


def run_qwen_p35_assignment_local_qkvo_residual(
    runtime: nn.Module,
    fit_prompt_input_ids: Sequence[Tensor],
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    *,
    split_provenance: Mapping[str, Any],
    redirect: base.AttentionRedirect = EXACT_REDIRECT,
    require_exact_p35: bool = True,
    rank: int = EXACT_RANK,
    horizon: int = EXACT_HORIZON,
    steps: int = EXACT_STEPS,
    micro_batch_size: int = EXACT_MICRO_BATCH_SIZE,
    q_learning_rate: float = EXACT_Q_LEARNING_RATE,
    k_learning_rate: float = EXACT_K_LEARNING_RATE,
    v_learning_rate: float = EXACT_V_LEARNING_RATE,
    o_learning_rate: float = EXACT_O_LEARNING_RATE,
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
    artifact_sink: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train/evaluate one QKVO child in memory and restore incumbent p35."""

    rank_value = o_runner._positive_int(rank, "rank")
    horizon_value = o_runner._positive_int(horizon, "horizon")
    step_count = o_runner._positive_int(steps, "steps")
    batch_size = o_runner._positive_int(micro_batch_size, "micro_batch_size")
    branch_lrs = {
        "q": o_runner._finite_positive(q_learning_rate, "q_learning_rate"),
        "k": o_runner._finite_positive(k_learning_rate, "k_learning_rate"),
        "v": o_runner._finite_positive(v_learning_rate, "v_learning_rate"),
        "o": o_runner._finite_positive(o_learning_rate, "o_learning_rate"),
    }
    gradient_bound = o_runner._finite_positive(max_gradient_norm, "max_gradient_norm")
    selection_steps = tuple(int(value) for value in calibration_steps)
    if require_exact_p35:
        exact_training_schedule = (
            step_count == EXACT_STEPS and selection_steps == EXACT_CALIBRATION_STEPS
        ) or (
            step_count == EXACT_SELECTED_REPLAY_STEPS
            and selection_steps == EXACT_SELECTED_REPLAY_CALIBRATION_STEPS
        )
        if redirect != EXACT_REDIRECT:
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Exact QKVO run is fixed to L19/G4 -> L21/G6."
            )
        if (
            rank_value != EXACT_RANK
            or horizon_value != EXACT_HORIZON
            or batch_size != EXACT_MICRO_BATCH_SIZE
            or not exact_training_schedule
            or branch_lrs
            != {
                "q": EXACT_Q_LEARNING_RATE,
                "k": EXACT_K_LEARNING_RATE,
                "v": EXACT_V_LEARNING_RATE,
                "o": EXACT_O_LEARNING_RATE,
            }
            or gradient_bound != 1.0
        ):
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Exact run requires rank8, cached horizon4, batch-2 updates, "
                "either the original 32-step 0/8/16/32 schedule or the exact "
                "selected-state 16-step 0/8/16 replay schedule, Q/K lr .001, "
                "V/O lr .002, and clip1."
            )
    if not isinstance(split_provenance, Mapping) or (
        require_exact_p35
        and split_provenance.get("verified_exact_fixture_provenance") is not True
    ):
        raise QwenP35AssignmentLocalQKVOResidualError(
            "Exact p35 mode requires verified three-split fixture provenance."
        )
    limits = trace_limits or FunctionalDecodeLimits(max_horizon=horizon_value)
    if horizon_value > limits.max_horizon:
        raise QwenP35AssignmentLocalQKVOResidualError(
            "horizon exceeds the functional decode bound."
        )
    fit_prompts, calibration_prompts, heldout_prompts, token_audit = (
        o_runner.behavior_runner._validate_token_splits(
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
    site = o_runner._redirected_site(parent_registry, redirect)
    if require_exact_p35 and site != o_runner.P35_L19G4_TO_L21G6:
        raise QwenP35AssignmentLocalQKVOResidualError(
            "Exact redirect provenance is not source156 -> survivor174."
        )
    _bridge, _proposal, proposal_candidates, tentative = base._tentative_codebook(
        runtime, (redirect,)
    )
    codebook, used_projections = tentative
    proposed_registry = codebook.registry
    affected_sites, sites_by_source = base._affected_sites(
        parent_registry, proposed_registry, (redirect,)
    )
    role_scale_audit = o_runner._incumbent_role_scale_audit(
        proposed_registry,
        redirect=redirect,
        require_exact_p35=require_exact_p35,
    )
    proposed_registry_state_before = {
        name: o_runner._tensor_sha256(value)
        for name, value in sorted(proposed_registry.state_dict().items())
    }
    events: list[dict[str, Any]] = []
    o_runner._event(events, "fixtures-validated", heldout_model_forward=False)

    adapter_handle: InstalledAssignmentLocalQKVOResidualAdapter | None = None
    second_handle: InstalledAssignmentLocalQKVOResidualAdapter | None = None
    proposed_snapshot: Qwen3AttentionRuntimeSnapshot | None = None
    lineage: dict[str, Any] | None = None
    training_report: Any = None
    installation_report: dict[str, Any] | None = None
    locked_state: dict[str, Tensor] | None = None
    locked_state_sha256: str | None = None
    heldout_baseline: FunctionalActionEvaluation | None = None
    heldout_locked: FunctionalActionEvaluation | None = None
    calibration_batches: tuple[Any, ...] = ()
    heldout_locked_batches: tuple[Any, ...] = ()
    locked_calibration_output_manifest: dict[str, Any] | None = None
    locked_heldout_output_manifest: dict[str, Any] | None = None
    parent_heldout_greedy = ()
    baseline_heldout_greedy = ()
    locked_heldout_greedy = ()
    parent_rollback_exact = False

    try:
        o_runner._event(
            events, "parent-fit-greedy-targets", heldout_model_forward=False
        )
        parent_fit_greedy, _parent_fit_drivers = o_runner._collect_greedy_split(
            runtime,
            fit_prompts,
            split="fit",
            horizon=horizon_value,
            limits=limits,
        )
        o_runner._event(
            events,
            "parent-calibration-greedy-targets",
            heldout_model_forward=False,
        )
        parent_calibration_greedy, _parent_calibration_drivers = (
            o_runner._collect_greedy_split(
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
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Parent fit/calibration collection changed p35."
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
        o_runner._event(
            events, "tentative-child-installed", heldout_model_forward=False
        )
        o_runner._event(events, "child-fit-greedy-drivers", heldout_model_forward=False)
        child_fit_greedy, child_fit_drivers = o_runner._collect_greedy_split(
            runtime,
            fit_prompts,
            split="fit",
            horizon=horizon_value,
            limits=limits,
        )
        o_runner._event(
            events,
            "child-calibration-greedy-drivers",
            heldout_model_forward=False,
        )
        child_calibration_greedy, child_calibration_drivers = (
            o_runner._collect_greedy_split(
                runtime,
                calibration_prompts,
                split="calibration-selection",
                horizon=horizon_value,
                limits=limits,
            )
        )
        if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Baseline child driver collection changed the tentative child."
            )
        if not o_runner._restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        ):
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Could not restore p35 before student-prefix targets."
            )
        o_runner._event(
            events,
            "parent-fit-student-prefix-targets",
            heldout_model_forward=False,
        )
        parent_fit_on_student = o_runner._collect_forced_split(
            runtime,
            fit_prompts,
            child_fit_drivers,
            split="fit",
            horizon=horizon_value,
            limits=limits,
        )
        o_runner._event(
            events,
            "parent-calibration-student-prefix-targets",
            heldout_model_forward=False,
        )
        parent_calibration_on_student = o_runner._collect_forced_split(
            runtime,
            calibration_prompts,
            child_calibration_drivers,
            split="calibration-selection",
            horizon=horizon_value,
            limits=limits,
        )

        fit_parent_batches = o_runner.build_functional_action_batches(
            parent_fit_greedy,
            split="fit",
            prefix_source="parent-greedy",
            micro_batch_size=batch_size,
        )
        fit_student_batches = o_runner.build_functional_action_batches(
            parent_fit_on_student,
            split="fit",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        fit_batches = o_runner._paired_batches(
            fit_parent_batches, fit_student_batches, split="fit"
        )
        calibration_parent_batches = o_runner.build_functional_action_batches(
            parent_calibration_greedy,
            split="calibration-selection",
            prefix_source="parent-greedy",
            micro_batch_size=batch_size,
        )
        calibration_student_batches = o_runner.build_functional_action_batches(
            parent_calibration_on_student,
            split="calibration-selection",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        calibration_batches = o_runner._paired_batches(
            calibration_parent_batches,
            calibration_student_batches,
            split="calibration-selection",
        )
        if require_exact_p35 and (
            len(fit_batches) != EXACT_STEPS
            or any(value.batch_size != 2 for value in fit_batches)
        ):
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Exact fit must be one complete 32-update batch-2 pass."
            )

        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Reinstalled tentative child differs before QKVO training."
            )
        o_runner._event(
            events,
            "tentative-child-installed-for-qkvo-training",
            heldout_model_forward=False,
        )
        adapter_handle = install_assignment_local_qkvo_residual_adapter(
            runtime,
            proposed_registry,
            sites=(site,),
            rank=rank_value,
            initialization_seed=initialization_seed,
            require_qwen3_4b_dimensions=require_exact_p35,
            role_scale_table_new_for_redirect=True,
        )
        installation_report = adapter_handle.report.as_dict()
        if (
            not adapter_handle.adapter_registry.starts_as_exact_zero_correction
            or not adapter_handle.adapter_registry.input_factors_are_nonzero
        ):
            raise QwenP35AssignmentLocalQKVOResidualError(
                "QKVO adapter initialization is not U=0/random-R."
            )
        o_runner._event(
            events,
            "qkvo-adapter-installed-zero-functional-correction",
            heldout_model_forward=False,
        )
        o_runner._event(events, "training-start", heldout_model_forward=False)
        training_report = train_assignment_local_qkvo_residual_disjoint(
            adapter_handle,
            fit_batches,
            calibration_batches,
            steps=step_count,
            q_learning_rate=branch_lrs["q"],
            k_learning_rate=branch_lrs["k"],
            v_learning_rate=branch_lrs["v"],
            o_learning_rate=branch_lrs["o"],
            beta1=beta1,
            beta2=beta2,
            adam_epsilon=adam_epsilon,
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            max_gradient_norm=gradient_bound,
            calibration_steps=selection_steps,
        )
        if require_exact_p35 and (
            tuple(value["step"] for value in training_report.calibration_history)
            != selection_steps
            or len(training_report.fit_step_history) != step_count
            or dict(training_report.branch_parameter_counts)
            != {"q": 24_576, "k": 21_504, "v": 21_504, "o": 24_576}
        ):
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Exact QKVO trainer omitted required movement diagnostics."
            )
        adapter_handle.assert_live()
        locked_state = o_runner._module_state_snapshot(adapter_handle.adapter_registry)
        locked_state_sha256 = o_runner._state_sha256(locked_state)
        if locked_state_sha256 != training_report.adapter_state_sha256:
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Runner lock hash disagrees with the QKVO trainer state."
            )
        o_runner._event(
            events,
            "winner-locked",
            heldout_model_forward=False,
            best_step=training_report.best_step,
            adapter_state_sha256=locked_state_sha256,
        )
        if artifact_sink is not None:
            locked_calibration_output_manifest = _action_output_manifest(
                runtime, calibration_batches
            )

        o_runner._event(
            events,
            "heldout-locked-qkvo-child-greedy-drivers",
            heldout_model_forward=True,
        )
        locked_heldout_greedy, locked_heldout_drivers = o_runner._collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        adapter_handle.rollback()
        adapter_handle = None
        if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
            raise QwenP35AssignmentLocalQKVOResidualError(
                "QKVO rollback did not recover the tentative child."
            )
        o_runner._event(
            events,
            "heldout-baseline-child-greedy-drivers",
            heldout_model_forward=True,
        )
        baseline_heldout_greedy, baseline_heldout_drivers = (
            o_runner._collect_greedy_split(
                runtime,
                heldout_prompts,
                split="heldout-evaluation",
                horizon=horizon_value,
                limits=limits,
            )
        )
        if not o_runner._restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        ):
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Could not restore p35 for post-lock heldout targets."
            )
        o_runner._event(
            events,
            "heldout-parent-greedy-targets",
            heldout_model_forward=True,
        )
        parent_heldout_greedy, _parent_heldout_drivers = o_runner._collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        o_runner._event(
            events,
            "heldout-parent-student-prefix-targets",
            heldout_model_forward=True,
        )
        parent_heldout_on_locked_student = o_runner._collect_forced_split(
            runtime,
            heldout_prompts,
            locked_heldout_drivers,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        parent_heldout_on_baseline_student = o_runner._collect_forced_split(
            runtime,
            heldout_prompts,
            baseline_heldout_drivers,
            split="heldout-evaluation",
            horizon=horizon_value,
            limits=limits,
        )
        heldout_parent_batches = o_runner.build_functional_action_batches(
            parent_heldout_greedy,
            split="heldout-evaluation",
            prefix_source="parent-greedy",
            micro_batch_size=batch_size,
        )
        heldout_locked_student_batches = o_runner.build_functional_action_batches(
            parent_heldout_on_locked_student,
            split="heldout-evaluation",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        heldout_baseline_student_batches = o_runner.build_functional_action_batches(
            parent_heldout_on_baseline_student,
            split="heldout-evaluation",
            prefix_source="student-greedy",
            micro_batch_size=batch_size,
        )
        heldout_locked_batches = o_runner._paired_batches(
            heldout_parent_batches,
            heldout_locked_student_batches,
            split="heldout-evaluation",
        )
        heldout_baseline_batches = o_runner._paired_batches(
            heldout_parent_batches,
            heldout_baseline_student_batches,
            split="heldout-evaluation",
        )

        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        o_runner._event(
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
        second_handle = install_assignment_local_qkvo_residual_adapter(
            runtime,
            proposed_registry,
            sites=(site,),
            rank=rank_value,
            initialization_seed=initialization_seed,
            require_qwen3_4b_dimensions=require_exact_p35,
            role_scale_table_new_for_redirect=True,
        )
        o_runner._load_exact_state(second_handle.adapter_registry, locked_state)
        o_runner._event(
            events,
            "heldout-locked-qkvo-action-evaluation",
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
        if artifact_sink is not None:
            locked_heldout_output_manifest = _action_output_manifest(
                runtime, heldout_locked_batches
            )
        second_handle.rollback()
        second_handle = None
        parent_rollback_exact = o_runner._restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        )
        if not parent_rollback_exact:
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Final QKVO transaction did not restore p35 exactly."
            )
        o_runner._event(events, "incumbent-p35-restored", heldout_model_forward=False)
    finally:
        failure_in_flight = sys.exc_info()[0] is not None
        cleanup_errors: list[Exception] = []
        for handle in (second_handle, adapter_handle):
            if handle is not None and handle.is_applied:
                try:
                    handle.rollback()
                except Exception as error:  # pragma: no cover
                    cleanup_errors.append(error)
        try:
            parent_rollback_exact = o_runner._restore_parent_exact(
                runtime, parent_installation, parent_snapshot, parent_identity
            )
            if not parent_rollback_exact:
                cleanup_errors.append(
                    QwenP35AssignmentLocalQKVOResidualError(
                        "Cleanup did not restore p35 exactly."
                    )
                )
        except Exception as error:  # pragma: no cover
            cleanup_errors.append(error)
        if cleanup_errors and not failure_in_flight:
            raise QwenP35AssignmentLocalQKVOResidualError(
                "QKVO cleanup failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            ) from cleanup_errors[0]

    if (
        any(
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
        )
        or not parent_rollback_exact
    ):
        raise QwenP35AssignmentLocalQKVOResidualError(
            "QKVO run completed without a full transaction proof."
        )
    assert proposed_snapshot is not None
    assert lineage is not None
    assert training_report is not None
    assert installation_report is not None
    assert locked_state_sha256 is not None
    assert heldout_baseline is not None
    assert heldout_locked is not None
    proposed_registry_state_after = {
        name: o_runner._tensor_sha256(value)
        for name, value in sorted(proposed_registry.state_dict().items())
    }
    if proposed_registry_state_after != proposed_registry_state_before:
        raise QwenP35AssignmentLocalQKVOResidualError(
            "QKVO training changed the tentative child base registry."
        )
    winner_lock = next(value for value in events if value["event"] == "winner-locked")
    first_heldout = next(value for value in events if value["heldout_model_forward"])
    if first_heldout["index"] <= winner_lock["index"]:
        raise QwenP35AssignmentLocalQKVOResidualError(
            "Heldout output was accessed before QKVO winner lock."
        )
    accounting = installation_report["accounting"]
    if require_exact_p35 and (
        accounting["adapter_parameters"] != 92_160
        or accounting["removed_site_parameters"] != 3_276_800
        or accounting["redirect_role_scale_parameters"] != 1_152
        or accounting["net_parameters_saved_vs_incumbent"] != 3_183_488
    ):
        raise QwenP35AssignmentLocalQKVOResidualError(
            "Exact rank-8 QKVO accounting differs from the declared fallback."
        )

    parent_bindings = base._materialized_site_bindings(parent_registry)
    proposed_bindings = base._materialized_site_bindings(proposed_registry)
    affected_payload = [
        {
            "layer_index": layer,
            "physical_group": group,
            "canonical_assignment_axis": proposed_bindings[
                (layer, group)
            ].assignment_axis,
            "parent_source_flat_index": parent_bindings[
                (layer, group)
            ].source_flat_index,
            "proposed_source_flat_index": proposed_bindings[
                (layer, group)
            ].source_flat_index,
            "proposed_active_slot": proposed_bindings[(layer, group)].active_slot,
        }
        for layer, group in affected_sites
    ]
    baseline_rollout = o_runner._greedy_rollout_comparison(
        parent_heldout_greedy, baseline_heldout_greedy
    )
    locked_rollout = o_runner._greedy_rollout_comparison(
        parent_heldout_greedy, locked_heldout_greedy
    )
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "redirect": redirect.as_dict(),
        "exact_p35_mode": require_exact_p35,
        "rank": rank_value,
        "horizon": horizon_value,
        "o_only_negative_provenance": dict(O_ONLY_NEGATIVE_PROVENANCE),
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
            str(source): [list(site) for site in sites]
            for source, sites in sites_by_source.items()
        },
        "used_redirect_projections": [value.as_dict() for value in used_projections],
        "proposal_candidate_count": len(proposal_candidates),
        "fit_action_batches": o_runner._batch_manifest(fit_batches),
        "calibration_action_batches": o_runner._batch_manifest(calibration_batches),
        "fixed_parent_trace_provenance": {
            "fit_parent_greedy": o_runner._trace_manifest(parent_fit_greedy),
            "fit_parent_on_student_greedy": o_runner._trace_manifest(
                parent_fit_on_student
            ),
            "calibration_parent_greedy": o_runner._trace_manifest(
                parent_calibration_greedy
            ),
            "calibration_parent_on_student_greedy": o_runner._trace_manifest(
                parent_calibration_on_student
            ),
            "baseline_child_fit_drivers": o_runner._trace_manifest(child_fit_greedy),
            "baseline_child_calibration_drivers": o_runner._trace_manifest(
                child_calibration_greedy
            ),
        },
        "adapter_installation": installation_report,
        "training": training_report.as_dict(),
        "decisive_training_diagnostics": {
            "calibration_history_steps": [
                value["step"] for value in training_report.calibration_history
            ],
            "fit_step_records": len(training_report.fit_step_history),
            "fit_loss_summary": dict(training_report.fit_loss_summary),
            "branch_gradient_summary": dict(training_report.branch_gradient_summary),
            "branch_parameter_delta_summary": dict(
                training_report.branch_parameter_delta_summary
            ),
            "parameter_gradient_activation": dict(
                training_report.parameter_gradient_activation
            ),
            "all_branches_received_nonzero_gradient": (
                training_report.all_branches_received_nonzero_gradient
            ),
            "all_parameter_tensors_received_nonzero_gradient": (
                training_report.all_parameter_tensors_received_nonzero_gradient
            ),
            "diagnostics_used_for_selection": False,
        },
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
            "locked_qkvo_action_function": heldout_locked.as_dict(),
            "locked_minus_baseline": o_runner._evaluation_delta(
                heldout_baseline,
                heldout_locked,
                student_prefix_domains_identical=(
                    baseline_heldout_drivers == locked_heldout_drivers
                ),
            ),
            "baseline_greedy_rollout": baseline_rollout,
            "locked_qkvo_greedy_rollout": locked_rollout,
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
            "q_correction_before_norm_rope": True,
            "k_correction_before_norm_rope_cache": True,
            "v_correction_before_cache": True,
            "o_correction_on_post_softmax_group_context": True,
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
            "optimizer_state_saved_or_counted": False,
            "structural_compaction_measured": True,
            "functional_compaction_certified": False,
        },
        "transaction": {
            "tentative_child_base_registry_unchanged": True,
            "adapter_parameters_only_trained": True,
            "locked_qkvo_reinstalled_exactly_for_heldout": True,
            "incumbent_p35_runtime_restored_exactly": parent_rollback_exact,
            "candidate_installed_after_return": False,
        },
        "checkpoint_saved": False,
        "fold_committed": False,
        "compaction_certified": False,
        "training_used": True,
        "training_scope": "rank-8-one-site-q-k-v-o-residual-factors-only",
    }
    report["report_sha256"] = base._sha256(report)
    if artifact_sink is not None:
        if (
            locked_state is None
            or locked_calibration_output_manifest is None
            or locked_heldout_output_manifest is None
        ):
            raise QwenP35AssignmentLocalQKVOResidualError(
                "Requested replay artifacts were not captured completely."
            )
        artifact_sink.clear()
        artifact_sink.update(
            {
                "locked_state": {
                    name: value.detach().cpu().clone()
                    for name, value in locked_state.items()
                },
                "calibration_batches": tuple(calibration_batches),
                "heldout_batches": tuple(heldout_locked_batches),
                "calibration_action_output_manifest": (
                    locked_calibration_output_manifest
                ),
                "heldout_action_output_manifest": locked_heldout_output_manifest,
                "heldout_greedy_trace_manifest": o_runner._trace_manifest(
                    locked_heldout_greedy
                ),
            }
        )
    return report


def load_verified_prompt_splits(*args: Any, **kwargs: Any):
    return o_runner.load_verified_prompt_splits(*args, **kwargs)


def _encode_prompts(*args: Any, **kwargs: Any):
    return o_runner._encode_prompts(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--fit-prompts-json", type=Path, default=DEFAULT_FIT_PROMPTS)
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
    parser.add_argument("--horizon", type=int, default=EXACT_HORIZON)
    parser.add_argument("--steps", type=int, default=EXACT_STEPS)
    parser.add_argument("--micro-batch-size", type=int, default=EXACT_MICRO_BATCH_SIZE)
    parser.add_argument("--q-learning-rate", type=float, default=EXACT_Q_LEARNING_RATE)
    parser.add_argument("--k-learning-rate", type=float, default=EXACT_K_LEARNING_RATE)
    parser.add_argument("--v-learning-rate", type=float, default=EXACT_V_LEARNING_RATE)
    parser.add_argument("--o-learning-rate", type=float, default=EXACT_O_LEARNING_RATE)
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
    report = run_qwen_p35_assignment_local_qkvo_residual(
        runtime,
        fit_prompts,
        calibration_prompts,
        heldout_prompts,
        split_provenance=fixtures.provenance,
        rank=args.rank,
        horizon=args.horizon,
        steps=args.steps,
        micro_batch_size=args.micro_batch_size,
        q_learning_rate=args.q_learning_rate,
        k_learning_rate=args.k_learning_rate,
        v_learning_rate=args.v_learning_rate,
        o_learning_rate=args.o_learning_rate,
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
        "update_rule": "manual-adam-per-qkvo-branch",
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
