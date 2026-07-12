#!/usr/bin/env python3
"""Jointly train uneven-rank Q/K/V/O residuals for all p35->p34 redirects.

The runner searches three fixed ragged-rank schedules on fit/calibration only,
locks one schedule and one training step, then begins the heldout stage.
Only a strict small-gate winner is written as a factor-only bundle. Every
candidate and the final reloaded bundle are rolled back to the exact p35
incumbent; no full model or optimizer checkpoint can be saved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import neural_forge.hf_assignment_local_qkvo_residual_adapter as qkvo_adapter_module
import neural_forge.hf_qkvo_factor_pack as factor_pack_module
import neural_forge.hf_ragged_assignment_local_qkvo_residual_adapter as ragged_adapter_module
import neural_forge.hf_ragged_qkvo_factor_bundle as ragged_bundle_module
import reproduce_pack_qwen_p35_qkvo_selected as replay_runner
import run_qwen_p35_assignment_local_qkvo_residual as single_runner
import run_qwen_p35_functional_residual_adapter as o_runner
import run_qwen_p35_vo_role_scale_search as role_search
import run_qwen_residual_attention_probe as base
from neural_forge.hf_assignment_local_qkvo_residual_adapter import (
    train_assignment_local_qkvo_residual_disjoint,
)
from neural_forge.hf_functional_decode_trace import FunctionalDecodeLimits
from neural_forge.hf_functional_residual_adapter import (
    FunctionalActionEvaluation,
    RedirectedAttentionSite,
    evaluate_functional_action_batches,
)
from neural_forge.hf_qkvo_factor_pack import (
    QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
    QKVO_SOURCE_PROVENANCE_SCHEMA,
    LoadedQKVOFactorPack,
    adapter_state_sha256,
    file_sha256,
    load_qkvo_factor_pack,
    tensor_sha256,
)
from neural_forge.hf_qwen_attention_surgery import (
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report
from neural_forge.hf_ragged_assignment_local_qkvo_residual_adapter import (
    InstalledRaggedAssignmentLocalQKVOResidualAdapter,
    RaggedAssignmentLocalQKVOAccounting,
    install_ragged_assignment_local_qkvo_residual_adapter,
)
from neural_forge.hf_ragged_qkvo_factor_bundle import (
    install_ragged_qkvo_factor_bundle,
    load_ragged_qkvo_factor_bundle,
    write_ragged_qkvo_factor_bundle,
)


RUN_SCHEMA = "neural-forge-qwen-p35-multisite-ragged-qkvo-run-v1"
EXACT_P35_PARAMETER_COUNT = 3_995_570_176
EXACT_EIGHT_SITE_REMOVED_PARAMETERS = 26_214_400
EXACT_ROLE_SCALE_PARAMETERS = 1_152
EXPECTED_ANCHOR_STATE_SHA256 = (
    "f4cad9b277826a6724d76c9727999dc614ae5c54ef6197b9902d8d4c26c20240"
)

# Exact p35->p34 lineage extracted read-only from the two preserved registries.
# Sorted by executable physical source site for canonical ragged state ordering.
P35_TO_P34_REDIRECTS = (
    base.AttentionRedirect(10, 2, 15, 5),
    base.AttentionRedirect(13, 2, 35, 0),
    base.AttentionRedirect(15, 2, 19, 5),
    base.AttentionRedirect(19, 4, 21, 6),
    base.AttentionRedirect(21, 3, 7, 6),
    base.AttentionRedirect(25, 1, 26, 4),
    base.AttentionRedirect(25, 5, 20, 6),
    base.AttentionRedirect(30, 1, 14, 4),
)
DEFAULT_RANK_SCHEDULES: Mapping[str, tuple[int, ...]] = {
    "anchor-economy-r36": (4, 4, 4, 8, 4, 4, 4, 4),
    "anchor-targeted-r40": (4, 4, 6, 8, 4, 4, 4, 6),
    "anchor-conservative-r50": (6, 6, 6, 8, 6, 6, 6, 6),
}
DEFAULT_STEPS = 32
DEFAULT_CALIBRATION_STEPS = (0, 8, 16, 32)


class QwenP35MultisiteRaggedQKVOError(RuntimeError):
    """A multi-site lineage, split, gate, bundle, or rollback failed."""


def _implementation_provenance() -> dict[str, Any]:
    files = {
        "multisite_runner": Path(__file__).resolve(),
        "single_site_fixture_runner": Path(single_runner.__file__).resolve(),
        "functional_fixture_runner": Path(o_runner.__file__).resolve(),
        "assignment_local_adapter_and_trainer": Path(
            qkvo_adapter_module.__file__
        ).resolve(),
        "ragged_adapter": Path(ragged_adapter_module.__file__).resolve(),
        "single_site_factor_pack": Path(factor_pack_module.__file__).resolve(),
        "ragged_factor_bundle": Path(ragged_bundle_module.__file__).resolve(),
    }
    rows = [
        {"role": role, "file": path.name, "sha256": file_sha256(path)}
        for role, path in sorted(files.items())
    ]
    return {
        "schema": "neural-forge-p35-multisite-ragged-qkvo-implementation-v1",
        "files": rows,
        "file_count": len(rows),
        "manifest_sha256": base._sha256(rows),
    }


def _assert_exact_source_contract(
    anchor: LoadedQKVOFactorPack,
    *,
    split_provenance: Mapping[str, Any],
    source_runtime: Mapping[str, Any],
    execution_environment: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = anchor.manifest.get("provenance")
    anchor_runtime = (
        provenance.get("source_runtime") if isinstance(provenance, Mapping) else None
    )
    expected_environment = {
        "python": replay_runner.EXPECTED_PYTHON_VERSION,
        "torch": replay_runner.EXPECTED_TORCH_VERSION,
        "device": "cuda:0",
        "dtype": "float16",
        "gpu_name": replay_runner.EXPECTED_GPU_NAME,
    }
    if (
        source_runtime.get("manifest_sha256")
        != replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
        or not isinstance(provenance, Mapping)
        or provenance.get("source_report_file_sha256")
        != replay_runner.EXPECTED_SOURCE_REPORT_FILE_SHA256
        or provenance.get("source_report_sha256")
        != replay_runner.EXPECTED_SOURCE_REPORT_SHA256
        or provenance.get("fixture_provenance_sha256")
        != split_provenance.get("provenance_sha256")
        or not isinstance(anchor_runtime, Mapping)
        or anchor_runtime.get("manifest_sha256")
        != replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
        or dict(execution_environment) != expected_environment
    ):
        raise QwenP35MultisiteRaggedQKVOError(
            "Anchor, p35 checkpoint, fixtures, or P40 execution environment differ "
            "from the exact selected replay contract."
        )
    return {
        "base_codebook_manifest_sha256": (
            replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
        ),
        "anchor_manifest_sha256": anchor.manifest["manifest_sha256"],
        "anchor_selected_state_sha256": anchor.manifest[
            "selected_adapter_state_sha256"
        ],
        "fixture_provenance_sha256": split_provenance["provenance_sha256"],
        "execution_environment": expected_environment,
    }


def _redirected_sites(
    parent_registry: Any,
    redirects: Sequence[base.AttentionRedirect],
) -> tuple[RedirectedAttentionSite, ...]:
    return tuple(
        o_runner._redirected_site(parent_registry, redirect) for redirect in redirects
    )


def _assert_multisite_lineage(
    parent: Any,
    proposed: Any,
    parent_registry: Any,
    proposed_registry: Any,
    redirects: Sequence[base.AttentionRedirect],
) -> dict[str, Any]:
    groups = int(parent.key_value_heads)
    removed = {value.source_layer * groups + value.source_group for value in redirects}
    survivors = {
        value.target_layer * groups + value.target_group for value in redirects
    }
    if (
        parent.logical_layers != 36
        or groups != 8
        or parent.prototype_count != 280
        or proposed.prototype_count != 272
        or len(removed) != 8
        or len(survivors) != 8
        or set(proposed.prototype_source_indices)
        != set(parent.prototype_source_indices).difference(removed)
        or not survivors.issubset(set(proposed.prototype_source_indices))
        or not torch.equal(
            parent_registry.group_permutations.detach().cpu(),
            proposed_registry.group_permutations.detach().cpu(),
        )
    ):
        raise QwenP35MultisiteRaggedQKVOError(
            "Tentative child is not the exact p35 280->272 eight-source lineage."
        )
    payload = {
        "parent_prototype_count": 280,
        "proposed_prototype_count": 272,
        "removed_source_flat_indices": sorted(removed),
        "survivor_source_flat_indices": sorted(survivors),
        "group_permutations_unchanged": True,
        "source_set_is_parent_minus_exactly_eight": True,
    }
    payload["lineage_sha256"] = base._sha256(payload)
    return payload


def _seed_anchor_site(
    handle: InstalledRaggedAssignmentLocalQKVOResidualAdapter,
    anchor: LoadedQKVOFactorPack,
) -> None:
    site_manifest = anchor.manifest.get("site")
    if (
        anchor.manifest.get("selected_adapter_state_sha256")
        != EXPECTED_ANCHOR_STATE_SHA256
        or anchor.manifest.get("rank") != 8
        or not isinstance(site_manifest, Mapping)
        or (site_manifest.get("layer_index"), site_manifest.get("physical_group"))
        != (19, 4)
    ):
        raise QwenP35MultisiteRaggedQKVOError(
            "Seed pack is not the exact selected L19/G4 rank-8 anchor."
        )
    rows = handle.adapter_registry.site_indices.detach().cpu().tolist()
    try:
        site_index = rows.index([19, 4])
    except ValueError as error:
        raise QwenP35MultisiteRaggedQKVOError(
            "Ragged schedule omitted the proven anchor site."
        ) from error
    if handle.adapter_registry.ranks[site_index] != 8:
        raise QwenP35MultisiteRaggedQKVOError(
            "Every schedule must preserve rank 8 at the proven anchor."
        )
    with torch.no_grad():
        for name, source in anchor.factors.items():
            destination = getattr(
                handle.adapter_registry,
                name.split("_")[0] + "_" + name.split("_")[1],
            )[site_index]
            value = source.squeeze(0)
            if (
                tuple(destination.shape) != tuple(value.shape)
                or destination.dtype != value.dtype
            ):
                raise QwenP35MultisiteRaggedQKVOError(
                    "Anchor factor shape/dtype differs from ragged site."
                )
            destination.copy_(value.to(device=destination.device))
    handle.adapter_registry._assert_finite()


def _schedule_key(
    evaluation: FunctionalActionEvaluation,
    parameter_count: int,
) -> tuple[float, float, int]:
    return (
        -float(evaluation.parent_action_top1_rate),
        float(evaluation.mean_parent_action_rank),
        int(parameter_count),
    )


def _discrete_evaluation_contract(
    evaluation: FunctionalActionEvaluation,
) -> dict[str, Any]:
    payload = evaluation.as_dict()
    keys = (
        "schema",
        "split",
        "example_count",
        "parent_action_top1_rate",
        "mean_parent_action_rank",
        "used_for_selection",
        "prefix_sources",
        "functional_outputs_only",
        "attention_residual_cache_weight_similarity_used",
    )
    return {name: payload[name] for name in keys}


def _continuous_evaluation_diagnostics(
    evaluation: FunctionalActionEvaluation,
) -> dict[str, Any]:
    payload = evaluation.as_dict()
    keys = (
        "total_loss",
        "hard_action_loss",
        "distribution_loss",
        "mean_parent_action_log_probability",
        "evaluation_sha256",
    )
    return {name: payload[name] for name in keys}


def _assert_candidate_accounting(
    handle: InstalledRaggedAssignmentLocalQKVOResidualAdapter,
) -> RaggedAssignmentLocalQKVOAccounting:
    report = handle.report
    accounting = report.accounting
    expected_before = (
        EXACT_P35_PARAMETER_COUNT
        - EXACT_EIGHT_SITE_REMOVED_PARAMETERS
        + EXACT_ROLE_SCALE_PARAMETERS
    )
    if (
        accounting.removed_parameters != EXACT_EIGHT_SITE_REMOVED_PARAMETERS
        or accounting.role_scale_parameters != EXACT_ROLE_SCALE_PARAMETERS
        or accounting.adapter_parameters != report.adapter_parameter_count
        or accounting.net_parameters_saved
        != EXACT_P35_PARAMETER_COUNT - report.model_parameter_count_installed
        or report.model_parameter_count_before != expected_before
        or report.model_parameter_count_installed
        != EXACT_P35_PARAMETER_COUNT - accounting.net_parameters_saved
    ):
        raise QwenP35MultisiteRaggedQKVOError(
            "Live joint child parameter accounting differs from exact p35."
        )
    return accounting


def _fit_eligible_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        value
        for value in rows
        if value.get("fit_gate_passed") is True
        and value.get("all_parameter_tensors_received_nonzero_gradient") is True
    )


def _load_state(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    current = module.state_dict()
    if set(current) != set(state):
        raise QwenP35MultisiteRaggedQKVOError(
            "Selected ragged state keys differ from fresh registry."
        )
    with torch.no_grad():
        for name, destination in current.items():
            source = state[name]
            if (
                tuple(destination.shape) != tuple(source.shape)
                or destination.dtype != source.dtype
            ):
                raise QwenP35MultisiteRaggedQKVOError(
                    "Selected ragged state shape/dtype changed."
                )
            destination.copy_(source.to(device=destination.device))
    if adapter_state_sha256(module.state_dict()) != adapter_state_sha256(state):
        raise QwenP35MultisiteRaggedQKVOError(
            "Fresh ragged registry did not reproduce selected state."
        )


def _role_scale_provenance_by_site(
    proposed_registry: Any,
    redirects: Sequence[base.AttentionRedirect],
) -> tuple[dict[str, Any], ...]:
    table = proposed_registry.role_scales
    if not isinstance(table, Tensor):
        raise QwenP35MultisiteRaggedQKVOError(
            "Tentative child lacks the explicit role-scale table."
        )
    table_sha = tensor_sha256(table)
    result = []
    for redirect in redirects:
        audit = o_runner._incumbent_role_scale_audit(
            proposed_registry,
            redirect=redirect,
            require_exact_p35=False,
        )
        result.append(
            {
                "schema": QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
                "q": audit["q_scale"],
                "k": audit["k_scale"],
                "v": audit["v_scale"],
                "o": audit["o_scale"],
                "role_scales_trainable": False,
                "table_sha256": table_sha,
                "canonical_assignment_axis": audit["canonical_assignment_axis"],
            }
        )
    return tuple(result)


def _greedy_gate(
    parent_traces: Sequence[Any],
    baseline_traces: Sequence[Any],
    locked_traces: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = o_runner._greedy_rollout_comparison(parent_traces, baseline_traces)
    locked = o_runner._greedy_rollout_comparison(parent_traces, locked_traces)
    return baseline, locked


def run_qwen_p35_multisite_ragged_qkvo(
    runtime: nn.Module,
    fit_prompt_input_ids: Sequence[Tensor],
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    *,
    split_provenance: Mapping[str, Any],
    anchor_pack: LoadedQKVOFactorPack,
    source_runtime: Mapping[str, Any],
    execution_environment: Mapping[str, Any],
    bundle_output: Path,
    rank_schedules: Mapping[str, Sequence[int]] = DEFAULT_RANK_SCHEDULES,
    steps: int = DEFAULT_STEPS,
    calibration_steps: Sequence[int] = DEFAULT_CALIBRATION_STEPS,
    horizon: int = 4,
    micro_batch_size: int = 2,
    initialization_seed: int = 0,
    trace_limits: FunctionalDecodeLimits | None = None,
) -> dict[str, Any]:
    """Search ragged schedules, heldout-gate one winner, pack, reload, restore."""

    if split_provenance.get("verified_exact_fixture_provenance") is not True:
        raise QwenP35MultisiteRaggedQKVOError(
            "Exact three-split fixture provenance is required."
        )
    observed_environment = replay_runner.assert_exact_execution_environment(
        runtime,
        device=str(execution_environment.get("device")),
        dtype=str(execution_environment.get("dtype")),
    )
    if observed_environment != dict(execution_environment):
        raise QwenP35MultisiteRaggedQKVOError(
            "Declared and observed exact execution environments differ."
        )
    source_contract = _assert_exact_source_contract(
        anchor_pack,
        split_provenance=split_provenance,
        source_runtime=source_runtime,
        execution_environment=execution_environment,
    )
    implementation_provenance = _implementation_provenance()
    schedules = {
        str(name): tuple(int(value) for value in ranks)
        for name, ranks in rank_schedules.items()
    }
    if (
        not schedules
        or len(set(schedules)) != len(schedules)
        or any(len(ranks) != 8 or ranks[3] != 8 for ranks in schedules.values())
        or any(rank <= 0 or rank > 8 for ranks in schedules.values() for rank in ranks)
    ):
        raise QwenP35MultisiteRaggedQKVOError(
            "Schedules need eight ranks in 1..8 with rank-8 at L19/G4."
        )
    fit_prompts, calibration_prompts, heldout_prompts, token_audit = (
        o_runner.behavior_runner._validate_token_splits(
            fit_prompt_input_ids,
            calibration_prompt_input_ids,
            heldout_prompt_input_ids,
            max_prompts_per_split=max(o_runner.EXPECTED_SPLIT_COUNTS.values()),
            require_exact_p35=True,
        )
    )
    limits = trace_limits or FunctionalDecodeLimits(max_horizon=horizon)
    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_identity = role_search._runtime_state_identity_sha256(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    if (
        qwen3_codebook_parameter_report(runtime).unique_parameter_count
        != EXACT_P35_PARAMETER_COUNT
    ):
        raise QwenP35MultisiteRaggedQKVOError(
            "Loaded incumbent is not the exact 3,995,570,176-parameter p35."
        )
    _bridge, _proposal, proposal_candidates, tentative = base._tentative_codebook(
        runtime, P35_TO_P34_REDIRECTS
    )
    codebook, used_projections = tentative
    proposed_registry = codebook.registry
    sites = _redirected_sites(parent_registry, P35_TO_P34_REDIRECTS)
    proposed_snapshot = None
    lineage = None
    events: list[dict[str, Any]] = []

    def event(name: str, *, heldout: bool = False, **payload: Any) -> None:
        events.append(
            {"index": len(events), "event": name, "heldout_forward": heldout, **payload}
        )

    active_handle: InstalledRaggedAssignmentLocalQKVOResidualAdapter | None = None
    parent_restored = False
    bundle_written = False
    bundle_manifest: dict[str, Any] | None = None
    try:
        event("parent-fit-calibration-targets")
        parent_fit_greedy, _ = o_runner._collect_greedy_split(
            runtime, fit_prompts, split="fit", horizon=horizon, limits=limits
        )
        parent_calibration_greedy, _ = o_runner._collect_greedy_split(
            runtime,
            calibration_prompts,
            split="calibration-selection",
            horizon=horizon,
            limits=limits,
        )
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        joint_snapshot = qwen3_attention_runtime_snapshot(runtime)
        proposed_snapshot = joint_snapshot
        lineage = _assert_multisite_lineage(
            parent_snapshot,
            proposed_snapshot,
            parent_registry,
            proposed_registry,
            P35_TO_P34_REDIRECTS,
        )
        event("joint-child-fit-calibration-drivers")
        child_fit_greedy, child_fit_drivers = o_runner._collect_greedy_split(
            runtime, fit_prompts, split="fit", horizon=horizon, limits=limits
        )
        child_calibration_greedy, child_calibration_drivers = (
            o_runner._collect_greedy_split(
                runtime,
                calibration_prompts,
                split="calibration-selection",
                horizon=horizon,
                limits=limits,
            )
        )
        if not o_runner._restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        ):
            raise QwenP35MultisiteRaggedQKVOError(
                "Could not restore p35 before forced targets."
            )
        parent_fit_on_student = o_runner._collect_forced_split(
            runtime,
            fit_prompts,
            child_fit_drivers,
            split="fit",
            horizon=horizon,
            limits=limits,
        )
        parent_calibration_on_student = o_runner._collect_forced_split(
            runtime,
            calibration_prompts,
            child_calibration_drivers,
            split="calibration-selection",
            horizon=horizon,
            limits=limits,
        )
        fit_batches = o_runner._paired_batches(
            o_runner.build_functional_action_batches(
                parent_fit_greedy,
                split="fit",
                prefix_source="parent-greedy",
                micro_batch_size=micro_batch_size,
            ),
            o_runner.build_functional_action_batches(
                parent_fit_on_student,
                split="fit",
                prefix_source="student-greedy",
                micro_batch_size=micro_batch_size,
            ),
            split="fit",
        )
        calibration_batches = o_runner._paired_batches(
            o_runner.build_functional_action_batches(
                parent_calibration_greedy,
                split="calibration-selection",
                prefix_source="parent-greedy",
                micro_batch_size=micro_batch_size,
            ),
            o_runner.build_functional_action_batches(
                parent_calibration_on_student,
                split="calibration-selection",
                prefix_source="student-greedy",
                micro_batch_size=micro_batch_size,
            ),
            split="calibration-selection",
        )
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        baseline_fit = evaluate_functional_action_batches(
            runtime,
            fit_batches,
            expected_split="fit",
            hard_action_weight=1.0,
            distribution_weight=0.1,
            temperature=1.0,
            used_for_selection=False,
        )
        baseline_calibration = evaluate_functional_action_batches(
            runtime,
            calibration_batches,
            expected_split="calibration-selection",
            hard_action_weight=1.0,
            distribution_weight=0.1,
            temperature=1.0,
            used_for_selection=True,
        )
        candidate_rows: list[dict[str, Any]] = []
        candidate_states: dict[str, dict[str, Tensor]] = {}
        candidate_training: dict[str, Any] = {}
        candidate_fit_evaluations: dict[str, FunctionalActionEvaluation] = {}
        for candidate_index, (name, ranks) in enumerate(schedules.items()):
            event("rank-schedule-training", schedule=name)
            active_handle = install_ragged_assignment_local_qkvo_residual_adapter(
                runtime,
                proposed_registry,
                sites=sites,
                site_ranks=ranks,
                owner_name=f"ragged_qkvo_candidate_{candidate_index}",
                initialization_seed=initialization_seed,
                require_qwen3_4b_dimensions=True,
                role_scale_table_new_for_redirects=True,
            )
            accounting = _assert_candidate_accounting(active_handle)
            _seed_anchor_site(active_handle, anchor_pack)
            training = train_assignment_local_qkvo_residual_disjoint(
                active_handle,
                fit_batches,
                calibration_batches,
                steps=steps,
                q_learning_rate=1e-3,
                k_learning_rate=1e-3,
                v_learning_rate=2e-3,
                o_learning_rate=2e-3,
                beta1=0.9,
                beta2=0.999,
                adam_epsilon=1e-8,
                hard_action_weight=1.0,
                distribution_weight=0.1,
                temperature=1.0,
                max_gradient_norm=1.0,
                calibration_steps=calibration_steps,
            )
            fit_evaluation = evaluate_functional_action_batches(
                runtime,
                fit_batches,
                expected_split="fit",
                hard_action_weight=1.0,
                distribution_weight=0.1,
                temperature=1.0,
                used_for_selection=False,
            )
            fit_gate_passed = _schedule_key(
                fit_evaluation, accounting.adapter_parameters
            ) <= _schedule_key(baseline_fit, accounting.adapter_parameters)
            state = o_runner._module_state_snapshot(active_handle.adapter_registry)
            state_sha = adapter_state_sha256(state)
            if state_sha != training.adapter_state_sha256:
                raise QwenP35MultisiteRaggedQKVOError(
                    "Candidate state hash differs from locked trainer state."
                )
            row = {
                "schedule": name,
                "site_ranks": list(ranks),
                "rank_sum": sum(ranks),
                "adapter_parameters": accounting.adapter_parameters,
                "net_parameters_saved": accounting.net_parameters_saved,
                "best_step": training.best_step,
                "fit": fit_evaluation.as_dict(),
                "fit_gate_passed": fit_gate_passed,
                "calibration": training.final_calibration.as_dict(),
                "selection_key": list(
                    _schedule_key(
                        training.final_calibration, accounting.adapter_parameters
                    )
                ),
                "adapter_state_sha256": state_sha,
                "all_parameter_tensors_received_nonzero_gradient": (
                    training.all_parameter_tensors_received_nonzero_gradient
                ),
            }
            candidate_rows.append(row)
            candidate_states[name] = state
            candidate_training[name] = training
            candidate_fit_evaluations[name] = fit_evaluation
            active_handle.rollback()
            active_handle = None
            if qwen3_attention_runtime_snapshot(runtime) != joint_snapshot:
                raise QwenP35MultisiteRaggedQKVOError(
                    "Candidate rollback changed the joint child."
                )
        eligible_rows = _fit_eligible_candidate_rows(candidate_rows)
        if not eligible_rows:
            raise QwenP35MultisiteRaggedQKVOError(
                "No ragged schedule passed the independent fit/gradient gate."
            )
        winner_row = min(
            eligible_rows,
            key=lambda value: tuple(value["selection_key"]),
        )
        winner_name = str(winner_row["schedule"])
        winner_state = candidate_states[winner_name]
        winner_training = candidate_training[winner_name]
        winner_fit = candidate_fit_evaluations[winner_name]
        if _schedule_key(
            winner_training.final_calibration,
            int(winner_row["adapter_parameters"]),
        ) > _schedule_key(baseline_calibration, int(winner_row["adapter_parameters"])):
            raise QwenP35MultisiteRaggedQKVOError(
                "No ragged schedule preserved the joint-child calibration gate."
            )
        event("winner-schedule-locked", schedule=winner_name)

        active_handle = install_ragged_assignment_local_qkvo_residual_adapter(
            runtime,
            proposed_registry,
            sites=sites,
            site_ranks=schedules[winner_name],
            owner_name="ragged_qkvo_locked_winner",
            initialization_seed=initialization_seed,
            require_qwen3_4b_dimensions=True,
            role_scale_table_new_for_redirects=True,
        )
        _assert_candidate_accounting(active_handle)
        _load_state(active_handle.adapter_registry, winner_state)
        event("locked-heldout-drivers", heldout=True)
        locked_heldout_greedy, locked_heldout_drivers = o_runner._collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        active_handle.rollback()
        active_handle = None
        event("baseline-heldout-drivers", heldout=True)
        baseline_heldout_greedy, baseline_heldout_drivers = (
            o_runner._collect_greedy_split(
                runtime,
                heldout_prompts,
                split="heldout-evaluation",
                horizon=horizon,
                limits=limits,
            )
        )
        if not o_runner._restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        ):
            raise QwenP35MultisiteRaggedQKVOError(
                "Could not restore p35 for heldout parent targets."
            )
        event("parent-heldout-targets", heldout=True)
        parent_heldout_greedy, _ = o_runner._collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        parent_on_locked = o_runner._collect_forced_split(
            runtime,
            heldout_prompts,
            locked_heldout_drivers,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        parent_on_baseline = o_runner._collect_forced_split(
            runtime,
            heldout_prompts,
            baseline_heldout_drivers,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        heldout_parent_batches = o_runner.build_functional_action_batches(
            parent_heldout_greedy,
            split="heldout-evaluation",
            prefix_source="parent-greedy",
            micro_batch_size=micro_batch_size,
        )
        heldout_locked_batches = o_runner._paired_batches(
            heldout_parent_batches,
            o_runner.build_functional_action_batches(
                parent_on_locked,
                split="heldout-evaluation",
                prefix_source="student-greedy",
                micro_batch_size=micro_batch_size,
            ),
            split="heldout-evaluation",
        )
        heldout_baseline_batches = o_runner._paired_batches(
            heldout_parent_batches,
            o_runner.build_functional_action_batches(
                parent_on_baseline,
                split="heldout-evaluation",
                prefix_source="student-greedy",
                micro_batch_size=micro_batch_size,
            ),
            split="heldout-evaluation",
        )
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        baseline_heldout = evaluate_functional_action_batches(
            runtime,
            heldout_baseline_batches,
            expected_split="heldout-evaluation",
            used_for_selection=False,
        )
        active_handle = install_ragged_assignment_local_qkvo_residual_adapter(
            runtime,
            proposed_registry,
            sites=sites,
            site_ranks=schedules[winner_name],
            owner_name="ragged_qkvo_bundle_source",
            initialization_seed=initialization_seed,
            require_qwen3_4b_dimensions=True,
            role_scale_table_new_for_redirects=True,
        )
        _assert_candidate_accounting(active_handle)
        _load_state(active_handle.adapter_registry, winner_state)
        locked_fit = evaluate_functional_action_batches(
            runtime,
            fit_batches,
            expected_split="fit",
            hard_action_weight=1.0,
            distribution_weight=0.1,
            temperature=1.0,
            used_for_selection=False,
        )
        if _discrete_evaluation_contract(locked_fit) != (
            _discrete_evaluation_contract(winner_fit)
        ):
            raise QwenP35MultisiteRaggedQKVOError(
                "Reinstalled winner changed its fit behavior contract."
            )
        locked_heldout = evaluate_functional_action_batches(
            runtime,
            heldout_locked_batches,
            expected_split="heldout-evaluation",
            used_for_selection=False,
        )
        baseline_rollout, locked_rollout = _greedy_gate(
            parent_heldout_greedy,
            baseline_heldout_greedy,
            locked_heldout_greedy,
        )
        heldout_gate_passed = bool(
            locked_heldout.parent_action_top1_rate == 1.0
            and locked_heldout.mean_parent_action_rank == 1.0
            and locked_heldout.parent_action_top1_rate
            >= baseline_heldout.parent_action_top1_rate
            and locked_rollout["exact_rollout_matches"] == len(heldout_prompts)
            and locked_rollout["token_matches"] == locked_rollout["token_count"]
            and locked_rollout["token_matches"] >= baseline_rollout["token_matches"]
        )
        if not heldout_gate_passed:
            raise QwenP35MultisiteRaggedQKVOError(
                "Locked winner failed strict heldout 100%/exact-rollout gate."
            )
        fit_output_manifest = single_runner._action_output_manifest(
            runtime, fit_batches
        )
        calibration_output_manifest = single_runner._action_output_manifest(
            runtime, calibration_batches
        )
        heldout_output_manifest = single_runner._action_output_manifest(
            runtime, heldout_locked_batches
        )
        role_provenance = _role_scale_provenance_by_site(
            proposed_registry, P35_TO_P34_REDIRECTS
        )
        seed_source = anchor_pack.manifest["provenance"]
        source_provenance = {
            "schema": QKVO_SOURCE_PROVENANCE_SCHEMA,
            "source_report_file_sha256": seed_source["source_report_file_sha256"],
            "source_report_sha256": seed_source["source_report_sha256"],
            "fixture_provenance_sha256": split_provenance["provenance_sha256"],
            "fit_evaluation_sha256": locked_fit.as_dict()["evaluation_sha256"],
            "calibration_evaluation_sha256": winner_training.final_calibration.as_dict()[
                "evaluation_sha256"
            ],
            "heldout_evaluation_sha256": locked_heldout.as_dict()["evaluation_sha256"],
            "anchor_pack_manifest_sha256": anchor_pack.manifest["manifest_sha256"],
            "source_runtime": dict(source_runtime),
            "exact_source_contract": source_contract,
            "multisite_implementation": implementation_provenance,
            "reproduced_environment": {
                **dict(execution_environment),
                "cuda_runtime": torch.version.cuda,
            },
            "rank_schedule": winner_name,
            "site_ranks": list(schedules[winner_name]),
            "training_schedule": {
                "steps": steps,
                "calibration_steps": list(calibration_steps),
                "horizon": horizon,
                "micro_batch_size": micro_batch_size,
                "q_k_learning_rate": 0.001,
                "v_o_learning_rate": 0.002,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "hard_action_weight": 1.0,
                "distribution_weight": 0.1,
                "temperature": 1.0,
                "max_gradient_norm": 1.0,
            },
        }
        bundle_manifest = write_ragged_qkvo_factor_bundle(
            bundle_output,
            active_handle.adapter_registry,
            selected_registry_state_sha256=adapter_state_sha256(winner_state),
            sites=sites,
            role_scale_provenance=role_provenance,
            source_provenance=source_provenance,
        )
        bundle_written = True
        active_handle.rollback()
        active_handle = None
        loaded_bundle = load_ragged_qkvo_factor_bundle(bundle_output)
        active_handle = install_ragged_assignment_local_qkvo_residual_adapter(
            runtime,
            proposed_registry,
            sites=sites,
            site_ranks=schedules[winner_name],
            owner_name="ragged_qkvo_bundle_reload",
            initialization_seed=initialization_seed + 999,
            require_qwen3_4b_dimensions=True,
            role_scale_table_new_for_redirects=True,
        )
        _assert_candidate_accounting(active_handle)
        reloaded_sha = install_ragged_qkvo_factor_bundle(
            active_handle.adapter_registry,
            loaded_bundle,
            attention_registry=proposed_registry,
        )
        reloaded_fit = evaluate_functional_action_batches(
            runtime,
            fit_batches,
            expected_split="fit",
            hard_action_weight=1.0,
            distribution_weight=0.1,
            temperature=1.0,
            used_for_selection=False,
        )
        reloaded_calibration = evaluate_functional_action_batches(
            runtime,
            calibration_batches,
            expected_split="calibration-selection",
            used_for_selection=True,
        )
        reloaded_heldout = evaluate_functional_action_batches(
            runtime,
            heldout_locked_batches,
            expected_split="heldout-evaluation",
            used_for_selection=False,
        )
        reloaded_fit_outputs = single_runner._action_output_manifest(
            runtime, fit_batches
        )
        reloaded_calibration_outputs = single_runner._action_output_manifest(
            runtime, calibration_batches
        )
        reloaded_heldout_outputs = single_runner._action_output_manifest(
            runtime, heldout_locked_batches
        )
        reloaded_greedy, _ = o_runner._collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        if (
            reloaded_sha != adapter_state_sha256(winner_state)
            or _discrete_evaluation_contract(reloaded_fit)
            != _discrete_evaluation_contract(locked_fit)
            or _discrete_evaluation_contract(reloaded_calibration)
            != _discrete_evaluation_contract(winner_training.final_calibration)
            or _discrete_evaluation_contract(reloaded_heldout)
            != _discrete_evaluation_contract(locked_heldout)
            or reloaded_fit_outputs != fit_output_manifest
            or reloaded_calibration_outputs != calibration_output_manifest
            or reloaded_heldout_outputs != heldout_output_manifest
            or o_runner._trace_manifest(reloaded_greedy)
            != o_runner._trace_manifest(locked_heldout_greedy)
        ):
            raise QwenP35MultisiteRaggedQKVOError(
                "Reloaded factor bundle changed fit/calibration/heldout behavior."
            )
        active_handle.rollback()
        active_handle = None
        parent_restored = o_runner._restore_parent_exact(
            runtime, parent_installation, parent_snapshot, parent_identity
        )
        if not parent_restored:
            raise QwenP35MultisiteRaggedQKVOError(
                "Final multi-site transaction did not restore p35."
            )
    finally:
        primary = sys.exc_info()[1]
        cleanup_error: Exception | None = None
        if active_handle is not None and active_handle.is_applied:
            try:
                active_handle.rollback()
            except Exception as error:  # pragma: no cover
                cleanup_error = error
        try:
            parent_restored = o_runner._restore_parent_exact(
                runtime, parent_installation, parent_snapshot, parent_identity
            )
            if not parent_restored and cleanup_error is None:
                cleanup_error = QwenP35MultisiteRaggedQKVOError(
                    "Cleanup could not restore p35."
                )
        except Exception as error:  # pragma: no cover
            cleanup_error = cleanup_error or error
        if (primary is not None or cleanup_error is not None) and bundle_written:
            if bundle_manifest is None or not _remove_owned_factor_bundle(
                bundle_output, bundle_manifest
            ):
                cleanup_error = cleanup_error or QwenP35MultisiteRaggedQKVOError(
                    "Could not prove ownership and remove the failed factor bundle."
                )
        if cleanup_error is not None:
            if primary is not None:
                raise QwenP35MultisiteRaggedQKVOError(
                    f"primary={primary}; cleanup={cleanup_error}"
                ) from cleanup_error
            raise cleanup_error

    winner_accounting = RaggedAssignmentLocalQKVOAccounting.build(
        site_ranks=schedules[winner_name],
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        logical_layers=36,
        physical_groups=8,
        checkpoint_element_size=2,
        role_scale_table_new_for_redirects=True,
    )
    first_heldout = next(value for value in events if value["heldout_forward"])
    lock_event = next(
        value for value in events if value["event"] == "winner-schedule-locked"
    )
    if first_heldout["index"] <= lock_event["index"]:
        raise QwenP35MultisiteRaggedQKVOError(
            "Heldout output was accessed before schedule lock."
        )
    report = {
        "schema": RUN_SCHEMA,
        "redirects": [value.as_dict() for value in P35_TO_P34_REDIRECTS],
        "redirected_sites": [value.as_dict() for value in sites],
        "lineage": lineage,
        "prompt_fixture_provenance": dict(split_provenance),
        "prompt_split_audit": token_audit,
        "rank_schedule_candidates": candidate_rows,
        "fit_baseline": baseline_fit.as_dict(),
        "calibration_baseline": baseline_calibration.as_dict(),
        "winner": {
            "schedule": winner_name,
            "site_ranks": list(schedules[winner_name]),
            "adapter_state_sha256": adapter_state_sha256(winner_state),
            "training": winner_training.as_dict(),
            "fit_evaluation": locked_fit.as_dict(),
            "fit_gate_passed": True,
            "selected_by": ("fit-gated-calibration-top1-rank-then-fewer-parameters"),
            "heldout_used_for_selection": False,
        },
        "heldout": {
            "baseline_action_evaluation": baseline_heldout.as_dict(),
            "winner_action_evaluation": locked_heldout.as_dict(),
            "baseline_greedy_rollout": baseline_rollout,
            "winner_greedy_rollout": locked_rollout,
            "strict_gate_passed": True,
            "used_for_selection": False,
        },
        "factor_bundle": bundle_manifest,
        "execution_provenance": {
            "source_contract": source_contract,
            "implementation": implementation_provenance,
            "environment": dict(execution_environment),
        },
        "reload_verification": {
            "selected_registry_state_sha256": reloaded_sha,
            "fit_discrete_contract_identical": True,
            "calibration_discrete_contract_identical": True,
            "heldout_discrete_contract_identical": True,
            "fit_outputs_identical": True,
            "calibration_outputs_identical": True,
            "heldout_outputs_identical": True,
            "heldout_greedy_traces_identical": True,
            "continuous_reduction_diagnostics": {
                "fit_before_pack": _continuous_evaluation_diagnostics(locked_fit),
                "fit_after_reload": _continuous_evaluation_diagnostics(reloaded_fit),
                "calibration_before_pack": _continuous_evaluation_diagnostics(
                    winner_training.final_calibration
                ),
                "calibration_after_reload": _continuous_evaluation_diagnostics(
                    reloaded_calibration
                ),
                "heldout_before_pack": _continuous_evaluation_diagnostics(
                    locked_heldout
                ),
                "heldout_after_reload": _continuous_evaluation_diagnostics(
                    reloaded_heldout
                ),
                "used_for_acceptance": False,
                "reason": (
                    "Repeated GPU reductions can differ while exact raw action "
                    "logits, discrete decisions, state, and greedy traces match."
                ),
            },
        },
        "accounting": winner_accounting.as_dict(),
        "events": events,
        "proposal_candidate_count": len(proposal_candidates),
        "used_redirect_projection_count": len(used_projections),
        "transaction": {
            "all_candidates_adapter_only": True,
            "winner_bundle_only": True,
            "optimizer_saved": False,
            "parent_tensors_saved": False,
            "base_model_tensors_saved": False,
            "full_model_checkpoint_saved": False,
            "p35_restored_exactly": parent_restored,
            "candidate_installed_after_return": False,
        },
        "scientific_status": {
            "small_disjoint_gate_passed": True,
            "broad_current_contract_gate_run": False,
            "compaction_certified": False,
            "reason": "factor bundle still requires independent broad evaluation",
        },
    }
    report["report_sha256"] = base._sha256(report)
    return report


def _exclusive_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise QwenP35MultisiteRaggedQKVOError(
                "Multi-site report already exists."
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _remove_owned_factor_bundle(directory: Path, manifest: Mapping[str, Any]) -> bool:
    """Remove only the complete, byte-verified flat bundle created by this run."""

    resolved = directory.resolve()
    if not resolved.exists():
        return True
    if not resolved.is_dir():
        return False
    rows = manifest.get("sites")
    if not isinstance(rows, list):
        return False
    expected_hashes: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        for name_key, hash_key in (
            ("factor_file", "factor_file_sha256"),
            ("manifest_file", "manifest_file_sha256"),
        ):
            name = row.get(name_key)
            digest = row.get(hash_key)
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(digest, str)
                or len(digest) != 64
                or name in expected_hashes
            ):
                return False
            expected_hashes[name] = digest
    bundle_manifest_path = resolved / "bundle.manifest.json"
    expected_names = set(expected_hashes) | {bundle_manifest_path.name}
    try:
        entries = tuple(resolved.iterdir())
        if (
            any(not path.is_file() for path in entries)
            or {path.name for path in entries} != expected_names
            or json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
            != dict(manifest)
            or any(
                file_sha256(resolved / name) != digest
                for name, digest in expected_hashes.items()
            )
        ):
            return False
        for name in sorted(expected_names):
            (resolved / name).unlink()
        resolved.rmdir()
        return True
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _preflight_output_paths(bundle_output: Path, report_output: Path) -> None:
    bundle = bundle_output.resolve()
    report = report_output.resolve()
    if (
        bundle.exists()
        or report.exists()
        or not bundle.parent.is_dir()
        or not report.parent.is_dir()
        or bundle == report
        or bundle in report.parents
        or report in bundle.parents
    ):
        raise QwenP35MultisiteRaggedQKVOError(
            "Bundle/report must be absent, disjoint, and under existing parents."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--anchor-factor-pack", type=Path, required=True)
    parser.add_argument("--anchor-factor-manifest", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fit-prompts-json", type=Path, default=single_runner.DEFAULT_FIT_PROMPTS
    )
    parser.add_argument(
        "--calibration-selection-prompts-json",
        type=Path,
        default=single_runner.DEFAULT_CALIBRATION_PROMPTS,
    )
    parser.add_argument(
        "--heldout-evaluation-prompts-json",
        type=Path,
        default=single_runner.DEFAULT_HELDOUT_PROMPTS,
    )
    parser.add_argument(
        "--fit-prompt-manifest",
        type=Path,
        default=single_runner.DEFAULT_FIT_MANIFEST,
    )
    parser.add_argument(
        "--audited-prompt-manifest",
        type=Path,
        default=single_runner.DEFAULT_AUDITED_MANIFEST,
    )
    parser.add_argument(
        "--transition-inventory",
        type=Path,
        default=single_runner.DEFAULT_TRANSITION_INVENTORY,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-vocabulary", type=int, default=200_000)
    parser.add_argument("--max-logit-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(model=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _preflight_output_paths(args.bundle_output, args.output)
    fixtures = single_runner.load_verified_prompt_splits(
        args.fit_prompts_json,
        args.calibration_selection_prompts_json,
        args.heldout_evaluation_prompts_json,
        fit_manifest_path=args.fit_prompt_manifest,
        audited_manifest_path=args.audited_prompt_manifest,
        transition_inventory_path=args.transition_inventory,
    )
    runtime, tokenizer, source = base._load_runtime_and_tokenizer(args)
    if source.get("manifest_sha256") != replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256:
        raise QwenP35MultisiteRaggedQKVOError(
            "Loaded checkpoint is not the exact successful p35 base artifact."
        )
    exact_environment = replay_runner.assert_exact_execution_environment(
        runtime, device=args.device, dtype=args.dtype
    )
    anchor = load_qkvo_factor_pack(args.anchor_factor_pack, args.anchor_factor_manifest)
    fit = single_runner._encode_prompts(
        tokenizer, fixtures.fit, max_prompt_tokens=args.max_prompt_tokens
    )
    calibration = single_runner._encode_prompts(
        tokenizer,
        fixtures.calibration,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    heldout = single_runner._encode_prompts(
        tokenizer, fixtures.heldout, max_prompt_tokens=args.max_prompt_tokens
    )
    limits = FunctionalDecodeLimits(
        max_horizon=4,
        max_prompt_tokens=args.max_prompt_tokens,
        max_vocabulary=args.max_vocabulary,
        max_logit_bytes=args.max_logit_bytes,
        max_model_devices=1,
    )
    report = run_qwen_p35_multisite_ragged_qkvo(
        runtime,
        fit,
        calibration,
        heldout,
        split_provenance=fixtures.provenance,
        anchor_pack=anchor,
        source_runtime=source,
        execution_environment=exact_environment,
        bundle_output=args.bundle_output,
        trace_limits=limits,
    )
    try:
        report["source"] = source
        report["parameter_report_after_restore"] = qwen3_codebook_parameter_report(
            runtime
        ).as_dict()
        report["environment"] = exact_environment
        report["report_sha256"] = base._sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
        _exclusive_report(args.output.resolve(), report)
    except Exception as primary:
        manifest = report.get("factor_bundle")
        if not isinstance(manifest, Mapping) or not _remove_owned_factor_bundle(
            args.bundle_output, manifest
        ):
            raise QwenP35MultisiteRaggedQKVOError(
                "Report write failed and the owned factor bundle could not be removed; "
                f"primary={primary}"
            ) from primary
        raise
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bundle": str(args.bundle_output.resolve()),
                "winner_schedule": report["winner"]["schedule"],
                "net_parameters_saved": report["accounting"]["net_parameters_saved"],
                "p35_restored": True,
                "full_checkpoint_saved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
