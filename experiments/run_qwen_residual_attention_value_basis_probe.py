#!/usr/bin/env python3
"""Run the exact p35 residual-targeted V-basis -> O four-control probe.

This runner is deliberately separate from the raw parent-V-cache probe.  It
hard-wires the physical prototype redirect ``L19/G4 -> L21/G6``, captures the
survivor-174 and removed-156 V directions before compaction, and fits their
bounded linear combination directly against isolated parent residual writes
under the tentative child's fixed Q/K attention and current O.

Four controls use identical disjoint calibration/held-out prompts, parent
drivers, cached forced and greedy lanes, horizons, and numeric tolerances:

1. uncompensated redirect;
2. block-sparse O-only;
3. residual-targeted basis-V-only;
4. basis V installed, actual child traces recollected, then block-sparse O
   refitted from the new ``A @ V`` contexts.

All mutations are in-memory and nested behind exact rollback audits.  This
runner never saves or commits a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_residual_attention_probe as base
import run_qwen_residual_attention_value_probe as raw_value
from neural_forge.hf_cached_attention_trace import (
    HFCachedAttentionTrace,
    HFCachedAttentionTraceLimits,
)
from neural_forge.hf_qwen_attention_surgery import qwen3_attention_runtime_snapshot
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report
from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionGatePolicy,
    ResidualAttentionScoringConfig,
)
from neural_forge.hf_residual_attention_functional import (
    build_residual_attention_functional_comparison,
)
from neural_forge.hf_residual_attention_value_basis import (
    RESIDUAL_VALUE_BASIS_INTEGRATION_CONTRACT,
    ParentOutputBlockBasisEntry,
    ResidualTargetedValueBasisFit,
    ValuePrototypeBasisEntry,
    build_residual_value_basis_site_calibration,
    capture_parent_output_block_basis_entry,
    capture_value_prototype_basis_entry,
    fit_residual_targeted_value_basis,
    install_residual_targeted_value_basis,
)
from neural_forge.hf_residual_attention_value_fit import InstalledValuePrototype


RUN_SCHEMA = "neural-forge-qwen-p35-residual-targeted-value-basis-probe-v3"
EXACT_REDIRECT = base.AttentionRedirect(19, 4, 21, 6)
EXACT_REMOVED_SOURCE = 19 * 8 + 4
EXACT_SURVIVOR_SOURCE = 21 * 8 + 6
EXACT_AFFECTED_SITES = ((2, 1), (19, 4), (21, 6))
SURVIVOR_BASIS_ID = "survivor-source-174-L21G6"
REMOVED_BASIS_ID = "removed-source-156-L19G4"
BASIS_COEFFICIENT_POLICY: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "prior": MappingProxyType({SURVIVOR_BASIS_ID: 1.0, REMOVED_BASIS_ID: 0.0}),
        "lower": MappingProxyType({SURVIVOR_BASIS_ID: 0.5, REMOVED_BASIS_ID: -0.5}),
        "upper": MappingProxyType({SURVIVOR_BASIS_ID: 1.5, REMOVED_BASIS_ID: 0.5}),
    }
)
SplitTraces = Mapping[base.TraceKey, HFCachedAttentionTrace]


class QwenResidualAttentionValueBasisProbeError(RuntimeError):
    """The exact basis-V/O probe failed an evidence or rollback contract."""


def _positive_ridge(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QwenResidualAttentionValueBasisProbeError(
            f"{name} must be a finite positive number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise QwenResidualAttentionValueBasisProbeError(
            f"{name} must be a finite positive number."
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise QwenResidualAttentionValueBasisProbeError(
            f"{name} must be a finite positive number."
        )
    return result


def _basis_ids_for_redirect(
    redirect: base.AttentionRedirect,
    groups: int,
    *,
    require_exact_p35: bool,
) -> tuple[int, str, int, str]:
    removed_source = redirect.source_layer * groups + redirect.source_group
    survivor_source = redirect.target_layer * groups + redirect.target_group
    if require_exact_p35:
        if (
            removed_source != EXACT_REMOVED_SOURCE
            or survivor_source != EXACT_SURVIVOR_SOURCE
        ):
            raise QwenResidualAttentionValueBasisProbeError(
                "Exact p35 basis provenance must be removed-156 and survivor-174."
            )
        return (
            survivor_source,
            SURVIVOR_BASIS_ID,
            removed_source,
            REMOVED_BASIS_ID,
        )
    return (
        survivor_source,
        f"survivor-source-{survivor_source}",
        removed_source,
        f"removed-source-{removed_source}",
    )


def _capture_redirect_basis(
    parent_registry: Any,
    redirect: base.AttentionRedirect,
    *,
    require_exact_p35: bool,
) -> tuple[ValuePrototypeBasisEntry, ValuePrototypeBasisEntry]:
    groups = int(parent_registry.key_value_heads)
    survivor_source, survivor_id, removed_source, removed_id = _basis_ids_for_redirect(
        redirect,
        groups,
        require_exact_p35=require_exact_p35,
    )
    survivor = capture_value_prototype_basis_entry(
        parent_registry,
        source_flat_index=survivor_source,
        basis_id=survivor_id,
        role="survivor",
    )
    removed = capture_value_prototype_basis_entry(
        parent_registry,
        source_flat_index=removed_source,
        basis_id=removed_id,
        role="removed-source",
    )
    return survivor, removed


def _coefficient_policy(
    basis: Sequence[ValuePrototypeBasisEntry],
    *,
    require_exact_p35: bool,
) -> dict[str, dict[str, float]]:
    survivor = next(value for value in basis if value.role == "survivor")
    removed = next(value for value in basis if value.role == "removed-source")
    if require_exact_p35:
        return {name: dict(values) for name, values in BASIS_COEFFICIENT_POLICY.items()}
    return {
        "prior": {survivor.basis_id: 1.0, removed.basis_id: 0.0},
        "lower": {survivor.basis_id: 0.5, removed.basis_id: -0.5},
        "upper": {survivor.basis_id: 1.5, removed.basis_id: 0.5},
    }


def _fit_basis_survivor(
    parent_calibration: SplitTraces,
    candidate_calibration: SplitTraces,
    parent_o_captures: Mapping[base.Site, ParentOutputBlockBasisEntry],
    proposed_registry: Any,
    sites_by_source: Mapping[int, tuple[base.Site, ...]],
    basis: Sequence[ValuePrototypeBasisEntry],
    *,
    prompt_count: int,
    basis_ridge: float,
    require_exact_p35: bool,
) -> tuple[int, ResidualTargetedValueBasisFit, dict[str, dict[str, float]]]:
    if len(sites_by_source) != 1:
        raise QwenResidualAttentionValueBasisProbeError(
            "The exact runner supports one survivor assignment set."
        )
    survivor_source, sites = next(iter(sites_by_source.items()))
    survivor_entry = next(value for value in basis if value.role == "survivor")
    if survivor_source != survivor_entry.source_flat_index:
        raise QwenResidualAttentionValueBasisProbeError(
            "Tentative survivor provenance differs from the captured basis."
        )
    bindings = base._materialized_site_bindings(proposed_registry)
    active_slots = {bindings[site].active_slot for site in sites}
    if len(active_slots) != 1:
        raise QwenResidualAttentionValueBasisProbeError(
            "One survivor provenance maps to multiple active V slots."
        )
    active_slot = next(iter(active_slots))
    rectangular = proposed_registry.prototype_v_weight.ndim == 4
    assignment_axes = {bindings[site].assignment_axis for site in sites}
    if rectangular:
        if len(assignment_axes) != 1:
            raise QwenResidualAttentionValueBasisProbeError(
                "A rectangular survivor cannot span canonical groups."
            )
        survivor_group = next(iter(assignment_axes))
        slots_per_group = int(proposed_registry.prototype_v_weight.shape[1])
        survivor_slot = active_slot - survivor_group * slots_per_group
    else:
        survivor_group = None
        survivor_slot = active_slot
    calibrations = []
    for layer_index, materialized_group in sites:
        canonical_group = bindings[(layer_index, materialized_group)].assignment_axis
        parent_o = parent_o_captures[(layer_index, materialized_group)]
        for prompt_index in range(prompt_count):
            key = ("forced", prompt_index, layer_index, materialized_group)
            calibrations.append(
                build_residual_value_basis_site_calibration(
                    candidate_calibration[key],
                    parent_calibration[key],
                    proposed_registry,
                    basis,
                    parent_o,
                    canonical_group=canonical_group,
                )
            )
    policy = _coefficient_policy(
        basis,
        require_exact_p35=require_exact_p35,
    )
    fit = fit_residual_targeted_value_basis(
        proposed_registry,
        calibrations,
        basis,
        survivor_basis_id=survivor_entry.basis_id,
        survivor_group=survivor_group,
        survivor_slot=survivor_slot,
        prior_coefficients=policy["prior"],
        lower_bounds=policy["lower"],
        upper_bounds=policy["upper"],
        ridge=basis_ridge,
    )
    return survivor_source, fit, policy


def _validate_prompt_splits(
    calibration: Sequence[Tensor],
    heldout: Sequence[Tensor],
    limits: HFCachedAttentionTraceLimits,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    try:
        return raw_value._validate_prompt_splits(calibration, heldout, limits)
    except raw_value.QwenResidualAttentionValueProbeError as error:
        raise QwenResidualAttentionValueBasisProbeError(str(error)) from error


def run_qwen_p35_residual_value_basis_probe(
    runtime: nn.Module,
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    *,
    redirect: base.AttentionRedirect = EXACT_REDIRECT,
    require_exact_p35: bool = True,
    horizon: int = 8,
    basis_ridge: float = 1e-4,
    o_ridge: float = 1e-6,
    trace_limits: HFCachedAttentionTraceLimits | None = None,
    max_affected_sites: int = 16,
    scoring: ResidualAttentionScoringConfig | None = None,
    gate_policy: ResidualAttentionGatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate the four residual-basis controls and restore p35 exactly."""

    limits = trace_limits or HFCachedAttentionTraceLimits()
    calibration_prompts, heldout_prompts = _validate_prompt_splits(
        calibration_prompt_input_ids,
        heldout_prompt_input_ids,
        limits,
    )
    basis_ridge = _positive_ridge(basis_ridge, "basis_ridge")
    o_ridge = _positive_ridge(o_ridge, "o_ridge")
    if require_exact_p35 and redirect != EXACT_REDIRECT:
        raise QwenResidualAttentionValueBasisProbeError(
            "The executable basis probe is fixed to L19/G4 -> L21/G6."
        )
    scoring_default, policy_default = raw_value._default_scoring()
    scoring_value = scoring or scoring_default
    policy_value = gate_policy or policy_default
    max_sites = base._positive_int(max_affected_sites, "max_affected_sites")
    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    if require_exact_p35 and (
        parent_snapshot.prototype_count != 280
        or parent_snapshot.key_value_heads != 8
        or parent_snapshot.logical_layers != 36
    ):
        raise QwenResidualAttentionValueBasisProbeError(
            "Exact mode requires the 36-layer, 8-group, 280-prototype p35 frontier."
        )
    basis = _capture_redirect_basis(
        parent_registry,
        redirect,
        require_exact_p35=require_exact_p35,
    )
    parent_installation = attention_surgery._capture_installed(runtime)
    _bridge, _proposal, candidates, tentative = base._tentative_codebook(
        runtime, (redirect,)
    )
    codebook, used_projections = tentative
    affected_sites, sites_by_source = base._affected_sites(
        parent_registry,
        codebook.registry,
        (redirect,),
    )
    raw_value._assert_block_o_layout(sites_by_source)
    if require_exact_p35 and (
        affected_sites != EXACT_AFFECTED_SITES
        or tuple(sites_by_source) != (EXACT_SURVIVOR_SOURCE,)
    ):
        raise QwenResidualAttentionValueBasisProbeError(
            "The p35 lineage does not expose exactly L2/G1, L19/G4, and L21/G6 "
            "for survivor-174."
        )
    if len(affected_sites) > max_sites:
        raise QwenResidualAttentionValueBasisProbeError(
            f"Affected sites exceed max_affected_sites={max_sites}."
        )
    if len({layer for layer, _group in affected_sites}) > limits.max_layers:
        raise QwenResidualAttentionValueBasisProbeError(
            "Affected layers exceed the cached-trace layer bound."
        )
    parent_bindings_for_capture = base._materialized_site_bindings(parent_registry)
    parent_o_captures = {
        (layer, physical): capture_parent_output_block_basis_entry(
            parent_registry,
            layer_index=layer,
            canonical_group=parent_bindings_for_capture[
                (layer, physical)
            ].assignment_axis,
        )
        for layer, physical in affected_sites
    }
    if any(
        capture.physical_group != physical
        for (_layer, physical), capture in parent_o_captures.items()
    ):
        raise QwenResidualAttentionValueBasisProbeError(
            "Captured parent O block physical coordinates disagree with the sites."
        )
    parent_o_blocks = {
        site: torch.from_numpy(np.array(capture.block, copy=True)).to(torch.float64)
        for site, capture in parent_o_captures.items()
    }
    candidate_ids = (redirect.candidate_id,)
    trace_state_bindings: dict[str, Any] = {}
    parent_calibration_before = raw_value._trace_state_binding(runtime)
    parent_calibration, calibration_drivers = base._collect_parent_traces(
        runtime,
        calibration_prompts,
        affected_sites,
        horizon=horizon,
        limits=limits,
    )
    trace_state_bindings["parent"] = {
        "calibration": raw_value._finish_trace_state_binding(
            runtime,
            parent_calibration_before,
            label="parent:calibration",
        )
    }
    parent_heldout_before = raw_value._trace_state_binding(runtime)
    parent_heldout, heldout_drivers = base._collect_parent_traces(
        runtime,
        heldout_prompts,
        affected_sites,
        horizon=horizon,
        limits=limits,
    )
    trace_state_bindings["parent"]["heldout"] = raw_value._finish_trace_state_binding(
        runtime,
        parent_heldout_before,
        label="parent:heldout",
    )

    variants: dict[str, Any] = {}
    functional_calibration_traces: dict[str, SplitTraces] = {}
    functional_heldout_traces: dict[str, SplitTraces] = {}
    trace_manifests: dict[str, Any] = {}
    o_only_fits: dict[int, dict[str, Any]] = {}
    post_basis_o_fits: dict[int, dict[str, Any]] = {}
    basis_fit: ResidualTargetedValueBasisFit | None = None
    basis_source: int | None = None
    coefficient_policy: dict[str, dict[str, float]] | None = None
    basis_installation: InstalledValuePrototype | None = None
    basis_trace_audit: dict[str, Any] | None = None
    proposed_snapshot = None
    proposed_reset_exact = False
    incumbent_rollback_exact = False
    o_backups: dict[int, Tensor] = {}
    proposed_o_sha256: str | None = None
    try:
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        proposed_snapshot = qwen3_attention_runtime_snapshot(runtime)
        (
            baseline_calibration,
            baseline_heldout,
            trace_state_bindings["uncompensated"],
        ) = raw_value._collect_both_splits(
            runtime,
            calibration_prompts,
            heldout_prompts,
            affected_sites,
            calibration_drivers,
            heldout_drivers,
            label="uncompensated",
            horizon=horizon,
            limits=limits,
        )
        variants["uncompensated"] = raw_value._split_variant_report(
            parent_calibration,
            parent_heldout,
            baseline_calibration,
            baseline_heldout,
            candidate_ids=candidate_ids,
            scoring=scoring_value,
            policy=policy_value,
        )
        functional_calibration_traces["uncompensated"] = baseline_calibration
        functional_heldout_traces["uncompensated"] = baseline_heldout
        trace_manifests["uncompensated"] = {
            "calibration": raw_value._trace_manifest(baseline_calibration),
            "heldout": raw_value._trace_manifest(baseline_heldout),
        }

        # Independent O-only control on the unchanged tentative V.
        o_only_fits = base._fit_survivor_blocks(
            parent_calibration,
            baseline_calibration,
            parent_o_blocks,
            codebook.registry,
            sites_by_source,
            prompt_count=len(calibration_prompts),
            ridge=o_ridge,
        )
        flat_o = base._flat_o(codebook.registry)
        target_o_slots = {int(value["active_slot"]) for value in o_only_fits.values()}
        o_backups = {
            slot: flat_o[slot].detach().clone() for slot in sorted(target_o_slots)
        }
        proposed_o_sha256 = base._o_slots_sha256(flat_o)
        o_only_audit = base._apply_compensation(
            codebook.registry,
            o_only_fits,
            kind="ridge",
        )
        o_calibration, o_heldout, trace_state_bindings["o_only"] = (
            raw_value._collect_both_splits(
                runtime,
                calibration_prompts,
                heldout_prompts,
                affected_sites,
                calibration_drivers,
                heldout_drivers,
                label="o_only",
                horizon=horizon,
                limits=limits,
            )
        )
        variants["o_only"] = {
            **raw_value._split_variant_report(
                parent_calibration,
                parent_heldout,
                o_calibration,
                o_heldout,
                candidate_ids=candidate_ids,
                scoring=scoring_value,
                policy=policy_value,
            ),
            "edit_audit": o_only_audit,
        }
        functional_calibration_traces["o_only"] = o_calibration
        functional_heldout_traces["o_only"] = o_heldout
        trace_manifests["o_only"] = {
            "calibration": raw_value._trace_manifest(o_calibration),
            "heldout": raw_value._trace_manifest(o_heldout),
        }
        raw_value._restore_o_exact(codebook.registry, o_backups, proposed_o_sha256)

        # Residual-targeted basis V sees only forced calibration traces.
        basis_source, basis_fit, coefficient_policy = _fit_basis_survivor(
            parent_calibration,
            baseline_calibration,
            parent_o_captures,
            codebook.registry,
            sites_by_source,
            basis,
            prompt_count=len(calibration_prompts),
            basis_ridge=basis_ridge,
            require_exact_p35=require_exact_p35,
        )
        basis_installation = install_residual_targeted_value_basis(
            codebook.registry,
            basis_fit,
        )
        (
            basis_v_calibration,
            basis_v_heldout,
            trace_state_bindings["basis_v_only"],
        ) = raw_value._collect_both_splits(
            runtime,
            calibration_prompts,
            heldout_prompts,
            affected_sites,
            calibration_drivers,
            heldout_drivers,
            label="basis_v_only",
            horizon=horizon,
            limits=limits,
        )
        basis_trace_audit = raw_value._post_value_trace_audit(
            codebook.registry,
            (basis_v_calibration, basis_v_heldout),
            affected_sites,
        )
        variants["basis_v_only"] = raw_value._split_variant_report(
            parent_calibration,
            parent_heldout,
            basis_v_calibration,
            basis_v_heldout,
            candidate_ids=candidate_ids,
            scoring=scoring_value,
            policy=policy_value,
        )
        functional_calibration_traces["basis_v_only"] = basis_v_calibration
        functional_heldout_traces["basis_v_only"] = basis_v_heldout
        trace_manifests["basis_v_only"] = {
            "calibration": raw_value._trace_manifest(basis_v_calibration),
            "heldout": raw_value._trace_manifest(basis_v_heldout),
        }

        # Revalidate V, discard the stale O-only fit, and solve O from newly
        # recollected actual child A@V contexts.
        basis_installation.assert_live()
        post_basis_o_fits = base._fit_survivor_blocks(
            parent_calibration,
            basis_v_calibration,
            parent_o_blocks,
            codebook.registry,
            sites_by_source,
            prompt_count=len(calibration_prompts),
            ridge=o_ridge,
        )
        if {
            int(value["active_slot"]) for value in post_basis_o_fits.values()
        } != target_o_slots:
            raise QwenResidualAttentionValueBasisProbeError(
                "Post-basis-V O targets differ from the O-only control slots."
            )
        basis_vo_audit = base._apply_compensation(
            codebook.registry,
            post_basis_o_fits,
            kind="ridge",
        )
        (
            basis_vo_calibration,
            basis_vo_heldout,
            trace_state_bindings["basis_v_then_o"],
        ) = raw_value._collect_both_splits(
            runtime,
            calibration_prompts,
            heldout_prompts,
            affected_sites,
            calibration_drivers,
            heldout_drivers,
            label="basis_v_then_o",
            horizon=horizon,
            limits=limits,
        )
        variants["basis_v_then_o"] = {
            **raw_value._split_variant_report(
                parent_calibration,
                parent_heldout,
                basis_vo_calibration,
                basis_vo_heldout,
                candidate_ids=candidate_ids,
                scoring=scoring_value,
                policy=policy_value,
            ),
            "edit_audit": basis_vo_audit,
        }
        functional_calibration_traces["basis_v_then_o"] = basis_vo_calibration
        functional_heldout_traces["basis_v_then_o"] = basis_vo_heldout
        trace_manifests["basis_v_then_o"] = {
            "calibration": raw_value._trace_manifest(basis_vo_calibration),
            "heldout": raw_value._trace_manifest(basis_vo_heldout),
        }
        basis_trace_audit = raw_value._post_value_trace_audit(
            codebook.registry,
            (
                basis_v_calibration,
                basis_v_heldout,
                basis_vo_calibration,
                basis_vo_heldout,
            ),
            affected_sites,
        )
        raw_value._restore_o_exact(codebook.registry, o_backups, proposed_o_sha256)
        basis_installation.rollback()
        proposed_reset_exact = (
            qwen3_attention_runtime_snapshot(runtime) == proposed_snapshot
        )
        if not proposed_reset_exact:
            raise QwenResidualAttentionValueBasisProbeError(
                "Basis V/O controls did not restore the tentative redirect exactly."
            )
    finally:
        failure_in_flight = sys.exc_info()[0] is not None
        restoration_errors: list[Exception] = []
        # O must roll back before V because the V handle certifies all non-V
        # registry state during its own rollback.
        if proposed_o_sha256 is not None and o_backups:
            try:
                raw_value._restore_o_exact(
                    codebook.registry,
                    o_backups,
                    proposed_o_sha256,
                )
            except Exception as error:  # pragma: no cover - adversarial cleanup
                restoration_errors.append(error)
        if (
            basis_installation is not None
            and basis_installation.is_applied
            and not basis_installation.is_committed
        ):
            try:
                basis_installation.rollback()
            except Exception as error:  # pragma: no cover - adversarial cleanup
                restoration_errors.append(error)
        try:
            restored = attention_surgery._restore_installed(
                runtime,
                parent_installation,
            )
            incumbent_rollback_exact = bool(
                restored
                and qwen3_attention_runtime_snapshot(runtime) == parent_snapshot
            )
            if not incumbent_rollback_exact:
                restoration_errors.append(
                    QwenResidualAttentionValueBasisProbeError(
                        "Probe did not restore the incumbent p35 registry exactly."
                    )
                )
        except Exception as error:  # pragma: no cover - adversarial cleanup
            restoration_errors.append(error)
        if restoration_errors and not failure_in_flight:
            raise QwenResidualAttentionValueBasisProbeError(
                "Probe cleanup failed after every rollback layer: "
                + "; ".join(str(error) for error in restoration_errors)
            ) from restoration_errors[0]

    if (
        proposed_snapshot is None
        or basis_fit is None
        or basis_installation is None
        or basis_source is None
        or coefficient_policy is None
        or basis_trace_audit is None
    ):
        raise QwenResidualAttentionValueBasisProbeError(
            "Probe completed without complete basis fit/trace evidence."
        )
    functional_comparison = build_residual_attention_functional_comparison(
        parent_calibration,
        parent_heldout,
        functional_calibration_traces,
        functional_heldout_traces,
        top_k=scoring_value.top_k,
    )
    functional_variant_reports = functional_comparison["variants"]
    for variant_id, functional_report in functional_variant_reports.items():
        variants[variant_id]["functional"] = functional_report
    parent_bindings = base._materialized_site_bindings(parent_registry)
    proposed_bindings = base._materialized_site_bindings(codebook.registry)
    affected_payload = [
        {
            "layer_index": layer,
            "materialized_group": group,
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
    result = {
        "schema": RUN_SCHEMA,
        "redirect": redirect.as_dict(),
        "exact_p35_mode": require_exact_p35,
        "candidate_ids": list(candidate_ids),
        "cached_trace_validation_mode": limits.cached_trace_validation_mode,
        "horizon": horizon,
        "basis_ridge": basis_ridge,
        "o_ridge": o_ridge,
        "prompt_splits": {
            "calibration": {
                "count": len(calibration_prompts),
                "prompt_sha256s": list(raw_value._prompt_sha256s(calibration_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(calibration_drivers.items())
                },
                "used_by_basis_v_fit": True,
                "used_by_o_fits": True,
                "used_by_functional_selection": True,
            },
            "heldout": {
                "count": len(heldout_prompts),
                "prompt_sha256s": list(raw_value._prompt_sha256s(heldout_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(heldout_drivers.items())
                },
                "used_by_basis_v_fit": False,
                "used_by_o_fits": False,
                "used_by_functional_selection": False,
                "used_by_functional_evaluation": True,
            },
            "token_rows_disjoint": True,
        },
        "parent_attention_snapshot": parent_snapshot.as_dict(),
        "proposed_attention_snapshot": proposed_snapshot.as_dict(),
        "accounting": {
            "parent_prototype_count": parent_snapshot.prototype_count,
            "proposed_prototype_count": proposed_snapshot.prototype_count,
            "prototype_count_removed": (
                parent_snapshot.prototype_count - proposed_snapshot.prototype_count
            ),
            "parent_attention_tensor_bytes": parent_snapshot.tensor_bytes,
            "proposed_attention_tensor_bytes": proposed_snapshot.tensor_bytes,
            "physical_attention_bytes_saved": (
                parent_snapshot.tensor_bytes - proposed_snapshot.tensor_bytes
            ),
            "prototype_pool_bytes_saved": (
                parent_snapshot.prototype_pool_bytes
                - proposed_snapshot.prototype_pool_bytes
            ),
            "index_bytes_delta": (
                proposed_snapshot.index_bytes - parent_snapshot.index_bytes
            ),
            "runtime_parameters_added_by_basis": 0,
            "parameter_count_unchanged_by_basis_v_o_fits": True,
        },
        "basis": {
            "survivor_source_flat_index": basis[0].source_flat_index,
            "removed_source_flat_index": basis[1].source_flat_index,
            "captured_before_tentative_compaction": True,
            "entries": [value.as_dict() for value in basis],
            "parent_o_blocks": {
                f"L{layer}G{physical}": capture.as_dict()
                for (layer, physical), capture in sorted(parent_o_captures.items())
            },
            "coefficient_policy": coefficient_policy,
            "fit_source": basis_source,
        },
        "affected_assignments": affected_payload,
        "affected_assignment_count": len(affected_payload),
        "affected_assignment_sha256": base._sha256(affected_payload),
        "survivor_assignment_sets": {
            str(source): [list(site) for site in sites]
            for source, sites in sites_by_source.items()
        },
        "used_projections": [value.as_dict() for value in used_projections],
        "proposal": {
            "candidate_count": len(candidates),
            "parent_registry_sha256": parent_snapshot.registry_sha256,
            "proposed_registry_sha256": proposed_snapshot.registry_sha256,
        },
        "fits": {
            "o_only": raw_value._fit_payload(o_only_fits),
            "residual_targeted_basis_v": basis_fit.report.as_dict(),
            "basis_v_installation": basis_installation.report.as_dict(),
            "post_basis_v_o": raw_value._fit_payload(post_basis_o_fits),
        },
        "basis_value_trace_audit": basis_trace_audit,
        "variants": variants,
        "functional_comparison": functional_comparison,
        "variant_trace_provenance": trace_manifests,
        "parent_trace_provenance": {
            "calibration": raw_value._trace_manifest(parent_calibration),
            "heldout": raw_value._trace_manifest(parent_heldout),
        },
        "trace_collection_state_bindings": trace_state_bindings,
        "fit_contract": {
            "four_controls": [
                "uncompensated",
                "o_only",
                "basis_v_only",
                "basis_v_then_o",
            ],
            "raw_parent_v_runner_preserved_and_not_called": True,
            "basis_contains_survivor_and_removed_source": True,
            "basis_fit_target_is_parent_isolated_group_residual_write": True,
            "basis_fit_design_uses_candidate_fixed_qk_and_current_o": True,
            "basis_coefficients_bounded_and_anchored": True,
            "basis_materialized_into_existing_survivor_v": True,
            "runtime_basis_coefficients_retained": False,
            "identical_parent_references_and_drivers_across_controls": True,
            "calibration_and_heldout_forced_and_greedy_lanes": True,
            "heldout_excluded_from_all_fits": True,
            "candidate_selection_uses_calibration_functional_outputs_only": True,
            "heldout_functional_outputs_are_evaluation_only": True,
            "heldout_excluded_from_candidate_selection": True,
            "attention_residual_cache_and_weight_similarity_excluded_from_selection": True,
            "all_survivor_assignments_in_basis_fit": True,
            "all_three_exact_p35_shared_sites_verified": require_exact_p35,
            "v_and_o_role_scales_and_group_permutations_honored": True,
            "o_only_design_uses_pre_basis_child_A_at_V": True,
            "post_basis_v_o_fit_recomputed_not_reused": True,
            "basis_v_revalidated_before_post_install_o_fit": True,
            "post_basis_v_o_design_uses_recollected_actual_child_A_at_V": True,
            "one_survivor_group_per_layer_for_block_o_verified": True,
            "attention_registry_content_hash_bound_around_every_trace_collection": True,
            "full_runtime_storage_version_hash_bound_around_every_trace_collection": True,
            "nested_o_then_v_then_incumbent_rollback": True,
            "cross_tokenizer_mapping_supported": False,
            "identical_raw_token_ids_required": True,
            "zero_gradient": True,
            "integration_contract": dict(RESIDUAL_VALUE_BASIS_INTEGRATION_CONTRACT),
        },
        "tentative_state_reset_exact": proposed_reset_exact,
        "incumbent_rollback_exact": incumbent_rollback_exact,
        "checkpoint_saved": False,
        "fold_committed": False,
        "compaction_certified": False,
        "gradient_free": True,
        "training_used": False,
    }
    result["report_sha256"] = base._sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--calibration-prompt", action="append", default=[])
    parser.add_argument("--calibration-prompts-json", type=Path)
    parser.add_argument("--heldout-prompt", action="append", default=[])
    parser.add_argument("--heldout-prompts-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--basis-ridge", type=float, default=1e-4)
    parser.add_argument("--o-ridge", type=float, default=1e-6)
    parser.add_argument("--max-prompts-per-split", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-affected-sites", type=int, default=16)
    parser.add_argument("--max-affected-layers", type=int, default=8)
    parser.add_argument("--max-capture-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-projection-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--logit-rtol", type=float, default=5e-3)
    parser.add_argument("--logit-atol", type=float, default=5e-3)
    parser.add_argument("--context-rtol", type=float, default=8e-3)
    parser.add_argument("--context-atol", type=float, default=8e-3)
    parser.add_argument(
        "--cached-trace-validation-mode",
        choices=("strict", "record"),
        default="strict",
        help=(
            "strict aborts on finite cached-trace reconstruction drift; record "
            "serializes drift/top-1 validity while keeping structure, finiteness, "
            "provenance, and exact state accounting fail-closed"
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(model=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibration_texts = raw_value._load_texts(
        args.calibration_prompt,
        args.calibration_prompts_json,
        "calibration prompt",
    )
    heldout_texts = raw_value._load_texts(
        args.heldout_prompt,
        args.heldout_prompts_json,
        "held-out prompt",
    )
    if max(len(calibration_texts), len(heldout_texts)) > args.max_prompts_per_split:
        raise QwenResidualAttentionValueBasisProbeError(
            "A prompt split exceeds --max-prompts-per-split."
        )
    runtime, tokenizer, source = base._load_runtime_and_tokenizer(args)
    calibration_prompts = raw_value._encode_prompts(
        tokenizer,
        calibration_texts,
        args.max_prompt_tokens,
    )
    heldout_prompts = raw_value._encode_prompts(
        tokenizer,
        heldout_texts,
        args.max_prompt_tokens,
    )
    limits = HFCachedAttentionTraceLimits(
        max_prompts=args.max_prompts_per_split,
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_layers=args.max_affected_layers,
        max_groups=1,
        max_model_devices=1,
        max_capture_bytes=args.max_capture_bytes,
        max_projection_bytes=args.max_projection_bytes,
        logit_rtol=args.logit_rtol,
        logit_atol=args.logit_atol,
        context_rtol=args.context_rtol,
        context_atol=args.context_atol,
        cached_trace_validation_mode=args.cached_trace_validation_mode,
    )
    report = run_qwen_p35_residual_value_basis_probe(
        runtime,
        calibration_prompts,
        heldout_prompts,
        redirect=EXACT_REDIRECT,
        require_exact_p35=True,
        horizon=args.horizon,
        basis_ridge=args.basis_ridge,
        o_ridge=args.o_ridge,
        trace_limits=limits,
        max_affected_sites=args.max_affected_sites,
    )
    report["source"] = source
    report["parameter_report_after_exact_rollback"] = qwen3_codebook_parameter_report(
        runtime
    ).as_dict()
    report["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "dtype": args.dtype,
        "cached_trace_validation_mode": args.cached_trace_validation_mode,
        "numeric_tolerances": {
            "logit_rtol": args.logit_rtol,
            "logit_atol": args.logit_atol,
            "context_rtol": args.context_rtol,
            "context_atol": args.context_atol,
        },
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
                "redirect": EXACT_REDIRECT.candidate_id,
                "variants": sorted(report["variants"]),
                "rollback_exact": report["incumbent_rollback_exact"],
                "checkpoint_saved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
