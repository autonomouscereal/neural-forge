#!/usr/bin/env python3
"""Run the exact p35 L19/G4 -> L21/G6 alternating V/O probe.

This is a bounded, zero-gradient, in-memory experiment.  It starts from the
preserved p35 attention codebook, previews one physical-prototype redirect,
and compares four controls:

1. uncompensated redirect;
2. O-only block ridge fitted on calibration traces;
3. globally fitted V-only;
4. V installed, real child ``A @ V`` recollected, then O refitted.

Calibration and held-out prompt sets are disjoint.  Both sets are evaluated on
forced-parent-token and greedy cached lanes with identical drivers, horizons,
and tolerances across controls.  Every V fit includes all logical sites which
consume the shared survivor.  The incumbent registry is restored exactly, no
checkpoint is written, and the JSON report carries tensor/trace/provenance
hashes and physical byte accounting.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_residual_attention_probe as base
from neural_forge.hf_cached_attention_trace import (
    HFCachedAttentionTrace,
    HFCachedAttentionTraceLimits,
)
from neural_forge.hf_qwen_attention_surgery import (
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report
from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionGatePolicy,
    ResidualAttentionScoringConfig,
)
from neural_forge.hf_residual_attention_functional import (
    build_residual_attention_functional_comparison,
)
from neural_forge.hf_residual_attention_value_fit import (
    InstalledValuePrototype,
    SharedValuePrototypeFit,
    build_value_site_calibration,
    fit_shared_codebook_value_prototype,
    install_shared_codebook_value_prototype,
)


RUN_SCHEMA = "neural-forge-qwen-p35-residual-attention-value-probe-v4"
EXACT_REDIRECT = base.AttentionRedirect(19, 4, 21, 6)
SplitTraces = Mapping[base.TraceKey, HFCachedAttentionTrace]


class QwenResidualAttentionValueProbeError(RuntimeError):
    """The four-control V/O probe failed a shape, isolation, or rollback proof."""


def _prompt_sha256s(prompts: Sequence[Tensor]) -> tuple[str, ...]:
    return tuple(
        base._tensor_sha256(torch.as_tensor(value, dtype=torch.long))
        for value in prompts
    )


def _validate_prompt_splits(
    calibration: Sequence[Tensor],
    heldout: Sequence[Tensor],
    limits: HFCachedAttentionTraceLimits,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    calibration_values = tuple(calibration)
    heldout_values = tuple(heldout)
    if not calibration_values or not heldout_values:
        raise QwenResidualAttentionValueProbeError(
            "Calibration and held-out prompt sets must both be non-empty."
        )
    if max(len(calibration_values), len(heldout_values)) > limits.max_prompts:
        raise QwenResidualAttentionValueProbeError(
            "A prompt split exceeds the trace max_prompts bound."
        )
    calibration_hashes = _prompt_sha256s(calibration_values)
    heldout_hashes = _prompt_sha256s(heldout_values)
    if len(set(calibration_hashes)) != len(calibration_hashes) or len(
        set(heldout_hashes)
    ) != len(heldout_hashes):
        raise QwenResidualAttentionValueProbeError(
            "Each prompt split must contain unique token rows."
        )
    overlap = sorted(set(calibration_hashes) & set(heldout_hashes))
    if overlap:
        raise QwenResidualAttentionValueProbeError(
            "Calibration and held-out token rows must be disjoint."
        )
    return calibration_values, heldout_values


def _default_scoring() -> tuple[
    ResidualAttentionScoringConfig, ResidualAttentionGatePolicy
]:
    scoring = ResidualAttentionScoringConfig(top_k=5)
    policy = ResidualAttentionGatePolicy(
        quantile_limits={
            "attention_js_divergence": 0.10,
            "residual_transition_error": 0.20,
            "cache_transition_error": 0.20,
        },
        worst_limits={
            "attention_js_divergence": 0.20,
            "residual_transition_error": 0.50,
            "cache_transition_error": 0.50,
            "action_mismatch": 0.10,
        },
        max_risk_score=0.25,
        minimum_horizon=2,
    )
    return scoring, policy


def _split_variant_report(
    parent_calibration: SplitTraces,
    parent_heldout: SplitTraces,
    candidate_calibration: SplitTraces,
    candidate_heldout: SplitTraces,
    *,
    candidate_ids: tuple[str, ...],
    scoring: ResidualAttentionScoringConfig,
    policy: ResidualAttentionGatePolicy,
) -> dict[str, Any]:
    return {
        "calibration": base._variant_report(
            parent_calibration,
            candidate_calibration,
            candidate_ids=candidate_ids,
            scoring=scoring,
            policy=policy,
        ),
        "heldout": base._variant_report(
            parent_heldout,
            candidate_heldout,
            candidate_ids=candidate_ids,
            scoring=scoring,
            policy=policy,
        ),
    }


def _trace_manifest(traces: SplitTraces) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, trace in sorted(traces.items()):
        name = ":".join(map(str, key))
        payload = trace.as_dict()
        result[name] = {
            "trace_sha256": payload["trace_sha256"],
            "lane": trace.lane,
            "sequence_id": trace.sequence_id,
            "layer_index": trace.layer_index,
            "group_indices": list(trace.group_indices),
            "horizon": trace.horizon,
            "prompt_token_ids_sha256": payload["prompt_token_ids_sha256"],
            "value_projection_weight_sha256": (trace.value_projection_weight_sha256),
            "value_projection_bias_sha256": trace.value_projection_bias_sha256,
            "maximum_value_projection_cache_error": (
                trace.maximum_value_projection_cache_error
            ),
            "maximum_cached_full_prefix_error": (
                trace.maximum_cached_full_prefix_error
            ),
            "cached_trace_validation_mode": trace.cached_trace_validation_mode,
            "finite_validation": payload["finite_validation"],
        }
    return result


def _runtime_state_identity_sha256(runtime: nn.Module) -> tuple[str, int]:
    """Cheap in-process binding for every registered runtime tensor.

    The content-addressed attention snapshot below covers the full tentative
    Q/K/V/O registry.  This second hash binds all other registered runtime
    tensors by storage identity, mutation version, shape, dtype, and device so
    a trace collection cannot silently change an upstream non-attention tensor.
    It is intentionally run-local rather than a portable checkpoint digest.
    """

    values = []
    for name, value in sorted(runtime.state_dict().items()):
        try:
            version = int(value._version)
        except RuntimeError:
            version = -1
        values.append(
            (
                name,
                int(value.untyped_storage().data_ptr()),
                version,
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        )
    return base._sha256(values), len(values)


def _trace_state_binding(runtime: nn.Module) -> dict[str, Any]:
    attention = qwen3_attention_runtime_snapshot(runtime)
    runtime_identity, tensor_count = _runtime_state_identity_sha256(runtime)
    return {
        "attention_registry_sha256": attention.registry_sha256,
        "attention_registry_tensor_bytes": attention.tensor_bytes,
        "runtime_state_identity_sha256": runtime_identity,
        "runtime_state_tensor_count": tensor_count,
    }


def _finish_trace_state_binding(
    runtime: nn.Module,
    before: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    after = _trace_state_binding(runtime)
    if dict(before) != after:
        raise QwenResidualAttentionValueProbeError(
            f"Trace collection {label!r} changed its bound registry/runtime state."
        )
    return {
        "label": label,
        "before": dict(before),
        "after": after,
        "attention_registry_content_bound": True,
        "full_runtime_storage_and_version_bound": True,
        "state_unchanged_during_collection": True,
    }


def _fit_payload(fits: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(source): {
            **{
                key: value
                for key, value in fit.items()
                if key not in {"scalar", "ridge"}
            },
            "scalar": fit["scalar"].as_dict(),
            "ridge": fit["ridge"].report.as_dict(),
        }
        for source, fit in fits.items()
    }


def _fit_value_survivors(
    parent_calibration: SplitTraces,
    candidate_calibration: SplitTraces,
    proposed_registry: Any,
    sites_by_source: Mapping[int, tuple[base.Site, ...]],
    *,
    prompt_count: int,
    ridge: float,
) -> dict[int, SharedValuePrototypeFit]:
    bindings = base._materialized_site_bindings(proposed_registry)
    rectangular = proposed_registry.prototype_v_weight.ndim == 4
    slots_per_group = (
        int(proposed_registry.prototype_v_weight.shape[1]) if rectangular else 0
    )
    result: dict[int, SharedValuePrototypeFit] = {}
    for source, sites in sorted(sites_by_source.items()):
        active_slots = {bindings[site].active_slot for site in sites}
        if len(active_slots) != 1:
            raise QwenResidualAttentionValueProbeError(
                "One survivor provenance maps to multiple active V slots."
            )
        active_slot = next(iter(active_slots))
        calibrations = []
        assignment_axes = {bindings[site].assignment_axis for site in sites}
        for layer_index, materialized_group in sites:
            canonical_group = bindings[
                (layer_index, materialized_group)
            ].assignment_axis
            for prompt_index in range(prompt_count):
                key = (
                    "forced",
                    prompt_index,
                    layer_index,
                    materialized_group,
                )
                calibrations.append(
                    build_value_site_calibration(
                        candidate_calibration[key],
                        parent_calibration[key],
                        proposed_registry,
                        canonical_group=canonical_group,
                    )
                )
        if rectangular:
            if len(assignment_axes) != 1:
                raise QwenResidualAttentionValueProbeError(
                    "A rectangular V survivor cannot span canonical groups."
                )
            survivor_group = next(iter(assignment_axes))
            survivor_slot = active_slot - survivor_group * slots_per_group
        else:
            survivor_group = None
            survivor_slot = active_slot
        result[source] = fit_shared_codebook_value_prototype(
            proposed_registry,
            calibrations,
            survivor_group=survivor_group,
            survivor_slot=survivor_slot,
            ridge=ridge,
        )
    return result


def _assert_block_o_layout(
    sites_by_source: Mapping[int, tuple[base.Site, ...]],
) -> None:
    """Fail closed on a shared-O case the inherited fitter does not model.

    If one survivor is consumed by multiple groups in the same layer, its
    installable design must sum those group contexts and use the layer residual
    target once.  The exact L19/G4 -> L21/G6 candidate has one affected group
    per layer, so the proven site-wise block fitter is valid here.  A different
    topology must get a separately tested grouped-layer solver.
    """

    for source, sites in sites_by_source.items():
        layers = [layer for layer, _group in sites]
        if len(layers) != len(set(layers)):
            raise QwenResidualAttentionValueProbeError(
                f"Survivor {source} is assigned to multiple groups in one layer; "
                "the exact probe refuses to duplicate a residual target."
            )


def _post_value_trace_audit(
    proposed_registry: Any,
    trace_sets: Sequence[SplitTraces],
    affected_sites: Sequence[base.Site],
) -> dict[str, Any]:
    expected: dict[int, tuple[str, str | None]] = {}
    for layer_index in sorted({layer for layer, _group in affected_sites}):
        weight, bias = proposed_registry.materialize_projection(layer_index, "v")
        expected[layer_index] = (
            base._tensor_sha256(weight),
            None if bias is None else base._tensor_sha256(bias),
        )
    checked = 0
    for traces in trace_sets:
        for trace in traces.values():
            expected_hash = expected.get(trace.layer_index)
            if expected_hash is None:
                continue
            if expected_hash != (
                trace.value_projection_weight_sha256,
                trace.value_projection_bias_sha256,
            ):
                raise QwenResidualAttentionValueProbeError(
                    "A post-V trace does not match the final materialized V hash."
                )
            checked += 1
    if checked == 0:
        raise QwenResidualAttentionValueProbeError(
            "The post-V trace audit checked no affected trace."
        )
    return {
        "checked_trace_count": checked,
        "materialized_v_sha256_by_layer": {
            str(layer): {"weight": hashes[0], "bias": hashes[1]}
            for layer, hashes in expected.items()
        },
        "all_trace_hashes_match_final_materialized_v": True,
        "all_traces_verify_U_times_Wv_equals_post_update_cache": True,
    }


def _restore_o_exact(
    proposed_registry: Any,
    backups: Mapping[int, Tensor],
    expected_sha256: str,
) -> None:
    flat_o = base._flat_o(proposed_registry)
    with torch.inference_mode():
        for slot, backup in backups.items():
            flat_o[slot].copy_(backup)
    if base._o_slots_sha256(flat_o) != expected_sha256:
        raise QwenResidualAttentionValueProbeError(
            "O control did not restore the tentative registry exactly."
        )


def _collect_both_splits(
    runtime: nn.Module,
    calibration_prompts: Sequence[Tensor],
    heldout_prompts: Sequence[Tensor],
    affected_sites: Sequence[base.Site],
    calibration_drivers: Mapping[int, tuple[int, ...]],
    heldout_drivers: Mapping[int, tuple[int, ...]],
    *,
    label: str,
    horizon: int,
    limits: HFCachedAttentionTraceLimits,
) -> tuple[
    dict[base.TraceKey, HFCachedAttentionTrace],
    dict[base.TraceKey, HFCachedAttentionTrace],
    dict[str, Any],
]:
    calibration_before = _trace_state_binding(runtime)
    calibration = base._collect_candidate_traces(
        runtime,
        calibration_prompts,
        affected_sites,
        calibration_drivers,
        horizon=horizon,
        limits=limits,
    )
    calibration_binding = _finish_trace_state_binding(
        runtime,
        calibration_before,
        label=f"{label}:calibration",
    )
    heldout_before = _trace_state_binding(runtime)
    heldout = base._collect_candidate_traces(
        runtime,
        heldout_prompts,
        affected_sites,
        heldout_drivers,
        horizon=horizon,
        limits=limits,
    )
    heldout_binding = _finish_trace_state_binding(
        runtime,
        heldout_before,
        label=f"{label}:heldout",
    )
    return (
        calibration,
        heldout,
        {
            "calibration": calibration_binding,
            "heldout": heldout_binding,
        },
    )


def run_qwen_p35_value_probe(
    runtime: nn.Module,
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    *,
    redirect: base.AttentionRedirect = EXACT_REDIRECT,
    require_exact_p35: bool = True,
    horizon: int = 8,
    ridge: float = 1e-6,
    trace_limits: HFCachedAttentionTraceLimits | None = None,
    max_affected_sites: int = 16,
    scoring: ResidualAttentionScoringConfig | None = None,
    gate_policy: ResidualAttentionGatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate four controls and restore the original p35 registry exactly."""

    limits = trace_limits or HFCachedAttentionTraceLimits()
    calibration_prompts, heldout_prompts = _validate_prompt_splits(
        calibration_prompt_input_ids,
        heldout_prompt_input_ids,
        limits,
    )
    if require_exact_p35 and redirect != EXACT_REDIRECT:
        raise QwenResidualAttentionValueProbeError(
            "The executable p35 proof is fixed to L19/G4 -> L21/G6."
        )
    scoring_controls, policy_controls = _default_scoring()
    scoring_value = scoring or scoring_controls
    policy_value = gate_policy or policy_controls
    max_sites = base._positive_int(max_affected_sites, "max_affected_sites")
    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    if require_exact_p35 and (
        parent_snapshot.prototype_count != 280
        or parent_snapshot.key_value_heads != 8
        or parent_snapshot.logical_layers != 36
    ):
        raise QwenResidualAttentionValueProbeError(
            "Exact mode requires the 36-layer, 8-group, 280-prototype p35 frontier."
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
    _assert_block_o_layout(sites_by_source)
    if require_exact_p35 and (
        affected_sites != ((2, 1), (19, 4), (21, 6))
        or tuple(sites_by_source) != (21 * 8 + 6,)
    ):
        raise QwenResidualAttentionValueProbeError(
            "The p35 lineage does not expose the expected L2/G1, L19/G4, and "
            "L21/G6 survivor assignments."
        )
    if len(affected_sites) > max_sites:
        raise QwenResidualAttentionValueProbeError(
            f"Affected sites exceed max_affected_sites={max_sites}."
        )
    if len({layer for layer, _group in affected_sites}) > limits.max_layers:
        raise QwenResidualAttentionValueProbeError(
            "Affected layers exceed the cached-trace layer bound."
        )
    parent_o_blocks = base._parent_o_blocks(parent_registry, affected_sites)
    candidate_ids = (redirect.candidate_id,)
    trace_state_bindings: dict[str, Any] = {}
    parent_calibration_before = _trace_state_binding(runtime)
    parent_calibration, calibration_drivers = base._collect_parent_traces(
        runtime,
        calibration_prompts,
        affected_sites,
        horizon=horizon,
        limits=limits,
    )
    parent_calibration_binding = _finish_trace_state_binding(
        runtime,
        parent_calibration_before,
        label="parent:calibration",
    )
    parent_heldout_before = _trace_state_binding(runtime)
    parent_heldout, heldout_drivers = base._collect_parent_traces(
        runtime,
        heldout_prompts,
        affected_sites,
        horizon=horizon,
        limits=limits,
    )
    parent_heldout_binding = _finish_trace_state_binding(
        runtime,
        parent_heldout_before,
        label="parent:heldout",
    )
    trace_state_bindings["parent"] = {
        "calibration": parent_calibration_binding,
        "heldout": parent_heldout_binding,
    }

    variants: dict[str, Any] = {}
    functional_calibration_traces: dict[str, SplitTraces] = {}
    functional_heldout_traces: dict[str, SplitTraces] = {}
    variant_trace_manifests: dict[str, Any] = {}
    o_only_fits: dict[int, dict[str, Any]] = {}
    post_v_o_fits: dict[int, dict[str, Any]] = {}
    value_fits: dict[int, SharedValuePrototypeFit] = {}
    value_installations: list[InstalledValuePrototype] = []
    value_trace_audit: dict[str, Any] | None = None
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
        ) = _collect_both_splits(
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
        variants["uncompensated"] = _split_variant_report(
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
        variant_trace_manifests["uncompensated"] = {
            "calibration": _trace_manifest(baseline_calibration),
            "heldout": _trace_manifest(baseline_heldout),
        }

        # O-only control: fit only on baseline calibration forced traces.
        o_only_fits = base._fit_survivor_blocks(
            parent_calibration,
            baseline_calibration,
            parent_o_blocks,
            codebook.registry,
            sites_by_source,
            prompt_count=len(calibration_prompts),
            ridge=ridge,
        )
        flat_o = base._flat_o(codebook.registry)
        target_o_slots = {int(value["active_slot"]) for value in o_only_fits.values()}
        o_backups = {
            slot: flat_o[slot].detach().clone() for slot in sorted(target_o_slots)
        }
        proposed_o_sha256 = base._o_slots_sha256(flat_o)
        o_only_audit = base._apply_compensation(
            codebook.registry, o_only_fits, kind="ridge"
        )
        o_calibration, o_heldout, o_trace_binding = _collect_both_splits(
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
        trace_state_bindings["o_only"] = o_trace_binding
        variants["o_only"] = {
            **_split_variant_report(
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
        variant_trace_manifests["o_only"] = {
            "calibration": _trace_manifest(o_calibration),
            "heldout": _trace_manifest(o_heldout),
        }
        _restore_o_exact(codebook.registry, o_backups, proposed_o_sha256)

        # V-only control: the global fit sees calibration forced traces only.
        value_fits = _fit_value_survivors(
            parent_calibration,
            baseline_calibration,
            codebook.registry,
            sites_by_source,
            prompt_count=len(calibration_prompts),
            ridge=ridge,
        )
        for source in sorted(value_fits):
            value_installations.append(
                install_shared_codebook_value_prototype(
                    codebook.registry, value_fits[source]
                )
            )
        v_calibration, v_heldout, v_trace_binding = _collect_both_splits(
            runtime,
            calibration_prompts,
            heldout_prompts,
            affected_sites,
            calibration_drivers,
            heldout_drivers,
            label="v_only",
            horizon=horizon,
            limits=limits,
        )
        trace_state_bindings["v_only"] = v_trace_binding
        value_trace_audit = _post_value_trace_audit(
            codebook.registry,
            (v_calibration, v_heldout),
            affected_sites,
        )
        variants["v_only"] = _split_variant_report(
            parent_calibration,
            parent_heldout,
            v_calibration,
            v_heldout,
            candidate_ids=candidate_ids,
            scoring=scoring_value,
            policy=policy_value,
        )
        functional_calibration_traces["v_only"] = v_calibration
        functional_heldout_traces["v_only"] = v_heldout
        variant_trace_manifests["v_only"] = {
            "calibration": _trace_manifest(v_calibration),
            "heldout": _trace_manifest(v_heldout),
        }

        # Alternating V -> O: discard the stale O-only fit and solve again from
        # the actual post-install child A@V calibration contexts.
        for installation in value_installations:
            installation.assert_live()
        post_v_o_fits = base._fit_survivor_blocks(
            parent_calibration,
            v_calibration,
            parent_o_blocks,
            codebook.registry,
            sites_by_source,
            prompt_count=len(calibration_prompts),
            ridge=ridge,
        )
        if {
            int(value["active_slot"]) for value in post_v_o_fits.values()
        } != target_o_slots:
            raise QwenResidualAttentionValueProbeError(
                "Post-V O fit targets differ from the O-only control slots."
            )
        vo_audit = base._apply_compensation(
            codebook.registry, post_v_o_fits, kind="ridge"
        )
        vo_calibration, vo_heldout, vo_trace_binding = _collect_both_splits(
            runtime,
            calibration_prompts,
            heldout_prompts,
            affected_sites,
            calibration_drivers,
            heldout_drivers,
            label="v_then_o",
            horizon=horizon,
            limits=limits,
        )
        trace_state_bindings["v_then_o"] = vo_trace_binding
        variants["v_then_o"] = {
            **_split_variant_report(
                parent_calibration,
                parent_heldout,
                vo_calibration,
                vo_heldout,
                candidate_ids=candidate_ids,
                scoring=scoring_value,
                policy=policy_value,
            ),
            "edit_audit": vo_audit,
        }
        functional_calibration_traces["v_then_o"] = vo_calibration
        functional_heldout_traces["v_then_o"] = vo_heldout
        variant_trace_manifests["v_then_o"] = {
            "calibration": _trace_manifest(vo_calibration),
            "heldout": _trace_manifest(vo_heldout),
        }
        value_trace_audit = _post_value_trace_audit(
            codebook.registry,
            (v_calibration, v_heldout, vo_calibration, vo_heldout),
            affected_sites,
        )
        _restore_o_exact(codebook.registry, o_backups, proposed_o_sha256)
        for installation in reversed(value_installations):
            installation.rollback()
        proposed_reset_exact = (
            qwen3_attention_runtime_snapshot(runtime) == proposed_snapshot
        )
        if not proposed_reset_exact:
            raise QwenResidualAttentionValueProbeError(
                "V/O controls did not restore the tentative redirect exactly."
            )
    finally:
        # Best-effort nested restoration before swapping the incumbent registry
        # back.  On the successful path each operation is also audited above.
        failure_in_flight = sys.exc_info()[0] is not None
        restoration_errors: list[Exception] = []
        if proposed_o_sha256 is not None and o_backups:
            try:
                _restore_o_exact(codebook.registry, o_backups, proposed_o_sha256)
            except Exception as error:  # pragma: no cover - adversarial cleanup
                restoration_errors.append(error)
        for installation in reversed(value_installations):
            if installation.is_applied and not installation.is_committed:
                try:
                    installation.rollback()
                except Exception as error:  # pragma: no cover - adversarial cleanup
                    restoration_errors.append(error)
        try:
            restored = attention_surgery._restore_installed(
                runtime, parent_installation
            )
            incumbent_rollback_exact = bool(
                restored
                and qwen3_attention_runtime_snapshot(runtime) == parent_snapshot
            )
            if not incumbent_rollback_exact:
                restoration_errors.append(
                    QwenResidualAttentionValueProbeError(
                        "Probe did not restore the incumbent p35 registry exactly."
                    )
                )
        except Exception as error:  # pragma: no cover - adversarial cleanup
            restoration_errors.append(error)
        if restoration_errors and not failure_in_flight:
            raise QwenResidualAttentionValueProbeError(
                "Probe cleanup failed after attempting every rollback layer: "
                + "; ".join(str(error) for error in restoration_errors)
            ) from restoration_errors[0]

    if proposed_snapshot is None or value_trace_audit is None:
        raise QwenResidualAttentionValueProbeError(
            "Probe completed without proposed/V trace evidence."
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
        "ridge": ridge,
        "prompt_splits": {
            "calibration": {
                "count": len(calibration_prompts),
                "prompt_sha256s": list(_prompt_sha256s(calibration_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(calibration_drivers.items())
                },
                "used_by_v_fit": True,
                "used_by_o_fits": True,
                "used_by_functional_selection": True,
            },
            "heldout": {
                "count": len(heldout_prompts),
                "prompt_sha256s": list(_prompt_sha256s(heldout_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(heldout_drivers.items())
                },
                "used_by_v_fit": False,
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
            "parameter_count_unchanged_by_v_o_fits": True,
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
            "o_only": _fit_payload(o_only_fits),
            "value": {
                str(source): fit.report.as_dict() for source, fit in value_fits.items()
            },
            "value_installations": {
                str(source): installation.report.as_dict()
                for source, installation in zip(
                    sorted(value_fits), value_installations, strict=True
                )
            },
            "post_v_o": _fit_payload(post_v_o_fits),
        },
        "value_trace_audit": value_trace_audit,
        "variants": variants,
        "functional_comparison": functional_comparison,
        "variant_trace_provenance": variant_trace_manifests,
        "parent_trace_provenance": {
            "calibration": _trace_manifest(parent_calibration),
            "heldout": _trace_manifest(parent_heldout),
        },
        "trace_collection_state_bindings": trace_state_bindings,
        "fit_contract": {
            "four_controls": [
                "uncompensated",
                "o_only",
                "v_only",
                "v_then_o",
            ],
            "identical_parent_references_and_drivers_across_controls": True,
            "calibration_and_heldout_forced_and_greedy_lanes": True,
            "heldout_excluded_from_all_fits": True,
            "v_global_across_every_survivor_assignment": True,
            "v_design_uses_exact_child_v_proj_input": True,
            "v_target_is_parent_post_update_value_cache": True,
            "raw_parent_v_cache_reconstruction_is_diagnostic": True,
            "v_only_control_is_always_retained_even_if_harmful": True,
            "v_role_scales_and_group_permutations_honored": True,
            "o_only_design_uses_pre_v_child_A_at_V": True,
            "post_v_o_fit_recomputed_not_reused": True,
            "live_v_installation_revalidated_immediately_before_o_refit": True,
            "post_v_o_design_uses_recollected_actual_child_A_at_V": True,
            "one_survivor_group_per_layer_for_block_o_verified": True,
            "multi_group_same_layer_o_target_duplication_refused": True,
            "attention_registry_content_hash_bound_around_every_trace_collection": True,
            "full_runtime_storage_version_hash_bound_around_every_trace_collection": True,
            "cross_tokenizer_mapping_supported": False,
            "identical_raw_token_ids_required": True,
            "candidate_selection_uses_calibration_functional_outputs_only": True,
            "heldout_functional_outputs_are_evaluation_only": True,
            "heldout_excluded_from_candidate_selection": True,
            "attention_residual_cache_and_weight_similarity_excluded_from_selection": True,
            "v_target_not_retargeted_from_parent_cache": True,
            "zero_gradient": True,
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


def _load_texts(
    inline: Sequence[str], json_path: Path | None, name: str
) -> tuple[str, ...]:
    values = [str(value) for value in inline]
    if json_path is not None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(
            not isinstance(item, str) for item in payload
        ):
            raise QwenResidualAttentionValueProbeError(
                f"{name} JSON must contain a list of strings."
            )
        values.extend(payload)
    if not values:
        raise QwenResidualAttentionValueProbeError(f"Provide at least one {name}.")
    return tuple(values)


def _encode_prompts(
    tokenizer: Any, texts: Sequence[str], max_prompt_tokens: int
) -> tuple[Tensor, ...]:
    result = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )["input_ids"]
        if not isinstance(encoded, Tensor) or tuple(encoded.shape[:1]) != (1,):
            raise QwenResidualAttentionValueProbeError(
                "Tokenizer did not return one input_ids row."
            )
        result.append(encoded.to(dtype=torch.long))
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--calibration-prompt", action="append", default=[])
    parser.add_argument("--calibration-prompts-json", type=Path)
    parser.add_argument("--heldout-prompt", action="append", default=[])
    parser.add_argument("--heldout-prompts-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-6)
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
    calibration_texts = _load_texts(
        args.calibration_prompt,
        args.calibration_prompts_json,
        "calibration prompt",
    )
    heldout_texts = _load_texts(
        args.heldout_prompt,
        args.heldout_prompts_json,
        "held-out prompt",
    )
    if max(len(calibration_texts), len(heldout_texts)) > (args.max_prompts_per_split):
        raise QwenResidualAttentionValueProbeError(
            "A prompt split exceeds --max-prompts-per-split."
        )
    runtime, tokenizer, source = base._load_runtime_and_tokenizer(args)
    calibration_prompts = _encode_prompts(
        tokenizer, calibration_texts, args.max_prompt_tokens
    )
    heldout_prompts = _encode_prompts(tokenizer, heldout_texts, args.max_prompt_tokens)
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
    report = run_qwen_p35_value_probe(
        runtime,
        calibration_prompts,
        heldout_prompts,
        redirect=EXACT_REDIRECT,
        require_exact_p35=True,
        horizon=args.horizon,
        ridge=args.ridge,
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
