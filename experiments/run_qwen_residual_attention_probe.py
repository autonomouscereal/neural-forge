#!/usr/bin/env python3
"""Probe Qwen codebook redirects with cached residual-attention evidence.

This is a measurement harness for the p36 -> p35/p34 attention frontier.  It
does not commit a fold or save a child checkpoint.  A coupled Q/K/V/O redirect
is installed tentatively, evaluated on forced and greedy cached lanes, and
rolled back exactly.

Optional scalar and ridge compensation edit only the existing survivor O
prototype blocks.  Every logical ``[layer, group]`` assigned to a survivor is
included in one global solve.  Each target is the corresponding parent group
contribution, never the full attention delta.  Because O group blocks add
linearly, different survivor slots use independent block-sparse solves; the
whole-model closed-loop trace is rerun after materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
from neural_forge.hf_cached_attention_trace import (
    HFCachedAttentionTrace,
    HFCachedAttentionTraceLimits,
    build_residual_attention_candidate_evidence,
    collect_hf_cached_attention_trace,
)
from neural_forge.hf_qwen_attention_surgery import (
    Qwen3AttentionSurgeryBridge,
    Qwen3AttentionSurgeryFeature,
    qwen3_attention_group_node_id,
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (
    load_qwen3_codebook_checkpoint,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
)
from neural_forge.hf_residual_attention_fit import (
    fit_attention_output_ridge,
    fit_residual_branch_scale,
)
from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionGatePolicy,
    ResidualAttentionScoringConfig,
    gate_residual_attention_candidate,
    score_residual_attention_candidate,
)


RUN_SCHEMA = "neural-forge-qwen-residual-attention-probe-v1"
CompensationMode = Literal["none", "scalar", "ridge", "both"]
Site = tuple[int, int]
TraceKey = tuple[str, int, int, int]
_CANDIDATE_PATTERN = re.compile(r"^(\d+):(\d+)=(\d+):(\d+)$")


class QwenResidualAttentionProbeError(RuntimeError):
    """The residual-attention probe cannot prove a required invariant."""


def _json_default(item: Any) -> Any:
    scalar = getattr(item, "item", None)
    if callable(scalar):
        normalized = scalar()
        if normalized is not item:
            return normalized
    if isinstance(item, Path):
        return str(item)
    raise TypeError(f"Unsupported canonical JSON value {type(item).__name__}.")


def _canonical_json(value: Any) -> bytes:

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    raw = (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenResidualAttentionProbeError(f"{name} must be positive.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QwenResidualAttentionProbeError(f"{name} must be positive.") from error
    if result <= 0 or result != value:
        raise QwenResidualAttentionProbeError(f"{name} must be positive.")
    return result


@dataclass(frozen=True, order=True, slots=True)
class AttentionRedirect:
    source_layer: int
    source_group: int
    target_layer: int
    target_group: int

    def __post_init__(self) -> None:
        for name in (
            "source_layer",
            "source_group",
            "target_layer",
            "target_group",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise QwenResidualAttentionProbeError(
                    f"{name} must be a non-negative integer."
                )
        if self.source == self.target:
            raise QwenResidualAttentionProbeError(
                "A redirect source and survivor cannot be the same node."
            )

    @property
    def source(self) -> tuple[int, int]:
        return self.source_layer, self.source_group

    @property
    def target(self) -> tuple[int, int]:
        return self.target_layer, self.target_group

    @property
    def candidate_id(self) -> str:
        return (
            f"L{self.source_layer}G{self.source_group}->"
            f"L{self.target_layer}G{self.target_group}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": list(self.source),
            "target": list(self.target),
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True, slots=True)
class _SiteBinding:
    layer_index: int
    materialized_group: int
    assignment_axis: int
    active_slot: int
    source_flat_index: int
    output_scale: float


def parse_attention_redirect(value: str) -> AttentionRedirect:
    match = _CANDIDATE_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "candidate must be SOURCE_LAYER:SOURCE_GROUP=TARGET_LAYER:TARGET_GROUP"
        )
    return AttentionRedirect(*(int(item) for item in match.groups()))


def _registry(runtime: nn.Module) -> Any:
    codebook = getattr(runtime, "attention_codebook", None)
    registry = getattr(codebook, "registry", None)
    if registry is None:
        raise QwenResidualAttentionProbeError(
            "Runtime must expose an installed attention_codebook.registry."
        )
    return registry


def _flat_sources(registry: Any) -> Tensor:
    return (
        registry.prototype_source_indices.detach()
        .reshape(-1)
        .to(device="cpu", dtype=torch.long)
    )


def _flat_o(registry: Any) -> Tensor:
    value = registry.prototype_o_weight
    if value.ndim == 3:
        return value
    if value.ndim == 4:
        return value.reshape(-1, *value.shape[2:])
    raise QwenResidualAttentionProbeError(
        "Attention O prototype tensor must be rectangular or ragged."
    )


def _global_assignments(registry: Any) -> Tensor:
    assignments = registry.assignments.detach().to(device="cpu", dtype=torch.long)
    if registry.prototype_q_weight.ndim == 3:
        return assignments
    groups, slots = map(int, registry.prototype_q_weight.shape[:2])
    offsets = torch.arange(groups, dtype=torch.long) * slots
    return assignments + offsets.unsqueeze(0)


def _materialized_site_bindings(registry: Any) -> dict[Site, _SiteBinding]:
    """Map each executable canonical group through its own layer permutation."""

    assignments = _global_assignments(registry)
    sources = _flat_sources(registry)
    permutations = registry.group_permutations.detach().to(
        device="cpu", dtype=torch.long
    )
    layers, groups = map(int, assignments.shape)
    if tuple(permutations.shape) != (layers, groups):
        raise QwenResidualAttentionProbeError(
            "Group permutations disagree with logical assignments."
        )
    result: dict[Site, _SiteBinding] = {}
    for layer_index in range(layers):
        # materialize_projection performs index_select(argsort(permutation)).
        materialized_to_assignment = torch.argsort(permutations[layer_index])
        for materialized_group in range(groups):
            assignment_axis = int(materialized_to_assignment[materialized_group].item())
            slot = int(assignments[layer_index, assignment_axis].item())
            source = int(sources[slot].item())
            scale = 1.0
            if registry.role_scales is not None:
                scale = float(
                    registry.role_scales[
                        layer_index,
                        assignment_axis,
                        3,
                    ]
                    .detach()
                    .to(torch.float64)
                    .item()
                )
            result[(layer_index, materialized_group)] = _SiteBinding(
                layer_index=layer_index,
                materialized_group=materialized_group,
                assignment_axis=assignment_axis,
                active_slot=slot,
                source_flat_index=source,
                output_scale=scale,
            )
    return result


def _attention_features(
    snapshot: Any,
    redirects: Sequence[AttentionRedirect],
) -> tuple[Qwen3AttentionSurgeryFeature, ...]:
    source_nodes = {redirect.source for redirect in redirects}
    target_nodes = {redirect.target for redirect in redirects}
    if source_nodes & target_nodes:
        raise QwenResidualAttentionProbeError(
            "Atomic redirect sources and survivors must be disjoint."
        )
    values = []
    for source_flat in snapshot.prototype_source_indices:
        layer, group = divmod(source_flat, snapshot.key_value_heads)
        node = (layer, group)
        importance = (
            1e9 if node in target_nodes else 1e-9 if node in source_nodes else 1.0
        )
        feature = float(source_flat)
        values.append(
            Qwen3AttentionSurgeryFeature(
                layer,
                group,
                weight=[feature],
                activation=[feature, feature + 0.01],
                jvp=[feature + 0.02],
                output=[feature + 0.03],
                importance=importance,
            )
        )
    return tuple(values)


def _tentative_codebook(
    runtime: nn.Module,
    redirects: Sequence[AttentionRedirect],
) -> tuple[Any, Any, tuple[Any, ...], tuple[Any, ...]]:
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    active = set(parent_snapshot.active_node_ids)
    for redirect in redirects:
        source = qwen3_attention_group_node_id(*redirect.source)
        target = qwen3_attention_group_node_id(*redirect.target)
        if source not in active or target not in active:
            raise QwenResidualAttentionProbeError(
                f"Redirect {redirect.candidate_id} references an inactive prototype."
            )
    bridge = Qwen3AttentionSurgeryBridge(
        runtime,
        _attention_features(parent_snapshot, redirects),
        max_nonconflicting_batch_size=max(len(redirects), 1),
        max_candidates=max(64, len(redirects) * 4),
    )
    candidates = []
    for redirect in redirects:
        scored = bridge.propose_pair(
            qwen3_attention_group_node_id(*redirect.source),
            qwen3_attention_group_node_id(*redirect.target),
        )
        expected_target = qwen3_attention_group_node_id(*redirect.target)
        if scored.candidate.resulting_medoid != expected_target:
            raise QwenResidualAttentionProbeError(
                f"Redirect {redirect.candidate_id} did not retain its requested survivor."
            )
        if bridge._removed_node(scored.candidate) != qwen3_attention_group_node_id(
            *redirect.source
        ):
            raise QwenResidualAttentionProbeError(
                f"Redirect {redirect.candidate_id} did not remove its requested source."
            )
        candidates.append(scored.candidate)
    proposal = bridge.graph.preview(tuple(candidates))

    def projection_for_pair(source: Any, target: Any) -> Any:
        return bridge._projection_for_pair_from(source, target, bridge.projections)

    codebook, projections = attention_surgery._compacted_attention_codebook(
        runtime,
        proposal.snapshot,
        projection_for_pair,
    )
    incumbent_registry = _registry(runtime)
    incumbent_reference = incumbent_registry.prototype_q_weight
    codebook.to(
        device=incumbent_reference.device,
        dtype=incumbent_reference.dtype,
    )
    codebook.eval()
    return bridge, proposal, tuple(candidates), (codebook, projections)


def _affected_sites(
    parent_registry: Any,
    proposed_registry: Any,
    redirects: Sequence[AttentionRedirect],
) -> tuple[tuple[Site, ...], Mapping[int, tuple[Site, ...]]]:
    parent = _materialized_site_bindings(parent_registry)
    proposed = _materialized_site_bindings(proposed_registry)
    if set(parent) != set(proposed):
        raise QwenResidualAttentionProbeError(
            "Tentative fold changed the logical attention site set."
        )
    groups = int(parent_registry.key_value_heads)
    survivor_sources = {
        redirect.target_layer * groups + redirect.target_group for redirect in redirects
    }
    changed = {
        site
        for site in parent
        if parent[site].source_flat_index != proposed[site].source_flat_index
    }
    by_source: dict[int, tuple[Site, ...]] = {}
    for source in sorted(survivor_sources):
        # Include every executable site now sharing this exact survivor slot,
        # including its original assignments.  Do not include other slots.
        sites = tuple(
            sorted(
                site
                for site, binding in proposed.items()
                if binding.source_flat_index == source
            )
        )
        if not sites:
            raise QwenResidualAttentionProbeError(
                f"Survivor source {source} has no proposed logical assignments."
            )
        slots = {proposed[site].active_slot for site in sites}
        if len(slots) != 1:
            raise QwenResidualAttentionProbeError(
                "One survivor provenance source maps to multiple active slots."
            )
        by_source[source] = sites
    affected = tuple(
        sorted(changed | {site for sites in by_source.values() for site in sites})
    )
    return affected, MappingProxyType(by_source)


def _collect_parent_traces(
    runtime: nn.Module,
    prompts: Sequence[Tensor],
    sites: Sequence[Site],
    *,
    horizon: int,
    limits: HFCachedAttentionTraceLimits,
) -> tuple[dict[TraceKey, HFCachedAttentionTrace], dict[int, tuple[int, ...]]]:
    traces: dict[TraceKey, HFCachedAttentionTrace] = {}
    drivers: dict[int, tuple[int, ...]] = {}
    for prompt_index, prompt in enumerate(prompts):
        for layer_index, group_index in sites:
            trace = collect_hf_cached_attention_trace(
                runtime,
                prompt,
                layer_index=layer_index,
                group_indices=(group_index,),
                horizon=horizon,
                sequence_id=f"prompt-{prompt_index}",
                lane="greedy",
                limits=limits,
            )
            prior = drivers.setdefault(prompt_index, trace.selected_token_ids)
            if prior != trace.selected_token_ids:
                raise QwenResidualAttentionProbeError(
                    "Parent greedy driver changed across observed attention sites."
                )
            traces[("greedy", prompt_index, layer_index, group_index)] = trace
            traces[("forced", prompt_index, layer_index, group_index)] = replace(
                trace,
                lane="forced",
            )
    return traces, drivers


def _collect_candidate_traces(
    runtime: nn.Module,
    prompts: Sequence[Tensor],
    sites: Sequence[Site],
    drivers: Mapping[int, tuple[int, ...]],
    *,
    horizon: int,
    limits: HFCachedAttentionTraceLimits,
) -> dict[TraceKey, HFCachedAttentionTrace]:
    traces: dict[TraceKey, HFCachedAttentionTrace] = {}
    for prompt_index, prompt in enumerate(prompts):
        for layer_index, group_index in sites:
            for lane in ("forced", "greedy"):
                trace = collect_hf_cached_attention_trace(
                    runtime,
                    prompt,
                    layer_index=layer_index,
                    group_indices=(group_index,),
                    horizon=horizon,
                    sequence_id=f"prompt-{prompt_index}",
                    lane=lane,
                    forced_token_ids=(
                        drivers[prompt_index] if lane == "forced" else None
                    ),
                    limits=limits,
                )
                traces[(lane, prompt_index, layer_index, group_index)] = trace
    return traces


def _lane_report(
    references: Mapping[TraceKey, HFCachedAttentionTrace],
    candidates: Mapping[TraceKey, HFCachedAttentionTrace],
    *,
    lane: str,
    candidate_ids: tuple[str, ...],
    scoring: ResidualAttentionScoringConfig,
    policy: ResidualAttentionGatePolicy,
) -> dict[str, Any]:
    keys = tuple(sorted(key for key in references if key[0] == lane))
    if not keys or set(keys) != {key for key in candidates if key[0] == lane}:
        raise QwenResidualAttentionProbeError(
            f"{lane} reference/candidate trace keys do not match."
        )
    divergent = [
        {
            "trace_key": list(key),
            "parent_consumed": list(references[key].consumed_token_ids),
            "candidate_consumed": list(candidates[key].consumed_token_ids),
            "parent_selected": list(references[key].selected_token_ids),
            "candidate_selected": list(candidates[key].selected_token_ids),
        }
        for key in keys
        if references[key].consumed_token_ids != candidates[key].consumed_token_ids
    ]
    if divergent:
        if lane != "greedy":
            raise QwenResidualAttentionProbeError(
                "Forced reference/candidate trajectories diverged."
            )
        return {
            "lane": lane,
            "comparable_internal_evidence": False,
            "reason": (
                "greedy trajectories consumed different tokens; fixed-parent-A "
                "readout and state-coordinate comparisons are invalid"
            ),
            "divergent_traces": divergent,
            "accepted": False,
        }
    evidence = build_residual_attention_candidate_evidence(
        tuple(references[key] for key in keys),
        tuple(candidates[key] for key in keys),
        candidate_ids=candidate_ids,
    )
    damage = score_residual_attention_candidate(evidence, scoring)
    decision = gate_residual_attention_candidate(damage, policy)
    return {
        "lane": lane,
        "comparable_internal_evidence": True,
        "damage": damage.as_dict(),
        "gate": decision.as_dict(),
        "accepted": decision.accepted,
    }


def _variant_report(
    references: Mapping[TraceKey, HFCachedAttentionTrace],
    candidates: Mapping[TraceKey, HFCachedAttentionTrace],
    *,
    candidate_ids: tuple[str, ...],
    scoring: ResidualAttentionScoringConfig,
    policy: ResidualAttentionGatePolicy,
) -> dict[str, Any]:
    forced = _lane_report(
        references,
        candidates,
        lane="forced",
        candidate_ids=candidate_ids,
        scoring=scoring,
        policy=policy,
    )
    greedy = _lane_report(
        references,
        candidates,
        lane="greedy",
        candidate_ids=candidate_ids,
        scoring=scoring,
        policy=policy,
    )
    return {
        "forced": forced,
        "greedy": greedy,
        "promotion_recommended": bool(forced["accepted"] and greedy["accepted"]),
    }


def _parent_o_blocks(registry: Any, sites: Sequence[Site]) -> dict[Site, Tensor]:
    q_rows = int(registry.query_rows_per_group)
    result = {}
    for layer_index, group_index in sites:
        weight, _bias = registry.materialize_projection(layer_index, "o")
        start = group_index * q_rows
        result[(layer_index, group_index)] = (
            weight[:, start : start + q_rows].detach().cpu().to(torch.float64).clone()
        )
    return result


def _fit_survivor_blocks(
    parent_traces: Mapping[TraceKey, HFCachedAttentionTrace],
    candidate_traces: Mapping[TraceKey, HFCachedAttentionTrace],
    parent_o_blocks: Mapping[Site, Tensor],
    proposed_registry: Any,
    sites_by_source: Mapping[int, tuple[Site, ...]],
    *,
    prompt_count: int,
    ridge: float,
) -> dict[int, dict[str, Any]]:
    bindings = _materialized_site_bindings(proposed_registry)
    flat_o = _flat_o(proposed_registry)
    result: dict[int, dict[str, Any]] = {}
    for source, sites in sites_by_source.items():
        slots = {bindings[site].active_slot for site in sites}
        if len(slots) != 1:
            raise QwenResidualAttentionProbeError(
                "A survivor global solve spans more than one physical O slot."
            )
        slot = next(iter(slots))
        prior_raw = flat_o[slot].detach().cpu().to(torch.float64).clone()
        design_rows: list[Tensor] = []
        target_rows: list[Tensor] = []
        actual_contribution_rows: list[Tensor] = []
        for site in sites:
            layer_index, group_index = site
            site_scale = bindings[site].output_scale
            parent_o = parent_o_blocks[site]
            for prompt_index in range(prompt_count):
                key = ("forced", prompt_index, layer_index, group_index)
                parent = parent_traces[key]
                candidate = candidate_traces[key]
                for parent_step, candidate_step in zip(parent.steps, candidate.steps):
                    target = (
                        parent_step.selected_attention_context.to(torch.float64)
                        @ parent_o.T
                    )
                    actual_context = candidate_step.selected_attention_context.to(
                        torch.float64
                    )
                    # Installable O sees the candidate runtime context
                    # A_child @ V_child.  A_parent @ V_fold remains an evaluator
                    # diagnostic and is never used as this regression design.
                    design_rows.append(actual_context * site_scale)
                    target_rows.append(target)
                    actual_contribution_rows.append(
                        (actual_context * site_scale) @ prior_raw.T
                    )
        design = torch.cat(design_rows, dim=0)
        targets = torch.cat(target_rows, dim=0)
        actual_contributions = torch.cat(actual_contribution_rows, dim=0)
        scalar = fit_residual_branch_scale(
            actual_contributions,
            targets,
            ridge_to_one=ridge,
            minimum_scale=0.0,
            maximum_scale=4.0,
        )
        output = fit_attention_output_ridge(
            design,
            targets,
            prior_raw,
            ridge=ridge,
        )
        result[source] = {
            "active_slot": slot,
            "affected_sites": [list(site) for site in sites],
            "row_count": int(design.shape[0]),
            "target_contract": "parent-selected-group-O-contribution",
            "design_contract": (
                "candidate-runtime-A-child-V-child-times-existing-logical-site-scale"
            ),
            "parent_fixed_attention_readout_used_as_o_design": False,
            "block_sparse": True,
            "full_attention_delta_used_as_regression_target": False,
            "scalar": scalar,
            "ridge": output,
        }
    return result


def _non_o_identity_manifest(registry: Any) -> tuple[tuple[Any, ...], ...]:
    result = []
    for name, value in sorted(registry.state_dict().items()):
        if name == "prototype_o_weight":
            continue
        try:
            version = int(value._version)
        except RuntimeError:
            version = -1
        result.append(
            (
                name,
                int(value.untyped_storage().data_ptr()),
                version,
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        )
    return tuple(result)


def _o_slots_sha256(flat_o: Tensor, *, excluded: set[int] | None = None) -> str:
    omitted = excluded or set()
    digest = hashlib.sha256()
    for slot in range(int(flat_o.shape[0])):
        if slot in omitted:
            continue
        digest.update(bytes.fromhex(_tensor_sha256(flat_o[slot])))
    return digest.hexdigest()


def _apply_compensation(
    proposed_registry: Any,
    fits: Mapping[int, Mapping[str, Any]],
    *,
    kind: Literal["scalar", "ridge"],
) -> dict[str, Any]:
    flat_o = _flat_o(proposed_registry)
    slots = {int(value["active_slot"]) for value in fits.values()}
    non_o_before = _non_o_identity_manifest(proposed_registry)
    unaffected_before = _o_slots_sha256(flat_o, excluded=slots)
    with torch.inference_mode():
        for value in fits.values():
            slot = int(value["active_slot"])
            if kind == "scalar":
                flat_o[slot].mul_(float(value["scalar"].scale))
            else:
                matrix = torch.tensor(
                    value["ridge"].weight,
                    dtype=flat_o.dtype,
                    device=flat_o.device,
                )
                if matrix.shape != flat_o[slot].shape:
                    raise QwenResidualAttentionProbeError(
                        "Fitted O block shape does not match survivor prototype storage."
                    )
                flat_o[slot].copy_(matrix)
    if _non_o_identity_manifest(proposed_registry) != non_o_before:
        raise QwenResidualAttentionProbeError(
            "O compensation mutated a non-O registry tensor."
        )
    unaffected_after = _o_slots_sha256(flat_o, excluded=slots)
    if unaffected_after != unaffected_before:
        raise QwenResidualAttentionProbeError(
            "O compensation changed an unaffected O prototype slot."
        )
    return {
        "kind": kind,
        "modified_active_slots": sorted(slots),
        "unaffected_o_slots_sha256_before": unaffected_before,
        "unaffected_o_slots_sha256_after": unaffected_after,
        "unaffected_o_slots_byte_exact": True,
        "non_o_tensor_identity_and_version_exact": True,
    }


def run_qwen_residual_attention_probe(
    runtime: nn.Module,
    prompt_input_ids: Sequence[Tensor],
    redirects: Sequence[AttentionRedirect],
    *,
    horizon: int = 8,
    compensation: CompensationMode = "both",
    ridge: float = 1e-6,
    trace_limits: HFCachedAttentionTraceLimits | None = None,
    max_affected_sites: int = 16,
    scoring: ResidualAttentionScoringConfig | None = None,
    gate_policy: ResidualAttentionGatePolicy | None = None,
) -> dict[str, Any]:
    """Measure one atomic redirect batch and restore the incumbent exactly."""

    prompts = tuple(prompt_input_ids)
    redirect_values = tuple(redirects)
    if not prompts or not redirect_values:
        raise QwenResidualAttentionProbeError(
            "At least one prompt and one redirect are required."
        )
    limits = trace_limits or HFCachedAttentionTraceLimits()
    if len(prompts) > limits.max_prompts:
        raise QwenResidualAttentionProbeError(
            f"Prompt count exceeds max_prompts={limits.max_prompts}."
        )
    max_sites = _positive_int(max_affected_sites, "max_affected_sites")
    controls = scoring or ResidualAttentionScoringConfig()
    policy = gate_policy or ResidualAttentionGatePolicy(
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
    mode = str(compensation).strip().lower()
    if mode not in {"none", "scalar", "ridge", "both"}:
        raise QwenResidualAttentionProbeError(
            "compensation must be none, scalar, ridge, or both."
        )
    parent_registry = _registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    bridge, proposal, candidates, tentative = _tentative_codebook(
        runtime, redirect_values
    )
    codebook, used_projections = tentative
    affected_sites, sites_by_source = _affected_sites(
        parent_registry,
        codebook.registry,
        redirect_values,
    )
    if len(affected_sites) > max_sites:
        raise QwenResidualAttentionProbeError(
            f"Affected assignment count exceeds max_affected_sites={max_sites}."
        )
    affected_layers = {layer for layer, _group in affected_sites}
    if len(affected_layers) > limits.max_layers:
        raise QwenResidualAttentionProbeError(
            f"Affected layer count exceeds max_layers={limits.max_layers}."
        )
    parent_o = _parent_o_blocks(parent_registry, affected_sites)
    candidate_ids = tuple(value.candidate_id for value in redirect_values)
    parent_traces, drivers = _collect_parent_traces(
        runtime,
        prompts,
        affected_sites,
        horizon=horizon,
        limits=limits,
    )
    variants: dict[str, Any] = {}
    proposed_snapshot = None
    rollback_exact = False
    reset_proposed_exact = False
    try:
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        proposed_snapshot = qwen3_attention_runtime_snapshot(runtime)
        baseline_traces = _collect_candidate_traces(
            runtime,
            prompts,
            affected_sites,
            drivers,
            horizon=horizon,
            limits=limits,
        )
        variants["uncompensated"] = _variant_report(
            parent_traces,
            baseline_traces,
            candidate_ids=candidate_ids,
            scoring=controls,
            policy=policy,
        )
        fits = _fit_survivor_blocks(
            parent_traces,
            baseline_traces,
            parent_o,
            codebook.registry,
            sites_by_source,
            prompt_count=len(prompts),
            ridge=ridge,
        )
        flat_o = _flat_o(codebook.registry)
        target_slots = {int(value["active_slot"]) for value in fits.values()}
        backups = {slot: flat_o[slot].detach().clone() for slot in sorted(target_slots)}
        proposed_o_sha256 = _o_slots_sha256(flat_o)
        for kind in ("scalar", "ridge"):
            if mode not in {kind, "both"}:
                continue
            edit_audit = _apply_compensation(codebook.registry, fits, kind=kind)
            traces = _collect_candidate_traces(
                runtime,
                prompts,
                affected_sites,
                drivers,
                horizon=horizon,
                limits=limits,
            )
            variants[kind] = {
                **_variant_report(
                    parent_traces,
                    traces,
                    candidate_ids=candidate_ids,
                    scoring=controls,
                    policy=policy,
                ),
                "edit_audit": edit_audit,
            }
            with torch.inference_mode():
                for slot, backup in backups.items():
                    flat_o[slot].copy_(backup)
            if _o_slots_sha256(flat_o) != proposed_o_sha256:
                raise QwenResidualAttentionProbeError(
                    f"{kind} O compensation did not reset exactly."
                )
        reset_proposed_exact = (
            qwen3_attention_runtime_snapshot(runtime) == proposed_snapshot
        )
        if not reset_proposed_exact:
            raise QwenResidualAttentionProbeError(
                "Compensation variants did not restore the tentative fold exactly."
            )
        fits_payload = {
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
    finally:
        restored = attention_surgery._restore_installed(runtime, parent_installation)
        rollback_exact = bool(
            restored and qwen3_attention_runtime_snapshot(runtime) == parent_snapshot
        )
        if not rollback_exact and sys.exc_info()[0] is None:
            raise QwenResidualAttentionProbeError(
                "Probe did not restore the incumbent attention registry exactly."
            )
    affected_payload = [
        {
            "layer_index": layer,
            "materialized_group": group,
            "parent_source_flat_index": _materialized_site_bindings(parent_registry)[
                (layer, group)
            ].source_flat_index,
            "proposed_source_flat_index": _materialized_site_bindings(
                codebook.registry
            )[(layer, group)].source_flat_index,
        }
        for layer, group in affected_sites
    ]
    result = {
        "schema": RUN_SCHEMA,
        "redirects": [value.as_dict() for value in redirect_values],
        "redirect_coordinate_semantics": (
            "prototype provenance layer:physical-group; executable affected groups "
            "are mapped through each layer's group_permutations"
        ),
        "candidate_ids": list(candidate_ids),
        "horizon": horizon,
        "prompt_count": len(prompts),
        "prompt_sha256s": [
            _tensor_sha256(torch.as_tensor(value, dtype=torch.long))
            for value in prompts
        ],
        "compensation_mode": mode,
        "ridge": ridge,
        "parent_attention_snapshot": parent_snapshot.as_dict(),
        "proposed_attention_snapshot": proposed_snapshot.as_dict(),
        "physical_attention_bytes_saved": (
            parent_snapshot.tensor_bytes - proposed_snapshot.tensor_bytes
        ),
        "used_projections": [value.as_dict() for value in used_projections],
        "affected_assignments": affected_payload,
        "affected_assignment_count": len(affected_payload),
        "affected_assignment_sha256": _sha256(affected_payload),
        "survivor_assignment_sets": {
            str(source): [list(site) for site in sites]
            for source, sites in sites_by_source.items()
        },
        "fit_contract": {
            "global_across_every_survivor_assignment": True,
            "selected_group_contribution_targets": True,
            "full_delta_single_group_regression": False,
            "o_design_uses_candidate_actual_context": True,
            "parent_fixed_attention_readout_is_diagnostic_only": True,
            "qk_averaged_before_softmax": False,
            "v_and_o_fold_remain_coupled": True,
            "compensation_edits_existing_o_prototypes_only": True,
        },
        "fits": fits_payload,
        "variants": variants,
        "gate_scope": {
            "class": "bounded-local-probe-only",
            "p35_numerical_floor_calibrated": False,
            "broad_behavior_generation_evaluation_performed": False,
            "warning": (
                "Default absolute limits are fail-closed engineering canaries, not "
                "a calibrated certification threshold; calibrate against known-safe "
                "p35 and dense-oracle lanes before promotion."
            ),
        },
        "parent_trace_provenance": [
            trace.as_dict()
            for key, trace in sorted(parent_traces.items())
            if key[0] == "greedy"
        ],
        "tentative_state_reset_exact": reset_proposed_exact,
        "incumbent_rollback_exact": rollback_exact,
        "checkpoint_saved": False,
        "fold_committed": False,
        "compaction_certified": False,
        "gradient_free": True,
        "training_used": False,
    }
    result["report_sha256"] = _sha256(result)
    return result


def _load_prompt_texts(args: argparse.Namespace) -> tuple[str, ...]:
    values = [str(value) for value in args.prompt]
    if args.prompts_json is not None:
        payload = json.loads(args.prompts_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(
            not isinstance(item, str) for item in payload
        ):
            raise QwenResidualAttentionProbeError(
                "--prompts-json must contain a JSON list of strings."
            )
        values.extend(payload)
    if not values:
        raise QwenResidualAttentionProbeError("Provide --prompt or --prompts-json.")
    return tuple(values)


def _dtype(value: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def _load_runtime_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    try:
        import transformers
    except ImportError as error:
        raise QwenResidualAttentionProbeError("transformers is required.") from error
    tokenizer_name = args.tokenizer or args.model
    if not tokenizer_name:
        raise QwenResidualAttentionProbeError(
            "--tokenizer is required with --codebook-checkpoint."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if args.codebook_checkpoint is not None:
        runtime = load_qwen3_codebook_checkpoint(
            args.codebook_checkpoint,
            device="cpu",
        )
        runtime.to(dtype=_dtype(args.dtype))
        runtime.to(args.device)
        manifest_path = args.codebook_checkpoint / "manifest.json"
        source = {
            "kind": "codebook-checkpoint",
            "path": str(args.codebook_checkpoint.resolve()),
            "manifest_sha256": _file_sha256(manifest_path),
        }
    else:
        runtime = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=_dtype(args.dtype),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        runtime.to(args.device)
        runtime.eval()
        layers = int(runtime.config.num_hidden_layers)
        runtime = prepare_qwen3_codebook_model(
            runtime,
            tokenizer,
            tokenizer,
            attention_prototypes_per_group=layers,
            neuron_prototypes_per_neuron=layers,
        )
        source = {"kind": "dense-prepared-identity", "model": args.model}
    return runtime.eval(), tokenizer, source


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--codebook-checkpoint", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--candidate",
        type=parse_attention_redirect,
        action="append",
        required=True,
        help=(
            "SOURCE_LAYER:SOURCE_PHYSICAL_GROUP="
            "TARGET_LAYER:TARGET_PHYSICAL_GROUP; repeat for an atomic batch"
        ),
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--compensation", choices=("none", "scalar", "ridge", "both"), default="both"
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-affected-sites", type=int, default=16)
    parser.add_argument("--max-affected-layers", type=int, default=8)
    parser.add_argument("--max-capture-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--logit-rtol", type=float, default=5e-3)
    parser.add_argument("--logit-atol", type=float, default=5e-3)
    parser.add_argument("--context-rtol", type=float, default=8e-3)
    parser.add_argument("--context-atol", type=float, default=8e-3)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime, tokenizer, source = _load_runtime_and_tokenizer(args)
    texts = _load_prompt_texts(args)
    if len(texts) > args.max_prompts:
        raise QwenResidualAttentionProbeError(
            f"Prompt count exceeds --max-prompts={args.max_prompts}."
        )
    prompts = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_tokens,
        )["input_ids"]
        prompts.append(encoded)
    limits = HFCachedAttentionTraceLimits(
        max_prompts=args.max_prompts,
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_layers=args.max_affected_layers,
        max_groups=1,
        max_model_devices=1,
        max_capture_bytes=args.max_capture_bytes,
        logit_rtol=args.logit_rtol,
        logit_atol=args.logit_atol,
        context_rtol=args.context_rtol,
        context_atol=args.context_atol,
    )
    report = run_qwen_residual_attention_probe(
        runtime,
        prompts,
        args.candidate,
        horizon=args.horizon,
        compensation=args.compensation,
        ridge=args.ridge,
        trace_limits=limits,
        max_affected_sites=args.max_affected_sites,
    )
    report["source"] = source
    report["parameter_report"] = qwen3_codebook_parameter_report(runtime).as_dict()
    report["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "dtype": args.dtype,
        "numeric_tolerances": {
            "logit_rtol": args.logit_rtol,
            "logit_atol": args.logit_atol,
            "context_rtol": args.context_rtol,
            "context_atol": args.context_atol,
        },
    }
    report["report_sha256"] = _sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    _atomic_write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "rollback_exact": report["incumbent_rollback_exact"],
                "affected_assignments": report["affected_assignment_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
