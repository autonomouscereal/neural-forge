#!/usr/bin/env python3
"""Search per-assignment V/O scales for the exact p35 L19/G4 redirect.

The tentative structural edit is fixed to ``L19/G4 -> L21/G6``.  Candidate
edits touch only the existing V/O role-scale scalars for the executable
physical assignment L19/G4.  Its current Q/K scales remain byte-equivalent.
Selection uses cached output function on calibration prompts only; held-out
function is ordered separately and cannot name or change the winner.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_residual_attention_probe as base
from neural_forge.hf_functional_decode_trace import (
    FunctionalDecodeLimits,
    FunctionalDecodeTrace,
    collect_functional_decode_trace,
    trace_mapping,
)
from neural_forge.hf_qwen_attention_surgery import (
    Qwen3AttentionRuntimeSnapshot,
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report
from neural_forge.hf_residual_attention_functional import (
    build_residual_attention_functional_comparison,
)
from neural_forge.hf_residual_attention_role_scale import (
    AttentionRoleScales,
    InstalledAttentionRoleScales,
    install_attention_role_scales,
)


RUN_SCHEMA = "neural-forge-qwen-p35-vo-role-scale-functional-search-v1"
EXACT_REDIRECT = base.AttentionRedirect(19, 4, 21, 6)
EXACT_SCALE_SITE = (19, 4)
EXACT_REMOVED_SOURCE = 19 * 8 + 4
EXACT_SURVIVOR_SOURCE = 21 * 8 + 6
EXACT_AFFECTED_SITES = ((2, 1), (19, 4), (21, 6))
FunctionalTraces = Mapping[
    tuple[str, str, tuple[int, ...]], FunctionalDecodeTrace
]


class QwenP35VORoleScaleSearchError(RuntimeError):
    """The bounded role-scale search failed a lineage or rollback contract."""


_FLOAT_TOKEN = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
_VO_SCALE_TOKEN = re.compile(rf"^{_FLOAT_TOKEN}:{_FLOAT_TOKEN}$")


class _VOScaleArgumentParser(argparse.ArgumentParser):
    """Let a negative ``V:O`` token serve as an option value.

    ``argparse`` recognizes a lone negative number but treats a negative
    colon-delimited pair as another option. The exact numeric grammar is safe
    to classify as a value before normal optional parsing.
    """

    def _parse_optional(self, arg_string: str) -> Any:
        if _VO_SCALE_TOKEN.fullmatch(arg_string):
            return None
        return super()._parse_optional(arg_string)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenP35VORoleScaleSearchError(f"{name} must be positive.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QwenP35VORoleScaleSearchError(
            f"{name} must be positive."
        ) from error
    if result <= 0 or result != value:
        raise QwenP35VORoleScaleSearchError(f"{name} must be positive.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QwenP35VORoleScaleSearchError(
            f"{name} must be finite and positive."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QwenP35VORoleScaleSearchError(
            f"{name} must be finite and positive."
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise QwenP35VORoleScaleSearchError(
            f"{name} must be finite and positive."
        )
    return result


@dataclass(frozen=True, order=True, slots=True)
class VOScalePair:
    v: float
    o: float

    def __post_init__(self) -> None:
        for name in ("v", "o"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise QwenP35VORoleScaleSearchError(
                    f"{name.upper()} scale must be finite."
                )
            object.__setattr__(self, name, float(value))

    def as_dict(self) -> dict[str, float]:
        return {"v": self.v, "o": self.o}


def parse_vo_scale(value: str) -> VOScalePair:
    parts = str(value).strip().split(":")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("V/O scale must be V:O.")
    try:
        return VOScalePair(float(parts[0]), float(parts[1]))
    except (ValueError, QwenP35VORoleScaleSearchError) as error:
        raise argparse.ArgumentTypeError(
            "V/O scale must contain two finite numbers as V:O."
        ) from error


@dataclass(frozen=True, slots=True)
class _ScaleCandidate:
    candidate_id: str
    pair: VOScalePair
    tentative_incumbent: bool
    explicit_request_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scales": self.pair.as_dict(),
            "tentative_incumbent": self.tentative_incumbent,
            "explicit_request_indices": list(self.explicit_request_indices),
        }


def _normalized_pair(pair: VOScalePair, *, dtype: torch.dtype) -> VOScalePair:
    values = torch.tensor((pair.v, pair.o), dtype=dtype).tolist()
    return VOScalePair(
        0.0 if float(values[0]) == 0.0 else float(values[0]),
        0.0 if float(values[1]) == 0.0 else float(values[1]),
    )


def _bounded_pair(pair: VOScalePair, *, maximum: float) -> VOScalePair:
    if abs(pair.v) > maximum or abs(pair.o) > maximum:
        raise QwenP35VORoleScaleSearchError(
            f"V/O scales must have absolute value <= {maximum}."
        )
    return pair


def _candidate_id(pair: VOScalePair) -> str:
    return f"vo-scale-{base._sha256(pair.as_dict())[:20]}"


def _prepare_scale_candidates(
    explicit_pairs: Sequence[VOScalePair],
    incumbent_pair: VOScalePair,
    *,
    dtype: torch.dtype,
    max_candidates: int,
    max_abs_scale: float,
) -> tuple[_ScaleCandidate, ...]:
    maximum_count = _positive_int(max_candidates, "max_candidates")
    maximum_scale = _positive_finite(max_abs_scale, "max_abs_scale")
    incumbent = _bounded_pair(
        _normalized_pair(incumbent_pair, dtype=dtype),
        maximum=maximum_scale,
    )
    indices: dict[VOScalePair, list[int]] = {}
    for index, raw in enumerate(explicit_pairs):
        if not isinstance(raw, VOScalePair):
            raise QwenP35VORoleScaleSearchError(
                "Explicit candidates must be VOScalePair values."
            )
        value = _bounded_pair(
            _normalized_pair(raw, dtype=dtype),
            maximum=maximum_scale,
        )
        indices.setdefault(value, []).append(index)
    indices.setdefault(incumbent, [])
    if len(indices) > maximum_count:
        raise QwenP35VORoleScaleSearchError(
            f"Unique candidates including the incumbent exceed {maximum_count}."
        )
    candidates = tuple(
        _ScaleCandidate(
            candidate_id=_candidate_id(pair),
            pair=pair,
            tentative_incumbent=pair == incumbent,
            explicit_request_indices=tuple(indices[pair]),
        )
        for pair in indices
    )
    if len({value.candidate_id for value in candidates}) != len(candidates):
        raise QwenP35VORoleScaleSearchError("Candidate ID collision detected.")
    incumbent_candidate = next(
        value for value in candidates if value.tentative_incumbent
    )
    return (
        incumbent_candidate,
        *sorted(
            (
                value
                for value in candidates
                if not value.tentative_incumbent
            ),
            key=lambda value: value.candidate_id,
        ),
    )


def _prompt_sha256s(prompts: Sequence[Tensor]) -> tuple[str, ...]:
    return tuple(
        base._tensor_sha256(torch.as_tensor(value, dtype=torch.long))
        for value in prompts
    )


def _validate_prompt_splits(
    calibration: Sequence[Tensor],
    heldout: Sequence[Tensor],
    *,
    max_prompts_per_split: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    maximum = _positive_int(max_prompts_per_split, "max_prompts_per_split")
    calibration_values = tuple(calibration)
    heldout_values = tuple(heldout)
    if not calibration_values or not heldout_values:
        raise QwenP35VORoleScaleSearchError(
            "Calibration and held-out prompt sets must both be non-empty."
        )
    if max(len(calibration_values), len(heldout_values)) > maximum:
        raise QwenP35VORoleScaleSearchError(
            "A prompt split exceeds max_prompts_per_split."
        )
    calibration_hashes = _prompt_sha256s(calibration_values)
    heldout_hashes = _prompt_sha256s(heldout_values)
    if len(set(calibration_hashes)) != len(calibration_hashes) or len(
        set(heldout_hashes)
    ) != len(heldout_hashes):
        raise QwenP35VORoleScaleSearchError(
            "Each prompt split must contain unique token rows."
        )
    if set(calibration_hashes) & set(heldout_hashes):
        raise QwenP35VORoleScaleSearchError(
            "Calibration and held-out token rows must be disjoint."
        )
    return calibration_values, heldout_values


def _canonical_assignment(
    registry: Any,
    *,
    layer_index: int,
    physical_group: int,
) -> int:
    if layer_index < 0 or layer_index >= int(registry.logical_layers):
        raise QwenP35VORoleScaleSearchError("Scale layer is out of range.")
    if physical_group < 0 or physical_group >= int(registry.key_value_heads):
        raise QwenP35VORoleScaleSearchError("Scale physical group is out of range.")
    permutation = (
        registry.group_permutations[layer_index]
        .detach()
        .cpu()
        .to(torch.long)
    )
    matches = torch.nonzero(
        permutation == physical_group,
        as_tuple=False,
    ).reshape(-1)
    if int(matches.numel()) != 1:
        raise QwenP35VORoleScaleSearchError(
            "Physical group does not resolve to exactly one canonical assignment."
        )
    return int(matches.item())


def _assignment_scales(
    registry: Any,
    *,
    layer_index: int,
    canonical_group: int,
) -> AttentionRoleScales:
    scales = registry.role_scales
    if scales is None or tuple(scales.shape) != (
        int(registry.logical_layers),
        int(registry.key_value_heads),
        4,
    ):
        raise QwenP35VORoleScaleSearchError(
            "Tentative p35 redirect must expose existing assignment role scales."
        )
    row = scales[layer_index, canonical_group].detach().cpu().tolist()
    return AttentionRoleScales(*map(float, row))


def _runtime_state_identity_sha256(runtime: nn.Module) -> str:
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
    return base._sha256(values)


def _assert_parent_driver_top1(
    traces: FunctionalTraces,
    *,
    split: str,
) -> None:
    forced = [trace for trace in traces.values() if trace.lane == "forced"]
    if not forced:
        raise QwenP35VORoleScaleSearchError(
            f"Parent {split} traces contain no forced lane."
        )
    invalid = [
        (trace.sequence_id, step.step_index)
        for trace in forced
        for step in trace.steps
        if step.selected_token_id != step.greedy_token_id
    ]
    if invalid:
        raise QwenP35VORoleScaleSearchError(
            f"Parent {split} forced drivers are not all parent top-1 actions: "
            f"{invalid[:8]!r}."
        )


def _collect_parent_split(
    runtime: nn.Module,
    prompts: Sequence[Tensor],
    *,
    split: str,
    horizon: int,
    limits: FunctionalDecodeLimits,
) -> tuple[FunctionalTraces, dict[int, tuple[int, ...]]]:
    traces: list[FunctionalDecodeTrace] = []
    drivers: dict[int, tuple[int, ...]] = {}
    for prompt_index, prompt in enumerate(prompts):
        sequence_id = f"prompt-{prompt_index}"
        greedy = collect_functional_decode_trace(
            runtime,
            prompt,
            horizon=horizon,
            sequence_id=sequence_id,
            lane="greedy",
            limits=limits,
        )
        drivers[prompt_index] = greedy.selected_token_ids
        forced = collect_functional_decode_trace(
            runtime,
            prompt,
            horizon=horizon,
            sequence_id=sequence_id,
            lane="forced",
            forced_token_ids=greedy.selected_token_ids,
            limits=limits,
        )
        traces.extend((forced, greedy))
    mapping = trace_mapping(traces)
    _assert_parent_driver_top1(mapping, split=split)
    return mapping, drivers


def _collect_candidate_split(
    runtime: nn.Module,
    prompts: Sequence[Tensor],
    drivers: Mapping[int, tuple[int, ...]],
    *,
    horizon: int,
    limits: FunctionalDecodeLimits,
) -> FunctionalTraces:
    traces: list[FunctionalDecodeTrace] = []
    if set(drivers) != set(range(len(prompts))):
        raise QwenP35VORoleScaleSearchError(
            "Candidate split drivers do not match its prompt rows."
        )
    for prompt_index, prompt in enumerate(prompts):
        sequence_id = f"prompt-{prompt_index}"
        forced = collect_functional_decode_trace(
            runtime,
            prompt,
            horizon=horizon,
            sequence_id=sequence_id,
            lane="forced",
            forced_token_ids=drivers[prompt_index],
            limits=limits,
        )
        greedy = collect_functional_decode_trace(
            runtime,
            prompt,
            horizon=horizon,
            sequence_id=sequence_id,
            lane="greedy",
            limits=limits,
        )
        traces.extend((forced, greedy))
    return trace_mapping(traces)


def _trace_manifest(traces: FunctionalTraces) -> dict[str, Any]:
    return {
        f"{lane}:{sequence_id}": trace.as_dict()
        for (lane, sequence_id, _prompt), trace in sorted(traces.items())
    }


def _build_functional_comparison(
    parent_calibration: FunctionalTraces,
    parent_heldout: FunctionalTraces,
    candidate_calibration: Mapping[str, FunctionalTraces],
    candidate_heldout: Mapping[str, FunctionalTraces],
    *,
    top_k: int,
) -> dict[str, Any]:
    return build_residual_attention_functional_comparison(
        parent_calibration,
        parent_heldout,
        candidate_calibration,
        candidate_heldout,
        top_k=top_k,
    )


def _scale_change_audit(
    before: Tensor,
    after: Tensor,
    *,
    layer_index: int,
    canonical_group: int,
) -> dict[str, Any]:
    if tuple(before.shape) != tuple(after.shape):
        raise QwenP35VORoleScaleSearchError(
            "Role-scale shape changed during one candidate."
        )
    changed = tuple(
        tuple(map(int, value))
        for value in torch.nonzero(
            before.detach().cpu() != after.detach().cpu(),
            as_tuple=False,
        ).tolist()
    )
    allowed = {
        (layer_index, canonical_group, 2),
        (layer_index, canonical_group, 3),
    }
    if not set(changed).issubset(allowed):
        raise QwenP35VORoleScaleSearchError(
            "Candidate changed a role scale outside target V/O scalars."
        )
    return {
        "changed_coordinates": [list(value) for value in changed],
        "allowed_coordinates": [list(value) for value in sorted(allowed)],
        "only_target_v_o_scales_changed": True,
        "target_q_k_scales_unchanged": True,
        "all_other_assignment_scales_unchanged": True,
    }


def _audit_lineage(
    parent: Qwen3AttentionRuntimeSnapshot,
    proposed: Qwen3AttentionRuntimeSnapshot,
    *,
    redirect: base.AttentionRedirect,
    affected_sites: Sequence[base.Site],
    require_exact_p35: bool,
) -> dict[str, Any]:
    groups = int(parent.key_value_heads)
    removed_source = redirect.source_layer * groups + redirect.source_group
    survivor_source = redirect.target_layer * groups + redirect.target_group
    parent_sources = set(parent.prototype_source_indices)
    proposed_sources = set(proposed.prototype_source_indices)
    if (
        proposed.prototype_count != parent.prototype_count - 1
        or len(proposed_sources) != proposed.prototype_count
        or proposed_sources != parent_sources.difference({removed_source})
        or survivor_source not in proposed_sources
    ):
        raise QwenP35VORoleScaleSearchError(
            "Tentative redirect is not an exact one-prototype source removal."
        )
    if require_exact_p35 and (
        parent.logical_layers != 36
        or groups != 8
        or parent.prototype_count != 280
        or proposed.prototype_count != 279
        or removed_source != EXACT_REMOVED_SOURCE
        or survivor_source != EXACT_SURVIVOR_SOURCE
        or tuple(affected_sites) != EXACT_AFFECTED_SITES
    ):
        raise QwenP35VORoleScaleSearchError(
            "Exact mode requires the p35 280 -> 279 L19/G4 -> L21/G6 lineage."
        )
    return {
        "parent_prototype_count": parent.prototype_count,
        "proposed_prototype_count": proposed.prototype_count,
        "prototype_count_removed": 1,
        "removed_source_flat_index": removed_source,
        "survivor_source_flat_index": survivor_source,
        "source_set_is_parent_minus_only_removed_source": True,
        "exact_p35_280_to_279_verified": require_exact_p35,
    }


def run_qwen_p35_vo_role_scale_search(
    runtime: nn.Module,
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    explicit_vo_scales: Sequence[VOScalePair] = (),
    *,
    redirect: base.AttentionRedirect = EXACT_REDIRECT,
    require_exact_p35: bool = True,
    horizon: int = 8,
    top_k: int = 5,
    max_candidates: int = 16,
    max_abs_scale: float = 16.0,
    max_prompts_per_split: int = 4,
    trace_limits: FunctionalDecodeLimits | None = None,
) -> dict[str, Any]:
    """Search explicit per-assignment V/O scales and restore p35 exactly."""

    if require_exact_p35 and redirect != EXACT_REDIRECT:
        raise QwenP35VORoleScaleSearchError(
            "Exact mode is fixed to L19/G4 -> L21/G6."
        )
    horizon_value = _positive_int(horizon, "horizon")
    top_k_value = _positive_int(top_k, "top_k")
    max_candidates_value = _positive_int(max_candidates, "max_candidates")
    max_abs_scale_value = _positive_finite(max_abs_scale, "max_abs_scale")
    max_prompts_value = _positive_int(
        max_prompts_per_split,
        "max_prompts_per_split",
    )
    explicit_pairs = tuple(explicit_vo_scales)
    limits = trace_limits or FunctionalDecodeLimits()
    if horizon_value > limits.max_horizon:
        raise QwenP35VORoleScaleSearchError(
            "horizon exceeds the functional trace limit."
        )
    calibration_prompts, heldout_prompts = _validate_prompt_splits(
        calibration_prompt_input_ids,
        heldout_prompt_input_ids,
        max_prompts_per_split=max_prompts_value,
    )
    if require_exact_p35 and redirect.source != EXACT_SCALE_SITE:
        raise QwenP35VORoleScaleSearchError(
            "Exact scale target must be executable physical L19/G4."
        )

    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_runtime_identity = _runtime_state_identity_sha256(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    _bridge, _proposal, proposal_candidates, tentative = base._tentative_codebook(
        runtime,
        (redirect,),
    )
    codebook, used_projections = tentative
    proposed_registry = codebook.registry
    affected_sites, sites_by_source = base._affected_sites(
        parent_registry,
        proposed_registry,
        (redirect,),
    )
    scale_layer, scale_physical = redirect.source
    if (scale_layer, scale_physical) not in affected_sites:
        raise QwenP35VORoleScaleSearchError(
            "Scale target is not an affected executable assignment."
        )
    canonical_group = _canonical_assignment(
        proposed_registry,
        layer_index=scale_layer,
        physical_group=scale_physical,
    )
    target_binding = base._materialized_site_bindings(proposed_registry)[
        (scale_layer, scale_physical)
    ]
    survivor_source = (
        redirect.target_layer * int(parent_snapshot.key_value_heads)
        + redirect.target_group
    )
    if (
        target_binding.assignment_axis != canonical_group
        or target_binding.source_flat_index != survivor_source
    ):
        raise QwenP35VORoleScaleSearchError(
            "Physical scale target does not resolve to the redirected survivor."
        )
    incumbent_scales = _assignment_scales(
        proposed_registry,
        layer_index=scale_layer,
        canonical_group=canonical_group,
    )
    role_scales = proposed_registry.role_scales
    assert role_scales is not None
    candidates = _prepare_scale_candidates(
        explicit_pairs,
        VOScalePair(incumbent_scales.v, incumbent_scales.o),
        dtype=role_scales.dtype,
        max_candidates=max_candidates_value,
        max_abs_scale=max_abs_scale_value,
    )

    parent_calibration, calibration_drivers = _collect_parent_split(
        runtime,
        calibration_prompts,
        split="calibration",
        horizon=horizon_value,
        limits=limits,
    )
    parent_heldout, heldout_drivers = _collect_parent_split(
        runtime,
        heldout_prompts,
        split="heldout",
        horizon=horizon_value,
        limits=limits,
    )
    if (
        qwen3_attention_runtime_snapshot(runtime) != parent_snapshot
        or _runtime_state_identity_sha256(runtime) != parent_runtime_identity
    ):
        raise QwenP35VORoleScaleSearchError(
            "Parent functional trace collection changed the p35 runtime."
        )

    candidate_reports: dict[str, Any] = {}
    candidate_calibration: dict[str, FunctionalTraces] = {}
    candidate_heldout: dict[str, FunctionalTraces] = {}
    proposed_snapshot: Qwen3AttentionRuntimeSnapshot | None = None
    proposed_scale_baseline = role_scales.detach().clone()
    live_scale: InstalledAttentionRoleScales | None = None
    proposed_reset_exact = False
    incumbent_rollback_exact = False
    try:
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        proposed_snapshot = qwen3_attention_runtime_snapshot(runtime)
        lineage = _audit_lineage(
            parent_snapshot,
            proposed_snapshot,
            redirect=redirect,
            affected_sites=affected_sites,
            require_exact_p35=require_exact_p35,
        )
        for candidate in candidates:
            live_scale = install_attention_role_scales(
                proposed_registry,
                layer_index=scale_layer,
                physical_group=scale_physical,
                scales=AttentionRoleScales(
                    q=incumbent_scales.q,
                    k=incumbent_scales.k,
                    v=candidate.pair.v,
                    o=candidate.pair.o,
                ),
                max_abs_scale=max_abs_scale_value,
            )
            try:
                installation = live_scale.assert_live()
                if (
                    installation.canonical_group != canonical_group
                    or installation.before_scales != incumbent_scales
                    or installation.installed_scales.q != incumbent_scales.q
                    or installation.installed_scales.k != incumbent_scales.k
                ):
                    raise QwenP35VORoleScaleSearchError(
                        "Candidate changed Q/K or resolved a different assignment."
                    )
                assert proposed_registry.role_scales is not None
                change_audit = _scale_change_audit(
                    proposed_scale_baseline,
                    proposed_registry.role_scales,
                    layer_index=scale_layer,
                    canonical_group=canonical_group,
                )
                calibration_traces = _collect_candidate_split(
                    runtime,
                    calibration_prompts,
                    calibration_drivers,
                    horizon=horizon_value,
                    limits=limits,
                )
                heldout_traces = _collect_candidate_split(
                    runtime,
                    heldout_prompts,
                    heldout_drivers,
                    horizon=horizon_value,
                    limits=limits,
                )
                live_scale.assert_live()
                candidate_calibration[candidate.candidate_id] = calibration_traces
                candidate_heldout[candidate.candidate_id] = heldout_traces
                candidate_reports[candidate.candidate_id] = {
                    **candidate.as_dict(),
                    "full_installed_scales": installation.installed_scales.as_dict(),
                    "installation_audit": installation.as_dict(),
                    "scale_change_audit": change_audit,
                    "trace_provenance": {
                        "calibration": _trace_manifest(calibration_traces),
                        "heldout": _trace_manifest(heldout_traces),
                    },
                }
            finally:
                if live_scale is not None and live_scale.is_applied:
                    live_scale.rollback()
                live_scale = None
            if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
                raise QwenP35VORoleScaleSearchError(
                    f"Candidate {candidate.candidate_id} did not roll back exactly."
                )
            candidate_reports[candidate.candidate_id][
                "tentative_reset_exact"
            ] = True
        proposed_reset_exact = (
            qwen3_attention_runtime_snapshot(runtime) == proposed_snapshot
        )
    finally:
        failure_in_flight = sys.exc_info()[0] is not None
        restoration_errors: list[Exception] = []
        if live_scale is not None and live_scale.is_applied:
            try:
                live_scale.rollback()
            except Exception as error:  # pragma: no cover - defensive cleanup
                restoration_errors.append(error)
        try:
            restored = attention_surgery._restore_installed(
                runtime,
                parent_installation,
            )
            incumbent_rollback_exact = bool(
                restored
                and qwen3_attention_runtime_snapshot(runtime) == parent_snapshot
                and _runtime_state_identity_sha256(runtime)
                == parent_runtime_identity
            )
            if not incumbent_rollback_exact:
                restoration_errors.append(
                    QwenP35VORoleScaleSearchError(
                        "Search did not restore the incumbent p35 runtime exactly."
                    )
                )
        except Exception as error:  # pragma: no cover - defensive cleanup
            restoration_errors.append(error)
        if restoration_errors and not failure_in_flight:
            raise QwenP35VORoleScaleSearchError(
                "Search cleanup failed: "
                + "; ".join(str(error) for error in restoration_errors)
            ) from restoration_errors[0]

    if proposed_snapshot is None or not proposed_reset_exact:
        raise QwenP35VORoleScaleSearchError(
            "Search completed without exact tentative-state rollback."
        )
    functional_comparison = _build_functional_comparison(
        parent_calibration,
        parent_heldout,
        candidate_calibration,
        candidate_heldout,
        top_k=top_k_value,
    )
    for candidate_id, functional in functional_comparison["variants"].items():
        candidate_reports[candidate_id]["functional"] = functional

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
            "proposed_active_slot": proposed_bindings[
                (layer, group)
            ].active_slot,
        }
        for layer, group in affected_sites
    ]
    result = {
        "schema": RUN_SCHEMA,
        "redirect": redirect.as_dict(),
        "exact_p35_mode": require_exact_p35,
        "horizon": horizon_value,
        "top_k": top_k_value,
        "max_candidates": max_candidates_value,
        "max_abs_scale": max_abs_scale_value,
        "candidate_count": len(candidates),
        "candidate_evaluation_order": [
            value.candidate_id for value in candidates
        ],
        "explicit_vo_scale_requests": [
            value.as_dict() for value in explicit_pairs
        ],
        "scale_target": {
            "layer_index": scale_layer,
            "physical_group": scale_physical,
            "canonical_assignment_axis": canonical_group,
            "proposed_source_flat_index": target_binding.source_flat_index,
            "proposed_active_slot": target_binding.active_slot,
            "tentative_incumbent_scales": incumbent_scales.as_dict(),
            "q_k_fixed_to_tentative_incumbent_for_every_candidate": True,
            "only_existing_v_o_scalars_searchable": True,
        },
        "prompt_splits": {
            "calibration": {
                "count": len(calibration_prompts),
                "prompt_sha256s": list(_prompt_sha256s(calibration_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(calibration_drivers.items())
                },
                "used_for_selection": True,
            },
            "heldout": {
                "count": len(heldout_prompts),
                "prompt_sha256s": list(_prompt_sha256s(heldout_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(heldout_drivers.items())
                },
                "used_for_selection": False,
                "evaluation_only": True,
            },
            "token_rows_disjoint": True,
            "parent_traces_collected_once_per_lane_and_prompt": True,
        },
        "parent_attention_snapshot": parent_snapshot.as_dict(),
        "proposed_attention_snapshot": proposed_snapshot.as_dict(),
        "lineage_audit": lineage,
        "accounting": {
            "parent_attention_tensor_bytes": parent_snapshot.tensor_bytes,
            "proposed_attention_tensor_bytes": proposed_snapshot.tensor_bytes,
            "physical_attention_bytes_saved": (
                parent_snapshot.tensor_bytes - proposed_snapshot.tensor_bytes
            ),
            "role_scale_search_adds_parameters": False,
            "role_scale_search_changes_parameter_count": False,
        },
        "affected_assignments": affected_payload,
        "affected_assignment_count": len(affected_payload),
        "affected_assignment_sha256": base._sha256(affected_payload),
        "survivor_assignment_sets": {
            str(source): [list(site) for site in sites]
            for source, sites in sites_by_source.items()
        },
        "used_redirect_projections": [
            value.as_dict() for value in used_projections
        ],
        "proposal_candidate_count": len(proposal_candidates),
        "parent_trace_provenance": {
            "calibration": _trace_manifest(parent_calibration),
            "heldout": _trace_manifest(parent_heldout),
        },
        "candidates": candidate_reports,
        "functional_comparison": functional_comparison,
        "selection_contract": {
            "calibration_output_function_only": True,
            "parent_driver_top1_rate_required": 1.0,
            "heldout_evaluation_only_and_winnerless": True,
            "heldout_excluded_from_selection": True,
            "internal_tensor_similarity_used": False,
            "functional_collector_records_tokens_and_logits_only": True,
            "candidate_input_order_not_a_ranking_input": True,
            "tentative_incumbent_always_included": True,
            "optimizer_constructed": False,
            "backward_called": False,
        },
        "transaction_contract": {
            "physical_target_resolved_to_one_canonical_assignment": True,
            "every_candidate_keeps_target_q_k_scales_unchanged": True,
            "every_candidate_changes_only_existing_target_v_o_scalars": True,
            "every_candidate_non_scale_registry_identity_verified": True,
            "every_candidate_parameter_identity_verified": True,
            "every_candidate_rolled_back_before_next_candidate": True,
            "parent_runtime_identity_restored_after_tentative_search": True,
        },
        "tentative_state_reset_exact": proposed_reset_exact,
        "incumbent_runtime_restored_exact": incumbent_rollback_exact,
        "checkpoint_saved": False,
        "fold_committed": False,
        "compaction_certified": False,
        "gradient_free": True,
        "training_used": False,
    }
    result["report_sha256"] = base._sha256(result)
    return result


def _load_texts(
    inline: Sequence[str],
    json_path: Path | None,
    name: str,
) -> tuple[str, ...]:
    values = [str(value) for value in inline]
    if json_path is not None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(
            not isinstance(item, str) for item in payload
        ):
            raise QwenP35VORoleScaleSearchError(
                f"{name} JSON must contain a list of strings."
            )
        values.extend(payload)
    if not values:
        raise QwenP35VORoleScaleSearchError(f"Provide at least one {name}.")
    return tuple(values)


def _encode_prompts(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_prompt_tokens: int,
) -> tuple[Tensor, ...]:
    prompts = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )["input_ids"]
        if not isinstance(encoded, Tensor) or tuple(encoded.shape[:1]) != (1,):
            raise QwenP35VORoleScaleSearchError(
                "Tokenizer did not return one input_ids row."
            )
        prompts.append(encoded.to(dtype=torch.long))
    return tuple(prompts)


def build_parser() -> argparse.ArgumentParser:
    parser = _VOScaleArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--vo-scale",
        type=parse_vo_scale,
        action="append",
        default=[],
        help="Explicit V:O pair; repeat for a bounded candidate set.",
    )
    parser.add_argument("--calibration-prompt", action="append", default=[])
    parser.add_argument("--calibration-prompts-json", type=Path)
    parser.add_argument("--heldout-prompt", action="append", default=[])
    parser.add_argument("--heldout-prompts-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-abs-scale", type=float, default=16.0)
    parser.add_argument("--max-prompts-per-split", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-vocabulary", type=int, default=1_000_000)
    parser.add_argument("--max-logit-bytes", type=int, default=256 * 1024 * 1024)
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
    if max(len(calibration_texts), len(heldout_texts)) > args.max_prompts_per_split:
        raise QwenP35VORoleScaleSearchError(
            "A prompt split exceeds --max-prompts-per-split."
        )
    runtime, tokenizer, source = base._load_runtime_and_tokenizer(args)
    calibration_prompts = _encode_prompts(
        tokenizer,
        calibration_texts,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    heldout_prompts = _encode_prompts(
        tokenizer,
        heldout_texts,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    limits = FunctionalDecodeLimits(
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_vocabulary=args.max_vocabulary,
        max_logit_bytes=args.max_logit_bytes,
        max_model_devices=1,
    )
    report = run_qwen_p35_vo_role_scale_search(
        runtime,
        calibration_prompts,
        heldout_prompts,
        args.vo_scale,
        horizon=args.horizon,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        max_abs_scale=args.max_abs_scale,
        max_prompts_per_split=args.max_prompts_per_split,
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
                "winner": report["functional_comparison"][
                    "calibration_selection"
                ]["winner_candidate_id"],
                "runtime_restored": report["incumbent_runtime_restored_exact"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
