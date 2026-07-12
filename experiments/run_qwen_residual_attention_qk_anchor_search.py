#!/usr/bin/env python3
"""Rank real Q/K anchors for the exact p35 L19/G4 -> L21/G6 redirect.

The p35 medoid redirect selects L21/G6 from static geometry.  This bounded
probe instead tests an explicit list of still-active *real* Q/K pairs under
cached multi-token execution.  For each anchor it:

* previews the same one-prototype p35 redirect (280 -> 279 prototypes);
* copies Q and K together from one active survivor into the L21/G6 slot;
* leaves V/O at the redirect incumbent, with an optional calibration-only O
  ridge control;
* measures all L21/G6-shared executable assignments on disjoint calibration
  and held-out forced-parent-token and greedy lanes; and
* ranks only from held-out attention distributions, residual/cache evolution,
  and action damage.

Q/K tensors are never averaged before softmax.  Every logical layer's actual
q_norm/k_norm, role scales, and group permutation are content-addressed in the
report.  The probe is zero-gradient, rolls back exactly, writes only JSON, and
cannot save or commit a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_residual_attention_probe as base
import run_qwen_residual_attention_value_probe as value_base
from neural_forge.hf_cached_attention_trace import (
    HFCachedAttentionTrace,
    HFCachedAttentionTraceLimits,
)
from neural_forge.hf_qwen_attention_surgery import (
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report
from neural_forge.hf_qwen_codebook_model import (
    Qwen3CodebookError,
    canonicalize_qwen3_tokenizer_contract,
)
from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionGatePolicy,
    ResidualAttentionScoringConfig,
)
from neural_forge.hf_residual_attention_qk_anchor import (
    InstalledRealQKAnchor,
    RealQKAnchor,
    ResidualAttentionQKAnchorError,
    audit_qk_anchor_assignments,
    install_real_qk_anchor,
)


RUN_SCHEMA = "neural-forge-qwen-p35-qk-anchor-search-v1"
RANKING_SCHEMA = "neural-forge-heldout-causal-qk-anchor-ranking-v1"
EXACT_REDIRECT = base.AttentionRedirect(19, 4, 21, 6)
EXACT_DESTINATION_SOURCE = 21 * 8 + 6
EXACT_REMOVED_SOURCE = 19 * 8 + 4
_ANCHOR_PATTERN = re.compile(r"^(?:L)?(\d+)(?::|/|G)(\d+)$", re.IGNORECASE)
SplitTraces = Mapping[base.TraceKey, HFCachedAttentionTrace]


class QwenResidualAttentionQKAnchorSearchError(RuntimeError):
    """The bounded Q/K anchor search failed a causal or rollback invariant."""


def parse_real_qk_anchor(value: str) -> RealQKAnchor:
    match = _ANCHOR_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "Q/K anchor must be LAYER:PHYSICAL_GROUP (for example 21:6)."
        )
    return RealQKAnchor(*(int(item) for item in match.groups()))


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QwenResidualAttentionQKAnchorSearchError(
            f"{name} must be positive and finite."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"{name} must be positive and finite."
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"{name} must be positive and finite."
        )
    return result


def _assert_zero_gradients(runtime: nn.Module, label: str) -> None:
    offenders = [
        name
        for name, parameter in runtime.named_parameters()
        if parameter.grad is not None
    ]
    if offenders:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"{label} found retained gradients: {offenders[:8]!r}."
        )


def _tokenizer_runtime_binding(runtime: nn.Module, tokenizer: Any) -> dict[str, Any]:
    """Prove that prompt IDs use the tokenizer bound into the checkpoint.

    A compatible vocabulary *size* is insufficient: a reordered vocabulary
    would make every prompt and greedy-token trace semantically unrelated while
    still remaining in range.  Canonical checkpoint loading already verifies
    the stored contract against ``parent_token_rows``; this function closes the
    other half by comparing the live tokenizer's complete token->ID map and
    special IDs with that verified runtime contract.
    """

    stored = getattr(runtime, "neural_forge_tokenizer_contract", None)
    if not isinstance(stored, Mapping):
        raise QwenResidualAttentionQKAnchorSearchError(
            "The codebook runtime lacks a resolved tokenizer contract."
        )
    try:
        contract = canonicalize_qwen3_tokenizer_contract(runtime, stored)
    except Qwen3CodebookError as error:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"The runtime tokenizer contract is invalid: {error}"
        ) from error
    try:
        vocabulary = tokenizer.get_vocab()
    except Exception as error:
        raise QwenResidualAttentionQKAnchorSearchError(
            "The live tokenizer cannot expose its complete vocabulary."
        ) from error
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise QwenResidualAttentionQKAnchorSearchError(
            "The live tokenizer vocabulary is empty or malformed."
        )
    try:
        normalized_vocab = {
            str(token): int(token_id) for token, token_id in vocabulary.items()
        }
        tokenizer_length = int(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as error:
        raise QwenResidualAttentionQKAnchorSearchError(
            "The live tokenizer vocabulary contains invalid token IDs."
        ) from error
    observed = {
        "vocab_size": tokenizer_length,
        "vocab_entries": len(normalized_vocab),
        "vocab_sha256": base._sha256(normalized_vocab),
        "special_token_ids": {
            name: getattr(tokenizer, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
    }
    if (
        observed["vocab_size"] != contract["vocab_size"]
        or observed["vocab_entries"] != contract["vocab_size"]
        or observed["vocab_sha256"] != contract["vocab_sha256"]
        or observed["special_token_ids"] != contract["special_token_ids"]
    ):
        raise QwenResidualAttentionQKAnchorSearchError(
            "The live tokenizer does not match the checkpoint's exact vocabulary "
            "and special-token contract."
        )
    stable_identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        **observed,
        "parent_token_rows_sha256": contract["parent_token_rows_sha256"],
    }
    return {
        **stable_identity,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "identity_sha256": base._sha256(stable_identity),
        "runtime_contract_sha256": base._sha256(contract),
        "exact_vocab_and_special_ids_verified": True,
        "runtime_parent_token_rows_contract_verified": True,
    }


def _validate_anchors(
    anchors: Sequence[RealQKAnchor],
    *,
    max_anchors: int,
) -> tuple[RealQKAnchor, ...]:
    values = tuple(anchors)
    maximum = base._positive_int(max_anchors, "max_anchors")
    if not values:
        raise QwenResidualAttentionQKAnchorSearchError(
            "Provide at least one explicit real Q/K anchor."
        )
    if len(values) > maximum:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"Anchor count exceeds max_anchors={maximum}."
        )
    if any(not isinstance(value, RealQKAnchor) for value in values):
        raise QwenResidualAttentionQKAnchorSearchError(
            "Every anchor must be RealQKAnchor."
        )
    if len(set(values)) != len(values):
        raise QwenResidualAttentionQKAnchorSearchError(
            "Explicit Q/K anchors must be unique."
        )
    return values


def _split_report(
    parent_calibration: SplitTraces,
    parent_heldout: SplitTraces,
    candidate_calibration: SplitTraces,
    candidate_heldout: SplitTraces,
    *,
    candidate_ids: tuple[str, ...],
    scoring: ResidualAttentionScoringConfig,
    policy: ResidualAttentionGatePolicy,
) -> dict[str, Any]:
    return value_base._split_variant_report(
        parent_calibration,
        parent_heldout,
        candidate_calibration,
        candidate_heldout,
        candidate_ids=candidate_ids,
        scoring=scoring,
        policy=policy,
    )


def _collect_splits(
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
    return value_base._collect_both_splits(
        runtime,
        calibration_prompts,
        heldout_prompts,
        affected_sites,
        calibration_drivers,
        heldout_drivers,
        label=label,
        horizon=horizon,
        limits=limits,
    )


def _fit_payload(fits: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(source): {
            **{
                key: value
                for key, value in fit.items()
                if key not in {"scalar", "ridge"}
            },
            "ridge": fit["ridge"].report.as_dict(),
        }
        for source, fit in fits.items()
    }


def _restore_o(
    registry: Any,
    backups: Mapping[int, Tensor],
    expected_sha256: str,
) -> None:
    flat_o = base._flat_o(registry)
    with torch.inference_mode():
        for slot, backup in backups.items():
            flat_o[slot].copy_(backup)
    if base._o_slots_sha256(flat_o) != expected_sha256:
        raise QwenResidualAttentionQKAnchorSearchError(
            "O-ridge control did not restore the Q/K candidate registry exactly."
        )


def _summary_metric(lane: Mapping[str, Any], metric: str, statistic: str) -> float:
    if not bool(lane.get("comparable_internal_evidence")):
        raise QwenResidualAttentionQKAnchorSearchError(
            "Cannot extract metrics from a divergent lane."
        )
    try:
        value = float(lane["damage"]["metrics"][metric][statistic])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"Missing {metric}.{statistic} in a causal lane report."
        ) from error
    if not math.isfinite(value) or value < 0.0:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"Invalid {metric}.{statistic} in a causal lane report."
        )
    return value


def _risk_score(lane: Mapping[str, Any]) -> float:
    try:
        value = float(lane["damage"]["risk_score"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise QwenResidualAttentionQKAnchorSearchError(
            "A causal lane lacks a finite non-negative risk score."
        ) from error
    if not math.isfinite(value) or value < 0.0:
        raise QwenResidualAttentionQKAnchorSearchError(
            "A causal lane lacks a finite non-negative risk score."
        )
    return value


def _lane_objective(lane: Mapping[str, Any]) -> float:
    """Fixed causal objective; no calibration rows or static weights enter it."""

    weights = {
        "attention_js_divergence": 0.25,
        "attention_score_geometry_error": 0.20,
        "residual_transition_error": 0.20,
        "cache_transition_error": 0.10,
        "action_margin_damage": 0.10,
        "topk_damage": 0.05,
        "action_mismatch": 0.10,
    }
    return sum(
        weight * _summary_metric(lane, metric, "quantile")
        for metric, weight in weights.items()
    )


def _causal_score(
    anchor: RealQKAnchor,
    control: str,
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    heldout = variant.get("heldout")
    if not isinstance(heldout, Mapping):
        raise QwenResidualAttentionQKAnchorSearchError(
            "Anchor variant lacks held-out evidence."
        )
    forced = heldout.get("forced")
    greedy = heldout.get("greedy")
    if not isinstance(forced, Mapping) or not bool(
        forced.get("comparable_internal_evidence")
    ):
        raise QwenResidualAttentionQKAnchorSearchError(
            "Held-out forced evidence must always be comparable."
        )
    if not isinstance(greedy, Mapping):
        raise QwenResidualAttentionQKAnchorSearchError(
            "Anchor variant lacks held-out greedy evidence."
        )
    forced_objective = _lane_objective(forced)
    forced_risk = _risk_score(forced)
    greedy_aligned = bool(greedy.get("comparable_internal_evidence"))
    greedy_objective = _lane_objective(greedy) if greedy_aligned else None
    greedy_risk = _risk_score(greedy) if greedy_aligned else None
    forced_mismatch = _summary_metric(forced, "action_mismatch", "worst")
    greedy_mismatch = (
        _summary_metric(greedy, "action_mismatch", "worst") if greedy_aligned else 1.0
    )
    combined = (
        forced_objective
        if greedy_objective is None
        else 0.5 * (forced_objective + greedy_objective)
    )
    return {
        "schema": RANKING_SCHEMA,
        "anchor_id": anchor.anchor_id,
        "control": control,
        "greedy_prefix_aligned": greedy_aligned,
        "heldout_forced_causal_objective": forced_objective,
        "heldout_greedy_causal_objective": greedy_objective,
        "heldout_combined_causal_objective": combined,
        "heldout_forced_risk_score": forced_risk,
        "heldout_greedy_risk_score": greedy_risk,
        "heldout_forced_action_mismatch_worst": forced_mismatch,
        "heldout_greedy_action_mismatch_worst": greedy_mismatch,
        "heldout_forced_attention_js_quantile": _summary_metric(
            forced, "attention_js_divergence", "quantile"
        ),
        "heldout_forced_attention_score_geometry_quantile": _summary_metric(
            forced, "attention_score_geometry_error", "quantile"
        ),
        "heldout_forced_residual_transition_quantile": _summary_metric(
            forced, "residual_transition_error", "quantile"
        ),
        "heldout_forced_cache_transition_quantile": _summary_metric(
            forced, "cache_transition_error", "quantile"
        ),
        "heldout_forced_action_margin_quantile": _summary_metric(
            forced, "action_margin_damage", "quantile"
        ),
        "ranking_uses_calibration_rows": False,
        "ranking_uses_static_weight_geometry": False,
    }


def _rank_candidates(
    candidate_reports: Mapping[str, Mapping[str, Any]], control: str
) -> list[dict[str, Any]]:
    scores = []
    for anchor_id, report in sorted(candidate_reports.items()):
        variants = report.get("variants")
        if not isinstance(variants, Mapping) or control not in variants:
            continue
        anchor_payload = report["anchor_edit"]["anchor"]
        anchor = RealQKAnchor(
            int(anchor_payload["layer_index"]),
            int(anchor_payload["physical_group"]),
        )
        scores.append(_causal_score(anchor, control, variants[control]))
    scores.sort(
        key=lambda item: (
            0 if item["greedy_prefix_aligned"] else 1,
            max(
                item["heldout_forced_action_mismatch_worst"],
                item["heldout_greedy_action_mismatch_worst"],
            ),
            item["heldout_combined_causal_objective"],
            item["heldout_forced_risk_score"],
            item["anchor_id"],
        )
    )
    for rank, item in enumerate(scores, start=1):
        item["rank"] = rank
    return scores


def run_qwen_p35_qk_anchor_search(
    runtime: nn.Module,
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    anchors: Sequence[RealQKAnchor],
    *,
    redirect: base.AttentionRedirect = EXACT_REDIRECT,
    require_exact_p35: bool = True,
    horizon: int = 8,
    fit_o_ridge: bool = True,
    ridge: float = 1e-6,
    trace_limits: HFCachedAttentionTraceLimits | None = None,
    max_anchors: int = 12,
    max_affected_sites: int = 16,
    scoring: ResidualAttentionScoringConfig | None = None,
    gate_policy: ResidualAttentionGatePolicy | None = None,
) -> dict[str, Any]:
    """Run the bounded real-anchor search and restore the incumbent exactly."""

    if not isinstance(fit_o_ridge, bool):
        raise QwenResidualAttentionQKAnchorSearchError("fit_o_ridge must be boolean.")
    ridge_value = _positive_finite(ridge, "ridge")
    limits = trace_limits or HFCachedAttentionTraceLimits()
    calibration_prompts, heldout_prompts = value_base._validate_prompt_splits(
        calibration_prompt_input_ids,
        heldout_prompt_input_ids,
        limits,
    )
    anchor_values = _validate_anchors(anchors, max_anchors=max_anchors)
    if require_exact_p35 and redirect != EXACT_REDIRECT:
        raise QwenResidualAttentionQKAnchorSearchError(
            "Exact p35 mode is fixed to L19/G4 -> L21/G6."
        )
    default_scoring, default_policy = value_base._default_scoring()
    scoring_value = scoring or default_scoring
    policy_value = gate_policy or default_policy
    max_sites = base._positive_int(max_affected_sites, "max_affected_sites")
    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    if require_exact_p35 and (
        parent_snapshot.logical_layers != 36
        or parent_snapshot.key_value_heads != 8
        or parent_snapshot.prototype_count != 280
    ):
        raise QwenResidualAttentionQKAnchorSearchError(
            "Exact mode requires the 36-layer, 8-group, 280-prototype p35 frontier."
        )
    _assert_zero_gradients(runtime, "preflight")
    parent_installation = attention_surgery._capture_installed(runtime)
    _bridge, _proposal, proposal_candidates, tentative = base._tentative_codebook(
        runtime, (redirect,)
    )
    codebook, used_projections = tentative
    proposed_registry = codebook.registry
    affected_sites, sites_by_source = base._affected_sites(
        parent_registry,
        proposed_registry,
        (redirect,),
    )
    value_base._assert_block_o_layout(sites_by_source)
    destination_source = (
        redirect.target_layer * int(parent_snapshot.key_value_heads)
        + redirect.target_group
    )
    if require_exact_p35:
        parent_sources = set(map(int, parent_snapshot.prototype_source_indices))
        proposed_sources = tuple(
            int(value) for value in base._flat_sources(proposed_registry).tolist()
        )
        if (
            int(proposed_registry.prototype_count) != 279
            or len(proposed_sources) != 279
            or len(set(proposed_sources)) != 279
            or EXACT_REMOVED_SOURCE not in parent_sources
            or EXACT_DESTINATION_SOURCE not in parent_sources
            or set(proposed_sources)
            != parent_sources.difference({EXACT_REMOVED_SOURCE})
        ):
            raise QwenResidualAttentionQKAnchorSearchError(
                "The tentative exact-p35 proposal is not the byte-addressable "
                "280 -> 279 lineage with only L19/G4 removed."
            )
    if require_exact_p35 and (
        destination_source != EXACT_DESTINATION_SOURCE
        or affected_sites != ((2, 1), (19, 4), (21, 6))
        or tuple(sites_by_source) != (EXACT_DESTINATION_SOURCE,)
    ):
        raise QwenResidualAttentionQKAnchorSearchError(
            "The p35 lineage does not expose the expected three L21/G6-shared "
            "executable assignments."
        )
    if len(affected_sites) > max_sites:
        raise QwenResidualAttentionQKAnchorSearchError(
            f"Affected sites exceed max_affected_sites={max_sites}."
        )
    if len({layer for layer, _group in affected_sites}) > limits.max_layers:
        raise QwenResidualAttentionQKAnchorSearchError(
            "Affected layers exceed the cached-trace layer bound."
        )
    # Resolve every explicit anchor against the *post-redirect* real survivors
    # before running any expensive trace.
    for anchor in anchor_values:
        try:
            installation = install_real_qk_anchor(
                proposed_registry,
                anchor,
                destination_source_flat_index=destination_source,
            )
            installation.rollback()
        except ResidualAttentionQKAnchorError as error:
            raise QwenResidualAttentionQKAnchorSearchError(
                f"Invalid explicit anchor {anchor.anchor_id}: {error}"
            ) from error

    parent_o_blocks = base._parent_o_blocks(parent_registry, affected_sites)
    parent_calibration_before = value_base._trace_state_binding(runtime)
    parent_calibration, calibration_drivers = base._collect_parent_traces(
        runtime,
        calibration_prompts,
        affected_sites,
        horizon=horizon,
        limits=limits,
    )
    parent_calibration_binding = value_base._finish_trace_state_binding(
        runtime,
        parent_calibration_before,
        label="parent:calibration",
    )
    parent_heldout_before = value_base._trace_state_binding(runtime)
    parent_heldout, heldout_drivers = base._collect_parent_traces(
        runtime,
        heldout_prompts,
        affected_sites,
        horizon=horizon,
        limits=limits,
    )
    parent_heldout_binding = value_base._finish_trace_state_binding(
        runtime,
        parent_heldout_before,
        label="parent:heldout",
    )

    candidate_reports: dict[str, Any] = {}
    proposed_snapshot = None
    proposed_reset_exact = False
    incumbent_rollback_exact = False
    live_anchor: InstalledRealQKAnchor | None = None
    live_o_backups: dict[int, Tensor] = {}
    live_o_sha256: str | None = None
    try:
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        proposed_snapshot = qwen3_attention_runtime_snapshot(runtime)
        for anchor in anchor_values:
            candidate_id = f"{redirect.candidate_id}/QK@{anchor.anchor_id}"
            live_o_backups = {}
            live_o_sha256 = None
            live_anchor = install_real_qk_anchor(
                proposed_registry,
                anchor,
                destination_source_flat_index=destination_source,
            )
            try:
                assignment_audit = audit_qk_anchor_assignments(
                    runtime,
                    proposed_registry,
                    anchor=anchor,
                    destination_source_flat_index=destination_source,
                )
                audited_sites = tuple(
                    tuple(value) for value in assignment_audit["assignment_coordinates"]
                )
                if audited_sites != affected_sites:
                    raise QwenResidualAttentionQKAnchorSearchError(
                        "The Q/K assignment audit did not cover the exact affected "
                        "survivor set in deterministic order."
                    )
                candidate_snapshot = qwen3_attention_runtime_snapshot(runtime)
                (
                    raw_calibration,
                    raw_heldout,
                    raw_trace_bindings,
                ) = _collect_splits(
                    runtime,
                    calibration_prompts,
                    heldout_prompts,
                    affected_sites,
                    calibration_drivers,
                    heldout_drivers,
                    label=f"{anchor.anchor_id}:qk_only",
                    horizon=horizon,
                    limits=limits,
                )
                variants: dict[str, Any] = {
                    "qk_only": _split_report(
                        parent_calibration,
                        parent_heldout,
                        raw_calibration,
                        raw_heldout,
                        candidate_ids=(candidate_id,),
                        scoring=scoring_value,
                        policy=policy_value,
                    )
                }
                trace_bindings: dict[str, Any] = {"qk_only": raw_trace_bindings}
                trace_manifests: dict[str, Any] = {
                    "qk_only": {
                        "calibration": value_base._trace_manifest(raw_calibration),
                        "heldout": value_base._trace_manifest(raw_heldout),
                    }
                }
                fits: dict[int, dict[str, Any]] = {}
                if fit_o_ridge:
                    fits = base._fit_survivor_blocks(
                        parent_calibration,
                        raw_calibration,
                        parent_o_blocks,
                        proposed_registry,
                        sites_by_source,
                        prompt_count=len(calibration_prompts),
                        ridge=ridge_value,
                    )
                    if set(map(int, fits)) != set(map(int, sites_by_source)):
                        raise QwenResidualAttentionQKAnchorSearchError(
                            "O-ridge fit escaped the exact shared-survivor source set."
                        )
                    flat_o = base._flat_o(proposed_registry)
                    target_slots = {
                        int(value["active_slot"]) for value in fits.values()
                    }
                    expected_o_slots = {live_anchor.report.destination_slot}
                    if target_slots != expected_o_slots:
                        raise QwenResidualAttentionQKAnchorSearchError(
                            "O-ridge fit targeted a slot outside the edited Q/K "
                            "destination survivor."
                        )
                    live_o_backups = {
                        slot: flat_o[slot].detach().clone()
                        for slot in sorted(target_slots)
                    }
                    live_o_sha256 = base._o_slots_sha256(flat_o)
                    o_audit = base._apply_compensation(
                        proposed_registry, fits, kind="ridge"
                    )
                    if set(map(int, o_audit["modified_active_slots"])) != target_slots:
                        raise QwenResidualAttentionQKAnchorSearchError(
                            "O-ridge edit audit disagrees with its bounded target slot."
                        )
                    (
                        o_calibration,
                        o_heldout,
                        o_trace_bindings,
                    ) = _collect_splits(
                        runtime,
                        calibration_prompts,
                        heldout_prompts,
                        affected_sites,
                        calibration_drivers,
                        heldout_drivers,
                        label=f"{anchor.anchor_id}:qk_plus_o_ridge",
                        horizon=horizon,
                        limits=limits,
                    )
                    variants["qk_plus_o_ridge"] = {
                        **_split_report(
                            parent_calibration,
                            parent_heldout,
                            o_calibration,
                            o_heldout,
                            candidate_ids=(f"{candidate_id}/O-ridge",),
                            scoring=scoring_value,
                            policy=policy_value,
                        ),
                        "edit_audit": o_audit,
                    }
                    trace_bindings["qk_plus_o_ridge"] = o_trace_bindings
                    trace_manifests["qk_plus_o_ridge"] = {
                        "calibration": value_base._trace_manifest(o_calibration),
                        "heldout": value_base._trace_manifest(o_heldout),
                    }
                    _restore_o(
                        proposed_registry,
                        live_o_backups,
                        live_o_sha256,
                    )
                    live_o_backups = {}
                    live_o_sha256 = None
                    live_anchor.assert_live()
                candidate_reports[anchor.anchor_id] = {
                    "candidate_id": candidate_id,
                    "anchor_edit": live_anchor.report.as_dict(
                        key_value_heads=parent_snapshot.key_value_heads
                    ),
                    "assignment_audit": assignment_audit,
                    "candidate_attention_snapshot": candidate_snapshot.as_dict(),
                    "fits": {"o_ridge": _fit_payload(fits)},
                    "variants": variants,
                    "trace_collection_state_bindings": trace_bindings,
                    "trace_provenance": trace_manifests,
                }
            finally:
                restoration_errors: list[Exception] = []
                if live_o_sha256 is not None and live_o_backups:
                    try:
                        _restore_o(proposed_registry, live_o_backups, live_o_sha256)
                    except Exception as error:  # pragma: no cover - cleanup defense
                        restoration_errors.append(error)
                live_o_backups = {}
                live_o_sha256 = None
                if live_anchor is not None and live_anchor.is_applied:
                    try:
                        live_anchor.rollback()
                    except Exception as error:  # pragma: no cover - cleanup defense
                        restoration_errors.append(error)
                live_anchor = None
                if qwen3_attention_runtime_snapshot(runtime) != proposed_snapshot:
                    restoration_errors.append(
                        QwenResidualAttentionQKAnchorSearchError(
                            f"Candidate {anchor.anchor_id} did not reset the "
                            "tentative redirect exactly."
                        )
                    )
                if restoration_errors and sys.exc_info()[0] is None:
                    raise QwenResidualAttentionQKAnchorSearchError(
                        "Q/K candidate cleanup failed: "
                        + "; ".join(str(error) for error in restoration_errors)
                    ) from restoration_errors[0]
            candidate_reports[anchor.anchor_id]["tentative_reset_exact"] = True
        proposed_reset_exact = (
            qwen3_attention_runtime_snapshot(runtime) == proposed_snapshot
        )
    finally:
        failure_in_flight = sys.exc_info()[0] is not None
        restoration_errors: list[Exception] = []
        if live_o_sha256 is not None and live_o_backups:
            try:
                _restore_o(proposed_registry, live_o_backups, live_o_sha256)
            except Exception as error:  # pragma: no cover - cleanup defense
                restoration_errors.append(error)
        if live_anchor is not None and live_anchor.is_applied:
            try:
                live_anchor.rollback()
            except Exception as error:  # pragma: no cover - cleanup defense
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
                    QwenResidualAttentionQKAnchorSearchError(
                        "Search did not restore the incumbent p35 registry exactly."
                    )
                )
        except Exception as error:  # pragma: no cover - cleanup defense
            restoration_errors.append(error)
        if restoration_errors and not failure_in_flight:
            raise QwenResidualAttentionQKAnchorSearchError(
                "Search cleanup failed after all rollback attempts: "
                + "; ".join(str(error) for error in restoration_errors)
            ) from restoration_errors[0]

    if proposed_snapshot is None or not proposed_reset_exact:
        raise QwenResidualAttentionQKAnchorSearchError(
            "Search completed without an exact tentative-state reset."
        )
    _assert_zero_gradients(runtime, "post-search")
    ranking: dict[str, Any] = {
        "objective": {
            "split": "heldout-only",
            "greedy_prefix_alignment_precedes_numeric_damage": True,
            "action_mismatch_precedes_weighted_causal_objective": True,
            "weighted_quantile_metrics": {
                "attention_js_divergence": 0.25,
                "attention_score_geometry_error": 0.20,
                "residual_transition_error": 0.20,
                "cache_transition_error": 0.10,
                "action_margin_damage": 0.10,
                "topk_damage": 0.05,
                "action_mismatch": 0.10,
            },
            "calibration_used_for_ranking": False,
            "static_weight_geometry_used_for_ranking": False,
        },
        "qk_only": _rank_candidates(candidate_reports, "qk_only"),
    }
    if fit_o_ridge:
        ranking["qk_plus_o_ridge"] = _rank_candidates(
            candidate_reports, "qk_plus_o_ridge"
        )
    ranking["winner_by_control"] = {
        control: (scores[0]["anchor_id"] if scores else None)
        for control, scores in ranking.items()
        if isinstance(scores, list)
    }
    parent_bindings = base._materialized_site_bindings(parent_registry)
    proposed_bindings = base._materialized_site_bindings(proposed_registry)
    affected_payload = [
        {
            "layer_index": layer,
            "materialized_physical_group": group,
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
        "explicit_anchor_count": len(anchor_values),
        "explicit_anchor_order": [anchor.anchor_id for anchor in anchor_values],
        "horizon": horizon,
        "fit_o_ridge": fit_o_ridge,
        "ridge": ridge_value,
        "prompt_splits": {
            "calibration": {
                "count": len(calibration_prompts),
                "prompt_sha256s": list(value_base._prompt_sha256s(calibration_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(calibration_drivers.items())
                },
                "used_by_o_fit": fit_o_ridge,
                "used_by_ranking": False,
            },
            "heldout": {
                "count": len(heldout_prompts),
                "prompt_sha256s": list(value_base._prompt_sha256s(heldout_prompts)),
                "driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(heldout_drivers.items())
                },
                "used_by_o_fit": False,
                "used_by_ranking": True,
            },
            "token_rows_disjoint": True,
        },
        "parent_attention_snapshot": parent_snapshot.as_dict(),
        "proposed_redirect_snapshot": proposed_snapshot.as_dict(),
        "accounting": {
            "parent_prototype_count": parent_snapshot.prototype_count,
            "proposed_prototype_count": proposed_snapshot.prototype_count,
            "prototype_count_removed": (
                parent_snapshot.prototype_count - proposed_snapshot.prototype_count
            ),
            "physical_attention_bytes_saved": (
                parent_snapshot.tensor_bytes - proposed_snapshot.tensor_bytes
            ),
            "qk_anchor_search_adds_parameters": False,
            "o_ridge_adds_parameters": False,
        },
        "affected_assignments": affected_payload,
        "affected_assignment_count": len(affected_payload),
        "affected_assignment_sha256": base._sha256(affected_payload),
        "survivor_assignment_sets": {
            str(source): [list(site) for site in sites]
            for source, sites in sites_by_source.items()
        },
        "used_redirect_projections": [value.as_dict() for value in used_projections],
        "proposal_candidate_count": len(proposal_candidates),
        "candidates": candidate_reports,
        "ranking": ranking,
        "parent_trace_provenance": {
            "calibration": value_base._trace_manifest(parent_calibration),
            "heldout": value_base._trace_manifest(parent_heldout),
        },
        "parent_trace_collection_state_bindings": {
            "calibration": parent_calibration_binding,
            "heldout": parent_heldout_binding,
        },
        "search_contract": {
            "exact_p35_280_to_279_source_lineage_verified": require_exact_p35,
            "explicit_bounded_real_survivors_only": True,
            "q_and_k_always_from_same_real_active_slot": True,
            "q_or_k_averaged_before_softmax": False,
            "v_remains_redirect_incumbent": True,
            "o_remains_redirect_incumbent_except_optional_control": True,
            "optional_o_fit_global_across_all_shared_assignments": fit_o_ridge,
            "optional_o_fit_uses_calibration_forced_rows_only": fit_o_ridge,
            "all_shared_assignments_evaluated": True,
            "group_permutations_and_role_scales_audited": True,
            "logical_layer_q_norm_k_norm_content_addressed": True,
            "causal_post_norm_attention_distributions_ranked": True,
            "calibration_and_heldout_forced_and_greedy_lanes": True,
            "heldout_only_ranking": True,
            "zero_gradient": True,
            "checkpoint_saved": False,
        },
        "scope": {
            "bounded_local_probe_only": True,
            "broad_behavior_generation_evaluation_performed": False,
            "compaction_certified": False,
            "warning": (
                "A winner identifies the best tested causal Q/K anchor for this "
                "prompt set and redirect.  It is not a saved-child or broad-suite "
                "compaction proof."
            ),
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
    return value_base._load_texts(inline, json_path, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--qk-anchor",
        type=parse_real_qk_anchor,
        action="append",
        required=True,
        help=(
            "Active real survivor LAYER:PHYSICAL_GROUP; repeat for a bounded "
            "explicit candidate set."
        ),
    )
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
    parser.add_argument("--skip-o-ridge", action="store_true")
    parser.add_argument("--max-anchors", type=int, default=12)
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
        raise QwenResidualAttentionQKAnchorSearchError(
            "A prompt split exceeds --max-prompts-per-split."
        )
    runtime, tokenizer, source = base._load_runtime_and_tokenizer(args)
    tokenizer_binding = _tokenizer_runtime_binding(runtime, tokenizer)
    calibration_prompts = value_base._encode_prompts(
        tokenizer, calibration_texts, args.max_prompt_tokens
    )
    heldout_prompts = value_base._encode_prompts(
        tokenizer, heldout_texts, args.max_prompt_tokens
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
    )
    report = run_qwen_p35_qk_anchor_search(
        runtime,
        calibration_prompts,
        heldout_prompts,
        args.qk_anchor,
        redirect=EXACT_REDIRECT,
        require_exact_p35=True,
        horizon=args.horizon,
        fit_o_ridge=not args.skip_o_ridge,
        ridge=args.ridge,
        trace_limits=limits,
        max_anchors=args.max_anchors,
        max_affected_sites=args.max_affected_sites,
    )
    report["source"] = source
    report["tokenizer_binding"] = tokenizer_binding
    report["parameter_report_after_exact_rollback"] = qwen3_codebook_parameter_report(
        runtime
    ).as_dict()
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
                "winners": report["ranking"]["winner_by_control"],
                "rollback_exact": report["incumbent_rollback_exact"],
                "checkpoint_saved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
