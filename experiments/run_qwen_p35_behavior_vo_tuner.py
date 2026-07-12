#!/usr/bin/env python3
"""Tune p35 L19/G4 V/O scales using causal-output behavior only.

This runner fixes the structural proposal to ``L19/G4 -> L21/G6`` and uses
three prompt-disjoint roles: fit gradients, calibration-only selection, and
winnerless heldout evaluation.  Parent functional traces are collected once
from the untouched p35 runtime.  The tentative ``280 -> 279`` child is then
installed, two existing V/O assignment scalars are tuned by the manual
behavior-space optimizer, and the original p35 runtime is restored exactly.

No weight, Q/K/V, attention, activation, residual, or cache-value similarity
is an objective.  This runner never commits a fold and never saves a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_p35_vo_role_scale_search as role_search
import run_qwen_residual_attention_probe as base
from neural_forge.hf_functional_decode_trace import FunctionalDecodeLimits
from neural_forge.hf_qwen_attention_surgery import (
    Qwen3AttentionRuntimeSnapshot,
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report
from neural_forge.hf_residual_attention_behavior_tuner import (
    BEHAVIOR_VO_TUNER_SCHEMA,
    BehaviorVOTuningControls,
    BehaviorVOTuningResult,
    tune_exact_p35_residual_attention_vo,
)


RUN_SCHEMA = "neural-forge-qwen-p35-behavior-vo-tuner-run-v1"
EXACT_REDIRECT = role_search.EXACT_REDIRECT
EXACT_AFFECTED_SITES = role_search.EXACT_AFFECTED_SITES
EXACT_PARENT_PROTOTYPE_COUNT = 280
EXACT_CHILD_PROTOTYPE_COUNT = 279

EXPERIMENTS = Path(__file__).resolve().parent
DEFAULT_FIT_PROMPTS = (
    EXPERIMENTS / "p35_p34_residual_attention_fit_prompts_20260711.json"
)
DEFAULT_CALIBRATION_PROMPTS = (
    EXPERIMENTS
    / "p35_p34_residual_attention_calibration_selection_prompts_20260711.json"
)
DEFAULT_HELDOUT_PROMPTS = (
    EXPERIMENTS
    / "p35_p34_residual_attention_heldout_evaluation_prompts_20260711.json"
)
DEFAULT_FIT_MANIFEST = (
    EXPERIMENTS
    / "p35_p34_residual_attention_fit_prompt_manifest_20260711.json"
)
DEFAULT_AUDITED_MANIFEST = (
    EXPERIMENTS / "p35_p34_residual_attention_prompt_manifest_20260711.json"
)
DEFAULT_TRANSITION_INVENTORY = (
    EXPERIMENTS / "p35_p34_generation_transitions_20260711.json"
)

EXPECTED_FILE_SHA256 = {
    "fit": "b67d140eb3e5f4bbaaa01a0e47ef755818a3379d4d25692d805bcb5c252003ca",
    "calibration-selection": (
        "a2eaea993c7f7f434e30744eeaf4b4da750b9a327d835deedb385dbaf154e406"
    ),
    "heldout-evaluation": (
        "cebf98bdbdfc1154c45758cdeed25e15141b6aeb910f06fa6cf349efdaf6d04f"
    ),
    "fit-manifest": (
        "f790d582f18861bdc40035eaabf4c0c1141da937e2cd8d3b393a32777f498735"
    ),
    "audited-manifest": (
        "d6d4d326cc55d864d388bda6220fe09df70a6ec7ffe4664d4277af53857a00d1"
    ),
    "transition-inventory": (
        "212ebf3d9350c1d48ad77a71616d39d5d163dd022a7ac95b9bb0d4c03f0b929a"
    ),
}
EXPECTED_SPLIT_COUNTS = {
    "fit": 8,
    "calibration-selection": 6,
    "heldout-evaluation": 8,
}
FIT_SELECTION_SEED = "neural-forge-p35-behavior-vo-fit-20260711"
FIT_CATEGORY_QUOTAS = {"code": 3, "tokenizer": 3, "model-surgery": 2}


class QwenP35BehaviorVOTunerError(RuntimeError):
    """A fixture, lineage, behavior objective, or rollback contract failed."""


@dataclass(frozen=True, slots=True)
class VerifiedPromptSplits:
    """Exact text fixtures and their fail-closed provenance audit."""

    fit: tuple[str, ...]
    calibration: tuple[str, ...]
    heldout: tuple[str, ...]
    provenance: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rank_fit_row(row: Mapping[str, Any]) -> str:
    payload = "\0".join(
        (
            FIT_SELECTION_SEED,
            "fit",
            "lost",
            str(row["category"]),
            str(row["id"]),
            str(row["prompt"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_exact_json(path: Path, *, name: str, expected_sha256: str) -> Any:
    resolved = path.resolve()
    if not resolved.is_file():
        raise QwenP35BehaviorVOTunerError(f"{name} does not exist: {resolved}")
    actual = _file_sha256(resolved)
    if actual != expected_sha256:
        raise QwenP35BehaviorVOTunerError(
            f"{name} SHA-256 changed: expected {expected_sha256}, got {actual}."
        )
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenP35BehaviorVOTunerError(
            f"{name} is not valid UTF-8 JSON."
        ) from error


def _prompt_list(payload: Any, *, name: str, expected_count: int) -> tuple[str, ...]:
    if (
        not isinstance(payload, list)
        or len(payload) != expected_count
        or any(not isinstance(value, str) or not value for value in payload)
        or len(set(payload)) != len(payload)
    ):
        raise QwenP35BehaviorVOTunerError(
            f"{name} must remain the exact unique {expected_count}-prompt list."
        )
    return tuple(payload)


def load_verified_prompt_splits(
    fit_path: Path,
    calibration_path: Path,
    heldout_path: Path,
    *,
    fit_manifest_path: Path = DEFAULT_FIT_MANIFEST,
    audited_manifest_path: Path = DEFAULT_AUDITED_MANIFEST,
    transition_inventory_path: Path = DEFAULT_TRANSITION_INVENTORY,
) -> VerifiedPromptSplits:
    """Load and independently verify all three immutable prompt fixtures."""

    fit = _prompt_list(
        _read_exact_json(
            fit_path,
            name="fit prompt fixture",
            expected_sha256=EXPECTED_FILE_SHA256["fit"],
        ),
        name="fit prompt fixture",
        expected_count=EXPECTED_SPLIT_COUNTS["fit"],
    )
    calibration = _prompt_list(
        _read_exact_json(
            calibration_path,
            name="calibration-selection prompt fixture",
            expected_sha256=EXPECTED_FILE_SHA256["calibration-selection"],
        ),
        name="calibration-selection prompt fixture",
        expected_count=EXPECTED_SPLIT_COUNTS["calibration-selection"],
    )
    heldout = _prompt_list(
        _read_exact_json(
            heldout_path,
            name="heldout-evaluation prompt fixture",
            expected_sha256=EXPECTED_FILE_SHA256["heldout-evaluation"],
        ),
        name="heldout-evaluation prompt fixture",
        expected_count=EXPECTED_SPLIT_COUNTS["heldout-evaluation"],
    )
    fit_manifest = _read_exact_json(
        fit_manifest_path,
        name="fit prompt manifest",
        expected_sha256=EXPECTED_FILE_SHA256["fit-manifest"],
    )
    audited_manifest = _read_exact_json(
        audited_manifest_path,
        name="audited calibration/heldout manifest",
        expected_sha256=EXPECTED_FILE_SHA256["audited-manifest"],
    )
    inventory = _read_exact_json(
        transition_inventory_path,
        name="p35/p34 transition inventory",
        expected_sha256=EXPECTED_FILE_SHA256["transition-inventory"],
    )

    try:
        fit_entries = fit_manifest["entries"]
        calibration_entries = audited_manifest["sets"]["calibration-selection"][
            "entries"
        ]
        heldout_entries = audited_manifest["sets"]["heldout-evaluation"][
            "entries"
        ]
        lost_rows = inventory["cases"]["lost"]
    except (KeyError, TypeError) as error:
        raise QwenP35BehaviorVOTunerError(
            "Prompt provenance manifests do not expose the required rows."
        ) from error
    if not all(isinstance(values, list) for values in (
        fit_entries,
        calibration_entries,
        heldout_entries,
        lost_rows,
    )):
        raise QwenP35BehaviorVOTunerError("Prompt provenance rows must be lists.")
    if [value.get("prompt") for value in fit_entries] != list(fit):
        raise QwenP35BehaviorVOTunerError(
            "Fit fixture order/content differs from its provenance manifest."
        )
    if [value.get("prompt") for value in calibration_entries] != list(calibration):
        raise QwenP35BehaviorVOTunerError(
            "Audited calibration fixture content changed."
        )
    if [value.get("prompt") for value in heldout_entries] != list(heldout):
        raise QwenP35BehaviorVOTunerError(
            "Audited heldout fixture content changed."
        )

    audited_ids = {
        str(value.get("id")) for value in (*calibration_entries, *heldout_entries)
    }
    audited_prompts = set(calibration) | set(heldout)
    inventory_by_id: dict[str, Mapping[str, Any]] = {}
    for row in lost_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise QwenP35BehaviorVOTunerError("Transition inventory row is malformed.")
        if row["id"] in inventory_by_id:
            raise QwenP35BehaviorVOTunerError(
                "Transition inventory contains a duplicate lost-case id."
            )
        inventory_by_id[row["id"]] = row

    manifest_ids: list[str] = []
    for index, entry in enumerate(fit_entries):
        if not isinstance(entry, Mapping):
            raise QwenP35BehaviorVOTunerError("Fit manifest row is malformed.")
        row_id = entry.get("id")
        source = inventory_by_id.get(str(row_id))
        if source is None:
            raise QwenP35BehaviorVOTunerError(
                f"Fit row {row_id!r} is absent from the lost transition inventory."
            )
        exact_fields = ("id", "category", "prompt", "prompt_sha256")
        if any(entry.get(name) != source.get(name) for name in exact_fields):
            raise QwenP35BehaviorVOTunerError(
                f"Fit row {row_id!r} differs from transition provenance."
            )
        prompt = str(entry["prompt"])
        if entry.get("index") != index or entry.get("transition") != "lost":
            raise QwenP35BehaviorVOTunerError("Fit row index/transition changed.")
        if entry.get("prompt_sha256") != _text_sha256(prompt):
            raise QwenP35BehaviorVOTunerError("Fit prompt text hash changed.")
        if entry.get("selection_rank_sha256") != _rank_fit_row(entry):
            raise QwenP35BehaviorVOTunerError("Fit selection rank changed.")
        if row_id in audited_ids or prompt in audited_prompts:
            raise QwenP35BehaviorVOTunerError(
                "Fit row overlaps audited calibration or heldout data."
            )
        manifest_ids.append(str(row_id))

    if Counter(str(value.get("category")) for value in fit_entries) != Counter(
        FIT_CATEGORY_QUOTAS
    ):
        raise QwenP35BehaviorVOTunerError("Fit category quotas changed.")
    expected_ids: set[str] = set()
    for category, count in FIT_CATEGORY_QUOTAS.items():
        eligible = [
            row
            for row in lost_rows
            if row.get("category") == category
            and row.get("id") not in audited_ids
            and row.get("prompt") not in audited_prompts
        ]
        ranked = sorted(eligible, key=lambda value: (_rank_fit_row(value), value["id"]))
        expected_ids.update(str(value["id"]) for value in ranked[:count])
    if set(manifest_ids) != expected_ids or len(manifest_ids) != len(expected_ids):
        raise QwenP35BehaviorVOTunerError(
            "Fit rows are not the deterministic quota winners from the inventory."
        )

    split_sets = {
        "fit": set(fit),
        "calibration-selection": set(calibration),
        "heldout-evaluation": set(heldout),
    }
    overlaps = {
        "fit_calibration_exact_prompt_overlap_count": len(
            split_sets["fit"] & split_sets["calibration-selection"]
        ),
        "fit_heldout_exact_prompt_overlap_count": len(
            split_sets["fit"] & split_sets["heldout-evaluation"]
        ),
        "calibration_heldout_exact_prompt_overlap_count": len(
            split_sets["calibration-selection"]
            & split_sets["heldout-evaluation"]
        ),
    }
    if any(overlaps.values()):
        raise QwenP35BehaviorVOTunerError(
            "Fit, calibration-selection, and heldout text fixtures overlap."
        )
    provenance = {
        "verified_exact_fixture_provenance": True,
        "file_sha256": dict(EXPECTED_FILE_SHA256),
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "fit_selection_seed": FIT_SELECTION_SEED,
        "fit_category_quotas": dict(FIT_CATEGORY_QUOTAS),
        "fit_transition": "lost",
        "fit_rows_match_transition_inventory_exactly": True,
        "fit_rows_are_deterministic_quota_winners": True,
        "audited_calibration_fixture_unchanged": True,
        "audited_heldout_fixture_unchanged": True,
        "prompt_text_sha256s": {
            "fit": [_text_sha256(value) for value in fit],
            "calibration-selection": [
                _text_sha256(value) for value in calibration
            ],
            "heldout-evaluation": [_text_sha256(value) for value in heldout],
        },
        "fit_case_ids": manifest_ids,
        "exact_text_overlap": overlaps,
        "all_three_text_sets_pairwise_disjoint": True,
        "broad_promotion_suite_modified": False,
        "broad_promotion_suite_used_as_live_fit_input": False,
    }
    provenance["provenance_sha256"] = base._sha256(provenance)
    return VerifiedPromptSplits(fit, calibration, heldout, provenance)


def _validate_token_splits(
    fit: Sequence[Tensor],
    calibration: Sequence[Tensor],
    heldout: Sequence[Tensor],
    *,
    max_prompts_per_split: int,
    require_exact_p35: bool,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], dict[str, Any]]:
    values = {
        "fit": tuple(fit),
        "calibration-selection": tuple(calibration),
        "heldout-evaluation": tuple(heldout),
    }
    maximum = role_search._positive_int(
        max_prompts_per_split, "max_prompts_per_split"
    )
    for name, rows in values.items():
        if not rows or len(rows) > maximum:
            raise QwenP35BehaviorVOTunerError(
                f"{name} is empty or exceeds max_prompts_per_split."
            )
        if require_exact_p35 and len(rows) != EXPECTED_SPLIT_COUNTS[name]:
            raise QwenP35BehaviorVOTunerError(
                f"Exact p35 mode requires {EXPECTED_SPLIT_COUNTS[name]} {name} prompts."
            )
    hashes = {
        name: role_search._prompt_sha256s(rows) for name, rows in values.items()
    }
    if any(len(set(row_hashes)) != len(row_hashes) for row_hashes in hashes.values()):
        raise QwenP35BehaviorVOTunerError(
            "Every tokenized prompt row must be unique within its split."
        )
    overlap = {
        "fit_calibration_token_overlap_count": len(
            set(hashes["fit"]) & set(hashes["calibration-selection"])
        ),
        "fit_heldout_token_overlap_count": len(
            set(hashes["fit"]) & set(hashes["heldout-evaluation"])
        ),
        "calibration_heldout_token_overlap_count": len(
            set(hashes["calibration-selection"])
            & set(hashes["heldout-evaluation"])
        ),
    }
    if any(overlap.values()):
        raise QwenP35BehaviorVOTunerError(
            "Tokenized fit, calibration-selection, and heldout rows must be pairwise disjoint."
        )
    audit = {
        "counts": {name: len(rows) for name, rows in values.items()},
        "token_row_sha256s": {
            name: list(row_hashes) for name, row_hashes in hashes.items()
        },
        **overlap,
        "all_three_token_row_sets_pairwise_disjoint": True,
    }
    return values["fit"], values["calibration-selection"], values[
        "heldout-evaluation"
    ], audit


def _assert_tuner_report_contract(result: BehaviorVOTuningResult) -> None:
    report = result.as_dict()
    try:
        valid = (
            report["schema"] == BEHAVIOR_VO_TUNER_SCHEMA
            and report["exact_fold_lineage"]["parent_prototype_count"]
            == EXACT_PARENT_PROTOTYPE_COUNT
            and report["exact_fold_lineage"]["child_prototype_count"]
            == EXACT_CHILD_PROTOTYPE_COUNT
            and report["accounting"]["prototype_count_removed"] == 1
            and report["accounting"]["this_report_covers_one_redirect_only"]
            is True
            and report["transaction"][
                "full_tentative_role_scales_restored_exactly"
            ]
            is True
            and report["transaction"]["selected_candidate_installed_after_return"]
            is False
            and report["checkpoint_saved"] is False
            and report["fold_committed"] is False
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise QwenP35BehaviorVOTunerError(
            "Behavior tuner did not return the exact one-redirect rollback contract."
        )


def run_qwen_p35_behavior_vo_tuner(
    runtime: nn.Module,
    fit_prompt_input_ids: Sequence[Tensor],
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    *,
    split_provenance: Mapping[str, Any],
    controls: BehaviorVOTuningControls | None = None,
    redirect: base.AttentionRedirect = EXACT_REDIRECT,
    require_exact_p35: bool = True,
    horizon: int = 8,
    trace_limits: FunctionalDecodeLimits | None = None,
) -> dict[str, Any]:
    """Tune the exact tentative child, then restore the incumbent p35 runtime."""

    if require_exact_p35 and redirect != EXACT_REDIRECT:
        raise QwenP35BehaviorVOTunerError(
            "Exact p35 behavior tuning is fixed to L19/G4 -> L21/G6."
        )
    if not isinstance(split_provenance, Mapping) or (
        require_exact_p35
        and split_provenance.get("verified_exact_fixture_provenance") is not True
    ):
        raise QwenP35BehaviorVOTunerError(
            "Exact p35 mode requires verified three-split fixture provenance."
        )
    settings = controls or BehaviorVOTuningControls(max_horizon=horizon)
    if not isinstance(settings, BehaviorVOTuningControls):
        raise QwenP35BehaviorVOTunerError(
            "controls must be BehaviorVOTuningControls."
        )
    horizon_value = role_search._positive_int(horizon, "horizon")
    limits = trace_limits or FunctionalDecodeLimits(max_horizon=horizon_value)
    if horizon_value > limits.max_horizon or horizon_value > settings.max_horizon:
        raise QwenP35BehaviorVOTunerError(
            "horizon exceeds the functional decoder or tuner bound."
        )
    fit_prompts, calibration_prompts, heldout_prompts, token_audit = (
        _validate_token_splits(
            fit_prompt_input_ids,
            calibration_prompt_input_ids,
            heldout_prompt_input_ids,
            max_prompts_per_split=settings.max_prompts_per_split,
            require_exact_p35=require_exact_p35,
        )
    )

    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_identity = role_search._runtime_state_identity_sha256(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    _bridge, _proposal, proposal_candidates, tentative = base._tentative_codebook(
        runtime, (redirect,)
    )
    codebook, used_projections = tentative
    proposed_registry = codebook.registry
    affected_sites, sites_by_source = base._affected_sites(
        parent_registry, proposed_registry, (redirect,)
    )

    parent_fit, fit_drivers = role_search._collect_parent_split(
        runtime,
        fit_prompts,
        split="fit",
        horizon=horizon_value,
        limits=limits,
    )
    parent_calibration, calibration_drivers = role_search._collect_parent_split(
        runtime,
        calibration_prompts,
        split="calibration-selection",
        horizon=horizon_value,
        limits=limits,
    )
    parent_heldout, heldout_drivers = role_search._collect_parent_split(
        runtime,
        heldout_prompts,
        split="heldout-evaluation",
        horizon=horizon_value,
        limits=limits,
    )
    if (
        qwen3_attention_runtime_snapshot(runtime) != parent_snapshot
        or role_search._runtime_state_identity_sha256(runtime) != parent_identity
    ):
        raise QwenP35BehaviorVOTunerError(
            "Collecting fixed parent functional traces changed the p35 runtime."
        )

    proposed_snapshot: Qwen3AttentionRuntimeSnapshot | None = None
    lineage: dict[str, Any] | None = None
    tuning_result: BehaviorVOTuningResult | None = None
    tentative_reset_exact = False
    incumbent_rollback_exact = False
    try:
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
        tuning_result = tune_exact_p35_residual_attention_vo(
            runtime,
            proposed_registry,
            parent_fit,
            parent_calibration,
            parent_heldout,
            controls=settings,
            trace_limits=limits,
        )
        _assert_tuner_report_contract(tuning_result)
        tentative_reset_exact = (
            qwen3_attention_runtime_snapshot(runtime) == proposed_snapshot
        )
        if not tentative_reset_exact:
            raise QwenP35BehaviorVOTunerError(
                "Behavior tuning did not restore the tentative 279-prototype child exactly."
            )
    finally:
        failure_in_flight = sys.exc_info()[0] is not None
        restoration_errors: list[Exception] = []
        try:
            restored = attention_surgery._restore_installed(
                runtime, parent_installation
            )
            incumbent_rollback_exact = bool(
                restored
                and qwen3_attention_runtime_snapshot(runtime) == parent_snapshot
                and role_search._runtime_state_identity_sha256(runtime)
                == parent_identity
            )
            if not incumbent_rollback_exact:
                restoration_errors.append(
                    QwenP35BehaviorVOTunerError(
                        "Runner did not restore the incumbent p35 runtime exactly."
                    )
                )
        except Exception as error:  # pragma: no cover - defensive cleanup
            restoration_errors.append(error)
        if restoration_errors and not failure_in_flight:
            raise QwenP35BehaviorVOTunerError(
                "Behavior-tuner cleanup failed: "
                + "; ".join(str(error) for error in restoration_errors)
            ) from restoration_errors[0]

    if (
        proposed_snapshot is None
        or lineage is None
        or tuning_result is None
        or not tentative_reset_exact
        or not incumbent_rollback_exact
    ):
        raise QwenP35BehaviorVOTunerError(
            "Behavior tuning completed without a full transaction proof."
        )
    tuner_report = tuning_result.as_dict()
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
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "redirect": redirect.as_dict(),
        "exact_p35_mode": require_exact_p35,
        "horizon": horizon_value,
        "prompt_fixture_provenance": dict(split_provenance),
        "prompt_split_audit": token_audit,
        "prompt_roles": {
            "fit": {
                "used_for_gradients": True,
                "used_for_selection": False,
                "used_for_heldout_evaluation": False,
                "parent_driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(fit_drivers.items())
                },
            },
            "calibration-selection": {
                "used_for_gradients": False,
                "used_for_selection": True,
                "used_for_heldout_evaluation": False,
                "parent_driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(calibration_drivers.items())
                },
            },
            "heldout-evaluation": {
                "used_for_gradients": False,
                "used_for_selection": False,
                "used_for_heldout_evaluation": True,
                "winner_named": False,
                "parent_driver_token_ids": {
                    str(index): list(value)
                    for index, value in sorted(heldout_drivers.items())
                },
            },
            "parent_traces_collected_once_per_lane_and_prompt": True,
        },
        "parent_trace_provenance": {
            "fit": role_search._trace_manifest(parent_fit),
            "calibration-selection": role_search._trace_manifest(
                parent_calibration
            ),
            "heldout-evaluation": role_search._trace_manifest(parent_heldout),
        },
        "parent_attention_snapshot": parent_snapshot.as_dict(),
        "proposed_attention_snapshot": proposed_snapshot.as_dict(),
        "lineage_audit": lineage,
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
        "behavior_tuning": tuner_report,
        "selected_candidate_id": tuning_result.selected_candidate_id,
        "selected_scales_data_only": tuning_result.selected_scales.as_dict(),
        "objective_contract": {
            "causal_output_action_space_only": True,
            "fixed_parent_top1_actions": True,
            "parent_output_logits_allowed_as_functional_targets": True,
            "weight_similarity_used": False,
            "qkv_similarity_used": False,
            "attention_similarity_used": False,
            "activation_similarity_used": False,
            "residual_similarity_used": False,
            "cache_value_similarity_used": False,
            "internal_trace_collector_used": False,
        },
        "accounting": {
            "parent_prototype_count": parent_snapshot.prototype_count,
            "tentative_child_prototype_count": proposed_snapshot.prototype_count,
            "prototype_count_removed": 1,
            "exact_280_to_279": bool(
                parent_snapshot.prototype_count == EXACT_PARENT_PROTOTYPE_COUNT
                and proposed_snapshot.prototype_count
                == EXACT_CHILD_PROTOTYPE_COUNT
            ),
            "parent_attention_tensor_bytes": parent_snapshot.tensor_bytes,
            "tentative_child_attention_tensor_bytes": proposed_snapshot.tensor_bytes,
            "physical_attention_bytes_saved": (
                parent_snapshot.tensor_bytes - proposed_snapshot.tensor_bytes
            ),
            "manual_tuner_runtime_parameters_added": 0,
            "manual_optimizer_state_saved_or_counted": False,
            "this_report_covers_one_redirect_only": True,
            "full_p35_to_p34_requires_redirect_count": 8,
            "tuner_accounting": tuner_report["accounting"],
        },
        "transaction": {
            "tentative_child_restored_exactly_after_tuning": tentative_reset_exact,
            "incumbent_p35_runtime_restored_exactly": incumbent_rollback_exact,
            "selected_candidate_installed_after_return": False,
            "parent_functional_traces_collected_before_tentative_install": True,
        },
        "checkpoint_saved": False,
        "fold_committed": False,
        "compaction_certified": False,
        "training_used": True,
        "training_scope": "two-existing-v-o-role-scale-scalars",
    }
    report["report_sha256"] = base._sha256(report)
    return report


def _encode_prompts(
    tokenizer: Any,
    values: Sequence[str],
    *,
    max_prompt_tokens: int,
) -> tuple[Tensor, ...]:
    return role_search._encode_prompts(
        tokenizer, values, max_prompt_tokens=max_prompt_tokens
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--fit-prompts-json", type=Path, required=True)
    parser.add_argument(
        "--calibration-selection-prompts-json", type=Path, required=True
    )
    parser.add_argument(
        "--heldout-evaluation-prompts-json", type=Path, required=True
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
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-abs-scale", type=float, default=16.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--action-ce-weight", type=float, default=1.0)
    parser.add_argument("--action-margin-weight", type=float, default=0.25)
    parser.add_argument("--action-margin", type=float, default=0.5)
    parser.add_argument("--topk-distill-weight", type=float, default=0.1)
    parser.add_argument("--topk-distill-tokens", type=int, default=8)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--functional-top-k", type=int, default=5)
    parser.add_argument("--max-total-fit-actions", type=int, default=512)
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
    controls = BehaviorVOTuningControls(
        steps=args.steps,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_epsilon=args.adam_epsilon,
        max_abs_scale=args.max_abs_scale,
        gradient_clip=args.gradient_clip,
        action_ce_weight=args.action_ce_weight,
        action_margin_weight=args.action_margin_weight,
        action_margin=args.action_margin,
        topk_distill_weight=args.topk_distill_weight,
        topk_distill_tokens=args.topk_distill_tokens,
        distill_temperature=args.distill_temperature,
        functional_top_k=args.functional_top_k,
        max_prompts_per_split=max(EXPECTED_SPLIT_COUNTS.values()),
        max_horizon=args.horizon,
        max_total_fit_actions=args.max_total_fit_actions,
    )
    limits = FunctionalDecodeLimits(
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_vocabulary=args.max_vocabulary,
        max_logit_bytes=args.max_logit_bytes,
        max_model_devices=1,
    )
    report = run_qwen_p35_behavior_vo_tuner(
        runtime,
        fit_prompts,
        calibration_prompts,
        heldout_prompts,
        split_provenance=fixtures.provenance,
        controls=controls,
        horizon=args.horizon,
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
        "tuner": "manual-two-scalar-adam",
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
                "selected_candidate_id": report["selected_candidate_id"],
                "selected_scales_data_only": report[
                    "selected_scales_data_only"
                ],
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
