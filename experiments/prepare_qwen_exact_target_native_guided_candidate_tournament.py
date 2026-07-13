#!/usr/bin/env python3
"""Prepare the model-free guided candidate-tournament contract.

The current guided fit144/candidate24 becomes fit168, a deterministic fresh
candidate24 is selected after exact and meaningful-prefix exclusion, and the
sealed32 artifact is reused without copying or reading it through a model.  Two
recovered v3 states and the audit-derived role-specific site mask are bound into
one immutable three-arm contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

import prepare_qwen_exact_target_native_continuation_splits as split_tools
import run_qwen_exact_target_native_guided_candidate_tournament as tournament
import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_guided_attention_site_mask import (
    build_guided_attention_site_mask,
)
from neural_forge.hf_guided_checkpoint_recovery import (
    GUIDED_CHECKPOINT_CANDIDATE_SCHEMA,
    GuidedCheckpointRecoverySink,
    canonical_sha256,
    portable_guided_source_binding_sha256,
    portable_v3_factor_state_sha256,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
)


BUILD_SCHEMA = "neural-forge-guided-candidate-tournament-build-v1"
SPLIT_SELECTION_SCHEMA = "neural-forge-guided-candidate-tournament-selection-v1"
DEFAULT_OUTPUT = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_guided_candidate_tournament_20260712"
)


class CandidateTournamentPreparationError(RuntimeError):
    """The model-free tournament package could not be authenticated."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload), allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _prompt_records(binding: stage.SplitBinding) -> tuple[Any, ...]:
    prompts = binding.payload.get("prompts")
    if not isinstance(prompts, list):
        raise CandidateTournamentPreparationError("Source prompt list is absent.")
    try:
        return tuple(
            split_tools._prompt_record(value, f"source.{binding.name}.prompts[{index}]")
            for index, value in enumerate(prompts)
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise CandidateTournamentPreparationError(
            "Source prompt authentication failed."
        ) from error


def prepare_tournament_splits(
    *,
    staging: Path,
    source: guided.AuthenticatedGuidedContinuation,
    corpus_path: Path,
    tokenizer_path: Path,
    tokenizer_loader: Callable[[Path], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create fit168/candidate24 bindings while leaving sealed32 untouched."""

    old = source.execution_contract.splits
    if {name: old[name].expected_count for name in old} != {
        "fit": 144,
        "candidate": 24,
        "sealed": 32,
    }:
        raise CandidateTournamentPreparationError(
            "Guided tournament source must be exact 144/24/32."
        )
    records = {name: _prompt_records(old[name]) for name in old}
    try:
        split_tools._assert_globally_disjoint(records, label="guided tournament source")
        corpus = split_tools._load_corpus(
            corpus_path.expanduser().resolve(),
            expected_sha256=split_tools.PINNED_CORPUS_SHA256,
            expected_line_count=split_tools.PINNED_CORPUS_LINE_COUNT,
        )
        tokenizer_root = tokenizer_path.expanduser().resolve()
        tokenizer_json = tokenizer_root / "tokenizer.json"
        if (
            not tokenizer_json.is_file()
            or split_tools._file_sha256(tokenizer_json)
            != split_tools.PINNED_TOKENIZER_JSON_SHA256
        ):
            raise CandidateTournamentPreparationError("Pinned tokenizer changed.")
        tokenizer = tokenizer_loader(tokenizer_root)
        fresh_candidate, selection_stats = split_tools._select_fresh_candidate(
            corpus,
            tokenizer,
            excluded_prompts=tuple(
                row for name in ("fit", "candidate", "sealed") for row in records[name]
            ),
            max_prompt_tokens=split_tools.MAX_PROMPT_TOKENS,
            expected_count=24,
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise CandidateTournamentPreparationError(
            "Fresh exact/prefix-disjoint candidate24 selection failed."
        ) from error
    active = {
        "fit": (*records["fit"], *records["candidate"]),
        "candidate": tuple(fresh_candidate),
        "sealed": records["sealed"],
    }
    if {name: len(value) for name, value in active.items()} != tournament.SPLIT_COUNTS:
        raise CandidateTournamentPreparationError("Active split count changed.")
    try:
        split_tools._assert_globally_disjoint(active, label="guided tournament active")
    except split_tools.ContinuationSplitPreparationError as error:
        raise CandidateTournamentPreparationError(
            "Active split hygiene failed."
        ) from error
    old_hashes = {name: old[name].sha256 for name in old}
    fit_prompts = [dict(value.payload) for value in active["fit"]]
    candidate_prompts = [dict(value.payload) for value in active["candidate"]]
    payloads = {
        "fit": split_tools._split_payload(
            "fit",
            fit_prompts,
            original_hashes=old_hashes,
            corpus_sha256=split_tools.PINNED_CORPUS_SHA256,
            corpus_line_count=split_tools.PINNED_CORPUS_LINE_COUNT,
            tokenizer_sha256=split_tools.PINNED_TOKENIZER_JSON_SHA256,
            components=(
                {
                    "name": "old-fit144",
                    "count": 144,
                    "prompt_manifest_sha256": split_tools._component_sha256(
                        [dict(value.payload) for value in records["fit"]]
                    ),
                },
                {
                    "name": "old-candidate24-promoted-to-fit",
                    "count": 24,
                    "prompt_manifest_sha256": split_tools._component_sha256(
                        [dict(value.payload) for value in records["candidate"]]
                    ),
                },
            ),
            selection_schema=SPLIT_SELECTION_SCHEMA,
        ),
        "candidate": split_tools._split_payload(
            "candidate",
            candidate_prompts,
            original_hashes=old_hashes,
            corpus_sha256=split_tools.PINNED_CORPUS_SHA256,
            corpus_line_count=split_tools.PINNED_CORPUS_LINE_COUNT,
            tokenizer_sha256=split_tools.PINNED_TOKENIZER_JSON_SHA256,
            components=(
                {
                    "name": "fresh-candidate24",
                    "count": 24,
                    "prompt_manifest_sha256": split_tools._component_sha256(
                        candidate_prompts
                    ),
                },
            ),
            selection_schema=SPLIT_SELECTION_SCHEMA,
        ),
    }
    split_dir = staging / "splits"
    split_dir.mkdir()
    for name, payload in payloads.items():
        _write_json(split_dir / f"{name}.json", payload)
    bindings = {
        name: {
            "path": f"splits/{name}.json",
            "sha256": stage._file_sha256(split_dir / f"{name}.json"),
            "expected_count": tournament.SPLIT_COUNTS[name],
        }
        for name in ("fit", "candidate")
    }
    bindings["sealed"] = {
        "path": str(old["sealed"].path),
        "sha256": old["sealed"].sha256,
        "expected_count": 32,
    }
    evidence = {
        "source_counts": {name: old[name].expected_count for name in old},
        "active_counts": dict(tournament.SPLIT_COUNTS),
        "old_candidate_promoted_to_fit": True,
        "new_candidate_exact_and_prefix_disjoint": True,
        "sealed_reused_by_exact_path_and_hash": True,
        "selection_statistics": selection_stats,
        "model_weights_loaded": False,
        "model_forward_calls": 0,
    }
    return bindings, evidence


def _recovered_binding(
    *, pack_path: Path, recovery_record_path: Path, expected_step: int
) -> dict[str, Any]:
    raw_pack = pack_path.expanduser()
    raw_record = recovery_record_path.expanduser()
    if raw_pack.is_symlink() or raw_record.is_symlink():
        raise CandidateTournamentPreparationError("Recovery inputs cannot be symlinks.")
    pack = raw_pack.resolve()
    record_path = raw_record.resolve()
    if not pack.is_file() or not record_path.is_file():
        raise CandidateTournamentPreparationError("Recovery inputs are absent.")
    record = stage._strict_json(record_path)
    if not isinstance(record, Mapping):
        raise CandidateTournamentPreparationError("Recovered record is not a mapping.")
    record_sha = record.get("candidate_record_sha256")
    if (
        not isinstance(record_sha, str)
        or canonical_sha256(
            {
                name: value
                for name, value in record.items()
                if name != "candidate_record_sha256"
            }
        )
        != record_sha
    ):
        raise CandidateTournamentPreparationError("Recovered record self-hash changed.")
    factor_sha = record.get("factor_state_sha256")
    if not isinstance(factor_sha, str):
        raise CandidateTournamentPreparationError("Recovery record lacks factor SHA.")
    try:
        loaded = torch.load(pack, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CandidateTournamentPreparationError(
            "Cannot safely load v3 recovery."
        ) from error
    payload: Any = loaded
    if isinstance(loaded, Mapping) and isinstance(loaded.get("factor_pack"), Mapping):
        payload = loaded["factor_pack"]
    if not isinstance(payload, Mapping):
        raise CandidateTournamentPreparationError("Recovered factor payload is absent.")
    try:
        GuidedCheckpointRecoverySink._validate_v3_factor_payload(
            payload, expected_factor_state_sha256=factor_sha
        )
    except Exception as error:
        raise CandidateTournamentPreparationError(
            "Recovered v3 pack is invalid."
        ) from error
    portable_sha = portable_v3_factor_state_sha256(payload)
    events = record.get("events")
    retained_steps = record.get("retained_steps")
    matching_events = (
        [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("step") == expected_step
        ]
        if isinstance(events, list)
        else []
    )
    pack_sha = stage._file_sha256(pack)
    if (
        record.get("schema") != GUIDED_CHECKPOINT_CANDIDATE_SCHEMA
        or record.get("factor_state_sha256") != factor_sha
        or record.get("portable_factor_state_sha256") != portable_sha
        or record.get("portable_guided_source_binding_sha256")
        != portable_guided_source_binding_sha256(payload["guided_source_binding"])
        or record.get("runtime_identity_excluded_from_portable_content")
        != ["inherited_installation_sha256"]
        or record.get("content_address") != factor_sha
        or record.get("contains_parent_state") is not False
        or record.get("contains_native_child_state") is not False
        or record.get("contains_optimizer_state") is not False
        or record.get("contains_model_checkpoint") is not False
        or record.get("factor_pack_relative_path") != pack.name
        or record_path.parent / str(record.get("factor_pack_relative_path")) != pack
        or record.get("factor_pack_file_sha256") != pack_sha
        or record.get("factor_pack_size_bytes") != pack.stat().st_size
        or not isinstance(events, list)
        or record.get("event_count") != len(events)
        or retained_steps
        != [event.get("step") for event in events if isinstance(event, Mapping)]
        or expected_step not in retained_steps
        or len(matching_events) != 1
        or matching_events[0].get("factor_state_sha256") != factor_sha
        or matching_events[0].get("portable_factor_state_sha256") != portable_sha
    ):
        raise CandidateTournamentPreparationError(
            f"Recovered record does not authenticate retained step {expected_step}."
        )
    return {
        "path": str(pack),
        "sha256": pack_sha,
        "factor_state_sha256": factor_sha,
        "portable_factor_state_sha256": portable_sha,
        "source_step": expected_step,
        "recovery_record": {
            "path": str(record_path),
            "sha256": stage._file_sha256(record_path),
        },
    }


def prepare_candidate_tournament(
    *,
    output_dir: Path,
    source_guided_preregistration: Path,
    source_guided_preregistration_sha256: str,
    step40_pack: Path,
    step40_recovery_record: Path,
    step64_pack: Path,
    step64_recovery_record: Path,
    corpus_path: Path = split_tools.DEFAULT_CORPUS,
    tokenizer_loader: Callable[[Path], Any] = split_tools._load_tokenizer,
) -> dict[str, Any]:
    """Build and authenticate one complete model-free tournament package."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise CandidateTournamentPreparationError("output_dir must not already exist.")
    try:
        source = guided.authenticate_guided_preregistration(
            source_guided_preregistration, source_guided_preregistration_sha256
        )
    except guided.GuidedContinuationError as error:
        raise CandidateTournamentPreparationError(
            "Source guided preregistration failed authentication."
        ) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    ).resolve()
    try:
        split_bindings, split_evidence = prepare_tournament_splits(
            staging=staging,
            source=source,
            corpus_path=corpus_path,
            tokenizer_path=source.execution_contract.parent.path,
            tokenizer_loader=tokenizer_loader,
        )
        initial_states = {
            tournament.ARM_IDS[0]: _recovered_binding(
                pack_path=step40_pack,
                recovery_record_path=step40_recovery_record,
                expected_step=40,
            ),
            tournament.ARM_IDS[1]: _recovered_binding(
                pack_path=step64_pack,
                recovery_record_path=step64_recovery_record,
                expected_step=64,
            ),
        }
        continuation = source.execution_contract.continuation
        if continuation is None:
            raise CandidateTournamentPreparationError("Guided source lost v2 state.")
        mask = build_guided_attention_site_mask(
            source.artifacts.audit_report,
            expected_report_sha256=source.artifacts.audit_report_self_sha256,
        )
        mask_path = staging / "guided_attention_site_mask.json"
        _write_json(mask_path, mask.as_dict())
        arms = [
            {
                "arm_id": tournament.ARM_IDS[0],
                "initial_schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
                "initial_factor_state_sha256": initial_states[tournament.ARM_IDS[0]][
                    "factor_state_sha256"
                ],
                "initial_portable_factor_state_sha256": initial_states[
                    tournament.ARM_IDS[0]
                ]["portable_factor_state_sha256"],
                "site_mask_enabled": False,
            },
            {
                "arm_id": tournament.ARM_IDS[1],
                "initial_schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
                "initial_factor_state_sha256": initial_states[tournament.ARM_IDS[1]][
                    "factor_state_sha256"
                ],
                "initial_portable_factor_state_sha256": initial_states[
                    tournament.ARM_IDS[1]
                ]["portable_factor_state_sha256"],
                "site_mask_enabled": False,
            },
            {
                "arm_id": tournament.ARM_IDS[2],
                "initial_schema": BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
                "initial_factor_state_sha256": continuation.factor_state_sha256,
                "initial_portable_factor_state_sha256": None,
                "site_mask_enabled": True,
            },
        ]
        selection = {
            "domain": "candidate-output-actions-and-free-rollouts-only",
            "rank_order": [
                "exact_rollout_count",
                "lcp_at_least_4_count",
                "lcp_token_count",
                "token_match_count",
                "repeat_fraction_excess",
                "adjacent_repeat_fraction_excess",
                "longest_run_excess",
                "action_agreement_count",
            ],
            "retain_distinct_full_v3_states": 3,
            "distinct_identity": "portable-v3-factor-content",
            "rank_step_zero": True,
            "later_states_require_pareto_safety_vs_step_zero": True,
            "sealed_disabled": True,
            "stop_after_candidate_winner": True,
        }
        runtime = {
            "dtype": "float16",
            "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
            "exact_source_reset_between_arms": True,
            "durable_callback_every_arm": True,
            "mask_verify_after_each_optimizer_step": True,
        }
        implementation = {
            "runner_sha256": stage._file_sha256(Path(tournament.__file__).resolve()),
            "builder_sha256": stage._file_sha256(Path(__file__).resolve()),
        }
        schedule = tournament.candidate_tournament_schedule().as_dict()
        preimage = {
            "source_guided_preregistration_sha256": source.file_sha256,
            "split_sha256s": {
                name: value["sha256"] for name, value in split_bindings.items()
            },
            "initial_state_sha256s": {
                name: {
                    "file": value["sha256"],
                    "factor": value["factor_state_sha256"],
                    "portable_factor": value["portable_factor_state_sha256"],
                    "record": value["recovery_record"]["sha256"],
                }
                for name, value in initial_states.items()
            },
            "site_mask_file_sha256": stage._file_sha256(mask_path),
            "site_mask_contract_sha256": mask.mask_contract_sha256,
            "arms": arms,
            "training": schedule,
            "selection": selection,
            "runtime": runtime,
            "implementation": implementation,
        }
        prereg = {
            "schema": tournament.PREREGISTRATION_SCHEMA,
            "sealed_before_model_access": True,
            "source_guided": {
                "path": str(source.path),
                "sha256": source.file_sha256,
            },
            "splits": split_bindings,
            "initial_states": initial_states,
            "site_mask": {
                "path": "guided_attention_site_mask.json",
                "sha256": stage._file_sha256(mask_path),
            },
            "arms": arms,
            "training": schedule,
            "selection": selection,
            "runtime": runtime,
            "implementation": implementation,
            "evaluation_contract_sha256": canonical_sha256(preimage),
        }
        prereg_path = staging / "preregistration.json"
        _write_json(prereg_path, prereg)
        prereg_sha = stage._file_sha256(prereg_path)
        contract = tournament.authenticate_candidate_tournament_preregistration(
            prereg_path, prereg_sha
        )
        report = {
            "schema": BUILD_SCHEMA,
            "preregistration": str(output / "preregistration.json"),
            "preregistration_sha256": prereg_sha,
            "evaluation_contract_sha256": contract.evaluation_contract_sha256,
            "split_evidence": split_evidence,
            "site_mask_contract_sha256": mask.mask_contract_sha256,
            "initial_factor_state_sha256s": {
                name: value.factor_state_sha256
                for name, value in contract.initial_states.items()
            },
            "models_loaded": False,
            "model_forward_calls": 0,
            "gpu_work_launched": False,
            "training_launched": False,
            "model_checkpoint_written": False,
            "sealed_evaluation_accessed": False,
            "ready_for_explicit_execution": True,
        }
        _write_json(staging / "build_report.json", report)
        if output.exists():
            raise CandidateTournamentPreparationError("output_dir appeared.")
        os.replace(staging, output)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-guided-preregistration", type=Path, required=True)
    parser.add_argument("--source-guided-preregistration-sha256", required=True)
    parser.add_argument("--step40-pack", type=Path, required=True)
    parser.add_argument("--step40-recovery-record", type=Path, required=True)
    parser.add_argument("--step64-pack", type=Path, required=True)
    parser.add_argument("--step64-recovery-record", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, default=split_tools.DEFAULT_CORPUS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_candidate_tournament(
        output_dir=args.output_dir,
        source_guided_preregistration=args.source_guided_preregistration,
        source_guided_preregistration_sha256=(
            args.source_guided_preregistration_sha256
        ),
        step40_pack=args.step40_pack,
        step40_recovery_record=args.step40_recovery_record,
        step64_pack=args.step64_pack,
        step64_recovery_record=args.step64_recovery_record,
        corpus_path=args.corpus_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
