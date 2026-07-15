#!/usr/bin/env python3
"""Prepare the model-free v4 guided-GUD candidate tournament package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import prepare_qwen_exact_target_native_continuation_splits as split_tools
import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament
import run_qwen_exact_target_native_stage as stage


BUILD_SCHEMA = "neural-forge-guided-gud-candidate-tournament-build-v1"
SPLIT_SELECTION_SCHEMA = "neural-forge-guided-gud-fresh-candidate-selection-v1"
DEFAULT_OUTPUT = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_guided_gud_candidate_tournament_v4"
)
DEFAULT_SOURCE_PREREGISTRATION = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_guided_candidate_tournament_portable_v3_20260713/"
    "preregistration.json"
)
DEFAULT_SOURCE_RUN_ROOT = Path(
    "/mnt/ai_data/neural-forge/runs/"
    "qwen_exact_target_native_guided_candidate_tournament_portable_v3_20260713"
)


class GuidedGUDCandidateTournamentPreparationError(RuntimeError):
    """The deterministic model-free v4 package could not be prepared."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _prompt_records(binding: stage.SplitBinding) -> tuple[Any, ...]:
    return tournament._prompt_records(binding)


def _source_row(
    *,
    tournament_preregistration: Path,
    tournament_preregistration_sha256: str,
    tournament_run_report: Path,
    winner_selection: Path,
) -> dict[str, Any]:
    winner = stage._strict_json(winner_selection)
    if not isinstance(winner, Mapping) or not isinstance(winner.get("winner"), Mapping):
        raise GuidedGUDCandidateTournamentPreparationError(
            "Source winner selection is malformed."
        )
    selected = winner["winner"]
    return {
        "tournament_preregistration": {
            "path": str(tournament_preregistration),
            "sha256": tournament_preregistration_sha256,
        },
        "tournament_run_report": {
            "path": str(tournament_run_report),
            "sha256": stage._file_sha256(tournament_run_report),
        },
        "winner_selection": {
            "path": str(winner_selection),
            "sha256": stage._file_sha256(winner_selection),
        },
        "expected_winner": {
            "arm_id": selected.get("arm_id"),
            "step": selected.get("step"),
            "runtime_factor_state_sha256": selected.get("runtime_factor_state_sha256"),
            "portable_factor_state_sha256": selected.get(
                "portable_factor_state_sha256"
            ),
            "factor_pack_file_sha256": selected.get("factor_pack_sha256"),
        },
    }


def prepare_guided_gud_splits(
    *,
    staging: Path,
    source: tournament.AuthenticatedV3WinnerSource,
    corpus_path: Path,
    tokenizer_loader: Callable[[Path], Any] = split_tools._load_tokenizer,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Promote fit168+candidate24 and select a fresh disjoint candidate24."""

    old = source.tournament_contract.splits
    if {name: old[name].expected_count for name in old} != {
        "fit": 168,
        "candidate": 24,
        "sealed": 32,
    }:
        raise GuidedGUDCandidateTournamentPreparationError(
            "Source tournament must be exact 168/24/32."
        )
    records = {name: _prompt_records(old[name]) for name in old}
    try:
        split_tools._assert_globally_disjoint(
            records,
            label="guided-GUD tournament source",
        )
        corpus = split_tools._load_corpus(
            corpus_path.expanduser().resolve(),
            expected_sha256=split_tools.PINNED_CORPUS_SHA256,
            expected_line_count=split_tools.PINNED_CORPUS_LINE_COUNT,
        )
        tokenizer_root = (
            source.tournament_contract.source.execution_contract.parent.path
        )
        tokenizer_json = tokenizer_root / "tokenizer.json"
        if (
            not tokenizer_json.is_file()
            or split_tools._file_sha256(tokenizer_json)
            != split_tools.PINNED_TOKENIZER_JSON_SHA256
        ):
            raise GuidedGUDCandidateTournamentPreparationError(
                "Pinned tokenizer changed."
            )
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
        raise GuidedGUDCandidateTournamentPreparationError(
            "Fresh exact/prefix-disjoint candidate24 selection failed."
        ) from error
    active = {
        "fit": (*records["fit"], *records["candidate"]),
        "candidate": tuple(fresh_candidate),
        "sealed": records["sealed"],
    }
    if {name: len(value) for name, value in active.items()} != tournament.SPLIT_COUNTS:
        raise GuidedGUDCandidateTournamentPreparationError(
            "Active v4 split counts changed."
        )
    try:
        split_tools._assert_globally_disjoint(
            active,
            label="guided-GUD tournament active",
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise GuidedGUDCandidateTournamentPreparationError(
            "Active v4 split hygiene failed."
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
                    "name": "old-fit168",
                    "count": 168,
                    "prompt_manifest_sha256": split_tools._component_sha256(
                        [dict(value.payload) for value in records["fit"]]
                    ),
                },
                {
                    "name": "exposed-candidate24-promoted-to-fit",
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


def prepare_guided_gud_candidate_tournament(
    *,
    output_dir: Path,
    source_tournament_preregistration: Path,
    source_tournament_preregistration_sha256: str,
    source_tournament_run_report: Path,
    source_winner_selection: Path,
    basis_bundle: Path,
    basis_bundle_sha256: str,
    site_mask: Path,
    site_mask_sha256: str,
    corpus_path: Path = split_tools.DEFAULT_CORPUS,
    tokenizer_loader: Callable[[Path], Any] = split_tools._load_tokenizer,
) -> dict[str, Any]:
    """Build, self-authenticate, and atomically publish one v4 preregistration."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise GuidedGUDCandidateTournamentPreparationError(
            "output_dir must not already exist."
        )
    prereg_source = source_tournament_preregistration.expanduser().resolve()
    run_report = source_tournament_run_report.expanduser().resolve()
    winner_selection = source_winner_selection.expanduser().resolve()
    source_row = _source_row(
        tournament_preregistration=prereg_source,
        tournament_preregistration_sha256=(source_tournament_preregistration_sha256),
        tournament_run_report=run_report,
        winner_selection=winner_selection,
    )
    source = tournament.authenticate_v3_winner_source(
        source_row,
        base=output.parent,
    )
    basis = tournament.authenticate_guided_gud_basis_bundle(
        basis_bundle,
        basis_bundle_sha256,
    )
    mask, mask_path, mask_file_sha = tournament.authenticate_guided_gud_layer_mask(
        site_mask,
        site_mask_sha256,
        expected_audit_report_sha256=basis.audit_report_sha256,
    )
    tournament.authenticate_guided_gud_mask_derivation(mask, basis)
    if (
        basis.source_v3_portable_factor_state_sha256
        != source.portable_factor_state_sha256
    ):
        raise GuidedGUDCandidateTournamentPreparationError(
            "Audit basis bundle is not bound to the exact v3 winner."
        )
    tournament.authenticate_guided_gud_basis_source(basis, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    ).resolve()
    try:
        split_rows, split_evidence = prepare_guided_gud_splits(
            staging=staging,
            source=source,
            corpus_path=corpus_path,
            tokenizer_loader=tokenizer_loader,
        )
        splits = {
            name: stage._load_split(
                split_rows[name],
                name=name,
                base=staging,
            )
            for name in ("fit", "candidate", "sealed")
        }
        tournament._authenticate_v4_split_evolution(splits, source)
        mask_copy = staging / "guided_gud_layer_mask.json"
        shutil.copyfile(mask_path, mask_copy)
        if stage._file_sha256(mask_copy) != mask_file_sha:
            raise GuidedGUDCandidateTournamentPreparationError(
                "Copied GUD mask changed bytes."
            )
        arms = tournament.expected_guided_gud_arms(source, basis, mask)
        training = {
            "schedule": tournament.guided_gud_candidate_schedule().as_dict(),
            "stages": tournament.guided_gud_stage_contract(),
        }
        selection = tournament.guided_gud_selection_contract()
        runtime = tournament.guided_gud_runtime_contract()
        implementation = tournament._implementation_contract()
        preimage = tournament.guided_gud_evaluation_preimage(
            source=source,
            splits=splits,
            basis=basis,
            mask=mask,
            mask_file_sha256=mask_file_sha,
            arms=arms,
            training=training,
            selection=selection,
            runtime=runtime,
            implementation=implementation,
        )
        prereg = {
            "schema": tournament.PREREGISTRATION_SCHEMA,
            "sealed_before_model_access": True,
            "source_v3_winner": source_row,
            "splits": split_rows,
            "basis_bundle": {
                "path": str(basis.path),
                "sha256": basis.file_sha256,
            },
            "site_mask": {
                "path": "guided_gud_layer_mask.json",
                "sha256": mask_file_sha,
            },
            "arms": [arm.as_dict() for arm in arms],
            "training": training,
            "selection": selection,
            "runtime": runtime,
            "implementation": implementation,
            "evaluation_contract_sha256": tournament.canonical_sha256(preimage),
        }
        prereg_path = staging / "preregistration.json"
        _write_json(prereg_path, prereg)
        prereg_sha = stage._file_sha256(prereg_path)
        contract = tournament.authenticate_guided_gud_candidate_preregistration(
            prereg_path,
            prereg_sha,
        )
        report = {
            "schema": BUILD_SCHEMA,
            "preregistration": str(output / "preregistration.json"),
            "preregistration_sha256": prereg_sha,
            "evaluation_contract_sha256": contract.evaluation_contract_sha256,
            "source_v3_runtime_factor_state_sha256": (
                source.runtime_factor_state_sha256
            ),
            "source_v3_portable_factor_state_sha256": (
                source.portable_factor_state_sha256
            ),
            "basis_payload_sha256": basis.payload_sha256,
            "audit_report_sha256": basis.audit_report_sha256,
            "site_mask_contract_sha256": mask.mask_contract_sha256,
            "split_evidence": split_evidence,
            "arm_count": len(arms),
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
            raise GuidedGUDCandidateTournamentPreparationError(
                "output_dir appeared during preparation."
            )
        os.replace(staging, output)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-tournament-preregistration",
        type=Path,
        default=DEFAULT_SOURCE_PREREGISTRATION,
    )
    parser.add_argument(
        "--source-tournament-preregistration-sha256",
        default=tournament.EXPECTED_SOURCE_TOURNAMENT_PREREGISTRATION_SHA256,
    )
    parser.add_argument(
        "--source-tournament-run-report",
        type=Path,
        default=DEFAULT_SOURCE_RUN_ROOT / "run_report.json",
    )
    parser.add_argument(
        "--source-winner-selection",
        type=Path,
        default=DEFAULT_SOURCE_RUN_ROOT / "winner_selection.json",
    )
    parser.add_argument("--basis-bundle", type=Path, required=True)
    parser.add_argument("--basis-bundle-sha256", required=True)
    parser.add_argument("--site-mask", type=Path, required=True)
    parser.add_argument("--site-mask-sha256", required=True)
    parser.add_argument("--corpus-path", type=Path, default=split_tools.DEFAULT_CORPUS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_guided_gud_candidate_tournament(
        output_dir=args.output_dir,
        source_tournament_preregistration=(args.source_tournament_preregistration),
        source_tournament_preregistration_sha256=(
            args.source_tournament_preregistration_sha256
        ),
        source_tournament_run_report=args.source_tournament_run_report,
        source_winner_selection=args.source_winner_selection,
        basis_bundle=args.basis_bundle,
        basis_bundle_sha256=args.basis_bundle_sha256,
        site_mask=args.site_mask,
        site_mask_sha256=args.site_mask_sha256,
        corpus_path=args.corpus_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
