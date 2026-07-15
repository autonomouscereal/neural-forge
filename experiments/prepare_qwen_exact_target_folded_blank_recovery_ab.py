#!/usr/bin/env python3
"""Build the model-free folded-vs-blank exact-target recovery A/B contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import run_qwen_exact_target_folded_blank_recovery_ab as recovery_ab
import run_qwen_exact_target_native_stage as stage


BUILD_SCHEMA = "neural-forge-folded-blank-recovery-ab-build-v1"


class FoldedBlankRecoveryABPreparationError(RuntimeError):
    """The model-free A/B preregistration could not be published safely."""


def prepare_recovery_ab(
    *,
    output_dir: Path | str,
    tournament_preregistration: Path | str,
    tournament_run_report: Path | str,
    winner_selection: Path | str,
    replay_bank: Path | str,
    recovery_plan: Path | str,
) -> dict[str, Any]:
    output = recovery_ab.publication._resolve(
        output_dir,
        base=Path.cwd(),
        name="recovery A/B output directory",
    )
    if output.exists() or output.name in {"", ".", ".."}:
        raise FoldedBlankRecoveryABPreparationError(
            "output_dir must be one absent named directory."
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FoldedBlankRecoveryABPreparationError(
            "output_dir parent must be an existing real directory."
        )
    paths = {
        "tournament_preregistration": recovery_ab.publication._resolve(
            tournament_preregistration,
            base=Path.cwd(),
            name="v4 candidate tournament preregistration",
        ),
        "tournament_run_report": recovery_ab.publication._resolve(
            tournament_run_report,
            base=Path.cwd(),
            name="v4 candidate tournament run report",
        ),
        "winner_selection": recovery_ab.publication._resolve(
            winner_selection,
            base=Path.cwd(),
            name="v4 retained winner selection",
        ),
        "replay_bank": recovery_ab.publication._resolve(
            replay_bank,
            base=Path.cwd(),
            name="frozen recovery replay bank",
        ),
        "recovery_plan": recovery_ab.publication._resolve(
            recovery_plan,
            base=Path.cwd(),
            name="recovery A/B plan",
        ),
    }
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise FoldedBlankRecoveryABPreparationError(
                f"{name} must be an existing real file."
            )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    ).resolve()
    renamed = False
    try:
        payload, _v4 = recovery_ab.build_recovery_ab_preregistration_payload(
            tournament_preregistration=paths["tournament_preregistration"],
            tournament_preregistration_sha256=stage._file_sha256(
                paths["tournament_preregistration"]
            ),
            tournament_run_report=paths["tournament_run_report"],
            tournament_run_report_sha256=stage._file_sha256(
                paths["tournament_run_report"]
            ),
            winner_selection=paths["winner_selection"],
            winner_selection_sha256=stage._file_sha256(paths["winner_selection"]),
            replay_bank=paths["replay_bank"],
            replay_bank_sha256=stage._file_sha256(paths["replay_bank"]),
            recovery_plan=paths["recovery_plan"],
            recovery_plan_sha256=stage._file_sha256(paths["recovery_plan"]),
            builder_path=Path(__file__).resolve(),
        )
        preregistration = staging / "preregistration.json"
        stage._atomic_json(preregistration, payload)
        preregistration_sha = stage._file_sha256(preregistration)
        authenticated = recovery_ab.authenticate_recovery_ab_preregistration(
            preregistration,
            preregistration_sha,
        )
        preflight = authenticated.preflight_report()
        report = {
            "schema": BUILD_SCHEMA,
            "status": "ready-experimental-unaccepted",
            "preregistration": str(output / "preregistration.json"),
            "preregistration_sha256": preregistration_sha,
            "semantic_contract_sha256": payload["semantic_contract_sha256"],
            "v4_candidate_only": True,
            "winner_portable_factor_state_sha256": payload["v4_candidate"]["winner"][
                "portable_factor_state_sha256"
            ],
            "compiled_blank_state_sha256": payload["compiled_blank"][
                "state_dict_sha256"
            ],
            "exact_unique_parameters": recovery_ab.EXACT_UNIQUE_PARAMETERS,
            "arms": list(recovery_ab.ARM_IDS),
            "shared_execution_contract_sha256": payload[
                "shared_execution_contract_sha256"
            ],
            "maximum_retained_checkpoints_per_arm": (
                recovery_ab.MAX_RETAINED_CHECKPOINTS_PER_ARM
            ),
            "maximum_total_retained_checkpoints": (
                recovery_ab.MAX_TOTAL_RETAINED_CHECKPOINTS
            ),
            "preflight_status": preflight["status"],
            "model_free_preregistration": True,
            "model_weights_loaded": False,
            "gpu_work_launched": False,
            "training_launched": False,
            "materialization_launched": False,
            "zero_step_evaluation_launched": False,
            "sealed_data_accessed": False,
            "publication_authorized": False,
            "accepted_artifact": False,
            "production_model_hook_implemented": True,
            "explicit_production_hook_required": False,
            "ready_for_explicit_preflight": True,
        }
        stage._atomic_json(staging / "build_report.json", report)
        staged_tree = stage._artifact_tree(staging)
        os.replace(staging, output)
        renamed = True
        final_tree = stage._artifact_tree(output)
        if any(
            final_tree.get(name) != staged_tree.get(name)
            for name in ("file_count", "total_bytes", "tree_sha256")
        ):
            raise FoldedBlankRecoveryABPreparationError(
                "Atomic recovery A/B tree verification failed."
            )
        final_prereg = output / "preregistration.json"
        recovery_ab.authenticate_recovery_ab_preregistration(
            final_prereg,
            stage._file_sha256(final_prereg),
        )
        return report
    except BaseException:
        if (
            not renamed
            and staging.exists()
            and staging.parent == output.parent
            and staging.name.startswith(f".{output.name}.")
            and staging.name.endswith(".staging")
            and not staging.is_symlink()
        ):
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tournament-preregistration", type=Path, required=True)
    parser.add_argument("--tournament-run-report", type=Path, required=True)
    parser.add_argument("--winner-selection", type=Path, required=True)
    parser.add_argument("--replay-bank", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_recovery_ab(
        output_dir=args.output_dir,
        tournament_preregistration=args.tournament_preregistration,
        tournament_run_report=args.tournament_run_report,
        winner_selection=args.winner_selection,
        replay_bank=args.replay_bank,
        recovery_plan=args.recovery_plan,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
