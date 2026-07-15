#!/usr/bin/env python3
"""Build the model-free one-shot guided-GUD winner publication contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import run_qwen_exact_target_native_guided_gud_winner_publication as publication
import run_qwen_exact_target_native_stage as stage


BUILD_SCHEMA = "neural-forge-qwen-guided-gud-winner-publication-build-v1"


class WinnerPublicationPreparationError(RuntimeError):
    """The model-free guided-GUD publication contract could not be written."""


def prepare_winner_publication(
    *,
    output_dir: Path | str,
    tournament_preregistration: Path | str,
    tournament_run_report: Path | str,
    winner_selection: Path | str,
    dtype: str,
    parent_device_map: str,
    target_device_map: str,
) -> dict[str, Any]:
    output = publication._resolve(
        output_dir,
        base=Path.cwd(),
        name="output_dir",
    )
    if output.exists() or output.name in {"", ".", ".."}:
        raise WinnerPublicationPreparationError(
            "output_dir must be one absent named directory."
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise WinnerPublicationPreparationError(
            "output_dir parent must be an existing real directory."
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    ).resolve()
    renamed = False
    try:
        tournament_prereg = publication._resolve(
            tournament_preregistration,
            base=Path.cwd(),
            name="guided-GUD tournament preregistration",
        )
        tournament_report = publication._resolve(
            tournament_run_report,
            base=Path.cwd(),
            name="guided-GUD tournament run report",
        )
        winner_path = publication._resolve(
            winner_selection,
            base=Path.cwd(),
            name="guided-GUD winner selection",
        )
        payload, _contract, winner, _factor_payload = (
            publication.build_winner_publication_preregistration_payload(
                tournament_preregistration=tournament_prereg,
                tournament_preregistration_sha256=stage._file_sha256(
                    tournament_prereg
                ),
                tournament_run_report=tournament_report,
                tournament_run_report_sha256=stage._file_sha256(tournament_report),
                winner_selection=winner_path,
                winner_selection_sha256=stage._file_sha256(winner_path),
                sealed_consumption_marker=output / "sealed_consumption.json",
                dtype=dtype,
                parent_device_map=parent_device_map,
                target_device_map=target_device_map,
                builder_path=Path(__file__).resolve(),
            )
        )
        preregistration = staging / "preregistration.json"
        stage._atomic_json(preregistration, payload)
        preregistration_sha = stage._file_sha256(preregistration)
        authenticated = publication.authenticate_winner_publication_preregistration(
            preregistration,
            preregistration_sha,
        )
        if authenticated.sealed_marker.exists():
            raise WinnerPublicationPreparationError(
                "Fresh sealed consumption marker unexpectedly exists."
            )
        report = {
            "schema": BUILD_SCHEMA,
            "status": "ready",
            "preregistration": str(output / "preregistration.json"),
            "preregistration_sha256": preregistration_sha,
            "semantic_contract_sha256": payload["semantic_contract_sha256"],
            "winner_portable_v4_factor_state_sha256": winner[
                "portable_factor_state_sha256"
            ],
            "winner_arm_id": winner["arm_id"],
            "winner_basis_method": winner["basis_method"],
            "winner_mask_mode": winner["mask_mode"],
            "authenticated_arm_count": len(publication.tournament.ARM_IDS),
            "authenticated_callback_count": (
                publication.tournament.EXPECTED_CALLBACK_COUNT
            ),
            "sealed_consumption_marker": str(authenticated.sealed_marker),
            "sealed_marker_absent": True,
            "sealed_expected_count": 32,
            "model_free_preregistration": True,
            "model_weights_loaded": False,
            "gpu_work_launched": False,
            "publication_attempted": False,
            "winner_locked_before_sealed": True,
            "sealed_reranking_forbidden": True,
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
            raise WinnerPublicationPreparationError(
                "Atomic guided-GUD publication tree verification failed."
            )
        final_preregistration = output / "preregistration.json"
        final_authenticated = (
            publication.authenticate_winner_publication_preregistration(
                final_preregistration,
                stage._file_sha256(final_preregistration),
            )
        )
        if final_authenticated.sealed_marker.exists():
            raise WinnerPublicationPreparationError(
                "Final sealed consumption marker unexpectedly exists."
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
    parser.add_argument("--dtype", choices=("float16",), required=True)
    parser.add_argument("--parent-device-map", required=True)
    parser.add_argument("--target-device-map", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_winner_publication(
        output_dir=args.output_dir,
        tournament_preregistration=args.tournament_preregistration,
        tournament_run_report=args.tournament_run_report,
        winner_selection=args.winner_selection,
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
