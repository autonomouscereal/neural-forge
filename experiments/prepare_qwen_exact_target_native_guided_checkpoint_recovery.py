#!/usr/bin/env python3
"""Prepare the model-free guided schemafix-v2 checkpoint recovery contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_qwen_exact_target_native_guided_checkpoint_recovery as recovery
from neural_forge.hf_guided_checkpoint_recovery import file_sha256


RECOVERY_BUILD_SCHEMA = "neural-forge-qwen-guided-checkpoint-recovery-build-v1"
DEFAULT_CURRENT_GUIDED_PREREGISTRATION = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_stage1_step24_guided_qkvo_checkpoint_recovery_v1_"
    "20260712/current_guided/preregistration.json"
)
DEFAULT_HISTORICAL_GUIDED_PREREGISTRATION = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_stage1_step24_guided_qkvo_continuation_"
    "schemafix_v2_20260712/preregistration.json"
)
DEFAULT_HISTORICAL_GUIDED_RUN_REPORT = Path(
    "/mnt/ai_data/neural-forge/runs/"
    "qwen_exact_target_native_stage1_step24_guided_qkvo_continuation_"
    "schemafix_v2_20260712/run_report.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_stage1_step24_guided_qkvo_checkpoint_recovery_v1_"
    "20260712/recovery"
)


class GuidedCheckpointRecoveryPreparationError(RuntimeError):
    """The model-free recovery contract could not be published safely."""


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _cleanup_owned_staging(staging: Path, *, output: Path) -> None:
    if not staging.exists():
        return
    resolved = staging.resolve()
    if (
        resolved.parent != output.parent.resolve()
        or not resolved.name.startswith(f".{output.name}.")
        or not resolved.name.endswith(".staging")
        or resolved.is_symlink()
    ):
        raise GuidedCheckpointRecoveryPreparationError(
            "Refusing to clean an unowned recovery staging path."
        )
    shutil.rmtree(resolved)


def prepare_guided_checkpoint_recovery(
    *,
    output_dir: Path,
    current_guided_preregistration: Path,
    historical_guided_preregistration: Path = (
        DEFAULT_HISTORICAL_GUIDED_PREREGISTRATION
    ),
    historical_guided_run_report: Path = DEFAULT_HISTORICAL_GUIDED_RUN_REPORT,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise GuidedCheckpointRecoveryPreparationError(
            "output_dir must not already exist."
        )
    current_path = current_guided_preregistration.expanduser().resolve()
    historical_prereg_path = historical_guided_preregistration.expanduser().resolve()
    historical_report_path = historical_guided_run_report.expanduser().resolve()
    for path, name in (
        (current_path, "current guided preregistration"),
        (historical_prereg_path, "historical guided preregistration"),
        (historical_report_path, "historical guided run report"),
    ):
        if path.is_symlink() or not path.is_file():
            raise GuidedCheckpointRecoveryPreparationError(f"{name} is absent.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    try:
        preregistration, _current = recovery.build_recovery_preregistration_payload(
            current_guided_preregistration=current_path,
            current_guided_preregistration_sha256=file_sha256(current_path),
            historical_guided_preregistration=historical_prereg_path,
            historical_guided_preregistration_sha256=file_sha256(
                historical_prereg_path
            ),
            historical_guided_run_report=historical_report_path,
            historical_guided_run_report_sha256=file_sha256(historical_report_path),
            builder_path=Path(__file__).resolve(),
        )
        prereg_path = staging / "preregistration.json"
        _write_json_exclusive(prereg_path, preregistration)
        prereg_sha = file_sha256(prereg_path)
        contract = recovery.authenticate_recovery_preregistration(
            prereg_path, prereg_sha
        )
        report = {
            "schema": RECOVERY_BUILD_SCHEMA,
            "status": "ready",
            "preregistration": str(output / "preregistration.json"),
            "preregistration_sha256": prereg_sha,
            "current_guided_preregistration_sha256": (
                contract.raw["current_guided"]["preregistration_sha256"]
            ),
            "historical_guided_preregistration_sha256": (
                contract.raw["historical_guided"]["preregistration_sha256"]
            ),
            "historical_guided_run_report_sha256": (
                contract.raw["historical_guided"]["run_report_sha256"]
            ),
            "historical_trainer_evidence_sha256": (
                contract.raw["historical_guided"]["trainer_evidence_sha256"]
            ),
            "checkpoint_expectations_sha256": (
                contract.raw["checkpoint_expectations_sha256"]
            ),
            "semantic_contract_sha256": contract.raw["semantic_contract_sha256"],
            "retained_steps": list(contract.retained_steps),
            "source_implementation": dict(contract.raw["source_implementation"]),
            "model_free_preregistration": True,
            "model_weights_loaded": False,
            "model_forward_calls": 0,
            "gpu_work_launched": False,
            "training_launched": False,
            "research_pack_written": False,
            "ready_for_explicit_preflight": True,
        }
        _write_json_exclusive(staging / "build_report.json", report)
        if output.exists():
            raise GuidedCheckpointRecoveryPreparationError(
                "output_dir appeared during preparation."
            )
        os.rename(staging, output)
        return report
    except BaseException:
        _cleanup_owned_staging(staging, output=output)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--current-guided-preregistration",
        type=Path,
        default=DEFAULT_CURRENT_GUIDED_PREREGISTRATION,
    )
    parser.add_argument(
        "--historical-guided-preregistration",
        type=Path,
        default=DEFAULT_HISTORICAL_GUIDED_PREREGISTRATION,
    )
    parser.add_argument(
        "--historical-guided-run-report",
        type=Path,
        default=DEFAULT_HISTORICAL_GUIDED_RUN_REPORT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_guided_checkpoint_recovery(
        output_dir=args.output_dir,
        current_guided_preregistration=args.current_guided_preregistration,
        historical_guided_preregistration=args.historical_guided_preregistration,
        historical_guided_run_report=args.historical_guided_run_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
