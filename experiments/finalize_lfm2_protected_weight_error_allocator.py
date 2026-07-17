#!/usr/bin/env python3
"""Finalize a protected-weight-error run from paired cached-output ratings.

This postflight loads no model, performs no forward or generation, and never
replaces the preliminary report.  The final report remains explicitly labeled
as a noncausal ablation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from experiments import run_lfm2_causal_value_allocator as shared_runner
    from experiments import run_lfm2_protected_weight_error_allocator as runner
    from experiments import run_lfm2_value_fold_control as control_runner
except ImportError:  # Direct execution from the experiments directory.
    import run_lfm2_causal_value_allocator as shared_runner
    import run_lfm2_protected_weight_error_allocator as runner
    import run_lfm2_value_fold_control as control_runner

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_parent_relative_manual_gate import (
    evaluate_parent_relative_manual_retention,
)


FINAL_SCHEMA = "neural-forge-lfm2-protected-weight-error-final-report-v1"


class Lfm2ProtectedWeightErrorFinalizeError(RuntimeError):
    """A preliminary ablation report, packet, or rating binding is invalid."""


def _read(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    try:
        return runner._read_json(
            path,
            expected_file_sha256=expected_sha256,
            name=name,
        )
    except runner.Lfm2ProtectedWeightErrorRunnerError as exc:
        raise Lfm2ProtectedWeightErrorFinalizeError(str(exc)) from exc


def finalize_lfm2_protected_weight_error_allocator(
    *,
    output_root: Path,
    expected_preliminary_report_file_sha256: str,
    ratings_path: Path,
    expected_ratings_file_sha256: str,
) -> dict[str, Any]:
    root = control_runner._local_directory(output_root, "output_root")
    preliminary_path = control_runner._local_file(
        root / "report.json",
        "preliminary_report",
    )
    packet_path = control_runner._local_file(
        root / "parent_relative_manual_packet.json",
        "parent_relative_manual_packet",
    )
    intelligence_path = control_runner._local_file(
        root / "intelligence_report.json",
        "intelligence_report",
    )
    preliminary = _read(
        preliminary_path,
        expected_preliminary_report_file_sha256,
        "preliminary_report",
    )
    preliminary_body = dict(preliminary)
    preliminary_seal = preliminary_body.pop("report_sha256", None)
    if preliminary_body.get(
        "schema"
    ) != runner.TARGET_REPORT_SCHEMA or preliminary_seal != canonical_sha256(
        preliminary_body
    ):
        raise Lfm2ProtectedWeightErrorFinalizeError(
            "Preliminary protected-weight-error report seal is invalid."
        )
    allocation_binding = preliminary.get("allocation_binding")
    if (
        not isinstance(allocation_binding, Mapping)
        or allocation_binding.get("ablation_label") != "protected_weight_error"
        or allocation_binding.get("cached_causal_frontier_used") is not False
        or allocation_binding.get("causal_evidence_used") is not False
        or allocation_binding.get("fixed_causal_multiplier") != 1.0
    ):
        raise Lfm2ProtectedWeightErrorFinalizeError(
            "Preliminary report relabels the noncausal ablation."
        )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    intelligence = json.loads(intelligence_path.read_text(encoding="utf-8"))
    if (
        not isinstance(packet, Mapping)
        or not isinstance(intelligence, Mapping)
        or packet.get("packet_sha256")
        != preliminary["holdout"]["parent_relative_manual_packet_sha256"]
        or intelligence.get("report_sha256")
        != preliminary["holdout"]["automatic_intelligence_report_sha256"]
    ):
        raise Lfm2ProtectedWeightErrorFinalizeError(
            "Cached packet/intelligence files differ from the preliminary report."
        )
    ratings = _read(
        control_runner._local_file(ratings_path, "ratings_path"),
        expected_ratings_file_sha256,
        "parent_relative_manual_ratings",
    )
    manual = evaluate_parent_relative_manual_retention(packet, ratings)
    decision = shared_runner._parent_relative_decision(intelligence, manual)
    manual_path = root / "parent_relative_manual_report.json"
    final_path = root / "final_report.json"
    if manual_path.exists() or final_path.exists():
        raise Lfm2ProtectedWeightErrorFinalizeError(
            "Refusing to replace an existing manual or final report."
        )
    control_runner._write_new_json(manual_path, manual)
    body = {
        "schema": FINAL_SCHEMA,
        "preliminary_report_sha256": preliminary_seal,
        "preliminary_report_file_sha256": (expected_preliminary_report_file_sha256),
        "status": decision["status"],
        "promoted": decision["promoted"],
        "decision": decision,
        "checkpoint": dict(preliminary["checkpoint"]),
        "fresh_reload_equivalence": dict(preliminary["fresh_reload_equivalence"]),
        "holdout": {
            **dict(preliminary["holdout"]),
            "parent_relative_manual_report_sha256": manual["report_sha256"],
        },
        "parent_binding": dict(preliminary["parent_binding"]),
        "allocation_binding": dict(allocation_binding),
        "execution_audit": {
            **dict(preliminary["execution_audit"]),
            "postflight_model_loads": 0,
            "postflight_forward_calls": 0,
            "postflight_generation_calls": 0,
            "postflight_optimizer_steps": 0,
            "postflight_backward_calls": 0,
            "postflight_cached_outputs_only": True,
        },
    }
    final = {**body, "report_sha256": canonical_sha256(body)}
    control_runner._write_new_json(final_path, final)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize an exact-50% protected-weight-error ablation from paired "
            "ratings without loading any model."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-preliminary-report-file-sha256",
        required=True,
    )
    parser.add_argument("--ratings-path", type=Path, required=True)
    parser.add_argument("--expected-ratings-file-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = finalize_lfm2_protected_weight_error_allocator(
        **vars(build_parser().parse_args(argv))
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted": report["promoted"],
                "parent_relative_lower_bound": report["decision"][
                    "certified_parent_relative_lower_bound"
                ],
                "report_sha256": report["report_sha256"],
                "ablation_label": "protected_weight_error",
                "causal_evidence_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
