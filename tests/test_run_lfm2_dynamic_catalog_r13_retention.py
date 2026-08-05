"""Runner side of the R13 retention lane: flags, carve-out, depth telemetry.

Kept in its own file so the hunks stay clearly separated from the in-flight
bridge-projection work in ``test_run_lfm2_dynamic_catalog.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_dynamic_catalog.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_dynamic_catalog_r13",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config(root: Path, **overrides):
    return MODULE.ProductionLfm2CatalogConfig(
        parent_model=root / "parent",
        calibration=None,
        sealed_parent_cache_directory=root / "cache",
        cached_parent_outputs=root / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=root / "runtime-cache",
        snapshot_root=root / "snapshots",
        publish_destination=root / "published",
        ledger_path=root / "ledger.jsonl",
        receipt_path=root / "receipt.json",
        device="cpu",
        dtype="float32",
        **overrides,
    )


def test_retention_lane_defaults_are_all_off(tmp_path) -> None:
    config = _config(tmp_path)
    assert config.dense_closed_loop_terminal_composition_floor is False
    assert config.dense_closed_loop_terminal_free_run_tripwire is False
    assert config.dense_closed_loop_input_error_covariance is False
    assert config.dense_closed_loop_input_error_multiplier == 0.0
    assert config.dense_closed_loop_margin_row_weights is False
    assert config.dense_closed_loop_solver_conditioning_diagnostics is False
    assert config.dense_closed_loop_gate_operating_point_diagnostics is False
    assert config.dense_closed_loop_depth_telemetry is False


def test_retention_lane_requires_the_closed_loop_body_fold(tmp_path) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires the dense closed-loop body fold",
    ):
        _config(
            tmp_path,
            dense_closed_loop_terminal_composition_floor=True,
        )


def test_tripwire_stop_rung_requires_the_tripwire(tmp_path) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="stop flag requires the tripwire",
    ):
        _config(
            tmp_path,
            dense_run_closed_loop=True,
            dense_closed_loop_terminal_free_run_tripwire_stops=True,
        )


def test_retention_lane_accepts_the_closed_loop_configuration(
    tmp_path,
) -> None:
    config = _config(
        tmp_path,
        dense_run_closed_loop=True,
        dense_closed_loop_terminal_composition_floor=True,
        dense_closed_loop_terminal_regression_tolerance=0.4,
        dense_closed_loop_terminal_absolute_floor_minimum_layer=3,
        dense_closed_loop_input_error_covariance=True,
        dense_closed_loop_input_error_multiplier=0.25,
        dense_closed_loop_margin_row_weights=True,
        dense_closed_loop_solver_conditioning_diagnostics=True,
        dense_closed_loop_gate_operating_point_diagnostics=True,
        dense_closed_loop_depth_telemetry=True,
    )
    assert config.dense_closed_loop_terminal_regression_tolerance == 0.4
    assert config.dense_closed_loop_terminal_absolute_floor_minimum_layer == 3
    assert config.dense_closed_loop_input_error_multiplier == 0.25


def test_negative_multiplier_is_refused(tmp_path) -> None:
    with pytest.raises(MODULE.ProductionLfm2CatalogError):
        _config(
            tmp_path,
            dense_run_closed_loop=True,
            dense_closed_loop_input_error_covariance=True,
            dense_closed_loop_input_error_multiplier=-1.0,
        )


def test_parser_exposes_every_retention_flag(tmp_path) -> None:
    parser = MODULE.build_parser()
    options = {
        action.option_strings[0]
        for action in parser._actions
        if action.option_strings
    }
    for flag in (
        "--dense-closed-loop-terminal-composition-floor",
        "--dense-closed-loop-terminal-regression-tolerance",
        "--dense-closed-loop-terminal-absolute-floor-minimum-layer",
        "--dense-closed-loop-terminal-free-run-tripwire",
        "--dense-closed-loop-terminal-free-run-tripwire-stops",
        "--dense-closed-loop-input-error-covariance",
        "--dense-closed-loop-input-error-multiplier",
        "--dense-closed-loop-margin-row-weights",
        "--dense-closed-loop-solver-conditioning-diagnostics",
        "--dense-closed-loop-gate-operating-point-diagnostics",
        "--dense-closed-loop-depth-telemetry",
    ):
        assert flag in options


# --- item 8: depth telemetry -------------------------------------------------


def _plan_with_closed_loop_report():
    return SimpleNamespace(
        report={
            "closed_loop": {
                "layers": [
                    {
                        "scope": {"layer_index": 0},
                        "validation": {
                            "accepted": True,
                            "reason": "held-out post-layer residual improved",
                            "decision_sha256": "a" * 64,
                            "metrics": {
                                "candidate_relative_rmse": 0.13,
                                "terminal_relative_rmse": 0.41,
                            },
                        },
                    },
                    {
                        "scope": {"layer_index": 1},
                        "validation": {
                            "accepted": True,
                            "reason": "held-out post-layer residual improved",
                            "decision_sha256": "b" * 64,
                            "metrics": {
                                "candidate_relative_rmse": 0.21,
                                "terminal_relative_rmse": 0.55,
                            },
                        },
                    },
                ],
                "body_reclosure": {
                    "validation": {"accepted": True, "metrics": {}},
                },
            }
        }
    )


def _evaluator_with_depth_table():
    trajectory = SimpleNamespace(
        events=[
            {
                "phase": "accepted_in_memory_child",
                "child_trace_rows_sha256": "c" * 64,
                "summary": {"free_run_token_match": 0.0044},
                "report": {
                    "mean_residual_cosine": 0.574,
                    "mean_residual_relative_rmse": 0.809,
                    "residual_layers": [
                        {"layer_index": 0, "relative_rmse": 0.13},
                        {"layer_index": 1, "relative_rmse": 0.55},
                    ],
                },
            }
        ]
    )
    return SimpleNamespace(trajectory_evaluator=trajectory)


def test_depth_telemetry_is_absent_by_default(tmp_path) -> None:
    config = _config(tmp_path)
    assert (
        MODULE._closed_loop_depth_telemetry(
            config,
            _plan_with_closed_loop_report(),
            _evaluator_with_depth_table(),
        )
        is None
    )


def test_depth_telemetry_reconstructs_the_depth_profile(tmp_path) -> None:
    config = _config(
        tmp_path,
        dense_run_closed_loop=True,
        dense_closed_loop_terminal_composition_floor=True,
        dense_closed_loop_depth_telemetry=True,
    )
    telemetry = MODULE._closed_loop_depth_telemetry(
        config,
        _plan_with_closed_loop_report(),
        _evaluator_with_depth_table(),
    )
    assert telemetry is not None
    assert telemetry["per_layer_validation_decision_count"] == 2
    decisions = telemetry["per_layer_validation_decisions"]
    assert [entry["layer_index"] for entry in decisions] == [0, 1]
    # The per-layer terminal metric r12's receipt could not carry.
    assert decisions[1]["metrics"]["terminal_relative_rmse"] == 0.55
    tables = telemetry["trajectory_gate_depth_tables"]
    assert telemetry["trajectory_gate_depth_table_count"] == 1
    assert tables[0]["residual_layers"] == [
        {"layer_index": 0, "relative_rmse": 0.13},
        {"layer_index": 1, "relative_rmse": 0.55},
    ]
    assert tables[0]["mean_residual_relative_rmse"] == 0.809
    assert telemetry["body_reclosure_validation"] == {
        "accepted": True,
        "metrics": {},
    }
    assert telemetry["terminal_composition_floor_enabled"] is True


def test_depth_telemetry_survives_a_missing_closed_loop_report(
    tmp_path,
) -> None:
    config = _config(
        tmp_path,
        dense_closed_loop_depth_telemetry=True,
    )
    telemetry = MODULE._closed_loop_depth_telemetry(
        config,
        SimpleNamespace(report={"closed_loop": None}),
        None,
    )
    assert telemetry["per_layer_validation_decisions"] == []
    assert telemetry["trajectory_gate_depth_tables"] == []


def test_config_replace_keeps_the_retention_lane_validated(tmp_path) -> None:
    config = _config(tmp_path, dense_run_closed_loop=True)
    with pytest.raises(MODULE.ProductionLfm2CatalogError):
        replace(
            config,
            dense_run_closed_loop=False,
            dense_closed_loop_margin_row_weights=True,
        )
