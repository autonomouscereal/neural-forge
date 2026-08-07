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

from neural_forge.dynamic_closed_loop_fold import (
    ChildClosedLoopCompiler,
    ClosedLoopCompilerConfig,
    FoldScope,
    ValidationDecision,
)
from neural_forge.dynamic_fold_compiler import DynamicFoldCompilation
from neural_forge.dynamic_fold_executor import (
    DynamicFoldRun,
    FoldAttempt,
    FoldDecision,
)
from neural_forge.dynamic_fold_planner import DynamicFoldPlan
from neural_forge.dynamic_fold_registry import ExperimentSignature

from test_dynamic_closed_loop_fold import _ToyAdapter


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
    assert (
        config.dense_closed_loop_intermediate_numeric_checks_diagnostic_only
        is False
    )
    assert (
        config.dense_closed_loop_all_layer_finite_numeric_checks_diagnostic_only
        is False
    )
    assert config.dense_closed_loop_terminal_free_run_tripwire is False
    assert config.dense_closed_loop_input_error_covariance is False
    assert config.dense_closed_loop_input_error_multiplier == 0.0
    assert config.dense_closed_loop_margin_row_weights is False
    assert config.dense_closed_loop_solver_conditioning_diagnostics is False
    assert config.dense_closed_loop_gate_operating_point_diagnostics is False
    assert config.dense_closed_loop_depth_telemetry is False
    assert (
        "dense_closed_loop_finite_numeric_policy"
        not in config.identity()["target"]
    )


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
        dense_closed_loop_intermediate_numeric_checks_diagnostic_only=True,
        dense_closed_loop_all_layer_finite_numeric_checks_diagnostic_only=True,
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
    assert config.dense_closed_loop_intermediate_numeric_checks_diagnostic_only
    assert config.dense_closed_loop_all_layer_finite_numeric_checks_diagnostic_only
    numeric_policy = config.identity()["target"][
        "dense_closed_loop_finite_numeric_policy"
    ]
    performance_policy = config.identity()["target"][
        "dense_closed_loop_performance_metrics"
    ]
    assert performance_policy["diagnostic_only"] is True
    assert performance_policy["numeric_decision_authority"] is False
    assert numeric_policy["intermediate_layers_diagnostic_only"] is True
    assert numeric_policy["all_layers_diagnostic_only"] is True
    assert numeric_policy["nonfinite_values_reject"] is True
    receipt_policy = MODULE._dense_closed_loop_performance_metrics_policy(
        config
    )
    assert receipt_policy["diagnostic_only"] is True
    assert receipt_policy["numeric_decision_authority"] is False
    assert receipt_policy[
        "intermediate_layers_finite_numeric_diagnostic_only"
    ] is True
    assert receipt_policy[
        "all_layers_finite_numeric_diagnostic_only"
    ] is True
    assert receipt_policy["final_finite_numeric_decision_authority"] is False


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
        "--dense-closed-loop-intermediate-numeric-checks-diagnostic-only",
        "--dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only",
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


def _real_closed_loop_report():
    """The closed-loop report exactly as the real compiler emits it.

    NOTES-W13M section 3 caught two independent defects at once: the telemetry
    read ``plan.report``, and ``DynamicFoldPlan`` has no ``report`` field, and
    it read ``closed_loop["layers"]``, and the compiler emits ``layer_reports``.
    The old fixture was a hand-made ``SimpleNamespace`` that matched neither, so
    the test passed against a function that could never work.  This one comes
    out of ``ChildClosedLoopCompiler`` itself, so the shape cannot drift here
    without failing.
    """

    return ChildClosedLoopCompiler(
        _ToyAdapter(),
        ClosedLoopCompilerConfig(
            validation_fraction=0.4,
            split_seed="r13-depth-telemetry",
            fold_body_reclosure=True,
        ),
    ).run()


def _fold_run_carrying(closed_loop_report):
    """Wrap one closed-loop report the way the dense budget fold really does.

    ``hf_lfm2_dense_budget_model_fold`` seals it under ``closed_loop`` on the
    transaction report of the fold attempt, and ``DynamicFoldRun``/
    ``FoldAttempt`` are the real containers the catalog receives.
    """

    signature = ExperimentSignature(
        method_id="lfm2-dense-budget-model-fold",
        method_version=1,
        family_id="lfm2-dense-budget",
        parent_graph_fingerprints=("1" * 64,),
        parent_state_fingerprints=(("parent", "2" * 64),),
        child_graph_fingerprint="3" * 64,
        child_state_fingerprint="4" * 64,
        evidence_fingerprint="5" * 64,
        source_unit_ids=("parent:body",),
        target_unit_ids=("child:body",),
        topology_transition="dense-budget",
        selection_policy="catalog",
        fit_objective="closed-form-least-squares",
        promotion_gate="held-out-residual",
        zero_gradient=True,
    )
    attempt = FoldAttempt(
        action_id="lfm2-dense-body",
        method_id="lfm2-dense-budget-model-fold",
        execution_signature=signature,
        evidence_fingerprint="5" * 64,
        evidence_root_fingerprint="6" * 64,
        incumbent_state_fingerprint="7" * 64,
        candidate_state_fingerprint="8" * 64,
        decision=FoldDecision.accept("held-out residual improved"),
        transaction_report={"closed_loop": closed_loop_report},
        rollback_exact=None,
        ledger_record_sha256="9" * 64,
    )
    return DynamicFoldRun(
        plan_fingerprint="a" * 64,
        status="complete",
        initial_state_fingerprint="b" * 64,
        final_state_fingerprint="c" * 64,
        bootstrap_report={},
        attempts=(attempt,),
        accepted_action_ids=("lfm2-dense-body",),
        completed_target_parameter_paths=(),
        final_decision=FoldDecision.accept("published"),
        final_evidence_fingerprint="5" * 64,
        final_evidence_root_fingerprint="6" * 64,
        ledger_fingerprint="d" * 64,
    )


def _run_with_closed_loop_report():
    return _fold_run_carrying(_real_closed_loop_report())


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


def test_depth_telemetry_reads_the_shapes_the_program_really_emits() -> None:
    """Pin the two shapes W13M caught the telemetry guessing wrong.

    Both defects were silent: ``getattr(plan, "report", None)`` returns ``None``
    for a ``slots=True`` dataclass without that field, and ``.get("layers")``
    returns ``None`` for a key that never existed.  If either shape drifts
    again, this fails instead of the telemetry going quietly empty.
    """

    # 1. The object the catalog used to read has no report at all.
    assert DynamicFoldCompilation.__dataclass_fields__["plan"].type in {
        "DynamicFoldPlan",
        DynamicFoldPlan,
    }
    assert "report" not in DynamicFoldPlan.__dataclass_fields__
    assert not hasattr(DynamicFoldPlan, "report")
    # 2. The report lives on the fold attempt's transaction report, and the
    #    call site now passes `compilation.run`, which is that container.
    assert DynamicFoldCompilation.__dataclass_fields__["run"].type in {
        "DynamicFoldRun",
        DynamicFoldRun,
    }
    assert "attempts" in DynamicFoldRun.__dataclass_fields__
    assert "transaction_report" in FoldAttempt.__dataclass_fields__
    # 3. The compiler's per-layer table is `layer_reports`, never `layers`.
    report = _real_closed_loop_report()
    assert "layer_reports" in report
    assert "layers" not in report
    assert "body_reclosure" in report
    # 4. And each entry carries the scope/validation shape the telemetry reads,
    #    built by the real classes rather than asserted against a literal.
    entry = report["layer_reports"][0]
    assert set(FoldScope("layer", 0).as_dict()) <= set(entry["scope"])
    decision = ValidationDecision(
        scope=FoldScope("layer", 0),
        accepted=True,
        validation_record_keys=(("text", "batch-0"),),
        reason="held-out post-layer residual improved",
        metrics=(("candidate_relative_rmse", 0.13),),
    )
    assert set(decision.as_dict()) <= set(entry["validation"])
    for key in ("accepted", "reason", "decision_sha256", "metrics"):
        assert key in entry["validation"], key

    run = _fold_run_carrying(report)
    assert MODULE._closed_loop_transaction_report(run) is report


def test_depth_telemetry_is_absent_by_default(tmp_path) -> None:
    config = _config(tmp_path)
    assert (
        MODULE._closed_loop_depth_telemetry(
            config,
            _run_with_closed_loop_report(),
            _evaluator_with_depth_table(),
        )
        is None
    )


def test_depth_telemetry_reconstructs_the_depth_profile(tmp_path) -> None:
    config = _config(
        tmp_path,
        dense_run_closed_loop=True,
        dense_closed_loop_terminal_composition_floor=True,
        dense_closed_loop_intermediate_numeric_checks_diagnostic_only=True,
        dense_closed_loop_all_layer_finite_numeric_checks_diagnostic_only=True,
        dense_closed_loop_depth_telemetry=True,
    )
    closed_loop = _real_closed_loop_report()
    telemetry = MODULE._closed_loop_depth_telemetry(
        config,
        _fold_run_carrying(closed_loop),
        _evaluator_with_depth_table(),
    )
    expected_layers = closed_loop["layer_reports"]
    assert telemetry is not None
    # Every layer the compiler validated, and no invented ones.
    assert telemetry["per_layer_validation_decision_count"] == len(
        expected_layers
    )
    assert telemetry["per_layer_validation_decision_count"] > 0
    decisions = telemetry["per_layer_validation_decisions"]
    assert [entry["layer_index"] for entry in decisions] == [
        entry["scope"]["layer_index"] for entry in expected_layers
    ]
    for entry, expected in zip(decisions, expected_layers, strict=True):
        assert entry["accepted"] == expected["validation"]["accepted"]
        assert entry["reason"] == expected["validation"]["reason"]
        assert entry["decision_sha256"] == (
            expected["validation"]["decision_sha256"]
        )
        # The per-layer metrics r12's receipt could not carry.
        assert entry["metrics"] == dict(expected["validation"]["metrics"])
        assert entry["metrics"]
    tables = telemetry["trajectory_gate_depth_tables"]
    assert telemetry["trajectory_gate_depth_table_count"] == 1
    assert tables[0]["residual_layers"] == [
        {"layer_index": 0, "relative_rmse": 0.13},
        {"layer_index": 1, "relative_rmse": 0.55},
    ]
    assert tables[0]["mean_residual_relative_rmse"] == 0.809
    assert telemetry["body_reclosure_validation"] == dict(
        closed_loop["body_reclosure"]["validation"]
    )
    assert telemetry["body_reclosure_validation"]["accepted"] is True
    assert telemetry["terminal_composition_floor_enabled"] is True
    assert telemetry["intermediate_numeric_checks_diagnostic_only"] is True
    assert telemetry["all_layer_finite_numeric_checks_diagnostic_only"] is True


@pytest.mark.parametrize(
    "run",
    [
        None,
        SimpleNamespace(attempts=()),
        SimpleNamespace(attempts=(SimpleNamespace(transaction_report={}),)),
        SimpleNamespace(
            attempts=(
                SimpleNamespace(transaction_report={"closed_loop": None}),
            )
        ),
    ],
)
def test_depth_telemetry_survives_a_missing_closed_loop_report(
    tmp_path,
    run,
) -> None:
    config = _config(
        tmp_path,
        dense_closed_loop_depth_telemetry=True,
    )
    telemetry = MODULE._closed_loop_depth_telemetry(config, run, None)
    assert MODULE._closed_loop_transaction_report(run) is None
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
