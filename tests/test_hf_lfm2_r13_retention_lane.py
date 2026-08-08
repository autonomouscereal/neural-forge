"""R13 retention lane: terminal composition floor, math pack, telemetry.

Every lane exercised here is additive and flag-gated.  The default-equivalence
tests are the contract: with no new argument supplied the adapter must produce
the identical sealed validation decision, the identical row alignment and the
identical compiled MLP state it produced before any of this existed.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from neural_forge.dynamic_closed_loop_fold import (  # noqa: E402
    ChildClosedLoopCompiler,
    ClosedLoopCompilerConfig,
    ClosedLoopFoldError,
    FoldScope,
    TrajectoryRecord,
)
from neural_forge.hf_closed_form_fit import FitMetrics  # noqa: E402
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (  # noqa: E402
    LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
    Lfm2DynamicClosedLoopAdapter,
    Lfm2DynamicClosedLoopAdapterConfig,
    Lfm2DynamicClosedLoopAdapterError,
    Lfm2TerminalFreeRunTripwire,
    _durable_diagnostic_json,
)
from neural_forge.hf_lfm2_macro_cell_attractor_gate import (  # noqa: E402
    Lfm2MacroCellAttractorGate,
    Lfm2MacroCellAttractorGateConfig,
)
from neural_forge.hf_lfm2_weight_complete_basis import (  # noqa: E402
    build_lfm2_weight_complete_basis,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedMlpFoldCase,
    _shifted_system_conditioning,
    _solved_system_conditioning,
    compile_lfm2_structural_residual_mlp_fold,
    lfm2_estimate_input_error_covariance,
)

from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)

# The adapter suite already owns the exact toy parent/child pair this lane has
# to be measured on; reusing its builders keeps the two suites describing one
# model rather than two that drift apart.
from test_hf_lfm2_dynamic_closed_loop_adapter import (  # noqa: E402
    _adapter,
    _batch,
    _model,
)


@pytest.fixture(scope="module")
def sealed_parent(tmp_path_factory):
    parent = _model(hidden=8, intermediate=12, heads=2, seed=1801)
    batches = [
        _batch((1, 2, 3, 4), (5,)),
        _batch((6, 7, 8, 9), (10,)),
        _batch((11, 12, 13, 14), (15,)),
        _batch((16, 17, 18, 19), (20,)),
    ]
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=2,
        max_decode_steps=2,
        max_capture_bytes=64 * 1024 * 1024,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("lfm2-r13-retention-parent"),
        parent,
        batches,
        config=config,
        input_device="cpu",
    )
    return parent, cache, config


def _metrics(relative_rmse: float, *, rows: int = 8) -> FitMetrics:
    return FitMetrics(
        rows=rows,
        output_width=4,
        rmse=relative_rmse,
        target_rms=1.0,
        relative_rmse=relative_rmse,
        mean_cosine_similarity=0.5,
        finite=True,
    )


def _layer_scope(layer_index: int) -> FoldScope:
    return FoldScope(kind="layer", layer_index=layer_index)


_RECORDS = (
    TrajectoryRecord(
        trajectory_id="trajectory-000",
        path_id="path-000",
        record_sha256="0" * 64,
    ),
)


# --- item 2: the terminal composition floor ----------------------------------


def test_disabled_terminal_floor_writes_no_metrics_and_no_reason(
    sealed_parent,
) -> None:
    """The unwired lane must seal the exact decision it sealed before."""

    adapter, _child, _mapping = _adapter(sealed_parent)
    decision = adapter._validation_decision(
        scope=_layer_scope(0),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        # Even handed a catastrophic composed endpoint, an off lane ignores it.
        terminal_baseline=_metrics(0.2),
        terminal_candidate=_metrics(9.0),
    )
    keys = dict(decision.metrics)
    assert decision.accepted is True
    assert not [name for name in keys if name.startswith("terminal_")]
    assert "TERMINAL" not in decision.reason


def test_terminal_floor_fails_closed_at_the_no_branch_baseline(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        performance_metrics_diagnostic_only=True,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.9),
        terminal_candidate=_metrics(1.0),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is False
    assert "no-branch baseline" in decision.reason
    assert metrics["terminal_relative_rmse"] == pytest.approx(1.0)
    assert metrics["terminal_no_branch_relative_rmse_baseline"] == 1.0
    assert metrics["terminal_absolute_floor_passed"] is False
    assert metrics["terminal_composition_floor_failed"] is True
    # The floor keeps authority even though the performance metrics are
    # explicitly diagnostic-only.  That is the whole point of the carve-out.
    assert metrics["performance_metrics_diagnostic_only"] is True
    assert (
        metrics["terminal_floor_retains_authority_under_diagnostic_only"]
        is True
    )


def test_terminal_floor_accepts_below_the_no_branch_baseline(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.80),
        terminal_candidate=_metrics(0.81),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is True
    assert metrics["terminal_absolute_floor_passed"] is True
    assert metrics["terminal_non_regression_passed"] is True
    assert metrics["terminal_relative_rmse_regression"] == pytest.approx(0.01)


def test_terminal_floor_rejects_a_depth_monotone_regression(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        terminal_composition_regression_tolerance=0.05,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.30),
        terminal_candidate=_metrics(0.50),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is False
    assert "regressed by" in decision.reason
    assert metrics["terminal_non_regression_passed"] is False
    assert metrics["terminal_absolute_floor_passed"] is True


def test_intermediate_numeric_policy_keeps_finite_passing_decision_accepted(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        intermediate_layer_numeric_checks_diagnostic_only=True,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(0),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.2),
        terminal_candidate=_metrics(0.3),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is True
    assert metrics["counterfactual_performance_threshold_passed"] is True
    assert metrics["layer_validation_is_incomplete"] is True
    assert metrics["intermediate_layer_numeric_checks_diagnostic_only"] is True
    assert metrics["all_layer_finite_numeric_checks_diagnostic_only"] is False
    assert (
        metrics[
            "local_performance_metrics_diagnostic_only_for_this_scope"
        ]
        is True
    )
    assert (
        metrics["finite_numeric_policy_diagnostic_only_for_this_scope"]
        is True
    )


def test_intermediate_numeric_policy_records_finite_reject_and_continues(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        terminal_composition_regression_tolerance=0.0,
        intermediate_layer_numeric_checks_diagnostic_only=True,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(0),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.8),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.2),
        terminal_candidate=_metrics(1.2),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is True
    assert metrics["counterfactual_performance_threshold_passed"] is False
    assert metrics["terminal_composition_floor_failed"] is True
    assert metrics["terminal_floor_failure_is_diagnostic_only"] is True
    assert metrics["terminal_floor_has_decision_authority"] is False
    assert "folded body is incomplete" in decision.reason


def test_intermediate_numeric_policy_keeps_final_layer_floor_strict(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        intermediate_layer_numeric_checks_diagnostic_only=True,
        performance_metrics_diagnostic_only=True,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.2),
        terminal_candidate=_metrics(1.2),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is False
    assert metrics["layer_validation_is_incomplete"] is False
    assert metrics["terminal_floor_failure_is_diagnostic_only"] is False
    assert metrics["terminal_floor_has_decision_authority"] is True
    assert (
        metrics[
            "local_performance_metrics_diagnostic_only_for_this_scope"
        ]
        is True
    )
    assert (
        metrics["finite_numeric_policy_diagnostic_only_for_this_scope"]
        is False
    )


def test_all_layer_numeric_policy_records_final_finite_reject_and_continues(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        terminal_composition_regression_tolerance=0.0,
        all_layer_finite_numeric_checks_diagnostic_only=True,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.8),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.2),
        terminal_candidate=_metrics(1.2),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is True
    assert metrics["layer_validation_is_incomplete"] is False
    assert metrics["all_layer_finite_numeric_checks_diagnostic_only"] is True
    assert metrics["intermediate_layer_numeric_checks_diagnostic_only"] is False
    assert metrics["counterfactual_performance_threshold_passed"] is False
    assert metrics["terminal_floor_failure_is_diagnostic_only"] is True
    assert metrics["terminal_floor_has_decision_authority"] is False
    assert (
        metrics["finite_numeric_policy_diagnostic_only_for_this_scope"]
        is True
    )
    assert "manual whole-output review" in decision.reason


def test_all_layer_numeric_policy_still_rejects_final_nonfinite_metrics(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        all_layer_finite_numeric_checks_diagnostic_only=True,
    )
    nonfinite = replace(
        _metrics(0.4),
        rmse=float("nan"),
        relative_rmse=float("nan"),
        mean_cosine_similarity=float("nan"),
        finite=False,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=nonfinite,
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.2),
        terminal_candidate=nonfinite,
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is False
    assert "nonfinite" in decision.reason
    assert metrics["performance_metrics_finite"] is False
    assert metrics["terminal_floor_metrics_finite"] is False
    assert metrics["terminal_floor_failure_is_diagnostic_only"] is False
    assert metrics["terminal_floor_has_decision_authority"] is True


def test_absolute_floor_engagement_depth_is_configurable(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        terminal_composition_non_regression_enabled=False,
        terminal_composition_absolute_floor_minimum_layer_index=3,
    )
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        terminal_baseline=_metrics(0.9),
        terminal_candidate=_metrics(2.0),
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is True
    assert metrics["terminal_absolute_floor_engaged"] is False
    assert metrics["terminal_relative_rmse"] == pytest.approx(2.0)


def test_free_run_tripwire_is_advisory_by_default(sealed_parent) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
    )
    adapter.terminal_free_run_match_probe = lambda layer_index: 0.0
    with pytest.warns(Lfm2TerminalFreeRunTripwire, match="TRIPWIRE"):
        decision = adapter._validation_decision(
            scope=_layer_scope(1),
            records=_RECORDS,
            baseline=_metrics(0.5),
            candidate=_metrics(0.4),
            minimum_improvement=0.0,
            endpoint="post-layer residual",
        )
    metrics = dict(decision.metrics)
    assert decision.accepted is True
    assert metrics["terminal_free_run_tripwire_tripped"] is True
    assert metrics["terminal_free_run_tripwire_is_advisory"] is True


def test_free_run_tripwire_can_be_promoted_to_a_stop(sealed_parent) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
        terminal_free_run_tripwire_stops_acceptance=True,
    )
    adapter.terminal_free_run_match_probe = lambda layer_index: 0.0
    with pytest.warns(Lfm2TerminalFreeRunTripwire):
        decision = adapter._validation_decision(
            scope=_layer_scope(1),
            records=_RECORDS,
            baseline=_metrics(0.5),
            candidate=_metrics(0.4),
            minimum_improvement=0.0,
            endpoint="post-layer residual",
        )
    assert decision.accepted is False
    assert "TERMINAL FREE-RUN TRIPWIRE" in decision.reason


def test_free_run_tripwire_stays_silent_above_chance(sealed_parent) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
        terminal_free_run_chance_match_fraction=0.01,
    )
    adapter.terminal_free_run_match_probe = lambda layer_index: 0.4
    with warnings.catch_warnings():
        warnings.simplefilter("error", Lfm2TerminalFreeRunTripwire)
        decision = adapter._validation_decision(
            scope=_layer_scope(1),
            records=_RECORDS,
            baseline=_metrics(0.5),
            candidate=_metrics(0.4),
            minimum_improvement=0.0,
            endpoint="post-layer residual",
        )
    assert dict(decision.metrics)["terminal_free_run_tripwire_tripped"] is False


# --- W13P item 5: the tripwire that never evaluated ---------------------------
#
# Through the whole sealed r13 run this clause reported
# ``terminal_free_run_tripwire_evaluated: false`` at all fourteen layers, cause
# unrecorded, while the run's own free-run sat at chance (0.102%).  Nothing in
# any production path has ever assigned ``terminal_free_run_match_probe``, so
# the clause could only ever record its own configuration.  It now reads the
# one counterfactual child-only rollout the adapter already performs, and when
# there is no rollout to read it says so in the sealed decision.


def test_disabled_free_run_tripwire_writes_no_metrics(sealed_parent) -> None:
    """An off tripwire seals nothing, probe assigned or not."""

    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.terminal_free_run_match_probe = lambda layer_index: 0.0
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
        free_run_match_observation=0.0,
    )
    metrics = dict(decision.metrics)
    assert not [
        name for name in metrics if name.startswith("terminal_free_run")
    ]


def test_free_run_tripwire_names_why_it_could_not_evaluate(
    sealed_parent,
) -> None:
    """The r13 silence is now self-diagnosing from the run's own evidence."""

    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
    )
    assert adapter.terminal_free_run_match_probe is None
    assert adapter.macro_cell_attractor_gate is None
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
    )
    metrics = dict(decision.metrics)
    assert decision.accepted is True
    assert metrics["terminal_free_run_tripwire_evaluated"] is False
    assert metrics["terminal_free_run_tripwire_observed"] is None
    assert metrics["terminal_free_run_tripwire_observation_source"] == "none"
    assert "no free-run observation source" in str(
        metrics["terminal_free_run_tripwire_unevaluated_reason"]
    )


def test_free_run_tripwire_names_an_assigned_probe_that_observed_nothing(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
    )
    adapter.terminal_free_run_match_probe = lambda layer_index: None
    decision = adapter._validation_decision(
        scope=_layer_scope(1),
        records=_RECORDS,
        baseline=_metrics(0.5),
        candidate=_metrics(0.4),
        minimum_improvement=0.0,
        endpoint="post-layer residual",
    )
    metrics = dict(decision.metrics)
    assert metrics["terminal_free_run_tripwire_evaluated"] is False
    assert "assigned free-run probe" in str(
        metrics["terminal_free_run_tripwire_unevaluated_reason"]
    )


def test_an_assigned_probe_outranks_the_supplied_rollout(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
        terminal_free_run_chance_match_fraction=0.01,
    )
    adapter.terminal_free_run_match_probe = lambda layer_index: 0.4
    with warnings.catch_warnings():
        warnings.simplefilter("error", Lfm2TerminalFreeRunTripwire)
        decision = adapter._validation_decision(
            scope=_layer_scope(1),
            records=_RECORDS,
            baseline=_metrics(0.5),
            candidate=_metrics(0.4),
            minimum_improvement=0.0,
            endpoint="post-layer residual",
            free_run_match_observation=0.0,
        )
    metrics = dict(decision.metrics)
    assert metrics["terminal_free_run_tripwire_observed"] == pytest.approx(0.4)
    assert metrics["terminal_free_run_tripwire_observation_source"] == (
        "assigned_probe"
    )
    assert metrics["terminal_free_run_tripwire_tripped"] is False


def test_the_macro_cell_rollout_is_read_as_the_free_run_observation() -> None:
    """The gate's own diagnostic is the observation the tripwire was spec'd on."""

    class _Snapshot:
        metrics = {
            "diagnostic_mean_parent_top1_exact_token_match_fraction": 0.375,
        }

    read = Lfm2DynamicClosedLoopAdapter._macro_cell_free_run_match
    assert read(None) is None
    assert read(_Snapshot()) == pytest.approx(0.375)


def _permissive_attractor_gate(cache):
    """A real rollout gate whose comparison cannot refuse a toy layer."""

    return Lfm2MacroCellAttractorGate(
        cache,
        config=Lfm2MacroCellAttractorGateConfig(
            maximum_records=2,
            maximum_steps=2,
            maximum_entropy_parent_distance_regression=1e9,
            maximum_repetition_excess_regression=1e9,
            maximum_dominant_token_excess_regression=1e9,
            maximum_eos_position_error_regression=1e9,
            maximum_new_collapsed_records=1_000_000,
        ),
    )


def test_the_free_run_tripwire_evaluates_inside_a_real_toy_fold(
    sealed_parent,
) -> None:
    """End to end: enabled tripwire, real rollout, an observation at every layer."""

    _parent, cache, _replay_config = sealed_parent
    gate = _permissive_attractor_gate(cache)
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        macro_cell_attractor_gate=gate,
    )
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
    )
    with warnings.catch_warnings():
        # The toy child free-runs at chance, so the advisory clause fires --
        # which is the whole point of it and must not stop the fold.
        warnings.simplefilter("ignore", Lfm2TerminalFreeRunTripwire)
        report = _run_toy_layers(adapter)
    layer_reports = report["layer_reports"]
    assert layer_reports
    for entry in layer_reports:
        metrics = dict(entry["validation"]["metrics"])
        assert entry["validation"]["accepted"] is True
        assert metrics["terminal_free_run_tripwire_enabled"] is True
        assert metrics["terminal_free_run_tripwire_evaluated"] is True
        assert metrics["terminal_free_run_tripwire_observation_source"] == (
            "macro_cell_attractor_free_rollout"
        )
        assert isinstance(
            metrics["terminal_free_run_tripwire_observed"], float
        )
        assert "terminal_free_run_tripwire_unevaluated_reason" not in metrics
        # Advisory only: the clause records and shouts, it never promotes or
        # demotes.  The gate next to it is the acceptance authority here.
        assert metrics["terminal_free_run_tripwire_is_advisory"] is True
        assert metrics["macro_cell_attractor_gate_accepted"] is True


def test_the_tripwire_rollout_is_the_gate_rollout_not_a_second_one(
    sealed_parent,
) -> None:
    """One child rollout per layer serves both instruments."""

    _parent, cache, _replay_config = sealed_parent
    gate = _permissive_attractor_gate(cache)
    phases: list[str] = []
    original_capture = gate.capture

    def _counted_capture(child_model, *, phase):
        phases.append(phase)
        return original_capture(child_model, phase=phase)

    gate.capture = _counted_capture
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        macro_cell_attractor_gate=gate,
    )
    adapter.config = replace(
        adapter.config,
        terminal_free_run_tripwire_enabled=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Lfm2TerminalFreeRunTripwire)
        report = _run_toy_layers(adapter)
    layers = len(report["layer_reports"])
    # Exactly one pre-cell and one post-cell capture per layer, unchanged from
    # the tripwire-off lane: the hoisted snapshot is reused, never re-taken.
    assert sum(1 for phase in phases if phase.startswith("pre-")) == layers
    assert sum(1 for phase in phases if phase.startswith("post-")) == layers


def test_terminal_floor_widens_the_layer_capture_selection(
    sealed_parent,
) -> None:
    from neural_forge.dynamic_closed_loop_fold import CapturePoint

    adapter, _child, _mapping = _adapter(sealed_parent)
    scope = _layer_scope(0)
    assert adapter._capture_layer_selection(
        scope, CapturePoint.PRE_PRODUCER
    ) == (0,)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
    )
    # A two-layer toy: layer 0 must also observe the terminal layer 1.
    assert adapter._capture_layer_selection(
        scope, CapturePoint.PRE_PRODUCER
    ) == (0, 1)
    assert adapter._capture_layer_selection(
        _layer_scope(1), CapturePoint.POST_DOWNSTREAM
    ) == (1,)
    profile = adapter._child_evidence_profile(
        scope,
        CapturePoint.POST_DOWNSTREAM,
        (0, 1),
        body_internal_role=None,
    )
    assert dict(profile.layer_suffixes)[1] == (
        "residual.post_layer",
        "residual.pre",
    )


def test_a_run_headed_past_the_no_branch_baseline_fails_closed_at_the_wall(
    sealed_parent,
) -> None:
    """A toy run whose composed endpoint crosses 1.0 aborts at that layer."""

    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        performance_metrics_diagnostic_only=True,
    )
    # Layer 0 composes fine; layer 1 walls at exactly the no-branch baseline.
    scripted = {0: 0.42, 1: 1.31}
    observed: list[int] = []

    def _scripted_terminal(paired, *, batch_indices):
        del paired, batch_indices
        # The scope currently under validation is the last producer recorded.
        layer_index = observed[-1] if observed else 0
        return _metrics(scripted[layer_index])

    original_producer = type(adapter).compile_producer
    events: list[tuple[str, dict[str, object]]] = []

    def _tracked_producer(self, scope, *args, **kwargs):
        observed.append(int(scope.layer_index))
        return original_producer(self, scope, *args, **kwargs)

    adapter.compile_producer = _tracked_producer.__get__(adapter)
    adapter._heldout_terminal_metrics = _scripted_terminal

    with pytest.raises(ClosedLoopFoldError) as failure:
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-r13-terminal-floor-wall-v1",
                fold_body_reclosure=False,
                fold_output_interface=False,
            ),
            progress_sink=(
                lambda stage, metadata: events.append((stage, dict(metadata)))
            ),
        ).run()
    message = str(failure.value)
    assert "no-branch baseline" in message
    assert "layer:000001 validation rejected" in message
    # Layer 0 was reached and accepted; the run stopped at the wall layer.
    assert observed == [0, 1]
    rejected = [
        metadata
        for stage, metadata in events
        if stage == "child_closed_loop_validation_decision_rejected"
    ]
    assert len(rejected) == 1
    metrics = json.loads(str(rejected[0]["validation_metrics_json"]))
    assert metrics["terminal_relative_rmse"] == pytest.approx(1.31)
    assert metrics["terminal_baseline_relative_rmse"] is not None
    assert metrics["terminal_relative_rmse_regression"] is not None
    assert metrics["terminal_absolute_floor_passed"] is False
    assert metrics["terminal_composition_floor_failed"] is True
    assert "baseline_relative_rmse" in metrics
    assert "candidate_relative_rmse" in metrics
    # NOTES-W13S §6.4: this layer publishes no restart capsule and no
    # completion receipt, so its per-solve prior-correction reports used to be
    # discarded here.  They now ride the refusal itself.
    assert metrics["refused_layer_index"] == 1
    assert metrics["refused_layer_prior_correction_solve_count"] > 0
    reports = json.loads(
        str(metrics["refused_layer_prior_correction_reports_json"])
    )
    clamped = [
        entry
        for stage_reports in reports.values()
        for entry in stage_reports
        if "prior_delta_clamp" in entry
    ]
    assert clamped, reports
    for entry in clamped:
        clamp = entry["prior_delta_clamp"]
        assert clamp["requested_prior_delta_ratio"] is not None
        assert clamp["allowed_prior_delta_ratio"] is not None
        assert isinstance(clamp["clamp_engaged"], bool)
        assert clamp["budget_units"] == "prior_frobenius_norm"


# --- W13P item 4: the A12 rule ported over the body reclosure ----------------
#
# A12 sealed the rule "reject any joint rewrite set whose closure regresses the
# preserved proven baseline" at the token interface on 2026-07-30.  The body
# reclosure was never covered by it: in the sealed r13 run it moved the composed
# held-out terminal from 0.806258 to 0.971148 -- 0.165 worse -- and was accepted,
# because the non-regression tolerance only guarded layer installs.  These three
# tests are that hole, closed.


def _toy_body_reclosure(sealed_parent, **config_overrides):
    """Run one real toy moving-gauge reclosure end to end and return the report."""

    parent, _cache, _replay_config = sealed_parent
    basis = build_lfm2_weight_complete_basis(parent, 4)
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=basis,
        body_reclosure_mode=LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
        parent_residual_operators=basis.inventory.operators,
        output_interface_mode="identity-skip",
    )
    adapter.config = replace(adapter.config, **config_overrides)
    return ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-moving-gauge-v3-contract-v1",
            fold_body_reclosure=True,
            fold_output_interface=False,
        ),
    ).run()


def test_body_reclosure_terminal_regression_is_absent_with_the_floor_off(
    sealed_parent,
) -> None:
    """The unwired lane seals no terminal evidence at the reclosure at all."""

    report = _toy_body_reclosure(
        sealed_parent,
        terminal_composition_floor_enabled=False,
    )
    validation = report["body_reclosure"]["validation"]
    metrics = dict(validation["metrics"])
    assert validation["accepted"] is True
    assert not [
        name
        for name in metrics
        if name.startswith("body_reclosure_terminal_")
    ]


def test_body_reclosure_records_a_terminal_regression_inside_tolerance(
    sealed_parent,
) -> None:
    """Within the arm's own 0.25 tolerance the reclosure stands, measured."""

    report = _toy_body_reclosure(
        sealed_parent,
        terminal_composition_floor_enabled=True,
        # The exact tolerance the sealed r13 arm ran.
        terminal_composition_regression_tolerance=0.25,
    )
    validation = report["body_reclosure"]["validation"]
    metrics = dict(validation["metrics"])
    before = metrics["body_reclosure_terminal_baseline_relative_rmse"]
    after = metrics["body_reclosure_terminal_relative_rmse"]
    assert validation["accepted"] is True
    # This toy reproduces r13's shape: the reclosure makes the composed
    # terminal endpoint worse, just not by enough to be refused.
    assert after > before
    assert metrics["body_reclosure_terminal_relative_rmse_regression"] == (
        pytest.approx(after - before)
    )
    assert metrics["body_reclosure_terminal_non_regression_engaged"] is True
    assert metrics["body_reclosure_terminal_non_regression_passed"] is True
    assert metrics["body_reclosure_terminal_composition_floor_failed"] is False
    assert metrics["body_reclosure_terminal_non_regression_tolerance"] == 0.25
    # The absolute no-branch clause is a per-install bound keyed on layer
    # depth; the ported rule is the non-regression one, and only that one.
    assert metrics["body_reclosure_terminal_absolute_floor_engaged"] is False
    assert metrics[
        "body_reclosure_terminal_non_regression_is_the_ported_a12_rule"
    ] is True
    assert metrics[
        "body_reclosure_terminal_non_regression_has_decision_authority"
    ] is True


def test_body_reclosure_fails_closed_on_a_terminal_regression(
    sealed_parent,
) -> None:
    """Beyond tolerance the run stops, naming both values and the stage."""

    events: list[tuple[str, dict[str, object]]] = []
    parent, _cache, _replay_config = sealed_parent
    basis = build_lfm2_weight_complete_basis(parent, 4)
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=basis,
        body_reclosure_mode=LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
        parent_residual_operators=basis.inventory.operators,
        output_interface_mode="identity-skip",
    )
    adapter.config = replace(
        adapter.config,
        terminal_composition_floor_enabled=True,
        # The toy reclosure regresses the terminal by ~0.0865; 0.05 refuses it.
        terminal_composition_regression_tolerance=0.05,
    )
    with pytest.raises(ClosedLoopFoldError) as failure:
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-moving-gauge-v3-contract-v1",
                fold_body_reclosure=True,
                fold_output_interface=False,
            ),
            progress_sink=(
                lambda stage, metadata: events.append((stage, dict(metadata)))
            ),
        ).run()
    message = str(failure.value)
    assert "body_reclosure validation rejected" in message
    assert "body reclosure refused" in message
    assert "before the reclosure" in message
    assert "after it" in message
    # The refusal keeps its evidence: the compiler publishes the complete
    # decision before it raises.
    rejected = [
        metadata
        for stage, metadata in events
        if stage == "child_closed_loop_validation_decision_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["scope_kind"] == "body_reclosure"
    metrics = json.loads(str(rejected[0]["validation_metrics_json"]))
    before = metrics["body_reclosure_terminal_baseline_relative_rmse"]
    after = metrics["body_reclosure_terminal_relative_rmse"]
    assert after > before
    assert metrics["body_reclosure_terminal_non_regression_passed"] is False
    assert metrics["body_reclosure_terminal_composition_floor_failed"] is True
    # Both numbers appear in the message a human reads, not only in the keys.
    assert f"{float(before):.9g}" in message
    assert f"{float(after):.9g}" in message


# --- item 7: errors-in-variables ridge ---------------------------------------


def _mlp_case(
    *,
    rows: int,
    hidden: int,
    intermediate: int,
    seed: int,
    endpoint_width: int | None = None,
):
    width = hidden if endpoint_width is None else endpoint_width
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        target_input = torch.randn(rows, hidden, dtype=torch.float64)
        parent_gate = torch.randn(rows, intermediate, dtype=torch.float64)
        parent_up = torch.randn(rows, intermediate, dtype=torch.float64)
        endpoint = torch.randn(rows, width, dtype=torch.float64)
        residual = torch.randn(rows, width, dtype=torch.float64)
    post = torch.nn.functional.silu(parent_gate) * parent_up
    return Lfm2CachedMlpFoldCase(
        target_input=target_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=residual,
    )


def _compile_toy_mlp(**overrides):
    source_hidden, source_intermediate = 6, 8
    target_hidden, target_intermediate = 3, 4
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4242)
        source_gate = torch.randn(
            source_intermediate, source_hidden, dtype=torch.float64
        )
        source_up = torch.randn(
            source_intermediate, source_hidden, dtype=torch.float64
        )
        source_down = torch.randn(
            source_hidden, source_intermediate, dtype=torch.float64
        )
        target_gate = torch.randn(
            target_intermediate, target_hidden, dtype=torch.float64
        )
        target_up = torch.randn(
            target_intermediate, target_hidden, dtype=torch.float64
        )
        target_down = torch.randn(
            target_hidden, target_intermediate, dtype=torch.float64
        )
    case = _mlp_case(
        rows=24,
        hidden=target_hidden,
        intermediate=source_intermediate,
        endpoint_width=target_hidden,
        seed=7,
    )
    return compile_lfm2_structural_residual_mlp_fold(
        source_gate_weight=source_gate,
        source_up_weight=source_up,
        source_down_weight=source_down,
        target_gate_weight_template=target_gate,
        target_up_weight_template=target_up,
        target_down_weight_template=target_down,
        cases=(case,),
        gate_weight_prior=target_gate,
        up_weight_prior=target_up,
        down_weight_prior=target_down,
        residual_assignment_sha256="ab" * 32,
        ridge=1e-6,
        **overrides,
    )


def test_zero_multiplier_input_error_covariance_is_bit_identical() -> None:
    """multiplier = 0 is the control arm and must change nothing at all."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        clean = torch.randn(32, 3, dtype=torch.float64)
        drifted = clean + 0.1 * torch.randn(32, 3, dtype=torch.float64)
    covariance, _report = lfm2_estimate_input_error_covariance(clean, drifted)
    baseline = _compile_toy_mlp()
    control = _compile_toy_mlp(
        input_error_covariance=covariance,
        input_error_scale=0.0,
    )
    assert control.compiled_state_sha256 == baseline.compiled_state_sha256
    assert torch.equal(control.gate_weight, baseline.gate_weight)
    assert torch.equal(control.up_weight, baseline.up_weight)
    assert torch.equal(control.down_weight, baseline.down_weight)


def test_nonzero_multiplier_input_error_covariance_moves_the_solution() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        clean = torch.randn(32, 3, dtype=torch.float64)
        drifted = clean + 0.1 * torch.randn(32, 3, dtype=torch.float64)
    covariance, report = lfm2_estimate_input_error_covariance(clean, drifted)
    baseline = _compile_toy_mlp()
    noise_aware = _compile_toy_mlp(
        input_error_covariance=covariance,
        input_error_scale=24.0,
    )
    assert noise_aware.compiled_state_sha256 != baseline.compiled_state_sha256
    assert not torch.equal(noise_aware.gate_weight, baseline.gate_weight)
    assert report["rank"] == 3
    assert report["rank_deficient"] is False


def test_adapter_resolves_no_covariance_on_the_unwired_lane(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    covariance, scale, evidence = adapter._mlp_input_error_covariance(
        layer_index=1,
        fit_cases=(),
    )
    assert (covariance, scale, evidence) == (None, 1.0, None)


def test_adapter_scales_the_covariance_by_the_fit_row_mass(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        mlp_input_error_covariance_enabled=True,
        mlp_input_error_robustness_multiplier=0.25,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        clean = torch.randn(16, 4, dtype=torch.float64)
        drifted = clean + 0.05 * torch.randn(16, 4, dtype=torch.float64)
    covariance, report = lfm2_estimate_input_error_covariance(clean, drifted)
    adapter._accepted_layer_input_error = (0, covariance, report)
    case = _mlp_case(rows=10, hidden=4, intermediate=6, seed=5)
    resolved, scale, evidence = adapter._mlp_input_error_covariance(
        layer_index=1,
        fit_cases=(case,),
    )
    assert resolved is covariance
    assert scale == pytest.approx(10 * 0.25)
    assert evidence["estimated_from_layer_index"] == 0
    assert evidence["applied_to_layer_index"] == 1
    assert evidence["scale_units"] == "sum_of_row_weights_times_multiplier"
    # A layer never consumes its own drift estimate.
    assert adapter._mlp_input_error_covariance(
        layer_index=0,
        fit_cases=(case,),
    ) == (None, 1.0, None)


# --- item 7: margin-derived moving-gauge row weights -------------------------


def test_margin_row_weights_default_off_leaves_the_uniform_profile(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    uniform = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
    entries = ({"batch_index": 0, "phase": "prefill", "row_count": 6},)
    weights, receipt = adapter._moving_gauge_margin_row_weights(
        entries, uniform
    )
    assert weights is None and receipt is None


def test_margin_row_weights_redistribute_without_changing_total_mass(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        moving_gauge_margin_row_weights_enabled=True,
    )
    uniform = torch.full((4,), 0.25, dtype=torch.float64)
    entries = ({"batch_index": 0, "phase": "prefill", "row_count": 4},)
    # One nearly-tied decision boundary and three comfortable ones.
    adapter.moving_gauge_margin_tables = {
        (0, "prefill"): torch.tensor(
            [[0.001], [4.0], [4.0], [4.0]], dtype=torch.float64
        )
    }
    weights, receipt = adapter._moving_gauge_margin_row_weights(
        entries, uniform
    )
    assert weights is not None
    assert float(weights.sum().item()) == pytest.approx(1.0)
    assert float(weights[0].item()) > float(weights[1].item())
    ratio = float(weights.max().item()) / float(weights.min().item())
    assert ratio <= (
        adapter.config.moving_gauge_margin_row_weight_ceiling
        / adapter.config.moving_gauge_margin_row_weight_floor
    ) + 1e-9
    assert receipt["terminal_and_lexical_writer_fits_excluded"] is True


def test_a_constant_margin_table_reproduces_the_uniform_profile(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        moving_gauge_margin_row_weights_enabled=True,
    )
    uniform = torch.full((5,), 0.2, dtype=torch.float64)
    entries = ({"batch_index": 3, "phase": "decode.000", "row_count": 5},)
    adapter.moving_gauge_margin_tables = {
        (3, "decode.000"): torch.full((5, 2), 0.75, dtype=torch.float64)
    }
    weights, _receipt = adapter._moving_gauge_margin_row_weights(
        entries, uniform
    )
    assert torch.allclose(weights, uniform, rtol=0.0, atol=1e-15)


def test_missing_margin_table_fails_closed(sealed_parent) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        moving_gauge_margin_row_weights_enabled=True,
    )
    adapter.moving_gauge_margin_tables = {
        (0, "prefill"): torch.full((4, 1), 1.0, dtype=torch.float64)
    }
    entries = ({"batch_index": 1, "phase": "prefill", "row_count": 4},)
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="missing or misaligned",
    ):
        adapter._moving_gauge_margin_row_weights(
            entries, torch.full((4,), 0.25, dtype=torch.float64)
        )


# --- item 8: solver and gate telemetry ---------------------------------------


def test_solver_conditioning_diagnostics_are_absent_by_default() -> None:
    compiled = _compile_toy_mlp()
    body = compiled.as_dict()
    assert all(
        "solver_conditioning" not in report
        for report in body["prior_correction_reports"]
    )
    assert "gate_operating_point" not in body


def test_solver_conditioning_diagnostics_do_not_change_the_state() -> None:
    baseline = _compile_toy_mlp()
    diagnostic = _compile_toy_mlp(solver_conditioning_diagnostics=True)
    assert diagnostic.compiled_state_sha256 == baseline.compiled_state_sha256
    assert torch.equal(diagnostic.gate_weight, baseline.gate_weight)
    reports = {
        report["role"]: report
        for report in diagnostic.as_dict()["prior_correction_reports"]
    }
    conditioning = reports["mlp_gate"]["solver_conditioning"]
    assert conditioning["solver_path"] == "primal_solve"
    assert conditioning["silent_gelsd_fallback_taken"] is False
    assert conditioning["system_singular_value_maximum"] > 0.0
    assert conditioning["system_singular_value_minimum"] > 0.0
    assert conditioning["system_condition_number"] == pytest.approx(
        conditioning["system_singular_value_maximum"]
        / conditioning["system_singular_value_minimum"]
    )


def test_nonfinite_solver_diagnostics_are_durable_and_explicit() -> None:
    encoded, marker_json = _durable_diagnostic_json(
        {
            "solver_conditioning": {
                "system_condition_number": float("inf"),
                "negative_probe": float("-inf"),
                "undefined_probe": float("nan"),
            }
        },
        "singular solver diagnostics",
    )
    conditioning = json.loads(encoded)["solver_conditioning"]
    assert conditioning == {
        "negative_probe": None,
        "system_condition_number": None,
        "undefined_probe": None,
    }
    assert marker_json is not None
    markers = json.loads(marker_json)
    assert markers["schema"] == "neural-forge-nonfinite-diagnostic-paths-v1"
    assert markers["entries"] == [
        {
            "kind": "negative_infinity",
            "path": ["solver_conditioning", "negative_probe"],
        },
        {
            "kind": "positive_infinity",
            "path": [
                "solver_conditioning",
                "system_condition_number",
            ],
        },
        {
            "kind": "nan",
            "path": ["solver_conditioning", "undefined_probe"],
        },
    ]


# --- W13R R1/R2: relative ridge and the body prior-delta clamp ---------------
#
# The Stage-A attribution
# (`reports/stage1-attribution-20260808/STAGE_A_ATTRIBUTION.md`) measured the
# mechanism these two remedies aim at: an absolute ridge of 1e-6 against a
# sigma_max of 1.4e5-7.1e5 leaves the body normal system effectively
# unregularized, the resulting near-singular solve asks for 1.4x-140x the
# prior's norm, and a hard 1.0 clamp truncates it -- at all 42 body MLP fits of
# the r13 refold.  The toy below reproduces that shape at 240 rows.


_TOY_SOURCE_HIDDEN = 8
_TOY_SOURCE_INTERMEDIATE = 10
_TOY_TARGET_HIDDEN = 4
_TOY_TARGET_INTERMEDIATE = 5
_TOY_ROWS = 240
# How hard the design's last direction is squeezed.  1e-5 puts cond(X^T X) in
# the 1e10 neighbourhood the run's layers 4/5/11 actually sat at.
_TOY_SQUEEZE = 1e-5
# The parent targets are not exactly representable in the child's chart; that
# unrepresentable part is what an unregularized inverse amplifies.
_TOY_NOISE = 5e-2
# A structural-transport prior small enough that even a well-regularized
# correction exceeds it, which is exactly when the clamp -- not the evidence --
# is the operative limiter.
_TOY_PRIOR_SCALE = 0.05


def _toy_design(rows: int, generator) -> torch.Tensor:
    raw = torch.randn(
        rows, _TOY_TARGET_HIDDEN, dtype=torch.float64, generator=generator
    )
    scale = torch.ones(_TOY_TARGET_HIDDEN, dtype=torch.float64)
    scale[-1] = _TOY_SQUEEZE
    return raw * scale


def _toy_endpoint(design, gate, up, down):
    post = torch.nn.functional.silu(design @ gate.T) * (design @ up.T)
    return post @ down.T


def _ill_conditioned_bundle():
    """One near-singular boundary with an unrepresentable target component."""

    generator = torch.Generator().manual_seed(20260808)

    def normal(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=generator)

    gate_map = normal(_TOY_SOURCE_INTERMEDIATE, _TOY_TARGET_HIDDEN)
    up_map = normal(_TOY_SOURCE_INTERMEDIATE, _TOY_TARGET_HIDDEN)
    teacher_gate = normal(_TOY_TARGET_INTERMEDIATE, _TOY_TARGET_HIDDEN)
    teacher_up = normal(_TOY_TARGET_INTERMEDIATE, _TOY_TARGET_HIDDEN)
    teacher_down = normal(_TOY_TARGET_HIDDEN, _TOY_TARGET_INTERMEDIATE)
    fit_design = _toy_design(_TOY_ROWS, generator)
    held_design = _toy_design(_TOY_ROWS, generator)
    parent_gate = fit_design @ gate_map.T + _TOY_NOISE * normal(
        _TOY_ROWS, _TOY_SOURCE_INTERMEDIATE
    )
    parent_up = fit_design @ up_map.T + _TOY_NOISE * normal(
        _TOY_ROWS, _TOY_SOURCE_INTERMEDIATE
    )
    case = Lfm2CachedMlpFoldCase(
        target_input=fit_design,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=(
            torch.nn.functional.silu(parent_gate) * parent_up
        ),
        parent_post_mlp_residual_target_space=_toy_endpoint(
            fit_design, teacher_gate, teacher_up, teacher_down
        ),
        target_pre_mlp_residual=torch.zeros(
            _TOY_ROWS, _TOY_TARGET_HIDDEN, dtype=torch.float64
        ),
    )
    return {
        "case": case,
        "held_design": held_design,
        "held_endpoint": _toy_endpoint(
            held_design, teacher_gate, teacher_up, teacher_down
        ),
        "source_gate": normal(
            _TOY_SOURCE_INTERMEDIATE, _TOY_SOURCE_HIDDEN
        ),
        "source_up": normal(_TOY_SOURCE_INTERMEDIATE, _TOY_SOURCE_HIDDEN),
        "source_down": normal(
            _TOY_SOURCE_HIDDEN, _TOY_SOURCE_INTERMEDIATE
        ),
        "prior_gate": _TOY_PRIOR_SCALE
        * normal(_TOY_TARGET_INTERMEDIATE, _TOY_TARGET_HIDDEN),
        "prior_up": _TOY_PRIOR_SCALE
        * normal(_TOY_TARGET_INTERMEDIATE, _TOY_TARGET_HIDDEN),
        "prior_down": _TOY_PRIOR_SCALE
        * normal(_TOY_TARGET_HIDDEN, _TOY_TARGET_INTERMEDIATE),
    }


def _compile_ill_conditioned(bundle, **overrides):
    return compile_lfm2_structural_residual_mlp_fold(
        source_gate_weight=bundle["source_gate"],
        source_up_weight=bundle["source_up"],
        source_down_weight=bundle["source_down"],
        target_gate_weight_template=bundle["prior_gate"],
        target_up_weight_template=bundle["prior_up"],
        target_down_weight_template=bundle["prior_down"],
        cases=(bundle["case"],),
        gate_weight_prior=bundle["prior_gate"],
        up_weight_prior=bundle["prior_up"],
        down_weight_prior=bundle["prior_down"],
        residual_assignment_sha256="ab" * 32,
        ridge=1e-6,
        solver_conditioning_diagnostics=True,
        **overrides,
    )


def _held_out_endpoint_error(compiled, bundle) -> float:
    """The boundary this branch actually has to hit, on rows it never saw."""

    design = bundle["held_design"]
    post = torch.nn.functional.silu(
        design @ compiled.gate_weight.to(torch.float64).T
    ) * (design @ compiled.up_weight.to(torch.float64).T)
    prediction = post @ compiled.down_weight.to(torch.float64).T
    endpoint = bundle["held_endpoint"]
    return float(
        torch.linalg.vector_norm(prediction - endpoint)
        / torch.linalg.vector_norm(endpoint)
    )


def _roles(compiled):
    return {
        value["role"]: value
        for value in compiled.as_dict()["prior_correction_reports"]
    }


def _condition_cap(relative_ridge_lambda: float) -> float:
    """The condition number a relative ridge can leave, at its worst.

    With ``ridge = lambda * sigma_max`` the solved spectrum runs from
    ``sigma_max * (1 + lambda)`` down to ``sigma_min + lambda * sigma_max``, so
    an exactly singular direction still lands at ``(1 + lambda) / lambda``.
    The reported cap ``1 / lambda`` is that bound to first order.
    """

    return (1.0 + relative_ridge_lambda) / relative_ridge_lambda


@pytest.fixture(scope="module")
def ill_conditioned_boundary():
    return _ill_conditioned_bundle()


def test_relative_ridge_default_off_solves_the_identical_system() -> None:
    """The byte-identity pin: unset must be the absolute lane, exactly."""

    baseline = _compile_toy_mlp()
    explicit_off = _compile_toy_mlp(relative_ridge_lambda=None)
    assert explicit_off.compiled_state_sha256 == baseline.compiled_state_sha256
    for name in ("gate_weight", "up_weight", "down_weight"):
        assert torch.equal(
            getattr(explicit_off, name), getattr(baseline, name)
        )
    for report in baseline.as_dict()["prior_correction_reports"]:
        assert "relative_ridge" not in report
    treated = _compile_toy_mlp(relative_ridge_lambda=1e-4)
    assert treated.compiled_state_sha256 != baseline.compiled_state_sha256


def test_the_relative_ridge_seals_lambda_sigma_max_and_the_mode() -> None:
    compiled = _compile_toy_mlp(
        relative_ridge_lambda=1e-4,
        solver_conditioning_diagnostics=True,
    )
    for role, report in _roles(compiled).items():
        relative = report["relative_ridge"]
        assert relative["mode"] == "relative_to_unregularized_sigma_max"
        assert relative["relative_ridge_lambda"] == 1e-4
        assert relative["absolute_ridge_configured"] == 1e-6
        sigma_max = relative["unregularized_system_singular_value_maximum"]
        assert sigma_max > 0.0, role
        assert relative["effective_ridge"] == pytest.approx(
            1e-4 * sigma_max, rel=1e-12
        )
        assert relative["effective_condition_number_cap"] == pytest.approx(1e4)
        # The cap is real: the solved system's condition number honours it.
        assert report["solver_conditioning"][
            "system_condition_number"
        ] <= _condition_cap(1e-4) * (1.0 + 1e-9)


@pytest.mark.parametrize("squeeze", [1.0, 1e-3, 1e-5, 1e-8])
@pytest.mark.parametrize("relative_ridge_lambda", [1e-6, 1e-4])
def test_the_reused_spectrum_measures_what_a_second_svd_would(
    squeeze: float,
    relative_ridge_lambda: float,
) -> None:
    """The relative arm reads conditioning off the spectrum it already took.

    That is only honest if the shifted spectrum is the solved system's
    spectrum.  It is, to ~1e-15 relative, because a symmetric PSD Gram's
    singular values are its eigenvalues and ``ridge*I`` shifts every one of
    them by exactly ``ridge``.
    """

    generator = torch.Generator().manual_seed(7)
    design = torch.randn(400, 6, dtype=torch.float64, generator=generator)
    scale = torch.ones(6, dtype=torch.float64)
    scale[-1] = squeeze
    gram = (design * scale).T @ (design * scale)
    singular = torch.linalg.svdvals(gram)
    ridge = relative_ridge_lambda * float(singular.max().item())
    direct = _solved_system_conditioning(
        gram + ridge * torch.eye(6, dtype=torch.float64)
    )
    derived = _shifted_system_conditioning(singular, ridge)
    for measured, reused in zip(direct, derived, strict=True):
        assert reused == pytest.approx(measured, rel=1e-12)
    assert derived[2] <= _condition_cap(relative_ridge_lambda) * (1.0 + 1e-9)


def test_every_solve_seals_its_requested_and_allowed_delta_ratio() -> None:
    """The diagnostic the attribution had to reconstruct from 1/update_scale."""

    compiled = _compile_toy_mlp()
    for role, report in _roles(compiled).items():
        clamp = report["prior_delta_clamp"]
        assert clamp["allowed_prior_delta_ratio"] == 1.0
        assert clamp["budget_units"] == "prior_frobenius_norm"
        assert clamp["requested_prior_delta_ratio"] > 0.0, role
        if clamp["clamp_engaged"]:
            # The clamp identity the run's 42/42 check verified: a bound solve
            # leaves exactly `max_prior_delta_ratio * ||prior||` of correction.
            assert clamp["requested_prior_delta_ratio"] > 1.0
            assert report["update_scale"] == pytest.approx(
                1.0 / clamp["requested_prior_delta_ratio"]
            )
        else:
            assert clamp["requested_prior_delta_ratio"] <= 1.0
            assert report["update_scale"] == 1.0


def test_mlp_down_now_carries_conditioning_like_the_other_two_roles() -> None:
    """R3(c): item 8 instrumented two of the three MLP roles."""

    compiled = _compile_toy_mlp(solver_conditioning_diagnostics=True)
    roles = _roles(compiled)
    assert set(roles) == {"mlp_gate", "mlp_up", "mlp_down"}
    for role, report in roles.items():
        conditioning = report["solver_conditioning"]
        assert conditioning["system_condition_number"] >= 1.0, role
        assert conditioning["system_singular_value_maximum"] > 0.0
        assert conditioning["silent_gelsd_fallback_taken"] is False
    assert (
        compiled.compiled_state_sha256
        == _compile_toy_mlp().compiled_state_sha256
    )


def test_the_clamp_not_the_evidence_limits_the_unregularized_boundary(
    ill_conditioned_boundary,
) -> None:
    """Reproduce the attributed shape: request >> allowed, boundary at ~1.0."""

    baseline = _compile_ill_conditioned(ill_conditioned_boundary)
    for role, report in _roles(baseline).items():
        clamp = report["prior_delta_clamp"]
        assert clamp["clamp_engaged"] is True, role
        assert clamp["allowed_prior_delta_ratio"] == 1.0
        assert clamp["requested_prior_delta_ratio"] > 10.0, role
        assert report["update_scale"] < 0.1
    # A boundary at the emit-zero floor, which is where arm A's 14 boundaries
    # sat (post-layer relL2 0.78-1.39).
    assert baseline.residual_endpoint_relative_rmse > 0.99
    assert _held_out_endpoint_error(baseline, ill_conditioned_boundary) > 0.99


def test_the_relative_ridge_shrinks_the_request_it_cannot_release(
    ill_conditioned_boundary,
) -> None:
    """R1 alone bounds the solve by the evidence; the clamp still binds.

    The comparison is made on ``mlp_gate``/``mlp_up`` only, and deliberately:
    those two are solved against the same design in both arms, so their
    conditioning and their request are comparable.  ``mlp_down`` is solved
    against the post-SwiGLU block *produced by* the gate/up weights, so its
    system is a different matrix in every arm -- which is itself worth knowing
    and is why its request can rise while the boundary improves.
    """

    baseline = _compile_ill_conditioned(ill_conditioned_boundary)
    treated = _compile_ill_conditioned(
        ill_conditioned_boundary, relative_ridge_lambda=1e-4
    )
    baseline_roles = _roles(baseline)
    treated_roles = _roles(treated)
    for role in ("mlp_gate", "mlp_up"):
        report = treated_roles[role]
        before = baseline_roles[role]
        assert (
            report["prior_delta_clamp"]["requested_prior_delta_ratio"]
            < before["prior_delta_clamp"]["requested_prior_delta_ratio"]
        ), role
        # The cap is 1/lambda, and it is the binding constraint here: the
        # unregularized system is five orders above it.
        assert before["solver_conditioning"]["system_condition_number"] > 1e7
        assert report["solver_conditioning"][
            "system_condition_number"
        ] <= _condition_cap(1e-4) * (1.0 + 1e-9)
    # Still clamped everywhere: a regularized solve of a boundary whose prior
    # is small is not a solve the 1.0 budget can pay for, so R1 alone leaves
    # the boundary exactly where it was.
    for role, report in treated_roles.items():
        assert report["prior_delta_clamp"]["clamp_engaged"] is True, role
    assert treated.residual_endpoint_relative_rmse > 0.99


def test_the_two_remedies_together_recover_the_boundary(
    ill_conditioned_boundary,
) -> None:
    """R1+R2: the correction survives and the endpoint improves, held out."""

    baseline = _compile_ill_conditioned(ill_conditioned_boundary)
    remedied = _compile_ill_conditioned(
        ill_conditioned_boundary,
        relative_ridge_lambda=1e-4,
        max_prior_delta_ratio=64.0,
    )
    for role, report in _roles(remedied).items():
        clamp = report["prior_delta_clamp"]
        assert clamp["clamp_engaged"] is False, role
        assert clamp["allowed_prior_delta_ratio"] == 64.0
        assert report["update_scale"] == 1.0
    assert remedied.residual_endpoint_relative_rmse < 0.5
    assert baseline.residual_endpoint_relative_rmse > 0.99
    held_out = _held_out_endpoint_error(remedied, ill_conditioned_boundary)
    assert held_out < 0.6
    assert held_out < _held_out_endpoint_error(
        baseline, ill_conditioned_boundary
    )


def test_relaxing_the_clamp_alone_leaves_the_singular_system_uncapped(
    ill_conditioned_boundary,
) -> None:
    """Why the coupling exists, measured rather than asserted.

    Relaxing the bound without regularizing does three bad things at once: the
    solved condition number stays at 2.5e8, the largest request is *still*
    clamped (it is 112x the prior, above even a 64x budget), and the boundary
    it does reach generalizes worse than the regularized arm's.  That is the
    A12/W12B shape, and it is why the catalog refuses this configuration.
    """

    remedied = _compile_ill_conditioned(
        ill_conditioned_boundary,
        relative_ridge_lambda=1e-4,
        max_prior_delta_ratio=64.0,
    )
    unregularized = _compile_ill_conditioned(
        ill_conditioned_boundary,
        max_prior_delta_ratio=64.0,
    )
    remedied_roles = _roles(remedied)
    unregularized_roles = _roles(unregularized)
    for role in ("mlp_gate", "mlp_up"):
        assert unregularized_roles[role]["solver_conditioning"][
            "system_condition_number"
        ] > 1e7
        assert remedied_roles[role]["solver_conditioning"][
            "system_condition_number"
        ] <= _condition_cap(1e-4) * (1.0 + 1e-9)
    assert (
        unregularized_roles["mlp_gate"]["prior_delta_clamp"]["clamp_engaged"]
        is True
    )
    assert (
        remedied_roles["mlp_gate"]["prior_delta_clamp"]["clamp_engaged"]
        is False
    )
    assert _held_out_endpoint_error(
        unregularized, ill_conditioned_boundary
    ) > _held_out_endpoint_error(remedied, ill_conditioned_boundary)


def test_gate_operating_point_diagnostics_do_not_change_the_state() -> None:
    baseline = _compile_toy_mlp()
    diagnostic = _compile_toy_mlp(gate_operating_point_diagnostics=True)
    assert diagnostic.compiled_state_sha256 == baseline.compiled_state_sha256
    operating_point = diagnostic.as_dict()["gate_operating_point"]
    assert diagnostic.uncorrected_gate_weight is None
    assert "drift" in operating_point
    assert "parent" in operating_point and "child" in operating_point


# --- end-to-end: the wired lanes engage inside a real toy fold ---------------


def _run_toy_layers(adapter):
    return ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-r13-wired-lane-v1",
            fold_body_reclosure=False,
            fold_output_interface=False,
        ),
    ).run()


def test_noise_aware_ridge_engages_after_the_first_accepted_layer(
    sealed_parent,
) -> None:
    control_adapter, _child, mapping = _adapter(sealed_parent)
    control_adapter.config = replace(
        control_adapter.config,
        mlp_input_error_covariance_enabled=True,
        mlp_input_error_robustness_multiplier=0.0,
    )
    control = _run_toy_layers(control_adapter)

    treated_adapter, _treated_child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=mapping,
    )
    treated_adapter.config = replace(
        treated_adapter.config,
        mlp_input_error_covariance_enabled=True,
        mlp_input_error_robustness_multiplier=4.0,
    )
    treated = _run_toy_layers(treated_adapter)

    def _downstream(report):
        return [
            dict(entry["downstream"]["compiled"]["metadata"])
            for entry in report["layer_reports"]
        ]

    control_meta = _downstream(control)
    treated_meta = _downstream(treated)
    # The zero-multiplier control arm never reaches the structural fold with a
    # covariance at all, so it is the untouched lane by construction.
    assert all(
        "noise_aware_ridge_applied" not in entry for entry in control_meta
    )
    # Layer 0 has no upstream drift; layer 1 consumes layer 0's.
    assert "noise_aware_ridge_applied" not in treated_meta[0]
    assert treated_meta[1]["noise_aware_ridge_applied"] is True
    assert treated_meta[1]["noise_aware_ridge_source_layer_index"] == 0
    assert treated_meta[1]["noise_aware_ridge_multiplier"] == 4.0
    assert treated_meta[1]["noise_aware_ridge_scale"] > 0.0
    assert len(treated_meta[1]["noise_aware_ridge_covariance_sha256"]) == 64
    # Layer 0 solves identically; layer 1's weights move.
    assert (
        treated_meta[0]["compiled_state_sha256"]
        == control_meta[0]["compiled_state_sha256"]
    )
    assert (
        treated_meta[1]["compiled_state_sha256"]
        != control_meta[1]["compiled_state_sha256"]
    )


def test_solver_and_gate_telemetry_reach_a_toy_layer_receipt(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    adapter.config = replace(
        adapter.config,
        mlp_solver_conditioning_diagnostics=True,
        mlp_gate_operating_point_diagnostics=True,
    )
    report = _run_toy_layers(adapter)
    for entry in report["layer_reports"]:
        metadata = dict(entry["downstream"]["compiled"]["metadata"])
        completion_receipt = entry["accepted_layer_completion_receipt"]
        sealed_metadata = dict(
            completion_receipt["stage_preimages"]["downstream"][
                "compiled"
            ]["metadata"]
        )
        assert sealed_metadata[
            "mlp_prior_correction_reports_json"
        ] == metadata["mlp_prior_correction_reports_json"]
        assert sealed_metadata[
            "mlp_gate_operating_point_json"
        ] == metadata["mlp_gate_operating_point_json"]
        body = {
            "prior_correction_reports": json.loads(
                metadata["mlp_prior_correction_reports_json"]
            ),
            "gate_operating_point": json.loads(
                metadata["mlp_gate_operating_point_json"]
            ),
        }
        roles = {
            value["role"]: value
            for value in body["prior_correction_reports"]
        }
        for role in ("mlp_gate", "mlp_up"):
            conditioning = roles[role]["solver_conditioning"]
            assert conditioning["solver_path"].endswith(
                ("solve", "fallback", "least_squares")
            )
            assert conditioning["system_condition_number"] >= 1.0
        assert "gate_operating_point" in body
        assert "drift" in body["gate_operating_point"]
        # R3(c): the third role is instrumented now, not two of three.
        for role in ("mlp_gate", "mlp_up", "mlp_down"):
            assert "solver_conditioning" in roles[role], role
    assert not adapter._downstream_payloads


# --- W13R R3: the telemetry that already existed and was thrown away ---------


def test_the_operator_branch_seals_its_prior_correction_reports(
    sealed_parent,
) -> None:
    """R3(a)/(b): the gap that made the operator stage unattributable.

    Every operator producer and consumer compile fits around the same
    structural-transport priors under the same clamp the MLP triple does, and
    the adapter dropped all of it.  These keys are unconditional: a telemetry
    flag nobody set is exactly how the Stage-A operator gap was created.
    """

    adapter, _child, _mapping = _adapter(sealed_parent)
    allowed = adapter.config.max_prior_delta_ratio
    report = _run_toy_layers(adapter)
    assert report["layer_reports"]
    for entry in report["layer_reports"]:
        producer = dict(entry["producer"]["compiled"]["metadata"])
        consumer = dict(entry["consumer"]["compiled"]["metadata"])
        downstream = dict(entry["downstream"]["compiled"]["metadata"])
        for stage in (producer, consumer):
            reports = json.loads(
                stage["operator_prior_correction_reports_json"]
            )
            assert reports
            for value in reports:
                assert value["role"]
                assert value["prior_norm"] >= 0.0
                assert value["correction_norm"] >= 0.0
                assert 0.0 < value["update_scale"] <= 1.0
                clamp = value["prior_delta_clamp"]
                assert clamp["allowed_prior_delta_ratio"] == allowed
                assert clamp["requested_prior_delta_ratio"] >= 0.0
                assert isinstance(clamp["clamp_engaged"], bool)
                if clamp["clamp_engaged"]:
                    assert clamp["requested_prior_delta_ratio"] > allowed
        # R3(b): both branch-local endpoint errors, which the run that raised
        # the question computed and discarded.
        assert consumer["operator_residual_endpoint_relative_rmse"] >= 0.0
        assert downstream["mlp_residual_endpoint_relative_rmse"] >= 0.0
        # ... and the composed post-layer endpoint they decompose.
        assert downstream["candidate_fit_relative_rmse"] >= 0.0


def test_the_body_solve_remedy_reaches_a_toy_layer_and_records_itself(
    sealed_parent,
) -> None:
    """R1/R2 end to end through the adapter, against an unset control."""

    control_adapter, _child, mapping = _adapter(sealed_parent)
    control = _run_toy_layers(control_adapter)

    treated_adapter, _treated_child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=mapping,
    )
    treated_adapter.config = replace(
        treated_adapter.config,
        mlp_solver_conditioning_diagnostics=True,
        relative_ridge_lambda=1e-4,
        body_max_prior_delta_ratio=8.0,
    )
    assert treated_adapter.config.mlp_max_prior_delta_ratio == 8.0
    assert control_adapter.config.mlp_max_prior_delta_ratio == (
        control_adapter.config.max_prior_delta_ratio
    )
    treated = _run_toy_layers(treated_adapter)

    for layer_ordinal, (control_entry, treated_entry) in enumerate(
        zip(control["layer_reports"], treated["layer_reports"], strict=True)
    ):
        control_meta = dict(control_entry["downstream"]["compiled"]["metadata"])
        treated_meta = dict(treated_entry["downstream"]["compiled"]["metadata"])
        # A different body system was solved, and the artifact says so rather
        # than being confusable with the control's layer.
        assert (
            treated_meta["compiled_state_sha256"]
            != control_meta["compiled_state_sha256"]
        )
        assert (
            treated_entry["downstream"]["compiled"]["artifact_sha256"]
            != control_entry["downstream"]["compiled"]["artifact_sha256"]
        )
        roles = {
            value["role"]: value
            for value in json.loads(
                treated_meta["mlp_prior_correction_reports_json"]
            )
        }
        for role, value in roles.items():
            relative = value["relative_ridge"]
            assert relative["relative_ridge_lambda"] == 1e-4
            assert relative["effective_ridge"] == pytest.approx(
                1e-4
                * relative["unregularized_system_singular_value_maximum"],
                rel=1e-12,
            )
            assert value["prior_delta_clamp"][
                "allowed_prior_delta_ratio"
            ] == 8.0, role
        # The operator branch never receives the relative ridge or the relaxed
        # clamp, at any depth: relaxing a bound on a system nothing regularizes
        # is the configuration the coupling exists to prevent.
        for stage in ("producer", "consumer"):
            for value in json.loads(
                dict(treated_entry[stage]["compiled"]["metadata"])[
                    "operator_prior_correction_reports_json"
                ]
            ):
                assert "relative_ridge" not in value
                assert value["prior_delta_clamp"][
                    "allowed_prior_delta_ratio"
                ] == treated_adapter.config.max_prior_delta_ratio
        if layer_ordinal == 0:
            # Layer 0's operator is fitted before any remedied MLP is
            # installed, so it is the one stage that must be byte-identical.
            # Deeper layers legitimately differ: this is a closed loop, and
            # layer k's operator evidence is the child layer k-1 produced.
            assert (
                treated_entry["producer"]["compiled"]["metadata"]
                == control_entry["producer"]["compiled"]["metadata"]
            )


def test_the_adapter_refuses_a_relaxed_clamp_without_a_relative_ridge() -> None:
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="requires relative_ridge_lambda",
    ) as failure:
        Lfm2DynamicClosedLoopAdapterConfig(body_max_prior_delta_ratio=4.0)
    assert "unbounded-step path" in str(failure.value)
    # Tightening the bound needs no companion; only relaxing it does.
    tightened = Lfm2DynamicClosedLoopAdapterConfig(
        body_max_prior_delta_ratio=0.25
    )
    assert tightened.mlp_max_prior_delta_ratio == 0.25
    assert Lfm2DynamicClosedLoopAdapterConfig().mlp_max_prior_delta_ratio == 1.0
