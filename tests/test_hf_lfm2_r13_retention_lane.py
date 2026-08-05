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
    Lfm2DynamicClosedLoopAdapterError,
    Lfm2TerminalFreeRunTripwire,
    _durable_diagnostic_json,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedMlpFoldCase,
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
    assert not adapter._downstream_payloads
