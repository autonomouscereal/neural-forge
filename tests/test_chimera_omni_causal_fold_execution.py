"""Execution proof: a chimera-omni parent folded 4x as a planned transaction.

``reports/chimera-omni-fold-readiness-20260804/`` proved the *numbers* -- exact
4x unique-parameter compaction, zero gradients, real emissions -- but it proved
them with driver code.  ``plan.json`` in that report says ``planner_ready =
false``, so the fold had no method id, no experiment signature and no
:class:`DynamicFoldExecutor` receipt.  This file is the same crossing as a
planned, ledgered run: one sealed parent forward, one global residual chart,
three registered operators, and an executor whose audits are the acceptance
record.

The parent here is randomly initialised rather than trained.  That is the point
of the file: it proves the *mechanism* -- planning, transaction, rollback,
zero-training audits -- and every endpoint number below is measured against the
chart-projected target the child can actually represent, on tokens no solve
saw.  Capability retention is a separate question that a trained parent and
whole-response review answer, not this test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_planner import (
    CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
    NATIVE_CAUSAL_LAYER_BRANCH_METHODS,
    NATIVE_RESIDUAL_CHART_INTERFACE_METHOD,
    DynamicFoldPlanner,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits


torch = pytest.importorskip("torch")

from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    KERNEL_SUPPORTS_ATTENTION_WINDOW,
    DynamicCausalOperatorPromotionError,
    NativeFusedGatedMLPBranchOperator,
    NativeResidualChartInterfaceOperator,
    NativeWindowedNoPEAttentionBranchOperator,
)
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    DynamicFoldExecutor,
    ExactCopyOperator,
    FoldDecision,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    fingerprint_model_state,
)
from neural_forge.dynamic_native_circuit_evidence import (  # noqa: E402
    NativeChildRecapture,
    NativeCircuitEvidenceError,
    NativeResidualChart,
    capture_native_circuit_evidence,
    compile_native_residual_chart,
)


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_chimera_omni_toy_fold.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_chimera_omni_toy_fold_execution", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

try:
    _REPO = runner.resolve_chimera_omni_repo()
    _CHIMERA = runner.import_chimera_omni(_REPO)
except Exception as exc:  # pragma: no cover - environment boundary
    pytest.skip(f"chimera-omni checkout unavailable: {exc}", allow_module_level=True)

chimera_config = _CHIMERA["config"]
chimera_modeling = _CHIMERA["modeling"]
chimera_tiny_proof = _CHIMERA["tiny_proof"]

TOKENIZER = canonical_sha256({"tokenizer": "shared-chimera-native-fold"})
PARENT_WIDTH = 128
CHILD_WIDTH = 32
BATCH = 12
SEQUENCE = 24
MEDIA_TOKENS = 16
#: Wider than the sealed sequence, so the band provably never binds on this
#: evidence.  ``banded_pair`` below uses a window that does bind.
WINDOW = 32


def _config(hidden_size: int, **changes):
    settings = {
        "vocab_size": 1024,
        "hidden_size": int(hidden_size),
        "intermediate_size": 320,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_sequence_length": 64,
        "block_pattern": ("local",),
        "residual_mode": "block_attnres",
        "attnres_block_size": 2,
        "attention_head_dim": 32,
        "feed_forward_kind": "dense",
        "activation_kind": "silu",
        "native_feature_dim": 64,
        "native_position_dim": 4,
        "native_type_count": 8,
        "mtp_heads": 1,
        "aux_heads": ("flow", "grounding", "sync"),
        "local_window": WINDOW,
    }
    settings.update(changes)
    config = chimera_config.ChimeraConfig(**settings)
    config.validate()
    return config


def _model(config, seed: int):
    torch.manual_seed(int(seed))
    model = chimera_modeling.ChimeraOmniModel(config)
    model.eval()
    return model


def _forward(model: Any, inputs: Any) -> Any:
    return model(**inputs, return_aux=True)


def _batch(config, seed: int = 11) -> dict[str, Any]:
    batch, _specification = runner.build_batch(
        chimera_tiny_proof,
        config,
        batch_size=BATCH,
        seq_len=SEQUENCE,
        max_media_tokens=MEDIA_TOKENS,
        seed=seed,
    )
    return {
        name: value
        for name, value in batch.items()
        if name not in {"native_media_targets", "native_media_loss_mask"}
    }


class _AcceptStructuralCandidate:
    """Behavior remains the outer authority; structure is admitted here."""

    def evaluate(self, context: Any) -> FoldDecision:
        del context
        return FoldDecision.accept("Structurally complete native branch install.")


class _ChimeraFold:
    """One prepared chimera-omni plan plus everything it needs to execute."""

    def __init__(self, *, window: int = WINDOW) -> None:
        self.parent_config = _config(PARENT_WIDTH, local_window=window)
        self.child_config = _config(CHILD_WIDTH, local_window=window)
        self.parent = _model(self.parent_config, 20260804)
        self.child = _model(self.child_config, 4711)
        self.inputs = _batch(self.parent_config)
        self.parent_graph = inspect_model_circuits(
            self.parent, model_name="parent", tokenizer_fingerprint=TOKENIZER
        )
        self.child_graph = inspect_model_circuits(
            self.child, model_name="child", tokenizer_fingerprint=TOKENIZER
        )
        self.evidence = capture_native_circuit_evidence(
            self.parent,
            self.parent_graph,
            inputs=self.inputs,
            forward=_forward,
            metadata={"test": "chimera-omni-4x-native-fold"},
        )
        self.chart = compile_native_residual_chart(
            self.evidence, self.parent_graph, target_width=CHILD_WIDTH
        )
        self.recapture = NativeChildRecapture(self.inputs, forward=_forward)
        self.blank_state = {
            name: value.detach().clone()
            for name, value in self.child.state_dict().items()
        }
        self.blank_fingerprint = fingerprint_model_state(self.child)
        self.plan = DynamicFoldPlanner(
            enabled_experimental_method_ids=NATIVE_CAUSAL_LAYER_BRANCH_METHODS
        ).plan(
            (self.parent_graph,),
            self.child_graph,
            child_state_fingerprint=self.blank_fingerprint,
            evidence_fingerprint=self.evidence.cache_key,
            parent_state_fingerprints={
                "parent": fingerprint_model_state(self.parent)
            },
        )

    def restore_blank(self) -> None:
        with torch.no_grad():
            for name, value in self.child.state_dict().items():
                value.copy_(self.blank_state[name])

    @property
    def context(self) -> FoldExecutionContext:
        return FoldExecutionContext(
            parent_models={"parent": self.parent},
            child_model=self.child,
            parent_graphs={"parent": self.parent_graph},
            child_graph=self.child_graph,
        )

    def registry(self, chart: NativeResidualChart | None = None):
        chart = self.chart if chart is None else chart
        return FoldOperatorRegistry(
            (
                ExactCopyOperator(),
                NativeWindowedNoPEAttentionBranchOperator(
                    chart=chart, recapture=self.recapture
                ),
                NativeFusedGatedMLPBranchOperator(
                    chart=chart, recapture=self.recapture
                ),
                NativeResidualChartInterfaceOperator(
                    chart=chart, recapture=self.recapture
                ),
            )
        )

    def execute(self, evaluator) -> Any:
        return DynamicFoldExecutor(
            operators=self.registry(),
            evidence_provider=StaticEvidenceProvider(
                self.evidence.cache_key, self.evidence
            ),
            evaluator=evaluator,
        ).execute(self.plan, self.context)

    def logits(self, model: Any) -> Any:
        with torch.no_grad():
            return model(**self.inputs, return_aux=True)["logits"]


@pytest.fixture(scope="module")
def fold() -> _ChimeraFold:
    return _ChimeraFold()


@pytest.fixture(scope="module")
def completed_run(fold: _ChimeraFold):
    parent_forwards = 0

    def counted(_module, _args, _output):
        nonlocal parent_forwards
        parent_forwards += 1

    fold.restore_blank()
    handle = fold.parent.register_forward_hook(counted)
    try:
        run = fold.execute(_AcceptStructuralCandidate())
    finally:
        handle.remove()
    return run, parent_forwards


def test_the_plan_is_ready_and_the_child_is_exactly_four_times_smaller(
    fold: _ChimeraFold,
) -> None:
    assert fold.plan.ready
    assert not fold.plan.gaps
    assert fold.parent_graph.unique_parameter_count == (
        4 * fold.child_graph.unique_parameter_count
    )
    methods = [action.method_id for action in fold.plan.actions]
    assert methods.count(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD) == 4
    assert methods.count(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD) == 4
    assert methods.count(NATIVE_RESIDUAL_CHART_INTERFACE_METHOD) == 14


def test_the_global_chart_is_a_contraction_of_the_observed_residual(
    fold: _ChimeraFold,
) -> None:
    report = dict(fold.chart.report)
    assert report["source_width"] == PARENT_WIDTH
    assert report["target_width"] == CHILD_WIDTH
    assert report["orthonormality_defect"] < 1e-9
    assert 0.0 < report["retained_energy_fraction"] <= 1.0
    assert report["per_boundary_chart_set"] is False
    assert "ResidualBoundaryChartSet" in report["per_boundary_chart_set_gap"]
    identity = fold.chart.encoder.transpose(0, 1) @ fold.chart.encoder
    assert torch.allclose(
        identity, torch.eye(CHILD_WIDTH, dtype=torch.float64), atol=1e-9
    )


def test_the_fold_executes_completely_without_a_parent_forward(
    fold: _ChimeraFold, completed_run
) -> None:
    run, parent_forwards = completed_run

    assert run.status == "complete"
    assert all(attempt.accepted for attempt in run.attempts)
    assert parent_forwards == 0
    assert run.observed_parent_root_forward_calls == 0
    assert run.observed_parent_module_forward_calls == 0
    for name in (
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
        "pruning_operations",
        "quantization_operations",
    ):
        assert getattr(run, name) == 0
    assert run.executed_fold_static_zero_training_audit_passed is True
    assert set(run.completed_target_parameter_paths) == set(
        fold.child_graph.parameter_paths
    )

    changed = [
        path
        for attempt in run.attempts
        for path in attempt.transaction_report["changed_state_paths"]
    ]
    assert len(changed) == len(set(changed))
    assert set(changed) <= set(fold.child_graph.parameter_paths)
    assert fingerprint_model_state(fold.child) != fold.blank_fingerprint

    logits = fold.logits(fold.child)
    assert tuple(logits.shape) == (BATCH, SEQUENCE, fold.child_config.vocab_size)
    assert bool(torch.isfinite(logits).all())


def test_every_branch_beats_the_no_branch_baseline_on_held_out_tokens(
    completed_run,
) -> None:
    run, _forwards = completed_run
    attention = [
        attempt.transaction_report
        for attempt in run.attempts
        if attempt.method_id == CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    ]
    mlp = [
        attempt.transaction_report
        for attempt in run.attempts
        if attempt.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    ]
    assert len(attention) == 4
    assert len(mlp) == 4
    for report in attention:
        endpoint = report["validation_attention_delta_relative_rmse"]
        assert 0.0 < endpoint < report["no_branch_endpoint_relative_rmse_baseline"]
        # The ridge is chosen on the selection partition and reported with the
        # whole ladder, so the number is auditable rather than tuned.
        assert report["selected_ridge"] in [
            entry["ridge"] for entry in report["ridge_ladder"]
        ]
        assert report["selection_endpoint_relative_rmse"] <= min(
            entry["selection_relative_rmse"] for entry in report["ridge_ladder"]
        )
    for report in mlp:
        endpoint = report["validation_mlp_delta_relative_rmse"]
        assert 0.0 < endpoint < report["no_branch_endpoint_relative_rmse_baseline"]
        assert report["fused_gate_up_split"] == "chunk_two_on_last_dimension"
        assert report["fused_gate_up_refused_in_transaction"] is True
        assert report["source_intermediate_size"] == report["target_intermediate_size"]


def test_the_nope_transport_and_the_window_binding_are_sealed(
    completed_run,
) -> None:
    run, _forwards = completed_run
    for attempt in run.attempts:
        if attempt.method_id != CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD:
            continue
        report = attempt.transaction_report
        assert report["position_policy"] == "nope"
        assert report["attention_window"] == WINDOW
        assert report["sealed_sequence_length"] == SEQUENCE
        if KERNEL_SUPPORTS_ATTENTION_WINDOW:
            assert report["attention_window_binding"] == "kernel_band_mask"
        else:
            # The band is vacuous on this evidence, which is the only case in
            # which the unbanded kernel is the parent's own operator.
            assert report["attention_window_binding"] == "inactive_on_sealed_evidence"
            assert report["position_transport"] == "nope_zero_position_identity_rope"
            assert report["nope_identity_transport_verified"] is True
        assert report["per_boundary_chart_set"] is False
        assert report["parent_forward_calls"] == 0


def test_the_installed_child_reproduces_every_reported_branch_endpoint(
    fold: _ChimeraFold, completed_run
) -> None:
    """The number in the receipt is the number the installed child computes.

    This is the check that makes the windowed-NoPE acceptance honest rather
    than plausible.  The kernel fitted an operator class -- band mask, no
    rotation, grouped-query softmax -- and the report scored *its own replay*.
    If the class the kernel replayed and the class the child runs differed by
    so much as an off-by-one in the band, the child's real branch delta would
    not reproduce that score.  So it is recomputed here from a fresh forward of
    the installed child, on the same held-out tokens, and compared.
    """

    run, _forwards = completed_run
    assert run.status == "complete"
    live = fold.recapture(fold.child, fold.child_graph)
    _valid, _fit, _selection, validation = fold.evidence.token_masks()
    flat_validation = validation.reshape(-1)

    def endpoint(module_path: str) -> float:
        installed = live.flat_rows(live.output(module_path))[flat_validation]
        expected = fold.chart.encode(
            fold.evidence.flat_rows(fold.evidence.output(module_path))
        )[flat_validation]
        difference = (installed - expected).square().mean().sqrt()
        reference = expected.square().mean().sqrt().clamp_min(1e-30)
        return float((difference / reference).item())

    reported_attention = [
        attempt.transaction_report["validation_attention_delta_relative_rmse"]
        for attempt in run.attempts
        if attempt.method_id == CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    ]
    reported_mlp = [
        attempt.transaction_report["validation_mlp_delta_relative_rmse"]
        for attempt in run.attempts
        if attempt.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    ]
    for index in range(4):
        assert endpoint(f"blocks.{index}.mixer") == pytest.approx(
            reported_attention[index], rel=1e-6, abs=1e-9
        )
        assert endpoint(f"blocks.{index}.ffn") == pytest.approx(
            reported_mlp[index], rel=1e-6, abs=1e-9
        )


def test_the_depth_selectors_are_folded_and_the_dead_one_is_recorded(
    completed_run,
) -> None:
    run, _forwards = completed_run
    selector_reports = [
        attempt.transaction_report
        for attempt in run.attempts
        if attempt.transaction_report.get("depth_selector_present") is True
    ]
    assert len(selector_reports) == 8
    invoked = [
        report for report in selector_reports if report["depth_selector_invoked"]
    ]
    dead = [
        report for report in selector_reports if not report["depth_selector_invoked"]
    ]
    # ``block_attnres`` never invokes the first block's attention selector: no
    # block has completed at that depth.
    assert len(dead) == 1
    assert dead[0]["depth_selector_method"] == (
        "unexercised_deterministic_chart_transport"
    )
    assert dead[0]["depth_source_count"] == 0
    for report in invoked:
        assert report["validation_depth_score_relative_rmse"] < 1.0


def test_the_exact_interface_transports_are_measured_not_asserted(
    fold: _ChimeraFold, completed_run
) -> None:
    run, _forwards = completed_run
    transports = {
        entry["node_id"]: entry
        for attempt in run.attempts
        if attempt.method_id == NATIVE_RESIDUAL_CHART_INTERFACE_METHOD
        for entry in attempt.transaction_report["interface_transports"]
    }

    assert transports["input.token_embedding"]["transport"] == "residual_row_table"
    assert transports["input.token_embedding"]["exact"] is True
    projection = transports["input.native_media.boundary_projection"]
    assert projection["transport"] == "residual_producer_projection"
    assert projection["transport_replay_relative_rmse"] < 1e-6
    assert transports["output.lm_head"]["transport"] == "residual_consumer_head"
    assert transports["output.lm_head"]["validation_relative_rmse"] < 1.0
    assert transports["output.native_media.emission_head"][
        "validation_relative_rmse"
    ] < 1.0

    # The row-table claim is checkable directly: the installed child table is
    # the parent's rows in chart coordinates, exactly.
    installed = fold.child.get_parameter("embed.weight").detach().to(torch.float64)
    expected = (
        fold.parent.get_parameter("embed.weight").detach().to(torch.float64)
        @ fold.chart.encoder
    )
    assert torch.allclose(installed, expected, atol=1e-6, rtol=0.0)


def test_the_folded_child_tracks_the_parent_where_a_blank_child_cannot(
    fold: _ChimeraFold, completed_run
) -> None:
    run, _forwards = completed_run
    assert run.status == "complete"
    parent_top1 = fold.logits(fold.parent).argmax(dim=-1)
    folded_top1 = fold.logits(fold.child).argmax(dim=-1)
    blank = _model(fold.child_config, 4711)
    blank.load_state_dict(fold.blank_state)
    blank.eval()
    blank_top1 = fold.logits(blank).argmax(dim=-1)

    folded_agreement = float((parent_top1 == folded_top1).double().mean())
    blank_agreement = float((parent_top1 == blank_top1).double().mean())
    # A machinery proof, not a capability claim: the parent is untrained, so
    # this says the installed circuits carry the parent's function, nothing
    # about intelligence.
    assert folded_agreement > 0.5
    assert folded_agreement > blank_agreement + 0.25


def test_a_rejected_branch_rolls_the_whole_run_back_to_the_blank_child(
    fold: _ChimeraFold,
) -> None:
    class _RejectTheFirstAttentionBranch:
        def evaluate(self, context: Any) -> FoldDecision:
            if context.action.method_id == (
                CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
            ):
                return FoldDecision.reject("Rejected for the rollback contract.")
            return FoldDecision.accept("Accepted.")

    fold.restore_blank()
    run = fold.execute(_RejectTheFirstAttentionBranch())

    assert run.status == "rejected"
    rejected = [attempt for attempt in run.attempts if not attempt.accepted]
    assert len(rejected) == 1
    assert rejected[0].method_id == CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    assert rejected[0].rollback_exact is True
    assert (
        rejected[0].transaction_report["final_state_fingerprint"]
        == rejected[0].incumbent_state_fingerprint
    )
    # Every interface transaction accepted before the rejection is undone too.
    assert fingerprint_model_state(fold.child) == fold.blank_fingerprint
    for name, value in fold.child.state_dict().items():
        assert torch.equal(value, fold.blank_state[name])
    fold.restore_blank()


def test_a_chart_of_the_wrong_width_fails_closed_before_any_mutation(
    fold: _ChimeraFold,
) -> None:
    fold.restore_blank()
    wrong = compile_native_residual_chart(
        fold.evidence, fold.parent_graph, target_width=CHILD_WIDTH // 2
    )
    action = next(
        value
        for value in fold.plan.actions
        if value.method_id == CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    )
    operator = NativeWindowedNoPEAttentionBranchOperator(
        chart=wrong, recapture=fold.recapture
    )
    before = fingerprint_model_state(fold.child)

    with pytest.raises(DynamicCausalOperatorPromotionError, match="chart maps"):
        operator.prepare(
            action,
            fold.context,
            FoldEvidence(
                fold.evidence.cache_key,
                fold.evidence,
                channels=action.evidence_requirements,
            ),
        )
    assert fingerprint_model_state(fold.child) == before


def test_foreign_evidence_fails_closed_before_any_mutation(
    fold: _ChimeraFold,
) -> None:
    fold.restore_blank()
    foreign = capture_native_circuit_evidence(
        fold.child,
        fold.child_graph,
        inputs=fold.inputs,
        forward=_forward,
        metadata={"test": "wrong-parent"},
    )
    action = next(
        value
        for value in fold.plan.actions
        if value.method_id == NATIVE_RESIDUAL_CHART_INTERFACE_METHOD
    )
    operator = NativeResidualChartInterfaceOperator(
        chart=fold.chart, recapture=fold.recapture
    )
    before = fingerprint_model_state(fold.child)

    with pytest.raises(
        DynamicCausalOperatorPromotionError, match="different parent graph"
    ):
        operator.prepare(
            action,
            fold.context,
            FoldEvidence(
                foreign.cache_key,
                foreign,
                channels=action.evidence_requirements,
            ),
        )
    assert fingerprint_model_state(fold.child) == before


def test_a_tampered_evidence_seal_is_refused() -> None:
    """The bundle verifies itself; a mutated capture is not the sealed one."""

    fold_evidence = _ChimeraFold.__new__(_ChimeraFold)
    del fold_evidence
    parent = _model(_config(PARENT_WIDTH), 20260804)
    graph = inspect_model_circuits(
        parent, model_name="parent", tokenizer_fingerprint=TOKENIZER
    )
    evidence = capture_native_circuit_evidence(
        parent,
        graph,
        inputs=_batch(_config(PARENT_WIDTH)),
        forward=_forward,
        metadata={"test": "seal"},
    )
    evidence.verify()
    capture = evidence.capture("blocks.0.mixer")
    with torch.no_grad():
        capture.output.add_(1.0)
    with pytest.raises(NativeCircuitEvidenceError, match="no longer matches its seal"):
        evidence.verify()


@pytest.fixture(scope="module")
def banded_fold() -> _ChimeraFold:
    """One pair whose window really does bind on the sealed evidence."""

    return _ChimeraFold(window=8)


@pytest.mark.skipif(
    KERNEL_SUPPORTS_ATTENTION_WINDOW,
    reason=(
        "HFCausalGQATopology exposes attention_window, so the band is threaded "
        "into the kernel mask and this refusal path is unreachable"
    ),
)
def test_a_binding_window_fails_closed_without_the_kernel_flag(
    banded_fold: _ChimeraFold,
) -> None:
    """Dormant guard: without the band flag a binding window must be refused.

    This path is unreachable while the kernel carries ``attention_window``.  It
    stays because the alternative -- folding a banded operator through an
    unbanded mask -- is exactly the kind of silent approximation this lane
    exists to prevent, and the guard should fire if the flag ever regresses.
    """

    action = next(
        value
        for value in banded_fold.plan.actions
        if value.method_id == CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    )
    operator = NativeWindowedNoPEAttentionBranchOperator(
        chart=banded_fold.chart, recapture=banded_fold.recapture
    )
    before = fingerprint_model_state(banded_fold.child)

    with pytest.raises(
        DynamicCausalOperatorPromotionError, match="attention_window",
    ):
        operator.prepare(
            action,
            banded_fold.context,
            FoldEvidence(
                banded_fold.evidence.cache_key,
                banded_fold.evidence,
                channels=action.evidence_requirements,
            ),
        )
    assert fingerprint_model_state(banded_fold.child) == before


def test_a_binding_window_folds_through_the_kernel_band_mask(
    banded_fold: _ChimeraFold,
) -> None:
    banded_fold.restore_blank()
    action = next(
        value
        for value in banded_fold.plan.actions
        if value.method_id == CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    )
    operator = NativeWindowedNoPEAttentionBranchOperator(
        chart=banded_fold.chart, recapture=banded_fold.recapture
    )
    transaction = operator.prepare(
        action,
        banded_fold.context,
        FoldEvidence(
            banded_fold.evidence.cache_key,
            banded_fold.evidence,
            channels=action.evidence_requirements,
        ),
    )
    try:
        report = transaction.report
        assert report["attention_window"] == 8
        assert report["attention_window_binding"] == "kernel_band_mask"
        assert report["sealed_sequence_length"] == SEQUENCE
        # The band really binds here: 8 < 24, so this is not the vacuous case.
        assert report["attention_window"] < report["sealed_sequence_length"]
        assert (
            report["validation_attention_delta_relative_rmse"]
            < report["no_branch_endpoint_relative_rmse_baseline"]
        )
    finally:
        transaction.rollback()
    assert fingerprint_model_state(banded_fold.child) == (
        banded_fold.blank_fingerprint
    )
