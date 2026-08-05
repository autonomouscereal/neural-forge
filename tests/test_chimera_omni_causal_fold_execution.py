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
    CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
    CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
    CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
    CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
    NATIVE_CAUSAL_LAYER_BRANCH_METHODS,
    NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
    NATIVE_RESIDUAL_CHART_INTERFACE_METHOD,
    DynamicFoldPlanner,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits


torch = pytest.importorskip("torch")

from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    KERNEL_SUPPORTS_ATTENTION_WINDOW,
    BoundedGatedDeltaRecurrenceLayerOperator,
    CompressedLatentKVAttentionLayerOperator,
    CompressedSummaryAttentionLayerOperator,
    DynamicCausalOperatorPromotionError,
    NativeFusedGatedMLPBranchOperator,
    NativeResidualChartInterfaceOperator,
    NativeWindowedNoPEAttentionBranchOperator,
    StableLatentMoEChannelMixerLayerOperator,
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
from neural_forge.hf_causal_kda_fold import ChimeraKDAFoldPolicy  # noqa: E402
from neural_forge.hf_causal_latent_kv_attention_fold import (  # noqa: E402
    ChimeraLatentKVFoldPolicy,
)
from neural_forge.hf_causal_sparse_latent_fold import (  # noqa: E402
    ChimeraSparseLatentFoldPolicy,
)
from neural_forge.hf_causal_stable_latent_moe_fold import (  # noqa: E402
    ChimeraStableLatentMoEFoldPolicy,
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
#: The step size ``_ChimeraFold(train_steps=...)`` trains a *parent* at, using
#: chimera-omni's own tiny-proof objective.  Small and slow on purpose: the
#: router's auxiliary-loss-free balancing is applied by training code between
#: optimizer steps, which the tiny-proof loop does not do, so a hot schedule
#: collapses the routing onto two experts and the MoE kernel's sparse row
#: budget then refuses -- correctly.
TRAIN_LEARNING_RATE = 5.0e-4

#: The whole-family crossing.  ``local`` alone is linear in the residual width,
#: so 128 -> 32 is exactly 4x by construction.  Add ``csa``/``hca`` and that
#: stops being true: their four summary projections are ``[hidden, hidden]``,
#: which shrinks by sixteen while everything else shrinks by four, so no
#: quarter-width child of a model containing them is ever exactly 4x.  This
#: pair is the exact integer solution instead -- 128 -> 34 with *every other*
#: geometry identical -- and it is a real solution, not a rounding: the counts
#: below are asserted, not approximated.
FAMILY_PARENT_WIDTH = 128
FAMILY_CHILD_WIDTH = 34
FAMILY_PARENT_PARAMETERS = 1_601_760
FAMILY_CHILD_PARAMETERS = 400_440
FAMILY_WINDOW = 8
FAMILY_CSA_STRIDE = 4
FAMILY_HCA_STRIDE = 8


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


def _family_config(hidden_size: int, **changes):
    """One chimera trunk running every sequence-mixer family that folds."""

    return _config(
        hidden_size,
        **{
            "vocab_size": 1534,
            "intermediate_size": 448,
            "block_pattern": ("local", "kda", "csa", "hca"),
            "kda_head_dim": 9,
            "kda_value_head_dim": 8,
            "kda_decay_rank": 4,
            "kda_short_conv_kernel_size": 4,
            "local_window": FAMILY_WINDOW,
            "csa_compress_stride": FAMILY_CSA_STRIDE,
            "hca_compress_stride": FAMILY_HCA_STRIDE,
            **changes,
        },
    )


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

    def __init__(
        self,
        *,
        window: int = WINDOW,
        factory=_config,
        parent_width: int = PARENT_WIDTH,
        child_width: int = CHILD_WIDTH,
        enabled_methods=NATIVE_CAUSAL_LAYER_BRANCH_METHODS,
        parent_seed: int = 20260804,
        train_steps: int = 0,
    ) -> None:
        self.enabled_methods = tuple(enabled_methods)
        self.parent_config = factory(parent_width, local_window=window)
        self.child_config = factory(child_width, local_window=window)
        self.child_width = int(child_width)
        self.parent = _model(self.parent_config, parent_seed)
        # Training the *parent* is allowed and is not a fold: the child is
        # still installed by closed-form solves alone, and every zero-training
        # audit below is asserted on the executed run.
        self.training = (
            runner.train_parent(
                chimera_tiny_proof,
                self.parent,
                self.parent_config,
                steps=int(train_steps),
                batch_size=8,
                seq_len=SEQUENCE,
                max_media_tokens=MEDIA_TOKENS,
                learning_rate=TRAIN_LEARNING_RATE,
                seed=int(parent_seed),
            )
            if train_steps
            else None
        )
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
            self.evidence, self.parent_graph, target_width=self.child_width
        )
        self.recapture = NativeChildRecapture(self.inputs, forward=_forward)
        self.blank_state = {
            name: value.detach().clone()
            for name, value in self.child.state_dict().items()
        }
        self.blank_fingerprint = fingerprint_model_state(self.child)
        self.plan = DynamicFoldPlanner(
            enabled_experimental_method_ids=self.enabled_methods
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
                BoundedGatedDeltaRecurrenceLayerOperator(
                    chart=chart, recapture=self.recapture
                ),
                CompressedSummaryAttentionLayerOperator(
                    chart=chart, recapture=self.recapture
                ),
                CompressedLatentKVAttentionLayerOperator(
                    chart=chart, recapture=self.recapture
                ),
                StableLatentMoEChannelMixerLayerOperator(
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


# ---------------------------------------------------------------------------
# The whole chimera sequence-mixer family: local + kda + csa + hca + fused MLP.
#
# ``NOTES-W11B.md`` left the bounded gated delta recurrence and the compressed
# summary attention as named planner gaps, and the coverage document said so.
# These are the same proof as above -- planned, ledgered, rolled back, audited
# -- for a parent that runs every one of those families in one trunk.
# ---------------------------------------------------------------------------


#: The receipt key each branch method reports its held-out endpoint under.
FAMILY_ENDPOINT_KEYS = {
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD: (
        "validation_attention_delta_relative_rmse"
    ),
    CAUSAL_KDA_RECURRENT_BRANCH_METHOD: "validation_recurrence_delta_relative_rmse",
    CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD: (
        "validation_composite_delta_relative_rmse"
    ),
    CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD: "validation_mlp_delta_relative_rmse",
}


@pytest.fixture(scope="module")
def family_fold() -> _ChimeraFold:
    return _ChimeraFold(
        window=FAMILY_WINDOW,
        factory=_family_config,
        parent_width=FAMILY_PARENT_WIDTH,
        child_width=FAMILY_CHILD_WIDTH,
        enabled_methods=NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
    )


@pytest.fixture(scope="module")
def completed_family_run(family_fold: _ChimeraFold):
    parent_forwards = 0

    def counted(_module, _args, _output):
        nonlocal parent_forwards
        parent_forwards += 1

    family_fold.restore_blank()
    handle = family_fold.parent.register_forward_hook(counted)
    try:
        run = family_fold.execute(_AcceptStructuralCandidate())
    finally:
        handle.remove()
    return run, parent_forwards


def _family_reports(run, method_id: str) -> list[dict]:
    return [
        attempt.transaction_report
        for attempt in run.attempts
        if attempt.method_id == method_id
    ]


def test_the_family_plan_is_ready_with_zero_gaps_at_exactly_four_times(
    family_fold: _ChimeraFold,
) -> None:
    assert family_fold.plan.ready
    assert not family_fold.plan.gaps
    assert family_fold.parent_graph.unique_parameter_count == FAMILY_PARENT_PARAMETERS
    assert family_fold.child_graph.unique_parameter_count == FAMILY_CHILD_PARAMETERS
    assert family_fold.parent_graph.unique_parameter_count == (
        4 * family_fold.child_graph.unique_parameter_count
    )

    methods = [action.method_id for action in family_fold.plan.actions]
    assert methods.count(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD) == 1
    assert methods.count(CAUSAL_KDA_RECURRENT_BRANCH_METHOD) == 1
    # One method id serves both csa and hca: they are the same
    # ``SparseLatentAttention`` executable at two strides.
    assert methods.count(CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD) == 2
    assert methods.count(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD) == 4
    assert methods.count(NATIVE_RESIDUAL_CHART_INTERFACE_METHOD) == 14
    assert methods.count("copy_exact") == 2

    owned = [
        path
        for action in family_fold.plan.actions
        for path in action.target_parameter_paths
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(family_fold.child_graph.parameter_paths)


def test_the_family_fold_executes_completely_without_a_parent_forward(
    family_fold: _ChimeraFold, completed_family_run
) -> None:
    run, parent_forwards = completed_family_run

    assert run.status == "complete"
    assert all(attempt.accepted for attempt in run.attempts)
    assert parent_forwards == 0
    assert run.observed_parent_root_forward_calls == 0
    # The two new operators trace the parent's own recurrence and read its own
    # weights, but they never call a parent module: every row they consume is a
    # sealed module-boundary capture from the one forward taken before the run
    # began.
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
        family_fold.child_graph.parameter_paths
    )

    changed = [
        path
        for attempt in run.attempts
        for path in attempt.transaction_report["changed_state_paths"]
    ]
    assert len(changed) == len(set(changed))
    logits = family_fold.logits(family_fold.child)
    assert tuple(logits.shape) == (
        BATCH,
        SEQUENCE,
        family_fold.child_config.vocab_size,
    )
    assert bool(torch.isfinite(logits).all())


def test_every_family_branch_beats_the_no_branch_baseline(
    completed_family_run,
) -> None:
    run, _forwards = completed_family_run
    counts = {
        CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD: 1,
        CAUSAL_KDA_RECURRENT_BRANCH_METHOD: 1,
        CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD: 2,
        CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD: 4,
    }
    for method_id, expected in counts.items():
        reports = _family_reports(run, method_id)
        assert len(reports) == expected
        for report in reports:
            endpoint = report[FAMILY_ENDPOINT_KEYS[method_id]]
            assert 0.0 < endpoint < report[
                "no_branch_endpoint_relative_rmse_baseline"
            ]
            assert report["selected_ridge"] in [
                entry["ridge"] for entry in report["ridge_ladder"]
            ]
            assert report["parent_forward_calls"] == 0
            assert report["numeric_weight_matching_objective"] is False


def test_the_installed_child_reproduces_every_family_branch_endpoint(
    family_fold: _ChimeraFold, completed_family_run
) -> None:
    """The number in the receipt is the number the installed child computes.

    This is what makes the two new operators honest rather than plausible.  A
    recurrence is not a linear map: if the class the KDA kernel replayed and the
    class the child's ``KimiDeltaAttention`` runs differed anywhere -- a conv
    tap, the decay chain, the gate -- twenty-four sequential steps would not
    land on the same held-out endpoint.  Likewise a compressed-summary branch
    whose stride or prepended zero summary were off by one would not.
    """

    run, _forwards = completed_family_run
    assert run.status == "complete"
    live = family_fold.recapture(family_fold.child, family_fold.child_graph)
    _valid, _fit, _selection, validation = family_fold.evidence.token_masks()
    flat_validation = validation.reshape(-1)

    def endpoint(module_path: str) -> float:
        installed = live.flat_rows(live.output(module_path))[flat_validation]
        expected = family_fold.chart.encode(
            family_fold.evidence.flat_rows(family_fold.evidence.output(module_path))
        )[flat_validation]
        difference = (installed - expected).square().mean().sqrt()
        reference = expected.square().mean().sqrt().clamp_min(1e-30)
        return float((difference / reference).item())

    checked = 0
    for attempt in run.attempts:
        key = FAMILY_ENDPOINT_KEYS.get(attempt.method_id)
        if key is None:
            continue
        report = attempt.transaction_report
        index = int(report["target_node_id"].split(".")[1])
        module = (
            f"blocks.{index}.ffn"
            if attempt.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
            else f"blocks.{index}.mixer"
        )
        assert endpoint(module) == pytest.approx(report[key], rel=1e-5, abs=1e-9)
        checked += 1
    assert checked == 8


def test_the_recurrence_receipt_carries_its_evidence_consistency_gates(
    completed_family_run,
) -> None:
    run, _forwards = completed_family_run
    (report,) = _family_reports(run, CAUSAL_KDA_RECURRENT_BRANCH_METHOD)

    # The rows no module boundary emits are named as derived, and the gate that
    # checks them is named too: the exact effective-weight unroll rebuilds the
    # parent's cached pre-norm recurrent output from those derived rows.
    assert report["captured_row_source"] == "sealed_native_module_boundary_capture"
    assert "retention_rows" in report["derived_rows"]
    assert "state_snapshots" in report["derived_rows"]
    assert "unroll" in report["derived_row_consistency_gate"]
    assert report["unroll_reconstruction_relative_rmse"] < 1e-6
    # The carried depthwise taps reproduce the parent's own post-conv rows.
    assert report["conv_tap_carry_relative_rmse"] < 1e-5
    # More than one chunk boundary, so the state-drift gate can measure growth
    # against the horizon at all.
    assert len(report["state_snapshot_positions"]) > 1
    assert len(report["state_snapshot_relative_rmse"]) == len(
        report["state_snapshot_positions"]
    )
    assert report["state_snapshot_source"] == (
        "parent_delta_rule_update_stepped_over_sealed_rows"
    )
    assert report["state_drift_growth_ratio"] <= 4.0
    assert report["source_geometry"]["heads"] == report["target_geometry"]["heads"]
    assert report["source_geometry"]["hidden_size"] == FAMILY_PARENT_WIDTH
    assert report["target_geometry"]["hidden_size"] == FAMILY_CHILD_WIDTH
    assert report["validation_residual_relative_rmse_bf16"] < 1.0


def test_the_compressed_summary_receipt_separates_its_two_sub_branches(
    completed_family_run,
) -> None:
    run, _forwards = completed_family_run
    reports = _family_reports(run, CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD)
    families = {report["mixer_family"]: report for report in reports}

    assert set(families) == {"csa", "hca"}
    assert families["csa"]["summary_stride"] == FAMILY_CSA_STRIDE
    assert families["hca"]["summary_stride"] == FAMILY_HCA_STRIDE
    for report in reports:
        assert report["one_executable_for_csa_and_hca"] is True
        assert report["method_id"] == CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD
        assert report["local_window"] == FAMILY_WINDOW
        # Two independently captured sub-branch values reconstruct the sealed
        # branch write; the split is measured, not assumed.
        assert report["split_reconstruction_relative_rmse"] < 1e-6
        assert 0.0 < report["local_energy_fraction"] <= 1.0
        assert 0.0 < report["summary_energy_fraction"] <= 1.0
        # Per-sub-branch endpoints are reported separately, because the bounded
        # alternation trades mass between them to serve the shared endpoint.
        assert report["local_sub_branch_validation_relative_rmse"] > 0.0
        assert report["summary_sub_branch_validation_relative_rmse"] > 0.0
        assert report["score_chart_used"] is True
        assert report["sub_branch_fit_coordinate_system"] == "parent_residual_width"
        assert report["endpoint_gate_coordinate_system"] == "child_residual_width"
        assert sorted(report["sub_branch_writers_transported_through_chart"]) == [
            "local.out.weight",
            "out.weight",
        ]
        assert report["summary_probability_overlap"] >= 0.25
        # The kernel's no-branch floor is not a hypothetical here: on this
        # crossing the unregularized rung of the ladder really does replay
        # worse than writing nothing in BF16, the kernel refuses to emit a
        # child at all, and the ladder records the refusal instead of the
        # operator quietly taking it.
        refusals = [
            entry["kernel_refusal"]
            for entry in report["ridge_ladder"]
            if "kernel_refusal" in entry
        ]
        assert refusals
        assert report["kernel_refused_ridge_rungs"] == len(refusals)
        assert all("no-branch baseline" in value for value in refusals)


def test_the_family_child_tracks_the_parent_where_a_blank_child_cannot(
    family_fold: _ChimeraFold, completed_family_run
) -> None:
    run, _forwards = completed_family_run
    assert run.status == "complete"
    parent_top1 = family_fold.logits(family_fold.parent).argmax(dim=-1)
    folded_top1 = family_fold.logits(family_fold.child).argmax(dim=-1)
    blank = _model(family_fold.child_config, 4711)
    blank.load_state_dict(family_fold.blank_state)
    blank.eval()
    blank_top1 = family_fold.logits(blank).argmax(dim=-1)

    folded_agreement = float((parent_top1 == folded_top1).double().mean())
    blank_agreement = float((parent_top1 == blank_top1).double().mean())
    # A machinery proof, not a capability claim: the parent is untrained.
    assert folded_agreement > 0.5
    assert folded_agreement > blank_agreement + 0.25


def test_a_rejected_recurrence_branch_rolls_the_family_run_back(
    family_fold: _ChimeraFold,
) -> None:
    class _RejectTheRecurrence:
        def evaluate(self, context: Any) -> FoldDecision:
            if context.action.method_id == CAUSAL_KDA_RECURRENT_BRANCH_METHOD:
                return FoldDecision.reject("Rejected for the rollback contract.")
            return FoldDecision.accept("Accepted.")

    family_fold.restore_blank()
    run = family_fold.execute(_RejectTheRecurrence())

    assert run.status == "rejected"
    rejected = [attempt for attempt in run.attempts if not attempt.accepted]
    assert len(rejected) == 1
    assert rejected[0].method_id == CAUSAL_KDA_RECURRENT_BRANCH_METHOD
    assert rejected[0].rollback_exact is True
    # Every transaction accepted before the rejection is undone too, tensor for
    # tensor -- including the whole first layer and the interface bundle.
    assert fingerprint_model_state(family_fold.child) == family_fold.blank_fingerprint
    for name, value in family_fold.child.state_dict().items():
        assert torch.equal(value, family_fold.blank_state[name])
    family_fold.restore_blank()


def _refuse(family_fold: _ChimeraFold, operator, method_id: str):
    action = next(
        value for value in family_fold.plan.actions if value.method_id == method_id
    )
    before = fingerprint_model_state(family_fold.child)
    with pytest.raises(
        DynamicCausalOperatorPromotionError, match="no installable branch solve"
    ) as error:
        operator.prepare(
            action,
            family_fold.context,
            FoldEvidence(
                family_fold.evidence.cache_key,
                family_fold.evidence,
                channels=action.evidence_requirements,
            ),
        )
    assert fingerprint_model_state(family_fold.child) == before
    return str(error.value)


@pytest.mark.parametrize(
    ("operator_class", "method_id", "policy", "measurement"),
    (
        (
            BoundedGatedDeltaRecurrenceLayerOperator,
            CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
            ChimeraKDAFoldPolicy(conv_carry_tolerance=1.0e-15),
            "carried depthwise conv taps",
        ),
        (
            CompressedSummaryAttentionLayerOperator,
            CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
            ChimeraSparseLatentFoldPolicy(
                minimum_summary_probability_overlap=0.999
            ),
            "block assignment",
        ),
    ),
)
def test_a_kernel_refusal_becomes_a_rejection_with_rollback(
    family_fold: _ChimeraFold, operator_class, method_id, policy, measurement
) -> None:
    """A kernel that refuses to emit is a planner-visible failure, not a pass.

    Both kernels carry the design's fail-closed thresholds in a sealed policy
    and raise rather than emit a child when one is breached.  Tightening one of
    those thresholds is the honest way to drive that path here: the kernel then
    refuses on *every* ridge rung, and the operator has to surface it as its own
    failure, carrying the kernel's own measurement in the message, and leave the
    child bit-identical.
    """

    family_fold.restore_blank()
    message = _refuse(
        family_fold,
        operator_class(
            chart=family_fold.chart,
            recapture=family_fold.recapture,
            fold_policy=policy,
        ),
        method_id,
    )
    assert measurement in message
    assert "ridge" in message


def test_the_summary_kernel_refuses_a_branch_that_loses_to_writing_nothing(
    family_fold: _ChimeraFold,
) -> None:
    """The design's 1.0 no-branch floor, exercised through the kernel itself.

    A ridge ladder of one absurd rung crushes the compressed-summary solve to
    almost nothing; the kernel's BF16 replay then lands at or above the
    no-branch baseline and it refuses to emit a child at all.  The operator
    surfaces that refusal by name.

    The recurrence kernel enforces the same floor, but a ridge-collapsed KDA
    solve lands *just under* it -- a huge ridge leaves the correlation
    direction, which still beats writing nothing -- so its floor is not
    reachable from the ridge axis and is covered by the policy case above
    instead.  Saying so is more useful than inventing a rung that happens to
    trip it.
    """

    family_fold.restore_blank()
    message = _refuse(
        family_fold,
        CompressedSummaryAttentionLayerOperator(
            chart=family_fold.chart,
            recapture=family_fold.recapture,
            ridge_ladder=(1.0e12,),
        ),
        CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
    )
    assert "no-branch baseline" in message


def test_a_recurrence_geometry_the_graph_does_not_declare_fails_closed(
    family_fold: _ChimeraFold,
) -> None:
    """The IR is what was contracted; the live module is what will be folded.

    If they disagree, the plan was approved for an operator that is not the one
    about to run, so the transaction is refused before any mutation instead of
    quietly trusting one of the two.
    """

    family_fold.restore_blank()
    action = next(
        value
        for value in family_fold.plan.actions
        if value.method_id == CAUSAL_KDA_RECURRENT_BRANCH_METHOD
    )
    operator = BoundedGatedDeltaRecurrenceLayerOperator(
        chart=family_fold.chart, recapture=family_fold.recapture
    )
    mixer = family_fold.parent.get_submodule("blocks.1.mixer")
    original = mixer.gate_lower_bound
    mixer.gate_lower_bound = original - 1.0
    before = fingerprint_model_state(family_fold.child)
    try:
        with pytest.raises(
            DynamicCausalOperatorPromotionError, match="gate_lower_bound"
        ):
            operator.prepare(
                action,
                family_fold.context,
                FoldEvidence(
                    family_fold.evidence.cache_key,
                    family_fold.evidence,
                    channels=action.evidence_requirements,
                ),
            )
    finally:
        mixer.gate_lower_bound = original
    assert fingerprint_model_state(family_fold.child) == before


# ---------------------------------------------------------------------------
# The last two chimera families: gated MLA and the Stable LatentMoE channel
# mixer, in one k3-aligned-style trunk that also runs two families the previous
# wave closed.
#
# ``NOTES-W12C.md`` section 3 left ``compressed_latent_attention`` and
# ``stable_latent_moe_*`` as the two remaining *named* planner gaps.  This is
# the same proof as above -- planned, ledgered, rolled back, audited -- for a
# parent whose every layer needs one of them.
# ---------------------------------------------------------------------------


#: Exact 4x for a gated-MLA + Stable-LatentMoE trunk, derived rather than
#: guessed.  This topology has **no** ``[hidden, hidden]`` tensor anywhere (no
#: csa/hca) and every latent width is pinned as an explicit constant, so the
#: unique parameter count is exactly affine in the residual width:
#:
#:     total(H) = 13299 * H + 17732
#:
#: (measured at four widths and asserted below).  ``total(H_p) = 4 total(H_c)``
#: is therefore ``13299 * (H_p - 4 H_c) = 3 * 17732 = 53196``, and
#: ``53196 / 13299 = 4`` exactly, so **H_p = 4 H_c + 4**.  At ``H_c = 31`` that
#: is ``H_p = 128``:
#:
#:     parent 13299 * 128 + 17732 = 1_720_004
#:     child  13299 *  31 + 17732 =   430_001   ->  exactly 4x
#:
#: The unround vocabulary and the MoE widths are what make the coefficient
#: divide three times the constant; they are the solution, not a rounding of
#: one.
LATENT_PARENT_WIDTH = 128
LATENT_CHILD_WIDTH = 31
LATENT_PARENT_PARAMETERS = 1_720_004
LATENT_CHILD_PARAMETERS = 430_001
LATENT_WIDTH_COEFFICIENT = 13_299
LATENT_WIDTH_CONSTANT = 17_732
LATENT_WINDOW = 8
LATENT_EXPERTS = 4
LATENT_EXPERTS_PER_TOKEN = 2
LATENT_SHARED_EXPERTS = 2
LATENT_MOE_LATENT = 16
LATENT_ROUTED_INTERMEDIATE = 8
LATENT_MLA_LATENT = 24
LATENT_MLA_EXTRA_HEAD_DIM = 8


def _latent_config(hidden_size: int, **changes):
    """One k3-aligned-style trunk: local + gated_mla + kda, MoE everywhere."""

    return _config(
        hidden_size,
        **{
            "vocab_size": 1459,
            "intermediate_size": 320,
            "block_pattern": ("local", "gated_mla", "kda", "gated_mla"),
            "activation_kind": "situ",
            "situ_gate_beta": 4.0,
            "situ_up_beta": 25.0,
            "kda_head_dim": 8,
            "kda_value_head_dim": 8,
            "kda_decay_rank": 4,
            "kda_short_conv_kernel_size": 4,
            "mla_latent_size": LATENT_MLA_LATENT,
            "mla_query_latent_size": LATENT_MLA_LATENT,
            "mla_key_head_dim": 16,
            "mla_extra_head_dim": LATENT_MLA_EXTRA_HEAD_DIM,
            "mla_value_head_dim": 16,
            "feed_forward_kind": "stable_latent_moe",
            "moe_num_experts": LATENT_EXPERTS,
            "moe_experts_per_token": LATENT_EXPERTS_PER_TOKEN,
            "moe_shared_experts": LATENT_SHARED_EXPERTS,
            "moe_latent_size": LATENT_MOE_LATENT,
            "moe_routed_intermediate_size": LATENT_ROUTED_INTERMEDIATE,
            "moe_first_dense_layers": 0,
            "local_window": LATENT_WINDOW,
            **changes,
        },
    )


LATENT_ENDPOINT_KEYS = {
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD: (
        "validation_attention_delta_relative_rmse"
    ),
    CAUSAL_KDA_RECURRENT_BRANCH_METHOD: "validation_recurrence_delta_relative_rmse",
    CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD: (
        "validation_latent_attention_delta_relative_rmse"
    ),
    CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD: (
        "validation_channel_mixer_delta_relative_rmse"
    ),
}


@pytest.fixture(scope="module")
def latent_fold() -> _ChimeraFold:
    return _ChimeraFold(
        window=LATENT_WINDOW,
        factory=_latent_config,
        parent_width=LATENT_PARENT_WIDTH,
        child_width=LATENT_CHILD_WIDTH,
        enabled_methods=NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
    )


@pytest.fixture(scope="module")
def completed_latent_run(latent_fold: _ChimeraFold):
    parent_forwards = 0

    def counted(_module, _args, _output):
        nonlocal parent_forwards
        parent_forwards += 1

    latent_fold.restore_blank()
    handle = latent_fold.parent.register_forward_hook(counted)
    try:
        run = latent_fold.execute(_AcceptStructuralCandidate())
    finally:
        handle.remove()
    return run, parent_forwards


def test_the_latent_topology_is_affine_in_the_residual_width() -> None:
    """The exact-4x arithmetic is measured, not assumed.

    Four widths pin the affine law; the pair then follows from it in closed
    form rather than from a search over configurations.
    """

    counts = {}
    for width in (64, 96, LATENT_CHILD_WIDTH, LATENT_PARENT_WIDTH):
        model = _model(_latent_config(width), 7)
        counts[width] = sum(int(value.numel()) for value in model.parameters())
    for width, total in counts.items():
        assert total == LATENT_WIDTH_COEFFICIENT * width + LATENT_WIDTH_CONSTANT
    assert 3 * LATENT_WIDTH_CONSTANT % LATENT_WIDTH_COEFFICIENT == 0
    assert LATENT_PARENT_WIDTH == 4 * LATENT_CHILD_WIDTH + (
        3 * LATENT_WIDTH_CONSTANT // LATENT_WIDTH_COEFFICIENT
    )
    assert counts[LATENT_PARENT_WIDTH] == LATENT_PARENT_PARAMETERS
    assert counts[LATENT_CHILD_WIDTH] == LATENT_CHILD_PARAMETERS
    assert LATENT_PARENT_PARAMETERS == 4 * LATENT_CHILD_PARAMETERS


def test_the_latent_plan_is_ready_with_zero_gaps_at_exactly_four_times(
    latent_fold: _ChimeraFold,
) -> None:
    assert latent_fold.plan.ready
    assert not latent_fold.plan.gaps
    assert latent_fold.parent_graph.unique_parameter_count == (
        LATENT_PARENT_PARAMETERS
    )
    assert latent_fold.child_graph.unique_parameter_count == LATENT_CHILD_PARAMETERS
    assert latent_fold.parent_graph.unique_parameter_count == (
        4 * latent_fold.child_graph.unique_parameter_count
    )

    methods = [action.method_id for action in latent_fold.plan.actions]
    assert methods.count(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD) == 1
    assert methods.count(CAUSAL_KDA_RECURRENT_BRANCH_METHOD) == 1
    # One method id serves both gated-MLA layers, and one serves every Stable
    # LatentMoE channel mixer in the trunk.
    assert methods.count(CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD) == 2
    assert methods.count(CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD) == 4
    assert methods.count(NATIVE_RESIDUAL_CHART_INTERFACE_METHOD) == 14

    owned = [
        path
        for action in latent_fold.plan.actions
        for path in action.target_parameter_paths
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(latent_fold.child_graph.parameter_paths)
    # Every tensor of one MoE channel mixer -- router, latent projections,
    # every routed expert, the accumulator norm and every shared expert --
    # travels in one transaction, because they are one operator.
    moe = next(
        action
        for action in latent_fold.plan.actions
        if action.method_id == CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD
    )
    paths = set(moe.target_parameter_paths)
    for suffix in (
        "ffn.router.weight",
        "ffn.latent_down.weight",
        "ffn.routed_experts.3.up.weight",
        "ffn.routed_norm.weight",
        "ffn.latent_up.weight",
        "ffn.shared_experts.1.down.weight",
    ):
        assert any(path.endswith(suffix) for path in paths)


def test_the_latent_fold_executes_completely_without_a_parent_forward(
    latent_fold: _ChimeraFold, completed_latent_run
) -> None:
    run, parent_forwards = completed_latent_run

    assert run.status == "complete"
    assert all(attempt.accepted for attempt in run.attempts)
    assert parent_forwards == 0
    assert run.observed_parent_root_forward_calls == 0
    # The two new operators recover a routing, read carried parent weights and
    # replay a softmax, but they never call a parent module: every row they
    # consume is a sealed module-boundary capture from the one forward taken
    # before the run began.
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
        latent_fold.child_graph.parameter_paths
    )

    changed = [
        path
        for attempt in run.attempts
        for path in attempt.transaction_report["changed_state_paths"]
    ]
    assert len(changed) == len(set(changed))
    logits = latent_fold.logits(latent_fold.child)
    assert tuple(logits.shape) == (
        BATCH,
        SEQUENCE,
        latent_fold.child_config.vocab_size,
    )
    assert bool(torch.isfinite(logits).all())


def test_every_latent_branch_beats_the_no_branch_baseline(
    completed_latent_run,
) -> None:
    run, _forwards = completed_latent_run
    counts = {
        CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD: 1,
        CAUSAL_KDA_RECURRENT_BRANCH_METHOD: 1,
        CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD: 2,
        CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD: 4,
    }
    for method_id, expected in counts.items():
        reports = _family_reports(run, method_id)
        assert len(reports) == expected
        for report in reports:
            endpoint = report[LATENT_ENDPOINT_KEYS[method_id]]
            assert 0.0 < endpoint < report[
                "no_branch_endpoint_relative_rmse_baseline"
            ]
            assert report["selected_ridge"] in [
                entry["ridge"] for entry in report["ridge_ladder"]
            ]
            assert report["parent_forward_calls"] == 0
            assert report["numeric_weight_matching_objective"] is False
            assert report["zero_gradient"] is True


def test_the_installed_child_reproduces_every_latent_branch_endpoint(
    latent_fold: _ChimeraFold, completed_latent_run
) -> None:
    """The number in the receipt is the number the installed child computes.

    This is what makes both new operators honest rather than plausible.  A
    top-k router is a *discrete* decision: if the installed router disagreed
    with the one the kernel replayed on a single row, that row would run
    through different experts and the recomputed endpoint would not match.  A
    compressed-latent attention is equally unforgiving -- a wrong latent basis,
    a mis-broadcast shared key block or an off-by-one in the causal mask would
    all show up here.
    """

    run, _forwards = completed_latent_run
    assert run.status == "complete"
    live = latent_fold.recapture(latent_fold.child, latent_fold.child_graph)
    _valid, _fit, _selection, validation = latent_fold.evidence.token_masks()
    flat_validation = validation.reshape(-1)

    def endpoint(module_path: str) -> float:
        installed = live.flat_rows(live.output(module_path))[flat_validation]
        expected = latent_fold.chart.encode(
            latent_fold.evidence.flat_rows(
                latent_fold.evidence.output(module_path)
            )
        )[flat_validation]
        difference = (installed - expected).square().mean().sqrt()
        reference = expected.square().mean().sqrt().clamp_min(1e-30)
        return float((difference / reference).item())

    checked = 0
    for attempt in run.attempts:
        key = LATENT_ENDPOINT_KEYS.get(attempt.method_id)
        if key is None:
            continue
        report = attempt.transaction_report
        index = int(report["target_node_id"].split(".")[1])
        module = (
            f"blocks.{index}.ffn"
            if attempt.method_id == CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD
            else f"blocks.{index}.mixer"
        )
        assert endpoint(module) == pytest.approx(report[key], rel=1e-5, abs=1e-9)
        checked += 1
    assert checked == 8


def test_the_latent_attention_receipt_carries_its_evidence_gates(
    completed_latent_run,
) -> None:
    run, _forwards = completed_latent_run
    reports = _family_reports(run, CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD)

    assert len(reports) == 2
    for report in reports:
        assert report["captured_row_source"] == (
            "sealed_native_module_boundary_capture"
        )
        # The key/value split, the head broadcast of the shared block, the
        # scores, the probabilities and the sigmoid gate are all derived, and
        # the gate that checks them is the reconstruction of the *captured*
        # branch delta.
        assert "attention_probabilities" in report["derived_rows"]
        assert "shared_key_broadcast" in report["derived_rows"]
        assert "captured_branch_delta" in report["derived_row_consistency_gate"]
        assert report["readout_identity_relative_rmse"] < 1e-5
        assert report["delta_reconstruction_relative_rmse"] < 1e-5
        # The shared key channels really do carry score mass, so folding them
        # as a head-independent block is not a vacuous claim.
        assert 0.0 < report["shared_key_channel_score_energy_fraction"] < 1.0
        assert report["one_executable_for_both_query_variants"] is True
        assert report["compressed_query_path"] is True
        assert report["kv_latent_basis"] == "identity"
        assert report["attention_probability_overlap"] >= 0.25
        # No score chart is taken: an MLA score is an inner product in head
        # coordinates, not in the residual width.
        assert report["score_chart_used"] is False
        assert report["source_geometry"]["extra_head_dim"] == (
            LATENT_MLA_EXTRA_HEAD_DIM
        )
        assert report["target_geometry"]["extra_head_dim"] == (
            LATENT_MLA_EXTRA_HEAD_DIM
        )
        assert report["source_geometry"]["hidden_size"] == LATENT_PARENT_WIDTH
        assert report["target_geometry"]["hidden_size"] == LATENT_CHILD_WIDTH
        assert report["validation_residual_relative_rmse_bf16"] < 1.0


def test_the_moe_receipt_carries_its_routing_identity_and_row_budget(
    completed_latent_run,
) -> None:
    run, _forwards = completed_latent_run
    reports = _family_reports(run, CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD)

    assert len(reports) == 4
    for report in reports:
        # The routing itself is derived; the gate that checks it compares the
        # recovered row sets against the captured per-expert row counts and the
        # rebuilt accumulator against the captured one.
        assert "expert_row_sets" in report["derived_rows"]
        assert "captured_expert_row_counts" in report[
            "derived_row_consistency_gate"
        ]
        assert report["routing_identity_relative_rmse"] < 1e-5
        assert report["expert_input_relative_rmse"] < 1e-5
        assert report["split_reconstruction_relative_rmse"] < 1e-5
        # Every routed token lands somewhere, and the sparse row budget is
        # reported per expert rather than assumed uniform.
        assert sum(report["expert_row_counts"]) == (
            BATCH * SEQUENCE * LATENT_EXPERTS_PER_TOKEN
        )
        assert len(report["expert_fit_row_counts"]) == LATENT_EXPERTS
        assert all(value > 0 for value in report["expert_fit_row_counts"])
        assert report["expected_expert_rows"] == pytest.approx(
            BATCH * SEQUENCE * LATENT_EXPERTS_PER_TOKEN / LATENT_EXPERTS
        )
        assert report["expert_rows_are_the_parents_own_assignment"] is True
        assert report["router_top_k_agreement"] >= 0.5
        # The balancing buffer is not a learned tensor, so no transaction owns
        # it; equality is required and recorded instead.
        assert report["router_bias_is_a_non_gradient_buffer"] is True
        assert report["router_bias_transported_by_this_transaction"] is False
        assert report["router_bias_equality_required"] is True
        assert report["latent_basis"] == "identity"
        # Both sub-branch endpoints are reported separately, because the
        # bounded alternation trades mass between them to serve the shared one.
        assert report["routed_sub_branch_validation_relative_rmse"] > 0.0
        assert report["shared_sub_branch_validation_relative_rmse"] > 0.0
        assert report["sub_branch_fit_coordinate_system"] == "child_residual_width"
        assert report["target_geometry"]["activation"] == "situ"
        assert report["validation_residual_relative_rmse_bf16"] < 1.0


def test_the_latent_child_tracks_the_parent_where_a_blank_child_cannot(
    latent_fold: _ChimeraFold, completed_latent_run
) -> None:
    run, _forwards = completed_latent_run
    assert run.status == "complete"
    parent_top1 = latent_fold.logits(latent_fold.parent).argmax(dim=-1)
    folded_top1 = latent_fold.logits(latent_fold.child).argmax(dim=-1)
    blank = _model(latent_fold.child_config, 4711)
    blank.load_state_dict(latent_fold.blank_state)
    blank.eval()
    blank_top1 = latent_fold.logits(blank).argmax(dim=-1)

    folded_agreement = float((parent_top1 == folded_top1).double().mean())
    blank_agreement = float((parent_top1 == blank_top1).double().mean())
    # A machinery proof, not a capability claim: the parent is untrained.
    assert folded_agreement > 0.5
    assert folded_agreement > blank_agreement + 0.25


def test_a_rejected_moe_branch_rolls_the_latent_run_back(
    latent_fold: _ChimeraFold,
) -> None:
    class _RejectTheChannelMixer:
        def evaluate(self, context: Any) -> FoldDecision:
            if context.action.method_id == CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD:
                return FoldDecision.reject("Rejected for the rollback contract.")
            return FoldDecision.accept("Accepted.")

    latent_fold.restore_blank()
    run = latent_fold.execute(_RejectTheChannelMixer())

    assert run.status == "rejected"
    rejected = [attempt for attempt in run.attempts if not attempt.accepted]
    assert len(rejected) == 1
    assert rejected[0].method_id == CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD
    assert rejected[0].rollback_exact is True
    # Every transaction accepted before the rejection is undone too, tensor for
    # tensor -- including the whole first layer and the interface bundle.
    assert fingerprint_model_state(latent_fold.child) == (
        latent_fold.blank_fingerprint
    )
    for name, value in latent_fold.child.state_dict().items():
        assert torch.equal(value, latent_fold.blank_state[name])
    latent_fold.restore_blank()


def _refuse_latent(latent_fold: _ChimeraFold, operator, method_id: str, match: str):
    action = next(
        value for value in latent_fold.plan.actions if value.method_id == method_id
    )
    before = fingerprint_model_state(latent_fold.child)
    with pytest.raises(DynamicCausalOperatorPromotionError, match=match) as error:
        operator.prepare(
            action,
            latent_fold.context,
            FoldEvidence(
                latent_fold.evidence.cache_key,
                latent_fold.evidence,
                channels=action.evidence_requirements,
            ),
        )
    assert fingerprint_model_state(latent_fold.child) == before
    return str(error.value)


@pytest.mark.parametrize(
    ("operator_class", "method_id", "policy", "measurement"),
    (
        (
            CompressedLatentKVAttentionLayerOperator,
            CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
            ChimeraLatentKVFoldPolicy(
                minimum_attention_probability_overlap=0.999999
            ),
            "attention mass",
        ),
        (
            StableLatentMoEChannelMixerLayerOperator,
            CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
            ChimeraStableLatentMoEFoldPolicy(
                minimum_router_top_k_agreement=1.0
            ),
            "top-k expert selection",
        ),
    ),
)
def test_a_new_kernel_refusal_becomes_a_rejection_with_rollback(
    latent_fold: _ChimeraFold, operator_class, method_id, policy, measurement
) -> None:
    """A kernel that refuses to emit is a planner-visible failure, not a pass.

    Both kernels carry the design's fail-closed thresholds in a sealed policy
    and raise rather than emit a child when one is breached.  Tightening one of
    those thresholds is the honest way to drive that path: the kernel then
    refuses on *every* ridge rung, and the operator has to surface it as its own
    failure, carrying the kernel's own measurement, and leave the child
    bit-identical.
    """

    latent_fold.restore_blank()
    message = _refuse_latent(
        latent_fold,
        operator_class(
            chart=latent_fold.chart,
            recapture=latent_fold.recapture,
            fold_policy=policy,
        ),
        method_id,
        "no installable branch solve",
    )
    assert measurement in message
    assert "ridge" in message


def test_the_sparse_expert_row_budget_refuses_before_any_mutation(
    latent_fold: _ChimeraFold,
) -> None:
    """Sparse routing decides how much evidence each expert's solve consumes.

    Each expert sees about ``rows * K / E`` rows and roughly half of those land
    in the fit partition.  A floor above that count refuses the whole channel
    mixer on every ridge rung, names the expert and prints the two numbers,
    rather than fitting a dictionary on rows the endpoint gate will never
    exercise.
    """

    latent_fold.restore_blank()
    message = _refuse_latent(
        latent_fold,
        StableLatentMoEChannelMixerLayerOperator(
            chart=latent_fold.chart,
            recapture=latent_fold.recapture,
            fold_policy=ChimeraStableLatentMoEFoldPolicy(
                minimum_expert_fit_rows=BATCH * SEQUENCE
            ),
        ),
        CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
        "no installable branch solve",
    )
    assert "Routed expert" in message
    assert "fit rows" in message
    assert "sealed floor" in message


def test_a_router_bias_the_child_does_not_share_fails_closed(
    latent_fold: _ChimeraFold,
) -> None:
    """The balancing buffer is a contract constant, not a folded tensor.

    ``router_bias`` is registered as a buffer, so it is not a learned tensor,
    not in the IR, and not inside the snapshot/rollback envelope every operator
    here relies on.  It moves the child's expert-selection boundary, so an
    unequal one is refused before any mutation instead of being folded across.
    """

    latent_fold.restore_blank()
    mixer = latent_fold.parent.get_submodule("blocks.0.ffn")
    original = mixer.router_bias.detach().clone()
    with torch.no_grad():
        mixer.router_bias[0] += 1.0
    try:
        message = _refuse_latent(
            latent_fold,
            StableLatentMoEChannelMixerLayerOperator(
                chart=latent_fold.chart, recapture=latent_fold.recapture
            ),
            CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
            "router_bias",
        )
    finally:
        with torch.no_grad():
            mixer.router_bias.copy_(original)
    assert "not a learned tensor" in message


def test_a_latent_geometry_the_graph_does_not_declare_fails_closed(
    latent_fold: _ChimeraFold,
) -> None:
    """The IR is what was contracted; the live module is what will be folded."""

    latent_fold.restore_blank()
    mixer = latent_fold.parent.get_submodule("blocks.1.mixer")
    original = mixer.extra_head_dim
    mixer.extra_head_dim = original - 1
    try:
        _refuse_latent(
            latent_fold,
            CompressedLatentKVAttentionLayerOperator(
                chart=latent_fold.chart, recapture=latent_fold.recapture
            ),
            CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
            "extra_head_dim",
        )
    finally:
        mixer.extra_head_dim = original


# ---------------------------------------------------------------------------
# The last named gap: the dense (non-MoE) channel mixer under SiTU.
#
# ``NOTES-W12F.md`` closed every chimera *family* and left this one lane's
# activation envelope open: the fused gate/up MLP lane admitted SiLU only, so
# the ``moe_first_dense_layers`` layer of the k3-aligned reference had no
# owning transaction.  The lane's activation is now a sealed contract constant
# over ``{silu, situ}`` -- one executable, one method id, two constants -- and
# this trunk proves it in the only way that counts: a complete planned exact-4x
# fold whose layer 0 is a dense SiTU mixer sitting next to gated-MLA and Stable
# LatentMoE neighbours.
# ---------------------------------------------------------------------------


#: Swapping layer 0's MoE channel mixer for a dense one is a pure change of the
#: affine law's coefficients -- the dense mixer is ``3 * intermediate * H`` and
#: the MoE it replaces is ``1956 H + 1552`` on these widths, both linear -- so
#: the exact-4x arithmetic survives with a different vocabulary:
#:
#:     total(H) = 12135 * H + 16180
#:
#: ``total(H_p) = 4 total(H_c)`` is ``12135 * (H_p - 4 H_c) = 3 * 16180``, and
#: ``48540 / 12135 = 4`` exactly, so **H_p = 4 H_c + 4** again.  At ``H_c = 31``
#: that is ``H_p = 128``:
#:
#:     parent 12135 * 128 + 16180 = 1_569_460
#:     child  12135 *  31 + 16180 =   392_365   ->  exactly 4x
DENSE_SITU_PARENT_WIDTH = 128
DENSE_SITU_CHILD_WIDTH = 31
DENSE_SITU_PARENT_PARAMETERS = 1_569_460
DENSE_SITU_CHILD_PARAMETERS = 392_365
DENSE_SITU_WIDTH_COEFFICIENT = 12_135
DENSE_SITU_WIDTH_CONSTANT = 16_180


def _dense_situ_config(hidden_size: int, **changes):
    """The latent trunk with its first channel mixer dense, under SiTU."""

    return _latent_config(
        hidden_size,
        **{
            "vocab_size": 1403,
            "moe_first_dense_layers": 1,
            **changes,
        },
    )


DENSE_SITU_ENDPOINT_KEYS = {
    **LATENT_ENDPOINT_KEYS,
    CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD: "validation_mlp_delta_relative_rmse",
}


#: A parent whose randomly initialised router clears the MoE kernel's sparse
#: expert row budget on every layer.  This is stated rather than hidden: at the
#: file's default seed the smallest routed expert of this trunk owns **9** fit
#: rows against a sealed floor of **15**, and the MoE kernel refuses the whole
#: channel mixer on every ridge rung -- correctly, because an expert fitted on
#: nine rows is an interpolation the endpoint gate cannot see.  Lowering the
#: floor to make a proof run would have disabled the gate this suite exists to
#: exercise, so the parent was chosen instead; the smallest expert here owns
#: 44 fit rows and the counts are asserted below.
DENSE_SITU_PARENT_SEED = 512


@pytest.fixture(scope="module")
def dense_situ_fold() -> _ChimeraFold:
    return _ChimeraFold(
        window=LATENT_WINDOW,
        factory=_dense_situ_config,
        parent_width=DENSE_SITU_PARENT_WIDTH,
        child_width=DENSE_SITU_CHILD_WIDTH,
        enabled_methods=NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
        parent_seed=DENSE_SITU_PARENT_SEED,
    )


@pytest.fixture(scope="module")
def completed_dense_situ_run(dense_situ_fold: _ChimeraFold):
    parent_forwards = 0

    def counted(_module, _args, _output):
        nonlocal parent_forwards
        parent_forwards += 1

    dense_situ_fold.restore_blank()
    handle = dense_situ_fold.parent.register_forward_hook(counted)
    try:
        run = dense_situ_fold.execute(_AcceptStructuralCandidate())
    finally:
        handle.remove()
    return run, parent_forwards


def test_the_dense_situ_topology_is_affine_in_the_residual_width() -> None:
    counts = {}
    for width in (64, 96, DENSE_SITU_CHILD_WIDTH, DENSE_SITU_PARENT_WIDTH):
        model = _model(_dense_situ_config(width), 7)
        counts[width] = sum(int(value.numel()) for value in model.parameters())
    for width, total in counts.items():
        assert total == DENSE_SITU_WIDTH_COEFFICIENT * width + (
            DENSE_SITU_WIDTH_CONSTANT
        )
    assert 3 * DENSE_SITU_WIDTH_CONSTANT % DENSE_SITU_WIDTH_COEFFICIENT == 0
    assert DENSE_SITU_PARENT_WIDTH == 4 * DENSE_SITU_CHILD_WIDTH + (
        3 * DENSE_SITU_WIDTH_CONSTANT // DENSE_SITU_WIDTH_COEFFICIENT
    )
    assert counts[DENSE_SITU_PARENT_WIDTH] == DENSE_SITU_PARENT_PARAMETERS
    assert counts[DENSE_SITU_CHILD_WIDTH] == DENSE_SITU_CHILD_PARAMETERS
    assert DENSE_SITU_PARENT_PARAMETERS == 4 * DENSE_SITU_CHILD_PARAMETERS


def test_the_dense_situ_plan_is_ready_with_zero_gaps_at_exactly_four_times(
    dense_situ_fold: _ChimeraFold,
) -> None:
    assert dense_situ_fold.plan.ready
    assert not dense_situ_fold.plan.gaps
    assert dict(dense_situ_fold.parent_graph.metadata)["kernel_gap_block_kinds"] == ""
    assert dense_situ_fold.parent_graph.unique_parameter_count == (
        DENSE_SITU_PARENT_PARAMETERS
    )
    assert dense_situ_fold.child_graph.unique_parameter_count == (
        DENSE_SITU_CHILD_PARAMETERS
    )
    assert dense_situ_fold.parent_graph.unique_parameter_count == (
        4 * dense_situ_fold.child_graph.unique_parameter_count
    )

    methods = [action.method_id for action in dense_situ_fold.plan.actions]
    assert methods.count(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD) == 1
    assert methods.count(CAUSAL_KDA_RECURRENT_BRANCH_METHOD) == 1
    assert methods.count(CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD) == 2
    # One dense SiTU channel mixer and three Stable LatentMoE ones, on the same
    # trunk, planned by two different method ids.
    assert methods.count(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD) == 1
    assert methods.count(CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD) == 3
    assert methods.count(NATIVE_RESIDUAL_CHART_INTERFACE_METHOD) == 14

    dense = next(
        action
        for action in dense_situ_fold.plan.actions
        if action.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    )
    # The SiTU constant is in the sealed action, not only in the operator.
    assert dict(dense.configuration)["fit_objective"] == (
        "cached_parent_endpoint_coupled_fused_gate_up_situ_residual_write"
    )
    owned = [
        path
        for action in dense_situ_fold.plan.actions
        for path in action.target_parameter_paths
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(dense_situ_fold.child_graph.parameter_paths)


def test_the_dense_situ_fold_executes_completely_without_a_parent_forward(
    dense_situ_fold: _ChimeraFold, completed_dense_situ_run
) -> None:
    run, parent_forwards = completed_dense_situ_run

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
        dense_situ_fold.child_graph.parameter_paths
    )
    logits = dense_situ_fold.logits(dense_situ_fold.child)
    assert bool(torch.isfinite(logits).all())


def test_every_dense_situ_branch_beats_the_no_branch_baseline(
    completed_dense_situ_run,
) -> None:
    run, _forwards = completed_dense_situ_run
    counts = {
        CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD: 1,
        CAUSAL_KDA_RECURRENT_BRANCH_METHOD: 1,
        CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD: 2,
        CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD: 1,
        CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD: 3,
    }
    for method_id, expected in counts.items():
        reports = _family_reports(run, method_id)
        assert len(reports) == expected
        for report in reports:
            endpoint = report[DENSE_SITU_ENDPOINT_KEYS[method_id]]
            assert 0.0 < endpoint < report[
                "no_branch_endpoint_relative_rmse_baseline"
            ]
            assert report["numeric_weight_matching_objective"] is False
            assert report["zero_gradient"] is True
            if method_id == CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD:
                # The MoE kernel's sparse expert row budget is live on this
                # trunk rather than loosened to let the proof run: the smallest
                # routed expert clears the sealed floor by a real margin.
                assert min(report["expert_fit_row_counts"]) >= 44


def test_the_dense_situ_receipt_names_the_activation_it_folded(
    completed_dense_situ_run,
) -> None:
    """A SiTU fold and a SiLU fold share a method id; the receipt separates them."""

    run, _forwards = completed_dense_situ_run
    reports = _family_reports(run, CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD)
    assert len(reports) == 1
    report = reports[0]

    assert report["gated_activation"] == {
        "kind": "situ",
        "gate_beta": 4.0,
        "up_beta": 25.0,
    }
    assert report["fused_gate_up_refused_in_transaction"] is True
    assert report["parent_residual_coordinate_endpoints"] is False


def test_the_installed_dense_situ_child_computes_situ_and_not_silu(
    dense_situ_fold: _ChimeraFold, completed_dense_situ_run
) -> None:
    """The dictionary the kernel compiled is SiTU's, measured against both.

    This is the gate that makes the widening honest rather than nominal, and it
    is worth being precise about *why* it is needed.  On a randomly initialised
    parent most pre-activations sit well inside SiTU's caps, so the two
    activations are numerically close -- SiLU applied to the installed tensors
    is only a couple of percent off.  A lane that had quietly kept SiLU's
    dictionary would therefore still fold, still beat the 1.0 no-branch
    baseline and still pass every count above.  What it could not do is make
    the receipt match the child: the receipt endpoint is computed through the
    kernel's own activation and the child computes SiTU, so
    ``test_the_installed_child_reproduces_every_dense_situ_branch_endpoint``
    would miss by three orders of magnitude.  Here the same fact is measured
    directly, at the module boundary.
    """

    run, _forwards = completed_dense_situ_run
    assert run.status == "complete"
    child = dense_situ_fold.child
    live = dense_situ_fold.recapture(child, dense_situ_fold.child_graph)
    module = child.get_submodule("blocks.0.ffn")
    rows = live.flat_rows(live.input("blocks.0.ffn"))
    written = live.flat_rows(live.output("blocks.0.ffn"))

    with torch.no_grad():
        up_weight = module.up.weight.detach().to(dtype=rows.dtype)
        down_weight = module.down.weight.detach().to(dtype=rows.dtype)
        gate, up = (rows @ up_weight.T).chunk(2, dim=-1)
        situ = (
            module.situ_gate_beta * torch.tanh(gate / module.situ_gate_beta)
        ) * torch.sigmoid(gate) * (
            module.situ_up_beta * torch.tanh(up / module.situ_up_beta)
        )
        silu = torch.nn.functional.silu(gate) * up

    def error(features: Any) -> float:
        prediction = features @ down_weight.T
        difference = (prediction - written).square().mean().sqrt()
        reference = written.square().mean().sqrt().clamp_min(1e-30)
        return float((difference / reference).item())

    assert module.activation_kind == "situ"
    assert error(situ) < 1.0e-6
    assert error(silu) > 1.0e-2
    assert error(silu) > 1000.0 * error(situ)


def test_the_installed_child_reproduces_every_dense_situ_branch_endpoint(
    dense_situ_fold: _ChimeraFold, completed_dense_situ_run
) -> None:
    run, _forwards = completed_dense_situ_run
    assert run.status == "complete"
    live = dense_situ_fold.recapture(dense_situ_fold.child, dense_situ_fold.child_graph)
    _valid, _fit, _selection, validation = dense_situ_fold.evidence.token_masks()
    flat_validation = validation.reshape(-1)

    def endpoint(module_path: str) -> float:
        installed = live.flat_rows(live.output(module_path))[flat_validation]
        expected = dense_situ_fold.chart.encode(
            dense_situ_fold.evidence.flat_rows(
                dense_situ_fold.evidence.output(module_path)
            )
        )[flat_validation]
        difference = (installed - expected).square().mean().sqrt()
        reference = expected.square().mean().sqrt().clamp_min(1e-30)
        return float((difference / reference).item())

    channel_mixers = {
        CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
        CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
    }
    checked = 0
    for attempt in run.attempts:
        key = DENSE_SITU_ENDPOINT_KEYS.get(attempt.method_id)
        if key is None:
            continue
        report = attempt.transaction_report
        index = int(report["target_node_id"].split(".")[1])
        module = (
            f"blocks.{index}.ffn"
            if attempt.method_id in channel_mixers
            else f"blocks.{index}.mixer"
        )
        assert endpoint(module) == pytest.approx(report[key], rel=1e-5, abs=1e-9)
        checked += 1
    assert checked == 8


def test_the_dense_situ_child_tracks_the_parent_where_a_blank_child_cannot(
    dense_situ_fold: _ChimeraFold, completed_dense_situ_run
) -> None:
    run, _forwards = completed_dense_situ_run
    assert run.status == "complete"
    parent_top1 = dense_situ_fold.logits(dense_situ_fold.parent).argmax(dim=-1)
    folded_top1 = dense_situ_fold.logits(dense_situ_fold.child).argmax(dim=-1)
    blank = _model(dense_situ_fold.child_config, 4711)
    blank.load_state_dict(dense_situ_fold.blank_state)
    blank.eval()
    blank_top1 = dense_situ_fold.logits(blank).argmax(dim=-1)

    folded_agreement = float((parent_top1 == folded_top1).double().mean())
    blank_agreement = float((parent_top1 == blank_top1).double().mean())
    # A machinery proof, not a capability claim: the parent is untrained.
    assert folded_agreement > 0.5
    assert folded_agreement > blank_agreement + 0.25


def test_a_rejected_dense_situ_branch_rolls_the_run_back(
    dense_situ_fold: _ChimeraFold,
) -> None:
    class _RejectTheDenseMixer:
        def evaluate(self, context: Any) -> FoldDecision:
            if context.action.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD:
                return FoldDecision.reject("Rejected for the rollback contract.")
            return FoldDecision.accept("Accepted.")

    dense_situ_fold.restore_blank()
    run = dense_situ_fold.execute(_RejectTheDenseMixer())

    assert run.status == "rejected"
    rejected = [attempt for attempt in run.attempts if not attempt.accepted]
    assert len(rejected) == 1
    assert rejected[0].method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    assert rejected[0].rollback_exact is True
    assert fingerprint_model_state(dense_situ_fold.child) == (
        dense_situ_fold.blank_fingerprint
    )
    for name, value in dense_situ_fold.child.state_dict().items():
        assert torch.equal(value, dense_situ_fold.blank_state[name])
    dense_situ_fold.restore_blank()


#: Small and slow on purpose (see ``TRAIN_LEARNING_RATE``), but a real drop
#: under chimera-omni's own pass criterion: mean-last-10 below mean-first-10.
DENSE_SITU_TRAIN_STEPS = 80


@pytest.fixture(scope="module")
def trained_dense_situ_fold() -> _ChimeraFold:
    return _ChimeraFold(
        window=LATENT_WINDOW,
        factory=_dense_situ_config,
        parent_width=DENSE_SITU_PARENT_WIDTH,
        child_width=DENSE_SITU_CHILD_WIDTH,
        enabled_methods=NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
        parent_seed=DENSE_SITU_PARENT_SEED,
        train_steps=DENSE_SITU_TRAIN_STEPS,
    )


def test_a_trained_parent_folds_and_the_child_tracks_its_next_token(
    trained_dense_situ_fold: _ChimeraFold,
) -> None:
    """The same trunk, folded from a parent that actually learned something.

    Every other case in this file measures mechanism against a randomly
    initialised parent, which is the honest thing to say about them.  This one
    trains the parent first under chimera-omni's own tiny-proof objective --
    allowed, because training a *parent* is not folding a child -- and then
    folds it by exactly the same closed-form transactions.  The executed run's
    zero-training audits are what separate the two: the parent's optimizer ran
    before the run began, and nothing inside the run constructs one.
    """

    fold = trained_dense_situ_fold
    training = fold.training
    assert training is not None
    assert training["chimera_tiny_proof_status"] == "passed"
    assert training["mean_last_10_loss"] < training["mean_first_10_loss"]

    fold.restore_blank()
    run = fold.execute(_AcceptStructuralCandidate())

    assert run.status == "complete"
    assert run.observed_parent_root_forward_calls == 0
    assert run.observed_parent_module_forward_calls == 0
    for name in (
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
    ):
        assert getattr(run, name) == 0
    assert run.executed_fold_static_zero_training_audit_passed is True

    dense = _family_reports(run, CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD)
    assert len(dense) == 1
    assert dense[0]["gated_activation"]["kind"] == "situ"
    assert 0.0 < dense[0]["validation_mlp_delta_relative_rmse"] < dense[0][
        "no_branch_endpoint_relative_rmse_baseline"
    ]

    parent_top1 = fold.logits(fold.parent).argmax(dim=-1)
    folded_top1 = fold.logits(fold.child).argmax(dim=-1)
    blank = _model(fold.child_config, 4711)
    blank.load_state_dict(fold.blank_state)
    blank.eval()
    blank_top1 = fold.logits(blank).argmax(dim=-1)

    folded_agreement = float((parent_top1 == folded_top1).double().mean())
    blank_agreement = float((parent_top1 == blank_top1).double().mean())
    assert folded_agreement > 0.75
    assert blank_agreement < 0.05


def _refuse_dense_situ(fold: _ChimeraFold, operator, match: str) -> str:
    action = next(
        value
        for value in fold.plan.actions
        if value.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    )
    before = fingerprint_model_state(fold.child)
    with pytest.raises(DynamicCausalOperatorPromotionError, match=match) as error:
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
    return str(error.value)


def test_an_activation_the_graph_does_not_declare_fails_closed(
    dense_situ_fold: _ChimeraFold,
) -> None:
    """The IR is what was contracted; the live module is what will be folded.

    The planner proved a SiTU crossing.  If the module the kernel is about to
    read says SiLU, the plan was approved for an operator that is not the one
    about to be folded, and the transaction is refused before any mutation
    rather than silently trusting either source.
    """

    dense_situ_fold.restore_blank()
    mixer = dense_situ_fold.parent.get_submodule("blocks.0.ffn")
    original = mixer.activation_kind
    mixer.activation_kind = "silu"
    try:
        message = _refuse_dense_situ(
            dense_situ_fold,
            NativeFusedGatedMLPBranchOperator(
                chart=dense_situ_fold.chart, recapture=dense_situ_fold.recapture
            ),
            "activation",
        )
    finally:
        mixer.activation_kind = original
    assert "situ" in message and "silu" in message


def test_a_situ_cap_the_child_does_not_share_fails_closed(
    dense_situ_fold: _ChimeraFold,
) -> None:
    """The betas are the constant, not decoration on the name."""

    dense_situ_fold.restore_blank()
    mixer = dense_situ_fold.child.get_submodule("blocks.0.ffn")
    original = mixer.situ_up_beta
    mixer.situ_up_beta = original * 2.0
    try:
        message = _refuse_dense_situ(
            dense_situ_fold,
            NativeFusedGatedMLPBranchOperator(
                chart=dense_situ_fold.chart, recapture=dense_situ_fold.recapture
            ),
            "situ_up_beta",
        )
    finally:
        mixer.situ_up_beta = original
    assert "target" in message
