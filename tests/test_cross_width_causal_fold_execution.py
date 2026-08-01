"""Cross-width execution proof: an H64 llama parent folded into an H48 child.

Nothing about the branch kernels changes here.  The parent and the child are
both llama-shaped decoders with identical head geometry and identical MLP
width; the *only* thing that differs is the residual coordinate system, 64
dimensions against 48.  That crossing had no executable path at all before the
per-boundary residual charts existed — the promotion preamble and the planner
contract both refused it — so this file is the first end-to-end evidence that
the crossing is a target-construction problem rather than a solver problem.

The parent is the design's stage-1 exact-subspace control: every residual-facing
weight is confined to one 48-dimensional subspace and every RMSNorm gain is
constant, so the parent's residual states lie exactly in 48 dimensions.  A chart
that finds that subspace discards nothing, which means any residual endpoint
error measured below belongs to the chart machinery, not to the contraction.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    CausalGQAAttentionBranchOperator,
    CausalGatedMLPBranchOperator,
    CrossWidthCausalBranchOperator,
    CrossWidthTokenHeadInterfaceOperator,
    DynamicCausalOperatorPromotionError,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
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
from neural_forge.dynamic_fold_planner import (  # noqa: E402
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_GATED_MLP_BRANCH_METHOD,
    CAUSAL_LAYER_BRANCH_METHODS,
    DynamicFoldPlanner,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_causal_cross_width_charts import (  # noqa: E402
    ATTENTION_INPUT_BOUNDARY,
    MLP_INPUT_BOUNDARY,
    LiveChildBoundaryRecapture,
    compile_residual_boundary_chart_set,
    identity_residual_boundary_chart_set,
    residual_boundary_id,
    residual_boundary_ids,
)


TOKEN_INTERFACE_BRIDGE_METHOD = "token_interface_bridge"
TOKENIZER = canonical_sha256({"tokenizer": "shared-cross-width-fold"})

PARENT_WIDTH = 64
CHILD_WIDTH = 48
INTERMEDIATE = 96
LAYERS = 2
VOCABULARY = 48


def _llama(seed: int, hidden: int, **changes: Any):
    torch.manual_seed(seed)
    configuration = {
        "vocab_size": VOCABULARY,
        "hidden_size": hidden,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    configuration.update(changes)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**configuration))
    model.eval()
    return model


def _confine(model, rank: int, seed: int = 913):
    """Confine every residual state of one parent to an exact ``rank`` subspace."""

    generator = torch.Generator().manual_seed(seed)
    width = int(model.config.hidden_size)
    basis, _ = torch.linalg.qr(torch.randn(width, rank, generator=generator))
    with torch.no_grad():

        def consume(path: str) -> None:
            weight = model.get_parameter(path)
            weight.copy_((weight @ basis) @ basis.T)

        def produce(path: str) -> None:
            weight = model.get_parameter(path)
            weight.copy_(basis @ (basis.T @ weight))

        consume("model.embed_tokens.weight")
        consume("lm_head.weight")
        model.get_parameter("model.norm.weight").fill_(1.0)
        for index in range(LAYERS):
            prefix = f"model.layers.{index}."
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
            ):
                consume(prefix + name + ".weight")
            for name in ("self_attn.o_proj", "mlp.down_proj"):
                produce(prefix + name + ".weight")
            model.get_parameter(prefix + "input_layernorm.weight").fill_(1.0)
            model.get_parameter(prefix + "post_attention_layernorm.weight").fill_(1.0)
    return basis


def _input_ids(sequences: int = 24, length: int = 24):
    return torch.randint(
        0,
        VOCABULARY,
        (sequences, length),
        generator=torch.Generator().manual_seed(7),
    )


class _AcceptStructuralCandidate:
    """Behavior remains the outer authority; structure is admitted here."""

    def evaluate(self, context: Any) -> FoldDecision:
        return FoldDecision.accept("Structurally complete cross-width branch install.")


class _CrossWidthFold:
    """One prepared cross-width plan plus everything it needs to execute."""

    def __init__(self) -> None:
        self.parent = _llama(20260801, PARENT_WIDTH)
        self.subspace = _confine(self.parent, CHILD_WIDTH)
        self.child = _llama(4711, CHILD_WIDTH)
        self.input_ids = _input_ids()
        self.parent_graph = inspect_model_circuits(
            self.parent,
            model_name="parent",
            tokenizer_fingerprint=TOKENIZER,
        )
        self.child_graph = inspect_model_circuits(
            self.child,
            model_name="child",
            tokenizer_fingerprint=TOKENIZER,
        )
        self.evidence = capture_hf_causal_circuit_evidence(
            self.parent,
            {"input_ids": self.input_ids},
            parent_graph_fingerprint=self.parent_graph.fingerprint,
            metadata={"test": "cross-width-h64-to-h48"},
        )
        self.charts = compile_residual_boundary_chart_set(
            self.evidence,
            self.parent,
            self.child,
        )
        self.blank_state = {
            name: value.detach().clone()
            for name, value in self.child.state_dict().items()
        }
        self.blank_fingerprint = fingerprint_model_state(self.child)
        self.plan = DynamicFoldPlanner(
            enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
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

    def recapture(self) -> LiveChildBoundaryRecapture:
        return LiveChildBoundaryRecapture({"input_ids": self.input_ids})

    def registry(self, recapture: LiveChildBoundaryRecapture) -> FoldOperatorRegistry:
        return FoldOperatorRegistry(
            (
                ExactCopyOperator(),
                CrossWidthCausalBranchOperator(
                    CausalGQAAttentionBranchOperator(),
                    chart_set=self.charts,
                    recapture=recapture,
                ),
                CrossWidthCausalBranchOperator(
                    CausalGatedMLPBranchOperator(),
                    chart_set=self.charts,
                    recapture=recapture,
                ),
                CrossWidthTokenHeadInterfaceOperator(chart_set=self.charts),
            )
        )

    def execute(self, evaluator, recapture=None) -> Any:
        recapture = self.recapture() if recapture is None else recapture
        return DynamicFoldExecutor(
            operators=self.registry(recapture),
            evidence_provider=StaticEvidenceProvider(
                self.evidence.cache_key, self.evidence
            ),
            evaluator=evaluator,
        ).execute(self.plan, self.context)


@pytest.fixture(scope="module")
def cross_width() -> _CrossWidthFold:
    return _CrossWidthFold()


@pytest.fixture(scope="module")
def completed_run(cross_width: _CrossWidthFold):
    parent_forwards = 0

    def counted(_module, _args, _output):
        nonlocal parent_forwards
        parent_forwards += 1

    cross_width.restore_blank()
    recapture = cross_width.recapture()
    handle = cross_width.parent.register_forward_hook(counted)
    try:
        run = cross_width.execute(_AcceptStructuralCandidate(), recapture)
    finally:
        handle.remove()
    return run, parent_forwards, recapture


def test_the_parent_residual_really_lives_in_the_child_width(
    cross_width: _CrossWidthFold,
) -> None:
    basis = cross_width.subspace.double()
    for layer in cross_width.evidence.layers:
        for name in ("block_input", "post_attention_residual", "block_output"):
            rows = getattr(layer, name).reshape(-1, PARENT_WIDTH).double()
            outside = rows - (rows @ basis) @ basis.T
            assert float(outside.norm() / rows.norm()) < 1e-6


def test_cross_width_plan_is_gap_free_and_partitions_the_child(
    cross_width: _CrossWidthFold,
) -> None:
    plan = cross_width.plan

    assert plan.ready
    assert not plan.gaps
    methods = [action.method_id for action in plan.actions]
    assert methods.count(TOKEN_INTERFACE_BRIDGE_METHOD) == 1
    assert methods.count(CAUSAL_ATTENTION_BRANCH_METHOD) == LAYERS
    assert methods.count(CAUSAL_GATED_MLP_BRANCH_METHOD) == LAYERS

    owned = [
        path for action in plan.actions for path in action.target_parameter_paths
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(cross_width.child_graph.parameter_paths)

    # Every action that crosses the width says so, and the interface-locked
    # boundaries travel together with the head.
    interface = next(
        action
        for action in plan.actions
        if action.method_id == TOKEN_INTERFACE_BRIDGE_METHOD
    )
    assert sorted(interface.target_parameter_paths) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    for action in plan.actions:
        configuration = dict(action.configuration)
        if action.method_id in {
            CAUSAL_ATTENTION_BRANCH_METHOD,
            CAUSAL_GATED_MLP_BRANCH_METHOD,
            TOKEN_INTERFACE_BRIDGE_METHOD,
        }:
            assert configuration["boundary_chart_required"] is True
    assert (
        dict(interface.configuration)["fit_objective"]
        == "chart_projected_token_head_interface"
    )


def test_the_same_crossing_stays_gapped_without_the_experimental_branches(
    cross_width: _CrossWidthFold,
) -> None:
    """The crossing is executable only because the branch lane was enabled."""

    plan = DynamicFoldPlanner().plan(
        (cross_width.parent_graph,),
        cross_width.child_graph,
        child_state_fingerprint=cross_width.blank_fingerprint,
        evidence_fingerprint=cross_width.evidence.cache_key,
        parent_state_fingerprints={
            "parent": fingerprint_model_state(cross_width.parent)
        },
    )

    assert not plan.ready
    assert {gap.required_capability for gap in plan.gaps} >= {
        "cross_family_closed_loop_layer_operator"
    }


def test_cross_width_fold_executes_completely_without_a_parent_forward(
    cross_width: _CrossWidthFold,
    completed_run,
) -> None:
    run, parent_forwards, recapture = completed_run

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
        cross_width.child_graph.parameter_paths
    )
    changed = [
        path
        for attempt in run.attempts
        for path in attempt.transaction_report["changed_state_paths"]
    ]
    assert len(changed) == len(set(changed))
    assert set(changed) <= set(cross_width.child_graph.parameter_paths)

    # One live-child recapture per branch action: the sequential discipline.
    assert recapture.recapture_calls == 2 * LAYERS

    assert fingerprint_model_state(cross_width.child) != cross_width.blank_fingerprint
    with torch.no_grad():
        logits = cross_width.child(
            input_ids=cross_width.input_ids,
            use_cache=False,
        ).logits
    assert tuple(logits.shape) == (
        int(cross_width.input_ids.shape[0]),
        int(cross_width.input_ids.shape[1]),
        VOCABULARY,
    )
    assert bool(torch.isfinite(logits).all())


def test_cross_width_endpoint_errors_and_chart_defects_are_inside_their_bounds(
    cross_width: _CrossWidthFold,
    completed_run,
) -> None:
    run, _forwards, _recapture = completed_run
    branch_methods = {CAUSAL_ATTENTION_BRANCH_METHOD, CAUSAL_GATED_MLP_BRANCH_METHOD}
    seen_boundaries: list[str] = []

    for attempt in run.attempts:
        report = attempt.transaction_report
        if attempt.method_id not in branch_methods:
            continue
        assert report["cross_width_residual_crossing"] is True
        assert report["parent_residual_width"] == PARENT_WIDTH
        assert report["child_residual_width"] == CHILD_WIDTH
        assert report["sequential_child_replay"] is True
        assert report["parent_residual_coordinate_endpoints"] is False
        assert report["parent_forward_calls"] == 0
        assert report["boundary_chart_set_sha256"] == cross_width.charts.set_sha256
        assert report["reverse_reclosure_sweep_executed"] is False
        seen_boundaries.append(report["input_boundary_id"])

        target = report["sequential_branch_target"]
        assert target["upstream_error_correction_included"] is True
        assert target["equation"] == "Y = h_next @ E_next - hhat_actual"
        # The sequential form and its three-term expansion agree, which is what
        # makes the per-responsibility energies below trustworthy.
        assert target["sequential_expansion_max_abs_error"] < 1e-6
        assert target["fit_transported_parent_delta_energy"] > 0.0
        assert target["fit_upstream_error_energy_ratio"] < 1e-2

        for side in ("input_boundary_chart", "output_boundary_chart"):
            chart = report[side]
            assert chart["cycle_defect"] < 1e-5
            assert chart["norm_commutation_row_fluctuation"] < 1e-9
            unabsorbed = chart["absorption_demand_unabsorbed_energy_fraction"]
            if unabsorbed is not None:
                assert unabsorbed < 1e-6

        key = (
            "validation_attention_delta_relative_rmse"
            if attempt.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
            else "validation_mlp_delta_relative_rmse"
        )
        # Held-out endpoint error of a real 64 -> 48 residual crossing, well
        # under the no-branch baseline of one.  Behavior, not this number,
        # remains the acceptance authority.
        assert 0.0 < report[key] < 5e-3
        assert report["no_branch_endpoint_relative_rmse_baseline"] == 1.0

    # Every residual-write boundary except the terminal one opened a branch.
    assert sorted(seen_boundaries) == sorted(residual_boundary_ids(LAYERS)[:-1])

    interface = next(
        attempt.transaction_report
        for attempt in run.attempts
        if attempt.method_id == TOKEN_INTERFACE_BRIDGE_METHOD
    )
    assert interface["kind"] == "chart_projected_token_head_interface"
    assert interface["final_norm_relative_rmse"] < 1e-2
    assert interface["embedding_boundary_chart"]["interface_locked"] is True
    assert interface["terminal_boundary_chart"]["interface_locked"] is True


def test_the_folded_child_reproduces_the_parent_logits_across_the_crossing(
    cross_width: _CrossWidthFold,
    completed_run,
) -> None:
    """The whole-model number the branch reports only bound indirectly.

    This is still a teacher-forced checkpoint metric on an exact-subspace toy,
    not a behavior verdict, but it is the end-to-end statement: an H64 parent
    and its H48 child produce the same logits through completely different
    residual coordinates.
    """

    run, _forwards, _recapture = completed_run
    assert run.status == "complete"
    with torch.no_grad():
        parent_logits = cross_width.parent(
            input_ids=cross_width.input_ids,
            use_cache=False,
        ).logits
        child_logits = cross_width.child(
            input_ids=cross_width.input_ids,
            use_cache=False,
        ).logits

    error = float(
        (child_logits - parent_logits).square().mean().sqrt()
        / parent_logits.square().mean().sqrt()
    )
    assert error < 5e-3


def test_cross_width_rejection_rolls_back_the_whole_run(
    cross_width: _CrossWidthFold,
) -> None:
    class _RejectTheSecondAttentionBranch:
        def __init__(self) -> None:
            self.seen = 0

        def evaluate(self, context: Any) -> FoldDecision:
            if context.action.method_id != CAUSAL_ATTENTION_BRANCH_METHOD:
                return FoldDecision.accept("Accepted.")
            self.seen += 1
            if self.seen == 2:
                return FoldDecision.reject("Rejected for the rollback contract.")
            return FoldDecision.accept("Accepted.")

    cross_width.restore_blank()
    run = cross_width.execute(_RejectTheSecondAttentionBranch())

    assert run.status == "rejected"
    rejected = [attempt for attempt in run.attempts if not attempt.accepted]
    assert len(rejected) == 1
    assert rejected[0].method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    assert rejected[0].rollback_exact is True
    assert (
        rejected[0].transaction_report["final_state_fingerprint"]
        == rejected[0].incumbent_state_fingerprint
    )
    assert fingerprint_model_state(cross_width.child) == (
        cross_width.blank_fingerprint
    )
    for name, value in cross_width.child.state_dict().items():
        assert torch.equal(value, cross_width.blank_state[name])


def _branch_action(plan, method_id: str, layer_index: int = 0):
    return next(
        action
        for action in plan.actions
        if action.method_id == method_id
        and dict(action.configuration)["target_layer_index"] == layer_index
    )


def _evidence(cross_width: _CrossWidthFold, action) -> FoldEvidence:
    return FoldEvidence(
        cross_width.evidence.cache_key,
        cross_width.evidence,
        channels=action.evidence_requirements,
    )


def test_the_plain_branch_operator_still_refuses_the_crossing(
    cross_width: _CrossWidthFold,
) -> None:
    """Fail-closed default: charts are the only door through the width change."""

    cross_width.restore_blank()
    action = _branch_action(cross_width.plan, CAUSAL_ATTENTION_BRANCH_METHOD)
    before = fingerprint_model_state(cross_width.child)

    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="no boundary-map operator is registered",
    ):
        CausalGQAAttentionBranchOperator().prepare(
            action,
            cross_width.context,
            _evidence(cross_width, action),
        )
    assert fingerprint_model_state(cross_width.child) == before


def test_the_cross_width_operator_refuses_a_foreign_chart_set(
    cross_width: _CrossWidthFold,
) -> None:
    cross_width.restore_blank()
    action = _branch_action(cross_width.plan, CAUSAL_ATTENTION_BRANCH_METHOD)
    other_child = _llama(4711, CHILD_WIDTH, intermediate_size=64)
    foreign = compile_residual_boundary_chart_set(
        cross_width.evidence,
        cross_width.parent,
        other_child,
    )
    before = fingerprint_model_state(cross_width.child)

    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="different child topology",
    ):
        CrossWidthCausalBranchOperator(
            CausalGQAAttentionBranchOperator(),
            chart_set=foreign,
            recapture=cross_width.recapture(),
        ).prepare(action, cross_width.context, _evidence(cross_width, action))
    assert fingerprint_model_state(cross_width.child) == before


def test_the_cross_width_operator_refuses_an_unplanned_crossing(
    cross_width: _CrossWidthFold,
) -> None:
    """An operator may not widen the envelope the planner did not open."""

    cross_width.restore_blank()
    action = _branch_action(cross_width.plan, CAUSAL_ATTENTION_BRANCH_METHOD)
    stripped = replace(
        action,
        configuration=tuple(
            (key, value)
            for key, value in action.configuration
            if key != "boundary_chart_required"
        ),
    )
    before = fingerprint_model_state(cross_width.child)

    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="did not mark boundary_chart_required",
    ):
        CrossWidthCausalBranchOperator(
            CausalGQAAttentionBranchOperator(),
            chart_set=cross_width.charts,
            recapture=cross_width.recapture(),
        ).prepare(stripped, cross_width.context, _evidence(cross_width, stripped))
    assert fingerprint_model_state(cross_width.child) == before


def test_a_branch_that_loses_to_writing_nothing_is_rolled_back(
    cross_width: _CrossWidthFold,
) -> None:
    """The endpoint fail-closed condition, exercised by tightening its bound."""

    cross_width.restore_blank()
    action = _branch_action(cross_width.plan, CAUSAL_ATTENTION_BRANCH_METHOD)
    before = fingerprint_model_state(cross_width.child)
    operator = CrossWidthCausalBranchOperator(
        CausalGQAAttentionBranchOperator(),
        chart_set=cross_width.charts,
        recapture=cross_width.recapture(),
        max_endpoint_relative_rmse=1e-12,
    )

    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="at or above the no-branch baseline",
    ):
        operator.prepare(action, cross_width.context, _evidence(cross_width, action))
    # The failing branch is undone exactly; a regression never survives.
    assert fingerprint_model_state(cross_width.child) == before
    for name, value in cross_width.child.state_dict().items():
        assert torch.equal(value, cross_width.blank_state[name])

    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="cannot exceed the no-branch baseline",
    ):
        CrossWidthCausalBranchOperator(
            CausalGQAAttentionBranchOperator(),
            chart_set=cross_width.charts,
            recapture=cross_width.recapture(),
            max_endpoint_relative_rmse=1.5,
        )


def test_the_cross_width_operator_refuses_a_missing_sequential_recapture(
    cross_width: _CrossWidthFold,
) -> None:
    """Design condition 6: no silent fallback to parent-coordinate inputs."""

    class _EmptyRecapture:
        recapture_calls = 0
        inputs_sha256 = None

        def __call__(self, child_model, boundary_ids):
            del child_model, boundary_ids
            return {}

    cross_width.restore_blank()
    action = _branch_action(cross_width.plan, CAUSAL_ATTENTION_BRANCH_METHOD)
    before = fingerprint_model_state(cross_width.child)

    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="A0025",
    ):
        CrossWidthCausalBranchOperator(
            CausalGQAAttentionBranchOperator(),
            chart_set=cross_width.charts,
            recapture=_EmptyRecapture(),
        ).prepare(action, cross_width.context, _evidence(cross_width, action))
    assert fingerprint_model_state(cross_width.child) == before


def test_the_token_head_operator_refuses_an_unshared_tokenizer_identity(
    cross_width: _CrossWidthFold,
) -> None:
    cross_width.restore_blank()
    action = next(
        value
        for value in cross_width.plan.actions
        if value.method_id == TOKEN_INTERFACE_BRIDGE_METHOD
    )
    foreign_child_graph = inspect_model_circuits(
        cross_width.child,
        model_name="child",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "not-the-target"}),
    )
    before = fingerprint_model_state(cross_width.child)

    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="shared tokenizer identity",
    ):
        CrossWidthTokenHeadInterfaceOperator(chart_set=cross_width.charts).prepare(
            action,
            FoldExecutionContext(
                parent_models={"parent": cross_width.parent},
                child_model=cross_width.child,
                parent_graphs={"parent": cross_width.parent_graph},
                child_graph=foreign_child_graph,
            ),
            _evidence(cross_width, action),
        )
    assert fingerprint_model_state(cross_width.child) == before


# ---------------------------------------------------------------------------
# Width-preserving control: identity charts must not change the arithmetic
# ---------------------------------------------------------------------------


class _StubRecapture:
    """Return a fixed child state per boundary, with no model execution."""

    def __init__(self, states) -> None:
        self.states = dict(states)
        self.recapture_calls = 0
        self.inputs_sha256 = canonical_sha256({"stub": "width-preserving-control"})

    def __call__(self, child_model, boundary_ids):
        del child_model
        self.recapture_calls += 1
        return {value: self.states[value] for value in boundary_ids}


@pytest.fixture(scope="module")
def width_preserving():
    """A same-width crossing whose layers still need the branch operators."""

    parent = _llama(20260801, PARENT_WIDTH)
    _confine(parent, CHILD_WIDTH)
    child = _llama(4711, PARENT_WIDTH, num_key_value_heads=2, intermediate_size=64)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    ids = _input_ids()
    evidence = capture_hf_causal_circuit_evidence(
        parent,
        {"input_ids": ids},
        parent_graph_fingerprint=parent_graph.fingerprint,
        metadata={"test": "width-preserving-identity-charts"},
    )
    plan = DynamicFoldPlanner(
        enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
    ).plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=fingerprint_model_state(child),
        evidence_fingerprint=evidence.cache_key,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
    )
    blank = {
        name: value.detach().clone() for name, value in child.state_dict().items()
    }
    charts = identity_residual_boundary_chart_set(evidence, child)
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=child,
        parent_graphs={"parent": parent_graph},
        child_graph=child_graph,
    )
    return parent, child, evidence, plan, charts, context, blank


def _restore(child, blank) -> None:
    with torch.no_grad():
        for name, value in child.state_dict().items():
            value.copy_(blank[name])


def _installed(child, action) -> dict[str, Any]:
    return {
        path: child.get_parameter(path).detach().clone()
        for path in action.target_parameter_paths
    }


@pytest.mark.parametrize(
    ("method_id", "operator_type", "branch", "write_path", "exact_paths"),
    (
        (
            CAUSAL_ATTENTION_BRANCH_METHOD,
            CausalGQAAttentionBranchOperator,
            "attention",
            "model.layers.0.self_attn.o_proj.weight",
            (
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ),
        ),
        (
            CAUSAL_GATED_MLP_BRANCH_METHOD,
            CausalGatedMLPBranchOperator,
            "gated_mlp",
            "model.layers.0.mlp.down_proj.weight",
            (
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            ),
        ),
    ),
)
def test_width_preserving_charts_change_nothing_but_the_endpoint_target(
    width_preserving,
    method_id: str,
    operator_type,
    branch: str,
    write_path: str,
    exact_paths: tuple[str, ...],
) -> None:
    """Identity charts must leave the whole solve alone, tensor by tensor.

    With ``E_b = E_{b+1} = I`` and the true width-preserving recapture
    ``hhat = h_b``, the chart transition and the upstream-error term are
    *exactly* zero, so every tensor whose fit does not read the endpoint target
    — the input norm and the Q/K/V (or gate/up) projections — is bit-identical
    to what the parent-coordinate operator installs.

    The one tensor that does read the endpoint target is the branch's write
    projection, and it cannot be bit-identical: the sequential target is
    ``h_{b+1} - h_b`` while the parent cached ``h_{b+1} = h_b + Delta`` in
    float32, so the two differ by the parent's own residual-sum rounding.  That
    is measured here against the float32 epsilon rather than waved at, and it
    is the exact price of implementing the design's sequential form instead of
    its diagnostic expansion.
    """

    _parent, child, evidence, plan, charts, context, blank = width_preserving
    action = _branch_action(plan, method_id)
    layer = evidence.layers[0]
    if branch == "attention":
        input_state = layer.block_input
        output_state = layer.post_attention_residual
        cached_delta = layer.attention_delta
        boundary = residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY)
    else:
        input_state = layer.post_attention_residual
        output_state = layer.block_output
        cached_delta = layer.mlp_delta
        boundary = residual_boundary_id(0, MLP_INPUT_BOUNDARY)

    _restore(child, blank)
    plain = operator_type().prepare(
        action,
        context,
        _evidence_for(evidence, action),
    )
    plain_report = dict(plain.report)
    plain_tensors = _installed(child, action)
    plain.rollback()

    _restore(child, blank)
    charted = CrossWidthCausalBranchOperator(
        operator_type(),
        chart_set=charts,
        recapture=_StubRecapture({boundary: input_state.double()}),
    ).prepare(action, context, _evidence_for(evidence, action))
    charted_report = dict(charted.report)
    charted_tensors = _installed(child, action)
    charted.rollback()

    assert set(plain_tensors) == set(charted_tensors) == set(exact_paths) | {
        write_path
    }
    for path in exact_paths:
        assert torch.equal(plain_tensors[path], charted_tensors[path]), path

    target = charted_report["sequential_branch_target"]
    assert target["fit_chart_transition_energy"] == 0.0
    assert target["fit_upstream_error_energy"] == 0.0

    # The endpoint target differs from the cached delta by the parent's own
    # float32 residual sum, and by nothing else.
    reconstructed = output_state.double() - input_state.double()
    drift = float((reconstructed - cached_delta.double()).abs().max())
    bound = 2.0 * float(torch.finfo(torch.float32).eps) * float(
        output_state.abs().max()
    )
    assert 0.0 < drift <= bound

    difference = float(
        (plain_tensors[write_path] - charted_tensors[write_path]).abs().max()
    )
    scale = float(plain_tensors[write_path].abs().max())
    assert 0.0 < difference < 1e-5 * scale

    assert plain_report["parent_residual_coordinate_endpoints"] is True
    assert charted_report["parent_residual_coordinate_endpoints"] is False
    assert charted_report["parent_residual_width"] == PARENT_WIDTH
    assert charted_report["child_residual_width"] == PARENT_WIDTH
    # The charted lane only adds provenance; it never drops a report key the
    # parent-coordinate lane already published.
    assert set(plain_report) <= set(charted_report)


def _evidence_for(evidence, action) -> FoldEvidence:
    return FoldEvidence(
        evidence.cache_key,
        evidence,
        channels=action.evidence_requirements,
    )
