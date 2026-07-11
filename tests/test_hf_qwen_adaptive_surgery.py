from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.adaptive_surgery_graph import AnchorSpec  # noqa: E402
from neural_forge.hf_qwen_adaptive_surgery import (  # noqa: E402
    Qwen3AdaptiveSurgeryBridge,
    Qwen3AdaptiveSurgeryFeature,
    Qwen3AdaptiveSurgeryStateError,
    Qwen3AdaptiveSurgeryTargetError,
    Qwen3ProjectionCoefficients,
    Qwen3SurgeryEvidenceRefresh,
    qwen3_adaptive_features_from_channels,
    qwen3_neuron_node_id,
    qwen3_runtime_assignment_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.tokens = tuple(f"token-{index}" for index in range(12))

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _runtime(*, neuron_use_scales: bool = False) -> torch.nn.Module:
    torch.manual_seed(914)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    tokenizer = _Tokenizer()
    return prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
        neuron_use_scales=neuron_use_scales,
    )


def test_optional_jvp_output_and_gqa_context_are_bounded_bridge_evidence() -> None:
    node = qwen3_neuron_node_id(1, 4)
    feature = Qwen3AdaptiveSurgeryFeature(
        1, 4, activation=[1.0, 2.0], gqa_features=[3.0]
    )
    from_channels = qwen3_adaptive_features_from_channels(
        activation_features={node: [1.0, 2.0]},
        gqa_features={node: [3.0]},
    )

    assert feature.rollout_jvp_features == (0.0,)
    assert feature.output_dual_features == (0.0,)
    assert feature.gqa_features == (3.0,)
    assert from_channels == (feature,)


def _features(
    runtime: torch.nn.Module,
    *,
    projected_source: tuple[int, int] | None = None,
    preferred_target: tuple[int, int] | None = None,
) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
    registry = runtime.neuron_codebook.registry
    width = registry.intermediate_width
    features = []
    for source in registry.prototype_source_indices.detach().to("cpu").tolist():
        layer, neuron = divmod(int(source), width)
        pair = (layer, neuron)
        value = (
            0.0
            if pair == projected_source
            else 0.001
            if pair == preferred_target
            else 100.0 + source
        )
        features.append(
            Qwen3AdaptiveSurgeryFeature(
                layer,
                neuron,
                weight=[value],
                activation=[value, value + 0.1],
                jacobian=[value + 0.2],
                dual=[value + 0.3],
                domain_features=[value + 0.4],
                importance=1_000.0 if pair == preferred_target else 0.01,
                anchor=(AnchorSpec.projected(1e-8) if pair == projected_source else None),
            )
        )
    return tuple(features)


def test_nonadjacent_qwen_nodes_score_on_demand_and_apply_role_projection() -> None:
    runtime = _runtime(neuron_use_scales=True)
    source = (1, 5)
    target = (0, 0)
    features = _features(
        runtime, projected_source=source, preferred_target=target
    )
    projection = Qwen3ProjectionCoefficients(source, target, 2.0, 3.0, 4.0)
    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime, features, projection_coefficients=(projection,)
    )

    source_id = qwen3_neuron_node_id(*source)
    target_id = qwen3_neuron_node_id(*target)
    candidate = bridge.graph.propose_candidate_for_pair(source_id, target_id)
    assert candidate.resulting_medoid == qwen3_neuron_node_id(*target)
    assert bridge.graph.score_pair(source_id, target_id).parameter_bytes_saved > 0

    before_codebook = runtime.neuron_codebook
    before = qwen3_runtime_assignment_snapshot(runtime)
    assert bridge._marginal_bytes_saved((candidate,)) == (
        candidate.score.parameter_bytes_saved
    )

    def select(context):
        assert context.projection_coefficients == (projection,)
        assert context.projection_role_scales
        return {"pairs": ((source_id, target_id),)}

    result = bridge.run(
        before.tensor_bytes - 1,
        max_attempts=1,
        candidate_selector=select,
    )
    assert result.target_reached

    after = qwen3_runtime_assignment_snapshot(runtime)
    assert runtime.neuron_codebook is not before_codebook
    assert after.prototype_count == before.prototype_count - 1
    assert after.assignment_source_indices[source[0]][source[1]] == (
        target[0] * after.intermediate_width + target[1]
    )
    torch.testing.assert_close(
        runtime.neuron_codebook.registry.neuron_scales[source[0], source[1]],
        torch.tensor([2.0, 3.0, 4.0]),
    )
    assert bridge.transactions[0].projection_coefficients == (projection,)
    assert all(parameter.grad is None for parameter in runtime.neuron_codebook.parameters())
    work = runtime.neuron_codebook.registry.compaction_work
    assert work.full_registry_clone_count == 0
    assert work.copied_tensor_bytes <= work.result_tensor_bytes
    assert work.peak_live_tensor_bytes_bound == (
        work.source_tensor_bytes + work.result_tensor_bytes + work.maximum_staging_bytes
    )


def test_projected_scale_setup_must_have_positive_amortized_physical_bytes() -> None:
    runtime = _runtime()
    source = (1, 5)
    target = (0, 0)
    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime,
        _features(runtime, projected_source=source, preferred_target=target),
        max_nonconflicting_batch_size=1,
    )
    candidate = bridge.graph.propose_candidate_for_pair(
        qwen3_neuron_node_id(*source), qwen3_neuron_node_id(*target)
    )
    before = qwen3_runtime_assignment_snapshot(runtime)
    source_id = qwen3_neuron_node_id(*source)
    target_id = qwen3_neuron_node_id(*target)

    assert bridge._marginal_bytes_saved((candidate,)) <= 0
    with pytest.raises(Qwen3AdaptiveSurgeryTargetError, match="No Qwen3 surgery candidate"):
        bridge.run(
            before.tensor_bytes - 1,
            candidate_selector=lambda _context: {"pairs": ((source_id, target_id),)},
        )
    assert qwen3_runtime_assignment_snapshot(runtime) == before


def test_automatic_projection_uses_post_swiglu_and_output_evidence() -> None:
    runtime = _runtime(neuron_use_scales=True)
    source = (1, 5)
    target = (0, 0)
    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime,
        _features(runtime, projected_source=source, preferred_target=target),
    )
    projection = bridge.projection_for_pair(source, target)

    assert projection.gate == 1.0
    assert projection.post_swiglu_residual is not None
    assert projection.output_residual is not None
    assert projection.fit_space == "post-swiglu-activation-and-output-residual"


def test_neuron_install_synchronizes_report_and_binds_improvement_evidence() -> None:
    runtime = _runtime(neuron_use_scales=True)
    source = (1, 5)
    target = (0, 0)
    projection = Qwen3ProjectionCoefficients(source, target, 1.0, 1.0, 1.0)
    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime,
        _features(runtime, projected_source=source, preferred_target=target),
        projection_coefficients=(projection,),
    )
    before = qwen3_runtime_assignment_snapshot(runtime)

    def gate(context):
        return {
            "accepted": True,
            "improved": True,
            "parent_score": 1.0,
            "candidate_score": 0.9,
            "higher_is_better": False,
            "evaluation_contract_sha256": "b" * 64,
            "parent_runtime_sha256": context.parent_runtime_snapshot.registry_sha256,
        }

    result = bridge.run(
        before.tensor_bytes - 1,
        gate=gate,
        candidate_selector=lambda _context: {
            "pairs": (
                (qwen3_neuron_node_id(*source), qwen3_neuron_node_id(*target)),
            )
        },
        max_attempts=1,
    )
    report = qwen3_codebook_parameter_report(runtime)
    assert result.transactions[0].measured_improvement_claim is True
    assert report.neuron_unique_parameter_count == (
        runtime.neuron_codebook.report.unique_learned_parameters
    )
    assert report.neuron_index_bytes == runtime.neuron_codebook.report.index_buffer_bytes


def test_rejected_rollout_restores_the_exact_installed_codebook() -> None:
    runtime = _runtime()
    bridge = Qwen3AdaptiveSurgeryBridge(runtime, _features(runtime))
    before_codebook = runtime.neuron_codebook
    before = qwen3_runtime_assignment_snapshot(runtime)

    with pytest.raises(Qwen3AdaptiveSurgeryTargetError, match="max_attempts=1"):
        bridge.run(
            before.tensor_bytes - 1,
            gate=lambda _context: {"accepted": False, "reason": "holdout failed"},
            max_attempts=1,
        )

    assert runtime.neuron_codebook is before_codebook
    assert qwen3_runtime_assignment_snapshot(runtime) == before
    transaction = bridge.transactions[0]
    assert transaction.accepted is False
    assert transaction.rollback_exact is True
    assert {action.kind for action in transaction.actions} == {"skip"}


def test_accepted_surgery_recomputes_bounded_real_evidence_sequentially() -> None:
    runtime = _runtime()
    initial_features = _features(runtime)
    bridge = Qwen3AdaptiveSurgeryBridge(runtime, initial_features)
    initial = qwen3_runtime_assignment_snapshot(runtime)
    recomputations = []

    def recompute(context):
        recomputations.append(context)
        assert tuple(context.active_node_ids) == tuple(sorted(context.active_node_ids))
        active = set(context.active_node_ids)
        refreshed = tuple(
            item
            for item in initial_features
            if item.node_id in active
        )
        return Qwen3SurgeryEvidenceRefresh(refreshed)

    result = bridge.run(
        initial.tensor_bytes - 200,
        recompute_features=recompute,
        max_attempts=4,
    )

    assert result.target_reached
    assert result.feature_recomputations == len(recomputations)
    assert result.feature_recomputations >= 1
    assert result.zero_gradient
    assert result.final_runtime_snapshot.deployable_neuron_codebook_bytes <= (
        result.target_deployable_neuron_codebook_bytes
    )
    assert len(bridge.features) == result.final_runtime_snapshot.prototype_count


def test_batched_gate_saves_multiple_prototypes_in_one_runtime_install() -> None:
    runtime = _runtime()
    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime, _features(runtime), max_nonconflicting_batch_size=3
    )
    before = qwen3_runtime_assignment_snapshot(runtime)
    batch = bridge.graph.propose_nonconflicting_batch(3)
    assert len(batch) >= 2
    target = before.tensor_bytes - sum(
        item.score.parameter_bytes_saved for item in batch[:2]
    )
    gate_sizes = []
    result = bridge.run(
        target,
        gate=lambda context: gate_sizes.append(len(context.graph_transaction.candidates)) or True,
        max_attempts=1,
    )

    assert gate_sizes == [2]
    assert result.accepted_transactions == 1
    assert result.transactions[0].proposed_runtime_snapshot.prototype_count == (
        before.prototype_count - 2
    )


def test_rejected_batch_bisects_and_keeps_accepted_singletons_available() -> None:
    runtime = _runtime()
    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime, _features(runtime), max_nonconflicting_batch_size=2
    )
    before = qwen3_runtime_assignment_snapshot(runtime)
    candidates = bridge.graph.propose_nonconflicting_batch(2)
    assert len(candidates) == 2
    target = before.tensor_bytes - sum(
        item.score.parameter_bytes_saved for item in candidates
    )
    batch_sizes = []
    result = bridge.run(
        target,
        gate=lambda context: batch_sizes.append(len(context.graph_transaction.candidates))
        or len(context.graph_transaction.candidates) == 1,
        max_attempts=3,
    )

    assert batch_sizes == [2, 1, 1]
    assert result.rejected_transactions == 1
    assert result.accepted_transactions == 2
    assert result.target_reached


def test_recompute_failure_is_atomic_and_preserves_graph_history() -> None:
    runtime = _runtime()
    bridge = Qwen3AdaptiveSurgeryBridge(runtime, _features(runtime))
    before_runtime = qwen3_runtime_assignment_snapshot(runtime)
    before_graph = bridge.graph.snapshot
    before_work = bridge.graph.work_diagnostics

    with pytest.raises(RuntimeError, match="refresh failed"):
        bridge.run(
            before_runtime.tensor_bytes - 1,
            recompute_features=lambda _context: (_ for _ in ()).throw(
                RuntimeError("refresh failed")
            ),
        )

    assert qwen3_runtime_assignment_snapshot(runtime) == before_runtime
    assert bridge.graph.snapshot == before_graph
    assert bridge.graph.work_diagnostics == before_work
    assert bridge._feature_recomputations == 0
    assert bridge.transactions == ()


@pytest.mark.parametrize("phase", ("selector", "gate", "recompute", "rebuild"))
def test_callback_exceptions_restore_runtime_and_graph_state(
    phase: str, monkeypatch
) -> None:
    runtime = _runtime()
    bridge = Qwen3AdaptiveSurgeryBridge(runtime, _features(runtime))
    before_runtime = qwen3_runtime_assignment_snapshot(runtime)
    before_graph = bridge.graph.snapshot
    before_work = bridge.graph.work_diagnostics
    before_bridge_counters = (
        bridge._attempt_count,
        bridge._accepted_transaction_count,
        bridge._rejected_transaction_count,
        bridge._feature_recomputations,
        bridge._graph_rounds,
    )
    target = before_runtime.tensor_bytes - 1
    if phase == "rebuild":
        saving = bridge.graph.propose_candidates(limit=1)[0].score.parameter_bytes_saved
        target = before_runtime.tensor_bytes - 2 * saving
        before_work = bridge.graph.work_diagnostics
        monkeypatch.setattr(
            bridge,
            "_build_graph",
            lambda _features: (_ for _ in ()).throw(RuntimeError("rebuild failed")),
        )
    kwargs = {
        "candidate_selector": (
            (lambda _context: (_ for _ in ()).throw(RuntimeError("selector failed")))
            if phase == "selector"
            else None
        ),
        "gate": (
            (lambda _context: (_ for _ in ()).throw(RuntimeError("gate failed")))
            if phase == "gate"
            else None
        ),
        "recompute_features": (
            (lambda _context: (_ for _ in ()).throw(RuntimeError("recompute failed")))
            if phase == "recompute"
            else None
        ),
    }

    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        bridge.run(target, **kwargs)

    assert qwen3_runtime_assignment_snapshot(runtime) == before_runtime
    assert bridge.graph.snapshot == before_graph
    assert bridge.graph.work_diagnostics == before_work
    assert (
        bridge._attempt_count,
        bridge._accepted_transaction_count,
        bridge._rejected_transaction_count,
        bridge._feature_recomputations,
        bridge._graph_rounds,
    ) == before_bridge_counters
    assert bridge.transactions == ()


def test_out_of_contract_callback_mutation_is_fatal_and_rollback_is_scoped() -> None:
    runtime = _runtime()
    bridge = Qwen3AdaptiveSurgeryBridge(runtime, _features(runtime))
    before_registry = qwen3_runtime_assignment_snapshot(runtime)
    before_embedding = runtime.get_input_embeddings().weight.detach().clone()

    def mutate_unowned_tensor(context):
        context.runtime.get_input_embeddings().weight[0, 0].add_(1.0)
        return True

    with pytest.raises(
        Qwen3AdaptiveSurgeryStateError, match="Fatal out-of-contract mutation"
    ):
        bridge.run(before_registry.tensor_bytes - 1, gate=mutate_unowned_tensor)

    assert qwen3_runtime_assignment_snapshot(runtime) == before_registry
    assert not torch.equal(runtime.get_input_embeddings().weight, before_embedding)
    assert bridge.transactions == ()


def test_refresh_cannot_change_anchor_or_importance() -> None:
    runtime = _runtime()
    original = _features(runtime)
    bridge = Qwen3AdaptiveSurgeryBridge(runtime, original)
    before = qwen3_runtime_assignment_snapshot(runtime)

    def refresh(context):
        active = set(context.active_node_ids)
        first = next(item for item in original if item.node_id in active)
        changed = Qwen3AdaptiveSurgeryFeature(
            first.layer_index,
            first.neuron_index,
            weight=first.weight_features,
            activation=first.activation_features,
            jacobian=first.jacobian_features,
            dual=first.dual_features,
            gqa_features=first.gqa_features,
            importance=first.importance + 1.0,
            anchor=first.anchor,
        )
        return tuple(
            changed if item.node_id == first.node_id else item
            for item in original
            if item.node_id in active
        )

    with pytest.raises(Exception, match="anchors and importance are immutable"):
        bridge.run(before.tensor_bytes - 1, recompute_features=refresh)
    assert qwen3_runtime_assignment_snapshot(runtime) == before


def test_transaction_history_is_compact_and_bounded() -> None:
    runtime = _runtime()
    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime, _features(runtime), transaction_history_limit=1
    )
    before = qwen3_runtime_assignment_snapshot(runtime)
    savings = bridge.graph.propose_candidates(limit=1)[0].score.parameter_bytes_saved
    bridge.run(before.tensor_bytes - 2 * savings, max_attempts=2)

    assert len(bridge.transactions) == 1
    audit = bridge.transactions[0].parent_runtime_snapshot
    assert audit.tensor_bytes > 0
    assert not hasattr(audit, "assignments")
