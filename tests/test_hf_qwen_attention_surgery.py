from __future__ import annotations

import hashlib
import json

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.adaptive_surgery_graph import AnchorSpec  # noqa: E402
from neural_forge.hf_qwen_attention_surgery import (  # noqa: E402
    Qwen3AttentionEvidenceRefresh,
    Qwen3AttentionProjection,
    Qwen3AttentionSurgeryBridge,
    Qwen3AttentionSurgeryFeature,
    Qwen3AttentionSurgeryShapeError,
    Qwen3AttentionSurgeryStateError,
    Qwen3AttentionSurgeryTargetError,
    qwen3_attention_group_node_id,
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    load_qwen3_codebook_checkpoint,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
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


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_contract(runtime: torch.nn.Module) -> dict[str, object]:
    tokenizer = _Tokenizer()
    return {
        "tokenizer_name_or_path": "identity-test-tokenizer",
        "vocab_size": len(tokenizer),
        "vocab_sha256": _canonical_sha256(tokenizer.get_vocab()),
        "special_token_ids": {
            name: getattr(runtime.config, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "parent_token_rows_sha256": _canonical_sha256(
            [int(value) for value in runtime.parent_token_rows.tolist()]
        ),
    }


def _runtime(
    *,
    seed: int = 1717,
    attention_bias: bool = False,
    group_permutations: torch.Tensor | None = None,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=attention_bias,
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
        attention_prototypes_per_group=4,
        neuron_prototypes_per_neuron=1,
        group_permutations=group_permutations,
    )


def _features(
    runtime: torch.nn.Module,
    *,
    projected: tuple[int, int] | None = None,
    exact: tuple[int, int] | None = None,
) -> tuple[Qwen3AttentionSurgeryFeature, ...]:
    snapshot = qwen3_attention_runtime_snapshot(runtime)
    groups = snapshot.key_value_heads
    result = []
    for source in snapshot.prototype_source_indices:
        layer, group = divmod(source, groups)
        pair = (layer, group)
        value = 0.0 if pair == projected else 0.0001 if pair == exact else 20.0 + source
        anchor = (
            AnchorSpec.projected(1e-8)
            if pair == projected
            else AnchorSpec.pinned()
            if pair == exact
            else None
        )
        result.append(
            Qwen3AttentionSurgeryFeature(
                layer,
                group,
                weight=[value],
                activation=[value, value + 0.05],
                jvp=[value + 0.1, value + 0.15],
                output=[value + 0.2],
                importance=1000.0 if pair == exact else 0.01,
                anchor=anchor,
            )
        )
    return tuple(result)


def _node(pair: tuple[int, int]):
    return qwen3_attention_group_node_id(*pair)


def test_nonlocal_cross_group_pair_installs_one_coupled_qkvo_redirect() -> None:
    runtime = _runtime()
    source = (3, 1)
    target = (0, 0)
    projection = Qwen3AttentionProjection(
        source,
        target,
        q=1.25,
        k=0.8,
        v=1.5,
        o=0.75,
        activation_residual=0.01,
        jvp_residual=0.02,
    )
    bridge = Qwen3AttentionSurgeryBridge(
        runtime,
        _features(runtime, projected=source, exact=target),
        projections=(projection,),
    )
    before = qwen3_attention_runtime_snapshot(runtime)
    scored = bridge.propose_pair(_node(source), _node(target))

    assert scored.candidate.resulting_medoid == _node(target)
    assert scored.scale_allocation_bytes > 0
    assert scored.physical_bytes_saved == (
        scored.prototype_bytes_saved - scored.scale_allocation_bytes
    )
    result = bridge.run(
        before.tensor_bytes - scored.physical_bytes_saved,
        candidate_selector=lambda _context: {
            "pairs": ((_node(source), _node(target)),)
        },
        max_attempts=1,
    )

    after = qwen3_attention_runtime_snapshot(runtime)
    target_source = target[0] * after.key_value_heads + target[1]
    assert result.target_reached
    assert after.prototype_count == before.prototype_count - 1
    assert after.assignment_source_indices[source[0]][source[1]] == target_source
    registry = runtime.attention_codebook.registry
    selected_slot = int(registry.assignments[source[0], source[1]])
    assert int(registry.prototype_source_indices[selected_slot]) == target_source
    torch.testing.assert_close(
        registry.role_scales[source[0], source[1]],
        torch.tensor(projection.values, dtype=registry.role_scales.dtype),
    )
    assert runtime.model.layers[source[0]].self_attn.q_proj.registry is registry
    assert runtime.model.layers[source[0]].self_attn.k_proj.registry is registry
    assert runtime.model.layers[source[0]].self_attn.v_proj.registry is registry
    assert runtime.model.layers[source[0]].self_attn.o_proj.registry is registry
    assert result.transactions[0].projections == (projection,)
    aggregate = qwen3_codebook_parameter_report(runtime)
    assert aggregate.attention_prototypes_per_group == 0
    assert aggregate.attention_unique_parameter_count == (
        runtime.attention_codebook.report.unique_learned_parameters
    )
    assert aggregate.attention_index_bytes == (
        runtime.attention_codebook.report.index_buffer_bytes
    )
    with torch.inference_mode():
        logits = runtime(torch.tensor([[1, 4, 7, 2]])).logits
    assert logits.shape == (1, 4, 12)
    assert bool(torch.isfinite(logits).all())


def test_projection_evidence_and_scale_cost_change_candidate_score() -> None:
    source = (3, 1)
    target = (0, 0)
    good_runtime = _runtime(seed=1818)
    bad_runtime = _runtime(seed=1818)
    features_good = _features(good_runtime, projected=source, exact=target)
    features_bad = _features(bad_runtime, projected=source, exact=target)
    good_projection = Qwen3AttentionProjection(
        source,
        target,
        1.0,
        1.0,
        1.0,
        1.0,
        activation_residual=0.0,
        jvp_residual=0.0,
    )
    bad_projection = Qwen3AttentionProjection(
        source,
        target,
        9.0,
        -7.0,
        6.0,
        -5.0,
        activation_residual=12.0,
        jvp_residual=8.0,
    )
    good = Qwen3AttentionSurgeryBridge(
        good_runtime, features_good, projections=(good_projection,)
    ).propose_pair(source, target)
    bad = Qwen3AttentionSurgeryBridge(
        bad_runtime, features_bad, projections=(bad_projection,)
    ).propose_pair(source, target)

    assert good.projection_damage < bad.projection_damage
    assert good.total_score > bad.total_score
    assert good.scale_allocation_bytes == 0
    assert bad.scale_allocation_bytes > 0
    assert good.physical_bytes_saved > bad.physical_bytes_saved


def test_mid_install_failure_restores_tensors_graph_history_epochs_and_diagnostics(
    monkeypatch,
) -> None:
    import neural_forge.hf_qwen_attention_surgery as surgery

    runtime = _runtime(seed=1919)
    source = (3, 1)
    target = (0, 0)
    bridge = Qwen3AttentionSurgeryBridge(
        runtime, _features(runtime, projected=source, exact=target)
    )
    before_codebook = runtime.attention_codebook
    before_runtime = qwen3_attention_runtime_snapshot(runtime)
    before_report = qwen3_codebook_parameter_report(runtime)
    before_graph = bridge.graph.snapshot
    before_graph_work = bridge.graph.work_diagnostics
    before_diagnostics = bridge.diagnostics
    original = surgery._wire_projection

    def fail_during_v(layer, name, value):
        if name == "v_proj":
            raise RuntimeError("mid-commit projection failure")
        original(layer, name, value)

    monkeypatch.setattr(surgery, "_wire_projection", fail_during_v)
    with pytest.raises(RuntimeError, match="mid-commit projection failure"):
        bridge.run(
            before_runtime.tensor_bytes - 1,
            candidate_selector=lambda _context: {
                "pairs": ((_node(source), _node(target)),)
            },
        )

    assert runtime.attention_codebook is before_codebook
    assert qwen3_attention_runtime_snapshot(runtime) == before_runtime
    assert qwen3_codebook_parameter_report(runtime) == before_report
    assert bridge.graph.snapshot == before_graph
    assert bridge.graph.work_diagnostics == before_graph_work
    assert bridge.diagnostics == before_diagnostics
    assert bridge.transactions == ()
    assert bridge.evidence_epoch == 0


def test_exact_anchor_bundle_survives_and_refresh_cannot_change_anchor_contract() -> (
    None
):
    source = (3, 1)
    target = (0, 0)
    runtime = _runtime(seed=2020)
    bridge = Qwen3AttentionSurgeryBridge(
        runtime, _features(runtime, projected=source, exact=target)
    )
    before = qwen3_attention_runtime_snapshot(runtime)
    scored = bridge.propose_pair(source, target)
    anchor_hash = before.bundle_hash(target)
    bridge.run(
        before.tensor_bytes - scored.physical_bytes_saved,
        candidate_selector=lambda _context: {"pairs": ((source, target),)},
    )
    assert qwen3_attention_runtime_snapshot(runtime).bundle_hash(target) == anchor_hash

    failing_runtime = _runtime(seed=2121)
    original_features = _features(failing_runtime, projected=source, exact=target)
    failing = Qwen3AttentionSurgeryBridge(failing_runtime, original_features)
    failing_before = qwen3_attention_runtime_snapshot(failing_runtime)

    def invalid_refresh(context):
        active = set(context.active_node_ids)
        values = []
        for feature in original_features:
            if feature.node_id not in active:
                continue
            values.append(
                Qwen3AttentionSurgeryFeature(
                    feature.layer_index,
                    feature.group_index,
                    weight=feature.weight_features,
                    activation=feature.activation_features,
                    jvp=feature.jvp_features,
                    output=feature.output_features,
                    importance=(
                        feature.importance + 1.0
                        if feature.node_id == _node(target)
                        else feature.importance
                    ),
                    anchor=feature.anchor,
                )
            )
        return Qwen3AttentionEvidenceRefresh(values)

    with pytest.raises(
        Qwen3AttentionSurgeryShapeError, match="anchors and importance are immutable"
    ):
        failing.run(
            failing_before.tensor_bytes - 1,
            refresh_features=invalid_refresh,
            candidate_selector=lambda _context: {"pairs": ((source, target),)},
        )
    assert qwen3_attention_runtime_snapshot(failing_runtime) == failing_before
    assert failing.transactions == ()
    assert failing.evidence_epoch == 0
    assert failing.diagnostics.feature_refreshes == 0


def test_rejected_four_candidate_batch_repeatedly_bisects_to_singletons() -> None:
    runtime = _runtime(seed=2222)
    features = _features(runtime)
    bridge = Qwen3AttentionSurgeryBridge(
        runtime,
        features,
        max_nonconflicting_batch_size=4,
        transaction_history_limit=3,
    )
    nodes = tuple(feature.node_id for feature in features)
    pairs = tuple((nodes[index], nodes[index + 1]) for index in range(0, 8, 2))
    candidates = tuple(bridge.graph.propose_candidate_for_pair(pair) for pair in pairs)
    before = qwen3_attention_runtime_snapshot(runtime)
    target = before.tensor_bytes - bridge._batch_physical_bytes_saved(candidates)
    batch_sizes = []

    result = bridge.run(
        target,
        gate=lambda context: (
            batch_sizes.append(len(context.graph_transaction.candidates))
            or len(context.graph_transaction.candidates) == 1
        ),
        candidate_selector=lambda _context: {"pairs": pairs},
        max_attempts=7,
    )

    assert batch_sizes == [4, 2, 1, 1, 2, 1, 1]
    assert result.target_reached
    assert result.diagnostics.accepted_transactions == 4
    assert result.diagnostics.rejected_transactions == 3
    assert result.diagnostics.bisections == 3
    assert result.diagnostics.evidence_epoch == 4
    assert len(result.transactions) == 3
    assert not hasattr(result.transactions[0].parent_runtime, "assignments")


def test_evidence_refresh_hook_runs_only_after_an_accepted_gate() -> None:
    source = (3, 1)
    target = (0, 0)
    rejected_runtime = _runtime(seed=2270)
    rejected_features = _features(rejected_runtime, projected=source, exact=target)
    rejected = Qwen3AttentionSurgeryBridge(rejected_runtime, rejected_features)
    rejected_before = qwen3_attention_runtime_snapshot(rejected_runtime)
    rejected_score = rejected.propose_pair(source, target)
    rejected_refreshes = []
    with pytest.raises(Qwen3AttentionSurgeryTargetError, match="max_attempts=1"):
        rejected.run(
            rejected_before.tensor_bytes - rejected_score.physical_bytes_saved,
            gate=lambda _context: False,
            refresh_features=lambda context: rejected_refreshes.append(context),
            candidate_selector=lambda _context: {"pairs": ((source, target),)},
            max_attempts=1,
        )
    assert rejected_refreshes == []
    assert rejected.evidence_epoch == 0

    accepted_runtime = _runtime(seed=2280)
    accepted_features = _features(accepted_runtime, projected=source, exact=target)
    accepted = Qwen3AttentionSurgeryBridge(accepted_runtime, accepted_features)
    accepted_before = qwen3_attention_runtime_snapshot(accepted_runtime)
    accepted_score = accepted.propose_pair(source, target)
    accepted_refreshes = []

    def refresh(context):
        accepted_refreshes.append(context)
        active = set(context.active_node_ids)
        return tuple(
            feature for feature in accepted_features if feature.node_id in active
        )

    result = accepted.run(
        accepted_before.tensor_bytes - accepted_score.physical_bytes_saved,
        gate=lambda _context: True,
        refresh_features=refresh,
        candidate_selector=lambda _context: {"pairs": ((source, target),)},
        max_attempts=1,
    )
    assert result.target_reached
    assert len(accepted_refreshes) == 1
    assert accepted.evidence_epoch == 1
    assert accepted.diagnostics.feature_refreshes == 1


def test_partial_projection_and_unmeasured_improvement_claim_fail_closed() -> None:
    source = (3, 1)
    target = (0, 0)
    with pytest.raises(Qwen3AttentionSurgeryShapeError, match="Partial Q/K/V/O"):
        Qwen3AttentionSurgeryBridge(
            _runtime(seed=2323),
            _features(_runtime(seed=2424)),
            projections={(source, target): {"q": 1.0, "k": 1.0, "v": 1.0}},
        )

    runtime = _runtime(seed=2525)
    bridge = Qwen3AttentionSurgeryBridge(
        runtime, _features(runtime, projected=source, exact=target)
    )
    before = qwen3_attention_runtime_snapshot(runtime)
    before_report = qwen3_codebook_parameter_report(runtime)
    with pytest.raises(
        Qwen3AttentionSurgeryStateError, match="improvement claim requires measured"
    ):
        bridge.run(
            before.tensor_bytes - 1,
            gate=lambda _context: {"accepted": True, "improved": True},
            candidate_selector=lambda _context: {"pairs": ((source, target),)},
        )
    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert qwen3_codebook_parameter_report(runtime) == before_report
    assert bridge.transactions == ()


def test_behavioral_improvement_claim_binds_contract_and_parent_hash() -> None:
    runtime = _runtime(seed=2575)
    source = (3, 1)
    target = (0, 0)
    bridge = Qwen3AttentionSurgeryBridge(
        runtime, _features(runtime, projected=source, exact=target)
    )
    before = qwen3_attention_runtime_snapshot(runtime)

    def gate(context):
        return {
            "accepted": True,
            "improved": True,
            "parent_score": 0.5,
            "candidate_score": 0.6,
            "higher_is_better": True,
            "evaluation_contract_sha256": "a" * 64,
            "parent_runtime_sha256": context.parent_runtime_snapshot.registry_sha256,
        }

    result = bridge.run(
        before.tensor_bytes - 1,
        gate=gate,
        candidate_selector=lambda _context: {"pairs": ((source, target),)},
        max_attempts=1,
    )
    assert result.transactions[0].measured_improvement_claim is True


def test_gate_cannot_mutate_only_one_projection_role() -> None:
    runtime = _runtime(seed=2626)
    source = (3, 1)
    target = (0, 0)
    bridge = Qwen3AttentionSurgeryBridge(
        runtime, _features(runtime, projected=source, exact=target)
    )
    before = qwen3_attention_runtime_snapshot(runtime)

    def mutate_q_only(context):
        context.runtime.attention_codebook.registry.prototype_q_weight[0].add_(1.0)
        return True

    with pytest.raises(
        Qwen3AttentionSurgeryStateError, match="Fatal out-of-contract mutation"
    ):
        bridge.run(
            before.tensor_bytes - 1,
            gate=mutate_q_only,
            candidate_selector=lambda _context: {"pairs": ((source, target),)},
        )
    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert bridge.transactions == ()


def test_automatic_qk_fit_is_post_norm_bias_aware() -> None:
    runtime = _runtime(seed=2727, attention_bias=True)
    bridge = Qwen3AttentionSurgeryBridge(runtime, _features(runtime))
    scored = bridge.propose_pair((3, 1), (0, 0))

    assert scored.projection.q == 1.0
    assert scored.projection.k == 1.0
    assert scored.projection.bias_included is True
    assert scored.projection.qk_normalized_residual is not None
    assert scored.projection.fit_space.startswith("post-qk-rmsnorm")


def test_nonidentity_permutation_audit_records_canonical_and_physical_groups() -> None:
    permutations = torch.tensor(
        [[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int64
    )
    runtime = _runtime(seed=2828, group_permutations=permutations)
    snapshot = qwen3_attention_runtime_snapshot(runtime)

    assert len(snapshot.prototype_coordinates) == snapshot.prototype_count
    swapped = [
        item
        for item in snapshot.prototype_coordinates
        if item.canonical_group != item.physical_group
    ]
    assert swapped
    for coordinate in snapshot.prototype_coordinates:
        assert permutations[
            coordinate.source_layer, coordinate.canonical_group
        ].item() == coordinate.physical_group
        assert coordinate.source_flat_index == (
            coordinate.source_layer * snapshot.key_value_heads
            + coordinate.physical_group
        )


def test_ragged_qwen_checkpoint_roundtrip_is_exact_and_standalone(tmp_path) -> None:
    runtime = _runtime(seed=2929)
    source = (3, 1)
    target = (0, 0)
    projection = Qwen3AttentionProjection(source, target, 1.0, 1.0, 1.0, 1.0)
    bridge = Qwen3AttentionSurgeryBridge(
        runtime,
        _features(runtime, projected=source, exact=target),
        projections=(projection,),
    )
    before = qwen3_attention_runtime_snapshot(runtime)
    score = bridge.propose_pair(source, target)
    bridge.run(
        before.tensor_bytes - score.physical_bytes_saved,
        candidate_selector=lambda _context: {"pairs": ((source, target),)},
        max_attempts=1,
    )
    expected_snapshot = qwen3_attention_runtime_snapshot(runtime)
    expected_report = qwen3_codebook_parameter_report(runtime)
    input_ids = torch.tensor([[1, 4, 7, 2]])
    with torch.inference_mode():
        expected_logits = runtime(input_ids, use_cache=False).logits.clone()
        expected_generation = runtime.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=3,
            use_cache=False,
        )
    checkpoint = tmp_path / "ragged-qwen"
    manifest = save_qwen3_codebook_checkpoint(
        runtime,
        checkpoint,
        tokenizer_contract=_checkpoint_contract(runtime),
    )
    restored = load_qwen3_codebook_checkpoint(checkpoint)
    restored_snapshot = qwen3_attention_runtime_snapshot(restored)

    assert int(restored.attention_codebook.registry._registry_layout) == 1
    assert restored_snapshot == expected_snapshot
    assert qwen3_codebook_parameter_report(restored) == expected_report
    assert manifest["assignment_hashes"]
    assert manifest["prototype_source_hashes"]
    with torch.inference_mode():
        actual_logits = restored(input_ids, use_cache=False).logits
        actual_generation = restored.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=3,
            use_cache=False,
        )
    torch.testing.assert_close(actual_logits, expected_logits, atol=0.0, rtol=0.0)
    assert torch.equal(actual_generation, expected_generation)
    work = runtime.attention_codebook.registry.compaction_work
    assert work.full_registry_clone_count == 0
    assert work.copied_tensor_bytes <= work.result_tensor_bytes
    assert work.peak_live_tensor_bytes_bound == (
        work.source_tensor_bytes + work.result_tensor_bytes + work.maximum_staging_bytes
    )
