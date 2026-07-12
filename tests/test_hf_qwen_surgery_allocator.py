from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.adaptive_surgery_graph import AnchorSpec  # noqa: E402
from neural_forge.hf_qwen_adaptive_surgery import (  # noqa: E402
    Qwen3AdaptiveSurgeryBridge,
    Qwen3AdaptiveSurgeryFeature,
    qwen3_neuron_node_id,
    qwen3_runtime_assignment_snapshot,
)
from neural_forge.hf_qwen_attention_surgery import (  # noqa: E402
    Qwen3AttentionSurgeryBridge,
    Qwen3AttentionSurgeryFeature,
    qwen3_attention_group_node_id,
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
)
from neural_forge.hf_qwen_sensitivity import (  # noqa: E402
    Qwen3ExactAnchorMasks,
    Qwen3SensitivityEvidence,
    Qwen3SensitivityFingerprints,
    Qwen3SensitivityReport,
)
import neural_forge.hf_qwen_surgery_allocator as surgery_allocator  # noqa: E402
from neural_forge.hf_qwen_surgery_allocator import (  # noqa: E402
    Qwen3CandidateValidationDamage,
    Qwen3MixedSurgeryBatch,
    Qwen3SurgeryAllocatorLimits,
    Qwen3SurgeryAllocatorShapeError,
    Qwen3SurgeryAllocatorStateError,
    Qwen3SurgeryAllocatorTargetError,
    Qwen3UnifiedSurgeryAllocator,
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, size: int = 24) -> None:
        self.tokens = tuple(f"token-{index}" for index in range(size))

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _runtime(*, seed: int = 7711, layers: int = 4) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=layers,
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
        attention_prototypes_per_group=layers,
        neuron_prototypes_per_neuron=layers,
    )


def _neuron_features(
    runtime: torch.nn.Module,
    *,
    exact: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
    snapshot = qwen3_runtime_assignment_snapshot(runtime)
    result = []
    for source in snapshot.prototype_source_indices:
        layer, neuron = divmod(source, snapshot.intermediate_width)
        value = float(source) / 100.0
        result.append(
            Qwen3AdaptiveSurgeryFeature(
                layer,
                neuron,
                weight=[value],
                activation=[value, value + 0.01],
                jacobian=[value + 0.02],
                dual=[value + 0.03],
                importance=1.0,
                anchor=(AnchorSpec.pinned() if (layer, neuron) in exact else None),
            )
        )
    return tuple(result)


def _attention_features(
    runtime: torch.nn.Module,
    *,
    exact: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[Qwen3AttentionSurgeryFeature, ...]:
    snapshot = qwen3_attention_runtime_snapshot(runtime)
    result = []
    for source in snapshot.prototype_source_indices:
        layer, group = divmod(source, snapshot.key_value_heads)
        value = float(source) / 100.0
        result.append(
            Qwen3AttentionSurgeryFeature(
                layer,
                group,
                weight=[value],
                activation=[value, value + 0.01],
                jvp=[value + 0.02],
                output=[value + 0.03],
                importance=1.0,
                anchor=(AnchorSpec.pinned() if (layer, group) in exact else None),
            )
        )
    return tuple(result)


def _allocator(
    runtime: torch.nn.Module,
    *,
    exact_neurons: frozenset[tuple[int, int]] = frozenset(),
    exact_attention: frozenset[tuple[int, int]] = frozenset(),
    limits: Qwen3SurgeryAllocatorLimits | None = None,
) -> Qwen3UnifiedSurgeryAllocator:
    return Qwen3UnifiedSurgeryAllocator(
        Qwen3AdaptiveSurgeryBridge(
            runtime,
            _neuron_features(runtime, exact=exact_neurons),
            max_nonconflicting_batch_size=4,
        ),
        Qwen3AttentionSurgeryBridge(
            runtime,
            _attention_features(runtime, exact=exact_attention),
            max_nonconflicting_batch_size=4,
        ),
        limits=limits,
    )


def _batch(allocator, proposals) -> Qwen3MixedSurgeryBatch:
    proposals = tuple(proposals)
    prototype_bytes = sum(item.prototype_bytes_saved for item in proposals)
    neuron_setup = max(
        (item.scale_setup_bytes for item in proposals if item.kind == "neuron"),
        default=0,
    )
    attention_setup = max(
        (item.scale_setup_bytes for item in proposals if item.kind == "attention"),
        default=0,
    )
    physical = prototype_bytes - neuron_setup - attention_setup
    return Qwen3MixedSurgeryBatch(
        proposals=proposals,
        parent_state_sha256=allocator._state_sha256(),
        requested_physical_bytes=physical,
        prototype_bytes_saved=prototype_bytes,
        scale_setup_bytes=neuron_setup + attention_setup,
        physical_bytes_saved=physical,
        total_risk=sum(item.risk for item in proposals),
        proposal_work=len(proposals),
        knapsack_states_considered=1,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _validated_proposals(
    allocator: Qwen3UnifiedSurgeryAllocator,
    *,
    explicit_neuron_pairs=(),
    explicit_attention_pairs=(),
):
    raw = allocator.discover_proposals(
        explicit_neuron_pairs=explicit_neuron_pairs,
        explicit_attention_pairs=explicit_attention_pairs,
    )
    parent = allocator._state_sha256()
    evidence = allocator._evidence
    discovery_rows = (
        (_digest("discovery-row"),)
        if evidence is None
        else (
            *evidence.fingerprints.calibration_row_fingerprints,
            *evidence.fingerprints.gate_row_fingerprints,
        )
    )
    allocator.install_candidate_validation(
        tuple(
            Qwen3CandidateValidationDamage(
                kind=item.kind,
                candidate_sha256=item.candidate.candidate_sha256,
                parent_state_sha256=parent,
                discovery_fingerprint=(
                    item.candidate.evidence_sha256
                    if evidence is None
                    else evidence.fingerprints.cache
                ),
                validation_fingerprint=_digest(
                    f"validation:{item.kind}:{item.candidate.candidate_sha256}"
                ),
                discovery_row_fingerprints=discovery_rows,
                validation_row_fingerprints=(_digest("validation-row"),),
                marginal_damage=item.risk,
            )
            for item in raw
        )
    )
    return allocator.discover_proposals(
        explicit_neuron_pairs=explicit_neuron_pairs,
        explicit_attention_pairs=explicit_attention_pairs,
    )


def _one_each(allocator):
    proposals = _validated_proposals(allocator)
    return (
        next(item for item in proposals if item.kind == "neuron"),
        next(item for item in proposals if item.kind == "attention"),
    )


def _accept(context, **metrics):
    return {
        "accepted": True,
        "parent_relative": True,
        "parent_state_sha256": context.parent_state_sha256,
        "evaluation_fingerprint": _digest("behavioral-evaluation"),
        **metrics,
    }


def _logits(runtime):
    with torch.inference_mode():
        return runtime(torch.tensor([[1, 3, 5, 7]], dtype=torch.long)).logits.clone()


def test_neuron_only_transaction_preserves_attention_and_tracks_real_bytes() -> None:
    runtime = _runtime(seed=7712)
    allocator = _allocator(runtime)
    neuron, _ = _one_each(allocator)
    batch = _batch(allocator, (neuron,))
    before_neuron = qwen3_runtime_assignment_snapshot(runtime)
    before_attention = qwen3_attention_runtime_snapshot(runtime)
    before_accounting = allocator.accounting()

    transaction = allocator.apply_batch(batch, evaluator=_accept)

    after_neuron = qwen3_runtime_assignment_snapshot(runtime)
    assert transaction.accepted is True
    assert after_neuron.prototype_count == before_neuron.prototype_count - 1
    assert qwen3_attention_runtime_snapshot(runtime) == before_attention
    assert (
        before_accounting.neuron_registry_bytes
        - transaction.committed_accounting.neuron_registry_bytes
        == batch.physical_bytes_saved
    )
    assert transaction.measured_behavioral_improvement_claim is False
    assert transaction.zero_recovery


def test_attention_only_transaction_is_atomic_qkvo_and_preserves_neurons() -> None:
    runtime = _runtime(seed=7713)
    allocator = _allocator(runtime)
    _, attention = _one_each(allocator)
    batch = _batch(allocator, (attention,))
    before_neuron = qwen3_runtime_assignment_snapshot(runtime)
    before_attention = qwen3_attention_runtime_snapshot(runtime)

    transaction = allocator.apply_batch(batch, evaluator=_accept)

    after_attention = qwen3_attention_runtime_snapshot(runtime)
    assert transaction.accepted is True
    assert qwen3_runtime_assignment_snapshot(runtime) == before_neuron
    assert after_attention.prototype_count == before_attention.prototype_count - 1
    assert all(
        getattr(runtime.model.layers[0].self_attn, role).registry
        is runtime.attention_codebook.registry
        for role in ("q_proj", "k_proj", "v_proj", "o_proj")
    )


def test_mixed_batch_changes_both_registries_under_one_evaluator_call() -> None:
    runtime = _runtime(seed=7714)
    allocator = _allocator(runtime)
    neuron, attention = _one_each(allocator)
    batch = _batch(allocator, (neuron, attention))
    before_neuron = qwen3_runtime_assignment_snapshot(runtime)
    before_attention = qwen3_attention_runtime_snapshot(runtime)
    calls = []

    def evaluator(context):
        calls.append(context)
        assert (
            context.proposed_neuron_snapshot.prototype_count
            == context.parent_neuron_snapshot.prototype_count - 1
        )
        assert (
            context.proposed_attention_snapshot.prototype_count
            == context.parent_attention_snapshot.prototype_count - 1
        )
        return _accept(context, parent_relative_gate="passed")

    transaction = allocator.apply_batch(batch, evaluator=evaluator)

    assert transaction.accepted
    assert len(calls) == 1
    assert (
        qwen3_runtime_assignment_snapshot(runtime).prototype_count
        == before_neuron.prototype_count - 1
    )
    assert (
        qwen3_attention_runtime_snapshot(runtime).prototype_count
        == before_attention.prototype_count - 1
    )
    report = qwen3_codebook_parameter_report(runtime)
    assert report.unique_parameter_count == sum(
        parameter.numel() for parameter in runtime.parameters()
    )


def test_bounded_byte_knapsack_uses_risk_and_one_time_setup_costs(monkeypatch) -> None:
    runtime = _runtime(seed=7715)
    allocator = _allocator(
        runtime,
        limits=Qwen3SurgeryAllocatorLimits(max_batch_size=2, max_knapsack_states=128),
    )
    proposals = allocator.discover_proposals()
    neurons = [item for item in proposals if item.kind == "neuron"]
    attention = next(item for item in proposals if item.kind == "attention")
    low_neuron = replace(
        neurons[0],
        risk=0.2,
        prototype_bytes_saved=60,
        scale_setup_bytes=10,
        individual_physical_bytes_saved=50,
        risk_per_individual_physical_byte=0.004,
    )
    expensive_neuron = replace(
        next(
            item
            for item in neurons[1:]
            if not (set(item.source_nodes) & set(low_neuron.source_nodes))
        ),
        risk=10.0,
        prototype_bytes_saved=100,
        scale_setup_bytes=10,
        individual_physical_bytes_saved=90,
        risk_per_individual_physical_byte=10.0 / 90.0,
    )
    low_attention = replace(
        attention,
        risk=0.1,
        prototype_bytes_saved=40,
        scale_setup_bytes=0,
        individual_physical_bytes_saved=40,
        risk_per_individual_physical_byte=0.0025,
    )
    monkeypatch.setattr(
        allocator,
        "discover_proposals",
        lambda **_kwargs: (low_attention, low_neuron, expensive_neuron),
    )

    batch = allocator.select_batch(80, unsafe_allow_unvalidated_candidates=True)

    assert set(batch.proposals) == {low_neuron, low_attention}
    assert batch.prototype_bytes_saved == 100
    assert batch.scale_setup_bytes == 10
    assert batch.physical_bytes_saved == 90
    assert batch.total_risk == pytest.approx(0.3)
    assert batch.knapsack_states_considered <= 8


def test_pareto_knapsack_minimizes_total_risk_not_risk_per_byte(monkeypatch) -> None:
    runtime = _runtime(seed=7727)
    allocator = _allocator(
        runtime,
        limits=Qwen3SurgeryAllocatorLimits(max_batch_size=2, max_knapsack_states=128),
    )
    raw = allocator.discover_proposals()
    neurons = [item for item in raw if item.kind == "neuron"]
    low_left = neurons[0]
    low_right = next(
        item
        for item in neurons[1:]
        if not (set(item.source_nodes) & set(low_left.source_nodes))
    )
    high_ratio_winner = next(item for item in raw if item.kind == "attention")
    high_ratio_winner = replace(
        high_ratio_winner,
        risk=9.0,
        prototype_bytes_saved=100,
        scale_setup_bytes=0,
        individual_physical_bytes_saved=100,
        risk_per_individual_physical_byte=0.09,
    )
    low_left = replace(
        low_left,
        risk=4.0,
        prototype_bytes_saved=40,
        scale_setup_bytes=0,
        individual_physical_bytes_saved=40,
        risk_per_individual_physical_byte=0.1,
    )
    low_right = replace(
        low_right,
        risk=4.0,
        prototype_bytes_saved=40,
        scale_setup_bytes=0,
        individual_physical_bytes_saved=40,
        risk_per_individual_physical_byte=0.1,
    )
    monkeypatch.setattr(
        allocator,
        "discover_proposals",
        lambda **_kwargs: (high_ratio_winner, low_left, low_right),
    )

    batch = allocator.select_batch(80, unsafe_allow_unvalidated_candidates=True)

    assert set(batch.proposals) == {low_left, low_right}
    assert batch.total_risk == pytest.approx(8.0)
    assert batch.physical_bytes_saved == 80


def test_knapsack_rejects_target_reaching_batches_beyond_overshoot(monkeypatch) -> None:
    runtime = _runtime(seed=7728)
    allocator = _allocator(
        runtime,
        limits=Qwen3SurgeryAllocatorLimits(
            max_batch_size=1,
            max_overshoot_bytes=1,
            max_overshoot_ratio=0.0,
        ),
    )
    proposal = replace(
        allocator.discover_proposals()[0],
        prototype_bytes_saved=100,
        scale_setup_bytes=0,
        individual_physical_bytes_saved=100,
    )
    monkeypatch.setattr(allocator, "discover_proposals", lambda **_kwargs: (proposal,))

    with pytest.raises(Qwen3SurgeryAllocatorTargetError, match="overshoot"):
        allocator.select_batch(10, unsafe_allow_unvalidated_candidates=True)


def test_allocator_limits_have_immutable_upper_ceilings() -> None:
    with pytest.raises(Qwen3SurgeryAllocatorShapeError, match="hard limit"):
        Qwen3SurgeryAllocatorLimits(max_knapsack_states=65_537)


def test_explicit_pairs_touching_exact_anchors_are_rejected() -> None:
    runtime = _runtime(seed=7716)
    allocator = _allocator(
        runtime,
        exact_neurons=frozenset({(0, 0)}),
        exact_attention=frozenset({(0, 0)}),
    )
    with pytest.raises(Qwen3SurgeryAllocatorShapeError, match="cannot touch"):
        allocator.discover_proposals(
            explicit_neuron_pairs=((qwen3_neuron_node_id(3, 5), (0, 0)),),
        )
    with pytest.raises(Qwen3SurgeryAllocatorShapeError, match="cannot touch"):
        allocator.discover_proposals(
            explicit_attention_pairs=((qwen3_attention_group_node_id(3, 1), (0, 0)),),
        )
    proposals = allocator.discover_proposals()
    exact = {
        qwen3_neuron_node_id(0, 0),
        qwen3_attention_group_node_id(0, 0),
    }
    assert all(not (set(item.candidate.member_ids) & exact) for item in proposals)


def test_stale_explicit_pairs_are_consumed_without_aborting_later_rounds() -> None:
    runtime = _runtime(seed=7729)
    allocator = _allocator(runtime)
    pair = (qwen3_neuron_node_id(3, 5), qwen3_neuron_node_id(0, 1))
    proposals = _validated_proposals(allocator, explicit_neuron_pairs=(pair,))
    explicit = next(
        item for item in proposals if item.kind == "neuron" and item.explicit_pair
    )
    allocator.apply_batch(_batch(allocator, (explicit,)), evaluator=_accept)
    skipped_before = allocator.diagnostics.stale_explicit_pairs_skipped

    allocator.discover_proposals(explicit_neuron_pairs=(pair,))

    assert allocator.diagnostics.stale_explicit_pairs_skipped == skipped_before + 1


def test_rejected_mixed_evaluation_restores_every_bridge_field_exactly() -> None:
    runtime = _runtime(seed=7717)
    allocator = _allocator(runtime)
    batch = _batch(allocator, _one_each(allocator))
    before_digest = allocator._state_sha256()
    before_neuron = runtime.neuron_codebook
    before_attention = runtime.attention_codebook
    before_report = qwen3_codebook_parameter_report(runtime)
    before_diagnostics = (
        allocator.diagnostics,
        allocator.attention_bridge.diagnostics,
        allocator.neuron_bridge.transactions,
    )

    transaction = allocator.apply_batch(
        batch,
        evaluator=lambda _: {"accepted": False, "reason": "held-out failed"},
    )

    assert transaction.accepted is False
    assert transaction.rollback_exact is True
    assert allocator._state_sha256() == before_digest
    assert runtime.neuron_codebook is before_neuron
    assert runtime.attention_codebook is before_attention
    assert qwen3_codebook_parameter_report(runtime) == before_report
    assert (
        allocator.diagnostics,
        allocator.attention_bridge.diagnostics,
        allocator.neuron_bridge.transactions,
    ) == before_diagnostics
    assert allocator.history == ()


def test_process_memory_ceiling_fails_closed_and_restores_runtime(monkeypatch) -> None:
    runtime = _runtime(seed=7730)
    allocator = _allocator(
        runtime,
        limits=Qwen3SurgeryAllocatorLimits(max_process_rss_increase_bytes=1),
    )
    batch = _batch(allocator, _one_each(allocator))
    before = _logits(runtime)
    calls = 0

    def rss():
        nonlocal calls
        calls += 1
        return 100 if calls == 1 else 102

    monkeypatch.setattr(surgery_allocator, "_process_rss_bytes", rss)
    with pytest.raises(Qwen3SurgeryAllocatorStateError, match="RSS"):
        allocator.apply_batch(batch, evaluator=_accept)

    assert torch.equal(_logits(runtime), before)
    assert allocator.history == ()


def test_stable_state_checks_do_not_rehash_full_tensor_contents(monkeypatch) -> None:
    runtime = _runtime(seed=7731)
    allocator = _allocator(runtime)
    first = allocator._state_sha256()

    def fail_hash(*_args, **_kwargs):
        raise AssertionError("unexpected tensor-content transfer")

    monkeypatch.setattr(surgery_allocator, "_logical_tensor_sha256", fail_hash)

    assert allocator._state_sha256() == first
    assert allocator._state_sha256() == first
    assert allocator.diagnostics.manifest_transfer_bytes == 0


def test_evaluator_exception_restores_both_registries_and_all_counters() -> None:
    runtime = _runtime(seed=7718)
    allocator = _allocator(runtime)
    batch = _batch(allocator, _one_each(allocator))
    before = allocator._state_sha256()
    neuron = runtime.neuron_codebook
    attention = runtime.attention_codebook

    def fail(_context):
        raise RuntimeError("injected evaluator failure")

    with pytest.raises(RuntimeError, match="injected evaluator failure"):
        allocator.apply_batch(batch, evaluator=fail)

    assert allocator._state_sha256() == before
    assert runtime.neuron_codebook is neuron
    assert runtime.attention_codebook is attention
    assert allocator.evidence_epoch == 0
    assert allocator.history == ()


def test_evaluator_cannot_mutate_ordinary_weights_and_logits_restore_bitwise() -> None:
    runtime = _runtime(seed=7722)
    allocator = _allocator(runtime)
    batch = _batch(allocator, _one_each(allocator))
    before_logits = _logits(runtime)
    before_digest = allocator._state_sha256()

    def mutate(_context):
        runtime.model.embed_tokens.weight[0].add_(17.0)
        return True

    with pytest.raises(Qwen3SurgeryAllocatorStateError, match="cannot mutate"):
        allocator.apply_batch(batch, evaluator=mutate)

    assert torch.equal(_logits(runtime), before_logits)
    assert allocator._state_sha256() == before_digest
    assert allocator.history == ()


def test_evaluator_mode_and_rng_mutation_is_restored_and_rejected() -> None:
    runtime = _runtime(seed=7723)
    allocator = _allocator(runtime)
    batch = _batch(allocator, _one_each(allocator))
    before_logits = _logits(runtime)
    before_rng = torch.random.get_rng_state().clone()
    before_modes = tuple(module.training for module in runtime.modules())

    def mutate_mode(context):
        runtime.train()
        runtime.model.embed_tokens.weight.data = torch.zeros_like(
            runtime.model.embed_tokens.weight
        )
        torch.rand(8)
        return _accept(context)

    with pytest.raises(Qwen3SurgeryAllocatorStateError, match="restored and rejected"):
        allocator.apply_batch(batch, evaluator=mutate_mode)

    assert tuple(module.training for module in runtime.modules()) == before_modes
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert torch.equal(_logits(runtime), before_logits)


def test_acceptance_requires_auditable_parent_relative_gate() -> None:
    runtime = _runtime(seed=7724)
    allocator = _allocator(runtime)
    batch = _batch(allocator, _one_each(allocator))
    before = _logits(runtime)

    with pytest.raises(Qwen3SurgeryAllocatorStateError, match="behavioral evaluator"):
        allocator.apply_batch(batch, evaluator=None)
    with pytest.raises(Qwen3SurgeryAllocatorStateError, match="parent-relative"):
        allocator.apply_batch(batch, evaluator=lambda _: True)

    assert torch.equal(_logits(runtime), before)
    assert allocator.history == ()


def test_candidate_validation_requires_disjoint_rows_and_current_parent() -> None:
    runtime = _runtime(seed=7725)
    allocator = _allocator(runtime)
    raw = allocator.discover_proposals()
    candidate = raw[0]
    parent = allocator._state_sha256()
    row = _digest("same-row")
    with pytest.raises(Qwen3SurgeryAllocatorShapeError, match="disjoint"):
        Qwen3CandidateValidationDamage(
            kind=candidate.kind,
            candidate_sha256=candidate.candidate.candidate_sha256,
            parent_state_sha256=parent,
            discovery_fingerprint=candidate.candidate.evidence_sha256,
            validation_fingerprint=_digest("different-validation"),
            discovery_row_fingerprints=(row,),
            validation_row_fingerprints=(row,),
            marginal_damage=0.1,
        )
    stale = Qwen3CandidateValidationDamage(
        kind=candidate.kind,
        candidate_sha256=candidate.candidate.candidate_sha256,
        parent_state_sha256="0" * 64,
        discovery_fingerprint=candidate.candidate.evidence_sha256,
        validation_fingerprint=_digest("validation-set"),
        discovery_row_fingerprints=(_digest("discovery-set"),),
        validation_row_fingerprints=(_digest("validation-row"),),
        marginal_damage=0.1,
    )
    with pytest.raises(Qwen3SurgeryAllocatorStateError, match="stale parent"):
        allocator.install_candidate_validation((stale,))


def test_candidate_validation_is_invalidated_after_accepted_edit() -> None:
    runtime = _runtime(seed=7726)
    allocator = _allocator(runtime)
    batch = _batch(allocator, _one_each(allocator))
    assert allocator.diagnostics.candidate_validation_records > 0

    transaction = allocator.apply_batch(batch, evaluator=_accept)

    assert transaction.accepted
    assert allocator.diagnostics.candidate_validation_records == 0
    with pytest.raises(Qwen3SurgeryAllocatorTargetError, match="validation"):
        allocator.select_batch(1)


def test_evidence_refresh_callbacks_run_only_after_accepted_mixed_gate() -> None:
    runtime = _runtime(seed=7719)
    allocator = _allocator(runtime)
    batch = _batch(allocator, _one_each(allocator))
    calls = {"neuron": 0, "attention": 0}

    def refresh_neurons(context):
        calls["neuron"] += 1
        active = set(context.active_node_ids)
        return tuple(
            item for item in allocator.neuron_bridge.features if item.node_id in active
        )

    def refresh_attention(context):
        calls["attention"] += 1
        active = set(context.active_node_ids)
        return tuple(
            item
            for item in allocator.attention_bridge.features
            if item.node_id in active
        )

    rejected = allocator.apply_batch(
        batch,
        evaluator=lambda _: False,
        refresh_neuron_features=refresh_neurons,
        refresh_attention_features=refresh_attention,
    )
    assert rejected.accepted is False
    assert calls == {"neuron": 0, "attention": 0}

    accepted = allocator.apply_batch(
        batch,
        evaluator=_accept,
        refresh_neuron_features=refresh_neurons,
        refresh_attention_features=refresh_attention,
    )
    assert accepted.accepted is True
    assert calls == {"neuron": 1, "attention": 1}
    assert allocator.evidence_epoch == 1
    assert allocator.neuron_bridge._feature_recomputations == 1
    assert allocator.attention_bridge.diagnostics.feature_refreshes == 1


def test_refresh_callback_cannot_mutate_live_runtime() -> None:
    runtime = _runtime(seed=7732)
    allocator = _allocator(runtime)
    neuron, _ = _one_each(allocator)
    batch = _batch(allocator, (neuron,))
    before = _logits(runtime)

    def mutate_refresh(_context):
        runtime.model.embed_tokens.weight[0].mul_(0.0)
        return allocator.neuron_bridge.features

    with pytest.raises(Qwen3SurgeryAllocatorStateError, match="cannot mutate"):
        allocator.apply_batch(
            batch,
            evaluator=_accept,
            refresh_neuron_features=mutate_refresh,
        )

    assert torch.equal(_logits(runtime), before)
    assert allocator.history == ()


def test_target_exhaustion_when_every_active_prototype_is_exact() -> None:
    runtime = _runtime(seed=7720, layers=2)
    neuron_snapshot = qwen3_runtime_assignment_snapshot(runtime)
    attention_snapshot = qwen3_attention_runtime_snapshot(runtime)
    exact_neurons = frozenset(
        (int(node[2]), int(node[3])) for node in neuron_snapshot.prototype_source_ids
    )
    exact_attention = frozenset(
        (int(node[2]), int(node[3])) for node in attention_snapshot.active_node_ids
    )
    allocator = _allocator(
        runtime,
        exact_neurons=exact_neurons,
        exact_attention=exact_attention,
    )

    with pytest.raises(Qwen3SurgeryAllocatorTargetError, match="No bounded"):
        allocator.select_batch(1)


def test_from_sensitivity_evidence_covers_unmeasured_active_nodes() -> None:
    runtime = _runtime(seed=7721, layers=2)
    manifest = surgery_allocator._logical_model_manifest(runtime)
    report_values = {}
    string_fields = {
        "schema",
        "attention_backend",
        "evidence_method",
        "directional_difference_reason",
        "model_fingerprint_mode",
    }
    tuple_fields = {"model_devices", "mlp_capture_methods"}
    bool_fields = {
        "exact_autodiff_used",
        "model_weights_mutated",
        "hooks_restored",
    }
    for field in fields(Qwen3SensitivityReport):
        if field.name in string_fields:
            report_values[field.name] = "synthetic-test-evidence"
        elif field.name in tuple_fields:
            report_values[field.name] = ("cpu",)
        elif field.name in bool_fields:
            report_values[field.name] = field.name == "hooks_restored"
        else:
            report_values[field.name] = 0
    evidence = Qwen3SensitivityEvidence(
        fingerprints=Qwen3SensitivityFingerprints(
            model=manifest.root,
            calibration="a" * 64,
            gate="b" * 64,
            runtime="1" * 64,
            measurement="c" * 64,
            cache="d" * 64,
            calibration_row_fingerprints=("e" * 64,),
            gate_row_fingerprints=("f" * 64,),
            model_manifest=manifest,
        ),
        neuron_activation_energy=tuple((0.1,) * 6 for _ in range(2)),
        attention_role_activation_energy=tuple(
            ((0.1, 0.1, 0.1, 0.1),) * 2 for _ in range(2)
        ),
        neurons=(),
        attention_groups=(),
        anchors=Qwen3ExactAnchorMasks(
            neuron_mask=tuple((False,) * 6 for _ in range(2)),
            attention_group_mask=tuple((False,) * 2 for _ in range(2)),
            explicit_neurons=frozenset(),
            measured_high_damage_neurons=frozenset(),
            explicit_attention_groups=frozenset(),
            measured_high_damage_attention_groups=frozenset(),
        ),
        report=Qwen3SensitivityReport(**report_values),
    )

    allocator = Qwen3UnifiedSurgeryAllocator.from_evidence(runtime, evidence)

    assert len(allocator.neuron_bridge.features) == (
        qwen3_runtime_assignment_snapshot(runtime).prototype_count
    )
    assert len(allocator.attention_bridge.features) == (
        qwen3_attention_runtime_snapshot(runtime).prototype_count
    )
    assert allocator.diagnostics.zero_recovery
    assert allocator.diagnostics.manifest_transfer_bytes == 0

    proposal = next(
        item for item in _validated_proposals(allocator) if item.kind == "neuron"
    )
    allocator.apply_batch(_batch(allocator, (proposal,)), evaluator=_accept)

    assert allocator._evidence is None
    assert allocator.diagnostics.candidate_validation_records == 0
