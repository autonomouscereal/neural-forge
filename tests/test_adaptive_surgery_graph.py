from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from neural_forge.adaptive_surgery_graph import (
    ACTION_KINDS,
    ActionRecord,
    AdaptiveSurgeryGraph,
    AnchorSpec,
    EvaluationDecision,
    ProjectionBinding,
    ScoreWeights,
    SurgeryNode,
    SurgeryTargetError,
)


def _node(
    node_id,
    value: float,
    *,
    parameter_bytes: int = 128,
    anchor: AnchorSpec | str | None = None,
    domain: list[float] | None = None,
    importance: float = 1.0,
) -> SurgeryNode:
    feature = np.array([value, value * 0.5 + 0.125], dtype=np.float64)
    return SurgeryNode(
        node_id=node_id,
        weight_features=feature,
        activation_features=feature + 0.01,
        rollout_jvp=feature - 0.02,
        output_dual=feature * 1.5,
        domain_features=[value] if domain is None else domain,
        importance=importance,
        parameter_byte_savings=parameter_bytes,
        anchor=anchor,
    )


def _grid_nodes() -> list[SurgeryNode]:
    # Logical rows/columns intentionally disagree with feature neighborhoods.
    return [
        _node((0, "mlp", 0), 0.00),
        _node((0, "mlp", 1), 10.00),
        _node((0, "mlp", 2), 20.00),
        _node((1, "mlp", 0), 30.00),
        _node((1, "mlp", 1), 0.01),
        _node((1, "mlp", 2), 10.01),
    ]


def test_grid_geometry_selects_diagonal_nonadjacent_batch_deterministically() -> None:
    first = AdaptiveSurgeryGraph(_grid_nodes(), projection_count=8, neighbor_window=2)
    second = AdaptiveSurgeryGraph(_grid_nodes(), projection_count=8, neighbor_window=2)

    first_candidates = first.propose_candidates()
    second_candidates = second.propose_candidates()
    expected_edges = {
        frozenset({(0, "mlp", 0), (1, "mlp", 1)}),
        frozenset({(0, "mlp", 1), (1, "mlp", 2)}),
    }
    actual_edges = {frozenset(candidate.sources) for candidate in first_candidates[:2]}

    assert actual_edges == expected_edges
    assert all(left[0] != right[0] for left, right in actual_edges)
    assert all(left[2] != right[2] for left, right in actual_edges)
    assert [candidate.candidate_sha256 for candidate in first_candidates] == [
        candidate.candidate_sha256 for candidate in second_candidates
    ]

    batch = first.propose_nonconflicting_batch(2)
    outcome = first.apply_batch(batch)

    assert outcome.accepted
    assert outcome.snapshot.prototype_count == 4
    assert outcome.snapshot.parameter_bytes_saved == 256
    assert len(set(outcome.snapshot.prototype_source_indices)) == 4
    assert {action.kind for action in outcome.transaction.actions} >= {
        "merge",
        "preserve",
        "share",
    }
    assert len(outcome.transaction.transaction_sha256) == 64
    assert len(outcome.snapshot.provenance_sha256) == 64


def test_exact_pinned_anchor_remains_its_own_real_medoid() -> None:
    pinned = ("depth-7", "attention", "head-A")
    neighbor = ("depth-2", "attention", "head-Z")
    graph = AdaptiveSurgeryGraph(
        [
            _node(pinned, 0.0, parameter_bytes=4096, anchor="exact"),
            _node(neighbor, 0.001, parameter_bytes=64),
        ]
    )

    candidate = graph.propose_candidates()[0]
    assert candidate.resulting_medoid == pinned
    assert candidate.exact_anchor == pinned

    snapshot = graph.apply_one(candidate).snapshot
    assert snapshot.assignments[pinned] == pinned
    assert snapshot.assignments[neighbor] == pinned
    assert snapshot.prototype_source_ids == (pinned,)

    two_pins = AdaptiveSurgeryGraph(
        [
            _node(("a", "role", 8), 0.0, anchor="exact"),
            _node(("b", "role", 3), 0.0, anchor="exact"),
        ]
    )
    with pytest.raises(SurgeryTargetError, match="exact anchors"):
        two_pins.fold_to(1)


def test_projected_anchor_retains_executable_coefficient_binding() -> None:
    anchored = ("encoder", "branch-9", "unit-left")
    survivor = ("decoder", "branch-1", "unit-right")
    anchor = AnchorSpec.projected(0.01, coefficient=0.25)
    graph = AdaptiveSurgeryGraph(
        [
            _node(
                anchored,
                1.0,
                parameter_bytes=4096,
                anchor=anchor,
                importance=0.01,
            ),
            _node(survivor, 1.0001, parameter_bytes=64, importance=10.0),
        ],
        score_weights=ScoreWeights(savings_byte_scale=64.0),
    )

    candidate = graph.propose_candidates()[0]
    assert candidate.resulting_medoid == survivor
    assert candidate.score.anchor_damage > 0.0
    assert candidate.score.activation_damage > 0.0
    assert candidate.score.rollout_jvp_damage > 0.0
    assert candidate.score.output_dual_damage > 0.0
    assert candidate.score.domain_damage > 0.0
    assert candidate.score.parameter_bytes_saved == 4096

    outcome = graph.apply_one(candidate)
    project_actions = [
        action for action in outcome.transaction.actions if action.kind == "project"
    ]
    assert len(project_actions) == 1
    assert project_actions[0].sources == (anchored,)
    assert project_actions[0].target == survivor
    assert project_actions[0].anchor_mode == "projected"
    binding = outcome.snapshot.projection_bindings[0]
    assert isinstance(binding, ProjectionBinding)
    assert binding.source == anchored
    assert binding.target == survivor
    assert binding.coefficient == 0.25
    assert binding.apply([8.0, -4.0]) == (2.0, -1.0)
    assert outcome.transaction.projection_bindings == (binding,)
    assert outcome.snapshot.as_dict()["projection_bindings"] == [binding.as_dict()]
    assert outcome.transaction.as_dict()["projection_binding_sha256s"] == [
        binding.binding_sha256
    ]


def test_soft_anchor_has_damage_but_no_projection_binding() -> None:
    anchored = ("encoder", "soft", "unit-left")
    survivor = ("decoder", "soft", "unit-right")
    graph = AdaptiveSurgeryGraph(
        [
            _node(
                anchored,
                1.0,
                parameter_bytes=4096,
                anchor=AnchorSpec.soft(0.01),
                importance=0.01,
            ),
            _node(survivor, 1.0001, parameter_bytes=64, importance=10.0),
        ],
        score_weights=ScoreWeights(savings_byte_scale=64.0),
    )

    candidate = graph.propose_candidates()[0]
    assert candidate.score.anchor_damage > 0.0
    outcome = graph.apply_one(candidate)

    assert outcome.snapshot.projection_bindings == ()
    assert outcome.transaction.projection_bindings == ()
    assert not any(action.kind == "project" for action in outcome.transaction.actions)
    soft_share = next(
        action
        for action in outcome.transaction.actions
        if action.kind == "share" and action.sources == (anchored,)
    )
    assert soft_share.anchor_mode == "soft"


def test_evaluator_rejection_rolls_back_and_exposes_alternate_choice() -> None:
    graph = AdaptiveSurgeryGraph(_grid_nodes(), projection_count=8, neighbor_window=2)
    parent = graph.snapshot
    rejected_candidate = graph.propose_candidates()[0]

    outcome = graph.apply_one(
        rejected_candidate,
        evaluator=lambda old, proposed: EvaluationDecision.reject(
            "parent-relative quality gate",
            metrics={
                "old_prototypes": old.prototype_count,
                "new_prototypes": proposed.prototype_count,
            },
        ),
    )

    assert outcome.rolled_back
    assert outcome.snapshot is parent
    assert graph.snapshot.snapshot_sha256 == parent.snapshot_sha256
    assert outcome.proposed_snapshot.prototype_count == parent.prototype_count - 1
    assert outcome.attempt.actions[0].kind == "skip"
    alternate = graph.propose_candidates()[0]
    assert alternate.candidate_sha256 != rejected_candidate.candidate_sha256

    accepted = graph.apply_one(alternate, evaluator=lambda old, new: True)
    assert accepted.accepted
    assert accepted.snapshot.prototype_count == parent.prototype_count - 1
    assert [attempt.accepted for attempt in graph.attempts] == [False, True]


def test_accepted_transaction_recomputes_and_changes_the_next_choice() -> None:
    nodes = [
        _node(("depth", 0, "u"), 0.00, domain=[0.00]),
        _node(("depth", 1, "u"), 0.01, domain=[0.01]),
        _node(("depth", 2, "u"), 0.02, domain=[0.02]),
        _node(("depth", 3, "u"), 10.0, domain=[10.0]),
    ]
    graph = AdaptiveSurgeryGraph(nodes, projection_count=4, neighbor_window=2)
    before = graph.propose_candidates()
    first = before[0]
    old_second = before[1]

    graph.apply_one(first)
    after = graph.propose_candidates()
    next_candidate = after[0]

    assert after.diagnostics.search_round == before.diagnostics.search_round + 1
    assert next_candidate.parent_snapshot_sha256 == graph.snapshot.snapshot_sha256
    assert next_candidate.candidate_sha256 != old_second.candidate_sha256
    assert set(first.member_ids) < set(next_candidate.member_ids)
    assert ("depth", 2, "u") in next_candidate.member_ids

    composed = graph.apply_one(next_candidate)
    compose_action = next(
        action for action in composed.transaction.actions if action.kind == "compose"
    )
    assert compose_action.parent_action_sha256s
    assert composed.snapshot.prototype_count == 2


def test_accepted_commit_refreshes_evidence_before_next_rescore() -> None:
    node_ids = tuple(("refresh", index, "unit") for index in range(5))
    initial_values = (0.0, 0.001, 0.002, 10.0, 10.01)
    refreshed_values = (0.0, 0.001, 50.0, 10.0, 10.01)
    nodes = [
        _node(node_id, value, domain=[value])
        for node_id, value in zip(node_ids, initial_values)
    ]

    control = AdaptiveSurgeryGraph(nodes, projection_count=2, neighbor_window=2)
    control.apply_one(control.propose_candidate_for_pair((0, 1)))
    control_next = frozenset(control.propose_candidates()[0].sources)

    refresh_calls: list[str] = []

    def refresh(snapshot, current_nodes):
        refresh_calls.append(snapshot.snapshot_sha256)
        return tuple(
            _node(
                node.node_id,
                refreshed_values[index],
                parameter_bytes=node.parameter_bytes,
                anchor=node.anchor,
                domain=[refreshed_values[index]],
                importance=node.importance,
            )
            for index, node in enumerate(current_nodes)
        )

    graph = AdaptiveSurgeryGraph(
        nodes,
        projection_count=2,
        neighbor_window=2,
        evidence_refresh=refresh,
    )
    original_evidence_sha256 = graph.evidence_sha256
    graph.apply_one(graph.propose_candidate_for_pair((0, 1)))
    refreshed_candidate = graph.propose_candidates()[0]
    refreshed_next = frozenset(refreshed_candidate.sources)

    assert len(refresh_calls) == 1
    assert graph.evidence_sha256 != original_evidence_sha256
    assert refreshed_candidate.evidence_sha256 == graph.evidence_sha256
    assert graph.nodes[2].weight_features[0] == 50.0
    assert graph.nodes[2].rollout_jvp[0] == 49.98
    assert refreshed_next == frozenset({node_ids[3], node_ids[4]})
    assert refreshed_next != control_next

    rejected = AdaptiveSurgeryGraph(nodes, evidence_refresh=refresh)
    rejected.apply_one(
        rejected.propose_candidate_for_pair((0, 1)),
        evaluator=lambda *_args: False,
    )
    assert len(refresh_calls) == 1
    assert rejected.nodes == tuple(nodes)


def test_restore_reinstates_the_snapshot_evidence_epoch_and_candidates() -> None:
    node_ids = tuple(("restore-refresh", index) for index in range(4))
    initial_values = (0.0, 0.001, 10.0, 10.001)
    refreshed_values = (0.0, 0.001, 50.0, 50.001)
    nodes = [_node(node_id, value) for node_id, value in zip(node_ids, initial_values)]

    def refresh(_snapshot, current_nodes):
        return tuple(
            _node(
                node.node_id,
                refreshed_values[index],
                parameter_bytes=node.parameter_bytes,
                anchor=node.anchor,
                importance=node.importance,
            )
            for index, node in enumerate(current_nodes)
        )

    graph = AdaptiveSurgeryGraph(
        nodes,
        projection_count=2,
        neighbor_window=2,
        history_limit=2,
        evidence_refresh=refresh,
    )
    root = graph.snapshot
    root_evidence_sha256 = graph.evidence_sha256
    root_candidates = graph.propose_candidates()
    root_candidate_hashes = [candidate.candidate_sha256 for candidate in root_candidates]

    accepted = graph.apply_one(graph.propose_candidate_for_pair((0, 1))).snapshot
    refreshed_evidence_sha256 = graph.evidence_sha256
    refreshed_candidates = graph.propose_candidates()
    refreshed_candidate_hashes = [
        candidate.candidate_sha256 for candidate in refreshed_candidates
    ]

    graph.restore(root)

    assert graph.nodes == tuple(nodes)
    assert graph.evidence_sha256 == root_evidence_sha256
    assert [candidate.candidate_sha256 for candidate in graph.propose_candidates()] == (
        root_candidate_hashes
    )
    assert graph.propose_candidates().candidates == root_candidates.candidates

    graph.restore(accepted)

    assert graph.nodes[2].weight_features[0] == 50.0
    assert graph.evidence_sha256 == refreshed_evidence_sha256
    assert [candidate.candidate_sha256 for candidate in graph.propose_candidates()] == (
        refreshed_candidate_hashes
    )
    assert graph.propose_candidates().candidates == refreshed_candidates.candidates


def test_projected_anchor_scoring_uses_binding_coefficient_and_callback() -> None:
    source = ("projected-score", "source")
    target = ("projected-score", "target")

    def score_for(coefficient: float) -> float:
        graph = AdaptiveSurgeryGraph(
            [
                _node(
                    source,
                    1.0,
                    parameter_bytes=4096,
                    anchor=AnchorSpec.projected(1.0, coefficient=coefficient),
                ),
                _node(target, 1.001, anchor="exact"),
            ]
        )
        return graph.propose_candidates()[0].score.anchor_damage

    default_damages = [score_for(coefficient) for coefficient in (0.0, 1.0, 100.0)]

    assert len(set(default_damages)) == 3
    assert default_damages[0] < default_damages[1] < default_damages[2]

    observed: list[ProjectionBinding] = []

    def domain_projection_damage(binding, _source, _target, _distances):
        observed.append(binding)
        assert binding.apply(3.0) == 12.0
        return 2.5

    graph = AdaptiveSurgeryGraph(
        [
            _node(
                source,
                1.0,
                parameter_bytes=4096,
                anchor=AnchorSpec.projected(2.0, coefficient=4.0),
            ),
            _node(target, 1.001, anchor="exact"),
        ],
        projection_damage_callback=domain_projection_damage,
    )
    candidate = graph.propose_candidates()[0]

    assert observed
    assert candidate.score.anchor_damage == 5.0


def test_explicit_pair_zero_six_is_scored_even_when_discovery_omits_it() -> None:
    node_ids = tuple(("nonadjacent", index * 13, "unit") for index in range(7))
    graph = AdaptiveSurgeryGraph(
        [_node(node_id, float(index)) for index, node_id in enumerate(node_ids)],
        projection_count=1,
        neighbor_window=1,
        max_candidates=2,
        candidate_pair_limit=2,
    )

    discovered = graph.propose_candidates()
    explicit_edge = frozenset({node_ids[0], node_ids[6]})

    assert explicit_edge not in {
        frozenset(candidate.sources) for candidate in discovered
    }
    before_pairs_scored = graph.work_diagnostics.candidate_pairs_scored
    candidate = graph.propose_candidate_for_pair((0, 6))

    assert frozenset(candidate.sources) == explicit_edge
    assert graph.score_pair(node_ids[0], node_ids[6]) == candidate.score
    assert graph.work_diagnostics.candidate_pairs_scored == before_pairs_scored + 2
    assert discovered.diagnostics.candidate_buffer_bound == 2
    assert discovered.diagnostics.materialized_full_pairwise_matrix is False
    assert graph.apply_one(candidate).snapshot.prototype_count == 6


def test_equal_damage_medoid_uses_total_utility_including_bytes() -> None:
    large = (0, "large-parameter-source")
    small = (1, "small-parameter-source")
    graph = AdaptiveSurgeryGraph(
        [
            _node(large, 1.0, parameter_bytes=4096),
            _node(small, 1.0, parameter_bytes=64),
        ],
        score_weights=ScoreWeights(savings_byte_scale=1.0),
    )

    candidate = graph.propose_candidates()[0]

    assert candidate.score.weighted_damage == 0.0
    assert candidate.resulting_medoid == small
    assert candidate.score.parameter_bytes_saved == 4096


def test_history_and_search_buffers_remain_capped_as_node_count_grows() -> None:
    history_limit = 3
    pair_limit = 7
    graph = AdaptiveSurgeryGraph(
        [_node(("history", index), index * 0.01) for index in range(12)],
        projection_count=2,
        neighbor_window=2,
        max_candidates=4,
        candidate_pair_limit=pair_limit,
        history_limit=history_limit,
    )

    snapshot = graph.fold_to(2)
    diagnostics = graph.work_diagnostics

    assert snapshot.transaction_count == 10
    assert snapshot.retained_transaction_count == history_limit
    assert len(snapshot.transactions) == history_limit
    assert [transaction.transaction_index for transaction in snapshot.transactions] == [
        8,
        9,
        10,
    ]
    assert len(graph.attempts) == history_limit
    assert len(graph.snapshot_history) == history_limit
    assert len(graph._evidence_states) == history_limit
    assert diagnostics.history_limit == history_limit
    assert diagnostics.retained_snapshots == history_limit
    assert diagnostics.retained_attempts == history_limit
    assert diagnostics.candidate_pair_buffer_limit == pair_limit

    larger = AdaptiveSurgeryGraph(
        [_node(("large-search", index), float(index)) for index in range(120)],
        projection_count=2,
        neighbor_window=2,
        max_candidates=4,
        candidate_pair_limit=pair_limit,
        history_limit=history_limit,
    )
    larger_candidates = larger.propose_candidates()

    assert larger_candidates.diagnostics.candidate_buffer_bound == pair_limit
    assert larger_candidates.diagnostics.unique_pair_count <= pair_limit
    assert larger.history_limit == history_limit


def test_rejected_batch_is_bisected_without_blacklisting_constituents() -> None:
    graph = AdaptiveSurgeryGraph(_grid_nodes(), projection_count=8, neighbor_window=2)
    batch = graph.propose_nonconflicting_batch(2)
    initial_prototypes = graph.snapshot.prototype_count

    def reject_batches(_parent, _snapshot, transaction):
        if len(transaction.candidates) > 1:
            return EvaluationDecision.reject("isolate batch damage")
        return EvaluationDecision.accept("singleton passes")

    outcomes = graph.apply_batch_with_retry(batch, evaluator=reject_batches)

    assert [outcome.accepted for outcome in outcomes] == [False, True, True]
    assert graph.snapshot.prototype_count == initial_prototypes - 2
    assert graph.work_diagnostics.rejected_transactions == 1
    assert graph.work_diagnostics.accepted_transactions == 2
    assert graph.work_diagnostics.singleton_rejections == 0


@pytest.mark.parametrize("resolution", ["commit", "reject"])
def test_commit_and_reject_are_atomic_when_recompute_raises(
    monkeypatch, resolution
) -> None:
    graph = AdaptiveSurgeryGraph(_grid_nodes(), history_limit=4)
    candidate = graph.propose_candidates()[0]
    proposal = graph.preview(candidate)
    parent = graph.snapshot
    attempts_before = graph.attempts
    history_before = graph.snapshot_history
    work_before = graph.work_diagnostics
    nodes_before = graph.nodes
    evidence_before = graph.evidence_sha256
    original_recompute = graph._compute_candidate_set

    def explode():
        raise RuntimeError("forced candidate recompute failure")

    def refreshed_nodes(_snapshot, current_nodes):
        return tuple(
            _node(
                node.node_id,
                100.0 + index,
                parameter_bytes=node.parameter_bytes,
                anchor=node.anchor,
                importance=node.importance,
            )
            for index, node in enumerate(current_nodes)
        )

    monkeypatch.setattr(graph, "_compute_candidate_set", explode)
    with pytest.raises(RuntimeError, match="forced candidate recompute failure"):
        if resolution == "commit":
            graph.commit(proposal, evidence_refresh=refreshed_nodes)
        else:
            graph.reject(proposal)

    assert graph.snapshot is parent
    assert graph.nodes == nodes_before
    assert graph.evidence_sha256 == evidence_before
    assert graph.attempts == attempts_before
    assert graph.snapshot_history == history_before
    assert graph.work_diagnostics == work_before

    monkeypatch.setattr(graph, "_compute_candidate_set", original_recompute)
    outcome = (
        graph.commit(proposal) if resolution == "commit" else graph.reject(proposal)
    )
    assert outcome.accepted is (resolution == "commit")


def test_exact_current_target_is_a_true_no_op() -> None:
    graph = AdaptiveSurgeryGraph(_grid_nodes())
    initial = graph.snapshot
    calls = 0

    def evaluator(_parent, _proposed):
        nonlocal calls
        calls += 1
        return True

    result = graph.fold_to(initial.prototype_count, evaluator=evaluator)

    assert result is initial
    assert result.is_exact
    assert result.transaction_count == 0
    assert graph.work_diagnostics.candidate_recomputations == 0
    assert calls == 0


def test_candidate_search_has_explicit_subquadratic_work_bound() -> None:
    node_count = 160
    nodes = [
        _node(
            ("stage", index % 11, "interleaved", index * 17),
            float(np.sin(index * 0.37) + index * 0.002),
            parameter_bytes=32 + index % 3,
        )
        for index in range(node_count)
    ]
    graph = AdaptiveSurgeryGraph(
        nodes,
        projection_count=3,
        neighbor_window=2,
        max_candidates=9,
        medoid_candidate_limit=4,
    )

    candidates = graph.propose_candidates()
    diagnostics = candidates.diagnostics

    assert len(candidates) <= 9
    assert diagnostics.materialized_full_pairwise_matrix is False
    assert diagnostics.no_full_pairwise_matrix
    assert diagnostics.bounded_work
    assert diagnostics.projection_neighbor_comparisons <= (
        diagnostics.projection_neighbor_comparison_bound
    )
    assert diagnostics.unique_pair_count <= diagnostics.candidate_buffer_bound
    assert diagnostics.unique_pair_count < node_count * (node_count - 1) // 2
    assert diagnostics.assignment_distance_evaluations <= (
        diagnostics.assignment_distance_evaluation_bound
    )
    assert diagnostics.distance_coordinate_operations <= (
        diagnostics.distance_coordinate_operation_bound
    )
    assert diagnostics.no_training
    assert graph.work_diagnostics.no_training


def test_arbitrary_tuple_ids_snapshots_and_registry_views_are_immutable() -> None:
    ids = (
        ("stage", ("branch", 2), "q", 17),
        (5, "cross-depth", None, b"left"),
        ("tail", False, -4, ("nested", "unit")),
    )
    nodes = [
        _node(ids[0], 0.0),
        _node(ids[1], 0.001),
        _node(ids[2], 8.0),
    ]
    first = AdaptiveSurgeryGraph(nodes)
    second = AdaptiveSurgeryGraph(nodes)

    assert first.graph_sha256 == second.graph_sha256
    first_candidate = first.propose_candidates()[0]
    second_candidate = second.propose_candidates()[0]
    assert first_candidate.candidate_sha256 == second_candidate.candidate_sha256

    swapped = replace(
        first_candidate,
        left_prototype=first_candidate.right_prototype,
        right_prototype=first_candidate.left_prototype,
        left_members=first_candidate.right_members,
        right_members=first_candidate.left_members,
    )
    assert swapped.operation_sha256 == first_candidate.operation_sha256
    assert swapped.candidate_sha256 != first_candidate.candidate_sha256

    snapshot = first.apply_one(first_candidate).snapshot
    assignments, source_indices = snapshot.registry_arrays()
    assert snapshot.node_ids == ids
    assert tuple(assignments) == snapshot.assignment_indices
    assert tuple(source_indices) == snapshot.prototype_source_indices
    for node_index, slot in enumerate(assignments):
        source_index = int(source_indices[slot])
        assert snapshot.assignment_pairs[node_index][1] == ids[source_index]
    with pytest.raises(TypeError):
        snapshot.assignments[ids[0]] = ids[2]  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        assignments[0] = 99
    with pytest.raises(FrozenInstanceError):
        snapshot.state_sha256 = "0" * 64  # type: ignore[misc]

    registry = snapshot.registry_payload()
    assert registry["assignments_sha256"] == snapshot.assignments_sha256
    assert registry["provenance_sha256"] == snapshot.provenance_sha256
    assert len(registry["registry_sha256"]) == 64


def test_candidate_discovery_is_cross_process_hashseed_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import json
        import numpy as np
        from neural_forge.adaptive_surgery_graph import AdaptiveSurgeryGraph, SurgeryNode

        nodes = []
        for index in range(24):
            value = float((index * 17) % 29) / 29.0
            feature = np.asarray([value, value * 0.25 + 0.125], dtype=np.float64)
            nodes.append(SurgeryNode(
                node_id=("node", index),
                weight_features=feature,
                activation_features=feature + 0.01,
                rollout_jvp=feature - 0.02,
                output_dual=feature * 1.5,
                domain_features=[float(index % 3)],
                importance=1.0 + index / 100.0,
                parameter_byte_savings=128,
            ))
        # Iteration order deliberately depends on PYTHONHASHSEED. The graph must
        # canonicalize these collections before they can affect discovery.
        pinned = {("node", value) for value in (3, 11)}
        soft = {key: 2.0 for key in {("node", value) for value in (5, 7, 19)}}
        graph = AdaptiveSurgeryGraph(
            nodes,
            pinned_anchors=pinned,
            soft_anchors=soft,
            projection_count=7,
            neighbor_window=3,
            max_candidates=12,
        )
        candidates = graph.propose_candidates()
        print(json.dumps({
            "graph": graph.graph_sha256,
            "exact": [item.candidate_sha256 for item in candidates],
            "semantic": [item.operation_sha256 for item in candidates],
        }, sort_keys=True))
        """
    )
    outputs = []
    for hash_seed in ("1", "777", "4294967295"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        environment["PYTHONPATH"] = str(root / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
            env=environment,
        )
        outputs.append(json.loads(completed.stdout))
    assert outputs[1:] == outputs[:-1]


def test_all_action_record_kinds_are_valid_and_hashed() -> None:
    source = ("layer-x", "role-y", "unit-z")
    records = [
        ActionRecord(kind=kind, sources=(source,), target=source)
        for kind in ACTION_KINDS
    ]

    assert {record.kind for record in records} == set(ACTION_KINDS)
    assert all(len(record.action_sha256) == 64 for record in records)
