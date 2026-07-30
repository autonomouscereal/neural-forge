from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
import neural_forge.hf_lfm2_semantic_step_token_fold as semantic_module
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    LFM2_SEMANTIC_STEP_PROGRESS_SCHEMA,
    POSITIONAL_CHANNEL_DESIGNS,
    Lfm2SemanticStepSolvedChunkCheckpoint,
    Lfm2SemanticStepTokenFold,
    Lfm2SemanticStepTokenFoldError,
    bind_lfm2_semantic_step_hard_anchor_runtime,
    bind_lfm2_sparse_semantic_code_runtime,
    compile_lfm2_semantic_step_token_fold,
    materialize_lfm2_semantic_step_design_matrix,
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.numeric_execution_policy import (
    NumericExecutionRequest,
    resolve_numeric_execution_policy,
)


SPANS = ([0], [1], [0, 1], [1, 0], [2, 3, 1])


def _evidence():
    return seal_lfm2_semantic_step_span_evidence(
        SPANS,
        target_row_count=4,
        lineage={"token_bridge_sha256": "a" * 64},
    )


def _source_table() -> tuple[torch.Tensor, torch.Tensor]:
    evidence = _evidence()
    design = materialize_lfm2_semantic_step_design_matrix(
        evidence,
        positional_design="linear-forward",
        dtype=torch.float64,
    )
    exact_target = torch.tensor(
        [
            [1.0, -0.25, 0.5],
            [-0.5, 1.5, 0.75],
            [0.25, 0.5, -1.0],
            [1.25, -0.75, 0.125],
        ],
        dtype=torch.float64,
    )
    return design @ exact_target, design


def test_sparse_ridge_fold_is_order_sensitive_chunk_deterministic_and_exact_at_runtime() -> (
    None
):
    source, diagnostic_design = _source_table()
    evidence = _evidence()
    progress = []
    first = compile_lfm2_semantic_step_token_fold(
        source,
        evidence,
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
        verification_donor_rows_per_chunk=1,
        progress_callback=progress.append,
    )
    second = compile_lfm2_semantic_step_token_fold(
        source,
        evidence,
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=99,
        verification_donor_rows_per_chunk=3,
    )

    assert torch.equal(first.target_table, second.target_table)
    assert first.as_dict()["learned_parameter_count"] == 4 * 3
    assert sum(parameter.numel() for parameter in first.parameters()) == 4 * 3
    assert len(tuple(first.parameters())) == 1
    assert first.target_table.requires_grad is False
    assert first.as_dict()["routing_parameter_count"] == 0
    assert first.as_dict()["parameter_alias_count"] == 0
    assert first.as_dict()["full_reconstructed_donor_table_materializations"] == 0
    assert first.as_dict()["full_reconstructed_donor_table_persisted"] is False
    assert first.as_dict()["pcg_operator_uses_strict_partial_donor_row_blocks"] is True
    assert first.as_dict()["maximum_matrix_free_operator_donor_rows_per_block"] < len(
        SPANS
    )
    assert first.as_dict()["operation_audit"]["training_steps"] == 0
    assert first.as_dict()["operation_audit"]["parent_forward_calls"] == 0

    # Reversed two-token spans have different fixed positional coefficients.
    assert not torch.equal(diagnostic_design[2], diagnostic_design[3])
    reversed_rows = first.reconstruct_selected_donor_embeddings([2, 3])
    assert not torch.allclose(reversed_rows[0], reversed_rows[1])

    selected = torch.tensor([4, 0, 3, 2], dtype=torch.int64)
    reconstructed = first.reconstruct_selected_donor_embeddings(
        selected,
        maximum_rows_per_chunk=2,
    )
    explicit_reconstructed = diagnostic_design @ first.target_table.detach().cpu()
    assert torch.allclose(
        reconstructed.cpu(),
        explicit_reconstructed.index_select(0, selected),
        atol=1e-12,
        rtol=0,
    )

    hidden = torch.tensor(
        [[0.5, -1.0, 0.25], [-0.75, 0.125, 1.5]],
        dtype=torch.float64,
    )
    target_scores = first.score_target_tokens(hidden)
    donor_logits = first.compute_all_donor_logits_from_target_scores(
        target_scores,
        maximum_donor_rows_per_chunk=2,
    )
    explicit_logits = target_scores.cpu() @ diagnostic_design.transpose(0, 1)
    assert torch.allclose(donor_logits.cpu(), explicit_logits, atol=1e-12, rtol=0)

    assert progress[0]["stage"] == "start"
    assert progress[-1]["stage"] == "publication-complete"
    previous = None
    for ordinal, immutable_record in enumerate(progress):
        record = dict(immutable_record)
        assert record["schema"] == LFM2_SEMANTIC_STEP_PROGRESS_SCHEMA
        assert record["ordinal"] == ordinal
        assert record["previous_progress_sha256"] == previous
        supplied = record.pop("progress_sha256")
        assert supplied == canonical_sha256(record)
        previous = supplied
        with pytest.raises(TypeError):
            immutable_record["stage"] = "tampered"
    assert first.as_dict()["progress_record_count"] == len(progress)
    assert first.as_dict()["progress_final_sha256"] == previous
    heartbeats = [
        dict(record) for record in progress if record["stage"] == "solver-heartbeat"
    ]
    assert heartbeats
    by_hidden_range = {}
    for record in heartbeats:
        key = (record["hidden_start"], record["hidden_stop"])
        by_hidden_range.setdefault(key, []).append(record)
        assert record["elapsed_seconds"] is None
        assert record["elapsed_timing_collected"] is False
    for records in by_hidden_range.values():
        observed_iterations = [record["solver_iteration"] for record in records]
        assert observed_iterations == sorted(observed_iterations)
        assert observed_iterations[0] == 0
        assert records[-1]["final_heartbeat"] is True


@pytest.mark.parametrize(
    "design",
    tuple(value for value in POSITIONAL_CHANNEL_DESIGNS if value != "uniform"),
)
def test_every_supported_positional_design_distinguishes_reversed_spans(
    design: str,
) -> None:
    matrix = materialize_lfm2_semantic_step_design_matrix(
        _evidence(),
        positional_design=design,
        dtype=torch.float64,
    )
    assert not torch.equal(matrix[2], matrix[3])
    assert torch.allclose(matrix.sum(dim=1), torch.ones(5, dtype=torch.float64))


def test_uniform_design_is_the_additive_mean_baseline() -> None:
    matrix = materialize_lfm2_semantic_step_design_matrix(
        _evidence(),
        positional_design="uniform",
        dtype=torch.float64,
    )
    assert torch.equal(matrix[2], matrix[3])
    assert torch.equal(
        matrix[4],
        torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3], dtype=torch.float64),
    )


def test_multi_token_span_emits_externally_but_advances_body_exactly_once() -> None:
    source, _design = _source_table()
    fold = compile_lfm2_semantic_step_token_fold(
        source,
        _evidence(),
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
    )
    body_calls = []
    emitted = []

    def body_step(embedding: torch.Tensor) -> torch.Tensor:
        body_calls.append(embedding.clone())
        return embedding + 2.0

    result = fold.semantic_step(
        body_step,
        donor_id=4,
        target_span_emitter=emitted.append,
    )
    assert result.target_token_ids == (2, 3, 1)
    assert emitted == [(2, 3, 1)]
    assert len(body_calls) == 1
    assert torch.equal(body_calls[0], result.semantic_embedding)
    assert torch.equal(result.body_output, result.semantic_embedding + 2.0)
    assert result.receipt["target_span_length"] == 3
    assert result.receipt["body_semantic_steps_advanced"] == 1
    assert result.receipt["target_span_tokens_used_as_body_steps"] == 0

    scores = torch.full((4,), -10.0, dtype=torch.float64)
    scores[0] = 5.0
    chosen = fold.semantic_step(body_step, target_scores=scores)
    expected_id = int(
        fold.compute_all_donor_logits_from_target_scores(scores).argmax().item()
    )
    assert chosen.donor_id == expected_id
    assert len(body_calls) == 2


class _TiedChild(torch.nn.Module):
    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        target_rows, hidden_size = table.shape
        self.embedding = torch.nn.Embedding(target_rows, hidden_size, dtype=table.dtype)
        self.lm_head = torch.nn.Linear(
            hidden_size,
            target_rows,
            bias=False,
            dtype=table.dtype,
        )
        self.lm_head.weight = self.embedding.weight
        with torch.no_grad():
            self.embedding.weight.copy_(table)


def test_parameterless_runtime_borrows_one_live_tied_child_table_without_alias_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _design = _source_table()
    fold = compile_lfm2_semantic_step_token_fold(
        source,
        _evidence(),
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
    )
    child = _TiedChild(fold.target_table.detach())
    runtime = fold.to_parameterless_runtime()

    assert not isinstance(runtime, torch.nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "target_table")
    assert sum(parameter.numel() for parameter in child.parameters()) == 4 * 3
    unique_parameters = {id(parameter) for parameter in child.parameters()}
    assert len(unique_parameters) == 1
    duplicate_bindings = tuple(child.named_parameters(remove_duplicate=False))
    assert len(duplicate_bindings) == 2
    assert duplicate_bindings[0][1] is duplicate_bindings[1][1]
    assert runtime.as_dict()["owned_learned_parameter_count"] == 0
    assert runtime.as_dict()["owns_or_registers_target_table"] is False
    assert runtime.as_dict()["parameter_alias_count"] == 0

    expected = fold.reconstruct_selected_donor_embeddings([4, 2])
    body_calls = []
    original_tensor_sha256 = semantic_module._tensor_sha256
    original_sparse_design_matrix = semantic_module._sparse_design_matrix
    full_table_hash_calls = 0
    routing_cache_builds = 0

    def counted_tensor_sha256(value):
        nonlocal full_table_hash_calls
        if tuple(value.shape) == (4, 3):
            full_table_hash_calls += 1
        return original_tensor_sha256(value)

    def counted_sparse_design_matrix(*args, **kwargs):
        nonlocal routing_cache_builds
        routing_cache_builds += 1
        return original_sparse_design_matrix(*args, **kwargs)

    monkeypatch.setattr(semantic_module, "_tensor_sha256", counted_tensor_sha256)
    monkeypatch.setattr(
        semantic_module,
        "_sparse_design_matrix",
        counted_sparse_design_matrix,
    )
    with runtime.authenticated_live_table_session(child.embedding.weight) as session:
        assert session.full_hash_verification_count == 1
        observed = runtime.reconstruct_selected_donor_embeddings(
            session,
            [4, 2],
            maximum_rows_per_chunk=1,
        )
        assert torch.allclose(observed, expected, atol=1e-12, rtol=0)
        hidden = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float64)
        target_scores = runtime.score_target_tokens(session, hidden)
        assert torch.equal(target_scores, fold.score_target_tokens(hidden))
        assert torch.allclose(
            runtime.compute_all_donor_logits_from_target_scores(
                session,
                target_scores,
                maximum_donor_rows_per_chunk=2,
            ),
            fold.compute_all_donor_logits_from_target_scores(
                target_scores,
                maximum_donor_rows_per_chunk=2,
            ),
            atol=1e-12,
            rtol=0,
        )
        for _ in range(65):
            runtime.semantic_step(
                session,
                lambda value: value,
                target_scores=target_scores,
                maximum_donor_rows_per_chunk=2,
            )
        result = runtime.semantic_step(
            session,
            lambda value: body_calls.append(value.clone()) or value,
            donor_id=4,
        )
        assert session.full_hash_verification_count == 1
    assert session.full_hash_verification_count == 2
    assert full_table_hash_calls == 2
    assert routing_cache_builds == 1
    assert session.as_dict()["full_hash_verification_count"] == 2
    assert session.as_dict()["live_table_cloned_or_registered"] is False
    assert session.as_dict()["routing_cache_build_count"] == 1
    assert session.as_dict()["routing_cache_use_count"] >= 132
    assert session.as_dict()["routing_tensor_rebuild_count_inside_session"] == 0
    assert session.as_dict()["routing_cache_released_on_close"] is True
    assert result.target_token_ids == (2, 3, 1)
    assert len(body_calls) == 1

    with torch.no_grad():
        child.embedding.weight[0, 0] += 1.0
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="Live child table"):
        runtime.reconstruct_selected_donor_embeddings(
            child.embedding.weight,
            [0],
        )


def test_hard_direct_anchor_factory_bypasses_solve_and_uses_existing_child_table() -> (
    None
):
    evidence = _evidence()
    anchor_table = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 2.0, -0.5],
            [1.5, -1.0, 0.25],
            [-0.5, 0.75, 1.0],
        ],
        dtype=torch.float32,
    )
    child = _TiedChild(anchor_table)
    runtime = bind_lfm2_semantic_step_hard_anchor_runtime(
        evidence,
        child.embedding.weight,
        positional_design="uniform",
    )
    assert runtime.table_origin == "hard-direct-anchor-table"
    assert runtime.as_dict()["least_squares_or_ridge_solve_calls"] == 0
    assert runtime.as_dict()["owned_learned_parameter_count"] == 0
    assert sum(parameter.numel() for parameter in child.parameters()) == 4 * 3
    design = materialize_lfm2_semantic_step_design_matrix(
        evidence,
        positional_design="uniform",
        dtype=torch.float32,
    )
    reconstructed = runtime.reconstruct_selected_donor_embeddings(
        child.embedding.weight,
        [0, 2, 4],
    )
    assert torch.equal(
        reconstructed,
        (design @ child.embedding.weight.detach()).index_select(
            0, torch.tensor([0, 2, 4])
        ),
    )


def test_dual_channel_sparse_code_is_the_only_semantic_route_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [2], [0, 1], [1]],
        target_row_count=3,
    )
    table = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [2.0, -1.0]],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 1, 2, 3, 5, 7], dtype=torch.int64)
    target_ids = torch.tensor([0, 1, 2, 0, 2, 0, 1], dtype=torch.int64)
    coefficients = torch.tensor(
        [1.0, 1.0, 1.0, 0.25, 0.75, -0.5, 1.5],
        dtype=torch.float32,
    )
    runtime = bind_lfm2_sparse_semantic_code_runtime(
        evidence,
        table,
        semantic_code_offsets=offsets,
        semantic_code_target_ids=target_ids,
        semantic_code_coefficients=coefficients,
        target_to_donor=torch.tensor([0, 1, 2], dtype=torch.int64),
    )
    receipt = runtime.as_dict()
    assert receipt["semantic_routing_mode"] == "dual-channel-sparse-semantic-code"
    assert receipt["routing_metadata_is_integer_only"] is False
    assert receipt["continuous_semantic_code_coefficient_count"] == 7
    assert receipt["owned_fixed_continuous_state_count"] == 7
    assert receipt["semantic_code_coefficients_counted_as_child_state"] is True

    # Donor row four has singleton spelling S=[1], but it is not a protected
    # direct anchor.  Its semantic code C=[-0.5*row0 + 1.5*row1] is independent.
    assert runtime.span_for_donor(4) == (1,)
    dense_code = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.0, 0.75],
            [-0.5, 1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_embeddings = dense_code @ table
    monkeypatch.setattr(
        semantic_module,
        "_sparse_design_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dual-channel runtime consulted spelling S for semantics")
        ),
    )
    with runtime.authenticated_live_table_session(table) as first:
        observed = runtime.reconstruct_selected_donor_embeddings(
            first,
            [3, 4],
        )
        target_scores = runtime.score_target_tokens(
            first,
            torch.tensor([[0.5, -0.25]], dtype=torch.float32),
        )
        donor_logits = runtime.compute_all_donor_logits_from_target_scores(
            first,
            target_scores,
        )
    assert torch.equal(observed, expected_embeddings[[3, 4]])
    assert torch.allclose(
        donor_logits,
        torch.tensor([[0.5, -0.25]], dtype=torch.float32) @ expected_embeddings.T,
    )
    assert first.as_dict()["routing_cache_build_count"] == 1
    assert first.as_dict()["routing_cache_hit"] is False
    assert first.as_dict()["routing_cache_released_on_close"] is False

    with runtime.authenticated_live_table_session(table) as second:
        runtime.reconstruct_selected_donor_embeddings(second, [4])
    assert second.as_dict()["routing_cache_build_count"] == 0
    assert second.as_dict()["routing_cache_hit"] is True
    assert runtime.prepared_semantic_code_cache_build_count == 1
    assert runtime.prepared_semantic_code_cache_hit_count == 1
    assert runtime.prepared_semantic_code_cache_use_count == 2


def test_dual_channel_sparse_code_requires_fp32_and_exact_protected_rows() -> None:
    evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [0, 1]],
        target_row_count=2,
    )
    table = torch.eye(2, dtype=torch.float32)
    offsets = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
    target_ids = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    coefficients = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float32)
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="FP32"):
        bind_lfm2_sparse_semantic_code_runtime(
            evidence,
            table,
            semantic_code_offsets=offsets,
            semantic_code_target_ids=target_ids,
            semantic_code_coefficients=coefficients.to(torch.float16),
            target_to_donor=[0, 1],
        )
    broken = coefficients.clone()
    broken[0] = 0.5
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="exact one-hot"):
        bind_lfm2_sparse_semantic_code_runtime(
            evidence,
            table,
            semantic_code_offsets=offsets,
            semantic_code_target_ids=target_ids,
            semantic_code_coefficients=broken,
            target_to_donor=[0, 1],
        )


def test_low_precision_runtime_routes_in_fp32_and_publishes_embeddings_in_table_dtype() -> (
    None
):
    evidence = _evidence()
    table = torch.arange(12, dtype=torch.float16).reshape(4, 3) / 8
    runtime = bind_lfm2_semantic_step_hard_anchor_runtime(
        evidence,
        table,
        positional_design="linear-forward",
    )
    with runtime.authenticated_live_table_session(table) as session:
        embeddings = runtime.reconstruct_selected_donor_embeddings(
            session,
            [2, 4],
        )
        logits = runtime.compute_all_donor_logits_from_target_scores(
            session,
            torch.ones(4, dtype=torch.float16),
        )
        assert embeddings.dtype == torch.float16
        assert logits.dtype == torch.float32
    receipt = session.as_dict()
    assert receipt["routing_weight_storage_dtype"] == "float32"
    assert receipt["routing_accumulation_dtype"] == "float32"
    assert receipt["semantic_embedding_publication_dtype"] == "float16"
    assert receipt["donor_logits_publication_dtype"] == "float32"
    assert receipt["low_precision_routing_accumulation_widened_to_fp32"] is True


def test_defined_donor_rows_may_be_smaller_than_model_capacity() -> None:
    source, _design = _source_table()
    padded_source = torch.cat(
        (
            source,
            torch.tensor(
                [[1000.0, -1000.0, 500.0], [-700.0, 900.0, -300.0]],
                dtype=source.dtype,
            ),
        ),
        dim=0,
    )
    base = compile_lfm2_semantic_step_token_fold(
        source,
        _evidence(),
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
    )
    padded = compile_lfm2_semantic_step_token_fold(
        padded_source,
        _evidence(),
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
    )
    assert torch.equal(base.target_table, padded.target_table)
    assert padded.as_dict()["donor_model_row_capacity"] == 7
    assert padded.as_dict()["donor_row_count"] == 5
    assert padded.as_dict()["undefined_donor_row_count"] == 2
    scores = torch.arange(4, dtype=torch.float64)
    logits = padded.compute_all_donor_logits_from_target_scores(scores)
    assert logits.shape == (5,)


def test_solved_hidden_chunks_are_sealed_published_and_reused_without_pcg() -> None:
    source, _design = _source_table()
    checkpoints: list[Lfm2SemanticStepSolvedChunkCheckpoint] = []
    first = compile_lfm2_semantic_step_token_fold(
        source,
        _evidence(),
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
        solved_chunk_publisher=checkpoints.append,
    )
    assert len(checkpoints) == 3
    assert first.as_dict()["solver_computed_chunk_count"] == 3
    assert first.as_dict()["checkpoint_publisher_invocation_count"] == 3
    by_bounds = {
        (
            checkpoint.as_dict()["hidden_start"],
            checkpoint.as_dict()["hidden_stop"],
        ): checkpoint
        for checkpoint in checkpoints
    }
    requests = []

    def loader(request):
        requests.append(dict(request))
        return by_bounds[(request["hidden_start"], request["hidden_stop"])]

    second = compile_lfm2_semantic_step_token_fold(
        source,
        _evidence(),
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
        solved_chunk_loader=loader,
    )
    assert torch.equal(first.target_table, second.target_table)
    assert len(requests) == 3
    assert second.as_dict()["solver_computed_chunk_count"] == 0
    assert second.as_dict()["checkpoint_reused_chunk_count"] == 3
    assert second.as_dict()["checkpoint_loader_invocation_count"] == 3
    assert all(
        report["chunk_source"] == "reused-authenticated-checkpoint"
        for report in second.as_dict()["solver_chunk_reports"]
    )


def test_positive_donor_row_weights_strengthen_anchor_without_dropping_rows() -> None:
    evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [0, 1], [1]],
        target_row_count=2,
    )
    source = torch.tensor([[10.0], [0.0], [0.0]], dtype=torch.float64)
    baseline = compile_lfm2_semantic_step_token_fold(
        source,
        evidence,
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
    )
    weighted = compile_lfm2_semantic_step_token_fold(
        source,
        evidence,
        donor_row_weights=torch.tensor([1000.0, 1.0, 1.0]),
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
    )
    baseline_anchor_error = abs(
        float(baseline.reconstruct_selected_donor_embeddings([0])[0, 0]) - 10.0
    )
    weighted_anchor_error = abs(
        float(weighted.reconstruct_selected_donor_embeddings([0])[0, 0]) - 10.0
    )
    assert weighted_anchor_error < baseline_anchor_error * 0.01
    receipt = weighted.as_dict()
    assert receipt["positive_weighted_donor_row_count"] == 3
    assert receipt["all_donor_rows_retained_in_weighted_solve"] is True
    assert receipt["minimum_donor_row_weight"] == 1.0
    assert receipt["maximum_donor_row_weight"] == 1000.0
    assert (
        receipt["donor_row_weights_sha256"]
        != baseline.as_dict()["donor_row_weights_sha256"]
    )
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="strictly positive"):
        compile_lfm2_semantic_step_token_fold(
            source,
            evidence,
            donor_row_weights=[1.0, 0.0, 1.0],
        )


def test_suffix_zero_width_unresolved_and_tampered_seals_fail_closed() -> None:
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="only exact"):
        seal_lfm2_semantic_step_span_evidence(
            [
                {"donor_id": 0, "kind": "append", "target_token_ids": [0]},
                {
                    "donor_id": 1,
                    "kind": "suffix-rewrite",
                    "reason": "target-suffix-rewrite",
                    "candidate_target_token_ids": [1],
                },
            ],
            target_row_count=2,
        )
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="zero-width"):
        seal_lfm2_semantic_step_span_evidence(
            [[0], []],
            target_row_count=2,
        )

    source, _design = _source_table()
    evidence = _evidence()
    tampered_evidence = deepcopy(evidence)
    tampered_evidence["spans"][0]["target_token_ids"] = [1]
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="seal"):
        compile_lfm2_semantic_step_token_fold(source, tampered_evidence)

    fold = compile_lfm2_semantic_step_token_fold(
        source,
        evidence,
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
    )
    tampered_receipt = fold.as_dict()
    tampered_receipt["learned_parameter_count"] += 1
    tampered_receipt["result_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered_receipt.items()
            if key != "result_sha256"
        }
    )
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="binding"):
        Lfm2SemanticStepTokenFold(
            fold.target_table.detach(),
            fold.span_offsets,
            fold.span_target_ids,
            tampered_receipt,
        )
    changed_table = fold.target_table.detach().clone()
    changed_table[0, 0] += 1.0
    with pytest.raises(Lfm2SemanticStepTokenFoldError, match="binding"):
        Lfm2SemanticStepTokenFold(
            changed_table,
            fold.span_offsets,
            fold.span_target_ids,
            fold.as_dict(),
        )


class _NumericModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))


def test_numeric_execution_policy_controls_solve_and_publication_placements() -> None:
    source, _design = _source_table()
    policy = resolve_numeric_execution_policy(
        source_models={"donor": _NumericModel()},
        target_model=_NumericModel(),
        request=NumericExecutionRequest(
            solve_device="cpu",
            solve_dtype="float64",
            publication_device="cpu",
            publication_dtype="float32",
        ),
    )
    fold = compile_lfm2_semantic_step_token_fold(
        source,
        _evidence(),
        positional_design="linear-forward",
        ridge=1e-8,
        maximum_solver_iterations=64,
        solver_tolerance=1e-12,
        hidden_columns_per_chunk=1,
        numeric_execution_policy=policy,
    )
    receipt = fold.as_dict()
    assert fold.target_table.dtype == torch.float32
    assert receipt["numeric_execution_policy_sha256"] == policy.policy_sha256
    assert receipt["solve_placement"] == {
        "device": "cpu",
        "dtype": "float64",
    }
    assert receipt["publication_placement"] == {
        "device": "cpu",
        "dtype": "float32",
    }
