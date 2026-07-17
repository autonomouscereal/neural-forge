from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_factorized_vocab import (  # noqa: E402
    analyze_tied_vocabulary_modules,
)
from neural_forge.hf_sparse_residual_vocab import (  # noqa: E402
    SparseResidualVocabularyError,
    build_sparse_residual_vocabulary_pair,
    solve_sparse_residual_target,
)


def _dense_tied_modules(
    vocab_size: int = 11,
    hidden_size: int = 6,
):
    torch.manual_seed(19)
    embedding = torch.nn.Embedding(
        vocab_size,
        hidden_size,
        dtype=torch.float64,
    )
    head = torch.nn.Linear(
        hidden_size,
        vocab_size,
        bias=False,
        dtype=torch.float64,
    )
    head.weight = embedding.weight
    return embedding, head


def test_selected_rows_are_exact_and_unselected_rows_are_the_base() -> None:
    embedding, head = _dense_tied_modules()
    analysis = analyze_tied_vocabulary_modules(embedding, head)
    selected = torch.tensor([0, 3, 8], dtype=torch.long)
    pair = build_sparse_residual_vocabulary_pair(
        analysis,
        embedding,
        head,
        rank=2,
        selected_token_ids=selected,
    )

    base = pair.core.materialize_base_weight()
    materialized = pair.core.materialize_weight()
    unselected = torch.tensor(
        [token for token in range(11) if token not in {0, 3, 8}],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        materialized.index_select(0, selected),
        embedding.weight.detach().index_select(0, selected),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        materialized.index_select(0, unselected),
        base.index_select(0, unselected),
        rtol=0.0,
        atol=0.0,
    )
    input_ids = torch.tensor([[0, 1, 3, 8, 10]], dtype=torch.long)
    torch.testing.assert_close(
        pair.embedding(input_ids),
        materialized.index_select(0, input_ids.reshape(-1)).reshape(1, 5, 6),
        rtol=1e-12,
        atol=1e-12,
    )
    hidden = torch.randn(2, 4, 6, dtype=torch.float64)
    torch.testing.assert_close(
        pair.lm_head(hidden),
        torch.nn.functional.linear(hidden, materialized),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pair_owns_one_shared_core_residual_and_integer_id_buffer() -> None:
    embedding, head = _dense_tied_modules()
    analysis = analyze_tied_vocabulary_modules(embedding, head)
    pair = build_sparse_residual_vocabulary_pair(
        analysis,
        embedding,
        head,
        rank=3,
        selected_token_ids=torch.tensor([1, 4, 9], dtype=torch.long),
    )
    report = pair.parameter_report()

    assert pair.embedding.core is pair.lm_head.core is pair.core
    assert (
        pair.embedding.core.residual_rows
        is pair.lm_head.core.residual_rows
        is pair.core.residual_rows
    )
    assert (
        pair.embedding.core.selected_token_ids
        is pair.lm_head.core.selected_token_ids
        is pair.core.selected_token_ids
    )
    assert report["one_shared_core_object"] is True
    assert report["one_shared_residual_parameter_object"] is True
    assert report["one_shared_selected_token_buffer_object"] is True
    assert report["unique_integer_buffer_objects"] == 1
    assert report["integer_buffer_count"] == 3
    assert report["integer_buffer_bytes"] == 3 * 8
    assert report["dense_vocabulary_matrix_retained"] is False
    assert pair.unique_parameter_count == 3 * (11 + 6) + 3 * 6
    assert all(
        tuple(parameter.shape) != (11, 6)
        for parameter in pair.parameters()
    )


def test_sparse_target_solver_spends_remaining_budget_on_exact_rows() -> None:
    target = solve_sparse_residual_target(
        original_parent_unique_parameters=1_000,
        source_unique_parameters=750,
        vocab_size=20,
        hidden_size=10,
        total_compression_fraction=0.35,
        rank=2,
        minimum_selected_rows=2,
    )

    assert target.requested_target_unique_parameters == 650
    assert target.selected_row_budget == 4
    assert target.physical_target_unique_parameters == 650
    assert target.learned_parameter_headroom == 0
    assert target.physical_total_compression_fraction == pytest.approx(0.35)
    assert target.integer_buffer_bytes == 4 * 8


def test_solver_refuses_rank_that_cannot_leave_required_residual_rows() -> None:
    with pytest.raises(
        SparseResidualVocabularyError,
        match="too little budget",
    ):
        solve_sparse_residual_target(
            original_parent_unique_parameters=1_000,
            source_unique_parameters=900,
            vocab_size=20,
            hidden_size=10,
            total_compression_fraction=0.50,
            rank=9,
            minimum_selected_rows=3,
        )


@pytest.mark.parametrize(
    "selected",
    [
        torch.tensor([], dtype=torch.long),
        torch.tensor([2, 1], dtype=torch.long),
        torch.tensor([1, 1], dtype=torch.long),
        torch.tensor([-1, 2], dtype=torch.long),
        torch.tensor([1, 11], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.int32),
    ],
)
def test_invalid_sparse_token_dictionary_fails_closed(selected) -> None:
    embedding, head = _dense_tied_modules()
    analysis = analyze_tied_vocabulary_modules(embedding, head)

    with pytest.raises(SparseResidualVocabularyError):
        build_sparse_residual_vocabulary_pair(
            analysis,
            embedding,
            head,
            rank=2,
            selected_token_ids=selected,
        )


def test_analysis_from_another_same_shape_matrix_is_rejected() -> None:
    embedding, head = _dense_tied_modules()
    other_embedding, other_head = _dense_tied_modules()
    with torch.no_grad():
        other_embedding.weight[0, 0].add_(1.0)
    analysis = analyze_tied_vocabulary_modules(other_embedding, other_head)

    with pytest.raises(
        SparseResidualVocabularyError,
        match="not bound",
    ):
        build_sparse_residual_vocabulary_pair(
            analysis,
            embedding,
            head,
            rank=2,
            selected_token_ids=torch.tensor([0], dtype=torch.long),
        )


def test_analysis_is_reusable_for_an_exact_storage_distinct_matrix_copy() -> None:
    embedding, head = _dense_tied_modules()
    other_embedding, other_head = _dense_tied_modules()
    analysis = analyze_tied_vocabulary_modules(other_embedding, other_head)

    pair = build_sparse_residual_vocabulary_pair(
        analysis,
        embedding,
        head,
        rank=2,
        selected_token_ids=torch.tensor([0, 5], dtype=torch.long),
    )

    torch.testing.assert_close(
        pair.core.materialize_weight()[[0, 5]],
        embedding.weight.detach()[[0, 5]],
    )
