from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_factorized_vocab import (  # noqa: E402
    FactorizationBudgetError,
    FactorizationFiniteError,
    FactorizationRankError,
    FactorizationShapeError,
    FactorizedEmbedding,
    FactorizedLMHead,
    count_unique_parameters,
    exact_parameter_report,
    factorize_embedding_and_lm_head,
    materialize_embedding_and_lm_head,
    plan_factorization,
)


def _matrix(rows: int, columns: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        rows,
        columns,
        generator=generator,
        dtype=torch.float64,
    )


def _embedding(vocab_size: int, hidden_size: int, seed: int):
    module = torch.nn.Embedding(vocab_size, hidden_size, dtype=torch.float64)
    with torch.no_grad():
        module.weight.copy_(_matrix(vocab_size, hidden_size, seed))
    return module


def _head(vocab_size: int, hidden_size: int, seed: int, *, bias: bool = False):
    module = torch.nn.Linear(
        hidden_size,
        vocab_size,
        bias=bias,
        dtype=torch.float64,
    )
    with torch.no_grad():
        module.weight.copy_(_matrix(vocab_size, hidden_size, seed))
        if module.bias is not None:
            module.bias.copy_(_matrix(1, vocab_size, seed + 1).squeeze(0))
    return module


def test_full_rank_embedding_and_head_match_dense_forward() -> None:
    embedding = _embedding(11, 5, 11)
    head = _head(11, 5, 12)
    factorized_embedding = FactorizedEmbedding.from_embedding(embedding)
    factorized_head = FactorizedLMHead.from_linear(head)
    token_ids = torch.tensor([[0, 7, 3], [10, 2, 2]])
    hidden_states = _matrix(12, 5, 13).reshape(2, 6, 5)

    torch.testing.assert_close(
        factorized_embedding(token_ids),
        embedding(token_ids),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        factorized_head(hidden_states),
        head(hidden_states),
        atol=1e-12,
        rtol=1e-12,
    )
    assert factorized_embedding.rank == 5
    assert factorized_head.rank == 5


def test_truncated_svd_strictly_beats_the_zero_matrix() -> None:
    embedding = _embedding(17, 7, 21)
    factorized = FactorizedEmbedding.from_embedding(embedding, rank=2)
    approximation = factorized.materialize_weight()
    approximation_error = torch.linalg.vector_norm(
        embedding.weight.detach() - approximation
    )
    zero_error = torch.linalg.vector_norm(embedding.weight.detach())

    assert approximation_error < zero_error
    assert factorized.plan is not None
    assert factorized.plan.rank == 2
    assert 0.0 < factorized.plan.retained_energy_fraction < 1.0


def test_tied_pair_shares_factor_objects_storage_and_exact_accounting() -> None:
    vocab_size = 13
    hidden_size = 6
    rank = 3
    embedding = _embedding(vocab_size, hidden_size, 31)
    head = torch.nn.Linear(
        hidden_size,
        vocab_size,
        bias=False,
        dtype=torch.float64,
    )
    head.weight = embedding.weight

    pair = factorize_embedding_and_lm_head(embedding, head, rank=rank)
    factor_cost = rank * (vocab_size + hidden_size)
    report = pair.parameter_report()

    assert pair.tied is True
    assert pair.embedding.vocab_factor is pair.lm_head.vocab_factor
    assert pair.embedding.hidden_factor is pair.lm_head.hidden_factor
    assert (
        pair.embedding.vocab_factor.untyped_storage().data_ptr()
        == pair.lm_head.vocab_factor.untyped_storage().data_ptr()
    )
    assert (
        pair.embedding.hidden_factor.untyped_storage().data_ptr()
        == pair.lm_head.hidden_factor.untyped_storage().data_ptr()
    )
    assert pair.unique_parameter_count == factor_cost
    assert pair.plan.unique_parameter_count == factor_cost
    assert report["unique_parameter_count"] == factor_cost
    assert report["logical_parameter_count"] == 2 * factor_cost
    assert report["shared_parameter_references"] == factor_cost
    assert count_unique_parameters(pair.embedding, pair.lm_head) == factor_cost


def test_untied_pair_keeps_independent_factors_and_counts_both() -> None:
    vocab_size = 12
    hidden_size = 5
    rank = 2
    embedding = _embedding(vocab_size, hidden_size, 41)
    head = _head(vocab_size, hidden_size, 42)

    pair = factorize_embedding_and_lm_head(embedding, head, rank=rank)
    expected = 2 * rank * (vocab_size + hidden_size)

    assert pair.tied is False
    assert pair.embedding.vocab_factor is not pair.lm_head.vocab_factor
    assert pair.embedding.hidden_factor is not pair.lm_head.hidden_factor
    assert (
        pair.embedding.vocab_factor.untyped_storage().data_ptr()
        != pair.lm_head.vocab_factor.untyped_storage().data_ptr()
    )
    assert pair.unique_parameter_count == expected
    assert pair.plan.logical_parameter_count == expected
    assert pair.plan.unique_parameter_count == expected


def test_optional_bias_is_preserved_in_forward_and_parameter_counts() -> None:
    vocab_size = 9
    hidden_size = 4
    head = _head(vocab_size, hidden_size, 51, bias=True)
    factorized = FactorizedLMHead.from_linear(head)
    hidden_states = _matrix(15, hidden_size, 52).reshape(3, 5, hidden_size)

    torch.testing.assert_close(
        factorized(hidden_states),
        head(hidden_states),
        atol=1e-12,
        rtol=1e-12,
    )
    assert factorized.bias is not None
    assert factorized.plan is not None
    assert factorized.plan.fixed_parameter_count == vocab_size
    expected = hidden_size * (vocab_size + hidden_size) + vocab_size
    assert count_unique_parameters(factorized) == expected


def test_float64_svd_factors_are_bitwise_deterministic_and_sign_canonical() -> None:
    embedding = _embedding(15, 6, 61)

    first = FactorizedEmbedding.from_embedding(embedding, rank=4)
    second = FactorizedEmbedding.from_embedding(embedding, rank=4)

    assert torch.equal(first.vocab_factor, second.vocab_factor)
    assert torch.equal(first.hidden_factor, second.hidden_factor)
    pivot_rows = torch.argmax(torch.abs(first.vocab_factor), dim=0)
    columns = torch.arange(first.rank)
    assert bool((first.vocab_factor[pivot_rows, columns] > 0.0).all())
    assert first.plan == second.plan


@pytest.mark.parametrize("tied", [False, True])
def test_dense_materialization_roundtrip_preserves_weights_and_tying(
    tied: bool,
) -> None:
    vocab_size = 10
    hidden_size = 4
    embedding = _embedding(vocab_size, hidden_size, 71)
    head = _head(vocab_size, hidden_size, 72, bias=True)
    if tied:
        head.weight = embedding.weight
    pair = factorize_embedding_and_lm_head(embedding, head)

    dense_embedding, dense_head = materialize_embedding_and_lm_head(pair)

    assert type(dense_embedding) is torch.nn.Embedding
    assert type(dense_head) is torch.nn.Linear
    torch.testing.assert_close(
        dense_embedding.weight,
        embedding.weight,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dense_head.weight,
        head.weight,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(dense_head.bias, head.bias, atol=0.0, rtol=0.0)
    assert (dense_embedding.weight is dense_head.weight) is tied
    assert dense_embedding.weight.requires_grad is False
    assert dense_head.bias is not None
    assert dense_head.bias.requires_grad is False


def test_rank_energy_and_budget_plans_have_exact_counts() -> None:
    weight = _matrix(14, 6, 81)
    per_rank = 14 + 6

    rank_plan = plan_factorization(weight, rank=3)
    energy_plan = plan_factorization(weight, energy=0.8)
    budget_plan = plan_factorization(
        weight,
        parameter_budget=2 * per_rank + 5,
        fixed_parameter_count=5,
    )

    assert rank_plan.rank == 3
    assert rank_plan.factor_parameter_count == 3 * per_rank
    assert energy_plan.retained_energy_fraction >= 0.8
    if energy_plan.rank > 1:
        previous_energy = sum(
            value * value
            for value in energy_plan.singular_values[: energy_plan.rank - 1]
        ) / sum(value * value for value in energy_plan.singular_values)
        assert previous_energy < 0.8
    assert budget_plan.rank == 2
    assert budget_plan.total_parameter_count == 2 * per_rank + 5
    assert budget_plan.budget_headroom == 0


def test_pair_budget_accounts_for_tying_and_bias_exactly() -> None:
    vocab_size = 11
    hidden_size = 5
    rank = 2
    embedding = _embedding(vocab_size, hidden_size, 91)
    head = _head(vocab_size, hidden_size, 92, bias=True)
    head.weight = embedding.weight
    budget = rank * (vocab_size + hidden_size) + vocab_size

    pair = factorize_embedding_and_lm_head(
        embedding,
        head,
        parameter_budget=budget,
    )

    assert pair.embedding.rank == rank
    assert pair.lm_head.rank == rank
    assert pair.plan.unique_parameter_count == budget
    assert pair.plan.budget_headroom == 0


def test_factorization_is_detached_frozen_and_accumulates_zero_gradients() -> None:
    embedding = _embedding(9, 4, 101)
    head = _head(9, 4, 102, bias=True)
    pair = factorize_embedding_and_lm_head(embedding, head, rank=3)

    parameters = tuple(pair.parameters())
    assert parameters
    assert all(parameter.requires_grad is False for parameter in parameters)
    assert all(parameter.grad is None for parameter in parameters)
    assert all(parameter.grad_fn is None for parameter in parameters)
    assert embedding.weight.grad is None
    assert head.weight.grad is None
    assert head.bias is not None
    assert head.bias.grad is None

    with torch.no_grad():
        embedded = pair.embedding(torch.tensor([1, 3, 5]))
        logits = pair.lm_head(_matrix(4, 4, 103))
    assert embedded.requires_grad is False
    assert logits.requires_grad is False
    assert all(parameter.grad is None for parameter in parameters)


def test_malformed_nonfinite_rank_and_budget_cases_fail_closed() -> None:
    embedding = _embedding(8, 4, 111)

    with pytest.raises(FactorizationRankError, match="positive integer"):
        FactorizedEmbedding.from_embedding(embedding, rank=0)
    with pytest.raises(FactorizationRankError, match="cannot exceed"):
        FactorizedEmbedding.from_embedding(embedding, rank=5)
    with pytest.raises(FactorizationRankError, match="interval"):
        FactorizedEmbedding.from_embedding(embedding, energy=float("nan"))
    with pytest.raises(FactorizationBudgetError, match="at least"):
        FactorizedEmbedding.from_embedding(embedding, parameter_budget=11)
    with pytest.raises(ValueError, match="at most one"):
        FactorizedEmbedding.from_embedding(embedding, rank=2, energy=0.5)

    embedding.weight.detach()[0, 0] = torch.inf
    with pytest.raises(FactorizationFiniteError, match="NaN or infinity"):
        FactorizedEmbedding.from_embedding(embedding, rank=2)

    mismatched_head = _head(7, 4, 113)
    with pytest.raises(FactorizationShapeError, match="matching vocabulary"):
        factorize_embedding_and_lm_head(_embedding(8, 4, 114), mismatched_head)

    with pytest.raises(FactorizationShapeError, match="columns"):
        FactorizedLMHead(
            torch.ones((8, 3), dtype=torch.float64),
            torch.ones((2, 4), dtype=torch.float64),
        )


def test_exact_report_rejects_missing_or_nonmodule_inputs() -> None:
    with pytest.raises(FactorizationShapeError, match="At least one"):
        exact_parameter_report()
    with pytest.raises(FactorizationShapeError, match="Module"):
        exact_parameter_report(object())  # type: ignore[arg-type]
