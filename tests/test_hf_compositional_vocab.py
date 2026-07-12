from __future__ import annotations

import io

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_compositional_vocab import (  # noqa: E402
    CompositionalEmbedding,
    CompositionalLMHead,
    TiedCompositionalVocabulary,
    fit_tied_compositional_vocabulary,
    serialized_tensor_state_bytes,
    split_hidden_subspaces,
)


def _matrix(rows: int, columns: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator, dtype=torch.float64)


def _clustered_product_fixture(
    *,
    codebook_size: int = 4,
    subspaces: int = 3,
    subspace_width: int = 3,
    noise: float = 0.01,
    seed: int = 100,
):
    generator = torch.Generator().manual_seed(seed)
    vocab_size = codebook_size**subspaces
    centers = []
    for subspace in range(subspaces):
        base = torch.randn(
            codebook_size,
            subspace_width,
            generator=generator,
            dtype=torch.float64,
        )
        centers.append(3.0 * base + 2.0 * subspace)
    token_ids = torch.arange(vocab_size, dtype=torch.int64)
    assignment_columns = []
    divisor = 1
    for _ in range(subspaces):
        assignment_columns.append((token_ids // divisor).remainder(codebook_size))
        divisor *= codebook_size
    assignments = torch.stack(assignment_columns, dim=1)
    parts = [
        centers[index].index_select(0, assignments[:, index])
        for index in range(subspaces)
    ]
    weight = torch.cat(parts, dim=1)
    weight += noise * torch.randn(weight.shape, generator=generator, dtype=weight.dtype)
    return weight, assignments


def test_full_codebook_exactness_and_separate_storage_accounting() -> None:
    weight = _matrix(7, 10, 11)
    assignments = torch.arange(7, dtype=torch.int64).unsqueeze(1).repeat(1, 3)
    vocabulary = fit_tied_compositional_vocabulary(
        weight,
        num_subspaces=3,
        codebook_size=7,
        assignments=assignments,
        token_chunk_size=3,
        codeword_chunk_size=2,
    )

    assert split_hidden_subspaces(10, 3) == ((0, 4), (4, 7), (7, 10))
    torch.testing.assert_close(
        vocabulary.materialize_weight(), weight, atol=0.0, rtol=0.0
    )
    assert vocabulary.registry.subspace_widths == (4, 3, 3)
    assert torch.equal(
        vocabulary.registry.assignments,
        assignments.to(torch.int32),
    )
    for subspace, (start, end) in enumerate(split_hidden_subspaces(10, 3)):
        source_indices = vocabulary.registry.medoid_source_indices[subspace].long()
        torch.testing.assert_close(
            vocabulary.registry.codebooks[subspace],
            weight[:, start:end].index_select(0, source_indices),
            atol=0.0,
            rtol=0.0,
        )

    report = vocabulary.report
    assert report.learned_parameter_count == 7 * 10
    assert report.dense_equivalent_parameter_count == 7 * 10
    assert report.assignment_index_bytes == 7 * 3 * 4
    assert report.medoid_source_index_bytes == 3 * 7 * 4
    assert report.index_buffer_bytes > report.assignment_index_bytes
    assert report.tensor_payload_bytes == (
        report.learned_parameter_bytes
        + report.index_buffer_bytes
        + report.other_tensor_buffer_bytes
    )
    stream = io.BytesIO()
    torch.save(vocabulary.state_dict(), stream)
    assert report.actual_serialized_bytes == stream.tell()
    assert report.actual_serialized_bytes == serialized_tensor_state_bytes(
        vocabulary.state_dict()
    )
    assert report.actual_serialized_bytes > report.tensor_payload_bytes


def test_compressed_clustered_reconstruction_beats_zero_and_best_rank_one() -> None:
    weight, assignments = _clustered_product_fixture(seed=20)
    vocabulary = fit_tied_compositional_vocabulary(
        weight,
        num_subspaces=3,
        codebook_size=4,
        assignments=assignments,
        token_chunk_size=11,
        codeword_chunk_size=2,
    )
    reconstructed = vocabulary.materialize_weight().detach()
    product_error = (weight - reconstructed).square().sum()
    zero_error = weight.square().sum()
    left, singular_values, right_t = torch.linalg.svd(weight, full_matrices=False)
    rank_one = (left[:, :1] * singular_values[:1].unsqueeze(0)).matmul(right_t[:1])
    rank_one_error = (weight - rank_one).square().sum()

    assert product_error < rank_one_error
    assert product_error < zero_error
    assert vocabulary.report.learned_parameter_count == 3 * 4 * 3
    assert vocabulary.report.learned_parameter_count < (
        vocabulary.report.dense_equivalent_parameter_count
    )
    assert vocabulary.diagnostics.squared_reconstruction_error == pytest.approx(
        float(product_error.item()), rel=1e-12, abs=1e-12
    )


def test_tied_embedding_and_head_match_one_logical_weight_without_dense_forward(
    monkeypatch,
) -> None:
    weight, assignments = _clustered_product_fixture(
        codebook_size=3,
        subspaces=2,
        subspace_width=4,
        seed=30,
    )
    bias = _matrix(1, weight.shape[0], 31).squeeze(0)
    vocabulary = fit_tied_compositional_vocabulary(
        weight,
        num_subspaces=2,
        codebook_size=3,
        assignments=assignments,
        bias=bias,
        token_chunk_size=7,
        codeword_chunk_size=2,
    )
    logical_weight = vocabulary.materialize_weight().detach()
    token_ids = torch.tensor([[0, 2, 4], [8, 1, 3]], dtype=torch.int64)
    hidden_states = _matrix(12, weight.shape[1], 32).reshape(2, 6, -1)

    def fail_dense_materialization():
        raise AssertionError("runtime forward attempted dense VxH materialization")

    monkeypatch.setattr(
        vocabulary.registry,
        "materialize_weight",
        fail_dense_materialization,
    )
    torch.testing.assert_close(
        vocabulary.embedding(token_ids),
        torch.nn.functional.embedding(token_ids, logical_weight),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        vocabulary.lm_head(hidden_states),
        torch.nn.functional.linear(hidden_states, logical_weight, bias),
        atol=1e-12,
        rtol=1e-12,
    )
    assert isinstance(vocabulary.embedding, CompositionalEmbedding)
    assert isinstance(vocabulary.lm_head, CompositionalLMHead)
    assert vocabulary.embedding.registry is vocabulary.registry
    assert vocabulary.lm_head.registry is vocabulary.registry
    assert list(vocabulary.embedding.parameters()) == []
    assert list(vocabulary.lm_head.parameters()) == []
    assert all(
        name.startswith("registry.") for name, _ in vocabulary.named_parameters()
    )


def test_hash_minibatch_fitting_is_deterministic_and_strictly_bounded() -> None:
    weight = _matrix(97, 11, 40)
    kwargs = {
        "num_subspaces": 4,
        "codebook_size": 7,
        "seed": 123,
        "max_iterations": 3,
        "token_chunk_size": 13,
        "codeword_chunk_size": 3,
        "initialization_minibatch_size": 31,
    }
    first = fit_tied_compositional_vocabulary(weight, **kwargs)
    second = fit_tied_compositional_vocabulary(weight, **kwargs)

    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
    assert first.diagnostics == second.diagnostics
    diagnostics = first.diagnostics
    assert diagnostics.initialization == "hash_minibatch_farthest_first"
    assert diagnostics.deterministic is True
    assert diagnostics.real_medoid_codewords is True
    assert diagnostics.supplied_assignments is False
    assert diagnostics.assignments_refined is True
    assert diagnostics.initialization_sample_sizes == (31, 31, 31, 31)
    assert diagnostics.iterations_run <= 4 * 3
    assert diagnostics.distance_values_computed <= diagnostics.distance_work_bound
    assert diagnostics.max_distance_block_elements <= 13 * 3
    assert (
        diagnostics.max_distance_block_elements
        < diagnostics.largest_vocabulary_by_codebook_matrix_elements
    )
    assert diagnostics.materialized_full_vocabulary_by_codebook_matrix is False
    assert diagnostics.no_full_vocabulary_by_codebook_matrix is True
    assert diagnostics.bounded_work is True
    assert diagnostics.fitting_requires_grad is False

    for subspace, (start, end) in enumerate(split_hidden_subspaces(11, 4)):
        source_indices = first.registry.medoid_source_indices[subspace].long()
        torch.testing.assert_close(
            first.registry.codebooks[subspace],
            weight[:, start:end].index_select(0, source_indices),
            atol=0.0,
            rtol=0.0,
        )


def test_tensor_only_state_roundtrip_preserves_outputs_reports_and_diagnostics() -> (
    None
):
    weight, assignments = _clustered_product_fixture(
        codebook_size=5,
        subspaces=3,
        subspace_width=2,
        seed=50,
    )
    varying_sizes = (3, 4, 5)
    varying_assignments = assignments.clone()
    varying_assignments[:, 0].remainder_(3)
    varying_assignments[:, 1].remainder_(4)
    bias = _matrix(1, weight.shape[0], 51).squeeze(0)
    vocabulary = fit_tied_compositional_vocabulary(
        weight,
        num_subspaces=3,
        codebook_size=varying_sizes,
        assignments=varying_assignments,
        bias=bias,
        token_chunk_size=17,
        codeword_chunk_size=2,
    )
    state = {
        key: value.detach().clone() for key, value in vocabulary.state_dict().items()
    }
    restored = TiedCompositionalVocabulary.from_state_dict(state)
    token_ids = torch.tensor([[0, 11, 42], [77, 3, 9]], dtype=torch.int64)
    hidden_states = _matrix(18, weight.shape[1], 52).reshape(3, 6, -1)

    assert state
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert not any("extra_state" in key for key in state)
    assert restored.state_dict().keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(restored.state_dict()[key], value)
    torch.testing.assert_close(
        restored.embedding(token_ids),
        vocabulary.embedding(token_ids),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        restored.lm_head(hidden_states),
        vocabulary.lm_head(hidden_states),
        atol=0.0,
        rtol=0.0,
    )
    assert restored.report == vocabulary.report
    assert restored.diagnostics == vocabulary.diagnostics
    assert all(parameter.requires_grad for parameter in restored.parameters())


def test_embedding_and_head_accumulate_normal_gradients_into_shared_codebooks() -> None:
    weight, assignments = _clustered_product_fixture(
        codebook_size=3,
        subspaces=2,
        subspace_width=3,
        seed=60,
    )
    embedding = torch.nn.Embedding(
        weight.shape[0], weight.shape[1], dtype=torch.float64
    )
    lm_head = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=True, dtype=torch.float64
    )
    with torch.no_grad():
        embedding.weight.copy_(weight)
        lm_head.weight = embedding.weight
        lm_head.bias.copy_(_matrix(1, weight.shape[0], 61).squeeze(0))
    vocabulary = fit_tied_compositional_vocabulary(
        embedding,
        lm_head,
        num_subspaces=2,
        codebook_size=3,
        assignments=assignments,
        token_chunk_size=5,
        codeword_chunk_size=2,
    )
    assert embedding.weight.grad is None
    assert lm_head.bias.grad is None

    token_ids = torch.arange(weight.shape[0], dtype=torch.int64).reshape(3, -1)
    hidden_states = _matrix(8, weight.shape[1], 62).requires_grad_(True)
    loss = vocabulary.embedding(token_ids).square().mean()
    loss = loss + vocabulary.lm_head(hidden_states).square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert bool(torch.isfinite(hidden_states.grad).all())
    parameters = list(vocabulary.parameters())
    assert parameters
    assert len(parameters) == 3  # two codebooks and one LM-head bias
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
    assert embedding.weight.grad is None
    assert lm_head.bias.grad is None
    assert sum(parameter.numel() for parameter in parameters) == (
        vocabulary.report.learned_parameter_count
    )
