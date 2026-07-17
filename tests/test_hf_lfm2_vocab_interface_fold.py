from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_factorized_vocab as factorized_vocab  # noqa: E402
from neural_forge.hf_lfm2_vocab_interface_fold import (  # noqa: E402
    Lfm2VocabularyInterfaceFoldError,
    audit_non_vocabulary_snapshot,
    exact_model_parameter_counts,
    install_lfm2_tied_vocabulary_pair,
    solve_tied_vocabulary_target,
)


class _TinyTiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(17, 8, dtype=torch.float64)
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8, bias=False, dtype=torch.float64)]
        )
        self.model.register_buffer(
            "runtime_buffer",
            torch.arange(8, dtype=torch.float64),
        )
        self.lm_head = torch.nn.Linear(8, 17, bias=False, dtype=torch.float64)
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value


@pytest.mark.parametrize(
    ("source", "fraction", "expected_rank", "expected_total"),
    [
        (206_726_912, 0.15, 835, 195_195_648),
        (206_726_912, 0.20, 663, 183_747_328),
        (206_726_912, 0.25, 490, 172_232_448),
        (206_726_912, 0.35, 145, 149_269_248),
        (229_693_184, 0.15, 490, 195_198_720),
        (229_693_184, 0.20, 318, 183_750_400),
    ],
)
def test_exact_target_solver_hits_large_parent_relative_frontier(
    source: int,
    fraction: float,
    expected_rank: int,
    expected_total: int,
) -> None:
    target = solve_tied_vocabulary_target(
        original_parent_unique_parameters=229_693_184,
        source_unique_parameters=source,
        vocab_size=65_536,
        hidden_size=1_024,
        total_compression_fraction=fraction,
    )

    assert target.rank == expected_rank
    assert target.physical_target_unique_parameters == expected_total
    assert target.physical_target_unique_parameters <= (
        target.requested_target_unique_parameters
    )
    assert target.physical_total_compression_fraction >= fraction


def test_one_svd_analysis_reuses_exact_rank_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyTiedModel()
    calls = 0
    original = factorized_vocab._analyze_weight

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(factorized_vocab, "_analyze_weight", counted)
    analysis = factorized_vocab.analyze_tied_vocabulary_modules(
        model.get_input_embeddings(),
        model.get_output_embeddings(),
    )
    pairs = tuple(analysis.factorize(rank) for rank in (7, 5, 2))

    assert calls == 1
    assert tuple(pair.embedding.rank for pair in pairs) == (7, 5, 2)
    assert all(pair.tied for pair in pairs)
    assert all(
        pair.embedding.vocab_factor is pair.lm_head.vocab_factor
        and pair.embedding.hidden_factor is pair.lm_head.hidden_factor
        for pair in pairs
    )


def test_physical_install_preserves_every_non_vocab_object_and_byte() -> None:
    model = _TinyTiedModel()
    source_counts = exact_model_parameter_counts(model)
    layer = model.model.layers[0]
    buffer = model.model.runtime_buffer
    analysis = factorized_vocab.analyze_tied_vocabulary_modules(
        model.get_input_embeddings(),
        model.get_output_embeddings(),
    )
    pair = analysis.factorize(3)

    installation = install_lfm2_tied_vocabulary_pair(model, pair)

    assert installation.active is True
    assert model.get_input_embeddings() is pair.embedding
    assert model.get_output_embeddings() is pair.lm_head
    assert model.model.layers[0] is layer
    assert model.model.runtime_buffer is buffer
    assert installation.installation_audit[
        "dense_vocabulary_weight_retained"
    ] is False
    assert installation.installation_audit[
        "non_vocabulary_preservation"
    ]["exact"] is True
    assert exact_model_parameter_counts(model)["unique"] == (
        source_counts["unique"] - 17 * 8 + 3 * (17 + 8)
    )


def test_rollback_restores_exact_dense_tie_counts_and_non_vocab_state() -> None:
    model = _TinyTiedModel()
    original_embedding = model.get_input_embeddings()
    original_head = model.get_output_embeddings()
    source_counts = exact_model_parameter_counts(model)
    analysis = factorized_vocab.analyze_tied_vocabulary_modules(
        original_embedding,
        original_head,
    )
    installation = install_lfm2_tied_vocabulary_pair(
        model,
        analysis.factorize(4),
    )

    assert installation.rollback() is True
    assert installation.active is False
    assert installation.rollback() is False
    assert model.get_input_embeddings() is original_embedding
    assert model.get_output_embeddings() is original_head
    assert original_embedding.weight is original_head.weight
    assert exact_model_parameter_counts(model) == source_counts


def test_non_vocab_tamper_is_detected_and_rollback_fails_closed() -> None:
    model = _TinyTiedModel()
    analysis = factorized_vocab.analyze_tied_vocabulary_modules(
        model.get_input_embeddings(),
        model.get_output_embeddings(),
    )
    installation = install_lfm2_tied_vocabulary_pair(
        model,
        analysis.factorize(3),
    )
    with torch.no_grad():
        model.model.layers[0].weight[0, 0].add_(1.0)

    with pytest.raises(
        Lfm2VocabularyInterfaceFoldError,
        match="non-vocabulary",
    ):
        audit_non_vocabulary_snapshot(
            model,
            installation.non_vocabulary_snapshot,
        )
    with pytest.raises(
        Lfm2VocabularyInterfaceFoldError,
        match="non-vocabulary",
    ):
        installation.rollback()


def test_untied_source_or_pair_fails_before_mutation() -> None:
    model = _TinyTiedModel()
    analysis = factorized_vocab.analyze_tied_vocabulary_modules(
        model.get_input_embeddings(),
        model.get_output_embeddings(),
    )
    pair = analysis.factorize(3)
    model.lm_head = torch.nn.Linear(8, 17, bias=False, dtype=torch.float64)

    with pytest.raises(
        Lfm2VocabularyInterfaceFoldError,
        match="not the same Parameter",
    ):
        install_lfm2_tied_vocabulary_pair(model, pair)
