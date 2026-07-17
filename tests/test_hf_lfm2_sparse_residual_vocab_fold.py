from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_factorized_vocab import (  # noqa: E402
    analyze_tied_vocabulary_modules,
)
from neural_forge.hf_lfm2_vocab_interface_fold import (  # noqa: E402
    Lfm2VocabularyInterfaceFoldError,
    audit_non_vocabulary_snapshot,
    exact_model_parameter_counts,
    install_lfm2_sparse_residual_vocabulary_pair,
)
from neural_forge.hf_sparse_residual_vocab import (  # noqa: E402
    build_sparse_residual_vocabulary_pair,
)


class _TinyTiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(
            17,
            8,
            dtype=torch.float64,
        )
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8, bias=False, dtype=torch.float64)]
        )
        self.model.register_buffer(
            "runtime_buffer",
            torch.arange(8, dtype=torch.float64),
        )
        self.lm_head = torch.nn.Linear(
            8,
            17,
            bias=False,
            dtype=torch.float64,
        )
        self.lm_head.weight = self.model.embed_tokens.weight
        self.raise_next_output_install = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        if self.raise_next_output_install:
            self.raise_next_output_install = False
            raise KeyboardInterrupt("synthetic BaseException")
        self.lm_head = value


def _pair(model: _TinyTiedModel):
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    analysis = analyze_tied_vocabulary_modules(embedding, head)
    return build_sparse_residual_vocabulary_pair(
        analysis,
        embedding,
        head,
        rank=3,
        selected_token_ids=torch.tensor([0, 3, 9], dtype=torch.long),
    )


def test_sparse_runtime_installs_with_exact_accounting_and_rolls_back() -> None:
    model = _TinyTiedModel()
    original_embedding = model.get_input_embeddings()
    original_head = model.get_output_embeddings()
    source_counts = exact_model_parameter_counts(model)
    pair = _pair(model)

    installation = install_lfm2_sparse_residual_vocabulary_pair(model, pair)

    assert installation.installation_audit["sparse_residual_runtime"] is True
    assert installation.installation_audit[
        "non_vocabulary_preservation"
    ]["exact"] is True
    assert model.get_input_embeddings() is pair.embedding
    assert model.get_output_embeddings() is pair.lm_head
    assert exact_model_parameter_counts(model)["unique"] == (
        source_counts["unique"] - 17 * 8 + pair.unique_parameter_count
    )
    assert installation.rollback() is True
    assert model.get_input_embeddings() is original_embedding
    assert model.get_output_embeddings() is original_head
    assert original_embedding.weight is original_head.weight
    assert exact_model_parameter_counts(model) == source_counts


def test_sparse_install_catches_baseexception_and_restores_source_surface() -> None:
    model = _TinyTiedModel()
    original_embedding = model.get_input_embeddings()
    original_head = model.get_output_embeddings()
    source_counts = exact_model_parameter_counts(model)
    pair = _pair(model)
    model.raise_next_output_install = True

    with pytest.raises(
        Lfm2VocabularyInterfaceFoldError,
        match="Could not install",
    ):
        install_lfm2_sparse_residual_vocabulary_pair(model, pair)

    assert model.get_input_embeddings() is original_embedding
    assert model.get_output_embeddings() is original_head
    assert exact_model_parameter_counts(model) == source_counts


def test_sparse_install_nonvocabulary_tamper_is_detected() -> None:
    model = _TinyTiedModel()
    installation = install_lfm2_sparse_residual_vocabulary_pair(
        model,
        _pair(model),
    )
    with torch.no_grad():
        model.model.runtime_buffer[0].add_(1.0)

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
