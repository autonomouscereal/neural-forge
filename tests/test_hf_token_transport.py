from __future__ import annotations

from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_token_align import (  # noqa: E402
    TransportAlignmentThresholds,
    build_token_alignment,
)
from neural_forge.hf_token_transport import (  # noqa: E402
    TokenTransportError,
    transport_vocabulary_rows,
)
from neural_forge.hf_transport_math import build_parent_residual_basis  # noqa: E402


class _Tokenizer:
    def __init__(
        self,
        surfaces: list[str],
        *,
        encodings: dict[str, list[int]] | None = None,
    ) -> None:
        self.surfaces = surfaces
        self.encodings = encodings or {}
        self.all_special_ids: list[int] = []

    def __len__(self) -> int:
        return len(self.surfaces)

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[index] for index in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self.encodings[text]


def _coordinates(source_dim: int, target_dim: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(
        source_dim + 8,
        source_dim,
        generator=generator,
        dtype=torch.float64,
    )
    return build_parent_residual_basis(rows, target_dim)


def test_exact_unique_rows_transport_embeddings_head_and_bias() -> None:
    donor = _Tokenizer(["a", " b", "c"])
    target = _Tokenizer(["c", "a", " b"])
    alignment = build_token_alignment(donor, target)
    embedding = torch.arange(15, dtype=torch.float64).reshape(3, 5)
    head = torch.arange(18, dtype=torch.float64).reshape(3, 6)
    bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    embedding_coordinates = _coordinates(5, 3, 1)
    head_coordinates = _coordinates(6, 4, 2)

    result = transport_vocabulary_rows(
        embedding,
        head,
        alignment,
        embedding_coordinates,
        head_coordinates,
        source_lm_head_bias=bias,
        thresholds=TransportAlignmentThresholds.strict(),
    )

    expected_embedding = embedding[[2, 0, 1]] @ embedding_coordinates.decode
    expected_head = head[[2, 0, 1]] @ head_coordinates.decode
    torch.testing.assert_close(result.embedding_weight, expected_embedding)
    torch.testing.assert_close(result.lm_head_weight, expected_head)
    torch.testing.assert_close(result.lm_head_bias, bias[[2, 0, 1]])
    assert result.report["random_rows"] == 0
    assert result.report["exact_token_rows"] == 3
    assert result.report["utf8_span_rows"] == 0


def test_embedding_span_is_explicit_but_head_span_fails_by_default() -> None:
    donor = _Tokenizer(["a", "b"], encodings={"ab": [0, 1]})
    target = _Tokenizer(["ab"])
    alignment = build_token_alignment(donor, target)
    embedding = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    head = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    coordinates = _coordinates(2, 2, 3)
    relaxed = TransportAlignmentThresholds.strict()

    with pytest.raises(TokenTransportError, match="cannot initialize one LM-head row"):
        transport_vocabulary_rows(
            embedding,
            head,
            alignment,
            coordinates,
            coordinates,
            thresholds=relaxed,
        )

    result = transport_vocabulary_rows(
        embedding,
        head,
        alignment,
        coordinates,
        coordinates,
        thresholds=relaxed,
        embedding_span_reduction="mean",
        allow_lm_head_spans=True,
        lm_head_span_reduction="sum",
    )
    torch.testing.assert_close(
        result.embedding_weight,
        embedding.to(torch.float64).mean(dim=0, keepdim=True) @ coordinates.decode,
    )
    torch.testing.assert_close(
        result.lm_head_weight,
        head.to(torch.float64).sum(dim=0, keepdim=True) @ coordinates.decode,
    )
    assert result.report["embedding_span_lengths"] == [2]
    assert result.report["lm_head_span_lengths"] == [2]


def test_ambiguity_and_partial_coverage_remain_fail_closed() -> None:
    ambiguous = build_token_alignment(_Tokenizer(["a", "a"]), _Tokenizer(["a"]))
    coordinates = _coordinates(2, 2, 4)
    weights = torch.ones((2, 2))
    with pytest.raises(ValueError, match="ambiguity_fraction"):
        transport_vocabulary_rows(
            weights,
            weights,
            ambiguous,
            coordinates,
            coordinates,
        )

    partial = build_token_alignment(_Tokenizer(["a"]), _Tokenizer(["a", "b"]))
    with pytest.raises(ValueError, match="coverage_fraction"):
        transport_vocabulary_rows(
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            partial,
            coordinates,
            coordinates,
        )


def test_shape_nonfinite_and_reduction_errors_fail_closed() -> None:
    alignment = build_token_alignment(_Tokenizer(["a"]), _Tokenizer(["a"]))
    coordinates = _coordinates(2, 2, 5)
    with pytest.raises(TokenTransportError, match="embedding rows"):
        transport_vocabulary_rows(
            torch.ones((2, 2)),
            torch.ones((1, 2)),
            alignment,
            coordinates,
            coordinates,
        )
    bad = torch.ones((1, 2))
    bad[0, 0] = torch.nan
    with pytest.raises(TokenTransportError, match="NaN or infinity"):
        transport_vocabulary_rows(
            bad,
            torch.ones((1, 2)),
            alignment,
            coordinates,
            coordinates,
        )
    with pytest.raises(TokenTransportError, match="must be mean or sum"):
        transport_vocabulary_rows(
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            alignment,
            coordinates,
            coordinates,
            embedding_span_reduction="first",  # type: ignore[arg-type]
        )


def test_transport_is_deterministic_detached_and_zero_gradient() -> None:
    alignment = build_token_alignment(_Tokenizer(["a", "b"]), _Tokenizer(["b", "a"]))
    coordinates = _coordinates(3, 2, 6)
    embedding = torch.arange(6, dtype=torch.float64).reshape(2, 3).requires_grad_()
    head = (embedding.detach() + 1.0).requires_grad_()

    first = transport_vocabulary_rows(
        embedding,
        head,
        alignment,
        coordinates,
        coordinates,
        thresholds=TransportAlignmentThresholds.strict(),
    )
    second = transport_vocabulary_rows(
        embedding,
        head,
        alignment,
        coordinates,
        coordinates,
        thresholds=TransportAlignmentThresholds.strict(),
    )

    assert torch.equal(first.embedding_weight, second.embedding_weight)
    assert torch.equal(first.lm_head_weight, second.lm_head_weight)
    assert first.report == second.report
    assert first.embedding_weight.requires_grad is False
    assert first.lm_head_weight.grad_fn is None
    assert embedding.grad is None
    assert head.grad is None
