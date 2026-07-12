from __future__ import annotations

from neural_forge.native_multimodal import COLORS, DIGITS, QUESTIONS, native_vocab


def test_native_vocab_contains_questions_answers_and_specials() -> None:
    vocab = native_vocab()
    for token in ["<pad>", "<bos>", "<sep>", "<eos>"]:
        assert token in vocab
    for color in COLORS:
        assert color in vocab
    for digit in DIGITS:
        assert digit in vocab
    for question, _ in QUESTIONS:
        for token in question.split():
            assert token in vocab


def test_native_vocab_ids_are_dense() -> None:
    vocab = native_vocab()
    assert sorted(vocab.values()) == list(range(len(vocab)))
