from __future__ import annotations

import json
from pathlib import Path

import pytest


tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_tokenizer_prune import HFTokenizerPruneError  # noqa: E402
from neural_forge.hf_tokenizer_prune import prune_hf_tokenizer  # noqa: E402


def _donor_tokenizer():
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: token_id for token_id, token in enumerate(alphabet)}
    merges = [("a", "b"), ("ab", "c"), ("x", "y")]
    for merged in ("ab", "abc", "xy"):
        vocabulary[merged] = len(vocabulary)

    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab=vocabulary, merges=merges)
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def _artifact_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def test_prune_builds_exact_strict_mapping_and_preserves_specials(
    tmp_path: Path,
) -> None:
    donor = _donor_tokenizer()

    result = prune_hf_tokenizer(
        donor,
        tmp_path / "target",
        target_vocab_size=262,
        calibration_texts=["abc abc", "UTF-8: cafe\u0301 \u4e16\u754c"],
    )

    assert len(result.tokenizer) == 262
    assert result.transport_profile.target_to_donor_row_indices == (
        result.selected_donor_token_ids
    )
    assert len(result.transport_fingerprint) == 64
    assert result.tokenizer.special_tokens_map == donor.special_tokens_map
    for target_id, donor_id in enumerate(result.selected_donor_token_ids):
        assert result.tokenizer.convert_ids_to_tokens(
            target_id
        ) == donor.convert_ids_to_tokens(donor_id)
    assert (tmp_path / "target" / "tokenizer.json").is_file()
    assert (tmp_path / "target" / "tokenizer_config.json").is_file()


def test_calibration_selection_keeps_structural_merge_closure(
    tmp_path: Path,
) -> None:
    result = prune_hf_tokenizer(
        _donor_tokenizer(),
        tmp_path / "target",
        target_vocab_size=262,
        calibration_texts=["abc"],
    )
    data = json.loads((tmp_path / "target" / "tokenizer.json").read_text("utf-8"))

    assert result.retained_merges == (("a", "b"), ("ab", "c"))
    assert data["model"]["merges"] == [["a", "b"], ["ab", "c"]]
    assert "ab" in data["model"]["vocab"]
    assert "abc" in data["model"]["vocab"]
    assert "xy" not in data["model"]["vocab"]
    for left, right in data["model"]["merges"]:
        assert left in data["model"]["vocab"]
        assert right in data["model"]["vocab"]
        assert left + right in data["model"]["vocab"]


def test_artifacts_and_selection_are_deterministic(tmp_path: Path) -> None:
    first = prune_hf_tokenizer(
        _donor_tokenizer(),
        tmp_path / "first",
        target_vocab_size=261,
    )
    second = prune_hf_tokenizer(
        _donor_tokenizer(),
        tmp_path / "second",
        target_vocab_size=261,
    )
    explicit_empty = prune_hf_tokenizer(
        _donor_tokenizer(),
        tmp_path / "explicit-empty",
        target_vocab_size=261,
        required_donor_token_ids=(),
    )

    assert first.selected_donor_token_ids == second.selected_donor_token_ids
    assert first.selected_donor_token_ids == explicit_empty.selected_donor_token_ids
    assert first.transport_fingerprint == second.transport_fingerprint
    assert _artifact_bytes(tmp_path / "first") == _artifact_bytes(tmp_path / "second")
    assert _artifact_bytes(tmp_path / "first") == _artifact_bytes(
        tmp_path / "explicit-empty"
    )


def test_required_donor_ids_and_complete_merge_closure_are_mandatory(
    tmp_path: Path,
) -> None:
    donor = _donor_tokenizer()
    donor_vocab = donor.get_vocab()

    result = prune_hf_tokenizer(
        donor,
        tmp_path / "target",
        target_vocab_size=262,
        calibration_texts=("xy",),
        required_donor_token_ids=(donor_vocab["abc"],),
    )

    expected_closure = {
        donor_vocab["a"],
        donor_vocab["b"],
        donor_vocab["c"],
        donor_vocab["ab"],
        donor_vocab["abc"],
    }
    assert result.required_donor_token_ids == (donor_vocab["abc"],)
    assert set(result.required_donor_token_dependency_ids) == expected_closure
    assert expected_closure.issubset(result.selected_donor_token_ids)
    assert set(result.mandatory_donor_token_ids).issubset(
        result.selected_donor_token_ids
    )
    assert len(result.mandatory_donor_token_ids) == 262
    assert donor_vocab["xy"] not in result.selected_donor_token_ids


def test_required_donor_ids_overflow_exact_budget_fails_closed(
    tmp_path: Path,
) -> None:
    donor = _donor_tokenizer()

    with pytest.raises(
        HFTokenizerPruneError,
        match=(
            "required donor token IDs, and BPE merge dependencies require "
            "at least 262 rows"
        ),
    ):
        prune_hf_tokenizer(
            donor,
            tmp_path / "target",
            target_vocab_size=261,
            required_donor_token_ids=(donor.get_vocab()["abc"],),
        )


@pytest.mark.parametrize(
    "required_ids, message",
    (
        ((True,), "only integers"),
        ((-1,), "outside donor vocabulary"),
        ((10_000,), "outside donor vocabulary"),
        ((1, 1), "duplicated"),
    ),
)
def test_required_donor_ids_are_strictly_validated(
    tmp_path: Path,
    required_ids: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(HFTokenizerPruneError, match=message):
        prune_hf_tokenizer(
            _donor_tokenizer(),
            tmp_path / "target",
            target_vocab_size=261,
            required_donor_token_ids=required_ids,
        )


def test_roundtrip_and_emitted_ids_fit_target_capacity(tmp_path: Path) -> None:
    text = "abc xy\nbytes:\t\x00 cafe\u0301 \u4e16\u754c"
    result = prune_hf_tokenizer(
        _donor_tokenizer(),
        tmp_path / "target",
        target_vocab_size=261,
        calibration_texts=[text],
    )

    token_ids = result.tokenizer.encode(text, add_special_tokens=False)
    assert (
        result.tokenizer.decode(
            token_ids,
            clean_up_tokenization_spaces=False,
        )
        == text
    )
    assert token_ids
    assert min(token_ids) >= 0
    assert max(token_ids) < 261


def test_roundtrip_matches_donor_normalization_semantics(tmp_path: Path) -> None:
    donor = _donor_tokenizer()
    donor.backend_tokenizer.normalizer = tokenizers.normalizers.NFC()
    text = "cafe\u0301"
    donor_ids = donor.encode(text, add_special_tokens=False)
    donor_roundtrip = donor.decode(
        donor_ids,
        clean_up_tokenization_spaces=False,
    )
    assert donor_roundtrip == "caf\u00e9"

    result = prune_hf_tokenizer(
        donor,
        tmp_path / "target",
        target_vocab_size=261,
        calibration_texts=[text],
    )
    target_ids = result.tokenizer.encode(text, add_special_tokens=False)
    assert (
        result.tokenizer.decode(
            target_ids,
            clean_up_tokenization_spaces=False,
        )
        == donor_roundtrip
    )


def test_target_too_small_for_byte_and_special_coverage_fails(tmp_path: Path) -> None:
    with pytest.raises(HFTokenizerPruneError, match="require at least 260 rows"):
        prune_hf_tokenizer(
            _donor_tokenizer(),
            tmp_path / "target",
            target_vocab_size=259,
        )
