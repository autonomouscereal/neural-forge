from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_tokenizer
from neural_forge.hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics
from neural_forge.lfm2_semantic_source_lexicon import (
    SEMANTIC_SOURCE_LEXICON_DIRECTORY,
    Lfm2SemanticSourceLexiconError,
    load_packaged_lfm2_semantic_source_lexicon,
    package_lfm2_semantic_source_lexicon,
)


pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


def _tokenizer_package(path: Path) -> tuple[object, str]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir()
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    tokenizer = load_lfm2_direct_local_tokenizer(path)
    return tokenizer, fingerprint_hf_tokenizer_semantics(tokenizer)


def _package(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "source-tokenizer"
    _tokenizer, semantics = _tokenizer_package(source)
    child = tmp_path / "child"
    child.mkdir()
    package_lfm2_semantic_source_lexicon(
        source,
        child,
        expected_tokenizer_semantics_sha256=semantics,
    )
    return source, child, semantics


def test_semantic_source_lexicon_fresh_loads_after_source_is_removed(
    tmp_path: Path,
) -> None:
    source, child, semantics = _package(tmp_path)
    shutil.rmtree(source)

    packaged = load_packaged_lfm2_semantic_source_lexicon(
        child,
        expected_tokenizer_semantics_sha256=semantics,
    )

    assert packaged.root == (child / SEMANTIC_SOURCE_LEXICON_DIRECTORY).resolve()
    assert fingerprint_hf_tokenizer_semantics(packaged.tokenizer) == semantics
    assert packaged.receipt["continuous_parameter_elements"] == 0
    assert packaged.receipt["parameter_count_credit"] is False


def test_semantic_source_lexicon_rejects_unsealed_additions(
    tmp_path: Path,
) -> None:
    _source, child, semantics = _package(tmp_path)
    package = child / SEMANTIC_SOURCE_LEXICON_DIRECTORY
    (package / "unsealed.bin").write_bytes(b"not part of the lexicon")

    with pytest.raises(
        Lfm2SemanticSourceLexiconError,
        match="unsealed additions",
    ):
        load_packaged_lfm2_semantic_source_lexicon(
            child,
            expected_tokenizer_semantics_sha256=semantics,
        )


def test_semantic_source_lexicon_rejects_byte_tampering(
    tmp_path: Path,
) -> None:
    _source, child, semantics = _package(tmp_path)
    package = child / SEMANTIC_SOURCE_LEXICON_DIRECTORY
    with (package / "tokenizer.json").open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(
        Lfm2SemanticSourceLexiconError,
        match="sealed inventory",
    ):
        load_packaged_lfm2_semantic_source_lexicon(
            child,
            expected_tokenizer_semantics_sha256=semantics,
        )
