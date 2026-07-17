from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments import compile_lfm2_bridge_folding_corpus as compiler  # noqa: E402


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        values = [ord(character) for character in text]
        return ([1] if add_special_tokens else []) + values

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces=False,
    ):
        values = [
            value
            for value in token_ids
            if not (skip_special_tokens and value == 1)
        ]
        return "".join(chr(value) for value in values)


class _Artifact:
    target_tokenizer_semantics_sha256 = "a" * 64
    artifact_sha256 = "b" * 64
    profile_fingerprint = "c" * 64

    def verify_files(self):
        return None


def test_compile_bridge_folding_corpus_is_broad_split_and_model_free(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
                [
                    {
                        "prompt": (
                            f"prompt-{index}-"
                            + ("x" * 80)
                            + f"-tail-{index}"
                        ),
                    "response": f"response-{index}-" + ("y" * 20),
                }
                for index in range(30)
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "trajectories.json"
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(compiler, "_load_local_tokenizer", lambda _path: _CharacterTokenizer())

    receipt = compiler.compile_bridge_folding_corpus(
        target_tokenizer_directory=target,
        source_json_paths=[source],
        output_path=output,
        receipt_path=receipt_path,
        maximum_rows=20,
        minimum_input_tokens=32,
        maximum_input_tokens=64,
        maximum_forced_tokens=8,
        bridge_artifact_loader=lambda _path: _Artifact(),
        tokenizer_semantics_fingerprint=lambda _tokenizer: "a" * 64,
    )

    rows = json.loads(output.read_text(encoding="ascii"))
    assert len(rows) == 20
    assert {row["split"] for row in rows} == {
        "selection",
        "fit",
        "validation",
    }
    assert all(row["schema"] == compiler.ROW_SCHEMA for row in rows)
    assert receipt["observed_tokens"] == {
        "minimum_input": 64,
        "maximum_input": 64,
        "minimum_forced": 8,
        "maximum_forced": 8,
    }
    assert receipt["parent_forward_calls"] == 0
    assert receipt["training_steps"] == 0
    assert receipt["model_parameters_pruned"] == 0
    assert receipt["model_parameters_quantized"] == 0
    assert len(receipt["receipt_sha256"]) == 64

    with pytest.raises(
        compiler.Lfm2BridgeFoldingCorpusError,
        match="Refusing to replace",
    ):
        compiler.compile_bridge_folding_corpus(
            target_tokenizer_directory=target,
            source_json_paths=[source],
            output_path=output,
            receipt_path=tmp_path / "other-receipt.json",
            maximum_rows=20,
            minimum_input_tokens=32,
            maximum_input_tokens=64,
            maximum_forced_tokens=8,
            bridge_artifact_loader=lambda _path: _Artifact(),
            tokenizer_semantics_fingerprint=lambda _tokenizer: "a" * 64,
        )


def test_compile_bridge_folding_corpus_rejects_inverted_input_bounds(
    tmp_path,
) -> None:
    with pytest.raises(
        compiler.Lfm2BridgeFoldingCorpusError,
        match="cannot exceed",
    ):
        compiler.compile_bridge_folding_corpus(
            target_tokenizer_directory=tmp_path,
            source_json_paths=[],
            output_path=tmp_path / "rows.json",
            receipt_path=tmp_path / "receipt.json",
            maximum_rows=3,
            minimum_input_tokens=65,
            maximum_input_tokens=64,
            maximum_forced_tokens=8,
        )
