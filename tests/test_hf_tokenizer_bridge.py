from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    STRICT_TOKENIZER_BRIDGE_FILENAME,
    StrictTokenizerBridgeError,
    compile_hf_target_tokenizer_bridge,
    compile_strict_vocabulary_operators,
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
    seal_strict_tokenizer_bridge_artifact,
)
from neural_forge.hf_transport_math import build_parent_residual_basis  # noqa: E402


class _Tokenizer:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = list(surfaces)
        self.all_special_ids: list[int] = []
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{token_id}-{surface}": token_id
            for token_id, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        exact = [
            token_id
            for token_id, surface in enumerate(self.surfaces)
            if surface == text
        ]
        if len(exact) != 1:
            raise KeyError(text)
        return exact


def _target_files(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "test"}}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"clean_up_tokenization_spaces": False}),
        encoding="utf-8",
    )


def _coordinates(source_dim: int, target_dim: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(
        source_dim + 4,
        source_dim,
        generator=generator,
        dtype=torch.float64,
    )
    return build_parent_residual_basis(rows, target_dim)


def test_sealed_bridge_roundtrips_and_binds_live_tokenizers(
    tmp_path: Path,
) -> None:
    donor = _Tokenizer(["zero", "one", "two", "three"])
    target = _Tokenizer(["three", "one"])
    root = tmp_path / "target"
    _target_files(root)

    artifact = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=8,
        target_row_capacity=2,
    )
    loaded = load_strict_tokenizer_bridge_artifact(
        root,
        donor_tokenizer=donor,
        target_tokenizer=target,
        expected_artifact_sha256=artifact.artifact_sha256,
    )

    assert artifact.to_dict() == loaded.to_dict()
    assert loaded.transport_profile.target_to_donor_row_indices == (3, 1)
    assert loaded.transport_profile.donor_row_capacity == 8
    assert loaded.target_vocabulary_size == 2
    assert len(loaded.artifact_sha256) == 64
    assert (root / STRICT_TOKENIZER_BRIDGE_FILENAME).is_file()
    assert loaded.to_dict()["model_parameter_operations"] == {
        "training_steps": 0,
        "backward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "random_target_rows": 0,
    }


def test_bridge_rejects_file_manifest_and_live_semantic_tampering(
    tmp_path: Path,
) -> None:
    donor = _Tokenizer(["a", "b", "c"])
    target = _Tokenizer(["c", "a"])
    root = tmp_path / "target"
    _target_files(root)
    artifact = seal_strict_tokenizer_bridge_artifact(donor, target, root)

    (root / "tokenizer.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(StrictTokenizerBridgeError, match="files differ"):
        load_strict_tokenizer_bridge_artifact(root)

    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "test"}}),
        encoding="utf-8",
    )
    changed_target = _Tokenizer(["c", "different"])
    with pytest.raises(
        StrictTokenizerBridgeError,
        match="target tokenizer differs",
    ):
        artifact.verify_live_tokenizers(donor, changed_target)

    manifest = json.loads((root / STRICT_TOKENIZER_BRIDGE_FILENAME).read_text("utf-8"))
    manifest["target_files"][0]["size_bytes"] += 1
    (root / STRICT_TOKENIZER_BRIDGE_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(StrictTokenizerBridgeError, match="seal is invalid"):
        load_strict_tokenizer_bridge_artifact(root)


def test_strict_operator_compiler_selects_rows_and_projects_coordinates(
    tmp_path: Path,
) -> None:
    donor = _Tokenizer(["zero", "one", "two", "three"])
    target = _Tokenizer(["three", "one"])
    root = tmp_path / "target"
    _target_files(root)
    artifact = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=6,
        target_row_capacity=2,
    )
    embedding = torch.arange(30, dtype=torch.float64).reshape(6, 5)
    head = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    bias = torch.arange(6, dtype=torch.float64)
    embedding_coordinates = _coordinates(5, 3, 1)
    head_coordinates = _coordinates(6, 4, 2)

    result = compile_strict_vocabulary_operators(
        embedding,
        head,
        artifact,
        embedding_coordinates,
        head_coordinates,
        source_lm_head_bias=bias,
    )

    torch.testing.assert_close(
        result.embedding_weight,
        embedding[[3, 1]] @ embedding_coordinates.decode,
    )
    torch.testing.assert_close(
        result.lm_head_weight,
        head[[3, 1]] @ head_coordinates.decode,
    )
    torch.testing.assert_close(result.lm_head_bias, bias[[3, 1]])
    assert result.embedding_weight.requires_grad is False
    assert result.report["random_target_rows"] == 0
    assert result.report["training_steps"] == 0
    assert result.report["pruned_model_parameters"] == 0
    assert len(result.report["report_sha256"]) == 64


def test_strict_operator_compiler_is_fail_closed_and_tie_aware(
    tmp_path: Path,
) -> None:
    donor = _Tokenizer(["a", "b", "c"])
    target = _Tokenizer(["c", "a"])
    root = tmp_path / "target"
    _target_files(root)
    artifact = seal_strict_tokenizer_bridge_artifact(donor, target, root)
    coordinates = _coordinates(4, 2, 4)
    weight = torch.arange(12, dtype=torch.float64).reshape(3, 4)

    tied = compile_strict_vocabulary_operators(
        weight,
        weight,
        artifact,
        coordinates,
        coordinates,
        require_tied_target=True,
    )
    assert tied.embedding_weight is tied.lm_head_weight

    with pytest.raises(StrictTokenizerBridgeError, match="exactly identical"):
        compile_strict_vocabulary_operators(
            weight,
            weight + 1.0,
            artifact,
            coordinates,
            coordinates,
            require_tied_target=True,
        )
    with pytest.raises(StrictTokenizerBridgeError, match="donor row capacity"):
        compile_strict_vocabulary_operators(
            weight[:2],
            weight,
            artifact,
            coordinates,
            coordinates,
        )
    bad = weight.clone()
    bad[0, 0] = torch.nan
    with pytest.raises(StrictTokenizerBridgeError, match="NaN or infinity"):
        compile_strict_vocabulary_operators(
            bad,
            weight,
            artifact,
            coordinates,
            coordinates,
        )

    (root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StrictTokenizerBridgeError, match="files differ"):
        compile_strict_vocabulary_operators(
            weight,
            weight,
            artifact,
            coordinates,
            coordinates,
        )


def test_semantic_fingerprint_is_path_independent_and_row_sensitive() -> None:
    first = _Tokenizer(["a", "b"])
    second = _Tokenizer(["a", "b"])
    changed = _Tokenizer(["b", "a"])

    assert fingerprint_hf_tokenizer_semantics(
        first
    ) == fingerprint_hf_tokenizer_semantics(second)
    assert fingerprint_hf_tokenizer_semantics(
        first
    ) != fingerprint_hf_tokenizer_semantics(changed)


def test_named_target_vocabulary_compiler_emits_sealed_bridge(
    tmp_path: Path,
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
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
    donor = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    result = compile_hf_target_tokenizer_bridge(
        donor,
        tmp_path / "compiled-target",
        target_vocab_size=262,
        calibration_texts=("abc abc",),
        donor_row_capacity=512,
    )

    assert len(result.tokenizer) == 262
    assert result.target_vocab_size == 262
    assert result.transport_profile.donor_row_capacity == 512
    assert result.transport_profile.target_row_capacity == 262
    assert result.transport_profile.target_to_donor_row_indices == (
        result.selected_donor_token_ids
    )
    assert result.tokenizer_fingerprint == (
        result.artifact.target_tokenizer_semantics_sha256
    )
    assert (result.output_dir / STRICT_TOKENIZER_BRIDGE_FILENAME).is_file()
    result.artifact.verify_live_tokenizers(donor, result.tokenizer)
