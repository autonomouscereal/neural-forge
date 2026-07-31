from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_tokenizer_pair_bridge import (  # noqa: E402
    TOKENIZER_PAIR_BRIDGE_CODES_FILENAME,
    TOKENIZER_PAIR_BRIDGE_FILENAME,
    TOKENIZER_PAIR_BRIDGE_KIND,
    TOKENIZER_PAIR_BRIDGE_SCHEMA,
    TokenizerPairBridgeArtifact,
    TokenizerPairBridgeError,
    build_hf_tokenizer_pair_code_profile,
    build_pair_bridge_row_incidences,
    compile_hf_tokenizer_pair_bridge,
    compile_pair_bridge_anchor_transport,
    compile_pair_bridge_length_normalized_transport,
    compile_pair_bridge_occurrence_weighted_transport,
    load_hf_tokenizer_pair_bridge_artifact,
    tokenizer_pair_bridge_configuration,
)


_SOURCE_SPECIALS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}
_TARGET_SPECIALS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad0>",
    "unk_token": "<unk0>",
}

_SOURCE_CORPUS = (
    "the quick brown fox jumps over the lazy dog",
    "the compiler folds the parent model into the child model",
    "tokenizer bridges align vocabulary rows without touching weights",
    "the forest is quiet and the river runs deep tonight",
    "another folded child inherits the parent interface exactly",
)
_TARGET_CORPUS = (
    "quantum superposition collapses whenever an observer measures it",
    "les jardins fleurissent au printemps quand la pluie tombe",
    "matrix factorization recovers low rank structure from noise",
    "神经网络 折叠 需要 词表 对齐",
    "eigenvalues drift when the operator loses its symmetry",
)


def _byte_to_character() -> tuple[str, ...]:
    """Independent reimplementation of the canonical ByteLevel byte map."""

    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    values = list(printable)
    characters = [chr(value) for value in printable]
    shift = 0
    for byte in range(256):
        if byte in printable:
            continue
        values.append(byte)
        characters.append(chr(256 + shift))
        shift += 1
    table = [""] * 256
    for byte, character in zip(values, characters):
        table[byte] = character
    return tuple(table)


_BYTE_TO_CHARACTER = _byte_to_character()
_CHARACTER_TO_BYTE = {
    character: byte for byte, character in enumerate(_BYTE_TO_CHARACTER)
}


def _train_byte_level_bpe(
    corpus: tuple[str, ...],
    vocab_size: int,
    specials: dict[str, str],
) -> Any:
    """Train one real ByteLevel-BPE tokenizer on a tiny corpus."""

    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.train_from_iterator(
        [line for line in corpus for _ in range(12)],
        trainer=tokenizers.trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            show_progress=False,
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=list(specials.values()),
        ),
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        **specials,
    )


def _row_tokens(tokenizer: Any) -> tuple[str, ...]:
    vocabulary = tokenizer.get_vocab()
    rows = [""] * len(vocabulary)
    for token, token_id in vocabulary.items():
        rows[token_id] = token
    return tuple(rows)


def _added_row_ids(tokenizer: Any) -> frozenset[int]:
    data = json.loads(tokenizer.backend_tokenizer.to_str())
    return frozenset(int(entry["id"]) for entry in data.get("added_tokens", ()))


def _row_bytes(tokenizer: Any) -> tuple[bytes, ...]:
    # An added token's row carries its literal content; every other row is the
    # ByteLevel rendering of its own bytes. A special token that is a plain BPE
    # model row stays on the ByteLevel path.
    added = _added_row_ids(tokenizer)
    rows = []
    for token_id, token in enumerate(_row_tokens(tokenizer)):
        if token_id in added:
            rows.append(token.encode("utf-8"))
        else:
            rows.append(bytes(_CHARACTER_TO_BYTE[value] for value in token))
    return tuple(rows)


@pytest.fixture(scope="module")
def tokenizer_pair() -> tuple[Any, Any]:
    source = _train_byte_level_bpe(_SOURCE_CORPUS, 420, _SOURCE_SPECIALS)
    target = _train_byte_level_bpe(_TARGET_CORPUS, 400, _TARGET_SPECIALS)
    assert len(source) != len(target)
    assert set(source.get_vocab()) != set(target.get_vocab())
    return source, target


@pytest.fixture(scope="module")
def compiled_bridge(
    tokenizer_pair: tuple[Any, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> TokenizerPairBridgeArtifact:
    source, target = tokenizer_pair
    root = tmp_path_factory.mktemp("pair-bridge") / "bridge"
    return compile_hf_tokenizer_pair_bridge(source, target, root)


def test_derived_byte_alphabet_matches_the_tokenizers_library() -> None:
    assert len(set(_BYTE_TO_CHARACTER)) == 256
    assert set(_BYTE_TO_CHARACTER) == set(
        tokenizers.pre_tokenizers.ByteLevel.alphabet()
    )


def test_every_row_segments_byte_exactly_in_both_directions(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, target = tokenizer_pair
    profile = compiled_bridge.profile
    source_bytes = _row_bytes(source)
    target_bytes = _row_bytes(target)
    source_tokens = _row_tokens(source)
    target_tokens = _row_tokens(target)
    source_added = _added_row_ids(source) | frozenset(source.all_special_ids)
    target_added = _added_row_ids(target) | frozenset(target.all_special_ids)

    assert profile.source_vocab_size == len(source)
    assert profile.target_vocab_size == len(target)
    assert len(profile.forward.codes) == len(source)
    assert len(profile.reverse.codes) == len(target)
    assert profile.forward.statistics["unresolved_rows"] == 0
    assert profile.reverse.statistics["unresolved_rows"] == 0

    plain_string_checks = 0
    for direction, codes, classes, in_bytes, out_bytes in (
        (
            "forward",
            profile.forward.codes,
            profile.forward.code_classes,
            source_bytes,
            target_bytes,
        ),
        (
            "reverse",
            profile.reverse.codes,
            profile.reverse.code_classes,
            target_bytes,
            source_bytes,
        ),
    ):
        for row_index, code in enumerate(codes):
            assert code, f"{direction} row {row_index} produced an empty code"
            assert all(0 <= entry < len(out_bytes) for entry in code)
            if classes[row_index] == "special-role":
                assert len(code) == 1
                continue
            rendered = b"".join(out_bytes[entry] for entry in code)
            assert rendered == in_bytes[row_index], (
                f"{direction} row {row_index} lost bytes"
            )

    for row_index, code in enumerate(profile.forward.codes):
        if profile.forward.code_classes[row_index] == "special-role":
            continue
        if row_index in source_added or any(entry in target_added for entry in code):
            continue
        assert (
            "".join(target_tokens[entry] for entry in code)
            == (source_tokens[row_index])
        )
        plain_string_checks += 1
    assert plain_string_checks > 200


def test_code_classes_are_exactly_the_one_row_resolutions(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, target = tokenizer_pair
    profile = compiled_bridge.profile
    source_bytes = _row_bytes(source)
    target_bytes = _row_bytes(target)
    target_rows = {}
    for row_index, raw in enumerate(target_bytes):
        target_rows.setdefault(raw, row_index)
    role_rows = {entry[1] for entry in profile.special_role_map}
    assert role_rows == {
        source.bos_token_id,
        source.eos_token_id,
        source.pad_token_id,
        source.unk_token_id,
    }

    anchors = 0
    multi = 0
    for row_index, code in enumerate(profile.forward.codes):
        code_class = profile.forward.code_classes[row_index]
        if code_class == "special-role":
            assert row_index in role_rows
            continue
        assert row_index not in role_rows
        if code_class == "anchor-exact":
            assert len(code) == 1
            assert target_bytes[code[0]] == source_bytes[row_index]
            assert source_bytes[row_index] in target_rows
            anchors += 1
        else:
            assert code_class == "multi-code"
            assert len(code) >= 2
            assert source_bytes[row_index] not in target_rows
            multi += 1
    assert anchors == profile.forward.statistics["anchor_exact_rows"]
    assert multi == profile.forward.statistics["multi_code_rows"]
    assert multi >= 32, "the tiny corpora must diverge enough to exercise codes"
    assert profile.forward.statistics["max_code_length"] >= 3
    assert profile.forward.statistics["pure_byte_fallback_rows"] >= 1
    assert profile.forward.statistics["byte_mismatch_rows"] == 0
    assert profile.forward.statistics["total_code_entries"] == sum(
        len(code) for code in profile.forward.codes
    )
    histogram = dict(
        (length, count)
        for length, count in profile.forward.statistics["code_length_histogram"]
    )
    assert sum(histogram.values()) == len(source)
    assert histogram[1] == (
        profile.forward.statistics["anchor_exact_rows"]
        + profile.forward.statistics["special_role_rows"]
    )


def test_special_role_mapping_can_be_disabled_for_pure_byte_semantics(
    tokenizer_pair: tuple[Any, Any],
) -> None:
    source, target = tokenizer_pair
    profile = build_hf_tokenizer_pair_code_profile(
        source,
        target,
        map_special_roles=False,
    )
    assert profile.special_role_map == ()
    assert profile.forward.statistics["special_role_rows"] == 0
    assert profile.forward.statistics["byte_exact_rows"] == len(source)
    assert profile.reverse.statistics["byte_exact_rows"] == len(target)
    source_bytes = _row_bytes(source)
    target_bytes = _row_bytes(target)
    for row_index, code in enumerate(profile.forward.codes):
        assert (
            b"".join(target_bytes[entry] for entry in code) == (source_bytes[row_index])
        )
    assert (
        profile.fingerprint()
        != build_hf_tokenizer_pair_code_profile(
            source,
            target,
        ).fingerprint()
    )


def test_sealed_artifact_round_trips_and_binds_live_tokenizers(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, target = tokenizer_pair
    root = compiled_bridge.root_dir
    loaded = load_hf_tokenizer_pair_bridge_artifact(
        root,
        source_tokenizer=source,
        target_tokenizer=target,
        expected_artifact_sha256=compiled_bridge.artifact_sha256,
    )

    assert loaded.to_dict() == compiled_bridge.to_dict()
    assert loaded.profile.forward.codes == compiled_bridge.profile.forward.codes
    assert loaded.profile.reverse.codes == compiled_bridge.profile.reverse.codes
    document = loaded.to_dict()
    assert document["schema"] == TOKENIZER_PAIR_BRIDGE_SCHEMA
    assert document["kind"] == TOKENIZER_PAIR_BRIDGE_KIND
    assert document["model_parameter_operations"] == {
        "training_steps": 0,
        "backward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "random_target_rows": 0,
    }
    assert document["pair_profile"]["model_weight_reads"] == 0
    assert document["pair_profile"]["model_weight_writes"] == 0
    assert document["pair_profile"]["model_weight_payloads_resident"] is False
    assert (
        document["pair_profile"]["segmentation"]["singleton_unicode_transport_used"]
        is False
    )
    assert [item["relative_path"] for item in document["payload_files"]] == [
        TOKENIZER_PAIR_BRIDGE_CODES_FILENAME
    ]
    assert (root / TOKENIZER_PAIR_BRIDGE_FILENAME).is_file()
    assert (root / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME).is_file()
    assert loaded.forward_code(3) == compiled_bridge.profile.forward.codes[3]
    assert loaded.reverse_code(3) == compiled_bridge.profile.reverse.codes[3]
    assert loaded.bridge_kind == TOKENIZER_PAIR_BRIDGE_KIND
    assert loaded.parent_tokenizer_semantics_sha256 == (
        loaded.source_tokenizer_semantics_sha256
    )
    assert loaded.child_tokenizer_semantics_sha256 == (
        loaded.target_tokenizer_semantics_sha256
    )
    assert loaded.source_vocabulary_size == len(source)
    assert loaded.target_vocabulary_size == len(target)
    assert set(tokenizer_pair_bridge_configuration(loaded).values()) == {
        loaded.artifact_sha256,
        loaded.profile_fingerprint,
        loaded.source_tokenizer_semantics_sha256,
        loaded.target_tokenizer_semantics_sha256,
    }
    with pytest.raises(TokenizerPairBridgeError, match="outside the defined"):
        loaded.forward_code(len(source))


def test_recompiling_is_idempotent_and_rejects_a_different_bridge(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, target = tokenizer_pair
    again = compile_hf_tokenizer_pair_bridge(
        source,
        target,
        compiled_bridge.root_dir,
    )
    assert again.artifact_sha256 == compiled_bridge.artifact_sha256
    with pytest.raises(TokenizerPairBridgeError, match="already sealed here"):
        compile_hf_tokenizer_pair_bridge(
            source,
            target,
            compiled_bridge.root_dir,
            source_row_capacity=len(source) + 8,
        )


def test_bridge_is_deterministic_across_independent_builds(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
    tmp_path: Path,
) -> None:
    source, target = tokenizer_pair
    reloaded_dir = tmp_path / "reloaded-tokenizer"
    reloaded_dir.mkdir()
    source.backend_tokenizer.save(str(reloaded_dir / "tokenizer.json"))
    reloaded_source = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(reloaded_dir / "tokenizer.json"),
        **_SOURCE_SPECIALS,
    )
    assert reloaded_source is not source

    first = compile_hf_tokenizer_pair_bridge(source, target, tmp_path / "first")
    second = compile_hf_tokenizer_pair_bridge(
        reloaded_source,
        target,
        tmp_path / "second",
    )
    assert first.artifact_sha256 == compiled_bridge.artifact_sha256
    assert second.artifact_sha256 == first.artifact_sha256
    assert second.profile.fingerprint() == first.profile.fingerprint()
    assert (tmp_path / "first" / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME).read_bytes() == (
        tmp_path / "second" / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME
    ).read_bytes()


def test_bridge_rejects_payload_manifest_and_live_tokenizer_tampering(
    tokenizer_pair: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    source, target = tokenizer_pair
    root = tmp_path / "sealed"
    artifact = compile_hf_tokenizer_pair_bridge(source, target, root)
    manifest_path = root / TOKENIZER_PAIR_BRIDGE_FILENAME
    payload_path = root / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME
    pristine_manifest = manifest_path.read_bytes()
    pristine_payload = payload_path.read_bytes()

    corrupted = bytearray(pristine_payload)
    corrupted[-1] ^= 0x01
    payload_path.write_bytes(bytes(corrupted))
    # A cold load hashes the code arrays before it rebuilds the profile; a
    # live artifact still holds its sealed file records and rejects earlier.
    with pytest.raises(TokenizerPairBridgeError, match="tampered with"):
        load_hf_tokenizer_pair_bridge_artifact(root)
    with pytest.raises(TokenizerPairBridgeError, match="payload files differ"):
        artifact.verify_files()
    payload_path.write_bytes(b"not a safetensors file")
    with pytest.raises(TokenizerPairBridgeError, match="readable safetensors"):
        load_hf_tokenizer_pair_bridge_artifact(root)
    payload_path.unlink()
    with pytest.raises(TokenizerPairBridgeError, match="missing or unsafe"):
        artifact.verify_packaged_files()
    payload_path.write_bytes(pristine_payload)
    artifact.verify_files()

    manifest = json.loads(pristine_manifest)
    manifest["payload_files"][0]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TokenizerPairBridgeError, match="seal is invalid"):
        load_hf_tokenizer_pair_bridge_artifact(root)
    manifest_path.write_bytes(pristine_manifest)

    manifest = json.loads(pristine_manifest)
    manifest.pop("payload_files")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TokenizerPairBridgeError, match="incomplete or unknown"):
        load_hf_tokenizer_pair_bridge_artifact(root)
    manifest_path.write_bytes(pristine_manifest)

    (root / "stowaway.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TokenizerPairBridgeError, match="payload files differ"):
        load_hf_tokenizer_pair_bridge_artifact(root)
    artifact.verify_packaged_files()
    (root / "stowaway.json").unlink()

    artifact.verify_files()
    with pytest.raises(TokenizerPairBridgeError, match="Live source tokenizer"):
        artifact.verify_live_tokenizers(target, target)
    with pytest.raises(TokenizerPairBridgeError, match="Live target tokenizer"):
        artifact.verify_live_tokenizers(source, source)
    with pytest.raises(TokenizerPairBridgeError, match="requires both"):
        load_hf_tokenizer_pair_bridge_artifact(root, source_tokenizer=source)


def test_bridge_rejects_a_substituted_code_payload(
    tokenizer_pair: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    source, target = tokenizer_pair
    honest = tmp_path / "honest"
    forged = tmp_path / "forged"
    compile_hf_tokenizer_pair_bridge(source, target, honest)
    compile_hf_tokenizer_pair_bridge(
        source,
        target,
        forged,
        map_special_roles=False,
    )

    substituted = forged / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME
    shutil.copyfile(substituted, honest / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME)
    manifest_path = honest / TOKENIZER_PAIR_BRIDGE_FILENAME
    manifest = json.loads(manifest_path.read_text("utf-8"))
    forged_manifest = json.loads(
        (forged / TOKENIZER_PAIR_BRIDGE_FILENAME).read_text("utf-8")
    )
    manifest["payload_files"] = forged_manifest["payload_files"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TokenizerPairBridgeError, match="tampered with"):
        load_hf_tokenizer_pair_bridge_artifact(honest)


def test_compiler_refuses_a_shared_output_directory(
    tokenizer_pair: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    source, target = tokenizer_pair
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TokenizerPairBridgeError, match="must be dedicated"):
        compile_hf_tokenizer_pair_bridge(source, target, shared)


def test_non_byte_level_tokenizers_fail_closed_naming_the_extension(
    tokenizer_pair: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    source, _target = tokenizer_pair

    word_piece = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(
            vocab={"[UNK]": 0, "the": 1, "##er": 2},
            unk_token="[UNK]",
        )
    )
    word_piece.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    unigram_style = transformers.PreTrainedTokenizerFast(
        tokenizer_object=word_piece,
        unk_token="[UNK]",
    )
    with pytest.raises(
        TokenizerPairBridgeError,
        match="SentencePiece byte-fallback extension",
    ) as excinfo:
        compile_hf_tokenizer_pair_bridge(
            source,
            unigram_style,
            tmp_path / "wordpiece",
        )
    assert "'WordPiece' model" in str(excinfo.value)

    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: index for index, token in enumerate(alphabet)}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.decoder = tokenizers.decoders.ByteLevel()
    no_byte_level = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
    )
    with pytest.raises(
        TokenizerPairBridgeError,
        match="does not use ByteLevel pre-tokenization",
    ):
        compile_hf_tokenizer_pair_bridge(
            source,
            no_byte_level,
            tmp_path / "no-byte-level",
        )


def test_incomplete_byte_alphabet_coverage_fails_closed(
    tokenizer_pair: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    source, _target = tokenizer_pair
    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.train_from_iterator(
        ["abc abc abc"] * 8,
        trainer=tokenizers.trainers.BpeTrainer(
            vocab_size=40,
            min_frequency=1,
            show_progress=False,
            special_tokens=["<unk>"],
        ),
    )
    starved = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
    )
    with pytest.raises(
        TokenizerPairBridgeError,
        match="does not cover the complete 256-row",
    ):
        compile_hf_tokenizer_pair_bridge(source, starved, tmp_path / "starved")


def test_incidence_index_reproduces_the_code_tables(
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    index = build_pair_bridge_row_incidences(compiled_bridge)
    profile = compiled_bridge.profile
    assert index.output_row_capacity == profile.target_row_capacity
    assert index.input_row_capacity == profile.source_row_capacity
    assert (
        index.report["total_incidences"]
        == profile.forward.statistics["total_code_entries"]
    )

    rebuilt: dict[int, list[tuple[int, int]]] = {}
    for output_row in range(index.output_row_capacity):
        for incidence in index.row(output_row):
            rebuilt.setdefault(incidence.input_row_index, []).append(
                (incidence.code_position, output_row)
            )
            assert incidence.weight_numerator == 1
            assert incidence.weight_denominator == incidence.code_length
            assert incidence.weight == pytest.approx(1.0 / incidence.code_length)
    for input_row, entries in rebuilt.items():
        entries.sort()
        assert (
            tuple(output for _position, output in entries)
            == (profile.forward.codes[input_row])
        )
    assert len(rebuilt) == len(profile.forward.codes)

    anchors = build_pair_bridge_row_incidences(
        compiled_bridge,
        weighting="anchor-exact",
    )
    assert anchors.report["contributing_input_rows"] == (
        profile.forward.statistics["anchor_exact_rows"]
        + profile.forward.statistics["special_role_rows"]
    )
    with pytest.raises(TokenizerPairBridgeError, match="outside the output row"):
        index.row(index.output_row_capacity)


def test_transport_helpers_produce_expected_shapes_and_exact_anchor_rows(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, target = tokenizer_pair
    profile = compiled_bridge.profile
    width = 6
    generator = torch.Generator().manual_seed(20260731)
    embedding = torch.randn(
        len(source),
        width,
        generator=generator,
        dtype=torch.float64,
    )

    anchor = compile_pair_bridge_anchor_transport(embedding, compiled_bridge)
    averaged = compile_pair_bridge_length_normalized_transport(
        embedding,
        compiled_bridge,
    )
    assert tuple(anchor.evidence.shape) == (len(target), width)
    assert tuple(averaged.evidence.shape) == (len(target), width)
    assert tuple(anchor.evidence_mask.shape) == (len(target),)
    assert anchor.evidence.dtype == torch.float64
    assert anchor.evidence.requires_grad is False
    assert anchor.report["training_steps"] == 0
    assert anchor.report["random_target_rows"] == 0
    assert anchor.report["model_weight_writes"] == 0
    assert anchor.report["evidence_only_no_parameter_install"] is True
    assert len(anchor.report["report_sha256"]) == 64
    assert anchor.report["report_sha256"] != averaged.report["report_sha256"]
    assert anchor.covered_rows <= averaged.covered_rows

    index = build_pair_bridge_row_incidences(compiled_bridge)
    exact_rows = 0
    for row_index, code in enumerate(profile.forward.codes):
        if profile.forward.code_classes[row_index] != "anchor-exact":
            continue
        output_row = code[0]
        assert bool(anchor.evidence_mask[output_row])
        if len(index.row(output_row)) != 1:
            continue
        torch.testing.assert_close(
            anchor.evidence[output_row],
            embedding[row_index],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            averaged.evidence[output_row],
            embedding[row_index],
            rtol=0.0,
            atol=0.0,
        )
        exact_rows += 1
    assert exact_rows > 100

    uncovered = (~averaged.evidence_mask).nonzero().flatten().tolist()
    for output_row in uncovered:
        assert torch.equal(
            averaged.evidence[output_row],
            torch.zeros(width, dtype=torch.float64),
        )
        assert int(averaged.incidence_counts[output_row]) == 0

    multi_row = next(
        row
        for row, code_class in enumerate(profile.forward.code_classes)
        if code_class == "multi-code"
    )
    output_row = profile.forward.codes[multi_row][0]
    expected = torch.zeros(width, dtype=torch.float64)
    total = 0.0
    for incidence in index.row(output_row):
        weight = 1.0 / incidence.code_length
        expected = expected + weight * embedding[incidence.input_row_index]
        total += weight
    torch.testing.assert_close(averaged.evidence[output_row], expected / total)
    assert float(averaged.weight_totals[output_row]) == pytest.approx(total)
    assert int(averaged.incidence_counts[output_row]) == len(index.row(output_row))


def test_occurrence_weighted_transport_honours_the_frequency_table(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, _target = tokenizer_pair
    profile = compiled_bridge.profile
    width = 4
    embedding = torch.arange(
        len(source) * width,
        dtype=torch.float64,
    ).reshape(len(source), width)
    frequencies = {row: (row % 7) + 1 for row in range(len(source))}

    weighted = compile_pair_bridge_occurrence_weighted_transport(
        embedding,
        compiled_bridge,
        frequencies,
    )
    index = build_pair_bridge_row_incidences(
        compiled_bridge,
        weighting="occurrence-weighted",
        token_frequencies=frequencies,
    )
    output_row = next(
        row for row in range(profile.target_row_capacity) if len(index.row(row)) >= 2
    )
    expected = torch.zeros(width, dtype=torch.float64)
    total = 0.0
    for incidence in index.row(output_row):
        weight = frequencies[incidence.input_row_index] / incidence.code_length
        assert incidence.weight_numerator == frequencies[incidence.input_row_index]
        expected = expected + weight * embedding[incidence.input_row_index]
        total += weight
    torch.testing.assert_close(weighted.evidence[output_row], expected / total)
    assert len(weighted.report["frequency_table_sha256"]) == 64

    silent = compile_pair_bridge_occurrence_weighted_transport(
        embedding,
        compiled_bridge,
        {0: 5},
    )
    assert silent.report["contributing_input_rows"] == 1
    assert silent.covered_rows == len(set(profile.forward.codes[0]))

    dense = compile_pair_bridge_occurrence_weighted_transport(
        embedding,
        compiled_bridge,
        [frequencies[row] for row in range(len(source))],
    )
    assert torch.equal(dense.evidence, weighted.evidence)
    assert dense.report["report_sha256"] == weighted.report["report_sha256"]


def test_transport_is_chunk_size_independent_and_supports_reverse(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, target = tokenizer_pair
    generator = torch.Generator().manual_seed(4096)
    embedding = torch.randn(
        len(source),
        5,
        generator=generator,
        dtype=torch.float64,
    )
    small = compile_pair_bridge_length_normalized_transport(
        embedding,
        compiled_bridge,
        chunk_rows=3,
    )
    large = compile_pair_bridge_length_normalized_transport(
        embedding,
        compiled_bridge,
        chunk_rows=8192,
    )
    assert torch.equal(small.evidence, large.evidence)
    assert torch.equal(small.weight_totals, large.weight_totals)

    reverse_embedding = torch.randn(
        len(target),
        7,
        generator=generator,
        dtype=torch.float64,
    )
    reverse = compile_pair_bridge_length_normalized_transport(
        reverse_embedding,
        compiled_bridge,
        direction="reverse",
    )
    assert tuple(reverse.evidence.shape) == (len(source), 7)
    assert reverse.report["code_direction"] == "target-row-to-source-code"
    assert reverse.covered_rows > 0


def test_transport_helpers_fail_closed_on_bad_inputs(
    tokenizer_pair: tuple[Any, Any],
    compiled_bridge: TokenizerPairBridgeArtifact,
) -> None:
    source, _target = tokenizer_pair
    embedding = torch.zeros(len(source), 3, dtype=torch.float64)

    with pytest.raises(TokenizerPairBridgeError, match="input row capacity"):
        compile_pair_bridge_length_normalized_transport(
            embedding[:-1],
            compiled_bridge,
        )
    with pytest.raises(TokenizerPairBridgeError, match="rank-two torch tensor"):
        compile_pair_bridge_length_normalized_transport(
            embedding[0],
            compiled_bridge,
        )
    broken = embedding.clone()
    broken[0, 0] = float("nan")
    with pytest.raises(TokenizerPairBridgeError, match="NaN or infinity"):
        compile_pair_bridge_length_normalized_transport(broken, compiled_bridge)
    with pytest.raises(TokenizerPairBridgeError, match="forward.*or.*reverse"):
        compile_pair_bridge_length_normalized_transport(
            embedding,
            compiled_bridge,
            direction="sideways",
        )
    with pytest.raises(TokenizerPairBridgeError, match="undefined vocabulary row"):
        compile_pair_bridge_occurrence_weighted_transport(
            embedding,
            compiled_bridge,
            {len(source): 3},
        )
    with pytest.raises(TokenizerPairBridgeError, match="at least one vocabulary"):
        compile_pair_bridge_occurrence_weighted_transport(
            embedding,
            compiled_bridge,
            {0: 0},
        )
    with pytest.raises(TokenizerPairBridgeError, match="cover every defined row"):
        compile_pair_bridge_occurrence_weighted_transport(
            embedding,
            compiled_bridge,
            [1, 2, 3],
        )
    with pytest.raises(TokenizerPairBridgeError, match="only applies to"):
        build_pair_bridge_row_incidences(
            compiled_bridge,
            token_frequencies={0: 1},
        )
    with pytest.raises(TokenizerPairBridgeError, match="verified"):
        compile_pair_bridge_length_normalized_transport(embedding, object())
