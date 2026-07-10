from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_native_raw_tokens import (  # noqa: E402
    SERIALIZATION_PATH,
    ByteLevelMappingError,
    HFNativeRawByteCodec,
    RawByteCodecLimits,
    RawTensorBoundsError,
    RawTensorContextError,
    RawTensorDecodeError,
    RawTensorEncodeError,
    discover_byte_token_mapping,
)


def _bytelevel_tokenizer(
    *,
    model_max_length: int = 61,
    missing_atom: str | None = None,
    bytelevel_pre_tokenizer: bool = True,
    bytelevel_decoder: bool = True,
) -> Any:
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    if missing_atom is not None:
        alphabet.remove(missing_atom)
    vocabulary = {token: token_id for token_id, token in enumerate(alphabet)}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocabulary, merges=[]))
    if bytelevel_pre_tokenizer:
        backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=False,
        )
    else:
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    if bytelevel_decoder:
        backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=model_max_length,
    )


def test_discovers_exact_existing_one_token_mapping_for_every_byte() -> None:
    tokenizer = _bytelevel_tokenizer()
    vocabulary_before = tokenizer.get_vocab()

    mapping = discover_byte_token_mapping(tokenizer)
    all_bytes = bytes(range(256))
    token_ids = mapping.encode_bytes(all_bytes)

    assert len(token_ids) == 256
    assert len(set(token_ids)) == 256
    assert mapping.decode_token_ids(token_ids) == all_bytes
    assert tuple(mapping.token_id_for_byte(value) for value in range(256)) == token_ids
    assert tuple(mapping.byte_for_token_id(value) for value in token_ids) == tuple(
        range(256)
    )
    assert max(token_ids) < len(tokenizer)
    assert tokenizer.get_vocab() == vocabulary_before
    report = mapping.report()
    assert report["all_bytes_exactly_one_existing_token"] is True
    assert report["all_byte_ids_are_ordinary_model_vocabulary_ids"] is True
    assert report["tokenizer_mutated"] is False
    assert len(report["byte_mapping_sha256"]) == 64


@pytest.mark.parametrize(
    ("modality", "value", "metadata"),
    [
        (
            "image",
            np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)[::-1],
            {"layout": "HWC"},
        ),
        (
            "audio.signed",
            np.array([[-32768, -7], [0, 19], [32767, 2]], dtype=np.int16),
            {"sample_rate": 48_000},
        ),
        (
            "audio.float",
            np.array([-1.0, -0.0, 0.25, 1.0], dtype=np.float32),
            {"sample_rate": 16_000},
        ),
        (
            "video",
            np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3),
            {"frame_rate": 29.97},
        ),
        (
            "point-cloud",
            np.array(
                [[-1.5, 0.0, 2.25], [3.5, -4.75, 8.0]],
                dtype=">f8",
            ),
            {"axes": ["x", "y", "z"]},
        ),
        (
            "tensor.3d",
            np.array(
                [[-(2**31), 0, 7], [19, -3, 2**31 - 1]],
                dtype=np.int32,
            ),
            {},
        ),
    ],
)
def test_same_generic_path_roundtrips_all_modalities_byte_exactly(
    modality: str,
    value: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    codec = HFNativeRawByteCodec(_bytelevel_tokenizer(), context_limit=53)

    encoded = codec.encode_tensor(value, modality=modality, metadata=metadata)
    decoded = codec.decode_chunks(encoded.chunks, expected_report=encoded.report())

    assert encoded.report()["serialization_path"] == SERIALIZATION_PATH
    assert decoded.report()["serialization_path"] == SERIALIZATION_PATH
    assert encoded.report()["learned_components"] == []
    assert encoded.report()["modality_specific_components"] == []
    assert encoded.report()["runtime_donor_required"] is False
    assert encoded.payload_input_ids == codec.mapping.encode_bytes(
        value.tobytes(order="C")
    )
    assert decoded.modality == modality
    assert decoded.shape == value.shape
    assert decoded.dtype == value.dtype.str
    assert decoded.metadata == metadata
    assert decoded.array.flags.c_contiguous
    assert decoded.array.tobytes(order="C") == value.tobytes(order="C")
    assert decoded.report()["exact_payload_roundtrip"] is True


def test_serialization_is_canonical_for_logically_identical_array_views() -> None:
    codec = HFNativeRawByteCodec(_bytelevel_tokenizer(), context_limit=47)
    base = np.arange(6 * 7, dtype=np.uint16).reshape(6, 7)
    noncontiguous = base[:, ::2]
    contiguous = np.ascontiguousarray(noncontiguous)
    metadata = {"nested": {"z": 3, "a": [True, None, "\u03bc"]}}

    first = codec.encode_tensor(
        noncontiguous,
        modality="tensor",
        metadata=metadata,
    )
    second = codec.encode_tensor(
        contiguous,
        modality="tensor",
        metadata=metadata,
    )

    assert first.input_ids == second.input_ids
    assert first.report() == second.report()
    assert json.dumps(first.report(), sort_keys=True, allow_nan=False)
    assert first.header_sha256 == second.header_sha256
    assert first.frame_sha256 == second.frame_sha256
    assert first.token_ids_sha256 == second.token_ids_sha256


def test_context_chunking_masks_and_reassembly_never_drop_bytes() -> None:
    context_limit = 31
    codec = HFNativeRawByteCodec(_bytelevel_tokenizer(model_max_length=context_limit))
    value = np.arange(257, dtype=np.uint8)

    encoded = codec.encode_tensor(value, modality="raw")
    reassembled = tuple(
        token_id for chunk in encoded.chunks for token_id in chunk.input_ids
    )

    assert len(encoded.chunks) > 1
    assert reassembled == encoded.input_ids
    assert all(0 < len(chunk.input_ids) <= context_limit for chunk in encoded.chunks)
    assert all(
        chunk.attention_mask == (1,) * len(chunk.input_ids) for chunk in encoded.chunks
    )
    assert all(
        chunk.to_hf_batch()
        == {
            "input_ids": [list(chunk.input_ids)],
            "attention_mask": [list(chunk.attention_mask)],
        }
        for chunk in encoded.chunks
    )
    report = encoded.report()
    assert report["context_limit"] == context_limit
    assert report["all_chunks_within_context"] is True
    assert report["tokens_reassembled_exactly"] is True
    assert report["bytes_dropped"] == 0
    assert sum(report["chunk_token_counts"]) == len(encoded.input_ids)
    decoded = codec.decode_chunks(encoded.chunks, expected_report=report)
    assert decoded.array.tobytes() == value.tobytes()
    assert decoded.report()["bytes_dropped"] == 0


def test_existing_lm_embedding_consumes_ids_without_new_parameters() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _bytelevel_tokenizer(model_max_length=512)
    codec = HFNativeRawByteCodec(tokenizer, context_limit=512)

    class Model:
        def __init__(self) -> None:
            self.embedding = torch.nn.Embedding(len(tokenizer), 7)
            self.config = SimpleNamespace(max_position_embeddings=512)

        def get_input_embeddings(self):
            return self.embedding

    model = Model()
    parameters_before = sum(
        parameter.numel() for parameter in model.embedding.parameters()
    )
    encoded = codec.encode_tensor(
        np.array([0, 127, 128, 255], dtype=np.uint8),
        modality="raw",
    )
    batch = encoded.chunks[0].to_hf_batch(return_tensors="pt")
    embeddings = model.get_input_embeddings()(batch["input_ids"])
    validation = codec.validate_model_input_embedding(model)

    assert embeddings.shape == (1, len(encoded.chunks[0].input_ids), 7)
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert sum(parameter.numel() for parameter in model.embedding.parameters()) == (
        parameters_before
    )
    assert validation["all_byte_ids_within_input_embedding"] is True
    assert validation["parameter_count_added"] == 0
    assert validation["learned_modules_added"] == []


def test_payload_corruption_and_wrong_expected_hash_fail_closed() -> None:
    codec = HFNativeRawByteCodec(_bytelevel_tokenizer(), context_limit=97)
    encoded = codec.encode_tensor(
        np.arange(20, dtype=np.int16),
        modality="audio",
    )
    corrupt = list(encoded.input_ids)
    original_byte = codec.mapping.byte_for_token_id(
        corrupt[encoded.payload_token_offset]
    )
    corrupt[encoded.payload_token_offset] = codec.mapping.token_id_for_byte(
        (original_byte + 1) % 256
    )

    with pytest.raises(RawTensorDecodeError, match="payload SHA-256"):
        codec.decode_tensor(corrupt)

    frame = codec.mapping.decode_token_ids(encoded.input_ids)
    header = json.loads(
        frame[
            encoded.payload_token_offset
            - encoded.header_bytes : encoded.payload_token_offset
        ].decode("ascii")
    )
    header["dtype"] = "<u2"
    changed_header = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    changed_frame = (
        b"NFHFRAW1:"
        + f"{len(changed_header):08x}".encode("ascii")
        + b":"
        + changed_header
        + frame[encoded.payload_token_offset :]
    )
    with pytest.raises(RawTensorDecodeError, match="content SHA-256"):
        codec.decode_tensor(codec.mapping.encode_bytes(changed_frame))

    wrong_report = encoded.report()
    wrong_report["frame_sha256"] = "0" * 64
    with pytest.raises(RawTensorDecodeError, match="frame_sha256"):
        codec.decode_tensor(encoded.input_ids, expected_report=wrong_report)

    unknown_id = max(codec.mapping.token_ids) + 1
    with pytest.raises(RawTensorDecodeError, match="not a byte atom"):
        codec.decode_tensor([unknown_id])


def test_missing_or_non_bytelevel_tokenizer_components_fail_closed() -> None:
    missing = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())[0]
    with pytest.raises(ByteLevelMappingError, match="missing 1 byte atoms"):
        discover_byte_token_mapping(_bytelevel_tokenizer(missing_atom=missing))
    with pytest.raises(ByteLevelMappingError, match="pre-tokenizer"):
        discover_byte_token_mapping(_bytelevel_tokenizer(bytelevel_pre_tokenizer=False))
    with pytest.raises(ByteLevelMappingError, match="decoder"):
        discover_byte_token_mapping(_bytelevel_tokenizer(bytelevel_decoder=False))


def test_bounds_dtype_metadata_and_context_are_strict() -> None:
    tokenizer = _bytelevel_tokenizer(model_max_length=10**20)
    codec_without_context = HFNativeRawByteCodec(tokenizer)
    value = np.arange(4, dtype=np.uint8)

    with pytest.raises(RawTensorContextError, match="context_limit is required"):
        codec_without_context.encode_tensor(value, modality="raw")
    with pytest.raises(RawTensorContextError, match="positive integer"):
        HFNativeRawByteCodec(tokenizer, context_limit=0)
    with pytest.raises(RawTensorEncodeError, match="NumPy ndarray"):
        HFNativeRawByteCodec(tokenizer, context_limit=64).encode_tensor(
            [1, 2, 3],  # type: ignore[arg-type]
            modality="raw",
        )
    with pytest.raises(RawTensorEncodeError, match="object"):
        HFNativeRawByteCodec(tokenizer, context_limit=64).encode_tensor(
            np.array([object()], dtype=object),
            modality="raw",
        )
    with pytest.raises(RawTensorEncodeError, match="finite"):
        HFNativeRawByteCodec(tokenizer, context_limit=64).encode_tensor(
            value,
            modality="raw",
            metadata={"bad": float("nan")},
        )

    bounded = HFNativeRawByteCodec(
        tokenizer,
        context_limit=64,
        limits=RawByteCodecLimits(max_payload_bytes=3),
    )
    with pytest.raises(RawTensorBoundsError, match="max_payload_bytes"):
        bounded.encode_tensor(value, modality="raw")


def test_truncated_or_reordered_chunks_are_rejected() -> None:
    codec = HFNativeRawByteCodec(_bytelevel_tokenizer(), context_limit=29)
    encoded = codec.encode_tensor(np.arange(80, dtype=np.uint8), modality="raw")
    assert len(encoded.chunks) > 2

    with pytest.raises(RawTensorDecodeError, match="truncated"):
        codec.decode_chunks(encoded.chunks[:-1])
    with pytest.raises(RawTensorDecodeError, match="missing or out of order"):
        codec.decode_chunks((encoded.chunks[1], encoded.chunks[0]))


@pytest.mark.parametrize(
    "value",
    [
        np.array(-0.0, dtype=np.float64),
        np.empty((0, 3), dtype=np.uint8),
    ],
)
def test_scalar_and_empty_shapes_roundtrip_exactly(value: np.ndarray) -> None:
    codec = HFNativeRawByteCodec(_bytelevel_tokenizer(), context_limit=43)

    encoded = codec.encode_tensor(value, modality="tensor")
    decoded = codec.decode_chunks(encoded.chunks, expected_report=encoded.report())

    assert decoded.shape == value.shape
    assert decoded.dtype == value.dtype.str
    assert decoded.array.tobytes() == value.tobytes()
    assert decoded.report()["rank"] == value.ndim


def test_protocol_report_explicitly_proves_parameter_free_generic_path() -> None:
    codec = HFNativeRawByteCodec(_bytelevel_tokenizer(), context_limit=128)

    report = codec.protocol_report()

    assert report["serialization_path"] == SERIALIZATION_PATH
    assert report["mapping"]["byte_token_count"] == 256
    assert report["parameter_count"] == 0
    assert report["learned_modules"] == []
    assert report["encoder"] is None
    assert report["decoder"] is None
    assert report["adapter"] is None
    assert report["projector"] is None
    assert report["bridge"] is None
    assert report["runtime_donor"] is None
    assert report["modality_specific_codecs"] == []
