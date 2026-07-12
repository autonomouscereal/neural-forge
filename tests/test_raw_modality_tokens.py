from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from neural_forge.raw_modality_tokens import (
    BYTE_TOKEN_STRINGS,
    FRAME_END_TOKEN,
    HEADER_END_TOKEN,
    RAW_TOKEN_STRINGS,
    CodecLimits,
    RawFrameStreamDecoder,
    RawModality,
    RawModalityCodec,
    RawTokenBoundsError,
    RawTokenDecodeError,
    RawTokenEncodeError,
    RawTokenVocabulary,
    RawVocabularyError,
    codec_from_hf_tokenizer,
    decode_token_chunks,
    expand_hf_raw_token_vocabulary,
    iter_token_chunks,
    plan_hf_raw_token_expansion,
)


def _payload_tokens(codec: RawModalityCodec, tokens: list[int]) -> list[int]:
    header_end = tokens.index(codec.vocabulary.id_for(HEADER_END_TOKEN))
    assert tokens[-1] == codec.vocabulary.id_for(FRAME_END_TOKEN)
    return tokens[header_end + 1 : -1]


def test_uint8_image_round_trip_is_exact_and_deterministic() -> None:
    codec = RawModalityCodec()
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)[::-1]

    first = codec.encode_image(image)
    second = codec.encode(RawModality.IMAGE, image)
    decoded = codec.decode(first)

    assert first == second
    assert decoded.modality is RawModality.IMAGE
    assert decoded.dtype == "uint8"
    assert decoded.shape == image.shape
    assert decoded.payload_length == image.nbytes
    assert len(decoded.checksum_sha256) == 64
    assert isinstance(decoded.value, np.ndarray)
    assert decoded.value.flags.c_contiguous
    assert np.array_equal(decoded.value, image)
    assert np.array_equal(codec.decode_image(first), image)

    channel_first = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    assert np.array_equal(
        codec.decode_image(codec.encode_image(channel_first)), channel_first
    )


def test_pcm16_audio_round_trip_preserves_samples_shape_and_rate() -> None:
    codec = RawModalityCodec()
    samples = np.array([[-32768, 32767], [-1234, 1234], [0, -1]], dtype=np.int16)

    tokens = codec.encode_audio(samples, sample_rate=48_000)
    decoded = codec.decode(tokens)
    direct_samples, sample_rate = codec.decode_audio(tokens)

    assert decoded.modality is RawModality.AUDIO
    assert decoded.dtype == "int16"
    assert decoded.metadata == {"sample_rate": 48_000}
    assert np.array_equal(decoded.value, samples)
    assert np.array_equal(direct_samples, samples)
    assert sample_rate == 48_000


def test_uint8_video_round_trip_preserves_frames_and_frame_rate() -> None:
    codec = RawModalityCodec()
    frames = np.arange(3 * 2 * 4 * 3, dtype=np.uint8).reshape(3, 2, 4, 3)

    tokens = codec.encode_video(frames, frame_rate=29.97)
    decoded = codec.decode(tokens)

    assert decoded.modality is RawModality.VIDEO
    assert decoded.metadata == {"frame_rate": 29.97}
    assert decoded.shape == frames.shape
    assert np.array_equal(decoded.value, frames)
    assert np.array_equal(codec.decode_video(tokens), frames)

    channel_first = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 3, 4, 5)
    assert np.array_equal(
        codec.decode_video(codec.encode_video(channel_first)), channel_first
    )


def test_utf8_text_and_ocr_are_distinct_frames_with_exact_strings() -> None:
    codec = RawModalityCodec()
    text = "plain text: caf\u00e9, \u6771\u4eac, \U0001f642"
    ocr = "invoice OCR\nline 2: \u20ac42.50"

    text_tokens = codec.encode_text(text)
    ocr_tokens = codec.encode_ocr(ocr)
    decoded_text = codec.decode(text_tokens)
    decoded_ocr = codec.decode(ocr_tokens)

    assert decoded_text.modality is RawModality.TEXT
    assert decoded_ocr.modality is RawModality.OCR
    assert decoded_text.dtype == decoded_ocr.dtype == "utf8"
    assert decoded_text.shape == (len(text.encode("utf-8")),)
    assert decoded_ocr.shape == (len(ocr.encode("utf-8")),)
    assert codec.decode_text(text_tokens) == text
    assert codec.decode_ocr(ocr_tokens) == ocr


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32])
def test_quantized_points_round_trip_exactly_for_supported_integer_dtypes(
    dtype: type[np.generic],
) -> None:
    codec = RawModalityCodec()
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
    scale = (0.25, 0.5, 2.0)
    zero_point = (1, 2, 3)

    tokens = codec.encode_points3d(points, scale=scale, zero_point=zero_point)
    decoded = codec.decode(tokens)
    direct_points, quantization = codec.decode_points3d(tokens)

    assert decoded.modality is RawModality.POINTS3D
    assert decoded.dtype == np.dtype(dtype).name
    assert decoded.metadata == {
        "scale": [0.25, 0.5, 2.0],
        "zero_point": [1, 2, 3],
    }
    assert np.array_equal(decoded.value, points)
    assert np.array_equal(direct_points, points)
    expected_float = (points.astype(np.float64) - np.array(zero_point)) * np.array(
        scale
    )
    assert np.array_equal(quantization.dequantize(direct_points), expected_float)


def test_every_modality_uses_one_shared_byte_vocabulary_and_payload_path() -> None:
    codec = RawModalityCodec(token_id_offset=1000)
    byte_ids = set(codec.vocabulary.byte_token_ids)
    image = np.array([[65]], dtype=np.uint8)
    audio = np.array([65], dtype=np.int16)
    video = np.array([[[65]]], dtype=np.uint8)
    points = np.array([[65, 0, -1]], dtype=np.int8)
    frames = [
        codec.encode_image(image),
        codec.encode_audio(audio),
        codec.encode_video(video),
        codec.encode_text("A"),
        codec.encode_ocr("A"),
        codec.encode_points3d(points),
    ]

    for frame in frames:
        assert set(_payload_tokens(codec, frame)) <= byte_ids
    expected_a = codec.byte_token_id(65)
    assert _payload_tokens(codec, frames[0]) == [expected_a]
    assert _payload_tokens(codec, frames[3]) == [expected_a]
    assert _payload_tokens(codec, frames[4]) == [expected_a]

    report = codec.protocol_report()
    assert report["shared_byte_token_count"] == 256
    assert report["modality_specific_payload_token_count"] == 0
    assert report["learned_components"] == []
    assert len(RAW_TOKEN_STRINGS) == len(BYTE_TOKEN_STRINGS) + 10


def test_payload_header_modality_truncation_and_trailing_corruption_are_rejected() -> (
    None
):
    codec = RawModalityCodec()
    original = codec.encode_image(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
    header_end = original.index(codec.vocabulary.id_for(HEADER_END_TOKEN))

    payload_corrupt = original.copy()
    payload_corrupt[header_end + 1] = codec.byte_token_id(255)
    with pytest.raises(RawTokenDecodeError, match="checksum"):
        codec.decode(payload_corrupt)

    digest_corrupt = original.copy()
    digest_corrupt[header_end - 1] = codec.byte_token_id(
        (codec.vocabulary.byte_token_ids.index(digest_corrupt[header_end - 1]) + 1)
        % 256
    )
    with pytest.raises(RawTokenDecodeError, match="checksum"):
        codec.decode(digest_corrupt)

    modality_corrupt = original.copy()
    modality_corrupt[2] = codec.vocabulary.id_for("<|raw:video|>")
    with pytest.raises(RawTokenDecodeError, match="modality"):
        codec.decode(modality_corrupt)

    truncated = original[: header_end + 2] + [original[-1]]
    with pytest.raises(RawTokenDecodeError, match="Payload token count"):
        codec.decode(truncated)

    trailing = original + [codec.byte_token_id(0)]
    with pytest.raises(RawTokenDecodeError, match="trailing"):
        codec.decode(trailing)

    control_in_payload = original.copy()
    control_in_payload[header_end + 1] = codec.vocabulary.id_for(HEADER_END_TOKEN)
    with pytest.raises(RawTokenDecodeError, match="Non-byte"):
        codec.decode(control_in_payload)


def test_strict_dtype_shape_metadata_and_unicode_validation() -> None:
    codec = RawModalityCodec()

    with pytest.raises(RawTokenEncodeError, match="uint8"):
        codec.encode_image(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(RawTokenEncodeError, match="int16"):
        codec.encode_audio(np.ones(4, dtype=np.int32))
    with pytest.raises(RawTokenEncodeError, match="rank 3 or rank 4"):
        codec.encode_video(np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(RawTokenEncodeError, match="point_count, 3"):
        codec.encode_points3d(np.ones((4, 4), dtype=np.int16))
    with pytest.raises(RawTokenEncodeError, match="greater than zero"):
        codec.encode_points3d(np.ones((2, 3), dtype=np.int16), scale=(1.0, 0.0, 1.0))
    with pytest.raises(RawTokenEncodeError, match="quantized dtype"):
        codec.encode_points3d(np.ones((2, 3), dtype=np.uint8), zero_point=256)
    with pytest.raises(RawTokenEncodeError, match="valid Unicode"):
        codec.encode_text("\ud800")
    with pytest.raises(RawTokenEncodeError, match="sample_rate"):
        codec.encode_audio(np.ones(2, dtype=np.int16), sample_rate=0)


def test_configured_bounds_apply_during_encoding_and_decoding() -> None:
    small = RawModalityCodec(limits=CodecLimits(max_payload_bytes=8, max_text_bytes=3))
    with pytest.raises(RawTokenBoundsError, match="max_payload_bytes"):
        small.encode_image(np.arange(9, dtype=np.uint8).reshape(3, 3))
    with pytest.raises(RawTokenBoundsError, match="max_text_bytes"):
        small.encode_text("four")

    normal = RawModalityCodec()
    frame = normal.encode_audio(np.arange(5, dtype=np.int16))
    with pytest.raises(RawTokenBoundsError, match="max_payload_bytes"):
        small.decode(frame)


def test_chunk_helpers_and_stream_decoder_handle_arbitrary_boundaries() -> None:
    codec = RawModalityCodec()
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    text = "streamed UTF-8: \u00e9"
    audio = np.array([-2, -1, 0, 1, 2], dtype=np.int16)
    encoded = [
        codec.encode_image(image),
        codec.encode_text(text),
        codec.encode_audio(audio, sample_rate=22_050),
    ]
    concatenated = [token for frame in encoded for token in frame]
    chunks = list(iter_token_chunks(concatenated, 7))

    decoded = decode_token_chunks(chunks, codec)

    assert [item.modality for item in decoded] == [
        RawModality.IMAGE,
        RawModality.TEXT,
        RawModality.AUDIO,
    ]
    assert np.array_equal(decoded[0].value, image)
    assert decoded[1].value == text
    assert np.array_equal(decoded[2].value, audio)
    assert decoded[2].metadata["sample_rate"] == 22_050
    assert [token for chunk in chunks for token in chunk] == concatenated


def test_stream_decoder_rejects_incomplete_unknown_and_post_failure_input() -> None:
    codec = RawModalityCodec()
    frame = codec.encode_text("incomplete")
    decoder = RawFrameStreamDecoder(codec)
    assert decoder.feed(frame[:-1]) == []
    with pytest.raises(RawTokenDecodeError, match="incomplete"):
        decoder.finish()
    with pytest.raises(RawTokenDecodeError, match="failed"):
        decoder.feed(frame)

    decoder.reset()
    with pytest.raises(RawTokenDecodeError, match="Unknown token ID"):
        decoder.feed([max(codec.vocabulary.allowed_ids) + 1])


def test_vocabulary_validation_rejects_duplicate_ids_and_bad_chunks() -> None:
    vocabulary = {token: index for index, token in enumerate(RAW_TOKEN_STRINGS)}
    vocabulary[RAW_TOKEN_STRINGS[-1]] = vocabulary[RAW_TOKEN_STRINGS[-2]]
    with pytest.raises(RawVocabularyError, match="unique"):
        RawTokenVocabulary(vocabulary)
    with pytest.raises(ValueError, match="positive"):
        list(iter_token_chunks([1, 2], 0))
    with pytest.raises(TypeError, match="integer"):
        list(iter_token_chunks([1, 1.5], 2))


class _FakeTokenizer:
    def __init__(self) -> None:
        self.vocabulary = {"base-a": 0, "base-b": 1, "base-c": 2}
        self.added_as_special: list[str] = []

    def __len__(self) -> int:
        return len(self.vocabulary)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def add_tokens(self, tokens: list[str], *, special_tokens: bool) -> int:
        assert special_tokens is True
        added = 0
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
                self.added_as_special.append(token)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocabulary[token]


def _fake_untied_model(torch, rows: int = 3, width: int = 8):
    class Model:
        def __init__(self) -> None:
            self.input_embeddings = torch.nn.Embedding(rows, width)
            self.output_head = torch.nn.Linear(width, rows, bias=True)
            self.config = SimpleNamespace(tie_word_embeddings=False, vocab_size=rows)
            with torch.no_grad():
                self.input_embeddings.weight.copy_(
                    torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
                    / 10.0
                )
                self.output_head.weight.copy_(
                    torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
                    / 7.0
                    - 1.0
                )
                self.output_head.bias.copy_(torch.linspace(-0.5, 0.5, rows))

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_head

        def resize_token_embeddings(self, size: int):
            old_input = self.input_embeddings
            old_output = self.output_head
            new_input = torch.nn.Embedding(size, width)
            new_output = torch.nn.Linear(width, size, bias=True)
            with torch.no_grad():
                new_input.weight.fill_(99.0)
                new_output.weight.fill_(-99.0)
                new_output.bias.fill_(42.0)
                new_input.weight[: old_input.weight.shape[0]].copy_(old_input.weight)
                new_output.weight[: old_output.weight.shape[0]].copy_(old_output.weight)
                new_output.bias[: old_output.bias.shape[0]].copy_(old_output.bias)
            self.input_embeddings = new_input
            self.output_head = new_output
            self.config.vocab_size = size
            return new_input

    return Model()


def test_hf_plan_and_expansion_add_only_shared_rows_to_an_untied_head() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _FakeTokenizer()
    model = _fake_untied_model(torch)
    input_before = model.get_input_embeddings().weight.detach().clone()
    output_before = model.get_output_embeddings().weight.detach().clone()
    bias_before = model.get_output_embeddings().bias.detach().clone()

    plan = plan_hf_raw_token_expansion(tokenizer, model)

    assert plan["mutated"] is False
    assert plan["missing_token_count"] == len(RAW_TOKEN_STRINGS)
    assert plan["shared_byte_token_count"] == 256
    assert plan["model"]["weights_tied"] is False
    assert plan["architecture"]["learned_modality_modules"] == []
    assert plan["architecture"]["only_token_rows_are_added"] is True
    assert len(tokenizer) == 3

    report = expand_hf_raw_token_vocabulary(tokenizer, model)

    expected_rows = 3 + len(RAW_TOKEN_STRINGS)
    assert report["status"] == "applied"
    assert report["added_token_count"] == len(RAW_TOKEN_STRINGS)
    assert report["all_tokens_atomic"] is True
    assert tokenizer.added_as_special == list(RAW_TOKEN_STRINGS)
    assert model.get_input_embeddings().weight.shape == (expected_rows, 8)
    assert model.get_output_embeddings().weight.shape == (expected_rows, 8)
    assert model.get_output_embeddings().bias.shape == (expected_rows,)
    assert (
        model.get_input_embeddings().weight.data_ptr()
        != model.get_output_embeddings().weight.data_ptr()
    )
    assert torch.equal(model.get_input_embeddings().weight[:3], input_before)
    assert torch.equal(model.get_output_embeddings().weight[:3], output_before)
    assert torch.equal(model.get_output_embeddings().bias[:3], bias_before)
    assert not torch.allclose(
        model.get_input_embeddings().weight[3],
        model.get_input_embeddings().weight[4],
    )
    assert json.dumps(report, sort_keys=True, allow_nan=False)

    codec = codec_from_hf_tokenizer(tokenizer)
    assert codec.decode_text(codec.encode_text("HF raw path")) == "HF raw path"


def test_hf_row_initialization_is_deterministic_and_expansion_is_idempotent() -> None:
    torch = pytest.importorskip("torch")
    tokenizer_a = _FakeTokenizer()
    tokenizer_b = _FakeTokenizer()
    model_a = _fake_untied_model(torch)
    model_b = _fake_untied_model(torch)

    expand_hf_raw_token_vocabulary(tokenizer_a, model_a)
    expand_hf_raw_token_vocabulary(tokenizer_b, model_b)

    assert tokenizer_a.get_vocab() == tokenizer_b.get_vocab()
    assert torch.equal(
        model_a.get_input_embeddings().weight,
        model_b.get_input_embeddings().weight,
    )
    assert torch.equal(
        model_a.get_output_embeddings().weight,
        model_b.get_output_embeddings().weight,
    )
    assert torch.equal(
        model_a.get_output_embeddings().bias,
        model_b.get_output_embeddings().bias,
    )

    input_before = model_a.get_input_embeddings().weight.detach().clone()
    output_before = model_a.get_output_embeddings().weight.detach().clone()
    second = expand_hf_raw_token_vocabulary(tokenizer_a, model_a)

    assert second["added_token_count"] == 0
    assert second["initialization"]["input_token_ids"] == []
    assert second["initialization"]["output_token_ids"] == []
    assert torch.equal(model_a.get_input_embeddings().weight, input_before)
    assert torch.equal(model_a.get_output_embeddings().weight, output_before)


def test_hf_expansion_rejects_tied_output_head_before_tokenizer_mutation() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _FakeTokenizer()

    class TiedModel:
        def __init__(self) -> None:
            self.shared = torch.nn.Embedding(3, 4)

        def get_input_embeddings(self):
            return self.shared

        def get_output_embeddings(self):
            return self.shared

    with pytest.raises(RawVocabularyError, match="untied"):
        expand_hf_raw_token_vocabulary(tokenizer, TiedModel())
    assert len(tokenizer) == 3
