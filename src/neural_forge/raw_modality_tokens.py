"""Deterministic raw multimodal framing over one shared token vocabulary.

The codec in this module is deliberately mechanical.  It does not contain a
learned encoder, decoder, projector, adapter, bridge, or modality-specific
payload vocabulary.  Every payload and header byte uses the same 256 byte
tokens; a small set of control tokens provides framing and modality identity.

Vocabulary-expansion utility, not the canonical folding transport
-----------------------------------------------------------------
This protocol (``NFRT``) adds 266 special tokens -- ten control tokens and 256
byte tokens -- to a tokenizer and to the model's embedding and output rows.
That is its purpose and it is fully supported here, but it means the tokenizer
fingerprint after expansion differs from the parent's.  Every folding lane
that compares parent and child ``ModelCircuitGraph.tokenizer_fingerprint``
values for equality, replays a sealed parent behavior cache, or requires exact
state restore is therefore **closed** to this protocol until a
fingerprint-lineage bridge exists.

The canonical raw-byte transport for folding lanes is
:mod:`neural_forge.hf_native_raw_tokens` (``NFHFRAW1``), which reuses the 256
ByteLevel byte tokens a tokenizer already has and adds zero rows.  Call
:func:`raw_token_lane_compatibility` -- re-exported here from that module so
both protocols quote one statement -- for the lane-by-lane matrix and for what
a bridge would require.

Modality identity uses the canonical names in
:mod:`neural_forge.dynamic_circuit_ir`; :class:`RawModality` members are
defined from those constants so the enum and the IR cannot drift apart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import operator
import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from .dynamic_circuit_ir import (
    MODALITY_AUDIO,
    MODALITY_IMAGE,
    MODALITY_OCR,
    MODALITY_POINTS3D,
    MODALITY_TEXT,
    MODALITY_VIDEO,
)
from .hf_native_raw_tokens import (
    CANONICAL_RAW_TRANSPORT,
    LANE_COMPATIBILITY_SCHEMA,
    raw_token_lane_compatibility,
)


PROTOCOL_NAME = "neural-forge-raw-token"
PROTOCOL_VERSION = 1

FRAME_START_TOKEN = "<|raw:frame-start|>"
VERSION_TOKEN = "<|raw:version-1|>"
IMAGE_TOKEN = "<|raw:image|>"
AUDIO_TOKEN = "<|raw:audio|>"
VIDEO_TOKEN = "<|raw:video|>"
TEXT_TOKEN = "<|raw:text|>"
OCR_TOKEN = "<|raw:ocr|>"
POINTS3D_TOKEN = "<|raw:points3d|>"
HEADER_END_TOKEN = "<|raw:header-end|>"
FRAME_END_TOKEN = "<|raw:frame-end|>"

CONTROL_TOKEN_STRINGS = (
    FRAME_START_TOKEN,
    VERSION_TOKEN,
    IMAGE_TOKEN,
    AUDIO_TOKEN,
    VIDEO_TOKEN,
    TEXT_TOKEN,
    OCR_TOKEN,
    POINTS3D_TOKEN,
    HEADER_END_TOKEN,
    FRAME_END_TOKEN,
)
BYTE_TOKEN_STRINGS = tuple(f"<|raw:byte:{value:02x}|>" for value in range(256))
RAW_TOKEN_STRINGS = CONTROL_TOKEN_STRINGS + BYTE_TOKEN_STRINGS

_MAGIC = b"NFRT"
_DIGEST_SIZE = hashlib.sha256().digest_size
_HEADER_PREFIX = struct.Struct(">4sBBBBQH")
_DIMENSION = struct.Struct(">I")
_MAX_STATISTIC_ROWS = 4096


class RawTokenError(ValueError):
    """Base error for invalid raw-token input or framing."""


class RawTokenEncodeError(RawTokenError):
    """Raised when a native value cannot be encoded without conversion."""


class RawTokenDecodeError(RawTokenError):
    """Raised when a frame is malformed, truncated, or corrupt."""


class RawTokenBoundsError(RawTokenError):
    """Raised when configured resource bounds would be exceeded."""


class RawVocabularyError(RawTokenError):
    """Raised for an invalid tokenizer/model vocabulary expansion."""


class RawModality(str, Enum):
    """Codec modality identity, spelled with canonical IR modality names."""

    IMAGE = MODALITY_IMAGE
    AUDIO = MODALITY_AUDIO
    VIDEO = MODALITY_VIDEO
    TEXT = MODALITY_TEXT
    OCR = MODALITY_OCR
    POINTS3D = MODALITY_POINTS3D

    @classmethod
    def coerce(cls, value: RawModality | str) -> RawModality:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise RawTokenEncodeError(
                f"Unsupported modality {value!r}; expected one of: {supported}."
            ) from exc


_MODALITY_CODE = {
    RawModality.IMAGE: 1,
    RawModality.AUDIO: 2,
    RawModality.VIDEO: 3,
    RawModality.TEXT: 4,
    RawModality.OCR: 5,
    RawModality.POINTS3D: 6,
}
_CODE_MODALITY = {code: modality for modality, code in _MODALITY_CODE.items()}
_MODALITY_TOKEN = {
    RawModality.IMAGE: IMAGE_TOKEN,
    RawModality.AUDIO: AUDIO_TOKEN,
    RawModality.VIDEO: VIDEO_TOKEN,
    RawModality.TEXT: TEXT_TOKEN,
    RawModality.OCR: OCR_TOKEN,
    RawModality.POINTS3D: POINTS3D_TOKEN,
}
_TOKEN_MODALITY = {token: modality for modality, token in _MODALITY_TOKEN.items()}

_DTYPE_UINT8 = 1
_DTYPE_INT16 = 2
_DTYPE_UTF8 = 3
_DTYPE_INT8 = 4
_DTYPE_UINT16 = 5
_DTYPE_INT32 = 6
_NUMPY_DTYPE_TO_CODE = {
    np.dtype(np.uint8): _DTYPE_UINT8,
    np.dtype(np.int16): _DTYPE_INT16,
    np.dtype(np.int8): _DTYPE_INT8,
    np.dtype(np.uint16): _DTYPE_UINT16,
    np.dtype(np.int32): _DTYPE_INT32,
}
_CODE_TO_NUMPY_DTYPE = {code: dtype for dtype, code in _NUMPY_DTYPE_TO_CODE.items()}
_POINT_DTYPES = frozenset(
    {
        np.dtype(np.int8),
        np.dtype(np.uint8),
        np.dtype(np.int16),
        np.dtype(np.uint16),
        np.dtype(np.int32),
    }
)


@dataclass(frozen=True)
class CodecLimits:
    """Hard resource bounds checked during both encoding and decoding."""

    max_payload_bytes: int = 64 * 1024 * 1024
    max_text_bytes: int = 16 * 1024 * 1024
    max_header_bytes: int = 4096
    max_metadata_bytes: int = 2048
    max_rank: int = 8
    max_dimension: int = 64_000_000
    max_elements: int = 64_000_000
    max_audio_samples: int = 10_000_000
    max_video_frames: int = 100_000
    max_points: int = 10_000_000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        minimum_header = _HEADER_PREFIX.size + _DIMENSION.size + _DIGEST_SIZE + 2
        if self.max_header_bytes < minimum_header:
            raise ValueError(
                f"max_header_bytes must be at least {minimum_header} bytes."
            )


@dataclass(frozen=True)
class PointQuantization:
    """Per-axis affine quantization metadata for integer XYZ points."""

    scale: tuple[float, float, float]
    zero_point: tuple[int, int, int]

    def dequantize(self, points: np.ndarray) -> np.ndarray:
        if not isinstance(points, np.ndarray) or points.ndim != 2:
            raise ValueError("points must be a rank-2 NumPy array.")
        if points.shape[1] != 3 or points.dtype not in _POINT_DTYPES:
            raise ValueError(
                "points must have shape (point_count, 3) and integer dtype."
            )
        scale = np.asarray(self.scale, dtype=np.float64)
        zero_point = np.asarray(self.zero_point, dtype=np.float64)
        return (points.astype(np.float64) - zero_point) * scale


@dataclass(frozen=True)
class DecodedRawFrame:
    """Validated value and provenance recovered from one complete frame."""

    modality: RawModality
    value: np.ndarray | str
    shape: tuple[int, ...]
    dtype: str
    metadata: Mapping[str, Any]
    payload_length: int
    checksum_sha256: str

    @property
    def quantization(self) -> PointQuantization | None:
        if self.modality is not RawModality.POINTS3D:
            return None
        return PointQuantization(
            scale=tuple(float(value) for value in self.metadata["scale"]),
            zero_point=tuple(int(value) for value in self.metadata["zero_point"]),
        )


@dataclass(frozen=True)
class RawTokenVocabulary:
    """Validated mapping from the fixed protocol strings to model token IDs."""

    token_to_id: Mapping[str, int]

    def __post_init__(self) -> None:
        source = dict(self.token_to_id)
        missing = [token for token in RAW_TOKEN_STRINGS if token not in source]
        if missing:
            raise RawVocabularyError(
                f"Raw token vocabulary is missing {len(missing)} required tokens."
            )
        normalized: dict[str, int] = {}
        for token in RAW_TOKEN_STRINGS:
            normalized[token] = _integer_id(source[token], f"ID for {token}")
            if normalized[token] < 0:
                raise RawVocabularyError(f"ID for {token} must be non-negative.")
        if len(set(normalized.values())) != len(normalized):
            raise RawVocabularyError("Every raw token must have a unique token ID.")
        object.__setattr__(self, "token_to_id", MappingProxyType(normalized))

    @classmethod
    def contiguous(cls, start_id: int = 0) -> RawTokenVocabulary:
        start = _integer_id(start_id, "start_id")
        if start < 0:
            raise RawVocabularyError("start_id must be non-negative.")
        return cls(
            {token: start + index for index, token in enumerate(RAW_TOKEN_STRINGS)}
        )

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> RawTokenVocabulary:
        vocabulary = _tokenizer_vocabulary(tokenizer)
        return cls(vocabulary)

    @property
    def id_to_token(self) -> Mapping[int, str]:
        return MappingProxyType(
            {token_id: token for token, token_id in self.token_to_id.items()}
        )

    @property
    def byte_token_ids(self) -> tuple[int, ...]:
        return tuple(self.token_to_id[token] for token in BYTE_TOKEN_STRINGS)

    @property
    def allowed_ids(self) -> frozenset[int]:
        return frozenset(self.token_to_id.values())

    def id_for(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise RawVocabularyError(f"Unknown raw protocol token: {token!r}.") from exc

    def byte_id(self, value: int) -> int:
        byte_value = _integer_id(value, "byte value")
        if not 0 <= byte_value <= 255:
            raise RawVocabularyError("byte value must be between 0 and 255.")
        return self.token_to_id[BYTE_TOKEN_STRINGS[byte_value]]


def raw_token_vocabulary(start_id: int = 0) -> dict[str, int]:
    """Return a mutable copy of the canonical contiguous raw vocabulary."""

    return dict(RawTokenVocabulary.contiguous(start_id).token_to_id)


class RawModalityCodec:
    """Encode native values into deterministic, checksummed token frames."""

    def __init__(
        self,
        *,
        vocabulary: RawTokenVocabulary | Mapping[str, int] | None = None,
        token_id_offset: int = 0,
        limits: CodecLimits | None = None,
    ) -> None:
        if vocabulary is not None and token_id_offset != 0:
            raise ValueError("Specify either vocabulary or token_id_offset, not both.")
        if vocabulary is None:
            self.vocabulary = RawTokenVocabulary.contiguous(token_id_offset)
        elif isinstance(vocabulary, RawTokenVocabulary):
            self.vocabulary = vocabulary
        else:
            self.vocabulary = RawTokenVocabulary(vocabulary)
        self.limits = limits or CodecLimits()
        self._id_to_byte = {
            self.vocabulary.id_for(token): value
            for value, token in enumerate(BYTE_TOKEN_STRINGS)
        }
        self._modality_id_to_modality = {
            self.vocabulary.id_for(token): modality
            for token, modality in _TOKEN_MODALITY.items()
        }

    @property
    def max_frame_tokens(self) -> int:
        return self.limits.max_payload_bytes + self.limits.max_header_bytes + 5

    def byte_token_id(self, value: int) -> int:
        return self.vocabulary.byte_id(value)

    def protocol_report(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "framing": [
                "frame_start",
                "version",
                "modality",
                "header_bytes",
                "header_end",
                "payload_bytes",
                "frame_end",
            ],
            "control_token_count": len(CONTROL_TOKEN_STRINGS),
            "shared_byte_token_count": len(BYTE_TOKEN_STRINGS),
            "total_raw_token_count": len(RAW_TOKEN_STRINGS),
            "modality_specific_payload_token_count": 0,
            "learned_components": [],
            "checksum": "sha256(header_core || payload)",
            "limits": dict(vars(self.limits)),
            "token_ids": dict(self.vocabulary.token_to_id),
        }

    def encode_image(self, image: np.ndarray) -> list[int]:
        array = _require_array(image, "image")
        if array.dtype != np.dtype(np.uint8):
            raise RawTokenEncodeError("image dtype must be exactly uint8.")
        return self._encode_array(RawModality.IMAGE, array, {})

    def encode_audio(
        self, samples: np.ndarray, *, sample_rate: int = 16_000
    ) -> list[int]:
        array = _require_array(samples, "audio samples")
        if array.dtype != np.dtype(np.int16):
            raise RawTokenEncodeError("PCM audio dtype must be exactly int16.")
        rate = _integer_id(sample_rate, "sample_rate")
        if not 1 <= rate <= 768_000:
            raise RawTokenEncodeError("sample_rate must be between 1 and 768000 Hz.")
        return self._encode_array(RawModality.AUDIO, array, {"sample_rate": rate})

    def encode_video(
        self, frames: np.ndarray, *, frame_rate: float | None = None
    ) -> list[int]:
        array = _require_array(frames, "video frames")
        if array.dtype != np.dtype(np.uint8):
            raise RawTokenEncodeError("video frame dtype must be exactly uint8.")
        metadata: dict[str, Any] = {}
        if frame_rate is not None:
            rate = _finite_number(frame_rate, "frame_rate")
            if not 0.0 < rate <= 1000.0:
                raise RawTokenEncodeError(
                    "frame_rate must be greater than zero and at most 1000."
                )
            metadata["frame_rate"] = rate
        return self._encode_array(RawModality.VIDEO, array, metadata)

    def encode_text(self, text: str) -> list[int]:
        return self._encode_text(RawModality.TEXT, text)

    def encode_ocr(self, text: str) -> list[int]:
        return self._encode_text(RawModality.OCR, text)

    def encode_points3d(
        self,
        points: np.ndarray,
        *,
        scale: float | Sequence[float] = 1.0,
        zero_point: int | Sequence[int] = 0,
    ) -> list[int]:
        array = _require_array(points, "quantized points")
        if array.dtype not in _POINT_DTYPES:
            accepted = ", ".join(sorted(dtype.name for dtype in _POINT_DTYPES))
            raise RawTokenEncodeError(
                f"quantized point dtype must be one of: {accepted}."
            )
        quantization = PointQuantization(
            scale=_normalize_scale(scale),
            zero_point=_normalize_zero_point(zero_point),
        )
        metadata = {
            "scale": list(quantization.scale),
            "zero_point": list(quantization.zero_point),
        }
        return self._encode_array(RawModality.POINTS3D, array, metadata)

    encode_point_cloud = encode_points3d

    def encode(
        self, modality: RawModality | str, value: np.ndarray | str, **kwargs: Any
    ) -> list[int]:
        resolved = RawModality.coerce(modality)
        if resolved is RawModality.IMAGE:
            if kwargs:
                raise RawTokenEncodeError("image encoding accepts no metadata options.")
            return self.encode_image(value)  # type: ignore[arg-type]
        if resolved is RawModality.AUDIO:
            return self.encode_audio(value, **kwargs)  # type: ignore[arg-type]
        if resolved is RawModality.VIDEO:
            return self.encode_video(value, **kwargs)  # type: ignore[arg-type]
        if resolved is RawModality.TEXT:
            if kwargs:
                raise RawTokenEncodeError("text encoding accepts no metadata options.")
            return self.encode_text(value)  # type: ignore[arg-type]
        if resolved is RawModality.OCR:
            if kwargs:
                raise RawTokenEncodeError("OCR encoding accepts no metadata options.")
            return self.encode_ocr(value)  # type: ignore[arg-type]
        return self.encode_points3d(value, **kwargs)  # type: ignore[arg-type]

    def iter_encode_chunks(
        self,
        modality: RawModality | str,
        value: np.ndarray | str,
        *,
        chunk_size: int,
        **kwargs: Any,
    ) -> Iterator[tuple[int, ...]]:
        yield from iter_token_chunks(self.encode(modality, value, **kwargs), chunk_size)

    def decode(self, tokens: Iterable[int]) -> DecodedRawFrame:
        frame = _bounded_token_list(tokens, self.max_frame_tokens)
        if len(frame) < 5:
            raise RawTokenDecodeError("Raw token frame is too short.")

        start_id = self.vocabulary.id_for(FRAME_START_TOKEN)
        version_id = self.vocabulary.id_for(VERSION_TOKEN)
        header_end_id = self.vocabulary.id_for(HEADER_END_TOKEN)
        frame_end_id = self.vocabulary.id_for(FRAME_END_TOKEN)
        if frame[0] != start_id:
            raise RawTokenDecodeError("Frame start token is missing or corrupt.")
        if frame[1] != version_id:
            raise RawTokenDecodeError("Unsupported or corrupt protocol version token.")
        try:
            outer_modality = self._modality_id_to_modality[frame[2]]
        except KeyError as exc:
            raise RawTokenDecodeError("Modality token is missing or corrupt.") from exc
        if frame[-1] != frame_end_id:
            raise RawTokenDecodeError(
                "Frame end token is missing, corrupt, or followed by trailing data."
            )
        try:
            header_end = frame.index(header_end_id, 3, len(frame) - 1)
        except ValueError as exc:
            raise RawTokenDecodeError(
                "Header end token is missing or corrupt."
            ) from exc

        header_tokens = frame[3:header_end]
        if len(header_tokens) > self.limits.max_header_bytes:
            raise RawTokenBoundsError("Header exceeds max_header_bytes.")
        header = self._tokens_to_bytes(header_tokens, "header")
        parsed = self._parse_header(header)
        if parsed["modality"] is not outer_modality:
            raise RawTokenDecodeError(
                "Outer modality token does not match the checksummed header."
            )

        payload_length = parsed["payload_length"]
        if payload_length > self.limits.max_payload_bytes:
            raise RawTokenBoundsError("Payload exceeds max_payload_bytes.")
        payload_tokens = frame[header_end + 1 : -1]
        if len(payload_tokens) != payload_length:
            raise RawTokenDecodeError(
                "Payload token count does not match the declared payload length."
            )
        payload = self._tokens_to_bytes(payload_tokens, "payload")
        expected_digest = hashlib.sha256(parsed["header_core"] + payload).digest()
        if not hmac.compare_digest(parsed["digest"], expected_digest):
            raise RawTokenDecodeError("Frame checksum validation failed.")

        metadata = _decode_metadata(parsed["metadata_bytes"])
        shape = parsed["shape"]
        dtype_code = parsed["dtype_code"]
        self._validate_semantics(
            outer_modality,
            dtype_code,
            shape,
            metadata,
            payload_length,
            decoding=True,
        )
        if dtype_code == _DTYPE_UTF8:
            try:
                value: np.ndarray | str = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise RawTokenDecodeError("Text payload is not valid UTF-8.") from exc
            if value.encode("utf-8") != payload:
                raise RawTokenDecodeError("Text payload has a noncanonical UTF-8 form.")
            dtype_name = "utf8"
        else:
            dtype = _CODE_TO_NUMPY_DTYPE[dtype_code]
            wire_dtype = dtype.newbyteorder("<") if dtype.itemsize > 1 else dtype
            array = np.frombuffer(payload, dtype=wire_dtype).reshape(shape)
            value = np.array(array, dtype=dtype, order="C", copy=True)
            dtype_name = dtype.name
        return DecodedRawFrame(
            modality=outer_modality,
            value=value,
            shape=shape,
            dtype=dtype_name,
            metadata=MappingProxyType(metadata),
            payload_length=payload_length,
            checksum_sha256=parsed["digest"].hex(),
        )

    def decode_image(self, tokens: Iterable[int]) -> np.ndarray:
        result = self._decode_as(tokens, RawModality.IMAGE)
        return result.value  # type: ignore[return-value]

    def decode_audio(self, tokens: Iterable[int]) -> tuple[np.ndarray, int]:
        result = self._decode_as(tokens, RawModality.AUDIO)
        return (
            result.value,  # type: ignore[return-value]
            int(result.metadata["sample_rate"]),
        )

    def decode_video(self, tokens: Iterable[int]) -> np.ndarray:
        result = self._decode_as(tokens, RawModality.VIDEO)
        return result.value  # type: ignore[return-value]

    def decode_text(self, tokens: Iterable[int]) -> str:
        result = self._decode_as(tokens, RawModality.TEXT)
        return result.value  # type: ignore[return-value]

    def decode_ocr(self, tokens: Iterable[int]) -> str:
        result = self._decode_as(tokens, RawModality.OCR)
        return result.value  # type: ignore[return-value]

    def decode_points3d(
        self, tokens: Iterable[int]
    ) -> tuple[np.ndarray, PointQuantization]:
        result = self._decode_as(tokens, RawModality.POINTS3D)
        quantization = result.quantization
        assert quantization is not None
        return result.value, quantization  # type: ignore[return-value]

    decode_point_cloud = decode_points3d

    def stream_decoder(self) -> RawFrameStreamDecoder:
        return RawFrameStreamDecoder(self)

    def decode_chunks(self, chunks: Iterable[Iterable[int]]) -> list[DecodedRawFrame]:
        decoder = self.stream_decoder()
        frames: list[DecodedRawFrame] = []
        for chunk in chunks:
            frames.extend(decoder.feed(chunk))
        decoder.finish()
        return frames

    def _encode_text(self, modality: RawModality, text: str) -> list[int]:
        if not isinstance(text, str):
            raise RawTokenEncodeError("text payload must be a Python str.")
        try:
            payload = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise RawTokenEncodeError("text payload is not valid Unicode.") from exc
        shape = (len(payload),)
        self._validate_semantics(
            modality,
            _DTYPE_UTF8,
            shape,
            {},
            len(payload),
            decoding=False,
        )
        return self._encode_frame(modality, _DTYPE_UTF8, shape, {}, payload)

    def _encode_array(
        self,
        modality: RawModality,
        array: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> list[int]:
        dtype_code = _NUMPY_DTYPE_TO_CODE[array.dtype]
        shape = tuple(int(dimension) for dimension in array.shape)
        wire_dtype = (
            array.dtype.newbyteorder("<") if array.dtype.itemsize > 1 else array.dtype
        )
        payload = np.ascontiguousarray(array, dtype=wire_dtype).tobytes(order="C")
        self._validate_semantics(
            modality,
            dtype_code,
            shape,
            dict(metadata),
            len(payload),
            decoding=False,
        )
        return self._encode_frame(modality, dtype_code, shape, dict(metadata), payload)

    def _encode_frame(
        self,
        modality: RawModality,
        dtype_code: int,
        shape: tuple[int, ...],
        metadata: Mapping[str, Any],
        payload: bytes,
    ) -> list[int]:
        metadata_bytes = _encode_metadata(metadata)
        if len(metadata_bytes) > self.limits.max_metadata_bytes:
            raise RawTokenBoundsError("Metadata exceeds max_metadata_bytes.")
        prefix = _HEADER_PREFIX.pack(
            _MAGIC,
            PROTOCOL_VERSION,
            _MODALITY_CODE[modality],
            dtype_code,
            len(shape),
            len(payload),
            len(metadata_bytes),
        )
        dimensions = b"".join(_DIMENSION.pack(value) for value in shape)
        header_core = prefix + dimensions + metadata_bytes
        digest = hashlib.sha256(header_core + payload).digest()
        header = header_core + digest
        if len(header) > self.limits.max_header_bytes:
            raise RawTokenBoundsError("Header exceeds max_header_bytes.")
        token_ids = self.vocabulary.token_to_id
        return [
            token_ids[FRAME_START_TOKEN],
            token_ids[VERSION_TOKEN],
            token_ids[_MODALITY_TOKEN[modality]],
            *(self._bytes_to_tokens(header)),
            token_ids[HEADER_END_TOKEN],
            *(self._bytes_to_tokens(payload)),
            token_ids[FRAME_END_TOKEN],
        ]

    def _bytes_to_tokens(self, payload: bytes) -> list[int]:
        byte_ids = self.vocabulary.byte_token_ids
        return [byte_ids[value] for value in payload]

    def _tokens_to_bytes(self, tokens: Sequence[int], section: str) -> bytes:
        values = bytearray()
        for token in tokens:
            try:
                values.append(self._id_to_byte[token])
            except KeyError as exc:
                raise RawTokenDecodeError(
                    f"Non-byte token found inside the {section} byte region."
                ) from exc
        return bytes(values)

    def _parse_header(self, header: bytes) -> dict[str, Any]:
        minimum = _HEADER_PREFIX.size + _DIGEST_SIZE
        if len(header) < minimum:
            raise RawTokenDecodeError("Header is truncated.")
        try:
            (
                magic,
                version,
                modality_code,
                dtype_code,
                rank,
                payload_length,
                metadata_length,
            ) = _HEADER_PREFIX.unpack_from(header)
        except struct.error as exc:
            raise RawTokenDecodeError("Header prefix is malformed.") from exc
        if magic != _MAGIC:
            raise RawTokenDecodeError("Header magic is invalid.")
        if version != PROTOCOL_VERSION:
            raise RawTokenDecodeError("Header protocol version is unsupported.")
        try:
            modality = _CODE_MODALITY[modality_code]
        except KeyError as exc:
            raise RawTokenDecodeError("Header modality code is invalid.") from exc
        if rank > self.limits.max_rank:
            raise RawTokenBoundsError("Header rank exceeds max_rank.")
        if metadata_length > self.limits.max_metadata_bytes:
            raise RawTokenBoundsError("Metadata exceeds max_metadata_bytes.")
        expected_length = (
            _HEADER_PREFIX.size
            + rank * _DIMENSION.size
            + metadata_length
            + _DIGEST_SIZE
        )
        if len(header) != expected_length:
            raise RawTokenDecodeError(
                "Header length does not match its rank and metadata length."
            )
        offset = _HEADER_PREFIX.size
        shape_values = []
        for _ in range(rank):
            shape_values.append(_DIMENSION.unpack_from(header, offset)[0])
            offset += _DIMENSION.size
        metadata_end = offset + metadata_length
        return {
            "modality": modality,
            "dtype_code": dtype_code,
            "shape": tuple(shape_values),
            "payload_length": payload_length,
            "metadata_bytes": header[offset:metadata_end],
            "header_core": header[:metadata_end],
            "digest": header[metadata_end:],
        }

    def _validate_semantics(
        self,
        modality: RawModality,
        dtype_code: int,
        shape: tuple[int, ...],
        metadata: Mapping[str, Any],
        payload_length: int,
        *,
        decoding: bool,
    ) -> None:
        error_type = RawTokenDecodeError if decoding else RawTokenEncodeError
        if len(shape) > self.limits.max_rank:
            raise RawTokenBoundsError("Shape rank exceeds max_rank.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise error_type("Shape dimensions must be non-negative integers.")
        if any(value > self.limits.max_dimension for value in shape):
            raise RawTokenBoundsError("Shape dimension exceeds max_dimension.")
        if payload_length > self.limits.max_payload_bytes:
            raise RawTokenBoundsError("Payload exceeds max_payload_bytes.")

        keys = set(metadata)
        if modality is RawModality.IMAGE:
            if dtype_code != _DTYPE_UINT8:
                raise error_type("Image header dtype must be uint8.")
            if len(shape) not in (2, 3) or any(value == 0 for value in shape):
                raise error_type("Image shape must be nonempty rank 2 or rank 3.")
            if keys:
                raise error_type("Image metadata must be empty.")
        elif modality is RawModality.AUDIO:
            if dtype_code != _DTYPE_INT16:
                raise error_type("PCM audio header dtype must be int16.")
            if len(shape) not in (1, 2) or any(value == 0 for value in shape):
                raise error_type("Audio shape must be nonempty rank 1 or rank 2.")
            if shape[0] > self.limits.max_audio_samples:
                raise RawTokenBoundsError("Audio exceeds max_audio_samples.")
            if len(shape) == 2 and not 1 <= shape[1] <= 32:
                raise error_type("Audio channel count must be between 1 and 32.")
            if keys != {"sample_rate"}:
                raise error_type("Audio metadata must contain only sample_rate.")
            sample_rate = metadata.get("sample_rate")
            if (
                isinstance(sample_rate, bool)
                or not isinstance(sample_rate, int)
                or not 1 <= sample_rate <= 768_000
            ):
                raise error_type("Audio sample_rate metadata is invalid.")
        elif modality is RawModality.VIDEO:
            if dtype_code != _DTYPE_UINT8:
                raise error_type("Video header dtype must be uint8.")
            if len(shape) not in (3, 4) or any(value == 0 for value in shape):
                raise error_type("Video shape must be nonempty rank 3 or rank 4.")
            if shape[0] > self.limits.max_video_frames:
                raise RawTokenBoundsError("Video exceeds max_video_frames.")
            if not keys <= {"frame_rate"}:
                raise error_type("Video metadata may contain only frame_rate.")
            if "frame_rate" in metadata:
                frame_rate = metadata["frame_rate"]
                if (
                    isinstance(frame_rate, bool)
                    or not isinstance(frame_rate, (int, float))
                    or not math.isfinite(float(frame_rate))
                    or not 0.0 < float(frame_rate) <= 1000.0
                ):
                    raise error_type("Video frame_rate metadata is invalid.")
        elif modality in (RawModality.TEXT, RawModality.OCR):
            if dtype_code != _DTYPE_UTF8:
                raise error_type("Text header dtype must be UTF-8.")
            if len(shape) != 1 or shape[0] != payload_length:
                raise error_type("Text shape must equal its UTF-8 byte length.")
            if payload_length > self.limits.max_text_bytes:
                raise RawTokenBoundsError("Text exceeds max_text_bytes.")
            if keys:
                raise error_type("Text metadata must be empty.")
        elif modality is RawModality.POINTS3D:
            dtype = _CODE_TO_NUMPY_DTYPE.get(dtype_code)
            if dtype not in _POINT_DTYPES:
                raise error_type("Quantized points must use a supported integer dtype.")
            if len(shape) != 2 or shape[1:] != (3,) or shape[0] == 0:
                raise error_type("Quantized point shape must be (point_count, 3).")
            if shape[0] > self.limits.max_points:
                raise RawTokenBoundsError("Point array exceeds max_points.")
            if keys != {"scale", "zero_point"}:
                raise error_type(
                    "Point metadata must contain only scale and zero_point."
                )
            raw_scale = metadata["scale"]
            raw_zero_point = metadata["zero_point"]
            if not isinstance(raw_scale, list) or not isinstance(raw_zero_point, list):
                raise error_type(
                    "Point scale and zero_point metadata must be JSON arrays."
                )
            try:
                scale = _normalize_scale(raw_scale)
                zero_point = _normalize_zero_point(raw_zero_point)
            except RawTokenEncodeError as exc:
                raise error_type(str(exc)) from exc
            dtype_limits = np.iinfo(dtype)
            if any(
                item < dtype_limits.min or item > dtype_limits.max
                for item in zero_point
            ):
                raise error_type(
                    "Point zero_point metadata must fit the quantized dtype."
                )
            if list(scale) != raw_scale:
                raise error_type("Point scale metadata is not canonical.")
            if list(zero_point) != raw_zero_point:
                raise error_type("Point zero_point metadata is not canonical.")
        else:  # pragma: no cover - enum exhaustiveness guard
            raise error_type(f"Unsupported modality: {modality!r}.")

        if dtype_code == _DTYPE_UTF8:
            expected_length = shape[0]
            element_count = shape[0]
        else:
            dtype = _CODE_TO_NUMPY_DTYPE.get(dtype_code)
            if dtype is None:
                raise error_type("Header dtype code is invalid.")
            element_count = _bounded_product(shape, self.limits.max_elements)
            expected_length = element_count * dtype.itemsize
        if element_count > self.limits.max_elements:
            raise RawTokenBoundsError("Shape exceeds max_elements.")
        if expected_length != payload_length:
            raise error_type(
                "Shape and dtype do not match the declared payload length."
            )

    def _decode_as(
        self, tokens: Iterable[int], expected: RawModality
    ) -> DecodedRawFrame:
        result = self.decode(tokens)
        if result.modality is not expected:
            raise RawTokenDecodeError(
                f"Expected {expected.value} frame, received {result.modality.value}."
            )
        return result


class RawFrameStreamDecoder:
    """Incrementally decode complete frames across arbitrary token chunks."""

    def __init__(self, codec: RawModalityCodec | None = None) -> None:
        self.codec = codec or RawModalityCodec()
        self._buffer: list[int] = []
        self._failed = False

    @property
    def buffered_token_count(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: Iterable[int]) -> list[DecodedRawFrame]:
        if self._failed:
            raise RawTokenDecodeError(
                "Stream decoder is in a failed state; call reset before reuse."
            )
        start_id = self.codec.vocabulary.id_for(FRAME_START_TOKEN)
        end_id = self.codec.vocabulary.id_for(FRAME_END_TOKEN)
        allowed = self.codec.vocabulary.allowed_ids
        decoded: list[DecodedRawFrame] = []
        try:
            for raw_token in chunk:
                token = _integer_id(raw_token, "stream token")
                if token not in allowed:
                    raise RawTokenDecodeError(
                        f"Unknown token ID {token} in raw frame stream."
                    )
                if not self._buffer and token != start_id:
                    raise RawTokenDecodeError(
                        "Stream data must begin with a frame start token."
                    )
                self._buffer.append(token)
                if len(self._buffer) > self.codec.max_frame_tokens:
                    raise RawTokenBoundsError(
                        "Incomplete stream frame exceeds the configured token bound."
                    )
                if token == end_id:
                    decoded.append(self.codec.decode(self._buffer))
                    self._buffer.clear()
        except (RawTokenError, TypeError) as exc:
            self._buffer.clear()
            self._failed = True
            if isinstance(exc, RawTokenError):
                raise
            raise RawTokenDecodeError("Stream contains a non-integer token.") from exc
        return decoded

    def finish(self) -> None:
        if self._failed:
            raise RawTokenDecodeError("Cannot finish a failed stream decoder.")
        if self._buffer:
            self._failed = True
            self._buffer.clear()
            raise RawTokenDecodeError("Stream ended with an incomplete raw frame.")

    def reset(self) -> None:
        self._buffer.clear()
        self._failed = False


def iter_token_chunks(
    tokens: Iterable[int], chunk_size: int
) -> Iterator[tuple[int, ...]]:
    """Yield fixed-size token tuples without changing token values or order."""

    size = _integer_id(chunk_size, "chunk_size")
    if size <= 0:
        raise ValueError("chunk_size must be positive.")
    chunk: list[int] = []
    for raw_token in tokens:
        chunk.append(_integer_id(raw_token, "token"))
        if len(chunk) == size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


def decode_token_chunks(
    chunks: Iterable[Iterable[int]], codec: RawModalityCodec | None = None
) -> list[DecodedRawFrame]:
    """Decode all complete frames from a finite iterable of token chunks."""

    return (codec or RawModalityCodec()).decode_chunks(chunks)


def plan_hf_raw_token_expansion(
    tokenizer: Any, model: Any | None = None
) -> dict[str, Any]:
    """Build a JSON-safe, non-mutating HF vocabulary expansion plan."""

    vocabulary = _tokenizer_vocabulary(tokenizer)
    tokenizer_size = _tokenizer_size(tokenizer)
    existing = [token for token in RAW_TOKEN_STRINGS if token in vocabulary]
    missing = [token for token in RAW_TOKEN_STRINGS if token not in vocabulary]
    proposed_ids: dict[str, int] = {token: vocabulary[token] for token in existing}
    proposed_ids.update(
        {token: tokenizer_size + index for index, token in enumerate(missing)}
    )
    model_plan = _inspect_hf_model(model) if model is not None else None
    return {
        "operation": "plan_hf_raw_token_expansion",
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "status": "planned",
        "mutated": False,
        "tokenizer_size_before": tokenizer_size,
        "tokenizer_size_after_expected": tokenizer_size + len(missing),
        "required_token_count": len(RAW_TOKEN_STRINGS),
        "control_token_count": len(CONTROL_TOKEN_STRINGS),
        "shared_byte_token_count": len(BYTE_TOKEN_STRINGS),
        "existing_token_count": len(existing),
        "missing_token_count": len(missing),
        "existing_tokens": existing,
        "tokens_to_add": missing,
        "proposed_token_ids": proposed_ids,
        "model": model_plan,
        "initialization": {
            "method": (
                "deterministic-existing-row-sample-column-mean-std-with-sha256-pattern"
            ),
            "deterministic": True,
            "random_seed": None,
            "pattern_scale": 0.05,
            "maximum_statistic_rows": _MAX_STATISTIC_ROWS,
            "input_and_output_statistics_are_independent": True,
            "existing_rows_are_preserved": True,
        },
        "architecture": {
            "learned_modality_modules": [],
            "encoder": False,
            "decoder": False,
            "projector": False,
            "adapter": False,
            "bridge": False,
            "only_token_rows_are_added": True,
            "requires_untied_output_head": model is not None,
        },
    }


def expand_hf_raw_token_vocabulary(
    tokenizer: Any, model: Any | None = None
) -> dict[str, Any]:
    """Add raw tokens and deterministically initialize new HF vocabulary rows.

    When a model is supplied, it must expose Hugging Face-style input and
    output embeddings and its output head must already be untied.  The helper
    adds no modules; it only calls ``resize_token_embeddings`` and overwrites
    newly assigned raw-token rows using statistics from pre-existing rows.
    """

    plan = plan_hf_raw_token_expansion(tokenizer, model)
    model_before = _inspect_hf_model(model) if model is not None else None
    if model_before is not None:
        if model_before["input_embedding_rows"] <= 0:
            raise RawVocabularyError("Model input embedding has no source rows.")
        if model_before["output_head_rows"] <= 0:
            raise RawVocabularyError("Model output head has no source rows.")
        if model_before["weights_tied"]:
            raise RawVocabularyError(
                "Raw-token expansion requires an already untied output head."
            )

    old_vocabulary = _tokenizer_vocabulary(tokenizer)
    missing = list(plan["tokens_to_add"])
    added_count = _add_special_tokens(tokenizer, missing)
    if added_count != len(missing):
        raise RawVocabularyError(
            "Tokenizer did not add the expected number of raw tokens."
        )
    new_vocabulary = _tokenizer_vocabulary(tokenizer)
    raw_ids = _validate_tokenizer_raw_tokens(tokenizer, new_vocabulary)

    initialization_report: dict[str, Any] = {
        "method": plan["initialization"]["method"],
        "pattern_scale": plan["initialization"]["pattern_scale"],
        "input_token_ids": [],
        "output_token_ids": [],
        "output_bias_token_ids": [],
    }
    if model is not None:
        input_before = model.get_input_embeddings().weight
        output_before = model.get_output_embeddings().weight
        input_rows_before = int(input_before.shape[0])
        output_rows_before = int(output_before.shape[0])
        input_stats = _tensor_row_statistics(input_before, input_rows_before)
        output_stats = _tensor_row_statistics(output_before, output_rows_before)
        output_bias = getattr(model.get_output_embeddings(), "bias", None)
        bias_stats = (
            _tensor_scalar_statistics(output_bias, output_rows_before)
            if output_bias is not None
            else None
        )

        target_rows = max(
            _tokenizer_size(tokenizer),
            max(raw_ids.values()) + 1,
            input_rows_before,
            output_rows_before,
        )
        if input_rows_before < target_rows or output_rows_before < target_rows:
            resize = getattr(model, "resize_token_embeddings", None)
            if not callable(resize):
                raise RawVocabularyError(
                    "Model must provide resize_token_embeddings for expansion."
                )
            try:
                resize(target_rows, mean_resizing=False)
            except TypeError:
                resize(target_rows)
        input_after = model.get_input_embeddings().weight
        output_module_after = model.get_output_embeddings()
        output_after = output_module_after.weight
        if int(input_after.shape[0]) < target_rows:
            raise RawVocabularyError(
                "Input embedding resize did not reach target size."
            )
        if int(output_after.shape[0]) < target_rows:
            raise RawVocabularyError("Output head resize did not reach target size.")
        if _weights_tied(input_after, output_after):
            raise RawVocabularyError("Output head became tied during model resize.")

        added = set(missing)
        input_tokens = [
            token
            for token in RAW_TOKEN_STRINGS
            if token in added or raw_ids[token] >= input_rows_before
        ]
        output_tokens = [
            token
            for token in RAW_TOKEN_STRINGS
            if token in added or raw_ids[token] >= output_rows_before
        ]
        _initialize_tensor_rows(input_after, input_tokens, raw_ids, input_stats)
        _initialize_tensor_rows(output_after, output_tokens, raw_ids, output_stats)
        bias_after = getattr(output_module_after, "bias", None)
        if bias_after is not None:
            if int(bias_after.shape[0]) < target_rows:
                raise RawVocabularyError(
                    "Output head bias resize did not reach target size."
                )
            assert bias_stats is not None
            _initialize_bias_rows(bias_after, output_tokens, raw_ids, bias_stats)
        initialization_report.update(
            {
                "input_source_row_count": input_rows_before,
                "output_source_row_count": output_rows_before,
                "input_statistic_row_count": input_stats["sample_row_count"],
                "output_statistic_row_count": output_stats["sample_row_count"],
                "input_token_ids": [raw_ids[token] for token in input_tokens],
                "output_token_ids": [raw_ids[token] for token in output_tokens],
                "output_bias_token_ids": (
                    [raw_ids[token] for token in output_tokens]
                    if bias_after is not None
                    else []
                ),
            }
        )
        model_after = _inspect_hf_model(model)
    else:
        model_after = None

    return {
        "operation": "expand_hf_raw_token_vocabulary",
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "status": "applied",
        "tokenizer_size_before": plan["tokenizer_size_before"],
        "tokenizer_size_after": _tokenizer_size(tokenizer),
        "added_token_count": added_count,
        "preexisting_raw_token_count": len(RAW_TOKEN_STRINGS) - added_count,
        "raw_token_ids": raw_ids,
        "all_tokens_atomic": True,
        "shared_byte_token_count": len(BYTE_TOKEN_STRINGS),
        "model_before": model_before,
        "model_after": model_after,
        "initialization": initialization_report,
        "architecture": plan["architecture"],
        "old_vocabulary_size": len(old_vocabulary),
    }


plan_hf_vocabulary_expansion = plan_hf_raw_token_expansion
apply_hf_raw_token_expansion = expand_hf_raw_token_vocabulary


def codec_from_hf_tokenizer(
    tokenizer: Any, *, limits: CodecLimits | None = None
) -> RawModalityCodec:
    """Construct a codec from raw tokens already installed in a tokenizer."""

    vocabulary = RawTokenVocabulary.from_tokenizer(tokenizer)
    _validate_tokenizer_raw_tokens(tokenizer, dict(vocabulary.token_to_id))
    return RawModalityCodec(vocabulary=vocabulary, limits=limits)


def _integer_id(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RawTokenEncodeError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise RawTokenEncodeError(f"{name} must be a finite number.")
    return result


def _require_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RawTokenEncodeError(f"{name} must be a NumPy ndarray.")
    return value


def _normalize_scale(value: float | Sequence[float]) -> tuple[float, float, float]:
    values: Sequence[Any]
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        values = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise RawTokenEncodeError("scale must be one number or three numbers.")
    if len(values) != 3:
        raise RawTokenEncodeError("scale must contain exactly three values.")
    normalized = tuple(_finite_number(item, "scale") for item in values)
    if any(item <= 0.0 for item in normalized):
        raise RawTokenEncodeError("scale values must be greater than zero.")
    return normalized  # type: ignore[return-value]


def _normalize_zero_point(
    value: int | Sequence[int],
) -> tuple[int, int, int]:
    values: Sequence[Any]
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        values = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise RawTokenEncodeError("zero_point must be one integer or three integers.")
    if len(values) != 3:
        raise RawTokenEncodeError("zero_point must contain exactly three values.")
    try:
        normalized = tuple(_integer_id(item, "zero_point") for item in values)
    except TypeError as exc:
        raise RawTokenEncodeError(str(exc)) from exc
    if any(not -(2**31) <= item < 2**31 for item in normalized):
        raise RawTokenEncodeError("zero_point values must fit signed int32.")
    return normalized  # type: ignore[return-value]


def _bounded_product(shape: Sequence[int], limit: int) -> int:
    product = 1
    for dimension in shape:
        if dimension == 0:
            return 0
        if product > limit // dimension:
            raise RawTokenBoundsError("Shape exceeds max_elements.")
        product *= dimension
    return product


def _bounded_token_list(tokens: Iterable[int], limit: int) -> list[int]:
    result = []
    try:
        iterator = iter(tokens)
    except TypeError as exc:
        raise RawTokenDecodeError("tokens must be an iterable of integers.") from exc
    for raw_token in iterator:
        try:
            result.append(_integer_id(raw_token, "token"))
        except TypeError as exc:
            raise RawTokenDecodeError(str(exc)) from exc
        if len(result) > limit:
            raise RawTokenBoundsError("Raw frame exceeds the configured token bound.")
    return result


def _encode_metadata(metadata: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            dict(metadata),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RawTokenEncodeError("Metadata is not canonical JSON data.") from exc
    return text.encode("ascii")


def _decode_metadata(payload: bytes) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RawTokenDecodeError(f"Duplicate metadata key: {key!r}.")
            result[key] = value
        return result

    def reject_constant(constant: str) -> Any:
        raise RawTokenDecodeError(
            f"Non-finite metadata constant is invalid: {constant}."
        )

    try:
        text = payload.decode("ascii", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except RawTokenDecodeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawTokenDecodeError("Metadata is not valid canonical JSON.") from exc
    if not isinstance(value, dict):
        raise RawTokenDecodeError("Metadata root must be a JSON object.")
    try:
        canonical = _encode_metadata(value)
    except RawTokenEncodeError as exc:
        raise RawTokenDecodeError("Metadata is not canonical JSON data.") from exc
    if canonical != payload:
        raise RawTokenDecodeError("Metadata JSON is not in canonical form.")
    return value


def _tokenizer_size(tokenizer: Any) -> int:
    try:
        size = _integer_id(len(tokenizer), "tokenizer length")
    except (TypeError, AttributeError) as exc:
        raise RawVocabularyError("Tokenizer must provide a valid __len__.") from exc
    if size < 0:
        raise RawVocabularyError("Tokenizer length must be non-negative.")
    return size


def _tokenizer_vocabulary(tokenizer: Any) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise RawVocabularyError("Tokenizer must provide get_vocab().")
    source = get_vocab()
    if not isinstance(source, Mapping):
        raise RawVocabularyError("Tokenizer get_vocab() must return a mapping.")
    result: dict[str, int] = {}
    for token, token_id in source.items():
        if not isinstance(token, str):
            raise RawVocabularyError("Tokenizer vocabulary keys must be strings.")
        try:
            result[token] = _integer_id(token_id, f"token ID for {token!r}")
        except TypeError as exc:
            raise RawVocabularyError(str(exc)) from exc
        if result[token] < 0:
            raise RawVocabularyError("Tokenizer token IDs must be non-negative.")
    return result


def _add_special_tokens(tokenizer: Any, tokens: Sequence[str]) -> int:
    if not tokens:
        return 0
    add_tokens = getattr(tokenizer, "add_tokens", None)
    if callable(add_tokens):
        try:
            return _integer_id(
                add_tokens(list(tokens), special_tokens=True), "added token count"
            )
        except TypeError:
            pass
    add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
    if not callable(add_special_tokens):
        raise RawVocabularyError(
            "Tokenizer must provide add_tokens(..., special_tokens=True)."
        )
    try:
        return _integer_id(
            add_special_tokens({"additional_special_tokens": list(tokens)}),
            "added token count",
        )
    except TypeError as exc:
        raise RawVocabularyError("Tokenizer could not add raw special tokens.") from exc


def _validate_tokenizer_raw_tokens(
    tokenizer: Any, vocabulary: Mapping[str, int]
) -> dict[str, int]:
    missing = [token for token in RAW_TOKEN_STRINGS if token not in vocabulary]
    if missing:
        raise RawVocabularyError(
            f"Tokenizer is missing {len(missing)} raw protocol tokens after expansion."
        )
    raw_ids = {token: int(vocabulary[token]) for token in RAW_TOKEN_STRINGS}
    if len(set(raw_ids.values())) != len(raw_ids):
        raise RawVocabularyError("Raw protocol tokens do not have unique IDs.")
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(convert) and not callable(encode):
        raise RawVocabularyError(
            "Tokenizer must expose token conversion or encoding for atomicity checks."
        )
    if callable(convert):
        for token, expected in raw_ids.items():
            observed = convert(token)
            if isinstance(observed, Sequence) and not isinstance(observed, str):
                if len(observed) != 1:
                    raise RawVocabularyError(f"Raw token {token!r} is not atomic.")
                observed = observed[0]
            try:
                observed_id = _integer_id(observed, f"ID for {token!r}")
            except TypeError as exc:
                raise RawVocabularyError(str(exc)) from exc
            if observed_id != expected:
                raise RawVocabularyError(f"Raw token {token!r} is not atomic.")
    if callable(encode):
        for token, expected in raw_ids.items():
            try:
                encoded = encode(token, add_special_tokens=False)
            except TypeError as exc:
                raise RawVocabularyError(
                    "Tokenizer encode() must support add_special_tokens=False."
                ) from exc
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
                encoded = list(encoded)
            if encoded != [expected]:
                raise RawVocabularyError(f"Raw token {token!r} is not atomic.")
    return raw_ids


def _inspect_hf_model(model: Any) -> dict[str, Any]:
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise RawVocabularyError(
            "Model must provide input and output embedding accessors."
        )
    input_module = get_input()
    output_module = get_output()
    if input_module is None or output_module is None:
        raise RawVocabularyError("Model input embedding and output head are required.")
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if input_weight is None or output_weight is None:
        raise RawVocabularyError("Model embedding and output head need weight tensors.")
    if getattr(input_weight, "ndim", None) != 2:
        raise RawVocabularyError("Input embedding weight must be rank 2.")
    if getattr(output_weight, "ndim", None) != 2:
        raise RawVocabularyError("Output head weight must be rank 2.")
    if int(input_weight.shape[0]) <= 0 or int(input_weight.shape[1]) <= 0:
        raise RawVocabularyError("Input embedding dimensions must be positive.")
    if int(output_weight.shape[0]) <= 0 or int(output_weight.shape[1]) <= 0:
        raise RawVocabularyError("Output head dimensions must be positive.")
    return {
        "input_embedding_rows": int(input_weight.shape[0]),
        "input_embedding_width": int(input_weight.shape[1]),
        "output_head_rows": int(output_weight.shape[0]),
        "output_head_width": int(output_weight.shape[1]),
        "weights_tied": _weights_tied(input_weight, output_weight),
        "output_head_has_bias": getattr(output_module, "bias", None) is not None,
    }


def _weights_tied(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_pointer = getattr(left, "data_ptr", None)
    right_pointer = getattr(right, "data_ptr", None)
    if callable(left_pointer) and callable(right_pointer):
        return int(left_pointer()) == int(right_pointer())
    return False


def _tensor_row_statistics(weight: Any, row_count: int) -> dict[str, Any]:
    torch = _import_torch()
    sample_row_count = min(row_count, _MAX_STATISTIC_ROWS)
    with torch.no_grad():
        if sample_row_count == row_count:
            source = weight[:row_count]
        else:
            positions = np.linspace(
                0, row_count - 1, num=sample_row_count, dtype=np.int64
            )
            indices = torch.as_tensor(positions, device=weight.device)
            source = weight.index_select(0, indices)
        source = source.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(source).all()):
            raise RawVocabularyError(
                "Existing vocabulary rows must contain only finite values."
            )
        column_mean = source.mean(dim=0).cpu()
        column_std = source.std(dim=0, unbiased=False).cpu()
        global_std = float(source.std(unbiased=False).cpu())
    if global_std > 0.0:
        floor = global_std * 1e-6
        column_std = torch.where(
            column_std > floor,
            column_std,
            torch.full_like(column_std, global_std),
        )
    return {
        "column_mean": column_mean,
        "column_std": column_std,
        "width": int(weight.shape[1]),
        "sample_row_count": sample_row_count,
    }


def _tensor_scalar_statistics(values: Any, row_count: int) -> tuple[float, float]:
    torch = _import_torch()
    with torch.no_grad():
        source = values[:row_count].detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(source).all()):
            raise RawVocabularyError(
                "Existing output bias rows must contain only finite values."
            )
        return (
            float(source.mean().cpu()),
            float(source.std(unbiased=False).cpu()),
        )


def _deterministic_pattern(token: str, width: int) -> np.ndarray:
    raw = hashlib.shake_256(token.encode("utf-8")).digest(width * 2)
    values = np.frombuffer(raw, dtype=">u2").astype(np.float32)
    values = values / np.float32(32767.5) - np.float32(1.0)
    values -= values.mean()
    root_mean_square = float(np.sqrt(np.mean(values * values)))
    if root_mean_square > 0.0:
        values /= np.float32(root_mean_square)
    return values


def _initialize_tensor_rows(
    weight: Any,
    tokens: Sequence[str],
    token_ids: Mapping[str, int],
    statistics: Mapping[str, Any],
) -> None:
    torch = _import_torch()
    mean = statistics["column_mean"]
    std = statistics["column_std"]
    width = int(statistics["width"])
    with torch.no_grad():
        for token in tokens:
            pattern = torch.from_numpy(_deterministic_pattern(token, width))
            row = mean + std * pattern * 0.05
            weight[token_ids[token]].copy_(
                row.to(device=weight.device, dtype=weight.dtype)
            )


def _initialize_bias_rows(
    bias: Any,
    tokens: Sequence[str],
    token_ids: Mapping[str, int],
    statistics: tuple[float, float],
) -> None:
    torch = _import_torch()
    mean, std = statistics
    with torch.no_grad():
        for token in tokens:
            pattern = float(_deterministic_pattern(token, 2)[0])
            bias[token_ids[token]].copy_(bias.new_tensor(mean + std * pattern * 0.05))


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RawVocabularyError(
            "PyTorch is required only when expanding model vocabulary rows."
        ) from exc
    return torch


__all__ = [
    "AUDIO_TOKEN",
    "BYTE_TOKEN_STRINGS",
    "CANONICAL_RAW_TRANSPORT",
    "CONTROL_TOKEN_STRINGS",
    "CodecLimits",
    "LANE_COMPATIBILITY_SCHEMA",
    "DecodedRawFrame",
    "FRAME_END_TOKEN",
    "FRAME_START_TOKEN",
    "HEADER_END_TOKEN",
    "IMAGE_TOKEN",
    "OCR_TOKEN",
    "POINTS3D_TOKEN",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "PointQuantization",
    "RAW_TOKEN_STRINGS",
    "RawFrameStreamDecoder",
    "RawModality",
    "RawModalityCodec",
    "RawTokenBoundsError",
    "RawTokenDecodeError",
    "RawTokenEncodeError",
    "RawTokenError",
    "RawTokenVocabulary",
    "RawVocabularyError",
    "TEXT_TOKEN",
    "VERSION_TOKEN",
    "VIDEO_TOKEN",
    "apply_hf_raw_token_expansion",
    "codec_from_hf_tokenizer",
    "decode_token_chunks",
    "expand_hf_raw_token_vocabulary",
    "iter_token_chunks",
    "plan_hf_raw_token_expansion",
    "plan_hf_vocabulary_expansion",
    "raw_token_lane_compatibility",
    "raw_token_vocabulary",
]
