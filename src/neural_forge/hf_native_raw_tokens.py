"""Parameter-free raw tensor bytes over existing Hugging Face token rows.

This module discovers the 256 byte atoms already present in a fast ByteLevel
tokenizer and uses those ordinary token IDs directly. It never adds vocabulary
rows or creates an encoder, decoder, adapter, projector, bridge, learned
module, or runtime donor dependency.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import operator
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


PROTOCOL_NAME = "neural-forge-hf-native-raw-bytes"
PROTOCOL_VERSION = 1
SERIALIZATION_PATH = "generic-ndarray-c-order-v1"

_MAGIC = b"NFHFRAW1"
_LENGTH_WIDTH = 8
_FRAME_PREFIX_SIZE = len(_MAGIC) + 1 + _LENGTH_WIDTH + 1
_HEADER_FIELDS = frozenset(
    {
        "byte_mapping_sha256",
        "content_sha256",
        "dtype",
        "metadata",
        "modality",
        "order",
        "payload_bytes",
        "payload_sha256",
        "rank",
        "schema",
        "shape",
        "version",
    }
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODALITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SUPPORTED_DTYPE_KINDS = frozenset("biufc")
_MAX_TOKEN_ID = (1 << 63) - 1
_MAX_TRUSTED_AUTO_CONTEXT = 10_000_000


class HFNativeRawTokenError(ValueError):
    """Base error for strict native raw-token operations."""


class ByteLevelMappingError(HFNativeRawTokenError):
    """Raised when a tokenizer cannot prove an exact 256-byte mapping."""


class RawTensorEncodeError(HFNativeRawTokenError):
    """Raised when a tensor cannot be serialized without conversion."""


class RawTensorDecodeError(HFNativeRawTokenError):
    """Raised when a raw-token stream is malformed, corrupt, or noncanonical."""


class RawTensorBoundsError(HFNativeRawTokenError):
    """Raised when configured tensor or frame bounds are exceeded."""


class RawTensorContextError(HFNativeRawTokenError):
    """Raised when a model-input context limit is absent or invalid."""


@dataclass(frozen=True)
class RawByteCodecLimits:
    """Hard bounds applied symmetrically during serialization and parsing."""

    max_payload_bytes: int = 64 * 1024 * 1024
    max_header_bytes: int = 64 * 1024
    max_rank: int = 16
    max_dimension: int = (1 << 31) - 1
    max_elements: int = 64 * 1024 * 1024
    max_metadata_depth: int = 16

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _positive_int(value, name, error_type=ValueError)
        if self.max_header_bytes >= 1 << (4 * _LENGTH_WIDTH):
            raise ValueError("max_header_bytes does not fit the frame length field")


@dataclass(frozen=True)
class ByteLevelByteMapping:
    """Validated one-token mapping from byte values to existing tokenizer IDs."""

    token_ids: tuple[int, ...]
    token_strings: tuple[str, ...]
    tokenizer_length: int
    tokenizer_class: str

    def __post_init__(self) -> None:
        if len(self.token_ids) != 256 or len(self.token_strings) != 256:
            raise ByteLevelMappingError("the ByteLevel mapping must contain 256 atoms")
        ids = tuple(_token_id(value, "byte token ID") for value in self.token_ids)
        if len(set(ids)) != 256:
            raise ByteLevelMappingError("all 256 byte atoms must have unique token IDs")
        if len(set(self.token_strings)) != 256:
            raise ByteLevelMappingError("all 256 ByteLevel atom strings must be unique")
        length = _positive_int(
            self.tokenizer_length,
            "tokenizer_length",
            error_type=ByteLevelMappingError,
        )
        if max(ids) >= length:
            raise ByteLevelMappingError(
                "a byte token ID is outside the existing tokenizer vocabulary"
            )
        if not isinstance(self.tokenizer_class, str) or not self.tokenizer_class:
            raise ByteLevelMappingError("tokenizer_class must be a non-empty string")
        object.__setattr__(self, "token_ids", ids)
        object.__setattr__(self, "tokenizer_length", length)

    @property
    def byte_to_token_id(self) -> tuple[int, ...]:
        return self.token_ids

    @property
    def token_id_to_byte(self) -> Mapping[int, int]:
        return MappingProxyType(
            {token_id: byte for byte, token_id in enumerate(self.token_ids)}
        )

    @property
    def mapping_sha256(self) -> str:
        return _token_ids_sha256(self.token_ids, domain=b"byte-map-v1")

    def token_id_for_byte(self, value: int) -> int:
        byte = _nonnegative_int(
            value,
            "byte value",
            error_type=ByteLevelMappingError,
        )
        if byte > 255:
            raise ByteLevelMappingError("byte value must be between 0 and 255")
        return self.token_ids[byte]

    def byte_for_token_id(self, value: int) -> int:
        token_id = _token_id(value, "token ID")
        try:
            return self.token_id_to_byte[token_id]
        except KeyError as exc:
            raise ByteLevelMappingError(
                f"token ID {token_id} is not one of the 256 ByteLevel atoms"
            ) from exc

    def encode_bytes(self, payload: bytes | bytearray | memoryview) -> tuple[int, ...]:
        try:
            raw = bytes(payload)
        except (TypeError, ValueError) as exc:
            raise TypeError("payload must be bytes-like") from exc
        return tuple(self.token_ids[value] for value in raw)

    def decode_token_ids(self, token_ids: Iterable[int]) -> bytes:
        inverse = self.token_id_to_byte
        result = bytearray()
        for position, raw_token_id in enumerate(token_ids):
            token_id = _token_id(raw_token_id, f"token ID at position {position}")
            try:
                result.append(inverse[token_id])
            except KeyError as exc:
                raise ByteLevelMappingError(
                    f"token ID {token_id} at position {position} is not a byte atom"
                ) from exc
        return bytes(result)

    def report(self) -> dict[str, Any]:
        return {
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_length": self.tokenizer_length,
            "byte_token_count": 256,
            "unique_byte_token_count": len(set(self.token_ids)),
            "all_bytes_exactly_one_existing_token": True,
            "all_byte_ids_are_ordinary_model_vocabulary_ids": True,
            "byte_token_ids": list(self.token_ids),
            "byte_mapping_sha256": self.mapping_sha256,
            "tokenizer_mutated": False,
        }


@dataclass(frozen=True)
class RawTokenChunk:
    """One context-bounded segment and its matching attention mask."""

    index: int
    start_token: int
    input_ids: tuple[int, ...]
    context_limit: int
    stream_token_count: int
    stream_token_ids_sha256: str

    def __post_init__(self) -> None:
        index = _nonnegative_int(self.index, "chunk index", error_type=ValueError)
        start = _nonnegative_int(
            self.start_token,
            "chunk start_token",
            error_type=ValueError,
        )
        context = _positive_int(
            self.context_limit,
            "chunk context_limit",
            error_type=ValueError,
        )
        count = _positive_int(
            self.stream_token_count,
            "stream_token_count",
            error_type=ValueError,
        )
        ids = tuple(
            _token_id(value, f"chunk token ID at position {position}")
            for position, value in enumerate(self.input_ids)
        )
        if not ids:
            raise ValueError("raw-token chunks must not be empty")
        if len(ids) > context:
            raise ValueError("raw-token chunk exceeds its context_limit")
        if start + len(ids) > count:
            raise ValueError("raw-token chunk extends beyond stream_token_count")
        if not _is_sha256(self.stream_token_ids_sha256):
            raise ValueError("stream_token_ids_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "start_token", start)
        object.__setattr__(self, "context_limit", context)
        object.__setattr__(self, "stream_token_count", count)
        object.__setattr__(self, "input_ids", ids)

    @property
    def attention_mask(self) -> tuple[int, ...]:
        return (1,) * len(self.input_ids)

    @property
    def stop_token(self) -> int:
        return self.start_token + len(self.input_ids)

    @property
    def token_ids_sha256(self) -> str:
        return _token_ids_sha256(self.input_ids, domain=b"chunk-token-ids-v1")

    def to_hf_batch(self, return_tensors: str | None = None) -> dict[str, Any]:
        """Return a batch accepted by ordinary Hugging Face LM input paths."""

        ids = [list(self.input_ids)]
        mask = [list(self.attention_mask)]
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": mask}
        if return_tensors == "np":
            return {
                "input_ids": np.asarray(ids, dtype=np.int64),
                "attention_mask": np.asarray(mask, dtype=np.int64),
            }
        if return_tensors == "pt":
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RawTensorContextError(
                    "PyTorch is required for return_tensors='pt'"
                ) from exc
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        raise ValueError("return_tensors must be None, 'np', or 'pt'")

    def report(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_token": self.start_token,
            "stop_token": self.stop_token,
            "token_count": len(self.input_ids),
            "context_limit": self.context_limit,
            "attention_mask_sum": sum(self.attention_mask),
            "attention_mask_is_binary": True,
            "token_ids_sha256": self.token_ids_sha256,
            "stream_token_ids_sha256": self.stream_token_ids_sha256,
        }


@dataclass(frozen=True)
class EncodedRawTensor:
    """Canonical byte-token stream plus context-safe model-input chunks."""

    modality: str
    shape: tuple[int, ...]
    dtype: str
    metadata: Mapping[str, Any]
    input_ids: tuple[int, ...]
    chunks: tuple[RawTokenChunk, ...]
    payload_token_offset: int
    payload_bytes: int
    payload_sha256: str
    content_sha256: str
    header_bytes: int
    header_sha256: str
    frame_sha256: str
    byte_mapping_sha256: str
    context_limit: int

    @property
    def attention_mask(self) -> tuple[int, ...]:
        return (1,) * len(self.input_ids)

    @property
    def token_ids_sha256(self) -> str:
        return _token_ids_sha256(self.input_ids)

    @property
    def payload_input_ids(self) -> tuple[int, ...]:
        return self.input_ids[self.payload_token_offset :]

    def to_hf_batches(self, return_tensors: str | None = None) -> list[dict[str, Any]]:
        return [chunk.to_hf_batch(return_tensors) for chunk in self.chunks]

    def report(self) -> dict[str, Any]:
        chunk_counts = [len(chunk.input_ids) for chunk in self.chunks]
        return {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "serialization_path": SERIALIZATION_PATH,
            "modality": self.modality,
            "rank": len(self.shape),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "metadata": _thaw_json(self.metadata),
            "order": "C",
            "payload_bytes": self.payload_bytes,
            "payload_token_offset": self.payload_token_offset,
            "header_bytes": self.header_bytes,
            "frame_bytes": len(self.input_ids),
            "token_count": len(self.input_ids),
            "bytes_per_token": 1,
            "payload_sha256": self.payload_sha256,
            "content_sha256": self.content_sha256,
            "header_sha256": self.header_sha256,
            "frame_sha256": self.frame_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "attention_mask_sha256": hashlib.sha256(
                bytes(self.attention_mask)
            ).hexdigest(),
            "byte_mapping_sha256": self.byte_mapping_sha256,
            "context_limit": self.context_limit,
            "chunk_count": len(self.chunks),
            "chunk_token_counts": chunk_counts,
            "max_chunk_tokens": max(chunk_counts),
            "all_chunks_within_context": all(
                count <= self.context_limit for count in chunk_counts
            ),
            "attention_masks_are_binary": True,
            "tokens_reassembled_exactly": sum(chunk_counts) == len(self.input_ids),
            "bytes_dropped": 0,
            "ordinary_existing_embedding_token_ids": True,
            "parameter_count": 0,
            "learned_components": [],
            "modality_specific_components": [],
            "runtime_donor_required": False,
        }


@dataclass(frozen=True)
class DecodedRawTensor:
    """An exactly reconstructed ndarray and its verified provenance."""

    modality: str
    array: np.ndarray
    metadata: Mapping[str, Any]
    payload_sha256: str
    content_sha256: str
    header_sha256: str
    frame_sha256: str
    token_ids_sha256: str
    byte_mapping_sha256: str
    payload_token_offset: int
    context_limit: int | None = None
    chunk_token_counts: tuple[int, ...] = ()

    @property
    def value(self) -> np.ndarray:
        return self.array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.array.shape)

    @property
    def dtype(self) -> str:
        return self.array.dtype.str

    def report(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "serialization_path": SERIALIZATION_PATH,
            "modality": self.modality,
            "rank": self.array.ndim,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "metadata": _thaw_json(self.metadata),
            "order": "C",
            "payload_bytes": self.array.nbytes,
            "payload_sha256": self.payload_sha256,
            "content_sha256": self.content_sha256,
            "header_sha256": self.header_sha256,
            "frame_sha256": self.frame_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "byte_mapping_sha256": self.byte_mapping_sha256,
            "payload_token_offset": self.payload_token_offset,
            "context_limit": self.context_limit,
            "chunk_count": len(self.chunk_token_counts),
            "chunk_token_counts": list(self.chunk_token_counts),
            "checksum_validation": "passed",
            "canonical_header_validation": "passed",
            "exact_payload_roundtrip": True,
            "bytes_dropped": 0,
            "learned_components": [],
        }


def discover_byte_token_mapping(tokenizer: Any) -> ByteLevelByteMapping:
    """Fail closed unless ``tokenizer`` proves all 256 ByteLevel atom rows."""

    if tokenizer is None:
        raise ByteLevelMappingError("tokenizer is required")
    if getattr(tokenizer, "is_fast", True) is not True:
        raise ByteLevelMappingError("an exact mapping requires a fast HF tokenizer")

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ByteLevelMappingError("tokenizer must provide get_vocab()")
    before_vocab = _validated_vocabulary(get_vocab())
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ByteLevelMappingError(
            "tokenizer must expose a structured fast-tokenizer backend"
        )
    try:
        backend_data = json.loads(backend.to_str())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ByteLevelMappingError("tokenizer backend JSON is malformed") from exc
    if not isinstance(backend_data, dict):
        raise ByteLevelMappingError("tokenizer backend JSON root must be an object")
    if not _contains_component(backend_data.get("pre_tokenizer"), "ByteLevel"):
        raise ByteLevelMappingError(
            "tokenizer pre-tokenizer must contain a ByteLevel component"
        )
    if not _contains_component(backend_data.get("decoder"), "ByteLevel"):
        raise ByteLevelMappingError(
            "tokenizer decoder must contain a ByteLevel component"
        )

    token_strings = _bytelevel_token_strings()
    model_vocab = _backend_model_vocabulary(backend_data)
    missing = [token for token in token_strings if token not in model_vocab]
    if missing:
        raise ByteLevelMappingError(
            "ByteLevel model vocabulary is missing "
            f"{len(missing)} byte atoms; first missing atom is {missing[0]!r}"
        )
    missing_public = [token for token in token_strings if token not in before_vocab]
    if missing_public:
        raise ByteLevelMappingError(
            "HF tokenizer vocabulary is missing "
            f"{len(missing_public)} ByteLevel atoms present in its backend model"
        )

    token_to_id = getattr(backend, "token_to_id", None)
    id_to_token = getattr(backend, "id_to_token", None)
    convert_to_id = getattr(tokenizer, "convert_tokens_to_ids", None)
    convert_to_token = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not all(
        callable(item)
        for item in (token_to_id, id_to_token, convert_to_id, convert_to_token)
    ):
        raise ByteLevelMappingError(
            "tokenizer must expose bidirectional token/ID conversion"
        )

    special_ids = {
        _token_id(value, "special token ID")
        for value in getattr(tokenizer, "all_special_ids", ())
    }
    token_ids: list[int] = []
    for byte, token in enumerate(token_strings):
        model_id = _token_id(model_vocab[token], f"model ID for byte {byte}")
        observed = {
            "model": model_id,
            "vocabulary": _token_id(
                before_vocab[token], f"vocabulary ID for byte {byte}"
            ),
            "backend": _token_id(token_to_id(token), f"backend ID for byte {byte}"),
            "hf": _token_id(convert_to_id(token), f"HF ID for byte {byte}"),
        }
        if len(set(observed.values())) != 1:
            raise ByteLevelMappingError(
                f"inconsistent token IDs for byte {byte}: {observed}"
            )
        if id_to_token(model_id) != token or convert_to_token(model_id) != token:
            raise ByteLevelMappingError(
                f"token ID {model_id} does not invert exactly for byte {byte}"
            )
        if model_id in special_ids:
            raise ByteLevelMappingError(
                f"byte {byte} resolves to special token ID {model_id}, not an ordinary row"
            )
        token_ids.append(model_id)

    after_vocab = _validated_vocabulary(get_vocab())
    if after_vocab != before_vocab:
        raise ByteLevelMappingError(
            "tokenizer vocabulary changed during mapping discovery"
        )
    try:
        tokenizer_length = operator.index(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ByteLevelMappingError("tokenizer must provide a valid length") from exc
    mapping = ByteLevelByteMapping(
        token_ids=tuple(token_ids),
        token_strings=token_strings,
        tokenizer_length=tokenizer_length,
        tokenizer_class=_type_name(tokenizer),
    )
    if mapping.decode_token_ids(mapping.encode_bytes(bytes(range(256)))) != bytes(
        range(256)
    ):
        raise ByteLevelMappingError("the discovered 256-byte mapping is not bijective")
    return mapping


class HFNativeRawByteCodec:
    """Serialize every ndarray modality through one parameter-free byte path."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        context_limit: int | None = None,
        limits: RawByteCodecLimits | None = None,
    ) -> None:
        self.mapping = discover_byte_token_mapping(tokenizer)
        self.limits = limits or RawByteCodecLimits()
        self.context_limit, self.context_limit_source = _resolve_context_limit(
            tokenizer,
            context_limit,
        )

    def protocol_report(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "serialization_path": SERIALIZATION_PATH,
            "framing": "ASCII magic + fixed-width header length + canonical ASCII JSON + raw payload",
            "payload_tokenization": "direct existing ByteLevel atom IDs",
            "context_limit": self.context_limit,
            "context_limit_source": self.context_limit_source,
            "limits": dict(vars(self.limits)),
            "mapping": self.mapping.report(),
            "parameter_count": 0,
            "learned_modules": [],
            "encoder": None,
            "decoder": None,
            "adapter": None,
            "projector": None,
            "bridge": None,
            "runtime_donor": None,
            "modality_specific_codecs": [],
        }

    def validate_model_input_embedding(self, model: Any) -> dict[str, Any]:
        """Prove the byte IDs fit an existing LM embedding without mutating it."""

        get_input = getattr(model, "get_input_embeddings", None)
        if not callable(get_input):
            raise ByteLevelMappingError("model must provide get_input_embeddings()")
        module = get_input()
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2:
            raise ByteLevelMappingError("model input embedding must have rank-2 weight")
        rows = int(weight.shape[0])
        width = int(weight.shape[1])
        if rows <= max(self.mapping.token_ids):
            raise ByteLevelMappingError(
                "model input embedding does not contain every discovered byte row"
            )
        model_context = _model_context_limit(model)
        if (
            self.context_limit is not None
            and model_context is not None
            and self.context_limit > model_context
        ):
            raise RawTensorContextError(
                f"codec context_limit {self.context_limit} exceeds model limit {model_context}"
            )
        return {
            "input_embedding_rows": rows,
            "input_embedding_width": width,
            "maximum_byte_token_id": max(self.mapping.token_ids),
            "all_byte_ids_within_input_embedding": True,
            "model_context_limit": model_context,
            "codec_context_limit": self.context_limit,
            "model_mutated": False,
            "parameter_count_added": 0,
            "learned_modules_added": [],
        }

    def encode_tensor(
        self,
        tensor: np.ndarray,
        *,
        modality: str,
        metadata: Mapping[str, Any] | None = None,
        context_limit: int | None = None,
    ) -> EncodedRawTensor:
        """Serialize any supported ndarray without modality-specific dispatch."""

        array = _validated_array(tensor, self.limits, error_type=RawTensorEncodeError)
        label = _validated_modality(modality, error_type=RawTensorEncodeError)
        normalized_metadata = _normalized_metadata(
            metadata,
            max_depth=self.limits.max_metadata_depth,
            error_type=RawTensorEncodeError,
        )
        payload = array.tobytes(order="C")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        header_core = {
            "byte_mapping_sha256": self.mapping.mapping_sha256,
            "dtype": array.dtype.str,
            "metadata": normalized_metadata,
            "modality": label,
            "order": "C",
            "payload_bytes": len(payload),
            "payload_sha256": payload_sha256,
            "rank": array.ndim,
            "schema": PROTOCOL_NAME,
            "shape": [int(value) for value in array.shape],
            "version": PROTOCOL_VERSION,
        }
        content_sha256 = hashlib.sha256(
            _canonical_json_bytes(header_core) + payload
        ).hexdigest()
        header = {**header_core, "content_sha256": content_sha256}
        header_payload = _canonical_json_bytes(header)
        if len(header_payload) > self.limits.max_header_bytes:
            raise RawTensorBoundsError("canonical header exceeds max_header_bytes")
        if any(value >= 128 for value in header_payload):
            raise AssertionError("canonical header must contain ASCII bytes only")
        prefix = _MAGIC + b":" + f"{len(header_payload):08x}".encode("ascii") + b":"
        frame = prefix + header_payload + payload
        input_ids = self.mapping.encode_bytes(frame)
        effective_context = self._effective_context_limit(context_limit)
        chunks = _chunk_token_ids(input_ids, effective_context)
        if tuple(value for chunk in chunks for value in chunk.input_ids) != input_ids:
            raise AssertionError("context chunking changed or dropped token IDs")
        return EncodedRawTensor(
            modality=label,
            shape=tuple(int(value) for value in array.shape),
            dtype=array.dtype.str,
            metadata=_copy_json(normalized_metadata),
            input_ids=input_ids,
            chunks=chunks,
            payload_token_offset=len(prefix) + len(header_payload),
            payload_bytes=len(payload),
            payload_sha256=payload_sha256,
            content_sha256=content_sha256,
            header_bytes=len(header_payload),
            header_sha256=hashlib.sha256(header_payload).hexdigest(),
            frame_sha256=hashlib.sha256(frame).hexdigest(),
            byte_mapping_sha256=self.mapping.mapping_sha256,
            context_limit=effective_context,
        )

    serialize_tensor = encode_tensor
    encode = encode_tensor

    def decode_tensor(
        self,
        token_ids: Iterable[int],
        *,
        expected_report: Mapping[str, Any] | None = None,
    ) -> DecodedRawTensor:
        ids = _bounded_token_ids(token_ids, self._max_frame_tokens)
        return self._decode_ids(ids, expected_report=expected_report)

    deserialize_tensor = decode_tensor
    decode = decode_tensor

    def decode_chunks(
        self,
        chunks: Iterable[RawTokenChunk],
        *,
        expected_report: Mapping[str, Any] | None = None,
    ) -> DecodedRawTensor:
        values = tuple(chunks)
        if not values:
            raise RawTensorDecodeError("at least one raw-token chunk is required")
        expected_start = 0
        stream_count = values[0].stream_token_count
        stream_hash = values[0].stream_token_ids_sha256
        context = values[0].context_limit
        ids: list[int] = []
        for expected_index, chunk in enumerate(values):
            if not isinstance(chunk, RawTokenChunk):
                raise RawTensorDecodeError("chunks must contain RawTokenChunk values")
            if chunk.index != expected_index or chunk.start_token != expected_start:
                raise RawTensorDecodeError(
                    "raw-token chunks are missing or out of order"
                )
            if chunk.stream_token_count != stream_count:
                raise RawTensorDecodeError("chunks disagree on stream_token_count")
            if chunk.stream_token_ids_sha256 != stream_hash:
                raise RawTensorDecodeError("chunks disagree on stream token hash")
            if chunk.context_limit != context:
                raise RawTensorDecodeError("chunks disagree on context_limit")
            if chunk.attention_mask != (1,) * len(chunk.input_ids):
                raise RawTensorDecodeError(
                    "chunk attention mask is not strict binary-one"
                )
            ids.extend(chunk.input_ids)
            expected_start = chunk.stop_token
        if len(ids) != stream_count:
            raise RawTensorDecodeError("chunk sequence is truncated")
        actual_stream_hash = _token_ids_sha256(ids)
        if not hmac.compare_digest(actual_stream_hash, stream_hash):
            raise RawTensorDecodeError("reassembled chunk token hash does not match")
        decoded = self._decode_ids(tuple(ids), expected_report=expected_report)
        return DecodedRawTensor(
            modality=decoded.modality,
            array=decoded.array,
            metadata=decoded.metadata,
            payload_sha256=decoded.payload_sha256,
            content_sha256=decoded.content_sha256,
            header_sha256=decoded.header_sha256,
            frame_sha256=decoded.frame_sha256,
            token_ids_sha256=decoded.token_ids_sha256,
            byte_mapping_sha256=decoded.byte_mapping_sha256,
            payload_token_offset=decoded.payload_token_offset,
            context_limit=context,
            chunk_token_counts=tuple(len(chunk.input_ids) for chunk in values),
        )

    def _effective_context_limit(self, override: int | None) -> int:
        if override is not None:
            return _positive_int(
                override,
                "context_limit",
                error_type=RawTensorContextError,
            )
        if self.context_limit is None:
            raise RawTensorContextError(
                "context_limit is required because tokenizer.model_max_length is not trustworthy"
            )
        return self.context_limit

    @property
    def _max_frame_tokens(self) -> int:
        return (
            _FRAME_PREFIX_SIZE
            + self.limits.max_header_bytes
            + self.limits.max_payload_bytes
        )

    def _decode_ids(
        self,
        ids: tuple[int, ...],
        *,
        expected_report: Mapping[str, Any] | None,
    ) -> DecodedRawTensor:
        try:
            frame = self.mapping.decode_token_ids(ids)
        except ByteLevelMappingError as exc:
            raise RawTensorDecodeError(str(exc)) from exc
        header_payload, payload, payload_offset = _split_frame(frame, self.limits)
        header = _parse_header(header_payload, self.mapping, self.limits)
        if len(payload) != header["payload_bytes"]:
            raise RawTensorDecodeError(
                "payload length does not match canonical header declaration"
            )
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(payload_sha256, header["payload_sha256"]):
            raise RawTensorDecodeError("payload SHA-256 validation failed")
        header_core = {
            name: value for name, value in header.items() if name != "content_sha256"
        }
        content_sha256 = hashlib.sha256(
            _canonical_json_bytes(header_core) + payload
        ).hexdigest()
        if not hmac.compare_digest(content_sha256, header["content_sha256"]):
            raise RawTensorDecodeError(
                "header-and-payload content SHA-256 validation failed"
            )

        dtype = _validated_dtype(header["dtype"], error_type=RawTensorDecodeError)
        shape = tuple(header["shape"])
        _validate_shape_and_size(
            shape,
            dtype,
            self.limits,
            declared_payload_bytes=header["payload_bytes"],
            error_type=RawTensorDecodeError,
        )
        try:
            array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy(order="C")
        except (TypeError, ValueError) as exc:
            raise RawTensorDecodeError(
                "payload cannot be reshaped as declared"
            ) from exc
        if array.tobytes(order="C") != payload:
            raise RawTensorDecodeError("decoded ndarray did not preserve payload bytes")

        actual = {
            "payload_sha256": payload_sha256,
            "content_sha256": content_sha256,
            "header_sha256": hashlib.sha256(header_payload).hexdigest(),
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
            "token_ids_sha256": _token_ids_sha256(ids),
            "byte_mapping_sha256": self.mapping.mapping_sha256,
        }
        _validate_expected_report(expected_report, actual)
        return DecodedRawTensor(
            modality=header["modality"],
            array=array,
            metadata=_copy_json(header["metadata"]),
            payload_sha256=payload_sha256,
            content_sha256=content_sha256,
            header_sha256=actual["header_sha256"],
            frame_sha256=actual["frame_sha256"],
            token_ids_sha256=actual["token_ids_sha256"],
            byte_mapping_sha256=actual["byte_mapping_sha256"],
            payload_token_offset=payload_offset,
        )


HFNativeRawTokenCodec = HFNativeRawByteCodec
NativeRawByteCodec = HFNativeRawByteCodec


def codec_from_bytelevel_tokenizer(
    tokenizer: Any,
    *,
    context_limit: int | None = None,
    limits: RawByteCodecLimits | None = None,
) -> HFNativeRawByteCodec:
    return HFNativeRawByteCodec(
        tokenizer,
        context_limit=context_limit,
        limits=limits,
    )


def _resolve_context_limit(
    tokenizer: Any,
    explicit: int | None,
) -> tuple[int | None, str]:
    if explicit is not None:
        return (
            _positive_int(
                explicit,
                "context_limit",
                error_type=RawTensorContextError,
            ),
            "explicit",
        )
    raw = getattr(tokenizer, "model_max_length", None)
    if raw is None or isinstance(raw, bool):
        return None, "unavailable"
    try:
        candidate = operator.index(raw)
    except (TypeError, ValueError, OverflowError):
        return None, "unavailable"
    if 0 < candidate <= _MAX_TRUSTED_AUTO_CONTEXT:
        return candidate, "tokenizer.model_max_length"
    return None, "untrusted_tokenizer_sentinel"


def _model_context_limit(model: Any) -> int | None:
    config = getattr(model, "config", None)
    candidates: list[int] = []
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            candidate = operator.index(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 < candidate <= _MAX_TRUSTED_AUTO_CONTEXT:
            candidates.append(candidate)
    return min(candidates) if candidates else None


def _chunk_token_ids(
    input_ids: tuple[int, ...],
    context_limit: int,
) -> tuple[RawTokenChunk, ...]:
    if not input_ids:
        raise RawTensorContextError("raw tensor frame must not be empty")
    stream_hash = _token_ids_sha256(input_ids)
    result = []
    for index, start in enumerate(range(0, len(input_ids), context_limit)):
        result.append(
            RawTokenChunk(
                index=index,
                start_token=start,
                input_ids=input_ids[start : start + context_limit],
                context_limit=context_limit,
                stream_token_count=len(input_ids),
                stream_token_ids_sha256=stream_hash,
            )
        )
    return tuple(result)


def _split_frame(
    frame: bytes,
    limits: RawByteCodecLimits,
) -> tuple[bytes, bytes, int]:
    if len(frame) < _FRAME_PREFIX_SIZE:
        raise RawTensorDecodeError("raw-token frame is truncated before its header")
    if not frame.startswith(_MAGIC + b":"):
        raise RawTensorDecodeError("raw-token frame magic/version is invalid")
    length_start = len(_MAGIC) + 1
    length_stop = length_start + _LENGTH_WIDTH
    raw_length = frame[length_start:length_stop]
    if frame[length_stop : length_stop + 1] != b":":
        raise RawTensorDecodeError("raw-token frame header separator is invalid")
    try:
        text_length = raw_length.decode("ascii")
        header_length = int(text_length, 16)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RawTensorDecodeError(
            "raw-token header length is not canonical hex"
        ) from exc
    if f"{header_length:08x}" != text_length:
        raise RawTensorDecodeError("raw-token header length is not canonical hex")
    if header_length > limits.max_header_bytes:
        raise RawTensorBoundsError("canonical header exceeds max_header_bytes")
    header_start = length_stop + 1
    payload_start = header_start + header_length
    if payload_start > len(frame):
        raise RawTensorDecodeError("raw-token frame is truncated inside its header")
    return frame[header_start:payload_start], frame[payload_start:], payload_start


def _parse_header(
    payload: bytes,
    mapping: ByteLevelByteMapping,
    limits: RawByteCodecLimits,
) -> dict[str, Any]:
    if any(value >= 128 for value in payload):
        raise RawTensorDecodeError("canonical tensor header must be ASCII")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawTensorDecodeError("canonical tensor header is invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != _HEADER_FIELDS:
        raise RawTensorDecodeError("canonical tensor header fields are invalid")
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise RawTensorDecodeError(
            "canonical tensor header values are invalid"
        ) from exc
    if canonical != payload:
        raise RawTensorDecodeError("tensor header JSON is not canonical")
    version = _nonnegative_int(
        value["version"],
        "version",
        error_type=RawTensorDecodeError,
    )
    if value["schema"] != PROTOCOL_NAME or version != PROTOCOL_VERSION:
        raise RawTensorDecodeError("tensor header schema/version is unsupported")
    if value["order"] != "C":
        raise RawTensorDecodeError(
            "only canonical C-order tensor payloads are supported"
        )
    if not _is_sha256(value["byte_mapping_sha256"]):
        raise RawTensorDecodeError(
            "byte_mapping_sha256 is not canonical lowercase SHA-256"
        )
    if value["byte_mapping_sha256"] != mapping.mapping_sha256:
        raise RawTensorDecodeError(
            "tensor header ByteLevel mapping hash does not match"
        )
    if not _is_sha256(value["payload_sha256"]):
        raise RawTensorDecodeError("payload_sha256 is not canonical lowercase SHA-256")
    if not _is_sha256(value["content_sha256"]):
        raise RawTensorDecodeError("content_sha256 is not canonical lowercase SHA-256")
    value["modality"] = _validated_modality(
        value["modality"],
        error_type=RawTensorDecodeError,
    )
    value["metadata"] = _normalized_metadata(
        value["metadata"],
        max_depth=limits.max_metadata_depth,
        error_type=RawTensorDecodeError,
    )
    value["payload_bytes"] = _nonnegative_int(
        value["payload_bytes"],
        "payload_bytes",
        error_type=RawTensorDecodeError,
    )
    if value["payload_bytes"] > limits.max_payload_bytes:
        raise RawTensorBoundsError("payload exceeds max_payload_bytes")
    rank = _nonnegative_int(value["rank"], "rank", error_type=RawTensorDecodeError)
    shape = value["shape"]
    if not isinstance(shape, list) or len(shape) != rank:
        raise RawTensorDecodeError("rank does not match the shape field")
    if rank > limits.max_rank:
        raise RawTensorBoundsError("tensor rank exceeds max_rank")
    value["shape"] = [
        _nonnegative_int(item, f"shape[{index}]", error_type=RawTensorDecodeError)
        for index, item in enumerate(shape)
    ]
    return value


def _validated_array(
    value: Any,
    limits: RawByteCodecLimits,
    *,
    error_type: type[HFNativeRawTokenError],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise error_type("tensor must be a NumPy ndarray")
    dtype = _validated_dtype(value.dtype, error_type=error_type)
    shape = tuple(int(item) for item in value.shape)
    _validate_shape_and_size(
        shape,
        dtype,
        limits,
        declared_payload_bytes=value.nbytes,
        error_type=error_type,
    )
    return value


def _validated_dtype(
    value: Any,
    *,
    error_type: type[HFNativeRawTokenError],
) -> np.dtype[Any]:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise error_type("dtype is not a valid NumPy dtype") from exc
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise error_type("object, structured, and subarray dtypes are not supported")
    if dtype.kind not in _SUPPORTED_DTYPE_KINDS or dtype.itemsize <= 0:
        raise error_type(
            "dtype must be a fixed-width bool, signed/unsigned integer, float, or complex"
        )
    if np.dtype(dtype.str).str != dtype.str:
        raise error_type("dtype does not have a canonical portable descriptor")
    return dtype


def _validate_shape_and_size(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    limits: RawByteCodecLimits,
    *,
    declared_payload_bytes: int,
    error_type: type[HFNativeRawTokenError],
) -> None:
    if len(shape) > limits.max_rank:
        raise RawTensorBoundsError("tensor rank exceeds max_rank")
    for index, dimension in enumerate(shape):
        if dimension < 0:
            raise error_type(f"shape[{index}] must be non-negative")
        if dimension > limits.max_dimension:
            raise RawTensorBoundsError(f"shape[{index}] exceeds max_dimension")
    elements = math.prod(shape) if shape else 1
    if elements > limits.max_elements:
        raise RawTensorBoundsError("tensor element count exceeds max_elements")
    expected_bytes = elements * dtype.itemsize
    if expected_bytes > limits.max_payload_bytes:
        raise RawTensorBoundsError("tensor payload exceeds max_payload_bytes")
    if declared_payload_bytes != expected_bytes:
        raise error_type("shape, dtype, and payload byte count are inconsistent")


def _normalized_metadata(
    value: Mapping[str, Any] | None,
    *,
    max_depth: int,
    error_type: type[HFNativeRawTokenError],
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise error_type("metadata must be a mapping with string keys")
    normalized = _normalize_json_value(
        value,
        depth=0,
        max_depth=max_depth,
        error_type=error_type,
    )
    if not isinstance(normalized, dict):
        raise error_type("metadata root must be an object")
    return normalized


def _normalize_json_value(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    error_type: type[HFNativeRawTokenError],
) -> Any:
    if depth > max_depth:
        raise error_type("metadata exceeds max_metadata_depth")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type("metadata floats must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise error_type("metadata keys must be strings")
            result[key] = _normalize_json_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                error_type=error_type,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                error_type=error_type,
            )
            for item in value
        ]
    raise error_type(
        f"metadata value {type(value).__name__} is not canonical JSON data"
    )


def _validated_modality(
    value: Any,
    *,
    error_type: type[HFNativeRawTokenError],
) -> str:
    if not isinstance(value, str) or _MODALITY.fullmatch(value) is None:
        raise error_type("modality must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _copy_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json(item) for item in value]
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _validate_expected_report(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, str],
) -> None:
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise RawTensorDecodeError("expected_report must be a mapping")
    for name, observed in actual.items():
        if name not in expected:
            raise RawTensorDecodeError(f"expected_report is missing strict hash {name}")
        claimed = expected[name]
        if not isinstance(claimed, str) or not hmac.compare_digest(claimed, observed):
            raise RawTensorDecodeError(f"expected_report {name} does not match")


def _bounded_token_ids(values: Iterable[int], limit: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise RawTensorDecodeError("token_ids must be an iterable of integers")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise RawTensorDecodeError("token_ids must be iterable") from exc
    result = []
    for position, value in enumerate(iterator):
        if position >= limit:
            raise RawTensorBoundsError("raw-token frame exceeds configured bounds")
        try:
            result.append(_token_id(value, f"token ID at position {position}"))
        except (TypeError, ValueError) as exc:
            raise RawTensorDecodeError(str(exc)) from exc
    if not result:
        raise RawTensorDecodeError("raw-token frame must not be empty")
    return tuple(result)


def _validated_vocabulary(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ByteLevelMappingError("tokenizer vocabulary must be a mapping")
    result: dict[str, int] = {}
    for token, raw_id in value.items():
        if not isinstance(token, str):
            raise ByteLevelMappingError("tokenizer vocabulary keys must be strings")
        result[token] = _token_id(raw_id, f"token ID for {token!r}")
    return result


def _backend_model_vocabulary(backend_data: Mapping[str, Any]) -> dict[str, int]:
    model = backend_data.get("model")
    if not isinstance(model, Mapping):
        raise ByteLevelMappingError("tokenizer backend model is missing")
    raw_vocab = model.get("vocab")
    if isinstance(raw_vocab, Mapping):
        return _validated_vocabulary(raw_vocab)
    if isinstance(raw_vocab, list):
        result: dict[str, int] = {}
        for token_id, item in enumerate(raw_vocab):
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) < 1
                or not isinstance(item[0], str)
            ):
                raise ByteLevelMappingError("tokenizer model vocabulary is malformed")
            result[item[0]] = token_id
        return result
    raise ByteLevelMappingError("tokenizer model vocabulary is malformed")


def _contains_component(value: Any, component_type: str) -> bool:
    if isinstance(value, Mapping):
        if value.get("type") == component_type:
            return True
        return any(_contains_component(item, component_type) for item in value.values())
    if isinstance(value, list):
        return any(_contains_component(item, component_type) for item in value)
    return False


def _bytelevel_token_strings() -> tuple[str, ...]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values.extend(range(ord("\u00a1"), ord("\u00ac") + 1))
    byte_values.extend(range(ord("\u00ae"), ord("\u00ff") + 1))
    codepoints = list(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    byte_to_codepoint = dict(zip(byte_values, codepoints, strict=True))
    tokens = tuple(chr(byte_to_codepoint[byte]) for byte in range(256))
    try:
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise ByteLevelMappingError(
            "tokenizers is required for ByteLevel validation"
        ) from exc
    installed_alphabet = ByteLevel.alphabet()
    if len(installed_alphabet) != 256 or set(installed_alphabet) != set(tokens):
        raise ByteLevelMappingError(
            "installed tokenizers ByteLevel alphabet differs from the exact byte map"
        )
    return tokens


def _token_ids_sha256(
    token_ids: Iterable[int],
    *,
    domain: bytes = b"raw-token-ids-v1",
) -> str:
    values = tuple(token_ids)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">Q", _token_id(value, "token ID")))
    return digest.hexdigest()


def _token_id(value: Any, label: str) -> int:
    return _nonnegative_int(value, label, error_type=ByteLevelMappingError)


def _positive_int(
    value: Any,
    label: str,
    *,
    error_type: type[Exception],
) -> int:
    result = _nonnegative_int(value, label, error_type=error_type)
    if result == 0:
        raise error_type(f"{label} must be a positive integer")
    return result


def _nonnegative_int(
    value: Any,
    label: str,
    *,
    error_type: type[Exception],
) -> int:
    if isinstance(value, bool):
        raise error_type(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error_type(f"{label} must be an integer") from exc
    if result < 0:
        raise error_type(f"{label} must be non-negative")
    if result > _MAX_TOKEN_ID and "token" in label.lower():
        raise error_type(f"{label} exceeds the supported 63-bit range")
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


__all__ = [
    "ByteLevelByteMapping",
    "ByteLevelMappingError",
    "DecodedRawTensor",
    "EncodedRawTensor",
    "HFNativeRawByteCodec",
    "HFNativeRawTokenCodec",
    "HFNativeRawTokenError",
    "NativeRawByteCodec",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "RawByteCodecLimits",
    "RawTensorBoundsError",
    "RawTensorContextError",
    "RawTensorDecodeError",
    "RawTensorEncodeError",
    "RawTokenChunk",
    "SERIALIZATION_PATH",
    "codec_from_bytelevel_tokenizer",
    "discover_byte_token_mapping",
]
