"""Binary codec and packed runtime for causal residual-entropy value folding.

This is the serialization foundation for CREVF.  It deliberately does not
choose important sites, estimate causal or Hessian sensitivity, or claim
behavioral retention.  It provides the byte-exact substrate those later
components need:

* independently variable four- through eight-bit symmetric groups;
* finite positive FP16 scales;
* deterministic fixed-DEFLATE entropy coding per tensor;
* sparse exact BF16 outliers and run-encoded exact BF16 anchors;
* exact-only BF16 records for norms and other small control tensors;
* bounded authenticated byte records for writer indexes and codebooks;
* tensor, record, and whole-artifact SHA-256 authentication; and
* a runtime that owns packed buffers but no persistent dense Parameters.

All serialized metadata is fixed-width binary or canonical minimal ULEB128.
Exact values are raw little-endian BF16 bits.  JSON and Python object
serialization are not part of the format.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import unicodedata
import zlib
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional HF dependency
    raise RuntimeError("PyTorch is required for the CREVF packed runtime.") from exc


CREVF_FILE_MAGIC = b"NFCREVF1"
CREVF_TENSOR_MAGIC = b"NFT1"
CREVF_AUXILIARY_MAGIC = b"NFA1"
CREVF_ENTROPY_MAGIC = b"DFX1"
CREVF_OUTLIER_MAGIC = b"OBF1"
CREVF_ANCHOR_MAGIC = b"ABF1"
CREVF_VERSION = 1

_FILE_HEADER = struct.Struct("<8sHHIQ")
_RECORD_ENVELOPE = struct.Struct("<Q32s")
_TENSOR_HEADER = struct.Struct("<4sBBBBHHIQQQQQQQ")
_AUXILIARY_HEADER = struct.Struct("<4sBBHHQ")
_ENTROPY_HEADER = struct.Struct("<4sBBBBQQ")
_OUTLIER_HEADER = struct.Struct("<4sQ")
_ANCHOR_HEADER = struct.Struct("<4sQI")
_SHA256_BYTES = 32
_DTYPE_BFLOAT16 = 1
_FLAG_EXACT_ONLY = 1
_FLAG_OUTLIERS = 2
_FLAG_ANCHORS = 4
_KNOWN_FLAGS = _FLAG_EXACT_ONLY | _FLAG_OUTLIERS | _FLAG_ANCHORS
_FILE_FLAG_AUXILIARY_RECORDS = 1
_KNOWN_FILE_FLAGS = _FILE_FLAG_AUXILIARY_RECORDS
_MAX_RANK = 16
_MAX_TEXT_BYTES = 65_535
_MAX_GROUP_SIZE = 1 << 20
_MAX_UINT64 = (1 << 64) - 1
_QUANT_CHUNK_GROUPS = 4096
_DEFAULT_MAX_TENSOR_DECODE_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_AUXILIARY_BYTES = 128 * 1024 * 1024
_TARGET_REDUCTION_PPM = frozenset({500_000, 550_000, 600_000, 650_000})

CREVF_ARTIFACT_FIXED_BYTES = _FILE_HEADER.size + _SHA256_BYTES
CREVF_RECORD_ENVELOPE_BYTES = _RECORD_ENVELOPE.size


def _zlib_runtime_profile() -> tuple[int, int, int, int]:
    components = tuple(
        int(value) for value in re.findall(r"\d+", zlib.ZLIB_RUNTIME_VERSION)
    )
    if (
        not components
        or len(components) > 4
        or any(value > 255 for value in components)
    ):
        raise RuntimeError("CREVF requires a four-byte-addressable zlib version.")
    return components + (0,) * (4 - len(components))


CREVF_ZLIB_PROFILE = _zlib_runtime_profile()


class CrevfError(ValueError):
    """The requested CREVF representation is invalid."""


class CrevfFormatError(CrevfError):
    """A CREVF tensor or artifact is malformed, noncanonical, or corrupted."""


class CrevfBudgetError(CrevfError):
    """A physical-byte target cannot contain the requested CREVF artifact."""


@dataclass(frozen=True, slots=True)
class CrevfExactValue:
    """One sparse exact BF16 value represented by its raw 16-bit payload."""

    index: int
    bf16_bits: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
            or isinstance(self.bf16_bits, bool)
            or not isinstance(self.bf16_bits, int)
            or not 0 <= self.bf16_bits <= 0xFFFF
        ):
            raise CrevfError(
                "Exact BF16 entries require a nonnegative index and uint16 bits."
            )


@dataclass(frozen=True, slots=True)
class CrevfExactRun:
    """One contiguous binary exact-anchor run."""

    start_index: int
    bf16_bytes: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or self.start_index < 0
            or not isinstance(self.bf16_bytes, bytes)
            or not self.bf16_bytes
            or len(self.bf16_bytes) % 2
        ):
            raise CrevfError(
                "Exact BF16 runs require a nonnegative start and nonempty "
                "little-endian uint16 bytes."
            )

    @classmethod
    def from_bits(
        cls,
        start_index: int,
        values: Sequence[int],
    ) -> "CrevfExactRun":
        bits = tuple(values)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 0xFFFF
            for value in bits
        ):
            raise CrevfError("Exact BF16 run words must be uint16 integers.")
        return cls(
            start_index,
            np.asarray(bits, dtype="<u2").tobytes(order="C"),
        )

    @property
    def value_count(self) -> int:
        return len(self.bf16_bytes) // 2

    @property
    def end_index(self) -> int:
        return self.start_index + self.value_count


@dataclass(frozen=True, slots=True)
class CrevfTensorSpec:
    """Canonical inputs for one encoded tensor.

    Quantized records provide one width and one padded code row per group.
    Exact-only records provide no quantized fields and cover the full tensor
    through exact anchor runs.
    """

    name: str
    role: str
    shape: tuple[int, ...]
    group_size: int = 0
    group_bit_widths: tuple[int, ...] = ()
    group_codes: Any = ()
    scales: Any = ()
    exact_outliers: tuple[CrevfExactValue, ...] = ()
    exact_anchors: tuple[CrevfExactRun, ...] = ()
    exact_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "group_bit_widths", tuple(self.group_bit_widths))
        if not isinstance(self.group_codes, (Tensor, np.ndarray)):
            object.__setattr__(self, "group_codes", tuple(self.group_codes))
        object.__setattr__(self, "exact_outliers", tuple(self.exact_outliers))
        object.__setattr__(self, "exact_anchors", tuple(self.exact_anchors))


@dataclass(frozen=True, slots=True)
class CrevfTensorMetadata:
    name: str
    role: str
    shape: tuple[int, ...]
    source_numel: int
    group_size: int
    group_count: int
    group_bit_widths: tuple[int, ...]
    exact_only: bool
    exact_outlier_count: int
    exact_anchor_count: int
    tensor_blob_bytes: int
    entropy_segment_bytes: int
    decoded_code_bytes: int
    persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    decoded_code_transient_bytes: int

    @property
    def runtime_packed_bytes(self) -> int:
        """Backward-compatible alias for persistent encoded runtime bytes."""

        return self.persistent_encoded_bytes

    @property
    def minimum_bit_width(self) -> int:
        return 16 if self.exact_only else min(self.group_bit_widths)

    @property
    def maximum_bit_width(self) -> int:
        return 16 if self.exact_only else max(self.group_bit_widths)

    @property
    def precision_kind(self) -> str:
        return "exact_bf16" if self.exact_only else "mixed_integer"


@dataclass(frozen=True, slots=True)
class CrevfAuxiliarySpec:
    """One bounded opaque runtime payload such as writer indexes or a codebook."""

    name: str
    kind: str
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise CrevfError("CREVF auxiliary payloads must be nonempty bytes.")


@dataclass(frozen=True, slots=True)
class CrevfAuxiliaryMetadata:
    """Authenticated physical and runtime accounting for an auxiliary record."""

    name: str
    kind: str
    payload_bytes: int
    auxiliary_blob_bytes: int
    persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    decoded_code_transient_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CrevfQuantizedCandidate:
    """One fully encoded, byte-chargeable quantization-policy candidate."""

    candidate_id: str
    group_size: int
    bit_width: int
    exact_outliers_per_group: int
    clipping_ratio_ppm: int
    rounding_offset_ppm: int
    tensor_blob: bytes

    @property
    def tensor_physical_bytes(self) -> int:
        return len(self.tensor_blob)


@dataclass(frozen=True, slots=True)
class CrevfReductionTarget:
    """One supported exact checkpoint reduction target."""

    reduction_ppm: int

    def __post_init__(self) -> None:
        if self.reduction_ppm not in _TARGET_REDUCTION_PPM:
            raise CrevfError(
                "CREVF reduction_ppm must be exactly 500000, 550000, 600000, or 650000."
            )

    @classmethod
    def from_reduction(cls, reduction: float) -> "CrevfReductionTarget":
        if isinstance(reduction, bool) or not isinstance(reduction, (int, float)):
            raise CrevfError(
                "target_reduction must be exactly 0.50, 0.55, 0.60, or 0.65."
            )
        ppm = int(round(float(reduction) * 1_000_000))
        if not math.isclose(float(reduction), ppm / 1_000_000, abs_tol=1e-12):
            raise CrevfError(
                "target_reduction must have an exact parts-per-million form."
            )
        return cls(ppm)

    @property
    def reduction(self) -> float:
        return self.reduction_ppm / 1_000_000

    def checkpoint_ceiling(self, dense_reference_bytes: int) -> int:
        _require_nonnegative_int(dense_reference_bytes, "dense_reference_bytes")
        if dense_reference_bytes == 0:
            raise CrevfError("dense_reference_bytes must be positive.")
        return dense_reference_bytes * (1_000_000 - self.reduction_ppm) // 1_000_000


@dataclass(frozen=True, slots=True)
class CrevfByteAccounting:
    """Exact physical, persistent, diagnostic, and transient accounting."""

    dense_reference_bytes: int
    reduction_ppm: int
    target_checkpoint_bytes: int
    effective_checkpoint_ceiling_bytes: int
    must_beat_physical_bytes: int | None
    artifact_physical_bytes: int
    shared_checkpoint_bytes: int
    checkpoint_physical_bytes: int
    persistent_encoded_bytes: int
    shared_persistent_encoded_bytes: int
    total_persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    shared_decoded_full_state_bytes: int
    total_decoded_full_state_bytes: int
    maximum_tensor_decode_transient_bytes: int
    maximum_layer_decode_transient_bytes: int
    layer_decode_transient_bytes: tuple[tuple[str, int], ...]
    headroom_bytes: int
    effective_headroom_bytes: int
    max_persistent_encoded_bytes: int | None
    persistent_encoded_headroom_bytes: int | None
    max_decoded_full_state_bytes: int | None
    decoded_full_state_headroom_bytes: int | None
    max_layer_transient_bytes: int | None
    layer_transient_headroom_bytes: int | None

    @property
    def runtime_packed_bytes(self) -> int:
        """Backward-compatible alias for artifact persistent encoded bytes."""

        return self.persistent_encoded_bytes

    @property
    def shared_runtime_bytes(self) -> int:
        """Backward-compatible alias for shared persistent encoded bytes."""

        return self.shared_persistent_encoded_bytes

    @property
    def total_runtime_bytes(self) -> int:
        """Backward-compatible alias for total persistent encoded bytes."""

        return self.total_persistent_encoded_bytes

    @property
    def max_runtime_bytes(self) -> int | None:
        """Backward-compatible alias for the persistent encoded ceiling."""

        return self.max_persistent_encoded_bytes

    @property
    def runtime_headroom_bytes(self) -> int | None:
        """Backward-compatible alias for persistent encoded headroom."""

        return self.persistent_encoded_headroom_bytes

    @property
    def realized_reduction(self) -> float:
        return 1.0 - self.checkpoint_physical_bytes / self.dense_reference_bytes


@dataclass(slots=True)
class _ParsedTensor:
    metadata: CrevfTensorMetadata
    width_segment: bytes
    scale_segment: bytes
    entropy_segment: bytes
    outlier_segment: bytes
    anchor_segment: bytes


@dataclass(frozen=True, slots=True)
class _ParsedAuxiliary:
    metadata: CrevfAuxiliaryMetadata
    payload: bytes


def _sha256(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CrevfError(f"{name} must be a nonnegative integer.")


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _canonical_text(value: str, name: str) -> bytes:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise CrevfError(f"{name} must be nonempty, NUL-free, canonical NFC text.")
    raw = value.encode("utf-8")
    if len(raw) > _MAX_TEXT_BYTES:
        raise CrevfError(f"{name} exceeds the binary header limit.")
    return raw


def _decode_canonical_text(raw: bytes, name: str) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CrevfFormatError(f"{name} is not valid UTF-8.") from exc
    try:
        expected = _canonical_text(value, name)
    except CrevfError as exc:
        raise CrevfFormatError(str(exc)) from exc
    if expected != raw:
        raise CrevfFormatError(f"{name} is not canonical UTF-8.")
    return value


def _encode_uleb128(value: int) -> bytes:
    _require_nonnegative_int(value, "ULEB128 value")
    if value > _MAX_UINT64:
        raise CrevfError("ULEB128 value exceeds the uint64 format limit.")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def _decode_uleb128(raw: bytes, offset: int) -> tuple[int, int]:
    start = offset
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(raw):
            raise CrevfFormatError("Truncated ULEB128 value.")
        byte = raw[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value > _MAX_UINT64:
                raise CrevfFormatError("ULEB128 value exceeds the uint64 format limit.")
            if raw[start:offset] != _encode_uleb128(value):
                raise CrevfFormatError("Nonminimal ULEB128 value.")
            return value, offset
        shift += 7
    raise CrevfFormatError("ULEB128 value exceeds the uint64 format limit.")


def _pack_unsigned(values: np.ndarray, bit_width: int) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
    if not 1 <= bit_width <= 8:
        raise CrevfError("Packed bit width must be in [1, 8].")
    maximum = (1 << bit_width) - 1
    if flat.size and int(flat.max()) > maximum:
        raise CrevfError("A code exceeds its declared bit width.")
    if not flat.size:
        return b""
    shifts = np.arange(bit_width, dtype=np.uint8)
    bits = ((flat[:, None] >> shifts[None, :]) & 1).reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def _unpack_unsigned(
    raw: bytes,
    *,
    value_count: int,
    bit_width: int,
    format_error: bool = True,
) -> np.ndarray:
    expected = (value_count * bit_width + 7) // 8
    error_type = CrevfFormatError if format_error else CrevfError
    if len(raw) != expected:
        raise error_type("Packed code byte count differs from its declared count.")
    if value_count == 0:
        return np.empty(0, dtype=np.uint8)
    bits = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8),
        bitorder="little",
    )
    used = value_count * bit_width
    if bool(bits[used:].any()):
        raise error_type("Nonzero packed padding bits are noncanonical.")
    rows = bits[:used].reshape(value_count, bit_width).astype(np.uint16)
    shifts = (1 << np.arange(bit_width, dtype=np.uint16))[None, :]
    return np.sum(rows * shifts, axis=1, dtype=np.uint16).astype(np.uint8)


def _group_code_matrix(
    values: Any,
    *,
    group_count: int,
    group_size: int,
    widths: Sequence[int],
) -> np.ndarray:
    if isinstance(values, Tensor):
        if (
            values.dtype == torch.bool
            or torch.is_floating_point(values)
            or torch.is_complex(values)
        ):
            raise CrevfError("Group codes must use an integer Tensor dtype.")
        source = values.detach().to(device="cpu").contiguous()
        if tuple(source.shape) != (group_count, group_size):
            raise CrevfError(
                "Group code matrix shape differs from (group_count, group_size)."
            )
        source_array = source.numpy()
        if source_array.size and (
            int(source_array.min()) < 0 or int(source_array.max()) > 255
        ):
            raise CrevfError("Group codes must be nonnegative uint8-range integers.")
        matrix = source_array.astype(np.uint8, copy=False)
    elif isinstance(values, np.ndarray):
        if values.dtype.kind not in {"i", "u"}:
            raise CrevfError("Group codes must use an integer NumPy dtype.")
        if values.shape != (group_count, group_size):
            raise CrevfError(
                "Group code matrix shape differs from (group_count, group_size)."
            )
        if values.size and (int(values.min()) < 0 or int(values.max()) > 255):
            raise CrevfError("Group codes must be nonnegative uint8-range integers.")
        matrix = np.ascontiguousarray(values, dtype=np.uint8)
    else:
        rows = tuple(values)
        if len(rows) != group_count:
            raise CrevfError("Every quantized group requires one padded code row.")
        checked = []
        for row in rows:
            if isinstance(row, Tensor):
                if (
                    row.dtype == torch.bool
                    or torch.is_floating_point(row)
                    or torch.is_complex(row)
                ):
                    raise CrevfError("Group code rows must use an integer dtype.")
                raw = row.detach().to(device="cpu").contiguous()
                raw_array = raw.numpy()
                if raw_array.size and (
                    int(raw_array.min()) < 0 or int(raw_array.max()) > 255
                ):
                    raise CrevfError(
                        "Group codes must be nonnegative uint8-range integers."
                    )
                array = raw_array.astype(np.uint8, copy=False)
            else:
                array = np.asarray(row)
                if array.dtype.kind not in {"i", "u"}:
                    raise CrevfError("Group code rows must use an integer dtype.")
                if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
                    raise CrevfError(
                        "Group codes must be nonnegative uint8-range integers."
                    )
                array = array.astype(np.uint8, copy=False)
            flat = np.ascontiguousarray(array.reshape(-1))
            if flat.size != group_size:
                raise CrevfError(
                    "Every group code row must include padded group_size values."
                )
            checked.append(flat)
        matrix = np.stack(checked, axis=0)
    if group_count:
        row_maxima = matrix.max(axis=1)
        allowed = np.asarray(
            [2 * ((1 << (width - 1)) - 1) for width in widths],
            dtype=np.uint16,
        )
        if bool((row_maxima.astype(np.uint16) > allowed).any()):
            raise CrevfError("A symmetric quantization code exceeds its group width.")
    return matrix


def _group_code_offsets(widths: np.ndarray, group_size: int) -> np.ndarray:
    byte_counts = widths.astype(np.int64) * group_size // 8
    return np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(byte_counts, dtype=np.int64))
    )


def _pack_group_code_stream(
    matrix: np.ndarray,
    widths: Sequence[int],
    group_size: int,
) -> bytes:
    width_array = np.asarray(widths, dtype=np.uint8)
    offsets = _group_code_offsets(width_array, group_size)
    output = np.empty(int(offsets[-1]), dtype=np.uint8)
    for width in range(4, 9):
        indices = np.flatnonzero(width_array == width)
        bytes_per_group = group_size * width // 8
        rows_per_chunk = max(1, 8_388_608 // (group_size * width))
        shifts = np.arange(width, dtype=np.uint8)
        byte_columns = np.arange(bytes_per_group, dtype=np.int64)
        for start in range(0, len(indices), rows_per_chunk):
            selected = indices[start : start + rows_per_chunk]
            rows = matrix[selected]
            bits = ((rows[:, :, None] >> shifts[None, None, :]) & 1).reshape(
                len(selected),
                group_size * width,
            )
            packed = np.packbits(bits, axis=1, bitorder="little")
            positions = offsets[selected, None] + byte_columns[None, :]
            output[positions] = packed
    return output.tobytes(order="C")


def _unpack_group_code_stream(
    raw: bytes,
    widths: Sequence[int],
    group_size: int,
) -> np.ndarray:
    width_array = np.asarray(widths, dtype=np.uint8)
    offsets = _group_code_offsets(width_array, group_size)
    if len(raw) != int(offsets[-1]):
        raise CrevfFormatError("Decoded group code stream has the wrong byte count.")
    source = np.frombuffer(raw, dtype=np.uint8)
    output = np.empty((len(width_array), group_size), dtype=np.uint8)
    for width in range(4, 9):
        indices = np.flatnonzero(width_array == width)
        bytes_per_group = group_size * width // 8
        rows_per_chunk = max(1, 8_388_608 // (group_size * width))
        byte_columns = np.arange(bytes_per_group, dtype=np.int64)
        value_shifts = (1 << np.arange(width, dtype=np.uint16))[None, None, :]
        for start in range(0, len(indices), rows_per_chunk):
            selected = indices[start : start + rows_per_chunk]
            positions = offsets[selected, None] + byte_columns[None, :]
            packed = source[positions]
            bits = np.unpackbits(packed, axis=1, bitorder="little").reshape(
                len(selected),
                group_size,
                width,
            )
            output[selected] = np.sum(
                bits.astype(np.uint16) * value_shifts,
                axis=2,
                dtype=np.uint16,
            ).astype(np.uint8)
    return output


def _fixed_deflate(raw: bytes) -> bytes:
    """Return the one permitted raw-DEFLATE encoding for ``raw``."""

    compressor = zlib.compressobj(
        level=9,
        method=zlib.DEFLATED,
        wbits=-15,
        memLevel=9,
        strategy=zlib.Z_FIXED,
    )
    return compressor.compress(raw) + compressor.flush(zlib.Z_FINISH)


def _encode_entropy(raw: bytes) -> bytes:
    payload = _fixed_deflate(raw)
    return (
        _ENTROPY_HEADER.pack(
            CREVF_ENTROPY_MAGIC,
            *CREVF_ZLIB_PROFILE,
            len(raw),
            len(payload),
        )
        + payload
    )


def _decode_entropy(
    segment: bytes,
    *,
    expected_raw_size: int,
) -> bytes:
    if len(segment) < _ENTROPY_HEADER.size:
        raise CrevfFormatError("Canonical fixed-DEFLATE segment is truncated.")
    (
        magic,
        zlib_major,
        zlib_minor,
        zlib_patch,
        zlib_build,
        raw_size,
        payload_size,
    ) = _ENTROPY_HEADER.unpack_from(segment)
    encoded_zlib_profile = (
        zlib_major,
        zlib_minor,
        zlib_patch,
        zlib_build,
    )
    if (
        magic != CREVF_ENTROPY_MAGIC
        or encoded_zlib_profile != CREVF_ZLIB_PROFILE
        or raw_size != expected_raw_size
        or payload_size != len(segment) - _ENTROPY_HEADER.size
    ):
        raise CrevfFormatError(
            "Canonical fixed-DEFLATE header differs from tensor geometry."
        )
    payload = segment[_ENTROPY_HEADER.size :]
    decoder = zlib.decompressobj(wbits=-15)
    try:
        decoded = decoder.decompress(payload, expected_raw_size + 1)
        if len(decoded) > expected_raw_size or decoder.unconsumed_tail:
            raise CrevfFormatError(
                "Canonical fixed-DEFLATE payload exceeds its decoded bound."
            )
        remaining = expected_raw_size - len(decoded)
        tail = decoder.flush(max(1, remaining + 1))
    except zlib.error as exc:
        raise CrevfFormatError("Canonical fixed-DEFLATE payload is invalid.") from exc
    decoded += tail
    if (
        len(decoded) != expected_raw_size
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise CrevfFormatError(
            "Canonical fixed-DEFLATE payload is truncated, oversized, or trailing."
        )
    if _encode_entropy(decoded) != segment:
        raise CrevfFormatError(
            "Fixed-DEFLATE payload is valid but not the canonical encoding."
        )
    return decoded


def _encode_outliers(values: Sequence[CrevfExactValue]) -> bytes:
    ordered = tuple(values)
    if not ordered:
        return b""
    previous = -1
    output = bytearray(_OUTLIER_HEADER.pack(CREVF_OUTLIER_MAGIC, len(ordered)))
    for value in ordered:
        if value.index <= previous:
            raise CrevfError("Exact BF16 outlier indices must be strictly increasing.")
        output.extend(_encode_uleb128(value.index - previous))
        output.extend(struct.pack("<H", value.bf16_bits))
        previous = value.index
    return bytes(output)


def _decode_outliers(segment: bytes) -> tuple[CrevfExactValue, ...]:
    if not segment:
        return ()
    if len(segment) < _OUTLIER_HEADER.size:
        raise CrevfFormatError("Exact BF16 outlier segment is truncated.")
    magic, count = _OUTLIER_HEADER.unpack_from(segment)
    if magic != CREVF_OUTLIER_MAGIC:
        raise CrevfFormatError("Exact BF16 outlier magic is invalid.")
    output = []
    offset = _OUTLIER_HEADER.size
    previous = -1
    for _ in range(count):
        delta, offset = _decode_uleb128(segment, offset)
        if delta < 1 or offset + 2 > len(segment):
            raise CrevfFormatError("Exact BF16 outlier entry is invalid.")
        index = previous + delta
        (bits,) = struct.unpack_from("<H", segment, offset)
        offset += 2
        output.append(CrevfExactValue(index, bits))
        previous = index
    if offset != len(segment) or _encode_outliers(output) != segment:
        raise CrevfFormatError("Exact BF16 outlier segment is noncanonical.")
    return tuple(output)


def _encode_anchors(runs: Sequence[CrevfExactRun]) -> bytes:
    ordered = tuple(runs)
    if not ordered:
        return b""
    total = sum(run.value_count for run in ordered)
    output = bytearray(_ANCHOR_HEADER.pack(CREVF_ANCHOR_MAGIC, total, len(ordered)))
    previous_end = 0
    for index, run in enumerate(ordered):
        if run.start_index < previous_end or (
            index > 0 and run.start_index == previous_end
        ):
            raise CrevfError(
                "Exact BF16 anchor runs must be sorted, disjoint, and nonadjacent."
            )
        output.extend(_encode_uleb128(run.start_index - previous_end))
        output.extend(_encode_uleb128(run.value_count))
        output.extend(run.bf16_bytes)
        previous_end = run.end_index
    return bytes(output)


def _decode_anchors(segment: bytes) -> tuple[CrevfExactRun, ...]:
    if not segment:
        return ()
    if len(segment) < _ANCHOR_HEADER.size:
        raise CrevfFormatError("Exact BF16 anchor segment is truncated.")
    magic, total, run_count = _ANCHOR_HEADER.unpack_from(segment)
    if magic != CREVF_ANCHOR_MAGIC or run_count == 0:
        raise CrevfFormatError("Exact BF16 anchor header is invalid.")
    output = []
    offset = _ANCHOR_HEADER.size
    previous_end = 0
    observed = 0
    for index in range(run_count):
        delta, offset = _decode_uleb128(segment, offset)
        length, offset = _decode_uleb128(segment, offset)
        start = previous_end + delta
        if (
            length == 0
            or (index > 0 and delta == 0)
            or offset + length * 2 > len(segment)
        ):
            raise CrevfFormatError("Exact BF16 anchor run is invalid or noncanonical.")
        bf16_bytes = segment[offset : offset + length * 2]
        offset += length * 2
        run = CrevfExactRun(start, bf16_bytes)
        output.append(run)
        observed += length
        previous_end = run.end_index
    if (
        offset != len(segment)
        or observed != total
        or _encode_anchors(output) != segment
    ):
        raise CrevfFormatError("Exact BF16 anchor segment is noncanonical.")
    return tuple(output)


def _exact_segments_overlap(
    outlier_indices: Sequence[int],
    anchors: Sequence[CrevfExactRun],
) -> bool:
    run_index = 0
    for index in outlier_indices:
        while run_index < len(anchors) and anchors[run_index].end_index <= index:
            run_index += 1
        if (
            run_index < len(anchors)
            and anchors[run_index].start_index <= index < anchors[run_index].end_index
        ):
            return True
    return False


def _validate_overlay_zero_codes(
    matrix: np.ndarray,
    widths: Sequence[int],
    group_size: int,
    outlier_indices: Sequence[int],
    anchors: Sequence[CrevfExactRun],
    *,
    format_error: bool,
) -> None:
    error_type = CrevfFormatError if format_error else CrevfError
    for index in outlier_indices:
        group_index, local_index = divmod(index, group_size)
        qmax = (1 << (widths[group_index] - 1)) - 1
        if int(matrix[group_index, local_index]) != qmax:
            raise error_type(
                "Exact overlay positions must carry the canonical zero base code."
            )
    for run in anchors:
        first_group = run.start_index // group_size
        last_group = (run.end_index - 1) // group_size
        for group_index in range(first_group, last_group + 1):
            group_start = group_index * group_size
            local_start = max(run.start_index, group_start) - group_start
            local_end = min(run.end_index, group_start + group_size) - group_start
            qmax = (1 << (widths[group_index] - 1)) - 1
            if bool((matrix[group_index, local_start:local_end] != qmax).any()):
                raise error_type(
                    "Exact overlay positions must carry the canonical zero base code."
                )


def _group_codes_empty(values: Any) -> bool:
    if isinstance(values, Tensor):
        return values.numel() == 0
    if isinstance(values, np.ndarray):
        return values.size == 0
    return len(values) == 0


def _scales_empty(values: Any) -> bool:
    if isinstance(values, Tensor):
        return values.numel() == 0
    return np.asarray(values).size == 0


def _float16_scale_bytes(scales: Any, expected_count: int) -> bytes:
    if isinstance(scales, Tensor):
        values = scales.detach().to(device="cpu", dtype=torch.float16).numpy()
    else:
        values = np.asarray(scales, dtype=np.float16)
    flat = np.ascontiguousarray(values.reshape(-1), dtype="<f2")
    if flat.size != expected_count:
        raise CrevfError("FP16 scale count differs from the quantized group count.")
    if not bool(np.isfinite(flat).all()) or bool((flat <= 0).any()):
        raise CrevfError("CREVF FP16 scales must be finite and positive.")
    return flat.tobytes(order="C")


def _canonical_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(shape)
    if not 1 <= len(result) <= _MAX_RANK or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > (1 << 63) - 1
        for value in result
    ):
        raise CrevfError(
            "Tensor shape must contain one to sixteen positive int64 dims."
        )
    numel = _product(result)
    if numel > (1 << 63) - 1:
        raise CrevfError("Tensor element count exceeds the format limit.")
    return result


def encode_crevf_tensor(spec: CrevfTensorSpec) -> bytes:
    """Encode one deterministic, self-authenticating CREVF tensor blob."""

    if not isinstance(spec, CrevfTensorSpec):
        raise CrevfError("spec must be a CrevfTensorSpec.")
    name_raw = _canonical_text(spec.name, "tensor name")
    role_raw = _canonical_text(spec.role, "tensor role")
    shape = _canonical_shape(spec.shape)
    source_numel = _product(shape)
    exact_only = spec.exact_only
    if not isinstance(exact_only, bool):
        raise CrevfError("exact_only must be bool.")
    outliers = tuple(spec.exact_outliers)
    anchors = tuple(spec.exact_anchors)
    outlier_segment = _encode_outliers(outliers)
    anchor_segment = _encode_anchors(anchors)
    outlier_indices = tuple(value.index for value in outliers)
    anchors_in_range = all(run.end_index <= source_numel for run in anchors)
    outlier_anchor_overlap = _exact_segments_overlap(outlier_indices, anchors)
    if (
        any(index >= source_numel for index in outlier_indices)
        or not anchors_in_range
        or outlier_anchor_overlap
    ):
        raise CrevfError(
            "Exact BF16 indices must be in range and outliers cannot overlap anchors."
        )
    if exact_only:
        if (
            spec.group_size != 0
            or spec.group_bit_widths
            or not _group_codes_empty(spec.group_codes)
            or not _scales_empty(spec.scales)
            or outliers
            or len(anchors) != 1
            or anchors[0].start_index != 0
            or anchors[0].end_index != source_numel
        ):
            raise CrevfError(
                "Exact-only tensors require only full, nonoverlapping BF16 anchor coverage."
            )
        group_size = 0
        group_count = 0
        width_segment = b""
        scale_segment = b""
        entropy_segment = b""
    else:
        group_size = spec.group_size
        if (
            isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or group_size < 8
            or group_size > _MAX_GROUP_SIZE
            or group_size % 8
        ):
            raise CrevfError(
                "Quantized CREVF group_size must be a multiple of eight in range."
            )
        group_count = (source_numel + group_size - 1) // group_size
        widths = tuple(spec.group_bit_widths)
        if len(widths) != group_count or any(
            isinstance(value, bool) or not isinstance(value, int) or not 4 <= value <= 8
            for value in widths
        ):
            raise CrevfError("Every quantized group requires a bit width in [4, 8].")
        width_segment = _pack_unsigned(
            np.asarray([value - 4 for value in widths], dtype=np.uint8),
            3,
        )
        scale_segment = _float16_scale_bytes(spec.scales, group_count)
        code_matrix = _group_code_matrix(
            spec.group_codes,
            group_count=group_count,
            group_size=group_size,
            widths=widths,
        )
        valid = source_numel - (group_count - 1) * group_size
        final_qmax = (1 << (widths[-1] - 1)) - 1
        if valid < group_size and bool((code_matrix[-1, valid:] != final_qmax).any()):
            raise CrevfError(
                "Final-group padded positions must use the canonical zero code."
            )
        _validate_overlay_zero_codes(
            code_matrix,
            widths,
            group_size,
            outlier_indices,
            anchors,
            format_error=False,
        )
        entropy_segment = _encode_entropy(
            _pack_group_code_stream(code_matrix, widths, group_size)
        )
    flags = (
        (_FLAG_EXACT_ONLY if exact_only else 0)
        | (_FLAG_OUTLIERS if outlier_segment else 0)
        | (_FLAG_ANCHORS if anchor_segment else 0)
    )
    header = _TENSOR_HEADER.pack(
        CREVF_TENSOR_MAGIC,
        CREVF_VERSION,
        flags,
        len(shape),
        _DTYPE_BFLOAT16,
        len(name_raw),
        len(role_raw),
        group_size,
        source_numel,
        group_count,
        len(width_segment),
        len(scale_segment),
        len(entropy_segment),
        len(outlier_segment),
        len(anchor_segment),
    )
    shape_raw = b"".join(struct.pack("<Q", value) for value in shape)
    core = (
        header
        + shape_raw
        + name_raw
        + role_raw
        + width_segment
        + scale_segment
        + entropy_segment
        + outlier_segment
        + anchor_segment
    )
    return core + _sha256(core)


def _parse_crevf_tensor(
    blob: bytes,
    *,
    max_decoded_code_bytes: int | None = _DEFAULT_MAX_TENSOR_DECODE_BYTES,
    preinflate_guard: Callable[[str, int, int, int], None] | None = None,
) -> _ParsedTensor:
    if max_decoded_code_bytes is not None:
        _require_nonnegative_int(
            max_decoded_code_bytes,
            "max_decoded_code_bytes",
        )
    if not isinstance(blob, bytes):
        raise CrevfFormatError("CREVF tensor blob must be immutable bytes.")
    if len(blob) < _TENSOR_HEADER.size + _SHA256_BYTES:
        raise CrevfFormatError("CREVF tensor blob is truncated.")
    core = blob[:-_SHA256_BYTES]
    if _sha256(core) != blob[-_SHA256_BYTES:]:
        raise CrevfFormatError("CREVF tensor SHA-256 mismatch.")
    (
        magic,
        version,
        flags,
        rank,
        dtype_code,
        name_length,
        role_length,
        group_size,
        source_numel,
        group_count,
        width_length,
        scale_length,
        entropy_length,
        outlier_length,
        anchor_length,
    ) = _TENSOR_HEADER.unpack_from(core)
    if (
        magic != CREVF_TENSOR_MAGIC
        or version != CREVF_VERSION
        or flags & ~_KNOWN_FLAGS
        or not 1 <= rank <= _MAX_RANK
        or dtype_code != _DTYPE_BFLOAT16
        or not name_length
        or not role_length
    ):
        raise CrevfFormatError("CREVF tensor header is invalid.")
    expected = (
        _TENSOR_HEADER.size
        + rank * 8
        + name_length
        + role_length
        + width_length
        + scale_length
        + entropy_length
        + outlier_length
        + anchor_length
    )
    if expected != len(core):
        raise CrevfFormatError("CREVF tensor segment lengths are inconsistent.")
    cursor = _TENSOR_HEADER.size
    shape = tuple(
        struct.unpack_from("<Q", core, cursor + index * 8)[0] for index in range(rank)
    )
    cursor += rank * 8
    try:
        canonical_shape = _canonical_shape(shape)
    except CrevfError as exc:
        raise CrevfFormatError(str(exc)) from exc
    if _product(canonical_shape) != source_numel:
        raise CrevfFormatError("CREVF tensor shape and source_numel disagree.")
    name = _decode_canonical_text(core[cursor : cursor + name_length], "tensor name")
    cursor += name_length
    role = _decode_canonical_text(core[cursor : cursor + role_length], "tensor role")
    cursor += role_length
    width_segment = core[cursor : cursor + width_length]
    cursor += width_length
    scale_segment = core[cursor : cursor + scale_length]
    cursor += scale_length
    entropy_segment = core[cursor : cursor + entropy_length]
    cursor += entropy_length
    outlier_segment = core[cursor : cursor + outlier_length]
    cursor += outlier_length
    anchor_segment = core[cursor : cursor + anchor_length]
    cursor += anchor_length
    if cursor != len(core):
        raise CrevfFormatError("CREVF tensor has unclaimed bytes.")
    exact_only = bool(flags & _FLAG_EXACT_ONLY)
    if bool(flags & _FLAG_OUTLIERS) != bool(outlier_segment) or bool(
        flags & _FLAG_ANCHORS
    ) != bool(anchor_segment):
        raise CrevfFormatError("CREVF exact-segment flags disagree with their bytes.")
    outliers = _decode_outliers(outlier_segment)
    anchors = _decode_anchors(anchor_segment)
    outlier_indices = tuple(value.index for value in outliers)
    anchors_in_range = all(run.end_index <= source_numel for run in anchors)
    outlier_anchor_overlap = _exact_segments_overlap(outlier_indices, anchors)
    if (
        any(index >= source_numel for index in outlier_indices)
        or not anchors_in_range
        or outlier_anchor_overlap
    ):
        raise CrevfFormatError("CREVF exact BF16 indices overlap or exceed the tensor.")
    persistent_encoded_bytes = (
        len(width_segment)
        + len(scale_segment)
        + len(entropy_segment)
        + len(outlier_segment)
        + len(anchor_segment)
    )
    if exact_only:
        if (
            group_size
            or group_count
            or width_segment
            or scale_segment
            or entropy_segment
            or outliers
            or len(anchors) != 1
            or anchors[0].start_index != 0
            or anchors[0].end_index != source_numel
        ):
            raise CrevfFormatError("Exact-only CREVF tensor coverage is inconsistent.")
        widths: tuple[int, ...] = ()
        decoded_codes = b""
        expected_code_bytes = 0
        decoded_full_state_bytes = (
            len(width_segment)
            + len(scale_segment)
            + len(outlier_segment)
            + len(anchor_segment)
        )
        if preinflate_guard is not None:
            preinflate_guard(
                name,
                persistent_encoded_bytes,
                expected_code_bytes,
                decoded_full_state_bytes,
            )
    else:
        if (
            group_size < 8
            or group_size > _MAX_GROUP_SIZE
            or group_size % 8
            or group_count != (source_numel + group_size - 1) // group_size
            or width_length != (group_count * 3 + 7) // 8
            or scale_length != group_count * 2
            or not entropy_segment
        ):
            raise CrevfFormatError("Quantized CREVF tensor geometry is inconsistent.")
        width_codes = _unpack_unsigned(
            width_segment,
            value_count=group_count,
            bit_width=3,
        )
        if bool((width_codes > 4).any()):
            raise CrevfFormatError("A CREVF group width lies outside [4, 8].")
        widths = tuple(int(value) + 4 for value in width_codes)
        scales = np.frombuffer(scale_segment, dtype="<f2")
        if (
            scales.size != group_count
            or not bool(np.isfinite(scales).all())
            or bool((scales <= 0).any())
        ):
            raise CrevfFormatError("CREVF FP16 scales are invalid.")
        expected_code_bytes = sum(group_size * width // 8 for width in widths)
        if (
            max_decoded_code_bytes is not None
            and expected_code_bytes > max_decoded_code_bytes
        ):
            raise CrevfBudgetError(
                f"CREVF tensor {name!r} declares {expected_code_bytes} decoded "
                "code bytes, exceeding the pre-inflate "
                f"{max_decoded_code_bytes}-byte tensor ceiling."
            )
        decoded_full_state_bytes = (
            len(width_segment)
            + len(scale_segment)
            + expected_code_bytes
            + len(outlier_segment)
            + len(anchor_segment)
        )
        if preinflate_guard is not None:
            preinflate_guard(
                name,
                persistent_encoded_bytes,
                expected_code_bytes,
                decoded_full_state_bytes,
            )
        decoded_codes = _decode_entropy(
            entropy_segment,
            expected_raw_size=expected_code_bytes,
        )
        code_matrix = _unpack_group_code_stream(
            decoded_codes,
            widths,
            group_size,
        )
        row_maxima = code_matrix.max(axis=1)
        allowed = np.asarray(
            [2 * ((1 << (width - 1)) - 1) for width in widths],
            dtype=np.uint16,
        )
        if bool((row_maxima.astype(np.uint16) > allowed).any()):
            raise CrevfFormatError("A CREVF symmetric code exceeds its width.")
        valid = source_numel - (group_count - 1) * group_size
        final_qmax = (1 << (widths[-1] - 1)) - 1
        if valid < group_size and bool((code_matrix[-1, valid:] != final_qmax).any()):
            raise CrevfFormatError("CREVF final-group padding is noncanonical.")
        _validate_overlay_zero_codes(
            code_matrix,
            widths,
            group_size,
            outlier_indices,
            anchors,
            format_error=True,
        )
    anchor_count = sum(run.value_count for run in anchors)
    return _ParsedTensor(
        metadata=CrevfTensorMetadata(
            name=name,
            role=role,
            shape=canonical_shape,
            source_numel=source_numel,
            group_size=group_size,
            group_count=group_count,
            group_bit_widths=widths,
            exact_only=exact_only,
            exact_outlier_count=len(outliers),
            exact_anchor_count=anchor_count,
            tensor_blob_bytes=len(blob),
            entropy_segment_bytes=len(entropy_segment),
            decoded_code_bytes=expected_code_bytes,
            persistent_encoded_bytes=persistent_encoded_bytes,
            decoded_full_state_bytes=decoded_full_state_bytes,
            decoded_code_transient_bytes=expected_code_bytes,
        ),
        width_segment=width_segment,
        scale_segment=scale_segment,
        entropy_segment=entropy_segment,
        outlier_segment=outlier_segment,
        anchor_segment=anchor_segment,
    )


def inspect_crevf_tensor(blob: bytes) -> CrevfTensorMetadata:
    """Authenticate and inspect one tensor without retaining dense values."""

    return _parse_crevf_tensor(blob).metadata


def encode_crevf_auxiliary(spec: CrevfAuxiliarySpec) -> bytes:
    """Encode one bounded opaque auxiliary payload with its own SHA-256 seal."""

    if not isinstance(spec, CrevfAuxiliarySpec):
        raise CrevfError("spec must be a CrevfAuxiliarySpec.")
    name_raw = _canonical_text(spec.name, "auxiliary name")
    kind_raw = _canonical_text(spec.kind, "auxiliary kind")
    payload = spec.payload
    if len(payload) > _DEFAULT_MAX_AUXILIARY_BYTES:
        raise CrevfBudgetError(
            "CREVF auxiliary payload exceeds the default per-record byte bound."
        )
    header = _AUXILIARY_HEADER.pack(
        CREVF_AUXILIARY_MAGIC,
        CREVF_VERSION,
        0,
        len(name_raw),
        len(kind_raw),
        len(payload),
    )
    core = header + name_raw + kind_raw + payload
    return core + _sha256(core)


def _parse_crevf_auxiliary(
    blob: bytes,
    *,
    max_payload_bytes: int | None = _DEFAULT_MAX_AUXILIARY_BYTES,
    preflight_guard: Callable[[str, int, int, int], None] | None = None,
) -> _ParsedAuxiliary:
    if not isinstance(blob, bytes):
        raise CrevfFormatError("CREVF auxiliary blob must be immutable bytes.")
    if max_payload_bytes is not None:
        _require_nonnegative_int(max_payload_bytes, "max_payload_bytes")
    minimum = _AUXILIARY_HEADER.size + _SHA256_BYTES
    if len(blob) < minimum:
        raise CrevfFormatError("CREVF auxiliary blob is truncated.")
    core = blob[:-_SHA256_BYTES]
    if _sha256(core) != blob[-_SHA256_BYTES:]:
        raise CrevfFormatError("CREVF auxiliary SHA-256 mismatch.")
    (
        magic,
        version,
        flags,
        name_length,
        kind_length,
        payload_length,
    ) = _AUXILIARY_HEADER.unpack_from(core)
    if (
        magic != CREVF_AUXILIARY_MAGIC
        or version != CREVF_VERSION
        or flags != 0
        or payload_length == 0
    ):
        raise CrevfFormatError("CREVF auxiliary header is invalid.")
    expected = (
        _AUXILIARY_HEADER.size
        + name_length
        + kind_length
        + payload_length
    )
    if expected != len(core):
        raise CrevfFormatError("CREVF auxiliary byte ranges are invalid.")
    if max_payload_bytes is not None and payload_length > max_payload_bytes:
        raise CrevfBudgetError(
            "CREVF auxiliary payload exceeds its authenticated decode bound."
        )
    cursor = _AUXILIARY_HEADER.size
    name = _decode_canonical_text(
        core[cursor : cursor + name_length],
        "auxiliary name",
    )
    cursor += name_length
    kind = _decode_canonical_text(
        core[cursor : cursor + kind_length],
        "auxiliary kind",
    )
    cursor += kind_length
    metadata = CrevfAuxiliaryMetadata(
        name=name,
        kind=kind,
        payload_bytes=payload_length,
        auxiliary_blob_bytes=len(blob),
        persistent_encoded_bytes=payload_length,
        decoded_full_state_bytes=payload_length,
    )
    if preflight_guard is not None:
        preflight_guard(
            name,
            metadata.persistent_encoded_bytes,
            metadata.decoded_code_transient_bytes,
            metadata.decoded_full_state_bytes,
        )
    payload = core[cursor:]
    parsed = _ParsedAuxiliary(metadata=metadata, payload=payload)
    if encode_crevf_auxiliary(CrevfAuxiliarySpec(name, kind, payload)) != blob:
        raise CrevfFormatError("CREVF auxiliary blob is noncanonical.")
    return parsed


def inspect_crevf_auxiliary(blob: bytes) -> CrevfAuxiliaryMetadata:
    """Authenticate an auxiliary blob and report exact resource accounting."""

    return _parse_crevf_auxiliary(blob).metadata


def _uint8_tensor(raw: bytes) -> Tensor:
    if not raw:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def _buffer_bytes(value: Tensor) -> bytes:
    if value.numel() == 0:
        return b""
    return (
        value.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
        .numpy()
        .tobytes(order="C")
    )


def _torch_unpack_unsigned(packed: Tensor, value_count: int, bit_width: int) -> Tensor:
    expected = (value_count * bit_width + 7) // 8
    if packed.dtype != torch.uint8 or packed.ndim != 1 or packed.numel() != expected:
        raise CrevfFormatError("Runtime packed code buffer is inconsistent.")
    if value_count == 0:
        return torch.empty(0, dtype=torch.uint8, device=packed.device)
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(1), shifts),
        1,
    ).reshape(-1)[: value_count * bit_width]
    value_shifts = torch.arange(
        bit_width,
        dtype=torch.int16,
        device=packed.device,
    )
    return torch.sum(
        bits.reshape(value_count, bit_width).to(torch.int16)
        * torch.bitwise_left_shift(torch.ones_like(value_shifts), value_shifts),
        dim=1,
    ).to(torch.uint8)


def _bf16_tensor_from_bits(bits: Sequence[int], device: torch.device) -> Tensor:
    if not bits:
        return torch.empty(0, dtype=torch.bfloat16, device=device)
    words = np.asarray(bits, dtype="<u2").astype(np.uint16, copy=True)
    uint16 = torch.from_numpy(words)
    return uint16.view(torch.bfloat16).to(device)


def _bf16_tensor_from_bytes(raw: bytes, device: torch.device) -> Tensor:
    if not raw:
        return torch.empty(0, dtype=torch.bfloat16, device=device)
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint16, copy=True)
    uint16 = torch.from_numpy(words)
    return uint16.view(torch.bfloat16).to(device)


_LAYER_PATH = re.compile(
    r"^(?P<prefix>.*?(?:layers?|blocks?|stages?|h)\.\d+)(?:\.|$)",
    flags=re.IGNORECASE,
)


def crevf_decode_transient_group(tensor_name: str) -> str:
    """Return the conservative layer-like unit decoded together at runtime."""

    match = _LAYER_PATH.match(tensor_name)
    return match.group("prefix") if match is not None else tensor_name


class CrevfPackedTensor(nn.Module):
    """One entropy-coded tensor store with transient code/dense reconstruction."""

    def __init__(self, parsed: _ParsedTensor) -> None:
        super().__init__()
        metadata = parsed.metadata
        self.name = metadata.name
        self.role = metadata.role
        self.shape = metadata.shape
        self.source_numel = metadata.source_numel
        self.group_size = metadata.group_size
        self.group_count = metadata.group_count
        self.exact_only = metadata.exact_only
        self.decoded_code_bytes = metadata.decoded_code_bytes
        self._persistent_encoded_bytes = metadata.persistent_encoded_bytes
        self._decoded_full_state_bytes = metadata.decoded_full_state_bytes
        self.register_buffer(
            "packed_group_widths",
            _uint8_tensor(parsed.width_segment),
            persistent=False,
        )
        self.register_buffer(
            "entropy_segment",
            _uint8_tensor(parsed.entropy_segment),
            persistent=False,
        )
        scales = (
            torch.from_numpy(np.frombuffer(parsed.scale_segment, dtype="<f2").copy())
            if parsed.scale_segment
            else torch.empty(0, dtype=torch.float16)
        )
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer(
            "exact_outlier_segment",
            _uint8_tensor(parsed.outlier_segment),
            persistent=False,
        )
        self.register_buffer(
            "exact_anchor_segment",
            _uint8_tensor(parsed.anchor_segment),
            persistent=False,
        )
        self.register_buffer(
            "_device_sentinel",
            torch.empty(0, dtype=torch.uint8),
            persistent=False,
        )

    @property
    def persistent_encoded_bytes(self) -> int:
        observed = sum(
            int(value.numel()) * int(value.element_size())
            for value in (
                self.packed_group_widths,
                self.entropy_segment,
                self.scales,
                self.exact_outlier_segment,
                self.exact_anchor_segment,
            )
        )
        if observed != self._persistent_encoded_bytes:
            raise CrevfFormatError(
                "Runtime persistent encoded bytes diverged from tensor metadata."
            )
        return observed

    @property
    def decoded_full_state_bytes(self) -> int:
        return self._decoded_full_state_bytes

    @property
    def decoded_code_transient_bytes(self) -> int:
        return self.decoded_code_bytes

    @property
    def packed_runtime_bytes(self) -> int:
        """Backward-compatible alias for persistent encoded bytes."""

        return self.persistent_encoded_bytes

    @property
    def device(self) -> torch.device:
        return self._device_sentinel.device

    def dequantize(self) -> Tensor:
        """Materialize BF16 values only for the duration chosen by the caller."""

        device = self.device
        if self.exact_only:
            dense = torch.empty(self.source_numel, dtype=torch.bfloat16, device=device)
        else:
            decoded_codes = _decode_entropy(
                _buffer_bytes(self.entropy_segment),
                expected_raw_size=self.decoded_code_bytes,
            )
            packed_codes = _uint8_tensor(decoded_codes).to(device=device)
            del decoded_codes
            width_codes = _torch_unpack_unsigned(
                self.packed_group_widths,
                self.group_count,
                3,
            )
            group_bit_widths = width_codes.to(torch.int64) + 4
            byte_counts = group_bit_widths * self.group_size // 8
            code_offsets = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(byte_counts, dim=0),
                )
            )
            if int(code_offsets[-1]) != packed_codes.numel():
                raise CrevfFormatError("Runtime CREVF code buffer has trailing bytes.")
            dense_groups = torch.empty(
                self.group_count,
                self.group_size,
                dtype=torch.float32,
                device=device,
            )
            for width in range(4, 9):
                indices = torch.nonzero(
                    group_bit_widths == width,
                    as_tuple=False,
                ).reshape(-1)
                bytes_per_group = self.group_size * width // 8
                rows_per_chunk = max(
                    1,
                    8_388_608 // (self.group_size * width),
                )
                byte_columns = torch.arange(
                    bytes_per_group,
                    dtype=torch.int64,
                    device=device,
                )
                qmax = (1 << (width - 1)) - 1
                for start in range(0, int(indices.numel()), rows_per_chunk):
                    selected = indices[start : start + rows_per_chunk]
                    positions = (
                        code_offsets.index_select(0, selected)[:, None]
                        + byte_columns[None, :]
                    )
                    packed = packed_codes[positions].reshape(-1)
                    codes = _torch_unpack_unsigned(
                        packed,
                        int(selected.numel()) * self.group_size,
                        width,
                    ).reshape(int(selected.numel()), self.group_size)
                    selected_values = (codes.to(torch.int16) - qmax).to(torch.float32)
                    selected_values *= self.scales.index_select(
                        0,
                        selected,
                    ).to(torch.float32)[:, None]
                    dense_groups.index_copy_(0, selected, selected_values)
            dense = dense_groups.reshape(-1)[: self.source_numel].to(torch.bfloat16)
            del packed_codes
        outliers = _decode_outliers(_buffer_bytes(self.exact_outlier_segment))
        if outliers:
            indices = torch.tensor(
                [value.index for value in outliers],
                dtype=torch.long,
                device=device,
            )
            values = _bf16_tensor_from_bits(
                [value.bf16_bits for value in outliers],
                device,
            )
            dense.index_copy_(0, indices, values)
        anchors = _decode_anchors(_buffer_bytes(self.exact_anchor_segment))
        for run in anchors:
            dense[run.start_index : run.end_index].copy_(
                _bf16_tensor_from_bytes(run.bf16_bytes, device)
            )
        return dense.reshape(self.shape)


class CrevfPackedAuxiliary(nn.Module):
    """One authenticated byte payload stored as a nonpersistent uint8 buffer."""

    def __init__(self, parsed: _ParsedAuxiliary) -> None:
        super().__init__()
        self.name = parsed.metadata.name
        self.kind = parsed.metadata.kind
        self._persistent_encoded_bytes = parsed.metadata.persistent_encoded_bytes
        self._decoded_full_state_bytes = parsed.metadata.decoded_full_state_bytes
        self.register_buffer(
            "encoded_payload",
            _uint8_tensor(parsed.payload),
            persistent=False,
        )

    @property
    def persistent_encoded_bytes(self) -> int:
        observed = int(self.encoded_payload.numel()) * int(
            self.encoded_payload.element_size()
        )
        if observed != self._persistent_encoded_bytes:
            raise CrevfFormatError(
                "Runtime auxiliary bytes diverged from authenticated metadata."
            )
        return observed

    @property
    def decoded_full_state_bytes(self) -> int:
        return self._decoded_full_state_bytes

    @property
    def decoded_code_transient_bytes(self) -> int:
        return 0

    def payload_bytes(self) -> bytes:
        return _buffer_bytes(self.encoded_payload)


class CrevfPackedRuntime(nn.Module):
    """A canonical collection of packed tensor stores with zero Parameters."""

    def __init__(
        self,
        stores: Sequence[CrevfPackedTensor],
        *,
        auxiliaries: Sequence[CrevfPackedAuxiliary] = (),
        physical_artifact_bytes: int,
        artifact_sha256: str,
    ) -> None:
        super().__init__()
        ordered = tuple(stores)
        names = tuple(store.name for store in ordered)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise CrevfFormatError(
                "Runtime tensor stores must have unique sorted names."
            )
        self.tensor_names = names
        self.physical_artifact_bytes = physical_artifact_bytes
        self.artifact_sha256 = artifact_sha256
        self.stores = nn.ModuleList(ordered)
        self._name_to_index = {name: index for index, name in enumerate(names)}
        ordered_auxiliaries = tuple(auxiliaries)
        auxiliary_names = tuple(value.name for value in ordered_auxiliaries)
        if (
            auxiliary_names != tuple(sorted(auxiliary_names))
            or len(auxiliary_names) != len(set(auxiliary_names))
            or set(names).intersection(auxiliary_names)
        ):
            raise CrevfFormatError(
                "Runtime auxiliary stores must have globally unique sorted names."
            )
        self.auxiliary_names = auxiliary_names
        self.auxiliaries = nn.ModuleList(ordered_auxiliaries)
        self._auxiliary_name_to_index = {
            name: index for index, name in enumerate(auxiliary_names)
        }

    def tensor(self, name: str) -> CrevfPackedTensor:
        try:
            return self.stores[self._name_to_index[name]]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"Unknown CREVF tensor {name!r}.") from exc

    def auxiliary(self, name: str) -> CrevfPackedAuxiliary:
        try:
            return self.auxiliaries[self._auxiliary_name_to_index[name]]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"Unknown CREVF auxiliary {name!r}.") from exc

    @property
    def persistent_encoded_bytes(self) -> int:
        return sum(
            store.persistent_encoded_bytes
            for store in (*tuple(self.stores), *tuple(self.auxiliaries))
        )

    @property
    def decoded_full_state_bytes(self) -> int:
        return sum(
            store.decoded_full_state_bytes
            for store in (*tuple(self.stores), *tuple(self.auxiliaries))
        )

    @property
    def maximum_tensor_decode_transient_bytes(self) -> int:
        return max(
            (store.decoded_code_transient_bytes for store in self.stores),
            default=0,
        )

    @property
    def layer_decode_transient_bytes(self) -> tuple[tuple[str, int], ...]:
        grouped: dict[str, int] = {}
        for store in self.stores:
            if store.decoded_code_transient_bytes == 0:
                continue
            group = crevf_decode_transient_group(store.name)
            grouped[group] = grouped.get(group, 0) + store.decoded_code_transient_bytes
        return tuple(sorted(grouped.items()))

    @property
    def maximum_layer_decode_transient_bytes(self) -> int:
        return max(
            (value for _group, value in self.layer_decode_transient_bytes), default=0
        )

    @property
    def packed_runtime_bytes(self) -> int:
        """Backward-compatible alias for persistent encoded bytes."""

        return self.persistent_encoded_bytes


def decode_crevf_tensor(
    blob: bytes,
    *,
    max_decoded_code_bytes: int | None = _DEFAULT_MAX_TENSOR_DECODE_BYTES,
) -> CrevfPackedTensor:
    """Decode one authenticated tensor into an entropy-coded runtime store."""

    return CrevfPackedTensor(
        _parse_crevf_tensor(
            blob,
            max_decoded_code_bytes=max_decoded_code_bytes,
        )
    )


def encode_crevf_artifact(
    tensor_blobs: Iterable[bytes],
    *,
    auxiliary_blobs: Iterable[bytes] = (),
) -> bytes:
    """Assemble canonical sorted tensor and optional auxiliary records."""

    parsed = [
        (_parse_crevf_tensor(blob).metadata.name, CREVF_TENSOR_MAGIC, blob)
        for blob in tensor_blobs
    ]
    parsed.extend(
        (
            _parse_crevf_auxiliary(blob).metadata.name,
            CREVF_AUXILIARY_MAGIC,
            blob,
        )
        for blob in auxiliary_blobs
    )
    parsed.sort(key=lambda value: value[0])
    names = [name for name, _magic, _blob in parsed]
    if len(names) != len(set(names)):
        raise CrevfError("A CREVF artifact cannot contain duplicate record names.")
    records = bytearray()
    for _name, _magic, blob in parsed:
        records.extend(_RECORD_ENVELOPE.pack(len(blob), _sha256(blob)))
        records.extend(blob)
    flags = (
        _FILE_FLAG_AUXILIARY_RECORDS
        if any(magic == CREVF_AUXILIARY_MAGIC for _name, magic, _blob in parsed)
        else 0
    )
    header = _FILE_HEADER.pack(
        CREVF_FILE_MAGIC,
        CREVF_VERSION,
        flags,
        len(parsed),
        len(records),
    )
    core = header + bytes(records)
    return core + _sha256(core)


def decode_crevf_artifact(
    artifact: bytes,
    *,
    max_tensor_decode_bytes: int | None = _DEFAULT_MAX_TENSOR_DECODE_BYTES,
    max_auxiliary_payload_bytes: int | None = _DEFAULT_MAX_AUXILIARY_BYTES,
    max_persistent_encoded_bytes: int | None = None,
    max_decoded_full_state_bytes: int | None = None,
    max_layer_transient_bytes: int | None = None,
) -> CrevfPackedRuntime:
    """Authenticate an artifact and build a zero-Parameter encoded runtime.

    Decoded-code limits are checked from authenticated tensor geometry before
    DEFLATE inflation. The full decoded-state value is diagnostic unless its
    optional ceiling is supplied; decoded streams are never retained together.
    """

    if not isinstance(artifact, bytes):
        raise CrevfFormatError("CREVF artifact must be immutable bytes.")
    for value, name in (
        (max_tensor_decode_bytes, "max_tensor_decode_bytes"),
        (max_auxiliary_payload_bytes, "max_auxiliary_payload_bytes"),
        (max_persistent_encoded_bytes, "max_persistent_encoded_bytes"),
        (max_decoded_full_state_bytes, "max_decoded_full_state_bytes"),
        (max_layer_transient_bytes, "max_layer_transient_bytes"),
    ):
        if value is not None:
            _require_nonnegative_int(value, name)
    if len(artifact) < CREVF_ARTIFACT_FIXED_BYTES:
        raise CrevfFormatError("CREVF artifact is truncated.")
    core = artifact[:-_SHA256_BYTES]
    if _sha256(core) != artifact[-_SHA256_BYTES:]:
        raise CrevfFormatError("CREVF artifact SHA-256 mismatch.")
    magic, version, flags, tensor_count, records_bytes = _FILE_HEADER.unpack_from(core)
    if (
        magic != CREVF_FILE_MAGIC
        or version != CREVF_VERSION
        or flags & ~_KNOWN_FILE_FLAGS
        or records_bytes != len(core) - _FILE_HEADER.size
    ):
        raise CrevfFormatError("CREVF artifact header is invalid.")
    stores = []
    auxiliary_stores = []
    persistent_encoded_bytes = 0
    decoded_full_state_bytes = 0
    layer_transient: dict[str, int] = {}

    def preinflate_guard(
        tensor_name: str,
        tensor_persistent_encoded_bytes: int,
        decoded_code_bytes: int,
        tensor_decoded_full_state_bytes: int,
    ) -> None:
        nonlocal decoded_full_state_bytes, persistent_encoded_bytes
        candidate_persistent = (
            persistent_encoded_bytes + tensor_persistent_encoded_bytes
        )
        if (
            max_persistent_encoded_bytes is not None
            and candidate_persistent > max_persistent_encoded_bytes
        ):
            raise CrevfBudgetError(
                "CREVF persistent encoded runtime would require "
                f"{candidate_persistent} bytes, exceeding the pre-inflate "
                f"{max_persistent_encoded_bytes}-byte ceiling."
            )
        candidate_full_state = (
            decoded_full_state_bytes + tensor_decoded_full_state_bytes
        )
        if (
            max_decoded_full_state_bytes is not None
            and candidate_full_state > max_decoded_full_state_bytes
        ):
            raise CrevfBudgetError(
                "CREVF decoded full-state diagnostic would require "
                f"{candidate_full_state} bytes, exceeding the pre-inflate "
                f"{max_decoded_full_state_bytes}-byte ceiling."
            )
        if decoded_code_bytes:
            group = crevf_decode_transient_group(tensor_name)
            candidate_layer = layer_transient.get(group, 0) + decoded_code_bytes
            if (
                max_layer_transient_bytes is not None
                and candidate_layer > max_layer_transient_bytes
            ):
                raise CrevfBudgetError(
                    f"CREVF layer group {group!r} would decode {candidate_layer} "
                    "code bytes, exceeding the pre-inflate "
                    f"{max_layer_transient_bytes}-byte layer ceiling."
                )
            layer_transient[group] = candidate_layer
        persistent_encoded_bytes = candidate_persistent
        decoded_full_state_bytes = candidate_full_state

    cursor = _FILE_HEADER.size
    previous_name: str | None = None
    saw_auxiliary = False
    for _ in range(tensor_count):
        if cursor + _RECORD_ENVELOPE.size > len(core):
            raise CrevfFormatError("CREVF record envelope is truncated.")
        blob_size, digest = _RECORD_ENVELOPE.unpack_from(core, cursor)
        cursor += _RECORD_ENVELOPE.size
        minimum_record_bytes = (
            min(_TENSOR_HEADER.size, _AUXILIARY_HEADER.size) + _SHA256_BYTES
        )
        if (
            blob_size < minimum_record_bytes
            or cursor + blob_size > len(core)
        ):
            raise CrevfFormatError("CREVF record byte range is invalid.")
        blob = core[cursor : cursor + blob_size]
        cursor += blob_size
        if _sha256(blob) != digest:
            raise CrevfFormatError("CREVF record SHA-256 mismatch.")
        record_magic = blob[:4]
        if record_magic == CREVF_TENSOR_MAGIC:
            parsed_tensor = _parse_crevf_tensor(
                blob,
                max_decoded_code_bytes=max_tensor_decode_bytes,
                preinflate_guard=preinflate_guard,
            )
            record_name = parsed_tensor.metadata.name
            stores.append(CrevfPackedTensor(parsed_tensor))
        elif record_magic == CREVF_AUXILIARY_MAGIC:
            if not flags & _FILE_FLAG_AUXILIARY_RECORDS:
                raise CrevfFormatError(
                    "CREVF auxiliary record is not declared by the file header."
                )
            parsed_auxiliary = _parse_crevf_auxiliary(
                blob,
                max_payload_bytes=max_auxiliary_payload_bytes,
                preflight_guard=preinflate_guard,
            )
            record_name = parsed_auxiliary.metadata.name
            auxiliary_stores.append(CrevfPackedAuxiliary(parsed_auxiliary))
            saw_auxiliary = True
        else:
            raise CrevfFormatError("CREVF artifact record magic is invalid.")
        if previous_name is not None and record_name <= previous_name:
            raise CrevfFormatError(
                "CREVF records are not uniquely sorted by record name."
            )
        previous_name = record_name
    if cursor != len(core):
        raise CrevfFormatError("CREVF artifact has trailing or unclaimed record bytes.")
    if bool(flags & _FILE_FLAG_AUXILIARY_RECORDS) != saw_auxiliary:
        raise CrevfFormatError(
            "CREVF auxiliary file flag does not match the record inventory."
        )
    runtime = CrevfPackedRuntime(
        stores,
        auxiliaries=auxiliary_stores,
        physical_artifact_bytes=len(artifact),
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
    )
    if tuple(runtime.parameters()):
        raise CrevfFormatError("CREVF packed runtime unexpectedly owns Parameters.")
    if runtime.persistent_encoded_bytes != persistent_encoded_bytes:
        raise CrevfFormatError(
            "CREVF persistent encoded accounting diverged during runtime load."
        )
    if runtime.decoded_full_state_bytes != decoded_full_state_bytes:
        raise CrevfFormatError(
            "CREVF decoded full-state accounting diverged during runtime load."
        )
    if runtime.layer_decode_transient_bytes != tuple(sorted(layer_transient.items())):
        raise CrevfFormatError(
            "CREVF layer transient accounting diverged during runtime load."
        )
    return runtime


def account_crevf_artifact(
    artifact: bytes,
    *,
    dense_reference_bytes: int,
    target_reduction: float,
    shared_checkpoint_bytes: int = 0,
    shared_persistent_encoded_bytes: int = 0,
    shared_decoded_full_state_bytes: int = 0,
    must_beat_physical_bytes: int | None = None,
    max_persistent_encoded_bytes: int | None = None,
    max_decoded_full_state_bytes: int | None = None,
    max_layer_transient_bytes: int | None = None,
    max_tensor_decode_bytes: int | None = _DEFAULT_MAX_TENSOR_DECODE_BYTES,
    shared_runtime_bytes: int | None = None,
    max_runtime_bytes: int | None = None,
    require_within_ceiling: bool = True,
) -> CrevfByteAccounting:
    """Authenticate and account all four independent CREVF byte budgets.

    ``shared_runtime_bytes`` and ``max_runtime_bytes`` remain compatibility
    aliases for the persistent-encoded fields. New callers should use the
    explicit names.
    """

    _require_nonnegative_int(shared_checkpoint_bytes, "shared_checkpoint_bytes")
    _require_nonnegative_int(
        shared_persistent_encoded_bytes,
        "shared_persistent_encoded_bytes",
    )
    _require_nonnegative_int(
        shared_decoded_full_state_bytes,
        "shared_decoded_full_state_bytes",
    )
    if shared_runtime_bytes is not None:
        _require_nonnegative_int(shared_runtime_bytes, "shared_runtime_bytes")
        if (
            shared_persistent_encoded_bytes
            and shared_persistent_encoded_bytes != shared_runtime_bytes
        ):
            raise CrevfError(
                "shared_runtime_bytes conflicts with shared_persistent_encoded_bytes."
            )
        shared_persistent_encoded_bytes = shared_runtime_bytes
    if max_runtime_bytes is not None:
        _require_nonnegative_int(max_runtime_bytes, "max_runtime_bytes")
        if (
            max_persistent_encoded_bytes is not None
            and max_persistent_encoded_bytes != max_runtime_bytes
        ):
            raise CrevfError(
                "max_runtime_bytes conflicts with max_persistent_encoded_bytes."
            )
        max_persistent_encoded_bytes = max_runtime_bytes
    if must_beat_physical_bytes is not None:
        _require_nonnegative_int(
            must_beat_physical_bytes,
            "must_beat_physical_bytes",
        )
        if must_beat_physical_bytes == 0:
            raise CrevfError("must_beat_physical_bytes must be positive when set.")
    for value, name in (
        (max_persistent_encoded_bytes, "max_persistent_encoded_bytes"),
        (max_decoded_full_state_bytes, "max_decoded_full_state_bytes"),
        (max_layer_transient_bytes, "max_layer_transient_bytes"),
        (max_tensor_decode_bytes, "max_tensor_decode_bytes"),
    ):
        if value is not None:
            _require_nonnegative_int(value, name)
    if (
        max_persistent_encoded_bytes is not None
        and shared_persistent_encoded_bytes > max_persistent_encoded_bytes
    ):
        raise CrevfBudgetError(
            "Shared persistent encoded bytes already exceed their ceiling."
        )
    if (
        max_decoded_full_state_bytes is not None
        and shared_decoded_full_state_bytes > max_decoded_full_state_bytes
    ):
        raise CrevfBudgetError(
            "Shared decoded full-state bytes already exceed their ceiling."
        )
    target = CrevfReductionTarget.from_reduction(target_reduction)
    ceiling = target.checkpoint_ceiling(dense_reference_bytes)
    effective_ceiling = min(
        ceiling,
        (
            must_beat_physical_bytes - 1
            if must_beat_physical_bytes is not None
            else ceiling
        ),
    )
    checkpoint_bytes = len(artifact) + shared_checkpoint_bytes
    if require_within_ceiling and checkpoint_bytes > effective_ceiling:
        raise CrevfBudgetError(
            f"CREVF checkpoint requires {checkpoint_bytes} bytes, exceeding "
            f"the exact effective {effective_ceiling}-byte ceiling."
        )
    artifact_decoded_ceiling = (
        None
        if max_decoded_full_state_bytes is None
        else max_decoded_full_state_bytes - shared_decoded_full_state_bytes
    )
    artifact_persistent_ceiling = (
        None
        if max_persistent_encoded_bytes is None
        else max_persistent_encoded_bytes - shared_persistent_encoded_bytes
    )
    runtime = decode_crevf_artifact(
        artifact,
        max_tensor_decode_bytes=max_tensor_decode_bytes,
        max_persistent_encoded_bytes=artifact_persistent_ceiling,
        max_decoded_full_state_bytes=artifact_decoded_ceiling,
        max_layer_transient_bytes=max_layer_transient_bytes,
    )
    total_persistent_encoded_bytes = (
        runtime.persistent_encoded_bytes + shared_persistent_encoded_bytes
    )
    if (
        max_persistent_encoded_bytes is not None
        and total_persistent_encoded_bytes > max_persistent_encoded_bytes
    ):
        raise CrevfBudgetError(
            "CREVF persistent encoded runtime requires "
            f"{total_persistent_encoded_bytes} bytes, exceeding the independent "
            f"{max_persistent_encoded_bytes}-byte persistent ceiling."
        )
    total_decoded_full_state_bytes = (
        runtime.decoded_full_state_bytes + shared_decoded_full_state_bytes
    )
    return CrevfByteAccounting(
        dense_reference_bytes=dense_reference_bytes,
        reduction_ppm=target.reduction_ppm,
        target_checkpoint_bytes=ceiling,
        effective_checkpoint_ceiling_bytes=effective_ceiling,
        must_beat_physical_bytes=must_beat_physical_bytes,
        artifact_physical_bytes=len(artifact),
        shared_checkpoint_bytes=shared_checkpoint_bytes,
        checkpoint_physical_bytes=checkpoint_bytes,
        persistent_encoded_bytes=runtime.persistent_encoded_bytes,
        shared_persistent_encoded_bytes=shared_persistent_encoded_bytes,
        total_persistent_encoded_bytes=total_persistent_encoded_bytes,
        decoded_full_state_bytes=runtime.decoded_full_state_bytes,
        shared_decoded_full_state_bytes=shared_decoded_full_state_bytes,
        total_decoded_full_state_bytes=total_decoded_full_state_bytes,
        maximum_tensor_decode_transient_bytes=(
            runtime.maximum_tensor_decode_transient_bytes
        ),
        maximum_layer_decode_transient_bytes=(
            runtime.maximum_layer_decode_transient_bytes
        ),
        layer_decode_transient_bytes=runtime.layer_decode_transient_bytes,
        headroom_bytes=ceiling - checkpoint_bytes,
        effective_headroom_bytes=effective_ceiling - checkpoint_bytes,
        max_persistent_encoded_bytes=max_persistent_encoded_bytes,
        persistent_encoded_headroom_bytes=(
            None
            if max_persistent_encoded_bytes is None
            else max_persistent_encoded_bytes - total_persistent_encoded_bytes
        ),
        max_decoded_full_state_bytes=max_decoded_full_state_bytes,
        decoded_full_state_headroom_bytes=(
            None
            if max_decoded_full_state_bytes is None
            else max_decoded_full_state_bytes - total_decoded_full_state_bytes
        ),
        max_layer_transient_bytes=max_layer_transient_bytes,
        layer_transient_headroom_bytes=(
            None
            if max_layer_transient_bytes is None
            else max_layer_transient_bytes
            - runtime.maximum_layer_decode_transient_bytes
        ),
    )


def bf16_bits(values: Tensor) -> tuple[int, ...]:
    """Return exact raw BF16 words without text or base64 conversion."""

    if not isinstance(values, Tensor) or values.dtype != torch.bfloat16:
        raise CrevfError("bf16_bits requires a torch.bfloat16 Tensor.")
    raw = (
        values.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
    )
    return tuple(int(value) for value in raw.reshape(-1))


def _bf16_raw_bytes(values: Tensor) -> bytes:
    if not isinstance(values, Tensor) or values.dtype != torch.bfloat16:
        raise CrevfError("Exact BF16 bytes require a torch.bfloat16 Tensor.")
    words = values.detach().to(device="cpu").contiguous().view(torch.uint16).numpy()
    return words.astype("<u2", copy=False).tobytes(order="C")


def select_crevf_group_topk_outliers(
    values: Tensor,
    *,
    group_size: int,
    outliers_per_group: int,
    require_row_aligned: bool = True,
) -> tuple[int, ...]:
    """Select stable per-group absolute-magnitude outliers without training."""

    if (
        not isinstance(values, Tensor)
        or values.dtype != torch.bfloat16
        or values.device.type == "meta"
        or not bool(torch.isfinite(values).all())
    ):
        raise CrevfError(
            "Top-k outlier selection requires materialized finite BF16 values."
        )
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 8
        or group_size > _MAX_GROUP_SIZE
        or group_size % 8
    ):
        raise CrevfError("group_size must be a supported multiple of eight.")
    if (
        isinstance(outliers_per_group, bool)
        or not isinstance(outliers_per_group, int)
        or not 0 <= outliers_per_group <= group_size
    ):
        raise CrevfError("outliers_per_group must be in [0, group_size].")
    if not isinstance(require_row_aligned, bool) or (
        require_row_aligned and values.ndim >= 2 and int(values.shape[-1]) % group_size
    ):
        raise CrevfError(
            "Row-aligned outlier selection requires group_size to divide "
            "the tensor's final dimension."
        )
    if outliers_per_group == 0:
        return ()
    source = (
        values.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    )
    source_numel = int(source.numel())
    group_count = (source_numel + group_size - 1) // group_size
    selected_indices: list[int] = []
    for group_start in range(0, group_count, _QUANT_CHUNK_GROUPS):
        group_end = min(group_count, group_start + _QUANT_CHUNK_GROUPS)
        row_count = group_end - group_start
        value_start = group_start * group_size
        value_end = min(source_numel, group_end * group_size)
        magnitudes = torch.full(
            (row_count, group_size),
            -torch.inf,
            dtype=torch.float32,
        )
        magnitudes.reshape(-1)[: value_end - value_start].copy_(
            source[value_start:value_end].abs()
        )
        local = torch.argsort(
            magnitudes,
            dim=1,
            descending=True,
            stable=True,
        )[:, :outliers_per_group]
        rows = torch.arange(row_count, dtype=torch.int64)[:, None]
        global_indices = (
            (rows + group_start) * group_size + local.to(torch.int64)
        ).reshape(-1)
        selected_indices.extend(
            int(index) for index in global_indices.tolist() if int(index) < source_numel
        )
    return tuple(sorted(selected_indices))


def _policy_ppm(value: float, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrevfError(f"{name} must have exact parts-per-million precision.")
    ppm = int(round(float(value) * 1_000_000))
    if not math.isclose(float(value), ppm / 1_000_000, abs_tol=1e-12):
        raise CrevfError(f"{name} must have exact parts-per-million precision.")
    return ppm


def iter_crevf_quantized_candidates(
    *,
    name: str,
    role: str,
    values: Tensor,
    group_sizes: Sequence[int],
    bit_widths: Sequence[int],
    outliers_per_group: Sequence[int],
    clipping_ratios: Sequence[float],
    rounding_offsets: Sequence[float],
    exact_anchor_ranges: Sequence[tuple[int, int]] = (),
    importance_weights: Tensor | None = None,
    require_row_aligned: bool = True,
) -> Iterable[CrevfQuantizedCandidate]:
    """Yield actual encoded candidates across group/outlier/rounding policies.

    This iterator intentionally does not assign behavioral loss. A caller must
    score each blob from sealed cached evidence before constructing planner
    options.
    """

    canonical_group_sizes = tuple(sorted(set(group_sizes)))
    canonical_widths = tuple(sorted(set(bit_widths)))
    canonical_outlier_counts = tuple(sorted(set(outliers_per_group)))
    ratio_ppm = {_policy_ppm(value, "clipping ratio") for value in clipping_ratios}
    offset_ppm = {_policy_ppm(value, "rounding offset") for value in rounding_offsets}
    canonical_ratios = tuple(
        (ppm, ppm / 1_000_000) for ppm in sorted(ratio_ppm, reverse=True)
    )
    canonical_offsets = tuple((ppm, ppm / 1_000_000) for ppm in sorted(offset_ppm))
    if (
        not canonical_group_sizes
        or not canonical_widths
        or not canonical_outlier_counts
        or not canonical_ratios
        or not canonical_offsets
    ):
        raise CrevfError("Candidate policy axes must all be nonempty.")
    anchor_ranges = tuple(exact_anchor_ranges)
    for group_size in canonical_group_sizes:
        outlier_cache: dict[int, tuple[int, ...]] = {}
        for outlier_count in canonical_outlier_counts:
            selected = select_crevf_group_topk_outliers(
                values,
                group_size=group_size,
                outliers_per_group=outlier_count,
                require_row_aligned=require_row_aligned,
            )
            if anchor_ranges:
                selected = tuple(
                    index
                    for index in selected
                    if not any(
                        start <= index < start + length
                        for start, length in anchor_ranges
                    )
                )
            outlier_cache[outlier_count] = selected
        for width in canonical_widths:
            for outlier_count in canonical_outlier_counts:
                for clip_ppm, ratio in canonical_ratios:
                    for offset_ppm, offset in canonical_offsets:
                        spec = build_crevf_quantized_spec(
                            name=name,
                            role=role,
                            values=values,
                            group_bit_widths=width,
                            group_size=group_size,
                            exact_outlier_indices=outlier_cache[outlier_count],
                            exact_anchor_ranges=anchor_ranges,
                            clipping_ratios=(ratio,),
                            rounding_offsets=(offset,),
                            importance_weights=importance_weights,
                        )
                        yield CrevfQuantizedCandidate(
                            candidate_id=(
                                f"g{group_size}-w{width}-k{outlier_count}"
                                f"-c{clip_ppm}-r{offset_ppm:+d}"
                            ),
                            group_size=group_size,
                            bit_width=width,
                            exact_outliers_per_group=outlier_count,
                            clipping_ratio_ppm=clip_ppm,
                            rounding_offset_ppm=offset_ppm,
                            tensor_blob=encode_crevf_tensor(spec),
                        )


def build_crevf_explicit_quantized_spec(
    *,
    name: str,
    role: str,
    shape: Sequence[int],
    group_size: int,
    group_bit_widths: Sequence[int],
    group_codes: Tensor | np.ndarray | Sequence[Sequence[int]],
    scales: Tensor | np.ndarray,
    exact_outliers: Sequence[CrevfExactValue] = (),
    exact_anchors: Sequence[CrevfExactRun] = (),
) -> CrevfTensorSpec:
    """Validate an externally solved integer representation without requantizing.

    Unlike :func:`build_crevf_quantized_spec`, this path never receives dense
    source values, chooses a scale, clips a value, or rounds a code.  It
    canonicalizes already solved uint codes and FP16 scales, then applies the
    exact same geometry, range, padding, overlay, and authentication
    preconditions used by the encoder.
    """

    _canonical_text(name, "tensor name")
    _canonical_text(role, "tensor role")
    canonical_shape = _canonical_shape(shape)
    source_numel = _product(canonical_shape)
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 8
        or group_size > _MAX_GROUP_SIZE
        or group_size % 8
    ):
        raise CrevfError(
            "Explicit CREVF group_size must be a multiple of eight in range."
        )
    group_count = (source_numel + group_size - 1) // group_size
    widths = tuple(group_bit_widths)
    if len(widths) != group_count or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 4 <= value <= 8
        for value in widths
    ):
        raise CrevfError(
            "Every explicit quantized group requires a bit width in [4, 8]."
        )
    if isinstance(group_codes, Tensor) and group_codes.device.type == "meta":
        raise CrevfError("Explicit CREVF group codes must be materialized.")
    code_matrix = _group_code_matrix(
        group_codes,
        group_count=group_count,
        group_size=group_size,
        widths=widths,
    ).copy()
    if isinstance(scales, Tensor):
        if scales.dtype != torch.float16 or scales.device.type == "meta":
            raise CrevfError(
                "Explicit CREVF scales must be materialized torch.float16 values."
            )
    elif isinstance(scales, np.ndarray):
        if scales.dtype != np.dtype(np.float16):
            raise CrevfError("Explicit CREVF scales must use NumPy float16.")
    else:
        raise CrevfError(
            "Explicit CREVF scales must be a torch or NumPy FP16 array."
        )
    scale_bytes = _float16_scale_bytes(scales, group_count)
    canonical_scales = np.frombuffer(scale_bytes, dtype="<f2").copy()
    outliers = tuple(exact_outliers)
    anchors = tuple(exact_anchors)
    if any(not isinstance(value, CrevfExactValue) for value in outliers):
        raise CrevfError("Explicit outliers must be CrevfExactValue entries.")
    if any(not isinstance(value, CrevfExactRun) for value in anchors):
        raise CrevfError("Explicit anchors must be CrevfExactRun entries.")
    # The segment encoders enforce canonical sorting, uniqueness, disjoint
    # anchor runs, and minimal binary representation.
    _encode_outliers(outliers)
    _encode_anchors(anchors)
    outlier_indices = tuple(value.index for value in outliers)
    if (
        any(index >= source_numel for index in outlier_indices)
        or any(run.end_index > source_numel for run in anchors)
        or _exact_segments_overlap(outlier_indices, anchors)
    ):
        raise CrevfError(
            "Exact BF16 indices must be in range and outliers cannot overlap anchors."
        )
    valid = source_numel - (group_count - 1) * group_size
    final_qmax = (1 << (widths[-1] - 1)) - 1
    if valid < group_size and bool((code_matrix[-1, valid:] != final_qmax).any()):
        raise CrevfError(
            "Final-group padded positions must use the canonical zero code."
        )
    _validate_overlay_zero_codes(
        code_matrix,
        widths,
        group_size,
        outlier_indices,
        anchors,
        format_error=False,
    )
    return CrevfTensorSpec(
        name=name,
        role=role,
        shape=canonical_shape,
        group_size=group_size,
        group_bit_widths=widths,
        group_codes=code_matrix,
        scales=canonical_scales,
        exact_outliers=outliers,
        exact_anchors=anchors,
    )


def build_crevf_quantized_spec(
    *,
    name: str,
    role: str,
    values: Tensor,
    group_bit_widths: int | Sequence[int],
    group_size: int = 256,
    exact_outlier_indices: Sequence[int] = (),
    exact_anchor_ranges: Sequence[tuple[int, int]] = (),
    clipping_ratios: Sequence[float] = (1.0,),
    rounding_offsets: Sequence[float] = (0.0,),
    importance_weights: Tensor | None = None,
) -> CrevfTensorSpec:
    """Deterministically quantize a local BF16 tensor into an encodable spec.

    Candidate clipping and rounding policies are selected independently per
    group by weighted reconstruction MSE. ``importance_weights`` may contain a
    sealed cached-causal diagonal; no parent execution or optimization occurs.
    """

    if not isinstance(values, Tensor) or values.dtype != torch.bfloat16:
        raise CrevfError("CREVF source values must be a local torch.bfloat16 Tensor.")
    if values.device.type == "meta" or not bool(torch.isfinite(values).all()):
        raise CrevfError("CREVF source values must be materialized and finite.")
    shape = tuple(int(value) for value in values.shape)
    _canonical_shape(shape)
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 8
        or group_size > _MAX_GROUP_SIZE
        or group_size % 8
    ):
        raise CrevfError("group_size must be a supported multiple of eight.")
    source = values.detach().to(device="cpu").contiguous().reshape(-1)
    source_numel = int(source.numel())
    group_count = (source_numel + group_size - 1) // group_size
    ratios = tuple(sorted(set(float(value) for value in clipping_ratios), reverse=True))
    if not ratios or any(
        not math.isfinite(value) or not 0 < value <= 1 for value in ratios
    ):
        raise CrevfError("clipping_ratios must be finite values in (0, 1].")
    offsets = tuple(sorted(set(float(value) for value in rounding_offsets)))
    if not offsets or any(
        not math.isfinite(value) or not -0.499 <= value <= 0.499 for value in offsets
    ):
        raise CrevfError("rounding_offsets must be finite values in [-0.499, 0.499].")
    if importance_weights is None:
        importance = None
    else:
        if (
            not isinstance(importance_weights, Tensor)
            or importance_weights.shape != values.shape
            or importance_weights.device.type == "meta"
            or not torch.is_floating_point(importance_weights)
        ):
            raise CrevfError(
                "importance_weights must be a materialized floating Tensor "
                "matching values."
            )
        importance = (
            importance_weights.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .reshape(-1)
        )
        if not bool(torch.isfinite(importance).all()) or bool((importance < 0).any()):
            raise CrevfError("importance_weights must be finite and nonnegative.")
    outlier_indices = tuple(exact_outlier_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < source_numel
        for index in outlier_indices
    ) or outlier_indices != tuple(sorted(set(outlier_indices))):
        raise CrevfError("exact_outlier_indices must be in range, unique, and sorted.")
    anchor_pairs: list[tuple[int, int]] = []
    previous_end = 0
    for range_index, pair in enumerate(exact_anchor_ranges):
        if (
            not isinstance(pair, Sequence)
            or len(pair) != 2
            or isinstance(pair[0], bool)
            or not isinstance(pair[0], int)
            or isinstance(pair[1], bool)
            or not isinstance(pair[1], int)
        ):
            raise CrevfError(
                "Exact anchor ranges must be integer (start, length) pairs."
            )
        anchor_start, anchor_length = pair
        if (
            anchor_start < 0
            or anchor_length <= 0
            or anchor_start + anchor_length > source_numel
            or anchor_start < previous_end
            or (range_index > 0 and anchor_start == previous_end)
        ):
            raise CrevfError(
                "Exact anchor ranges must be sorted, in range, and nonadjacent."
            )
        anchor_pairs.append((anchor_start, anchor_length))
        previous_end = anchor_start + anchor_length
    source_words = source.view(torch.uint16)
    outliers = tuple(
        CrevfExactValue(index, int(source_words[index])) for index in outlier_indices
    )
    anchors = tuple(
        CrevfExactRun(
            anchor_start,
            source_words[anchor_start : anchor_start + anchor_length]
            .numpy()
            .astype("<u2", copy=False)
            .tobytes(order="C"),
        )
        for anchor_start, anchor_length in anchor_pairs
    )
    if _exact_segments_overlap(outlier_indices, anchors):
        raise CrevfError("Exact outliers cannot overlap exact anchor ranges.")
    if isinstance(group_bit_widths, int) and not isinstance(group_bit_widths, bool):
        widths = (group_bit_widths,) * group_count
    else:
        widths = tuple(group_bit_widths)
    if len(widths) != group_count or any(
        isinstance(width, bool) or not isinstance(width, int) or not 4 <= width <= 8
        for width in widths
    ):
        raise CrevfError("group_bit_widths must provide one width in [4, 8] per group.")
    width_array = np.asarray(widths, dtype=np.uint8)
    scales = torch.empty(group_count, dtype=torch.float16)
    codes = torch.empty((group_count, group_size), dtype=torch.uint8)
    for group_start in range(0, group_count, _QUANT_CHUNK_GROUPS):
        group_end = min(group_count, group_start + _QUANT_CHUNK_GROUPS)
        row_count = group_end - group_start
        value_start = group_start * group_size
        value_end = min(source_numel, group_end * group_size)
        chunk = torch.zeros((row_count, group_size), dtype=torch.float32)
        chunk.reshape(-1)[: value_end - value_start].copy_(
            source[value_start:value_end].to(torch.float32)
        )
        chunk_importance = torch.ones_like(chunk)
        if importance is not None:
            chunk_importance.zero_()
            chunk_importance.reshape(-1)[: value_end - value_start].copy_(
                importance[value_start:value_end]
            )
        outlier_start = bisect_left(outlier_indices, value_start)
        outlier_end = bisect_left(outlier_indices, value_end)
        if outlier_end > outlier_start:
            local_indices = torch.tensor(
                [
                    index - value_start
                    for index in outlier_indices[outlier_start:outlier_end]
                ],
                dtype=torch.long,
            )
            chunk.reshape(-1).index_fill_(0, local_indices, 0)
        for run in anchors:
            if run.start_index >= value_end:
                break
            if run.end_index > value_start:
                local_start = max(run.start_index, value_start) - value_start
                local_end = min(run.end_index, value_end) - value_start
                chunk.reshape(-1)[local_start:local_end] = 0
        chunk_widths = width_array[group_start:group_end]
        for width in range(4, 9):
            selected_array = np.flatnonzero(chunk_widths == width)
            if not len(selected_array):
                continue
            selected = torch.from_numpy(selected_array)
            selected_values = chunk.index_select(0, selected)
            selected_importance = chunk_importance.index_select(0, selected)
            qmax = (1 << (width - 1)) - 1
            maxima = selected_values.abs().amax(dim=1)
            best_loss = torch.full(
                (int(selected.numel()),),
                torch.inf,
                dtype=torch.float64,
            )
            best_scale = torch.empty(
                int(selected.numel()),
                dtype=torch.float16,
            )
            best_codes = torch.empty(
                (int(selected.numel()), group_size),
                dtype=torch.uint8,
            )
            for ratio in ratios:
                scale32 = torch.where(
                    maxima > 0,
                    maxima * ratio / float(qmax),
                    torch.ones_like(maxima),
                )
                scale = scale32.to(torch.float16)
                valid_scale = torch.isfinite(scale) & (scale > 0)
                if not bool(valid_scale.any()):
                    continue
                effective = scale.to(torch.float32)
                normalized = selected_values.abs() / effective[:, None]
                signs = torch.sign(selected_values)
                for offset in offsets:
                    signed = (signs * torch.floor(normalized + 0.5 + offset)).clamp(
                        -qmax, qmax
                    )
                    reconstructed = signed * effective[:, None]
                    error = (selected_values - reconstructed).to(torch.float64)
                    loss = torch.sum(
                        selected_importance.to(torch.float64) * error.square(),
                        dim=1,
                    )
                    loss = torch.where(
                        valid_scale,
                        loss,
                        torch.full_like(loss, torch.inf),
                    )
                    better = loss < best_loss
                    if bool(better.any()):
                        best_loss = torch.where(better, loss, best_loss)
                        best_scale[better] = scale[better]
                        candidate_codes = (signed.to(torch.int16) + qmax).to(
                            torch.uint8
                        )
                        best_codes[better] = candidate_codes[better]
            if not bool(torch.isfinite(best_loss).all()):
                raise CrevfError("A quantization scale is not representable in FP16.")
            destination = selected + group_start
            codes.index_copy_(
                0,
                destination,
                best_codes,
            )
            scales.index_copy_(0, destination, best_scale)
    return CrevfTensorSpec(
        name=name,
        role=role,
        shape=shape,
        group_size=group_size,
        group_bit_widths=widths,
        group_codes=codes,
        scales=scales,
        exact_outliers=outliers,
        exact_anchors=anchors,
    )


def build_crevf_exact_spec(
    *,
    name: str,
    role: str,
    values: Tensor,
) -> CrevfTensorSpec:
    """Build an exact-only binary BF16 control-tensor specification."""

    raw = _bf16_raw_bytes(values)
    return CrevfTensorSpec(
        name=name,
        role=role,
        shape=tuple(int(value) for value in values.shape),
        exact_anchors=(CrevfExactRun(0, raw),),
        exact_only=True,
    )


__all__ = [
    "CREVF_ANCHOR_MAGIC",
    "CREVF_ARTIFACT_FIXED_BYTES",
    "CREVF_AUXILIARY_MAGIC",
    "CREVF_ENTROPY_MAGIC",
    "CREVF_FILE_MAGIC",
    "CREVF_OUTLIER_MAGIC",
    "CREVF_RECORD_ENVELOPE_BYTES",
    "CREVF_TENSOR_MAGIC",
    "CREVF_ZLIB_PROFILE",
    "CrevfBudgetError",
    "CrevfByteAccounting",
    "CrevfAuxiliaryMetadata",
    "CrevfAuxiliarySpec",
    "CrevfError",
    "CrevfExactRun",
    "CrevfExactValue",
    "CrevfFormatError",
    "CrevfPackedAuxiliary",
    "CrevfPackedRuntime",
    "CrevfPackedTensor",
    "CrevfQuantizedCandidate",
    "CrevfReductionTarget",
    "CrevfTensorMetadata",
    "CrevfTensorSpec",
    "account_crevf_artifact",
    "bf16_bits",
    "build_crevf_exact_spec",
    "build_crevf_explicit_quantized_spec",
    "build_crevf_quantized_spec",
    "crevf_decode_transient_group",
    "decode_crevf_artifact",
    "decode_crevf_tensor",
    "encode_crevf_artifact",
    "encode_crevf_auxiliary",
    "encode_crevf_tensor",
    "inspect_crevf_auxiliary",
    "inspect_crevf_tensor",
    "iter_crevf_quantized_candidates",
    "select_crevf_group_topk_outliers",
]
