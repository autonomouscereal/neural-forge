"""Packed LFM2 value-quantization controls for large byte reductions.

This module is deliberately a *control*, not a structural folding result.
Every logical layer, head, channel, neuron, tokenizer entry, and tied
embedding/output-head role remains present.  Only the serialized scalar values
are represented by groupwise symmetric quantization.

The 50% control starts from eight-bit groups and uses a distortion-ranked
minority of seven-bit groups.  The 75% control starts from four-bit groups and
uses a distortion-ranked minority of three-bit groups.  The lower-width groups
pay for every scale, one-bit group selector, payload header, JSON manifest,
configuration file, generation configuration, and copied tokenizer artifact.
Consequently the final directory, rather than an idealized tensor-only count,
must fit the requested byte ceiling.

The standalone runtime owns only packed ``uint8`` buffers and ``float16``
scales.  Dense weights are materialized transiently inside an operator call;
they are never stored in the checkpoint, a sidecar, a Parameter, or a
persistent runtime cache.  The input embedding and LM head reference the same
single packed tensor store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional HF dependency
    raise RuntimeError(
        "PyTorch is required for the LFM2 packed value-fold control."
    ) from exc

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256


CHECKPOINT_SCHEMA = "neural-forge-lfm2-packed-value-fold-control-v1"
MANIFEST_FILENAME = "neural_forge_lfm2_value_fold_control_manifest.json"
PAYLOAD_FILENAME = "model.nfpq"
CONFIG_FILENAME = "config.json"
GENERATION_CONFIG_FILENAME = "generation_config.json"
PAYLOAD_MAGIC = b"NF_LFM2_PQ_V1\n"
CONTROL_KIND = "groupwise_symmetric_packed_value_quantization"
TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
)
_MODEL_MODULE = "transformers.models.lfm2.modeling_lfm2"
_DIGEST_LENGTH = 64
_PLACEHOLDER_DIGEST = "0" * _DIGEST_LENGTH
_SOURCE_DTYPES = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}
_TARGET_BITS = {
    500_000: (8, 7),
    750_000: (4, 3),
}


class Lfm2ValueFoldControlError(ValueError):
    """The source, target, or control configuration is invalid."""


class Lfm2ValueFoldControlBudgetError(Lfm2ValueFoldControlError):
    """The requested physical byte ceiling cannot contain the control."""


class Lfm2ValueFoldControlFormatError(Lfm2ValueFoldControlError):
    """A packed checkpoint is incomplete, inconsistent, or corrupted."""


@dataclass(frozen=True, slots=True)
class Lfm2ValueFoldTarget:
    """An exact serialized-directory byte target for one control."""

    reduction_ppm: int
    group_size: int = 256
    analysis_chunk_groups: int = 4096

    def __post_init__(self) -> None:
        if self.reduction_ppm not in _TARGET_BITS:
            raise Lfm2ValueFoldControlError(
                "reduction_ppm must encode exactly 0.50 or 0.75 (500000 or 750000)."
            )
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 8
            or self.group_size % 8
        ):
            raise Lfm2ValueFoldControlError(
                "group_size must be an integer multiple of eight."
            )
        if (
            isinstance(self.analysis_chunk_groups, bool)
            or not isinstance(self.analysis_chunk_groups, int)
            or self.analysis_chunk_groups < 1
        ):
            raise Lfm2ValueFoldControlError(
                "analysis_chunk_groups must be a positive integer."
            )

    @classmethod
    def from_reduction(
        cls,
        reduction: float,
        *,
        group_size: int = 256,
        analysis_chunk_groups: int = 4096,
    ) -> "Lfm2ValueFoldTarget":
        if isinstance(reduction, bool) or not isinstance(reduction, (int, float)):
            raise Lfm2ValueFoldControlError("target_reduction must be 0.50 or 0.75.")
        ppm = int(round(float(reduction) * 1_000_000))
        if not math.isclose(float(reduction), ppm / 1_000_000, abs_tol=1e-12):
            raise Lfm2ValueFoldControlError(
                "target_reduction must have an exact parts-per-million form."
            )
        return cls(
            ppm,
            group_size=group_size,
            analysis_chunk_groups=analysis_chunk_groups,
        )

    @property
    def requested_reduction(self) -> float:
        return self.reduction_ppm / 1_000_000

    @property
    def base_bit_width(self) -> int:
        return _TARGET_BITS[self.reduction_ppm][0]

    @property
    def lower_bit_width(self) -> int:
        return _TARGET_BITS[self.reduction_ppm][1]

    @property
    def bytes_saved_per_lower_group(self) -> int:
        return self.group_size * (self.base_bit_width - self.lower_bit_width) // 8


@dataclass(frozen=True, slots=True)
class _TensorAnalysis:
    canonical_name: str
    logical_names: tuple[str, ...]
    parameter: nn.Parameter
    shape: tuple[int, ...]
    source_dtype: str
    source_numel: int
    source_nbytes: int
    source_sha256: str
    group_count: int
    base_sse: np.ndarray
    lower_sse: np.ndarray
    source_square_sum: np.ndarray

    @property
    def distortion_penalty(self) -> np.ndarray:
        denominator = np.maximum(self.source_square_sum, 1e-30)
        return np.maximum(self.lower_sse - self.base_sse, 0.0) / denominator


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise Lfm2ValueFoldControlError(f"Refusing to replace {path}.")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new(path, canonical_json_bytes(dict(value)) + b"\n")


def _safe_local_directory(
    value: str | os.PathLike[str],
    name: str,
) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ValueFoldControlError(f"{name} must be an explicit local directory.")
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2ValueFoldControlError(f"{name} does not exist.") from exc
    if not path.is_dir() or path.is_symlink():
        raise Lfm2ValueFoldControlError(
            f"{name} is missing, not a directory, or a symlink."
        )
    return path


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2ValueFoldControlFormatError(f"Duplicate JSON key {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise Lfm2ValueFoldControlFormatError(
            f"Non-finite JSON constant {value!r} is forbidden."
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except Lfm2ValueFoldControlFormatError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2ValueFoldControlFormatError(
            f"Could not read strict JSON artifact {path.name!r}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2ValueFoldControlFormatError(
            f"JSON artifact {path.name!r} must contain an object."
        )
    return value


def _model_parameters(
    model: nn.Module,
) -> tuple[dict[str, nn.Parameter], dict[str, str]]:
    try:
        logical = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - supported torch is required
        raise Lfm2ValueFoldControlError(
            "PyTorch must support remove_duplicate=False."
        ) from exc
    if not logical:
        raise Lfm2ValueFoldControlError("The source LFM2 has no Parameters.")
    if (
        "model.embed_tokens.weight" not in logical
        or "lm_head.weight" not in logical
        or logical["model.embed_tokens.weight"] is not logical["lm_head.weight"]
    ):
        raise Lfm2ValueFoldControlError(
            "The source must tie model.embed_tokens.weight and lm_head.weight "
            "by exact Parameter identity."
        )
    identity_groups: dict[int, list[str]] = {}
    for name, parameter in logical.items():
        if (
            not isinstance(name, str)
            or not isinstance(parameter, nn.Parameter)
            or parameter.device.type == "meta"
            or parameter.layout != torch.strided
        ):
            raise Lfm2ValueFoldControlError(
                "Every logical source weight must be a materialized dense Parameter."
            )
        dtype_name = str(parameter.dtype)
        if dtype_name not in _SOURCE_DTYPES or parameter.element_size() != 2:
            raise Lfm2ValueFoldControlError(
                "The 50%/75% controls require a float16 or bfloat16 source "
                f"checkpoint; {name!r} has {dtype_name}."
            )
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise Lfm2ValueFoldControlError(
                f"Source Parameter {name!r} contains non-finite values."
            )
        identity_groups.setdefault(id(parameter), []).append(name)
    aliases: dict[str, str] = {}
    for names in identity_groups.values():
        if "model.embed_tokens.weight" in names:
            canonical = "model.embed_tokens.weight"
        else:
            canonical = sorted(names)[0]
        for name in names:
            aliases[name] = canonical
    if aliases["lm_head.weight"] != "model.embed_tokens.weight":
        raise Lfm2ValueFoldControlError(
            "The tied vocabulary did not canonicalize to the embedding."
        )
    return logical, aliases


def _validate_source_lfm2(
    model: nn.Module,
) -> tuple[dict[str, nn.Parameter], dict[str, str]]:
    if (
        not isinstance(model, nn.Module)
        or type(model).__name__ != "Lfm2ForCausalLM"
        or type(model).__module__ != _MODEL_MODULE
        or getattr(getattr(model, "config", None), "model_type", None) != "lfm2"
    ):
        raise Lfm2ValueFoldControlError(
            "The packed control requires Transformers Lfm2ForCausalLM."
        )
    config = model.config
    layers = getattr(getattr(model, "model", None), "layers", None)
    if (
        not isinstance(layers, nn.ModuleList)
        or len(layers) != getattr(config, "num_hidden_layers", None)
        or tuple(getattr(config, "layer_types", ()))
        != tuple(
            "full_attention" if layer.is_attention_layer else "conv" for layer in layers
        )
        or getattr(config, "tie_word_embeddings", None) is not True
    ):
        raise Lfm2ValueFoldControlError(
            "The concrete LFM2 topology disagrees with its configuration."
        )
    return _model_parameters(model)


def pack_unsigned_codes(codes: Tensor | np.ndarray, bit_width: int) -> bytes:
    """Pack unsigned codes LSB-first without Python object serialization."""

    if isinstance(bit_width, bool) or not isinstance(bit_width, int):
        raise Lfm2ValueFoldControlError("bit_width must be an integer.")
    if bit_width < 1 or bit_width > 8:
        raise Lfm2ValueFoldControlError("bit_width must be in [1, 8].")
    if isinstance(codes, Tensor):
        values = codes.detach().to(device="cpu", dtype=torch.uint8).numpy()
    else:
        values = np.asarray(codes)
    if values.dtype != np.uint8:
        values = values.astype(np.uint8, copy=False)
    flat = np.ascontiguousarray(values.reshape(-1))
    maximum = (1 << bit_width) - 1
    if flat.size and int(flat.max()) > maximum:
        raise Lfm2ValueFoldControlError(
            "A code exceeds the selected unsigned bit width."
        )
    if flat.size == 0:
        return b""
    shifts = np.arange(bit_width, dtype=np.uint8)
    bits = ((flat[:, None] >> shifts[None, :]) & 1).reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def unpack_unsigned_codes(
    packed: Tensor,
    *,
    value_count: int,
    bit_width: int,
) -> Tensor:
    """Unpack LSB-first codes on the packed tensor's current device."""

    if (
        not isinstance(packed, Tensor)
        or packed.dtype != torch.uint8
        or packed.ndim != 1
    ):
        raise Lfm2ValueFoldControlFormatError(
            "packed must be a one-dimensional uint8 Tensor."
        )
    if (
        isinstance(value_count, bool)
        or not isinstance(value_count, int)
        or value_count < 0
    ):
        raise Lfm2ValueFoldControlFormatError(
            "value_count must be a nonnegative integer."
        )
    if (
        isinstance(bit_width, bool)
        or not isinstance(bit_width, int)
        or bit_width < 1
        or bit_width > 8
    ):
        raise Lfm2ValueFoldControlFormatError("bit_width must be in [1, 8].")
    expected = (value_count * bit_width + 7) // 8
    if packed.numel() != expected:
        raise Lfm2ValueFoldControlFormatError(
            "Packed code byte count differs from its declared shape."
        )
    if value_count == 0:
        return torch.empty(0, dtype=torch.uint8, device=packed.device)
    bit_shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(1), bit_shifts),
        1,
    ).reshape(-1)[: value_count * bit_width]
    value_shifts = torch.arange(
        bit_width,
        dtype=torch.int16,
        device=packed.device,
    )
    values = torch.sum(
        bits.reshape(value_count, bit_width).to(torch.int16)
        * torch.bitwise_left_shift(
            torch.ones_like(value_shifts),
            value_shifts,
        ),
        dim=1,
    )
    return values.to(torch.uint8)


def _pack_mask(mask: np.ndarray) -> bytes:
    values = np.asarray(mask, dtype=np.uint8).reshape(-1)
    return np.packbits(values, bitorder="little").tobytes()


def _unpack_mask(mask: Tensor, count: int) -> Tensor:
    unpacked = unpack_unsigned_codes(
        mask, value_count=((count + 7) // 8) * 8, bit_width=1
    )
    return unpacked[:count].to(torch.bool)


def _quantize_groups(
    values: Tensor,
    bit_width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if values.ndim != 2 or values.dtype != torch.float32 or values.device.type != "cpu":
        raise Lfm2ValueFoldControlError(
            "Quantization analysis values must be a two-dimensional CPU float32 Tensor."
        )
    qmax = (1 << (bit_width - 1)) - 1
    if qmax < 1:
        raise Lfm2ValueFoldControlError(
            "Symmetric value quantization requires at least two bits."
        )
    maxima = values.abs().amax(dim=1)
    scales32 = torch.where(
        maxima > 0,
        maxima / float(qmax),
        torch.ones_like(maxima),
    )
    scales = scales32.to(torch.float16)
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise Lfm2ValueFoldControlError(
            "A group scale is non-finite or not representable in float16."
        )
    effective = scales.to(torch.float32)
    signed = torch.round(values / effective[:, None]).clamp(-qmax, qmax)
    codes = (signed.to(torch.int16) + qmax).to(torch.uint8)
    reconstructed = (codes.to(torch.int16) - qmax).to(torch.float32)
    reconstructed = reconstructed * effective[:, None]
    return codes, scales, reconstructed


def _parameter_chunks(
    parameter: nn.Parameter,
    *,
    group_size: int,
    chunk_groups: int,
):
    flat = parameter.detach().reshape(-1)
    numel = int(flat.numel())
    group_count = (numel + group_size - 1) // group_size
    for group_start in range(0, group_count, chunk_groups):
        group_end = min(group_count, group_start + chunk_groups)
        value_start = group_start * group_size
        value_end = min(numel, group_end * group_size)
        source = flat[value_start:value_end].to(
            device="cpu",
            dtype=torch.float32,
        )
        padded = torch.zeros(
            group_end - group_start,
            group_size,
            dtype=torch.float32,
        )
        padded.reshape(-1)[: int(source.numel())].copy_(source)
        yield group_start, group_end, value_start, value_end, padded


def _source_tensor_sha256(parameter: nn.Parameter, chunk_values: int) -> str:
    digest = hashlib.sha256()
    flat = parameter.detach().reshape(-1)
    for start in range(0, int(flat.numel()), chunk_values):
        piece = flat[start : start + chunk_values].to(device="cpu").contiguous()
        digest.update(piece.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _analyze_tensor(
    canonical_name: str,
    logical_names: Sequence[str],
    parameter: nn.Parameter,
    target: Lfm2ValueFoldTarget,
) -> _TensorAnalysis:
    numel = int(parameter.numel())
    groups = (numel + target.group_size - 1) // target.group_size
    base_sse = np.empty(groups, dtype=np.float64)
    lower_sse = np.empty(groups, dtype=np.float64)
    square_sum = np.empty(groups, dtype=np.float64)
    for start, end, _value_start, _value_end, values in _parameter_chunks(
        parameter,
        group_size=target.group_size,
        chunk_groups=target.analysis_chunk_groups,
    ):
        _base_codes, _base_scales, base = _quantize_groups(
            values,
            target.base_bit_width,
        )
        _lower_codes, _lower_scales, lower = _quantize_groups(
            values,
            target.lower_bit_width,
        )
        base_sse[start:end] = (
            (values - base).square().sum(dim=1).to(torch.float64).numpy()
        )
        lower_sse[start:end] = (
            (values - lower).square().sum(dim=1).to(torch.float64).numpy()
        )
        square_sum[start:end] = values.square().sum(dim=1).to(torch.float64).numpy()
    return _TensorAnalysis(
        canonical_name=canonical_name,
        logical_names=tuple(sorted(logical_names)),
        parameter=parameter,
        shape=tuple(int(value) for value in parameter.shape),
        source_dtype=str(parameter.dtype),
        source_numel=numel,
        source_nbytes=numel * int(parameter.element_size()),
        source_sha256=_source_tensor_sha256(
            parameter,
            target.group_size * target.analysis_chunk_groups,
        ),
        group_count=groups,
        base_sse=base_sse,
        lower_sse=lower_sse,
        source_square_sum=square_sum,
    )


def _analyze_model(
    model: nn.Module,
    target: Lfm2ValueFoldTarget,
) -> tuple[tuple[_TensorAnalysis, ...], dict[str, str]]:
    logical, aliases = _validate_source_lfm2(model)
    logical_by_canonical: dict[str, list[str]] = {}
    for name, canonical in aliases.items():
        logical_by_canonical.setdefault(canonical, []).append(name)
    analyses = []
    with torch.no_grad():
        for canonical in sorted(logical_by_canonical):
            analyses.append(
                _analyze_tensor(
                    canonical,
                    logical_by_canonical[canonical],
                    logical[canonical],
                    target,
                )
            )
    return tuple(analyses), aliases


def _selection_order(
    analyses: Sequence[_TensorAnalysis],
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    offsets = []
    cursor = 0
    penalties = []
    for analysis in analyses:
        start = cursor
        cursor += analysis.group_count
        offsets.append((start, cursor))
        penalties.append(analysis.distortion_penalty)
    combined = np.concatenate(penalties)
    order = np.argsort(combined, kind="stable")
    return order, tuple(offsets)


def _selection_masks(
    analyses: Sequence[_TensorAnalysis],
    offsets: Sequence[tuple[int, int]],
    order: np.ndarray,
    lower_group_count: int,
) -> tuple[np.ndarray, ...]:
    total = sum(value.group_count for value in analyses)
    if lower_group_count < 0 or lower_group_count > total:
        raise Lfm2ValueFoldControlBudgetError(
            "Lower-bit group count exceeds the available groups."
        )
    selected = np.zeros(total, dtype=np.bool_)
    selected[order[:lower_group_count]] = True
    return tuple(selected[start:end].copy() for start, end in offsets)


def _tensor_entry_layout(
    analyses: Sequence[_TensorAnalysis],
    masks: Sequence[np.ndarray],
    target: Lfm2ValueFoldTarget,
    *,
    segment_digests: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], int]:
    offset = len(PAYLOAD_MAGIC)
    entries: dict[str, Any] = {}
    for analysis, mask in zip(analyses, masks, strict=True):
        low_count = int(np.count_nonzero(mask))
        high_count = analysis.group_count - low_count
        mask_bytes = (analysis.group_count + 7) // 8
        high_bytes = high_count * target.group_size * target.base_bit_width // 8
        low_bytes = low_count * target.group_size * target.lower_bit_width // 8
        scale_bytes = analysis.group_count * 2
        segments: dict[str, Any] = {}
        for role, size in (
            ("lower_group_mask", mask_bytes),
            ("base_bit_values", high_bytes),
            ("lower_bit_values", low_bytes),
            ("scales", scale_bytes),
        ):
            digest = (
                _PLACEHOLDER_DIGEST
                if segment_digests is None
                else segment_digests[analysis.canonical_name][role]
            )
            segments[role] = {
                "byte_offset": offset,
                "size_bytes": size,
                "sha256": digest,
            }
            offset += size
        chosen_sse = np.where(
            mask,
            analysis.lower_sse,
            analysis.base_sse,
        )
        entries[analysis.canonical_name] = {
            "logical_names": list(analysis.logical_names),
            "shape": list(analysis.shape),
            "source_dtype": analysis.source_dtype,
            "source_numel": analysis.source_numel,
            "source_nbytes": analysis.source_nbytes,
            "source_sha256": analysis.source_sha256,
            "group_size": target.group_size,
            "group_count": analysis.group_count,
            "base_bit_width": target.base_bit_width,
            "lower_bit_width": target.lower_bit_width,
            "base_group_count": high_count,
            "lower_group_count": low_count,
            "scale_dtype": "torch.float16",
            "symmetric_zero_point": 0,
            "padded_value_count": analysis.group_count * target.group_size,
            "sum_squared_error": float(chosen_sse.sum()),
            "source_square_sum": float(analysis.source_square_sum.sum()),
            "segments": segments,
        }
    return entries, offset


def _config_documents(model: nn.Module) -> tuple[dict[str, Any], dict[str, Any]]:
    config = dict(model.config.to_dict())
    if config.get("model_type") != "lfm2":
        raise Lfm2ValueFoldControlError("Source config is not LFM2.")
    config["_name_or_path"] = ""
    config.pop("auto_map", None)
    config["tie_word_embeddings"] = True
    generation = getattr(model, "generation_config", None)
    generation_dict = (
        dict(generation.to_dict())
        if generation is not None and callable(getattr(generation, "to_dict", None))
        else {}
    )
    generation_dict["_from_model_config"] = True
    canonical_json_bytes(config)
    canonical_json_bytes(generation_dict)
    return config, generation_dict


def _copy_tokenizer_artifacts(
    source: Path,
    destination: Path,
) -> tuple[dict[str, Any], ...]:
    files = []
    for name in TOKENIZER_ARTIFACTS:
        path = source / name
        if path.is_symlink():
            raise Lfm2ValueFoldControlError(
                f"Tokenizer artifact cannot be a symlink: {path}."
            )
        if path.is_file():
            target = destination / name
            shutil.copyfile(path, target)
            files.append(_artifact(target))
    if not files or files[0]["file"] != "tokenizer.json":
        raise Lfm2ValueFoldControlError(
            "The local tokenizer package lacks tokenizer.json."
        )
    return tuple(files)


def _tokenizer_fingerprint(
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
) -> str:
    files = [
        {
            "path": value["file"],
            "bytes": value["size_bytes"],
            "sha256": value["sha256"],
        }
        for value in tokenizer_artifacts
    ]
    return canonical_sha256(
        {
            "schema": "neural-forge-local-tokenizer-artifacts-v1",
            "files": files,
        }
    )


def _manifest_document(
    *,
    model: nn.Module,
    target: Lfm2ValueFoldTarget,
    aliases: Mapping[str, str],
    entries: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    dense_tensor_bytes: int,
    shared_nonweight_bytes: int,
    payload_bytes: int,
    compressed_checkpoint_bytes: int,
    manifest_bytes: int,
    lower_group_count: int,
    total_group_count: int,
    tokenizer_fingerprint: str,
) -> tuple[dict[str, Any], bytes]:
    target_bytes = (
        (dense_tensor_bytes + shared_nonweight_bytes)
        * (1_000_000 - target.reduction_ppm)
        // 1_000_000
    )
    packed_value_bytes = sum(
        value["segments"]["base_bit_values"]["size_bytes"]
        + value["segments"]["lower_bit_values"]["size_bytes"]
        for value in entries.values()
    )
    index_bytes = sum(
        value["segments"]["lower_group_mask"]["size_bytes"]
        for value in entries.values()
    )
    scale_bytes = sum(
        value["segments"]["scales"]["size_bytes"] for value in entries.values()
    )
    source_numel = sum(value["source_numel"] for value in entries.values())
    logical_parameter_count = sum(
        value["source_numel"] * len(value["logical_names"])
        for value in entries.values()
    )
    layer_types = list(model.config.layer_types)
    core = {
        "schema": CHECKPOINT_SCHEMA,
        "control_label": (
            "SERIALIZED-BYTE VALUE-QUANTIZATION CONTROL; "
            "NOT STRUCTURAL PARAMETER COMPACTION"
        ),
        "method": {
            "kind": CONTROL_KIND,
            "structural_parameter_count_compaction": False,
            "logical_neurons_removed": 0,
            "logical_heads_removed": 0,
            "logical_layers_removed": 0,
            "tokenizer_entries_removed": 0,
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
            "parent_traces_used": False,
            "value_only_approximation": True,
        },
        "target": {
            "requested_reduction_ppm": target.reduction_ppm,
            "requested_reduction": target.requested_reduction,
            "target_checkpoint_bytes": target_bytes,
            "ceiling_is_exact": True,
            "headroom_bytes": target_bytes - compressed_checkpoint_bytes,
            "base_bit_width": target.base_bit_width,
            "lower_bit_width": target.lower_bit_width,
            "bytes_saved_per_lower_group": target.bytes_saved_per_lower_group,
        },
        "topology": {
            "model_class": "transformers.Lfm2ForCausalLM",
            "model_type": "lfm2",
            "hidden_size": int(model.config.hidden_size),
            "logical_layer_count": int(model.config.num_hidden_layers),
            "layer_types": layer_types,
            "query_heads": int(model.config.num_attention_heads),
            "key_value_heads": int(model.config.num_key_value_heads),
            "vocabulary_size": int(model.config.vocab_size),
            "intermediate_size": int(model.config.intermediate_size),
            "all_logical_operators_preserved": True,
            "tied_input_output_embeddings": True,
        },
        "state": {
            "logical_to_canonical": dict(sorted(aliases.items())),
            "logical_tensor_count": len(aliases),
            "canonical_tensor_count": len(entries),
            "logical_parameter_count": logical_parameter_count,
            "canonical_source_parameter_count": source_numel,
            "tied_vocabulary_stored_once": (
                aliases.get("lm_head.weight") == "model.embed_tokens.weight"
            ),
            "parameter_count_reduction_claimed": False,
        },
        "quantization": {
            "scheme": "symmetric_per_group_mixed_integer_bitwidth",
            "group_size": target.group_size,
            "scale_dtype": "torch.float16",
            "selector_encoding": "one_lsb_first_bit_per_group",
            "selector_index_bytes_counted": True,
            "scale_bytes_counted": True,
            "padded_final_groups_counted": True,
            "distortion_allocator": (
                "lowest incremental normalized reconstruction SSE first"
            ),
            "total_group_count": total_group_count,
            "base_group_count": total_group_count - lower_group_count,
            "lower_group_count": lower_group_count,
        },
        "tensors": dict(entries),
        "artifacts": dict(artifacts),
        "storage": {
            "dense_source_tensor_bytes": dense_tensor_bytes,
            "shared_nonweight_artifact_bytes": shared_nonweight_bytes,
            "accounted_dense_reference_bytes": (
                dense_tensor_bytes + shared_nonweight_bytes
            ),
            "payload_header_bytes": len(PAYLOAD_MAGIC),
            "packed_value_bytes": packed_value_bytes,
            "group_selector_index_bytes": index_bytes,
            "scale_bytes": scale_bytes,
            "payload_bytes": payload_bytes,
            "compressed_checkpoint_bytes": compressed_checkpoint_bytes,
            "manifest_bytes_included_in_compressed_checkpoint_bytes": True,
            "manifest_bytes_derivation": (
                "compressed_checkpoint_bytes - payload_bytes - "
                "shared_nonweight_artifact_bytes"
            ),
            "all_checkpoint_files_counted": True,
            "hidden_dense_checkpoint_copy": False,
            "external_weight_sidecars": [],
        },
        "tokenizer": {
            "preserved_verbatim": True,
            "fingerprint": tokenizer_fingerprint,
            "artifacts": [value["file"] for value in artifacts["tokenizer"]],
        },
        "runtime": {
            "persistent_dense_weight_parameter_elements": 0,
            "zero_element_device_sentinel": True,
            "packed_buffers_only": True,
            "transient_operator_dequantization": True,
            "dense_weight_cache": False,
            "fresh_reload_required_for_certification": True,
        },
        "execution_audit": {
            "network_access": False,
            "hub_loader_used": False,
            "from_pretrained_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_forward_passes": 0,
            "parent_generate_invocations": 0,
        },
        "independent_holdout": {
            "required_suite": "sealed-independent-64-case-lfm2-holdout",
            "parent_responses_must_come_from_existing_sealed_cache": True,
            "new_parent_responses_allowed": False,
            "candidate_outputs_scored_by_cached_evaluator": True,
        },
        "serialization": {
            "payload_format": "neural-forge-packed-quantized-raw-v1",
            "payload_magic_hex": PAYLOAD_MAGIC.hex(),
            "python_pickle": False,
            "serialized_modules": False,
            "dense_safetensors": False,
            "byte_order": "little-bit-order-within-byte",
        },
        "standalone": True,
        "parent_model_required_for_reload": False,
        "external_weight_dependencies": [],
    }
    document = {**core, "manifest_sha256": canonical_sha256(core)}
    raw = canonical_json_bytes(document) + b"\n"
    return document, raw


def _stable_manifest(
    *,
    other_bytes: int,
    factory: Any,
) -> tuple[dict[str, Any], bytes]:
    total = other_bytes
    for _ in range(32):
        manifest_bytes = max(0, total - other_bytes)
        document, raw = factory(total, manifest_bytes)
        actual = other_bytes + len(raw)
        if actual == total:
            return document, raw
        total = actual
    raise Lfm2ValueFoldControlError("Manifest byte accounting did not converge.")


def _placeholder_artifacts(
    config_artifact: Mapping[str, Any],
    generation_artifact: Mapping[str, Any],
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
    payload_bytes: int,
) -> dict[str, Any]:
    return {
        "config": dict(config_artifact),
        "generation_config": dict(generation_artifact),
        "payload": {
            "file": PAYLOAD_FILENAME,
            "size_bytes": payload_bytes,
            "sha256": _PLACEHOLDER_DIGEST,
        },
        "tokenizer": [dict(value) for value in tokenizer_artifacts],
    }


def _budget_layout(
    *,
    model: nn.Module,
    target: Lfm2ValueFoldTarget,
    analyses: Sequence[_TensorAnalysis],
    aliases: Mapping[str, str],
    masks: Sequence[np.ndarray],
    config_artifact: Mapping[str, Any],
    generation_artifact: Mapping[str, Any],
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
    tokenizer_fingerprint: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], int]:
    entries, payload_bytes = _tensor_entry_layout(analyses, masks, target)
    shared_nonweight_bytes = (
        int(config_artifact["size_bytes"])
        + int(generation_artifact["size_bytes"])
        + sum(int(value["size_bytes"]) for value in tokenizer_artifacts)
    )
    dense_bytes = sum(value.source_nbytes for value in analyses)
    lower_count = sum(int(np.count_nonzero(value)) for value in masks)
    total_groups = sum(value.group_count for value in analyses)
    artifacts = _placeholder_artifacts(
        config_artifact,
        generation_artifact,
        tokenizer_artifacts,
        payload_bytes,
    )
    other_bytes = shared_nonweight_bytes + payload_bytes

    def factory(total: int, manifest_bytes: int):
        return _manifest_document(
            model=model,
            target=target,
            aliases=aliases,
            entries=entries,
            artifacts=artifacts,
            dense_tensor_bytes=dense_bytes,
            shared_nonweight_bytes=shared_nonweight_bytes,
            payload_bytes=payload_bytes,
            compressed_checkpoint_bytes=total,
            manifest_bytes=manifest_bytes,
            lower_group_count=lower_count,
            total_group_count=total_groups,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )

    document, raw = _stable_manifest(other_bytes=other_bytes, factory=factory)
    return document, raw, entries, payload_bytes


def _solve_budget(
    *,
    model: nn.Module,
    target: Lfm2ValueFoldTarget,
    analyses: Sequence[_TensorAnalysis],
    aliases: Mapping[str, str],
    order: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    config_artifact: Mapping[str, Any],
    generation_artifact: Mapping[str, Any],
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
    tokenizer_fingerprint: str,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    total_groups = len(order)
    lower_count = 0
    while True:
        masks = _selection_masks(
            analyses,
            offsets,
            order,
            lower_count,
        )
        document, _raw, _entries, _payload_bytes = _budget_layout(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            masks=masks,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        total = int(document["storage"]["compressed_checkpoint_bytes"])
        ceiling = int(document["target"]["target_checkpoint_bytes"])
        if total <= ceiling:
            break
        deficit = total - ceiling
        additional = max(
            1,
            math.ceil(deficit / target.bytes_saved_per_lower_group),
        )
        lower_count += additional
        if lower_count > total_groups:
            raise Lfm2ValueFoldControlBudgetError(
                "The fixed configuration/tokenizer/manifest overhead plus the "
                "lowest supported group widths cannot meet the byte ceiling."
            )
    if lower_count:
        previous_masks = _selection_masks(
            analyses,
            offsets,
            order,
            lower_count - 1,
        )
        previous, _raw, _entries, _payload_bytes = _budget_layout(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            masks=previous_masks,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if (
            previous["storage"]["compressed_checkpoint_bytes"]
            <= previous["target"]["target_checkpoint_bytes"]
        ):
            raise AssertionError(
                "The mixed-width allocator did not choose the minimal lower-group count."
            )
    return masks, document


def _copy_stream(
    source: Path,
    destination: Any,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            destination.write(block)
            digest.update(block)
            count += len(block)
    return count, digest.hexdigest()


def _write_quantized_payload(
    path: Path,
    analyses: Sequence[_TensorAnalysis],
    masks: Sequence[np.ndarray],
    target: Lfm2ValueFoldTarget,
    work_directory: Path,
) -> tuple[dict[str, Mapping[str, str]], dict[str, Any]]:
    segment_digests: dict[str, Mapping[str, str]] = {}
    reconstructed_sse = 0.0
    source_square_sum = 0.0
    with path.open("xb") as payload:
        payload.write(PAYLOAD_MAGIC)
        for tensor_index, (analysis, mask) in enumerate(
            zip(analyses, masks, strict=True)
        ):
            prefix = f"tensor-{tensor_index:04d}"
            high_path = work_directory / f".{prefix}.high.tmp"
            low_path = work_directory / f".{prefix}.low.tmp"
            scale_path = work_directory / f".{prefix}.scales.tmp"
            mask_bytes = _pack_mask(mask)
            payload.write(mask_bytes)
            mask_digest = _sha256_bytes(mask_bytes)
            try:
                with (
                    high_path.open("xb") as high_handle,
                    low_path.open("xb") as low_handle,
                    scale_path.open("xb") as scale_handle,
                ):
                    for start, end, _vs, _ve, values in _parameter_chunks(
                        analysis.parameter,
                        group_size=target.group_size,
                        chunk_groups=target.analysis_chunk_groups,
                    ):
                        chunk_mask = mask[start:end]
                        base_codes, base_scales, base_reconstructed = _quantize_groups(
                            values,
                            target.base_bit_width,
                        )
                        low_codes, low_scales, low_reconstructed = _quantize_groups(
                            values,
                            target.lower_bit_width,
                        )
                        high_rows = np.flatnonzero(~chunk_mask)
                        low_rows = np.flatnonzero(chunk_mask)
                        if len(high_rows):
                            high_handle.write(
                                pack_unsigned_codes(
                                    base_codes[torch.from_numpy(high_rows)],
                                    target.base_bit_width,
                                )
                            )
                        if len(low_rows):
                            low_handle.write(
                                pack_unsigned_codes(
                                    low_codes[torch.from_numpy(low_rows)],
                                    target.lower_bit_width,
                                )
                            )
                        chosen_scales = base_scales.clone()
                        if len(low_rows):
                            chosen_scales[torch.from_numpy(low_rows)] = low_scales[
                                torch.from_numpy(low_rows)
                            ]
                        scale_handle.write(
                            chosen_scales.contiguous()
                            .view(torch.uint8)
                            .numpy()
                            .tobytes()
                        )
                        chosen = base_reconstructed
                        if len(low_rows):
                            chosen = chosen.clone()
                            chosen[torch.from_numpy(low_rows)] = low_reconstructed[
                                torch.from_numpy(low_rows)
                            ]
                        reconstructed_sse += float((values - chosen).square().sum())
                        source_square_sum += float(values.square().sum())
                high_size, high_digest = _copy_stream(high_path, payload)
                low_size, low_digest = _copy_stream(low_path, payload)
                scale_size, scale_digest = _copy_stream(scale_path, payload)
            finally:
                for temporary in (high_path, low_path, scale_path):
                    if temporary.exists():
                        temporary.unlink()
            expected_high = (
                int(np.count_nonzero(~mask))
                * target.group_size
                * target.base_bit_width
                // 8
            )
            expected_low = (
                int(np.count_nonzero(mask))
                * target.group_size
                * target.lower_bit_width
                // 8
            )
            expected_scale = analysis.group_count * 2
            if (
                high_size != expected_high
                or low_size != expected_low
                or scale_size != expected_scale
            ):
                raise AssertionError("Packed segment size diverged from its plan.")
            segment_digests[analysis.canonical_name] = {
                "lower_group_mask": mask_digest,
                "base_bit_values": high_digest,
                "lower_bit_values": low_digest,
                "scales": scale_digest,
            }
        payload.flush()
        os.fsync(payload.fileno())
    return segment_digests, {
        "sum_squared_error": reconstructed_sse,
        "source_square_sum": source_square_sum,
        "relative_rmse": (
            math.sqrt(reconstructed_sse / source_square_sum)
            if source_square_sum > 0
            else 0.0
        ),
    }


def export_lfm2_value_fold_control(
    model: nn.Module,
    destination: str | os.PathLike[str],
    *,
    tokenizer_directory: str | os.PathLike[str],
    target_reduction: float,
    group_size: int = 256,
    analysis_chunk_groups: int = 4096,
) -> dict[str, Any]:
    """Export one exact-budget, standalone packed-value LFM2 control."""

    target = Lfm2ValueFoldTarget.from_reduction(
        target_reduction,
        group_size=group_size,
        analysis_chunk_groups=analysis_chunk_groups,
    )
    output = Path(destination).expanduser().absolute()
    if output.exists():
        raise Lfm2ValueFoldControlError(
            f"Refusing to replace existing destination {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_source = _safe_local_directory(
        tokenizer_directory,
        "tokenizer_directory",
    )
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        config, generation = _config_documents(model)
        _write_canonical_json(temporary / CONFIG_FILENAME, config)
        _write_canonical_json(
            temporary / GENERATION_CONFIG_FILENAME,
            generation,
        )
        tokenizer_artifacts = _copy_tokenizer_artifacts(
            tokenizer_source,
            temporary,
        )
        tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer_artifacts)
        config_artifact = _artifact(temporary / CONFIG_FILENAME)
        generation_artifact = _artifact(temporary / GENERATION_CONFIG_FILENAME)
        analyses, aliases = _analyze_model(model, target)
        order, offsets = _selection_order(analyses)
        masks, planned = _solve_budget(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            order=order,
            offsets=offsets,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        segment_digests, _error = _write_quantized_payload(
            temporary / PAYLOAD_FILENAME,
            analyses,
            masks,
            target,
            temporary,
        )
        entries, payload_bytes = _tensor_entry_layout(
            analyses,
            masks,
            target,
            segment_digests=segment_digests,
        )
        payload_artifact = _artifact(temporary / PAYLOAD_FILENAME)
        if payload_artifact["size_bytes"] != payload_bytes:
            raise AssertionError("Physical payload size differs from the plan.")
        shared_nonweight_bytes = (
            config_artifact["size_bytes"]
            + generation_artifact["size_bytes"]
            + sum(value["size_bytes"] for value in tokenizer_artifacts)
        )
        dense_bytes = sum(value.source_nbytes for value in analyses)
        lower_count = sum(int(np.count_nonzero(value)) for value in masks)
        total_groups = sum(value.group_count for value in analyses)
        artifacts = {
            "config": config_artifact,
            "generation_config": generation_artifact,
            "payload": payload_artifact,
            "tokenizer": list(tokenizer_artifacts),
        }
        other_bytes = shared_nonweight_bytes + payload_bytes

        def factory(total: int, manifest_bytes: int):
            return _manifest_document(
                model=model,
                target=target,
                aliases=aliases,
                entries=entries,
                artifacts=artifacts,
                dense_tensor_bytes=dense_bytes,
                shared_nonweight_bytes=shared_nonweight_bytes,
                payload_bytes=payload_bytes,
                compressed_checkpoint_bytes=total,
                manifest_bytes=manifest_bytes,
                lower_group_count=lower_count,
                total_group_count=total_groups,
                tokenizer_fingerprint=tokenizer_fingerprint,
            )

        manifest, manifest_raw = _stable_manifest(
            other_bytes=other_bytes,
            factory=factory,
        )
        if (
            manifest["storage"]["compressed_checkpoint_bytes"]
            > manifest["target"]["target_checkpoint_bytes"]
        ):
            raise Lfm2ValueFoldControlBudgetError(
                "Final hashes changed the manifest enough to exceed the target."
            )
        if (
            planned["quantization"]["lower_group_count"]
            != manifest["quantization"]["lower_group_count"]
        ):
            raise AssertionError("Final lower-group selection changed.")
        _write_new(temporary / MANIFEST_FILENAME, manifest_raw)
        observed_total = sum(
            path.stat().st_size for path in temporary.iterdir() if path.is_file()
        )
        if observed_total != manifest["storage"]["compressed_checkpoint_bytes"]:
            raise AssertionError(
                "On-disk checkpoint byte count differs from the manifest."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class PackedQuantizedTensor(nn.Module):
    """One canonical packed tensor store with transient dequantization."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, Any],
        lower_group_mask: Tensor,
        base_bit_values: Tensor,
        lower_bit_values: Tensor,
        scales: Tensor,
    ) -> None:
        super().__init__()
        self.shape = tuple(int(value) for value in metadata["shape"])
        self.source_numel = int(metadata["source_numel"])
        self.group_size = int(metadata["group_size"])
        self.group_count = int(metadata["group_count"])
        self.base_bit_width = int(metadata["base_bit_width"])
        self.lower_bit_width = int(metadata["lower_bit_width"])
        self.source_dtype_name = str(metadata["source_dtype"])
        if self.source_dtype_name not in _SOURCE_DTYPES:
            raise Lfm2ValueFoldControlFormatError(
                "Packed tensor source dtype is unsupported."
            )
        self.register_buffer(
            "lower_group_mask",
            lower_group_mask,
            persistent=False,
        )
        self.register_buffer(
            "base_bit_values",
            base_bit_values,
            persistent=False,
        )
        self.register_buffer(
            "lower_bit_values",
            lower_bit_values,
            persistent=False,
        )
        self.register_buffer("scales", scales, persistent=False)

    @property
    def source_dtype(self) -> torch.dtype:
        return _SOURCE_DTYPES[self.source_dtype_name]

    @property
    def packed_nbytes(self) -> int:
        return sum(
            int(value.numel()) * int(value.element_size())
            for value in (
                self.lower_group_mask,
                self.base_bit_values,
                self.lower_bit_values,
                self.scales,
            )
        )

    def dequantize(self) -> Tensor:
        mask = _unpack_mask(self.lower_group_mask, self.group_count)
        high_indices = torch.nonzero(~mask, as_tuple=False).reshape(-1)
        low_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        output = torch.empty(
            self.group_count,
            self.group_size,
            dtype=torch.float32,
            device=self.scales.device,
        )
        if high_indices.numel():
            codes = unpack_unsigned_codes(
                self.base_bit_values,
                value_count=int(high_indices.numel()) * self.group_size,
                bit_width=self.base_bit_width,
            ).reshape(int(high_indices.numel()), self.group_size)
            qmax = (1 << (self.base_bit_width - 1)) - 1
            values = (codes.to(torch.int16) - qmax).to(torch.float32)
            values *= self.scales.index_select(0, high_indices).to(torch.float32)[
                :, None
            ]
            output.index_copy_(0, high_indices, values)
        if low_indices.numel():
            codes = unpack_unsigned_codes(
                self.lower_bit_values,
                value_count=int(low_indices.numel()) * self.group_size,
                bit_width=self.lower_bit_width,
            ).reshape(int(low_indices.numel()), self.group_size)
            qmax = (1 << (self.lower_bit_width - 1)) - 1
            values = (codes.to(torch.int16) - qmax).to(torch.float32)
            values *= self.scales.index_select(0, low_indices).to(torch.float32)[
                :, None
            ]
            output.index_copy_(0, low_indices, values)
        return (
            output.reshape(-1)[: self.source_numel]
            .reshape(self.shape)
            .to(self.source_dtype)
        )


class PackedLinear(nn.Module):
    def __init__(
        self,
        weight: PackedQuantizedTensor,
        bias: PackedQuantizedTensor | None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_packed_weight", weight)
        object.__setattr__(self, "_packed_bias", bias)
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    @property
    def bias(self) -> Tensor | None:
        return None if self._packed_bias is None else self._packed_bias.dequantize()

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight, self.bias)


class PackedEmbedding(nn.Module):
    def __init__(
        self,
        weight: PackedQuantizedTensor,
        *,
        padding_idx: int | None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_packed_weight", weight)
        self.num_embeddings = weight.shape[0]
        self.embedding_dim = weight.shape[1]
        self.padding_idx = padding_idx

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.embedding(
            input_ids,
            self.weight,
            padding_idx=self.padding_idx,
        )


class PackedConv1d(nn.Module):
    def __init__(
        self,
        weight: PackedQuantizedTensor,
        bias: PackedQuantizedTensor | None,
        source: nn.Conv1d,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_packed_weight", weight)
        object.__setattr__(self, "_packed_bias", bias)
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    @property
    def bias(self) -> Tensor | None:
        return None if self._packed_bias is None else self._packed_bias.dequantize()

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv1d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PackedLfm2RMSNorm(nn.Module):
    def __init__(
        self,
        weight: PackedQuantizedTensor,
        *,
        variance_epsilon: float,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_packed_weight", weight)
        self.variance_epsilon = float(variance_epsilon)

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.to(torch.float32)
        variance = normalized.square().mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


def _resolve_owner(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    owner_path, separator, child_name = path.rpartition(".")
    if not separator:
        return model, path
    owner: nn.Module = model
    try:
        for part in owner_path.split("."):
            owner = getattr(owner, part)
    except (AttributeError, KeyError, IndexError) as exc:
        raise Lfm2ValueFoldControlFormatError(
            f"Module path {path!r} does not exist."
        ) from exc
    return owner, child_name


def _replace_modules_with_packed(
    model: nn.Module,
    stores: Mapping[str, PackedQuantizedTensor],
    aliases: Mapping[str, str],
) -> None:
    original_modules = tuple(model.named_modules())
    for module_name, module in reversed(original_modules):
        if not module_name:
            continue
        replacement: nn.Module | None = None
        if isinstance(module, nn.Linear):
            weight_name = f"{module_name}.weight"
            bias_name = f"{module_name}.bias"
            if weight_name not in aliases:
                raise Lfm2ValueFoldControlFormatError(
                    f"Linear {module_name!r} lacks a packed weight."
                )
            replacement = PackedLinear(
                stores[aliases[weight_name]],
                (stores[aliases[bias_name]] if module.bias is not None else None),
            )
        elif isinstance(module, nn.Embedding):
            weight_name = f"{module_name}.weight"
            if weight_name not in aliases:
                raise Lfm2ValueFoldControlFormatError(
                    f"Embedding {module_name!r} lacks a packed weight."
                )
            replacement = PackedEmbedding(
                stores[aliases[weight_name]],
                padding_idx=module.padding_idx,
            )
        elif isinstance(module, nn.Conv1d):
            weight_name = f"{module_name}.weight"
            bias_name = f"{module_name}.bias"
            if weight_name not in aliases:
                raise Lfm2ValueFoldControlFormatError(
                    f"Conv1d {module_name!r} lacks a packed weight."
                )
            replacement = PackedConv1d(
                stores[aliases[weight_name]],
                (stores[aliases[bias_name]] if module.bias is not None else None),
                module,
            )
        elif (
            type(module).__name__ == "Lfm2RMSNorm"
            and type(module).__module__ == _MODEL_MODULE
        ):
            weight_name = f"{module_name}.weight"
            if weight_name not in aliases:
                raise Lfm2ValueFoldControlFormatError(
                    f"RMSNorm {module_name!r} lacks a packed weight."
                )
            replacement = PackedLfm2RMSNorm(
                stores[aliases[weight_name]],
                variance_epsilon=module.variance_epsilon,
            )
        if replacement is not None:
            owner, child_name = _resolve_owner(model, module_name)
            setattr(owner, child_name, replacement)


def _checked_artifact(
    root: Path,
    value: Any,
    name: str,
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "file",
        "size_bytes",
        "sha256",
    }:
        raise Lfm2ValueFoldControlFormatError(f"{name} artifact metadata is malformed.")
    filename = value["file"]
    size = value["size_bytes"]
    digest = value["sha256"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise Lfm2ValueFoldControlFormatError(f"{name} artifact metadata is invalid.")
    path = root / filename
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != size
        or _sha256_file(path) != digest
    ):
        raise Lfm2ValueFoldControlFormatError(
            f"{name} artifact failed path, size, or SHA-256 validation."
        )
    return path


def _read_segment(
    payload: bytes,
    metadata: Mapping[str, Any],
    *,
    expected_size: int,
    name: str,
) -> Tensor:
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "byte_offset",
        "size_bytes",
        "sha256",
    }:
        raise Lfm2ValueFoldControlFormatError(
            f"Packed segment {name!r} metadata is malformed."
        )
    offset = metadata["byte_offset"]
    size = metadata["size_bytes"]
    digest = metadata["sha256"]
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < len(PAYLOAD_MAGIC)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size != expected_size
        or not isinstance(digest, str)
        or len(digest) != 64
        or offset + size > len(payload)
    ):
        raise Lfm2ValueFoldControlFormatError(
            f"Packed segment {name!r} range is invalid."
        )
    raw = payload[offset : offset + size]
    if _sha256_bytes(raw) != digest:
        raise Lfm2ValueFoldControlFormatError(
            f"Packed segment {name!r} failed SHA-256 validation."
        )
    if not raw:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def _manifest_and_artifacts(
    checkpoint_directory: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    root = _safe_local_directory(
        checkpoint_directory,
        "checkpoint_directory",
    )
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Lfm2ValueFoldControlFormatError(f"Missing {MANIFEST_FILENAME}.")
    manifest = _strict_json(manifest_path)
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise Lfm2ValueFoldControlFormatError(
            "Unsupported packed value-fold checkpoint schema."
        )
    claimed = manifest.get("manifest_sha256")
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or claimed != canonical_sha256(core)
    ):
        raise Lfm2ValueFoldControlFormatError("Manifest SHA-256 mismatch.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "config",
        "generation_config",
        "payload",
        "tokenizer",
    }:
        raise Lfm2ValueFoldControlFormatError(
            "Manifest artifact inventory is malformed."
        )
    paths = {
        "config": _checked_artifact(root, artifacts["config"], "config"),
        "generation_config": _checked_artifact(
            root,
            artifacts["generation_config"],
            "generation_config",
        ),
        "payload": _checked_artifact(root, artifacts["payload"], "payload"),
    }
    tokenizer = artifacts["tokenizer"]
    if not isinstance(tokenizer, list) or not tokenizer:
        raise Lfm2ValueFoldControlFormatError("Tokenizer artifact inventory is empty.")
    for index, value in enumerate(tokenizer):
        path = _checked_artifact(root, value, f"tokenizer[{index}]")
        paths[f"tokenizer:{path.name}"] = path
    expected_files = {MANIFEST_FILENAME, *(path.name for path in paths.values())}
    actual_files = {
        path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()
    }
    if actual_files != expected_files or any(
        path.is_dir() or path.is_symlink() for path in root.iterdir()
    ):
        raise Lfm2ValueFoldControlFormatError(
            "Checkpoint contains missing, unclaimed, nested, or symlink artifacts."
        )
    observed_total = sum(path.stat().st_size for path in root.iterdir())
    storage = manifest.get("storage")
    target = manifest.get("target")
    if (
        not isinstance(storage, Mapping)
        or not isinstance(target, Mapping)
        or storage.get("compressed_checkpoint_bytes") != observed_total
        or storage.get("payload_bytes") != paths["payload"].stat().st_size
        or storage.get("manifest_bytes_included_in_compressed_checkpoint_bytes")
        is not True
        or storage.get("all_checkpoint_files_counted") is not True
        or storage.get("hidden_dense_checkpoint_copy") is not False
        or storage.get("external_weight_sidecars") != []
        or target.get("ceiling_is_exact") is not True
        or observed_total > target.get("target_checkpoint_bytes", -1)
        or target.get("headroom_bytes")
        != target.get("target_checkpoint_bytes") - observed_total
    ):
        raise Lfm2ValueFoldControlFormatError(
            "Checkpoint physical-byte accounting is inconsistent."
        )
    dense = storage.get("dense_source_tensor_bytes")
    shared = storage.get("shared_nonweight_artifact_bytes")
    reference = storage.get("accounted_dense_reference_bytes")
    reduction_ppm = target.get("requested_reduction_ppm")
    if (
        isinstance(dense, bool)
        or not isinstance(dense, int)
        or dense <= 0
        or isinstance(shared, bool)
        or not isinstance(shared, int)
        or shared <= 0
        or reference != dense + shared
        or reduction_ppm not in _TARGET_BITS
        or target.get("target_checkpoint_bytes")
        != reference * (1_000_000 - reduction_ppm) // 1_000_000
    ):
        raise Lfm2ValueFoldControlFormatError(
            "Dense-reference or target arithmetic is inconsistent."
        )
    method = manifest.get("method")
    runtime = manifest.get("runtime")
    if (
        not isinstance(method, Mapping)
        or method.get("kind") != CONTROL_KIND
        or method.get("structural_parameter_count_compaction") is not False
        or method.get("training_used") is not False
        or method.get("optimizer_used") is not False
        or method.get("backward_used") is not False
        or not isinstance(runtime, Mapping)
        or runtime.get("persistent_dense_weight_parameter_elements") != 0
        or runtime.get("zero_element_device_sentinel") is not True
        or runtime.get("dense_weight_cache") is not False
    ):
        raise Lfm2ValueFoldControlFormatError(
            "Artifact is not the declared zero-training value-only control."
        )
    return root, manifest, paths


def load_lfm2_value_fold_control(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_checkpoint_bytes: int | None = None,
    max_packed_runtime_bytes: int | None = None,
) -> nn.Module:
    """Freshly reload a standalone packed LFM2 control without dense weights."""

    root, manifest, paths = _manifest_and_artifacts(checkpoint_directory)
    checkpoint_bytes = int(manifest["storage"]["compressed_checkpoint_bytes"])
    for limit, name, required in (
        (max_checkpoint_bytes, "max_checkpoint_bytes", checkpoint_bytes),
        (
            max_packed_runtime_bytes,
            "max_packed_runtime_bytes",
            int(manifest["storage"]["payload_bytes"]) - len(PAYLOAD_MAGIC),
        ),
    ):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise Lfm2ValueFoldControlError(
                f"{name} must be a nonnegative integer or None."
            )
        if limit is not None and required > limit:
            raise Lfm2ValueFoldControlBudgetError(
                f"Checkpoint requires {required} bytes for {name}, exceeding {limit}."
            )
    config_data = _strict_json(paths["config"])
    generation_data = _strict_json(paths["generation_config"])
    topology = manifest.get("topology")
    if (
        not isinstance(topology, Mapping)
        or topology.get("model_type") != "lfm2"
        or topology.get("all_logical_operators_preserved") is not True
        or topology.get("tied_input_output_embeddings") is not True
        or config_data.get("model_type") != "lfm2"
        or config_data.get("tie_word_embeddings") is not True
        or config_data.get("num_hidden_layers") != topology.get("logical_layer_count")
        or config_data.get("layer_types") != topology.get("layer_types")
        or config_data.get("_name_or_path") not in {"", None}
        or "auto_map" in config_data
    ):
        raise Lfm2ValueFoldControlFormatError(
            "Fresh LFM2 configuration differs from the sealed topology."
        )
    try:
        from transformers import GenerationConfig, Lfm2Config, Lfm2ForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2ValueFoldControlError(
            "Loading requires Transformers with LFM2 support."
        ) from exc
    config = Lfm2Config.from_dict(config_data)
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise Lfm2ValueFoldControlError(
            "device='meta' is not a materialized runtime target."
        )
    with torch.device("meta"):
        model = Lfm2ForCausalLM(config)
    logical_meta, aliases_meta = _model_parameters_meta(model)
    declared_aliases = manifest.get("state", {}).get("logical_to_canonical")
    if aliases_meta != declared_aliases:
        raise Lfm2ValueFoldControlFormatError(
            "Fresh LFM2 logical aliases differ from the manifest."
        )
    payload = paths["payload"].read_bytes()
    if not payload.startswith(PAYLOAD_MAGIC):
        raise Lfm2ValueFoldControlFormatError("Packed payload magic is invalid.")
    entries = manifest.get("tensors")
    canonical_names = set(aliases_meta.values())
    if not isinstance(entries, Mapping) or set(entries) != canonical_names:
        raise Lfm2ValueFoldControlFormatError(
            "Packed tensor inventory differs from the canonical state."
        )
    stores: dict[str, PackedQuantizedTensor] = {}
    expected_offset = len(PAYLOAD_MAGIC)
    for name in sorted(entries):
        entry = entries[name]
        if not isinstance(entry, Mapping):
            raise Lfm2ValueFoldControlFormatError(
                f"Packed tensor {name!r} metadata is malformed."
            )
        if entry.get("logical_names") != sorted(
            logical for logical, canonical in aliases_meta.items() if canonical == name
        ):
            raise Lfm2ValueFoldControlFormatError(
                f"Packed tensor {name!r} logical aliases are inconsistent."
            )
        meta_parameter = logical_meta[name]
        if (
            entry.get("shape") != list(meta_parameter.shape)
            or entry.get("source_numel") != int(meta_parameter.numel())
            or entry.get("source_dtype") not in _SOURCE_DTYPES
            or entry.get("group_size") != manifest["quantization"]["group_size"]
            or entry.get("base_bit_width") != manifest["target"]["base_bit_width"]
            or entry.get("lower_bit_width") != manifest["target"]["lower_bit_width"]
            or entry.get("scale_dtype") != "torch.float16"
            or entry.get("symmetric_zero_point") != 0
        ):
            raise Lfm2ValueFoldControlFormatError(
                f"Packed tensor {name!r} shape or quantization contract is invalid."
            )
        group_count = int(entry["group_count"])
        group_size = int(entry["group_size"])
        high_count = int(entry["base_group_count"])
        low_count = int(entry["lower_group_count"])
        if (
            group_count != (int(meta_parameter.numel()) + group_size - 1) // group_size
            or high_count + low_count != group_count
            or entry.get("padded_value_count") != group_count * group_size
        ):
            raise Lfm2ValueFoldControlFormatError(
                f"Packed tensor {name!r} group counts are inconsistent."
            )
        segments = entry.get("segments")
        if not isinstance(segments, Mapping) or set(segments) != {
            "lower_group_mask",
            "base_bit_values",
            "lower_bit_values",
            "scales",
        }:
            raise Lfm2ValueFoldControlFormatError(
                f"Packed tensor {name!r} segment inventory is invalid."
            )
        expected_sizes = {
            "lower_group_mask": (group_count + 7) // 8,
            "base_bit_values": (
                high_count * group_size * int(entry["base_bit_width"]) // 8
            ),
            "lower_bit_values": (
                low_count * group_size * int(entry["lower_bit_width"]) // 8
            ),
            "scales": group_count * 2,
        }
        loaded: dict[str, Tensor] = {}
        for role in (
            "lower_group_mask",
            "base_bit_values",
            "lower_bit_values",
            "scales",
        ):
            segment = segments[role]
            if segment.get("byte_offset") != expected_offset:
                raise Lfm2ValueFoldControlFormatError(
                    "Packed tensor segments must be contiguous and ordered."
                )
            loaded[role] = _read_segment(
                payload,
                segment,
                expected_size=expected_sizes[role],
                name=f"{name}:{role}",
            )
            expected_offset += expected_sizes[role]
        scale_bytes = loaded["scales"]
        scales = scale_bytes.view(torch.float16)
        if (
            scales.numel() != group_count
            or not bool(torch.isfinite(scales).all())
            or bool((scales <= 0).any())
        ):
            raise Lfm2ValueFoldControlFormatError(
                f"Packed tensor {name!r} scales are invalid."
            )
        mask = _unpack_mask(loaded["lower_group_mask"], group_count)
        if int(mask.sum()) != low_count:
            raise Lfm2ValueFoldControlFormatError(
                f"Packed tensor {name!r} selector count is invalid."
            )
        stores[name] = PackedQuantizedTensor(
            metadata=entry,
            lower_group_mask=loaded["lower_group_mask"].to(target_device),
            base_bit_values=loaded["base_bit_values"].to(target_device),
            lower_bit_values=loaded["lower_bit_values"].to(target_device),
            scales=scales.clone().to(target_device),
        )
    if expected_offset != len(payload):
        raise Lfm2ValueFoldControlFormatError(
            "Packed payload has unclaimed trailing bytes."
        )
    runtime_dtype = stores[sorted(stores)[0]].source_dtype
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(
        config,
        device=target_device,
    ).to(dtype=runtime_dtype)
    _replace_modules_with_packed(model, stores, aliases_meta)
    registry = nn.ModuleList([stores[name] for name in sorted(stores)])
    model.add_module("neural_forge_packed_registry", registry)
    sentinel_dtype = runtime_dtype
    model.register_parameter(
        "neural_forge_device_sentinel",
        nn.Parameter(
            torch.empty(0, device=target_device, dtype=sentinel_dtype),
            requires_grad=False,
        ),
    )
    model.generation_config = GenerationConfig.from_dict(generation_data)
    model.eval()
    parameters = tuple(model.named_parameters())
    if tuple(name for name, _value in parameters) != (
        "neural_forge_device_sentinel",
    ) or any(parameter.numel() for _name, parameter in parameters):
        raise Lfm2ValueFoldControlFormatError(
            "Packed runtime retained a persistent dense weight Parameter."
        )
    packed_runtime_bytes = sum(store.packed_nbytes for store in stores.values())
    expected_runtime_bytes = len(payload) - len(PAYLOAD_MAGIC)
    if packed_runtime_bytes != expected_runtime_bytes:
        raise Lfm2ValueFoldControlFormatError(
            "Packed runtime buffers differ from physical payload accounting."
        )
    if (
        model.model.embed_tokens._packed_weight is not model.lm_head._packed_weight
        or model.model.embed_tokens._packed_weight
        is not stores["model.embed_tokens.weight"]
    ):
        raise Lfm2ValueFoldControlFormatError(
            "Fresh runtime did not restore tied vocabulary storage once."
        )
    setattr(model, "neural_forge_checkpoint_manifest", manifest)
    setattr(
        model,
        "neural_forge_load_report",
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "constructed_on_meta": True,
            "persistent_dense_weight_parameter_elements": 0,
            "zero_element_device_sentinel": True,
            "canonical_packed_tensor_count": len(stores),
            "packed_runtime_bytes": packed_runtime_bytes,
            "checkpoint_bytes": checkpoint_bytes,
            "tied_vocabulary_stored_once": True,
            "from_pretrained_used": False,
            "network_access": False,
        },
    )
    return model


def _model_parameters_meta(
    model: nn.Module,
) -> tuple[dict[str, nn.Parameter], dict[str, str]]:
    logical = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    if not logical:
        raise Lfm2ValueFoldControlFormatError(
            "Fresh meta LFM2 has no logical Parameters."
        )
    groups: dict[int, list[str]] = {}
    for name, parameter in logical.items():
        groups.setdefault(id(parameter), []).append(name)
    aliases: dict[str, str] = {}
    for names in groups.values():
        canonical = (
            "model.embed_tokens.weight"
            if "model.embed_tokens.weight" in names
            else sorted(names)[0]
        )
        for name in names:
            aliases[name] = canonical
    return logical, aliases


def load_lfm2_value_fold_tokenizer(
    checkpoint_directory: str | os.PathLike[str],
) -> Any:
    """Load the verbatim copied tokenizer directly, with no hub loader."""

    root, manifest, _paths = _manifest_and_artifacts(checkpoint_directory)
    from transformers import PreTrainedTokenizerFast

    tokenizer_path = root / "tokenizer.json"
    kwargs: dict[str, Any] = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = root / name
        if path.is_file():
            value = _strict_json(path)
            kwargs.update(value)
    for name in (
        "tokenizer_class",
        "backend",
        "is_local",
        "use_fast",
        "sp_model_kwargs",
    ):
        kwargs.pop(name, None)
    for name in (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ):
        value = kwargs.get(name)
        if isinstance(value, Mapping) and "content" in value:
            kwargs[name] = str(value["content"])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        **kwargs,
    )
    chat_template = root / "chat_template.jinja"
    if chat_template.is_file():
        tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
    setattr(
        tokenizer,
        "neural_forge_tokenizer_fingerprint",
        manifest["tokenizer"]["fingerprint"],
    )
    return tokenizer


def inspect_lfm2_value_fold_control(
    checkpoint_directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Authenticate an artifact and return its exact accounting summary."""

    root, manifest, _paths = _manifest_and_artifacts(checkpoint_directory)
    return {
        "checkpoint_directory": str(root),
        "manifest_sha256": manifest["manifest_sha256"],
        "control_label": manifest["control_label"],
        "requested_reduction": manifest["target"]["requested_reduction"],
        "realized_serialized_reduction": (
            1.0
            - manifest["storage"]["compressed_checkpoint_bytes"]
            / manifest["storage"]["accounted_dense_reference_bytes"]
        ),
        "target_checkpoint_bytes": manifest["target"]["target_checkpoint_bytes"],
        "compressed_checkpoint_bytes": manifest["storage"][
            "compressed_checkpoint_bytes"
        ],
        "headroom_bytes": manifest["target"]["headroom_bytes"],
        "packed_value_bytes": manifest["storage"]["packed_value_bytes"],
        "scale_bytes": manifest["storage"]["scale_bytes"],
        "group_selector_index_bytes": manifest["storage"]["group_selector_index_bytes"],
        "manifest_bytes": (
            manifest["storage"]["compressed_checkpoint_bytes"]
            - manifest["storage"]["payload_bytes"]
            - manifest["storage"]["shared_nonweight_artifact_bytes"]
        ),
        "shared_nonweight_artifact_bytes": manifest["storage"][
            "shared_nonweight_artifact_bytes"
        ],
        "structural_parameter_count_compaction": False,
        "logical_neurons_removed": 0,
        "logical_heads_removed": 0,
        "logical_layers_removed": 0,
        "tied_vocabulary_stored_once": manifest["state"]["tied_vocabulary_stored_once"],
        "no_hidden_dense_copy": True,
        "all_checkpoint_files_counted": True,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_FILENAME",
    "CONTROL_KIND",
    "GENERATION_CONFIG_FILENAME",
    "Lfm2ValueFoldControlBudgetError",
    "Lfm2ValueFoldControlError",
    "Lfm2ValueFoldControlFormatError",
    "Lfm2ValueFoldTarget",
    "MANIFEST_FILENAME",
    "PAYLOAD_FILENAME",
    "PackedConv1d",
    "PackedEmbedding",
    "PackedLfm2RMSNorm",
    "PackedLinear",
    "PackedQuantizedTensor",
    "export_lfm2_value_fold_control",
    "inspect_lfm2_value_fold_control",
    "load_lfm2_value_fold_control",
    "load_lfm2_value_fold_tokenizer",
    "pack_unsigned_codes",
    "unpack_unsigned_codes",
]
