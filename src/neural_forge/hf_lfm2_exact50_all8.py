"""Exact-50% standalone LFM2 checkpoint with every logical value at eight bits.

This is a serialization/value-compaction control, not structural neuron
compaction.  It preserves every logical tensor, layer, head, channel, neuron,
and tied vocabulary role.  Every stored quantized value uses symmetric
groupwise int8 (represented as unsigned codes); no value uses seven bits.

The format reaches the full-directory 50% ceiling losslessly around those
eight-bit codes:

* a deterministic storage-class mask splits codes into a zlib-compressed
  stream and a raw fallback stream;
* both streams contain eight-bit codes;
* float16 scales and the storage-class mask are compressed losslessly when
  that is smaller than raw storage;
* config, generation config, and tokenizer files live in one authenticated
  compressed package and reload without a hub loader;
* the runtime reconstructs one ordered uint8 code buffer per tensor, so the
  storage-class mask is not retained at runtime.

The exporter fails closed unless both the complete on-disk directory and the
persistent packed runtime fit the exact same 50% byte ceiling.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_lfm2_value_fold_control import (
    CONFIG_FILENAME,
    GENERATION_CONFIG_FILENAME,
    TOKENIZER_ARTIFACTS,
    Lfm2ValueFoldControlBudgetError,
    Lfm2ValueFoldControlError,
    Lfm2ValueFoldControlFormatError,
    Lfm2ValueFoldTarget,
    _analyze_model,
    _config_documents,
    _model_parameters_meta,
    _pack_mask,
    _parameter_chunks,
    _quantize_groups,
    _replace_modules_with_packed,
    _safe_local_directory,
    _selection_masks,
    _selection_order,
    _sha256_bytes,
    _sha256_file,
    _tokenizer_fingerprint,
    _unpack_mask,
    _validate_source_lfm2,
    _write_new,
)


CHECKPOINT_SCHEMA = "neural-forge-lfm2-exact50-all8-zlib-v1"
MANIFEST_FILENAME = "neural_forge_lfm2_exact50_all8_manifest.json"
PAYLOAD_FILENAME = "model.nf8z"
PACKAGE_FILENAME = "standalone.nfpkgz"
PAYLOAD_MAGIC = b"NF_LFM2_ALL8_V1\n"
PACKAGE_MAGIC = b"NF_LFM2_PACKAGE_V1\n"
METHOD_KIND = "uniform_int8_lossless_container_exact50"
GROUP_SIZE_DEFAULT = 256
_DIGEST_LENGTH = 64
_SOURCE_DTYPES = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}
_SEGMENT_ROLES = ("storage_mask", "compressed_class_codes", "raw_class_codes", "scales")
_PACKAGE_MEMBER_ORDER = (
    CONFIG_FILENAME,
    GENERATION_CONFIG_FILENAME,
    *TOKENIZER_ARTIFACTS,
)
_SEALED_RAW_CLASS_GROUPS = 138_815
_SEALED_TOTAL_GROUPS = 897_248


class Lfm2Exact50All8Error(Lfm2ValueFoldControlError):
    """The exact50 all8 source or configuration is invalid."""


class Lfm2Exact50All8BudgetError(Lfm2ValueFoldControlBudgetError):
    """The exact50 all8 artifact or runtime cannot meet the ceiling."""


class Lfm2Exact50All8FormatError(Lfm2ValueFoldControlFormatError):
    """The exact50 all8 checkpoint is incomplete or corrupted."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _strict_json_bytes(raw: bytes, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2Exact50All8FormatError(
                    f"Duplicate JSON key {key!r} in {name}."
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise Lfm2Exact50All8FormatError(
            f"Non-finite JSON constant {value!r} in {name}."
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except Lfm2Exact50All8FormatError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact50All8FormatError(
            f"Could not read strict JSON member {name!r}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2Exact50All8FormatError(
            f"JSON member {name!r} must contain an object."
        )
    return value


def _strict_json_file(path: Path) -> dict[str, Any]:
    try:
        return _strict_json_bytes(path.read_bytes(), path.name)
    except OSError as exc:
        raise Lfm2Exact50All8FormatError(
            f"Could not read {path.name!r}."
        ) from exc


def _source_documents(
    model: nn.Module,
    tokenizer_directory: Path,
) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...], str]:
    config, generation = _config_documents(model)
    members = {
        CONFIG_FILENAME: canonical_json_bytes(config) + b"\n",
        GENERATION_CONFIG_FILENAME: canonical_json_bytes(generation) + b"\n",
    }
    tokenizer_artifacts: list[dict[str, Any]] = []
    for name in TOKENIZER_ARTIFACTS:
        path = tokenizer_directory / name
        if path.is_symlink():
            raise Lfm2Exact50All8Error(
                f"Tokenizer artifact cannot be a symlink: {path}."
            )
        if not path.is_file():
            continue
        raw = path.read_bytes()
        members[name] = raw
        tokenizer_artifacts.append(
            {
                "file": name,
                "size_bytes": len(raw),
                "sha256": _digest_bytes(raw),
            }
        )
    if (
        not tokenizer_artifacts
        or tokenizer_artifacts[0]["file"] != "tokenizer.json"
    ):
        raise Lfm2Exact50All8Error(
            "The local tokenizer package lacks tokenizer.json."
        )
    fingerprint = _tokenizer_fingerprint(tokenizer_artifacts)
    return members, tuple(tokenizer_artifacts), fingerprint


def _choose_lossless_codec(raw: bytes) -> tuple[str, bytes]:
    compressed = zlib.compress(raw, level=9)
    if len(compressed) < len(raw):
        return "zlib-9", compressed
    return "raw", raw


def _write_package(
    path: Path,
    members: Mapping[str, bytes],
) -> tuple[dict[str, Any], int]:
    entries: dict[str, Any] = {}
    with path.open("xb") as handle:
        handle.write(PACKAGE_MAGIC)
        offset = len(PACKAGE_MAGIC)
        for name in _PACKAGE_MEMBER_ORDER:
            if name not in members:
                continue
            raw = members[name]
            codec, stored = _choose_lossless_codec(raw)
            handle.write(stored)
            entries[name] = {
                "byte_offset": offset,
                "stored_size_bytes": len(stored),
                "stored_sha256": _digest_bytes(stored),
                "raw_size_bytes": len(raw),
                "raw_sha256": _digest_bytes(raw),
                "codec": codec,
            }
            offset += len(stored)
        handle.flush()
        os.fsync(handle.fileno())
    if set(entries) != set(members):
        raise AssertionError("Standalone package member inventory changed.")
    return entries, sum(len(value) for value in members.values())


def _stream_digest(path: Path) -> str:
    return _sha256_file(path)


def _compress_file(source: Path, destination: Path) -> None:
    compressor = zlib.compressobj(level=9)
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            encoded = compressor.compress(block)
            if encoded:
                writer.write(encoded)
        writer.write(compressor.flush())
        writer.flush()
        os.fsync(writer.fileno())


def _copy_file(source: Path, destination: Any) -> None:
    with source.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            destination.write(block)


def _encode_raw_segment(
    source: Path,
    work_directory: Path,
) -> tuple[str, Path]:
    compressed = work_directory / f".{source.name}.zlib"
    _compress_file(source, compressed)
    if compressed.stat().st_size < source.stat().st_size:
        return "zlib-9", compressed
    compressed.unlink()
    return "raw", source


def _write_all8_raw_segments(
    analyses: Sequence[Any],
    masks: Sequence[np.ndarray],
    *,
    group_size: int,
    analysis_chunk_groups: int,
    work_directory: Path,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    paths = {
        role: work_directory / f".{role}.raw" for role in _SEGMENT_ROLES
    }
    tensor_entries: dict[str, Any] = {}
    offsets = {role: 0 for role in _SEGMENT_ROLES}
    reconstructed_sse = 0.0
    source_square_sum = 0.0
    with (
        paths["storage_mask"].open("xb") as mask_handle,
        paths["compressed_class_codes"].open("xb") as high_handle,
        paths["raw_class_codes"].open("xb") as low_handle,
        paths["scales"].open("xb") as scale_handle,
    ):
        for analysis, mask in zip(analyses, masks, strict=True):
            mask_raw = _pack_mask(mask)
            mask_handle.write(mask_raw)
            high_count = int(np.count_nonzero(~mask))
            low_count = int(np.count_nonzero(mask))
            entry = {
                "logical_names": list(analysis.logical_names),
                "shape": list(analysis.shape),
                "source_dtype": analysis.source_dtype,
                "source_numel": analysis.source_numel,
                "source_nbytes": analysis.source_nbytes,
                "source_sha256": analysis.source_sha256,
                "group_size": group_size,
                "group_count": analysis.group_count,
                "code_bit_width": 8,
                "scale_dtype": "torch.float16",
                "symmetric_zero_point": 0,
                "padded_value_count": analysis.group_count * group_size,
                "storage_class_counts": {
                    "compressed": high_count,
                    "raw_fallback": low_count,
                },
                "raw_segment_ranges": {
                    "storage_mask": {
                        "byte_offset": offsets["storage_mask"],
                        "size_bytes": len(mask_raw),
                    },
                    "compressed_class_codes": {
                        "byte_offset": offsets["compressed_class_codes"],
                        "size_bytes": high_count * group_size,
                    },
                    "raw_class_codes": {
                        "byte_offset": offsets["raw_class_codes"],
                        "size_bytes": low_count * group_size,
                    },
                    "scales": {
                        "byte_offset": offsets["scales"],
                        "size_bytes": analysis.group_count * 2,
                    },
                },
            }
            offsets["storage_mask"] += len(mask_raw)
            offsets["compressed_class_codes"] += high_count * group_size
            offsets["raw_class_codes"] += low_count * group_size
            offsets["scales"] += analysis.group_count * 2
            for start, end, _vs, _ve, values in _parameter_chunks(
                analysis.parameter,
                group_size=group_size,
                chunk_groups=analysis_chunk_groups,
            ):
                codes, scales, reconstructed = _quantize_groups(values, 8)
                chunk_mask = mask[start:end]
                high_rows = np.flatnonzero(~chunk_mask)
                low_rows = np.flatnonzero(chunk_mask)
                if len(high_rows):
                    high_handle.write(
                        codes[torch.from_numpy(high_rows)]
                        .contiguous()
                        .numpy()
                        .tobytes()
                    )
                if len(low_rows):
                    low_handle.write(
                        codes[torch.from_numpy(low_rows)]
                        .contiguous()
                        .numpy()
                        .tobytes()
                    )
                scale_handle.write(
                    scales.contiguous().view(torch.uint8).numpy().tobytes()
                )
                reconstructed_sse += float((values - reconstructed).square().sum())
                source_square_sum += float(values.square().sum())
            entry["sum_squared_error"] = float(analysis.base_sse.sum())
            entry["source_square_sum"] = float(analysis.source_square_sum.sum())
            tensor_entries[analysis.canonical_name] = entry
        for handle in (mask_handle, high_handle, low_handle, scale_handle):
            handle.flush()
            os.fsync(handle.fileno())
    for role, expected in offsets.items():
        if paths[role].stat().st_size != expected:
            raise AssertionError(f"Raw all8 segment {role!r} changed size.")
    error = {
        "sum_squared_error": reconstructed_sse,
        "source_square_sum": source_square_sum,
        "relative_rmse": (
            math.sqrt(reconstructed_sse / source_square_sum)
            if source_square_sum > 0
            else 0.0
        ),
    }
    return paths, tensor_entries, error


def _write_payload(
    path: Path,
    raw_paths: Mapping[str, Path],
    work_directory: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    encoded: dict[str, tuple[str, Path]] = {}
    for role in _SEGMENT_ROLES:
        encoded[role] = _encode_raw_segment(raw_paths[role], work_directory)
    entries: dict[str, Any] = {}
    raw_sizes: dict[str, int] = {}
    with path.open("xb") as handle:
        handle.write(PAYLOAD_MAGIC)
        offset = len(PAYLOAD_MAGIC)
        for role in _SEGMENT_ROLES:
            codec, stored_path = encoded[role]
            raw_path = raw_paths[role]
            _copy_file(stored_path, handle)
            stored_size = stored_path.stat().st_size
            raw_size = raw_path.stat().st_size
            entries[role] = {
                "byte_offset": offset,
                "stored_size_bytes": stored_size,
                "stored_sha256": _stream_digest(stored_path),
                "raw_size_bytes": raw_size,
                "raw_sha256": _stream_digest(raw_path),
                "codec": codec,
            }
            raw_sizes[role] = raw_size
            offset += stored_size
        handle.flush()
        os.fsync(handle.fileno())
    for codec, stored_path in encoded.values():
        if codec == "zlib-9" and stored_path.exists():
            stored_path.unlink()
    return entries, raw_sizes


def _manifest_document(
    *,
    model: nn.Module,
    aliases: Mapping[str, str],
    tensor_entries: Mapping[str, Any],
    payload_segments: Mapping[str, Any],
    package_members: Mapping[str, Any],
    payload_artifact: Mapping[str, Any],
    package_artifact: Mapping[str, Any],
    dense_source_bytes: int,
    source_nonweight_bytes: int,
    manifest_bytes: int,
    checkpoint_bytes: int,
    raw_sizes: Mapping[str, int],
    tokenizer_fingerprint: str,
    error: Mapping[str, Any],
    auxiliary_runtime_buffer_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    reference = dense_source_bytes + source_nonweight_bytes
    ceiling = reference // 2
    source_numel = sum(
        int(value["source_numel"]) for value in tensor_entries.values()
    )
    logical_parameter_count = sum(
        int(value["source_numel"]) * len(value["logical_names"])
        for value in tensor_entries.values()
    )
    total_groups = sum(
        int(value["group_count"]) for value in tensor_entries.values()
    )
    raw_code_bytes = (
        int(raw_sizes["compressed_class_codes"])
        + int(raw_sizes["raw_class_codes"])
    )
    packed_weight_runtime_bytes = raw_code_bytes + int(raw_sizes["scales"])
    materialized_runtime_tensor_bytes = (
        packed_weight_runtime_bytes + auxiliary_runtime_buffer_bytes
    )
    core = {
        "schema": CHECKPOINT_SCHEMA,
        "control_label": (
            "EXACT-50% ALL-8-BIT SERIALIZATION CONTROL; "
            "NOT STRUCTURAL PARAMETER COMPACTION"
        ),
        "method": {
            "kind": METHOD_KIND,
            "structural_parameter_count_compaction": False,
            "logical_neurons_removed": 0,
            "logical_heads_removed": 0,
            "logical_layers_removed": 0,
            "tokenizer_entries_removed": 0,
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
            "parent_traces_used": False,
            "all_logical_weight_codes_are_8bit": True,
            "seven_bit_weight_codes": 0,
            "lossy_container_compression": False,
        },
        "target": {
            "requested_reduction_ppm": 500_000,
            "requested_reduction": 0.5,
            "target_checkpoint_bytes": ceiling,
            "ceiling_is_exact": True,
            "headroom_bytes": ceiling - checkpoint_bytes,
            "runtime_headroom_bytes": ceiling - materialized_runtime_tensor_bytes,
        },
        "topology": {
            "model_class": "transformers.Lfm2ForCausalLM",
            "model_type": "lfm2",
            "hidden_size": int(model.config.hidden_size),
            "logical_layer_count": int(model.config.num_hidden_layers),
            "layer_types": list(model.config.layer_types),
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
            "canonical_tensor_count": len(tensor_entries),
            "logical_parameter_count": logical_parameter_count,
            "canonical_source_parameter_count": source_numel,
            "tied_vocabulary_stored_once": (
                aliases.get("lm_head.weight") == "model.embed_tokens.weight"
            ),
            "parameter_count_reduction_claimed": False,
        },
        "quantization": {
            "scheme": "symmetric_per_group_uniform_int8",
            "group_size": int(
                next(iter(tensor_entries.values()))["group_size"]
            ),
            "code_bit_width": 8,
            "minimum_code_bit_width": 8,
            "maximum_code_bit_width": 8,
            "seven_bit_group_count": 0,
            "scale_dtype": "torch.float16",
            "total_group_count": total_groups,
            "padded_final_groups_counted": True,
            "sum_squared_error": float(error["sum_squared_error"]),
            "source_square_sum": float(error["source_square_sum"]),
            "relative_rmse": float(error["relative_rmse"]),
        },
        "storage_allocator": {
            "selector_semantics": (
                "lossless storage codec class only; never quantization bit width"
            ),
            "selection_source": (
                "legacy exact50 normalized-SSE order reused only to establish "
                "a conservative compressed/raw storage split"
            ),
            "compressed_class_group_count": sum(
                int(value["storage_class_counts"]["compressed"])
                for value in tensor_entries.values()
            ),
            "raw_fallback_group_count": sum(
                int(value["storage_class_counts"]["raw_fallback"])
                for value in tensor_entries.values()
            ),
            "mask_persistent_at_runtime": False,
        },
        "tensors": dict(tensor_entries),
        "payload_segments": dict(payload_segments),
        "package_members": dict(package_members),
        "artifacts": {
            "payload": dict(payload_artifact),
            "standalone_package": dict(package_artifact),
        },
        "storage": {
            "dense_source_tensor_bytes": dense_source_bytes,
            "source_nonweight_artifact_bytes": source_nonweight_bytes,
            "accounted_dense_reference_bytes": reference,
            "raw_8bit_value_bytes": raw_code_bytes,
            "raw_float16_scale_bytes": int(raw_sizes["scales"]),
            "raw_storage_selector_bytes": int(raw_sizes["storage_mask"]),
            "packed_weight_runtime_bytes": packed_weight_runtime_bytes,
            "auxiliary_runtime_buffer_bytes": auxiliary_runtime_buffer_bytes,
            "materialized_runtime_tensor_bytes": (
                materialized_runtime_tensor_bytes
            ),
            "packed_runtime_bytes": materialized_runtime_tensor_bytes,
            "payload_bytes": int(payload_artifact["size_bytes"]),
            "standalone_package_bytes": int(package_artifact["size_bytes"]),
            "manifest_bytes": manifest_bytes,
            "compressed_checkpoint_bytes": checkpoint_bytes,
            "all_checkpoint_files_counted": True,
            "manifest_bytes_included_in_compressed_checkpoint_bytes": True,
            "runtime_bytes_under_same_ceiling": (
                materialized_runtime_tensor_bytes <= ceiling
            ),
            "hidden_dense_checkpoint_copy": False,
            "external_weight_sidecars": [],
        },
        "tokenizer": {
            "preserved_verbatim_after_lossless_decompression": True,
            "standalone_compressed_loading": True,
            "fingerprint": tokenizer_fingerprint,
            "artifacts": [
                name for name in TOKENIZER_ARTIFACTS if name in package_members
            ],
        },
        "runtime": {
            "persistent_dense_weight_parameter_elements": 0,
            "zero_element_device_sentinel": True,
            "ordered_uint8_code_buffers": True,
            "storage_class_mask_retained": False,
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
            "payload_magic_hex": PAYLOAD_MAGIC.hex(),
            "package_magic_hex": PACKAGE_MAGIC.hex(),
            "lossless_codecs": ["raw", "zlib-9"],
            "python_pickle": False,
            "serialized_modules": False,
            "dense_safetensors": False,
        },
        "standalone": True,
        "parent_model_required_for_reload": False,
        "external_weight_dependencies": [],
    }
    document = {**core, "manifest_sha256": canonical_sha256(core)}
    return document, canonical_json_bytes(document) + b"\n"


def _stable_manifest(*, other_bytes: int, factory: Any) -> tuple[dict[str, Any], bytes]:
    total = other_bytes
    for _ in range(32):
        manifest_bytes = max(0, total - other_bytes)
        document, raw = factory(total, manifest_bytes)
        actual = other_bytes + len(raw)
        if actual == total:
            return document, raw
        total = actual
    raise Lfm2Exact50All8Error("Manifest byte accounting did not converge.")


def _auxiliary_runtime_buffer_bytes(model: nn.Module) -> int:
    """Count the source runtime buffers the packed shell must reproduce exactly."""

    source = getattr(getattr(model, "model", None), "rotary_emb", None)
    if not isinstance(source, nn.Module):
        raise Lfm2Exact50All8Error("Source LFM2 lacks its rotary embedding.")
    seen: set[int] = set()
    total = 0
    for _name, value in source.named_buffers(recurse=True, remove_duplicate=True):
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.numel()) * int(value.element_size())
    return total


def export_lfm2_exact50_all8(
    model: nn.Module,
    destination: str | os.PathLike[str],
    *,
    tokenizer_directory: str | os.PathLike[str],
    group_size: int = GROUP_SIZE_DEFAULT,
    analysis_chunk_groups: int = 4096,
) -> dict[str, Any]:
    """Export a standalone all-int8 checkpoint under the exact 50% ceiling."""

    target = Lfm2ValueFoldTarget(
        500_000,
        group_size=group_size,
        analysis_chunk_groups=analysis_chunk_groups,
    )
    _validate_source_lfm2(model)
    output = Path(destination).expanduser().absolute()
    if output.exists():
        raise Lfm2Exact50All8Error(
            f"Refusing to replace existing destination {output}."
        )
    tokenizer_source = _safe_local_directory(
        tokenizer_directory, "tokenizer_directory"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        members, _tokenizer_artifacts, tokenizer_fingerprint = _source_documents(
            model, tokenizer_source
        )
        package_members, source_nonweight_bytes = _write_package(
            temporary / PACKAGE_FILENAME, members
        )
        package_artifact = _artifact(temporary / PACKAGE_FILENAME)
        analyses, aliases = _analyze_model(model, target)
        order, offsets = _selection_order(analyses)
        raw_class_count = math.ceil(
            len(order) * _SEALED_RAW_CLASS_GROUPS / _SEALED_TOTAL_GROUPS
        )
        masks = _selection_masks(
            analyses,
            offsets,
            order,
            raw_class_count,
        )
        raw_paths, tensor_entries, error = _write_all8_raw_segments(
            analyses,
            masks,
            group_size=group_size,
            analysis_chunk_groups=analysis_chunk_groups,
            work_directory=temporary,
        )
        payload_segments, raw_sizes = _write_payload(
            temporary / PAYLOAD_FILENAME,
            raw_paths,
            temporary,
        )
        for path in raw_paths.values():
            path.unlink()
        payload_artifact = _artifact(temporary / PAYLOAD_FILENAME)
        dense_source_bytes = sum(value.source_nbytes for value in analyses)
        reference = dense_source_bytes + source_nonweight_bytes
        ceiling = reference // 2
        packed_weight_runtime_bytes = (
            raw_sizes["compressed_class_codes"]
            + raw_sizes["raw_class_codes"]
            + raw_sizes["scales"]
        )
        auxiliary_runtime_buffer_bytes = _auxiliary_runtime_buffer_bytes(model)
        materialized_runtime_tensor_bytes = (
            packed_weight_runtime_bytes + auxiliary_runtime_buffer_bytes
        )
        if materialized_runtime_tensor_bytes > ceiling:
            raise Lfm2Exact50All8BudgetError(
                "Uniform int8 codes, float16 scales, and all auxiliary model "
                "buffers exceed the exact 50% runtime ceiling."
            )
        other_bytes = (
            int(payload_artifact["size_bytes"])
            + int(package_artifact["size_bytes"])
        )

        def factory(total: int, manifest_bytes: int):
            return _manifest_document(
                model=model,
                aliases=aliases,
                tensor_entries=tensor_entries,
                payload_segments=payload_segments,
                package_members=package_members,
                payload_artifact=payload_artifact,
                package_artifact=package_artifact,
                dense_source_bytes=dense_source_bytes,
                source_nonweight_bytes=source_nonweight_bytes,
                manifest_bytes=manifest_bytes,
                checkpoint_bytes=total,
                raw_sizes=raw_sizes,
                tokenizer_fingerprint=tokenizer_fingerprint,
                error=error,
                auxiliary_runtime_buffer_bytes=auxiliary_runtime_buffer_bytes,
            )

        manifest, manifest_raw = _stable_manifest(
            other_bytes=other_bytes,
            factory=factory,
        )
        if manifest["storage"]["compressed_checkpoint_bytes"] > ceiling:
            raise Lfm2Exact50All8BudgetError(
                "Lossless all8 container exceeds the exact 50% directory ceiling."
            )
        _write_new(temporary / MANIFEST_FILENAME, manifest_raw)
        observed = sum(path.stat().st_size for path in temporary.iterdir())
        if observed != manifest["storage"]["compressed_checkpoint_bytes"]:
            raise AssertionError(
                "On-disk exact50 all8 bytes differ from the manifest."
            )
        temporary.replace(output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class All8PackedQuantizedTensor(nn.Module):
    """One canonical ordered uint8 tensor plus float16 group scales."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, Any],
        codes: Tensor,
        scales: Tensor,
    ) -> None:
        super().__init__()
        self.shape = tuple(int(value) for value in metadata["shape"])
        self.source_numel = int(metadata["source_numel"])
        self.group_size = int(metadata["group_size"])
        self.group_count = int(metadata["group_count"])
        self.source_dtype_name = str(metadata["source_dtype"])
        if (
            metadata.get("code_bit_width") != 8
            or self.source_dtype_name not in _SOURCE_DTYPES
            or codes.dtype != torch.uint8
            or codes.ndim != 1
            or codes.numel() != self.group_count * self.group_size
            or scales.dtype != torch.float16
            or scales.ndim != 1
            or scales.numel() != self.group_count
        ):
            raise Lfm2Exact50All8FormatError(
                "All8 packed tensor buffers are inconsistent."
            )
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)

    @property
    def source_dtype(self) -> torch.dtype:
        return _SOURCE_DTYPES[self.source_dtype_name]

    @property
    def packed_nbytes(self) -> int:
        return int(self.codes.numel()) + int(self.scales.numel()) * 2

    def dequantize(self) -> Tensor:
        values = (self.codes.reshape(self.group_count, self.group_size).to(torch.int16) - 127)
        values = values.to(torch.float32) * self.scales.to(torch.float32)[:, None]
        return (
            values.reshape(-1)[: self.source_numel]
            .reshape(self.shape)
            .to(self.source_dtype)
        )


def _checked_artifact(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "file",
        "size_bytes",
        "sha256",
    }:
        raise Lfm2Exact50All8FormatError(f"{name} artifact metadata is malformed.")
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
        or len(digest) != _DIGEST_LENGTH
    ):
        raise Lfm2Exact50All8FormatError(f"{name} artifact metadata is invalid.")
    path = root / filename
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != size
        or _sha256_file(path) != digest
    ):
        raise Lfm2Exact50All8FormatError(
            f"{name} artifact failed path, size, or SHA-256 validation."
        )
    return path


def _manifest_and_artifacts(
    checkpoint_directory: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    root = _safe_local_directory(checkpoint_directory, "checkpoint_directory")
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Lfm2Exact50All8FormatError(f"Missing {MANIFEST_FILENAME}.")
    manifest = _strict_json_file(manifest_path)
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise Lfm2Exact50All8FormatError(
            "Unsupported exact50 all8 checkpoint schema."
        )
    claimed = manifest.get("manifest_sha256")
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    if (
        not isinstance(claimed, str)
        or len(claimed) != _DIGEST_LENGTH
        or claimed != canonical_sha256(core)
    ):
        raise Lfm2Exact50All8FormatError("Manifest SHA-256 mismatch.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "payload",
        "standalone_package",
    }:
        raise Lfm2Exact50All8FormatError(
            "Exact50 all8 artifact inventory is malformed."
        )
    paths = {
        "payload": _checked_artifact(root, artifacts["payload"], "payload"),
        "package": _checked_artifact(
            root, artifacts["standalone_package"], "standalone_package"
        ),
    }
    expected_files = {MANIFEST_FILENAME, *(path.name for path in paths.values())}
    children = tuple(root.iterdir())
    actual_files = {
        path.name for path in children if path.is_file() and not path.is_symlink()
    }
    if actual_files != expected_files or any(
        path.is_dir() or path.is_symlink() for path in children
    ):
        raise Lfm2Exact50All8FormatError(
            "Checkpoint contains missing, unclaimed, nested, or symlink artifacts."
        )
    observed = sum(path.stat().st_size for path in children)
    storage = manifest.get("storage")
    target = manifest.get("target")
    method = manifest.get("method")
    quantization = manifest.get("quantization")
    runtime = manifest.get("runtime")
    if (
        not isinstance(storage, Mapping)
        or not isinstance(target, Mapping)
        or not isinstance(method, Mapping)
        or not isinstance(quantization, Mapping)
        or not isinstance(runtime, Mapping)
        or storage.get("compressed_checkpoint_bytes") != observed
        or storage.get("manifest_bytes") != manifest_path.stat().st_size
        or storage.get("payload_bytes") != paths["payload"].stat().st_size
        or storage.get("standalone_package_bytes")
        != paths["package"].stat().st_size
        or storage.get("all_checkpoint_files_counted") is not True
        or storage.get("hidden_dense_checkpoint_copy") is not False
        or storage.get("external_weight_sidecars") != []
        or storage.get("runtime_bytes_under_same_ceiling") is not True
        or target.get("ceiling_is_exact") is not True
        or target.get("requested_reduction_ppm") != 500_000
        or target.get("target_checkpoint_bytes")
        != storage.get("accounted_dense_reference_bytes") // 2
        or observed > target.get("target_checkpoint_bytes", -1)
        or target.get("headroom_bytes")
        != target.get("target_checkpoint_bytes") - observed
        or storage.get("materialized_runtime_tensor_bytes")
        != storage.get("packed_runtime_bytes")
        or storage.get("packed_weight_runtime_bytes")
        + storage.get("auxiliary_runtime_buffer_bytes")
        != storage.get("materialized_runtime_tensor_bytes")
        or storage.get("materialized_runtime_tensor_bytes")
        > target.get("target_checkpoint_bytes")
        or target.get("runtime_headroom_bytes")
        != target.get("target_checkpoint_bytes")
        - storage.get("materialized_runtime_tensor_bytes")
        or method.get("kind") != METHOD_KIND
        or method.get("all_logical_weight_codes_are_8bit") is not True
        or method.get("seven_bit_weight_codes") != 0
        or method.get("training_used") is not False
        or method.get("optimizer_used") is not False
        or method.get("backward_used") is not False
        or quantization.get("minimum_code_bit_width") != 8
        or quantization.get("maximum_code_bit_width") != 8
        or quantization.get("seven_bit_group_count") != 0
        or runtime.get("persistent_dense_weight_parameter_elements") != 0
        or runtime.get("dense_weight_cache") is not False
        or runtime.get("storage_class_mask_retained") is not False
    ):
        raise Lfm2Exact50All8FormatError(
            "Exact50 all8 physical or method accounting is inconsistent."
        )
    dense = storage.get("dense_source_tensor_bytes")
    shared = storage.get("source_nonweight_artifact_bytes")
    reference = storage.get("accounted_dense_reference_bytes")
    if (
        isinstance(dense, bool)
        or not isinstance(dense, int)
        or dense <= 0
        or isinstance(shared, bool)
        or not isinstance(shared, int)
        or shared <= 0
        or reference != dense + shared
    ):
        raise Lfm2Exact50All8FormatError(
            "Exact50 all8 dense reference arithmetic is inconsistent."
        )
    tensors = manifest.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise Lfm2Exact50All8FormatError(
            "Exact50 all8 tensor inventory is empty."
        )
    try:
        source_bytes = sum(int(value["source_nbytes"]) for value in tensors.values())
        group_count = sum(int(value["group_count"]) for value in tensors.values())
        padded_values = sum(
            int(value["padded_value_count"]) for value in tensors.values()
        )
        mask_bytes = sum(
            (int(value["group_count"]) + 7) // 8 for value in tensors.values()
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2Exact50All8FormatError(
            "Exact50 all8 tensor accounting is malformed."
        ) from exc
    if (
        source_bytes != dense
        or group_count != quantization.get("total_group_count")
        or padded_values != storage.get("raw_8bit_value_bytes")
        or group_count * 2 != storage.get("raw_float16_scale_bytes")
        or mask_bytes != storage.get("raw_storage_selector_bytes")
        or storage.get("packed_weight_runtime_bytes")
        != storage.get("raw_8bit_value_bytes")
        + storage.get("raw_float16_scale_bytes")
    ):
        raise Lfm2Exact50All8FormatError(
            "Exact50 all8 tensor, padding, or runtime totals are inconsistent."
        )
    return root, manifest, paths


def _decompress_exact(
    stored: bytes,
    *,
    codec: str,
    raw_size: int,
    name: str,
) -> bytes:
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise Lfm2Exact50All8FormatError(f"{name} raw size is invalid.")
    if codec == "raw":
        raw = stored
    elif codec == "zlib-9":
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(stored, raw_size + 1)
            if len(raw) <= raw_size:
                raw += decoder.flush(raw_size + 1 - len(raw))
        except zlib.error as exc:
            raise Lfm2Exact50All8FormatError(
                f"{name} zlib stream is invalid."
            ) from exc
        if decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:
            raise Lfm2Exact50All8FormatError(
                f"{name} zlib stream has trailing or unconsumed data."
            )
    else:
        raise Lfm2Exact50All8FormatError(f"{name} codec is unsupported.")
    if len(raw) != raw_size:
        raise Lfm2Exact50All8FormatError(
            f"{name} decompressed size differs from its declaration."
        )
    return raw


def _load_container_segments(
    path: Path,
    *,
    magic: bytes,
    entries: Mapping[str, Any],
    expected_order: Sequence[str],
    label: str,
) -> dict[str, bytes]:
    payload = path.read_bytes()
    if not payload.startswith(magic):
        raise Lfm2Exact50All8FormatError(f"{label} magic is invalid.")
    if not isinstance(entries, Mapping):
        raise Lfm2Exact50All8FormatError(
            f"{label} segment inventory is invalid."
        )
    try:
        physical_order = sorted(
            entries, key=lambda name: int(entries[name]["byte_offset"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2Exact50All8FormatError(
            f"{label} segment offsets are malformed."
        ) from exc
    if physical_order != list(expected_order):
        raise Lfm2Exact50All8FormatError(
            f"{label} segment order or inventory is invalid."
        )
    output: dict[str, bytes] = {}
    offset = len(magic)
    for name in expected_order:
        entry = entries[name]
        required = {
            "byte_offset",
            "stored_size_bytes",
            "stored_sha256",
            "raw_size_bytes",
            "raw_sha256",
            "codec",
        }
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise Lfm2Exact50All8FormatError(
                f"{label} segment {name!r} metadata is malformed."
            )
        stored_size = entry["stored_size_bytes"]
        raw_size = entry["raw_size_bytes"]
        if (
            entry["byte_offset"] != offset
            or isinstance(stored_size, bool)
            or not isinstance(stored_size, int)
            or stored_size < 0
            or not isinstance(entry["stored_sha256"], str)
            or len(entry["stored_sha256"]) != _DIGEST_LENGTH
            or not isinstance(entry["raw_sha256"], str)
            or len(entry["raw_sha256"]) != _DIGEST_LENGTH
            or offset + stored_size > len(payload)
        ):
            raise Lfm2Exact50All8FormatError(
                f"{label} segment {name!r} range is invalid."
            )
        stored = payload[offset : offset + stored_size]
        if _sha256_bytes(stored) != entry["stored_sha256"]:
            raise Lfm2Exact50All8FormatError(
                f"{label} segment {name!r} stored SHA-256 mismatch."
            )
        raw = _decompress_exact(
            stored,
            codec=entry["codec"],
            raw_size=raw_size,
            name=f"{label}:{name}",
        )
        if _sha256_bytes(raw) != entry["raw_sha256"]:
            raise Lfm2Exact50All8FormatError(
                f"{label} segment {name!r} raw SHA-256 mismatch."
            )
        output[name] = raw
        offset += stored_size
    if offset != len(payload):
        raise Lfm2Exact50All8FormatError(
            f"{label} has unclaimed trailing bytes."
        )
    return output


def _package_members(
    manifest: Mapping[str, Any],
    package_path: Path,
) -> dict[str, bytes]:
    entries = manifest.get("package_members")
    if not isinstance(entries, Mapping):
        raise Lfm2Exact50All8FormatError(
            "Standalone package inventory is malformed."
        )
    expected = [name for name in _PACKAGE_MEMBER_ORDER if name in entries]
    if (
        CONFIG_FILENAME not in entries
        or GENERATION_CONFIG_FILENAME not in entries
        or "tokenizer.json" not in entries
    ):
        raise Lfm2Exact50All8FormatError(
            "Standalone package lacks a required member."
        )
    return _load_container_segments(
        package_path,
        magic=PACKAGE_MAGIC,
        entries=entries,
        expected_order=expected,
        label="standalone package",
    )


def _slice_raw(
    raw: bytes,
    value: Mapping[str, Any],
    *,
    expected_offset: int,
    expected_size: int,
    name: str,
) -> tuple[bytes, int]:
    if not isinstance(value, Mapping) or set(value) != {"byte_offset", "size_bytes"}:
        raise Lfm2Exact50All8FormatError(f"{name} range metadata is malformed.")
    if value["byte_offset"] != expected_offset or value["size_bytes"] != expected_size:
        raise Lfm2Exact50All8FormatError(f"{name} range is inconsistent.")
    end = expected_offset + expected_size
    if end > len(raw):
        raise Lfm2Exact50All8FormatError(f"{name} range exceeds its segment.")
    return raw[expected_offset:end], end


def load_lfm2_exact50_all8(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_checkpoint_bytes: int | None = None,
    max_packed_runtime_bytes: int | None = None,
) -> nn.Module:
    """Freshly reload a standalone exact50 all8 LFM2 without dense weights."""

    _root, manifest, paths = _manifest_and_artifacts(checkpoint_directory)
    checkpoint_bytes = int(manifest["storage"]["compressed_checkpoint_bytes"])
    runtime_bytes = int(manifest["storage"]["materialized_runtime_tensor_bytes"])
    for limit, name, required in (
        (max_checkpoint_bytes, "max_checkpoint_bytes", checkpoint_bytes),
        (max_packed_runtime_bytes, "max_packed_runtime_bytes", runtime_bytes),
    ):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise Lfm2Exact50All8Error(
                f"{name} must be a nonnegative integer or None."
            )
        if limit is not None and required > limit:
            raise Lfm2Exact50All8BudgetError(
                f"Checkpoint requires {required} bytes for {name}, exceeding {limit}."
            )
    members = _package_members(manifest, paths["package"])
    config_data = _strict_json_bytes(members[CONFIG_FILENAME], CONFIG_FILENAME)
    generation_data = _strict_json_bytes(
        members[GENERATION_CONFIG_FILENAME], GENERATION_CONFIG_FILENAME
    )
    topology = manifest.get("topology")
    if (
        not isinstance(topology, Mapping)
        or topology.get("model_type") != "lfm2"
        or topology.get("all_logical_operators_preserved") is not True
        or topology.get("tied_input_output_embeddings") is not True
        or config_data.get("model_type") != "lfm2"
        or config_data.get("tie_word_embeddings") is not True
        or config_data.get("num_hidden_layers")
        != topology.get("logical_layer_count")
        or config_data.get("layer_types") != topology.get("layer_types")
        or config_data.get("_name_or_path") not in {"", None}
        or "auto_map" in config_data
    ):
        raise Lfm2Exact50All8FormatError(
            "Fresh LFM2 configuration differs from the sealed topology."
        )
    try:
        from transformers import GenerationConfig, Lfm2Config, Lfm2ForCausalLM
    except ImportError as exc:
        raise Lfm2Exact50All8Error(
            "Loading requires Transformers with LFM2 support."
        ) from exc
    config = Lfm2Config.from_dict(config_data)
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise Lfm2Exact50All8Error(
            "device='meta' is not a materialized runtime target."
        )
    with torch.device("meta"):
        model = Lfm2ForCausalLM(config)
    logical_meta, aliases_meta = _model_parameters_meta(model)
    if aliases_meta != manifest.get("state", {}).get("logical_to_canonical"):
        raise Lfm2Exact50All8FormatError(
            "Fresh LFM2 logical aliases differ from the manifest."
        )
    raw_segments = _load_container_segments(
        paths["payload"],
        magic=PAYLOAD_MAGIC,
        entries=manifest.get("payload_segments"),
        expected_order=_SEGMENT_ROLES,
        label="model payload",
    )
    entries = manifest.get("tensors")
    canonical_names = set(aliases_meta.values())
    if not isinstance(entries, Mapping) or set(entries) != canonical_names:
        raise Lfm2Exact50All8FormatError(
            "All8 tensor inventory differs from the canonical state."
        )
    cursors = {role: 0 for role in _SEGMENT_ROLES}
    stores: dict[str, All8PackedQuantizedTensor] = {}
    for name in sorted(entries):
        entry = entries[name]
        meta_parameter = logical_meta[name]
        if (
            not isinstance(entry, Mapping)
            or entry.get("logical_names")
            != sorted(
                logical
                for logical, canonical in aliases_meta.items()
                if canonical == name
            )
            or entry.get("shape") != list(meta_parameter.shape)
            or entry.get("source_numel") != int(meta_parameter.numel())
            or entry.get("source_nbytes") != int(meta_parameter.numel()) * 2
            or not isinstance(entry.get("source_sha256"), str)
            or len(entry.get("source_sha256")) != _DIGEST_LENGTH
            or entry.get("source_dtype") not in _SOURCE_DTYPES
            or entry.get("group_size") != manifest["quantization"]["group_size"]
            or entry.get("code_bit_width") != 8
            or entry.get("scale_dtype") != "torch.float16"
            or entry.get("symmetric_zero_point") != 0
        ):
            raise Lfm2Exact50All8FormatError(
                f"All8 tensor {name!r} metadata is inconsistent."
            )
        group_count = int(entry["group_count"])
        group_size = int(entry["group_size"])
        if (
            group_count
            != (int(meta_parameter.numel()) + group_size - 1) // group_size
            or entry.get("padded_value_count") != group_count * group_size
        ):
            raise Lfm2Exact50All8FormatError(
                f"All8 tensor {name!r} group count is inconsistent."
            )
        counts = entry.get("storage_class_counts")
        ranges = entry.get("raw_segment_ranges")
        if (
            not isinstance(counts, Mapping)
            or set(counts) != {"compressed", "raw_fallback"}
            or not isinstance(ranges, Mapping)
            or set(ranges) != set(_SEGMENT_ROLES)
        ):
            raise Lfm2Exact50All8FormatError(
                f"All8 tensor {name!r} storage ranges are malformed."
            )
        high_count = counts["compressed"]
        low_count = counts["raw_fallback"]
        if (
            isinstance(high_count, bool)
            or not isinstance(high_count, int)
            or high_count < 0
            or isinstance(low_count, bool)
            or not isinstance(low_count, int)
            or low_count < 0
            or high_count + low_count != group_count
        ):
            raise Lfm2Exact50All8FormatError(
                f"All8 tensor {name!r} storage counts are invalid."
            )
        expected_sizes = {
            "storage_mask": (group_count + 7) // 8,
            "compressed_class_codes": high_count * group_size,
            "raw_class_codes": low_count * group_size,
            "scales": group_count * 2,
        }
        pieces: dict[str, bytes] = {}
        for role in _SEGMENT_ROLES:
            pieces[role], cursors[role] = _slice_raw(
                raw_segments[role],
                ranges[role],
                expected_offset=cursors[role],
                expected_size=expected_sizes[role],
                name=f"{name}:{role}",
            )
        mask_tensor = torch.frombuffer(
            bytearray(pieces["storage_mask"]), dtype=torch.uint8
        )
        mask = _unpack_mask(mask_tensor, group_count)
        if int(mask.sum()) != low_count:
            raise Lfm2Exact50All8FormatError(
                f"All8 tensor {name!r} storage selector count is invalid."
            )
        high_codes = (
            torch.frombuffer(
                bytearray(pieces["compressed_class_codes"]), dtype=torch.uint8
            ).reshape(high_count, group_size)
            if high_count
            else torch.empty((0, group_size), dtype=torch.uint8)
        )
        low_codes = (
            torch.frombuffer(
                bytearray(pieces["raw_class_codes"]), dtype=torch.uint8
            ).reshape(low_count, group_size)
            if low_count
            else torch.empty((0, group_size), dtype=torch.uint8)
        )
        codes = torch.empty(group_count, group_size, dtype=torch.uint8)
        high_indices = torch.nonzero(~mask, as_tuple=False).reshape(-1)
        low_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if high_count:
            codes.index_copy_(0, high_indices, high_codes)
        if low_count:
            codes.index_copy_(0, low_indices, low_codes)
        scales = torch.frombuffer(
            bytearray(pieces["scales"]), dtype=torch.float16
        ).clone()
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise Lfm2Exact50All8FormatError(
                f"All8 tensor {name!r} scales are invalid."
            )
        stores[name] = All8PackedQuantizedTensor(
            metadata=entry,
            codes=codes.reshape(-1).to(target_device),
            scales=scales.to(target_device),
        )
    if any(cursors[role] != len(raw_segments[role]) for role in _SEGMENT_ROLES):
        raise Lfm2Exact50All8FormatError(
            "All8 payload has unclaimed raw segment bytes."
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
        raise Lfm2Exact50All8FormatError(
            "All8 runtime retained a persistent dense weight Parameter."
        )
    observed_weight_runtime = sum(store.packed_nbytes for store in stores.values())
    seen_buffers: set[int] = set()
    observed_runtime = 0
    for _name, value in model.named_buffers(recurse=True, remove_duplicate=True):
        if id(value) in seen_buffers:
            continue
        seen_buffers.add(id(value))
        observed_runtime += int(value.numel()) * int(value.element_size())
    if (
        observed_weight_runtime
        != manifest["storage"]["packed_weight_runtime_bytes"]
        or observed_runtime != runtime_bytes
    ):
        raise Lfm2Exact50All8FormatError(
            "All8 runtime buffers differ from physical runtime accounting."
        )
    if (
        model.model.embed_tokens._packed_weight is not model.lm_head._packed_weight
        or model.model.embed_tokens._packed_weight
        is not stores["model.embed_tokens.weight"]
    ):
        raise Lfm2Exact50All8FormatError(
            "Fresh all8 runtime did not restore tied vocabulary storage once."
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
            "packed_weight_runtime_bytes": observed_weight_runtime,
            "auxiliary_runtime_buffer_bytes": (
                observed_runtime - observed_weight_runtime
            ),
            "materialized_runtime_tensor_bytes": observed_runtime,
            "packed_runtime_bytes": observed_runtime,
            "checkpoint_bytes": checkpoint_bytes,
            "all_logical_weight_codes_are_8bit": True,
            "seven_bit_weight_codes": 0,
            "storage_class_mask_retained": False,
            "tied_vocabulary_stored_once": True,
            "from_pretrained_used": False,
            "network_access": False,
        },
    )
    return model


def load_lfm2_exact50_all8_tokenizer(
    checkpoint_directory: str | os.PathLike[str],
) -> Any:
    """Load the tokenizer directly from the authenticated compressed package."""

    _root, manifest, paths = _manifest_and_artifacts(checkpoint_directory)
    members = _package_members(manifest, paths["package"])
    try:
        import tokenizers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise Lfm2Exact50All8Error(
            "Tokenizer loading requires tokenizers and Transformers."
        ) from exc
    try:
        backend = tokenizers.Tokenizer.from_str(
            members["tokenizer.json"].decode("utf-8")
        )
    except (UnicodeError, Exception) as exc:
        if isinstance(exc, Lfm2Exact50All8Error):
            raise
        raise Lfm2Exact50All8FormatError(
            "Compressed tokenizer.json could not be loaded."
        ) from exc
    kwargs: dict[str, Any] = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        if name in members:
            kwargs.update(_strict_json_bytes(members[name], name))
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
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)
    if "chat_template.jinja" in members:
        tokenizer.chat_template = members["chat_template.jinja"].decode("utf-8")
    setattr(
        tokenizer,
        "neural_forge_tokenizer_fingerprint",
        manifest["tokenizer"]["fingerprint"],
    )
    return tokenizer


def inspect_lfm2_exact50_all8(
    checkpoint_directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Authenticate an all8 artifact and return exact disk/runtime accounting."""

    root, manifest, _paths = _manifest_and_artifacts(checkpoint_directory)
    storage = manifest["storage"]
    target = manifest["target"]
    return {
        "checkpoint_directory": str(root),
        "manifest_sha256": manifest["manifest_sha256"],
        "control_label": manifest["control_label"],
        "requested_reduction": 0.5,
        "realized_serialized_reduction": (
            1.0
            - storage["compressed_checkpoint_bytes"]
            / storage["accounted_dense_reference_bytes"]
        ),
        "target_checkpoint_bytes": target["target_checkpoint_bytes"],
        "compressed_checkpoint_bytes": storage["compressed_checkpoint_bytes"],
        "headroom_bytes": target["headroom_bytes"],
        "packed_weight_runtime_bytes": storage["packed_weight_runtime_bytes"],
        "auxiliary_runtime_buffer_bytes": storage[
            "auxiliary_runtime_buffer_bytes"
        ],
        "materialized_runtime_tensor_bytes": storage[
            "materialized_runtime_tensor_bytes"
        ],
        "packed_runtime_bytes": storage["packed_runtime_bytes"],
        "runtime_headroom_bytes": target["runtime_headroom_bytes"],
        "raw_8bit_value_bytes": storage["raw_8bit_value_bytes"],
        "raw_float16_scale_bytes": storage["raw_float16_scale_bytes"],
        "raw_storage_selector_bytes": storage["raw_storage_selector_bytes"],
        "payload_bytes": storage["payload_bytes"],
        "standalone_package_bytes": storage["standalone_package_bytes"],
        "manifest_bytes": storage["manifest_bytes"],
        "minimum_code_bit_width": 8,
        "maximum_code_bit_width": 8,
        "seven_bit_group_count": 0,
        "structural_parameter_count_compaction": False,
        "tied_vocabulary_stored_once": True,
        "no_hidden_dense_copy": True,
        "all_checkpoint_files_counted": True,
    }


__all__ = [
    "All8PackedQuantizedTensor",
    "CHECKPOINT_SCHEMA",
    "Lfm2Exact50All8BudgetError",
    "Lfm2Exact50All8Error",
    "Lfm2Exact50All8FormatError",
    "MANIFEST_FILENAME",
    "METHOD_KIND",
    "PACKAGE_FILENAME",
    "PACKAGE_MAGIC",
    "PAYLOAD_FILENAME",
    "PAYLOAD_MAGIC",
    "export_lfm2_exact50_all8",
    "inspect_lfm2_exact50_all8",
    "load_lfm2_exact50_all8",
    "load_lfm2_exact50_all8_tokenizer",
]
