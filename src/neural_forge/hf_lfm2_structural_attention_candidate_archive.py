"""Immutable archive for pristine structural attention fold candidates.

Post-Q/K output reclosure needs the exact Q-DEIM tensors that produced the
saved body: not only Q/K/V/O, but also the residual chart and authenticated
interpolation objects.  Those objects are compiler evidence, not runtime
parameters, so a standalone child checkpoint does not contain them.  This
module persists them once during the original fold and reconstructs the typed
candidate later without reopening parent weights or regenerating traces.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_attention_group_qdeim import (
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_candidate_state_sha256,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_STRUCTURAL_ATTENTION_CANDIDATE_ARCHIVE_SCHEMA = (
    "neural-forge-lfm2-structural-attention-candidate-archive-v1"
)
MANIFEST_FILENAME = "manifest.json"
TENSOR_FILENAME = "candidates.safetensors"
_TENSOR_FIELDS = (
    "query_weight",
    "key_weight",
    "value_weight",
    "out_weight",
    "group_pod_basis",
    "group_interpolation",
    "context_reconstruction",
    "normalized_input_decoder",
    "residual_output_chart",
)


class Lfm2StructuralAttentionCandidateArchiveError(RuntimeError):
    """Candidate archive path, bytes, receipt, or tensor lineage is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Structural candidate archives require PyTorch.") from exc
    return torch


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    return value


def _verified_receipt(
    value: Any,
    *,
    schema: str,
    seal_field: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop(seal_field, None)
    if body.get("schema") != schema or supplied != canonical_sha256(body):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            f"{label} is stale, malformed, or unauthenticated."
        )
    return MappingProxyType({**body, seal_field: supplied})


def _ordinary_root(value: str | os.PathLike[str], *, create: bool) -> Path:
    path = Path(value).expanduser().absolute()
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise Lfm2StructuralAttentionCandidateArchiveError(
                "Candidate archive root must be one ordinary directory."
            )
    elif create:
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise Lfm2StructuralAttentionCandidateArchiveError(
                "Candidate archive parent must be one ordinary directory."
            )
    else:
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate archive directory does not exist."
        )
    return path


def _compiler_receipts_from_stack(
    layer_stack_receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[int, Mapping[str, Any]]]:
    stack = _verified_receipt(
        layer_stack_receipt,
        schema=LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
        seal_field="result_sha256",
        label="semantic layer-stack receipt",
    )
    layers = stack.get("layer_installations")
    if not isinstance(layers, list) or not layers:
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Semantic stack contains no layer installations."
        )
    result: dict[int, Mapping[str, Any]] = {}
    for raw_layer in layers:
        if not isinstance(raw_layer, Mapping):
            raise Lfm2StructuralAttentionCandidateArchiveError(
                "Semantic stack layer receipt is malformed."
            )
        if raw_layer.get("operator_kind") != "full_attention":
            continue
        layer_index = raw_layer.get("layer_index")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or layer_index in result
        ):
            raise Lfm2StructuralAttentionCandidateArchiveError(
                "Semantic stack attention indices are invalid or duplicated."
            )
        operator = raw_layer.get("attention_operator_installation_receipt")
        if not isinstance(operator, Mapping):
            raise Lfm2StructuralAttentionCandidateArchiveError(
                f"Layer {layer_index} has no attention operator receipt."
            )
        structural = operator.get("structural_attention_operator_receipt")
        if isinstance(structural, Mapping):
            operator = structural
        compiler = _verified_receipt(
            operator.get("coupled_attention_compiler_receipt"),
            schema=LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
            seal_field="result_sha256",
            label=f"layer {layer_index} structural attention compiler receipt",
        )
        result[layer_index] = compiler
    if not result:
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Semantic stack contains no structural attention candidates."
        )
    return stack, MappingProxyType(result)


def _tensor_record(value: Any, *, label: str) -> dict[str, Any]:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or not value.dtype.is_floating_point
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            f"{label} must be one finite dense floating rank-2 tensor."
        )
    return {
        "sha256": lfm2_authenticated_tensor_sha256(value),
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype).removeprefix("torch."),
    }


def _candidate_tensor_map(
    candidates: Mapping[int, Lfm2CompiledAttentionGroupQdeim],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensors: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for layer_index in sorted(candidates):
        candidate = candidates[layer_index]
        if not isinstance(candidate, Lfm2CompiledAttentionGroupQdeim):
            raise Lfm2StructuralAttentionCandidateArchiveError(
                f"Layer {layer_index} is not one typed structural candidate."
            )
        for field in _TENSOR_FIELDS:
            key = f"layer.{layer_index:03d}.{field}"
            tensor = (
                getattr(candidate, field).detach().to(device="cpu").contiguous().clone()
            )
            tensors[key] = tensor
            records[key] = _tensor_record(tensor, label=key)
    return tensors, records


def _authenticate_candidate(
    candidate: Lfm2CompiledAttentionGroupQdeim,
    *,
    compiler_receipt: Mapping[str, Any],
    layer_index: int,
) -> None:
    report = _verified_receipt(
        candidate.report,
        schema=LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        seal_field="result_sha256",
        label=f"layer {layer_index} candidate report",
    )
    interpolation = report.get("interpolation")
    authentication = report.get("authentication")
    materialization = report.get("materialization")
    selection = report.get("selection")
    if not all(
        isinstance(value, Mapping)
        for value in (interpolation, authentication, materialization, selection)
    ):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            f"Layer {layer_index} compiler receipt lacks candidate authentication."
        )
    state_sha = lfm2_attention_group_qdeim_candidate_state_sha256(
        candidate.query_weight,
        candidate.key_weight,
        candidate.value_weight,
        candidate.out_weight,
    )
    if (
        dict(report) != dict(compiler_receipt)
        or materialization.get("candidate_state_sha256") != state_sha
        or interpolation.get("group_pod_basis_sha256")
        != lfm2_authenticated_tensor_sha256(candidate.group_pod_basis)
        or interpolation.get("group_interpolation_sha256")
        != lfm2_authenticated_tensor_sha256(candidate.group_interpolation)
        or interpolation.get("context_reconstruction_sha256")
        != lfm2_authenticated_tensor_sha256(candidate.context_reconstruction)
        or authentication.get("normalized_input_decoder_sha256")
        != lfm2_authenticated_tensor_sha256(candidate.normalized_input_decoder)
        or authentication.get("residual_output_chart_sha256")
        != lfm2_authenticated_tensor_sha256(candidate.residual_output_chart)
        or selection.get("selected_source_key_value_groups")
        != list(candidate.selected_source_key_value_groups)
        or selection.get("selected_source_query_heads")
        != list(candidate.selected_source_query_heads)
    ):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            f"Layer {layer_index} candidate tensors differ from the semantic stack."
        )


def write_lfm2_structural_attention_candidate_archive(
    root: str | os.PathLike[str],
    candidates: Mapping[int, Lfm2CompiledAttentionGroupQdeim],
    *,
    semantic_layer_stack_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Atomically publish and freshly reload all pristine attention candidates."""

    if not isinstance(candidates, Mapping):
        raise TypeError("candidates must be a mapping.")
    destination = _ordinary_root(root, create=True)
    if destination.exists():
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate archive destination already exists."
        )
    stack, compilers = _compiler_receipts_from_stack(semantic_layer_stack_receipt)
    if set(candidates) != set(compilers):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate set does not cover the semantic stack attention topology."
        )
    for index in sorted(candidates):
        _authenticate_candidate(
            candidates[index],
            compiler_receipt=compilers[index],
            layer_index=index,
        )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - required dependency
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "safetensors is required for structural candidate archives."
        ) from exc
    tensors, records = _candidate_tensor_map(candidates)
    stage = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        tensor_path = stage / TENSOR_FILENAME
        save_file(tensors, str(tensor_path))
        body = {
            "schema": LFM2_STRUCTURAL_ATTENTION_CANDIDATE_ARCHIVE_SCHEMA,
            "status": "pristine-structural-attention-candidates-persisted",
            "semantic_layer_stack_result_sha256": stack["result_sha256"],
            "attention_layer_indices": sorted(candidates),
            "attention_layer_count": len(candidates),
            "tensor_filename": TENSOR_FILENAME,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "tensor_records": records,
            "compiler_receipts": {
                str(index): _deep_thaw(compilers[index])
                for index in sorted(compilers)
            },
            "candidate_state_sha256": {
                str(index): lfm2_attention_group_qdeim_candidate_state_sha256(
                    candidates[index].query_weight,
                    candidates[index].key_weight,
                    candidates[index].value_weight,
                    candidates[index].out_weight,
                )
                for index in sorted(candidates)
            },
            "parent_model_objects_constructed": 0,
            "parent_weight_tensor_loads": 0,
            "parent_trace_recaptures": 0,
            "model_forward_calls": 0,
            "training_steps": 0,
        }
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        manifest_path = stage / MANIFEST_FILENAME
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    _loaded, authenticated = load_lfm2_structural_attention_candidate_archive(
        destination,
        semantic_layer_stack_receipt=semantic_layer_stack_receipt,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    return authenticated


def load_lfm2_structural_attention_candidate_archive(
    root: str | os.PathLike[str],
    *,
    semantic_layer_stack_receipt: Mapping[str, Any],
    expected_manifest_sha256: str,
) -> tuple[Mapping[int, Lfm2CompiledAttentionGroupQdeim], Mapping[str, Any]]:
    """Authenticate and reconstruct exact structural candidates without parent I/O."""

    destination = _ordinary_root(root, create=False)
    expected_manifest = _digest(
        expected_manifest_sha256,
        label="expected candidate archive manifest",
    )
    manifest_path = destination / MANIFEST_FILENAME
    tensor_path = destination / TENSOR_FILENAME
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not tensor_path.is_file()
        or tensor_path.is_symlink()
    ):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate archive files are missing or indirect."
        )
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate archive manifest is invalid JSON."
        ) from exc
    if not isinstance(manifest, dict):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate archive manifest must be one JSON object."
        )
    body = dict(manifest)
    supplied = body.pop("manifest_sha256", None)
    stack, compilers = _compiler_receipts_from_stack(semantic_layer_stack_receipt)
    layer_indices = tuple(sorted(compilers))
    if (
        body.get("schema") != LFM2_STRUCTURAL_ATTENTION_CANDIDATE_ARCHIVE_SCHEMA
        or supplied != expected_manifest
        or supplied != canonical_sha256(body)
        or body.get("semantic_layer_stack_result_sha256")
        != stack["result_sha256"]
        or body.get("attention_layer_indices") != list(layer_indices)
        or body.get("attention_layer_count") != len(layer_indices)
        or body.get("tensor_filename") != TENSOR_FILENAME
        or body.get("tensor_file_sha256") != _file_sha256(tensor_path)
        or body.get("compiler_receipts")
        != {str(index): _deep_thaw(compilers[index]) for index in layer_indices}
    ):
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate archive manifest, stack ancestry, or tensor file changed."
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "safetensors is required for structural candidate archives."
        ) from exc
    tensors = load_file(str(tensor_path), device="cpu")
    expected_keys = {
        f"layer.{index:03d}.{field}"
        for index in layer_indices
        for field in _TENSOR_FIELDS
    }
    records = body.get("tensor_records")
    if set(tensors) != expected_keys or not isinstance(records, Mapping) or set(
        records
    ) != expected_keys:
        raise Lfm2StructuralAttentionCandidateArchiveError(
            "Candidate archive tensor inventory is incomplete."
        )
    for key in sorted(expected_keys):
        if records[key] != _tensor_record(tensors[key], label=key):
            raise Lfm2StructuralAttentionCandidateArchiveError(
                f"Candidate archive tensor {key} differs from its manifest."
            )
    result: dict[int, Lfm2CompiledAttentionGroupQdeim] = {}
    for index in layer_indices:
        compiler = compilers[index]
        selection = compiler["selection"]

        def tensor(field: str) -> Any:
            return tensors[f"layer.{index:03d}.{field}"]

        candidate = Lfm2CompiledAttentionGroupQdeim(
            selected_source_key_value_groups=tuple(
                selection["selected_source_key_value_groups"]
            ),
            selected_source_query_heads=tuple(
                selection["selected_source_query_heads"]
            ),
            query_weight=tensor("query_weight"),
            key_weight=tensor("key_weight"),
            value_weight=tensor("value_weight"),
            out_weight=tensor("out_weight"),
            group_pod_basis=tensor("group_pod_basis"),
            group_interpolation=tensor("group_interpolation"),
            context_reconstruction=tensor("context_reconstruction"),
            normalized_input_decoder=tensor("normalized_input_decoder"),
            residual_output_chart=tensor("residual_output_chart"),
            report=compiler,
        )
        _authenticate_candidate(
            candidate,
            compiler_receipt=compiler,
            layer_index=index,
        )
        if body.get("candidate_state_sha256", {}).get(str(index)) != (
            lfm2_attention_group_qdeim_candidate_state_sha256(
                candidate.query_weight,
                candidate.key_weight,
                candidate.value_weight,
                candidate.out_weight,
            )
        ):
            raise Lfm2StructuralAttentionCandidateArchiveError(
                f"Layer {index} candidate state differs from its manifest."
            )
        result[index] = candidate
    authenticated = MappingProxyType(
        {**body, "manifest_sha256": supplied, "fresh_reload_verified": True}
    )
    return MappingProxyType(result), authenticated


__all__ = [
    "LFM2_STRUCTURAL_ATTENTION_CANDIDATE_ARCHIVE_SCHEMA",
    "Lfm2StructuralAttentionCandidateArchiveError",
    "MANIFEST_FILENAME",
    "TENSOR_FILENAME",
    "load_lfm2_structural_attention_candidate_archive",
    "write_lfm2_structural_attention_candidate_archive",
]
