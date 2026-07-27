"""Immutable cached-evidence seeds for postfold LFM2 body reclosure.

Residual reclosure performed after target-lattice attention treatment needs three
compiler objects which are deliberately absent from a standalone child
checkpoint: the contextual parent vocabulary rows used by the foundation
interface solve, that solve's producer encoder, and the live terminal W2 seed
which the semantic layer stack left deferred.  The original exact-4x compiler
has all three at once, so it persists them exactly once without reopening parent
weights or regenerating traces.

The archive is a non-replaceable safetensors payload plus canonical manifest.
Loading requires an independently supplied manifest digest and all three sealed
ancestry receipts.  Consequently, caller-supplied tensors cannot be relabelled
as evidence from another foundation, layer stack, or deferred-W2 installation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_semantic_foundation import (
    LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_POSTFOLD_RECLOSURE_SEED_ARCHIVE_SCHEMA = (
    "neural-forge-lfm2-postfold-reclosure-seed-archive-v1"
)
LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-deferred-final-w2-installation-v1"
)
MANIFEST_FILENAME = "manifest.json"
TENSOR_FILENAME = "reclosure-seeds.safetensors"
_TENSOR_NAMES = (
    "mapped_parent_tied_table",
    "producer_encoder",
    "deferred_final_w2_seed",
)


class Lfm2PostfoldReclosureSeedArchiveError(RuntimeError):
    """The archive bytes, digest pin, tensor inventory, or ancestry is invalid."""


@dataclass(frozen=True, slots=True)
class Lfm2PostfoldReclosureSeedArchive:
    """Freshly loaded reclosure seeds with their authenticated manifest."""

    mapped_parent_tied_table: Any = field(repr=False, compare=False)
    producer_encoder: Any = field(repr=False, compare=False)
    deferred_final_w2_seed: Any = field(repr=False, compare=False)
    manifest: Mapping[str, Any]

    @property
    def contextual_parent_rows(self) -> Any:
        """Alias documenting the exact role of ``mapped_parent_tied_table``."""

        return self.mapped_parent_tied_table


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime
        raise RuntimeError("Postfold reclosure seed archives require PyTorch.") from exc
    return torch


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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
        raise Lfm2PostfoldReclosureSeedArchiveError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop(seal_field, None)
    if body.get("schema") != schema or supplied != canonical_sha256(body):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            f"{label} is stale, malformed, or unauthenticated."
        )
    return MappingProxyType({**body, seal_field: supplied})


def _verified_nested_report(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop("report_sha256", None)
    if supplied != canonical_sha256(body):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            f"{label} is stale, malformed, or unauthenticated."
        )
    return MappingProxyType({**body, "report_sha256": supplied})


def _ordinary_root(value: str | os.PathLike[str], *, create: bool) -> Path:
    path = Path(value).expanduser().absolute()
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise Lfm2PostfoldReclosureSeedArchiveError(
                "Reclosure seed archive root must be one ordinary directory."
            )
    elif create:
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise Lfm2PostfoldReclosureSeedArchiveError(
                "Reclosure seed archive parent must be one ordinary directory."
            )
    else:
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed archive directory does not exist."
        )
    return path


def _materialized_matrix(value: Any, *, label: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or not value.dtype.is_floating_point
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            f"{label} must be one finite dense floating rank-2 tensor."
        )
    return value.detach().to(device="cpu").contiguous()


def _foundation_tensor_sha256(value: Any) -> str:
    """Match the semantic-foundation descriptor-prefix tensor digest."""

    torch = _torch()
    tensor = _materialized_matrix(value, label="foundation tensor")
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_record(value: Any, *, label: str) -> dict[str, Any]:
    tensor = _materialized_matrix(value, label=label)
    return {
        "sha256": lfm2_authenticated_tensor_sha256(tensor),
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _authenticate_ancestry(
    *,
    mapped_parent_tied_table: Any,
    producer_encoder: Any,
    deferred_final_w2_seed: Any,
    foundation_interface_solve_receipt: Mapping[str, Any],
    semantic_layer_stack_receipt: Mapping[str, Any],
    deferred_final_w2_installation_receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    mapped = _materialized_matrix(
        mapped_parent_tied_table,
        label="mapped_parent_tied_table",
    )
    encoder = _materialized_matrix(producer_encoder, label="producer_encoder")
    seed = _materialized_matrix(
        deferred_final_w2_seed,
        label="deferred_final_w2_seed",
    )
    if (
        int(mapped.shape[1]) != int(encoder.shape[0])
        or int(seed.shape[0]) != int(encoder.shape[1])
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed matrices do not share the foundation child coordinate."
        )

    foundation = _verified_receipt(
        foundation_interface_solve_receipt,
        schema=LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
        seal_field="result_sha256",
        label="foundation interface solve receipt",
    )
    active = _verified_nested_report(
        foundation.get("active_interface_report"),
        label="foundation active interface report",
    )
    stack = _verified_receipt(
        semantic_layer_stack_receipt,
        schema=LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
        seal_field="result_sha256",
        label="semantic layer-stack receipt",
    )
    installation = _verified_receipt(
        deferred_final_w2_installation_receipt,
        schema=LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
        seal_field="result_sha256",
        label="deferred final-W2 installation receipt",
    )

    seed_archive_sha256 = lfm2_authenticated_tensor_sha256(seed)
    seed_terminal_sha256 = lfm2_cpcf_terminal_action_tensor_sha256(seed)
    lineage = stack.get("tied_interface_lineage")
    if (
        foundation.get("active_interface_report_sha256")
        != active["report_sha256"]
        or foundation.get("source_tied_table_sha256")
        != _foundation_tensor_sha256(mapped)
        or foundation.get("producer_encoder_sha256")
        != _foundation_tensor_sha256(encoder)
        or not isinstance(lineage, Mapping)
        or lineage.get("interface_report_sha256") != active["report_sha256"]
        or lineage.get("producer_encoder_sha256")
        != active.get("producer_encoder_sha256")
        or stack.get("deferred_final_w2_sha256") != seed_archive_sha256
        or stack.get("final_layer_mlp_down_weight_prior_live_w2_sha256")
        != seed_terminal_sha256
        or installation.get("semantic_layer_stack_result_sha256")
        != stack["result_sha256"]
        or installation.get("prior_authentication_sha256")
        != stack.get("final_layer_mlp_down_weight_prior_authentication_sha256")
        or installation.get("prior_tensor_sha256")
        != stack.get("final_layer_mlp_down_weight_prior_tensor_sha256")
        or installation.get("live_w2_sha256_before") != seed_terminal_sha256
        or installation.get("live_w2_sha256_after")
        != stack.get("final_layer_mlp_down_weight_prior_tensor_sha256")
        or installation.get("model_state_before_sha256")
        != stack.get("model_state_after_sha256")
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seeds do not match their foundation, stack, and W2 ancestry."
        )
    return foundation, stack, installation


def write_lfm2_postfold_reclosure_seed_archive(
    root: str | os.PathLike[str],
    *,
    mapped_parent_tied_table: Any,
    producer_encoder: Any,
    deferred_final_w2_seed: Any,
    foundation_interface_solve_receipt: Mapping[str, Any],
    semantic_layer_stack_receipt: Mapping[str, Any],
    deferred_final_w2_installation_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Atomically publish and freshly reload exact postfold reclosure seeds."""

    destination = _ordinary_root(root, create=True)
    if destination.exists():
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed archive destination already exists."
        )
    foundation, stack, installation = _authenticate_ancestry(
        mapped_parent_tied_table=mapped_parent_tied_table,
        producer_encoder=producer_encoder,
        deferred_final_w2_seed=deferred_final_w2_seed,
        foundation_interface_solve_receipt=foundation_interface_solve_receipt,
        semantic_layer_stack_receipt=semantic_layer_stack_receipt,
        deferred_final_w2_installation_receipt=(
            deferred_final_w2_installation_receipt
        ),
    )
    tensors = {
        "mapped_parent_tied_table": _materialized_matrix(
            mapped_parent_tied_table,
            label="mapped_parent_tied_table",
        ).clone(),
        "producer_encoder": _materialized_matrix(
            producer_encoder,
            label="producer_encoder",
        ).clone(),
        "deferred_final_w2_seed": _materialized_matrix(
            deferred_final_w2_seed,
            label="deferred_final_w2_seed",
        ).clone(),
    }
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - required dependency
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "safetensors is required for postfold reclosure seed archives."
        ) from exc

    stage = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        tensor_path = stage / TENSOR_FILENAME
        save_file(tensors, str(tensor_path))
        records = {
            name: _tensor_record(tensors[name], label=name) for name in _TENSOR_NAMES
        }
        body = {
            "schema": LFM2_POSTFOLD_RECLOSURE_SEED_ARCHIVE_SCHEMA,
            "status": "exact-postfold-reclosure-seeds-persisted",
            "tensor_filename": TENSOR_FILENAME,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "tensor_names": list(_TENSOR_NAMES),
            "tensor_records": records,
            "mapped_parent_tied_table_role": "contextual_parent_rows",
            "contextual_parent_rows_are_exact_mapped_parent_tied_table": True,
            "mapped_parent_tied_table_foundation_sha256": (
                _foundation_tensor_sha256(tensors["mapped_parent_tied_table"])
            ),
            "producer_encoder_foundation_sha256": (
                _foundation_tensor_sha256(tensors["producer_encoder"])
            ),
            "deferred_final_w2_seed_terminal_action_sha256": (
                lfm2_cpcf_terminal_action_tensor_sha256(
                    tensors["deferred_final_w2_seed"]
                )
            ),
            "foundation_interface_solve_result_sha256": foundation["result_sha256"],
            "foundation_active_interface_report_sha256": foundation[
                "active_interface_report_sha256"
            ],
            "semantic_layer_stack_result_sha256": stack["result_sha256"],
            "semantic_layer_stack_model_state_after_sha256": stack[
                "model_state_after_sha256"
            ],
            "deferred_final_w2_installation_result_sha256": installation[
                "result_sha256"
            ],
            "deferred_final_w2_installation_model_state_before_sha256": installation[
                "model_state_before_sha256"
            ],
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

    _loaded, authenticated = load_lfm2_postfold_reclosure_seed_archive(
        destination,
        foundation_interface_solve_receipt=foundation_interface_solve_receipt,
        semantic_layer_stack_receipt=semantic_layer_stack_receipt,
        deferred_final_w2_installation_receipt=(
            deferred_final_w2_installation_receipt
        ),
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    return authenticated


def load_lfm2_postfold_reclosure_seed_archive(
    root: str | os.PathLike[str],
    *,
    foundation_interface_solve_receipt: Mapping[str, Any],
    semantic_layer_stack_receipt: Mapping[str, Any],
    deferred_final_w2_installation_receipt: Mapping[str, Any],
    expected_manifest_sha256: str,
) -> tuple[Lfm2PostfoldReclosureSeedArchive, Mapping[str, Any]]:
    """Freshly load seeds after authenticating bytes and all receipt ancestry."""

    destination = _ordinary_root(root, create=False)
    expected_manifest = _digest(
        expected_manifest_sha256,
        label="expected reclosure seed archive manifest",
    )
    manifest_path = destination / MANIFEST_FILENAME
    tensor_path = destination / TENSOR_FILENAME
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not tensor_path.is_file()
        or tensor_path.is_symlink()
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed archive files are missing or indirect."
        )
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed archive manifest is invalid JSON."
        ) from exc
    if (
        not isinstance(manifest, dict)
        or raw_manifest != _canonical_json(manifest) + b"\n"
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed archive manifest is not canonical JSON."
        )
    body = dict(manifest)
    supplied = body.pop("manifest_sha256", None)
    if (
        body.get("schema") != LFM2_POSTFOLD_RECLOSURE_SEED_ARCHIVE_SCHEMA
        or supplied != expected_manifest
        or supplied != canonical_sha256(body)
        or body.get("tensor_filename") != TENSOR_FILENAME
        or body.get("tensor_file_sha256") != _file_sha256(tensor_path)
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed manifest, ancestry, or tensor file changed."
        )
    tensors = _load_tensor_payload(tensor_path)
    foundation, stack, installation = _authenticate_ancestry(
        mapped_parent_tied_table=tensors["mapped_parent_tied_table"],
        producer_encoder=tensors["producer_encoder"],
        deferred_final_w2_seed=tensors["deferred_final_w2_seed"],
        foundation_interface_solve_receipt=foundation_interface_solve_receipt,
        semantic_layer_stack_receipt=semantic_layer_stack_receipt,
        deferred_final_w2_installation_receipt=(
            deferred_final_w2_installation_receipt
        ),
    )
    expected_bindings = {
        "foundation_interface_solve_result_sha256": foundation["result_sha256"],
        "foundation_active_interface_report_sha256": foundation[
            "active_interface_report_sha256"
        ],
        "semantic_layer_stack_result_sha256": stack["result_sha256"],
        "semantic_layer_stack_model_state_after_sha256": stack[
            "model_state_after_sha256"
        ],
        "deferred_final_w2_installation_result_sha256": installation[
            "result_sha256"
        ],
        "deferred_final_w2_installation_model_state_before_sha256": installation[
            "model_state_before_sha256"
        ],
    }
    records = body.get("tensor_records")
    if (
        body.get("tensor_names") != list(_TENSOR_NAMES)
        or not isinstance(records, Mapping)
        or set(records) != set(_TENSOR_NAMES)
        or any(body.get(key) != value for key, value in expected_bindings.items())
        or body.get("mapped_parent_tied_table_foundation_sha256")
        != _foundation_tensor_sha256(tensors["mapped_parent_tied_table"])
        or body.get("producer_encoder_foundation_sha256")
        != _foundation_tensor_sha256(tensors["producer_encoder"])
        or body.get("deferred_final_w2_seed_terminal_action_sha256")
        != lfm2_cpcf_terminal_action_tensor_sha256(
            tensors["deferred_final_w2_seed"]
        )
    ):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed manifest, ancestry, or tensor file changed."
        )
    for name in _TENSOR_NAMES:
        if records[name] != _tensor_record(tensors[name], label=name):
            raise Lfm2PostfoldReclosureSeedArchiveError(
                f"Reclosure seed tensor {name} differs from its manifest."
            )

    authenticated = MappingProxyType(
        {**body, "manifest_sha256": supplied, "fresh_reload_verified": True}
    )
    archive = Lfm2PostfoldReclosureSeedArchive(
        mapped_parent_tied_table=tensors["mapped_parent_tied_table"],
        producer_encoder=tensors["producer_encoder"],
        deferred_final_w2_seed=tensors["deferred_final_w2_seed"],
        manifest=authenticated,
    )
    return archive, authenticated


def _load_tensor_payload(path: Path) -> Mapping[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - required dependency
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "safetensors is required for postfold reclosure seed archives."
        ) from exc
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != set(_TENSOR_NAMES):
        raise Lfm2PostfoldReclosureSeedArchiveError(
            "Reclosure seed archive tensor inventory is incomplete."
        )
    return MappingProxyType(
        {
            name: _materialized_matrix(tensors[name], label=name).clone()
            for name in _TENSOR_NAMES
        }
    )


__all__ = [
    "LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA",
    "LFM2_POSTFOLD_RECLOSURE_SEED_ARCHIVE_SCHEMA",
    "Lfm2PostfoldReclosureSeedArchive",
    "Lfm2PostfoldReclosureSeedArchiveError",
    "MANIFEST_FILENAME",
    "TENSOR_FILENAME",
    "load_lfm2_postfold_reclosure_seed_archive",
    "write_lfm2_postfold_reclosure_seed_archive",
]
