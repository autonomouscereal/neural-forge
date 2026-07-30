"""Authenticate and package fixed LFM2 carrier-route child state.

The offline carrier compiler publishes a complete evidence artifact containing
the table against which its routes were solved.  That source table is provenance,
not a weight replacement for a differently shaped production child.  Production
packaging therefore keeps only the independent ``C_in``/``C_out`` vectors and
their integer CSR metadata, then binds those routes to the already-folded,
untied input and readout tables of the standalone child.

This module is the package-layer authority for the publication ABI.  Repository
experiment runners may expose compatibility wrappers, but production code must
not import from ``experiments``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_counted_child_state import ContinuousChildStateAllocation


STANDALONE_BRIDGE_SCHEMA = "neural-forge-lfm2-carrier-reclosed-bridge-v1"
PACKAGED_ROUTE_SCHEMA = "neural-forge-lfm2-packaged-carrier-routes-v1"
PACKAGED_ROUTE_DIRECTORY = "semantic-carrier-routes"
PACKAGED_ROUTE_RECEIPT_FILENAME = "carrier-routes-receipt.json"
PACKAGED_ROUTE_TENSOR_FILENAME = "carrier-routes.safetensors"
PACKAGED_TARGET_ROUTE_KEY = "direct_target_row_to_semantic_id"
CARRIER_MATERIALIZATION_TENSOR_HASH_DOMAIN = (
    b"neural-forge-lfm2-semantic-carrier-tensor-v1\0"
)
CHILD_TABLE_TENSOR_HASH_DOMAIN = b"neural-forge-lfm2-child-table-v1\0"
PUBLISHED_TENSOR_KEYS = (
    "target_table",
    "target_to_donor",
    "input_offsets",
    "input_target_ids",
    "input_coefficients",
    "output_offsets",
    "output_target_ids",
    "output_coefficients",
)
PACKAGED_ROUTE_TENSOR_KEYS = (
    PACKAGED_TARGET_ROUTE_KEY,
    "input_offsets",
    "input_target_ids",
    "input_coefficients",
    "output_offsets",
    "output_target_ids",
    "output_coefficients",
    "span_offsets",
    "span_target_ids",
)
_STRUCTURAL_ROUTE_KEYS = frozenset(
    {
        PACKAGED_TARGET_ROUTE_KEY,
        "input_offsets",
        "input_target_ids",
        "output_offsets",
        "output_target_ids",
        "span_offsets",
        "span_target_ids",
    }
)
_COEFFICIENT_KEYS = ("input_coefficients", "output_coefficients")
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_pruning_calls": 0,
        "parameter_quantization_calls": 0,
        "parameter_sparsification_calls": 0,
        "parameter_sharing_calls": 0,
        "runtime_sidecar_parameters": 0,
    }
)
_PACKAGED_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_pruning_calls": 0,
        "parameter_quantization_calls": 0,
        "parameter_sparsification_calls": 0,
        "parameter_sharing_calls": 0,
    }
)


class Lfm2CarrierBridgePublicationError(ValueError):
    """A carrier publication or packaged route bundle is not authentic."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Lfm2CarrierBridgePublicationError(
            "PyTorch is required for LFM2 carrier-route authentication."
        ) from exc
    return torch


def _load_file(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Lfm2CarrierBridgePublicationError(
            "safetensors is required for LFM2 carrier-route authentication."
        ) from exc
    try:
        return load_file(str(path), device="cpu")
    except Exception as exc:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier tensor payload is not a readable local safetensors artifact."
        ) from exc


def _save_file(tensors: Mapping[str, Any], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Lfm2CarrierBridgePublicationError(
            "safetensors is required for LFM2 carrier-route packaging."
        ) from exc
    save_file(
        {
            str(name): tensor.detach().to(device="cpu").contiguous()
            for name, tensor in tensors.items()
        },
        str(path),
    )


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CarrierBridgePublicationError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest."
        )
    return value


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2CarrierBridgePublicationError(
                    f"{label} contains duplicate JSON key {key!r}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Lfm2CarrierBridgePublicationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CarrierBridgePublicationError(
            f"{label} is not strict UTF-8 JSON."
        ) from exc


def _ordinary_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2CarrierBridgePublicationError(f"{label} does not exist.") from exc
    if value.is_symlink() or not resolved.is_file():
        raise Lfm2CarrierBridgePublicationError(
            f"{label} must be an ordinary file."
        )
    return resolved


def _ordinary_directory(path: str | Path, label: str) -> Path:
    value = Path(path)
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2CarrierBridgePublicationError(f"{label} does not exist.") from exc
    if value.is_symlink() or not resolved.is_dir():
        raise Lfm2CarrierBridgePublicationError(
            f"{label} must be an ordinary directory."
        )
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable_file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _load_authenticated_tensors(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    source = _ordinary_file(path, label)
    before = _stable_file_identity(source)
    observed_sha256 = _file_sha256(source)
    if observed_sha256 != _digest(expected_sha256, f"{label} SHA-256"):
        raise Lfm2CarrierBridgePublicationError(f"{label} SHA-256 mismatch.")
    if _stable_file_identity(source) != before:
        raise Lfm2CarrierBridgePublicationError(
            f"{label} changed while being hashed."
        )
    tensors = _load_file(source)
    if _stable_file_identity(source) != before:
        raise Lfm2CarrierBridgePublicationError(
            f"{label} changed while being loaded."
        )
    return tensors, observed_sha256


def _sealed_json(
    value: Mapping[str, Any],
    *,
    schema: str,
    seal_field: str,
    label: str,
) -> dict[str, Any]:
    result = dict(value)
    supplied = result.pop(seal_field, None)
    if result.get("schema") != schema or supplied != canonical_sha256(result):
        raise Lfm2CarrierBridgePublicationError(
            f"{label} self-seal does not verify."
        )
    return {**result, seal_field: supplied}


def carrier_materialization_tensor_sha256(value: Any) -> str:
    """Hash a tensor under the carrier materialization receipt's v1 domain."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier materialization value must be a tensor."
        )
    tensor = value.detach()
    digest = hashlib.sha256()
    digest.update(CARRIER_MATERIALIZATION_TENSOR_HASH_DOMAIN)
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        tensor.to(device="cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return digest.hexdigest()


def child_table_tensor_sha256(value: Any) -> str:
    """Hash one already-folded live child table without changing its dtype."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise Lfm2CarrierBridgePublicationError("Child table must be a tensor.")
    tensor = value.detach()
    digest = hashlib.sha256()
    digest.update(CHILD_TABLE_TENSOR_HASH_DOMAIN)
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        tensor.to(device="cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return digest.hexdigest()


def _route_allocations(
    *,
    artifact_sha256: str,
    input_coefficients: Any,
    output_coefficients: Any,
) -> tuple[ContinuousChildStateAllocation, ContinuousChildStateAllocation]:
    return (
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_in",
            parameter_elements=int(input_coefficients.numel()),
            tensor_sha256=carrier_materialization_tensor_sha256(input_coefficients),
            artifact_sha256=artifact_sha256,
            dtype="float32",
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_out",
            parameter_elements=int(output_coefficients.numel()),
            tensor_sha256=carrier_materialization_tensor_sha256(output_coefficients),
            artifact_sha256=artifact_sha256,
            dtype="float32",
        ),
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2CarrierBridgePublication:
    root: Path
    receipt: Mapping[str, Any]
    tensors: Mapping[str, Any]
    allocations: tuple[
        ContinuousChildStateAllocation,
        ContinuousChildStateAllocation,
    ]

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])


@dataclass(frozen=True, slots=True)
class AuthenticatedPackagedLfm2CarrierRoutes:
    root: Path
    receipt: Mapping[str, Any]
    tensors: Mapping[str, Any]
    allocations: tuple[
        ContinuousChildStateAllocation,
        ContinuousChildStateAllocation,
    ]

    @property
    def package_sha256(self) -> str:
        return str(self.receipt["package_sha256"])


def load_lfm2_carrier_bridge_publication(
    directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
) -> AuthenticatedLfm2CarrierBridgePublication:
    """Fresh-load and authenticate one complete offline carrier publication."""

    torch = _torch()
    root = _ordinary_directory(directory, "carrier bridge publication")
    receipt_path = _ordinary_file(
        root / "bridge-receipt.json",
        "carrier bridge receipt",
    )
    before = _stable_file_identity(receipt_path)
    raw_receipt = _strict_json_bytes(
        receipt_path.read_bytes(),
        "carrier bridge receipt",
    )
    if _stable_file_identity(receipt_path) != before:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge receipt changed while being read."
        )
    if not isinstance(raw_receipt, Mapping):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge receipt must be an object."
        )
    receipt = _sealed_json(
        raw_receipt,
        schema=STANDALONE_BRIDGE_SCHEMA,
        seal_field="artifact_sha256",
        label="carrier bridge receipt",
    )
    artifact_sha256 = _digest(
        receipt.get("artifact_sha256"),
        "carrier bridge artifact SHA-256",
    )
    if expected_artifact_sha256 is not None and artifact_sha256 != _digest(
        expected_artifact_sha256,
        "expected carrier bridge artifact SHA-256",
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge publication belongs to a different artifact."
        )
    if (
        receipt.get("status") != "complete"
        or receipt.get("runtime_requires_parent") is not False
        or receipt.get("runtime_requires_external_sidecar") is not False
        or receipt.get("all_continuous_bridge_state_is_in_bridge_tensor_file")
        is not True
        or receipt.get("integer_csr_routes_are_structural_metadata") is not True
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge publication is not a complete standalone child artifact."
        )
    operation_audit = receipt.get("operation_audit")
    if not isinstance(operation_audit, Mapping) or any(
        isinstance(value, bool) or not isinstance(value, int) or value != 0
        for value in operation_audit.values()
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge publication does not have a zero-operation fold audit."
        )
    if receipt.get("bridge_tensor_file") != "bridge.safetensors":
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge receipt names an unsupported tensor payload."
        )
    tensors, observed_file_sha256 = _load_authenticated_tensors(
        root / "bridge.safetensors",
        expected_sha256=_digest(
            receipt.get("bridge_tensor_file_sha256"),
            "carrier bridge tensor file SHA-256",
        ),
        label="carrier bridge tensors",
    )
    if observed_file_sha256 != receipt["bridge_tensor_file_sha256"]:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge tensor file binding changed during fresh reload."
        )
    if set(tensors) != set(PUBLISHED_TENSOR_KEYS):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge tensor payload has an unexpected key set."
        )
    input_coefficients = tensors["input_coefficients"]
    output_coefficients = tensors["output_coefficients"]
    if (
        input_coefficients.dtype is not torch.float32
        or output_coefficients.dtype is not torch.float32
        or input_coefficients.ndim != 1
        or output_coefficients.ndim != 1
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge routes are not independent one-dimensional FP32 tensors."
        )
    materialization = receipt.get("materialization_receipt")
    if not isinstance(materialization, Mapping):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge materialization receipt is missing."
        )
    allocations = _route_allocations(
        artifact_sha256=artifact_sha256,
        input_coefficients=input_coefficients,
        output_coefficients=output_coefficients,
    )
    input_count = allocations[0].parameter_elements
    output_count = allocations[1].parameter_elements
    checks = {
        "input_coefficients_sha256": allocations[0].tensor_sha256
        == materialization.get("input_coefficients_sha256"),
        "output_coefficients_sha256": allocations[1].tensor_sha256
        == materialization.get("output_coefficients_sha256"),
        "continuous_semantic_coefficient_count": input_count + output_count
        == receipt.get("continuous_semantic_coefficient_count"),
        "continuous_fold_coefficient_count": input_count + output_count
        == materialization.get("continuous_fold_coefficient_count"),
        "physical_target_table_parameter_count": int(tensors["target_table"].numel())
        == receipt.get("physical_target_table_parameter_count"),
        "target_table_dtype": str(tensors["target_table"].dtype)
        == receipt.get("target_table_dtype"),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier bridge counted-state metadata does not match its loaded "
            "tensors: "
            + ", ".join(failed_checks)
            + "."
        )
    return AuthenticatedLfm2CarrierBridgePublication(
        root=root,
        receipt=MappingProxyType(dict(receipt)),
        tensors=MappingProxyType(dict(tensors)),
        allocations=allocations,
    )


def carrier_bridge_allocations_from_publication(
    directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
) -> tuple[ContinuousChildStateAllocation, ContinuousChildStateAllocation]:
    """Compatibility observer returning the two authenticated counted routes."""

    return load_lfm2_carrier_bridge_publication(
        directory,
        expected_artifact_sha256=expected_artifact_sha256,
    ).allocations


def load_authenticated_lfm2_span_evidence(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
    expected_evidence_sha256: str,
) -> dict[str, Any]:
    """Load one sealed donor-emission-span artifact for child packaging."""

    source = _ordinary_file(path, "semantic-step span evidence")
    before = _stable_file_identity(source)
    observed_file_sha256 = _file_sha256(source)
    if observed_file_sha256 != _digest(
        expected_file_sha256,
        "semantic-step span evidence file SHA-256",
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Semantic-step span evidence file SHA-256 mismatch."
        )
    raw = _strict_json_bytes(source.read_bytes(), "semantic-step span evidence")
    if _stable_file_identity(source) != before:
        raise Lfm2CarrierBridgePublicationError(
            "Semantic-step span evidence changed while being read."
        )
    if not isinstance(raw, Mapping):
        raise Lfm2CarrierBridgePublicationError(
            "Semantic-step span evidence must be an object."
        )
    try:
        from .hf_lfm2_semantic_step_token_fold import (
            verify_lfm2_semantic_step_span_evidence,
        )

        verified = verify_lfm2_semantic_step_span_evidence(raw)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CarrierBridgePublicationError(
            "Semantic-step span evidence is not authentic."
        ) from exc
    if verified.get("evidence_sha256") != _digest(
        expected_evidence_sha256,
        "semantic-step span evidence SHA-256",
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Semantic-step span evidence belongs to a different artifact."
        )
    return verified


def _child_tables(
    child_model: Any,
    *,
    expected_target_rows: int | None,
    expected_hidden_size: int | None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    torch = _torch()
    get_input = getattr(child_model, "get_input_embeddings", None)
    get_output = getattr(child_model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier-bound child must expose input and output embedding accessors."
        )
    input_weight = getattr(get_input(), "weight", None)
    output_weight = getattr(get_output(), "weight", None)
    expected_shape = None
    if expected_target_rows is not None or expected_hidden_size is not None:
        if (
            isinstance(expected_target_rows, bool)
            or not isinstance(expected_target_rows, int)
            or expected_target_rows <= 0
            or isinstance(expected_hidden_size, bool)
            or not isinstance(expected_hidden_size, int)
            or expected_hidden_size <= 0
        ):
            raise Lfm2CarrierBridgePublicationError(
                "Expected carrier-bound child table dimensions must be positive."
            )
        expected_shape = (expected_target_rows, expected_hidden_size)
    if (
        not isinstance(input_weight, torch.Tensor)
        or not isinstance(output_weight, torch.Tensor)
        or input_weight.ndim != 2
        or output_weight.ndim != 2
        or tuple(input_weight.shape) != tuple(output_weight.shape)
        or (expected_shape is not None and tuple(input_weight.shape) != expected_shape)
        or not torch.is_floating_point(input_weight)
        or not torch.is_floating_point(output_weight)
        or input_weight.is_complex()
        or output_weight.is_complex()
        or input_weight.layout != torch.strided
        or output_weight.layout != torch.strided
        or input_weight.device.type == "meta"
        or output_weight.device.type == "meta"
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier-bound child input/readout tables have the wrong materialized ABI."
        )
    input_storage = input_weight.untyped_storage().data_ptr()
    output_storage = output_weight.untyped_storage().data_ptr()
    if input_weight is output_weight or input_storage == output_storage:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier-bound C_in/C_out require independently stored untied child tables."
        )
    input_binding = {
        "parameter_path": "model.embed_tokens.weight",
        "route_state_id": "semantic-carrier-route.c_in",
        "shape": [int(size) for size in input_weight.shape],
        "dtype": str(input_weight.dtype).removeprefix("torch."),
        "tensor_sha256": child_table_tensor_sha256(input_weight),
    }
    output_binding = {
        "parameter_path": "lm_head.weight",
        "route_state_id": "semantic-carrier-route.c_out",
        "shape": [int(size) for size in output_weight.shape],
        "dtype": str(output_weight.dtype).removeprefix("torch."),
        "tensor_sha256": child_table_tensor_sha256(output_weight),
    }
    return input_weight, output_weight, input_binding, output_binding


def _package_root(child_directory: str | os.PathLike[str]) -> Path:
    child = _ordinary_directory(child_directory, "standalone child directory")
    return child / PACKAGED_ROUTE_DIRECTORY


def package_lfm2_carrier_routes_with_child(
    publication_directory: str | os.PathLike[str],
    child_directory: str | os.PathLike[str],
    child_model: Any,
    *,
    expected_artifact_sha256: str,
    span_evidence: Mapping[str, Any],
    expected_target_rows: int | None = None,
    expected_hidden_size: int | None = None,
) -> AuthenticatedPackagedLfm2CarrierRoutes:
    """Package only routes/CSR and bind them to an untied folded child."""

    source = load_lfm2_carrier_bridge_publication(
        publication_directory,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    child = _ordinary_directory(child_directory, "standalone child directory")
    package_root = child / PACKAGED_ROUTE_DIRECTORY
    if package_root.exists() or package_root.is_symlink():
        raise Lfm2CarrierBridgePublicationError(
            "Standalone child already contains carrier-route state."
        )
    _input, _output, input_binding, output_binding = _child_tables(
        child_model,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )
    try:
        from .hf_lfm2_semantic_step_token_fold import (
            lfm2_semantic_step_span_routing_tensors,
        )

        verified_span_evidence, span_offsets, span_target_ids = (
            lfm2_semantic_step_span_routing_tensors(span_evidence)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier-route packaging requires authentic complete donor spans."
        ) from exc
    temporary = Path(
        tempfile.mkdtemp(prefix=".carrier-routes.", dir=child)
    ).resolve(strict=True)
    try:
        route_tensors = {
            name: source.tensors[name]
            for name in PACKAGED_ROUTE_TENSOR_KEYS
            if name in source.tensors
        }
        route_tensors[PACKAGED_TARGET_ROUTE_KEY] = source.tensors[
            "target_to_donor"
        ]
        route_tensors["span_offsets"] = span_offsets
        route_tensors["span_target_ids"] = span_target_ids
        tensor_path = temporary / PACKAGED_ROUTE_TENSOR_FILENAME
        _save_file(route_tensors, tensor_path)
        tensor_file_sha256 = _file_sha256(tensor_path)
        body = {
            "schema": PACKAGED_ROUTE_SCHEMA,
            "status": "complete",
            "source_bridge_artifact_sha256": source.artifact_sha256,
            "source_bridge_tensor_file_sha256": source.receipt[
                "bridge_tensor_file_sha256"
            ],
            "route_tensor_file": PACKAGED_ROUTE_TENSOR_FILENAME,
            "route_tensor_file_sha256": tensor_file_sha256,
            "route_tensor_keys": list(PACKAGED_ROUTE_TENSOR_KEYS),
            "input_route_allocation": source.allocations[0].as_dict(),
            "output_route_allocation": source.allocations[1].as_dict(),
            "span_evidence_sha256": verified_span_evidence["evidence_sha256"],
            "input_table_binding": input_binding,
            "output_table_binding": output_binding,
            "input_output_tables_are_untied": True,
            "input_output_table_storage_is_independent": True,
            "source_target_table_packaged": False,
            "source_target_table_used_as_child_weight": False,
            "integer_csr_routes_are_structural_metadata": True,
            "continuous_route_state_counted_exactly_once": True,
            "continuous_semantic_coefficient_count": sum(
                value.parameter_elements for value in source.allocations
            ),
            "standalone_runtime": True,
            "external_runtime_state_required": False,
            "parameter_sharing_credit": False,
            "quantization_credit": False,
            "pruning_credit": False,
            "operation_audit": dict(_PACKAGED_ZERO_OPERATION_AUDIT),
        }
        receipt = {**body, "package_sha256": canonical_sha256(body)}
        (temporary / PACKAGED_ROUTE_RECEIPT_FILENAME).write_text(
            json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        authenticated = load_packaged_lfm2_carrier_routes(
            temporary,
            expected_source_artifact_sha256=source.artifact_sha256,
            child_model=child_model,
            expected_target_rows=expected_target_rows,
            expected_hidden_size=expected_hidden_size,
        )
        if authenticated.allocations != source.allocations:
            raise Lfm2CarrierBridgePublicationError(
                "Packaged carrier routes differ from their source publication."
            )
        os.rename(temporary, package_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return load_packaged_lfm2_carrier_routes(
        package_root,
        expected_source_artifact_sha256=source.artifact_sha256,
        child_model=child_model,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )


def load_packaged_lfm2_carrier_routes(
    package_or_child_directory: str | os.PathLike[str],
    *,
    expected_source_artifact_sha256: str,
    child_model: Any | None = None,
    expected_target_rows: int | None = None,
    expected_hidden_size: int | None = None,
) -> AuthenticatedPackagedLfm2CarrierRoutes:
    """Authenticate packaged route state, optionally against a fresh child load."""

    torch = _torch()
    supplied = _ordinary_directory(
        package_or_child_directory,
        "packaged carrier-route path",
    )
    direct_payload = (
        (supplied / PACKAGED_ROUTE_RECEIPT_FILENAME).is_file()
        and not (supplied / PACKAGED_ROUTE_RECEIPT_FILENAME).is_symlink()
        and (supplied / PACKAGED_ROUTE_TENSOR_FILENAME).is_file()
        and not (supplied / PACKAGED_ROUTE_TENSOR_FILENAME).is_symlink()
    )
    root = (
        supplied
        if supplied.name == PACKAGED_ROUTE_DIRECTORY or direct_payload
        else _ordinary_directory(
            supplied / PACKAGED_ROUTE_DIRECTORY,
            "packaged carrier-route directory",
        )
    )
    expected_inventory = {
        PACKAGED_ROUTE_RECEIPT_FILENAME,
        PACKAGED_ROUTE_TENSOR_FILENAME,
    }
    entries = tuple(root.iterdir())
    if (
        {path.name for path in entries} != expected_inventory
        or any(
            path.is_symlink()
            or not path.is_file()
            or getattr(path.stat(), "st_nlink", 1) != 1
            for path in entries
        )
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route directory contains unsealed state."
        )
    receipt_path = _ordinary_file(
        root / PACKAGED_ROUTE_RECEIPT_FILENAME,
        "packaged carrier-route receipt",
    )
    before = _stable_file_identity(receipt_path)
    raw_receipt = _strict_json_bytes(
        receipt_path.read_bytes(),
        "packaged carrier-route receipt",
    )
    if _stable_file_identity(receipt_path) != before:
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route receipt changed while being read."
        )
    if not isinstance(raw_receipt, Mapping):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route receipt must be an object."
        )
    receipt = _sealed_json(
        raw_receipt,
        schema=PACKAGED_ROUTE_SCHEMA,
        seal_field="package_sha256",
        label="packaged carrier-route receipt",
    )
    source_sha256 = _digest(
        receipt.get("source_bridge_artifact_sha256"),
        "packaged source bridge artifact SHA-256",
    )
    if source_sha256 != _digest(
        expected_source_artifact_sha256,
        "expected packaged source bridge artifact SHA-256",
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes belong to a different source publication."
        )
    if (
        receipt.get("status") != "complete"
        or receipt.get("route_tensor_file") != PACKAGED_ROUTE_TENSOR_FILENAME
        or receipt.get("route_tensor_keys") != list(PACKAGED_ROUTE_TENSOR_KEYS)
        or not isinstance(receipt.get("span_evidence_sha256"), str)
        or receipt.get("input_output_tables_are_untied") is not True
        or receipt.get("input_output_table_storage_is_independent") is not True
        or receipt.get("source_target_table_packaged") is not False
        or receipt.get("source_target_table_used_as_child_weight") is not False
        or receipt.get("integer_csr_routes_are_structural_metadata") is not True
        or receipt.get("continuous_route_state_counted_exactly_once") is not True
        or receipt.get("standalone_runtime") is not True
        or receipt.get("external_runtime_state_required") is not False
        or receipt.get("parameter_sharing_credit") is not False
        or receipt.get("quantization_credit") is not False
        or receipt.get("pruning_credit") is not False
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route policy is incomplete or unsafe."
        )
    operation_audit = receipt.get("operation_audit")
    if not isinstance(operation_audit, Mapping) or dict(operation_audit) != dict(
        _PACKAGED_ZERO_OPERATION_AUDIT
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes do not have a zero-operation fold audit."
        )
    tensors, _observed_sha256 = _load_authenticated_tensors(
        root / PACKAGED_ROUTE_TENSOR_FILENAME,
        expected_sha256=_digest(
            receipt.get("route_tensor_file_sha256"),
            "packaged route tensor file SHA-256",
        ),
        label="packaged carrier-route tensors",
    )
    if set(tensors) != set(PACKAGED_ROUTE_TENSOR_KEYS):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes have an unexpected tensor key set."
        )
    if "target_table" in tensors:
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes must not contain the source target table."
        )
    if any(
        not isinstance(tensors[name], torch.Tensor)
        or tensors[name].dtype is not torch.int64
        or tensors[name].ndim != 1
        for name in _STRUCTURAL_ROUTE_KEYS
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier CSR metadata must be one-dimensional int64 tensors."
        )
    input_coefficients = tensors["input_coefficients"]
    output_coefficients = tensors["output_coefficients"]
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.float32
        or value.ndim != 1
        or not bool(torch.isfinite(value).all())
        for value in (input_coefficients, output_coefficients)
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged C_in/C_out must be finite independent FP32 vectors."
        )
    if (
        int(tensors["input_offsets"][0].item()) != 0
        or int(tensors["input_offsets"][-1].item())
        != int(input_coefficients.numel())
        or int(tensors["output_offsets"][0].item()) != 0
        or int(tensors["output_offsets"][-1].item())
        != int(output_coefficients.numel())
        or int(tensors["input_target_ids"].numel())
        != int(input_coefficients.numel())
        or int(tensors["output_target_ids"].numel())
        != int(output_coefficients.numel())
        or int(tensors["span_offsets"][0].item()) != 0
        or int(tensors["span_offsets"][-1].item())
        != int(tensors["span_target_ids"].numel())
        or int(tensors["span_offsets"].numel())
        != int(tensors["input_offsets"].numel())
        or int(tensors["span_offsets"].numel())
        != int(tensors["output_offsets"].numel())
        or bool((tensors["span_offsets"][1:] <= tensors["span_offsets"][:-1]).any())
        or bool((tensors["input_offsets"][1:] < tensors["input_offsets"][:-1]).any())
        or bool((tensors["output_offsets"][1:] < tensors["output_offsets"][:-1]).any())
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier CSR offsets, indices, and coefficients disagree."
        )
    target_row_count = int(tensors[PACKAGED_TARGET_ROUTE_KEY].numel())
    if target_row_count <= 0 or any(
        bool((tensors[name] < 0).any())
        or bool((tensors[name] >= target_row_count).any())
        for name in ("input_target_ids", "output_target_ids", "span_target_ids")
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier route or span IDs leave the target vocabulary."
        )
    allocations = _route_allocations(
        artifact_sha256=source_sha256,
        input_coefficients=input_coefficients,
        output_coefficients=output_coefficients,
    )
    serialized_allocations = (
        receipt.get("input_route_allocation"),
        receipt.get("output_route_allocation"),
    )
    if any(
        not isinstance(serialized, Mapping)
        or serialized.get("state_id") != allocation.state_id
        or serialized.get("parameter_elements") != allocation.parameter_elements
        or serialized.get("tensor_sha256") != allocation.tensor_sha256
        or serialized.get("artifact_sha256") != allocation.artifact_sha256
        or serialized.get("dtype") != allocation.dtype
        for serialized, allocation in zip(
            serialized_allocations,
            allocations,
            strict=True,
        )
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route allocations differ from loaded tensors."
        )
    if receipt.get("continuous_semantic_coefficient_count") != sum(
        value.parameter_elements for value in allocations
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route count differs from its tensor payload."
        )
    input_binding = receipt.get("input_table_binding")
    output_binding = receipt.get("output_table_binding")
    if (
        not isinstance(input_binding, Mapping)
        or not isinstance(output_binding, Mapping)
        or input_binding.get("parameter_path") != "model.embed_tokens.weight"
        or output_binding.get("parameter_path") != "lm_head.weight"
        or input_binding.get("route_state_id") != allocations[0].state_id
        or output_binding.get("route_state_id") != allocations[1].state_id
        or input_binding.get("shape") != output_binding.get("shape")
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged routes are not bound to separate input/readout tables."
        )
    if child_model is not None:
        _input, _output, observed_input, observed_output = _child_tables(
            child_model,
            expected_target_rows=expected_target_rows,
            expected_hidden_size=expected_hidden_size,
        )
        if dict(input_binding) != observed_input or dict(output_binding) != observed_output:
            raise Lfm2CarrierBridgePublicationError(
                "Fresh child input/readout tables differ from carrier bindings."
            )
    elif expected_target_rows is not None or expected_hidden_size is not None:
        expected_shape = [expected_target_rows, expected_hidden_size]
        if input_binding.get("shape") != expected_shape:
            raise Lfm2CarrierBridgePublicationError(
                "Packaged carrier table binding has the wrong exact-body shape."
            )
    return AuthenticatedPackagedLfm2CarrierRoutes(
        root=root,
        receipt=MappingProxyType(dict(receipt)),
        tensors=MappingProxyType(dict(tensors)),
        allocations=allocations,
    )


def _bind_lfm2_carrier_runtime(
    child_model: Any,
    *,
    span_evidence: Mapping[str, Any],
    input_offsets: Any,
    input_target_ids: Any,
    input_coefficients: Any,
    output_offsets: Any,
    output_target_ids: Any,
    output_coefficients: Any,
    direct_target_row_to_semantic_id: Any,
    expected_allocations: tuple[
        ContinuousChildStateAllocation,
        ContinuousChildStateAllocation,
    ],
    expected_target_rows: int | None,
    expected_hidden_size: int | None,
) -> Any:
    input_table, output_table, _input_binding, _output_binding = _child_tables(
        child_model,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )
    try:
        from .hf_lfm2_semantic_step_token_fold import (
            bind_lfm2_separate_sparse_semantic_code_runtime,
        )

        runtime = bind_lfm2_separate_sparse_semantic_code_runtime(
            span_evidence,
            input_table,
            live_child_output_table=output_table,
            input_semantic_code_offsets=input_offsets,
            input_semantic_code_target_ids=input_target_ids,
            input_semantic_code_coefficients=input_coefficients,
            output_semantic_code_offsets=output_offsets,
            output_semantic_code_target_ids=output_target_ids,
            output_semantic_code_coefficients=output_coefficients,
            target_to_donor=direct_target_row_to_semantic_id,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier routes could not bind to the child's live untied tables."
        ) from exc
    runtime_receipt = runtime.as_dict()
    if (
        runtime_receipt.get("untied_live_tables") is not True
        or runtime_receipt.get("input_output_table_storage_shared") is not False
        or runtime_receipt.get(
            "input_continuous_semantic_code_coefficient_count"
        )
        != expected_allocations[0].parameter_elements
        or runtime_receipt.get(
            "output_continuous_semantic_code_coefficient_count"
        )
        != expected_allocations[1].parameter_elements
        or runtime_receipt.get("owned_fixed_continuous_state_count")
        != sum(value.parameter_elements for value in expected_allocations)
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Fresh dual-table carrier runtime changed its counted route ABI."
        )
    return runtime


def bind_lfm2_carrier_bridge_publication_to_child(
    child_model: Any,
    publication_directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str,
    span_evidence: Mapping[str, Any],
    expected_target_rows: int | None = None,
    expected_hidden_size: int | None = None,
) -> Any:
    """Bind authenticated source routes transiently for in-memory review.

    The source publication's target table is authenticated as provenance but is
    never installed into or used by the differently shaped child.  Only its two
    route vectors and integer structure are bound to the child's live tables.
    """

    source = load_lfm2_carrier_bridge_publication(
        publication_directory,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    try:
        from .hf_lfm2_semantic_step_token_fold import (
            verify_lfm2_semantic_step_span_evidence,
        )

        verified_spans = verify_lfm2_semantic_step_span_evidence(span_evidence)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CarrierBridgePublicationError(
            "Carrier routes require authentic complete semantic spans."
        ) from exc
    tensors = source.tensors
    return _bind_lfm2_carrier_runtime(
        child_model,
        span_evidence=verified_spans,
        input_offsets=tensors["input_offsets"],
        input_target_ids=tensors["input_target_ids"],
        input_coefficients=tensors["input_coefficients"],
        output_offsets=tensors["output_offsets"],
        output_target_ids=tensors["output_target_ids"],
        output_coefficients=tensors["output_coefficients"],
        direct_target_row_to_semantic_id=tensors["target_to_donor"],
        expected_allocations=source.allocations,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )


def bind_packaged_lfm2_carrier_routes(
    child_model: Any,
    packaged: AuthenticatedPackagedLfm2CarrierRoutes,
    *,
    expected_target_rows: int | None = None,
    expected_hidden_size: int | None = None,
) -> Any:
    """Bind packaged C_in/C_out to the fresh child's independent live tables."""

    if not isinstance(packaged, AuthenticatedPackagedLfm2CarrierRoutes):
        raise Lfm2CarrierBridgePublicationError(
            "packaged must be an authenticated carrier-route bundle."
        )
    # Never trust a retained/mutable tensor mapping at bind time.  Reopen the
    # exact package bytes and bind its table seals to this live fresh child.
    # This simultaneously rejects coefficient mutation, package replacement,
    # and same-shape input/readout table swaps.
    authenticated = load_packaged_lfm2_carrier_routes(
        packaged.root,
        expected_source_artifact_sha256=str(
            packaged.receipt["source_bridge_artifact_sha256"]
        ),
        child_model=child_model,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )
    torch = _torch()
    if (
        authenticated.package_sha256 != packaged.package_sha256
        or authenticated.allocations != packaged.allocations
        or set(authenticated.tensors) != set(packaged.tensors)
        or any(
            not isinstance(packaged.tensors[name], torch.Tensor)
            or packaged.tensors[name].dtype != authenticated.tensors[name].dtype
            or tuple(packaged.tensors[name].shape)
            != tuple(authenticated.tensors[name].shape)
            or not bool(
                torch.equal(
                    packaged.tensors[name].detach().to(device="cpu"),
                    authenticated.tensors[name],
                )
            )
            for name in authenticated.tensors
        )
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier routes changed between authentication and live binding."
        )
    packaged = authenticated
    tensors = packaged.tensors
    offsets = tensors["span_offsets"]
    target_ids = tensors["span_target_ids"]
    spans = []
    for donor_id in range(int(offsets.numel()) - 1):
        start = int(offsets[donor_id].item())
        stop = int(offsets[donor_id + 1].item())
        spans.append([int(value) for value in target_ids[start:stop].tolist()])
    try:
        from .hf_lfm2_semantic_step_token_fold import (
            seal_lfm2_semantic_step_span_evidence,
        )

        span_evidence = seal_lfm2_semantic_step_span_evidence(
            spans,
            target_row_count=int(tensors[PACKAGED_TARGET_ROUTE_KEY].numel()),
            lineage={
                "packaged_span_evidence_sha256": packaged.receipt[
                    "span_evidence_sha256"
                ],
                "carrier_route_package_sha256": packaged.package_sha256,
                "derived_only_from_packaged_integer_span_tensors": True,
            },
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes could not bind to fresh child tables."
        ) from exc
    return _bind_lfm2_carrier_runtime(
        child_model,
        span_evidence=span_evidence,
        input_offsets=tensors["input_offsets"],
        input_target_ids=tensors["input_target_ids"],
        input_coefficients=tensors["input_coefficients"],
        output_offsets=tensors["output_offsets"],
        output_target_ids=tensors["output_target_ids"],
        output_coefficients=tensors["output_coefficients"],
        direct_target_row_to_semantic_id=tensors[PACKAGED_TARGET_ROUTE_KEY],
        expected_allocations=packaged.allocations,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )


__all__ = [
    "CARRIER_MATERIALIZATION_TENSOR_HASH_DOMAIN",
    "PACKAGED_ROUTE_DIRECTORY",
    "PACKAGED_ROUTE_RECEIPT_FILENAME",
    "PACKAGED_ROUTE_SCHEMA",
    "PACKAGED_ROUTE_TENSOR_FILENAME",
    "PACKAGED_ROUTE_TENSOR_KEYS",
    "PACKAGED_TARGET_ROUTE_KEY",
    "PUBLISHED_TENSOR_KEYS",
    "STANDALONE_BRIDGE_SCHEMA",
    "AuthenticatedLfm2CarrierBridgePublication",
    "AuthenticatedPackagedLfm2CarrierRoutes",
    "Lfm2CarrierBridgePublicationError",
    "carrier_bridge_allocations_from_publication",
    "carrier_materialization_tensor_sha256",
    "bind_lfm2_carrier_bridge_publication_to_child",
    "bind_packaged_lfm2_carrier_routes",
    "child_table_tensor_sha256",
    "load_lfm2_carrier_bridge_publication",
    "load_authenticated_lfm2_span_evidence",
    "load_packaged_lfm2_carrier_routes",
    "package_lfm2_carrier_routes_with_child",
]
