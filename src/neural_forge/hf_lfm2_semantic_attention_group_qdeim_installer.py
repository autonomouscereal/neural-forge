"""Transactional installer for sealed-cache intact-group attention Q-DEIM.

The cache adapter compiles one coupled dense Q/K/V/O candidate.  This module
binds that candidate to an authenticated blank target attention block, carries
the direct parent Q/K RMSNorm weights alongside it, and publishes all six
ordinary Parameters in the existing atomic attention transaction.  Blank
Q/K/V/O matrices are exactly zero.  Q/K RMSNorm seeds are either the legacy
exact-zero blank or the canonical exact-one neutral initialization.

Preparation and installation are deliberately separate.  The prepared object
authenticates the live blank target, the cache candidate, the direct-parent
norm inventory, topology, Parameter identities/storage, and gradient state.
Any evidence deficit is returned before installation and is never replaced by
an incumbent, random, structural-proxy, or split QK/VO fallback.

Numeric diagnostics and manual score thresholds have no publication or veto
authority here.  The dense candidate is installed when its structural and
authentication contracts are valid; complete decoded behavior remains the
primary intelligence evidence downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_attention_group_qdeim import (
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_source_state_sha256,
    lfm2_attention_group_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2DenseQkvoSourceInventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_semantic_attention_group_qdeim_adapter import (
    LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_ADAPTER_SCHEMA,
    Lfm2SemanticAttentionGroupQdeimCandidate,
    Lfm2SemanticAttentionGroupQdeimEvidenceDeficit,
    compile_lfm2_semantic_attention_group_qdeim_from_cache,
)
from neural_forge import hf_lfm2_semantic_attention_refold_adapter as _transaction
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_NORM_INVENTORY_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-group-qdeim-parent-norm-inventory-v1"
)
LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_PREPARATION_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-group-qdeim-preparation-v1"
)
LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-group-qdeim-installation-v1"
)

_QKVO_ROLES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
)
_NORM_ROLES = ("q_layernorm.weight", "k_layernorm.weight")
_ALL_ROLES = (*_QKVO_ROLES, *_NORM_ROLES)
_QKVO_SEED_POLICY = "exact-zero-blank-required"
_NORM_SEED_POLICY = "exact-zero-legacy-blank-or-exact-one-canonical-neutral"
_QKVO_ZERO_SEED_KIND = "exact-zero-blank"
_NORM_ZERO_SEED_KIND = "exact-zero-legacy-blank"
_NORM_ONE_SEED_KIND = "exact-one-canonical-neutral"
_ZERO_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "parameter_aliases_created": 0,
    }
)


class Lfm2SemanticAttentionGroupQdeimInstallationError(RuntimeError):
    """Candidate, target, norm lineage, or transactional installation failed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Semantic attention group-QDEIM installation requires PyTorch."
        ) from exc
    return torch


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _authenticated_qdeim_compute_execution(
    value: Any,
    *,
    expected_requested_device: str,
    label: str,
) -> Mapping[str, Any]:
    if expected_requested_device == "cpu":
        backend = "cpu"
        device_index = None
    elif isinstance(expected_requested_device, str):
        prefix, separator, index_text = expected_requested_device.partition(":")
        if prefix != "cuda" or separator != ":" or not index_text.isdecimal():
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                f"{label} requested an invalid fold compute device."
            )
        backend = "cuda"
        device_index = int(index_text)
    else:
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} requested an invalid fold compute device."
        )
    if not isinstance(value, Mapping):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} omitted its Q-DEIM compute execution receipt."
        )
    body = dict(value)
    supplied = body.pop("receipt_sha256", None)
    _digest(supplied, label=f"{label} compute receipt SHA-256")
    expected_cuda = backend == "cuda"
    if (
        supplied != canonical_sha256(body)
        or body.get("schema") != "neural-forge-lfm2-attention-qdeim-compute-v1"
        or body.get("requested_device") != expected_requested_device
        or body.get("resolved_device") != expected_requested_device
        or body.get("backend") != backend
        or body.get("compute_dtype") != "float64"
        or body.get("authenticated_source_archive_device") != "cpu"
        or body.get("active_layer_staged_to_compute_device") is not expected_cuda
        or body.get("materialized_candidate_device") != "cpu"
        or body.get("system_memory_is_immutable_archive") is not True
        or body.get("gpu_is_ephemeral_dense_algebra_workspace") is not expected_cuda
        or body.get("silent_device_fallback_performed") is not False
        or body.get("training_steps") != 0
        or body.get("backward_calls") != 0
        or body.get("optimizer_steps") != 0
        or (
            expected_cuda
            and (
                body.get("device_index") != device_index
                or not isinstance(body.get("device_name"), str)
                or not body.get("device_name")
                or not isinstance(body.get("compute_capability"), list)
                or len(body["compute_capability"]) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, int) or part < 0
                    for part in body["compute_capability"]
                )
            )
        )
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} did not authenticate requested-device FP64 Q-DEIM execution."
        )
    return MappingProxyType({**body, "receipt_sha256": supplied})


def _tensor_sha256(value: Any, *, label: str) -> str:
    try:
        return lfm2_authenticated_tensor_sha256(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} is not one hashable materialized tensor."
        ) from exc


def _candidate_state_sha256(state: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-attention-group-qdeim-candidate-state-v1",
            "tensor_sha256": {
                name: _tensor_sha256(state[name], label=f"candidate {name}")
                for name in sorted(_QKVO_ROLES)
            },
        }
    )


def _detached_tensor(
    value: Any,
    *,
    label: str,
    ndim: int,
    dtype: Any | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() <= 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or value.grad_fn is not None
        or (dtype is not None and value.dtype != dtype)
        or not bool(torch.isfinite(value).all().item())
    ):
        expected = "" if dtype is None else f" {dtype}"
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} must be one detached finite dense rank-{ndim}{expected} tensor."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _self_sealed_report(
    value: Mapping[str, Any],
    *,
    seal_name: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} must be one sealed mapping."
        )
    thawed = _deep_thaw(value)
    supplied = thawed.pop(seal_name, None)
    if supplied != canonical_sha256(thawed):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} self-seal is invalid."
        )
    return MappingProxyType({**thawed, seal_name: supplied})


def _target_seed_authentication(
    snapshot: _TargetSnapshot,
) -> Mapping[str, Mapping[str, Any]]:
    """Authenticate the only blank seeds this installer is allowed to replace."""

    if not _verify_target_snapshot(snapshot):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Target seed authentication requires one intact target snapshot."
        )
    torch = _torch()
    prefix = f"model.layers.{snapshot.layer_index}.self_attn."
    authentication: dict[str, Mapping[str, Any]] = {}
    for role in _QKVO_ROLES:
        name = prefix + role
        value = snapshot.parameter_values[name]
        if not torch.equal(value, torch.zeros_like(value)):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Coupled attention group-QDEIM installation requires exact-zero "
                f"blank Q/K/V/O matrices; {role!r} is not exact-zero."
            )
        authentication[role] = MappingProxyType(
            {
                "state_key": name,
                "seed_policy": _QKVO_SEED_POLICY,
                "seed_kind": _QKVO_ZERO_SEED_KIND,
                "seed_tensor_sha256": snapshot.parameter_sha256[name],
            }
        )
    for role in _NORM_ROLES:
        name = prefix + role
        value = snapshot.parameter_values[name]
        if torch.equal(value, torch.zeros_like(value)):
            seed_kind = _NORM_ZERO_SEED_KIND
        elif torch.equal(value, torch.ones_like(value)):
            seed_kind = _NORM_ONE_SEED_KIND
        else:
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Coupled attention group-QDEIM Q/K norm seeds must each be "
                "exact-zero legacy blanks or exact-one canonical neutral seeds; "
                f"{role!r} is neither."
            )
        authentication[role] = MappingProxyType(
            {
                "state_key": name,
                "seed_policy": _NORM_SEED_POLICY,
                "seed_kind": seed_kind,
                "seed_tensor_sha256": snapshot.parameter_sha256[name],
            }
        )
    return MappingProxyType(authentication)


def _target_seed_authentication_body(
    snapshot: _TargetSnapshot,
) -> dict[str, dict[str, Any]]:
    return {
        role: dict(receipt)
        for role, receipt in _target_seed_authentication(snapshot).items()
    }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionGroupQdeimNormInventory:
    """Sealed direct-parent Q/K RMSNorm source lineage."""

    layer_index: int
    source_parent_identity_sha256: str
    source_qkvo_inventory_sha256: str
    query_norm_receipt: Mapping[str, Any]
    key_norm_receipt: Mapping[str, Any]
    inventory_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Norm inventory layer_index must be a nonnegative integer."
            )
        parent_sha = _digest(
            self.source_parent_identity_sha256,
            label="source parent identity SHA-256",
        )
        qkvo_sha = _digest(
            self.source_qkvo_inventory_sha256,
            label="source Q/K/V/O inventory SHA-256",
        )
        normalized: dict[str, Mapping[str, Any]] = {}
        for role, supplied in (
            ("q_layernorm.weight", self.query_norm_receipt),
            ("k_layernorm.weight", self.key_norm_receipt),
        ):
            receipt = dict(supplied)
            shape = receipt.get("shape")
            dtype = receipt.get("dtype")
            tensor_sha = receipt.get("tensor_sha256")
            if (
                not isinstance(shape, list)
                or len(shape) != 1
                or isinstance(shape[0], bool)
                or not isinstance(shape[0], int)
                or shape[0] <= 0
                or dtype != "torch.bfloat16"
                or _digest(tensor_sha, label=f"{role} tensor SHA-256") != tensor_sha
            ):
                raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                    f"Direct-parent norm receipt {role!r} is invalid."
                )
            normalized[role] = MappingProxyType(
                {
                    "shape": list(shape),
                    "dtype": dtype,
                    "tensor_sha256": tensor_sha,
                    "logical_parameters": shape[0],
                }
            )
        if (
            normalized["q_layernorm.weight"]["shape"]
            != normalized["k_layernorm.weight"]["shape"]
        ):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Direct-parent Q/K norm receipts must share one head dimension."
            )
        body = {
            "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_NORM_INVENTORY_SCHEMA,
            "layer_index": self.layer_index,
            "source_parent_identity_sha256": parent_sha,
            "source_qkvo_inventory_sha256": qkvo_sha,
            "tensor_receipts": {role: dict(normalized[role]) for role in _NORM_ROLES},
            "direct_parent_norms_only": True,
            "head_averaging_or_refit_performed": False,
        }
        computed = canonical_sha256(body)
        if self.inventory_sha256 and self.inventory_sha256 != computed:
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Direct-parent norm inventory does not match its supplied seal."
            )
        object.__setattr__(self, "source_parent_identity_sha256", parent_sha)
        object.__setattr__(self, "source_qkvo_inventory_sha256", qkvo_sha)
        object.__setattr__(
            self,
            "query_norm_receipt",
            normalized["q_layernorm.weight"],
        )
        object.__setattr__(
            self,
            "key_norm_receipt",
            normalized["k_layernorm.weight"],
        )
        object.__setattr__(self, "inventory_sha256", computed)

    @property
    def head_dim(self) -> int:
        return int(self.query_norm_receipt["shape"][0])

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_NORM_INVENTORY_SCHEMA,
            "layer_index": self.layer_index,
            "source_parent_identity_sha256": self.source_parent_identity_sha256,
            "source_qkvo_inventory_sha256": self.source_qkvo_inventory_sha256,
            "tensor_receipts": {
                "q_layernorm.weight": dict(self.query_norm_receipt),
                "k_layernorm.weight": dict(self.key_norm_receipt),
            },
            "direct_parent_norms_only": True,
            "head_averaging_or_refit_performed": False,
        }

    def verify(self) -> str:
        if canonical_sha256(self.authentication_body()) != self.inventory_sha256:
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Direct-parent norm inventory changed after sealing."
            )
        return self.inventory_sha256

    def as_dict(self) -> dict[str, Any]:
        return {**self.authentication_body(), "inventory_sha256": self.inventory_sha256}


def seal_lfm2_semantic_attention_group_qdeim_norm_inventory(
    *,
    layer_index: int,
    source_parent_identity_sha256: str,
    source_inventory: Lfm2DenseQkvoSourceInventory,
    source_query_norm_weight: Any,
    source_key_norm_weight: Any,
) -> Lfm2SemanticAttentionGroupQdeimNormInventory:
    """Seal the exact direct-parent Q/K norm tensors before target preparation."""

    torch = _torch()
    if (
        not isinstance(source_inventory, Lfm2DenseQkvoSourceInventory)
        or source_inventory.verify() != source_inventory.inventory_sha256
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "source_inventory is not one authenticated dense Q/K/V/O inventory."
        )
    query = _detached_tensor(
        source_query_norm_weight,
        label="source_query_norm_weight",
        ndim=1,
        dtype=torch.bfloat16,
    )
    key = _detached_tensor(
        source_key_norm_weight,
        label="source_key_norm_weight",
        ndim=1,
        dtype=torch.bfloat16,
    )
    if (
        tuple(query.shape) != tuple(key.shape)
        or int(torch.count_nonzero(query).item()) == 0
        or int(torch.count_nonzero(key).item()) == 0
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Direct-parent Q/K norm tensors must share one BF16 head axis and "
            "must not be all-zero."
        )

    def receipt(value: Any) -> Mapping[str, Any]:
        return {
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype),
            "tensor_sha256": _tensor_sha256(value, label="direct-parent norm"),
        }

    return Lfm2SemanticAttentionGroupQdeimNormInventory(
        layer_index=layer_index,
        source_parent_identity_sha256=source_parent_identity_sha256,
        source_qkvo_inventory_sha256=source_inventory.inventory_sha256,
        query_norm_receipt=receipt(query),
        key_norm_receipt=receipt(key),
    )


@dataclass(slots=True)
class _TargetSnapshot:
    layer_index: int
    model_state_sha256: str
    topology_sha256: str
    parameter_count: int
    parameter_values: Mapping[str, Any]
    parameter_sha256: Mapping[str, str]
    parameter_ids: Mapping[str, int]
    parameter_pointers: Mapping[str, int]
    parameter_shapes: Mapping[str, tuple[int, ...]]
    parameter_dtypes: Mapping[str, Any]
    parameter_requires_grad: Mapping[str, bool]
    gradient_references: Mapping[str, Any | None]
    gradient_values: Mapping[str, Any | None]
    gradient_ids: Mapping[str, int | None]
    gradient_sha256: Mapping[str, str | None]
    bundle_sha256: str


def _parameter_storage_span(value: Any, *, label: str) -> tuple[str, int, int]:
    """Return one conservative byte span for a live strided Parameter view."""

    torch = _torch()
    if (
        not isinstance(value, torch.nn.Parameter)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() <= 0
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} is not one materialized strided Parameter."
        )
    element_size = int(value.element_size())
    minimum_offset = int(value.storage_offset())
    maximum_offset = int(value.storage_offset())
    for size, stride in zip(value.shape, value.stride(), strict=True):
        extent = (int(size) - 1) * int(stride)
        minimum_offset += min(0, extent)
        maximum_offset += max(0, extent)
    storage = value.untyped_storage()
    storage_base = int(storage.data_ptr())
    start = storage_base + minimum_offset * element_size
    end = storage_base + (maximum_offset + 1) * element_size
    if (
        start < storage_base
        or end <= start
        or end > storage_base + int(storage.nbytes())
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            f"{label} exposes an invalid Parameter storage span."
        )
    device = f"{value.device.type}:{value.device.index}"
    return device, start, end


def _storage_spans_overlap(
    left: tuple[str, int, int],
    right: tuple[str, int, int],
) -> bool:
    return bool(left[0] == right[0] and left[1] < right[2] and right[1] < left[2])


def _require_independent_target_storage(
    target_model: Any,
    parameters: Mapping[str, Any],
) -> None:
    """Reject any model Parameter whose storage overlaps the six write targets."""

    target_spans = {
        name: _parameter_storage_span(value, label=f"target {name}")
        for name, value in parameters.items()
    }
    all_named = tuple(target_model.named_parameters(remove_duplicate=False))
    for target_name, target_span in target_spans.items():
        for other_name, other in all_named:
            if other_name == target_name:
                continue
            other_span = _parameter_storage_span(
                other,
                label=f"model Parameter {other_name}",
            )
            if _storage_spans_overlap(target_span, other_span):
                raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                    "Target attention Parameters must have independent storage; "
                    f"{target_name!r} overlaps {other_name!r}."
                )


def _target_snapshot(target_model: Any, *, layer_index: int) -> _TargetSnapshot:
    try:
        parameters = _transaction._parameter_bundle(target_model, layer_index)
        topology = _transaction._topology_sha256(target_model)
        count = _transaction._parameter_count(target_model)
        model_sha = fingerprint_lfm2_parent(target_model).identity_sha256
    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Target does not expose one ordinary six-Parameter LFM2 attention block."
        ) from exc
    _require_independent_target_storage(target_model, parameters)
    torch = _torch()
    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    gradient_references: dict[str, Any | None] = {}
    gradient_values: dict[str, Any | None] = {}
    gradient_hashes: dict[str, str | None] = {}
    for name, parameter in parameters.items():
        expected_ndim = 1 if name.endswith("layernorm.weight") else 2
        if (
            parameter.dtype != torch.bfloat16
            or parameter.ndim != expected_ndim
            or parameter.layout != torch.strided
            or parameter.device.type == "meta"
            or not bool(torch.isfinite(parameter).all().item())
        ):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                f"Target attention Parameter {name!r} is not finite dense BF16."
            )
        frozen = parameter.detach().to(device="cpu").contiguous().clone()
        values[name] = frozen
        hashes[name] = _tensor_sha256(frozen, label=f"target {name}")
        gradient_references[name] = parameter.grad
        gradient_values[name] = (
            None
            if parameter.grad is None
            else parameter.grad.detach().to(device="cpu").contiguous().clone()
        )
        gradient_hashes[name] = (
            None
            if parameter.grad is None
            else _tensor_sha256(parameter.grad, label=f"target gradient {name}")
        )
    body = {
        "schema": "neural-forge-lfm2-attention-six-parameter-live-bundle-v1",
        "layer_index": layer_index,
        "model_state_sha256": model_sha,
        "topology_sha256": topology,
        "parameter_count": count,
        "parameter_sha256": dict(sorted(hashes.items())),
        "parameter_ids": {
            name: id(parameter) for name, parameter in parameters.items()
        },
        "parameter_pointers": {
            name: int(parameter.data_ptr()) for name, parameter in parameters.items()
        },
        "parameter_shapes": {
            name: list(parameter.shape) for name, parameter in parameters.items()
        },
        "parameter_dtypes": {
            name: str(parameter.dtype) for name, parameter in parameters.items()
        },
        "parameter_requires_grad": {
            name: bool(parameter.requires_grad)
            for name, parameter in parameters.items()
        },
        "gradient_ids": {
            name: None if parameter.grad is None else id(parameter.grad)
            for name, parameter in parameters.items()
        },
        "gradient_sha256": dict(sorted(gradient_hashes.items())),
    }
    snapshot = _TargetSnapshot(
        layer_index=layer_index,
        model_state_sha256=model_sha,
        topology_sha256=topology,
        parameter_count=count,
        parameter_values=MappingProxyType(values),
        parameter_sha256=MappingProxyType(hashes),
        parameter_ids=MappingProxyType(
            {name: id(parameter) for name, parameter in parameters.items()}
        ),
        parameter_pointers=MappingProxyType(
            {name: int(parameter.data_ptr()) for name, parameter in parameters.items()}
        ),
        parameter_shapes=MappingProxyType(
            {name: tuple(parameter.shape) for name, parameter in parameters.items()}
        ),
        parameter_dtypes=MappingProxyType(
            {name: parameter.dtype for name, parameter in parameters.items()}
        ),
        parameter_requires_grad=MappingProxyType(
            {
                name: bool(parameter.requires_grad)
                for name, parameter in parameters.items()
            }
        ),
        gradient_references=MappingProxyType(gradient_references),
        gradient_values=MappingProxyType(gradient_values),
        gradient_ids=MappingProxyType(
            {
                name: None if parameter.grad is None else id(parameter.grad)
                for name, parameter in parameters.items()
            }
        ),
        gradient_sha256=MappingProxyType(gradient_hashes),
        bundle_sha256=canonical_sha256(body),
    )
    if not _verify_target_snapshot(snapshot):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Captured target snapshot failed its internal authentication."
        )
    return snapshot


def _snapshot_authentication_body(snapshot: _TargetSnapshot) -> dict[str, Any]:
    return {
        "schema": "neural-forge-lfm2-attention-six-parameter-live-bundle-v1",
        "layer_index": snapshot.layer_index,
        "model_state_sha256": snapshot.model_state_sha256,
        "topology_sha256": snapshot.topology_sha256,
        "parameter_count": snapshot.parameter_count,
        "parameter_sha256": dict(sorted(snapshot.parameter_sha256.items())),
        "parameter_ids": dict(snapshot.parameter_ids),
        "parameter_pointers": dict(snapshot.parameter_pointers),
        "parameter_shapes": {
            name: list(shape) for name, shape in snapshot.parameter_shapes.items()
        },
        "parameter_dtypes": {
            name: str(dtype) for name, dtype in snapshot.parameter_dtypes.items()
        },
        "parameter_requires_grad": dict(snapshot.parameter_requires_grad),
        "gradient_ids": dict(snapshot.gradient_ids),
        "gradient_sha256": dict(sorted(snapshot.gradient_sha256.items())),
    }


def _verify_target_snapshot(snapshot: _TargetSnapshot) -> bool:
    """Rehash every mutable backup before it can authorize or drive rollback."""

    try:
        if not isinstance(snapshot, _TargetSnapshot):
            return False
        expected_names = {
            f"model.layers.{snapshot.layer_index}.self_attn.{role}"
            for role in _ALL_ROLES
        }
        maps = (
            snapshot.parameter_values,
            snapshot.parameter_sha256,
            snapshot.parameter_ids,
            snapshot.parameter_pointers,
            snapshot.parameter_shapes,
            snapshot.parameter_dtypes,
            snapshot.parameter_requires_grad,
            snapshot.gradient_references,
            snapshot.gradient_values,
            snapshot.gradient_ids,
            snapshot.gradient_sha256,
        )
        if any(set(value) != expected_names for value in maps):
            return False
        for name in expected_names:
            frozen = snapshot.parameter_values[name]
            if (
                tuple(frozen.shape) != snapshot.parameter_shapes[name]
                or frozen.dtype != snapshot.parameter_dtypes[name]
                or _tensor_sha256(frozen, label=f"snapshot {name}")
                != snapshot.parameter_sha256[name]
            ):
                return False
            reference = snapshot.gradient_references[name]
            frozen_gradient = snapshot.gradient_values[name]
            gradient_id = snapshot.gradient_ids[name]
            gradient_sha = snapshot.gradient_sha256[name]
            if gradient_sha is None:
                if (
                    reference is not None
                    or frozen_gradient is not None
                    or gradient_id is not None
                ):
                    return False
            elif (
                reference is None
                or frozen_gradient is None
                or id(reference) != gradient_id
                or _tensor_sha256(
                    frozen_gradient,
                    label=f"snapshot frozen gradient {name}",
                )
                != gradient_sha
            ):
                return False
        return bool(
            canonical_sha256(_snapshot_authentication_body(snapshot))
            == snapshot.bundle_sha256
        )
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
        return False


def _snapshot_matches(target_model: Any, snapshot: _TargetSnapshot) -> bool:
    if not _verify_target_snapshot(snapshot):
        return False
    try:
        current = _target_snapshot(target_model, layer_index=snapshot.layer_index)
    except Lfm2SemanticAttentionGroupQdeimInstallationError:
        return False
    return bool(
        current.model_state_sha256 == snapshot.model_state_sha256
        and current.topology_sha256 == snapshot.topology_sha256
        and current.parameter_count == snapshot.parameter_count
        and current.parameter_sha256 == snapshot.parameter_sha256
        and current.parameter_ids == snapshot.parameter_ids
        and current.parameter_pointers == snapshot.parameter_pointers
        and current.parameter_shapes == snapshot.parameter_shapes
        and current.parameter_dtypes == snapshot.parameter_dtypes
        and current.parameter_requires_grad == snapshot.parameter_requires_grad
        and current.gradient_ids == snapshot.gradient_ids
        and current.gradient_sha256 == snapshot.gradient_sha256
        and current.bundle_sha256 == snapshot.bundle_sha256
    )


def _restore_compile_snapshot(target_model: Any, snapshot: _TargetSnapshot) -> None:
    """Undo an impossible compiler-side target mutation before failing closed."""

    if not _verify_target_snapshot(snapshot):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Target rollback snapshot is unauthenticated or has mutable-backup drift."
        )
    torch = _torch()
    try:
        parameters = _transaction._parameter_bundle(target_model, snapshot.layer_index)
    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Cache compiler changed target topology and exact rollback is impossible."
        ) from exc
    if any(
        id(parameters[name]) != snapshot.parameter_ids[name]
        or int(parameters[name].data_ptr()) != snapshot.parameter_pointers[name]
        or tuple(parameters[name].shape) != snapshot.parameter_shapes[name]
        or parameters[name].dtype != snapshot.parameter_dtypes[name]
        for name in parameters
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Cache compiler changed Parameter identity/storage and exact rollback is impossible."
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.requires_grad_(snapshot.parameter_requires_grad[name])
            parameter.copy_(
                snapshot.parameter_values[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
            reference = snapshot.gradient_references[name]
            frozen_gradient = snapshot.gradient_values[name]
            if reference is None:
                parameter.grad = None
            else:
                reference.copy_(
                    frozen_gradient.to(device=reference.device, dtype=reference.dtype)
                )
                parameter.grad = reference
    if not _snapshot_matches(target_model, snapshot):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Cache compiler touched the target and exact rollback was not proven."
        )


def _require_snapshot_unchanged(
    target_model: Any,
    snapshot: _TargetSnapshot,
    *,
    phase: str,
) -> None:
    if _snapshot_matches(target_model, snapshot):
        return
    _restore_compile_snapshot(target_model, snapshot)
    raise Lfm2SemanticAttentionGroupQdeimInstallationError(
        f"Target state changed during {phase}; no candidate was installed."
    )


def _candidate_state(
    candidate: Lfm2SemanticAttentionGroupQdeimCandidate,
    *,
    layer_index: int,
    source_inventory_sha256: str,
    source_parent_identity_sha256: str,
    source_state_sha256: str,
    target_template_sha256: str,
    target_templates: Mapping[str, Any],
    normalized_input_decoder_sha256: str,
    residual_output_chart_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    torch = _torch()
    if (
        not isinstance(candidate, Lfm2SemanticAttentionGroupQdeimCandidate)
        or candidate.layer_index != layer_index
        or not isinstance(candidate.compiled, Lfm2CompiledAttentionGroupQdeim)
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Cache compiler returned no typed intact-group attention candidate."
        )
    adapter_report = _self_sealed_report(
        candidate.report,
        seal_name="result_sha256",
        label="semantic attention group-QDEIM adapter report",
    )
    capability_value = adapter_report.get("cache_capability_receipt")
    context_bank_value = adapter_report.get("context_bank_receipt")
    if not isinstance(capability_value, Mapping) or not isinstance(
        context_bank_value,
        Mapping,
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Cache adapter omitted its capability or true-context receipt."
        )
    capability_report = _self_sealed_report(
        capability_value,
        seal_name="result_sha256",
        label="semantic attention group-QDEIM cache capability",
    )
    context_bank_report = _self_sealed_report(
        context_bank_value,
        seal_name="result_sha256",
        label="semantic attention group-QDEIM true-context bank",
    )
    tangent_bank_value = adapter_report.get("attention_tangent_bank")
    tangent_loader_value = adapter_report.get("attention_tangent_fit_loader_receipt")
    tangent_bank_report = None
    tangent_loader_report = None
    if tangent_bank_value is not None or tangent_loader_value is not None:
        if not isinstance(tangent_bank_value, Mapping) or not isinstance(
            tangent_loader_value,
            Mapping,
        ):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Cache adapter emitted an incomplete attention tangent lineage."
            )
        tangent_bank_report = _self_sealed_report(
            tangent_bank_value,
            seal_name="bank_sha256",
            label="semantic attention causal tangent bank",
        )
        tangent_loader_report = _self_sealed_report(
            tangent_loader_value,
            seal_name="result_sha256",
            label="semantic attention tangent FIT loader",
        )
        for name in (
            "query_local_reduction_receipt",
            "sampled_suffix_history_key_value_reduction_receipt",
            "joint_sampled_token_query_key_value_reduction_receipt",
            "older_prefix_key_value_reduction_receipt",
        ):
            reduction = tangent_bank_report.get(name)
            if not isinstance(reduction, Mapping):
                raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                    "Attention tangent bank omitted a covariance reduction receipt."
                )
            _self_sealed_report(
                reduction,
                seal_name="result_sha256",
                label=f"semantic attention tangent {name}",
            )
    compiled_report = _self_sealed_report(
        candidate.compiled.report,
        seal_name="result_sha256",
        label="standalone attention group-QDEIM compiler report",
    )
    requested_fold_compute_device = adapter_report.get("fold_compute_device")
    adapter_compute_execution = _authenticated_qdeim_compute_execution(
        adapter_report.get("qdeim_compute_execution"),
        expected_requested_device=requested_fold_compute_device,
        label="semantic attention group-QDEIM adapter",
    )
    compiler_compute_execution = _authenticated_qdeim_compute_execution(
        compiled_report.get("compute_execution"),
        expected_requested_device=requested_fold_compute_device,
        label="standalone attention group-QDEIM compiler",
    )
    if dict(adapter_compute_execution) != dict(compiler_compute_execution):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Adapter and compiler Q-DEIM compute execution receipts differ."
        )
    if (
        lfm2_attention_group_qdeim_target_template_sha256(
            target_templates["q_proj.weight"],
            target_templates["k_proj.weight"],
            target_templates["v_proj.weight"],
            target_templates["out_proj.weight"],
        )
        != target_template_sha256
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Prepared target templates changed after their blank-target seal."
        )
    state = candidate.state_dict()
    if set(state) != set(_QKVO_ROLES):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Intact-group candidate must contain exactly Q/K/V/O."
        )
    frozen: dict[str, Any] = {}
    for role in _QKVO_ROLES:
        tensor = _detached_tensor(
            state[role],
            label=f"candidate {role}",
            ndim=2,
            dtype=torch.bfloat16,
        )
        if (
            tuple(tensor.shape) != tuple(target_templates[role].shape)
            or int(torch.count_nonzero(tensor).item()) == 0
        ):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                f"Candidate {role} does not densely plant the blank target topology."
            )
        frozen[role] = tensor
    tensor_hashes = {
        role: _tensor_sha256(frozen[role], label=f"candidate {role}")
        for role in _QKVO_ROLES
    }
    candidate_state_sha = _candidate_state_sha256(frozen)
    decoder = _detached_tensor(
        candidate.compiled.normalized_input_decoder,
        label="compiled normalized input decoder",
        ndim=2,
    )
    chart = _detached_tensor(
        candidate.compiled.residual_output_chart,
        label="compiled residual output chart",
        ndim=2,
    )
    basis = _detached_tensor(
        candidate.compiled.group_pod_basis,
        label="compiled GQA-group POD basis",
        ndim=2,
    )
    group_interpolation = _detached_tensor(
        candidate.compiled.group_interpolation,
        label="compiled GQA-group interpolation",
        ndim=2,
    )
    context_reconstruction = _detached_tensor(
        candidate.compiled.context_reconstruction,
        label="compiled context reconstruction",
        ndim=2,
    )
    adapter_digest_fields = (
        "cache_manifest_sha256",
        "cache_selective_session_sha256",
        "cache_capability_sha256",
        "residual_chart_receipt_sha256",
        "normalized_input_decoder_receipt_sha256",
        "context_bank_receipt_sha256",
        "standalone_context_evidence_sha256",
        "standalone_compiler_result_sha256",
    )
    for name in adapter_digest_fields:
        _digest(adapter_report.get(name), label=f"adapter {name}")
    materialization = compiled_report.get("materialization")
    semantic_policy = compiled_report.get("semantic_policy")
    compiler_authentication = compiled_report.get("authentication")
    compiler_topology = compiled_report.get("topology")
    compiler_evidence = compiled_report.get("evidence")
    compiler_selection = compiled_report.get("selection")
    compiler_interpolation = compiled_report.get("interpolation")
    prohibited = compiled_report.get("prohibited_operations")
    if isinstance(compiler_authentication, Mapping):
        _digest(
            compiler_authentication.get("functional_snapshot_evidence_sha256"),
            label="compiler functional snapshot evidence SHA-256",
        )
    compiler_snapshot_sha256 = (
        None
        if not isinstance(compiler_authentication, Mapping)
        else compiler_authentication.get("functional_snapshot_evidence_sha256")
    )
    source_contract = adapter_report.get("source_attention_contract")
    if not isinstance(source_contract, Mapping):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Adapter report has no authenticated source attention contract."
        )
    head_dim = source_contract.get("head_dim")
    source_q_heads = source_contract.get("query_heads")
    source_kv_heads = source_contract.get("key_value_heads")
    if (
        isinstance(head_dim, bool)
        or not isinstance(head_dim, int)
        or head_dim <= 0
        or isinstance(source_q_heads, bool)
        or not isinstance(source_q_heads, int)
        or source_q_heads <= 0
        or isinstance(source_kv_heads, bool)
        or not isinstance(source_kv_heads, int)
        or source_kv_heads <= 0
        or source_q_heads % source_kv_heads
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Adapter source attention contract has invalid GQA topology."
        )
    target_q_rows = int(target_templates["q_proj.weight"].shape[0])
    target_k_rows = int(target_templates["k_proj.weight"].shape[0])
    target_v_rows = int(target_templates["v_proj.weight"].shape[0])
    if (
        target_q_rows % head_dim
        or target_k_rows % head_dim
        or target_v_rows != target_k_rows
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Target templates do not preserve the authenticated GQA head axis."
        )
    target_q_heads = target_q_rows // head_dim
    target_kv_heads = target_k_rows // head_dim
    queries_per_key_value = source_q_heads // source_kv_heads
    group_block_width = queries_per_key_value * head_dim
    selected_groups = tuple(candidate.compiled.selected_source_key_value_groups)
    selected_query_heads = tuple(candidate.compiled.selected_source_query_heads)
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in (*selected_groups, *selected_query_heads)
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Compiled whole-group selection contains a non-integer index."
        )
    expected_selected_query_heads = tuple(
        group * queries_per_key_value + offset
        for group in selected_groups
        for offset in range(queries_per_key_value)
    )
    selected_payloads = context_bank_report.get("selected_payloads")
    tangent_contract_declared = "attention_tangent_bank_materialized" in adapter_report
    if (
        adapter_report.get("schema")
        != LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_ADAPTER_SCHEMA
        or adapter_report.get("layer_index") != layer_index
        or adapter_report.get("source_inventory_sha256") != source_inventory_sha256
        or capability_report.get("schema")
        != "neural-forge-lfm2-semantic-attention-group-qdeim-cache-capability-v1"
        or capability_report.get("result_sha256")
        != adapter_report.get("cache_capability_sha256")
        or capability_report.get("cache_key") != adapter_report.get("cache_key")
        or capability_report.get("cache_manifest_sha256")
        != adapter_report.get("cache_manifest_sha256")
        or capability_report.get("cache_selective_session_sha256")
        != adapter_report.get("cache_selective_session_sha256")
        or capability_report.get("layer_index") != layer_index
        or capability_report.get("fit_batch_indices")
        != adapter_report.get("fit_batch_indices")
        or capability_report.get("true_post_softmax_context_complete") is not True
        or capability_report.get("tensor_payloads_loaded") != 0
        or capability_report.get("parent_forward_calls") != 0
        or capability_report.get("trace_recaptures") != 0
        or capability_report.get("training_steps") != 0
        or (
            tangent_contract_declared
            and (
                capability_report.get(
                    "analytic_attention_tangent_phase_bundle_complete"
                )
                is not adapter_report.get("attention_tangent_bank_materialized")
                or adapter_report.get("functional_snapshot_block_count")
                != (2 if tangent_bank_report is not None else 0)
                or adapter_report.get("attention_tangent_bank_sha256")
                != (
                    None
                    if tangent_bank_report is None
                    else tangent_bank_report.get("bank_sha256")
                )
                or adapter_report.get("attention_tangent_fit_loader_receipt_sha256")
                != (
                    None
                    if tangent_loader_report is None
                    else tangent_loader_report.get("result_sha256")
                )
                or adapter_report.get("functional_snapshot_evidence_sha256")
                != compiler_snapshot_sha256
                or adapter_report.get("cross_phase_older_key_anchor_stitch_required")
                != capability_report.get("cross_phase_older_key_anchor_stitch_required")
                or adapter_report.get(
                    "cross_phase_older_key_anchor_stitch_feasible_without_recapture"
                )
                is not True
                or adapter_report.get("cross_phase_older_key_anchor_stitch_implemented")
                != capability_report.get(
                    "cross_phase_older_key_anchor_stitch_implemented"
                )
                or (
                    tangent_bank_report is not None
                    and (
                        tangent_bank_report.get("fit_case_evidence_sha256")
                        != tangent_loader_report.get("fit_case_evidence_sha256")
                        or tangent_bank_report.get(
                            "query_local_and_history_key_value_evidence_"
                            "separately_labelled"
                        )
                        is not True
                        or tangent_bank_report.get(
                            "cross_phase_older_key_anchor_stitch_implemented"
                        )
                        is not True
                        or tangent_bank_report.get(
                            "partial_diagnostic_family_scope_is_explicit_and_not_"
                            "promoted_to_functional_complete"
                        )
                        is not True
                        or tangent_bank_report.get(
                            "functional_pod_uses_joint_sampled_token_qkv_cross_terms"
                        )
                        is not True
                        or tangent_bank_report.get(
                            "functional_pod_double_counts_partial_q_or_kv_covariance"
                        )
                        is not False
                        or tangent_bank_report.get(
                            "complete_value_history_tangent_coverage"
                        )
                        is not True
                        or tangent_bank_report.get(
                            "source_o_and_residual_chart_applied_before_group_"
                            "covariance"
                        )
                        is not True
                        or tangent_bank_report.get(
                            "functional_group_pod_is_residual_output_attention_aware"
                        )
                        is not True
                        or adapter_report.get(
                            "analytic_attention_tangent_evidence_complete"
                        )
                        is not tangent_bank_report.get(
                            "complete_key_value_history_tangent_coverage"
                        )
                        or tangent_bank_report.get("heldout_tensor_payloads_used") != 0
                    )
                )
            )
        )
        or context_bank_report.get("schema")
        != "neural-forge-lfm2-semantic-attention-qdeim-context-bank-v1"
        or context_bank_report.get("result_sha256")
        != adapter_report.get("context_bank_receipt_sha256")
        or context_bank_report.get("route") != "captured_true"
        or context_bank_report.get("fit_batch_indices")
        != adapter_report.get("fit_batch_indices")
        or not isinstance(selected_payloads, list)
        or not selected_payloads
        or any(
            not isinstance(payload, Mapping)
            or payload.get("session_sha256")
            != adapter_report.get("cache_selective_session_sha256")
            for payload in selected_payloads
        )
        or context_bank_report.get("heldout_tensor_payload_loads") != 0
        or context_bank_report.get("parent_forward_calls") != 0
        or context_bank_report.get("trace_recaptures") != 0
        or context_bank_report.get("training_steps") != 0
        or adapter_report.get("source_state_sha256") != source_state_sha256
        or adapter_report.get("cache_parent_fingerprint")
        != source_parent_identity_sha256
        or adapter_report.get("target_template_sha256") != target_template_sha256
        or adapter_report.get("normalized_input_decoder_sha256")
        != normalized_input_decoder_sha256
        or adapter_report.get("residual_chart_sha256") != residual_output_chart_sha256
        or adapter_report.get("standalone_compiler_result_sha256")
        != compiled_report["result_sha256"]
        or adapter_report.get("candidate_tensor_sha256") != tensor_hashes
        or not isinstance(materialization, Mapping)
        or adapter_report.get("candidate_state_sha256") != candidate_state_sha
        or materialization.get("candidate_state_sha256") != candidate_state_sha
        or adapter_report.get("candidate_dense_bfloat16_standalone") is not True
        or adapter_report.get("qkv_producer_averaging_performed") is not False
        or adapter_report.get("dense_all_parent_output_columns_used") is not True
        or adapter_report.get("captured_true_post_softmax_empirical_context_route")
        is not True
        or adapter_report.get("heldout_tensor_payload_loads") != 0
        or adapter_report.get("selected_source_key_value_groups")
        != list(selected_groups)
        or adapter_report.get("selected_source_query_heads")
        != list(selected_query_heads)
        or adapter_report.get("automatic_semantic_acceptance_or_rejection_permitted")
        is not False
        or adapter_report.get("numeric_metrics_have_semantic_decision_authority")
        is not False
        or adapter_report.get("binary_manual_threshold_has_iteration_veto_authority")
        is not False
        or adapter_report.get("whole_decoded_behavior_is_primary_intelligence_evidence")
        is not True
        or compiled_report.get("schema") != LFM2_ATTENTION_GROUP_QDEIM_SCHEMA
        or compiled_report.get("status")
        != "materialized-for-holistic-whole-response-intelligence-evaluation"
        or compiled_report.get("method")
        != "deterministic-block-constrained-qdeim-intact-gqa-producers"
        or not isinstance(compiler_authentication, Mapping)
        or compiler_authentication.get("source_state_sha256_before")
        != source_state_sha256
        or compiler_authentication.get("source_state_sha256_after")
        != source_state_sha256
        or compiler_authentication.get("source_state_immutable") is not True
        or compiler_authentication.get("target_template_sha256")
        != target_template_sha256
        or compiler_authentication.get("normalized_input_decoder_sha256")
        != normalized_input_decoder_sha256
        or compiler_authentication.get("residual_output_chart_sha256")
        != residual_output_chart_sha256
        or compiler_authentication.get("fit_context_evidence_sha256")
        != adapter_report.get("standalone_context_evidence_sha256")
        or not isinstance(
            compiler_authentication.get("functional_snapshot_evidence_sha256"),
            str,
        )
        or not isinstance(compiler_topology, Mapping)
        or compiler_topology.get("source_query_head_count") != source_q_heads
        or compiler_topology.get("source_key_value_head_count") != source_kv_heads
        or compiler_topology.get("target_query_head_count") != target_q_heads
        or compiler_topology.get("target_key_value_head_count") != target_kv_heads
        or compiler_topology.get("queries_per_key_value") != queries_per_key_value
        or compiler_topology.get("head_dim") != head_dim
        or compiler_topology.get("source_group_block_width") != group_block_width
        or not isinstance(compiler_evidence, Mapping)
        or compiler_evidence.get("fit_partition_only") is not True
        or compiler_evidence.get("heldout_rows_used") is not False
        or not isinstance(compiler_selection, Mapping)
        or compiler_selection.get("selected_source_key_value_groups")
        != list(selected_groups)
        or compiler_selection.get("selected_source_query_heads")
        != list(selected_query_heads)
        or compiler_selection.get("whole_gqa_groups_only") is not True
        or compiler_selection.get("qkv_producer_averaging_performed") is not False
        or compiler_selection.get(
            "qkv_producer_rows_are_single_source_copies_through_decoder"
        )
        is not True
        or selected_query_heads != expected_selected_query_heads
        or len(selected_groups) != target_kv_heads
        or len(set(selected_groups)) != target_kv_heads
        or any(group < 0 or group >= source_kv_heads for group in selected_groups)
        or len(selected_query_heads) != target_q_heads
        or tuple(basis.shape) != (source_kv_heads, target_kv_heads)
        or tuple(group_interpolation.shape) != (target_kv_heads, source_kv_heads)
        or tuple(context_reconstruction.shape)
        != (
            target_kv_heads * group_block_width,
            source_kv_heads * group_block_width,
        )
        or not isinstance(compiler_interpolation, Mapping)
        or compiler_interpolation.get("group_pod_basis_sha256")
        != _tensor_sha256(basis, label="compiled GQA-group POD basis")
        or compiler_interpolation.get("group_interpolation_sha256")
        != _tensor_sha256(
            group_interpolation,
            label="compiled GQA-group interpolation",
        )
        or compiler_interpolation.get("context_reconstruction_sha256")
        != _tensor_sha256(
            context_reconstruction,
            label="compiled context reconstruction",
        )
        or compiler_interpolation.get("group_interpolation_shape")
        != list(group_interpolation.shape)
        or compiler_interpolation.get("context_reconstruction_shape")
        != list(context_reconstruction.shape)
        or compiler_interpolation.get(
            "numeric_diagnostics_have_semantic_decision_authority"
        )
        is not False
        or materialization.get("dense_all_parent_output_columns_used") is not True
        or materialization.get("pre_softmax_qkv_averaging") is not False
        or materialization.get("candidate_dtype") != "bfloat16"
        or materialization.get("standalone_dense_tensors") is not True
        or materialization.get("decoder_shape") != list(decoder.shape)
        or materialization.get("residual_chart_shape") != list(chart.shape)
        or _tensor_sha256(decoder, label="compiled normalized input decoder")
        != normalized_input_decoder_sha256
        or _tensor_sha256(chart, label="compiled residual output chart")
        != residual_output_chart_sha256
        or not isinstance(semantic_policy, Mapping)
        or semantic_policy.get("numeric_threshold_auto_accept_permitted") is not False
        or semantic_policy.get("numeric_threshold_auto_veto_permitted") is not False
        or semantic_policy.get("automatic_semantic_publication_permitted") is not False
        or semantic_policy.get("whole_response_intelligence_evaluation_is_primary")
        is not True
        or semantic_policy.get("binary_manual_acceptance_gate_required") is not False
        or semantic_policy.get("retention_targets_are_objectives_not_cutoffs")
        is not True
        or not isinstance(prohibited, Mapping)
        or prohibited.get("model_forward_calls") != 0
        or prohibited.get("model_generate_calls") != 0
        or prohibited.get("training_steps") != 0
        or prohibited.get("backward_calls") != 0
        or prohibited.get("optimizer_steps") != 0
        or prohibited.get("pruning_used") is not False
        or prohibited.get("quantization_used") is not False
        or prohibited.get("sparsity_used") is not False
        or prohibited.get("parameter_sharing_used") is not False
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Intact-group candidate lost cache, source, target, or semantic-policy authentication."
        )
    return (
        MappingProxyType(frozen),
        adapter_report,
        compiled_report,
    )


def _verified_norms(
    *,
    layer_index: int,
    source_inventory: Lfm2DenseQkvoSourceInventory,
    source_query_norm_weight: Any,
    source_key_norm_weight: Any,
    source_norm_inventory: Lfm2SemanticAttentionGroupQdeimNormInventory,
    target_parameters: Mapping[str, Any],
) -> tuple[Any, Any]:
    torch = _torch()
    if (
        not isinstance(source_inventory, Lfm2DenseQkvoSourceInventory)
        or source_inventory.verify() != source_inventory.inventory_sha256
        or not isinstance(
            source_norm_inventory,
            Lfm2SemanticAttentionGroupQdeimNormInventory,
        )
        or source_norm_inventory.verify() != source_norm_inventory.inventory_sha256
        or source_norm_inventory.layer_index != layer_index
        or source_norm_inventory.source_qkvo_inventory_sha256
        != source_inventory.inventory_sha256
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Direct-parent norm lineage is not bound to the source Q/K/V/O inventory."
        )
    query = _detached_tensor(
        source_query_norm_weight,
        label="source_query_norm_weight",
        ndim=1,
        dtype=torch.bfloat16,
    )
    key = _detached_tensor(
        source_key_norm_weight,
        label="source_key_norm_weight",
        ndim=1,
        dtype=torch.bfloat16,
    )
    prefix = f"model.layers.{layer_index}.self_attn."
    target_query = target_parameters[prefix + "q_layernorm.weight"]
    target_key = target_parameters[prefix + "k_layernorm.weight"]
    if (
        tuple(query.shape) != tuple(target_query.shape)
        or tuple(key.shape) != tuple(target_key.shape)
        or _tensor_sha256(query, label="source query norm")
        != source_norm_inventory.query_norm_receipt["tensor_sha256"]
        or _tensor_sha256(key, label="source key norm")
        != source_norm_inventory.key_norm_receipt["tensor_sha256"]
        or int(torch.count_nonzero(query).item()) == 0
        or int(torch.count_nonzero(key).item()) == 0
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Direct-parent norm tensors are stale, all-zero, or incompatible with "
            "the target."
        )
    return query, key


@dataclass(frozen=True, slots=True)
class Lfm2PreparedSemanticAttentionGroupQdeimInstallation:
    """Authenticated coupled Q/K/V/O/norm state awaiting one transaction."""

    layer_index: int
    candidate: Lfm2SemanticAttentionGroupQdeimCandidate = field(
        repr=False,
        compare=False,
    )
    query_norm_weight: Any = field(repr=False, compare=False)
    key_norm_weight: Any = field(repr=False, compare=False)
    source_norm_inventory: Lfm2SemanticAttentionGroupQdeimNormInventory
    target_snapshot: _TargetSnapshot = field(repr=False, compare=False)
    source_state_sha256: str
    target_template_sha256: str
    normalized_input_decoder_sha256: str
    residual_output_chart_sha256: str
    cache_manifest_sha256: str
    cache_selective_session_sha256: str
    cache_capability_sha256: str
    residual_chart_receipt_sha256: str
    normalized_input_decoder_receipt_sha256: str
    context_bank_receipt_sha256: str
    candidate_adapter_result_sha256: str
    candidate_compiler_result_sha256: str
    candidate_state_sha256: str
    fold_compute_device: str
    qdeim_compute_execution_sha256: str
    preparation_sha256: str = ""

    def __post_init__(self) -> None:
        if not _verify_target_snapshot(self.target_snapshot):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Prepared target snapshot is unauthenticated or mutable."
            )
        for label, value in (
            ("source state SHA-256", self.source_state_sha256),
            ("target template SHA-256", self.target_template_sha256),
            (
                "normalized input decoder SHA-256",
                self.normalized_input_decoder_sha256,
            ),
            ("residual output chart SHA-256", self.residual_output_chart_sha256),
            ("cache manifest SHA-256", self.cache_manifest_sha256),
            ("cache session SHA-256", self.cache_selective_session_sha256),
            ("cache capability SHA-256", self.cache_capability_sha256),
            ("residual chart receipt SHA-256", self.residual_chart_receipt_sha256),
            (
                "normalized decoder receipt SHA-256",
                self.normalized_input_decoder_receipt_sha256,
            ),
            ("context bank receipt SHA-256", self.context_bank_receipt_sha256),
            ("adapter result SHA-256", self.candidate_adapter_result_sha256),
            ("compiler result SHA-256", self.candidate_compiler_result_sha256),
            ("candidate state SHA-256", self.candidate_state_sha256),
            (
                "Q-DEIM compute execution SHA-256",
                self.qdeim_compute_execution_sha256,
            ),
        ):
            _digest(value, label=label)
        query = _detached_tensor(
            self.query_norm_weight,
            label="prepared query norm",
            ndim=1,
            dtype=_torch().bfloat16,
        )
        key = _detached_tensor(
            self.key_norm_weight,
            label="prepared key norm",
            ndim=1,
            dtype=_torch().bfloat16,
        )
        object.__setattr__(self, "query_norm_weight", query)
        object.__setattr__(self, "key_norm_weight", key)
        if not self._prepared_norms_match_source_inventory():
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Prepared Q/K norm tensors are not the exact sealed direct-parent "
                "norm inventory for this layer."
            )
        computed = canonical_sha256(self.authentication_body())
        if self.preparation_sha256 and self.preparation_sha256 != computed:
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Prepared attention group-QDEIM state does not match its seal."
            )
        object.__setattr__(self, "preparation_sha256", computed)

    def _prepared_norms_match_source_inventory(self) -> bool:
        try:
            inventory = self.source_norm_inventory
            return bool(
                isinstance(
                    inventory,
                    Lfm2SemanticAttentionGroupQdeimNormInventory,
                )
                and inventory.verify() == inventory.inventory_sha256
                and inventory.layer_index == self.layer_index
                and _tensor_sha256(
                    self.query_norm_weight,
                    label="prepared query norm",
                )
                == inventory.query_norm_receipt["tensor_sha256"]
                and _tensor_sha256(
                    self.key_norm_weight,
                    label="prepared key norm",
                )
                == inventory.key_norm_receipt["tensor_sha256"]
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            return False

    def authentication_body(self) -> dict[str, Any]:
        target_seed_authentication = _target_seed_authentication_body(
            self.target_snapshot
        )
        torch = _torch()
        prefix = f"model.layers.{self.layer_index}.self_attn."
        transaction_covered_state_keys = sorted(prefix + role for role in _ALL_ROLES)
        expected_changed_state_keys = sorted(
            [
                *(prefix + role for role in _QKVO_ROLES),
                *(
                    prefix + role
                    for role, source_norm in (
                        ("q_layernorm.weight", self.query_norm_weight),
                        ("k_layernorm.weight", self.key_norm_weight),
                    )
                    if not torch.equal(
                        self.target_snapshot.parameter_values[prefix + role],
                        source_norm,
                    )
                ),
            ]
        )
        expected_unchanged_state_keys = sorted(
            set(transaction_covered_state_keys) - set(expected_changed_state_keys)
        )
        return {
            "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_PREPARATION_SCHEMA,
            "layer_index": self.layer_index,
            "candidate_adapter_result_sha256": self.candidate_adapter_result_sha256,
            "candidate_compiler_result_sha256": self.candidate_compiler_result_sha256,
            "candidate_state_sha256": self.candidate_state_sha256,
            "fold_compute_device": self.fold_compute_device,
            "qdeim_compute_execution_sha256": (self.qdeim_compute_execution_sha256),
            "source_state_sha256": self.source_state_sha256,
            "query_norm_weight_sha256": _tensor_sha256(
                self.query_norm_weight,
                label="prepared query norm",
            ),
            "key_norm_weight_sha256": _tensor_sha256(
                self.key_norm_weight,
                label="prepared key norm",
            ),
            "source_norm_inventory_sha256": self.source_norm_inventory.inventory_sha256,
            "source_parent_identity_sha256": (
                self.source_norm_inventory.source_parent_identity_sha256
            ),
            "source_qkvo_inventory_sha256": (
                self.source_norm_inventory.source_qkvo_inventory_sha256
            ),
            "prepared_norm_tensors_match_source_inventory": True,
            "target_template_sha256": self.target_template_sha256,
            "normalized_input_decoder_sha256": self.normalized_input_decoder_sha256,
            "residual_output_chart_sha256": self.residual_output_chart_sha256,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cache_selective_session_sha256": self.cache_selective_session_sha256,
            "cache_capability_sha256": self.cache_capability_sha256,
            "residual_chart_receipt_sha256": self.residual_chart_receipt_sha256,
            "normalized_input_decoder_receipt_sha256": (
                self.normalized_input_decoder_receipt_sha256
            ),
            "context_bank_receipt_sha256": self.context_bank_receipt_sha256,
            "captured_blank_target_bundle_sha256": self.target_snapshot.bundle_sha256,
            "captured_child_model_state_sha256": self.target_snapshot.model_state_sha256,
            "captured_target_topology_sha256": self.target_snapshot.topology_sha256,
            "captured_target_parameter_count": self.target_snapshot.parameter_count,
            "captured_parameter_ids": dict(self.target_snapshot.parameter_ids),
            "captured_parameter_pointers": dict(
                self.target_snapshot.parameter_pointers
            ),
            "captured_gradient_ids": dict(self.target_snapshot.gradient_ids),
            "captured_gradient_sha256": dict(self.target_snapshot.gradient_sha256),
            "target_seed_authentication": target_seed_authentication,
            "target_seed_authentication_sha256": canonical_sha256(
                target_seed_authentication
            ),
            "exact_zero_qkvo_blank_target_required_and_proven": True,
            "qk_norm_seed_policy": _NORM_SEED_POLICY,
            "qk_norm_seed_policy_authenticated": True,
            "six_role_atomic_transaction_coverage_required": True,
            "transaction_covered_parameter_roles": list(_ALL_ROLES),
            "transaction_covered_state_keys": transaction_covered_state_keys,
            "transaction_covered_parameter_role_count": len(_ALL_ROLES),
            "expected_changed_state_keys_after_installation": (
                expected_changed_state_keys
            ),
            "expected_changed_state_key_count_after_installation": len(
                expected_changed_state_keys
            ),
            "expected_unchanged_seed_equal_candidate_state_keys_after_installation": (
                expected_unchanged_state_keys
            ),
            "actual_changed_bytes_may_cover_four_to_six_roles": True,
            "candidate_frozen_before_installation": True,
            "numeric_metrics_have_semantic_decision_authority": False,
            "binary_manual_threshold_has_iteration_veto_authority": False,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }

    def verify(self) -> bool:
        try:
            return bool(
                _verify_target_snapshot(self.target_snapshot)
                and self._prepared_norms_match_source_inventory()
                and self.source_norm_inventory.verify()
                == self.source_norm_inventory.inventory_sha256
                and self.preparation_sha256
                == canonical_sha256(self.authentication_body())
            )
        except (TypeError, ValueError, RuntimeError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionGroupQdeimInstallationResult:
    """One sealed six-Parameter installation receipt and its prepared source."""

    receipt: Mapping[str, Any]
    prepared: Lfm2PreparedSemanticAttentionGroupQdeimInstallation = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.prepared.verify():
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Installed attention candidate lost its preparation authentication."
            )
        receipt = _self_sealed_report(
            self.receipt,
            seal_name="result_sha256",
            label="attention group-QDEIM installation receipt",
        )
        if (
            receipt.get("schema")
            != LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_INSTALLATION_SCHEMA
            or receipt.get("preparation_sha256") != self.prepared.preparation_sha256
        ):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Installation receipt is not bound to its prepared candidate."
            )
        object.__setattr__(self, "receipt", receipt)

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.receipt)


def prepare_lfm2_semantic_attention_group_qdeim_installation(
    *,
    target_model: Any,
    session: Any,
    layer_index: int,
    fit_batch_indices: Sequence[int],
    fit_batch_sample_weights: Mapping[int, float],
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_out_weight: Any,
    source_inventory: Lfm2DenseQkvoSourceInventory,
    source_query_norm_weight: Any,
    source_key_norm_weight: Any,
    source_norm_inventory: Lfm2SemanticAttentionGroupQdeimNormInventory,
    normalized_input_decoder: Any,
    normalized_input_decoder_receipt: Mapping[str, Any],
    residual_output_chart: Any,
    residual_chart_receipt: Mapping[str, Any],
    pod_rank_relative_tolerance: float | None = None,
    fold_compute_device: str = "cpu",
) -> (
    Lfm2PreparedSemanticAttentionGroupQdeimInstallation
    | Lfm2SemanticAttentionGroupQdeimEvidenceDeficit
):
    """Compile and freeze a coupled candidate without mutating the blank target."""

    snapshot = _target_snapshot(target_model, layer_index=layer_index)
    _target_seed_authentication(snapshot)
    target_parameters = _transaction._parameter_bundle(target_model, layer_index)
    query_norm, key_norm = _verified_norms(
        layer_index=layer_index,
        source_inventory=source_inventory,
        source_query_norm_weight=source_query_norm_weight,
        source_key_norm_weight=source_key_norm_weight,
        source_norm_inventory=source_norm_inventory,
        target_parameters=target_parameters,
    )
    prefix = f"model.layers.{layer_index}.self_attn."
    templates = {
        role: snapshot.parameter_values[prefix + role].clone() for role in _QKVO_ROLES
    }
    template_sha = lfm2_attention_group_qdeim_target_template_sha256(
        templates["q_proj.weight"],
        templates["k_proj.weight"],
        templates["v_proj.weight"],
        templates["out_proj.weight"],
    )
    source_state_sha = lfm2_attention_group_qdeim_source_state_sha256(
        source_query_weight,
        source_key_weight,
        source_value_weight,
        source_out_weight,
    )
    decoder_sha = _tensor_sha256(
        normalized_input_decoder,
        label="normalized input decoder",
    )
    chart_sha = _tensor_sha256(
        residual_output_chart,
        label="residual output chart",
    )
    try:
        compiled = compile_lfm2_semantic_attention_group_qdeim_from_cache(
            session=session,
            layer_index=layer_index,
            fit_batch_indices=fit_batch_indices,
            fit_batch_sample_weights=fit_batch_sample_weights,
            source_query_weight=source_query_weight,
            source_key_weight=source_key_weight,
            source_value_weight=source_value_weight,
            source_out_weight=source_out_weight,
            source_inventory=source_inventory,
            source_query_norm_weight=source_query_norm_weight,
            source_key_norm_weight=source_key_norm_weight,
            expected_source_query_norm_weight_sha256=(
                source_norm_inventory.query_norm_receipt["tensor_sha256"]
            ),
            expected_source_key_norm_weight_sha256=(
                source_norm_inventory.key_norm_receipt["tensor_sha256"]
            ),
            target_query_weight_template=templates["q_proj.weight"],
            target_key_weight_template=templates["k_proj.weight"],
            target_value_weight_template=templates["v_proj.weight"],
            target_out_weight_template=templates["out_proj.weight"],
            normalized_input_decoder=normalized_input_decoder,
            normalized_input_decoder_receipt=normalized_input_decoder_receipt,
            residual_output_chart=residual_output_chart,
            residual_chart_receipt=residual_chart_receipt,
            pod_rank_relative_tolerance=pod_rank_relative_tolerance,
            fold_compute_device=fold_compute_device,
        )
    except BaseException:
        _require_snapshot_unchanged(
            target_model,
            snapshot,
            phase="cache candidate compilation",
        )
        raise
    _require_snapshot_unchanged(
        target_model,
        snapshot,
        phase="cache candidate compilation",
    )
    if isinstance(compiled, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit):
        if compiled.candidate_materialized:
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Typed attention evidence deficit falsely claims a candidate."
            )
        return compiled
    state, adapter_report, compiler_report = _candidate_state(
        compiled,
        layer_index=layer_index,
        source_inventory_sha256=source_inventory.inventory_sha256,
        source_parent_identity_sha256=(
            source_norm_inventory.source_parent_identity_sha256
        ),
        source_state_sha256=source_state_sha,
        target_template_sha256=template_sha,
        target_templates=templates,
        normalized_input_decoder_sha256=decoder_sha,
        residual_output_chart_sha256=chart_sha,
    )
    source_contract = adapter_report.get("source_attention_contract", {})
    if (
        not isinstance(source_contract, Mapping)
        or source_contract.get("head_dim") != source_norm_inventory.head_dim
        or adapter_report.get("fold_compute_device") != fold_compute_device
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Direct-parent norm head dimension or fold compute device differs "
            "from the cache attention contract."
        )
    qdeim_compute_execution = adapter_report["qdeim_compute_execution"]
    del state
    return Lfm2PreparedSemanticAttentionGroupQdeimInstallation(
        layer_index=layer_index,
        candidate=compiled,
        query_norm_weight=query_norm,
        key_norm_weight=key_norm,
        source_norm_inventory=source_norm_inventory,
        target_snapshot=snapshot,
        source_state_sha256=source_state_sha,
        target_template_sha256=template_sha,
        normalized_input_decoder_sha256=decoder_sha,
        residual_output_chart_sha256=chart_sha,
        cache_manifest_sha256=adapter_report["cache_manifest_sha256"],
        cache_selective_session_sha256=adapter_report["cache_selective_session_sha256"],
        cache_capability_sha256=adapter_report["cache_capability_sha256"],
        residual_chart_receipt_sha256=adapter_report["residual_chart_receipt_sha256"],
        normalized_input_decoder_receipt_sha256=adapter_report[
            "normalized_input_decoder_receipt_sha256"
        ],
        context_bank_receipt_sha256=adapter_report["context_bank_receipt_sha256"],
        candidate_adapter_result_sha256=adapter_report["result_sha256"],
        candidate_compiler_result_sha256=compiler_report["result_sha256"],
        candidate_state_sha256=adapter_report["candidate_state_sha256"],
        fold_compute_device=fold_compute_device,
        qdeim_compute_execution_sha256=qdeim_compute_execution["receipt_sha256"],
    )


def install_lfm2_prepared_semantic_attention_group_qdeim(
    *,
    target_model: Any,
    prepared: Lfm2PreparedSemanticAttentionGroupQdeimInstallation,
) -> Lfm2SemanticAttentionGroupQdeimInstallationResult:
    """Atomically install Q/K/V/O plus direct-parent Q/K norms."""

    if (
        not isinstance(
            prepared,
            Lfm2PreparedSemanticAttentionGroupQdeimInstallation,
        )
        or not prepared.verify()
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Prepared attention group-QDEIM candidate is unauthenticated."
        )
    if not _snapshot_matches(target_model, prepared.target_snapshot):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Live blank target is stale relative to the prepared attention candidate."
        )
    prefix = f"model.layers.{prepared.layer_index}.self_attn."
    templates = {
        role: prepared.target_snapshot.parameter_values[prefix + role]
        for role in _QKVO_ROLES
    }
    state, adapter_report, compiler_report = _candidate_state(
        prepared.candidate,
        layer_index=prepared.layer_index,
        source_inventory_sha256=(
            prepared.source_norm_inventory.source_qkvo_inventory_sha256
        ),
        source_parent_identity_sha256=(
            prepared.source_norm_inventory.source_parent_identity_sha256
        ),
        source_state_sha256=prepared.source_state_sha256,
        target_template_sha256=prepared.target_template_sha256,
        target_templates=templates,
        normalized_input_decoder_sha256=(prepared.normalized_input_decoder_sha256),
        residual_output_chart_sha256=prepared.residual_output_chart_sha256,
    )
    if (
        adapter_report["result_sha256"] != prepared.candidate_adapter_result_sha256
        or compiler_report["result_sha256"] != prepared.candidate_compiler_result_sha256
        or adapter_report["candidate_state_sha256"] != prepared.candidate_state_sha256
        or adapter_report["cache_manifest_sha256"] != prepared.cache_manifest_sha256
        or adapter_report["cache_selective_session_sha256"]
        != prepared.cache_selective_session_sha256
        or adapter_report["cache_capability_sha256"] != prepared.cache_capability_sha256
        or adapter_report["residual_chart_receipt_sha256"]
        != prepared.residual_chart_receipt_sha256
        or adapter_report["normalized_input_decoder_receipt_sha256"]
        != prepared.normalized_input_decoder_receipt_sha256
        or adapter_report["context_bank_receipt_sha256"]
        != prepared.context_bank_receipt_sha256
        or adapter_report.get("fold_compute_device") != prepared.fold_compute_device
        or adapter_report.get("qdeim_compute_execution", {}).get("receipt_sha256")
        != prepared.qdeim_compute_execution_sha256
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Prepared candidate state or compiler lineage is stale."
        )
    install_state = {
        **state,
        "q_layernorm.weight": prepared.query_norm_weight,
        "k_layernorm.weight": prepared.key_norm_weight,
    }
    target_seed_authentication = _target_seed_authentication_body(
        prepared.target_snapshot
    )
    transaction_covered_keys = tuple(sorted(prefix + role for role in _ALL_ROLES))
    expected_changed = tuple(
        name
        for name in transaction_covered_keys
        if not _torch().equal(
            prepared.target_snapshot.parameter_values[name],
            install_state[name.removeprefix(prefix)],
        )
    )
    prepared_authentication = prepared.authentication_body()
    prepared_expected_changed = tuple(
        prepared_authentication["expected_changed_state_keys_after_installation"]
    )
    expected_qkvo_changes = {prefix + role for role in _QKVO_ROLES}
    if (
        prepared_expected_changed != expected_changed
        or not expected_qkvo_changes.issubset(expected_changed)
        or not 4 <= len(expected_changed) <= 6
    ):
        raise Lfm2SemanticAttentionGroupQdeimInstallationError(
            "Authenticated attention candidate must change all four exact-zero "
            "Q/K/V/O matrices and may change zero, one, or both neutral norm seeds."
        )
    try:
        before, after, changed, parameter_count = (
            _transaction._install_qkvo_transaction(
                target_model,
                layer_index=prepared.layer_index,
                state=install_state,
            )
        )
        current_topology = _transaction._topology_sha256(target_model)
        current_count = _transaction._parameter_count(target_model)
        if (
            before != prepared.target_snapshot.model_state_sha256
            or after == before
            or changed != expected_changed
            or parameter_count != prepared.target_snapshot.parameter_count
            or current_count != prepared.target_snapshot.parameter_count
            or current_topology != prepared.target_snapshot.topology_sha256
        ):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Coupled attention transaction did not preserve six-role coverage "
                "and publish its exact four-to-six-role byte-change inventory."
            )
    except BaseException as exc:
        if not _snapshot_matches(target_model, prepared.target_snapshot):
            try:
                _restore_compile_snapshot(target_model, prepared.target_snapshot)
            except BaseException as rollback_exc:
                raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                    "Six-Parameter attention transaction failed and exact rollback was not proven."
                ) from rollback_exc
        if not _snapshot_matches(target_model, prepared.target_snapshot):
            raise Lfm2SemanticAttentionGroupQdeimInstallationError(
                "Six-Parameter attention transaction failed and exact rollback was not proven."
            ) from exc
        raise
    changed_roles = tuple(name.removeprefix(prefix) for name in changed)
    unchanged_roles = tuple(
        sorted(role for role in _ALL_ROLES if prefix + role not in changed)
    )
    source_norm_equal_to_seed_roles = tuple(
        sorted(role for role in _NORM_ROLES if prefix + role not in changed)
    )
    source_norm_changed_from_seed_roles = tuple(
        sorted(role for role in _NORM_ROLES if prefix + role in changed)
    )
    norm_seed_transitions = {
        role: {
            **target_seed_authentication[role],
            "source_norm_tensor_sha256": _tensor_sha256(
                install_state[role],
                label=f"installed {role}",
            ),
            "candidate_equal_to_seed": prefix + role not in changed,
            "bytes_changed": prefix + role in changed,
        }
        for role in _NORM_ROLES
    }
    receipt_body = {
        "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_INSTALLATION_SCHEMA,
        "status": "installed-coupled-intact-group-qdeim-attention",
        "installed": True,
        "install_count": 1,
        "layer_index": prepared.layer_index,
        "preparation_sha256": prepared.preparation_sha256,
        "candidate_adapter_result_sha256": adapter_report["result_sha256"],
        "candidate_compiler_result_sha256": compiler_report["result_sha256"],
        "candidate_state_sha256": prepared.candidate_state_sha256,
        "fold_compute_device": prepared.fold_compute_device,
        "qdeim_compute_execution": dict(adapter_report["qdeim_compute_execution"]),
        "qdeim_compute_execution_sha256": (prepared.qdeim_compute_execution_sha256),
        "source_norm_inventory_sha256": (
            prepared.source_norm_inventory.inventory_sha256
        ),
        "source_parent_identity_sha256": (
            prepared.source_norm_inventory.source_parent_identity_sha256
        ),
        "source_qkvo_inventory_sha256": (
            prepared.source_norm_inventory.source_qkvo_inventory_sha256
        ),
        "source_state_sha256": prepared.source_state_sha256,
        "target_template_sha256": prepared.target_template_sha256,
        "normalized_input_decoder_sha256": (prepared.normalized_input_decoder_sha256),
        "residual_output_chart_sha256": prepared.residual_output_chart_sha256,
        "cache_manifest_sha256": prepared.cache_manifest_sha256,
        "cache_selective_session_sha256": (prepared.cache_selective_session_sha256),
        "cache_capability_sha256": prepared.cache_capability_sha256,
        "residual_chart_receipt_sha256": (prepared.residual_chart_receipt_sha256),
        "normalized_input_decoder_receipt_sha256": (
            prepared.normalized_input_decoder_receipt_sha256
        ),
        "context_bank_receipt_sha256": prepared.context_bank_receipt_sha256,
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "target_topology_sha256_before": prepared.target_snapshot.topology_sha256,
        "target_topology_sha256_after": current_topology,
        "parameter_count_before": parameter_count,
        "parameter_count_after": current_count,
        "parameter_count_changed": False,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "changed_state_keys": list(changed),
        "changed_state_key_count": len(changed),
        "prepared_expected_changed_state_keys": list(prepared_expected_changed),
        "prepared_expected_changed_state_keys_matched": True,
        "changed_parameter_roles": list(changed_roles),
        "unchanged_transaction_parameter_roles": list(unchanged_roles),
        "candidate_parameter_roles": list(_ALL_ROLES),
        "transaction_covered_parameter_roles": list(_ALL_ROLES),
        "transaction_covered_state_keys": list(transaction_covered_keys),
        "transaction_covered_parameter_role_count": len(_ALL_ROLES),
        "all_six_roles_supplied_to_atomic_transaction": True,
        "six_role_transaction_coverage_proven": True,
        "all_six_roles_changed": len(changed) == len(_ALL_ROLES),
        "actual_changed_bytes_cover_four_to_six_roles": 4 <= len(changed) <= 6,
        "target_seed_authentication": target_seed_authentication,
        "target_seed_authentication_sha256": canonical_sha256(
            target_seed_authentication
        ),
        "exact_zero_qkvo_blank_target_required_and_proven": True,
        "qk_norm_seed_policy": _NORM_SEED_POLICY,
        "qk_norm_seed_policy_authenticated": True,
        "norm_seed_transitions": norm_seed_transitions,
        "source_norm_equal_to_target_seed_roles": list(source_norm_equal_to_seed_roles),
        "source_norm_changed_from_target_seed_roles": list(
            source_norm_changed_from_seed_roles
        ),
        "complete_six_parameter_bundle_installed_transactionally": True,
        "qkvo_installed_from_one_coupled_intact_group_candidate": True,
        "query_and_key_norms_installed_from_direct_parent_tensors": True,
        "blank_query_norm_zero_escape_proven": (
            target_seed_authentication["q_layernorm.weight"]["seed_kind"]
            == _NORM_ZERO_SEED_KIND
            and "q_layernorm.weight" in changed_roles
        ),
        "blank_key_norm_zero_escape_proven": (
            target_seed_authentication["k_layernorm.weight"]["seed_kind"]
            == _NORM_ZERO_SEED_KIND
            and "k_layernorm.weight" in changed_roles
        ),
        "split_qk_or_vo_fallback_used": False,
        "incumbent_noop_fallback_used": False,
        "structural_proxy_fallback_used": False,
        "random_fallback_used": False,
        "heldout_payload_used_for_compilation": False,
        "numeric_diagnostics_are_advisory": True,
        "numeric_metrics_have_semantic_decision_authority": False,
        "numeric_metric_used_for_publication_or_rollback": False,
        "binary_manual_threshold_has_iteration_veto_authority": False,
        "binary_manual_acceptance_gate_required": False,
        "intelligence_retention_target_is_objective_not_numeric_cutoff": True,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "downstream_complete_response_review_required": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    receipt = MappingProxyType(
        {
            **receipt_body,
            "result_sha256": canonical_sha256(receipt_body),
        }
    )
    return Lfm2SemanticAttentionGroupQdeimInstallationResult(
        receipt=receipt,
        prepared=prepared,
    )


def compile_install_lfm2_semantic_attention_group_qdeim(
    **arguments: Any,
) -> (
    Lfm2SemanticAttentionGroupQdeimInstallationResult
    | Lfm2SemanticAttentionGroupQdeimEvidenceDeficit
):
    """Compile, authenticate, and install one coupled blank attention block."""

    prepared = prepare_lfm2_semantic_attention_group_qdeim_installation(**arguments)
    if isinstance(prepared, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit):
        return prepared
    target_model = arguments.get("target_model")
    return install_lfm2_prepared_semantic_attention_group_qdeim(
        target_model=target_model,
        prepared=prepared,
    )


__all__ = [
    "LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_INSTALLATION_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_NORM_INVENTORY_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_PREPARATION_SCHEMA",
    "Lfm2PreparedSemanticAttentionGroupQdeimInstallation",
    "Lfm2SemanticAttentionGroupQdeimInstallationError",
    "Lfm2SemanticAttentionGroupQdeimInstallationResult",
    "Lfm2SemanticAttentionGroupQdeimNormInventory",
    "compile_install_lfm2_semantic_attention_group_qdeim",
    "install_lfm2_prepared_semantic_attention_group_qdeim",
    "prepare_lfm2_semantic_attention_group_qdeim_installation",
    "seal_lfm2_semantic_attention_group_qdeim_norm_inventory",
]
