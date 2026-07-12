"""Target-native dense GQA Q/K/V/O residual corrections.

This module is deliberately separate from the attention-codebook runtime.  It
wraps ordinary dense ``nn.Linear`` Q/K/V/O projections in an explicitly
declared target architecture, exposes only sparse low-rank factors to an
output-space trainer, and can bake every correction back into the target's
ordinary dense matrices for publication.

The operator has no parent tensors, source-width bases, cache sidecars, or
internal-similarity objective.  Q/K/V corrections execute at the projection
boundary, before the target attention implementation applies normalization,
RoPE, or cache updates.  O corrections consume the real post-softmax physical
GQA-group context passed to the target's dense output projection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ProjectionRole = Literal["q", "k", "v", "o"]
GuidedQKVOTrainingStage = Literal["guided-left", "guided", "joint"]

DENSE_GQA_QKVO_ACCOUNTING_SCHEMA = "neural-forge-dense-gqa-qkvo-residual-accounting-v1"
DENSE_GQA_QKVO_REGISTRY_SCHEMA = "neural-forge-dense-gqa-qkvo-residual-registry-v1"
DENSE_GQA_QKVO_INSTALLATION_SCHEMA = (
    "neural-forge-dense-gqa-qkvo-residual-installation-v2"
)
DENSE_GQA_QKVO_TRAINING_SCOPE_SCHEMA = (
    "neural-forge-dense-gqa-qkvo-output-training-scope-v1"
)
DENSE_GQA_QKVO_BAKE_SCHEMA = "neural-forge-dense-gqa-qkvo-bake-v1"
GUIDED_DENSE_GQA_QKVO_REGISTRY_SCHEMA = (
    "neural-forge-guided-dense-gqa-qkvo-residual-registry-v1"
)
GUIDED_DENSE_GQA_QKVO_INSTALLATION_SCHEMA = (
    "neural-forge-guided-dense-gqa-qkvo-residual-installation-v1"
)
GUIDED_DENSE_GQA_QKVO_BAKE_SCHEMA = (
    "neural-forge-guided-dense-gqa-qkvo-residual-bake-v1"
)
QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA = "neural-forge-qkvo-output-jacobian-audit-v2"
GUIDED_QKVO_RIGHT_BASIS_V2_SCHEMA = "neural-forge-guided-qkvo-right-basis-v2"

_GUIDED_STAGE1_LOGICAL_LAYERS = 30
_GUIDED_STAGE1_HIDDEN_SIZE = 1536
_GUIDED_STAGE1_QUERY_HEADS = 16
_GUIDED_STAGE1_KEY_VALUE_HEADS = 4
_GUIDED_STAGE1_HEAD_DIM = 128
_GUIDED_STAGE1_RANK = 8
_GUIDED_STAGE1_SITE_COUNT = 120
_GUIDED_STAGE1_RIGHT_PARAMETER_COUNT = 4_915_200
_GUIDED_STAGE1_PARAMETER_COUNT = 7_127_040
_TENSOR_HASH_STAGING_BYTES = 4 * 1024 * 1024
_GUIDED_BASIS_NAMES: dict[ProjectionRole, str] = {
    "q": "q_orthogonal_right",
    "k": "k_orthogonal_right",
    "v": "v_orthogonal_right",
    "o": "o_orthogonal_right",
}


class DenseGQAQKVOResidualError(RuntimeError):
    """Base failure for target-native dense QKVO correction."""


class DenseGQAQKVOResidualTopologyError(DenseGQAQKVOResidualError):
    """The declared target geometry or live module topology is invalid."""


class DenseGQAQKVOResidualAccountingError(DenseGQAQKVOResidualError):
    """Closed-form and live target accounting disagree."""


class DenseGQAQKVOResidualTrainingError(DenseGQAQKVOResidualError):
    """The adapter-only output-training boundary was violated."""


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise DenseGQAQKVOResidualTopologyError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DenseGQAQKVOResidualTopologyError(
            f"{name} must be an integer."
        ) from error
    if result != value or (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive " if positive else "nonnegative "
        raise DenseGQAQKVOResidualTopologyError(f"{name} must be a {qualifier}integer.")
    return result


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DenseGQAQKVOResidualError(
            "QKVO provenance must be finite canonical JSON data."
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(
    value: Tensor,
    *,
    staging_observer: Callable[[int], None] | None = None,
) -> str:
    if not isinstance(value, Tensor):
        raise DenseGQAQKVOResidualTopologyError("Expected a tensor to hash.")
    detached = value.detach()
    if detached.layout != torch.strided or detached.device.type == "meta":
        raise DenseGQAQKVOResidualTopologyError(
            "Tensor hashing requires a dense non-meta tensor."
        )
    if not detached.is_contiguous():
        tensor_nbytes = int(detached.numel() * detached.element_size())
        if detached.device.type != "cpu" or tensor_nbytes > _TENSOR_HASH_STAGING_BYTES:
            raise DenseGQAQKVOResidualTopologyError(
                "Large or accelerator tensor hashing requires contiguous storage."
            )
        detached = detached.contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii")
    )
    flat = detached.view(-1)
    elements_per_chunk = max(
        1,
        _TENSOR_HASH_STAGING_BYTES // max(1, detached.element_size()),
    )
    for start in range(0, int(flat.numel()), elements_per_chunk):
        staged = flat[start : start + elements_per_chunk].to(device="cpu").contiguous()
        staged_nbytes = int(staged.numel() * staged.element_size())
        if staged_nbytes > _TENSOR_HASH_STAGING_BYTES:
            raise DenseGQAQKVOResidualTopologyError(
                "Tensor hash staging exceeded its hard chunk bound."
            )
        if staging_observer is not None:
            staging_observer(staged_nbytes)
        digest.update(memoryview(staged.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DenseGQAQKVOResidualTopologyError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _qkvo_factor_state_sha256(
    registry: SparseDenseGQAQKVOResidualRegistry,
) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
            for name, value in registry.named_parameters(recurse=False)
        ]
    )


def _guided_stage1_sites() -> tuple[DenseGQASite, ...]:
    return tuple(
        DenseGQASite(layer_index, physical_group)
        for layer_index in range(_GUIDED_STAGE1_LOGICAL_LAYERS)
        for physical_group in range(_GUIDED_STAGE1_KEY_VALUE_HEADS)
    )


def _guided_basis_tensor_manifest(
    tensors: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": _tensor_sha256(value),
            "numel": int(value.numel()),
            "nbytes": int(value.numel() * value.element_size()),
        }
        for name, value in tensors.items()
    ]


def guided_qkvo_basis_payload_sha256(basis_pack: Mapping[str, Any]) -> str:
    """Hash a v2 basis payload without relying on ``torch.save`` bytes.

    The pack's tensors are represented by an ordered, content-addressed
    manifest.  This makes the digest stable across serialization containers
    while still binding every tensor byte and every non-tensor metadata field.
    Installation independently revalidates the producer's manifest as well.
    """

    if not isinstance(basis_pack, Mapping):
        raise DenseGQAQKVOResidualTopologyError("basis_pack must be a mapping.")
    factor_tensors = basis_pack.get("factor_tensors")
    if not isinstance(factor_tensors, Mapping):
        raise DenseGQAQKVOResidualTopologyError(
            "The guided basis pack has no factor_tensors mapping."
        )
    names = tuple(_GUIDED_BASIS_NAMES.values())
    if set(factor_tensors) != set(names) or any(
        not isinstance(factor_tensors[name], Tensor) for name in names
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The guided basis pack tensor names or order changed."
        )
    metadata = {
        key: value for key, value in basis_pack.items() if key != "factor_tensors"
    }
    metadata["factor_tensors_manifest_for_pack_sha256"] = _guided_basis_tensor_manifest(
        {name: factor_tensors[name] for name in names}
    )
    return _canonical_sha256(metadata)


def _unique_parameter_count(module: nn.Module) -> int:
    return sum(int(value.numel()) for value in module.parameters())


@dataclass(frozen=True, slots=True)
class DenseGQAGeometry:
    """The complete target-native dense GQA projection geometry."""

    logical_layers: int
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    query_group_layout: str = "kv-major-contiguous"

    def __post_init__(self) -> None:
        for name in (
            "logical_layers",
            "hidden_size",
            "query_heads",
            "key_value_heads",
            "head_dim",
        ):
            object.__setattr__(
                self,
                name,
                _index(getattr(self, name), name, positive=True),
            )
        if self.query_heads % self.key_value_heads:
            raise DenseGQAQKVOResidualTopologyError(
                "query_heads must be divisible by key_value_heads."
            )
        if self.query_group_layout != "kv-major-contiguous":
            raise DenseGQAQKVOResidualTopologyError(
                "Only explicit kv-major-contiguous query-to-KV grouping is supported."
            )

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.key_value_heads * self.head_dim

    @property
    def query_heads_per_group(self) -> int:
        return self.query_heads // self.key_value_heads

    @property
    def query_group_width(self) -> int:
        return self.query_heads_per_group * self.head_dim

    def as_dict(self) -> dict[str, int | str]:
        return {
            "logical_layers": self.logical_layers,
            "hidden_size": self.hidden_size,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "query_width": self.query_width,
            "key_value_width": self.key_value_width,
            "query_heads_per_group": self.query_heads_per_group,
            "query_group_width": self.query_group_width,
            "query_group_layout": self.query_group_layout,
        }


@dataclass(frozen=True, slots=True, order=True)
class DenseGQASite:
    """One target physical ``(layer, KV group)`` correction site."""

    layer_index: int
    physical_group: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_index",
            _index(self.layer_index, "layer_index"),
        )
        object.__setattr__(
            self,
            "physical_group",
            _index(self.physical_group, "physical_group"),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "layer_index": self.layer_index,
            "physical_group": self.physical_group,
        }


def _validated_sites(
    sites: Sequence[DenseGQASite], geometry: DenseGQAGeometry
) -> tuple[DenseGQASite, ...]:
    values = tuple(sites)
    if not values or any(not isinstance(site, DenseGQASite) for site in values):
        raise DenseGQAQKVOResidualTopologyError(
            "sites must be a nonempty sequence of DenseGQASite values."
        )
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise DenseGQAQKVOResidualTopologyError("sites must be sorted and unique.")
    if any(
        site.layer_index >= geometry.logical_layers
        or site.physical_group >= geometry.key_value_heads
        for site in values
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "A dense QKVO site is outside the declared target geometry."
        )
    return values


@dataclass(frozen=True, slots=True)
class DenseGQAQKVOResidualAccounting:
    """Exact installed and baked accounting for target-native corrections."""

    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    query_group_width: int
    logical_layers: int
    rank: int
    site_count: int
    checkpoint_element_size: int
    q_adapter_parameters: int
    k_adapter_parameters: int
    v_adapter_parameters: int
    o_adapter_parameters: int
    adapter_parameters: int
    installed_parameter_delta: int
    baked_publication_parameter_delta: int
    installed_learned_parameter_bytes: int
    adapter_site_index_bytes: int
    adapter_schema_buffer_bytes: int
    installed_tensor_payload_bytes: int
    baked_publication_tensor_payload_bytes: int
    flops_per_site_token_while_installed: int
    baked_publication_adapter_flops_per_token: int

    @classmethod
    def build(
        cls,
        *,
        logical_layers: int,
        hidden_size: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
        rank: int,
        site_count: int,
        checkpoint_element_size: int = 2,
        schema_buffer_bytes: int = 8,
    ) -> DenseGQAQKVOResidualAccounting:
        geometry = DenseGQAGeometry(
            logical_layers=logical_layers,
            hidden_size=hidden_size,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
        )
        rank_value = _index(rank, "rank", positive=True)
        sites = _index(site_count, "site_count", positive=True)
        element_size = _index(
            checkpoint_element_size,
            "checkpoint_element_size",
            positive=True,
        )
        schema_bytes = _index(schema_buffer_bytes, "schema_buffer_bytes")
        query_group = geometry.query_group_width
        hidden = geometry.hidden_size
        head = geometry.head_dim
        q_parameters = sites * rank_value * (hidden + query_group)
        k_parameters = sites * rank_value * (hidden + head)
        v_parameters = sites * rank_value * (hidden + head)
        o_parameters = sites * rank_value * (query_group + hidden)
        adapter = q_parameters + k_parameters + v_parameters + o_parameters
        site_index_bytes = (
            sites * 2 * torch.tensor([], dtype=torch.int32).element_size()
        )
        learned_bytes = adapter * element_size
        return cls(
            hidden_size=hidden,
            query_heads=geometry.query_heads,
            key_value_heads=geometry.key_value_heads,
            head_dim=head,
            query_group_width=query_group,
            logical_layers=geometry.logical_layers,
            rank=rank_value,
            site_count=sites,
            checkpoint_element_size=element_size,
            q_adapter_parameters=q_parameters,
            k_adapter_parameters=k_parameters,
            v_adapter_parameters=v_parameters,
            o_adapter_parameters=o_parameters,
            adapter_parameters=adapter,
            installed_parameter_delta=adapter,
            baked_publication_parameter_delta=0,
            installed_learned_parameter_bytes=learned_bytes,
            adapter_site_index_bytes=site_index_bytes,
            adapter_schema_buffer_bytes=schema_bytes,
            installed_tensor_payload_bytes=(
                learned_bytes + site_index_bytes + schema_bytes
            ),
            baked_publication_tensor_payload_bytes=0,
            flops_per_site_token_while_installed=2 * (adapter // sites),
            baked_publication_adapter_flops_per_token=0,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_GQA_QKVO_ACCOUNTING_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "installed_factors_are_target_shaped": True,
            "source_sized_basis_parameters": 0,
            "parent_runtime_reference_count": 0,
            "bake_rule": "dense_weight += left_factor @ right_factor",
            "published_adapter_registry_retained": False,
            "published_dense_parameter_topology_unchanged": True,
            "artifact_header_and_manifest_delta_included": False,
        }
        payload["accounting_sha256"] = _canonical_sha256(payload)
        return payload


class SparseDenseGQAQKVOResidualRegistry(nn.Module):
    """Sparse target-site factors with one registered model owner."""

    _ROLES: tuple[ProjectionRole, ...] = ("q", "k", "v", "o")

    def __init__(
        self,
        sites: Sequence[DenseGQASite],
        *,
        geometry: DenseGQAGeometry,
        rank: int,
        dtype: torch.dtype,
        device: torch.device | str,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(geometry, DenseGQAGeometry):
            raise DenseGQAQKVOResidualTopologyError(
                "geometry must be DenseGQAGeometry."
            )
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise DenseGQAQKVOResidualTopologyError(
                "Adapter dtype must be floating point."
            )
        values = _validated_sites(sites, geometry)
        self.geometry = geometry
        self.rank = _index(rank, "rank", positive=True)
        rows_by_layer: list[list[tuple[int, int]]] = [
            [] for _ in range(geometry.logical_layers)
        ]
        for row_index, site in enumerate(values):
            rows_by_layer[site.layer_index].append((row_index, site.physical_group))
        # Forward dispatch must not copy the complete registry index from the
        # accelerator for every projection.  The sites have already passed the
        # sorted/unique/geometry checks above, so retain their derived lookup as
        # immutable host metadata.  ``site_indices`` remains the authenticated,
        # serialized buffer and is still hashed by the live/publication gates.
        self._site_rows_by_layer = tuple(tuple(rows) for rows in rows_by_layer)
        self.register_buffer(
            "site_indices",
            torch.tensor(
                [[site.layer_index, site.physical_group] for site in values],
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            "_registry_schema_version",
            torch.tensor(1, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "_registry_layout",
            torch.tensor(1, dtype=torch.int32, device=device),
        )
        seed = _index(initialization_seed, "initialization_seed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        widths = {
            "q": (geometry.query_group_width, geometry.hidden_size),
            "k": (geometry.head_dim, geometry.hidden_size),
            "v": (geometry.head_dim, geometry.hidden_size),
            "o": (geometry.hidden_size, geometry.query_group_width),
        }
        for role in self._ROLES:
            output_width, input_width = widths[role]
            left = torch.zeros(
                (len(values), output_width, self.rank),
                dtype=dtype,
                device=device,
            )
            right = torch.empty(
                (len(values), self.rank, input_width),
                dtype=torch.float32,
                device="cpu",
            )
            right.uniform_(
                -1.0 / math.sqrt(input_width),
                1.0 / math.sqrt(input_width),
                generator=generator,
            )
            setattr(self, f"{role}_left", nn.Parameter(left))
            setattr(
                self,
                f"{role}_right",
                nn.Parameter(right.to(device=device, dtype=dtype)),
            )
        self._factor_parameter_names = tuple(
            name for name, _value in self.named_parameters()
        )
        self._finite_parameter_signature_cache: (
            tuple[tuple[int, int, int, tuple[int, ...], torch.dtype, str], ...] | None
        ) = None
        self._assert_finite()

    @property
    def site_count(self) -> int:
        return int(self.site_indices.shape[0])

    @property
    def learned_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    @property
    def index_buffer_bytes(self) -> int:
        return int(self.site_indices.numel() * self.site_indices.element_size())

    @property
    def schema_buffer_bytes(self) -> int:
        return int(
            self._registry_schema_version.numel()
            * self._registry_schema_version.element_size()
            + self._registry_layout.numel() * self._registry_layout.element_size()
        )

    @property
    def starts_as_exact_zero_correction(self) -> bool:
        return all(
            int(torch.count_nonzero(getattr(self, f"{role}_left")).item()) == 0
            for role in self._ROLES
        )

    @property
    def input_factors_are_nonzero(self) -> bool:
        return all(
            int(torch.count_nonzero(getattr(self, f"{role}_right")).item()) > 0
            for role in self._ROLES
        )

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return factors only; no internal target API exists."""

        return tuple(self.parameters())

    def _factor_parameter_items(self) -> tuple[tuple[str, nn.Parameter], ...]:
        items = tuple(self.named_parameters())
        if tuple(name for name, _value in items) != self._factor_parameter_names:
            raise DenseGQAQKVOResidualTopologyError(
                "Adapter factor parameter topology changed."
            )
        return items

    def _scan_factor_finiteness(
        self,
        items: tuple[tuple[str, nn.Parameter], ...],
    ) -> None:
        for name, value in items:
            if not bool(torch.isfinite(value.detach()).all()):
                raise DenseGQAQKVOResidualError(
                    f"Adapter parameter {name!r} contains NaN or infinity."
                )

    @staticmethod
    def _finite_parameter_signature(
        items: tuple[tuple[str, nn.Parameter], ...],
    ) -> tuple[tuple[int, int, int, tuple[int, ...], torch.dtype, str], ...]:
        return tuple(
            (
                id(value),
                int(value.data_ptr()),
                int(value._version),
                tuple(value.shape),
                value.dtype,
                str(value.device),
            )
            for _name, value in items
        )

    def _assert_finite(self) -> None:
        """Validate factors once per numeric parameter version.

        PyTorch increments ``Parameter._version`` for optimizer updates,
        authenticated ``copy_`` restores, and other ordinary in-place writes.
        A complete factor scan is therefore necessary only after that compact
        ownership/storage/version signature changes. Failed or unstable scans
        are deliberately never cached.
        """

        items = self._factor_parameter_items()
        before = self._finite_parameter_signature(items)
        if before == self._finite_parameter_signature_cache:
            return
        self._scan_factor_finiteness(items)
        after_items = self._factor_parameter_items()
        after = self._finite_parameter_signature(after_items)
        if after != before:
            raise DenseGQAQKVOResidualError(
                "Adapter factors changed during finite validation."
            )
        self._finite_parameter_signature_cache = after

    def _site_rows(self, layer_index: int) -> tuple[tuple[int, int], ...]:
        layer = _index(layer_index, "layer_index")
        if layer >= self.geometry.logical_layers:
            raise DenseGQAQKVOResidualTopologyError(
                "layer_index is outside the target geometry."
            )
        return self._site_rows_by_layer[layer]

    def _correction(
        self,
        role: ProjectionRole,
        row_index: int,
        inputs: Tensor,
    ) -> Tensor:
        left = getattr(self, f"{role}_left")[row_index]
        right = getattr(self, f"{role}_right")[row_index]
        latent = F.linear(inputs.to(dtype=right.dtype), right)
        return F.linear(latent, left)

    def dense_delta(self, role: ProjectionRole, row_index: int) -> Tensor:
        """Materialize the target-local ``U @ R`` matrix for publication."""

        if role not in self._ROLES:
            raise DenseGQAQKVOResidualTopologyError(
                "Projection role must be q, k, v, or o."
            )
        row = _index(row_index, "row_index")
        if row >= self.site_count:
            raise DenseGQAQKVOResidualTopologyError(
                "row_index is outside the sparse site registry."
            )
        self._assert_finite()
        left = getattr(self, f"{role}_left")[row]
        right = getattr(self, f"{role}_right")[row]
        result = left @ right
        if not bool(torch.isfinite(result.detach()).all()):
            raise DenseGQAQKVOResidualError(
                f"Materialized {role.upper()} correction is non-finite."
            )
        return result

    def apply_to_projection(
        self,
        role: Literal["q", "k", "v"],
        inputs: Tensor,
        base_output: Tensor,
        *,
        layer_index: int,
    ) -> Tensor:
        if role not in {"q", "k", "v"}:
            raise DenseGQAQKVOResidualTopologyError(
                "apply_to_projection accepts q, k, or v."
            )
        geometry = self.geometry
        role_width = geometry.query_group_width if role == "q" else geometry.head_dim
        expected_output = (
            geometry.query_width if role == "q" else geometry.key_value_width
        )
        if (
            not isinstance(inputs, Tensor)
            or inputs.ndim == 0
            or int(inputs.shape[-1]) != geometry.hidden_size
            or not isinstance(base_output, Tensor)
            or tuple(base_output.shape[:-1]) != tuple(inputs.shape[:-1])
            or int(base_output.shape[-1]) != expected_output
        ):
            raise DenseGQAQKVOResidualTopologyError(
                f"{role.upper()} tensors do not match the target GQA geometry."
            )
        self._assert_finite()
        result = base_output
        for row_index, physical_group in self._site_rows(layer_index):
            correction = self._correction(role, row_index, inputs).to(
                dtype=base_output.dtype
            )
            start = physical_group * role_width
            stop = start + role_width
            result = result + F.pad(correction, (start, expected_output - stop))
        return result

    def apply_to_output(
        self,
        contexts: Tensor,
        base_output: Tensor,
        *,
        layer_index: int,
    ) -> Tensor:
        geometry = self.geometry
        if (
            not isinstance(contexts, Tensor)
            or contexts.ndim == 0
            or int(contexts.shape[-1]) != geometry.query_width
            or not isinstance(base_output, Tensor)
            or tuple(base_output.shape[:-1]) != tuple(contexts.shape[:-1])
            or int(base_output.shape[-1]) != geometry.hidden_size
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "O tensors do not match the target GQA geometry."
            )
        self._assert_finite()
        result = base_output
        width = geometry.query_group_width
        for row_index, physical_group in self._site_rows(layer_index):
            start = physical_group * width
            stop = start + width
            correction = self._correction("o", row_index, contexts[..., start:stop]).to(
                dtype=base_output.dtype
            )
            result = result + correction
        return result

    def as_dict(self) -> dict[str, Any]:
        role_parameters = {
            role: int(
                getattr(self, f"{role}_left").numel()
                + getattr(self, f"{role}_right").numel()
            )
            for role in self._ROLES
        }
        payload = {
            "schema": DENSE_GQA_QKVO_REGISTRY_SCHEMA,
            "geometry": self.geometry.as_dict(),
            "site_indices": self.site_indices.detach().cpu().tolist(),
            "rank": self.rank,
            "site_count": self.site_count,
            "learned_parameter_count": self.learned_parameter_count,
            "role_parameter_counts": role_parameters,
            "zero_output_factor_initialization": self.starts_as_exact_zero_correction,
            "random_input_factor_initialization": self.input_factors_are_nonzero,
            "target_shaped_factors_only": True,
            "source_sized_basis_parameters": 0,
            "parent_runtime_references": 0,
            "cross_token_mixing_inside_adapter": False,
            "output_function_training_parameters_only": True,
            "internal_numeric_matching_api_exposed": False,
        }
        payload["registry_sha256"] = _canonical_sha256(payload)
        return payload


class GuidedSparseDenseGQAQKVOResidualRegistry(SparseDenseGQAQKVOResidualRegistry):
    """A second QKVO product whose right rows come from the v2 audit.

    This is intentionally a distinct module.  Its factors are never stacked,
    concatenated, factorized, or otherwise regrouped with the inherited
    rank-eight product.
    """

    def __init__(
        self,
        sites: Sequence[DenseGQASite],
        *,
        geometry: DenseGQAGeometry,
        rank: int,
        guided_right_rows: Mapping[ProjectionRole, Tensor],
        source_basis_tensor_sha256s: Mapping[ProjectionRole, str],
        audit_report_sha256: str,
        basis_envelope_file_sha256: str,
        basis_payload_sha256: str,
        basis_manifest_sha256: str,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if (
            not isinstance(guided_right_rows, Mapping)
            or tuple(guided_right_rows) != self._ROLES
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "guided_right_rows must be ordered Q, K, V, O rows."
            )
        if (
            not isinstance(source_basis_tensor_sha256s, Mapping)
            or tuple(source_basis_tensor_sha256s) != self._ROLES
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Source basis hashes must be ordered Q, K, V, O digests."
            )
        values = _validated_sites(sites, geometry)
        rank_value = _index(rank, "rank", positive=True)
        expected_input_width = {
            "q": geometry.hidden_size,
            "k": geometry.hidden_size,
            "v": geometry.hidden_size,
            "o": geometry.query_group_width,
        }
        authenticated_rows: dict[ProjectionRole, Tensor] = {}
        authenticated_hashes: dict[ProjectionRole, str] = {}
        for role in self._ROLES:
            value = guided_right_rows[role]
            expected_shape = (
                len(values),
                rank_value,
                expected_input_width[role],
            )
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or not value.is_contiguous()
                or tuple(value.shape) != expected_shape
                or not bool(torch.isfinite(value).all())
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    f"Guided {role.upper()} right rows must be finite contiguous "
                    f"CPU FP32 with shape {expected_shape}."
                )
            digest = _tensor_sha256(value)
            expected_digest = _required_sha256(
                source_basis_tensor_sha256s[role],
                f"source {role} basis tensor SHA-256",
            )
            if digest != expected_digest:
                raise DenseGQAQKVOResidualTopologyError(
                    f"Guided {role.upper()} right rows failed source authentication."
                )
            authenticated_rows[role] = value
            authenticated_hashes[role] = digest
        super().__init__(
            values,
            geometry=geometry,
            rank=rank_value,
            dtype=dtype,
            device=device,
            initialization_seed=0,
        )
        with torch.no_grad():
            for role in self._ROLES:
                getattr(self, f"{role}_right").copy_(
                    authenticated_rows[role].to(device=device, dtype=dtype)
                )
        self.audit_report_sha256 = _required_sha256(
            audit_report_sha256, "audit_report_sha256"
        )
        self.basis_envelope_file_sha256 = _required_sha256(
            basis_envelope_file_sha256, "basis_envelope_file_sha256"
        )
        self.basis_payload_sha256 = _required_sha256(
            basis_payload_sha256, "basis_payload_sha256"
        )
        self.basis_manifest_sha256 = _required_sha256(
            basis_manifest_sha256, "basis_manifest_sha256"
        )
        self.source_basis_tensor_sha256s = dict(authenticated_hashes)
        self._assert_finite()
        if not self.starts_as_exact_zero_correction:
            raise DenseGQAQKVOResidualTopologyError(
                "The guided left factors must start at exact zero."
            )
        if any(
            _tensor_sha256(getattr(self, f"{role}_right")) != authenticated_hashes[role]
            for role in self._ROLES
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Device installation changed an authenticated guided right basis."
            )

    def as_dict(self) -> dict[str, Any]:
        role_parameters = {
            role: int(
                getattr(self, f"{role}_left").numel()
                + getattr(self, f"{role}_right").numel()
            )
            for role in self._ROLES
        }
        payload = {
            "schema": GUIDED_DENSE_GQA_QKVO_REGISTRY_SCHEMA,
            "geometry": self.geometry.as_dict(),
            "site_indices": self.site_indices.detach().cpu().tolist(),
            "rank": self.rank,
            "site_count": self.site_count,
            "learned_parameter_count": self.learned_parameter_count,
            "role_parameter_counts": role_parameters,
            "zero_guided_left_initialization": (self.starts_as_exact_zero_correction),
            "authenticated_guided_right_initialization": True,
            "audit_report_sha256": self.audit_report_sha256,
            "basis_envelope_file_sha256": self.basis_envelope_file_sha256,
            "basis_payload_sha256": self.basis_payload_sha256,
            "basis_manifest_sha256": self.basis_manifest_sha256,
            "source_basis_tensor_sha256s": dict(self.source_basis_tensor_sha256s),
            "separate_from_inherited_registry": True,
            "old_and_guided_factors_concatenated": False,
            "old_and_guided_factors_regrouped": False,
            "target_shaped_factors_only": True,
            "parent_runtime_references": 0,
            "cross_token_mixing_inside_adapter": False,
        }
        payload["registry_sha256"] = _canonical_sha256(payload)
        return payload


class DenseGQAQKVOResidualProjection(nn.Module):
    """Linear-compatible target projection plus a sparse correction view."""

    def __init__(
        self,
        base_projection: nn.Linear,
        adapter_registry: SparseDenseGQAQKVOResidualRegistry,
        *,
        layer_index: int,
        role: ProjectionRole,
    ) -> None:
        super().__init__()
        if type(base_projection) is not nn.Linear:
            raise DenseGQAQKVOResidualTopologyError(
                "Dense QKVO correction requires an ordinary nn.Linear projection."
            )
        if role not in SparseDenseGQAQKVOResidualRegistry._ROLES:
            raise DenseGQAQKVOResidualTopologyError(
                "Projection role must be q, k, v, or o."
            )
        object.__setattr__(self, "_adapter_registry", adapter_registry)
        object.__setattr__(self, "_guided_adapter_registry", None)
        self.base_projection = base_projection
        self.layer_index = _index(layer_index, "layer_index")
        self.role = role
        self.in_features = base_projection.in_features
        self.out_features = base_projection.out_features
        self.train(base_projection.training)

    @property
    def adapter_registry(self) -> SparseDenseGQAQKVOResidualRegistry:
        return self._adapter_registry

    @property
    def guided_adapter_registry(
        self,
    ) -> GuidedSparseDenseGQAQKVOResidualRegistry | None:
        return self._guided_adapter_registry

    def _attach_guided_registry(
        self,
        registry: GuidedSparseDenseGQAQKVOResidualRegistry,
    ) -> None:
        if not isinstance(registry, GuidedSparseDenseGQAQKVOResidualRegistry):
            raise DenseGQAQKVOResidualTopologyError(
                "Only an authenticated guided QKVO registry can be attached."
            )
        if self._guided_adapter_registry is not None:
            raise DenseGQAQKVOResidualTopologyError(
                "A guided QKVO tranche is already attached."
            )
        if (
            registry.geometry != self.adapter_registry.geometry
            or registry.rank != self.adapter_registry.rank
            or not torch.equal(
                registry.site_indices.detach().cpu(),
                self.adapter_registry.site_indices.detach().cpu(),
            )
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Guided and inherited QKVO registry topology differs."
            )
        object.__setattr__(self, "_guided_adapter_registry", registry)

    def _detach_guided_registry(
        self,
        registry: GuidedSparseDenseGQAQKVOResidualRegistry,
    ) -> None:
        if self._guided_adapter_registry is not registry:
            raise DenseGQAQKVOResidualTopologyError(
                "The attached guided QKVO registry changed."
            )
        object.__setattr__(self, "_guided_adapter_registry", None)

    @property
    def weight(self) -> Tensor:
        return self.base_projection.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_projection.bias

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base_projection(inputs)
        if self.role == "o":
            result = self.adapter_registry.apply_to_output(
                inputs,
                base_output,
                layer_index=self.layer_index,
            )
            if self.guided_adapter_registry is not None:
                result = self.guided_adapter_registry.apply_to_output(
                    inputs,
                    result,
                    layer_index=self.layer_index,
                )
            return result
        result = self.adapter_registry.apply_to_projection(
            self.role, inputs, base_output, layer_index=self.layer_index
        )
        if self.guided_adapter_registry is not None:
            result = self.guided_adapter_registry.apply_to_projection(
                self.role, inputs, result, layer_index=self.layer_index
            )
        return result


@dataclass(frozen=True, slots=True)
class DenseGQAQKVOResidualInstallationReport:
    sites: tuple[DenseGQASite, ...]
    geometry: DenseGQAGeometry
    rank: int
    factor_dtype: str
    owner_name: str
    projection_names: Mapping[str, str]
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_installed: int
    adapter_site_indices_sha256: str
    base_projection_state_sha256: str
    accounting: DenseGQAQKVOResidualAccounting

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_GQA_QKVO_INSTALLATION_SCHEMA,
            "sites": [site.as_dict() for site in self.sites],
            "geometry": self.geometry.as_dict(),
            "rank": self.rank,
            "factor_dtype": self.factor_dtype,
            "owner_name": self.owner_name,
            "projection_names": dict(self.projection_names),
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "adapter_site_indices_sha256": self.adapter_site_indices_sha256,
            "base_projection_state_sha256": self.base_projection_state_sha256,
            "accounting": self.accounting.as_dict(),
            "q_correction_boundary": "dense-projection-output-before-q-norm-and-rope",
            "k_correction_boundary": (
                "dense-projection-output-before-k-norm-rope-and-cache-append"
            ),
            "v_correction_boundary": "dense-projection-output-before-cache-append",
            "o_correction_boundary": (
                "post-softmax-physical-group-context-before-residual-add"
            ),
            "pointwise_causal_operator": True,
            "future_token_access": False,
            "cache_sidecar_required": False,
            "parent_runtime_reference_count": 0,
            "source_sized_basis_parameters": 0,
            "internal_numeric_similarity_used": False,
            "zero_initialization_preserves_dense_child": True,
            "bake_to_plain_dense_weights_supported": True,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class DenseGQAQKVOBakeReport:
    sites: tuple[DenseGQASite, ...]
    geometry: DenseGQAGeometry
    rank: int
    adapter_parameters_removed: int
    model_parameter_count_before_installation: int
    model_parameter_count_before_bake: int
    model_parameter_count_after_bake: int
    dense_delta_sha256: str
    affected_slice_count: int

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_GQA_QKVO_BAKE_SCHEMA,
            "sites": [site.as_dict() for site in self.sites],
            "geometry": self.geometry.as_dict(),
            "rank": self.rank,
            "adapter_parameters_removed": self.adapter_parameters_removed,
            "model_parameter_count_before_installation": (
                self.model_parameter_count_before_installation
            ),
            "model_parameter_count_before_bake": self.model_parameter_count_before_bake,
            "model_parameter_count_after_bake": self.model_parameter_count_after_bake,
            "dense_delta_sha256": self.dense_delta_sha256,
            "affected_slice_count": self.affected_slice_count,
            "bake_rule": "dense_weight += left_factor @ right_factor",
            "published_projection_type": "torch.nn.Linear",
            "published_adapter_parameter_count": 0,
            "published_parameter_delta_vs_original_dense_target": 0,
            "published_adapter_flops_per_token": 0,
            "adapter_registry_removed": True,
            "projection_wrappers_removed": True,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_retained": False,
            "internal_numeric_matching_used": False,
            "functional_publication_gate_still_required": True,
        }
        payload["bake_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class GuidedDenseGQAQKVOResidualInstallationReport:
    """Authenticated installation evidence for old plus guided products."""

    sites: tuple[DenseGQASite, ...]
    geometry: DenseGQAGeometry
    rank: int
    factor_dtype: str
    owner_name: str
    projection_names: Mapping[str, str]
    inherited_adapter_parameter_count: int
    guided_adapter_parameter_count: int
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_before_guided_installation: int
    model_parameter_count_installed: int
    adapter_site_indices_sha256: str
    audit_site_indices_sha256: str
    base_projection_state_sha256: str
    audit_report_sha256: str
    basis_envelope_file_sha256: str
    basis_payload_sha256: str
    basis_manifest_sha256: str
    source_basis_tensor_sha256s: Mapping[str, str]
    inherited_installation_sha256: str
    inherited_factor_state_sha256_at_install: str
    inherited_handle_identity: int
    inherited_registry_identity: int

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": GUIDED_DENSE_GQA_QKVO_INSTALLATION_SCHEMA,
            "sites": [site.as_dict() for site in self.sites],
            "geometry": self.geometry.as_dict(),
            "rank": self.rank,
            "factor_dtype": self.factor_dtype,
            "owner_name": self.owner_name,
            "projection_names": dict(self.projection_names),
            "inherited_adapter_parameter_count": (
                self.inherited_adapter_parameter_count
            ),
            "guided_adapter_parameter_count": self.guided_adapter_parameter_count,
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_before_guided_installation": (
                self.model_parameter_count_before_guided_installation
            ),
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "adapter_site_indices_sha256": self.adapter_site_indices_sha256,
            "audit_site_indices_sha256": self.audit_site_indices_sha256,
            "base_projection_state_sha256": self.base_projection_state_sha256,
            "audit_report_sha256": self.audit_report_sha256,
            "basis_envelope_file_sha256": self.basis_envelope_file_sha256,
            "basis_payload_sha256": self.basis_payload_sha256,
            "basis_manifest_sha256": self.basis_manifest_sha256,
            "source_basis_tensor_sha256s": dict(self.source_basis_tensor_sha256s),
            "inherited_installation_sha256": self.inherited_installation_sha256,
            "inherited_factor_state_sha256_at_install": (
                self.inherited_factor_state_sha256_at_install
            ),
            "inherited_handle_identity": self.inherited_handle_identity,
            "inherited_registry_identity": self.inherited_registry_identity,
            "initial_training_stage": "guided-left",
            "forward_product_order": ["inherited", "guided"],
            "bake_product_order": ["inherited", "guided"],
            "factor_concatenation_used": False,
            "factor_regrouping_used": False,
            "zero_guided_left_preserves_inherited_function_exactly": True,
            "q_correction_boundary": ("dense-projection-output-before-q-norm-and-rope"),
            "k_correction_boundary": (
                "dense-projection-output-before-k-norm-rope-and-cache-append"
            ),
            "v_correction_boundary": ("dense-projection-output-before-cache-append"),
            "o_correction_boundary": (
                "post-softmax-physical-group-context-before-residual-add"
            ),
            "pointwise_causal_operator": True,
            "cache_sidecar_required": False,
            "parent_runtime_reference_count": 0,
            "bake_to_plain_dense_weights_supported": True,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class GuidedDenseGQAQKVOBakeReport:
    """Publication evidence for the sequential inherited/guided bake."""

    sites: tuple[DenseGQASite, ...]
    geometry: DenseGQAGeometry
    rank: int
    inherited_adapter_parameters_removed: int
    guided_adapter_parameters_removed: int
    adapter_parameters_removed: int
    model_parameter_count_before_installation: int
    model_parameter_count_before_bake: int
    model_parameter_count_after_bake: int
    inherited_dense_delta_sha256: str
    guided_dense_delta_sha256: str
    sequential_dense_delta_sha256: str
    affected_slice_count: int

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": GUIDED_DENSE_GQA_QKVO_BAKE_SCHEMA,
            "sites": [site.as_dict() for site in self.sites],
            "geometry": self.geometry.as_dict(),
            "rank": self.rank,
            "inherited_adapter_parameters_removed": (
                self.inherited_adapter_parameters_removed
            ),
            "guided_adapter_parameters_removed": (
                self.guided_adapter_parameters_removed
            ),
            "adapter_parameters_removed": self.adapter_parameters_removed,
            "model_parameter_count_before_installation": (
                self.model_parameter_count_before_installation
            ),
            "model_parameter_count_before_bake": self.model_parameter_count_before_bake,
            "model_parameter_count_after_bake": self.model_parameter_count_after_bake,
            "inherited_dense_delta_sha256": self.inherited_dense_delta_sha256,
            "guided_dense_delta_sha256": self.guided_dense_delta_sha256,
            "sequential_dense_delta_sha256": self.sequential_dense_delta_sha256,
            "dense_delta_sha256": self.sequential_dense_delta_sha256,
            "affected_slice_count": self.affected_slice_count,
            "bake_rule": [
                "dense_weight += inherited_left @ inherited_right",
                "dense_weight += guided_left @ guided_right",
            ],
            "factor_concatenation_used": False,
            "factor_regrouping_used": False,
            "published_projection_type": "torch.nn.Linear",
            "published_adapter_parameter_count": 0,
            "published_parameter_delta_vs_original_dense_target": 0,
            "published_adapter_flops_per_token": 0,
            "inherited_registry_removed": True,
            "guided_registry_removed": True,
            "projection_wrappers_removed": True,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_retained": False,
            "functional_publication_gate_still_required": True,
        }
        payload["bake_sha256"] = _canonical_sha256(payload)
        return payload


def _authenticate_guided_qkvo_sources(
    inherited_handle: InstalledDenseGQAQKVOResidualAdapter,
    *,
    audit_report: Mapping[str, Any],
    basis_pack: Mapping[str, Any],
    expected_audit_report_sha256: str,
    expected_basis_envelope_file_sha256: str,
    expected_basis_payload_sha256: str,
    expected_basis_manifest_sha256: str,
) -> tuple[
    dict[ProjectionRole, Tensor],
    dict[ProjectionRole, str],
    str,
    str,
    str,
    str,
]:
    """Fail closed on both v2 payloads and their external hash pins."""

    if not isinstance(inherited_handle, InstalledDenseGQAQKVOResidualAdapter):
        raise DenseGQAQKVOResidualTopologyError(
            "Guided installation requires the live inherited QKVO handle."
        )
    inherited_handle.assert_live()
    report_sha256 = _required_sha256(
        expected_audit_report_sha256, "expected_audit_report_sha256"
    )
    envelope_file_sha256 = _required_sha256(
        expected_basis_envelope_file_sha256,
        "expected_basis_envelope_file_sha256",
    )
    payload_sha256 = _required_sha256(
        expected_basis_payload_sha256, "expected_basis_payload_sha256"
    )
    manifest_sha256 = _required_sha256(
        expected_basis_manifest_sha256,
        "expected_basis_manifest_sha256",
    )
    if not isinstance(audit_report, Mapping):
        raise DenseGQAQKVOResidualTopologyError("audit_report must be a mapping.")
    report = dict(audit_report)
    embedded_report_sha256 = report.pop("report_sha256", None)
    if (
        audit_report.get("schema") != QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA
        or embedded_report_sha256 != report_sha256
        or _canonical_sha256(report) != report_sha256
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The output-Jacobian audit v2 report failed hash authentication."
        )
    expected_sites = _guided_stage1_sites()
    expected_site_indices = [
        [site.layer_index, site.physical_group] for site in expected_sites
    ]
    expected_site_indices_sha256 = _canonical_sha256(expected_site_indices)
    if (
        inherited_handle.report.geometry
        != DenseGQAGeometry(
            logical_layers=_GUIDED_STAGE1_LOGICAL_LAYERS,
            hidden_size=_GUIDED_STAGE1_HIDDEN_SIZE,
            query_heads=_GUIDED_STAGE1_QUERY_HEADS,
            key_value_heads=_GUIDED_STAGE1_KEY_VALUE_HEADS,
            head_dim=_GUIDED_STAGE1_HEAD_DIM,
        )
        or inherited_handle.report.rank != _GUIDED_STAGE1_RANK
        or inherited_handle.report.sites != expected_sites
        or audit_report.get("site_count") != _GUIDED_STAGE1_SITE_COUNT
        or audit_report.get("site_indices") != expected_site_indices
        or audit_report.get("site_indices_sha256") != expected_site_indices_sha256
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The guided audit is not the ordered 120-site stage-one topology."
        )
    roles = list(SparseDenseGQAQKVOResidualRegistry._ROLES)
    per_role = audit_report.get("per_role")
    if (
        audit_report.get("passed") is not True
        or audit_report.get("diagnostic_passed_roles") != roles
        or audit_report.get("authorized_roles") != roles
        or audit_report.get("installable_roles") != roles
        or not isinstance(per_role, Mapping)
        or tuple(per_role) != tuple(roles)
        or any(
            not isinstance(per_role[role], Mapping)
            or per_role[role].get("passed") is not True
            for role in roles
        )
        or audit_report.get("installer_must_ignore_unauthorized_role_tensors")
        is not True
        or audit_report.get("all_four_role_tensors_retained_for_diagnostics")
        is not True
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The audit did not authorize all four installable QKVO roles."
        )
    if not isinstance(basis_pack, Mapping):
        raise DenseGQAQKVOResidualTopologyError("basis_pack must be a mapping.")
    expected_pack_keys = {
        "schema",
        "factor_state_sha256",
        "site_indices",
        "site_indices_sha256",
        "basis_manifest_sha256",
        "tensor_count",
        "total_numel",
        "total_nbytes",
        "factors",
        "factor_tensors",
        "diagnostic_passed_roles",
        "authorized_roles",
        "installable_roles",
        "authorized_for_training",
        "installer_must_ignore_unauthorized_role_tensors",
        "all_four_role_tensors_retained_for_diagnostics",
        "contains_model_state",
        "contains_parent_state",
        "contains_optimizer_state",
        "contains_gradients",
        "contains_activations",
        "contains_hidden_states",
    }
    if set(basis_pack) != expected_pack_keys:
        raise DenseGQAQKVOResidualTopologyError(
            "The basis pack contains missing, extra, or unauthorized payload fields."
        )
    if guided_qkvo_basis_payload_sha256(basis_pack) != payload_sha256:
        raise DenseGQAQKVOResidualTopologyError(
            "The guided basis payload failed its canonical SHA-256 pin."
        )
    if (
        basis_pack.get("schema") != GUIDED_QKVO_RIGHT_BASIS_V2_SCHEMA
        or basis_pack.get("factor_state_sha256")
        != audit_report.get("factor_state_sha256_before")
        or basis_pack.get("site_indices") != expected_site_indices
        or basis_pack.get("site_indices_sha256") != expected_site_indices_sha256
        or basis_pack.get("diagnostic_passed_roles") != roles
        or basis_pack.get("authorized_roles") != roles
        or basis_pack.get("installable_roles") != roles
        or basis_pack.get("authorized_for_training") is not True
        or basis_pack.get("installer_must_ignore_unauthorized_role_tensors") is not True
        or basis_pack.get("all_four_role_tensors_retained_for_diagnostics") is not True
        or any(
            basis_pack.get(name) is not False
            for name in (
                "contains_model_state",
                "contains_parent_state",
                "contains_optimizer_state",
                "contains_gradients",
                "contains_activations",
                "contains_hidden_states",
            )
        )
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The guided basis v2 authorization contract is malformed."
        )
    factor_tensors = basis_pack.get("factor_tensors")
    factor_names = tuple(_GUIDED_BASIS_NAMES.values())
    if not isinstance(factor_tensors, Mapping) or set(factor_tensors) != set(
        factor_names
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The basis pack must contain only ordered Q/K/V/O diagnostic tensors."
        )
    typed_tensors: dict[str, Tensor] = {}
    for name in factor_names:
        value = factor_tensors[name]
        if not isinstance(value, Tensor):
            raise DenseGQAQKVOResidualTopologyError(
                f"Basis factor {name!r} is not a tensor."
            )
        typed_tensors[name] = value
    manifest = _guided_basis_tensor_manifest(typed_tensors)
    computed_manifest_sha256 = _canonical_sha256(manifest)
    if (
        computed_manifest_sha256 != manifest_sha256
        or basis_pack.get("basis_manifest_sha256") != manifest_sha256
        or audit_report.get("basis_manifest_sha256") != manifest_sha256
        or basis_pack.get("factors") != manifest
        or audit_report.get("basis_manifest") != manifest
        or basis_pack.get("tensor_count") != 4
        or basis_pack.get("total_numel") != _GUIDED_STAGE1_RIGHT_PARAMETER_COUNT
        or basis_pack.get("total_nbytes") != _GUIDED_STAGE1_RIGHT_PARAMETER_COUNT * 4
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The guided basis tensor manifest failed authentication."
        )
    inherited_factor_sha256 = _qkvo_factor_state_sha256(
        inherited_handle.adapter_registry
    )
    if (
        audit_report.get("inherited_qkvo_state_sha256_before")
        != inherited_factor_sha256
        or audit_report.get("inherited_qkvo_state_sha256_after")
        != inherited_factor_sha256
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "The live inherited QKVO factors differ from the audited product."
        )
    guided_rows = {
        role: typed_tensors[_GUIDED_BASIS_NAMES[role]]
        for role in SparseDenseGQAQKVOResidualRegistry._ROLES
    }
    tensor_sha256s = {
        role: manifest[index]["tensor_sha256"]
        for index, role in enumerate(SparseDenseGQAQKVOResidualRegistry._ROLES)
    }
    return (
        guided_rows,
        tensor_sha256s,
        report_sha256,
        envelope_file_sha256,
        payload_sha256,
        manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class _ProjectionTensorSnapshot:
    tensor: Tensor
    data_pointer: int
    version: int
    sha256: str


def _projection_state(
    projections: Mapping[tuple[int, ProjectionRole], nn.Linear],
) -> dict[tuple[int, ProjectionRole, str], _ProjectionTensorSnapshot]:
    result: dict[tuple[int, ProjectionRole, str], _ProjectionTensorSnapshot] = {}
    for (layer_index, role), projection in projections.items():
        for name, value in (("weight", projection.weight), ("bias", projection.bias)):
            if value is None:
                continue
            result[(layer_index, role, name)] = _ProjectionTensorSnapshot(
                tensor=value,
                data_pointer=int(value.data_ptr()),
                version=int(value._version),
                sha256=_tensor_sha256(value),
            )
    return result


def _projection_state_sha256(
    state: Mapping[tuple[int, ProjectionRole, str], _ProjectionTensorSnapshot],
) -> str:
    return _canonical_sha256(
        [
            {
                "key": [layer, role, name],
                "data_pointer": snapshot.data_pointer,
                "version": snapshot.version,
                "tensor_sha256": snapshot.sha256,
            }
            for (layer, role, name), snapshot in sorted(state.items())
        ]
    )


def _normalized_projection_names(
    values: Mapping[str, str] | None,
) -> dict[ProjectionRole, str]:
    result: dict[ProjectionRole, str] = {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "o_proj",
    }
    if values is not None:
        if not isinstance(values, Mapping) or set(values) != set(result):
            raise DenseGQAQKVOResidualTopologyError(
                "projection_names must map exactly q, k, v, and o."
            )
        result = {role: str(values[role]) for role in result}
    if (
        any(not name.isidentifier() for name in result.values())
        or len(set(result.values())) != 4
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "Projection attribute names must be distinct identifiers."
        )
    return result


def _attention_module_map(
    values: Mapping[int, nn.Module] | Sequence[nn.Module],
    geometry: DenseGQAGeometry,
) -> dict[int, nn.Module]:
    if isinstance(values, Mapping):
        result: dict[int, nn.Module] = {}
        for raw_index, module in values.items():
            index = _index(raw_index, "attention module layer index")
            if index >= geometry.logical_layers or not isinstance(module, nn.Module):
                raise DenseGQAQKVOResidualTopologyError(
                    "attention_modules contains an invalid target layer."
                )
            result[index] = module
        return result
    if isinstance(values, (str, bytes)):
        raise DenseGQAQKVOResidualTopologyError(
            "attention_modules must contain target attention modules."
        )
    try:
        sequence = tuple(values)
    except TypeError as error:
        raise DenseGQAQKVOResidualTopologyError(
            "attention_modules must be a mapping or sequence."
        ) from error
    if len(sequence) != geometry.logical_layers or any(
        not isinstance(module, nn.Module) for module in sequence
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "An attention-module sequence must cover every declared target layer."
        )
    return dict(enumerate(sequence))


def _validate_projection_shape(
    role: ProjectionRole,
    projection: nn.Linear,
    geometry: DenseGQAGeometry,
) -> None:
    expected = {
        "q": (geometry.query_width, geometry.hidden_size),
        "k": (geometry.key_value_width, geometry.hidden_size),
        "v": (geometry.key_value_width, geometry.hidden_size),
        "o": (geometry.hidden_size, geometry.query_width),
    }[role]
    actual = (projection.out_features, projection.in_features)
    if actual != expected or tuple(projection.weight.shape) != expected:
        raise DenseGQAQKVOResidualTopologyError(
            f"Dense {role.upper()} projection shape {actual} does not match {expected}."
        )
    if projection.bias is not None and tuple(projection.bias.shape) != (expected[0],):
        raise DenseGQAQKVOResidualTopologyError(
            f"Dense {role.upper()} projection bias has the wrong shape."
        )


def _assert_unaliased_bake_weights(
    model: nn.Module,
    projections: Mapping[tuple[int, ProjectionRole], nn.Linear],
) -> None:
    try:
        all_named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - supported Torch versions expose this.
        all_named = tuple(model.named_parameters())
    for (layer_index, role), projection in projections.items():
        aliases = [name for name, value in all_named if value is projection.weight]
        if len(aliases) != 1:
            raise DenseGQAQKVOResidualTopologyError(
                f"Dense layer {layer_index} {role.upper()} weight is aliased; "
                "publication bake would change undeclared consumers."
            )


class InstalledDenseGQAQKVOResidualAdapter:
    """Live target-native adapter with rejection rollback and dense bake."""

    def __init__(
        self,
        *,
        model: nn.Module,
        attention_modules: Mapping[int, nn.Module],
        adapter_registry: SparseDenseGQAQKVOResidualRegistry,
        report: DenseGQAQKVOResidualInstallationReport,
        original_projections: Mapping[tuple[int, ProjectionRole], nn.Linear],
        installed_views: Mapping[
            tuple[int, ProjectionRole], DenseGQAQKVOResidualProjection
        ],
        projection_state_before: Mapping[
            tuple[int, ProjectionRole, str], _ProjectionTensorSnapshot
        ],
    ) -> None:
        self.model = model
        self.attention_modules = dict(attention_modules)
        self.adapter_registry = adapter_registry
        self.report = report
        self._original_projections = dict(original_projections)
        self._installed_views = dict(installed_views)
        self._projection_state_before = dict(projection_state_before)
        self._state: Literal["live", "rolled-back", "baked"] = "live"
        self._bake_report: DenseGQAQKVOBakeReport | None = None
        self._published_projections: dict[tuple[int, ProjectionRole], nn.Linear] = {}

    @property
    def is_applied(self) -> bool:
        return self._state == "live"

    @property
    def is_baked(self) -> bool:
        return self._state == "baked"

    @property
    def bake_report(self) -> DenseGQAQKVOBakeReport | None:
        return self._bake_report

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        return self.adapter_registry.output_trainable_parameters()

    def _assert_base_unchanged(self) -> None:
        current = _projection_state(self._original_projections)
        if set(current) != set(self._projection_state_before):
            raise DenseGQAQKVOResidualTopologyError(
                "Dense base projection topology changed."
            )
        for key, before in self._projection_state_before.items():
            after = current[key]
            if (
                after.tensor is not before.tensor
                or after.data_pointer != before.data_pointer
                or after.version != before.version
                or after.sha256 != before.sha256
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "A dense base projection changed during the QKVO experiment."
                )

    def _assert_unique_registry_owner(self) -> None:
        paths = [
            name
            for name, module in self.model.named_modules(remove_duplicate=False)
            if module is self.adapter_registry
        ]
        if paths != [self.report.owner_name]:
            raise DenseGQAQKVOResidualTopologyError(
                "Dense QKVO registry must have exactly one registered owner."
            )

    def assert_structurally_live(self) -> DenseGQAQKVOResidualInstallationReport:
        """Validate ownership/base topology without reading disposable values."""

        if self._state != "live":
            raise DenseGQAQKVOResidualTopologyError(
                "Dense QKVO adapter is no longer live."
            )
        if (
            getattr(self.model, self.report.owner_name, None)
            is not self.adapter_registry
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Model lost the dense QKVO registry owner."
            )
        self._assert_unique_registry_owner()
        for (layer_index, role), installed in self._installed_views.items():
            name = self.report.projection_names[role]
            if (
                getattr(self.attention_modules[layer_index], name, None)
                is not installed
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "A live dense QKVO projection wrapper was replaced."
                )
        if _tensor_sha256(self.adapter_registry.site_indices) != (
            self.report.adapter_site_indices_sha256
        ):
            raise DenseGQAQKVOResidualTopologyError("Dense QKVO site indices changed.")
        self._assert_base_unchanged()
        return self.report

    def assert_live(self) -> DenseGQAQKVOResidualInstallationReport:
        self.assert_structurally_live()
        self.adapter_registry._assert_finite()
        return self.report

    def output_training_scope(self) -> DenseGQAAdapterOnlyOutputTrainingScope:
        return DenseGQAAdapterOnlyOutputTrainingScope(self)

    def rollback(self) -> None:
        """Reject the candidate and restore exact original module objects."""

        if self._state == "rolled-back":
            return
        if self._state == "baked":
            raise DenseGQAQKVOResidualTopologyError(
                "A published dense bake cannot be rejected through adapter rollback."
            )
        owner = getattr(self.model, self.report.owner_name, None)
        if owner is not None and owner is not self.adapter_registry:
            raise DenseGQAQKVOResidualTopologyError(
                "Dense QKVO registry owner was replaced; rollback is unsafe."
            )
        if owner is self.adapter_registry:
            self._assert_unique_registry_owner()
        for (layer_index, role), installed in self._installed_views.items():
            name = self.report.projection_names[role]
            if (
                getattr(self.attention_modules[layer_index], name, None)
                is not installed
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "Dense QKVO projection wrapper was replaced; rollback is unsafe."
                )
        if _tensor_sha256(self.adapter_registry.site_indices) != (
            self.report.adapter_site_indices_sha256
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Dense QKVO site indices changed; rollback is unsafe."
            )
        # Adapter factors may be trained or non-finite.  They are isolated and
        # removed; only the dense base must remain exact.
        self._assert_base_unchanged()
        for (layer_index, role), original in self._original_projections.items():
            name = self.report.projection_names[role]
            setattr(self.attention_modules[layer_index], name, original)
        if getattr(self.model, self.report.owner_name, None) is self.adapter_registry:
            delattr(self.model, self.report.owner_name)
        elif hasattr(self.model, self.report.owner_name):
            raise DenseGQAQKVOResidualTopologyError(
                "Dense QKVO registry owner changed during rollback."
            )
        self._assert_base_unchanged()
        if (
            _unique_parameter_count(self.model)
            != self.report.model_parameter_count_before
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Dense QKVO rollback parameter accounting failed."
            )
        for (layer_index, role), original in self._original_projections.items():
            name = self.report.projection_names[role]
            if getattr(self.attention_modules[layer_index], name, None) is not original:
                raise DenseGQAQKVOResidualTopologyError(
                    "Dense QKVO projection rollback failed."
                )
        self._state = "rolled-back"

    def _bake_updates(
        self,
    ) -> list[tuple[int, ProjectionRole, slice, slice, Tensor]]:
        geometry = self.report.geometry
        updates: list[tuple[int, ProjectionRole, slice, slice, Tensor]] = []
        sites = self.adapter_registry.site_indices.detach().cpu().tolist()
        for row_index, (layer_index, group) in enumerate(sites):
            layer = int(layer_index)
            physical_group = int(group)
            for role in SparseDenseGQAQKVOResidualRegistry._ROLES:
                delta = self.adapter_registry.dense_delta(role, row_index).detach()
                if role == "q":
                    start = physical_group * geometry.query_group_width
                    row_slice = slice(start, start + geometry.query_group_width)
                    column_slice = slice(0, geometry.hidden_size)
                elif role in {"k", "v"}:
                    start = physical_group * geometry.head_dim
                    row_slice = slice(start, start + geometry.head_dim)
                    column_slice = slice(0, geometry.hidden_size)
                else:
                    start = physical_group * geometry.query_group_width
                    row_slice = slice(0, geometry.hidden_size)
                    column_slice = slice(start, start + geometry.query_group_width)
                projection = self._original_projections[(layer, role)]
                expected = projection.weight[row_slice, column_slice]
                if tuple(delta.shape) != tuple(expected.shape):
                    raise DenseGQAQKVOResidualTopologyError(
                        "Materialized dense correction has the wrong target slice shape."
                    )
                updates.append(
                    (
                        layer,
                        role,
                        row_slice,
                        column_slice,
                        delta.to(
                            device=projection.weight.device,
                            dtype=projection.weight.dtype,
                        ),
                    )
                )
        return updates

    def bake_for_publication(self) -> DenseGQAQKVOBakeReport:
        """Absorb all target-local ``U @ R`` matrices and remove the adapter."""

        self.assert_live()
        before_bake_count = _unique_parameter_count(self.model)
        updates = self._bake_updates()
        if len(updates) != 4 * len(self.report.sites):
            raise DenseGQAQKVOResidualTopologyError(
                "Dense bake did not produce four updates per declared site."
            )
        delta_manifest = [
            {
                "layer_index": layer,
                "role": role,
                "shape": list(delta.shape),
                "sha256": _tensor_sha256(delta),
            }
            for layer, role, _rows, _columns, delta in updates
        ]
        delta_sha256 = _canonical_sha256(delta_manifest)
        # Build replacement dense modules off-model.  A failed materialization
        # therefore leaves the live base tensors, their versions, wrappers,
        # and registry bit-exact instead of trying to undo floating-point
        # additions in place.
        published: dict[tuple[int, ProjectionRole], nn.Linear] = {}
        try:
            with torch.no_grad():
                for key, original in self._original_projections.items():
                    replacement = nn.Linear(
                        original.in_features,
                        original.out_features,
                        bias=original.bias is not None,
                        device=original.weight.device,
                        dtype=original.weight.dtype,
                    )
                    replacement.weight.copy_(original.weight)
                    replacement.weight.requires_grad_(
                        bool(original.weight.requires_grad)
                    )
                    if original.bias is not None:
                        assert replacement.bias is not None
                        replacement.bias.copy_(original.bias)
                        replacement.bias.requires_grad_(
                            bool(original.bias.requires_grad)
                        )
                    replacement.train(original.training)
                    published[key] = replacement
                for layer, role, rows, columns, delta in updates:
                    published[(layer, role)].weight[rows, columns].add_(delta)
            if any(
                not bool(torch.isfinite(projection.weight.detach()).all())
                or (
                    projection.bias is not None
                    and not bool(torch.isfinite(projection.bias.detach()).all())
                )
                for projection in published.values()
            ):
                raise DenseGQAQKVOResidualError(
                    "Publication bake produced a non-finite dense projection."
                )
            for (layer_index, role), replacement in published.items():
                name = self.report.projection_names[role]
                setattr(self.attention_modules[layer_index], name, replacement)
            if (
                getattr(self.model, self.report.owner_name, None)
                is self.adapter_registry
            ):
                delattr(self.model, self.report.owner_name)
            else:
                raise DenseGQAQKVOResidualTopologyError(
                    "Dense QKVO registry owner changed during publication bake."
                )
            after_count = _unique_parameter_count(self.model)
            if after_count != self.report.model_parameter_count_before:
                raise DenseGQAQKVOResidualAccountingError(
                    "Baked dense model did not return to its original parameter count."
                )
            if any(
                type(
                    getattr(
                        self.attention_modules[layer],
                        self.report.projection_names[role],
                    )
                )
                is not nn.Linear
                for layer, role in self._original_projections
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "Publication bake left a non-dense projection wrapper."
                )
        except Exception:
            if (
                getattr(self.model, self.report.owner_name, None)
                is not self.adapter_registry
            ):
                if hasattr(self.model, self.report.owner_name):
                    delattr(self.model, self.report.owner_name)
                self.model.add_module(self.report.owner_name, self.adapter_registry)
            for (layer_index, role), installed in self._installed_views.items():
                setattr(
                    self.attention_modules[layer_index],
                    self.report.projection_names[role],
                    installed,
                )
            raise
        report = DenseGQAQKVOBakeReport(
            sites=self.report.sites,
            geometry=self.report.geometry,
            rank=self.report.rank,
            adapter_parameters_removed=self.report.adapter_parameter_count,
            model_parameter_count_before_installation=(
                self.report.model_parameter_count_before
            ),
            model_parameter_count_before_bake=before_bake_count,
            model_parameter_count_after_bake=after_count,
            dense_delta_sha256=delta_sha256,
            affected_slice_count=len(updates),
        )
        self._state = "baked"
        self._bake_report = report
        self._published_projections = published
        return report

    def restore_prepublication_native(
        self,
        *,
        expected_model_parameter_count: int | None = None,
    ) -> None:
        """Undo a bake while a larger atomic publication transaction fails.

        Normal rejection still uses :meth:`rollback` before publication.  This
        narrower escape hatch exists so a transaction that composes QKVO with
        another independently baked native operator can restore the exact
        pre-transaction child if the later bake or final certificate fails.
        """

        if self._state == "live":
            self.rollback()
            return
        if self._state == "rolled-back":
            return
        for (layer_index, role), published in self._published_projections.items():
            name = self.report.projection_names[role]
            if (
                getattr(self.attention_modules[layer_index], name, None)
                is not published
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "Published dense QKVO ownership changed; restoration is unsafe."
                )
        for (layer_index, role), original in self._original_projections.items():
            name = self.report.projection_names[role]
            setattr(self.attention_modules[layer_index], name, original)
        self._assert_base_unchanged()
        expected_count = (
            self.report.model_parameter_count_before
            if expected_model_parameter_count is None
            else _index(
                expected_model_parameter_count,
                "expected_model_parameter_count",
                positive=True,
            )
        )
        if _unique_parameter_count(self.model) != expected_count:
            raise DenseGQAQKVOResidualTopologyError(
                "Prepublication QKVO restoration changed the native budget."
            )
        self._state = "rolled-back"

    def __enter__(self) -> InstalledDenseGQAQKVOResidualAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._state == "live":
            self.rollback()


def _dense_updates_for_registry(
    registry: SparseDenseGQAQKVOResidualRegistry,
    *,
    geometry: DenseGQAGeometry,
    projections: Mapping[tuple[int, ProjectionRole], nn.Linear],
) -> list[tuple[int, ProjectionRole, slice, slice, Tensor]]:
    updates: list[tuple[int, ProjectionRole, slice, slice, Tensor]] = []
    sites = registry.site_indices.detach().cpu().tolist()
    for row_index, (raw_layer, raw_group) in enumerate(sites):
        layer = int(raw_layer)
        physical_group = int(raw_group)
        for role in SparseDenseGQAQKVOResidualRegistry._ROLES:
            delta = registry.dense_delta(role, row_index).detach()
            if role == "q":
                start = physical_group * geometry.query_group_width
                row_slice = slice(start, start + geometry.query_group_width)
                column_slice = slice(0, geometry.hidden_size)
            elif role in {"k", "v"}:
                start = physical_group * geometry.head_dim
                row_slice = slice(start, start + geometry.head_dim)
                column_slice = slice(0, geometry.hidden_size)
            else:
                start = physical_group * geometry.query_group_width
                row_slice = slice(0, geometry.hidden_size)
                column_slice = slice(start, start + geometry.query_group_width)
            projection = projections[(layer, role)]
            expected = projection.weight[row_slice, column_slice]
            if tuple(delta.shape) != tuple(expected.shape):
                raise DenseGQAQKVOResidualTopologyError(
                    "Materialized guided correction has the wrong target slice shape."
                )
            updates.append(
                (
                    layer,
                    role,
                    row_slice,
                    column_slice,
                    delta.to(
                        device=projection.weight.device,
                        dtype=projection.weight.dtype,
                    ),
                )
            )
    return updates


class InstalledGuidedDenseGQAQKVOResidualTranche:
    """Transaction-compatible owner of inherited plus guided QKVO products."""

    _STAGES: tuple[GuidedQKVOTrainingStage, ...] = (
        "guided-left",
        "guided",
        "joint",
    )

    def __init__(
        self,
        *,
        inherited_handle: InstalledDenseGQAQKVOResidualAdapter,
        guided_registry: GuidedSparseDenseGQAQKVOResidualRegistry,
        report: GuidedDenseGQAQKVOResidualInstallationReport,
        inherited_requires_grad: Mapping[str, bool],
    ) -> None:
        self.inherited_handle = inherited_handle
        self.guided_registry = guided_registry
        self.report = report
        self.model = inherited_handle.model
        self.attention_modules = inherited_handle.attention_modules
        self._inherited_requires_grad = dict(inherited_requires_grad)
        self._state: Literal["live", "rolled-back-guided", "rolled-back", "baked"] = (
            "live"
        )
        self._training_stage: GuidedQKVOTrainingStage = "guided-left"
        self._guided_right_training_started = False
        self._bake_report: GuidedDenseGQAQKVOBakeReport | None = None
        self._published_projections: dict[tuple[int, ProjectionRole], nn.Linear] = {}
        self._source_binding = MappingProxyType(
            {
                "audit_report_sha256": report.audit_report_sha256,
                "basis_envelope_file_sha256": (report.basis_envelope_file_sha256),
                "basis_payload_sha256": report.basis_payload_sha256,
                "basis_manifest_sha256": report.basis_manifest_sha256,
                "source_basis_tensor_sha256s": MappingProxyType(
                    dict(report.source_basis_tensor_sha256s)
                ),
                "inherited_installation_sha256": (report.inherited_installation_sha256),
                "inherited_factor_state_sha256_at_install": (
                    report.inherited_factor_state_sha256_at_install
                ),
                "inherited_handle_identity": report.inherited_handle_identity,
                "inherited_registry_identity": report.inherited_registry_identity,
                "adapter_site_indices_sha256": (report.adapter_site_indices_sha256),
                "audit_site_indices_sha256": report.audit_site_indices_sha256,
            }
        )
        self._set_requires_grad_for_stage("guided-left")

    @property
    def adapter_registry(self) -> SparseDenseGQAQKVOResidualRegistry:
        """Compatibility alias for the inherited transaction registry."""

        return self.inherited_handle.adapter_registry

    @property
    def is_applied(self) -> bool:
        return self._state == "live"

    @property
    def is_baked(self) -> bool:
        return self._state == "baked"

    @property
    def bake_report(self) -> GuidedDenseGQAQKVOBakeReport | None:
        return self._bake_report

    @property
    def training_stage(self) -> GuidedQKVOTrainingStage:
        return self._training_stage

    def factor_source_binding(self) -> Mapping[str, Any]:
        return self._source_binding

    def _set_requires_grad_for_stage(self, stage: GuidedQKVOTrainingStage) -> None:
        inherited_trainable = stage == "joint"
        for _name, value in self.adapter_registry.named_parameters(recurse=False):
            value.requires_grad_(inherited_trainable)
        guided_right_trainable = stage in {"guided", "joint"}
        for name, value in self.guided_registry.named_parameters(recurse=False):
            value.requires_grad_(name.endswith("_left") or guided_right_trainable)

    def set_training_stage(
        self, stage: GuidedQKVOTrainingStage
    ) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        if stage not in self._STAGES:
            raise DenseGQAQKVOResidualTrainingError(
                "Guided QKVO stage must be guided-left, guided, or joint."
            )
        current_index = self._STAGES.index(self._training_stage)
        requested_index = self._STAGES.index(stage)
        if requested_index < current_index:
            raise DenseGQAQKVOResidualTrainingError(
                "Guided QKVO training stages cannot move backward."
            )
        self._training_stage = stage
        if stage in {"guided", "joint"}:
            self._guided_right_training_started = True
        self._set_requires_grad_for_stage(stage)
        return self.output_trainable_parameters()

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        inherited = tuple(self.adapter_registry.parameters(recurse=False))
        guided = tuple(self.guided_registry.parameters(recurse=False))
        if self._training_stage == "guided-left":
            result = tuple(
                value
                for name, value in self.guided_registry.named_parameters(recurse=False)
                if name.endswith("_left")
            )
        elif self._training_stage == "guided":
            result = guided
        else:
            result = (*inherited, *guided)
        if not result or any(not value.requires_grad for value in result):
            raise DenseGQAQKVOResidualTrainingError(
                "The guided QKVO trainable stage is internally inconsistent."
            )
        return result

    def _assert_unique_guided_owner(self) -> None:
        paths = [
            name
            for name, module in self.model.named_modules(remove_duplicate=False)
            if module is self.guided_registry
        ]
        if paths != [self.report.owner_name]:
            raise DenseGQAQKVOResidualTopologyError(
                "Guided QKVO registry must have exactly one registered owner."
            )

    def assert_structurally_live(
        self,
    ) -> GuidedDenseGQAQKVOResidualInstallationReport:
        if self._state != "live":
            raise DenseGQAQKVOResidualTopologyError(
                "The guided QKVO tranche is no longer live."
            )
        if (
            id(self.inherited_handle) != self.report.inherited_handle_identity
            or id(self.adapter_registry) != self.report.inherited_registry_identity
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "The inherited QKVO handle identity changed."
            )
        self.inherited_handle.assert_structurally_live()
        if (
            getattr(self.model, self.report.owner_name, None)
            is not self.guided_registry
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Model lost the guided QKVO registry owner."
            )
        self._assert_unique_guided_owner()
        for installed in self.inherited_handle._installed_views.values():
            if installed.guided_adapter_registry is not self.guided_registry:
                raise DenseGQAQKVOResidualTopologyError(
                    "A projection lost the separate guided QKVO product."
                )
        if _tensor_sha256(self.guided_registry.site_indices) != (
            self.report.adapter_site_indices_sha256
        ):
            raise DenseGQAQKVOResidualTopologyError("Guided QKVO site indices changed.")
        if (
            self._training_stage != "joint"
            and _qkvo_factor_state_sha256(self.adapter_registry)
            != self.report.inherited_factor_state_sha256_at_install
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "The inherited product changed before joint training was enabled."
            )
        if not self._guided_right_training_started and any(
            _tensor_sha256(getattr(self.guided_registry, f"{role}_right"))
            != self.report.source_basis_tensor_sha256s[role]
            for role in SparseDenseGQAQKVOResidualRegistry._ROLES
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "A guided right basis changed during guided-left-only training."
            )
        return self.report

    def assert_live(self) -> GuidedDenseGQAQKVOResidualInstallationReport:
        self.assert_structurally_live()
        self.adapter_registry._assert_finite()
        self.guided_registry._assert_finite()
        return self.report

    def output_training_scope(
        self,
        stage: GuidedQKVOTrainingStage | None = None,
    ) -> GuidedDenseGQAQKVOOutputTrainingScope:
        if stage is not None:
            self.set_training_stage(stage)
        return GuidedDenseGQAQKVOOutputTrainingScope(self)

    @contextmanager
    def suspend_guided_registry_owner_for_nested_publication(
        self,
    ) -> Iterator[None]:
        """Temporarily hide only the new registry from a legacy nested bake.

        The inherited compiler transaction installed its lexical adapter after
        the original QKVO product.  Consequently the lexical adapter's exact
        state-key and parameter-count baseline cannot include this later
        guided registry.  Publication still has to bake lexical first.  This
        narrow scope unregisters only the authenticated guided owner while the
        lexical adapter restores its legacy baseline, then immediately
        re-registers the *same* registry before the sequential old+guided QKVO
        bake.  Projection wrappers retain their non-owning references, and no
        forward pass or checkpoint is permitted at this intermediate boundary.
        """

        self.assert_live()
        owner_name = self.report.owner_name
        if getattr(self.model, owner_name, None) is not self.guided_registry:
            raise DenseGQAQKVOResidualTopologyError(
                "Guided registry owner changed before nested publication."
            )
        guided_state_sha256 = _qkvo_factor_state_sha256(self.guided_registry)
        guided_parameter_ownership = tuple(
            (
                name,
                id(value),
                int(value.data_ptr()),
                bool(value.requires_grad),
            )
            for name, value in self.guided_registry.named_parameters(recurse=False)
        )
        before = _unique_parameter_count(self.model)
        delattr(self.model, owner_name)
        suspended = _unique_parameter_count(self.model)
        if suspended != before - self.report.guided_adapter_parameter_count:
            self.model.add_module(owner_name, self.guided_registry)
            raise DenseGQAQKVOResidualAccountingError(
                "Suspending the guided registry changed the wrong parameter count."
            )
        if any(
            module is self.guided_registry
            for _name, module in self.model.named_modules(remove_duplicate=False)
        ):
            self.model.add_module(owner_name, self.guided_registry)
            raise DenseGQAQKVOResidualTopologyError(
                "Guided registry retained an undeclared registered owner."
            )
        try:
            yield
        finally:
            if hasattr(self.model, owner_name):
                raise DenseGQAQKVOResidualTopologyError(
                    "Guided registry owner path was occupied during nested publication."
                )
            before_restore = _unique_parameter_count(self.model)
            self.model.add_module(owner_name, self.guided_registry)
            if (
                _unique_parameter_count(self.model)
                != before_restore + self.report.guided_adapter_parameter_count
                or tuple(
                    (
                        name,
                        id(value),
                        int(value.data_ptr()),
                        bool(value.requires_grad),
                    )
                    for name, value in self.guided_registry.named_parameters(
                        recurse=False
                    )
                )
                != guided_parameter_ownership
                or _qkvo_factor_state_sha256(self.guided_registry)
                != guided_state_sha256
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "Nested publication did not restore the exact guided registry."
                )
            self.assert_live()

    def rollback_guided_only(
        self,
    ) -> InstalledDenseGQAQKVOResidualAdapter:
        """Remove only the new tranche and leave the inherited product live."""

        if self._state == "rolled-back-guided":
            return self.inherited_handle
        if self._state != "live":
            raise DenseGQAQKVOResidualTopologyError(
                "Only a live guided tranche can be removed independently."
            )
        self.assert_structurally_live()
        count_before_removal = _unique_parameter_count(self.model)
        for installed in self.inherited_handle._installed_views.values():
            installed._detach_guided_registry(self.guided_registry)
        if getattr(self.model, self.report.owner_name, None) is self.guided_registry:
            delattr(self.model, self.report.owner_name)
        elif hasattr(self.model, self.report.owner_name):
            raise DenseGQAQKVOResidualTopologyError(
                "Guided QKVO registry owner changed during rollback."
            )
        for name, value in self.adapter_registry.named_parameters(recurse=False):
            value.requires_grad_(self._inherited_requires_grad[name])
        if (
            _unique_parameter_count(self.model)
            != count_before_removal - self.report.guided_adapter_parameter_count
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Guided-only rollback parameter accounting failed."
            )
        self.inherited_handle.assert_live()
        self._state = "rolled-back-guided"
        return self.inherited_handle

    def rollback(self) -> None:
        """Reject the combined candidate and restore the exact native child."""

        if self._state == "rolled-back":
            return
        if self._state == "baked":
            raise DenseGQAQKVOResidualTopologyError(
                "A published guided bake cannot use live rollback."
            )
        if self._state == "live":
            self.rollback_guided_only()
        if self.inherited_handle.is_applied:
            self.inherited_handle.rollback()
        self._state = "rolled-back"

    @staticmethod
    def _delta_manifest(
        updates: Sequence[tuple[int, ProjectionRole, slice, slice, Tensor]],
    ) -> tuple[list[dict[str, Any]], str]:
        manifest = [
            {
                "layer_index": layer,
                "role": role,
                "shape": list(delta.shape),
                "sha256": _tensor_sha256(delta),
            }
            for layer, role, _rows, _columns, delta in updates
        ]
        return manifest, _canonical_sha256(manifest)

    def bake_for_publication(self) -> GuidedDenseGQAQKVOBakeReport:
        """Bake inherited first, guided second, then remove both sidecars."""

        self.assert_live()
        before_bake_count = _unique_parameter_count(self.model)
        original = self.inherited_handle._original_projections
        inherited_updates = _dense_updates_for_registry(
            self.adapter_registry,
            geometry=self.report.geometry,
            projections=original,
        )
        guided_updates = _dense_updates_for_registry(
            self.guided_registry,
            geometry=self.report.geometry,
            projections=original,
        )
        expected_updates = 4 * len(self.report.sites)
        if (
            len(inherited_updates) != expected_updates
            or len(guided_updates) != expected_updates
        ):
            raise DenseGQAQKVOResidualTopologyError(
                "Sequential QKVO bake did not produce four updates per site."
            )
        inherited_manifest, inherited_sha256 = self._delta_manifest(inherited_updates)
        guided_manifest, guided_sha256 = self._delta_manifest(guided_updates)
        sequential_sha256 = _canonical_sha256(
            {
                "order": ["inherited", "guided"],
                "inherited": inherited_manifest,
                "guided": guided_manifest,
            }
        )
        published: dict[tuple[int, ProjectionRole], nn.Linear] = {}
        try:
            with torch.no_grad():
                for key, base in original.items():
                    replacement = nn.Linear(
                        base.in_features,
                        base.out_features,
                        bias=base.bias is not None,
                        device=base.weight.device,
                        dtype=base.weight.dtype,
                    )
                    replacement.weight.copy_(base.weight)
                    replacement.weight.requires_grad_(bool(base.weight.requires_grad))
                    if base.bias is not None:
                        assert replacement.bias is not None
                        replacement.bias.copy_(base.bias)
                        replacement.bias.requires_grad_(bool(base.bias.requires_grad))
                    replacement.train(base.training)
                    published[key] = replacement
                # The two loops are deliberately separate.  Their floating
                # point addition order is part of the publication contract.
                for layer, role, rows, columns, delta in inherited_updates:
                    published[(layer, role)].weight[rows, columns].add_(delta)
                for layer, role, rows, columns, delta in guided_updates:
                    published[(layer, role)].weight[rows, columns].add_(delta)
            if any(
                not bool(torch.isfinite(projection.weight).all())
                or (
                    projection.bias is not None
                    and not bool(torch.isfinite(projection.bias).all())
                )
                for projection in published.values()
            ):
                raise DenseGQAQKVOResidualError(
                    "Sequential QKVO bake produced a non-finite projection."
                )
            for (layer_index, role), replacement in published.items():
                name = self.report.projection_names[role]
                setattr(self.attention_modules[layer_index], name, replacement)
            if getattr(self.model, self.report.owner_name, None) is not (
                self.guided_registry
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "Guided registry owner changed during publication."
                )
            delattr(self.model, self.report.owner_name)
            old_owner_name = self.inherited_handle.report.owner_name
            if getattr(self.model, old_owner_name, None) is not self.adapter_registry:
                raise DenseGQAQKVOResidualTopologyError(
                    "Inherited registry owner changed during publication."
                )
            delattr(self.model, old_owner_name)
            after_count = _unique_parameter_count(self.model)
            if after_count != self.report.model_parameter_count_before:
                raise DenseGQAQKVOResidualAccountingError(
                    "Sequential QKVO bake did not restore the inherited native budget."
                )
            if any(
                type(
                    getattr(
                        self.attention_modules[layer],
                        self.report.projection_names[role],
                    )
                )
                is not nn.Linear
                for layer, role in original
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "Sequential QKVO publication left a projection wrapper."
                )
        except Exception:
            old_owner_name = self.inherited_handle.report.owner_name
            if getattr(self.model, old_owner_name, None) is not self.adapter_registry:
                if hasattr(self.model, old_owner_name):
                    delattr(self.model, old_owner_name)
                self.model.add_module(old_owner_name, self.adapter_registry)
            if getattr(self.model, self.report.owner_name, None) is not (
                self.guided_registry
            ):
                if hasattr(self.model, self.report.owner_name):
                    delattr(self.model, self.report.owner_name)
                self.model.add_module(self.report.owner_name, self.guided_registry)
            for (
                layer_index,
                role,
            ), installed in self.inherited_handle._installed_views.items():
                setattr(
                    self.attention_modules[layer_index],
                    self.report.projection_names[role],
                    installed,
                )
            raise
        for installed in self.inherited_handle._installed_views.values():
            installed._detach_guided_registry(self.guided_registry)
        report = GuidedDenseGQAQKVOBakeReport(
            sites=self.report.sites,
            geometry=self.report.geometry,
            rank=self.report.rank,
            inherited_adapter_parameters_removed=(
                self.report.inherited_adapter_parameter_count
            ),
            guided_adapter_parameters_removed=(
                self.report.guided_adapter_parameter_count
            ),
            adapter_parameters_removed=self.report.adapter_parameter_count,
            model_parameter_count_before_installation=(
                self.report.model_parameter_count_before
            ),
            model_parameter_count_before_bake=before_bake_count,
            model_parameter_count_after_bake=after_count,
            inherited_dense_delta_sha256=inherited_sha256,
            guided_dense_delta_sha256=guided_sha256,
            sequential_dense_delta_sha256=sequential_sha256,
            affected_slice_count=len(inherited_updates) + len(guided_updates),
        )
        inherited_report = DenseGQAQKVOBakeReport(
            sites=self.inherited_handle.report.sites,
            geometry=self.inherited_handle.report.geometry,
            rank=self.inherited_handle.report.rank,
            adapter_parameters_removed=(
                self.inherited_handle.report.adapter_parameter_count
            ),
            model_parameter_count_before_installation=(
                self.inherited_handle.report.model_parameter_count_before
            ),
            model_parameter_count_before_bake=before_bake_count,
            model_parameter_count_after_bake=after_count,
            dense_delta_sha256=inherited_sha256,
            affected_slice_count=len(inherited_updates),
        )
        self.inherited_handle._state = "baked"
        self.inherited_handle._bake_report = inherited_report
        self.inherited_handle._published_projections = published
        self._published_projections = published
        self._bake_report = report
        self._state = "baked"
        return report

    def restore_prepublication_native(
        self,
        *,
        expected_model_parameter_count: int | None = None,
    ) -> None:
        """Undo a successful combined bake after a later publication failure."""

        if self._state == "live":
            self.rollback()
            return
        if self._state in {"rolled-back", "rolled-back-guided"}:
            return
        for (layer_index, role), published in self._published_projections.items():
            name = self.report.projection_names[role]
            if (
                getattr(self.attention_modules[layer_index], name, None)
                is not published
            ):
                raise DenseGQAQKVOResidualTopologyError(
                    "Published guided QKVO ownership changed; restoration is unsafe."
                )
        for (
            layer_index,
            role,
        ), original in self.inherited_handle._original_projections.items():
            setattr(
                self.attention_modules[layer_index],
                self.report.projection_names[role],
                original,
            )
        self.inherited_handle._assert_base_unchanged()
        expected = (
            self.inherited_handle.report.model_parameter_count_before
            if expected_model_parameter_count is None
            else _index(
                expected_model_parameter_count,
                "expected_model_parameter_count",
                positive=True,
            )
        )
        if _unique_parameter_count(self.model) != expected:
            raise DenseGQAQKVOResidualTopologyError(
                "Prepublication guided QKVO restoration changed the native budget."
            )
        # Both handles intentionally share this map while a successful bake is
        # still rollback-capable.  Once native restoration succeeds, retaining
        # it would keep a complete off-model copy of every baked Q/K/V/O
        # projection alive.  Clear both attributes explicitly so this failure
        # path owns no hidden dense clone.
        self._published_projections.clear()
        self.inherited_handle._published_projections.clear()
        self.inherited_handle._state = "rolled-back"
        self._state = "rolled-back"

    def __enter__(self) -> InstalledGuidedDenseGQAQKVOResidualTranche:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._state == "live":
            self.rollback()


class GuidedDenseGQAQKVOOutputTrainingScope:
    """Freeze every unselected parameter for one guided training stage."""

    def __init__(self, handle: InstalledGuidedDenseGQAQKVOResidualTranche) -> None:
        if not isinstance(handle, InstalledGuidedDenseGQAQKVOResidualTranche):
            raise DenseGQAQKVOResidualTrainingError(
                "Guided training scope requires a live guided handle."
            )
        self.handle = handle
        self._frozen: tuple[
            tuple[nn.Parameter, bool, Tensor | None, int, int], ...
        ] = ()
        self._module_modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise DenseGQAQKVOResidualTrainingError(
                "Guided output training scope cannot be re-entered."
            )
        trainable = self.handle.output_trainable_parameters()
        trainable_ids = {id(value) for value in trainable}
        self._frozen = tuple(
            (
                value,
                bool(value.requires_grad),
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in self.handle.model.parameters()
            if id(value) not in trainable_ids
        )
        self._module_modes = tuple(
            (module, bool(module.training)) for module in self.handle.model.modules()
        )
        for value, _requires_grad, _gradient, _version, _pointer in self._frozen:
            value.requires_grad_(False)
        self.handle.model.eval()
        self._entered = True
        return trainable

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._frozen:
            value.requires_grad_(requires_grad)
            if value.grad is not gradient:
                value.grad = gradient
        for module, training in self._module_modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            or bool(value.requires_grad) != requires_grad
            for value, requires_grad, gradient, version, pointer in self._frozen
        ):
            raise DenseGQAQKVOResidualTrainingError(
                "An unselected parameter changed inside guided QKVO training."
            )
        self.handle.assert_live()


class DenseGQAAdapterOnlyOutputTrainingScope:
    """Freeze the dense target and expose only sparse factors to autograd."""

    def __init__(self, handle: InstalledDenseGQAQKVOResidualAdapter) -> None:
        if not isinstance(handle, InstalledDenseGQAQKVOResidualAdapter):
            raise DenseGQAQKVOResidualTrainingError(
                "Training scope requires a live dense QKVO handle."
            )
        self.handle = handle
        self._base_state: tuple[
            tuple[nn.Parameter, bool, Tensor | None, int, int], ...
        ] = ()
        self._module_modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise DenseGQAQKVOResidualTrainingError(
                "Output training scope cannot be re-entered."
            )
        self.handle.assert_live()
        trainable = self.handle.output_trainable_parameters()
        adapter_ids = {id(value) for value in trainable}
        if not trainable or any(not value.requires_grad for value in trainable):
            raise DenseGQAQKVOResidualTrainingError(
                "Every dense QKVO factor must be trainable."
            )
        self._base_state = tuple(
            (
                value,
                bool(value.requires_grad),
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in self.handle.model.parameters()
            if id(value) not in adapter_ids
        )
        self._module_modes = tuple(
            (module, bool(module.training)) for module in self.handle.model.modules()
        )
        for value, _requires_grad, _gradient, _version, _pointer in self._base_state:
            value.requires_grad_(False)
        self.handle.model.eval()
        self._entered = True
        return trainable

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._base_state:
            value.requires_grad_(requires_grad)
            if value.grad is not gradient:
                value.grad = gradient
        for module, training in self._module_modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            or bool(value.requires_grad) != requires_grad
            for value, requires_grad, gradient, version, pointer in self._base_state
        ):
            raise DenseGQAQKVOResidualTrainingError(
                "A dense base parameter changed inside adapter-only output training."
            )
        self.handle.assert_live()

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_GQA_QKVO_TRAINING_SCOPE_SCHEMA,
            "adapter_parameters_only": True,
            "base_parameters_frozen": True,
            "accepted_target_space": "causal-model-output-actions-or-logits",
            "internal_similarity_target_api_exposed": False,
            "optimizer_constructed": False,
            "optimizer_state_saved": False,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["training_scope_sha256"] = _canonical_sha256(payload)
        return payload


def install_dense_gqa_qkvo_residual_adapter(
    model: nn.Module,
    attention_modules: Mapping[int, nn.Module] | Sequence[nn.Module],
    *,
    geometry: DenseGQAGeometry,
    sites: Sequence[DenseGQASite],
    rank: int,
    projection_names: Mapping[str, str] | None = None,
    owner_name: str = "dense_gqa_qkvo_residual_registry",
    initialization_seed: int = 0,
) -> InstalledDenseGQAQKVOResidualAdapter:
    """Install target-native corrections around ordinary dense projections."""

    if not isinstance(model, nn.Module):
        raise DenseGQAQKVOResidualTopologyError("model must be an nn.Module.")
    if not isinstance(geometry, DenseGQAGeometry):
        raise DenseGQAQKVOResidualTopologyError("geometry must be DenseGQAGeometry.")
    if (
        not isinstance(owner_name, str)
        or not owner_name.isidentifier()
        or hasattr(model, owner_name)
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "owner_name must be an unused model attribute identifier."
        )
    site_values = _validated_sites(sites, geometry)
    names = _normalized_projection_names(projection_names)
    module_map = _attention_module_map(attention_modules, geometry)
    affected_layers = sorted({site.layer_index for site in site_values})
    if any(layer not in module_map for layer in affected_layers):
        raise DenseGQAQKVOResidualTopologyError(
            "attention_modules does not contain every affected target layer."
        )
    model_module_ids = {id(module) for module in model.modules()}
    if any(id(module_map[layer]) not in model_module_ids for layer in affected_layers):
        raise DenseGQAQKVOResidualTopologyError(
            "An affected attention module is not owned by the target model."
        )
    original: dict[tuple[int, ProjectionRole], nn.Linear] = {}
    for layer_index in affected_layers:
        attention = module_map[layer_index]
        for role in SparseDenseGQAQKVOResidualRegistry._ROLES:
            projection = getattr(attention, names[role], None)
            if type(projection) is not nn.Linear:
                raise DenseGQAQKVOResidualTopologyError(
                    f"Layer {layer_index} {role.upper()} is not an ordinary nn.Linear."
                )
            _validate_projection_shape(role, projection, geometry)
            original[(layer_index, role)] = projection
    dtype_devices = {
        (projection.weight.dtype, projection.weight.device)
        for projection in original.values()
    }
    if len(dtype_devices) != 1:
        raise DenseGQAQKVOResidualTopologyError(
            "Affected dense Q/K/V/O weights must share one dtype and device."
        )
    _assert_unaliased_bake_weights(model, original)
    dtype, device = next(iter(dtype_devices))
    # Temporary factors are optimizer state, not published model weights.  AdamW
    # keeps its moments in the parameter dtype, and an FP16 epsilon such as 1e-8
    # underflows to zero.  Keep all temporary factor parameters and optimizer
    # math in FP32 when the native child is low precision; corrections are cast
    # back to the dense projection dtype at the residual boundary and at bake.
    factor_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    registry = SparseDenseGQAQKVOResidualRegistry(
        site_values,
        geometry=geometry,
        rank=rank,
        dtype=factor_dtype,
        device=device,
        initialization_seed=initialization_seed,
    )
    if (
        not registry.starts_as_exact_zero_correction
        or not registry.input_factors_are_nonzero
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "Dense QKVO installation requires zero U and nonzero deterministic R."
        )
    projection_state = _projection_state(original)
    projection_state_hash = _projection_state_sha256(projection_state)
    before_count = _unique_parameter_count(model)
    installed: dict[tuple[int, ProjectionRole], DenseGQAQKVOResidualProjection] = {
        key: DenseGQAQKVOResidualProjection(
            projection,
            registry,
            layer_index=key[0],
            role=key[1],
        )
        for key, projection in original.items()
    }
    try:
        model.add_module(owner_name, registry)
        for (layer_index, role), view in installed.items():
            setattr(module_map[layer_index], names[role], view)
        installed_count = _unique_parameter_count(model)
        if installed_count != before_count + registry.learned_parameter_count:
            raise DenseGQAQKVOResidualAccountingError(
                "Installed dense target count is not base plus sparse factors."
            )
        accounting = DenseGQAQKVOResidualAccounting.build(
            logical_layers=geometry.logical_layers,
            hidden_size=geometry.hidden_size,
            query_heads=geometry.query_heads,
            key_value_heads=geometry.key_value_heads,
            head_dim=geometry.head_dim,
            rank=rank,
            site_count=len(site_values),
            checkpoint_element_size=torch.tensor([], dtype=factor_dtype).element_size(),
            schema_buffer_bytes=registry.schema_buffer_bytes,
        )
        if accounting.adapter_parameters != registry.learned_parameter_count:
            raise DenseGQAQKVOResidualAccountingError(
                "Live dense factors disagree with closed-form accounting."
            )
        report = DenseGQAQKVOResidualInstallationReport(
            sites=site_values,
            geometry=geometry,
            rank=_index(rank, "rank", positive=True),
            factor_dtype=str(factor_dtype),
            owner_name=owner_name,
            projection_names=dict(names),
            adapter_parameter_count=registry.learned_parameter_count,
            model_parameter_count_before=before_count,
            model_parameter_count_installed=installed_count,
            adapter_site_indices_sha256=_tensor_sha256(registry.site_indices),
            base_projection_state_sha256=projection_state_hash,
            accounting=accounting,
        )
        handle = InstalledDenseGQAQKVOResidualAdapter(
            model=model,
            attention_modules=module_map,
            adapter_registry=registry,
            report=report,
            original_projections=original,
            installed_views=installed,
            projection_state_before=projection_state,
        )
        handle.assert_live()
        return handle
    except Exception:
        for (layer_index, role), projection in original.items():
            setattr(module_map[layer_index], names[role], projection)
        if getattr(model, owner_name, None) is registry:
            delattr(model, owner_name)
        raise


def install_guided_dense_gqa_qkvo_residual_tranche(
    inherited_handle: InstalledDenseGQAQKVOResidualAdapter,
    *,
    audit_report: Mapping[str, Any],
    basis_pack: Mapping[str, Any],
    expected_audit_report_sha256: str,
    expected_basis_envelope_file_sha256: str,
    expected_basis_payload_sha256: str,
    expected_basis_manifest_sha256: str,
    owner_name: str = "guided_dense_gqa_qkvo_residual_registry",
) -> InstalledGuidedDenseGQAQKVOResidualTranche:
    """Install the authenticated exact-zero guided stage-one QKVO product.

    ``expected_basis_envelope_file_sha256`` belongs to the outer torch artifact and
    must be verified by the artifact contract before this nested payload is
    passed here.  The independently recomputed payload and tensor-manifest
    hashes prevent that external file hash from being conflated with a
    different serialization domain.
    """

    if not isinstance(inherited_handle, InstalledDenseGQAQKVOResidualAdapter):
        raise DenseGQAQKVOResidualTopologyError(
            "inherited_handle must be a live dense QKVO handle."
        )
    inherited_handle.assert_live()
    model = inherited_handle.model
    if (
        not isinstance(owner_name, str)
        or not owner_name.isidentifier()
        or hasattr(model, owner_name)
        or owner_name == inherited_handle.report.owner_name
    ):
        raise DenseGQAQKVOResidualTopologyError(
            "owner_name must be a distinct unused model attribute identifier."
        )
    (
        guided_rows,
        source_tensor_sha256s,
        report_sha256,
        envelope_file_sha256,
        payload_sha256,
        manifest_sha256,
    ) = _authenticate_guided_qkvo_sources(
        inherited_handle,
        audit_report=audit_report,
        basis_pack=basis_pack,
        expected_audit_report_sha256=expected_audit_report_sha256,
        expected_basis_envelope_file_sha256=(expected_basis_envelope_file_sha256),
        expected_basis_payload_sha256=expected_basis_payload_sha256,
        expected_basis_manifest_sha256=expected_basis_manifest_sha256,
    )
    registry = inherited_handle.adapter_registry
    inherited_parameters = tuple(registry.named_parameters(recurse=False))
    if not inherited_parameters:
        raise DenseGQAQKVOResidualTopologyError(
            "The inherited QKVO registry has no factors."
        )
    dtype_devices = {
        (value.dtype, value.device) for _name, value in inherited_parameters
    }
    if len(dtype_devices) != 1:
        raise DenseGQAQKVOResidualTopologyError(
            "Inherited QKVO factors must share one dtype and device."
        )
    factor_dtype, device = next(iter(dtype_devices))
    if factor_dtype != torch.float32:
        raise DenseGQAQKVOResidualTopologyError(
            "The exact guided stage-one tranche requires FP32 temporary factors."
        )
    inherited_factor_sha256 = _qkvo_factor_state_sha256(registry)
    inherited_requires_grad = {
        name: bool(value.requires_grad) for name, value in inherited_parameters
    }
    before_count = _unique_parameter_count(model)
    guided_registry = GuidedSparseDenseGQAQKVOResidualRegistry(
        inherited_handle.report.sites,
        geometry=inherited_handle.report.geometry,
        rank=inherited_handle.report.rank,
        guided_right_rows=guided_rows,
        source_basis_tensor_sha256s=source_tensor_sha256s,
        audit_report_sha256=report_sha256,
        basis_envelope_file_sha256=envelope_file_sha256,
        basis_payload_sha256=payload_sha256,
        basis_manifest_sha256=manifest_sha256,
        dtype=factor_dtype,
        device=device,
    )
    if (
        guided_registry.learned_parameter_count != _GUIDED_STAGE1_PARAMETER_COUNT
        or sum(
            int(getattr(guided_registry, f"{role}_right").numel())
            for role in SparseDenseGQAQKVOResidualRegistry._ROLES
        )
        != _GUIDED_STAGE1_RIGHT_PARAMETER_COUNT
    ):
        raise DenseGQAQKVOResidualAccountingError(
            "The guided stage-one tranche is not exactly 7,127,040 parameters."
        )
    attached: list[DenseGQAQKVOResidualProjection] = []
    try:
        model.add_module(owner_name, guided_registry)
        for installed in inherited_handle._installed_views.values():
            installed._attach_guided_registry(guided_registry)
            attached.append(installed)
        installed_count = _unique_parameter_count(model)
        if installed_count != before_count + _GUIDED_STAGE1_PARAMETER_COUNT:
            raise DenseGQAQKVOResidualAccountingError(
                "Guided installation count is not inherited plus 7,127,040."
            )
        if _qkvo_factor_state_sha256(registry) != inherited_factor_sha256:
            raise DenseGQAQKVOResidualTopologyError(
                "Guided installation changed the inherited rank-eight factors."
            )
        inherited_payload = inherited_handle.report.as_dict()
        report = GuidedDenseGQAQKVOResidualInstallationReport(
            sites=inherited_handle.report.sites,
            geometry=inherited_handle.report.geometry,
            rank=inherited_handle.report.rank,
            factor_dtype=str(factor_dtype),
            owner_name=owner_name,
            projection_names=dict(inherited_handle.report.projection_names),
            inherited_adapter_parameter_count=(
                inherited_handle.report.adapter_parameter_count
            ),
            guided_adapter_parameter_count=_GUIDED_STAGE1_PARAMETER_COUNT,
            adapter_parameter_count=(
                inherited_handle.report.adapter_parameter_count
                + _GUIDED_STAGE1_PARAMETER_COUNT
            ),
            model_parameter_count_before=(
                inherited_handle.report.model_parameter_count_before
            ),
            model_parameter_count_before_guided_installation=before_count,
            model_parameter_count_installed=installed_count,
            adapter_site_indices_sha256=(
                inherited_handle.report.adapter_site_indices_sha256
            ),
            audit_site_indices_sha256=str(audit_report["site_indices_sha256"]),
            base_projection_state_sha256=(
                inherited_handle.report.base_projection_state_sha256
            ),
            audit_report_sha256=report_sha256,
            basis_envelope_file_sha256=envelope_file_sha256,
            basis_payload_sha256=payload_sha256,
            basis_manifest_sha256=manifest_sha256,
            source_basis_tensor_sha256s=dict(source_tensor_sha256s),
            inherited_installation_sha256=str(inherited_payload["installation_sha256"]),
            inherited_factor_state_sha256_at_install=(inherited_factor_sha256),
            inherited_handle_identity=id(inherited_handle),
            inherited_registry_identity=id(registry),
        )
        handle = InstalledGuidedDenseGQAQKVOResidualTranche(
            inherited_handle=inherited_handle,
            guided_registry=guided_registry,
            report=report,
            inherited_requires_grad=inherited_requires_grad,
        )
        handle.assert_live()
        return handle
    except Exception:
        for installed in reversed(attached):
            if installed.guided_adapter_registry is guided_registry:
                installed._detach_guided_registry(guided_registry)
        if getattr(model, owner_name, None) is guided_registry:
            delattr(model, owner_name)
        for name, value in inherited_parameters:
            value.requires_grad_(inherited_requires_grad[name])
        if _unique_parameter_count(model) != before_count:
            raise DenseGQAQKVOResidualTopologyError(
                "Failed guided installation could not restore exact accounting."
            )
        raise


__all__ = [
    "DENSE_GQA_QKVO_ACCOUNTING_SCHEMA",
    "DENSE_GQA_QKVO_BAKE_SCHEMA",
    "DENSE_GQA_QKVO_INSTALLATION_SCHEMA",
    "DENSE_GQA_QKVO_REGISTRY_SCHEMA",
    "DENSE_GQA_QKVO_TRAINING_SCOPE_SCHEMA",
    "GUIDED_DENSE_GQA_QKVO_BAKE_SCHEMA",
    "GUIDED_DENSE_GQA_QKVO_INSTALLATION_SCHEMA",
    "GUIDED_DENSE_GQA_QKVO_REGISTRY_SCHEMA",
    "GUIDED_QKVO_RIGHT_BASIS_V2_SCHEMA",
    "QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA",
    "DenseGQAAdapterOnlyOutputTrainingScope",
    "DenseGQAGeometry",
    "DenseGQAQKVOBakeReport",
    "DenseGQAQKVOResidualAccounting",
    "DenseGQAQKVOResidualAccountingError",
    "DenseGQAQKVOResidualError",
    "DenseGQAQKVOResidualInstallationReport",
    "DenseGQAQKVOResidualProjection",
    "DenseGQAQKVOResidualTopologyError",
    "DenseGQAQKVOResidualTrainingError",
    "DenseGQASite",
    "GuidedDenseGQAQKVOBakeReport",
    "GuidedDenseGQAQKVOOutputTrainingScope",
    "GuidedDenseGQAQKVOResidualInstallationReport",
    "GuidedQKVOTrainingStage",
    "GuidedSparseDenseGQAQKVOResidualRegistry",
    "InstalledDenseGQAQKVOResidualAdapter",
    "InstalledGuidedDenseGQAQKVOResidualTranche",
    "SparseDenseGQAQKVOResidualRegistry",
    "guided_qkvo_basis_payload_sha256",
    "install_dense_gqa_qkvo_residual_adapter",
    "install_guided_dense_gqa_qkvo_residual_tranche",
]
