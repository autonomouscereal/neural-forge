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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ProjectionRole = Literal["q", "k", "v", "o"]

DENSE_GQA_QKVO_ACCOUNTING_SCHEMA = "neural-forge-dense-gqa-qkvo-residual-accounting-v1"
DENSE_GQA_QKVO_REGISTRY_SCHEMA = "neural-forge-dense-gqa-qkvo-residual-registry-v1"
DENSE_GQA_QKVO_INSTALLATION_SCHEMA = (
    "neural-forge-dense-gqa-qkvo-residual-installation-v1"
)
DENSE_GQA_QKVO_TRAINING_SCOPE_SCHEMA = (
    "neural-forge-dense-gqa-qkvo-output-training-scope-v1"
)
DENSE_GQA_QKVO_BAKE_SCHEMA = "neural-forge-dense-gqa-qkvo-bake-v1"


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


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise DenseGQAQKVOResidualTopologyError("Expected a tensor to hash.")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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

    def _assert_finite(self) -> None:
        for name, value in self.named_parameters():
            if not bool(torch.isfinite(value.detach()).all()):
                raise DenseGQAQKVOResidualError(
                    f"Adapter parameter {name!r} contains NaN or infinity."
                )

    def _site_rows(self, layer_index: int) -> tuple[tuple[int, int], ...]:
        layer = _index(layer_index, "layer_index")
        if layer >= self.geometry.logical_layers:
            raise DenseGQAQKVOResidualTopologyError(
                "layer_index is outside the target geometry."
            )
        return tuple(
            (row_index, int(group))
            for row_index, (site_layer, group) in enumerate(
                self.site_indices.detach().cpu().to(torch.int64).tolist()
            )
            if int(site_layer) == layer
        )

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
    def weight(self) -> Tensor:
        return self.base_projection.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_projection.bias

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base_projection(inputs)
        if self.role == "o":
            return self.adapter_registry.apply_to_output(
                inputs,
                base_output,
                layer_index=self.layer_index,
            )
        return self.adapter_registry.apply_to_projection(
            self.role,
            inputs,
            base_output,
            layer_index=self.layer_index,
        )


@dataclass(frozen=True, slots=True)
class DenseGQAQKVOResidualInstallationReport:
    sites: tuple[DenseGQASite, ...]
    geometry: DenseGQAGeometry
    rank: int
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

    def assert_live(self) -> DenseGQAQKVOResidualInstallationReport:
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
        self.adapter_registry._assert_finite()
        self._assert_base_unchanged()
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
    registry = SparseDenseGQAQKVOResidualRegistry(
        site_values,
        geometry=geometry,
        rank=rank,
        dtype=dtype,
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
            checkpoint_element_size=torch.tensor([], dtype=dtype).element_size(),
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


__all__ = [
    "DENSE_GQA_QKVO_ACCOUNTING_SCHEMA",
    "DENSE_GQA_QKVO_BAKE_SCHEMA",
    "DENSE_GQA_QKVO_INSTALLATION_SCHEMA",
    "DENSE_GQA_QKVO_REGISTRY_SCHEMA",
    "DENSE_GQA_QKVO_TRAINING_SCOPE_SCHEMA",
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
    "InstalledDenseGQAQKVOResidualAdapter",
    "SparseDenseGQAQKVOResidualRegistry",
    "install_dense_gqa_qkvo_residual_adapter",
]
