"""Ragged-rank assignment-local Q/K/V/O residual adapters.

Each redirected physical site owns its own rank. The registry stores no padded
max-rank tensors: a rank-4 site consumes exactly half the learned factor values
of a rank-8 site. Corrections remain pointwise and execute at the same proven
Q/K/V pre-normalization/cache and O post-softmax-context boundaries as the
uniform-rank assignment-local operator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import neural_forge.hf_assignment_local_qkvo_residual_adapter as uniform
from neural_forge.hf_attention_codebook import (
    AttentionCodebookProjectionView,
    AttentionCodebookRegistry,
)
from neural_forge.hf_functional_residual_adapter import (
    RedirectAudit,
    RedirectedAttentionSite,
    audit_redirected_attention_site,
)


RAGGED_QKVO_RESIDUAL_SCHEMA = "neural-forge-ragged-assignment-local-qkvo-residual-v1"
RAGGED_QKVO_ACCOUNTING_SCHEMA = (
    "neural-forge-ragged-assignment-local-qkvo-accounting-v1"
)
RAGGED_QKVO_INSTALLATION_SCHEMA = (
    "neural-forge-ragged-assignment-local-qkvo-installation-v1"
)


class RaggedAssignmentLocalQKVOResidualError(RuntimeError):
    """A ragged QKVO topology, accounting, or transaction is invalid."""


def _positive_int(value: Any, name: str) -> int:
    try:
        result = uniform._index(value, name, positive=True)
    except uniform.AssignmentLocalQKVOResidualError as error:
        raise RaggedAssignmentLocalQKVOResidualError(str(error)) from error
    return result


def _canonical_sha256(value: Any) -> str:
    return uniform._canonical_sha256(value)


@dataclass(frozen=True, slots=True)
class RaggedAssignmentLocalQKVOAccounting:
    site_count: int
    site_ranks: tuple[int, ...]
    rank_sum: int
    removed_parameters_per_site: int
    removed_parameters: int
    role_scale_parameters: int
    adapter_parameters: int
    net_parameters_saved: int
    checkpoint_element_size: int
    adapter_tensor_bytes: int
    net_tensor_bytes_saved: int
    per_site: tuple[Mapping[str, int], ...]

    @classmethod
    def build(
        cls,
        *,
        site_ranks: Sequence[int],
        hidden_size: int,
        query_group_width: int,
        head_dim: int,
        logical_layers: int,
        physical_groups: int,
        checkpoint_element_size: int,
        role_scale_table_new_for_redirects: bool,
    ) -> RaggedAssignmentLocalQKVOAccounting:
        ranks = tuple(_positive_int(value, "site rank") for value in site_ranks)
        if not ranks:
            raise RaggedAssignmentLocalQKVOResidualError(
                "Accounting requires at least one site rank."
            )
        hidden = _positive_int(hidden_size, "hidden_size")
        query = _positive_int(query_group_width, "query_group_width")
        head = _positive_int(head_dim, "head_dim")
        layers = _positive_int(logical_layers, "logical_layers")
        groups = _positive_int(physical_groups, "physical_groups")
        element_size = _positive_int(checkpoint_element_size, "checkpoint_element_size")
        removed_per_site = 2 * hidden * query + 2 * hidden * head
        per_site = tuple(
            MappingProxyType(
                {
                    "rank": rank,
                    "q_parameters": rank * (query + hidden),
                    "k_parameters": rank * (head + hidden),
                    "v_parameters": rank * (head + hidden),
                    "o_parameters": rank * (hidden + query),
                    "adapter_parameters": rank
                    * (2 * (query + hidden) + 2 * (head + hidden)),
                }
            )
            for rank in ranks
        )
        adapter = sum(value["adapter_parameters"] for value in per_site)
        removed = len(ranks) * removed_per_site
        role_scales = layers * groups * 4 if role_scale_table_new_for_redirects else 0
        net = removed - role_scales - adapter
        if net <= 0:
            raise RaggedAssignmentLocalQKVOResidualError(
                "Ragged adapter exceeds the removed-prototype budget."
            )
        return cls(
            site_count=len(ranks),
            site_ranks=ranks,
            rank_sum=sum(ranks),
            removed_parameters_per_site=removed_per_site,
            removed_parameters=removed,
            role_scale_parameters=role_scales,
            adapter_parameters=adapter,
            net_parameters_saved=net,
            checkpoint_element_size=element_size,
            adapter_tensor_bytes=adapter * element_size,
            net_tensor_bytes_saved=net * element_size,
            per_site=per_site,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": RAGGED_QKVO_ACCOUNTING_SCHEMA,
            "site_count": self.site_count,
            "site_ranks": list(self.site_ranks),
            "rank_sum": self.rank_sum,
            "removed_parameters_per_site": self.removed_parameters_per_site,
            "removed_parameters": self.removed_parameters,
            "role_scale_parameters": self.role_scale_parameters,
            "adapter_parameters": self.adapter_parameters,
            "net_parameters_saved": self.net_parameters_saved,
            "checkpoint_element_size": self.checkpoint_element_size,
            "adapter_tensor_bytes": self.adapter_tensor_bytes,
            "net_tensor_bytes_saved": self.net_tensor_bytes_saved,
            "per_site": [dict(value) for value in self.per_site],
            "optimizer_state_counted": False,
            "padded_to_max_rank": False,
        }
        payload["accounting_sha256"] = _canonical_sha256(payload)
        return payload


class RaggedAssignmentLocalQKVOResidualRegistry(nn.Module):
    """Sparse per-site Q/K/V/O factors with a distinct rank at every site."""

    _ROLES = ("q", "k", "v", "o")

    def __init__(
        self,
        site_indices: Tensor | Sequence[Sequence[int]],
        site_ranks: Sequence[int],
        *,
        logical_layers: int,
        physical_groups: int,
        hidden_size: int,
        query_group_width: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.logical_layers = _positive_int(logical_layers, "logical_layers")
        self.physical_groups = _positive_int(physical_groups, "physical_groups")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.query_group_width = _positive_int(query_group_width, "query_group_width")
        self.head_dim = _positive_int(head_dim, "head_dim")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise RaggedAssignmentLocalQKVOResidualError(
                "Ragged adapter dtype must be floating point."
            )
        raw_sites = torch.as_tensor(site_indices)
        if (
            raw_sites.dtype == torch.bool
            or raw_sites.is_floating_point()
            or raw_sites.is_complex()
            or raw_sites.ndim != 2
            or int(raw_sites.shape[1]) != 2
            or int(raw_sites.shape[0]) == 0
        ):
            raise RaggedAssignmentLocalQKVOResidualError(
                "site_indices must be nonempty integer [site,2]."
            )
        rows = tuple(
            tuple(map(int, row))
            for row in raw_sites.detach().cpu().to(torch.int64).tolist()
        )
        ranks = tuple(_positive_int(value, "site rank") for value in site_ranks)
        if (
            len(rows) != len(ranks)
            or list(rows) != sorted(rows)
            or len(set(rows)) != len(rows)
            or any(
                layer >= self.logical_layers or group >= self.physical_groups
                for layer, group in rows
            )
        ):
            raise RaggedAssignmentLocalQKVOResidualError(
                "Sites/ranks must align and sites must be sorted, unique, and in range."
            )
        self.register_buffer(
            "site_indices", raw_sites.detach().to(device=device, dtype=torch.int32)
        )
        self.register_buffer(
            "site_ranks", torch.tensor(ranks, dtype=torch.int32, device=device)
        )
        self.register_buffer(
            "_registry_schema_version",
            torch.tensor(1, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "_registry_layout", torch.tensor(2, dtype=torch.int32, device=device)
        )
        seed = uniform._index(initialization_seed, "initialization_seed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        widths = {
            "q": (self.query_group_width, self.hidden_size),
            "k": (self.head_dim, self.hidden_size),
            "v": (self.head_dim, self.hidden_size),
            "o": (self.hidden_size, self.query_group_width),
        }
        for role in self._ROLES:
            output_width, input_width = widths[role]
            left_values = nn.ParameterList()
            right_values = nn.ParameterList()
            for rank in ranks:
                left_values.append(
                    nn.Parameter(
                        torch.zeros((output_width, rank), dtype=dtype, device=device)
                    )
                )
                bound = 1.0 / math.sqrt(input_width)
                right = torch.empty((rank, input_width), dtype=torch.float32)
                right.uniform_(-bound, bound, generator=generator)
                right_values.append(nn.Parameter(right.to(device=device, dtype=dtype)))
            setattr(self, f"{role}_left", left_values)
            setattr(self, f"{role}_right", right_values)
        self._assert_finite()

    @property
    def site_count(self) -> int:
        return int(self.site_indices.shape[0])

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(map(int, self.site_ranks.detach().cpu().tolist()))

    @property
    def max_rank(self) -> int:
        return max(self.ranks)

    @property
    def rank_sum(self) -> int:
        return sum(self.ranks)

    @property
    def query_width(self) -> int:
        return self.physical_groups * self.query_group_width

    @property
    def learned_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    @property
    def starts_as_exact_zero_correction(self) -> bool:
        return all(
            int(torch.count_nonzero(value).item()) == 0
            for role in self._ROLES
            for value in getattr(self, f"{role}_left")
        )

    @property
    def input_factors_are_nonzero(self) -> bool:
        return all(
            int(torch.count_nonzero(value).item()) > 0
            for role in self._ROLES
            for value in getattr(self, f"{role}_right")
        )

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.parameters())

    def _assert_finite(self) -> None:
        for name, value in self.named_parameters():
            if not bool(torch.isfinite(value.detach()).all()):
                raise RaggedAssignmentLocalQKVOResidualError(
                    f"Ragged factor {name!r} contains NaN or infinity."
                )

    def _site_rows(self, layer_index: int) -> tuple[tuple[int, int], ...]:
        layer = uniform._index(layer_index, "layer_index")
        return tuple(
            (index, int(group))
            for index, (site_layer, group) in enumerate(
                self.site_indices.detach().cpu().tolist()
            )
            if int(site_layer) == layer
        )

    def _correction(self, role: str, site_index: int, inputs: Tensor) -> Tensor:
        left = getattr(self, f"{role}_left")[site_index]
        right = getattr(self, f"{role}_right")[site_index]
        return F.linear(F.linear(inputs.to(dtype=right.dtype), right), left)

    def apply_to_projection(
        self,
        role: str,
        inputs: Tensor,
        base_output: Tensor,
        *,
        layer_index: int,
    ) -> Tensor:
        if role not in {"q", "k", "v"}:
            raise RaggedAssignmentLocalQKVOResidualError("Unknown Q/K/V role.")
        role_width = self.query_group_width if role == "q" else self.head_dim
        expected_output = self.physical_groups * role_width
        if (
            inputs.ndim == 0
            or int(inputs.shape[-1]) != self.hidden_size
            or tuple(base_output.shape[:-1]) != tuple(inputs.shape[:-1])
            or int(base_output.shape[-1]) != expected_output
        ):
            raise RaggedAssignmentLocalQKVOResidualError(
                "Ragged Q/K/V projection tensors have incompatible geometry."
            )
        self._assert_finite()
        result = base_output
        for site_index, group in self._site_rows(layer_index):
            correction = self._correction(role, site_index, inputs).to(
                dtype=base_output.dtype
            )
            start = group * role_width
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
        if (
            contexts.ndim == 0
            or int(contexts.shape[-1]) != self.query_width
            or tuple(base_output.shape[:-1]) != tuple(contexts.shape[:-1])
            or int(base_output.shape[-1]) != self.hidden_size
        ):
            raise RaggedAssignmentLocalQKVOResidualError(
                "Ragged O projection tensors have incompatible geometry."
            )
        self._assert_finite()
        result = base_output
        for site_index, group in self._site_rows(layer_index):
            start = group * self.query_group_width
            stop = start + self.query_group_width
            correction = self._correction("o", site_index, contexts[..., start:stop])
            result = result + correction.to(dtype=base_output.dtype)
        return result

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": RAGGED_QKVO_RESIDUAL_SCHEMA,
            "site_indices": self.site_indices.detach().cpu().tolist(),
            "site_ranks": list(self.ranks),
            "site_count": self.site_count,
            "rank_sum": self.rank_sum,
            "max_rank": self.max_rank,
            "logical_layers": self.logical_layers,
            "physical_groups": self.physical_groups,
            "hidden_size": self.hidden_size,
            "query_group_width": self.query_group_width,
            "head_dim": self.head_dim,
            "learned_parameter_count": self.learned_parameter_count,
            "padded_to_max_rank": False,
            "zero_output_factor_initialization": (self.starts_as_exact_zero_correction),
            "random_input_factor_initialization": self.input_factors_are_nonzero,
            "output_function_training_only": True,
            "weight_similarity_used": False,
        }
        payload["registry_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class RaggedAssignmentLocalQKVOInstallationReport:
    sites: tuple[RedirectedAttentionSite, ...]
    redirect_audits: tuple[RedirectAudit, ...]
    site_ranks: tuple[int, ...]
    rank: int
    owner_name: str
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_installed: int
    attention_registry_state_sha256: str
    attention_registry_identity_sha256: str
    adapter_site_indices_sha256: str
    adapter_site_ranks_sha256: str
    accounting: RaggedAssignmentLocalQKVOAccounting

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": RAGGED_QKVO_INSTALLATION_SCHEMA,
            "sites": [value.as_dict() for value in self.sites],
            "redirect_audits": [value.as_dict() for value in self.redirect_audits],
            "site_ranks": list(self.site_ranks),
            "max_rank": self.rank,
            "owner_name": self.owner_name,
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "attention_registry_state_sha256": self.attention_registry_state_sha256,
            "attention_registry_identity_sha256": (
                self.attention_registry_identity_sha256
            ),
            "adapter_site_indices_sha256": self.adapter_site_indices_sha256,
            "adapter_site_ranks_sha256": self.adapter_site_ranks_sha256,
            "accounting": self.accounting.as_dict(),
            "correction_boundaries_identical_to_proven_single_site_operator": True,
            "checkpoint_saved": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


class InstalledRaggedAssignmentLocalQKVOResidualAdapter(
    uniform.InstalledAssignmentLocalQKVOResidualAdapter
):
    """Uniform-handle-compatible ragged installation with rank-map auditing."""

    def __init__(self, *, site_ranks_sha256: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._site_ranks_sha256 = site_ranks_sha256

    @property
    def adapter_registry(self) -> RaggedAssignmentLocalQKVOResidualRegistry:
        return self.__dict__["adapter_registry"]

    @adapter_registry.setter
    def adapter_registry(
        self, value: RaggedAssignmentLocalQKVOResidualRegistry
    ) -> None:
        self.__dict__["adapter_registry"] = value

    def _assert_rank_map(self) -> None:
        if uniform._tensor_sha256(self.adapter_registry.site_ranks) != (
            self._site_ranks_sha256
        ):
            raise RaggedAssignmentLocalQKVOResidualError(
                "Ragged site ranks changed during the live transaction."
            )

    def assert_live(self) -> RaggedAssignmentLocalQKVOInstallationReport:
        report = super().assert_live()
        self._assert_rank_map()
        return report  # type: ignore[return-value]

    def rollback(self) -> None:
        if self.is_applied:
            self._assert_rank_map()
        super().rollback()


def install_ragged_assignment_local_qkvo_residual_adapter(
    model: nn.Module,
    attention_registry: AttentionCodebookRegistry,
    *,
    sites: Sequence[RedirectedAttentionSite],
    site_ranks: Sequence[int],
    owner_name: str = "ragged_assignment_local_qkvo_residual_registry",
    initialization_seed: int = 0,
    require_qwen3_4b_dimensions: bool = False,
    role_scale_table_new_for_redirects: bool = True,
) -> InstalledRaggedAssignmentLocalQKVOResidualAdapter:
    """Install one exact ragged registry over several redirected sites."""

    if (
        not isinstance(model, nn.Module)
        or not isinstance(attention_registry, AttentionCodebookRegistry)
        or not isinstance(owner_name, str)
        or not owner_name.isidentifier()
        or hasattr(model, owner_name)
    ):
        raise RaggedAssignmentLocalQKVOResidualError(
            "Model, attention registry, and unused owner name are required."
        )
    site_values = tuple(sites)
    ranks = tuple(_positive_int(value, "site rank") for value in site_ranks)
    if (
        not site_values
        or len(site_values) != len(ranks)
        or any(not isinstance(value, RedirectedAttentionSite) for value in site_values)
        or [value.site_pair for value in site_values]
        != sorted(value.site_pair for value in site_values)
    ):
        raise RaggedAssignmentLocalQKVOResidualError(
            "Sorted redirected sites and aligned positive ranks are required."
        )
    indices = uniform._site_indices(
        site_values,
        logical_layers=attention_registry.logical_layers,
        physical_groups=attention_registry.key_value_heads,
    )
    audits = tuple(
        audit_redirected_attention_site(attention_registry, site)
        for site in site_values
    )
    query_group_width = attention_registry.query_rows_per_group
    hidden = attention_registry.output_width
    dimensions = (
        hidden,
        query_group_width,
        attention_registry.head_dim,
        attention_registry.logical_layers,
        attention_registry.key_value_heads,
    )
    if require_qwen3_4b_dimensions and dimensions != (2560, 512, 128, 36, 8):
        raise RaggedAssignmentLocalQKVOResidualError(
            "Exact Qwen3-4B ragged adapter dimensions differ."
        )
    dtype = attention_registry.prototype_q_weight.dtype
    device = attention_registry.prototype_q_weight.device
    adapter = RaggedAssignmentLocalQKVOResidualRegistry(
        indices,
        ranks,
        logical_layers=attention_registry.logical_layers,
        physical_groups=attention_registry.key_value_heads,
        hidden_size=hidden,
        query_group_width=query_group_width,
        head_dim=attention_registry.head_dim,
        dtype=dtype,
        device=device,
        initialization_seed=initialization_seed,
    )
    layers = uniform._model_layers(model)
    original: dict[
        tuple[int, uniform.ProjectionRole], AttentionCodebookProjectionView
    ] = {}
    installed: dict[
        tuple[int, uniform.ProjectionRole],
        uniform.AssignmentLocalQKVOResidualProjection,
    ] = {}
    for layer_index in sorted({site.layer_index for site in site_values}):
        attention = getattr(layers[layer_index], "self_attn", None)
        if not isinstance(attention, nn.Module):
            raise RaggedAssignmentLocalQKVOResidualError(
                "Target layer has no self-attention module."
            )
        for role in adapter._ROLES:
            projection = getattr(attention, f"{role}_proj", None)
            if not isinstance(projection, AttentionCodebookProjectionView):
                raise RaggedAssignmentLocalQKVOResidualError(
                    "Every target projection must be a live codebook view."
                )
            key = (layer_index, role)
            original[key] = projection
            installed[key] = uniform.AssignmentLocalQKVOResidualProjection(
                projection,
                adapter,  # type: ignore[arg-type]
                attention_registry,
                layer_index,
                role,
            )
    registry_hashes = uniform._state_hashes(attention_registry)
    registry_identity = uniform._tensor_identity_sha256(attention_registry)
    before_count = uniform._unique_parameter_count(model)
    try:
        model.add_module(owner_name, adapter)
        for (layer_index, role), view in installed.items():
            setattr(layers[layer_index].self_attn, f"{role}_proj", view)
        installed_count = uniform._unique_parameter_count(model)
        if installed_count != before_count + adapter.learned_parameter_count:
            raise RaggedAssignmentLocalQKVOResidualError(
                "Installed parameter count is not base plus ragged factors."
            )
        accounting = RaggedAssignmentLocalQKVOAccounting.build(
            site_ranks=ranks,
            hidden_size=hidden,
            query_group_width=query_group_width,
            head_dim=attention_registry.head_dim,
            logical_layers=attention_registry.logical_layers,
            physical_groups=attention_registry.key_value_heads,
            checkpoint_element_size=torch.tensor([], dtype=dtype).element_size(),
            role_scale_table_new_for_redirects=(role_scale_table_new_for_redirects),
        )
        if accounting.adapter_parameters != adapter.learned_parameter_count:
            raise RaggedAssignmentLocalQKVOResidualError(
                "Ragged runtime factors disagree with accounting."
            )
        rank_sha = uniform._tensor_sha256(adapter.site_ranks)
        report = RaggedAssignmentLocalQKVOInstallationReport(
            sites=site_values,
            redirect_audits=audits,
            site_ranks=ranks,
            rank=max(ranks),
            owner_name=owner_name,
            adapter_parameter_count=adapter.learned_parameter_count,
            model_parameter_count_before=before_count,
            model_parameter_count_installed=installed_count,
            attention_registry_state_sha256=uniform._state_sha256(registry_hashes),
            attention_registry_identity_sha256=registry_identity,
            adapter_site_indices_sha256=uniform._tensor_sha256(adapter.site_indices),
            adapter_site_ranks_sha256=rank_sha,
            accounting=accounting,
        )
        handle = InstalledRaggedAssignmentLocalQKVOResidualAdapter(
            model=model,
            attention_registry=attention_registry,
            adapter_registry=adapter,
            report=report,  # type: ignore[arg-type]
            original_projections=original,
            installed_views=installed,
            registry_hashes_before=registry_hashes,
            registry_identity_before=registry_identity,
            site_ranks_sha256=rank_sha,
        )
        handle.assert_live()
        return handle
    except Exception:
        for (layer_index, role), projection in original.items():
            setattr(layers[layer_index].self_attn, f"{role}_proj", projection)
        if getattr(model, owner_name, None) is adapter:
            delattr(model, owner_name)
        raise


__all__ = [
    "RAGGED_QKVO_ACCOUNTING_SCHEMA",
    "RAGGED_QKVO_INSTALLATION_SCHEMA",
    "RAGGED_QKVO_RESIDUAL_SCHEMA",
    "InstalledRaggedAssignmentLocalQKVOResidualAdapter",
    "RaggedAssignmentLocalQKVOAccounting",
    "RaggedAssignmentLocalQKVOInstallationReport",
    "RaggedAssignmentLocalQKVOResidualError",
    "RaggedAssignmentLocalQKVOResidualRegistry",
    "install_ragged_assignment_local_qkvo_residual_adapter",
]
