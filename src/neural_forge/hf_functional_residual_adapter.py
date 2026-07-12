"""Sparse, function-trained residual-write adapters for attention folds.

This module implements the smallest first-class correction that can repair one
shared attention assignment without refitting the shared survivor globally.
For one logical layer and one *physical* GQA group it computes::

    base_o_proj(z_all) + U @ (V @ z_group)

``z_group`` is the post-softmax context slice immediately before the output
projection and residual add.  The registry is sparse: it stores only active
``(layer, physical_group)`` rows, never a dense ``[layers, groups, ...]`` bank.

The factors are deliberately unrelated to parent weights.  The optional
trainer consumes only fixed parent output logits/actions on parent-greedy and
student-greedy prefixes.  It has no API for weight, activation, residual,
attention, or cache similarity targets.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_forge.hf_attention_codebook import (
    AttentionCodebookProjectionView,
    AttentionCodebookRegistry,
)


FUNCTIONAL_RESIDUAL_ADAPTER_SCHEMA = (
    "neural-forge-functional-residual-adapter-v1"
)
FUNCTIONAL_RESIDUAL_ADAPTER_ACCOUNTING_SCHEMA = (
    "neural-forge-functional-residual-adapter-accounting-v1"
)
FUNCTIONAL_RESIDUAL_ADAPTER_TRAINING_SCHEMA = (
    "neural-forge-functional-residual-adapter-training-v1"
)
FUNCTIONAL_RESIDUAL_ADAPTER_DISJOINT_TRAINING_SCHEMA = (
    "neural-forge-functional-residual-adapter-disjoint-training-v1"
)
FUNCTIONAL_RESIDUAL_ADAPTER_EVALUATION_SCHEMA = (
    "neural-forge-functional-residual-adapter-evaluation-v1"
)

FunctionalActionSplit = Literal[
    "fit", "calibration-selection", "heldout-evaluation"
]
FunctionalPrefixSource = Literal["parent-greedy", "student-greedy"]


class FunctionalResidualAdapterError(RuntimeError):
    """A sparse adapter, redirect, accounting, or functional run is invalid."""


class FunctionalResidualAdapterTopologyError(FunctionalResidualAdapterError):
    """The requested physical assignment or runtime topology is invalid."""


class FunctionalResidualAdapterAccountingError(FunctionalResidualAdapterError):
    """A claimed adapter compaction budget is invalid."""


class FunctionalResidualAdapterTrainingError(FunctionalResidualAdapterError):
    """Functional-only adapter training or evaluation is invalid."""


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise FunctionalResidualAdapterError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FunctionalResidualAdapterError(f"{name} must be an integer.") from error
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise FunctionalResidualAdapterError(f"{name} must be {qualifier}.")
    return result


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise FunctionalResidualAdapterError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FunctionalResidualAdapterError(f"{name} must be finite.") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise FunctionalResidualAdapterError(f"{name} must be {qualifier}.")
    return result


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FunctionalResidualAdapterError(
            "Adapter provenance is not canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_hashes(module: nn.Module) -> Mapping[str, str]:
    return MappingProxyType(
        {
            name: _tensor_sha256(value)
            for name, value in sorted(module.state_dict().items())
        }
    )


def _state_sha256(hashes: Mapping[str, str]) -> str:
    return _canonical_sha256(dict(hashes))


def _tensor_identity_sha256(module: nn.Module) -> str:
    payload = {
        "parameters": [
            {
                "name": name,
                "identity": id(value),
                "data_pointer": int(value.data_ptr()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "requires_grad": bool(value.requires_grad),
            }
            for name, value in module.named_parameters()
        ],
        "buffers": [
            {
                "name": name,
                "identity": id(value),
                "data_pointer": int(value.data_ptr()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            for name, value in module.named_buffers()
        ],
    }
    return _canonical_sha256(payload)


def _unique_parameter_count(module: nn.Module) -> int:
    seen: set[int] = set()
    result = 0
    for value in module.parameters():
        if id(value) not in seen:
            seen.add(id(value))
            result += int(value.numel())
    return result


def _validate_float_tensor(
    value: Tensor,
    name: str,
    shape: tuple[int, ...],
) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise FunctionalResidualAdapterError(
            f"{name} must have shape {list(shape)}."
        )
    if not value.dtype.is_floating_point:
        raise FunctionalResidualAdapterError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value.detach()).all()):
        raise FunctionalResidualAdapterError(f"{name} contains NaN or infinity.")
    return value


def _site_tensor(
    site_indices: Tensor | Sequence[Sequence[int]],
    *,
    logical_layers: int,
    physical_groups: int,
) -> Tensor:
    raw = torch.as_tensor(site_indices)
    if raw.dtype == torch.bool or raw.is_floating_point() or raw.is_complex():
        raise FunctionalResidualAdapterTopologyError(
            "site_indices must be an integer [site,2] tensor."
        )
    result = raw.detach().to(device="cpu", dtype=torch.int64)
    if result.ndim != 2 or int(result.shape[1]) != 2 or int(result.shape[0]) == 0:
        raise FunctionalResidualAdapterTopologyError(
            "site_indices must be a nonempty integer [site,2] tensor."
        )
    rows = [tuple(map(int, row)) for row in result.tolist()]
    if rows != sorted(rows) or len(rows) != len(set(rows)):
        raise FunctionalResidualAdapterTopologyError(
            "site_indices must be sorted and unique."
        )
    if any(
        layer < 0
        or layer >= logical_layers
        or group < 0
        or group >= physical_groups
        for layer, group in rows
    ):
        raise FunctionalResidualAdapterTopologyError(
            "site_indices contains an out-of-range layer or physical group."
        )
    return result.to(torch.int32)


def _validated_permutation(
    value: Tensor,
    *,
    physical_groups: int,
    name: str,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise FunctionalResidualAdapterTopologyError(
            f"{name} must be an integer physical-group permutation."
        )
    permutation = value.detach().to(device="cpu", dtype=torch.int64)
    expected = torch.arange(physical_groups, dtype=torch.int64)
    if tuple(permutation.shape) != (physical_groups,) or not torch.equal(
        torch.sort(permutation).values, expected
    ):
        raise FunctionalResidualAdapterTopologyError(
            f"{name} must be a complete physical-group bijection."
        )
    return permutation


@dataclass(frozen=True, slots=True)
class RedirectedAttentionSite:
    """One removed physical assignment redirected to one surviving prototype."""

    layer_index: int
    physical_group: int
    removed_source_flat_index: int
    survivor_layer_index: int
    survivor_physical_group: int
    survivor_source_flat_index: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _index(getattr(self, name), name))
        if (self.layer_index, self.physical_group) == (
            self.survivor_layer_index,
            self.survivor_physical_group,
        ):
            raise FunctionalResidualAdapterTopologyError(
                "Removed and surviving sites must be different."
            )
        if self.removed_source_flat_index == self.survivor_source_flat_index:
            raise FunctionalResidualAdapterTopologyError(
                "Removed and surviving provenance IDs must be different."
            )

    @property
    def site_pair(self) -> tuple[int, int]:
        return self.layer_index, self.physical_group

    @property
    def survivor_pair(self) -> tuple[int, int]:
        return self.survivor_layer_index, self.survivor_physical_group

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "physical_group": self.physical_group,
            "removed_source_flat_index": self.removed_source_flat_index,
            "survivor_layer_index": self.survivor_layer_index,
            "survivor_physical_group": self.survivor_physical_group,
            "survivor_source_flat_index": self.survivor_source_flat_index,
        }


P35_L19G4_TO_L21G6 = RedirectedAttentionSite(
    layer_index=19,
    physical_group=4,
    removed_source_flat_index=19 * 8 + 4,
    survivor_layer_index=21,
    survivor_physical_group=6,
    survivor_source_flat_index=21 * 8 + 6,
)


@dataclass(frozen=True, slots=True)
class RedirectAudit:
    site: RedirectedAttentionSite
    removed_canonical_group: int
    survivor_canonical_group: int
    shared_prototype_slot: int
    shared_prototype_source: int
    removed_source_absent: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "site": self.site.as_dict(),
            "removed_canonical_group": self.removed_canonical_group,
            "survivor_canonical_group": self.survivor_canonical_group,
            "shared_prototype_slot": self.shared_prototype_slot,
            "shared_prototype_source": self.shared_prototype_source,
            "removed_source_absent": self.removed_source_absent,
            "physical_group_mapping_verified": True,
            "redirect_provenance_verified": True,
        }


def _physical_assignment(
    registry: AttentionCodebookRegistry,
    layer_index: int,
    physical_group: int,
) -> tuple[int, int, int]:
    permutation = _validated_permutation(
        registry.group_permutations[layer_index],
        physical_groups=registry.key_value_heads,
        name=f"group_permutations[{layer_index}]",
    )
    matches = torch.nonzero(permutation == physical_group, as_tuple=False).reshape(-1)
    if int(matches.numel()) != 1:
        raise FunctionalResidualAdapterTopologyError(
            "Physical group does not map to exactly one canonical group."
        )
    canonical = int(matches.item())
    slot = int(registry.assignments[layer_index, canonical].detach().cpu().item())
    sources = registry.prototype_source_indices.detach().cpu()
    if sources.ndim == 1:
        source = int(sources[slot].item())
    elif sources.ndim == 2:
        source = int(sources[canonical, slot].item())
    else:
        raise FunctionalResidualAdapterTopologyError(
            "Prototype provenance must be rank one or two."
        )
    return canonical, slot, source


def audit_redirected_attention_site(
    registry: AttentionCodebookRegistry,
    site: RedirectedAttentionSite,
) -> RedirectAudit:
    """Prove that a physical assignment now uses the declared survivor."""

    if not isinstance(registry, AttentionCodebookRegistry):
        raise FunctionalResidualAdapterTopologyError(
            "registry must be an AttentionCodebookRegistry."
        )
    if site.layer_index >= registry.logical_layers or (
        site.survivor_layer_index >= registry.logical_layers
    ):
        raise FunctionalResidualAdapterTopologyError(
            "Redirect layer is outside the live registry."
        )
    if site.physical_group >= registry.key_value_heads or (
        site.survivor_physical_group >= registry.key_value_heads
    ):
        raise FunctionalResidualAdapterTopologyError(
            "Redirect physical group is outside the live registry."
        )
    expected_removed_source = (
        site.layer_index * registry.key_value_heads + site.physical_group
    )
    expected_survivor_source = (
        site.survivor_layer_index * registry.key_value_heads
        + site.survivor_physical_group
    )
    if (
        site.removed_source_flat_index != expected_removed_source
        or site.survivor_source_flat_index != expected_survivor_source
    ):
        raise FunctionalResidualAdapterTopologyError(
            "Redirect provenance IDs do not encode their declared physical sites."
        )
    removed_canonical, removed_slot, removed_source = _physical_assignment(
        registry, site.layer_index, site.physical_group
    )
    survivor_canonical, survivor_slot, survivor_source = _physical_assignment(
        registry, site.survivor_layer_index, site.survivor_physical_group
    )
    sources = registry.prototype_source_indices.detach().cpu().reshape(-1)
    removed_absent = not bool(
        torch.any(sources.to(torch.int64) == site.removed_source_flat_index)
    )
    if (
        removed_slot != survivor_slot
        or removed_source != survivor_source
        or survivor_source != site.survivor_source_flat_index
        or not removed_absent
    ):
        raise FunctionalResidualAdapterTopologyError(
            "Live assignments do not prove the declared removed-to-survivor redirect."
        )
    return RedirectAudit(
        site=site,
        removed_canonical_group=removed_canonical,
        survivor_canonical_group=survivor_canonical,
        shared_prototype_slot=removed_slot,
        shared_prototype_source=survivor_source,
        removed_source_absent=removed_absent,
    )


@dataclass(frozen=True, slots=True)
class FunctionalResidualAdapterAccounting:
    """Exact learned-parameter and tensor-payload accounting for one redirect."""

    hidden_size: int
    group_context_width: int
    head_dim: int
    logical_layers: int
    physical_groups: int
    rank: int
    site_count: int
    checkpoint_element_size: int
    added_scalar_gate: bool
    removed_site_weight_parameters: int
    removed_site_bias_parameters: int
    removed_site_parameters: int
    redirect_role_scale_parameters: int
    adapter_factor_parameters: int
    adapter_gate_parameters: int
    adapter_parameters: int
    net_parameters_saved_vs_incumbent: int
    removed_prototype_index_bytes: int
    adapter_site_index_bytes: int
    adapter_schema_buffer_bytes: int
    learned_payload_bytes_saved: int
    index_adjusted_payload_bytes_saved: int
    tensor_payload_bytes_saved: int
    flops_per_site_token: int

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        group_context_width: int,
        head_dim: int,
        logical_layers: int,
        physical_groups: int,
        rank: int,
        site_count: int = 1,
        checkpoint_element_size: int = 2,
        added_scalar_gate: bool = False,
        q_prototype_bias: bool = False,
        k_prototype_bias: bool = False,
        v_prototype_bias: bool = False,
        role_scale_table_new_for_redirect: bool = True,
        removed_prototype_index_bytes: int = 4,
        schema_buffer_bytes: int = 8,
    ) -> FunctionalResidualAdapterAccounting:
        values = {
            name: _index(value, name, positive=True)
            for name, value in {
                "hidden_size": hidden_size,
                "group_context_width": group_context_width,
                "head_dim": head_dim,
                "logical_layers": logical_layers,
                "physical_groups": physical_groups,
                "rank": rank,
                "site_count": site_count,
                "checkpoint_element_size": checkpoint_element_size,
            }.items()
        }
        if not all(
            isinstance(value, bool)
            for value in (
                added_scalar_gate,
                q_prototype_bias,
                k_prototype_bias,
                v_prototype_bias,
                role_scale_table_new_for_redirect,
            )
        ):
            raise FunctionalResidualAdapterAccountingError(
                "Accounting feature flags must be boolean."
            )
        removed_index_per_site = _index(
            removed_prototype_index_bytes, "removed_prototype_index_bytes"
        )
        schema_bytes = _index(schema_buffer_bytes, "schema_buffer_bytes")
        hidden = values["hidden_size"]
        group_width = values["group_context_width"]
        head = values["head_dim"]
        sites = values["site_count"]
        # One coupled physical-group prototype owns Q_group, K, V, and O_group.
        # Q/K/V prototype biases, when present, are removed with that prototype;
        # the per-layer O bias is retained and therefore is not counted here.
        removed_weights = sites * (2 * hidden * group_width + 2 * hidden * head)
        removed_biases = sites * (
            group_width * int(q_prototype_bias)
            + head * int(k_prototype_bias)
            + head * int(v_prototype_bias)
        )
        removed = removed_weights + removed_biases
        role_scales = (
            values["logical_layers"] * values["physical_groups"] * 4
            if role_scale_table_new_for_redirect
            else 0
        )
        factors = sites * values["rank"] * (hidden + group_width)
        gates = sites if added_scalar_gate else 0
        adapter = factors + gates
        net = removed - role_scales - adapter
        if net <= 0:
            raise FunctionalResidualAdapterAccountingError(
                "The adapter consumes the complete removed-prototype budget."
            )
        element_size = values["checkpoint_element_size"]
        learned_payload = net * element_size
        removed_index = sites * removed_index_per_site
        site_index_bytes = sites * 2 * torch.tensor([], dtype=torch.int32).element_size()
        index_adjusted = learned_payload + removed_index - site_index_bytes
        tensor_payload = index_adjusted - schema_bytes
        return cls(
            hidden_size=hidden,
            group_context_width=group_width,
            head_dim=head,
            logical_layers=values["logical_layers"],
            physical_groups=values["physical_groups"],
            rank=values["rank"],
            site_count=sites,
            checkpoint_element_size=element_size,
            added_scalar_gate=bool(added_scalar_gate),
            removed_site_weight_parameters=removed_weights,
            removed_site_bias_parameters=removed_biases,
            removed_site_parameters=removed,
            redirect_role_scale_parameters=role_scales,
            adapter_factor_parameters=factors,
            adapter_gate_parameters=gates,
            adapter_parameters=adapter,
            net_parameters_saved_vs_incumbent=net,
            removed_prototype_index_bytes=removed_index,
            adapter_site_index_bytes=site_index_bytes,
            adapter_schema_buffer_bytes=schema_bytes,
            learned_payload_bytes_saved=learned_payload,
            index_adjusted_payload_bytes_saved=index_adjusted,
            tensor_payload_bytes_saved=tensor_payload,
            flops_per_site_token=2 * values["rank"] * (hidden + group_width),
        )

    @classmethod
    def qwen3_4b_one_site(
        cls,
        *,
        rank: int,
        checkpoint_element_size: int = 2,
        added_scalar_gate: bool = False,
    ) -> FunctionalResidualAdapterAccounting:
        return cls.build(
            hidden_size=2560,
            group_context_width=512,
            head_dim=128,
            logical_layers=36,
            physical_groups=8,
            rank=rank,
            checkpoint_element_size=checkpoint_element_size,
            added_scalar_gate=added_scalar_gate,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FUNCTIONAL_RESIDUAL_ADAPTER_ACCOUNTING_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            "one_site_formula": (
                "2*hidden_size*(group_context_width+head_dim)"
                "+present_qkv_prototype_biases"
            ),
            "adapter_formula": (
                "site_count*rank*(hidden_size+group_context_width)"
                "+optional_site_gate"
            ),
            "artifact_header_and_manifest_delta_included": False,
            "removed_prototype_index_bytes_are_total_for_all_sites": True,
            "artifact_plan_required_before_checkpoint_claim": True,
            "prior_p36_to_p35_observed_saving_included": False,
            "full_p35_to_p34_step_hardcoded": False,
        }
        payload["accounting_sha256"] = _canonical_sha256(payload)
        return payload


class SparseFunctionalResidualAdapterRegistry(nn.Module):
    """Sparse native registry of assignment-specific ``U @ V`` corrections."""

    def __init__(
        self,
        site_indices: Tensor | Sequence[Sequence[int]],
        *,
        logical_layers: int,
        physical_groups: int,
        hidden_size: int,
        group_context_width: int,
        rank: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        left: Tensor | None = None,
        right: Tensor | None = None,
        added_scalar_gates: bool = False,
        gates: Tensor | None = None,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.logical_layers = _index(logical_layers, "logical_layers", positive=True)
        self.physical_groups = _index(
            physical_groups, "physical_groups", positive=True
        )
        self.hidden_size = _index(hidden_size, "hidden_size", positive=True)
        self.group_context_width = _index(
            group_context_width, "group_context_width", positive=True
        )
        self.rank = _index(rank, "rank", positive=True)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise FunctionalResidualAdapterError("dtype must be floating point.")
        indices = _site_tensor(
            site_indices,
            logical_layers=self.logical_layers,
            physical_groups=self.physical_groups,
        )
        self.register_buffer("site_indices", indices.to(device=device))
        self.register_buffer(
            "_registry_schema_version",
            torch.tensor(1, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "_registry_layout",
            torch.tensor(1, dtype=torch.int32, device=device),
        )
        sites = int(indices.shape[0])
        left_shape = (sites, self.hidden_size, self.rank)
        right_shape = (sites, self.rank, self.group_context_width)
        if left is None:
            left_value = torch.zeros(left_shape, dtype=dtype, device=device)
        else:
            left_value = _validate_float_tensor(left, "left", left_shape).to(
                device=device, dtype=dtype
            )
        if right is None:
            seed = _index(initialization_seed, "initialization_seed")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            bound = 1.0 / math.sqrt(self.group_context_width)
            right_value = torch.empty(right_shape, dtype=torch.float32, device="cpu")
            right_value.uniform_(-bound, bound, generator=generator)
            right_value = right_value.to(device=device, dtype=dtype)
        else:
            right_value = _validate_float_tensor(right, "right", right_shape).to(
                device=device, dtype=dtype
            )
        self.left = nn.Parameter(left_value.detach().clone())
        self.right = nn.Parameter(right_value.detach().clone())
        self.added_scalar_gates = bool(added_scalar_gates)
        if self.added_scalar_gates:
            gate_value = (
                torch.ones((sites,), dtype=dtype, device=device)
                if gates is None
                else _validate_float_tensor(gates, "gates", (sites,)).to(
                    device=device, dtype=dtype
                )
            )
            self.gates = nn.Parameter(gate_value.detach().clone())
        else:
            if gates is not None:
                raise FunctionalResidualAdapterError(
                    "gates requires added_scalar_gates=True."
                )
            self.register_parameter("gates", None)

    @property
    def site_count(self) -> int:
        return int(self.site_indices.shape[0])

    @property
    def query_width(self) -> int:
        return self.physical_groups * self.group_context_width

    @property
    def learned_parameter_count(self) -> int:
        return int(sum(value.numel() for value in self.parameters()))

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
        return bool(torch.count_nonzero(self.left.detach()).item() == 0)

    def _assert_finite(self) -> None:
        for name, value in self.named_parameters():
            if not bool(torch.isfinite(value.detach()).all()):
                raise FunctionalResidualAdapterError(
                    f"Adapter parameter {name!r} contains NaN or infinity."
                )

    def apply_to_output(
        self,
        contexts: Tensor,
        base_output: Tensor,
        *,
        layer_index: int,
        group_permutation: Tensor,
    ) -> Tensor:
        """Add sparse physical-group corrections to an existing O output."""

        layer = _index(layer_index, "layer_index")
        if layer >= self.logical_layers:
            raise FunctionalResidualAdapterTopologyError(
                "layer_index is outside the adapter registry."
            )
        _validated_permutation(
            group_permutation,
            physical_groups=self.physical_groups,
            name=f"group_permutations[{layer}]",
        )
        if not isinstance(contexts, Tensor) or contexts.ndim == 0 or (
            int(contexts.shape[-1]) != self.query_width
        ):
            raise FunctionalResidualAdapterTopologyError(
                "Post-softmax contexts have the wrong flattened physical width."
            )
        if not isinstance(base_output, Tensor) or tuple(base_output.shape[:-1]) != tuple(
            contexts.shape[:-1]
        ) or int(base_output.shape[-1]) != self.hidden_size:
            raise FunctionalResidualAdapterTopologyError(
                "Base O output is incompatible with the context tensor."
            )
        self._assert_finite()
        result = base_output
        rows = self.site_indices.detach().to(device="cpu", dtype=torch.int64).tolist()
        for row_index, (site_layer, physical_group) in enumerate(rows):
            if site_layer != layer:
                continue
            start = physical_group * self.group_context_width
            stop = start + self.group_context_width
            selected = contexts[..., start:stop].to(dtype=self.right.dtype)
            latent = F.linear(selected, self.right[row_index])
            correction = F.linear(latent, self.left[row_index])
            if self.gates is not None:
                correction = correction * self.gates[row_index]
            result = result + correction.to(dtype=base_output.dtype)
        return result

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FUNCTIONAL_RESIDUAL_ADAPTER_SCHEMA,
            "site_indices": self.site_indices.detach().cpu().tolist(),
            "logical_layers": self.logical_layers,
            "physical_groups": self.physical_groups,
            "hidden_size": self.hidden_size,
            "group_context_width": self.group_context_width,
            "query_width": self.query_width,
            "rank": self.rank,
            "site_count": self.site_count,
            "added_scalar_gates": self.added_scalar_gates,
            "learned_parameter_count": self.learned_parameter_count,
            "factor_parameter_count": int(self.left.numel() + self.right.numel()),
            "gate_parameter_count": 0 if self.gates is None else int(self.gates.numel()),
            "index_buffer_bytes": self.index_buffer_bytes,
            "schema_buffer_bytes": self.schema_buffer_bytes,
            "sparse_sites_only": True,
            "bias_parameters": 0,
            "parent_runtime_references": 0,
            "zero_correction_initialization": self.starts_as_exact_zero_correction,
            "correction_boundary": (
                "post-softmax-physical-group-context-before-residual-add"
            ),
            "excluded_objectives": [
                "weight-similarity",
                "activation-similarity",
                "attention-similarity",
                "residual-similarity",
                "cache-similarity",
            ],
        }
        payload["registry_sha256"] = _canonical_sha256(payload)
        return payload


class FunctionalResidualAdapterOutputProjection(nn.Module):
    """Parameterless view that adds one sparse registry after the shared O."""

    def __init__(
        self,
        base_projection: AttentionCodebookProjectionView,
        adapter_registry: SparseFunctionalResidualAdapterRegistry,
        attention_registry: AttentionCodebookRegistry,
        layer_index: int,
    ) -> None:
        super().__init__()
        if not isinstance(base_projection, AttentionCodebookProjectionView) or (
            base_projection.registry is not attention_registry
            or base_projection.role != "o"
            or base_projection.layer_index != layer_index
        ):
            raise FunctionalResidualAdapterTopologyError(
                "Base projection must be the live parameterless codebook O view."
            )
        object.__setattr__(self, "_adapter_registry", adapter_registry)
        object.__setattr__(self, "_attention_registry", attention_registry)
        self.base_projection = base_projection
        self.layer_index = _index(layer_index, "layer_index")
        self.in_features = base_projection.in_features
        self.out_features = base_projection.out_features
        self.train(base_projection.training)

    @property
    def adapter_registry(self) -> SparseFunctionalResidualAdapterRegistry:
        return self._adapter_registry

    @property
    def attention_registry(self) -> AttentionCodebookRegistry:
        return self._attention_registry

    @property
    def weight(self) -> Tensor:
        return self.base_projection.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_projection.bias

    def forward(self, contexts: Tensor) -> Tensor:
        base_output = self.base_projection(contexts)
        return self.adapter_registry.apply_to_output(
            contexts,
            base_output,
            layer_index=self.layer_index,
            group_permutation=self.attention_registry.group_permutations[
                self.layer_index
            ],
        )


@dataclass(frozen=True, slots=True)
class FunctionalResidualAdapterInstallationReport:
    site: RedirectedAttentionSite
    redirect_audit: RedirectAudit
    rank: int
    owner_name: str
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_installed: int
    attention_registry_state_sha256: str
    attention_registry_identity_sha256: str
    adapter_site_indices_sha256: str
    accounting: FunctionalResidualAdapterAccounting

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FUNCTIONAL_RESIDUAL_ADAPTER_SCHEMA,
            "site": self.site.as_dict(),
            "redirect_audit": self.redirect_audit.as_dict(),
            "rank": self.rank,
            "owner_name": self.owner_name,
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "attention_registry_state_sha256": self.attention_registry_state_sha256,
            "attention_registry_identity_sha256": (
                self.attention_registry_identity_sha256
            ),
            "adapter_site_indices_sha256": self.adapter_site_indices_sha256,
            "accounting": self.accounting.as_dict(),
            "base_projection_reused": True,
            "shared_survivor_globally_refit": False,
            "zero_initialization_preserves_tentative_child": True,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


def _model_layers(model: nn.Module) -> Sequence[nn.Module]:
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or not layers:
        raise FunctionalResidualAdapterTopologyError(
            "Model must expose model.layers or layers."
        )
    return layers


class InstalledFunctionalResidualAdapter:
    """A live, uncommittable experimental adapter with exact rejection rollback."""

    def __init__(
        self,
        *,
        model: nn.Module,
        attention_registry: AttentionCodebookRegistry,
        adapter_registry: SparseFunctionalResidualAdapterRegistry,
        report: FunctionalResidualAdapterInstallationReport,
        original_projections: Mapping[int, AttentionCodebookProjectionView],
        installed_views: Mapping[int, FunctionalResidualAdapterOutputProjection],
        registry_hashes_before: Mapping[str, str],
        registry_identity_before: str,
    ) -> None:
        self.model = model
        self.attention_registry = attention_registry
        self.adapter_registry = adapter_registry
        self.report = report
        self._original_projections = dict(original_projections)
        self._installed_views = dict(installed_views)
        self._registry_hashes_before = dict(registry_hashes_before)
        self._registry_identity_before = registry_identity_before
        self._rolled_back = False

    @property
    def is_applied(self) -> bool:
        return not self._rolled_back

    def assert_base_unchanged(self) -> None:
        if (
            dict(_state_hashes(self.attention_registry))
            != self._registry_hashes_before
            or _tensor_identity_sha256(self.attention_registry)
            != self._registry_identity_before
        ):
            raise FunctionalResidualAdapterTopologyError(
                "Base attention registry changed during the adapter experiment."
            )

    def _assert_unique_registry_owner(self) -> None:
        owner_paths = [
            name
            for name, module in self.model.named_modules(remove_duplicate=False)
            if module is self.adapter_registry
        ]
        if owner_paths != [self.report.owner_name]:
            raise FunctionalResidualAdapterTopologyError(
                "Sparse adapter registry must have exactly one registered owner."
            )

    def assert_live(self) -> FunctionalResidualAdapterInstallationReport:
        if self._rolled_back:
            raise FunctionalResidualAdapterTopologyError(
                "A rolled-back adapter is no longer live."
            )
        if getattr(self.model, self.report.owner_name, None) is not self.adapter_registry:
            raise FunctionalResidualAdapterTopologyError(
                "Model lost the unique sparse adapter registry owner."
            )
        self._assert_unique_registry_owner()
        layers = _model_layers(self.model)
        for layer_index, installed in self._installed_views.items():
            attention = getattr(layers[layer_index], "self_attn", None)
            if getattr(attention, "o_proj", None) is not installed:
                raise FunctionalResidualAdapterTopologyError(
                    "A live adapter output view was replaced."
                )
        if _tensor_sha256(self.adapter_registry.site_indices) != (
            self.report.adapter_site_indices_sha256
        ):
            raise FunctionalResidualAdapterTopologyError(
                "Sparse adapter site indices changed."
            )
        self.adapter_registry._assert_finite()
        self.assert_base_unchanged()
        return self.report

    def rollback(self) -> None:
        if self._rolled_back:
            return
        layers = _model_layers(self.model)
        owner = getattr(self.model, self.report.owner_name, None)
        if owner is not None and owner is not self.adapter_registry:
            raise FunctionalResidualAdapterTopologyError(
                "Adapter registry owner was replaced; exact rollback is unsafe."
            )
        if owner is self.adapter_registry:
            self._assert_unique_registry_owner()
        for layer_index, installed in self._installed_views.items():
            attention = getattr(layers[layer_index], "self_attn", None)
            if getattr(attention, "o_proj", None) is not installed:
                raise FunctionalResidualAdapterTopologyError(
                    "Adapter output view was replaced; exact rollback is unsafe."
                )
        self.assert_base_unchanged()
        # Adapter values may be trained or even corrupted.  They are isolated,
        # so rejection removes them rather than requiring their initial value.
        for layer_index, original in self._original_projections.items():
            attention = getattr(layers[layer_index], "self_attn", None)
            if not isinstance(attention, nn.Module):
                raise FunctionalResidualAdapterTopologyError(
                    "Cannot restore a missing self-attention module."
                )
            attention.o_proj = original
        if getattr(self.model, self.report.owner_name, None) is self.adapter_registry:
            delattr(self.model, self.report.owner_name)
        elif hasattr(self.model, self.report.owner_name):
            raise FunctionalResidualAdapterTopologyError(
                "Adapter registry owner changed during exact rollback."
            )
        self.assert_base_unchanged()
        if _unique_parameter_count(self.model) != self.report.model_parameter_count_before:
            raise FunctionalResidualAdapterTopologyError(
                "Exact adapter rollback parameter accounting failed."
            )
        for layer_index, original in self._original_projections.items():
            attention = getattr(layers[layer_index], "self_attn", None)
            if getattr(attention, "o_proj", None) is not original:
                raise FunctionalResidualAdapterTopologyError(
                    "Exact adapter projection rollback failed."
                )
        self._rolled_back = True

    def __enter__(self) -> InstalledFunctionalResidualAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.rollback()


def install_functional_residual_adapter(
    model: nn.Module,
    attention_registry: AttentionCodebookRegistry,
    *,
    site: RedirectedAttentionSite,
    rank: int,
    owner_name: str = "functional_residual_adapter_registry",
    added_scalar_gate: bool = False,
    initialization_seed: int = 0,
    require_qwen3_4b_dimensions: bool = False,
) -> InstalledFunctionalResidualAdapter:
    """Install one sparse adapter after auditing an exact physical redirect."""

    if not isinstance(model, nn.Module):
        raise FunctionalResidualAdapterTopologyError("model must be an nn.Module.")
    if not isinstance(owner_name, str) or not owner_name.isidentifier() or (
        hasattr(model, owner_name)
    ):
        raise FunctionalResidualAdapterTopologyError(
            "owner_name must be one unused Python identifier."
        )
    redirect = audit_redirected_attention_site(attention_registry, site)
    if attention_registry.query_heads % attention_registry.key_value_heads:
        raise FunctionalResidualAdapterTopologyError(
            "Query heads must divide evenly across physical KV groups."
        )
    group_width = (
        attention_registry.query_heads
        // attention_registry.key_value_heads
        * attention_registry.head_dim
    )
    hidden = attention_registry.output_width
    if attention_registry.query_width != (
        attention_registry.key_value_heads * group_width
    ):
        raise FunctionalResidualAdapterTopologyError(
            "Flattened O input cannot be partitioned into physical GQA groups."
        )
    if require_qwen3_4b_dimensions and (
        hidden,
        group_width,
        attention_registry.head_dim,
        attention_registry.logical_layers,
        attention_registry.key_value_heads,
    ) != (2560, 512, 128, 36, 8):
        raise FunctionalResidualAdapterTopologyError(
            "Exact Qwen3-4B adapter requires h=2560,m=512,d=128,L=36,G=8."
        )
    dtype = attention_registry.prototype_q_weight.dtype
    device = attention_registry.prototype_q_weight.device
    adapter = SparseFunctionalResidualAdapterRegistry(
        [site.site_pair],
        logical_layers=attention_registry.logical_layers,
        physical_groups=attention_registry.key_value_heads,
        hidden_size=hidden,
        group_context_width=group_width,
        rank=rank,
        dtype=dtype,
        device=device,
        added_scalar_gates=added_scalar_gate,
        initialization_seed=initialization_seed,
    )
    if not adapter.starts_as_exact_zero_correction:
        raise FunctionalResidualAdapterTopologyError(
            "New adapter must start as an exact zero correction."
        )
    layers = _model_layers(model)
    if len(layers) != attention_registry.logical_layers:
        raise FunctionalResidualAdapterTopologyError(
            "Model layer count disagrees with the attention registry."
        )
    original: dict[int, AttentionCodebookProjectionView] = {}
    installed: dict[int, FunctionalResidualAdapterOutputProjection] = {}
    target_layers = sorted({int(row[0]) for row in adapter.site_indices.tolist()})
    for layer_index in target_layers:
        attention = getattr(layers[layer_index], "self_attn", None)
        projection = getattr(attention, "o_proj", None)
        if not isinstance(projection, AttentionCodebookProjectionView):
            raise FunctionalResidualAdapterTopologyError(
                "Target layer O projection is not a parameterless codebook view."
            )
        original[layer_index] = projection
        installed[layer_index] = FunctionalResidualAdapterOutputProjection(
            projection,
            adapter,
            attention_registry,
            layer_index,
        )
    registry_hashes = _state_hashes(attention_registry)
    registry_identity = _tensor_identity_sha256(attention_registry)
    before_count = _unique_parameter_count(model)
    try:
        model.add_module(owner_name, adapter)
        for layer_index, view in installed.items():
            layers[layer_index].self_attn.o_proj = view
        installed_count = _unique_parameter_count(model)
        if installed_count != before_count + adapter.learned_parameter_count:
            raise FunctionalResidualAdapterTopologyError(
                "Installed model parameter count does not equal base plus sparse factors."
            )
        accounting = FunctionalResidualAdapterAccounting.build(
            hidden_size=hidden,
            group_context_width=group_width,
            head_dim=attention_registry.head_dim,
            logical_layers=attention_registry.logical_layers,
            physical_groups=attention_registry.key_value_heads,
            rank=rank,
            checkpoint_element_size=torch.tensor([], dtype=dtype).element_size(),
            added_scalar_gate=added_scalar_gate,
            q_prototype_bias=attention_registry.prototype_q_bias is not None,
            k_prototype_bias=attention_registry.prototype_k_bias is not None,
            v_prototype_bias=attention_registry.prototype_v_bias is not None,
            role_scale_table_new_for_redirect=True,
            schema_buffer_bytes=adapter.schema_buffer_bytes,
        )
        if accounting.adapter_parameters != adapter.learned_parameter_count:
            raise FunctionalResidualAdapterAccountingError(
                "Runtime sparse parameters disagree with the closed-form accounting."
            )
        report = FunctionalResidualAdapterInstallationReport(
            site=site,
            redirect_audit=redirect,
            rank=_index(rank, "rank", positive=True),
            owner_name=owner_name,
            adapter_parameter_count=adapter.learned_parameter_count,
            model_parameter_count_before=before_count,
            model_parameter_count_installed=installed_count,
            attention_registry_state_sha256=_state_sha256(registry_hashes),
            attention_registry_identity_sha256=registry_identity,
            adapter_site_indices_sha256=_tensor_sha256(adapter.site_indices),
            accounting=accounting,
        )
        handle = InstalledFunctionalResidualAdapter(
            model=model,
            attention_registry=attention_registry,
            adapter_registry=adapter,
            report=report,
            original_projections=original,
            installed_views=installed,
            registry_hashes_before=registry_hashes,
            registry_identity_before=registry_identity,
        )
        handle.assert_live()
        return handle
    except Exception:
        for layer_index, projection in original.items():
            layers[layer_index].self_attn.o_proj = projection
        if getattr(model, owner_name, None) is adapter:
            delattr(model, owner_name)
        raise


def install_p35_l19g4_to_l21g6_functional_residual_adapter(
    model: nn.Module,
    attention_registry: AttentionCodebookRegistry,
    *,
    rank: int = 8,
    owner_name: str = "functional_residual_adapter_registry",
    added_scalar_gate: bool = False,
    initialization_seed: int = 0,
) -> InstalledFunctionalResidualAdapter:
    """Install the exact production Qwen p35 L19/G4 -> L21/G6 adapter."""

    return install_functional_residual_adapter(
        model,
        attention_registry,
        site=P35_L19G4_TO_L21G6,
        rank=rank,
        owner_name=owner_name,
        added_scalar_gate=added_scalar_gate,
        initialization_seed=initialization_seed,
        require_qwen3_4b_dimensions=True,
    )


@dataclass(frozen=True, slots=True)
class FunctionalActionBatch:
    """Fixed parent output/action targets on one family of rollout prefixes."""

    sequence_id: str
    split: FunctionalActionSplit
    prefix_source: FunctionalPrefixSource
    input_ids: Tensor
    attention_mask: Tensor
    parent_action_ids: Tensor
    parent_action_logits: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise FunctionalResidualAdapterTrainingError(
                "sequence_id must be nonempty."
            )
        if self.split not in {
            "fit",
            "calibration-selection",
            "heldout-evaluation",
        }:
            raise FunctionalResidualAdapterTrainingError("Unknown functional split.")
        if self.prefix_source not in {"parent-greedy", "student-greedy"}:
            raise FunctionalResidualAdapterTrainingError("Unknown prefix source.")
        inputs = torch.as_tensor(self.input_ids)
        mask = torch.as_tensor(self.attention_mask)
        actions = torch.as_tensor(self.parent_action_ids)
        logits = torch.as_tensor(self.parent_action_logits)
        if (
            inputs.dtype == torch.bool
            or inputs.is_floating_point()
            or inputs.is_complex()
            or inputs.ndim != 2
            or int(inputs.shape[0]) == 0
            or int(inputs.shape[1]) == 0
            or int(inputs.min()) < 0
        ):
            raise FunctionalResidualAdapterTrainingError(
                "input_ids must be a nonempty integer [batch,sequence] tensor."
            )
        if tuple(mask.shape) != tuple(inputs.shape) or mask.is_complex():
            raise FunctionalResidualAdapterTrainingError(
                "attention_mask must match input_ids."
            )
        if not bool(torch.isfinite(mask).all()):
            raise FunctionalResidualAdapterTrainingError(
                "attention_mask must contain only finite zero/one values."
            )
        mask01 = mask.to(torch.int64)
        if (
            not torch.equal(mask, mask01.to(device=mask.device, dtype=mask.dtype))
            or not torch.equal(mask01, mask01.clamp(0, 1))
            or bool(
            torch.any(mask01[:, 1:] > mask01[:, :-1])
            )
            or bool(torch.any(mask01.sum(dim=1) == 0))
        ):
            raise FunctionalResidualAdapterTrainingError(
                "attention_mask rows must be nonempty one-prefix then zero-padding."
            )
        if (
            actions.dtype == torch.bool
            or actions.is_floating_point()
            or actions.is_complex()
            or tuple(actions.shape) != (int(inputs.shape[0]),)
        ):
            raise FunctionalResidualAdapterTrainingError(
                "parent_action_ids must be one integer per prefix."
            )
        if (
            logits.ndim != 2
            or int(logits.shape[0]) != int(inputs.shape[0])
            or int(logits.shape[1]) < 2
            or not logits.dtype.is_floating_point
            or not bool(torch.isfinite(logits.detach()).all())
        ):
            raise FunctionalResidualAdapterTrainingError(
                "parent_action_logits must be finite [batch,vocabulary] outputs."
            )
        action64 = actions.to(torch.int64)
        if int(action64.min()) < 0 or int(action64.max()) >= int(logits.shape[1]):
            raise FunctionalResidualAdapterTrainingError(
                "parent_action_ids contains an out-of-vocabulary action."
            )
        if not torch.equal(action64.cpu(), logits.argmax(dim=-1).to(torch.int64).cpu()):
            raise FunctionalResidualAdapterTrainingError(
                "Fixed parent action IDs must be parent-logit top1 actions."
            )
        object.__setattr__(self, "input_ids", inputs.detach().clone().to(torch.long))
        object.__setattr__(self, "attention_mask", mask01.detach().clone())
        object.__setattr__(self, "parent_action_ids", action64.detach().clone())
        object.__setattr__(self, "parent_action_logits", logits.detach().clone())

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def vocabulary_size(self) -> int:
        return int(self.parent_action_logits.shape[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "split": self.split,
            "prefix_source": self.prefix_source,
            "batch_size": self.batch_size,
            "prefix_width": int(self.input_ids.shape[1]),
            "vocabulary_size": self.vocabulary_size,
            "input_ids_sha256": _tensor_sha256(self.input_ids),
            "attention_mask_sha256": _tensor_sha256(self.attention_mask),
            "parent_action_ids_sha256": _tensor_sha256(self.parent_action_ids),
            "parent_action_logits_sha256": _tensor_sha256(
                self.parent_action_logits
            ),
            "parent_top1_anchor_verified": True,
            "output_function_only": True,
        }


def _forward_kwargs(model: nn.Module, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return values
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {name: value for name, value in values.items() if name in signature.parameters}


def _model_device(model: nn.Module) -> torch.device:
    embeddings = getattr(model, "get_input_embeddings", None)
    if callable(embeddings):
        module = embeddings()
        weight = getattr(module, "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    for value in (*model.parameters(), *model.buffers()):
        return value.device
    raise FunctionalResidualAdapterTrainingError("model has no tensor device.")


def _action_logits(model: nn.Module, batch: FunctionalActionBatch) -> Tensor:
    device = _model_device(model)
    original_positions = batch.attention_mask.sum(dim=1) - 1
    input_width = int(batch.input_ids.shape[1])
    # Qwen can project only the shortest trailing suffix that contains every
    # row's final real token.  For adjacent batch-2 rollout prefixes this is
    # normally two positions instead of [sequence,vocabulary] for the entire
    # prompt.  Models without logits_to_keep simply receive the full-output
    # call after signature filtering.
    logits_to_keep = input_width - int(original_positions.min().item())
    values = {
        "input_ids": batch.input_ids.to(device=device),
        "attention_mask": batch.attention_mask.to(device=device),
        "use_cache": False,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "logits_to_keep": logits_to_keep,
    }
    output = model(**_forward_kwargs(model, values))
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or int(logits.shape[0]) != batch.batch_size
        or int(logits.shape[-1]) != batch.vocabulary_size
        or not logits.dtype.is_floating_point
        or not bool(torch.isfinite(logits.detach()).all())
    ):
        raise FunctionalResidualAdapterTrainingError(
            "Model must return finite [batch,sequence,vocabulary] logits."
        )
    positions = original_positions.to(device=logits.device)
    output_width = int(logits.shape[1])
    if output_width < input_width:
        suffix_start = input_width - output_width
        if bool(torch.any(positions < suffix_start)):
            raise FunctionalResidualAdapterTrainingError(
                "Short model output omitted one row's final action position."
            )
        positions = positions - suffix_start
    elif output_width > input_width:
        raise FunctionalResidualAdapterTrainingError(
            "Model output sequence is wider than its action-batch input."
        )
    row = torch.arange(batch.batch_size, device=logits.device)
    return logits[row, positions]


def _losses_from_action_logits(
    student: Tensor,
    batch: FunctionalActionBatch,
    *,
    hard_action_weight: float,
    distribution_weight: float,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    student = student.to(torch.float32)
    actions = batch.parent_action_ids.to(device=student.device)
    parent = batch.parent_action_logits.to(device=student.device, dtype=torch.float32)
    hard = F.cross_entropy(student, actions)
    log_student = F.log_softmax(student / temperature, dim=-1)
    parent_probabilities = F.softmax(parent / temperature, dim=-1)
    distribution = F.kl_div(
        log_student,
        parent_probabilities,
        reduction="batchmean",
    ) * (temperature * temperature)
    total = hard_action_weight * hard + distribution_weight * distribution
    if not bool(torch.isfinite(total.detach())):
        raise FunctionalResidualAdapterTrainingError(
            "Functional action loss became non-finite."
        )
    return total, hard, distribution


def _functional_losses(
    model: nn.Module,
    batch: FunctionalActionBatch,
    *,
    hard_action_weight: float,
    distribution_weight: float,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return _losses_from_action_logits(
        _action_logits(model, batch),
        batch,
        hard_action_weight=hard_action_weight,
        distribution_weight=distribution_weight,
        temperature=temperature,
    )


@dataclass(frozen=True, slots=True)
class FunctionalActionEvaluation:
    split: FunctionalActionSplit
    example_count: int
    total_loss: float
    hard_action_loss: float
    distribution_loss: float
    parent_action_top1_rate: float
    mean_parent_action_log_probability: float
    mean_parent_action_rank: float
    used_for_selection: bool
    prefix_sources: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FUNCTIONAL_RESIDUAL_ADAPTER_EVALUATION_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            "functional_outputs_only": True,
            "attention_residual_cache_weight_similarity_used": False,
        }
        payload["evaluation_sha256"] = _canonical_sha256(payload)
        return payload


def evaluate_functional_action_batches(
    model: nn.Module,
    batches: Sequence[FunctionalActionBatch],
    *,
    expected_split: FunctionalActionSplit,
    hard_action_weight: float = 1.0,
    distribution_weight: float = 1.0,
    temperature: float = 1.0,
    used_for_selection: bool = False,
) -> FunctionalActionEvaluation:
    """Evaluate actual output actions; heldout can never select a candidate."""

    if not batches or any(batch.split != expected_split for batch in batches):
        raise FunctionalResidualAdapterTrainingError(
            "Every evaluation batch must belong to expected_split."
        )
    if used_for_selection and expected_split != "calibration-selection":
        raise FunctionalResidualAdapterTrainingError(
            "Only calibration-selection evaluation may select a candidate."
        )
    hard_weight = _finite_float(hard_action_weight, "hard_action_weight")
    distribution_weight = _finite_float(
        distribution_weight, "distribution_weight"
    )
    temperature = _finite_float(temperature, "temperature", positive=True)
    if hard_weight < 0.0 or distribution_weight < 0.0 or (
        hard_weight + distribution_weight <= 0.0
    ):
        raise FunctionalResidualAdapterTrainingError(
            "At least one nonnegative output loss weight must be positive."
        )
    totals = {"count": 0, "total": 0.0, "hard": 0.0, "distribution": 0.0}
    top1 = 0
    log_probability = 0.0
    rank = 0.0
    module_modes = tuple((module, bool(module.training)) for module in model.modules())
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                student = _action_logits(model, batch).to(torch.float32)
                total, hard, distribution = _losses_from_action_logits(
                    student,
                    batch,
                    hard_action_weight=hard_weight,
                    distribution_weight=distribution_weight,
                    temperature=temperature,
                )
                actions = batch.parent_action_ids.to(device=student.device)
                selected = student.gather(1, actions[:, None]).squeeze(1)
                totals["count"] += batch.batch_size
                totals["total"] += float(total.item()) * batch.batch_size
                totals["hard"] += float(hard.item()) * batch.batch_size
                totals["distribution"] += float(distribution.item()) * batch.batch_size
                top1 += int((student.argmax(dim=-1) == actions).sum().item())
                log_probability += float(
                    F.log_softmax(student, dim=-1)
                    .gather(1, actions[:, None])
                    .sum()
                    .item()
                )
                rank += float(
                    (1 + (student > selected[:, None]).sum(dim=-1)).sum().item()
                )
    finally:
        for module, training in module_modes:
            module.training = training
    count = int(totals["count"])
    return FunctionalActionEvaluation(
        split=expected_split,
        example_count=count,
        total_loss=totals["total"] / count,
        hard_action_loss=totals["hard"] / count,
        distribution_loss=totals["distribution"] / count,
        parent_action_top1_rate=top1 / count,
        mean_parent_action_log_probability=log_probability / count,
        mean_parent_action_rank=rank / count,
        used_for_selection=bool(used_for_selection),
        prefix_sources=tuple(sorted({batch.prefix_source for batch in batches})),
    )


@dataclass(frozen=True, slots=True)
class FunctionalResidualAdapterTrainingReport:
    steps_requested: int
    best_step: int
    batch_count: int
    example_count: int
    parent_greedy_examples: int
    student_greedy_examples: int
    learning_rate: float
    hard_action_weight: float
    distribution_weight: float
    temperature: float
    initial_calibration_loss: float
    best_calibration_loss: float
    final_calibration_loss: float
    calibration_evaluations: int
    autograd_calls: int
    optimizer_constructed: bool
    heldout_evaluation_calls: int
    adapter_state_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FUNCTIONAL_RESIDUAL_ADAPTER_TRAINING_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            "selection_split": "calibration-selection",
            "fixed_parent_outputs_only": True,
            "fixed_parent_and_student_rollout_prefixes": True,
            "heldout_used_for_fitting": False,
            "heldout_used_for_selection": False,
            "base_parameters_updated": 0,
            "learned_compaction_not_fold_only": True,
            "checkpoint_saved": False,
            "candidate_committed": False,
            "excluded_objectives": [
                "weight-similarity",
                "activation-similarity",
                "attention-similarity",
                "residual-similarity",
                "cache-similarity",
            ],
        }
        payload["training_sha256"] = _canonical_sha256(payload)
        return payload


def train_functional_residual_adapter(
    handle: InstalledFunctionalResidualAdapter,
    batches: Sequence[FunctionalActionBatch],
    *,
    steps: int,
    learning_rate: float = 1e-2,
    hard_action_weight: float = 1.0,
    distribution_weight: float = 1.0,
    temperature: float = 1.0,
    max_gradient_norm: float = 10.0,
) -> FunctionalResidualAdapterTrainingReport:
    """Train/select sparse factors from calibration output actions only.

    The best in-memory factor state is selected on the fixed calibration set.
    No optimizer object or optimizer state is constructed, and no checkpoint is
    written or committed.  Heldout batches are rejected at this boundary.
    """

    if not isinstance(handle, InstalledFunctionalResidualAdapter):
        raise FunctionalResidualAdapterTrainingError(
            "handle must be an InstalledFunctionalResidualAdapter."
        )
    handle.assert_live()
    step_count = _index(steps, "steps", positive=True)
    if step_count > 100_000:
        raise FunctionalResidualAdapterTrainingError("steps exceeds the hard bound.")
    learning_rate = _finite_float(learning_rate, "learning_rate", positive=True)
    hard_weight = _finite_float(hard_action_weight, "hard_action_weight")
    distribution_weight = _finite_float(
        distribution_weight, "distribution_weight"
    )
    temperature = _finite_float(temperature, "temperature", positive=True)
    max_gradient_norm = _finite_float(
        max_gradient_norm, "max_gradient_norm", positive=True
    )
    if hard_weight < 0.0 or distribution_weight < 0.0 or (
        hard_weight + distribution_weight <= 0.0
    ):
        raise FunctionalResidualAdapterTrainingError(
            "At least one nonnegative output loss weight must be positive."
        )
    if not batches or any(batch.split != "calibration-selection" for batch in batches):
        raise FunctionalResidualAdapterTrainingError(
            "Training and candidate selection accept calibration batches only."
        )
    sources = {batch.prefix_source for batch in batches}
    if sources != {"parent-greedy", "student-greedy"}:
        raise FunctionalResidualAdapterTrainingError(
            "Calibration must include fixed parent-greedy and student-greedy prefixes."
        )
    if len({batch.vocabulary_size for batch in batches}) != 1:
        raise FunctionalResidualAdapterTrainingError(
            "Calibration batches disagree on vocabulary size."
        )
    model = handle.model
    adapter = handle.adapter_registry
    parameters = tuple(adapter.parameters())
    if not parameters or any(not value.requires_grad for value in parameters):
        raise FunctionalResidualAdapterTrainingError(
            "Every sparse adapter factor must be trainable."
        )
    module_modes = tuple((module, bool(module.training)) for module in model.modules())
    model.eval()
    try:
        initial = evaluate_functional_action_batches(
            model,
            batches,
            expected_split="calibration-selection",
            hard_action_weight=hard_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=True,
        )
        best_loss = initial.total_loss
        best_step = 0
        best_state = tuple(value.detach().cpu().clone() for value in parameters)
        evaluations = 1
        for step in range(1, step_count + 1):
            batch = batches[(step - 1) % len(batches)]
            total, _hard, _distribution = _functional_losses(
                model,
                batch,
                hard_action_weight=hard_weight,
                distribution_weight=distribution_weight,
                temperature=temperature,
            )
            gradients = torch.autograd.grad(
                total,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            norm = torch.sqrt(
                sum(
                    torch.sum(gradient.detach().to(torch.float32) ** 2)
                    for gradient in gradients
                )
            )
            scale = min(1.0, max_gradient_norm / max(float(norm.item()), 1e-12))
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients, strict=True):
                    parameter.add_(
                        gradient.to(device=parameter.device, dtype=parameter.dtype),
                        alpha=-learning_rate * scale,
                    )
            adapter._assert_finite()
            calibration = evaluate_functional_action_batches(
                model,
                batches,
                expected_split="calibration-selection",
                hard_action_weight=hard_weight,
                distribution_weight=distribution_weight,
                temperature=temperature,
                used_for_selection=True,
            )
            evaluations += 1
            if calibration.total_loss < best_loss:
                best_loss = calibration.total_loss
                best_step = step
                best_state = tuple(value.detach().cpu().clone() for value in parameters)
        with torch.no_grad():
            for parameter, best in zip(parameters, best_state, strict=True):
                parameter.copy_(best.to(device=parameter.device, dtype=parameter.dtype))
        final = evaluate_functional_action_batches(
            model,
            batches,
            expected_split="calibration-selection",
            hard_action_weight=hard_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=True,
        )
        evaluations += 1
        handle.assert_live()
    finally:
        for module, training in module_modes:
            module.training = training
    adapter_state_hash = _state_sha256(_state_hashes(adapter))
    parent_examples = sum(
        batch.batch_size for batch in batches if batch.prefix_source == "parent-greedy"
    )
    student_examples = sum(
        batch.batch_size for batch in batches if batch.prefix_source == "student-greedy"
    )
    return FunctionalResidualAdapterTrainingReport(
        steps_requested=step_count,
        best_step=best_step,
        batch_count=len(batches),
        example_count=sum(batch.batch_size for batch in batches),
        parent_greedy_examples=parent_examples,
        student_greedy_examples=student_examples,
        learning_rate=learning_rate,
        hard_action_weight=hard_weight,
        distribution_weight=distribution_weight,
        temperature=temperature,
        initial_calibration_loss=initial.total_loss,
        best_calibration_loss=best_loss,
        final_calibration_loss=final.total_loss,
        calibration_evaluations=evaluations,
        autograd_calls=step_count,
        optimizer_constructed=False,
        heldout_evaluation_calls=0,
        adapter_state_sha256=adapter_state_hash,
    )


def _prefix_row_sha256s(
    batches: Sequence[FunctionalActionBatch],
) -> frozenset[str]:
    result: set[str] = set()
    for batch in batches:
        for row_index in range(batch.batch_size):
            length = int(batch.attention_mask[row_index].sum().item())
            row = batch.input_ids[row_index, :length].detach().cpu().tolist()
            digest = _canonical_sha256([int(value) for value in row])
            result.add(digest)
    return frozenset(result)


def _functional_selection_key(
    evaluation: FunctionalActionEvaluation,
) -> tuple[float, float]:
    """Discrete lexicographic calibration key; lower is better.

    Log probability and output loss remain useful diagnostics, but neither may
    promote an adapter when the actual parent action ordering is unchanged.
    """

    return (
        -evaluation.parent_action_top1_rate,
        evaluation.mean_parent_action_rank,
    )


@dataclass(frozen=True, slots=True)
class DisjointFunctionalResidualAdapterTrainingReport:
    steps_requested: int
    best_step: int
    fit_batch_count: int
    fit_example_count: int
    calibration_batch_count: int
    calibration_example_count: int
    fit_parent_greedy_examples: int
    fit_student_greedy_examples: int
    calibration_parent_greedy_examples: int
    calibration_student_greedy_examples: int
    learning_rate: float
    beta1: float
    beta2: float
    adam_epsilon: float
    max_gradient_norm: float
    hard_action_weight: float
    distribution_weight: float
    temperature: float
    initial_calibration: FunctionalActionEvaluation
    best_calibration: FunctionalActionEvaluation
    final_calibration: FunctionalActionEvaluation
    calibration_evaluations: int
    calibration_selection_steps: tuple[int, ...]
    winner_reproduction_evaluations: int
    fit_gradient_calls: int
    autograd_calls: int
    optimizer_constructed: bool
    optimizer_state_saved: bool
    manual_optimizer_state_tensors: int
    heldout_evaluation_calls: int
    adapter_state_sha256: str
    fit_prefix_domain_sha256: str
    calibration_prefix_domain_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": FUNCTIONAL_RESIDUAL_ADAPTER_DISJOINT_TRAINING_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "initial_calibration",
                    "best_calibration",
                    "final_calibration",
                }
            },
            "initial_calibration": self.initial_calibration.as_dict(),
            "best_calibration": self.best_calibration.as_dict(),
            "final_calibration": self.final_calibration.as_dict(),
            "selection_split": "calibration-selection",
            "gradient_split": "fit",
            "selection_key_order": [
                "maximize-parent-action-top1-rate",
                "minimize-mean-parent-action-rank",
            ],
            "continuous_log_probability_or_loss_can_promote": False,
            "continuous_output_metrics_are_diagnostic_only": True,
            "fit_and_calibration_prefixes_disjoint": True,
            "fixed_parent_outputs_only": True,
            "fixed_parent_and_student_rollout_prefixes": True,
            "heldout_used_for_fitting": False,
            "heldout_used_for_selection": False,
            "base_parameters_updated": 0,
            "adapter_parameters_only": True,
            "update_rule": "manual-adam",
            "learned_compaction_not_fold_only": True,
            "checkpoint_saved": False,
            "candidate_committed": False,
            "excluded_objectives": [
                "weight-similarity",
                "activation-similarity",
                "attention-similarity",
                "residual-similarity",
                "cache-similarity",
            ],
        }
        payload["training_sha256"] = _canonical_sha256(payload)
        return payload


def train_functional_residual_adapter_disjoint(
    handle: InstalledFunctionalResidualAdapter,
    fit_batches: Sequence[FunctionalActionBatch],
    calibration_batches: Sequence[FunctionalActionBatch],
    *,
    steps: int,
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    hard_action_weight: float = 1.0,
    distribution_weight: float = 0.1,
    temperature: float = 1.0,
    max_gradient_norm: float = 1.0,
    calibration_steps: Sequence[int] | None = None,
) -> DisjointFunctionalResidualAdapterTrainingReport:
    """Fit adapter factors on ``fit`` and lock only by calibration outputs.

    Heldout batches cannot enter this API.  Base parameters are temporarily
    frozen, retain their exact tensor versions/gradient objects, and are never
    updated.  The selected in-memory factor state is the lexicographic winner
    on actual calibration action behavior, never an internal similarity.
    """

    if not isinstance(handle, InstalledFunctionalResidualAdapter):
        raise FunctionalResidualAdapterTrainingError(
            "handle must be an InstalledFunctionalResidualAdapter."
        )
    handle.assert_live()
    fit_values = tuple(fit_batches)
    calibration_values = tuple(calibration_batches)
    if not fit_values or any(batch.split != "fit" for batch in fit_values):
        raise FunctionalResidualAdapterTrainingError(
            "Gradient batches must belong only to the fit split."
        )
    if not calibration_values or any(
        batch.split != "calibration-selection" for batch in calibration_values
    ):
        raise FunctionalResidualAdapterTrainingError(
            "Selection batches must belong only to calibration-selection."
        )
    for name, values in (
        ("fit", fit_values),
        ("calibration-selection", calibration_values),
    ):
        if {batch.prefix_source for batch in values} != {
            "parent-greedy",
            "student-greedy",
        }:
            raise FunctionalResidualAdapterTrainingError(
                f"{name} requires paired parent-greedy and student-greedy prefixes."
            )
    if len(
        {batch.vocabulary_size for batch in (*fit_values, *calibration_values)}
    ) != 1:
        raise FunctionalResidualAdapterTrainingError(
            "Fit and calibration batches disagree on vocabulary size."
        )
    fit_prefixes = _prefix_row_sha256s(fit_values)
    calibration_prefixes = _prefix_row_sha256s(calibration_values)
    if fit_prefixes & calibration_prefixes:
        raise FunctionalResidualAdapterTrainingError(
            "Fit and calibration contain an overlapping exact prefix row."
        )
    step_count = _index(steps, "steps", positive=True)
    if step_count > 100_000:
        raise FunctionalResidualAdapterTrainingError("steps exceeds the hard bound.")
    learning_rate = _finite_float(learning_rate, "learning_rate", positive=True)
    beta1 = _finite_float(beta1, "beta1")
    beta2 = _finite_float(beta2, "beta2")
    adam_epsilon = _finite_float(
        adam_epsilon, "adam_epsilon", positive=True
    )
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise FunctionalResidualAdapterTrainingError(
            "Adam beta values must be in [0,1)."
        )
    hard_weight = _finite_float(hard_action_weight, "hard_action_weight")
    distribution_weight = _finite_float(
        distribution_weight, "distribution_weight"
    )
    temperature = _finite_float(temperature, "temperature", positive=True)
    max_gradient_norm = _finite_float(
        max_gradient_norm, "max_gradient_norm", positive=True
    )
    if hard_weight < 0.0 or distribution_weight < 0.0 or (
        hard_weight + distribution_weight <= 0.0
    ):
        raise FunctionalResidualAdapterTrainingError(
            "At least one nonnegative output loss weight must be positive."
        )
    if calibration_steps is None:
        selection_steps = tuple(range(step_count + 1))
    else:
        try:
            selection_steps = tuple(
                _index(value, "calibration_steps entry")
                for value in calibration_steps
            )
        except TypeError as error:
            raise FunctionalResidualAdapterTrainingError(
                "calibration_steps must be an integer sequence."
            ) from error
        if (
            not selection_steps
            or selection_steps != tuple(sorted(set(selection_steps)))
            or selection_steps[0] != 0
            or selection_steps[-1] != step_count
        ):
            raise FunctionalResidualAdapterTrainingError(
                "calibration_steps must be sorted, unique, and include 0 and steps."
            )
    selection_step_set = set(selection_steps)
    model = handle.model
    adapter = handle.adapter_registry
    adapter_parameters = tuple(adapter.parameters())
    adapter_ids = {id(value) for value in adapter_parameters}
    if not adapter_parameters or any(
        not value.requires_grad for value in adapter_parameters
    ):
        raise FunctionalResidualAdapterTrainingError(
            "Every sparse adapter factor must be trainable."
        )
    base_state = tuple(
        (
            value,
            bool(value.requires_grad),
            value.grad,
            int(value._version),
            int(value.data_ptr()),
        )
        for value in model.parameters()
        if id(value) not in adapter_ids
    )
    module_modes = tuple((module, bool(module.training)) for module in model.modules())
    model.eval()
    initial: FunctionalActionEvaluation | None = None
    best: FunctionalActionEvaluation | None = None
    final: FunctionalActionEvaluation | None = None
    best_step = 0
    calibration_evaluations = 0
    first_moments = [
        torch.zeros_like(value, dtype=torch.float32)
        for value in adapter_parameters
    ]
    second_moments = [
        torch.zeros_like(value, dtype=torch.float32)
        for value in adapter_parameters
    ]
    try:
        for parameter, _requires_grad, _gradient, _version, _pointer in base_state:
            parameter.requires_grad_(False)
        initial = evaluate_functional_action_batches(
            model,
            calibration_values,
            expected_split="calibration-selection",
            hard_action_weight=hard_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=True,
        )
        calibration_evaluations = 1
        best = initial
        best_key = _functional_selection_key(initial)
        best_state = tuple(
            value.detach().cpu().clone() for value in adapter_parameters
        )
        for step in range(1, step_count + 1):
            batch = fit_values[(step - 1) % len(fit_values)]
            total, _hard, _distribution = _functional_losses(
                model,
                batch,
                hard_action_weight=hard_weight,
                distribution_weight=distribution_weight,
                temperature=temperature,
            )
            gradients = torch.autograd.grad(
                total,
                adapter_parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            norm = torch.sqrt(
                sum(
                    torch.sum(gradient.detach().to(torch.float32) ** 2)
                    for gradient in gradients
                )
            )
            scale = min(1.0, max_gradient_norm / max(float(norm.item()), 1e-12))
            with torch.no_grad():
                for parameter, gradient, first, second in zip(
                    adapter_parameters,
                    gradients,
                    first_moments,
                    second_moments,
                    strict=True,
                ):
                    clipped = gradient.detach().to(torch.float32) * scale
                    first.mul_(beta1).add_(clipped, alpha=1.0 - beta1)
                    second.mul_(beta2).addcmul_(
                        clipped, clipped, value=1.0 - beta2
                    )
                    corrected_first = first / (1.0 - beta1**step)
                    corrected_second = second / (1.0 - beta2**step)
                    update = corrected_first / (
                        torch.sqrt(corrected_second) + adam_epsilon
                    )
                    parameter.add_(
                        update.to(
                            device=parameter.device, dtype=parameter.dtype
                        ),
                        alpha=-learning_rate,
                    )
            adapter._assert_finite()
            if step not in selection_step_set:
                continue
            calibration = evaluate_functional_action_batches(
                model,
                calibration_values,
                expected_split="calibration-selection",
                hard_action_weight=hard_weight,
                distribution_weight=distribution_weight,
                temperature=temperature,
                used_for_selection=True,
            )
            calibration_evaluations += 1
            key = _functional_selection_key(calibration)
            if key < best_key:
                best_key = key
                best = calibration
                best_step = step
                best_state = tuple(
                    value.detach().cpu().clone() for value in adapter_parameters
                )
        with torch.no_grad():
            for parameter, selected in zip(
                adapter_parameters, best_state, strict=True
            ):
                parameter.copy_(
                    selected.to(device=parameter.device, dtype=parameter.dtype)
                )
        final = evaluate_functional_action_batches(
            model,
            calibration_values,
            expected_split="calibration-selection",
            hard_action_weight=hard_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=True,
        )
        calibration_evaluations += 1
        if _functional_selection_key(final) != best_key:
            raise FunctionalResidualAdapterTrainingError(
                "Locked adapter state does not reproduce its calibration winner."
            )
    finally:
        for parameter, requires_grad, gradient, _version, _pointer in base_state:
            parameter.requires_grad_(requires_grad)
            if parameter.grad is not gradient:
                parameter.grad = gradient
        for module, training in module_modes:
            module.training = training
    if initial is None or best is None or final is None:
        raise FunctionalResidualAdapterTrainingError(
            "Disjoint functional training did not produce a locked winner."
        )
    if any(
        int(parameter._version) != version
        or int(parameter.data_ptr()) != pointer
        or parameter.grad is not gradient
        or bool(parameter.requires_grad) != requires_grad
        for parameter, requires_grad, gradient, version, pointer in base_state
    ):
        raise FunctionalResidualAdapterTrainingError(
            "A base parameter changed during adapter-only training."
        )
    handle.assert_live()
    adapter_state_hash = _state_sha256(_state_hashes(adapter))
    return DisjointFunctionalResidualAdapterTrainingReport(
        steps_requested=step_count,
        best_step=best_step,
        fit_batch_count=len(fit_values),
        fit_example_count=sum(value.batch_size for value in fit_values),
        calibration_batch_count=len(calibration_values),
        calibration_example_count=sum(
            value.batch_size for value in calibration_values
        ),
        fit_parent_greedy_examples=sum(
            value.batch_size
            for value in fit_values
            if value.prefix_source == "parent-greedy"
        ),
        fit_student_greedy_examples=sum(
            value.batch_size
            for value in fit_values
            if value.prefix_source == "student-greedy"
        ),
        calibration_parent_greedy_examples=sum(
            value.batch_size
            for value in calibration_values
            if value.prefix_source == "parent-greedy"
        ),
        calibration_student_greedy_examples=sum(
            value.batch_size
            for value in calibration_values
            if value.prefix_source == "student-greedy"
        ),
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        adam_epsilon=adam_epsilon,
        max_gradient_norm=max_gradient_norm,
        hard_action_weight=hard_weight,
        distribution_weight=distribution_weight,
        temperature=temperature,
        initial_calibration=initial,
        best_calibration=best,
        final_calibration=final,
        calibration_evaluations=calibration_evaluations,
        calibration_selection_steps=selection_steps,
        winner_reproduction_evaluations=1,
        fit_gradient_calls=step_count,
        autograd_calls=step_count,
        optimizer_constructed=False,
        optimizer_state_saved=False,
        manual_optimizer_state_tensors=2 * len(adapter_parameters),
        heldout_evaluation_calls=0,
        adapter_state_sha256=adapter_state_hash,
        fit_prefix_domain_sha256=_canonical_sha256(sorted(fit_prefixes)),
        calibration_prefix_domain_sha256=_canonical_sha256(
            sorted(calibration_prefixes)
        ),
    )


__all__ = [
    "FUNCTIONAL_RESIDUAL_ADAPTER_ACCOUNTING_SCHEMA",
    "FUNCTIONAL_RESIDUAL_ADAPTER_EVALUATION_SCHEMA",
    "FUNCTIONAL_RESIDUAL_ADAPTER_SCHEMA",
    "FUNCTIONAL_RESIDUAL_ADAPTER_TRAINING_SCHEMA",
    "FUNCTIONAL_RESIDUAL_ADAPTER_DISJOINT_TRAINING_SCHEMA",
    "P35_L19G4_TO_L21G6",
    "FunctionalActionBatch",
    "FunctionalActionEvaluation",
    "FunctionalActionSplit",
    "FunctionalPrefixSource",
    "FunctionalResidualAdapterAccounting",
    "FunctionalResidualAdapterAccountingError",
    "FunctionalResidualAdapterError",
    "FunctionalResidualAdapterInstallationReport",
    "FunctionalResidualAdapterOutputProjection",
    "FunctionalResidualAdapterTopologyError",
    "FunctionalResidualAdapterTrainingError",
    "FunctionalResidualAdapterTrainingReport",
    "DisjointFunctionalResidualAdapterTrainingReport",
    "InstalledFunctionalResidualAdapter",
    "RedirectAudit",
    "RedirectedAttentionSite",
    "SparseFunctionalResidualAdapterRegistry",
    "audit_redirected_attention_site",
    "evaluate_functional_action_batches",
    "install_functional_residual_adapter",
    "install_p35_l19g4_to_l21g6_functional_residual_adapter",
    "train_functional_residual_adapter",
    "train_functional_residual_adapter_disjoint",
]
