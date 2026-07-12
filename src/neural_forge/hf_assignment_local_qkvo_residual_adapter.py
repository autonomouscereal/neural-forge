"""Assignment-local Q/K/V/O residual adapters for attention codebook folds.

For each sparse physical ``(layer, GQA-group)`` site this module adds four
independent low-rank residuals without numerically matching parent weights::

    q_group = q_shared_group(x) + Uq @ (Rq @ x)
    k_group = k_shared_group(x) + Uk @ (Rk @ x)
    v_group = v_shared_group(x) + Uv @ (Rv @ x)
    y       = O_shared(z_all) + Uo @ (Ro @ z_group)

Q/K/V corrections are installed directly at the projection attributes used by
Qwen, so they execute before Q/K normalization, RoPE, and K/V cache append.
The O correction consumes the actual post-softmax group context immediately
before the residual add.  Every output factor starts at zero while its input
factor is deterministic random, making installation an exact-zero functional
perturbation with a useful first gradient.

The API intentionally exposes only adapter parameters for output-function
training.  It contains no parent-weight, activation, attention, residual, or
cache-value similarity objective and no checkpoint/commit operation.
"""

from __future__ import annotations

import hashlib
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
from neural_forge.hf_functional_residual_adapter import (
    P35_L19G4_TO_L21G6,
    FunctionalActionBatch,
    FunctionalActionEvaluation,
    RedirectAudit,
    RedirectedAttentionSite,
    _functional_losses,
    evaluate_functional_action_batches,
    audit_redirected_attention_site,
)


ASSIGNMENT_LOCAL_QKVO_RESIDUAL_SCHEMA = "neural-forge-assignment-local-qkvo-residual-v1"
ASSIGNMENT_LOCAL_QKVO_ACCOUNTING_SCHEMA = (
    "neural-forge-assignment-local-qkvo-residual-accounting-v1"
)
ASSIGNMENT_LOCAL_QKVO_INSTALLATION_SCHEMA = (
    "neural-forge-assignment-local-qkvo-residual-installation-v1"
)
ASSIGNMENT_LOCAL_QKVO_TRAINING_SCOPE_SCHEMA = (
    "neural-forge-assignment-local-qkvo-output-training-scope-v1"
)
ASSIGNMENT_LOCAL_QKVO_DISJOINT_TRAINING_SCHEMA = (
    "neural-forge-assignment-local-qkvo-disjoint-output-training-v1"
)

ProjectionRole = Literal["q", "k", "v", "o"]


class AssignmentLocalQKVOResidualError(RuntimeError):
    """An assignment-local operator or proof contract is invalid."""


class AssignmentLocalQKVOResidualTopologyError(AssignmentLocalQKVOResidualError):
    """A physical site, projection boundary, or live installation is invalid."""


class AssignmentLocalQKVOResidualAccountingError(AssignmentLocalQKVOResidualError):
    """A parameter or payload compaction claim is invalid."""


class AssignmentLocalQKVOResidualTrainingError(AssignmentLocalQKVOResidualError):
    """Adapter-only output training changed a frozen base parameter."""


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise AssignmentLocalQKVOResidualError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AssignmentLocalQKVOResidualError(f"{name} must be an integer.") from error
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise AssignmentLocalQKVOResidualError(f"{name} must be {qualifier}.")
    return result


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise AssignmentLocalQKVOResidualError(
            "QKVO residual provenance is not canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
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
    count = 0
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            count += int(parameter.numel())
    return count


def _model_layers(model: nn.Module) -> Sequence[nn.Module]:
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or not layers:
        raise AssignmentLocalQKVOResidualTopologyError(
            "Model must expose model.layers or layers."
        )
    return layers


def _site_indices(
    sites: Sequence[RedirectedAttentionSite],
    *,
    logical_layers: int,
    physical_groups: int,
) -> Tensor:
    pairs = [site.site_pair for site in sites]
    if not pairs:
        raise AssignmentLocalQKVOResidualTopologyError(
            "At least one redirected site is required."
        )
    if pairs != sorted(pairs) or len(pairs) != len(set(pairs)):
        raise AssignmentLocalQKVOResidualTopologyError(
            "Redirected sites must be sorted and unique by physical site."
        )
    if any(
        layer < 0 or layer >= logical_layers or group < 0 or group >= physical_groups
        for layer, group in pairs
    ):
        raise AssignmentLocalQKVOResidualTopologyError(
            "A redirected physical site is outside the live topology."
        )
    return torch.tensor(pairs, dtype=torch.int32)


@dataclass(frozen=True, slots=True)
class AssignmentLocalQKVOResidualAccounting:
    """Closed-form parameter and tensor accounting for sparse Q/K/V/O repair."""

    hidden_size: int
    query_group_width: int
    head_dim: int
    logical_layers: int
    physical_groups: int
    rank: int
    site_count: int
    checkpoint_element_size: int
    removed_site_weight_parameters: int
    removed_site_bias_parameters: int
    removed_site_parameters: int
    redirect_role_scale_parameters: int
    q_adapter_parameters: int
    k_adapter_parameters: int
    v_adapter_parameters: int
    o_adapter_parameters: int
    adapter_parameters: int
    net_parameters_saved_vs_incumbent: int
    learned_payload_bytes_saved: int
    removed_prototype_index_bytes: int
    adapter_site_index_bytes: int
    adapter_schema_buffer_bytes: int
    index_adjusted_payload_bytes_saved: int
    tensor_payload_bytes_saved: int
    flops_per_site_token: int

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        query_group_width: int,
        head_dim: int,
        logical_layers: int,
        physical_groups: int,
        rank: int,
        site_count: int = 1,
        checkpoint_element_size: int = 2,
        q_prototype_bias: bool = False,
        k_prototype_bias: bool = False,
        v_prototype_bias: bool = False,
        role_scale_table_new_for_redirect: bool = True,
        removed_prototype_index_bytes: int = 4,
        schema_buffer_bytes: int = 8,
    ) -> AssignmentLocalQKVOResidualAccounting:
        values = {
            name: _index(value, name, positive=True)
            for name, value in {
                "hidden_size": hidden_size,
                "query_group_width": query_group_width,
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
                q_prototype_bias,
                k_prototype_bias,
                v_prototype_bias,
                role_scale_table_new_for_redirect,
            )
        ):
            raise AssignmentLocalQKVOResidualAccountingError(
                "Accounting feature flags must be boolean."
            )
        hidden = values["hidden_size"]
        query_group = values["query_group_width"]
        head = values["head_dim"]
        rank_value = values["rank"]
        sites = values["site_count"]
        removed_weights = sites * (2 * hidden * query_group + 2 * hidden * head)
        removed_biases = sites * (
            query_group * int(q_prototype_bias)
            + head * int(k_prototype_bias)
            + head * int(v_prototype_bias)
        )
        removed = removed_weights + removed_biases
        role_scales = (
            values["logical_layers"] * values["physical_groups"] * 4
            if role_scale_table_new_for_redirect
            else 0
        )
        q_parameters = sites * rank_value * (hidden + query_group)
        k_parameters = sites * rank_value * (hidden + head)
        v_parameters = sites * rank_value * (hidden + head)
        o_parameters = sites * rank_value * (query_group + hidden)
        adapter = q_parameters + k_parameters + v_parameters + o_parameters
        net = removed - role_scales - adapter
        if net <= 0:
            raise AssignmentLocalQKVOResidualAccountingError(
                "QKVO adapter consumes the removed-prototype budget."
            )
        element_size = values["checkpoint_element_size"]
        learned_payload = net * element_size
        removed_index = sites * _index(
            removed_prototype_index_bytes,
            "removed_prototype_index_bytes",
        )
        site_index = sites * 2 * torch.tensor([], dtype=torch.int32).element_size()
        schema_bytes = _index(schema_buffer_bytes, "schema_buffer_bytes")
        index_adjusted = learned_payload + removed_index
        tensor_payload = index_adjusted - site_index - schema_bytes
        return cls(
            hidden_size=hidden,
            query_group_width=query_group,
            head_dim=head,
            logical_layers=values["logical_layers"],
            physical_groups=values["physical_groups"],
            rank=rank_value,
            site_count=sites,
            checkpoint_element_size=element_size,
            removed_site_weight_parameters=removed_weights,
            removed_site_bias_parameters=removed_biases,
            removed_site_parameters=removed,
            redirect_role_scale_parameters=role_scales,
            q_adapter_parameters=q_parameters,
            k_adapter_parameters=k_parameters,
            v_adapter_parameters=v_parameters,
            o_adapter_parameters=o_parameters,
            adapter_parameters=adapter,
            net_parameters_saved_vs_incumbent=net,
            learned_payload_bytes_saved=learned_payload,
            removed_prototype_index_bytes=removed_index,
            adapter_site_index_bytes=site_index,
            adapter_schema_buffer_bytes=schema_bytes,
            index_adjusted_payload_bytes_saved=index_adjusted,
            tensor_payload_bytes_saved=tensor_payload,
            flops_per_site_token=2 * (adapter // sites),
        )

    @classmethod
    def qwen3_4b_one_site(
        cls,
        *,
        rank: int = 8,
        checkpoint_element_size: int = 2,
    ) -> AssignmentLocalQKVOResidualAccounting:
        return cls.build(
            hidden_size=2560,
            query_group_width=512,
            head_dim=128,
            logical_layers=36,
            physical_groups=8,
            rank=rank,
            checkpoint_element_size=checkpoint_element_size,
            role_scale_table_new_for_redirect=True,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": ASSIGNMENT_LOCAL_QKVO_ACCOUNTING_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "one_coupled_group_prototype_removed_per_site": True,
            "role_scale_table_counted_at_most_once": True,
            "role_scale_table_new_for_redirect": (
                self.redirect_role_scale_parameters > 0
            ),
            "o_bias_removed": False,
            "optimizer_state_counted": False,
            "artifact_plan_required_before_checkpoint_claim": True,
            "checkpoint_claimed": False,
        }
        payload["accounting_sha256"] = _canonical_sha256(payload)
        return payload


class SparseAssignmentLocalQKVOResidualRegistry(nn.Module):
    """One sparse registry owner for assignment-local Q/K/V/O factors."""

    _ROLES = ("q", "k", "v", "o")

    def __init__(
        self,
        site_indices: Tensor | Sequence[Sequence[int]],
        *,
        logical_layers: int,
        physical_groups: int,
        hidden_size: int,
        query_group_width: int,
        head_dim: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device | str,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.logical_layers = _index(logical_layers, "logical_layers", positive=True)
        self.physical_groups = _index(physical_groups, "physical_groups", positive=True)
        self.hidden_size = _index(hidden_size, "hidden_size", positive=True)
        self.query_group_width = _index(
            query_group_width, "query_group_width", positive=True
        )
        self.head_dim = _index(head_dim, "head_dim", positive=True)
        self.rank = _index(rank, "rank", positive=True)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise AssignmentLocalQKVOResidualTopologyError(
                "Adapter dtype must be floating point."
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
            raise AssignmentLocalQKVOResidualTopologyError(
                "site_indices must be a nonempty integer [site,2] tensor."
            )
        rows = [
            tuple(map(int, row))
            for row in raw_sites.detach().cpu().to(torch.int64).tolist()
        ]
        if (
            rows != sorted(rows)
            or len(rows) != len(set(rows))
            or any(
                layer < 0
                or layer >= self.logical_layers
                or group < 0
                or group >= self.physical_groups
                for layer, group in rows
            )
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "site_indices must be sorted, unique, and in range."
            )
        self.register_buffer(
            "site_indices",
            raw_sites.detach().to(device=device, dtype=torch.int32),
        )
        self.register_buffer(
            "_registry_schema_version",
            torch.tensor(1, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "_registry_layout",
            torch.tensor(1, dtype=torch.int32, device=device),
        )
        sites = len(rows)
        seed = _index(initialization_seed, "initialization_seed")
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
            left = torch.zeros(
                (sites, output_width, self.rank),
                dtype=dtype,
                device=device,
            )
            bound = 1.0 / math.sqrt(input_width)
            right = torch.empty(
                (sites, self.rank, input_width),
                dtype=torch.float32,
                device="cpu",
            )
            right.uniform_(-bound, bound, generator=generator)
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
    def query_width(self) -> int:
        return self.physical_groups * self.query_group_width

    @property
    def key_value_width(self) -> int:
        return self.physical_groups * self.head_dim

    @property
    def learned_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

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

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return only sparse factors; the API accepts no internal targets."""

        return tuple(self.parameters())

    def _assert_finite(self) -> None:
        for name, value in self.named_parameters():
            if not bool(torch.isfinite(value.detach()).all()):
                raise AssignmentLocalQKVOResidualError(
                    f"Adapter parameter {name!r} contains NaN or infinity."
                )

    def _site_rows(self, layer_index: int) -> tuple[tuple[int, int], ...]:
        layer = _index(layer_index, "layer_index")
        if layer >= self.logical_layers:
            raise AssignmentLocalQKVOResidualTopologyError(
                "layer_index is outside the adapter registry."
            )
        return tuple(
            (row_index, int(group))
            for row_index, (site_layer, group) in enumerate(
                self.site_indices.detach().cpu().to(torch.int64).tolist()
            )
            if int(site_layer) == layer
        )

    def _correction(
        self, role: ProjectionRole, row_index: int, inputs: Tensor
    ) -> Tensor:
        left = getattr(self, f"{role}_left")[row_index]
        right = getattr(self, f"{role}_right")[row_index]
        latent = F.linear(inputs.to(dtype=right.dtype), right)
        return F.linear(latent, left)

    def apply_to_projection(
        self,
        role: ProjectionRole,
        inputs: Tensor,
        base_output: Tensor,
        *,
        layer_index: int,
    ) -> Tensor:
        """Add pointwise physical-group Q/K/V corrections before Qwen transforms."""

        if role not in {"q", "k", "v"}:
            raise AssignmentLocalQKVOResidualTopologyError(
                "apply_to_projection accepts only q, k, or v."
            )
        expected_input = self.hidden_size
        role_width = self.query_group_width if role == "q" else self.head_dim
        expected_output = self.physical_groups * role_width
        if (
            not isinstance(inputs, Tensor)
            or inputs.ndim == 0
            or int(inputs.shape[-1]) != expected_input
            or not isinstance(base_output, Tensor)
            or tuple(base_output.shape[:-1]) != tuple(inputs.shape[:-1])
            or int(base_output.shape[-1]) != expected_output
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                f"{role.upper()} projection tensors do not match the physical topology."
            )
        self._assert_finite()
        result = base_output
        for row_index, physical_group in self._site_rows(layer_index):
            correction = self._correction(role, row_index, inputs).to(
                dtype=base_output.dtype
            )
            start = physical_group * role_width
            stop = start + role_width
            result = result + F.pad(
                correction,
                (start, expected_output - stop),
            )
        return result

    def apply_to_output(
        self,
        contexts: Tensor,
        base_output: Tensor,
        *,
        layer_index: int,
    ) -> Tensor:
        """Add pointwise O residuals from real post-softmax physical contexts."""

        if (
            not isinstance(contexts, Tensor)
            or contexts.ndim == 0
            or int(contexts.shape[-1]) != self.query_width
            or not isinstance(base_output, Tensor)
            or tuple(base_output.shape[:-1]) != tuple(contexts.shape[:-1])
            or int(base_output.shape[-1]) != self.hidden_size
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "O projection tensors do not match the physical topology."
            )
        self._assert_finite()
        result = base_output
        for row_index, physical_group in self._site_rows(layer_index):
            start = physical_group * self.query_group_width
            stop = start + self.query_group_width
            correction = self._correction("o", row_index, contexts[..., start:stop])
            result = result + correction.to(dtype=base_output.dtype)
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
            "schema": ASSIGNMENT_LOCAL_QKVO_RESIDUAL_SCHEMA,
            "site_indices": self.site_indices.detach().cpu().tolist(),
            "logical_layers": self.logical_layers,
            "physical_groups": self.physical_groups,
            "hidden_size": self.hidden_size,
            "query_group_width": self.query_group_width,
            "head_dim": self.head_dim,
            "rank": self.rank,
            "site_count": self.site_count,
            "learned_parameter_count": self.learned_parameter_count,
            "role_parameter_counts": role_parameters,
            "zero_output_factor_initialization": self.starts_as_exact_zero_correction,
            "random_input_factor_initialization": self.input_factors_are_nonzero,
            "sparse_sites_only": True,
            "pointwise_over_sequence_positions": True,
            "cross_token_mixing_inside_adapter": False,
            "parent_runtime_references": 0,
            "output_function_training_parameters_only": True,
            "excluded_objectives": [
                "weight-similarity",
                "qkv-similarity",
                "activation-similarity",
                "attention-similarity",
                "residual-similarity",
                "cache-value-similarity",
            ],
        }
        payload["registry_sha256"] = _canonical_sha256(payload)
        return payload


class AssignmentLocalQKVOResidualProjection(nn.Module):
    """Linear-compatible projection view with one sparse residual registry."""

    def __init__(
        self,
        base_projection: AttentionCodebookProjectionView,
        adapter_registry: SparseAssignmentLocalQKVOResidualRegistry,
        attention_registry: AttentionCodebookRegistry,
        layer_index: int,
        role: ProjectionRole,
    ) -> None:
        super().__init__()
        if (
            not isinstance(base_projection, AttentionCodebookProjectionView)
            or base_projection.registry is not attention_registry
            or base_projection.role != role
            or base_projection.layer_index != layer_index
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Base projection is not the declared live codebook view."
            )
        if role not in {"q", "k", "v", "o"}:
            raise AssignmentLocalQKVOResidualTopologyError(
                "Projection role must be q, k, v, or o."
            )
        object.__setattr__(self, "_adapter_registry", adapter_registry)
        object.__setattr__(self, "_attention_registry", attention_registry)
        self.base_projection = base_projection
        self.layer_index = _index(layer_index, "layer_index")
        self.role = role
        self.in_features = base_projection.in_features
        self.out_features = base_projection.out_features
        self.train(base_projection.training)

    @property
    def adapter_registry(self) -> SparseAssignmentLocalQKVOResidualRegistry:
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
class AssignmentLocalQKVOResidualInstallationReport:
    sites: tuple[RedirectedAttentionSite, ...]
    redirect_audits: tuple[RedirectAudit, ...]
    rank: int
    owner_name: str
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_installed: int
    attention_registry_state_sha256: str
    attention_registry_identity_sha256: str
    adapter_site_indices_sha256: str
    accounting: AssignmentLocalQKVOResidualAccounting

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": ASSIGNMENT_LOCAL_QKVO_INSTALLATION_SCHEMA,
            "sites": [site.as_dict() for site in self.sites],
            "redirect_audits": [audit.as_dict() for audit in self.redirect_audits],
            "rank": self.rank,
            "owner_name": self.owner_name,
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "attention_registry_state_sha256": self.attention_registry_state_sha256,
            "attention_registry_identity_sha256": self.attention_registry_identity_sha256,
            "adapter_site_indices_sha256": self.adapter_site_indices_sha256,
            "accounting": self.accounting.as_dict(),
            "q_correction_boundary": "projection-output-before-q-norm-and-rope",
            "k_correction_boundary": (
                "projection-output-before-k-norm-rope-and-cache-append"
            ),
            "v_correction_boundary": "projection-output-before-cache-append",
            "o_correction_boundary": (
                "post-softmax-physical-group-context-before-residual-add"
            ),
            "pointwise_causal_operator": True,
            "adapter_cross_token_state": False,
            "future_token_access": False,
            "k_v_corrections_enter_child_cache": True,
            "cache_sidecar_required": False,
            "base_projection_reused": True,
            "shared_survivor_globally_refit": False,
            "zero_initialization_preserves_tentative_child": True,
            "parent_or_internal_numeric_similarity_used": False,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


class InstalledAssignmentLocalQKVOResidualAdapter:
    """Live experimental QKVO adapter with exact removal rollback."""

    def __init__(
        self,
        *,
        model: nn.Module,
        attention_registry: AttentionCodebookRegistry,
        adapter_registry: SparseAssignmentLocalQKVOResidualRegistry,
        report: AssignmentLocalQKVOResidualInstallationReport,
        original_projections: Mapping[
            tuple[int, ProjectionRole], AttentionCodebookProjectionView
        ],
        installed_views: Mapping[
            tuple[int, ProjectionRole], AssignmentLocalQKVOResidualProjection
        ],
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

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        return self.adapter_registry.output_trainable_parameters()

    def assert_base_unchanged(self) -> None:
        if (
            dict(_state_hashes(self.attention_registry)) != self._registry_hashes_before
            or _tensor_identity_sha256(self.attention_registry)
            != self._registry_identity_before
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Base attention registry changed during the QKVO experiment."
            )

    def _assert_unique_registry_owner(self) -> None:
        paths = [
            name
            for name, module in self.model.named_modules(remove_duplicate=False)
            if module is self.adapter_registry
        ]
        if paths != [self.report.owner_name]:
            raise AssignmentLocalQKVOResidualTopologyError(
                "Sparse QKVO registry must have exactly one registered owner."
            )

    def assert_live(self) -> AssignmentLocalQKVOResidualInstallationReport:
        if self._rolled_back:
            raise AssignmentLocalQKVOResidualTopologyError(
                "A rolled-back QKVO adapter is no longer live."
            )
        if (
            getattr(self.model, self.report.owner_name, None)
            is not self.adapter_registry
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Model lost the sparse QKVO registry owner."
            )
        self._assert_unique_registry_owner()
        layers = _model_layers(self.model)
        for (layer_index, role), installed in self._installed_views.items():
            if (
                getattr(layers[layer_index].self_attn, f"{role}_proj", None)
                is not installed
            ):
                raise AssignmentLocalQKVOResidualTopologyError(
                    "A live assignment-local projection view was replaced."
                )
        if _tensor_sha256(self.adapter_registry.site_indices) != (
            self.report.adapter_site_indices_sha256
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Sparse QKVO site indices changed."
            )
        self.adapter_registry._assert_finite()
        self.assert_base_unchanged()
        return self.report

    def output_training_scope(self) -> AdapterOnlyOutputTrainingScope:
        return AdapterOnlyOutputTrainingScope(self)

    def rollback(self) -> None:
        if self._rolled_back:
            return
        layers = _model_layers(self.model)
        owner = getattr(self.model, self.report.owner_name, None)
        if owner is not None and owner is not self.adapter_registry:
            raise AssignmentLocalQKVOResidualTopologyError(
                "QKVO registry owner was replaced; exact rollback is unsafe."
            )
        if owner is self.adapter_registry:
            self._assert_unique_registry_owner()
        for (layer_index, role), installed in self._installed_views.items():
            if (
                getattr(layers[layer_index].self_attn, f"{role}_proj", None)
                is not installed
            ):
                raise AssignmentLocalQKVOResidualTopologyError(
                    "QKVO projection view was replaced; exact rollback is unsafe."
                )
        if _tensor_sha256(self.adapter_registry.site_indices) != (
            self.report.adapter_site_indices_sha256
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Sparse QKVO site indices changed; exact rollback is unsafe."
            )
        # Adapter factors may be trained or non-finite. They are isolated and
        # are removed, not trusted, during rejection rollback.
        self.assert_base_unchanged()
        for (layer_index, role), original in self._original_projections.items():
            setattr(layers[layer_index].self_attn, f"{role}_proj", original)
        if getattr(self.model, self.report.owner_name, None) is self.adapter_registry:
            delattr(self.model, self.report.owner_name)
        elif hasattr(self.model, self.report.owner_name):
            raise AssignmentLocalQKVOResidualTopologyError(
                "QKVO registry owner changed during exact rollback."
            )
        self.assert_base_unchanged()
        if (
            _unique_parameter_count(self.model)
            != self.report.model_parameter_count_before
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Exact QKVO rollback parameter accounting failed."
            )
        for (layer_index, role), original in self._original_projections.items():
            if (
                getattr(layers[layer_index].self_attn, f"{role}_proj", None)
                is not original
            ):
                raise AssignmentLocalQKVOResidualTopologyError(
                    "Exact QKVO projection rollback failed."
                )
        self._rolled_back = True

    def __enter__(self) -> InstalledAssignmentLocalQKVOResidualAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.rollback()


class AdapterOnlyOutputTrainingScope:
    """Temporarily expose only adapter factors to an output-space trainer."""

    def __init__(self, handle: InstalledAssignmentLocalQKVOResidualAdapter) -> None:
        if not isinstance(handle, InstalledAssignmentLocalQKVOResidualAdapter):
            raise AssignmentLocalQKVOResidualTrainingError(
                "Training scope requires a live QKVO adapter handle."
            )
        self.handle = handle
        self._base_state: tuple[
            tuple[nn.Parameter, bool, Tensor | None, int, int], ...
        ] = ()
        self._module_modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise AssignmentLocalQKVOResidualTrainingError(
                "Output training scope cannot be re-entered."
            )
        self.handle.assert_live()
        trainable = self.handle.output_trainable_parameters()
        adapter_ids = {id(value) for value in trainable}
        if not trainable or any(not value.requires_grad for value in trainable):
            raise AssignmentLocalQKVOResidualTrainingError(
                "Every QKVO factor must be trainable."
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
            raise AssignmentLocalQKVOResidualTrainingError(
                "A base parameter changed inside adapter-only output training."
            )
        self.handle.assert_live()

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": ASSIGNMENT_LOCAL_QKVO_TRAINING_SCOPE_SCHEMA,
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


def _functional_selection_key(
    evaluation: FunctionalActionEvaluation,
) -> tuple[float, float]:
    """Discrete calibration behavior only; continuous values are diagnostics."""

    return (
        -evaluation.parent_action_top1_rate,
        evaluation.mean_parent_action_rank,
    )


def _prefix_domain_sha256(
    batches: Sequence[FunctionalActionBatch],
) -> tuple[frozenset[str], str]:
    hashes: set[str] = set()
    for batch in batches:
        for row_index in range(batch.batch_size):
            length = int(batch.attention_mask[row_index].sum().item())
            row = [
                int(value)
                for value in batch.input_ids[row_index, :length].detach().cpu().tolist()
            ]
            hashes.add(_canonical_sha256(row))
    frozen = frozenset(hashes)
    return frozen, _canonical_sha256(sorted(frozen))


@dataclass(frozen=True, slots=True)
class AssignmentLocalQKVODisjointTrainingReport:
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
    q_learning_rate: float
    k_learning_rate: float
    v_learning_rate: float
    o_learning_rate: float
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
    calibration_selection_steps: tuple[int, ...]
    calibration_evaluations: int
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
    branch_parameter_counts: Mapping[str, int]
    calibration_history: tuple[Mapping[str, Any], ...]
    fit_step_history: tuple[Mapping[str, Any], ...]
    fit_loss_summary: Mapping[str, Any]
    branch_gradient_summary: Mapping[str, Any]
    branch_parameter_delta_summary: Mapping[str, Any]
    parameter_gradient_activation: Mapping[str, bool]
    all_branches_received_nonzero_gradient: bool
    all_parameter_tensors_received_nonzero_gradient: bool
    prelock_final_adapter_state_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": ASSIGNMENT_LOCAL_QKVO_DISJOINT_TRAINING_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "initial_calibration",
                    "best_calibration",
                    "final_calibration",
                    "branch_parameter_counts",
                    "calibration_history",
                    "fit_step_history",
                    "fit_loss_summary",
                    "branch_gradient_summary",
                    "branch_parameter_delta_summary",
                    "parameter_gradient_activation",
                }
            },
            "initial_calibration": self.initial_calibration.as_dict(),
            "best_calibration": self.best_calibration.as_dict(),
            "final_calibration": self.final_calibration.as_dict(),
            "branch_parameter_counts": dict(self.branch_parameter_counts),
            "calibration_history": [dict(value) for value in self.calibration_history],
            "fit_step_history": [dict(value) for value in self.fit_step_history],
            "fit_loss_summary": dict(self.fit_loss_summary),
            "branch_gradient_summary": dict(self.branch_gradient_summary),
            "branch_parameter_delta_summary": dict(self.branch_parameter_delta_summary),
            "parameter_gradient_activation": dict(self.parameter_gradient_activation),
            "gradient_split": "fit",
            "selection_split": "calibration-selection",
            "selection_key_order": [
                "maximize-parent-action-top1-rate",
                "minimize-mean-parent-action-rank",
            ],
            "continuous_log_probability_or_loss_can_promote": False,
            "continuous_output_metrics_are_diagnostic_only": True,
            "diagnostics_can_select_or_promote": False,
            "fit_and_calibration_prefixes_disjoint": True,
            "fixed_parent_outputs_only": True,
            "fixed_parent_and_student_rollout_prefixes": True,
            "adapter_parameters_only": True,
            "base_parameters_updated": 0,
            "update_rule": "manual-adam-per-qkvo-branch-learning-rate",
            "heldout_used_for_fitting": False,
            "heldout_used_for_selection": False,
            "excluded_objectives": [
                "weight-similarity",
                "qkv-similarity",
                "activation-similarity",
                "attention-similarity",
                "residual-similarity",
                "cache-value-similarity",
            ],
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["training_sha256"] = _canonical_sha256(payload)
        return payload


def train_assignment_local_qkvo_residual_disjoint(
    handle: InstalledAssignmentLocalQKVOResidualAdapter,
    fit_batches: Sequence[FunctionalActionBatch],
    calibration_batches: Sequence[FunctionalActionBatch],
    *,
    steps: int,
    q_learning_rate: float = 1e-3,
    k_learning_rate: float = 1e-3,
    v_learning_rate: float = 2e-3,
    o_learning_rate: float = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    hard_action_weight: float = 1.0,
    distribution_weight: float = 0.1,
    temperature: float = 1.0,
    max_gradient_norm: float = 1.0,
    calibration_steps: Sequence[int] | None = None,
) -> AssignmentLocalQKVODisjointTrainingReport:
    """Fit Q/K/V/O factors on fit and lock only by calibration actions."""

    if not isinstance(handle, InstalledAssignmentLocalQKVOResidualAdapter):
        raise AssignmentLocalQKVOResidualTrainingError(
            "handle must be an installed assignment-local QKVO adapter."
        )
    handle.assert_live()
    fit_values = tuple(fit_batches)
    calibration_values = tuple(calibration_batches)
    if not fit_values or any(value.split != "fit" for value in fit_values):
        raise AssignmentLocalQKVOResidualTrainingError(
            "Gradient batches must belong only to fit."
        )
    if not calibration_values or any(
        value.split != "calibration-selection" for value in calibration_values
    ):
        raise AssignmentLocalQKVOResidualTrainingError(
            "Selection batches must belong only to calibration-selection."
        )
    for name, values in (
        ("fit", fit_values),
        ("calibration-selection", calibration_values),
    ):
        if {value.prefix_source for value in values} != {
            "parent-greedy",
            "student-greedy",
        }:
            raise AssignmentLocalQKVOResidualTrainingError(
                f"{name} requires paired parent-greedy and student-greedy prefixes."
            )
    if (
        len({value.vocabulary_size for value in (*fit_values, *calibration_values)})
        != 1
    ):
        raise AssignmentLocalQKVOResidualTrainingError(
            "Fit and calibration vocabulary sizes disagree."
        )
    fit_prefixes, fit_prefix_hash = _prefix_domain_sha256(fit_values)
    calibration_prefixes, calibration_prefix_hash = _prefix_domain_sha256(
        calibration_values
    )
    if fit_prefixes & calibration_prefixes:
        raise AssignmentLocalQKVOResidualTrainingError(
            "Fit and calibration contain an overlapping exact prefix row."
        )
    step_count = _index(steps, "steps", positive=True)
    if step_count > 100_000:
        raise AssignmentLocalQKVOResidualTrainingError("steps exceeds the hard bound.")

    def finite_positive(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise AssignmentLocalQKVOResidualTrainingError(
                f"{name} must be finite and positive."
            )
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise AssignmentLocalQKVOResidualTrainingError(
                f"{name} must be finite and positive."
            ) from error
        if not math.isfinite(result) or result <= 0.0:
            raise AssignmentLocalQKVOResidualTrainingError(
                f"{name} must be finite and positive."
            )
        return result

    branch_lrs = {
        "q": finite_positive(q_learning_rate, "q_learning_rate"),
        "k": finite_positive(k_learning_rate, "k_learning_rate"),
        "v": finite_positive(v_learning_rate, "v_learning_rate"),
        "o": finite_positive(o_learning_rate, "o_learning_rate"),
    }
    beta1_value = float(beta1)
    beta2_value = float(beta2)
    if (
        not math.isfinite(beta1_value)
        or not math.isfinite(beta2_value)
        or not 0.0 <= beta1_value < 1.0
        or not 0.0 <= beta2_value < 1.0
    ):
        raise AssignmentLocalQKVOResidualTrainingError(
            "Adam beta values must be finite and in [0,1)."
        )
    epsilon = finite_positive(adam_epsilon, "adam_epsilon")
    gradient_bound = finite_positive(max_gradient_norm, "max_gradient_norm")
    temperature_value = finite_positive(temperature, "temperature")
    hard_weight = float(hard_action_weight)
    distribution_weight_value = float(distribution_weight)
    if (
        not math.isfinite(hard_weight)
        or not math.isfinite(distribution_weight_value)
        or hard_weight < 0.0
        or distribution_weight_value < 0.0
        or hard_weight + distribution_weight_value <= 0.0
    ):
        raise AssignmentLocalQKVOResidualTrainingError(
            "At least one finite nonnegative output loss weight must be positive."
        )
    if calibration_steps is None:
        selection_steps = tuple(range(step_count + 1))
    else:
        selection_steps = tuple(
            _index(value, "calibration_steps entry") for value in calibration_steps
        )
        if (
            not selection_steps
            or selection_steps != tuple(sorted(set(selection_steps)))
            or selection_steps[0] != 0
            or selection_steps[-1] != step_count
        ):
            raise AssignmentLocalQKVOResidualTrainingError(
                "calibration_steps must be sorted, unique, and include 0 and steps."
            )
    selection_step_set = set(selection_steps)
    named_parameters = tuple(handle.adapter_registry.named_parameters())
    parameter_names = tuple(name for name, _value in named_parameters)
    parameters = tuple(value for _name, value in named_parameters)
    parameter_lrs = tuple(branch_lrs[name[0]] for name in parameter_names)
    branch_counts = MappingProxyType(
        {
            role: sum(
                int(value.numel())
                for name, value in named_parameters
                if name.startswith(f"{role}_")
            )
            for role in ("q", "k", "v", "o")
        }
    )
    first_moments = [
        torch.zeros_like(value, dtype=torch.float32) for value in parameters
    ]
    second_moments = [
        torch.zeros_like(value, dtype=torch.float32) for value in parameters
    ]
    initial_parameter_state = tuple(
        value.detach().cpu().to(torch.float32).clone() for value in parameters
    )
    parameter_gradient_activation = {name: False for name in parameter_names}
    branch_gradient_values: dict[str, list[float]] = {
        role: [] for role in ("q", "k", "v", "o")
    }
    branch_delta_values: dict[str, list[float]] = {
        role: [] for role in ("q", "k", "v", "o")
    }
    fit_history: list[dict[str, Any]] = []
    calibration_history: list[dict[str, Any]] = []

    def parameter_delta_norms() -> dict[str, float]:
        result: dict[str, float] = {}
        for role in ("q", "k", "v", "o"):
            squared = sum(
                torch.sum(
                    (
                        parameter.detach().cpu().to(torch.float32)
                        - initial_parameter_state[index]
                    )
                    ** 2
                )
                for index, (name, parameter) in enumerate(named_parameters)
                if name.startswith(f"{role}_")
            )
            result[role] = float(torch.sqrt(squared).item())
        return result

    def adapter_state_sha256() -> str:
        return _state_sha256(_state_hashes(handle.adapter_registry))

    initial: FunctionalActionEvaluation | None = None
    best: FunctionalActionEvaluation | None = None
    final: FunctionalActionEvaluation | None = None
    best_step = 0
    calibration_evaluations = 0
    with handle.output_training_scope() as scoped_parameters:
        if tuple(scoped_parameters) != parameters:
            raise AssignmentLocalQKVOResidualTrainingError(
                "Output training scope returned an unexpected parameter order."
            )
        initial = evaluate_functional_action_batches(
            handle.model,
            calibration_values,
            expected_split="calibration-selection",
            hard_action_weight=hard_weight,
            distribution_weight=distribution_weight_value,
            temperature=temperature_value,
            used_for_selection=True,
        )
        calibration_evaluations = 1
        best = initial
        best_key = _functional_selection_key(initial)
        best_state = tuple(value.detach().cpu().clone() for value in parameters)
        calibration_history.append(
            {
                "step": 0,
                "evaluation": initial.as_dict(),
                "discrete_selection_key": list(best_key),
                "promoted": False,
                "incumbent": True,
                "adapter_state_sha256": adapter_state_sha256(),
                "parameter_delta_norm_by_branch": parameter_delta_norms(),
            }
        )
        for step in range(1, step_count + 1):
            batch = fit_values[(step - 1) % len(fit_values)]
            total, _hard, _distribution = _functional_losses(
                handle.model,
                batch,
                hard_action_weight=hard_weight,
                distribution_weight=distribution_weight_value,
                temperature=temperature_value,
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
                    torch.sum(value.detach().to(torch.float32) ** 2)
                    for value in gradients
                )
            )
            total_gradient_norm = float(norm.item())
            scale = min(1.0, gradient_bound / max(total_gradient_norm, 1e-12))
            branch_gradient_norms: dict[str, float] = {}
            for role in ("q", "k", "v", "o"):
                squared = sum(
                    torch.sum(gradient.detach().to(torch.float32) ** 2)
                    for (name, _parameter), gradient in zip(
                        named_parameters, gradients, strict=True
                    )
                    if name.startswith(f"{role}_")
                )
                branch_norm = float(torch.sqrt(squared).item())
                branch_gradient_norms[role] = branch_norm
                branch_gradient_values[role].append(branch_norm)
            for name, gradient in zip(parameter_names, gradients, strict=True):
                if (
                    float(
                        torch.linalg.vector_norm(
                            gradient.detach().to(torch.float32)
                        ).item()
                    )
                    > 0.0
                ):
                    parameter_gradient_activation[name] = True
            with torch.no_grad():
                for parameter, gradient, first, second, learning_rate in zip(
                    parameters,
                    gradients,
                    first_moments,
                    second_moments,
                    parameter_lrs,
                    strict=True,
                ):
                    clipped = gradient.detach().to(torch.float32) * scale
                    first.mul_(beta1_value).add_(clipped, alpha=1.0 - beta1_value)
                    second.mul_(beta2_value).addcmul_(
                        clipped, clipped, value=1.0 - beta2_value
                    )
                    corrected_first = first / (1.0 - beta1_value**step)
                    corrected_second = second / (1.0 - beta2_value**step)
                    update = corrected_first / (torch.sqrt(corrected_second) + epsilon)
                    parameter.add_(
                        update.to(device=parameter.device, dtype=parameter.dtype),
                        alpha=-learning_rate,
                    )
            handle.adapter_registry._assert_finite()
            delta_norms = parameter_delta_norms()
            for role, value in delta_norms.items():
                branch_delta_values[role].append(value)
            fit_history.append(
                {
                    "step": step,
                    "sequence_id": batch.sequence_id,
                    "prefix_source": batch.prefix_source,
                    "batch_size": batch.batch_size,
                    "total_loss": float(total.detach().item()),
                    "hard_action_loss": float(_hard.detach().item()),
                    "distribution_loss": float(_distribution.detach().item()),
                    "gradient_norm_total": total_gradient_norm,
                    "gradient_clip_scale": scale,
                    "gradient_norm_by_branch": branch_gradient_norms,
                    "parameter_delta_norm_by_branch": delta_norms,
                    "nonzero_gradient_by_branch": {
                        role: value > 0.0
                        for role, value in branch_gradient_norms.items()
                    },
                }
            )
            if step not in selection_step_set:
                continue
            calibration = evaluate_functional_action_batches(
                handle.model,
                calibration_values,
                expected_split="calibration-selection",
                hard_action_weight=hard_weight,
                distribution_weight=distribution_weight_value,
                temperature=temperature_value,
                used_for_selection=True,
            )
            calibration_evaluations += 1
            key = _functional_selection_key(calibration)
            promoted = key < best_key
            if promoted:
                best_key = key
                best = calibration
                best_step = step
                best_state = tuple(value.detach().cpu().clone() for value in parameters)
            calibration_history.append(
                {
                    "step": step,
                    "evaluation": calibration.as_dict(),
                    "discrete_selection_key": list(key),
                    "promoted": promoted,
                    "incumbent": False,
                    "adapter_state_sha256": adapter_state_sha256(),
                    "parameter_delta_norm_by_branch": delta_norms,
                }
            )
        prelock_final_state_sha256 = adapter_state_sha256()
        prelock_final_delta_norms = parameter_delta_norms()
        with torch.no_grad():
            for parameter, selected in zip(parameters, best_state, strict=True):
                parameter.copy_(
                    selected.to(device=parameter.device, dtype=parameter.dtype)
                )
        final = evaluate_functional_action_batches(
            handle.model,
            calibration_values,
            expected_split="calibration-selection",
            hard_action_weight=hard_weight,
            distribution_weight=distribution_weight_value,
            temperature=temperature_value,
            used_for_selection=True,
        )
        calibration_evaluations += 1
        if _functional_selection_key(final) != best_key:
            raise AssignmentLocalQKVOResidualTrainingError(
                "Locked QKVO state does not reproduce its calibration winner."
            )
    if initial is None or best is None or final is None:
        raise AssignmentLocalQKVOResidualTrainingError(
            "QKVO training did not produce a locked calibration state."
        )
    handle.assert_live()
    locked_delta_norms = parameter_delta_norms()

    def scalar_summary(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {"minimum": 0.0, "maximum": 0.0, "mean": 0.0, "final": 0.0}
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
            "final": values[-1],
        }

    fit_loss_summary = {
        name: scalar_summary([float(row[name]) for row in fit_history])
        for name in ("total_loss", "hard_action_loss", "distribution_loss")
    }
    branch_gradient_summary = {
        role: {
            **scalar_summary(values),
            "nonzero_step_count": sum(value > 0.0 for value in values),
            "all_steps_nonzero": all(value > 0.0 for value in values),
        }
        for role, values in branch_gradient_values.items()
    }
    branch_parameter_delta_summary = {
        role: {
            "maximum_over_fit": max(values, default=0.0),
            "prelock_final": prelock_final_delta_norms[role],
            "locked": locked_delta_norms[role],
        }
        for role, values in branch_delta_values.items()
    }
    all_branches_active = all(
        summary["nonzero_step_count"] > 0
        for summary in branch_gradient_summary.values()
    )
    all_tensors_active = all(parameter_gradient_activation.values())
    return AssignmentLocalQKVODisjointTrainingReport(
        steps_requested=step_count,
        best_step=best_step,
        fit_batch_count=len(fit_values),
        fit_example_count=sum(value.batch_size for value in fit_values),
        calibration_batch_count=len(calibration_values),
        calibration_example_count=sum(value.batch_size for value in calibration_values),
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
        q_learning_rate=branch_lrs["q"],
        k_learning_rate=branch_lrs["k"],
        v_learning_rate=branch_lrs["v"],
        o_learning_rate=branch_lrs["o"],
        beta1=beta1_value,
        beta2=beta2_value,
        adam_epsilon=epsilon,
        max_gradient_norm=gradient_bound,
        hard_action_weight=hard_weight,
        distribution_weight=distribution_weight_value,
        temperature=temperature_value,
        initial_calibration=initial,
        best_calibration=best,
        final_calibration=final,
        calibration_selection_steps=selection_steps,
        calibration_evaluations=calibration_evaluations,
        winner_reproduction_evaluations=1,
        fit_gradient_calls=step_count,
        autograd_calls=step_count,
        optimizer_constructed=False,
        optimizer_state_saved=False,
        manual_optimizer_state_tensors=2 * len(parameters),
        heldout_evaluation_calls=0,
        adapter_state_sha256=_state_sha256(_state_hashes(handle.adapter_registry)),
        fit_prefix_domain_sha256=fit_prefix_hash,
        calibration_prefix_domain_sha256=calibration_prefix_hash,
        branch_parameter_counts=branch_counts,
        calibration_history=tuple(calibration_history),
        fit_step_history=tuple(fit_history),
        fit_loss_summary=MappingProxyType(fit_loss_summary),
        branch_gradient_summary=MappingProxyType(branch_gradient_summary),
        branch_parameter_delta_summary=MappingProxyType(branch_parameter_delta_summary),
        parameter_gradient_activation=MappingProxyType(parameter_gradient_activation),
        all_branches_received_nonzero_gradient=all_branches_active,
        all_parameter_tensors_received_nonzero_gradient=all_tensors_active,
        prelock_final_adapter_state_sha256=prelock_final_state_sha256,
    )


def install_assignment_local_qkvo_residual_adapter(
    model: nn.Module,
    attention_registry: AttentionCodebookRegistry,
    *,
    sites: Sequence[RedirectedAttentionSite],
    rank: int,
    owner_name: str = "assignment_local_qkvo_residual_registry",
    initialization_seed: int = 0,
    require_qwen3_4b_dimensions: bool = False,
    role_scale_table_new_for_redirect: bool = True,
) -> InstalledAssignmentLocalQKVOResidualAdapter:
    """Audit physical redirects and install sparse Q/K/V/O correction views."""

    if not isinstance(model, nn.Module):
        raise AssignmentLocalQKVOResidualTopologyError("model must be an nn.Module.")
    if (
        not isinstance(attention_registry, AttentionCodebookRegistry)
        or not isinstance(owner_name, str)
        or not owner_name.isidentifier()
        or hasattr(model, owner_name)
    ):
        raise AssignmentLocalQKVOResidualTopologyError(
            "Registry and unused owner_name are required."
        )
    site_values = tuple(sites)
    if any(not isinstance(site, RedirectedAttentionSite) for site in site_values):
        raise AssignmentLocalQKVOResidualTopologyError(
            "sites must contain RedirectedAttentionSite values."
        )
    indices = _site_indices(
        site_values,
        logical_layers=attention_registry.logical_layers,
        physical_groups=attention_registry.key_value_heads,
    )
    audits = tuple(
        audit_redirected_attention_site(attention_registry, site)
        for site in site_values
    )
    if attention_registry.query_heads % attention_registry.key_value_heads:
        raise AssignmentLocalQKVOResidualTopologyError(
            "Query heads must divide evenly across physical KV groups."
        )
    query_group_width = (
        attention_registry.query_heads
        // attention_registry.key_value_heads
        * attention_registry.head_dim
    )
    hidden = attention_registry.output_width
    dimensions = (
        hidden,
        query_group_width,
        attention_registry.head_dim,
        attention_registry.logical_layers,
        attention_registry.key_value_heads,
    )
    if require_qwen3_4b_dimensions and dimensions != (2560, 512, 128, 36, 8):
        raise AssignmentLocalQKVOResidualTopologyError(
            "Exact Qwen3-4B QKVO adapter requires h=2560,m=512,d=128,L=36,G=8."
        )
    dtype = attention_registry.prototype_q_weight.dtype
    device = attention_registry.prototype_q_weight.device
    adapter = SparseAssignmentLocalQKVOResidualRegistry(
        indices,
        logical_layers=attention_registry.logical_layers,
        physical_groups=attention_registry.key_value_heads,
        hidden_size=hidden,
        query_group_width=query_group_width,
        head_dim=attention_registry.head_dim,
        rank=rank,
        dtype=dtype,
        device=device,
        initialization_seed=initialization_seed,
    )
    if (
        not adapter.starts_as_exact_zero_correction
        or not adapter.input_factors_are_nonzero
    ):
        raise AssignmentLocalQKVOResidualTopologyError(
            "QKVO adapter requires zero output factors and random nonzero input factors."
        )
    layers = _model_layers(model)
    if len(layers) != attention_registry.logical_layers:
        raise AssignmentLocalQKVOResidualTopologyError(
            "Model layer count disagrees with the attention registry."
        )
    original: dict[tuple[int, ProjectionRole], AttentionCodebookProjectionView] = {}
    installed: dict[
        tuple[int, ProjectionRole], AssignmentLocalQKVOResidualProjection
    ] = {}
    for layer_index in sorted({site.layer_index for site in site_values}):
        attention = getattr(layers[layer_index], "self_attn", None)
        if not isinstance(attention, nn.Module):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Target layer has no self-attention module."
            )
        if require_qwen3_4b_dimensions and (
            not isinstance(getattr(attention, "q_norm", None), nn.Module)
            or not isinstance(getattr(attention, "k_norm", None), nn.Module)
            or getattr(attention, "layer_idx", None) != layer_index
        ):
            raise AssignmentLocalQKVOResidualTopologyError(
                "Exact Qwen installation cannot prove the pre-norm/cache boundary."
            )
        for role in SparseAssignmentLocalQKVOResidualRegistry._ROLES:
            projection = getattr(attention, f"{role}_proj", None)
            if not isinstance(projection, AttentionCodebookProjectionView):
                raise AssignmentLocalQKVOResidualTopologyError(
                    f"Target {role.upper()} projection is not a codebook view."
                )
            key = (layer_index, role)
            original[key] = projection
            installed[key] = AssignmentLocalQKVOResidualProjection(
                projection,
                adapter,
                attention_registry,
                layer_index,
                role,
            )
    registry_hashes = _state_hashes(attention_registry)
    registry_identity = _tensor_identity_sha256(attention_registry)
    before_count = _unique_parameter_count(model)
    try:
        model.add_module(owner_name, adapter)
        for (layer_index, role), view in installed.items():
            setattr(layers[layer_index].self_attn, f"{role}_proj", view)
        installed_count = _unique_parameter_count(model)
        if installed_count != before_count + adapter.learned_parameter_count:
            raise AssignmentLocalQKVOResidualTopologyError(
                "Installed model count is not base plus sparse QKVO factors."
            )
        accounting = AssignmentLocalQKVOResidualAccounting.build(
            hidden_size=hidden,
            query_group_width=query_group_width,
            head_dim=attention_registry.head_dim,
            logical_layers=attention_registry.logical_layers,
            physical_groups=attention_registry.key_value_heads,
            rank=rank,
            site_count=len(site_values),
            checkpoint_element_size=torch.tensor([], dtype=dtype).element_size(),
            q_prototype_bias=attention_registry.prototype_q_bias is not None,
            k_prototype_bias=attention_registry.prototype_k_bias is not None,
            v_prototype_bias=attention_registry.prototype_v_bias is not None,
            role_scale_table_new_for_redirect=role_scale_table_new_for_redirect,
            schema_buffer_bytes=adapter.schema_buffer_bytes,
        )
        if accounting.adapter_parameters != adapter.learned_parameter_count:
            raise AssignmentLocalQKVOResidualAccountingError(
                "Runtime QKVO factors disagree with closed-form accounting."
            )
        report = AssignmentLocalQKVOResidualInstallationReport(
            sites=site_values,
            redirect_audits=audits,
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
        handle = InstalledAssignmentLocalQKVOResidualAdapter(
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
        for (layer_index, role), projection in original.items():
            setattr(layers[layer_index].self_attn, f"{role}_proj", projection)
        if getattr(model, owner_name, None) is adapter:
            delattr(model, owner_name)
        raise


def install_p35_l19g4_to_l21g6_qkvo_residual_adapter(
    model: nn.Module,
    attention_registry: AttentionCodebookRegistry,
    *,
    rank: int = 8,
    owner_name: str = "assignment_local_qkvo_residual_registry",
    initialization_seed: int = 0,
) -> InstalledAssignmentLocalQKVOResidualAdapter:
    """Install the exact first fallback operator for p35 L19/G4."""

    return install_assignment_local_qkvo_residual_adapter(
        model,
        attention_registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=rank,
        owner_name=owner_name,
        initialization_seed=initialization_seed,
        require_qwen3_4b_dimensions=True,
        role_scale_table_new_for_redirect=True,
    )


__all__ = [
    "ASSIGNMENT_LOCAL_QKVO_ACCOUNTING_SCHEMA",
    "ASSIGNMENT_LOCAL_QKVO_DISJOINT_TRAINING_SCHEMA",
    "ASSIGNMENT_LOCAL_QKVO_INSTALLATION_SCHEMA",
    "ASSIGNMENT_LOCAL_QKVO_RESIDUAL_SCHEMA",
    "ASSIGNMENT_LOCAL_QKVO_TRAINING_SCOPE_SCHEMA",
    "AdapterOnlyOutputTrainingScope",
    "AssignmentLocalQKVODisjointTrainingReport",
    "AssignmentLocalQKVOResidualAccounting",
    "AssignmentLocalQKVOResidualAccountingError",
    "AssignmentLocalQKVOResidualError",
    "AssignmentLocalQKVOResidualInstallationReport",
    "AssignmentLocalQKVOResidualProjection",
    "AssignmentLocalQKVOResidualTopologyError",
    "AssignmentLocalQKVOResidualTrainingError",
    "InstalledAssignmentLocalQKVOResidualAdapter",
    "ProjectionRole",
    "SparseAssignmentLocalQKVOResidualRegistry",
    "install_assignment_local_qkvo_residual_adapter",
    "install_p35_l19g4_to_l21g6_qkvo_residual_adapter",
    "train_assignment_local_qkvo_residual_disjoint",
]
