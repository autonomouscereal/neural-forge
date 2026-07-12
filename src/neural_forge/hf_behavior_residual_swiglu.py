"""Behavior-trained residual SwiGLU paths for folded MLP layers.

The operator in this module deliberately has no parent-weight or internal-state
target.  A selected folded MLP receives a small parallel path::

    y = folded_mlp(x) + down_r(silu(gate_r(x)) * up_r(x))

``down_r`` starts at exact zero while ``gate_r`` and ``up_r`` use deterministic
random features.  Installation is therefore functionally identical to the
folded child, but the first output-space gradient is useful.  A promoted path
can be represented by only three ordinary tensors.  The dense bake helper is a
local equivalence proof for an already dense target MLP; it is not a compact
codebook publication implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_forge.hf_functional_residual_adapter import (
    FunctionalActionBatch,
    FunctionalActionEvaluation,
    _functional_losses,
    evaluate_functional_action_batches,
)
from neural_forge.hf_neuron_codebook import (
    NeuronCodebookLayerView,
    NeuronCodebookRegistry,
)
from neural_forge.hf_qwen_codebook_model import _tensor_sha256 as _checkpoint_hash


BEHAVIOR_RESIDUAL_SWIGLU_SCHEMA = "neural-forge-behavior-residual-swiglu-v1"
BEHAVIOR_RESIDUAL_SWIGLU_ACCOUNTING_SCHEMA = (
    "neural-forge-behavior-residual-swiglu-accounting-v1"
)
BEHAVIOR_RESIDUAL_SWIGLU_INSTALLATION_SCHEMA = (
    "neural-forge-behavior-residual-swiglu-installation-v1"
)
BEHAVIOR_RESIDUAL_SWIGLU_TRAINING_SCHEMA = (
    "neural-forge-behavior-residual-swiglu-output-training-v3"
)


class BehaviorResidualSwiGLUError(RuntimeError):
    """A residual SwiGLU contract is invalid."""


class BehaviorResidualSwiGLUTopologyError(BehaviorResidualSwiGLUError):
    """A model layer, codebook, installation, or rollback is invalid."""


class BehaviorResidualSwiGLUTrainingError(BehaviorResidualSwiGLUError):
    """Output-only training violated its bounded or frozen-base contract."""


class BehaviorResidualSwiGLUAccountingError(BehaviorResidualSwiGLUError):
    """The residual consumes its claimed structural saving."""


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise BehaviorResidualSwiGLUError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise BehaviorResidualSwiGLUError(f"{name} must be an integer.") from error
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise BehaviorResidualSwiGLUError(f"{name} must be {qualifier}.")
    return result


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BehaviorResidualSwiGLUTrainingError(
            f"{name} must be finite and positive."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise BehaviorResidualSwiGLUTrainingError(
            f"{name} must be finite and positive."
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise BehaviorResidualSwiGLUTrainingError(
            f"{name} must be finite and positive."
        )
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
        raise BehaviorResidualSwiGLUError(
            "Residual SwiGLU provenance is not canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    payload = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes": hashlib.sha256(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    return _canonical_sha256(payload)


def _unique_parameter_count(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def _model_layers(model: nn.Module) -> Sequence[nn.Module]:
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or not layers:
        raise BehaviorResidualSwiGLUTopologyError(
            "Model must expose model.layers or layers."
        )
    return layers


def _parameter_guard(
    parameters: Sequence[nn.Parameter],
) -> tuple[tuple[nn.Parameter, int, int, int], ...]:
    return tuple(
        (value, id(value), int(value.data_ptr()), int(value._version))
        for value in parameters
    )


def _guard_sha256(
    values: Sequence[tuple[nn.Parameter, int, int, int]],
) -> str:
    return _canonical_sha256(
        [
            {
                "identity": identity,
                "data_pointer": pointer,
                "version": version,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
            }
            for parameter, identity, pointer, version in values
        ]
    )


def _assert_guard(
    values: Sequence[tuple[nn.Parameter, int, int, int]],
) -> None:
    if any(
        id(parameter) != identity
        or int(parameter.data_ptr()) != pointer
        or int(parameter._version) != version
        for parameter, identity, pointer, version in values
    ):
        raise BehaviorResidualSwiGLUTopologyError(
            "A frozen base parameter changed during residual SwiGLU work."
        )


@dataclass(frozen=True, slots=True)
class MLPCodebookRedirectAudit:
    logical_layers: int
    intermediate_size: int
    prototypes_per_neuron: int
    prototype_count: int
    redirected_assignments_by_layer: tuple[int, ...]
    selected_layers: tuple[int, ...]
    selected_redirected_assignments: int
    assignment_sha256: str
    prototype_source_indices_sha256: str

    @property
    def total_redirected_assignments(self) -> int:
        return sum(self.redirected_assignments_by_layer)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "selected_layers"
            },
            "selected_layers": list(self.selected_layers),
            "total_redirected_assignments": self.total_redirected_assignments,
            "each_missing_prototype_redirected_once": (
                self.total_redirected_assignments == self.intermediate_size
            ),
        }
        payload["audit_sha256"] = _canonical_sha256(payload)
        return payload


def audit_mlp_codebook_redirects(
    registry: NeuronCodebookRegistry,
    *,
    selected_layers: Sequence[int],
    expected_assignment_sha256: str | None = None,
    expected_prototype_source_sha256: str | None = None,
) -> MLPCodebookRedirectAudit:
    """Decode real source-layer redirects from a neuron codebook registry."""

    if not isinstance(registry, NeuronCodebookRegistry):
        raise BehaviorResidualSwiGLUTopologyError(
            "registry must be a NeuronCodebookRegistry."
        )
    layers = registry.logical_layers
    intermediate = registry.intermediate_width
    per_neuron = registry.prototypes_per_neuron
    if per_neuron is None or per_neuron >= layers:
        raise BehaviorResidualSwiGLUTopologyError(
            "Audit requires a compressed aligned-index neuron codebook."
        )
    selected = tuple(_index(value, "selected layer") for value in selected_layers)
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(value >= layers for value in selected)
    ):
        raise BehaviorResidualSwiGLUTopologyError(
            "selected_layers must be sorted, unique, nonempty, and in range."
        )
    assignments = registry.assignments.detach().cpu().to(torch.int64)
    sources = registry.prototype_source_indices.detach().cpu().to(torch.int64)
    if sources.ndim != 1 or int(sources.numel()) != registry.prototype_count:
        raise BehaviorResidualSwiGLUTopologyError(
            "MLP codebook source provenance must be one global rank-1 vector."
        )
    counts: list[int] = []
    for layer_index in range(layers):
        source_flat = sources.index_select(0, assignments[layer_index])
        source_layers = torch.div(source_flat, intermediate, rounding_mode="floor")
        counts.append(int((source_layers != layer_index).sum().item()))
    assignment_hash = _checkpoint_hash(registry.assignments)
    source_hash = _checkpoint_hash(registry.prototype_source_indices)
    if expected_assignment_sha256 is not None and (
        assignment_hash != expected_assignment_sha256
    ):
        raise BehaviorResidualSwiGLUTopologyError(
            "Neuron assignment hash differs from the required p35 artifact."
        )
    if expected_prototype_source_sha256 is not None and (
        source_hash != expected_prototype_source_sha256
    ):
        raise BehaviorResidualSwiGLUTopologyError(
            "Neuron prototype-source hash differs from the required p35 artifact."
        )
    audit = MLPCodebookRedirectAudit(
        logical_layers=layers,
        intermediate_size=intermediate,
        prototypes_per_neuron=per_neuron,
        prototype_count=registry.prototype_count,
        redirected_assignments_by_layer=tuple(counts),
        selected_layers=selected,
        selected_redirected_assignments=sum(counts[index] for index in selected),
        assignment_sha256=assignment_hash,
        prototype_source_indices_sha256=source_hash,
    )
    missing_per_neuron = layers - per_neuron
    if audit.total_redirected_assignments != intermediate * missing_per_neuron:
        raise BehaviorResidualSwiGLUTopologyError(
            "Redirect count does not equal the number of missing depth prototypes."
        )
    return audit


@dataclass(frozen=True, slots=True)
class BehaviorResidualSwiGLUAccounting:
    hidden_size: int
    residual_width: int
    site_count: int
    removed_prototype_count: int
    selected_redirected_assignments: int
    checkpoint_element_size: int
    removed_prototype_parameters: int
    selected_redirect_parameter_budget: int
    residual_parameters_per_site: int
    residual_parameters: int
    net_parameters_saved_vs_unfolded_codebook: int
    selected_budget_after_residual: int
    learned_payload_bytes_saved: int
    layer_index_buffer_bytes: int
    flops_per_token: int

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        residual_width: int,
        site_count: int,
        removed_prototype_count: int,
        selected_redirected_assignments: int,
        checkpoint_element_size: int = 2,
    ) -> BehaviorResidualSwiGLUAccounting:
        values = {
            name: _index(value, name, positive=True)
            for name, value in {
                "hidden_size": hidden_size,
                "residual_width": residual_width,
                "site_count": site_count,
                "removed_prototype_count": removed_prototype_count,
                "selected_redirected_assignments": (selected_redirected_assignments),
                "checkpoint_element_size": checkpoint_element_size,
            }.items()
        }
        hidden = values["hidden_size"]
        width = values["residual_width"]
        sites = values["site_count"]
        removed = values["removed_prototype_count"] * 3 * hidden
        selected_budget = values["selected_redirected_assignments"] * 3 * hidden
        per_site = 3 * hidden * width
        residual = sites * per_site
        net = removed - residual
        selected_net = selected_budget - residual
        if net <= 0 or selected_net <= 0:
            raise BehaviorResidualSwiGLUAccountingError(
                "Residual SwiGLU consumes its removed-prototype budget."
            )
        element_size = values["checkpoint_element_size"]
        return cls(
            hidden_size=hidden,
            residual_width=width,
            site_count=sites,
            removed_prototype_count=values["removed_prototype_count"],
            selected_redirected_assignments=values["selected_redirected_assignments"],
            checkpoint_element_size=element_size,
            removed_prototype_parameters=removed,
            selected_redirect_parameter_budget=selected_budget,
            residual_parameters_per_site=per_site,
            residual_parameters=residual,
            net_parameters_saved_vs_unfolded_codebook=net,
            selected_budget_after_residual=selected_net,
            learned_payload_bytes_saved=net * element_size,
            layer_index_buffer_bytes=sites
            * torch.tensor([], dtype=torch.int32).element_size(),
            flops_per_token=sites * (6 * hidden * width + 2 * width),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": BEHAVIOR_RESIDUAL_SWIGLU_ACCOUNTING_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "removed_qwen_neuron_formula": "3*hidden_size",
            "residual_formula": "site_count*3*hidden_size*residual_width",
            "gate_up_down_bias_parameters": 0,
            "optimizer_state_counted": False,
            "checkpoint_claimed": False,
        }
        payload["accounting_sha256"] = _canonical_sha256(payload)
        return payload


class ResidualSwiGLUBranch(nn.Module):
    """A tiny native SwiGLU path with exact-zero output initialization."""

    def __init__(
        self,
        hidden_size: int,
        residual_width: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        self.hidden_size = _index(hidden_size, "hidden_size", positive=True)
        self.residual_width = _index(residual_width, "residual_width", positive=True)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise BehaviorResidualSwiGLUTopologyError(
                "Residual dtype must be floating point."
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_index(initialization_seed, "initialization_seed"))
        bound = 1.0 / math.sqrt(self.hidden_size)
        gate = torch.empty((self.residual_width, self.hidden_size), dtype=torch.float32)
        up = torch.empty_like(gate)
        gate.uniform_(-bound, bound, generator=generator)
        up.uniform_(-bound, bound, generator=generator)
        self.gate_weight = nn.Parameter(gate.to(device=device, dtype=dtype))
        self.up_weight = nn.Parameter(up.to(device=device, dtype=dtype))
        self.down_weight = nn.Parameter(
            torch.zeros(
                (self.hidden_size, self.residual_width),
                dtype=dtype,
                device=device,
            )
        )
        self.assert_finite()

    @property
    def starts_as_exact_zero_correction(self) -> bool:
        return int(torch.count_nonzero(self.down_weight.detach()).item()) == 0

    @property
    def learned_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    def assert_finite(self) -> None:
        if any(
            not bool(torch.isfinite(value.detach()).all())
            for value in self.parameters()
        ):
            raise BehaviorResidualSwiGLUError(
                "Residual SwiGLU contains NaN or infinity."
            )

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor) or int(inputs.shape[-1]) != self.hidden_size:
            raise BehaviorResidualSwiGLUTopologyError(
                "Residual SwiGLU input has the wrong hidden width."
            )
        gate = F.linear(inputs, self.gate_weight)
        up = F.linear(inputs, self.up_weight)
        return F.linear(F.silu(gate) * up, self.down_weight)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": BEHAVIOR_RESIDUAL_SWIGLU_SCHEMA,
            "hidden_size": self.hidden_size,
            "residual_width": self.residual_width,
            "learned_parameter_count": self.learned_parameter_count,
            "gate_weight_shape": list(self.gate_weight.shape),
            "up_weight_shape": list(self.up_weight.shape),
            "down_weight_shape": list(self.down_weight.shape),
            "bias_parameters": 0,
            "exact_zero_output_initialization": (self.starts_as_exact_zero_correction),
            "parent_runtime_references": 0,
            "parent_or_internal_numeric_target_api": False,
        }
        payload["branch_sha256"] = _canonical_sha256(payload)
        return payload


class BehaviorResidualSwiGLUMLP(nn.Module):
    """One folded MLP plus one behavior-trained parallel SwiGLU path."""

    def __init__(
        self,
        base_mlp: nn.Module,
        residual: ResidualSwiGLUBranch,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        if not isinstance(base_mlp, nn.Module) or not isinstance(
            residual, ResidualSwiGLUBranch
        ):
            raise BehaviorResidualSwiGLUTopologyError(
                "A base MLP and residual branch are required."
            )
        self.base_mlp = base_mlp
        self.residual = residual
        self.layer_index = _index(layer_index, "layer_index")
        self.train(base_mlp.training)

    def forward(self, hidden_states: Tensor) -> Tensor:
        base = self.base_mlp(hidden_states)
        correction = self.residual(hidden_states).to(dtype=base.dtype)
        if tuple(base.shape) != tuple(correction.shape):
            raise BehaviorResidualSwiGLUTopologyError(
                "Base and residual MLP outputs do not align."
            )
        return base + correction


class BakedDenseSwiGLU(nn.Module):
    """A plain dense SwiGLU used to prove residual-neuron bake equivalence."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype, device=device
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def _materialized_base_weights(
    base_mlp: nn.Module,
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(base_mlp, NeuronCodebookLayerView):
        raise BehaviorResidualSwiGLUTopologyError(
            "Dense baking a codebook view would duplicate a full logical layer; "
            "compact codebook publication is not implemented."
        )
    projections = tuple(
        getattr(base_mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj")
    )
    if not all(isinstance(value, nn.Linear) for value in projections):
        raise BehaviorResidualSwiGLUTopologyError(
            "Baking requires a dense or neuron-codebook SwiGLU base."
        )
    gate, up, down = projections
    if any(value.bias is not None for value in (gate, up, down)):
        raise BehaviorResidualSwiGLUTopologyError(
            "Baking currently requires bias-free projections."
        )
    return gate.weight, up.weight, down.weight


def _assert_silu_base(base_mlp: nn.Module) -> None:
    if isinstance(base_mlp, NeuronCodebookLayerView):
        if base_mlp.registry.activation != "silu":
            raise BehaviorResidualSwiGLUTopologyError(
                "Residual SwiGLU requires a SiLU neuron-codebook base."
            )
        return
    activation = getattr(base_mlp, "act_fn", None)
    name = (
        ""
        if activation is None
        else f"{type(activation).__module__}.{type(activation).__qualname__}".lower()
    )
    if not isinstance(activation, nn.SiLU) and "silu" not in name:
        raise BehaviorResidualSwiGLUTopologyError(
            "Dense residual SwiGLU requires an explicitly identifiable SiLU act_fn."
        )


def bake_behavior_residual_swiglu(
    installed: BehaviorResidualSwiGLUMLP,
) -> BakedDenseSwiGLU:
    """Prove local equivalence by concatenating into an already dense MLP.

    This helper deliberately rejects a neuron-codebook view.  Materializing
    that view while retaining the global registry would duplicate a complete
    logical layer and invalidate the residual-only parameter accounting.
    """

    if not isinstance(installed, BehaviorResidualSwiGLUMLP):
        raise BehaviorResidualSwiGLUTopologyError(
            "installed must be a BehaviorResidualSwiGLUMLP."
        )
    _assert_silu_base(installed.base_mlp)
    gate, up, down = _materialized_base_weights(installed.base_mlp)
    residual = installed.residual
    if (
        tuple(gate.shape) != tuple(up.shape)
        or int(gate.shape[1]) != residual.hidden_size
        or tuple(down.shape) != (residual.hidden_size, int(gate.shape[0]))
    ):
        raise BehaviorResidualSwiGLUTopologyError(
            "Base and residual dimensions cannot be baked together."
        )
    baked = BakedDenseSwiGLU(
        residual.hidden_size,
        int(gate.shape[0]) + residual.residual_width,
        dtype=gate.dtype,
        device=gate.device,
    )
    with torch.no_grad():
        baked.gate_proj.weight.copy_(
            torch.cat((gate.detach(), residual.gate_weight.detach()), dim=0)
        )
        baked.up_proj.weight.copy_(
            torch.cat((up.detach(), residual.up_weight.detach()), dim=0)
        )
        baked.down_proj.weight.copy_(
            torch.cat((down.detach(), residual.down_weight.detach()), dim=1)
        )
    baked.train(installed.training)
    return baked


@dataclass(frozen=True, slots=True)
class BehaviorResidualSwiGLUInstallationReport:
    layer_indices: tuple[int, ...]
    residual_width: int
    residual_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_installed: int
    base_parameter_guard_sha256: str
    accounting: BehaviorResidualSwiGLUAccounting

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": BEHAVIOR_RESIDUAL_SWIGLU_INSTALLATION_SCHEMA,
            "layer_indices": list(self.layer_indices),
            "residual_width": self.residual_width,
            "residual_parameter_count": self.residual_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_installed": (self.model_parameter_count_installed),
            "base_parameter_guard_sha256": self.base_parameter_guard_sha256,
            "accounting": self.accounting.as_dict(),
            "correction_boundary": "parallel-mlp-output-before-residual-add",
            "exact_zero_initialization": True,
            "dense_target_equivalence_bake_helper": True,
            "compact_codebook_publication_bake_implemented": False,
            "native_qwen_config_save_reload_publication_proven": False,
            "parent_or_internal_numeric_similarity_used": False,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


class InstalledBehaviorResidualSwiGLU:
    """Live residual MLP experiment with exact topology rollback."""

    def __init__(
        self,
        *,
        model: nn.Module,
        originals: Mapping[int, nn.Module],
        installed: Mapping[int, BehaviorResidualSwiGLUMLP],
        base_guard: Sequence[tuple[nn.Parameter, int, int, int]],
        parameter_ids_before: frozenset[int],
        report: BehaviorResidualSwiGLUInstallationReport,
    ) -> None:
        self.model = model
        self.originals = dict(originals)
        self.installed = dict(installed)
        self._base_guard = tuple(base_guard)
        self._parameter_ids_before = parameter_ids_before
        self.report = report
        self._rolled_back = False

    @property
    def is_applied(self) -> bool:
        return not self._rolled_back

    def residual_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        self.assert_live()
        return tuple(
            (f"layer_{layer_index}.{name}", parameter)
            for layer_index, wrapper in sorted(self.installed.items())
            for name, parameter in wrapper.residual.named_parameters()
        )

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(value for _name, value in self.residual_named_parameters())

    def residual_state(self) -> dict[str, Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.residual_named_parameters()
        }

    def residual_state_sha256(self) -> str:
        return _canonical_sha256(
            {
                name: _tensor_sha256(value)
                for name, value in sorted(self.residual_state().items())
            }
        )

    def load_residual_state(self, state: Mapping[str, Tensor]) -> None:
        expected = dict(self.residual_named_parameters())
        if set(state) != set(expected):
            raise BehaviorResidualSwiGLUTopologyError(
                "Residual state keys do not exactly match the live installation."
            )
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if (
                    not isinstance(value, Tensor)
                    or tuple(value.shape) != tuple(parameter.shape)
                    or not value.dtype.is_floating_point
                    or not bool(torch.isfinite(value).all())
                ):
                    raise BehaviorResidualSwiGLUTopologyError(
                        f"Residual state tensor {name!r} is invalid."
                    )
                parameter.copy_(
                    value.to(device=parameter.device, dtype=parameter.dtype)
                )
        self.assert_live()

    def zero_output_correction(self) -> None:
        self.assert_live()
        with torch.no_grad():
            for wrapper in self.installed.values():
                wrapper.residual.down_weight.zero_()

    def assert_base_unchanged(self) -> None:
        _assert_guard(self._base_guard)

    def assert_live(self) -> BehaviorResidualSwiGLUInstallationReport:
        if self._rolled_back:
            raise BehaviorResidualSwiGLUTopologyError(
                "A rolled-back residual SwiGLU is no longer live."
            )
        layers = _model_layers(self.model)
        if any(
            layers[index].mlp is not wrapper
            for index, wrapper in self.installed.items()
        ):
            raise BehaviorResidualSwiGLUTopologyError(
                "A live residual SwiGLU wrapper was replaced."
            )
        self.assert_base_unchanged()
        for wrapper in self.installed.values():
            wrapper.residual.assert_finite()
        return self.report

    def output_training_scope(self) -> ResidualSwiGLUOutputTrainingScope:
        return ResidualSwiGLUOutputTrainingScope(self)

    def rollback(self) -> None:
        if self._rolled_back:
            return
        layers = _model_layers(self.model)
        if any(
            layers[index].mlp is not wrapper
            for index, wrapper in self.installed.items()
        ):
            raise BehaviorResidualSwiGLUTopologyError(
                "Residual wrapper ownership changed; exact rollback is unsafe."
            )
        # Disposable residual factors may be non-finite after a failed update.
        # Exact rollback depends only on wrapper ownership and the frozen base.
        self.assert_base_unchanged()
        for index, original in self.originals.items():
            layers[index].mlp = original
        self.assert_base_unchanged()
        if frozenset(id(value) for value in self.model.parameters()) != (
            self._parameter_ids_before
        ):
            raise BehaviorResidualSwiGLUTopologyError(
                "Exact residual SwiGLU rollback changed base parameter ownership."
            )
        if (
            _unique_parameter_count(self.model)
            != self.report.model_parameter_count_before
        ):
            raise BehaviorResidualSwiGLUTopologyError(
                "Exact residual SwiGLU rollback parameter accounting failed."
            )
        if any(
            layers[index].mlp is not original
            for index, original in self.originals.items()
        ):
            raise BehaviorResidualSwiGLUTopologyError(
                "Exact residual SwiGLU topology rollback failed."
            )
        self._rolled_back = True

    def __enter__(self) -> InstalledBehaviorResidualSwiGLU:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.rollback()


class ResidualSwiGLUOutputTrainingScope:
    """Freeze every base tensor and expose only residual-neuron parameters."""

    def __init__(self, handle: InstalledBehaviorResidualSwiGLU) -> None:
        if not isinstance(handle, InstalledBehaviorResidualSwiGLU):
            raise BehaviorResidualSwiGLUTrainingError(
                "Training scope requires a live residual SwiGLU handle."
            )
        self.handle = handle
        self._base_state: tuple[
            tuple[nn.Parameter, bool, Tensor | None, int, int], ...
        ] = ()
        self._modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise BehaviorResidualSwiGLUTrainingError(
                "Residual output training scope cannot be re-entered."
            )
        self.handle.assert_live()
        trainable = self.handle.output_trainable_parameters()
        trainable_ids = {id(value) for value in trainable}
        self._base_state = tuple(
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
        self._modes = tuple(
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
        for module, training in self._modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            or bool(value.requires_grad) != requires_grad
            for value, requires_grad, gradient, version, pointer in self._base_state
        ):
            raise BehaviorResidualSwiGLUTrainingError(
                "A base parameter changed during residual-only output training."
            )
        self.handle.assert_live()


def install_behavior_residual_swiglu(
    model: nn.Module,
    *,
    layer_indices: Sequence[int],
    residual_width: int,
    accounting: BehaviorResidualSwiGLUAccounting,
    initialization_seed: int = 0,
) -> InstalledBehaviorResidualSwiGLU:
    """Install exact-zero residual SwiGLU paths at selected logical layers."""

    if not isinstance(model, nn.Module) or not isinstance(
        accounting, BehaviorResidualSwiGLUAccounting
    ):
        raise BehaviorResidualSwiGLUTopologyError(
            "A model and explicit accounting record are required."
        )
    layers = _model_layers(model)
    selected = tuple(_index(value, "layer index") for value in layer_indices)
    width = _index(residual_width, "residual_width", positive=True)
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(value >= len(layers) for value in selected)
        or accounting.site_count != len(selected)
        or accounting.residual_width != width
    ):
        raise BehaviorResidualSwiGLUTopologyError(
            "Layer selection and residual accounting disagree."
        )
    parameters_before = tuple(model.parameters())
    ids_before = frozenset(id(value) for value in parameters_before)
    guard = _parameter_guard(parameters_before)
    count_before = _unique_parameter_count(model)
    originals: dict[int, nn.Module] = {}
    installed: dict[int, BehaviorResidualSwiGLUMLP] = {}
    try:
        for offset, layer_index in enumerate(selected):
            original = getattr(layers[layer_index], "mlp", None)
            if not isinstance(original, nn.Module):
                raise BehaviorResidualSwiGLUTopologyError(
                    f"Layer {layer_index} lacks an MLP module."
                )
            if isinstance(original, BehaviorResidualSwiGLUMLP):
                raise BehaviorResidualSwiGLUTopologyError(
                    f"Layer {layer_index} already has a residual SwiGLU."
                )
            _assert_silu_base(original)
            if isinstance(original, NeuronCodebookLayerView):
                hidden = original.in_features
                reference = original.registry.prototype_gate_weight
            else:
                gate = getattr(original, "gate_proj", None)
                if not isinstance(gate, nn.Linear):
                    raise BehaviorResidualSwiGLUTopologyError(
                        "Residual installation requires a dense or codebook SwiGLU."
                    )
                hidden = gate.in_features
                reference = gate.weight
            if hidden != accounting.hidden_size:
                raise BehaviorResidualSwiGLUTopologyError(
                    "Live hidden size differs from residual accounting."
                )
            residual = ResidualSwiGLUBranch(
                hidden,
                width,
                dtype=reference.dtype,
                device=reference.device,
                initialization_seed=_index(initialization_seed, "initialization_seed")
                + offset,
            )
            wrapper = BehaviorResidualSwiGLUMLP(
                original, residual, layer_index=layer_index
            )
            originals[layer_index] = original
            installed[layer_index] = wrapper
            layers[layer_index].mlp = wrapper
    except Exception:
        for layer_index, original in originals.items():
            layers[layer_index].mlp = original
        raise
    residual_parameters = sum(
        wrapper.residual.learned_parameter_count for wrapper in installed.values()
    )
    count_installed = _unique_parameter_count(model)
    if (
        residual_parameters != accounting.residual_parameters
        or count_installed != count_before + residual_parameters
        or not all(
            wrapper.residual.starts_as_exact_zero_correction
            for wrapper in installed.values()
        )
    ):
        for layer_index, original in originals.items():
            layers[layer_index].mlp = original
        raise BehaviorResidualSwiGLUTopologyError(
            "Residual installation failed exact-zero parameter accounting."
        )
    report = BehaviorResidualSwiGLUInstallationReport(
        layer_indices=selected,
        residual_width=width,
        residual_parameter_count=residual_parameters,
        model_parameter_count_before=count_before,
        model_parameter_count_installed=count_installed,
        base_parameter_guard_sha256=_guard_sha256(guard),
        accounting=accounting,
    )
    return InstalledBehaviorResidualSwiGLU(
        model=model,
        originals=originals,
        installed=installed,
        base_guard=guard,
        parameter_ids_before=ids_before,
        report=report,
    )


def _action_selection_key(
    evaluation: FunctionalActionEvaluation,
) -> tuple[float, float]:
    return (-evaluation.parent_action_top1_rate, evaluation.mean_parent_action_rank)


def _calibration_rollout_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("output_tokens_only") is not True:
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout selection requires an output-token-only mapping."
        )

    counts: dict[str, int] = {}
    for name in (
        "prompt_count",
        "exact_rollout_matches",
        "token_matches",
        "token_count",
    ):
        candidate = value.get(name)
        if isinstance(candidate, bool):
            raise BehaviorResidualSwiGLUTrainingError(
                f"Calibration rollout {name} must be an integer."
            )
        try:
            counts[name] = operator.index(candidate)
        except (TypeError, ValueError, OverflowError) as error:
            raise BehaviorResidualSwiGLUTrainingError(
                f"Calibration rollout {name} must be an integer."
            ) from error
    if counts["prompt_count"] <= 0 or counts["token_count"] <= 0:
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout prompt_count and token_count must be positive."
        )
    if (
        counts["exact_rollout_matches"] < 0
        or counts["exact_rollout_matches"] > counts["prompt_count"]
        or counts["token_matches"] < 0
        or counts["token_matches"] > counts["token_count"]
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout match counts are outside their declared bounds."
        )
    try:
        mean_lcp = float(value.get("mean_longest_common_prefix"))
    except (TypeError, ValueError, OverflowError) as error:
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout mean_longest_common_prefix must be finite."
        ) from error
    maximum_mean_lcp = counts["token_count"] / counts["prompt_count"]
    if not math.isfinite(mean_lcp) or not 0.0 <= mean_lcp <= maximum_mean_lcp:
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout mean_longest_common_prefix is outside its bounds."
        )
    summary = {
        **counts,
        "exact_rollout_rate": (
            counts["exact_rollout_matches"] / counts["prompt_count"]
        ),
        "token_match_rate": counts["token_matches"] / counts["token_count"],
        "mean_longest_common_prefix": mean_lcp,
        "output_tokens_only": True,
    }
    summary["rollout_summary_sha256"] = _canonical_sha256(summary)
    return summary


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_rollout_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout provenance must be a mapping."
        )
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout provenance must be canonical JSON."
        ) from error
    prompt_hashes = normalized.get("prompt_token_row_sha256s")
    if (
        normalized.get("selection_split") != "calibration-selection"
        or not isinstance(prompt_hashes, list)
        or not prompt_hashes
        or any(not _is_sha256(item) for item in prompt_hashes)
        or normalized.get("prompt_count") != len(prompt_hashes)
        or normalized.get("prompt_token_domain_sha256")
        != _canonical_sha256(prompt_hashes)
        or not _is_sha256(normalized.get("heldout_prompt_token_domain_sha256"))
        or normalized.get("prompt_token_domain_sha256")
        == normalized.get("heldout_prompt_token_domain_sha256")
        or normalized.get("calibration_heldout_token_overlap_count") != 0
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout provenance does not prove its disjoint prompt domain."
        )
    normalized["provenance_sha256"] = _canonical_sha256(
        {key: item for key, item in normalized.items() if key != "provenance_sha256"}
    )
    return normalized


def _calibration_rollout_record(
    value: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _calibration_rollout_summary(value)
    rows = value.get("rows")
    trace_hashes = value.get("trace_hashes")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != summary["prompt_count"]
        or not isinstance(trace_hashes, Sequence)
        or isinstance(trace_hashes, (str, bytes))
        or len(trace_hashes) != summary["prompt_count"]
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout requires one metric and trace-hash row per prompt."
        )

    normalized_rows: list[dict[str, Any]] = []
    normalized_trace_hashes: list[dict[str, Any]] = []
    sequence_ids: set[str] = set()
    aggregate_exact = 0
    aggregate_matches = 0
    aggregate_tokens = 0
    aggregate_lcp = 0
    for row, trace_row in zip(rows, trace_hashes, strict=True):
        if not isinstance(row, Mapping) or not isinstance(trace_row, Mapping):
            raise BehaviorResidualSwiGLUTrainingError(
                "Calibration rollout per-prompt rows must be mappings."
            )
        sequence_id = row.get("sequence_id")
        exact = row.get("exact_rollout_match")
        if (
            not isinstance(sequence_id, str)
            or not sequence_id
            or sequence_id in sequence_ids
            or not isinstance(exact, bool)
        ):
            raise BehaviorResidualSwiGLUTrainingError(
                "Calibration rollout sequence IDs or exact flags are invalid."
            )
        sequence_ids.add(sequence_id)
        row_counts: dict[str, int] = {}
        for name in ("token_matches", "token_count", "longest_common_prefix"):
            candidate = row.get(name)
            if isinstance(candidate, bool):
                raise BehaviorResidualSwiGLUTrainingError(
                    f"Calibration rollout row {name} must be an integer."
                )
            try:
                row_counts[name] = operator.index(candidate)
            except (TypeError, ValueError, OverflowError) as error:
                raise BehaviorResidualSwiGLUTrainingError(
                    f"Calibration rollout row {name} must be an integer."
                ) from error
        if (
            row_counts["token_count"] <= 0
            or not 0 <= row_counts["token_matches"] <= row_counts["token_count"]
            or not 0 <= row_counts["longest_common_prefix"] <= row_counts["token_count"]
            or (exact and row_counts["token_matches"] != row_counts["token_count"])
        ):
            raise BehaviorResidualSwiGLUTrainingError(
                "Calibration rollout per-prompt counts are inconsistent."
            )
        trace_sequence_id = trace_row.get("sequence_id")
        hash_names = (
            "prompt_token_ids_sha256",
            "parent_selected_token_ids_sha256",
            "child_selected_token_ids_sha256",
        )
        if trace_sequence_id != sequence_id or any(
            not _is_sha256(trace_row.get(name)) for name in hash_names
        ):
            raise BehaviorResidualSwiGLUTrainingError(
                "Calibration rollout per-prompt trace provenance is invalid."
            )
        normalized_rows.append(
            {
                "sequence_id": sequence_id,
                "exact_rollout_match": exact,
                **row_counts,
            }
        )
        normalized_trace_hashes.append(
            {
                "sequence_id": sequence_id,
                **{name: trace_row[name] for name in hash_names},
            }
        )
        aggregate_exact += int(exact)
        aggregate_matches += row_counts["token_matches"]
        aggregate_tokens += row_counts["token_count"]
        aggregate_lcp += row_counts["longest_common_prefix"]
    if (
        aggregate_exact != summary["exact_rollout_matches"]
        or aggregate_matches != summary["token_matches"]
        or aggregate_tokens != summary["token_count"]
        or aggregate_lcp / summary["prompt_count"]
        != summary["mean_longest_common_prefix"]
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout aggregates do not reproduce their per-prompt rows."
        )

    comparison = {
        key: value[key]
        for key in (
            "prompt_count",
            "exact_rollout_matches",
            "exact_rollout_rate",
            "token_matches",
            "token_count",
            "token_match_rate",
            "mean_longest_common_prefix",
            "output_tokens_only",
        )
    }
    comparison["rows"] = normalized_rows
    comparison_sha256 = value.get("comparison_sha256")
    if comparison_sha256 != _canonical_sha256(comparison):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout comparison hash does not match its full rows."
        )
    provenance = _normalize_rollout_provenance(value.get("selection_provenance"))
    if (
        provenance != expected_provenance
        or provenance["prompt_count"] != summary["prompt_count"]
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout callback changed its sealed selection provenance."
        )
    record = {
        **comparison,
        "comparison_sha256": comparison_sha256,
        "trace_hashes": normalized_trace_hashes,
        "selection_provenance": provenance,
    }
    if value.get("rollout_evaluation_sha256") != _canonical_sha256(record):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout evaluation hash does not match its full record."
        )
    record["rollout_evaluation_sha256"] = value["rollout_evaluation_sha256"]
    return record


def calibration_rollout_selection_key(
    evaluation: FunctionalActionEvaluation,
    rollout: Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    """Return the rollout-first lexicographic key used by v2 winner locking."""

    if not isinstance(evaluation, FunctionalActionEvaluation):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration selection requires a functional action evaluation."
        )
    summary = _calibration_rollout_summary(rollout)
    return (
        -float(summary["exact_rollout_matches"]),
        -float(summary["token_matches"]),
        -float(summary["mean_longest_common_prefix"]),
        -evaluation.parent_action_top1_rate,
        evaluation.mean_parent_action_rank,
    )


def calibration_rollout_candidate_can_promote(
    incumbent_evaluation: FunctionalActionEvaluation,
    incumbent_rollout: Mapping[str, Any],
    candidate_evaluation: FunctionalActionEvaluation,
    candidate_rollout: Mapping[str, Any],
) -> bool:
    """Require Pareto-safe rollout behavior before action-order tie-breaking."""

    incumbent = _calibration_rollout_summary(incumbent_rollout)
    candidate = _calibration_rollout_summary(candidate_rollout)
    rollout_fields = (
        "exact_rollout_matches",
        "token_matches",
        "mean_longest_common_prefix",
    )
    if any(candidate[name] < incumbent[name] for name in rollout_fields):
        return False
    return calibration_rollout_selection_key(
        candidate_evaluation, candidate
    ) < calibration_rollout_selection_key(incumbent_evaluation, incumbent)


def _selection_key(
    evaluation: FunctionalActionEvaluation,
    rollout: Mapping[str, Any] | None,
) -> tuple[float, ...]:
    if rollout is None:
        return _action_selection_key(evaluation)
    return calibration_rollout_selection_key(evaluation, rollout)


def _assert_same_rollout_domain(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    if any(
        candidate[name] != reference[name]
        for name in (
            "prompt_count",
            "token_count",
            "selection_provenance",
        )
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout evaluator changed its bounded prompt/token domain."
        )


def _prefix_domain(
    batches: Sequence[FunctionalActionBatch],
) -> tuple[frozenset[str], str]:
    hashes: set[str] = set()
    for batch in batches:
        for row_index in range(batch.batch_size):
            length = int(batch.attention_mask[row_index].sum().item())
            row = batch.input_ids[row_index, :length].detach().cpu().tolist()
            hashes.add(_canonical_sha256([int(value) for value in row]))
    result = frozenset(hashes)
    return result, _canonical_sha256(sorted(result))


@dataclass(frozen=True, slots=True)
class BehaviorResidualSwiGLUFitRefresh:
    """One current-student fit phase and its fixed parent output targets."""

    refresh_after_step: int
    batches: tuple[FunctionalActionBatch, ...]
    provenance: Mapping[str, Any]


def _optimizer_moment_sha256(
    first_moments: Sequence[Tensor], second_moments: Sequence[Tensor]
) -> str:
    return _canonical_sha256(
        {
            "first": [_tensor_sha256(value) for value in first_moments],
            "second": [_tensor_sha256(value) for value in second_moments],
        }
    )


def _normalize_fit_phase(
    batches: Sequence[FunctionalActionBatch],
    provenance: Mapping[str, Any] | None,
    *,
    refresh_after_step: int,
    calibration_domain: frozenset[str],
    require_on_policy_provenance: bool,
) -> tuple[tuple[FunctionalActionBatch, ...], Mapping[str, Any]]:
    values = tuple(batches)
    if (
        not values
        or any(value.split != "fit" for value in values)
        or {value.prefix_source for value in values}
        != {"parent-greedy", "student-greedy"}
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Every fit phase requires paired fit-only parent/student prefixes."
        )
    domain, domain_sha256 = _prefix_domain(values)
    if domain & calibration_domain:
        raise BehaviorResidualSwiGLUTrainingError(
            "A refreshed fit phase overlaps the calibration prefix domain."
        )
    batch_manifest_sha256 = _canonical_sha256([value.as_dict() for value in values])
    if provenance is None:
        if require_on_policy_provenance:
            raise BehaviorResidualSwiGLUTrainingError(
                "On-policy fit phases require sealed refresh provenance."
            )
        normalized = {
            "schema": "neural-forge-behavior-residual-fit-phase-derived-v1",
            "refresh_after_step": refresh_after_step,
            "fit_split": "fit",
            "fit_prefix_domain_sha256": domain_sha256,
            "batch_manifest_sha256": batch_manifest_sha256,
            "refreshed_from_current_student_rollout": False,
            "calibration_used_for_training": False,
            "calibration_model_forwards": 0,
            "heldout_model_forwards": 0,
            "parent_output_targets_only": True,
        }
    else:
        try:
            normalized = json.loads(
                json.dumps(
                    dict(provenance),
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as error:
            raise BehaviorResidualSwiGLUTrainingError(
                "Fit refresh provenance must be canonical JSON."
            ) from error
        required_hashes = (
            "fit_prompt_token_domain_sha256",
            "student_driver_trace_sha256",
            "parent_target_trace_sha256",
        )
        if (
            normalized.get("schema")
            != "neural-forge-behavior-residual-on-policy-fit-phase-v1"
            or normalized.get("refresh_after_step") != refresh_after_step
            or normalized.get("fit_split") != "fit"
            or normalized.get("fit_prefix_domain_sha256") != domain_sha256
            or normalized.get("batch_manifest_sha256") != batch_manifest_sha256
            or any(not _is_sha256(normalized.get(name)) for name in required_hashes)
            or normalized.get("refreshed_from_current_student_rollout") is not True
            or normalized.get("calibration_used_for_training") is not False
            or normalized.get("calibration_model_forwards") != 0
            or normalized.get("heldout_model_forwards") != 0
            or normalized.get("parent_output_targets_only") is not True
        ):
            raise BehaviorResidualSwiGLUTrainingError(
                "On-policy fit refresh provenance or batch binding is invalid."
            )
    normalized["phase_provenance_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in normalized.items()
            if key != "phase_provenance_sha256"
        }
    )
    return values, MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class BehaviorResidualSwiGLUTrainingReport:
    steps_requested: int
    best_step: int
    gate_up_learning_rate: float
    down_learning_rate: float
    max_gradient_norm: float
    initial_calibration: FunctionalActionEvaluation
    best_calibration: FunctionalActionEvaluation
    final_calibration: FunctionalActionEvaluation
    initial_calibration_rollout: Mapping[str, Any] | None
    best_calibration_rollout: Mapping[str, Any] | None
    final_calibration_rollout: Mapping[str, Any] | None
    calibration_rollout_provenance: Mapping[str, Any] | None
    calibration_steps: tuple[int, ...]
    fit_gradient_calls: int
    calibration_evaluations: int
    calibration_rollout_evaluations: int
    fit_refresh_steps: tuple[int, ...]
    fit_refresh_evaluations: int
    manual_adam_moment_reset_count: int
    residual_state_sha256: str
    fit_prefix_domain_sha256: str
    calibration_prefix_domain_sha256: str
    parameter_gradient_activation: Mapping[str, bool]
    fit_history: tuple[Mapping[str, Any], ...]
    fit_phase_history: tuple[Mapping[str, Any], ...]
    calibration_history: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": BEHAVIOR_RESIDUAL_SWIGLU_TRAINING_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "initial_calibration",
                    "best_calibration",
                    "final_calibration",
                    "initial_calibration_rollout",
                    "best_calibration_rollout",
                    "final_calibration_rollout",
                    "calibration_rollout_provenance",
                    "parameter_gradient_activation",
                    "fit_history",
                    "fit_phase_history",
                    "calibration_history",
                }
            },
            "initial_calibration": self.initial_calibration.as_dict(),
            "best_calibration": self.best_calibration.as_dict(),
            "final_calibration": self.final_calibration.as_dict(),
            "initial_calibration_rollout": (
                None
                if self.initial_calibration_rollout is None
                else dict(self.initial_calibration_rollout)
            ),
            "best_calibration_rollout": (
                None
                if self.best_calibration_rollout is None
                else dict(self.best_calibration_rollout)
            ),
            "final_calibration_rollout": (
                None
                if self.final_calibration_rollout is None
                else dict(self.final_calibration_rollout)
            ),
            "calibration_rollout_provenance": (
                None
                if self.calibration_rollout_provenance is None
                else dict(self.calibration_rollout_provenance)
            ),
            "parameter_gradient_activation": dict(self.parameter_gradient_activation),
            "fit_history": [dict(value) for value in self.fit_history],
            "fit_phase_history": [dict(value) for value in self.fit_phase_history],
            "calibration_history": [dict(value) for value in self.calibration_history],
            "selection_key": (
                [
                    "maximize-parent-action-top1-rate",
                    "minimize-mean-parent-action-rank",
                ]
                if self.initial_calibration_rollout is None
                else [
                    "maximize-exact-calibration-rollout-matches",
                    "maximize-calibration-rollout-token-matches",
                    "maximize-calibration-mean-longest-common-prefix",
                    "maximize-parent-action-top1-rate",
                    "minimize-mean-parent-action-rank",
                ]
            ),
            "calibration_free_rollout_selection": (
                self.initial_calibration_rollout is not None
            ),
            "rollout_promotion_requires_pareto_nonregression": (
                self.initial_calibration_rollout is not None
            ),
            "selection_prompt_split": "calibration-selection",
            "continuous_loss_can_promote": False,
            "fixed_parent_output_targets_only": True,
            "base_parameters_updated": 0,
            "torch_optimizer_object_constructed": False,
            "manual_adam_state_tensors": 2 * len(self.parameter_gradient_activation),
            "manual_adam_update_steps": self.steps_requested,
            "manual_adam_moments_preserved_across_fit_refreshes": (
                self.manual_adam_moment_reset_count == 0
            ),
            "manual_optimizer_state_saved": False,
            "gradient_recovery_training": True,
            "fold_only": False,
            "generic_trainer_does_not_claim_unobserved_heldout_call_count": True,
            "excluded_objectives": [
                "weight-similarity",
                "activation-similarity",
                "hidden-state-similarity",
                "mlp-output-similarity",
                "residual-stream-similarity",
            ],
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["training_sha256"] = _canonical_sha256(payload)
        return payload


def train_behavior_residual_swiglu_disjoint(
    handle: InstalledBehaviorResidualSwiGLU,
    fit_batches: Sequence[FunctionalActionBatch],
    calibration_batches: Sequence[FunctionalActionBatch],
    *,
    steps: int,
    gate_up_learning_rate: float = 1e-3,
    down_learning_rate: float = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    max_gradient_norm: float = 1.0,
    hard_action_weight: float = 1.0,
    distribution_weight: float = 0.1,
    temperature: float = 1.0,
    calibration_steps: Sequence[int] | None = None,
    calibration_rollout_evaluator: Callable[[], Mapping[str, Any]] | None = None,
    calibration_rollout_provenance: Mapping[str, Any] | None = None,
    fit_refresh_steps: Sequence[int] | None = None,
    fit_batch_refresher: Callable[[int], BehaviorResidualSwiGLUFitRefresh]
    | None = None,
    initial_fit_provenance: Mapping[str, Any] | None = None,
) -> BehaviorResidualSwiGLUTrainingReport:
    """Fit on one prefix domain and lock only by calibration output behavior.

    When supplied, ``calibration_rollout_evaluator`` must evaluate the current
    residual state on calibration prompts only.  It is invoked at every bounded
    selection checkpoint and once after the selected state is reloaded.
    """

    if not isinstance(handle, InstalledBehaviorResidualSwiGLU):
        raise BehaviorResidualSwiGLUTrainingError(
            "Training requires a live residual SwiGLU installation."
        )
    if (calibration_rollout_evaluator is None) != (
        calibration_rollout_provenance is None
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Calibration rollout evaluator and provenance must be supplied together."
        )
    if calibration_rollout_evaluator is not None and not callable(
        calibration_rollout_evaluator
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "calibration_rollout_evaluator must be callable."
        )
    normalized_rollout_provenance = (
        None
        if calibration_rollout_provenance is None
        else MappingProxyType(
            _normalize_rollout_provenance(calibration_rollout_provenance)
        )
    )
    fit = tuple(fit_batches)
    calibration = tuple(calibration_batches)
    if (
        not fit
        or not calibration
        or any(value.split != "fit" for value in fit)
        or any(value.split != "calibration-selection" for value in calibration)
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Fit and calibration batches must use their declared disjoint splits."
        )
    for name, values in (("fit", fit), ("calibration", calibration)):
        if {value.prefix_source for value in values} != {
            "parent-greedy",
            "student-greedy",
        }:
            raise BehaviorResidualSwiGLUTrainingError(
                f"{name} requires paired parent- and student-greedy prefixes."
            )
    fit_domain, fit_hash = _prefix_domain(fit)
    calibration_domain, calibration_hash = _prefix_domain(calibration)
    if fit_domain & calibration_domain:
        raise BehaviorResidualSwiGLUTrainingError(
            "Fit and calibration contain an overlapping exact prefix."
        )
    step_count = _index(steps, "steps", positive=True)
    if step_count > 256:
        raise BehaviorResidualSwiGLUTrainingError(
            "The bounded residual experiment caps training at 256 steps."
        )
    gate_up_lr = _finite_positive(gate_up_learning_rate, "gate_up_learning_rate")
    down_lr = _finite_positive(down_learning_rate, "down_learning_rate")
    epsilon = _finite_positive(adam_epsilon, "adam_epsilon")
    gradient_bound = _finite_positive(max_gradient_norm, "max_gradient_norm")
    temperature_value = _finite_positive(temperature, "temperature")
    beta1_value = float(beta1)
    beta2_value = float(beta2)
    if (
        not 0.0 <= beta1_value < 1.0
        or not 0.0 <= beta2_value < 1.0
        or not math.isfinite(beta1_value)
        or not math.isfinite(beta2_value)
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "Adam beta values must be finite and in [0,1)."
        )
    if calibration_steps is None:
        selection_steps = tuple(range(step_count + 1))
    else:
        selection_steps = tuple(
            _index(value, "calibration step") for value in calibration_steps
        )
        if (
            not selection_steps
            or selection_steps != tuple(sorted(set(selection_steps)))
            or selection_steps[0] != 0
            or selection_steps[-1] != step_count
        ):
            raise BehaviorResidualSwiGLUTrainingError(
                "calibration_steps must be sorted, unique, and include 0 and steps."
            )
    refresh_steps = tuple(
        _index(value, "fit refresh step") for value in (fit_refresh_steps or ())
    )
    if refresh_steps != tuple(sorted(set(refresh_steps))) or any(
        value <= 0 or value >= step_count for value in refresh_steps
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "fit_refresh_steps must be sorted, unique, positive, and below steps."
        )
    if (fit_batch_refresher is None) != (not refresh_steps):
        raise BehaviorResidualSwiGLUTrainingError(
            "A fit_batch_refresher is required exactly when refresh steps exist."
        )
    if fit_batch_refresher is not None and (
        not callable(fit_batch_refresher) or initial_fit_provenance is None
    ):
        raise BehaviorResidualSwiGLUTrainingError(
            "On-policy fit refresh requires a callable and initial phase provenance."
        )
    active_fit, active_fit_provenance = _normalize_fit_phase(
        fit,
        initial_fit_provenance,
        refresh_after_step=0,
        calibration_domain=calibration_domain,
        require_on_policy_provenance=fit_batch_refresher is not None,
    )
    named = handle.residual_named_parameters()
    names = tuple(name for name, _value in named)
    parameters = tuple(value for _name, value in named)
    learning_rates = tuple(
        down_lr if name.endswith("down_weight") else gate_up_lr for name in names
    )
    first_moments = [
        torch.zeros_like(value, dtype=torch.float32) for value in parameters
    ]
    second_moments = [
        torch.zeros_like(value, dtype=torch.float32) for value in parameters
    ]
    gradient_activation = {name: False for name in names}
    fit_history: list[dict[str, Any]] = []
    fit_phase_history: list[dict[str, Any]] = []
    calibration_history: list[dict[str, Any]] = []
    selection_set = set(selection_steps)

    initial: FunctionalActionEvaluation | None = None
    best: FunctionalActionEvaluation | None = None
    final: FunctionalActionEvaluation | None = None
    initial_rollout: Mapping[str, Any] | None = None
    best_rollout: Mapping[str, Any] | None = None
    final_rollout: Mapping[str, Any] | None = None
    best_step = 0
    evaluations = 0
    rollout_evaluations = 0
    refresh_evaluations = 0
    active_fit_cursor = 0
    active_fit_phase = 0
    fit_phase_history.append(
        {
            "phase_index": active_fit_phase,
            "refresh_after_step": 0,
            "batch_count": len(active_fit),
            "batch_cursor_start": 0,
            "phase_provenance": dict(active_fit_provenance),
            "residual_state_sha256_before": handle.residual_state_sha256(),
            "residual_state_sha256_after": handle.residual_state_sha256(),
            "manual_adam_moment_sha256_before": _optimizer_moment_sha256(
                first_moments, second_moments
            ),
            "manual_adam_moment_sha256_after": _optimizer_moment_sha256(
                first_moments, second_moments
            ),
            "manual_adam_moments_reset": False,
        }
    )

    def evaluate_calibration_rollout() -> Mapping[str, Any] | None:
        if calibration_rollout_evaluator is None:
            return None
        if normalized_rollout_provenance is None:
            raise BehaviorResidualSwiGLUTrainingError(
                "Normalized calibration rollout provenance is missing."
            )
        state_before = handle.residual_state_sha256()
        record = _calibration_rollout_record(
            calibration_rollout_evaluator(),
            expected_provenance=normalized_rollout_provenance,
        )
        handle.assert_live()
        if handle.residual_state_sha256() != state_before:
            raise BehaviorResidualSwiGLUTrainingError(
                "Calibration rollout callback mutated the candidate residual state."
            )
        return MappingProxyType(record)

    with handle.output_training_scope() as scoped:
        if tuple(scoped) != parameters:
            raise BehaviorResidualSwiGLUTrainingError(
                "Training scope returned an unexpected residual parameter order."
            )
        initial = evaluate_functional_action_batches(
            handle.model,
            calibration,
            expected_split="calibration-selection",
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature_value,
            used_for_selection=True,
        )
        evaluations = 1
        if calibration_rollout_evaluator is not None:
            initial_rollout = evaluate_calibration_rollout()
            rollout_evaluations = 1
        best = initial
        best_rollout = initial_rollout
        best_key = _selection_key(initial, initial_rollout)
        best_state = handle.residual_state()
        best_state_sha256 = handle.residual_state_sha256()
        calibration_history.append(
            {
                "step": 0,
                "evaluation": initial.as_dict(),
                "rollout_evaluation": (
                    None if initial_rollout is None else dict(initial_rollout)
                ),
                "selection_key": list(best_key),
                "promoted": False,
                "incumbent": True,
                "residual_state_sha256": handle.residual_state_sha256(),
            }
        )
        for step in range(1, step_count + 1):
            fit_batch_schedule_position = active_fit_cursor % len(active_fit)
            batch = active_fit[fit_batch_schedule_position]
            active_fit_cursor += 1
            total, hard, distribution = _functional_losses(
                handle.model,
                batch,
                hard_action_weight=hard_action_weight,
                distribution_weight=distribution_weight,
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
                    torch.sum(value.detach().to(torch.float32).square())
                    for value in gradients
                )
            )
            norm_value = float(norm.item())
            scale = min(1.0, gradient_bound / max(norm_value, 1e-12))
            for name, gradient in zip(names, gradients, strict=True):
                if float(torch.linalg.vector_norm(gradient.float()).item()) > 0.0:
                    gradient_activation[name] = True
            with torch.no_grad():
                for parameter, gradient, first, second, learning_rate in zip(
                    parameters,
                    gradients,
                    first_moments,
                    second_moments,
                    learning_rates,
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
            handle.assert_live()
            fit_history.append(
                {
                    "step": step,
                    "fit_phase_index": active_fit_phase,
                    "fit_batch_schedule_position": fit_batch_schedule_position,
                    "fit_phase_provenance_sha256": active_fit_provenance[
                        "phase_provenance_sha256"
                    ],
                    "sequence_id": batch.sequence_id,
                    "prefix_source": batch.prefix_source,
                    "total_loss": float(total.detach().item()),
                    "hard_action_loss": float(hard.detach().item()),
                    "distribution_loss": float(distribution.detach().item()),
                    "gradient_norm": norm_value,
                    "gradient_clip_scale": scale,
                }
            )
            if step in selection_set:
                evaluation = evaluate_functional_action_batches(
                    handle.model,
                    calibration,
                    expected_split="calibration-selection",
                    hard_action_weight=hard_action_weight,
                    distribution_weight=distribution_weight,
                    temperature=temperature_value,
                    used_for_selection=True,
                )
                evaluations += 1
                rollout = None
                if calibration_rollout_evaluator is not None:
                    rollout = evaluate_calibration_rollout()
                    if initial_rollout is None:
                        raise BehaviorResidualSwiGLUTrainingError(
                            "Initial calibration rollout summary is missing."
                        )
                    _assert_same_rollout_domain(initial_rollout, rollout)
                    rollout_evaluations += 1
                key = _selection_key(evaluation, rollout)
                if rollout is None:
                    promoted = key < best_key
                else:
                    if best_rollout is None or best is None:
                        raise BehaviorResidualSwiGLUTrainingError(
                            "Rollout-aware selection lost its incumbent state."
                        )
                    promoted = calibration_rollout_candidate_can_promote(
                        best,
                        best_rollout,
                        evaluation,
                        rollout,
                    )
                if promoted:
                    best_key = key
                    best = evaluation
                    best_rollout = rollout
                    best_step = step
                    best_state = handle.residual_state()
                    best_state_sha256 = handle.residual_state_sha256()
                calibration_history.append(
                    {
                        "step": step,
                        "evaluation": evaluation.as_dict(),
                        "rollout_evaluation": (
                            None if rollout is None else dict(rollout)
                        ),
                        "selection_key": list(key),
                        "promoted": promoted,
                        "incumbent": False,
                        "residual_state_sha256": handle.residual_state_sha256(),
                    }
                )
            if step in refresh_steps:
                if fit_batch_refresher is None:
                    raise BehaviorResidualSwiGLUTrainingError(
                        "Fit refresh callback disappeared during training."
                    )
                state_before = handle.residual_state_sha256()
                moments_before = _optimizer_moment_sha256(first_moments, second_moments)
                refresh = fit_batch_refresher(step)
                handle.assert_live()
                state_after = handle.residual_state_sha256()
                moments_after = _optimizer_moment_sha256(first_moments, second_moments)
                if (
                    not isinstance(refresh, BehaviorResidualSwiGLUFitRefresh)
                    or refresh.refresh_after_step != step
                    or state_after != state_before
                    or moments_after != moments_before
                ):
                    raise BehaviorResidualSwiGLUTrainingError(
                        "Fit refresh changed candidate/Adam state or returned wrong phase."
                    )
                prior_fit_batch_count = len(active_fit)
                refreshed_fit, refreshed_fit_provenance = _normalize_fit_phase(
                    refresh.batches,
                    refresh.provenance,
                    refresh_after_step=step,
                    calibration_domain=calibration_domain,
                    require_on_policy_provenance=True,
                )
                if len(refreshed_fit) != prior_fit_batch_count:
                    raise BehaviorResidualSwiGLUTrainingError(
                        "Fit refresh changed the preserved update schedule length."
                    )
                active_fit = refreshed_fit
                active_fit_provenance = refreshed_fit_provenance
                active_fit_phase += 1
                refresh_evaluations += 1
                fit_phase_history.append(
                    {
                        "phase_index": active_fit_phase,
                        "refresh_after_step": step,
                        "batch_count": len(active_fit),
                        "batch_cursor_start": active_fit_cursor,
                        "phase_provenance": dict(active_fit_provenance),
                        "residual_state_sha256_before": state_before,
                        "residual_state_sha256_after": state_after,
                        "manual_adam_moment_sha256_before": moments_before,
                        "manual_adam_moment_sha256_after": moments_after,
                        "manual_adam_moments_reset": False,
                    }
                )
        handle.load_residual_state(best_state)
        if handle.residual_state_sha256() != best_state_sha256:
            raise BehaviorResidualSwiGLUTrainingError(
                "Locked residual tensors did not reproduce the selected state hash."
            )
        final = evaluate_functional_action_batches(
            handle.model,
            calibration,
            expected_split="calibration-selection",
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature_value,
            used_for_selection=True,
        )
        evaluations += 1
        if calibration_rollout_evaluator is not None:
            final_rollout = evaluate_calibration_rollout()
            if initial_rollout is None:
                raise BehaviorResidualSwiGLUTrainingError(
                    "Initial calibration rollout summary is missing."
                )
            _assert_same_rollout_domain(initial_rollout, final_rollout)
            rollout_evaluations += 1
        if (
            final != best
            or final_rollout != best_rollout
            or _selection_key(final, final_rollout) != best_key
        ):
            raise BehaviorResidualSwiGLUTrainingError(
                "Locked state did not reproduce full action and per-prompt rollout data."
            )
    if initial is None or best is None or final is None:
        raise BehaviorResidualSwiGLUTrainingError(
            "Residual training did not produce a locked calibration state."
        )
    handle.assert_live()
    return BehaviorResidualSwiGLUTrainingReport(
        steps_requested=step_count,
        best_step=best_step,
        gate_up_learning_rate=gate_up_lr,
        down_learning_rate=down_lr,
        max_gradient_norm=gradient_bound,
        initial_calibration=initial,
        best_calibration=best,
        final_calibration=final,
        initial_calibration_rollout=initial_rollout,
        best_calibration_rollout=best_rollout,
        final_calibration_rollout=final_rollout,
        calibration_rollout_provenance=normalized_rollout_provenance,
        calibration_steps=selection_steps,
        fit_gradient_calls=step_count,
        calibration_evaluations=evaluations,
        calibration_rollout_evaluations=rollout_evaluations,
        fit_refresh_steps=refresh_steps,
        fit_refresh_evaluations=refresh_evaluations,
        manual_adam_moment_reset_count=0,
        residual_state_sha256=handle.residual_state_sha256(),
        fit_prefix_domain_sha256=fit_hash,
        calibration_prefix_domain_sha256=calibration_hash,
        parameter_gradient_activation=MappingProxyType(gradient_activation),
        fit_history=tuple(MappingProxyType(value) for value in fit_history),
        fit_phase_history=tuple(MappingProxyType(value) for value in fit_phase_history),
        calibration_history=tuple(
            MappingProxyType(value) for value in calibration_history
        ),
    )


__all__ = [
    "BEHAVIOR_RESIDUAL_SWIGLU_ACCOUNTING_SCHEMA",
    "BEHAVIOR_RESIDUAL_SWIGLU_INSTALLATION_SCHEMA",
    "BEHAVIOR_RESIDUAL_SWIGLU_SCHEMA",
    "BEHAVIOR_RESIDUAL_SWIGLU_TRAINING_SCHEMA",
    "BakedDenseSwiGLU",
    "BehaviorResidualSwiGLUAccounting",
    "BehaviorResidualSwiGLUAccountingError",
    "BehaviorResidualSwiGLUError",
    "BehaviorResidualSwiGLUFitRefresh",
    "BehaviorResidualSwiGLUInstallationReport",
    "BehaviorResidualSwiGLUMLP",
    "BehaviorResidualSwiGLUTopologyError",
    "BehaviorResidualSwiGLUTrainingError",
    "BehaviorResidualSwiGLUTrainingReport",
    "InstalledBehaviorResidualSwiGLU",
    "MLPCodebookRedirectAudit",
    "ResidualSwiGLUBranch",
    "audit_mlp_codebook_redirects",
    "bake_behavior_residual_swiglu",
    "calibration_rollout_candidate_can_promote",
    "calibration_rollout_selection_key",
    "install_behavior_residual_swiglu",
    "train_behavior_residual_swiglu_disjoint",
]
