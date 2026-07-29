"""Derive heterogeneous LFM2 capacity options from sealed cached traces.

This module does not fold, train, or execute either model.  It converts the
relative energy of each cached operator and MLP residual write into a sealed
per-layer capacity pressure.  The heterogeneous exact-budget solver remains
responsible for choosing a globally legal whole-model layout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)
from .hf_lfm2_heterogeneous_layout_selection import (
    Lfm2AuthenticatedLayerPhysicalOption,
    build_lfm2_attention_layer_option,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
)

CAPACITY_EVIDENCE_SCHEMA = "neural-forge-lfm2-cached-capacity-evidence-v2"
CAPACITY_OPTION_BANK_SCHEMA = "neural-forge-lfm2-cached-capacity-option-bank-v2"
MLP_CAPACITY_RISK_METHOD = "lfm2-cached-joint-swiglu-activation-cluster-surrogate-v1"
AUTOMATIC_MLP_OPTION_POLICY = "dynamic-exact-budget-broad-integer-lattice-v1"
EXPLICIT_MLP_OPTION_POLICY = "caller-supplied-width-grid-v1"


class Lfm2CachedHeterogeneousCapacityError(RuntimeError):
    """Cached evidence cannot authenticate a heterogeneous option bank."""


@dataclass(frozen=True, slots=True)
class Lfm2CachedMlpCapacityRisk:
    """Width-conditioned cached nonlinear defect for one physical SwiGLU.

    This is a fold-selection surrogate, not a trained loss.  It is bound to the
    production activation-cluster SwiGLU compiler and combines the cached
    post-nonlinearity, sign-regime, analytic tangent, residual-writer, and
    downstream-residual tails which that compiler must merge at the proposed
    physical width.
    """

    intermediate_size: int
    source_intermediate_size: int
    fold_method_id: str
    post_swiglu_tail_fraction: float
    signed_regime_tail_fraction: float
    tangent_tail_fraction: float
    downstream_weighted_tail_fraction: float
    collision_fraction: float
    predicted_defect: float

    def __post_init__(self) -> None:
        width = _positive_int(self.intermediate_size, "intermediate_size")
        source = _positive_int(
            self.source_intermediate_size,
            "source_intermediate_size",
        )
        if width >= source:
            raise Lfm2CachedHeterogeneousCapacityError(
                "A compact MLP capacity candidate must be narrower than its source."
            )
        if self.fold_method_id != MLP_CAPACITY_RISK_METHOD:
            raise Lfm2CachedHeterogeneousCapacityError(
                "MLP capacity risk is not bound to the production fold method."
            )
        for name in (
            "post_swiglu_tail_fraction",
            "signed_regime_tail_fraction",
            "tangent_tail_fraction",
            "downstream_weighted_tail_fraction",
            "collision_fraction",
            "predicted_defect",
        ):
            value = _finite_nonnegative(getattr(self, name), name)
            if name != "predicted_defect" and value > 1.0 + 1e-12:
                raise Lfm2CachedHeterogeneousCapacityError(f"{name} cannot exceed one.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "intermediate_size": self.intermediate_size,
            "source_intermediate_size": self.source_intermediate_size,
            "fold_method_id": self.fold_method_id,
            "post_swiglu_tail_fraction": self.post_swiglu_tail_fraction,
            "signed_regime_tail_fraction": self.signed_regime_tail_fraction,
            "tangent_tail_fraction": self.tangent_tail_fraction,
            "downstream_weighted_tail_fraction": (
                self.downstream_weighted_tail_fraction
            ),
            "collision_fraction": self.collision_fraction,
            "predicted_defect": self.predicted_defect,
        }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedHeterogeneousCapacityError(
            f"{name} must be a positive integer."
        )
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CachedHeterogeneousCapacityError(
            f"{name} must be finite and non-negative."
        )
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


@dataclass(frozen=True, slots=True)
class Lfm2CachedLayerCapacityEvidence:
    """One source layer's normalized residual-write pressure."""

    layer_index: int
    layer_type: str
    source_layer_sha256: str
    operator_delta_energy_fraction: float
    mlp_delta_energy_fraction: float
    sampled_scalar_count: int
    nonlinear_sampled_row_count: int
    source_mlp_intermediate_size: int
    mlp_capacity_risks: tuple[Lfm2CachedMlpCapacityRisk, ...]
    selection_evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2CachedHeterogeneousCapacityError(
                "layer_index must be a non-negative integer."
            )
        if self.layer_type not in {"conv", "full_attention"}:
            raise Lfm2CachedHeterogeneousCapacityError(
                "layer_type must be conv or full_attention."
            )
        _digest(self.source_layer_sha256, "source_layer_sha256")
        _finite_nonnegative(
            self.operator_delta_energy_fraction,
            "operator_delta_energy_fraction",
        )
        _finite_nonnegative(
            self.mlp_delta_energy_fraction,
            "mlp_delta_energy_fraction",
        )
        _positive_int(self.sampled_scalar_count, "sampled_scalar_count")
        _positive_int(
            self.nonlinear_sampled_row_count,
            "nonlinear_sampled_row_count",
        )
        source_width = _positive_int(
            self.source_mlp_intermediate_size,
            "source_mlp_intermediate_size",
        )
        if not self.mlp_capacity_risks:
            raise Lfm2CachedHeterogeneousCapacityError(
                "Layer capacity evidence must contain nonlinear width risks."
            )
        widths = tuple(value.intermediate_size for value in self.mlp_capacity_risks)
        if widths != tuple(sorted(set(widths))) or any(
            value.source_intermediate_size != source_width
            for value in self.mlp_capacity_risks
        ):
            raise Lfm2CachedHeterogeneousCapacityError(
                "Layer nonlinear width risks are not canonical for the source MLP."
            )
        defects = tuple(value.predicted_defect for value in self.mlp_capacity_risks)
        if any(left + 1e-12 < right for left, right in pairwise(defects)):
            raise Lfm2CachedHeterogeneousCapacityError(
                "Predicted MLP defect must not increase with physical width."
            )
        _digest(self.selection_evidence_sha256, "selection_evidence_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_type": self.layer_type,
            "source_layer_sha256": self.source_layer_sha256,
            "operator_delta_energy_fraction": (self.operator_delta_energy_fraction),
            "mlp_delta_energy_fraction": self.mlp_delta_energy_fraction,
            "sampled_scalar_count": self.sampled_scalar_count,
            "nonlinear_sampled_row_count": self.nonlinear_sampled_row_count,
            "source_mlp_intermediate_size": self.source_mlp_intermediate_size,
            "mlp_capacity_risks": [
                value.as_dict() for value in self.mlp_capacity_risks
            ],
            "selection_evidence_sha256": self.selection_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CachedHeterogeneousCapacityOptionBank:
    """Authenticated option grid consumed by the exact layout selector."""

    source_graph_fingerprint: str
    parent_cache_key: str
    mlp_option_policy: str
    exact_budget_lattice_sha256: str
    layer_evidence: tuple[Lfm2CachedLayerCapacityEvidence, ...]
    layer_options: tuple[Lfm2AuthenticatedLayerPhysicalOption, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        _digest(self.source_graph_fingerprint, "source_graph_fingerprint")
        _digest(self.parent_cache_key, "parent_cache_key")
        if self.mlp_option_policy not in {
            AUTOMATIC_MLP_OPTION_POLICY,
            EXPLICIT_MLP_OPTION_POLICY,
        }:
            raise Lfm2CachedHeterogeneousCapacityError(
                "Unknown MLP capacity option policy."
            )
        _digest(self.exact_budget_lattice_sha256, "exact_budget_lattice_sha256")
        _digest(self.evidence_sha256, "evidence_sha256")
        if not self.layer_evidence or not self.layer_options:
            raise Lfm2CachedHeterogeneousCapacityError(
                "Capacity option bank cannot be empty."
            )
        indices = tuple(value.layer_index for value in self.layer_evidence)
        if indices != tuple(range(len(indices))):
            raise Lfm2CachedHeterogeneousCapacityError(
                "Capacity evidence must cover every layer in order."
            )
        evidence_by_source = {
            value.source_layer_sha256: value for value in self.layer_evidence
        }
        if len(evidence_by_source) != len(self.layer_evidence) or any(
            option.source_layer_sha256 not in evidence_by_source
            for option in self.layer_options
        ):
            raise Lfm2CachedHeterogeneousCapacityError(
                "Capacity options are not bound to the ordered source layers."
            )
        if canonical_sha256(self._body()) != self.evidence_sha256:
            raise Lfm2CachedHeterogeneousCapacityError(
                "Capacity option bank seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": CAPACITY_OPTION_BANK_SCHEMA,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "parent_cache_key": self.parent_cache_key,
            "mlp_option_policy": self.mlp_option_policy,
            "exact_budget_lattice_sha256": self.exact_budget_lattice_sha256,
            "layer_evidence": [value.as_dict() for value in self.layer_evidence],
            "layer_options": [value.as_dict() for value in self.layer_options],
            "selection_contract": {
                "cached_parent_traces_only": True,
                "parent_model_calls": 0,
                "child_model_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "global_exact_budget_solver_selects_layout": True,
                "hardcoded_layer_indices": False,
                "fixed_566_567_production_default": False,
                "mlp_width_risk_depends_on_cached_nonlinear_defect": True,
                "dense_physical_weights": True,
                "parameter_sharing": False,
                "quantization": False,
            },
        }

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {**self._body(), "evidence_sha256": self.evidence_sha256}
        )


def _energy(value: Any) -> tuple[float, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cached capacity evidence requires PyTorch.") from exc
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached residual endpoints must be floating tensors."
        )
    rows = value.detach().to(device="cpu", dtype=torch.float64)
    if rows.numel() <= 0 or not bool(torch.isfinite(rows).all().item()):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached residual endpoints must be finite and non-empty."
        )
    return float(rows.square().sum().item()), int(rows.numel())


def _delta_energy(left: Any, right: Any) -> tuple[float, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cached capacity evidence requires PyTorch.") from exc
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or tuple(left.shape) != tuple(right.shape)
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached residual endpoints changed shape within a layer."
        )
    delta = left.detach().to(device="cpu", dtype=torch.float64) - right.detach().to(
        device="cpu", dtype=torch.float64
    )
    if delta.numel() <= 0 or not bool(torch.isfinite(delta).all().item()):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached residual deltas must be finite and non-empty."
        )
    return float(delta.square().sum().item()), int(delta.numel())


def _normalized_pressures(
    values: Sequence[float],
    indices: Sequence[int],
) -> dict[int, float]:
    selected = tuple(
        _finite_nonnegative(values[index], "capacity pressure") for index in indices
    )
    maximum = max(selected, default=0.0)
    if maximum <= 0.0:
        return {index: 0.0 for index in indices}
    return {index: values[index] / maximum for index in indices}


def _ordered_parent_phases(record: Mapping[str, Any]) -> tuple[str, ...]:
    capture = record.get("capture")
    diagnostics = (
        capture.get("phase_diagnostics") if isinstance(capture, Mapping) else None
    )
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise Lfm2CachedHeterogeneousCapacityError(
            "Parent cache batch has no authenticated causal phases."
        )

    def order_key(phase: Any) -> tuple[int, int]:
        if phase in {"prefill", "cached_prefill"}:
            return (0, 0)
        if isinstance(phase, str) and phase.startswith("decode."):
            suffix = phase.removeprefix("decode.")
            if suffix.isdigit():
                return (1, int(suffix))
        raise Lfm2CachedHeterogeneousCapacityError(
            f"Parent cache contains an unknown causal phase {phase!r}."
        )

    phases = tuple(sorted(diagnostics, key=order_key))
    if len(phases) != len(set(phases)):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Parent cache causal phases are not unique."
        )
    return phases


def _operator_parameter_count(
    layer_type: str,
    geometry: int | tuple[int, int, int],
    *,
    hidden_size: int,
    conv_kernel_size: int,
) -> int:
    if layer_type == "conv":
        if not isinstance(geometry, int):
            raise Lfm2CachedHeterogeneousCapacityError(
                "A short-convolution budget geometry must be an integer width."
            )
        return geometry * (4 * hidden_size + conv_kernel_size)
    if layer_type != "full_attention" or not isinstance(geometry, tuple):
        raise Lfm2CachedHeterogeneousCapacityError(
            "An attention budget geometry must be a Q/KV/head tuple."
        )
    query, key_value, head_dim = geometry
    return 2 * hidden_size * head_dim * (query + key_value) + 2 * head_dim


def _balanced_width_allocation(
    total_width: int,
    minimums: Sequence[int],
    maximums: Sequence[int],
) -> tuple[int, ...] | None:
    if len(minimums) != len(maximums) or not minimums:
        raise Lfm2CachedHeterogeneousCapacityError(
            "MLP width allocation bounds must be nonempty and aligned."
        )
    values = [int(value) for value in minimums]
    limits = [int(value) for value in maximums]
    if total_width < sum(values) or total_width > sum(limits):
        return None
    remaining = total_width - sum(values)
    while remaining:
        active = [index for index, value in enumerate(values) if value < limits[index]]
        if not active:  # pragma: no cover - protected by the total bound
            return None
        selected = min(
            active,
            key=lambda index: (values[index] / limits[index], index),
        )
        values[selected] += 1
        remaining -= 1
    return tuple(values)


@dataclass(frozen=True, slots=True)
class Lfm2MlpWidthOptionPolicy:
    policy_id: str
    widths_by_layer: tuple[tuple[int, ...], ...]
    exact_budget_lattice_sha256: str

    def __post_init__(self) -> None:
        if self.policy_id not in {
            AUTOMATIC_MLP_OPTION_POLICY,
            EXPLICIT_MLP_OPTION_POLICY,
        }:
            raise Lfm2CachedHeterogeneousCapacityError(
                "Unknown MLP width-option policy."
            )
        if not self.widths_by_layer or any(
            not widths
            or widths != tuple(sorted(set(widths)))
            or any(
                isinstance(width, bool) or not isinstance(width, int) or width <= 0
                for width in widths
            )
            for widths in self.widths_by_layer
        ):
            raise Lfm2CachedHeterogeneousCapacityError(
                "MLP width-option policy rows must be nonempty, positive, and canonical."
            )
        _digest(
            self.exact_budget_lattice_sha256,
            "exact_budget_lattice_sha256",
        )


def _explicit_mlp_width_policy(
    supplied: Sequence[int],
    source_widths: Sequence[int],
) -> Lfm2MlpWidthOptionPolicy:
    widths = tuple(
        sorted({_positive_int(value, "mlp_intermediate_size") for value in supplied})
    )
    if len(widths) < 2:
        raise Lfm2CachedHeterogeneousCapacityError(
            "Capacity discovery needs at least two MLP choices."
        )
    rows = []
    for layer_index, source_width in enumerate(source_widths):
        if any(value >= source_width for value in widths):
            raise Lfm2CachedHeterogeneousCapacityError(
                "Caller-supplied MLP width is not a compact physical candidate for "
                f"source layer {layer_index}."
            )
        rows.append(widths)
    body = {
        "schema": "neural-forge-lfm2-explicit-mlp-width-grid-v1",
        "policy_id": EXPLICIT_MLP_OPTION_POLICY,
        "source_widths": list(source_widths),
        "widths_by_layer": [list(value) for value in rows],
        "exact_budget_witness": None,
    }
    return Lfm2MlpWidthOptionPolicy(
        policy_id=EXPLICIT_MLP_OPTION_POLICY,
        widths_by_layer=tuple(rows),
        exact_budget_lattice_sha256=canonical_sha256(body),
    )


def _automatic_mlp_width_policy(
    *,
    layer_types: Sequence[str],
    source_widths: Sequence[int],
    short_conv_internal_widths: Sequence[int],
    attention_head_options: Sequence[tuple[int, int, int]],
    target_hidden_size: int,
    target_vocab_size: int,
    exact_target_unique_parameters: int,
    conv_kernel_size: int,
) -> Lfm2MlpWidthOptionPolicy:
    """Generate broad widths plus an authenticated exact-count witness.

    Symmetric one-neuron through large-radius offsets let the global solver
    redistribute nonlinear capacity between fragile and robust layers without
    exploding its exact-state frontier.  Source-relative anchors allow large
    capacity trades, and a balanced witness is derived from the requested
    whole-child count, never from a fixed 566/567 production constant.
    """

    hidden = _positive_int(target_hidden_size, "target_hidden_size")
    vocab = _positive_int(target_vocab_size, "target_vocab_size")
    target = _positive_int(
        exact_target_unique_parameters,
        "exact_target_unique_parameters",
    )
    kernel = _positive_int(conv_kernel_size, "conv_kernel_size")
    conv_widths = tuple(
        sorted(
            {
                _positive_int(value, "short_conv_internal_width")
                for value in short_conv_internal_widths
            }
        )
    )
    attention_options = tuple(
        sorted(
            {
                (
                    _positive_int(query, "query_heads"),
                    _positive_int(key_value, "key_value_heads"),
                    _positive_int(head_dim, "head_dim"),
                )
                for query, key_value, head_dim in attention_head_options
            }
        )
    )
    if len(conv_widths) < 2 or len(attention_options) < 2:
        raise Lfm2CachedHeterogeneousCapacityError(
            "Automatic MLP policy needs at least two canonical operator choices."
        )
    if any(
        query % key_value or head_dim % 2
        for query, key_value, head_dim in attention_options
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Automatic attention choices require divisible Q/KV groups and even head_dim."
        )
    types = tuple(layer_types)
    sources = tuple(
        _positive_int(value, "source_mlp_intermediate_size") for value in source_widths
    )
    if not types or len(types) != len(sources):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Automatic MLP policy requires aligned source layers and widths."
        )
    minimums = tuple(max(3, math.ceil(value / 8)) for value in sources)
    maximums = tuple(value - 1 for value in sources)
    if any(minimum > maximum for minimum, maximum in zip(minimums, maximums)):
        raise Lfm2CachedHeterogeneousCapacityError(
            "A source MLP is too narrow for compact automatic capacity discovery."
        )

    states: dict[int, tuple[int | tuple[int, int, int], ...]] = {0: ()}
    for layer_type in types:
        geometries: tuple[int | tuple[int, int, int], ...]
        if layer_type == "conv":
            geometries = conv_widths
        elif layer_type == "full_attention":
            geometries = attention_options
        else:
            raise Lfm2CachedHeterogeneousCapacityError(
                f"Unsupported source layer type {layer_type!r}."
            )
        next_states: dict[int, tuple[int | tuple[int, int, int], ...]] = {}
        for subtotal, path in sorted(states.items()):
            for geometry in geometries:
                operator_count = _operator_parameter_count(
                    layer_type,
                    geometry,
                    hidden_size=hidden,
                    conv_kernel_size=kernel,
                )
                next_states.setdefault(subtotal + operator_count, path + (geometry,))
        states = next_states

    fixed_count = 2 * vocab * hidden + hidden + 2 * hidden * len(types)
    mlp_parameter_quantum = 3 * hidden
    feasible: list[
        tuple[
            int,
            int,
            tuple[int | tuple[int, int, int], ...],
            tuple[int, ...],
        ]
    ] = []
    for operator_count, path in sorted(states.items()):
        remaining = target - fixed_count - operator_count
        if remaining <= 0 or remaining % mlp_parameter_quantum:
            continue
        total_width = remaining // mlp_parameter_quantum
        allocation = _balanced_width_allocation(
            total_width,
            minimums,
            maximums,
        )
        if allocation is not None:
            feasible.append((total_width, operator_count, path, allocation))
    if not feasible:
        raise Lfm2CachedHeterogeneousCapacityError(
            "The exact whole-child budget has no dense compact MLP/operator "
            "integer-lattice witness under the discovered source widths."
        )

    # Prefer the feasible witness which leaves the largest aggregate nonlinear
    # capacity.  Risk-aware selection may still spend those parameters on a
    # more expensive operator geometry because all feasible alternatives feed
    # the same exact global solver.
    witness = max(feasible, key=lambda value: (value[0], -value[1], value[3]))
    candidate_sets = [set() for _ in types]
    for _total, _operator, _path, allocation in feasible:
        for index, width in enumerate(allocation):
            candidate_sets[index].add(width)
    source_relative_ratios = (
        (1, 8),
        (1, 4),
        (1, 2),
        (3, 4),
    )
    for index, (source, center, minimum, maximum) in enumerate(
        zip(sources, witness[3], minimums, maximums, strict=True)
    ):
        for offset in (1, 2, 4, 8, 16, 32, 64):
            for candidate in (center - offset, center + offset):
                if minimum <= candidate <= maximum:
                    candidate_sets[index].add(candidate)
        for numerator, denominator in source_relative_ratios:
            candidate = min(
                maximum,
                max(minimum, round(source * numerator / denominator)),
            )
            candidate_sets[index].add(candidate)
        for numerator, denominator in ((1, 2), (3, 2)):
            candidate = min(
                maximum,
                max(minimum, round(center * numerator / denominator)),
            )
            candidate_sets[index].add(candidate)
        candidate_sets[index].update((minimum, maximum, center))
        geometries = conv_widths if types[index] == "conv" else attention_options
        operator_counts = tuple(
            _operator_parameter_count(
                types[index],
                geometry,
                hidden_size=hidden,
                conv_kernel_size=kernel,
            )
            for geometry in geometries
        )
        exchange_widths = {
            abs(left - right) // mlp_parameter_quantum
            for left in operator_counts
            for right in operator_counts
            if left != right and abs(left - right) % mlp_parameter_quantum == 0
        }
        for exchange in exchange_widths:
            for candidate in (center - exchange, center + exchange):
                if minimum <= candidate <= maximum:
                    candidate_sets[index].add(candidate)
    widths_by_layer = tuple(tuple(sorted(values)) for values in candidate_sets)
    if any(len(values) < 3 for values in widths_by_layer):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Automatic MLP capacity discovery did not produce a broad option bank."
        )
    witness_count = fixed_count + witness[1] + mlp_parameter_quantum * sum(witness[3])
    if witness_count != target or any(
        width not in widths_by_layer[index] for index, width in enumerate(witness[3])
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Automatic MLP integer-lattice witness failed exact accounting."
        )
    body = {
        "schema": "neural-forge-lfm2-automatic-mlp-width-lattice-v1",
        "policy_id": AUTOMATIC_MLP_OPTION_POLICY,
        "target_hidden_size": hidden,
        "target_vocab_size": vocab,
        "conv_kernel_size": kernel,
        "short_conv_internal_widths": list(conv_widths),
        "attention_head_options": [list(value) for value in attention_options],
        "exact_target_unique_parameters": target,
        "fixed_boundary_and_norm_parameters": fixed_count,
        "mlp_parameter_quantum": mlp_parameter_quantum,
        "source_widths": list(sources),
        "minimum_widths": list(minimums),
        "maximum_widths": list(maximums),
        "feasible_operator_state_count": len(feasible),
        "widths_by_layer": [list(value) for value in widths_by_layer],
        "exact_budget_witness": {
            "operator_parameter_count": witness[1],
            "operator_geometries": [
                list(value) if isinstance(value, tuple) else value
                for value in witness[2]
            ],
            "mlp_widths": list(witness[3]),
            "whole_child_unique_parameters": witness_count,
        },
        "dense_physical_only": True,
        "shared_parameters": 0,
        "quantized_parameters": 0,
        "training_steps": 0,
    }
    return Lfm2MlpWidthOptionPolicy(
        policy_id=AUTOMATIC_MLP_OPTION_POLICY,
        widths_by_layer=widths_by_layer,
        exact_budget_lattice_sha256=canonical_sha256(body),
    )


def derive_lfm2_automatic_mlp_width_policy(
    *,
    layer_types: Sequence[str],
    source_widths: Sequence[int],
    short_conv_internal_widths: Sequence[int],
    attention_head_options: Sequence[tuple[int, int, int]],
    target_hidden_size: int,
    target_vocab_size: int,
    exact_target_unique_parameters: int,
    conv_kernel_size: int,
) -> Lfm2MlpWidthOptionPolicy:
    """Expose pure exact-lattice derivation for multi-H frontier orchestration.

    Callers can derive and union width sets for several hidden-size candidates
    before performing one authenticated nonlinear-cache scan.
    """

    return _automatic_mlp_width_policy(
        layer_types=layer_types,
        source_widths=source_widths,
        short_conv_internal_widths=short_conv_internal_widths,
        attention_head_options=attention_head_options,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        exact_target_unique_parameters=exact_target_unique_parameters,
        conv_kernel_size=conv_kernel_size,
    )


def _flatten_cached_rows(value: Any, name: str, *, width: int | None = None) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cached capacity evidence requires PyTorch.") from exc
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim < 2
        or value.numel() <= 0
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            f"{name} must be a nonempty floating cached endpoint."
        )
    if width is not None and int(value.shape[-1]) != width:
        raise Lfm2CachedHeterogeneousCapacityError(
            f"{name} has the wrong cached feature width."
        )
    rows = (
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(
            -1,
            int(value.shape[-1]),
        )
    )
    if not bool(torch.isfinite(rows).all().item()):
        raise Lfm2CachedHeterogeneousCapacityError(
            f"{name} contains non-finite cached values."
        )
    return rows


def _normalized_row_weight(value: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cached capacity evidence requires PyTorch.") from exc
    mean = value.mean(dtype=torch.float64)
    tiny = torch.finfo(torch.float64).tiny
    if not bool(torch.isfinite(mean).item()) or float(mean.item()) <= tiny:
        return torch.ones_like(value, dtype=torch.float64)
    return value.to(torch.float64) / mean


def _cached_mlp_nonlinear_statistics(
    *,
    operator_input: Any,
    mlp_input: Any,
    gate: Any,
    up: Any,
    post_swiglu: Any,
    mlp_down: Any,
    post_operator: Any,
    post_layer: Any,
    final_post_layer: Any,
) -> tuple[dict[str, Any], int]:
    """Return streaming per-neuron nonlinear and downstream sufficient stats."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cached capacity evidence requires PyTorch.") from exc
    operator_rows = _flatten_cached_rows(operator_input, "operator.input")
    hidden = int(operator_rows.shape[1])
    mlp_input_rows = _flatten_cached_rows(mlp_input, "mlp.input", width=hidden)
    gate_rows = _flatten_cached_rows(gate, "mlp.gate")
    source_width = int(gate_rows.shape[1])
    up_rows = _flatten_cached_rows(up, "mlp.up", width=source_width)
    post_rows = _flatten_cached_rows(
        post_swiglu,
        "mlp.post_swiglu",
        width=source_width,
    )
    down_rows = _flatten_cached_rows(mlp_down, "mlp.down", width=hidden)
    post_operator_rows = _flatten_cached_rows(
        post_operator,
        "residual.post_operator",
        width=hidden,
    )
    post_layer_rows = _flatten_cached_rows(
        post_layer,
        "residual.post_layer",
        width=hidden,
    )
    final_rows = _flatten_cached_rows(
        final_post_layer,
        "final residual.post_layer",
        width=hidden,
    )
    row_count = int(gate_rows.shape[0])
    if any(
        int(value.shape[0]) != row_count
        for value in (
            operator_rows,
            mlp_input_rows,
            up_rows,
            post_rows,
            down_rows,
            post_operator_rows,
            post_layer_rows,
            final_rows,
        )
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached nonlinear endpoints do not have aligned token rows."
        )
    if not torch.allclose(
        post_operator_rows + down_rows,
        post_layer_rows,
        rtol=2e-3,
        atol=2e-5,
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached MLP writer does not close its residual endpoint."
        )
    sigmoid = torch.sigmoid(gate_rows)
    silu_gate = gate_rows * sigmoid
    if not torch.allclose(
        silu_gate * up_rows,
        post_rows,
        rtol=2e-3,
        atol=2e-5,
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached post-SwiGLU evidence does not close its gate/up nonlinearity."
        )
    silu_derivative = sigmoid * (1.0 + gate_rows * (1.0 - sigmoid))
    tangent = (up_rows * silu_derivative).square() + silu_gate.square()
    post_square = post_rows.square()
    positive_square = torch.where(post_rows > 0.0, post_square, 0.0)
    negative_square = torch.where(post_rows < 0.0, post_square, 0.0)
    writer_weight = _normalized_row_weight(down_rows.square().mean(dim=1))
    input_shift_weight = _normalized_row_weight(
        (mlp_input_rows - operator_rows).square().mean(dim=1)
    )
    downstream_weight = _normalized_row_weight(
        (final_rows - post_layer_rows).square().mean(dim=1)
    )
    combined_row_weight = (
        1.0 + writer_weight + input_shift_weight + downstream_weight
    ) / 4.0
    return (
        {
            "post": post_square.sum(dim=0, dtype=torch.float64),
            "positive": positive_square.sum(dim=0, dtype=torch.float64),
            "negative": negative_square.sum(dim=0, dtype=torch.float64),
            "tangent": tangent.sum(dim=0, dtype=torch.float64),
            "downstream": (
                post_square.to(torch.float64) * combined_row_weight.reshape(-1, 1)
            ).sum(dim=0, dtype=torch.float64),
        },
        row_count,
    )


def _probability_vector(value: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cached capacity evidence requires PyTorch.") from exc
    vector = value.detach().to(device="cpu", dtype=torch.float64).clamp_min(0.0)
    total = vector.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
        return torch.full_like(vector, 1.0 / int(vector.numel()))
    return vector / total


def _cached_mlp_capacity_risks(
    *,
    source_width: int,
    candidate_widths: Sequence[int],
    nonlinear_statistics: Mapping[str, Any],
    layer_pressure: float,
) -> tuple[Lfm2CachedMlpCapacityRisk, ...]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Cached capacity evidence requires PyTorch.") from exc
    source = _positive_int(source_width, "source_width")
    widths = tuple(
        sorted({_positive_int(value, "candidate_width") for value in candidate_widths})
    )
    if not widths or widths[-1] >= source:
        raise Lfm2CachedHeterogeneousCapacityError(
            "MLP risk widths must be nonempty and narrower than their source."
        )
    post = _probability_vector(nonlinear_statistics["post"])
    tangent = _probability_vector(nonlinear_statistics["tangent"])
    downstream = _probability_vector(nonlinear_statistics["downstream"])
    signed = _probability_vector(
        2.0
        * torch.minimum(
            nonlinear_statistics["positive"].to(torch.float64),
            nonlinear_statistics["negative"].to(torch.float64),
        )
    )
    if any(
        int(value.numel()) != source for value in (post, tangent, downstream, signed)
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Cached nonlinear statistics differ from the source MLP width."
        )
    joint = 0.25 * post + 0.20 * signed + 0.25 * tangent + 0.30 * downstream
    order = tuple(
        sorted(range(source), key=lambda index: (-float(joint[index].item()), index))
    )
    pressure = 0.05 + 0.95 * _finite_nonnegative(layer_pressure, "layer_pressure")

    def tail(vector: Any, retained: Sequence[int]) -> float:
        if not retained:
            return 1.0
        kept = float(vector[list(retained)].sum().item())
        return min(1.0, max(0.0, 1.0 - kept))

    result = []
    for width in widths:
        retained = order[:width]
        post_tail = tail(post, retained)
        signed_tail = tail(signed, retained)
        tangent_tail = tail(tangent, retained)
        downstream_tail = tail(downstream, retained)
        collision = (source - width) / source
        predicted = pressure * (
            0.25 * post_tail
            + 0.20 * signed_tail
            + 0.25 * tangent_tail
            + 0.25 * downstream_tail
            + 0.05 * collision
        )
        result.append(
            Lfm2CachedMlpCapacityRisk(
                intermediate_size=width,
                source_intermediate_size=source,
                fold_method_id=MLP_CAPACITY_RISK_METHOD,
                post_swiglu_tail_fraction=post_tail,
                signed_regime_tail_fraction=signed_tail,
                tangent_tail_fraction=tangent_tail,
                downstream_weighted_tail_fraction=downstream_tail,
                collision_fraction=collision,
                predicted_defect=predicted,
            )
        )
    return tuple(result)


def build_lfm2_cached_heterogeneous_capacity_option_bank(
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    *,
    selective_session: (Lfm2CircuitTraceSelectiveAuthenticatedSession | None) = None,
    short_conv_internal_widths: Sequence[int] = (704, 1024),
    attention_head_options: Sequence[tuple[int, int, int]] = (
        (12, 6, 64),
        (16, 8, 64),
    ),
    mlp_intermediate_sizes: Sequence[int] | None = None,
    target_hidden_size: int | None = None,
    target_vocab_size: int | None = None,
    exact_target_unique_parameters: int | None = None,
    conv_kernel_size: int = 3,
) -> Lfm2CachedHeterogeneousCapacityOptionBank:
    """Build a sealed, layer-agnostic capacity grid from the parent cache.

    The production caller fully verifies the cache once at ingress.  This
    consumer reauthenticates the immutable cache files through a selective
    session and materializes only the residual boundary tensors it uses.  A
    caller may share an already-open session across cache consumers; its
    lifetime remains owned by that caller.
    """

    if not isinstance(source_graph, ModelCircuitGraph):
        raise TypeError("source_graph must be ModelCircuitGraph.")
    if not isinstance(parent_cache, Lfm2CircuitTraceCache):
        raise TypeError("parent_cache must be Lfm2CircuitTraceCache.")
    if selective_session is not None and not isinstance(
        selective_session,
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise TypeError(
            "selective_session must be "
            "Lfm2CircuitTraceSelectiveAuthenticatedSession or None."
        )
    cache = parent_cache
    conv_widths = tuple(
        sorted(
            {
                _positive_int(value, "short_conv_internal_width")
                for value in short_conv_internal_widths
            }
        )
    )
    attention_options = tuple(
        sorted(
            {
                (
                    _positive_int(query, "query_heads"),
                    _positive_int(key_value, "key_value_heads"),
                    _positive_int(head_dim, "head_dim"),
                )
                for query, key_value, head_dim in attention_head_options
            }
        )
    )
    if len(conv_widths) < 2 or len(attention_options) < 2:
        raise Lfm2CachedHeterogeneousCapacityError(
            "Capacity discovery needs at least two operator choices."
        )
    automatic_widths = mlp_intermediate_sizes is None
    if automatic_widths and any(
        value is None
        for value in (
            target_hidden_size,
            target_vocab_size,
            exact_target_unique_parameters,
        )
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Automatic production MLP options require the target hidden size, "
            "target vocabulary size, and exact whole-child parameter count."
        )
    discovered = discover_lfm2_source_layers(source_graph)
    operator_delta = [0.0 for _ in discovered]
    mlp_delta = [0.0 for _ in discovered]
    operator_reference = [0.0 for _ in discovered]
    mlp_reference = [0.0 for _ in discovered]
    scalar_counts = [0 for _ in discovered]
    nonlinear_row_counts = [0 for _ in discovered]
    source_mlp_widths: list[int | None] = [None for _ in discovered]
    nonlinear_statistics: list[dict[str, Any] | None] = [None for _ in discovered]
    record_bindings = []
    session_context = (
        cache.selective_authenticated_session()
        if selective_session is None
        else nullcontext(selective_session)
    )
    with session_context as session:
        session.verify()
        if (
            session.path != cache.path
            or session.cache_key != cache.cache_key
            or session.batch_count != cache.batch_count
            or session.manifest != cache.manifest
        ):
            raise Lfm2CachedHeterogeneousCapacityError(
                "Selective session differs from the parent cache."
            )
        for expected_index, batch_reference in enumerate(session.batch_references):
            if batch_reference.index != expected_index:
                raise Lfm2CachedHeterogeneousCapacityError(
                    "Selective cache references are not in canonical batch order."
                )
            record = batch_reference.record
            phases = _ordered_parent_phases(record)
            record_bindings.append(
                {
                    "batch_index": batch_reference.index,
                    "input_sha256": batch_reference.input_sha256,
                    "record_sha256": record.get("record_sha256"),
                    "phase_order": list(phases),
                }
            )
            available_names = frozenset(batch_reference.ordered_tensor_names)
            for layer in discovered:
                selected_name_values = []
                for phase in phases:
                    prefix = f"{phase}.layer.{layer.ordinal:03d}."
                    endpoint_names = tuple(
                        prefix + endpoint
                        for endpoint in (
                            "residual.pre",
                            "operator.input",
                            "residual.post_operator",
                            "mlp.input",
                            "mlp.gate",
                            "mlp.up",
                            "mlp.post_swiglu",
                            "mlp.down",
                            "residual.post_layer",
                        )
                    )
                    final_name = (
                        f"{phase}.layer.{len(discovered) - 1:03d}.residual.post_layer"
                    )
                    required_names = (*endpoint_names, final_name)
                    if any(name not in available_names for name in required_names):
                        raise Lfm2CachedHeterogeneousCapacityError(
                            "Parent cache lacks authenticated operator-norm, "
                            "feed-forward, post-SwiGLU, writer, or downstream "
                            f"evidence for layer {layer.ordinal}, phase {phase}."
                        )
                    selected_name_values.extend(required_names)
                selected_names = tuple(sorted(set(selected_name_values)))
                payload = session.load_selected_tensors(
                    batch_reference.index,
                    selected_names,
                )
                try:
                    tensors = payload.tensors
                    for phase in phases:
                        prefix = f"{phase}.layer.{layer.ordinal:03d}."
                        pre = tensors[prefix + "residual.pre"]
                        post_operator = tensors[prefix + "residual.post_operator"]
                        post_layer = tensors[prefix + "residual.post_layer"]
                        value, count = _delta_energy(post_operator, pre)
                        reference, _ = _energy(pre)
                        operator_delta[layer.ordinal] += value
                        operator_reference[layer.ordinal] += reference
                        value, _ = _delta_energy(post_layer, post_operator)
                        reference, _ = _energy(post_operator)
                        mlp_delta[layer.ordinal] += value
                        mlp_reference[layer.ordinal] += reference
                        scalar_counts[layer.ordinal] += count
                        final_name = (
                            f"{phase}.layer.{len(discovered) - 1:03d}."
                            "residual.post_layer"
                        )
                        statistics, row_count = _cached_mlp_nonlinear_statistics(
                            operator_input=tensors[prefix + "operator.input"],
                            mlp_input=tensors[prefix + "mlp.input"],
                            gate=tensors[prefix + "mlp.gate"],
                            up=tensors[prefix + "mlp.up"],
                            post_swiglu=tensors[prefix + "mlp.post_swiglu"],
                            mlp_down=tensors[prefix + "mlp.down"],
                            post_operator=post_operator,
                            post_layer=post_layer,
                            final_post_layer=tensors[final_name],
                        )
                        width = int(statistics["post"].numel())
                        current_width = source_mlp_widths[layer.ordinal]
                        if current_width is None:
                            source_mlp_widths[layer.ordinal] = width
                        elif current_width != width:
                            raise Lfm2CachedHeterogeneousCapacityError(
                                "Source MLP width changed across cached causal phases."
                            )
                        aggregate = nonlinear_statistics[layer.ordinal]
                        if aggregate is None:
                            nonlinear_statistics[layer.ordinal] = {
                                name: tensor.clone()
                                for name, tensor in statistics.items()
                            }
                        else:
                            for name, tensor in statistics.items():
                                aggregate[name].add_(tensor)
                        nonlinear_row_counts[layer.ordinal] += row_count
                finally:
                    session.release_selected_tensors(payload)
        session.verify()
    if any(value is None for value in source_mlp_widths) or any(
        value is None for value in nonlinear_statistics
    ):
        raise Lfm2CachedHeterogeneousCapacityError(
            "Authenticated nonlinear MLP cache evidence is incomplete."
        )
    tiny = float.fromhex("0x1.0p-1022")
    operator_fraction = [
        value / max(reference, tiny)
        for value, reference in zip(operator_delta, operator_reference, strict=True)
    ]
    mlp_fraction = [
        value / max(reference, tiny)
        for value, reference in zip(mlp_delta, mlp_reference, strict=True)
    ]
    conv_indices = tuple(
        layer.ordinal for layer in discovered if layer.layer_type == "conv"
    )
    attention_indices = tuple(
        layer.ordinal for layer in discovered if layer.layer_type == "full_attention"
    )
    conv_pressure = _normalized_pressures(operator_fraction, conv_indices)
    attention_pressure = _normalized_pressures(operator_fraction, attention_indices)
    mlp_pressure = _normalized_pressures(
        mlp_fraction,
        tuple(layer.ordinal for layer in discovered),
    )
    resolved_source_widths = tuple(int(value) for value in source_mlp_widths)
    if automatic_widths:
        assert target_hidden_size is not None
        assert target_vocab_size is not None
        assert exact_target_unique_parameters is not None
        width_policy = _automatic_mlp_width_policy(
            layer_types=tuple(layer.layer_type for layer in discovered),
            source_widths=resolved_source_widths,
            short_conv_internal_widths=conv_widths,
            attention_head_options=attention_options,
            target_hidden_size=target_hidden_size,
            target_vocab_size=target_vocab_size,
            exact_target_unique_parameters=exact_target_unique_parameters,
            conv_kernel_size=conv_kernel_size,
        )
    else:
        assert mlp_intermediate_sizes is not None
        width_policy = _explicit_mlp_width_policy(
            mlp_intermediate_sizes,
            resolved_source_widths,
        )
    layer_evidence = []
    for layer in discovered:
        statistics = nonlinear_statistics[layer.ordinal]
        assert statistics is not None
        capacity_risks = _cached_mlp_capacity_risks(
            source_width=resolved_source_widths[layer.ordinal],
            candidate_widths=width_policy.widths_by_layer[layer.ordinal],
            nonlinear_statistics=statistics,
            layer_pressure=mlp_pressure[layer.ordinal],
        )
        body = {
            "schema": CAPACITY_EVIDENCE_SCHEMA,
            "source_graph_fingerprint": source_graph.fingerprint,
            "parent_cache_key": cache.cache_key,
            "record_bindings": record_bindings,
            "layer_index": layer.ordinal,
            "layer_type": layer.layer_type,
            "source_layer_sha256": layer.source_layer_sha256,
            "operator_delta_energy_fraction": operator_fraction[layer.ordinal],
            "mlp_delta_energy_fraction": mlp_fraction[layer.ordinal],
            "sampled_scalar_count": scalar_counts[layer.ordinal],
            "nonlinear_sampled_row_count": nonlinear_row_counts[layer.ordinal],
            "source_mlp_intermediate_size": (resolved_source_widths[layer.ordinal]),
            "nonlinear_endpoint_contract": [
                "operator.input",
                "mlp.input",
                "mlp.gate",
                "mlp.up",
                "mlp.post_swiglu",
                "mlp.down",
                "residual.post_operator",
                "residual.post_layer",
                "final.residual.post_layer",
            ],
            "mlp_capacity_risk_method": MLP_CAPACITY_RISK_METHOD,
            "mlp_capacity_risks": [value.as_dict() for value in capacity_risks],
            "mlp_option_policy": width_policy.policy_id,
            "exact_budget_lattice_sha256": (width_policy.exact_budget_lattice_sha256),
            "parent_model_calls": 0,
            "child_model_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
        layer_evidence.append(
            Lfm2CachedLayerCapacityEvidence(
                layer_index=layer.ordinal,
                layer_type=layer.layer_type,
                source_layer_sha256=layer.source_layer_sha256,
                operator_delta_energy_fraction=operator_fraction[layer.ordinal],
                mlp_delta_energy_fraction=mlp_fraction[layer.ordinal],
                sampled_scalar_count=scalar_counts[layer.ordinal],
                nonlinear_sampled_row_count=nonlinear_row_counts[layer.ordinal],
                source_mlp_intermediate_size=(resolved_source_widths[layer.ordinal]),
                mlp_capacity_risks=capacity_risks,
                selection_evidence_sha256=canonical_sha256(body),
            )
        )
    options = []
    for layer, evidence in zip(discovered, layer_evidence, strict=True):
        risk_by_width = {
            value.intermediate_size: value for value in evidence.mlp_capacity_risks
        }
        for intermediate in width_policy.widths_by_layer[layer.ordinal]:
            mlp_capacity_risk = risk_by_width[intermediate]
            mlp_risk = mlp_capacity_risk.predicted_defect
            if layer.layer_type == "conv":
                maximum_internal = max(conv_widths)
                for internal in conv_widths:
                    operator_loss = 1.0 - internal / maximum_internal
                    operator_risk = conv_pressure[layer.ordinal] * (
                        operator_loss * operator_loss
                    )
                    option_evidence = canonical_sha256(
                        {
                            "layer_evidence_sha256": (
                                evidence.selection_evidence_sha256
                            ),
                            "mlp_intermediate_size": intermediate,
                            "mlp_capacity_risk": mlp_capacity_risk.as_dict(),
                            "mlp_option_policy": width_policy.policy_id,
                            "exact_budget_lattice_sha256": (
                                width_policy.exact_budget_lattice_sha256
                            ),
                            "short_conv_internal_width": internal,
                            "operator_capacity_risk": operator_risk,
                        }
                    )
                    options.append(
                        build_lfm2_short_conv_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=intermediate,
                            internal_width=internal,
                            selection_evidence_sha256=option_evidence,
                            structural_risk=operator_risk + mlp_risk,
                        )
                    )
            else:
                maximum_attention_capacity = max(
                    head_dim * (query + key_value)
                    for query, key_value, head_dim in attention_options
                )
                for query, key_value, head_dim in attention_options:
                    attention_capacity = head_dim * (query + key_value)
                    operator_loss = (
                        1.0 - attention_capacity / maximum_attention_capacity
                    )
                    operator_risk = attention_pressure[layer.ordinal] * (
                        operator_loss * operator_loss
                    )
                    option_evidence = canonical_sha256(
                        {
                            "layer_evidence_sha256": (
                                evidence.selection_evidence_sha256
                            ),
                            "mlp_intermediate_size": intermediate,
                            "mlp_capacity_risk": mlp_capacity_risk.as_dict(),
                            "mlp_option_policy": width_policy.policy_id,
                            "exact_budget_lattice_sha256": (
                                width_policy.exact_budget_lattice_sha256
                            ),
                            "query_heads": query,
                            "key_value_heads": key_value,
                            "head_dim": head_dim,
                            "operator_capacity_risk": operator_risk,
                        }
                    )
                    options.append(
                        build_lfm2_attention_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=intermediate,
                            query_heads=query,
                            key_value_heads=key_value,
                            head_dim=head_dim,
                            selection_evidence_sha256=option_evidence,
                            structural_risk=operator_risk + mlp_risk,
                        )
                    )
    provisional = object.__new__(Lfm2CachedHeterogeneousCapacityOptionBank)
    object.__setattr__(
        provisional,
        "source_graph_fingerprint",
        source_graph.fingerprint,
    )
    object.__setattr__(provisional, "parent_cache_key", cache.cache_key)
    object.__setattr__(provisional, "mlp_option_policy", width_policy.policy_id)
    object.__setattr__(
        provisional,
        "exact_budget_lattice_sha256",
        width_policy.exact_budget_lattice_sha256,
    )
    object.__setattr__(provisional, "layer_evidence", tuple(layer_evidence))
    object.__setattr__(provisional, "layer_options", tuple(options))
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    evidence_sha256 = canonical_sha256(provisional._body())
    return Lfm2CachedHeterogeneousCapacityOptionBank(
        source_graph_fingerprint=source_graph.fingerprint,
        parent_cache_key=cache.cache_key,
        mlp_option_policy=width_policy.policy_id,
        exact_budget_lattice_sha256=(width_policy.exact_budget_lattice_sha256),
        layer_evidence=tuple(layer_evidence),
        layer_options=tuple(options),
        evidence_sha256=evidence_sha256,
    )


__all__ = [
    "AUTOMATIC_MLP_OPTION_POLICY",
    "CAPACITY_EVIDENCE_SCHEMA",
    "CAPACITY_OPTION_BANK_SCHEMA",
    "EXPLICIT_MLP_OPTION_POLICY",
    "MLP_CAPACITY_RISK_METHOD",
    "Lfm2CachedHeterogeneousCapacityError",
    "Lfm2CachedHeterogeneousCapacityOptionBank",
    "Lfm2CachedLayerCapacityEvidence",
    "Lfm2CachedMlpCapacityRisk",
    "Lfm2MlpWidthOptionPolicy",
    "build_lfm2_cached_heterogeneous_capacity_option_bank",
    "derive_lfm2_automatic_mlp_width_policy",
]
