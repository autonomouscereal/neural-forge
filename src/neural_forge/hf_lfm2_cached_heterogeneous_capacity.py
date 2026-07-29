"""Derive heterogeneous LFM2 capacity options from sealed cached traces.

This module does not fold, train, or execute either model.  It converts the
relative energy of each cached operator and MLP residual write into a sealed
per-layer capacity pressure.  The heterogeneous exact-budget solver remains
responsible for choosing a globally legal whole-model layout.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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


CAPACITY_EVIDENCE_SCHEMA = "neural-forge-lfm2-cached-capacity-evidence-v1"
CAPACITY_OPTION_BANK_SCHEMA = "neural-forge-lfm2-cached-capacity-option-bank-v1"


class Lfm2CachedHeterogeneousCapacityError(RuntimeError):
    """Cached evidence cannot authenticate a heterogeneous option bank."""


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
        _digest(self.selection_evidence_sha256, "selection_evidence_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_type": self.layer_type,
            "source_layer_sha256": self.source_layer_sha256,
            "operator_delta_energy_fraction": (self.operator_delta_energy_fraction),
            "mlp_delta_energy_fraction": self.mlp_delta_energy_fraction,
            "sampled_scalar_count": self.sampled_scalar_count,
            "selection_evidence_sha256": self.selection_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CachedHeterogeneousCapacityOptionBank:
    """Authenticated option grid consumed by the exact layout selector."""

    source_graph_fingerprint: str
    parent_cache_key: str
    layer_evidence: tuple[Lfm2CachedLayerCapacityEvidence, ...]
    layer_options: tuple[Lfm2AuthenticatedLayerPhysicalOption, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        _digest(self.source_graph_fingerprint, "source_graph_fingerprint")
        _digest(self.parent_cache_key, "parent_cache_key")
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
    mlp_intermediate_sizes: Sequence[int] = (566, 567),
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
    mlp_widths = tuple(
        sorted(
            {
                _positive_int(value, "mlp_intermediate_size")
                for value in mlp_intermediate_sizes
            }
        )
    )
    if len(conv_widths) < 2 or len(attention_options) < 2 or len(mlp_widths) < 2:
        raise Lfm2CachedHeterogeneousCapacityError(
            "Capacity discovery needs at least two operator and MLP choices."
        )
    discovered = discover_lfm2_source_layers(source_graph)
    operator_delta = [0.0 for _ in discovered]
    mlp_delta = [0.0 for _ in discovered]
    operator_reference = [0.0 for _ in discovered]
    mlp_reference = [0.0 for _ in discovered]
    scalar_counts = [0 for _ in discovered]
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
            selected_name_values = []
            for layer in discovered:
                for phase in phases:
                    prefix = f"{phase}.layer.{layer.ordinal:03d}.residual."
                    boundary_names = tuple(
                        prefix + endpoint
                        for endpoint in (
                            "pre",
                            "post_operator",
                            "post_layer",
                        )
                    )
                    if any(name not in available_names for name in boundary_names):
                        raise Lfm2CachedHeterogeneousCapacityError(
                            "Parent cache lacks a complete residual boundary "
                            f"triple for layer {layer.ordinal}, phase {phase}."
                        )
                    selected_name_values.extend(boundary_names)
            selected_names = tuple(sorted(selected_name_values))
            payload = session.load_selected_tensors(
                batch_reference.index,
                selected_names,
            )
            try:
                tensors = payload.tensors
                for layer in discovered:
                    for phase in phases:
                        prefix = f"{phase}.layer.{layer.ordinal:03d}.residual."
                        pre = tensors[prefix + "pre"]
                        post_operator = tensors[prefix + "post_operator"]
                        post_layer = tensors[prefix + "post_layer"]
                        value, count = _delta_energy(post_operator, pre)
                        reference, _ = _energy(pre)
                        operator_delta[layer.ordinal] += value
                        operator_reference[layer.ordinal] += reference
                        value, _ = _delta_energy(post_layer, post_operator)
                        reference, _ = _energy(post_operator)
                        mlp_delta[layer.ordinal] += value
                        mlp_reference[layer.ordinal] += reference
                        scalar_counts[layer.ordinal] += count
            finally:
                session.release_selected_tensors(payload)
        session.verify()
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
    layer_evidence = []
    for layer in discovered:
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
            "parent_model_calls": 0,
            "training_steps": 0,
        }
        layer_evidence.append(
            Lfm2CachedLayerCapacityEvidence(
                layer_index=layer.ordinal,
                layer_type=layer.layer_type,
                source_layer_sha256=layer.source_layer_sha256,
                operator_delta_energy_fraction=operator_fraction[layer.ordinal],
                mlp_delta_energy_fraction=mlp_fraction[layer.ordinal],
                sampled_scalar_count=scalar_counts[layer.ordinal],
                selection_evidence_sha256=canonical_sha256(body),
            )
        )
    options = []
    for layer, evidence in zip(discovered, layer_evidence, strict=True):
        for intermediate in mlp_widths:
            mlp_risk = (
                mlp_pressure[layer.ordinal] if intermediate == mlp_widths[0] else 0.0
            )
            if layer.layer_type == "conv":
                for internal in conv_widths:
                    operator_risk = (
                        conv_pressure[layer.ordinal]
                        if internal == conv_widths[0]
                        else 0.0
                    )
                    option_evidence = canonical_sha256(
                        {
                            "layer_evidence_sha256": (
                                evidence.selection_evidence_sha256
                            ),
                            "mlp_intermediate_size": intermediate,
                            "short_conv_internal_width": internal,
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
                for query, key_value, head_dim in attention_options:
                    operator_risk = (
                        attention_pressure[layer.ordinal]
                        if (query, key_value, head_dim) == attention_options[0]
                        else 0.0
                    )
                    option_evidence = canonical_sha256(
                        {
                            "layer_evidence_sha256": (
                                evidence.selection_evidence_sha256
                            ),
                            "mlp_intermediate_size": intermediate,
                            "query_heads": query,
                            "key_value_heads": key_value,
                            "head_dim": head_dim,
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
    object.__setattr__(provisional, "layer_evidence", tuple(layer_evidence))
    object.__setattr__(provisional, "layer_options", tuple(options))
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    evidence_sha256 = canonical_sha256(provisional._body())
    return Lfm2CachedHeterogeneousCapacityOptionBank(
        source_graph_fingerprint=source_graph.fingerprint,
        parent_cache_key=cache.cache_key,
        layer_evidence=tuple(layer_evidence),
        layer_options=tuple(options),
        evidence_sha256=evidence_sha256,
    )


__all__ = [
    "CAPACITY_EVIDENCE_SCHEMA",
    "CAPACITY_OPTION_BANK_SCHEMA",
    "Lfm2CachedHeterogeneousCapacityError",
    "Lfm2CachedHeterogeneousCapacityOptionBank",
    "Lfm2CachedLayerCapacityEvidence",
    "build_lfm2_cached_heterogeneous_capacity_option_bank",
]
