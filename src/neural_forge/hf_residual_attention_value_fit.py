"""Installable V-side regression for residual-attention codebook folds.

The cached trace collector proves the exact value projection relation

``V_cache = concat(U_prefill, U_cached) @ W_v.T + bias``.

This module uses the *child* model's captured ``U`` rows as the design matrix
and the aligned parent site's appended V-cache rows as targets.  When a V
prototype is shared, one anchored ridge system is fitted across every logical
site which reads that prototype.  Per-site V role scales and canonical-to-
physical GQA permutations are part of the design; they are not ignored or
folded into an undocumented post-processing step.

Installing V is only the first half of an alternating experiment.  V changes
the actual child attention context ``A @ V``.  Therefore an O fit made before
the V installation is stale.  :func:`refit_output_after_value_installation`
accepts only newly collected child traces whose materialized V hash matches
the installed projection, and reruns the existing anchored O ridge on those
actual contexts.  That generic full-materialized-O result is diagnostic only;
it is not installable as one shared codebook O prototype.  The exact Qwen p35
runner instead uses its separately tested block-sparse survivor O fitter.  None
of this is training: there is no optimizer, gradient, or call to ``backward()``.

The module is deliberately narrower than the project's eventual universal
bridge: child/parent traces must use identical raw token IDs, and callers must
bind Q/K/O plus upstream runtime state externally around trace collection.
The executable Qwen p35 probe supplies that transaction-level snapshot audit.
"""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from neural_forge.hf_attention_codebook import (
    AttentionCodebookRegistry,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_cached_attention_trace import HFCachedAttentionTrace
from neural_forge.hf_residual_attention_fit import (
    ResidualAttentionFitLimits,
    ResidualAttentionRidgeFit,
    fit_attention_output_ridge,
)


VALUE_FIT_SCHEMA = "neural-forge-residual-attention-value-fit-v2"
_V_ROLE_INDEX = 2
VALUE_FIT_RUNTIME_CONTRACT: Mapping[str, Any] = MappingProxyType(
    {
        "supported_runtime_path": "prepared-Qwen3-codebook-identical-token-ids",
        "raw_token_id_alignment_required": True,
        "cross_tokenizer_mapping_supported": False,
        "full_qk_o_upstream_snapshot_bound_by_module": False,
        "external_full_runtime_snapshot_binding_required": True,
        "materialized_v_trace_binding_required": True,
    }
)


class ResidualAttentionValueFitError(RuntimeError):
    """A V calibration, global coverage proof, or installation is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionValueFitError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionValueFitError(f"{name} must be positive.") from error
    if result <= 0:
        raise ResidualAttentionValueFitError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionValueFitError(f"{name} cannot be negative.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionValueFitError(f"{name} cannot be negative.") from error
    if result < 0:
        raise ResidualAttentionValueFitError(f"{name} cannot be negative.")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ResidualAttentionValueFitError(f"{name} must be positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ResidualAttentionValueFitError(f"{name} must be positive.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ResidualAttentionValueFitError(f"{name} must be positive.")
    return result


def _array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    try:
        result = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ResidualAttentionValueFitError(
            f"{name} must be a finite numeric array."
        ) from error
    if result.size == 0 or (ndim is not None and result.ndim != ndim):
        suffix = "" if ndim is None else f" with rank {ndim}"
        raise ResidualAttentionValueFitError(f"{name} must be non-empty{suffix}.")
    if not bool(np.isfinite(result).all()):
        raise ResidualAttentionValueFitError(f"{name} contains NaN or infinity.")
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(
        contiguous.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _optional_tensor_sha256(value: Tensor | None) -> str | None:
    return None if value is None else _tensor_sha256(value)


def _relative_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = float(np.sqrt(np.mean(np.square(target - prediction))))
    scale = float(np.sqrt(np.mean(np.square(target))))
    return error / max(scale, np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class ResidualAttentionValueFitLimits:
    """Hard site, row, solve, and rollback-snapshot bounds."""

    max_sites: int = 4_096
    max_calibrations: int = 16_384
    max_rows: int = 65_536
    max_input_dimensions: int = 4_096
    max_head_dimensions: int = 512
    max_gram_elements: int = 16_777_216
    max_rhs_elements: int = 33_554_432
    max_snapshot_bytes: int = 512 * 1024 * 1024
    minimum_absolute_role_scale: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "max_sites",
            "max_calibrations",
            "max_rows",
            "max_input_dimensions",
            "max_head_dimensions",
            "max_gram_elements",
            "max_rhs_elements",
            "max_snapshot_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "minimum_absolute_role_scale",
            _positive_float(
                self.minimum_absolute_role_scale,
                "minimum_absolute_role_scale",
            ),
        )


@dataclass(frozen=True, slots=True)
class ValueSiteCalibration:
    """Aligned child-input/parent-value rows for one logical codebook site."""

    layer_index: int
    canonical_group: int
    physical_group: int
    input_rows: Any
    target_value_rows: Any
    sequence_id: str
    child_trace_sha256: str
    parent_trace_sha256: str
    child_v_weight_sha256: str
    child_v_bias_sha256: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "layer_index", _nonnegative_int(self.layer_index, "layer_index")
        )
        object.__setattr__(
            self,
            "canonical_group",
            _nonnegative_int(self.canonical_group, "canonical_group"),
        )
        object.__setattr__(
            self,
            "physical_group",
            _nonnegative_int(self.physical_group, "physical_group"),
        )
        inputs = _array(self.input_rows, "input_rows", ndim=2)
        targets = _array(self.target_value_rows, "target_value_rows", ndim=2)
        if int(inputs.shape[0]) != int(targets.shape[0]):
            raise ResidualAttentionValueFitError(
                "input_rows and target_value_rows must have equal row counts."
            )
        object.__setattr__(self, "input_rows", inputs)
        object.__setattr__(self, "target_value_rows", targets)
        for name in (
            "sequence_id",
            "child_trace_sha256",
            "parent_trace_sha256",
            "child_v_weight_sha256",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ResidualAttentionValueFitError(f"{name} cannot be blank.")
            object.__setattr__(self, name, value)
        if self.child_v_bias_sha256 is not None:
            bias_sha256 = str(self.child_v_bias_sha256).strip()
            if not bias_sha256:
                raise ResidualAttentionValueFitError(
                    "child_v_bias_sha256 cannot be blank when provided."
                )
            object.__setattr__(self, "child_v_bias_sha256", bias_sha256)

    @property
    def row_count(self) -> int:
        return int(self.input_rows.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "canonical_group": self.canonical_group,
            "physical_group": self.physical_group,
            "row_count": self.row_count,
            "input_width": int(self.input_rows.shape[1]),
            "head_dim": int(self.target_value_rows.shape[1]),
            "sequence_id": self.sequence_id,
            "child_trace_sha256": self.child_trace_sha256,
            "parent_trace_sha256": self.parent_trace_sha256,
            "child_v_weight_sha256": self.child_v_weight_sha256,
            "child_v_bias_sha256": self.child_v_bias_sha256,
            "input_rows_sha256": _array_sha256(self.input_rows),
            "target_value_rows_sha256": _array_sha256(self.target_value_rows),
            "design_source": "exact-child-v-proj-input",
            "target_source": "parent-post-update-v-cache",
            "runtime_contract": dict(VALUE_FIT_RUNTIME_CONTRACT),
        }


def _trace_sha256(trace: HFCachedAttentionTrace) -> str:
    value = trace.as_dict().get("trace_sha256")
    if not isinstance(value, str) or not value:
        raise ResidualAttentionValueFitError("Cached trace has no provenance hash.")
    return value


def _aligned_trace_pair(
    child: HFCachedAttentionTrace,
    parent: HFCachedAttentionTrace,
) -> None:
    if not isinstance(child, HFCachedAttentionTrace) or not isinstance(
        parent, HFCachedAttentionTrace
    ):
        raise ResidualAttentionValueFitError(
            "child and parent evidence must be cached attention traces."
        )
    if (
        child.sequence_id != parent.sequence_id
        or child.lane != parent.lane
        or child.layer_index != parent.layer_index
        or child.prompt_token_ids != parent.prompt_token_ids
        or child.consumed_token_ids != parent.consumed_token_ids
        or child.horizon != parent.horizon
        or child.key_value_heads != parent.key_value_heads
        or child.head_dim != parent.head_dim
    ):
        raise ResidualAttentionValueFitError(
            "Child and parent traces must share sequence, lane, layer, driven "
            "prefix, horizon, and KV topology."
        )
    if child.lane == "forced" and (
        child.selected_token_ids != parent.selected_token_ids
    ):
        raise ResidualAttentionValueFitError(
            "Forced child and parent traces must use identical driving tokens."
        )


def build_value_site_calibration(
    child_trace: HFCachedAttentionTrace,
    parent_trace: HFCachedAttentionTrace,
    registry: AttentionCodebookRegistry,
    *,
    canonical_group: int,
) -> ValueSiteCalibration:
    """Pair exact child ``U`` rows with aligned parent appended V-cache rows."""

    if not isinstance(registry, AttentionCodebookRegistry):
        raise ResidualAttentionValueFitError(
            "registry must be an AttentionCodebookRegistry."
        )
    _aligned_trace_pair(child_trace, parent_trace)
    layer = child_trace.layer_index
    if layer >= registry.logical_layers:
        raise ResidualAttentionValueFitError(
            "Trace layer is outside the supplied codebook registry."
        )
    live_v_weight, live_v_bias = registry.materialize_projection(layer, "v")
    live_v_weight_sha256 = _tensor_sha256(live_v_weight)
    live_v_bias_sha256 = _optional_tensor_sha256(live_v_bias)
    if (
        child_trace.value_projection_weight_sha256 != live_v_weight_sha256
        or child_trace.value_projection_bias_sha256 != live_v_bias_sha256
    ):
        raise ResidualAttentionValueFitError(
            "Child trace V hashes do not match the supplied live registry's "
            "materialized V projection."
        )
    canonical = _nonnegative_int(canonical_group, "canonical_group")
    if canonical >= registry.key_value_heads:
        raise ResidualAttentionValueFitError(
            "canonical_group is outside the codebook GQA topology."
        )
    physical = int(registry.group_permutations[layer, canonical].item())
    if physical not in child_trace.group_indices or physical not in (
        parent_trace.group_indices
    ):
        raise ResidualAttentionValueFitError(
            "Both traces must explicitly select the physical group named by "
            "group_permutations."
        )
    input_chunks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for child_step, parent_step in zip(
        child_trace.steps, parent_trace.steps, strict=True
    ):
        if child_step.prefix_length != parent_step.prefix_length:
            raise ResidualAttentionValueFitError(
                "Aligned trace steps have different cache prefix lengths."
            )
        chunk = _array(
            child_step.value_projection_input,
            "value_projection_input",
        )
        if chunk.ndim != 3 or int(chunk.shape[0]) != 1:
            raise ResidualAttentionValueFitError(
                "Cached v_proj input chunks must have shape [1, rows, width]."
            )
        row_count = int(chunk.shape[1])
        start = child_step.prefix_length - row_count
        if start < 0:
            raise ResidualAttentionValueFitError(
                "A v_proj input chunk is longer than its post-update cache."
            )
        parent_cache = parent_step.layer_value_cache
        if (
            parent_cache.ndim != 4
            or int(parent_cache.shape[0]) != 1
            or int(parent_cache.shape[1]) != parent_trace.key_value_heads
            or int(parent_cache.shape[2]) != parent_step.prefix_length
        ):
            raise ResidualAttentionValueFitError(
                "Parent trace does not contain an exact post-update V cache."
            )
        target = parent_cache[0, physical, start : parent_step.prefix_length, :]
        if int(target.shape[0]) != row_count:
            raise ResidualAttentionValueFitError(
                "Parent appended V-cache rows do not align with child U rows."
            )
        input_chunks.append(np.array(chunk[0], copy=True))
        targets.append(_array(target, "parent appended value rows", ndim=2))
    inputs = np.concatenate(input_chunks, axis=0)
    target_rows = np.concatenate(targets, axis=0)
    return ValueSiteCalibration(
        layer_index=layer,
        canonical_group=canonical,
        physical_group=physical,
        input_rows=inputs,
        target_value_rows=target_rows,
        sequence_id=child_trace.sequence_id,
        child_trace_sha256=_trace_sha256(child_trace),
        parent_trace_sha256=_trace_sha256(parent_trace),
        child_v_weight_sha256=live_v_weight_sha256,
        child_v_bias_sha256=live_v_bias_sha256,
    )


@dataclass(frozen=True, slots=True)
class SharedValueSiteReport:
    layer_index: int
    canonical_group: int
    physical_group: int
    row_count: int
    role_scale: float
    calibration_count: int

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class SharedValuePrototypeReport:
    layout: str
    survivor_group: int | None
    survivor_slot: int
    affected_sites: tuple[SharedValueSiteReport, ...]
    calibration_count: int
    row_count: int
    input_width: int
    head_dim: int
    has_bias: bool
    assignment_sha256: str
    group_permutations_sha256: str
    role_scales_sha256: str | None
    calibration_v_sha256_by_layer: tuple[tuple[int, str, str | None], ...]
    anchor_weight_sha256: str
    anchor_bias_sha256: str | None
    fitted_weight_sha256: str
    fitted_bias_sha256: str | None
    before_relative_rmse: float
    after_relative_rmse: float
    relative_anchor_drift: float
    ridge: float
    condition_number: float
    design_sha256: str
    target_sha256: str
    calibration_sha256s: tuple[str, ...]
    optimizer_steps: int = 0
    backward_calls: int = 0

    @property
    def zero_gradient(self) -> bool:
        return self.optimizer_steps == 0 and self.backward_calls == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": VALUE_FIT_SCHEMA,
            "layout": self.layout,
            "survivor_group": self.survivor_group,
            "survivor_slot": self.survivor_slot,
            "affected_sites": [site.as_dict() for site in self.affected_sites],
            "calibration_count": self.calibration_count,
            "row_count": self.row_count,
            "input_width": self.input_width,
            "head_dim": self.head_dim,
            "has_bias": self.has_bias,
            "assignment_sha256": self.assignment_sha256,
            "group_permutations_sha256": self.group_permutations_sha256,
            "role_scales_sha256": self.role_scales_sha256,
            "calibration_v_sha256_by_layer": [
                {
                    "layer_index": layer,
                    "weight_sha256": weight,
                    "bias_sha256": bias,
                }
                for layer, weight, bias in self.calibration_v_sha256_by_layer
            ],
            "anchor_weight_sha256": self.anchor_weight_sha256,
            "anchor_bias_sha256": self.anchor_bias_sha256,
            "fitted_weight_sha256": self.fitted_weight_sha256,
            "fitted_bias_sha256": self.fitted_bias_sha256,
            "before_relative_rmse": self.before_relative_rmse,
            "after_relative_rmse": self.after_relative_rmse,
            "relative_anchor_drift": self.relative_anchor_drift,
            "ridge": self.ridge,
            "condition_number": self.condition_number,
            "design_sha256": self.design_sha256,
            "target_sha256": self.target_sha256,
            "calibration_sha256s": list(self.calibration_sha256s),
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "zero_gradient": self.zero_gradient,
            "global_all_affected_site_fit": True,
            "role_scales_applied_in_design": True,
            "group_permutations_honored": True,
            "alternating_order": "install-V -> recollect-actual-child-A@V -> refit-O",
            "experimental_not_compaction_proof": True,
            "runtime_contract": dict(VALUE_FIT_RUNTIME_CONTRACT),
        }


@dataclass(frozen=True, slots=True)
class SharedValuePrototypeFit:
    weight: np.ndarray
    bias: np.ndarray | None
    report: SharedValuePrototypeReport

    def __post_init__(self) -> None:
        if not isinstance(self.report, SharedValuePrototypeReport):
            raise ResidualAttentionValueFitError(
                "report must be a SharedValuePrototypeReport."
            )
        weight = _array(self.weight, "fitted V weight", ndim=2)
        bias = None if self.bias is None else _array(self.bias, "fitted V bias", ndim=1)
        if (
            _array_sha256(weight) != self.report.fitted_weight_sha256
            or (None if bias is None else _array_sha256(bias))
            != self.report.fitted_bias_sha256
        ):
            raise ResidualAttentionValueFitError(
                "Fitted V payload hashes disagree with the fit report."
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)


def _registry_layout(registry: AttentionCodebookRegistry) -> str:
    return (
        "ragged"
        if isinstance(registry, RaggedAttentionCodebookRegistry)
        else "rectangular"
    )


def _target_prototype(
    registry: AttentionCodebookRegistry,
    *,
    survivor_group: int | None,
    survivor_slot: int,
) -> tuple[Tensor, Tensor | None]:
    if _registry_layout(registry) == "rectangular":
        assert survivor_group is not None
        weight = registry.prototype_v_weight[survivor_group, survivor_slot]
        bias = (
            None
            if registry.prototype_v_bias is None
            else registry.prototype_v_bias[survivor_group, survivor_slot]
        )
    else:
        weight = registry.prototype_v_weight[survivor_slot]
        bias = (
            None
            if registry.prototype_v_bias is None
            else registry.prototype_v_bias[survivor_slot]
        )
    return weight, bias


def _affected_site_keys(
    registry: AttentionCodebookRegistry,
    *,
    survivor_group: int | None,
    survivor_slot: int,
) -> tuple[tuple[int, int], ...]:
    assignments = registry.assignments.detach().cpu()
    if _registry_layout(registry) == "rectangular":
        assert survivor_group is not None
        return tuple(
            (layer, survivor_group)
            for layer in range(registry.logical_layers)
            if int(assignments[layer, survivor_group].item()) == survivor_slot
        )
    return tuple(
        (layer, group)
        for layer in range(registry.logical_layers)
        for group in range(registry.key_value_heads)
        if int(assignments[layer, group].item()) == survivor_slot
    )


def _validated_survivor(
    registry: AttentionCodebookRegistry,
    survivor_group: int | None,
    survivor_slot: int,
) -> tuple[int | None, int]:
    slot = _nonnegative_int(survivor_slot, "survivor_slot")
    if _registry_layout(registry) == "rectangular":
        if survivor_group is None:
            raise ResidualAttentionValueFitError(
                "survivor_group is required for a rectangular codebook."
            )
        group = _nonnegative_int(survivor_group, "survivor_group")
        if group >= registry.key_value_heads or slot >= (registry.prototypes_per_group):
            raise ResidualAttentionValueFitError(
                "The rectangular survivor group/slot is out of range."
            )
        return group, slot
    if survivor_group is not None:
        raise ResidualAttentionValueFitError(
            "survivor_group must be omitted for a ragged global prototype."
        )
    if slot >= registry.prototype_count:
        raise ResidualAttentionValueFitError(
            "The ragged survivor slot is out of range."
        )
    return None, slot


def fit_shared_codebook_value_prototype(
    registry: AttentionCodebookRegistry,
    calibrations: Sequence[ValueSiteCalibration],
    *,
    survivor_slot: int,
    survivor_group: int | None = None,
    ridge: float = 1e-3,
    limits: ResidualAttentionValueFitLimits | None = None,
) -> SharedValuePrototypeFit:
    """Fit one survivor V prototype across every site currently assigned to it."""

    if not isinstance(registry, AttentionCodebookRegistry):
        raise ResidualAttentionValueFitError(
            "registry must be an AttentionCodebookRegistry."
        )
    controls = limits or ResidualAttentionValueFitLimits()
    if not isinstance(controls, ResidualAttentionValueFitLimits):
        raise ResidualAttentionValueFitError(
            "limits must be ResidualAttentionValueFitLimits."
        )
    group, slot = _validated_survivor(registry, survivor_group, survivor_slot)
    evidence = tuple(calibrations)
    if (
        not evidence
        or len(evidence) > controls.max_calibrations
        or any(not isinstance(value, ValueSiteCalibration) for value in evidence)
    ):
        raise ResidualAttentionValueFitError(
            "calibrations must be a bounded non-empty ValueSiteCalibration sequence."
        )
    expected_sites = _affected_site_keys(
        registry, survivor_group=group, survivor_slot=slot
    )
    if not expected_sites or len(expected_sites) > controls.max_sites:
        raise ResidualAttentionValueFitError(
            "The survivor has no affected sites or exceeds max_sites."
        )
    supplied_sites = {(value.layer_index, value.canonical_group) for value in evidence}
    if supplied_sites != set(expected_sites):
        missing = sorted(set(expected_sites) - supplied_sites)
        extra = sorted(supplied_sites - set(expected_sites))
        raise ResidualAttentionValueFitError(
            "Global V fitting requires calibration for every and only affected "
            f"logical site; missing={missing}, extra={extra}."
        )
    anchor_weight_tensor, anchor_bias_tensor = _target_prototype(
        registry, survivor_group=group, survivor_slot=slot
    )
    calibration_v_sha256_by_layer = tuple(
        (
            layer,
            _tensor_sha256(weight),
            _optional_tensor_sha256(bias),
        )
        for layer in sorted({layer for layer, _canonical in expected_sites})
        for weight, bias in (registry.materialize_projection(layer, "v"),)
    )
    live_v_hashes = {
        layer: (weight_sha256, bias_sha256)
        for layer, weight_sha256, bias_sha256 in calibration_v_sha256_by_layer
    }
    input_width = int(anchor_weight_tensor.shape[1])
    head_dim = int(anchor_weight_tensor.shape[0])
    if (
        input_width > controls.max_input_dimensions
        or head_dim > controls.max_head_dimensions
    ):
        raise ResidualAttentionValueFitError(
            "V prototype dimensions exceed the declared solve bounds."
        )
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    site_reports: list[SharedValueSiteReport] = []
    calibration_hashes: list[str] = []
    for layer, canonical in expected_sites:
        site_evidence = tuple(
            value
            for value in evidence
            if (value.layer_index, value.canonical_group) == (layer, canonical)
        )
        physical = int(registry.group_permutations[layer, canonical].item())
        if any(value.physical_group != physical for value in site_evidence):
            raise ResidualAttentionValueFitError(
                "Calibration physical groups disagree with group_permutations."
            )
        if any(
            (
                value.child_v_weight_sha256,
                value.child_v_bias_sha256,
            )
            != live_v_hashes[layer]
            for value in site_evidence
        ):
            raise ResidualAttentionValueFitError(
                "A V calibration is stale or foreign to the supplied live registry."
            )
        scale = (
            1.0
            if registry.role_scales is None
            else float(
                registry.role_scales[layer, canonical, _V_ROLE_INDEX]
                .detach()
                .cpu()
                .item()
            )
        )
        if not math.isfinite(scale) or abs(scale) < (
            controls.minimum_absolute_role_scale
        ):
            raise ResidualAttentionValueFitError(
                "An affected site's V role scale is non-finite or too close to zero."
            )
        site_rows = 0
        for calibration in site_evidence:
            if tuple(calibration.input_rows.shape[1:]) != (input_width,) or tuple(
                calibration.target_value_rows.shape[1:]
            ) != (head_dim,):
                raise ResidualAttentionValueFitError(
                    "Calibration widths do not match the survivor V prototype."
                )
            design = np.array(calibration.input_rows * scale, copy=True)
            if anchor_bias_tensor is not None:
                design = np.concatenate(
                    (
                        design,
                        np.full((design.shape[0], 1), scale, dtype=np.float64),
                    ),
                    axis=1,
                )
            designs.append(design)
            targets.append(np.array(calibration.target_value_rows, copy=True))
            site_rows += calibration.row_count
            calibration_hashes.append(
                hashlib.sha256(repr(calibration.as_dict()).encode("utf-8")).hexdigest()
            )
        site_reports.append(
            SharedValueSiteReport(
                layer_index=layer,
                canonical_group=canonical,
                physical_group=physical,
                row_count=site_rows,
                role_scale=scale,
                calibration_count=len(site_evidence),
            )
        )
    design_rows = np.concatenate(designs, axis=0)
    target_rows = np.concatenate(targets, axis=0)
    if int(design_rows.shape[0]) > controls.max_rows:
        raise ResidualAttentionValueFitError("Global V calibration exceeds max_rows.")
    anchor_weight = _array(anchor_weight_tensor, "anchor V weight", ndim=2)
    if anchor_bias_tensor is None:
        anchor_augmented = anchor_weight
    else:
        anchor_bias = _array(anchor_bias_tensor, "anchor V bias", ndim=1)
        anchor_augmented = np.concatenate((anchor_weight, anchor_bias[:, None]), axis=1)
    ridge_fit = fit_attention_output_ridge(
        design_rows,
        target_rows,
        anchor_augmented,
        ridge=ridge,
        limits=ResidualAttentionFitLimits(
            max_rows=controls.max_rows,
            max_input_dimensions=controls.max_input_dimensions + 1,
            max_output_dimensions=controls.max_head_dimensions,
            max_gram_elements=controls.max_gram_elements,
            max_rhs_elements=controls.max_rhs_elements,
        ),
        method="global-scaled-child-U-to-parent-V-anchored-ridge",
    )
    if anchor_bias_tensor is None:
        fitted_weight = _array(ridge_fit.weight, "fitted V weight", ndim=2)
        fitted_bias = None
    else:
        fitted_weight = _array(ridge_fit.weight[:, :-1], "fitted V weight", ndim=2)
        fitted_bias = _array(ridge_fit.weight[:, -1], "fitted V bias", ndim=1)
    report = SharedValuePrototypeReport(
        layout=_registry_layout(registry),
        survivor_group=group,
        survivor_slot=slot,
        affected_sites=tuple(site_reports),
        calibration_count=len(evidence),
        row_count=int(design_rows.shape[0]),
        input_width=input_width,
        head_dim=head_dim,
        has_bias=anchor_bias_tensor is not None,
        assignment_sha256=_tensor_sha256(registry.assignments),
        group_permutations_sha256=_tensor_sha256(registry.group_permutations),
        role_scales_sha256=_optional_tensor_sha256(registry.role_scales),
        calibration_v_sha256_by_layer=calibration_v_sha256_by_layer,
        anchor_weight_sha256=_tensor_sha256(anchor_weight_tensor),
        anchor_bias_sha256=_optional_tensor_sha256(anchor_bias_tensor),
        fitted_weight_sha256=_array_sha256(fitted_weight),
        fitted_bias_sha256=(
            None if fitted_bias is None else _array_sha256(fitted_bias)
        ),
        before_relative_rmse=ridge_fit.report.before_relative_rmse,
        after_relative_rmse=ridge_fit.report.after_relative_rmse,
        relative_anchor_drift=ridge_fit.report.relative_anchor_drift,
        ridge=ridge_fit.report.ridge,
        condition_number=ridge_fit.report.condition_number,
        design_sha256=ridge_fit.report.design_sha256,
        target_sha256=ridge_fit.report.target_sha256,
        calibration_sha256s=tuple(calibration_hashes),
    )
    return SharedValuePrototypeFit(fitted_weight, fitted_bias, report)


def _non_v_hashes(registry: AttentionCodebookRegistry) -> Mapping[str, str]:
    hashes = {
        name: _tensor_sha256(value)
        for name, value in registry.state_dict().items()
        if name not in {"prototype_v_weight", "prototype_v_bias"}
    }
    return MappingProxyType(hashes)


def _unaffected_v_hashes(
    registry: AttentionCodebookRegistry,
    *,
    survivor_group: int | None,
    survivor_slot: int,
) -> Mapping[str, str]:
    hashes: dict[str, str] = {}
    if _registry_layout(registry) == "rectangular":
        assert survivor_group is not None
        for group in range(registry.key_value_heads):
            for slot in range(registry.prototypes_per_group):
                if (group, slot) == (survivor_group, survivor_slot):
                    continue
                key = f"prototype_v_weight[{group},{slot}]"
                hashes[key] = _tensor_sha256(registry.prototype_v_weight[group, slot])
                if registry.prototype_v_bias is not None:
                    hashes[f"prototype_v_bias[{group},{slot}]"] = _tensor_sha256(
                        registry.prototype_v_bias[group, slot]
                    )
    else:
        for slot in range(registry.prototype_count):
            if slot == survivor_slot:
                continue
            hashes[f"prototype_v_weight[{slot}]"] = _tensor_sha256(
                registry.prototype_v_weight[slot]
            )
            if registry.prototype_v_bias is not None:
                hashes[f"prototype_v_bias[{slot}]"] = _tensor_sha256(
                    registry.prototype_v_bias[slot]
                )
    return MappingProxyType(hashes)


@dataclass(frozen=True, slots=True)
class ValuePrototypeInstallationReport:
    layout: str
    survivor_group: int | None
    survivor_slot: int
    affected_sites: tuple[tuple[int, int], ...]
    before_weight_sha256: str
    before_bias_sha256: str | None
    installed_weight_sha256: str
    installed_bias_sha256: str | None
    assignment_sha256: str
    materialized_v_sha256_by_layer: tuple[tuple[int, str, str | None], ...]
    exact_unaffected_v_identity: bool
    exact_non_v_identity: bool
    parameter_identity_preserved: bool
    rollback_snapshot_bytes: int
    installation_token: str
    optimizer_steps: int = 0
    backward_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": VALUE_FIT_SCHEMA,
            "layout": self.layout,
            "survivor_group": self.survivor_group,
            "survivor_slot": self.survivor_slot,
            "affected_sites": [list(value) for value in self.affected_sites],
            "before_weight_sha256": self.before_weight_sha256,
            "before_bias_sha256": self.before_bias_sha256,
            "installed_weight_sha256": self.installed_weight_sha256,
            "installed_bias_sha256": self.installed_bias_sha256,
            "assignment_sha256": self.assignment_sha256,
            "materialized_v_sha256_by_layer": [
                {"layer_index": layer, "weight_sha256": weight, "bias_sha256": bias}
                for layer, weight, bias in self.materialized_v_sha256_by_layer
            ],
            "exact_unaffected_v_identity": self.exact_unaffected_v_identity,
            "exact_non_v_identity": self.exact_non_v_identity,
            "parameter_identity_preserved": self.parameter_identity_preserved,
            "rollback_snapshot_bytes": self.rollback_snapshot_bytes,
            "installation_token": self.installation_token,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "zero_gradient": self.optimizer_steps == self.backward_calls == 0,
            "alternating_order": "install-V -> recollect-actual-child-A@V -> refit-O",
            "experimental_not_compaction_proof": True,
            "runtime_contract": dict(VALUE_FIT_RUNTIME_CONTRACT),
        }


class InstalledValuePrototype:
    """Applied V mutation with exact target-slice rollback until committed."""

    def __init__(
        self,
        registry: AttentionCodebookRegistry,
        report: ValuePrototypeInstallationReport,
        original_weight: Tensor,
        original_bias: Tensor | None,
        original_non_v_hashes: Mapping[str, str],
        original_unaffected_hashes: Mapping[str, str],
    ) -> None:
        self.registry = registry
        self.report = report
        self._original_weight = original_weight
        self._original_bias = original_bias
        self._original_non_v_hashes = original_non_v_hashes
        self._original_unaffected_hashes = original_unaffected_hashes
        self._rolled_back = False
        self._committed = False

    @property
    def is_applied(self) -> bool:
        return not self._rolled_back

    @property
    def is_committed(self) -> bool:
        return self._committed

    def commit(self) -> ValuePrototypeInstallationReport:
        if self._rolled_back:
            raise ResidualAttentionValueFitError(
                "A rolled-back V installation cannot be committed."
            )
        self.assert_live()
        self._committed = True
        return self.report

    def assert_live(self) -> ValuePrototypeInstallationReport:
        """Prove without mutation that the installed V still materializes exactly."""

        if self._rolled_back:
            raise ResidualAttentionValueFitError(
                "A rolled-back V installation is no longer live."
            )
        self._assert_live_installation()
        return self.report

    def _assert_live_installation(self) -> None:
        """Fail closed unless the registry still materializes this installation."""

        group = self.report.survivor_group
        slot = self.report.survivor_slot
        current_weight, current_bias = _target_prototype(
            self.registry, survivor_group=group, survivor_slot=slot
        )
        if _tensor_sha256(current_weight) != self.report.installed_weight_sha256 or (
            _optional_tensor_sha256(current_bias) != self.report.installed_bias_sha256
        ):
            raise ResidualAttentionValueFitError(
                "The live registry no longer contains the installed V prototype."
            )
        if (
            _tensor_sha256(self.registry.assignments) != self.report.assignment_sha256
            or _affected_site_keys(
                self.registry,
                survivor_group=group,
                survivor_slot=slot,
            )
            != self.report.affected_sites
        ):
            raise ResidualAttentionValueFitError(
                "The live registry assignment map no longer matches the V installation."
            )
        current_materialized = tuple(
            (
                layer,
                _tensor_sha256(weight),
                _optional_tensor_sha256(bias),
            )
            for layer in sorted({layer for layer, _group in self.report.affected_sites})
            for weight, bias in (self.registry.materialize_projection(layer, "v"),)
        )
        if current_materialized != self.report.materialized_v_sha256_by_layer:
            raise ResidualAttentionValueFitError(
                "The live materialized V projections no longer match the installation."
            )

    def rollback(self) -> None:
        if self._committed:
            raise ResidualAttentionValueFitError(
                "A committed V installation cannot be rolled back by this handle."
            )
        if self._rolled_back:
            return
        group = self.report.survivor_group
        slot = self.report.survivor_slot
        current_weight, current_bias = _target_prototype(
            self.registry, survivor_group=group, survivor_slot=slot
        )
        if _tensor_sha256(current_weight) != self.report.installed_weight_sha256 or (
            _optional_tensor_sha256(current_bias) != self.report.installed_bias_sha256
        ):
            raise ResidualAttentionValueFitError(
                "The installed V target changed externally; rollback refuses to "
                "overwrite the newer state."
            )
        if dict(_non_v_hashes(self.registry)) != dict(
            self._original_non_v_hashes
        ) or dict(
            _unaffected_v_hashes(
                self.registry,
                survivor_group=group,
                survivor_slot=slot,
            )
        ) != dict(self._original_unaffected_hashes):
            raise ResidualAttentionValueFitError(
                "Unrelated registry state changed after V installation; rollback "
                "refuses before mutating the installed target."
            )
        with torch.no_grad():
            current_weight.copy_(self._original_weight)
            if current_bias is not None and self._original_bias is not None:
                current_bias.copy_(self._original_bias)
        restored_weight, restored_bias = _target_prototype(
            self.registry, survivor_group=group, survivor_slot=slot
        )
        target_restored = (
            _tensor_sha256(restored_weight) == self.report.before_weight_sha256
            and _optional_tensor_sha256(restored_bias) == self.report.before_bias_sha256
        )
        exact_identity = (
            target_restored
            and dict(_non_v_hashes(self.registry)) == dict(self._original_non_v_hashes)
            and dict(
                _unaffected_v_hashes(
                    self.registry,
                    survivor_group=group,
                    survivor_slot=slot,
                )
            )
            == dict(self._original_unaffected_hashes)
        )
        if target_restored:
            # Even if unrelated state changes concurrently after the preflight,
            # the handle must truthfully report that its V target is no longer
            # installed.
            self._rolled_back = True
        if not exact_identity:
            raise ResidualAttentionValueFitError(
                "Exact V rollback identity audit failed."
            )

    def __enter__(self) -> InstalledValuePrototype:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        if exc_type is not None or not self._committed:
            self.rollback()


def install_shared_codebook_value_prototype(
    registry: AttentionCodebookRegistry,
    fit: SharedValuePrototypeFit,
    *,
    limits: ResidualAttentionValueFitLimits | None = None,
) -> InstalledValuePrototype:
    """Install only the fitted V slice and certify every other tensor unchanged."""

    if not isinstance(registry, AttentionCodebookRegistry) or not isinstance(
        fit, SharedValuePrototypeFit
    ):
        raise ResidualAttentionValueFitError(
            "registry and fit must be codebook/value-fit objects."
        )
    controls = limits or ResidualAttentionValueFitLimits()
    report = fit.report
    group, slot = _validated_survivor(
        registry, report.survivor_group, report.survivor_slot
    )
    if (
        report.layout != _registry_layout(registry)
        or _tensor_sha256(registry.assignments) != report.assignment_sha256
        or _tensor_sha256(registry.group_permutations)
        != report.group_permutations_sha256
        or _optional_tensor_sha256(registry.role_scales) != report.role_scales_sha256
    ):
        raise ResidualAttentionValueFitError(
            "Registry assignments, permutations, scales, or layout changed after fit."
        )
    current_calibration_v_hashes = tuple(
        (
            layer,
            _tensor_sha256(weight),
            _optional_tensor_sha256(bias),
        )
        for layer, _weight_sha256, _bias_sha256 in (
            report.calibration_v_sha256_by_layer
        )
        for weight, bias in (registry.materialize_projection(layer, "v"),)
    )
    if current_calibration_v_hashes != report.calibration_v_sha256_by_layer:
        raise ResidualAttentionValueFitError(
            "The live materialized V projections changed after calibration fitting."
        )
    target_weight, target_bias = _target_prototype(
        registry, survivor_group=group, survivor_slot=slot
    )
    if (
        _tensor_sha256(target_weight) != report.anchor_weight_sha256
        or _optional_tensor_sha256(target_bias) != report.anchor_bias_sha256
    ):
        raise ResidualAttentionValueFitError(
            "Survivor V anchor changed after the global fit."
        )
    fitted_weight_array = _array(fit.weight, "fitted V weight", ndim=2)
    fitted_bias_array = (
        None if fit.bias is None else _array(fit.bias, "fitted V bias", ndim=1)
    )
    if (
        _array_sha256(fitted_weight_array) != report.fitted_weight_sha256
        or (None if fitted_bias_array is None else _array_sha256(fitted_bias_array))
        != report.fitted_bias_sha256
    ):
        raise ResidualAttentionValueFitError(
            "Fitted V payload changed after the report was created."
        )
    fitted_weight = torch.as_tensor(
        np.array(fitted_weight_array, copy=True),
        device=target_weight.device,
        dtype=target_weight.dtype,
    )
    if tuple(fitted_weight.shape) != tuple(target_weight.shape):
        raise ResidualAttentionValueFitError(
            "Fitted V weight does not match the survivor slice shape."
        )
    if target_bias is None:
        if fit.bias is not None:
            raise ResidualAttentionValueFitError(
                "A bias fit cannot be installed into a bias-free V prototype."
            )
        fitted_bias = None
    else:
        if fit.bias is None:
            raise ResidualAttentionValueFitError(
                "A biased V prototype requires a fitted bias."
            )
        fitted_bias = torch.as_tensor(
            np.array(fitted_bias_array, copy=True),
            device=target_bias.device,
            dtype=target_bias.dtype,
        )
        if tuple(fitted_bias.shape) != tuple(target_bias.shape):
            raise ResidualAttentionValueFitError(
                "Fitted V bias does not match the survivor slice shape."
            )
    original_weight = target_weight.detach().clone()
    original_bias = None if target_bias is None else target_bias.detach().clone()
    snapshot_bytes = int(original_weight.numel() * original_weight.element_size()) + (
        0
        if original_bias is None
        else int(original_bias.numel() * original_bias.element_size())
    )
    if snapshot_bytes > controls.max_snapshot_bytes:
        raise ResidualAttentionValueFitError(
            "V rollback snapshot exceeds max_snapshot_bytes."
        )
    before_non_v = _non_v_hashes(registry)
    before_unaffected = _unaffected_v_hashes(
        registry, survivor_group=group, survivor_slot=slot
    )
    parameter_ids = tuple(id(parameter) for parameter in registry.parameters())
    try:
        with torch.no_grad():
            target_weight.copy_(fitted_weight)
            if target_bias is not None and fitted_bias is not None:
                target_bias.copy_(fitted_bias)
        installed_weight_hash = _tensor_sha256(target_weight)
        installed_bias_hash = _optional_tensor_sha256(target_bias)
        exact_non_v = dict(before_non_v) == dict(_non_v_hashes(registry))
        exact_unaffected = dict(before_unaffected) == dict(
            _unaffected_v_hashes(registry, survivor_group=group, survivor_slot=slot)
        )
        parameter_identity = parameter_ids == tuple(
            id(parameter) for parameter in registry.parameters()
        )
        if not exact_non_v or not exact_unaffected or not parameter_identity:
            raise ResidualAttentionValueFitError(
                "V installation changed a non-V tensor, unaffected V slot, or "
                "parameter object identity."
            )
        affected = _affected_site_keys(
            registry, survivor_group=group, survivor_slot=slot
        )
        materialized: list[tuple[int, str, str | None]] = []
        for layer in sorted({layer for layer, _group in affected}):
            weight, bias = registry.materialize_projection(layer, "v")
            materialized.append(
                (layer, _tensor_sha256(weight), _optional_tensor_sha256(bias))
            )
        token = hashlib.sha256(
            repr(
                (
                    report.assignment_sha256,
                    installed_weight_hash,
                    installed_bias_hash,
                    tuple(materialized),
                )
            ).encode("utf-8")
        ).hexdigest()
        installation_report = ValuePrototypeInstallationReport(
            layout=report.layout,
            survivor_group=group,
            survivor_slot=slot,
            affected_sites=affected,
            before_weight_sha256=report.anchor_weight_sha256,
            before_bias_sha256=report.anchor_bias_sha256,
            installed_weight_sha256=installed_weight_hash,
            installed_bias_sha256=installed_bias_hash,
            assignment_sha256=report.assignment_sha256,
            materialized_v_sha256_by_layer=tuple(materialized),
            exact_unaffected_v_identity=exact_unaffected,
            exact_non_v_identity=exact_non_v,
            parameter_identity_preserved=parameter_identity,
            rollback_snapshot_bytes=snapshot_bytes,
            installation_token=token,
        )
    except Exception:
        with torch.no_grad():
            target_weight.copy_(original_weight)
            if target_bias is not None and original_bias is not None:
                target_bias.copy_(original_bias)
        raise
    return InstalledValuePrototype(
        registry,
        installation_report,
        original_weight,
        original_bias,
        before_non_v,
        before_unaffected,
    )


@dataclass(frozen=True, slots=True)
class AlternatingValueOutputFit:
    """O ridge explicitly fitted after V from actual child ``A @ V`` rows."""

    value_installation_token: str
    output_fit: ResidualAttentionRidgeFit
    child_context_sha256: str
    parent_delta_sha256: str
    trace_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": VALUE_FIT_SCHEMA,
            "value_installation_token": self.value_installation_token,
            "output_fit": self.output_fit.report.as_dict(),
            "child_context_sha256": self.child_context_sha256,
            "parent_delta_sha256": self.parent_delta_sha256,
            "trace_count": self.trace_count,
            "actual_child_attention_context_used": True,
            "v_installed_before_o_fit": True,
            "o_fit_scope": "full-materialized-projection-diagnostic",
            "shared_codebook_o_installable": False,
            "requires_block_sparse_shared_o_fitter": True,
            "alternating_order": "install-V -> recollect-actual-child-A@V -> refit-O",
            "experimental_not_compaction_proof": True,
            "runtime_contract": dict(VALUE_FIT_RUNTIME_CONTRACT),
        }


def refit_output_after_value_installation(
    installation: InstalledValuePrototype,
    child_traces: Sequence[HFCachedAttentionTrace],
    parent_traces: Sequence[HFCachedAttentionTrace],
    anchor_o_weight: Any,
    *,
    ridge: float = 1e-3,
    limits: ResidualAttentionFitLimits | None = None,
) -> AlternatingValueOutputFit:
    """Fit a diagnostic full O from post-install contexts verified against V.

    This helper does not install O and its full-width result is not a shared
    codebook-prototype update.  Installable folds must use a block-sparse O
    fitter which isolates the affected physical group columns and honors the
    survivor's global affected-site and role-scale contract.
    """

    if not isinstance(installation, InstalledValuePrototype) or not (
        installation.is_applied
    ):
        raise ResidualAttentionValueFitError(
            "A currently applied V installation is required before refitting O."
        )
    installation.assert_live()
    children = tuple(child_traces)
    parents = tuple(parent_traces)
    if not children or len(children) != len(parents):
        raise ResidualAttentionValueFitError(
            "Post-V child and parent trace collections must be equal and non-empty."
        )
    expected_hashes = {
        layer: (weight, bias)
        for layer, weight, bias in (installation.report.materialized_v_sha256_by_layer)
    }
    context_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for child, parent in zip(children, parents, strict=True):
        _aligned_trace_pair(child, parent)
        expected = expected_hashes.get(child.layer_index)
        if expected is None or expected != (
            child.value_projection_weight_sha256,
            child.value_projection_bias_sha256,
        ):
            raise ResidualAttentionValueFitError(
                "Child trace was not recollected from the installed materialized V "
                "projection; a pre-V O fit is stale."
            )
        for child_step, parent_step in zip(child.steps, parent.steps, strict=True):
            context_rows.append(
                _array(
                    child_step.full_attention_context,
                    "actual child A@V context",
                    ndim=2,
                )
            )
            target_rows.append(
                _array(
                    parent_step.attention_delta,
                    "parent attention residual delta",
                    ndim=2,
                )
            )
    contexts = np.concatenate(context_rows, axis=0)
    targets = np.concatenate(target_rows, axis=0)
    output_fit = fit_attention_output_ridge(
        contexts,
        targets,
        anchor_o_weight,
        ridge=ridge,
        limits=limits,
        method="post-V-actual-child-A@V-to-parent-residual-O-ridge",
    )
    return AlternatingValueOutputFit(
        value_installation_token=installation.report.installation_token,
        output_fit=output_fit,
        child_context_sha256=_array_sha256(contexts),
        parent_delta_sha256=_array_sha256(targets),
        trace_count=len(children),
    )


__all__ = [
    "AlternatingValueOutputFit",
    "InstalledValuePrototype",
    "ResidualAttentionValueFitError",
    "ResidualAttentionValueFitLimits",
    "SharedValuePrototypeFit",
    "SharedValuePrototypeReport",
    "SharedValueSiteReport",
    "VALUE_FIT_SCHEMA",
    "VALUE_FIT_RUNTIME_CONTRACT",
    "ValuePrototypeInstallationReport",
    "ValueSiteCalibration",
    "build_value_site_calibration",
    "fit_shared_codebook_value_prototype",
    "install_shared_codebook_value_prototype",
    "refit_output_after_value_installation",
]
