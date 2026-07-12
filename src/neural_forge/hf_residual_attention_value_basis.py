"""Residual-targeted basis fitting for a shared attention V prototype.

Raw parent-V transplantation is not necessarily a useful objective after Q/K/O
have already been folded.  This module instead keeps the candidate's captured
Q/K attention probabilities and current O projection fixed, evaluates a small
explicit basis of *existing* V prototypes through that residual path, and fits
only the basis coefficients against the parent's isolated selected-group
residual contribution.

The solve is deliberately tiny and zero-gradient.  A box-constrained anchored
ridge is solved by finite active-set enumeration, then the selected linear
combination is materialized into the existing survivor V tensor.  No runtime
coefficient, adapter, optimizer state, or new parameter is retained.  Every
logical site sharing the survivor must participate in the fit, including its
V/O role scales and canonical-to-physical GQA permutation.

This is the V half of an alternating operation.  After installation callers
must recollect the real child traces and run the existing block-sparse O fit on
the new ``A @ V`` contexts.  The fixed-attention calibration is not a claim
that downstream Q/K states remain fixed after V is installed.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from neural_forge.hf_attention_codebook import (
    AttentionCodebookRegistry,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_cached_attention_trace import HFCachedAttentionTrace
from neural_forge.hf_residual_attention_value_fit import (
    InstalledValuePrototype,
    ResidualAttentionValueFitLimits,
    SharedValuePrototypeFit,
    SharedValuePrototypeReport,
    SharedValueSiteReport,
    install_shared_codebook_value_prototype,
)


RESIDUAL_VALUE_BASIS_SCHEMA = "neural-forge-residual-targeted-value-basis-v2"
_V_ROLE_INDEX = 2
_O_ROLE_INDEX = 3
BasisRole = Literal["survivor", "removed-source", "auxiliary"]


class ResidualAttentionValueBasisError(RuntimeError):
    """A basis, trace, global coverage proof, solve, or installation is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionValueBasisError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionValueBasisError(f"{name} must be positive.") from error
    if result <= 0:
        raise ResidualAttentionValueBasisError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionValueBasisError(f"{name} cannot be negative.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionValueBasisError(f"{name} cannot be negative.") from error
    if result < 0:
        raise ResidualAttentionValueBasisError(f"{name} cannot be negative.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ResidualAttentionValueBasisError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ResidualAttentionValueBasisError(f"{name} must be finite.") from error
    if not math.isfinite(result):
        raise ResidualAttentionValueBasisError(f"{name} must be finite.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ResidualAttentionValueBasisError(f"{name} must be positive.")
    return result


def _array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        value = value.numpy()
    try:
        result = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ResidualAttentionValueBasisError(
            f"{name} must be a finite numeric array."
        ) from error
    if result.size == 0 or (ndim is not None and result.ndim != ndim):
        suffix = "" if ndim is None else f" with rank {ndim}"
        raise ResidualAttentionValueBasisError(f"{name} must be non-empty{suffix}.")
    if not bool(np.isfinite(result).all()):
        raise ResidualAttentionValueBasisError(f"{name} contains NaN or infinity.")
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


def _source_dtype(value: Any, name: str) -> torch.dtype:
    normalized = str(value).strip().removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ResidualAttentionValueBasisError(
            f"{name} must identify float16, bfloat16, float32, or float64."
        )
    return dtype


def _payload_sha256(weight: np.ndarray, bias: np.ndarray | None) -> str:
    return hashlib.sha256(
        repr(
            (
                _array_sha256(weight),
                None if bias is None else _array_sha256(bias),
            )
        ).encode("utf-8")
    ).hexdigest()


def _array_source_sha256(value: np.ndarray, dtype: torch.dtype) -> str:
    restored = torch.as_tensor(np.array(value, copy=True), dtype=dtype)
    return _tensor_sha256(restored)


def _registry_sha256(registry: AttentionCodebookRegistry) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(registry.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _trace_sha256(trace: HFCachedAttentionTrace) -> str:
    value = trace.as_dict().get("trace_sha256")
    if not isinstance(value, str) or not value:
        raise ResidualAttentionValueBasisError("Cached trace has no provenance hash.")
    return value


def _relative_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = float(np.sqrt(np.mean(np.square(target - prediction))))
    scale = float(np.sqrt(np.mean(np.square(target))))
    return error / max(scale, np.finfo(np.float64).eps)


def _registry_layout(registry: AttentionCodebookRegistry) -> str:
    return (
        "ragged"
        if isinstance(registry, RaggedAttentionCodebookRegistry)
        else "rectangular"
    )


def _flat_prototype_tensor(value: Tensor) -> Tensor:
    if value.ndim == 3:
        return value
    if value.ndim == 4:
        return value.reshape(-1, *value.shape[2:])
    raise ResidualAttentionValueBasisError(
        "Prototype tensor must use ragged [prototype,...] or rectangular "
        "[group,slot,...] storage."
    )


def _target_prototype(
    registry: AttentionCodebookRegistry,
    *,
    survivor_group: int | None,
    survivor_slot: int,
) -> tuple[Tensor, Tensor | None, int]:
    slot = _nonnegative_int(survivor_slot, "survivor_slot")
    if _registry_layout(registry) == "rectangular":
        if survivor_group is None:
            raise ResidualAttentionValueBasisError(
                "survivor_group is required for a rectangular codebook."
            )
        group = _nonnegative_int(survivor_group, "survivor_group")
        if group >= registry.key_value_heads or slot >= registry.prototypes_per_group:
            raise ResidualAttentionValueBasisError(
                "The rectangular survivor group/slot is out of range."
            )
        weight = registry.prototype_v_weight[group, slot]
        bias = (
            None
            if registry.prototype_v_bias is None
            else registry.prototype_v_bias[group, slot]
        )
        active_slot = group * registry.prototypes_per_group + slot
    else:
        if survivor_group is not None or slot >= registry.prototype_count:
            raise ResidualAttentionValueBasisError(
                "A ragged survivor uses only one in-range global slot."
            )
        weight = registry.prototype_v_weight[slot]
        bias = (
            None
            if registry.prototype_v_bias is None
            else registry.prototype_v_bias[slot]
        )
        active_slot = slot
    return weight, bias, active_slot


def _affected_site_keys(
    registry: AttentionCodebookRegistry,
    *,
    survivor_group: int | None,
    survivor_slot: int,
) -> tuple[tuple[int, int], ...]:
    _weight, _bias, active_slot = _target_prototype(
        registry,
        survivor_group=survivor_group,
        survivor_slot=survivor_slot,
    )
    assignments = registry.assignments.detach().cpu().to(torch.long)
    if _registry_layout(registry) == "rectangular":
        assert survivor_group is not None
        return tuple(
            (layer, survivor_group)
            for layer in range(registry.logical_layers)
            if int(assignments[layer, survivor_group].item()) == survivor_slot
        )
    return tuple(
        (layer, canonical)
        for layer in range(registry.logical_layers)
        for canonical in range(registry.key_value_heads)
        if int(assignments[layer, canonical].item()) == active_slot
    )


@dataclass(frozen=True, slots=True)
class ResidualValueBasisLimits:
    """Hard bounds for evidence, basis width, active sets, and scalar solve size."""

    max_sites: int = 4_096
    max_calibrations: int = 16_384
    max_rows: int = 65_536
    max_basis_size: int = 8
    max_active_sets: int = 6_561
    max_scalar_observations: int = 67_108_864
    max_input_dimensions: int = 4_096
    max_head_dimensions: int = 512
    max_output_dimensions: int = 16_384
    max_condition_number: float = 1e14
    minimum_absolute_role_scale: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "max_sites",
            "max_calibrations",
            "max_rows",
            "max_basis_size",
            "max_active_sets",
            "max_scalar_observations",
            "max_input_dimensions",
            "max_head_dimensions",
            "max_output_dimensions",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_condition_number",
            _positive_float(self.max_condition_number, "max_condition_number"),
        )
        object.__setattr__(
            self,
            "minimum_absolute_role_scale",
            _positive_float(
                self.minimum_absolute_role_scale,
                "minimum_absolute_role_scale",
            ),
        )


@dataclass(frozen=True, slots=True)
class ValuePrototypeBasisEntry:
    """One immutable, raw (unscaled) existing V prototype in the small basis."""

    basis_id: str
    role: BasisRole
    source_flat_index: int
    source_active_slot: int
    source_layout: str
    source_registry_sha256: str
    source_dtype: str
    weight: Any
    bias: Any | None
    source_weight_sha256: str
    source_bias_sha256: str | None
    _captured_payload_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        basis_id = str(self.basis_id).strip()
        if not basis_id:
            raise ResidualAttentionValueBasisError("basis_id cannot be blank.")
        role = str(self.role).strip().lower()
        if role not in {"survivor", "removed-source", "auxiliary"}:
            raise ResidualAttentionValueBasisError(
                "basis role must be survivor, removed-source, or auxiliary."
            )
        object.__setattr__(self, "basis_id", basis_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "source_flat_index",
            _nonnegative_int(self.source_flat_index, "source_flat_index"),
        )
        object.__setattr__(
            self,
            "source_active_slot",
            _nonnegative_int(self.source_active_slot, "source_active_slot"),
        )
        layout = str(self.source_layout).strip().lower()
        if layout not in {"ragged", "rectangular"}:
            raise ResidualAttentionValueBasisError(
                "source_layout must be ragged or rectangular."
            )
        object.__setattr__(self, "source_layout", layout)
        registry_hash = str(self.source_registry_sha256).strip()
        source_weight_hash = str(self.source_weight_sha256).strip()
        if not registry_hash or not source_weight_hash:
            raise ResidualAttentionValueBasisError(
                "Basis source registry/weight hashes cannot be blank."
            )
        object.__setattr__(self, "source_registry_sha256", registry_hash)
        object.__setattr__(self, "source_weight_sha256", source_weight_hash)
        source_dtype = _source_dtype(self.source_dtype, "source_dtype")
        object.__setattr__(self, "source_dtype", str(source_dtype))
        weight = _array(self.weight, "basis weight", ndim=2)
        bias = None if self.bias is None else _array(self.bias, "basis bias", ndim=1)
        if bias is not None and tuple(bias.shape) != (int(weight.shape[0]),):
            raise ResidualAttentionValueBasisError(
                "Basis bias width must match its V output width."
            )
        if (bias is None) != (self.source_bias_sha256 is None):
            raise ResidualAttentionValueBasisError(
                "Basis payload/source bias presence must agree."
            )
        if self.source_bias_sha256 is not None:
            bias_hash = str(self.source_bias_sha256).strip()
            if not bias_hash:
                raise ResidualAttentionValueBasisError(
                    "source_bias_sha256 cannot be blank when present."
                )
            object.__setattr__(self, "source_bias_sha256", bias_hash)
        if (
            _array_source_sha256(weight, source_dtype) != source_weight_hash
            or (None if bias is None else _array_source_sha256(bias, source_dtype))
            != self.source_bias_sha256
        ):
            raise ResidualAttentionValueBasisError(
                "Basis payload does not round-trip to its captured source hashes."
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(
            self,
            "_captured_payload_sha256",
            _payload_sha256(weight, bias),
        )

    @property
    def payload_sha256(self) -> str:
        return self._captured_payload_sha256

    def assert_untampered(self) -> None:
        source_dtype = _source_dtype(self.source_dtype, "source_dtype")
        if (
            _payload_sha256(self.weight, self.bias) != self._captured_payload_sha256
            or _array_source_sha256(self.weight, source_dtype)
            != self.source_weight_sha256
            or (
                None
                if self.bias is None
                else _array_source_sha256(self.bias, source_dtype)
            )
            != self.source_bias_sha256
        ):
            raise ResidualAttentionValueBasisError(
                f"Basis entry {self.basis_id!r} changed after capture."
            )

    def as_dict(self) -> dict[str, Any]:
        self.assert_untampered()
        return {
            "basis_id": self.basis_id,
            "role": self.role,
            "source_flat_index": self.source_flat_index,
            "source_active_slot": self.source_active_slot,
            "source_layout": self.source_layout,
            "source_registry_sha256": self.source_registry_sha256,
            "source_dtype": self.source_dtype,
            "source_weight_sha256": self.source_weight_sha256,
            "source_bias_sha256": self.source_bias_sha256,
            "weight_payload_sha256": _array_sha256(self.weight),
            "bias_payload_sha256": (
                None if self.bias is None else _array_sha256(self.bias)
            ),
            "payload_sha256": self.payload_sha256,
            "shape": list(self.weight.shape),
        }


def capture_value_prototype_basis_entry(
    registry: AttentionCodebookRegistry,
    *,
    source_flat_index: int,
    basis_id: str,
    role: BasisRole,
) -> ValuePrototypeBasisEntry:
    """Capture one real V prototype by immutable source provenance.

    Capture removed-source entries from the incumbent registry *before* a
    tentative redirect deletes them.  The returned CPU float64 payload remains
    usable after the proposed registry is installed.
    """

    if not isinstance(registry, AttentionCodebookRegistry):
        raise ResidualAttentionValueBasisError(
            "registry must be an AttentionCodebookRegistry."
        )
    source = _nonnegative_int(source_flat_index, "source_flat_index")
    sources = (
        registry.prototype_source_indices.detach().cpu().reshape(-1).to(torch.long)
    )
    matches = torch.nonzero(sources == source, as_tuple=False).reshape(-1)
    if int(matches.numel()) != 1:
        raise ResidualAttentionValueBasisError(
            "source_flat_index must identify exactly one active real prototype."
        )
    active_slot = int(matches[0].item())
    flat_weight = _flat_prototype_tensor(registry.prototype_v_weight)
    flat_bias = (
        None
        if registry.prototype_v_bias is None
        else _flat_prototype_tensor(registry.prototype_v_bias.unsqueeze(-1)).squeeze(-1)
    )
    weight = flat_weight[active_slot]
    bias = None if flat_bias is None else flat_bias[active_slot]
    return ValuePrototypeBasisEntry(
        basis_id=basis_id,
        role=role,
        source_flat_index=source,
        source_active_slot=active_slot,
        source_layout=_registry_layout(registry),
        source_registry_sha256=_registry_sha256(registry),
        source_dtype=str(weight.dtype),
        weight=weight,
        bias=bias,
        source_weight_sha256=_tensor_sha256(weight),
        source_bias_sha256=_optional_tensor_sha256(bias),
    )


@dataclass(frozen=True, slots=True)
class ParentOutputBlockBasisEntry:
    """One incumbent materialized O block bound to its exact logical source site."""

    source_registry_sha256: str
    layer_index: int
    canonical_group: int
    physical_group: int
    source_flat_index: int
    source_active_slot: int
    source_dtype: str
    block: Any
    source_block_sha256: str
    _captured_payload_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        registry_hash = str(self.source_registry_sha256).strip()
        block_hash = str(self.source_block_sha256).strip()
        if not registry_hash or not block_hash:
            raise ResidualAttentionValueBasisError(
                "Parent O registry and block hashes cannot be blank."
            )
        object.__setattr__(self, "source_registry_sha256", registry_hash)
        object.__setattr__(self, "source_block_sha256", block_hash)
        for name in (
            "layer_index",
            "canonical_group",
            "physical_group",
            "source_flat_index",
            "source_active_slot",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        source_dtype = _source_dtype(self.source_dtype, "source_dtype")
        object.__setattr__(self, "source_dtype", str(source_dtype))
        block = _array(self.block, "parent materialized O block", ndim=2)
        if _array_source_sha256(block, source_dtype) != block_hash:
            raise ResidualAttentionValueBasisError(
                "Parent O block does not round-trip to its captured source hash."
            )
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "_captured_payload_sha256", _array_sha256(block))

    @property
    def payload_sha256(self) -> str:
        return self._captured_payload_sha256

    @property
    def source_site(self) -> tuple[int, int, int]:
        return self.layer_index, self.canonical_group, self.physical_group

    def assert_untampered(self) -> None:
        source_dtype = _source_dtype(self.source_dtype, "source_dtype")
        if (
            _array_sha256(self.block) != self._captured_payload_sha256
            or _array_source_sha256(self.block, source_dtype)
            != self.source_block_sha256
        ):
            raise ResidualAttentionValueBasisError(
                "Parent O block changed after capture."
            )

    def as_dict(self) -> dict[str, Any]:
        self.assert_untampered()
        payload = {
            "schema": RESIDUAL_VALUE_BASIS_SCHEMA,
            "source_registry_sha256": self.source_registry_sha256,
            "layer_index": self.layer_index,
            "canonical_group": self.canonical_group,
            "physical_group": self.physical_group,
            "source_flat_index": self.source_flat_index,
            "source_active_slot": self.source_active_slot,
            "source_dtype": self.source_dtype,
            "source_block_sha256": self.source_block_sha256,
            "block_payload_sha256": self.payload_sha256,
            "shape": list(self.block.shape),
            "bias_excluded": True,
        }
        payload["capture_sha256"] = hashlib.sha256(
            repr(payload).encode("utf-8")
        ).hexdigest()
        return payload


def capture_parent_output_block_basis_entry(
    registry: AttentionCodebookRegistry,
    *,
    layer_index: int,
    canonical_group: int,
) -> ParentOutputBlockBasisEntry:
    """Capture one incumbent materialized physical O block with full provenance."""

    if not isinstance(registry, AttentionCodebookRegistry):
        raise ResidualAttentionValueBasisError(
            "registry must be an AttentionCodebookRegistry."
        )
    layer = _nonnegative_int(layer_index, "layer_index")
    canonical = _nonnegative_int(canonical_group, "canonical_group")
    if layer >= registry.logical_layers or canonical >= registry.key_value_heads:
        raise ResidualAttentionValueBasisError(
            "Parent O source layer/group is outside the registry topology."
        )
    physical = int(registry.group_permutations[layer, canonical].item())
    assignment_slot = int(registry.assignments[layer, canonical].item())
    if _registry_layout(registry) == "rectangular":
        active_slot = canonical * registry.prototypes_per_group + assignment_slot
        source_flat = int(
            registry.prototype_source_indices[canonical, assignment_slot].item()
        )
    else:
        active_slot = assignment_slot
        source_flat = int(registry.prototype_source_indices[active_slot].item())
    weight, _bias = registry.materialize_projection(layer, "o")
    q_rows = registry.query_rows_per_group
    block = weight[:, physical * q_rows : (physical + 1) * q_rows]
    return ParentOutputBlockBasisEntry(
        source_registry_sha256=_registry_sha256(registry),
        layer_index=layer,
        canonical_group=canonical,
        physical_group=physical,
        source_flat_index=source_flat,
        source_active_slot=active_slot,
        source_dtype=str(block.dtype),
        block=block,
        source_block_sha256=_tensor_sha256(block),
    )


@dataclass(frozen=True, slots=True)
class ResidualValueBasisSiteCalibration:
    """Fixed-candidate-attention basis writes and parent target for one site."""

    layer_index: int
    canonical_group: int
    physical_group: int
    sequence_id: str
    basis_ids: tuple[str, ...]
    basis_payload_sha256s: tuple[str, ...]
    basis_contribution_rows: Any
    target_contribution_rows: Any
    candidate_trace_sha256: str
    parent_trace_sha256: str
    parent_o_provenance: ParentOutputBlockBasisEntry
    candidate_registry_sha256: str
    candidate_v_weight_sha256: str
    candidate_v_bias_sha256: str | None
    candidate_o_block_sha256: str
    parent_o_block_sha256: str
    parent_registry_sha256: str
    parent_o_capture_sha256: str
    parent_o_source_flat_index: int
    parent_o_source_active_slot: int
    value_role_scale: float
    output_role_scale: float
    _captured_payload_sha256: str = field(init=False, repr=False)

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
        sequence_id = str(self.sequence_id).strip()
        if not sequence_id:
            raise ResidualAttentionValueBasisError("sequence_id cannot be blank.")
        object.__setattr__(self, "sequence_id", sequence_id)
        basis_ids = tuple(str(value).strip() for value in self.basis_ids)
        hashes = tuple(str(value).strip() for value in self.basis_payload_sha256s)
        if (
            len(basis_ids) < 2
            or len(set(basis_ids)) != len(basis_ids)
            or len(hashes) != len(basis_ids)
            or any(not value for value in (*basis_ids, *hashes))
        ):
            raise ResidualAttentionValueBasisError(
                "Calibration requires at least two unique basis IDs and hashes."
            )
        object.__setattr__(self, "basis_ids", basis_ids)
        object.__setattr__(self, "basis_payload_sha256s", hashes)
        if not isinstance(self.parent_o_provenance, ParentOutputBlockBasisEntry):
            raise ResidualAttentionValueBasisError(
                "parent_o_provenance must be a ParentOutputBlockBasisEntry."
            )
        self.parent_o_provenance.assert_untampered()
        contributions = _array(
            self.basis_contribution_rows, "basis_contribution_rows", ndim=3
        )
        targets = _array(
            self.target_contribution_rows, "target_contribution_rows", ndim=2
        )
        if (
            int(contributions.shape[0]) != int(targets.shape[0])
            or int(contributions.shape[1]) != len(basis_ids)
            or int(contributions.shape[2]) != int(targets.shape[1])
        ):
            raise ResidualAttentionValueBasisError(
                "Basis contributions must be [rows,basis,hidden] aligned with "
                "[rows,hidden] parent targets."
            )
        object.__setattr__(self, "basis_contribution_rows", contributions)
        object.__setattr__(self, "target_contribution_rows", targets)
        for name in (
            "candidate_trace_sha256",
            "parent_trace_sha256",
            "candidate_registry_sha256",
            "candidate_v_weight_sha256",
            "candidate_o_block_sha256",
            "parent_o_block_sha256",
            "parent_registry_sha256",
            "parent_o_capture_sha256",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ResidualAttentionValueBasisError(f"{name} cannot be blank.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "parent_o_source_flat_index",
            _nonnegative_int(
                self.parent_o_source_flat_index,
                "parent_o_source_flat_index",
            ),
        )
        object.__setattr__(
            self,
            "parent_o_source_active_slot",
            _nonnegative_int(
                self.parent_o_source_active_slot,
                "parent_o_source_active_slot",
            ),
        )
        parent_payload = self.parent_o_provenance.as_dict()
        if (
            self.parent_o_provenance.source_site
            != (self.layer_index, self.canonical_group, self.physical_group)
            or self.parent_registry_sha256
            != self.parent_o_provenance.source_registry_sha256
            or self.parent_o_capture_sha256 != parent_payload["capture_sha256"]
            or self.parent_o_source_flat_index
            != self.parent_o_provenance.source_flat_index
            or self.parent_o_source_active_slot
            != self.parent_o_provenance.source_active_slot
            or self.parent_o_block_sha256
            != _array_sha256(self.parent_o_provenance.block)
        ):
            raise ResidualAttentionValueBasisError(
                "Parent O calibration fields disagree with captured provenance."
            )
        if self.candidate_v_bias_sha256 is not None:
            bias_hash = str(self.candidate_v_bias_sha256).strip()
            if not bias_hash:
                raise ResidualAttentionValueBasisError(
                    "candidate_v_bias_sha256 cannot be blank when present."
                )
            object.__setattr__(self, "candidate_v_bias_sha256", bias_hash)
        object.__setattr__(
            self,
            "value_role_scale",
            _finite_float(self.value_role_scale, "value_role_scale"),
        )
        object.__setattr__(
            self,
            "output_role_scale",
            _finite_float(self.output_role_scale, "output_role_scale"),
        )
        object.__setattr__(
            self,
            "_captured_payload_sha256",
            self._current_payload_sha256(),
        )

    def _current_payload_sha256(self) -> str:
        payload = (
            self.layer_index,
            self.canonical_group,
            self.physical_group,
            self.sequence_id,
            self.basis_ids,
            self.basis_payload_sha256s,
            _array_sha256(self.basis_contribution_rows),
            _array_sha256(self.target_contribution_rows),
            self.candidate_trace_sha256,
            self.parent_trace_sha256,
            self.parent_o_provenance.as_dict()["capture_sha256"],
            self.candidate_registry_sha256,
            self.candidate_v_weight_sha256,
            self.candidate_v_bias_sha256,
            self.candidate_o_block_sha256,
            self.parent_o_block_sha256,
            self.parent_registry_sha256,
            self.parent_o_capture_sha256,
            self.parent_o_source_flat_index,
            self.parent_o_source_active_slot,
            self.value_role_scale,
            self.output_role_scale,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def assert_untampered(self) -> None:
        self.parent_o_provenance.assert_untampered()
        if self._current_payload_sha256() != self._captured_payload_sha256:
            raise ResidualAttentionValueBasisError(
                "Residual value-basis calibration changed after capture."
            )

    @property
    def row_count(self) -> int:
        return int(self.target_contribution_rows.shape[0])

    @property
    def output_width(self) -> int:
        return int(self.target_contribution_rows.shape[1])

    def as_dict(self) -> dict[str, Any]:
        self.assert_untampered()
        payload = {
            "schema": RESIDUAL_VALUE_BASIS_SCHEMA,
            "layer_index": self.layer_index,
            "canonical_group": self.canonical_group,
            "physical_group": self.physical_group,
            "sequence_id": self.sequence_id,
            "row_count": self.row_count,
            "output_width": self.output_width,
            "basis_ids": list(self.basis_ids),
            "basis_payload_sha256s": list(self.basis_payload_sha256s),
            "basis_contribution_sha256": _array_sha256(self.basis_contribution_rows),
            "target_contribution_sha256": _array_sha256(self.target_contribution_rows),
            "candidate_trace_sha256": self.candidate_trace_sha256,
            "parent_trace_sha256": self.parent_trace_sha256,
            "parent_o_provenance": self.parent_o_provenance.as_dict(),
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "candidate_v_weight_sha256": self.candidate_v_weight_sha256,
            "candidate_v_bias_sha256": self.candidate_v_bias_sha256,
            "candidate_o_block_sha256": self.candidate_o_block_sha256,
            "parent_o_block_sha256": self.parent_o_block_sha256,
            "parent_registry_sha256": self.parent_registry_sha256,
            "parent_o_capture_sha256": self.parent_o_capture_sha256,
            "parent_o_source_flat_index": self.parent_o_source_flat_index,
            "parent_o_source_active_slot": self.parent_o_source_active_slot,
            "value_role_scale": self.value_role_scale,
            "output_role_scale": self.output_role_scale,
            "design_contract": (
                "candidate-fixed-QK-A @ scaled-existing-V-basis, then current-O-block"
            ),
            "target_contract": "parent-selected-group-context @ parent-O-block",
        }
        payload["calibration_sha256"] = self._captured_payload_sha256
        return payload


def _validate_trace_pair(
    candidate: HFCachedAttentionTrace,
    parent: HFCachedAttentionTrace,
) -> None:
    if not isinstance(candidate, HFCachedAttentionTrace) or not isinstance(
        parent, HFCachedAttentionTrace
    ):
        raise ResidualAttentionValueBasisError(
            "candidate and parent evidence must be cached attention traces."
        )
    if candidate.lane != "forced" or parent.lane != "forced":
        raise ResidualAttentionValueBasisError(
            "Residual-targeted basis calibration requires the aligned forced lane."
        )
    if (
        candidate.sequence_id != parent.sequence_id
        or candidate.layer_index != parent.layer_index
        or candidate.group_indices != parent.group_indices
        or candidate.prompt_token_ids != parent.prompt_token_ids
        or candidate.consumed_token_ids != parent.consumed_token_ids
        or candidate.selected_token_ids != parent.selected_token_ids
        or candidate.horizon != parent.horizon
        or candidate.key_value_heads != parent.key_value_heads
        or candidate.query_heads_per_group != parent.query_heads_per_group
        or candidate.head_dim != parent.head_dim
    ):
        raise ResidualAttentionValueBasisError(
            "Candidate and parent forced traces must share sequence, layer/group, "
            "driven prefix, horizon, and attention topology."
        )


def build_residual_value_basis_site_calibration(
    candidate_trace: HFCachedAttentionTrace,
    parent_trace: HFCachedAttentionTrace,
    registry: AttentionCodebookRegistry,
    basis_entries: Sequence[ValuePrototypeBasisEntry],
    parent_o_block: ParentOutputBlockBasisEntry,
    *,
    canonical_group: int,
    limits: ResidualValueBasisLimits | None = None,
) -> ResidualValueBasisSiteCalibration:
    """Build one site's residual-space basis under captured candidate Q/K.

    ``parent_o_block`` is a captured incumbent materialized O column block for
    this physical GQA group, shaped ``[hidden, q_rows_per_group]``.  Its bias is
    intentionally excluded because a layer O bias cannot be uniquely assigned
    to one selected group.
    """

    if not isinstance(registry, AttentionCodebookRegistry):
        raise ResidualAttentionValueBasisError(
            "registry must be an AttentionCodebookRegistry."
        )
    controls = limits or ResidualValueBasisLimits()
    if not isinstance(controls, ResidualValueBasisLimits):
        raise ResidualAttentionValueBasisError(
            "limits must be ResidualValueBasisLimits."
        )
    _validate_trace_pair(candidate_trace, parent_trace)
    if not isinstance(parent_o_block, ParentOutputBlockBasisEntry):
        raise ResidualAttentionValueBasisError(
            "parent_o_block must be a captured ParentOutputBlockBasisEntry."
        )
    parent_o_block.assert_untampered()
    entries = tuple(basis_entries)
    if (
        len(entries) < 2
        or len(entries) > controls.max_basis_size
        or any(not isinstance(value, ValuePrototypeBasisEntry) for value in entries)
        or len({value.basis_id for value in entries}) != len(entries)
    ):
        raise ResidualAttentionValueBasisError(
            "basis_entries must contain two to max_basis_size unique entries."
        )
    for entry in entries:
        entry.assert_untampered()
    source_registry_hashes = {value.source_registry_sha256 for value in entries}
    if (
        len(source_registry_hashes) != 1
        or len({value.source_layout for value in entries}) != 1
        or len({value.source_flat_index for value in entries}) != len(entries)
        or parent_o_block.source_registry_sha256 not in source_registry_hashes
    ):
        raise ResidualAttentionValueBasisError(
            "Every distinct basis entry and parent O block must come from one "
            "incumbent registry."
        )
    layer = candidate_trace.layer_index
    if layer >= registry.logical_layers:
        raise ResidualAttentionValueBasisError(
            "Trace layer is outside the supplied codebook registry."
        )
    canonical = _nonnegative_int(canonical_group, "canonical_group")
    if canonical >= registry.key_value_heads:
        raise ResidualAttentionValueBasisError(
            "canonical_group is outside the codebook GQA topology."
        )
    physical = int(registry.group_permutations[layer, canonical].item())
    if candidate_trace.group_indices != (physical,):
        raise ResidualAttentionValueBasisError(
            "The trace must select exactly the physical group named by "
            "group_permutations."
        )
    if parent_o_block.source_site != (layer, canonical, physical):
        raise ResidualAttentionValueBasisError(
            "Parent O provenance does not match the calibrated logical/physical site."
        )
    if (
        candidate_trace.query_heads_per_group != registry.query_heads_per_group
        or candidate_trace.head_dim != registry.head_dim
        or candidate_trace.key_value_heads != registry.key_value_heads
    ):
        raise ResidualAttentionValueBasisError(
            "Trace attention topology disagrees with the candidate registry."
        )
    input_width = int(entries[0].weight.shape[1])
    head_dim = int(entries[0].weight.shape[0])
    has_bias = entries[0].bias is not None
    if any(
        tuple(value.weight.shape) != (head_dim, input_width)
        or (value.bias is not None) != has_bias
        for value in entries
    ):
        raise ResidualAttentionValueBasisError(
            "Every basis entry needs one common V shape and bias contract."
        )
    if (
        input_width > controls.max_input_dimensions
        or head_dim > controls.max_head_dimensions
    ):
        raise ResidualAttentionValueBasisError(
            "Basis dimensions exceed the declared limits."
        )
    live_v_weight, live_v_bias = registry.materialize_projection(layer, "v")
    if candidate_trace.value_projection_weight_sha256 != _tensor_sha256(
        live_v_weight
    ) or candidate_trace.value_projection_bias_sha256 != _optional_tensor_sha256(
        live_v_bias
    ):
        raise ResidualAttentionValueBasisError(
            "Candidate trace V hashes do not match the live registry."
        )
    live_o_weight, _live_o_bias = registry.materialize_projection(layer, "o")
    q_rows = registry.query_rows_per_group
    start = physical * q_rows
    candidate_o = _array(
        live_o_weight[:, start : start + q_rows],
        "candidate materialized O block",
        ndim=2,
    )
    parent_o = _array(parent_o_block.block, "parent_o_block", ndim=2)
    if candidate_o.shape != parent_o.shape or tuple(candidate_o.shape) != (
        registry.output_width,
        q_rows,
    ):
        raise ResidualAttentionValueBasisError(
            "Candidate and parent O blocks must match [hidden,q_rows_per_group]."
        )
    if registry.output_width > controls.max_output_dimensions:
        raise ResidualAttentionValueBasisError(
            "Residual output width exceeds max_output_dimensions."
        )
    value_scale = (
        1.0
        if registry.role_scales is None
        else float(
            registry.role_scales[layer, canonical, _V_ROLE_INDEX].detach().cpu().item()
        )
    )
    output_scale = (
        1.0
        if registry.role_scales is None
        else float(
            registry.role_scales[layer, canonical, _O_ROLE_INDEX].detach().cpu().item()
        )
    )
    if (
        not math.isfinite(value_scale)
        or not math.isfinite(output_scale)
        or abs(value_scale) < controls.minimum_absolute_role_scale
        or abs(output_scale) < controls.minimum_absolute_role_scale
    ):
        raise ResidualAttentionValueBasisError(
            "V/O role scales must be finite and bounded away from zero."
        )
    history_chunks: list[np.ndarray] = []
    contribution_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for candidate_step, parent_step in zip(
        candidate_trace.steps, parent_trace.steps, strict=True
    ):
        if candidate_step.prefix_length != parent_step.prefix_length:
            raise ResidualAttentionValueBasisError(
                "Aligned trace steps have different prefix lengths."
            )
        chunk = _array(
            candidate_step.value_projection_input,
            "candidate value_projection_input",
        )
        if (
            chunk.ndim != 3
            or int(chunk.shape[0]) != 1
            or int(chunk.shape[2]) != input_width
        ):
            raise ResidualAttentionValueBasisError(
                "Each V input chunk must be [1,rows,input_width]."
            )
        history_chunks.append(np.array(chunk[0], copy=True))
        history = np.concatenate(history_chunks, axis=0)
        if int(history.shape[0]) != candidate_step.prefix_length:
            raise ResidualAttentionValueBasisError(
                "Accumulated V input history does not equal the cache prefix."
            )
        probabilities = _array(
            candidate_step.attention_probabilities,
            "candidate fixed attention probabilities",
        )
        expected_probability_shape = (
            1,
            registry.query_heads_per_group,
            1,
            candidate_step.prefix_length,
        )
        if tuple(probabilities.shape) != expected_probability_shape:
            raise ResidualAttentionValueBasisError(
                "Candidate attention probabilities have the wrong selected-group shape."
            )
        attention = probabilities[0, :, 0, :]
        basis_residuals: list[np.ndarray] = []
        for entry in entries:
            values = history @ entry.weight.T
            if entry.bias is not None:
                values = values + entry.bias
            values = values * value_scale
            context = attention @ values
            context_flat = context.reshape(1, q_rows)
            basis_residuals.append((context_flat @ candidate_o.T)[0])
        contribution_rows.append(np.stack(basis_residuals, axis=0))
        parent_context = _array(
            parent_step.selected_attention_context,
            "parent selected attention context",
            ndim=2,
        )
        if tuple(parent_context.shape) != (1, q_rows):
            raise ResidualAttentionValueBasisError(
                "Parent selected context must be [1,q_rows_per_group]."
            )
        target_rows.append((parent_context @ parent_o.T)[0])
    contributions = np.stack(contribution_rows, axis=0)
    targets = np.stack(target_rows, axis=0)
    if int(contributions.shape[0]) > controls.max_rows:
        raise ResidualAttentionValueBasisError("Site calibration exceeds max_rows.")
    return ResidualValueBasisSiteCalibration(
        layer_index=layer,
        canonical_group=canonical,
        physical_group=physical,
        sequence_id=candidate_trace.sequence_id,
        basis_ids=tuple(value.basis_id for value in entries),
        basis_payload_sha256s=tuple(value.payload_sha256 for value in entries),
        basis_contribution_rows=contributions,
        target_contribution_rows=targets,
        candidate_trace_sha256=_trace_sha256(candidate_trace),
        parent_trace_sha256=_trace_sha256(parent_trace),
        parent_o_provenance=parent_o_block,
        candidate_registry_sha256=_registry_sha256(registry),
        candidate_v_weight_sha256=_tensor_sha256(live_v_weight),
        candidate_v_bias_sha256=_optional_tensor_sha256(live_v_bias),
        candidate_o_block_sha256=_array_sha256(candidate_o),
        parent_o_block_sha256=_array_sha256(parent_o),
        parent_registry_sha256=parent_o_block.source_registry_sha256,
        parent_o_capture_sha256=parent_o_block.as_dict()["capture_sha256"],
        parent_o_source_flat_index=parent_o_block.source_flat_index,
        parent_o_source_active_slot=parent_o_block.source_active_slot,
        value_role_scale=value_scale,
        output_role_scale=output_scale,
    )


@dataclass(frozen=True, slots=True)
class ResidualValueBasisSiteReport:
    layer_index: int
    canonical_group: int
    physical_group: int
    calibration_count: int
    row_count: int
    value_role_scale: float
    output_role_scale: float
    candidate_o_block_sha256s: tuple[str, ...]
    parent_o_block_sha256s: tuple[str, ...]
    parent_o_capture_sha256s: tuple[str, ...]
    parent_registry_sha256s: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in (
                (field, getattr(self, field)) for field in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class ResidualTargetedValueBasisReport:
    layout: str
    survivor_group: int | None
    survivor_slot: int
    survivor_active_slot: int
    survivor_basis_id: str
    basis_entries: tuple[ValuePrototypeBasisEntry, ...]
    basis_source_registry_sha256: str
    basis_payload_sha256s: tuple[str, ...]
    coefficients: tuple[float, ...]
    prior_coefficients: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    active_set: tuple[str, ...]
    affected_sites: tuple[ResidualValueBasisSiteReport, ...]
    calibration_count: int
    row_count: int
    scalar_observation_count: int
    ridge: float
    effective_ridge: float
    condition_number: float
    before_relative_rmse: float
    after_relative_rmse: float
    relative_anchor_drift: float
    candidate_registry_sha256: str
    assignment_sha256: str
    group_permutations_sha256: str
    role_scales_sha256: str | None
    design_sha256: str
    target_sha256: str
    fitted_weight_sha256: str
    fitted_bias_sha256: str | None
    calibration_sha256s: tuple[str, ...]
    enumerated_active_sets: int
    optimizer_steps: int = 0
    backward_calls: int = 0

    @property
    def zero_gradient(self) -> bool:
        return self.optimizer_steps == self.backward_calls == 0

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": RESIDUAL_VALUE_BASIS_SCHEMA,
            "layout": self.layout,
            "survivor_group": self.survivor_group,
            "survivor_slot": self.survivor_slot,
            "survivor_active_slot": self.survivor_active_slot,
            "survivor_basis_id": self.survivor_basis_id,
            "basis_entries": [value.as_dict() for value in self.basis_entries],
            "basis_source_registry_sha256": self.basis_source_registry_sha256,
            "basis_payload_sha256s": list(self.basis_payload_sha256s),
            "coefficients": list(self.coefficients),
            "prior_coefficients": list(self.prior_coefficients),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "active_set": list(self.active_set),
            "affected_sites": [value.as_dict() for value in self.affected_sites],
            "calibration_count": self.calibration_count,
            "row_count": self.row_count,
            "scalar_observation_count": self.scalar_observation_count,
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "condition_number": self.condition_number,
            "before_relative_rmse": self.before_relative_rmse,
            "after_relative_rmse": self.after_relative_rmse,
            "relative_anchor_drift": self.relative_anchor_drift,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "assignment_sha256": self.assignment_sha256,
            "group_permutations_sha256": self.group_permutations_sha256,
            "role_scales_sha256": self.role_scales_sha256,
            "design_sha256": self.design_sha256,
            "target_sha256": self.target_sha256,
            "fitted_weight_sha256": self.fitted_weight_sha256,
            "fitted_bias_sha256": self.fitted_bias_sha256,
            "calibration_sha256s": list(self.calibration_sha256s),
            "enumerated_active_sets": self.enumerated_active_sets,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "zero_gradient": self.zero_gradient,
            "coefficient_solver": "finite-box-active-set-anchored-ridge",
            "design_contract": (
                "candidate-fixed-QK-A @ scaled-existing-V-basis, then current-O-block"
            ),
            "target_contract": "parent-selected-group-residual-contribution",
            "global_all_survivor_sites": True,
            "group_permutations_honored": True,
            "v_and_o_role_scales_honored": True,
            "runtime_coefficients_retained": False,
            "new_runtime_parameters": 0,
            "materialization": "sum(coeff_i * existing_raw_V_prototype_i)",
            "required_next_step": (
                "recollect actual child A@V and run existing block-sparse O fit"
            ),
            "experimental_not_compaction_proof": True,
        }
        payload["report_sha256"] = hashlib.sha256(
            repr(payload).encode("utf-8")
        ).hexdigest()
        return payload


@dataclass(frozen=True, slots=True)
class ResidualTargetedValueBasisFit:
    weight: Any
    bias: Any | None
    report: ResidualTargetedValueBasisReport
    installable_fit: SharedValuePrototypeFit

    def __post_init__(self) -> None:
        if not isinstance(self.report, ResidualTargetedValueBasisReport):
            raise ResidualAttentionValueBasisError(
                "report must be a ResidualTargetedValueBasisReport."
            )
        weight = _array(self.weight, "fitted basis V weight", ndim=2)
        bias = (
            None
            if self.bias is None
            else _array(self.bias, "fitted basis V bias", ndim=1)
        )
        if (
            _array_sha256(weight) != self.report.fitted_weight_sha256
            or (None if bias is None else _array_sha256(bias))
            != self.report.fitted_bias_sha256
        ):
            raise ResidualAttentionValueBasisError(
                "Fitted basis payload hashes disagree with its report."
            )
        if not isinstance(self.installable_fit, SharedValuePrototypeFit):
            raise ResidualAttentionValueBasisError(
                "installable_fit must use the rollback-safe shared V installer."
            )
        if _array_sha256(self.installable_fit.weight) != _array_sha256(weight) or (
            None
            if self.installable_fit.bias is None
            else _array_sha256(self.installable_fit.bias)
        ) != (None if bias is None else _array_sha256(bias)):
            raise ResidualAttentionValueBasisError(
                "Installable shared V payload differs from the basis fit."
            )
        for entry in self.report.basis_entries:
            entry.assert_untampered()
        if tuple(
            value.payload_sha256 for value in self.report.basis_entries
        ) != self.report.basis_payload_sha256s or {
            value.source_registry_sha256 for value in self.report.basis_entries
        } != {self.report.basis_source_registry_sha256}:
            raise ResidualAttentionValueBasisError(
                "Basis report provenance changed after coefficient fitting."
            )
        basis_size = len(self.report.basis_entries)
        if any(
            len(value) != basis_size
            for value in (
                self.report.coefficients,
                self.report.prior_coefficients,
                self.report.lower_bounds,
                self.report.upper_bounds,
                self.report.active_set,
                self.report.basis_payload_sha256s,
            )
        ) or any(
            label not in {"lower", "free", "upper"} for label in self.report.active_set
        ):
            raise ResidualAttentionValueBasisError(
                "Basis coefficient, bound, active-set, and payload metadata widths "
                "must agree."
            )
        coefficients = np.array(self.report.coefficients, dtype=np.float64)
        prior = np.array(self.report.prior_coefficients, dtype=np.float64)
        lower = np.array(self.report.lower_bounds, dtype=np.float64)
        upper = np.array(self.report.upper_bounds, dtype=np.float64)
        if (
            not bool(np.isfinite(coefficients).all())
            or not bool(np.isfinite(prior).all())
            or not bool(np.isfinite(lower).all())
            or not bool(np.isfinite(upper).all())
            or bool(np.any(lower > upper))
            or bool(np.any(coefficients < lower))
            or bool(np.any(coefficients > upper))
            or bool(np.any(prior < lower))
            or bool(np.any(prior > upper))
        ):
            raise ResidualAttentionValueBasisError(
                "Basis coefficients/prior must be finite and satisfy ordered bounds."
            )
        reconstructed_weight = np.zeros_like(weight)
        reconstructed_bias = None if bias is None else np.zeros_like(bias)
        for coefficient, entry in zip(
            coefficients,
            self.report.basis_entries,
            strict=True,
        ):
            reconstructed_weight += coefficient * entry.weight
            if reconstructed_bias is not None and entry.bias is not None:
                reconstructed_bias += coefficient * entry.bias
        if _array_sha256(reconstructed_weight) != _array_sha256(weight) or (
            None if reconstructed_bias is None else _array_sha256(reconstructed_bias)
        ) != (None if bias is None else _array_sha256(bias)):
            raise ResidualAttentionValueBasisError(
                "Reported basis coefficients do not materialize the fitted V payload."
            )
        install_report = self.installable_fit.report
        basis_sites = tuple(
            (
                value.layer_index,
                value.canonical_group,
                value.physical_group,
                value.row_count,
                value.calibration_count,
                value.value_role_scale,
            )
            for value in self.report.affected_sites
        )
        install_sites = tuple(
            (
                value.layer_index,
                value.canonical_group,
                value.physical_group,
                value.row_count,
                value.calibration_count,
                value.role_scale,
            )
            for value in install_report.affected_sites
        )
        survivor_entries = tuple(
            value
            for value in self.report.basis_entries
            if value.basis_id == self.report.survivor_basis_id
        )
        if len(survivor_entries) != 1:
            raise ResidualAttentionValueBasisError(
                "Basis report does not contain exactly one named survivor."
            )
        survivor_entry = survivor_entries[0]
        if (
            install_report.layout != self.report.layout
            or install_report.survivor_group != self.report.survivor_group
            or install_report.survivor_slot != self.report.survivor_slot
            or basis_sites != install_sites
            or install_report.calibration_count != self.report.calibration_count
            or install_report.row_count != self.report.row_count
            or install_report.assignment_sha256 != self.report.assignment_sha256
            or install_report.group_permutations_sha256
            != self.report.group_permutations_sha256
            or install_report.role_scales_sha256 != self.report.role_scales_sha256
            or install_report.anchor_weight_sha256
            != survivor_entry.source_weight_sha256
            or install_report.anchor_bias_sha256 != survivor_entry.source_bias_sha256
            or install_report.fitted_weight_sha256 != self.report.fitted_weight_sha256
            or install_report.fitted_bias_sha256 != self.report.fitted_bias_sha256
            or install_report.before_relative_rmse != self.report.before_relative_rmse
            or install_report.after_relative_rmse != self.report.after_relative_rmse
            or install_report.relative_anchor_drift != self.report.relative_anchor_drift
            or install_report.ridge != self.report.ridge
            or install_report.condition_number != self.report.condition_number
            or install_report.design_sha256 != self.report.design_sha256
            or install_report.target_sha256 != self.report.target_sha256
            or install_report.calibration_sha256s != self.report.calibration_sha256s
        ):
            raise ResidualAttentionValueBasisError(
                "Installable shared V report is not exactly bound to the basis fit."
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)


def _coefficient_vector(
    basis_ids: tuple[str, ...],
    values: Mapping[str, float] | Sequence[float] | None,
    *,
    default: Sequence[float],
    name: str,
) -> np.ndarray:
    if values is None:
        payload = tuple(default)
    elif isinstance(values, Mapping):
        unknown = sorted(set(values) - set(basis_ids))
        if unknown:
            raise ResidualAttentionValueBasisError(
                f"{name} contains unknown basis IDs: {unknown}."
            )
        payload = tuple(
            values.get(basis_id, default[index])
            for index, basis_id in enumerate(basis_ids)
        )
    else:
        payload = tuple(values)
    if len(payload) != len(basis_ids):
        raise ResidualAttentionValueBasisError(
            f"{name} must provide one value per basis entry."
        )
    result = np.array(
        [
            _finite_float(value, f"{name}[{index}]")
            for index, value in enumerate(payload)
        ],
        dtype=np.float64,
    )
    return result


def _solve_box_anchored_ridge(
    design: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    ridge: float,
    max_active_sets: int,
) -> tuple[np.ndarray, float, float, tuple[str, ...], int]:
    basis_size = int(design.shape[1])
    active_set_count = 3**basis_size
    if active_set_count > max_active_sets:
        raise ResidualAttentionValueBasisError(
            "Basis box solve exceeds max_active_sets."
        )
    gram = design.T @ design
    trace_scale = float(np.trace(gram)) / max(basis_size, 1)
    effective_ridge = ridge * max(trace_scale, np.finfo(np.float64).tiny)
    system = gram + effective_ridge * np.eye(basis_size, dtype=np.float64)
    right = design.T @ target + effective_ridge * prior
    condition_number = float(np.linalg.cond(system))
    if not math.isfinite(condition_number):
        raise ResidualAttentionValueBasisError(
            "Anchored basis ridge system has a non-finite condition number."
        )
    best: np.ndarray | None = None
    best_objective = math.inf
    best_state: tuple[int, ...] | None = None
    tolerance = 1e-10
    enumerated = 0
    for state in itertools.product((-1, 0, 1), repeat=basis_size):
        enumerated += 1
        fixed = [index for index, value in enumerate(state) if value]
        free = [index for index, value in enumerate(state) if not value]
        coefficients = np.empty(basis_size, dtype=np.float64)
        for index in fixed:
            coefficients[index] = lower[index] if state[index] < 0 else upper[index]
        if free:
            free_system = system[np.ix_(free, free)]
            free_right = right[free].copy()
            if fixed:
                free_right -= system[np.ix_(free, fixed)] @ coefficients[fixed]
            try:
                coefficients[free] = np.linalg.solve(free_system, free_right)
            except np.linalg.LinAlgError:
                continue
        if bool(np.any(coefficients < lower - tolerance)) or bool(
            np.any(coefficients > upper + tolerance)
        ):
            continue
        coefficients = np.minimum(np.maximum(coefficients, lower), upper)
        residual = design @ coefficients - target
        anchor = coefficients - prior
        objective = float(residual @ residual + effective_ridge * (anchor @ anchor))
        if objective < best_objective:
            best = coefficients.copy()
            best_objective = objective
            best_state = state
    if best is None or best_state is None or not bool(np.isfinite(best).all()):
        raise ResidualAttentionValueBasisError(
            "No finite feasible coefficient active set was found."
        )
    labels = tuple(
        "lower" if value < 0 else "upper" if value > 0 else "free"
        for value in best_state
    )
    return best, effective_ridge, condition_number, labels, enumerated


def fit_residual_targeted_value_basis(
    registry: AttentionCodebookRegistry,
    calibrations: Sequence[ResidualValueBasisSiteCalibration],
    basis_entries: Sequence[ValuePrototypeBasisEntry],
    *,
    survivor_basis_id: str,
    survivor_slot: int,
    survivor_group: int | None = None,
    prior_coefficients: Mapping[str, float] | Sequence[float] | None = None,
    lower_bounds: Mapping[str, float] | Sequence[float] | None = None,
    upper_bounds: Mapping[str, float] | Sequence[float] | None = None,
    ridge: float = 1e-4,
    limits: ResidualValueBasisLimits | None = None,
) -> ResidualTargetedValueBasisFit:
    """Fit and materialize a bounded V basis in residual-contribution space."""

    if not isinstance(registry, AttentionCodebookRegistry):
        raise ResidualAttentionValueBasisError(
            "registry must be an AttentionCodebookRegistry."
        )
    controls = limits or ResidualValueBasisLimits()
    if not isinstance(controls, ResidualValueBasisLimits):
        raise ResidualAttentionValueBasisError(
            "limits must be ResidualValueBasisLimits."
        )
    entries = tuple(basis_entries)
    if (
        len(entries) < 2
        or len(entries) > controls.max_basis_size
        or any(not isinstance(value, ValuePrototypeBasisEntry) for value in entries)
        or len({value.basis_id for value in entries}) != len(entries)
    ):
        raise ResidualAttentionValueBasisError(
            "basis_entries must contain two to max_basis_size unique entries."
        )
    for entry in entries:
        entry.assert_untampered()
    source_registry_hashes = {value.source_registry_sha256 for value in entries}
    if (
        len(source_registry_hashes) != 1
        or len({value.source_layout for value in entries}) != 1
        or len({value.source_flat_index for value in entries}) != len(entries)
    ):
        raise ResidualAttentionValueBasisError(
            "Every distinct basis entry must come from one incumbent registry."
        )
    basis_source_registry_sha256 = next(iter(source_registry_hashes))
    survivor_id = str(survivor_basis_id).strip()
    survivor_matches = [
        index for index, value in enumerate(entries) if value.basis_id == survivor_id
    ]
    if len(survivor_matches) != 1 or entries[survivor_matches[0]].role != "survivor":
        raise ResidualAttentionValueBasisError(
            "survivor_basis_id must identify exactly one survivor-role entry."
        )
    if sum(value.role == "survivor" for value in entries) != 1 or not any(
        value.role == "removed-source" for value in entries
    ):
        raise ResidualAttentionValueBasisError(
            "The explicit basis requires exactly one survivor and at least one "
            "removed-source prototype."
        )
    basis_ids = tuple(value.basis_id for value in entries)
    basis_hashes = tuple(value.payload_sha256 for value in entries)
    head_dim, input_width = map(int, entries[0].weight.shape)
    has_bias = entries[0].bias is not None
    if any(
        tuple(value.weight.shape) != (head_dim, input_width)
        or (value.bias is not None) != has_bias
        for value in entries
    ):
        raise ResidualAttentionValueBasisError(
            "Every basis entry needs one common shape and bias contract."
        )
    target_weight, target_bias, active_slot = _target_prototype(
        registry,
        survivor_group=survivor_group,
        survivor_slot=survivor_slot,
    )
    survivor_entry = entries[survivor_matches[0]]
    if (
        _tensor_sha256(target_weight) != survivor_entry.source_weight_sha256
        or _optional_tensor_sha256(target_bias) != survivor_entry.source_bias_sha256
        or _array_sha256(_array(target_weight, "live survivor V weight", ndim=2))
        != _array_sha256(survivor_entry.weight)
        or (
            None
            if target_bias is None
            else _array_sha256(_array(target_bias, "live survivor V bias", ndim=1))
        )
        != (None if survivor_entry.bias is None else _array_sha256(survivor_entry.bias))
    ):
        raise ResidualAttentionValueBasisError(
            "The survivor basis entry is not the live raw survivor V prototype."
        )
    evidence = tuple(calibrations)
    if (
        not evidence
        or len(evidence) > controls.max_calibrations
        or any(
            not isinstance(value, ResidualValueBasisSiteCalibration)
            for value in evidence
        )
    ):
        raise ResidualAttentionValueBasisError(
            "calibrations must be a bounded non-empty site-calibration sequence."
        )
    expected_sites = _affected_site_keys(
        registry,
        survivor_group=survivor_group,
        survivor_slot=survivor_slot,
    )
    supplied_sites = {(value.layer_index, value.canonical_group) for value in evidence}
    if (
        not expected_sites
        or len(expected_sites) > controls.max_sites
        or supplied_sites != set(expected_sites)
    ):
        missing = sorted(set(expected_sites) - supplied_sites)
        extra = sorted(supplied_sites - set(expected_sites))
        raise ResidualAttentionValueBasisError(
            "Global residual-basis fitting requires every and only survivor site; "
            f"missing={missing}, extra={extra}."
        )
    registry_hash = _registry_sha256(registry)
    for calibration in evidence:
        calibration.assert_untampered()
        calibration.parent_o_provenance.assert_untampered()
        if (
            calibration.parent_o_capture_sha256
            != calibration.parent_o_provenance.as_dict()["capture_sha256"]
        ):
            raise ResidualAttentionValueBasisError(
                "Parent O provenance changed after site calibration."
            )
    if any(
        value.basis_ids != basis_ids
        or value.basis_payload_sha256s != basis_hashes
        or value.candidate_registry_sha256 != registry_hash
        or value.parent_registry_sha256 != basis_source_registry_sha256
        for value in evidence
    ):
        raise ResidualAttentionValueBasisError(
            "A calibration basis order/hash or candidate registry hash is stale."
        )
    for calibration in evidence:
        physical = int(
            registry.group_permutations[
                calibration.layer_index, calibration.canonical_group
            ].item()
        )
        if calibration.physical_group != physical:
            raise ResidualAttentionValueBasisError(
                "Calibration physical group disagrees with group_permutations."
            )
        live_v_weight, live_v_bias = registry.materialize_projection(
            calibration.layer_index,
            "v",
        )
        live_o_weight, _live_o_bias = registry.materialize_projection(
            calibration.layer_index,
            "o",
        )
        q_rows = registry.query_rows_per_group
        live_o_block = _array(
            live_o_weight[:, physical * q_rows : (physical + 1) * q_rows],
            "live candidate O block",
            ndim=2,
        )
        expected_value_scale = (
            1.0
            if registry.role_scales is None
            else float(
                registry.role_scales[
                    calibration.layer_index,
                    calibration.canonical_group,
                    _V_ROLE_INDEX,
                ]
                .detach()
                .cpu()
                .item()
            )
        )
        expected_output_scale = (
            1.0
            if registry.role_scales is None
            else float(
                registry.role_scales[
                    calibration.layer_index,
                    calibration.canonical_group,
                    _O_ROLE_INDEX,
                ]
                .detach()
                .cpu()
                .item()
            )
        )
        if (
            calibration.candidate_v_weight_sha256 != _tensor_sha256(live_v_weight)
            or calibration.candidate_v_bias_sha256
            != _optional_tensor_sha256(live_v_bias)
            or calibration.candidate_o_block_sha256 != _array_sha256(live_o_block)
            or calibration.value_role_scale != expected_value_scale
            or calibration.output_role_scale != expected_output_scale
        ):
            raise ResidualAttentionValueBasisError(
                "Calibration V/O hashes or role scales are stale or forged."
            )
    contributions = np.concatenate(
        [value.basis_contribution_rows for value in evidence], axis=0
    )
    targets = np.concatenate(
        [value.target_contribution_rows for value in evidence], axis=0
    )
    row_count = int(contributions.shape[0])
    output_width = int(contributions.shape[2])
    scalar_observations = row_count * output_width
    if (
        row_count > controls.max_rows
        or scalar_observations > controls.max_scalar_observations
    ):
        raise ResidualAttentionValueBasisError(
            "Global residual-basis calibration exceeds declared row bounds."
        )
    design = contributions.transpose(0, 2, 1).reshape(-1, len(entries))
    target = targets.reshape(-1)
    prior_default = [1.0 if basis_id == survivor_id else 0.0 for basis_id in basis_ids]
    prior = _coefficient_vector(
        basis_ids,
        prior_coefficients,
        default=prior_default,
        name="prior_coefficients",
    )
    lower = _coefficient_vector(
        basis_ids,
        lower_bounds,
        default=[-2.0] * len(entries),
        name="lower_bounds",
    )
    upper = _coefficient_vector(
        basis_ids,
        upper_bounds,
        default=[2.0] * len(entries),
        name="upper_bounds",
    )
    if (
        bool(np.any(lower > upper))
        or bool(np.any(prior < lower))
        or bool(np.any(prior > upper))
    ):
        raise ResidualAttentionValueBasisError(
            "Coefficient bounds must be ordered and contain the anchor prior."
        )
    ridge_value = _positive_float(ridge, "ridge")
    coefficients, effective_ridge, condition, active_set, enumerated = (
        _solve_box_anchored_ridge(
            design,
            target,
            prior,
            lower,
            upper,
            ridge=ridge_value,
            max_active_sets=controls.max_active_sets,
        )
    )
    if condition > controls.max_condition_number:
        raise ResidualAttentionValueBasisError(
            "Anchored basis ridge condition number exceeds max_condition_number."
        )
    fitted_weight = np.zeros((head_dim, input_width), dtype=np.float64)
    fitted_bias = np.zeros(head_dim, dtype=np.float64) if has_bias else None
    for coefficient, entry in zip(coefficients, entries, strict=True):
        fitted_weight += coefficient * entry.weight
        if fitted_bias is not None and entry.bias is not None:
            fitted_bias += coefficient * entry.bias
    fitted_weight = _array(fitted_weight, "materialized basis V weight", ndim=2)
    if fitted_bias is not None:
        fitted_bias = _array(fitted_bias, "materialized basis V bias", ndim=1)
    anchor_weight = _array(target_weight, "survivor anchor weight", ndim=2)
    anchor_bias = (
        None
        if target_bias is None
        else _array(target_bias, "survivor anchor bias", ndim=1)
    )
    anchor_vector = anchor_weight.reshape(-1)
    fitted_vector = fitted_weight.reshape(-1)
    if anchor_bias is not None and fitted_bias is not None:
        anchor_vector = np.concatenate((anchor_vector, anchor_bias))
        fitted_vector = np.concatenate((fitted_vector, fitted_bias))
    relative_anchor_drift = float(
        np.linalg.norm(fitted_vector - anchor_vector)
        / max(np.linalg.norm(anchor_vector), np.finfo(np.float64).eps)
    )
    site_reports: list[ResidualValueBasisSiteReport] = []
    shared_site_reports: list[SharedValueSiteReport] = []
    for layer, canonical in expected_sites:
        site_evidence = tuple(
            value
            for value in evidence
            if (value.layer_index, value.canonical_group) == (layer, canonical)
        )
        rows = sum(value.row_count for value in site_evidence)
        physical = int(registry.group_permutations[layer, canonical].item())
        value_scale = site_evidence[0].value_role_scale
        output_scale = site_evidence[0].output_role_scale
        if any(
            value.value_role_scale != value_scale
            or value.output_role_scale != output_scale
            for value in site_evidence
        ):
            raise ResidualAttentionValueBasisError(
                "One logical site has inconsistent role-scale provenance."
            )
        site_reports.append(
            ResidualValueBasisSiteReport(
                layer_index=layer,
                canonical_group=canonical,
                physical_group=physical,
                calibration_count=len(site_evidence),
                row_count=rows,
                value_role_scale=value_scale,
                output_role_scale=output_scale,
                candidate_o_block_sha256s=tuple(
                    sorted({value.candidate_o_block_sha256 for value in site_evidence})
                ),
                parent_o_block_sha256s=tuple(
                    sorted({value.parent_o_block_sha256 for value in site_evidence})
                ),
                parent_o_capture_sha256s=tuple(
                    sorted({value.parent_o_capture_sha256 for value in site_evidence})
                ),
                parent_registry_sha256s=tuple(
                    sorted({value.parent_registry_sha256 for value in site_evidence})
                ),
            )
        )
        shared_site_reports.append(
            SharedValueSiteReport(
                layer_index=layer,
                canonical_group=canonical,
                physical_group=physical,
                row_count=rows,
                role_scale=value_scale,
                calibration_count=len(site_evidence),
            )
        )
    calibration_hashes = tuple(
        value.as_dict()["calibration_sha256"] for value in evidence
    )
    report = ResidualTargetedValueBasisReport(
        layout=_registry_layout(registry),
        survivor_group=survivor_group,
        survivor_slot=survivor_slot,
        survivor_active_slot=active_slot,
        survivor_basis_id=survivor_id,
        basis_entries=entries,
        basis_source_registry_sha256=basis_source_registry_sha256,
        basis_payload_sha256s=basis_hashes,
        coefficients=tuple(map(float, coefficients)),
        prior_coefficients=tuple(map(float, prior)),
        lower_bounds=tuple(map(float, lower)),
        upper_bounds=tuple(map(float, upper)),
        active_set=active_set,
        affected_sites=tuple(site_reports),
        calibration_count=len(evidence),
        row_count=row_count,
        scalar_observation_count=scalar_observations,
        ridge=ridge_value,
        effective_ridge=effective_ridge,
        condition_number=condition,
        before_relative_rmse=_relative_rmse(target, design @ prior),
        after_relative_rmse=_relative_rmse(target, design @ coefficients),
        relative_anchor_drift=relative_anchor_drift,
        candidate_registry_sha256=registry_hash,
        assignment_sha256=_tensor_sha256(registry.assignments),
        group_permutations_sha256=_tensor_sha256(registry.group_permutations),
        role_scales_sha256=_optional_tensor_sha256(registry.role_scales),
        design_sha256=_array_sha256(design),
        target_sha256=_array_sha256(target),
        fitted_weight_sha256=_array_sha256(fitted_weight),
        fitted_bias_sha256=(
            None if fitted_bias is None else _array_sha256(fitted_bias)
        ),
        calibration_sha256s=calibration_hashes,
        enumerated_active_sets=enumerated,
    )
    calibration_v_hashes = tuple(
        (
            layer,
            _tensor_sha256(weight),
            _optional_tensor_sha256(bias),
        )
        for layer in sorted({layer for layer, _canonical in expected_sites})
        for weight, bias in (registry.materialize_projection(layer, "v"),)
    )
    shared_report = SharedValuePrototypeReport(
        layout=report.layout,
        survivor_group=survivor_group,
        survivor_slot=survivor_slot,
        affected_sites=tuple(shared_site_reports),
        calibration_count=len(evidence),
        row_count=row_count,
        input_width=input_width,
        head_dim=head_dim,
        has_bias=has_bias,
        assignment_sha256=report.assignment_sha256,
        group_permutations_sha256=report.group_permutations_sha256,
        role_scales_sha256=report.role_scales_sha256,
        calibration_v_sha256_by_layer=calibration_v_hashes,
        anchor_weight_sha256=_tensor_sha256(target_weight),
        anchor_bias_sha256=_optional_tensor_sha256(target_bias),
        fitted_weight_sha256=report.fitted_weight_sha256,
        fitted_bias_sha256=report.fitted_bias_sha256,
        before_relative_rmse=report.before_relative_rmse,
        after_relative_rmse=report.after_relative_rmse,
        relative_anchor_drift=report.relative_anchor_drift,
        ridge=report.ridge,
        condition_number=report.condition_number,
        design_sha256=report.design_sha256,
        target_sha256=report.target_sha256,
        calibration_sha256s=report.calibration_sha256s,
    )
    installable = SharedValuePrototypeFit(fitted_weight, fitted_bias, shared_report)
    return ResidualTargetedValueBasisFit(
        fitted_weight,
        fitted_bias,
        report,
        installable,
    )


def install_residual_targeted_value_basis(
    registry: AttentionCodebookRegistry,
    fit: ResidualTargetedValueBasisFit,
    *,
    limits: ResidualAttentionValueFitLimits | None = None,
) -> InstalledValuePrototype:
    """Install through the existing exact rollback handle after provenance checks."""

    if not isinstance(registry, AttentionCodebookRegistry) or not isinstance(
        fit, ResidualTargetedValueBasisFit
    ):
        raise ResidualAttentionValueBasisError(
            "registry and fit must be residual-targeted value-basis objects."
        )
    validated_fit = ResidualTargetedValueBasisFit(
        fit.weight,
        fit.bias,
        fit.report,
        fit.installable_fit,
    )
    if _registry_sha256(registry) != validated_fit.report.candidate_registry_sha256:
        raise ResidualAttentionValueBasisError(
            "Candidate registry changed after residual-basis calibration."
        )
    if (
        _array_sha256(validated_fit.weight) != validated_fit.report.fitted_weight_sha256
        or (None if validated_fit.bias is None else _array_sha256(validated_fit.bias))
        != validated_fit.report.fitted_bias_sha256
    ):
        raise ResidualAttentionValueBasisError(
            "Residual-basis payload changed after its report was created."
        )
    return install_shared_codebook_value_prototype(
        registry,
        validated_fit.installable_fit,
        limits=limits,
    )


RESIDUAL_VALUE_BASIS_INTEGRATION_CONTRACT: Mapping[str, Any] = MappingProxyType(
    {
        "capture_order": (
            "capture survivor+removed raw V basis and parent O blocks from incumbent; "
            "then create/install tentative redirect"
        ),
        "calibration_lane": "forced-parent-token",
        "fit_target": "parent-selected-group-residual-contribution",
        "parent_o_provenance": "captured-incumbent-registry-materialized-block",
        "fit_design": "candidate-fixed-QK-attention-through-current-O",
        "global_coverage": "every logical site sharing survivor",
        "post_install_order": (
            "recollect actual child traces; then existing block-sparse O fit"
        ),
        "runtime_parameters_added": 0,
        "optimizer_or_gradients": False,
        "raw_parent_v_probe_replaced": False,
    }
)


__all__ = [
    "RESIDUAL_VALUE_BASIS_INTEGRATION_CONTRACT",
    "RESIDUAL_VALUE_BASIS_SCHEMA",
    "ResidualAttentionValueBasisError",
    "ParentOutputBlockBasisEntry",
    "ResidualTargetedValueBasisFit",
    "ResidualTargetedValueBasisReport",
    "ResidualValueBasisLimits",
    "ResidualValueBasisSiteCalibration",
    "ResidualValueBasisSiteReport",
    "ValuePrototypeBasisEntry",
    "build_residual_value_basis_site_calibration",
    "capture_parent_output_block_basis_entry",
    "capture_value_prototype_basis_entry",
    "fit_residual_targeted_value_basis",
    "install_residual_targeted_value_basis",
]
