"""Sealed-cache adapter for LFM2 CPCF predictive-rank feasibility.

This module is deliberately isolated from the production LFM2 runner.  It
accepts only an already completed :class:`Lfm2CircuitTraceCache`-compatible
object, verifies that cache, and derives whole-batch-disjoint fit/validation
evidence for every residual boundary.  It never accepts a parent model and
cannot capture, regenerate, or extend parent traces.

The evidence is intentionally functional:

* future residual endpoints and residual writes;
* MLP post-SwiGLU features;
* actual attention Q/K score summaries and pre-O context;
* actual short-convolution B/C/x gated-path values and context; and
* token-identity-aware top-logit margin sketches.

Large producer and vocabulary features use deterministic CountSketch
projections.  Every projection and cache field is disclosed in the report, so
the resulting minimum width is explicitly the rank of this sealed evidence
contract rather than an unsupported universal information bound.

Attention Q/K score summaries are observability evidence only.  LFM installs
shared Q/K weights before Q/K normalization and RoPE, while the current cache
persists ``q_linear``/``k_linear`` and post-RoPE ``q``/``k`` but not the
normalized pre-RoPE rows or the per-position RoPE cos/sin tensors.  Therefore
this adapter fails closed on claiming that the family-neutral CPCF bilinear
factor can be installed directly into LFM.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_closed_form_causal_circuit_fold import (
    CausalFeatureBlock,
    PredictiveBoundaryMap,
    fit_predictive_boundary_map,
)


LFM2_CPCF_CACHE_ADAPTER_SCHEMA = (
    "neural-forge-lfm2-cpcf-cached-evidence-v1"
)
LFM2_CPCF_SPLIT_SCHEMA = "neural-forge-lfm2-cpcf-whole-batch-split-v1"
LFM2_CPCF_BOUNDARY_SCHEMA = "neural-forge-lfm2-cpcf-boundary-evidence-v1"
LFM2_CPCF_RANK_AUDIT_SCHEMA = "neural-forge-lfm2-cpcf-predictive-rank-audit-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2CpcfCachedEvidenceError(ValueError):
    """The sealed cache cannot support a valid CPCF feasibility audit."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF cached evidence."
        ) from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfCachedEvidenceError(
            "CPCF cache evidence metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfCachedEvidenceError(f"{name} must be a positive integer.")
    return value


def _fraction(
    value: Any,
    name: str,
    *,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfCachedEvidenceError(f"{name} must be a finite fraction.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} must be a finite fraction."
        ) from exc
    lower_ok = result > 0.0 if strictly_positive else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        bracket = "(0, 1]" if strictly_positive else "[0, 1]"
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} must be within {bracket}."
        )
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} must be finite and nonnegative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} must be finite and nonnegative."
        )
    return result


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        _canonical_json([int(size) for size in tensor.shape])
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def _float_tensor(value: Any, name: str, *, dimensions: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} must be a nonempty dense real rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} contains NaN or infinity."
        )
    return result


def _sequence_rows(
    value: Any,
    name: str,
    *,
    batch_size: int | None = None,
    sequence_size: int | None = None,
) -> Any:
    tensor = _float_tensor(value, name, dimensions=3)
    batch, sequence, width = (int(size) for size in tensor.shape)
    if batch_size is not None and batch != batch_size:
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} has a different batch axis from its residual boundary."
        )
    if sequence_size is not None and sequence != sequence_size:
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} has a different sequence axis from its residual boundary."
        )
    return tensor.reshape(batch * sequence, width).contiguous()


def _head_sequence_rows(
    value: Any,
    name: str,
    *,
    batch_size: int,
    sequence_size: int,
) -> Any:
    tensor = _float_tensor(value, name, dimensions=4)
    batch, heads, sequence, width = (int(size) for size in tensor.shape)
    if batch != batch_size or sequence != sequence_size:
        raise Lfm2CpcfCachedEvidenceError(
            f"{name} is not aligned to its residual boundary."
        )
    return (
        tensor.permute(0, 2, 1, 3)
        .reshape(batch * sequence, heads * width)
        .contiguous()
    )


def _field(
    tensors: Mapping[str, Any],
    name: str,
    *,
    layer_index: int,
) -> Any:
    try:
        return tensors[name]
    except KeyError as exc:
        raise Lfm2CpcfCachedEvidenceError(
            f"Layer {layer_index} cache evidence is missing {name!r}."
        ) from exc


@dataclass(frozen=True, slots=True)
class Lfm2CpcfEvidenceConfig:
    """Deterministic split, sketch, and predictive-rank controls."""

    validation_fraction: float = 0.25
    predictive_energy_fraction: float = 0.90
    state_reconstruction_weight: float = 0.25
    ridge: float = 1e-6
    producer_sketch_width: int = 64
    logit_topk: int = 8
    logit_identity_sketch_width: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_fraction",
            _fraction(
                self.validation_fraction,
                "validation_fraction",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "predictive_energy_fraction",
            _fraction(
                self.predictive_energy_fraction,
                "predictive_energy_fraction",
                strictly_positive=True,
            ),
        )
        if abs(self.predictive_energy_fraction - 0.90) > 1e-12:
            raise Lfm2CpcfCachedEvidenceError(
                "This feasibility contract is fixed at 90% predictive energy."
            )
        object.__setattr__(
            self,
            "state_reconstruction_weight",
            _nonnegative_float(
                self.state_reconstruction_weight,
                "state_reconstruction_weight",
            ),
        )
        object.__setattr__(
            self,
            "ridge",
            _nonnegative_float(self.ridge, "ridge"),
        )
        for name in (
            "producer_sketch_width",
            "logit_topk",
            "logit_identity_sketch_width",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation_fraction": self.validation_fraction,
            "predictive_energy_fraction": self.predictive_energy_fraction,
            "state_reconstruction_weight": self.state_reconstruction_weight,
            "ridge": self.ridge,
            "producer_sketch_width": self.producer_sketch_width,
            "logit_topk": self.logit_topk,
            "logit_identity_sketch_width": (
                self.logit_identity_sketch_width
            ),
            "split_unit": "whole_cache_batch",
            "parent_execution_permitted": False,
            "cache_regeneration_permitted": False,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCacheSplit:
    """Content-bound whole-batch fit/validation partition."""

    cache_key: str
    fit_batch_indices: tuple[int, ...]
    validation_batch_indices: tuple[int, ...]
    fit_input_sha256: tuple[str, ...]
    validation_input_sha256: tuple[str, ...]
    split_sha256: str

    def __post_init__(self) -> None:
        _digest(self.cache_key, "cache_key")
        _digest(self.split_sha256, "split_sha256")
        if not self.fit_batch_indices or not self.validation_batch_indices:
            raise Lfm2CpcfCachedEvidenceError(
                "Both whole-batch evidence splits must be nonempty."
            )
        fit = set(self.fit_batch_indices)
        validation = set(self.validation_batch_indices)
        if (
            fit & validation
            or len(fit) != len(self.fit_batch_indices)
            or len(validation) != len(self.validation_batch_indices)
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Fit and validation cache batches must be unique and disjoint."
            )
        if (
            len(self.fit_batch_indices) != len(self.fit_input_sha256)
            or len(self.validation_batch_indices)
            != len(self.validation_input_sha256)
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Split batch indices and input identities are misaligned."
            )
        for digest in (*self.fit_input_sha256, *self.validation_input_sha256):
            _digest(digest, "split input identity")
        if _sha(self._body()) != self.split_sha256:
            raise Lfm2CpcfCachedEvidenceError(
                "CPCF cache split changed after sealing."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SPLIT_SCHEMA,
            "cache_key": self.cache_key,
            "fit": [
                [index, identity]
                for index, identity in zip(
                    self.fit_batch_indices,
                    self.fit_input_sha256,
                )
            ],
            "validation": [
                [index, identity]
                for index, identity in zip(
                    self.validation_batch_indices,
                    self.validation_input_sha256,
                )
            ],
            "split_unit": "whole_cache_batch",
            "batch_sets_disjoint": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "split_sha256": self.split_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2CpcfBoundaryEvidence:
    """One layer's disjoint residual-state and causal-action evidence."""

    cache_key: str
    split_sha256: str
    layer_index: int
    operator_kind: str
    source_width: int
    fit_states: Any
    validation_states: Any
    fit_features: tuple[CausalFeatureBlock, ...]
    validation_features: tuple[CausalFeatureBlock, ...]
    fit_weights: Any
    validation_weights: Any
    fit_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    feature_contracts: Mapping[str, Mapping[str, Any]]
    evidence_sha256: str

    def __post_init__(self) -> None:
        torch = _torch()
        _digest(self.cache_key, "cache_key")
        _digest(self.split_sha256, "split_sha256")
        _digest(self.evidence_sha256, "evidence_sha256")
        _positive_int(self.source_width, "source_width")
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "layer_index must be a nonnegative integer."
            )
        if self.operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfCachedEvidenceError(
                "operator_kind must be full_attention or short_conv."
            )
        fit = _float_tensor(self.fit_states, "fit_states", dimensions=2)
        validation = _float_tensor(
            self.validation_states,
            "validation_states",
            dimensions=2,
        )
        if (
            int(fit.shape[1]) != self.source_width
            or int(validation.shape[1]) != self.source_width
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Boundary state width differs from the topology."
            )
        if (
            len(self.fit_row_ids) != int(fit.shape[0])
            or len(self.validation_row_ids) != int(validation.shape[0])
            or set(self.fit_row_ids) & set(self.validation_row_ids)
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Boundary row identities are incomplete or overlap."
            )
        if len(set(self.fit_row_ids)) != len(self.fit_row_ids) or len(
            set(self.validation_row_ids)
        ) != len(self.validation_row_ids):
            raise Lfm2CpcfCachedEvidenceError(
                "Boundary row identities must be unique within each split."
            )
        fit_names = tuple(block.name for block in self.fit_features)
        validation_names = tuple(
            block.name for block in self.validation_features
        )
        if not fit_names or fit_names != validation_names:
            raise Lfm2CpcfCachedEvidenceError(
                "Fit and validation causal feature contracts differ."
            )
        if set(fit_names) != set(self.feature_contracts):
            raise Lfm2CpcfCachedEvidenceError(
                "Feature provenance does not cover every causal block."
            )
        for block in self.fit_features:
            if int(block.values.shape[0]) != int(fit.shape[0]):
                raise Lfm2CpcfCachedEvidenceError(
                    f"Fit feature {block.name!r} is row-misaligned."
                )
        for block in self.validation_features:
            if int(block.values.shape[0]) != int(validation.shape[0]):
                raise Lfm2CpcfCachedEvidenceError(
                    f"Validation feature {block.name!r} is row-misaligned."
                )
        for value, rows, name in (
            (self.fit_weights, int(fit.shape[0]), "fit_weights"),
            (
                self.validation_weights,
                int(validation.shape[0]),
                "validation_weights",
            ),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or int(value.numel()) != rows
                or not torch.is_floating_point(value)
                or not bool(torch.isfinite(value).all().item())
                or bool((value <= 0).any().item())
            ):
                raise Lfm2CpcfCachedEvidenceError(
                    f"{name} must contain one finite positive value per row."
                )
        object.__setattr__(self, "fit_states", fit)
        object.__setattr__(self, "validation_states", validation)
        object.__setattr__(
            self,
            "fit_weights",
            self.fit_weights.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous(),
        )
        object.__setattr__(
            self,
            "validation_weights",
            self.validation_weights.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous(),
        )
        object.__setattr__(
            self,
            "feature_contracts",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(value))
                    for name, value in self.feature_contracts.items()
                }
            ),
        )

    @property
    def fit_rows(self) -> int:
        return int(self.fit_states.shape[0])

    @property
    def validation_rows(self) -> int:
        return int(self.validation_states.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_BOUNDARY_SCHEMA,
            "cache_key": self.cache_key,
            "split_sha256": self.split_sha256,
            "layer_index": self.layer_index,
            "operator_kind": self.operator_kind,
            "source_width": self.source_width,
            "fit_rows": self.fit_rows,
            "validation_rows": self.validation_rows,
            "fit_validation_row_ids_disjoint": True,
            "fit_states_sha256": _tensor_sha256(self.fit_states),
            "validation_states_sha256": _tensor_sha256(
                self.validation_states
            ),
            "feature_contracts": {
                name: dict(value)
                for name, value in self.feature_contracts.items()
            },
            "feature_names": [
                block.name for block in self.fit_features
            ],
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfBoundaryRankResult:
    """Minimum predictive width and held-out retention for one boundary."""

    layer_index: int
    operator_kind: str
    source_width: int
    minimum_width_at_90_predictive_energy: int
    predictive_energy_floor: float
    validation_state_retained_energy_fraction: float
    validation_state_relative_rmse: float
    minimum_validation_feature_retention: float
    mean_validation_feature_retention: float
    validation_blocks: Mapping[str, Mapping[str, Any]]
    report_sha256: str

    def __post_init__(self) -> None:
        _digest(self.report_sha256, "report_sha256")
        if not (
            1
            <= self.minimum_width_at_90_predictive_energy
            <= self.source_width
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Predictive-rank result has an invalid minimum width."
            )
        object.__setattr__(
            self,
            "validation_blocks",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(value))
                    for name, value in self.validation_blocks.items()
                }
            ),
        )

    @property
    def maximum_residual_width_compression_ratio(self) -> float:
        return (
            self.source_width
            / self.minimum_width_at_90_predictive_energy
        )

    def supports_width(self, width: int) -> bool:
        return (
            not isinstance(width, bool)
            and isinstance(width, int)
            and width >= self.minimum_width_at_90_predictive_energy
            and width <= self.source_width
        )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "layer_index": self.layer_index,
            "operator_kind": self.operator_kind,
            "source_width": self.source_width,
            "minimum_width_at_90_predictive_energy": (
                self.minimum_width_at_90_predictive_energy
            ),
            "maximum_residual_width_compression_ratio": (
                self.maximum_residual_width_compression_ratio
            ),
            "predictive_energy_floor": self.predictive_energy_floor,
            "validation_state_retained_energy_fraction": (
                self.validation_state_retained_energy_fraction
            ),
            "validation_state_relative_rmse": (
                self.validation_state_relative_rmse
            ),
            "minimum_validation_feature_retention": (
                self.minimum_validation_feature_retention
            ),
            "mean_validation_feature_retention": (
                self.mean_validation_feature_retention
            ),
            "validation_blocks": {
                name: dict(value)
                for name, value in self.validation_blocks.items()
            },
        }
        return {**body, "report_sha256": self.report_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2CpcfPredictiveRankAudit:
    """Whole-model per-boundary CPCF rank-feasibility result."""

    cache_key: str
    layer_topology_sha256: str
    split: Lfm2CpcfCacheSplit
    config: Lfm2CpcfEvidenceConfig
    boundaries: tuple[Lfm2CpcfBoundaryRankResult, ...]
    candidate_widths: Mapping[int, Mapping[str, Any]]
    missing_cache_fields_for_full_compilation: tuple[
        Mapping[str, Any], ...
    ]
    external_requirements_for_full_compilation: tuple[
        Mapping[str, Any], ...
    ]
    report_sha256: str

    def __post_init__(self) -> None:
        _digest(self.cache_key, "cache_key")
        _digest(self.layer_topology_sha256, "layer_topology_sha256")
        _digest(self.report_sha256, "report_sha256")
        if not self.boundaries:
            raise Lfm2CpcfCachedEvidenceError(
                "Predictive-rank audit must cover at least one boundary."
            )
        expected = tuple(range(len(self.boundaries)))
        observed = tuple(value.layer_index for value in self.boundaries)
        if observed != expected:
            raise Lfm2CpcfCachedEvidenceError(
                "Predictive-rank audit does not cover every ordered layer."
            )
        object.__setattr__(
            self,
            "candidate_widths",
            MappingProxyType(
                {
                    int(width): MappingProxyType(dict(report))
                    for width, report in self.candidate_widths.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "missing_cache_fields_for_full_compilation",
            tuple(
                MappingProxyType(dict(value))
                for value in self.missing_cache_fields_for_full_compilation
            ),
        )
        object.__setattr__(
            self,
            "external_requirements_for_full_compilation",
            tuple(
                MappingProxyType(dict(value))
                for value in self.external_requirements_for_full_compilation
            ),
        )

    @property
    def source_width(self) -> int:
        widths = {value.source_width for value in self.boundaries}
        if len(widths) != 1:
            raise Lfm2CpcfCachedEvidenceError(
                "LFM2 boundary source widths are not uniform."
            )
        return next(iter(widths))

    @property
    def minimum_uniform_width_at_90_predictive_energy(self) -> int:
        return max(
            value.minimum_width_at_90_predictive_energy
            for value in self.boundaries
        )

    @property
    def maximum_uniform_residual_width_compression_ratio(self) -> float:
        return (
            self.source_width
            / self.minimum_uniform_width_at_90_predictive_energy
        )

    @property
    def full_compilation_ready(self) -> bool:
        return not (
            self.missing_cache_fields_for_full_compilation
            or self.external_requirements_for_full_compilation
        )

    def supports_uniform_width(self, width: int) -> bool:
        return all(value.supports_width(width) for value in self.boundaries)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": LFM2_CPCF_RANK_AUDIT_SCHEMA,
            "cache_key": self.cache_key,
            "layer_topology_sha256": self.layer_topology_sha256,
            "split": self.split.as_dict(),
            "config": self.config.as_dict(),
            "source_width": self.source_width,
            "layer_count": len(self.boundaries),
            "minimum_uniform_width_at_90_predictive_energy": (
                self.minimum_uniform_width_at_90_predictive_energy
            ),
            "maximum_uniform_residual_width_compression_ratio": (
                self.maximum_uniform_residual_width_compression_ratio
            ),
            "boundaries": [value.as_dict() for value in self.boundaries],
            "candidate_widths": {
                str(width): dict(report)
                for width, report in self.candidate_widths.items()
            },
            "missing_cache_fields_for_full_compilation": [
                dict(value)
                for value in self.missing_cache_fields_for_full_compilation
            ],
            "external_requirements_for_full_compilation": [
                dict(value)
                for value in self.external_requirements_for_full_compilation
            ],
            "full_compilation_ready": self.full_compilation_ready,
            "operation_audit": {
                "parent_forward_count": 0,
                "parent_generation_count": 0,
                "parent_cache_regeneration_count": 0,
                "training_step_count": 0,
                "backward_call_count": 0,
                "optimizer_step_count": 0,
            },
            "parameter_compaction_claimed": False,
            "measurement_only": True,
            "installability_contract_evaluated": False,
            "parent_side_compiler_inputs_ready": False,
            "scope": "residual_width_predictive_rank_feasibility_only",
        }
        return {**body, "report_sha256": self.report_sha256}


def _build_split(
    cache_key: str,
    ordered_input_sha256: Sequence[str],
    validation_fraction: float,
) -> Lfm2CpcfCacheSplit:
    identities = tuple(
        _digest(value, f"batch {index} input identity")
        for index, value in enumerate(ordered_input_sha256)
    )
    if len(identities) < 2:
        raise Lfm2CpcfCachedEvidenceError(
            "CPCF validation requires at least two complete cache batches."
        )
    validation_count = min(
        len(identities) - 1,
        max(1, int(round(len(identities) * validation_fraction))),
    )
    ranked = sorted(
        range(len(identities)),
        key=lambda index: (
            hashlib.sha256(
                (
                    f"{cache_key}\0{identities[index]}\0"
                    "lfm2-cpcf-validation-v1"
                ).encode("ascii")
            ).hexdigest(),
            index,
        ),
    )
    validation_set = frozenset(ranked[:validation_count])
    fit_indices = tuple(
        index for index in range(len(identities)) if index not in validation_set
    )
    validation_indices = tuple(
        index for index in range(len(identities)) if index in validation_set
    )
    body = {
        "schema": LFM2_CPCF_SPLIT_SCHEMA,
        "cache_key": cache_key,
        "fit": [
            [index, identities[index]] for index in fit_indices
        ],
        "validation": [
            [index, identities[index]] for index in validation_indices
        ],
        "split_unit": "whole_cache_batch",
        "batch_sets_disjoint": True,
    }
    return Lfm2CpcfCacheSplit(
        cache_key=cache_key,
        fit_batch_indices=fit_indices,
        validation_batch_indices=validation_indices,
        fit_input_sha256=tuple(identities[index] for index in fit_indices),
        validation_input_sha256=tuple(
            identities[index] for index in validation_indices
        ),
        split_sha256=_sha(body),
    )


def _countsketch_columns(
    rows: Any,
    target_width: int,
    *,
    seed_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    source = _float_tensor(rows, "CountSketch rows", dimensions=2)
    source_width = int(source.shape[1])
    selected_width = min(source_width, target_width)
    if selected_width == source_width:
        return source, {
            "transform": "identity",
            "source_width": source_width,
            "output_width": source_width,
        }
    seed = int(_digest(seed_sha256, "CountSketch seed")[:16], 16)
    columns = torch.arange(source_width, dtype=torch.int64)
    buckets = (
        columns * 104729 + (seed % 1_000_003)
    ).remainder(selected_width)
    signs = torch.where(
        (
            columns * 13007 + ((seed >> 17) % 1_000_033)
        ).remainder(2)
        == 0,
        torch.ones(source_width, dtype=torch.float64),
        -torch.ones(source_width, dtype=torch.float64),
    )
    result = torch.zeros(
        (int(source.shape[0]), selected_width),
        dtype=torch.float64,
    )
    result.scatter_add_(
        1,
        buckets.unsqueeze(0).expand(int(source.shape[0]), -1),
        source * signs.unsqueeze(0),
    )
    result.mul_(math.sqrt(selected_width / source_width))
    return result.contiguous(), {
        "transform": "deterministic_signed_column_countsketch",
        "source_width": source_width,
        "output_width": selected_width,
        "seed_sha256": seed_sha256,
        "energy_scale": math.sqrt(selected_width / source_width),
    }


def _logit_margin_features(
    logits: Any,
    config: Lfm2CpcfEvidenceConfig,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    rows = _float_tensor(logits, "cached logits", dimensions=2)
    vocabulary = int(rows.shape[1])
    topk = min(config.logit_topk, vocabulary)
    if topk < 2:
        raise Lfm2CpcfCachedEvidenceError(
            "Logit-margin evidence requires at least two vocabulary rows."
        )
    values, indices = torch.topk(rows, k=topk, dim=1)
    margins = values[:, :1] - values[:, 1:]
    sketch_width = min(config.logit_identity_sketch_width, vocabulary)
    identity = torch.zeros(
        (int(rows.shape[0]), sketch_width),
        dtype=torch.float64,
    )

    def token_bucket(token_ids: Any) -> Any:
        return (token_ids * 104729 + 8191).remainder(sketch_width)

    def token_sign(token_ids: Any) -> Any:
        return torch.where(
            (token_ids * 13007 + 4099).remainder(2) == 0,
            torch.ones_like(token_ids, dtype=torch.float64),
            -torch.ones_like(token_ids, dtype=torch.float64),
        )

    winner = indices[:, :1].expand(-1, topk - 1)
    competitors = indices[:, 1:]
    weighted = margins / math.sqrt(max(1, topk - 1))
    identity.scatter_add_(
        1,
        token_bucket(winner),
        weighted * token_sign(winner),
    )
    identity.scatter_add_(
        1,
        token_bucket(competitors),
        -weighted * token_sign(competitors),
    )
    return torch.cat((margins, identity), dim=1).contiguous(), {
        "source_fields": ["<phase>.logits"],
        "transform": (
            "top1_vs_topk_margin_values_plus_signed_token_identity_countsketch"
        ),
        "vocabulary_size": vocabulary,
        "topk": topk,
        "raw_margin_width": topk - 1,
        "identity_sketch_width": sketch_width,
        "output_width": topk - 1 + sketch_width,
        "full_logit_vector_retained": False,
        "token_identity_included": True,
    }


def _attention_score_summary(
    query: Any,
    key_history: Any,
    *,
    phase: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    q_value = _float_tensor(query, "attention query", dimensions=4)
    k_value = _float_tensor(
        key_history,
        "attention key history",
        dimensions=4,
    )
    batch, query_heads, query_sequence, head_dim = (
        int(size) for size in q_value.shape
    )
    key_batch, key_heads, key_sequence, key_dim = (
        int(size) for size in k_value.shape
    )
    if (
        key_batch != batch
        or key_dim != head_dim
        or key_sequence < query_sequence
    ):
        raise Lfm2CpcfCachedEvidenceError(
            "Attention Q/K score evidence has incompatible shapes."
        )
    if query_heads % key_heads != 0:
        raise Lfm2CpcfCachedEvidenceError(
            "LFM2 GQA query heads are not divisible by key/value heads."
        )
    group = query_heads // key_heads
    mapped = k_value.index_select(
        1,
        torch.arange(query_heads, dtype=torch.long) // group,
    )
    summaries = []
    query_start = key_sequence - query_sequence
    scale = math.sqrt(head_dim)
    for position in range(query_sequence):
        allowed = query_start + position + 1
        q_row = q_value[:, :, position, :]
        scores = torch.einsum(
            "bhd,bhkd->bhk",
            q_row,
            mapped[:, :, :allowed, :],
        ) / scale
        mean = scores.mean(dim=2)
        centered = scores - mean.unsqueeze(2)
        current_minus_mean = centered[:, :, -1]
        centered_rms = centered.square().mean(dim=2).sqrt()
        maximum_minus_mean = centered.max(dim=2).values
        summaries.append(
            torch.stack(
                (
                    mean,
                    current_minus_mean,
                    centered_rms,
                    maximum_minus_mean,
                ),
                dim=2,
            )
        )
    result = (
        torch.stack(summaries, dim=2)
        .permute(0, 2, 1, 3)
        .reshape(batch * query_sequence, query_heads * 4)
        .contiguous()
    )
    return result, {
        "source_fields": [
            "<phase>.layer.<index>.attention.q",
            (
                "<phase>.layer.<index>.attention.key_cache"
                " or attention.k"
            ),
        ],
        "transform": (
            "scaled_qk_history_mean_current_centered_rms_and_centered_max"
        ),
        "phase": phase,
        "query_heads": query_heads,
        "key_value_heads": key_heads,
        "head_dim": head_dim,
        "query_rows": query_sequence,
        "sampled_key_rows": key_sequence,
        "output_width": query_heads * 4,
        "post_rope_observability_only": True,
        "installable_pre_rope_qk_factor_claimed": False,
    }


def _phase_names(record: Mapping[str, Any], batch_index: int) -> tuple[str, ...]:
    capture = record.get("capture")
    if not isinstance(capture, Mapping):
        raise Lfm2CpcfCachedEvidenceError(
            f"Cache batch {batch_index} has no capture record."
        )
    decode_steps = capture.get("decode_steps")
    if (
        isinstance(decode_steps, bool)
        or not isinstance(decode_steps, int)
        or decode_steps < 0
    ):
        raise Lfm2CpcfCachedEvidenceError(
            f"Cache batch {batch_index} has an invalid decode count."
        )
    return (
        "prefill",
        *(f"decode.{step:03d}" for step in range(decode_steps)),
    )


def _feature_specifications(
    cache_key: str,
    config: Lfm2CpcfEvidenceConfig,
    tensors: Mapping[str, Any],
    *,
    phase: str,
    layer_index: int,
    final_layer_index: int,
    operator_kind: str,
) -> tuple[Any, dict[str, tuple[str, str, float, Any, dict[str, Any]]]]:
    prefix = f"{phase}.layer.{layer_index:03d}."
    final_prefix = f"{phase}.layer.{final_layer_index:03d}."
    state_tensor = _float_tensor(
        _field(
            tensors,
            prefix + "residual.pre",
            layer_index=layer_index,
        ),
        prefix + "residual.pre",
        dimensions=3,
    )
    batch_size, sequence_size, source_width = (
        int(size) for size in state_tensor.shape
    )
    state = state_tensor.reshape(-1, source_width).contiguous()

    def sequence(name: str) -> Any:
        return _sequence_rows(
            _field(tensors, name, layer_index=layer_index),
            name,
            batch_size=batch_size,
            sequence_size=sequence_size,
        )

    next_residual = sequence(prefix + "residual.post_layer")
    final_residual = _sequence_rows(
        _field(
            tensors,
            final_prefix + "residual.post_layer",
            layer_index=layer_index,
        ),
        final_prefix + "residual.post_layer",
        batch_size=batch_size,
        sequence_size=sequence_size,
    )
    operator_write = sequence(prefix + "operator.output")
    mlp_write = sequence(prefix + "mlp.down")
    mlp_features = sequence(prefix + "mlp.post_swiglu")
    mlp_sketch_seed = _sha(
        {
            "cache_key": cache_key,
            "layer_index": layer_index,
            "field": "mlp.post_swiglu",
            "schema": LFM2_CPCF_CACHE_ADAPTER_SCHEMA,
        }
    )
    mlp_sketch, mlp_contract = _countsketch_columns(
        mlp_features,
        config.producer_sketch_width,
        seed_sha256=mlp_sketch_seed,
    )
    logits = _sequence_rows(
        _field(tensors, f"{phase}.logits", layer_index=layer_index),
        f"{phase}.logits",
        batch_size=batch_size,
        sequence_size=sequence_size,
    )
    logit_features, logit_contract = _logit_margin_features(logits, config)
    specifications: dict[
        str,
        tuple[str, str, float, Any, dict[str, Any]],
    ] = {
        "future.next_residual": (
            "future.next_residual",
            "future_residual",
            1.0,
            next_residual,
            {
                "source_fields": [prefix + "residual.post_layer"],
                "transform": "identity",
                "output_width": source_width,
            },
        ),
        "future.final_residual_delta": (
            "future.final_residual_delta",
            "future_residual",
            1.0,
            final_residual - state,
            {
                "source_fields": [
                    prefix + "residual.pre",
                    final_prefix + "residual.post_layer",
                ],
                "transform": "final_residual_minus_boundary_state",
                "output_width": source_width,
            },
        ),
        "action.operator_residual_write": (
            "action.operator_residual_write",
            "operator_residual_write",
            1.0,
            operator_write,
            {
                "source_fields": [prefix + "operator.output"],
                "transform": "identity",
                "output_width": source_width,
            },
        ),
        "action.mlp_residual_write": (
            "action.mlp_residual_write",
            "mlp_residual_write",
            1.0,
            mlp_write,
            {
                "source_fields": [prefix + "mlp.down"],
                "transform": "identity",
                "output_width": source_width,
            },
        ),
        "action.mlp_post_swiglu": (
            "action.mlp_post_swiglu",
            "mlp_nonlinear_producer",
            0.5,
            mlp_sketch,
            {
                "source_fields": [prefix + "mlp.post_swiglu"],
                **mlp_contract,
            },
        ),
        "action.logit_margins": (
            "action.logit_margins",
            "output_decision_margin",
            2.0,
            logit_features,
            logit_contract,
        ),
    }
    if operator_kind == "full_attention":
        query_name = prefix + "attention.q"
        query = _field(tensors, query_name, layer_index=layer_index)
        cache_prefix = (
            "cached_prefill" if phase == "prefill" else phase
        )
        key_cache_name = (
            f"{cache_prefix}.layer.{layer_index:03d}."
            "attention.key_cache"
        )
        key_local_name = prefix + "attention.k"
        if key_cache_name in tensors:
            key = tensors[key_cache_name]
            history_source = key_cache_name
        else:
            key = _field(
                tensors,
                key_local_name,
                layer_index=layer_index,
            )
            history_source = key_local_name
        score_summary, score_contract = _attention_score_summary(
            query,
            key,
            phase=phase,
        )
        score_contract = {
            **score_contract,
            "resolved_key_history_field": history_source,
            "key_history_is_tail_sample_only": True,
        }
        context_name = prefix + "attention.context"
        context = _head_sequence_rows(
            _field(tensors, context_name, layer_index=layer_index),
            context_name,
            batch_size=batch_size,
            sequence_size=sequence_size,
        )
        specifications["action.attention_qk_scores"] = (
            "action.attention_qk_scores",
            "attention_score_geometry",
            1.0,
            score_summary,
            score_contract,
        )
        specifications["action.attention_context"] = (
            "action.attention_context",
            "attention_value_context",
            1.0,
            context,
            {
                "source_fields": [context_name],
                "transform": "head_sequence_flatten",
                "output_width": int(context.shape[1]),
            },
        )
    else:
        path_names = [
            prefix + "short_conv.b",
            prefix + "short_conv.c",
            prefix + "short_conv.x",
            prefix + "short_conv.bx",
        ]
        path = _torch().cat(
            [sequence(name) for name in path_names],
            dim=1,
        )
        conv_seed = _sha(
            {
                "cache_key": cache_key,
                "layer_index": layer_index,
                "field": "short_conv.b_c_x_bx",
                "schema": LFM2_CPCF_CACHE_ADAPTER_SCHEMA,
            }
        )
        path_sketch, path_contract = _countsketch_columns(
            path,
            config.producer_sketch_width,
            seed_sha256=conv_seed,
        )
        context_name = prefix + "short_conv.context"
        context = sequence(context_name)
        specifications["action.short_conv_bcx_path"] = (
            "action.short_conv_bcx_path",
            "short_conv_multiplicative_path",
            1.0,
            path_sketch,
            {
                "source_fields": path_names,
                **path_contract,
            },
        )
        specifications["action.short_conv_context"] = (
            "action.short_conv_context",
            "short_conv_filtered_context",
            1.0,
            context,
            {
                "source_fields": [context_name],
                "transform": "identity",
                "output_width": int(context.shape[1]),
            },
        )
    return state, specifications


def _balanced_phase_weights(group_ids: Sequence[str]) -> Any:
    torch = _torch()
    if not group_ids:
        raise Lfm2CpcfCachedEvidenceError(
            "Cannot weight an empty boundary split."
        )
    counts: dict[str, int] = {}
    for group in group_ids:
        counts[group] = counts.get(group, 0) + 1
    raw = torch.tensor(
        [1.0 / counts[group] for group in group_ids],
        dtype=torch.float64,
    )
    return (raw * (len(group_ids) / float(raw.sum().item()))).contiguous()


def _stable_feature_contract(
    value: Any,
    *,
    phase: str,
    layer_index: int,
    final_layer_index: int,
) -> Any:
    """Remove phase-specific cache paths while retaining exact provenance."""

    if isinstance(value, str):
        result = value
        if result.startswith("cached_prefill."):
            result = "<phase_cache>." + result[len("cached_prefill.") :]
        elif result.startswith(phase + "."):
            result = "<phase>." + result[len(phase) + 1 :]
        result = result.replace(
            f"layer.{layer_index:03d}",
            "layer.<index>",
        )
        result = result.replace(
            f"layer.{final_layer_index:03d}",
            "layer.<final>",
        )
        return result
    if isinstance(value, Mapping):
        return {
            str(key): _stable_feature_contract(
                item,
                phase=phase,
                layer_index=layer_index,
                final_layer_index=final_layer_index,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _stable_feature_contract(
                item,
                phase=phase,
                layer_index=layer_index,
                final_layer_index=final_layer_index,
            )
            for item in value
        ]
    return value


def _minimum_width_validation(
    boundary_map: PredictiveBoundaryMap,
    evidence: Lfm2CpcfBoundaryEvidence,
    *,
    minimum_width: int,
    ridge: float,
) -> dict[str, Any]:
    torch = _torch()
    encoder = boundary_map.encoder[:, :minimum_width].contiguous()
    fit_latent = evidence.fit_states @ encoder
    validation_latent = evidence.validation_states @ encoder

    def ridge_fit(inputs: Any, targets: Any, weights: Any) -> Any:
        root = torch.sqrt(weights).unsqueeze(1)
        design = inputs * root
        response = targets * root
        gram = design.T @ design
        return torch.linalg.solve(
            gram
            + ridge
            * torch.eye(int(gram.shape[0]), dtype=torch.float64),
            design.T @ response,
        )

    def metrics(prediction: Any, target: Any, weights: Any) -> tuple[float, float]:
        scale = weights.unsqueeze(1)
        target_energy = (scale * target.square()).sum().clamp_min(1e-30)
        error_energy = (scale * (prediction - target).square()).sum()
        retention = float((1.0 - error_energy / target_energy).item())
        relative_rmse = float(
            torch.sqrt(error_energy / target_energy).item()
        )
        return retention, relative_rmse

    decoder = ridge_fit(
        fit_latent,
        evidence.fit_states,
        evidence.fit_weights,
    )
    state_retention, state_rmse = metrics(
        validation_latent @ decoder,
        evidence.validation_states,
        evidence.validation_weights,
    )
    validation_table = {
        block.name: block for block in evidence.validation_features
    }
    block_reports: dict[str, dict[str, Any]] = {}
    retentions = []
    for block in evidence.fit_features:
        predictor = ridge_fit(
            fit_latent,
            block.values,
            evidence.fit_weights,
        )
        target = validation_table[block.name].values
        retention, relative_rmse = metrics(
            validation_latent @ predictor,
            target,
            evidence.validation_weights,
        )
        retentions.append(retention)
        block_reports[block.name] = {
            "role": block.role,
            "retained_energy_fraction": retention,
            "relative_rmse": relative_rmse,
        }
    return {
        "validation_state_retained_energy_fraction": state_retention,
        "validation_state_relative_rmse": state_rmse,
        "minimum_validation_feature_retention": min(retentions),
        "mean_validation_feature_retention": sum(retentions) / len(retentions),
        "validation_blocks": block_reports,
    }


def _full_compilation_gaps(
    topology: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    kinds = {
        str(entry["contract"]["kind"])
        for entry in topology
    }
    missing: list[dict[str, Any]] = [
        {
            "scope": "model_output",
            "missing_cache_fields": [
                "model.final_norm.input",
                "model.final_norm.output",
            ],
            "present_alternative": ["<phase>.logits", "residual.post_layer"],
            "reason": (
                "Direct final-norm/head compilation cannot distinguish the "
                "final norm from the vocabulary writer using the current "
                "cache endpoints alone."
            ),
        },
        {
            "scope": "behavior_observability",
            "missing_cache_fields": [
                "output.eos_token_id",
                "output.repetition_direction_labels",
                "output.task_capability_direction_labels",
            ],
            "present_alternative": ["<phase>.logits"],
            "reason": (
                "Raw logits permit generic top-token margins, but the cache "
                "does not identify EOS, repetition, or task-specific actions."
            ),
        },
    ]
    if "full_attention" in kinds:
        missing.extend(
            [
                {
                    "scope": "attention_qk_installability",
                    "missing_cache_fields": [
                        "attention.q_normalized_pre_rope",
                        "attention.k_normalized_pre_rope",
                        "attention.rope_cos",
                        "attention.rope_sin",
                    ],
                    "present_alternative": [
                        "attention.q_linear",
                        "attention.k_linear",
                        "attention.q",
                        "attention.k",
                    ],
                    "reason": (
                        "LFM Q/K weights are shared, pre-normalization/pre-RoPE "
                        "operators. A position-independent factor of cached "
                        "post-RoPE scores is not an installable LFM weight "
                        "solution without the missing transforms and a shared-"
                        "weight realizability solve."
                    ),
                },
                {
                    "scope": "attention_score_coverage",
                    "missing_cache_fields": [
                        "attention.full_causal_score_matrix",
                        "attention.probabilities",
                    ],
                    "present_alternative": [
                        "attention.q",
                        "attention.k",
                        "attention.key_cache (tail sample in cached mode)",
                        "attention.context",
                    ],
                    "reason": (
                        "The adapter can form sampled score summaries, but the "
                        "cache does not persist every causal key score or the "
                        "actual softmax probability matrix."
                    ),
                },
            ]
        )
    if "short_conv" in kinds:
        missing.append(
            {
                "scope": "short_conv_compilation",
                "missing_cache_fields": [
                    "short_conv.kernel_weights",
                    "short_conv.complete_pre_sample_bx_history",
                ],
                "present_alternative": [
                    "short_conv.b",
                    "short_conv.c",
                    "short_conv.x",
                    "short_conv.bx",
                    "short_conv.context",
                    "short_conv.conv_state (cached mode)",
                    "short_conv.kernel_size (cached mode)",
                ],
                "reason": (
                    "Functional FIR compilation needs either the exact source "
                    "kernel or complete causal B*x histories. The current "
                    "sampled cache supplies neither for every prefill row."
                ),
            }
        )
    external = (
        {
            "scope": "target_closed_loop",
            "requirement": (
                "Child-only full-prefix replay after each compiled macrocell"
            ),
            "reason": (
                "Parent evidence selects predictive coordinates; it cannot "
                "substitute for target-width closed-loop child states."
            ),
        },
        {
            "scope": "tokenizer_and_vocabulary",
            "requirement": (
                "Sealed donor-to-target token bridge, target special-token "
                "identities, and projected target-vocabulary margins"
            ),
            "reason": (
                "The parent circuit cache is tokenizer-local and does not "
                "define a cross-tokenizer target decision space."
            ),
        },
        {
            "scope": "target_topology",
            "requirement": (
                "Exact blank-target tensor inventory and tied-parameter audit"
            ),
            "reason": (
                "Predictive rank is a width feasibility measurement, not an "
                "exact parameter-count or installation proof."
            ),
        },
    )
    return tuple(missing), external


class Lfm2CpcfCachedEvidenceAdapter:
    """Selectable read-only adapter over one immutable LFM2 circuit cache."""

    def __init__(
        self,
        cache: Any,
        *,
        config: Lfm2CpcfEvidenceConfig | None = None,
    ) -> None:
        self.cache = cache
        self.config = (
            Lfm2CpcfEvidenceConfig() if config is None else config
        )
        if not isinstance(self.config, Lfm2CpcfEvidenceConfig):
            raise Lfm2CpcfCachedEvidenceError(
                "config must be Lfm2CpcfEvidenceConfig."
            )
        verify = getattr(cache, "verify", None)
        if not callable(verify):
            raise Lfm2CpcfCachedEvidenceError(
                "A CPCF parent cache must expose the sealed-cache verify() API."
            )
        verify()
        self.cache_key = _digest(
            getattr(cache, "cache_key", None),
            "verified cache_key",
        )
        manifest = getattr(cache, "manifest", None)
        if not isinstance(manifest, Mapping):
            raise Lfm2CpcfCachedEvidenceError(
                "Verified cache manifest is unavailable."
            )
        topology_section = manifest.get("layer_topology")
        corpus = manifest.get("corpus")
        if not isinstance(topology_section, Mapping) or not isinstance(
            corpus,
            Mapping,
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Cache manifest lacks topology or corpus identity."
            )
        self.layer_topology_sha256 = _digest(
            topology_section.get("layer_topology_sha256"),
            "layer_topology_sha256",
        )
        layers = topology_section.get("layers")
        if not isinstance(layers, Sequence) or isinstance(
            layers,
            (str, bytes),
        ) or not layers:
            raise Lfm2CpcfCachedEvidenceError(
                "Cache topology has no ordered LFM2 layers."
            )
        self.topology = tuple(layers)
        for index, entry in enumerate(self.topology):
            if not isinstance(entry, Mapping):
                raise Lfm2CpcfCachedEvidenceError(
                    f"Layer topology entry {index} is invalid."
                )
            contract = entry.get("contract")
            if (
                not isinstance(contract, Mapping)
                or contract.get("layer_index") != index
                or contract.get("kind")
                not in {"full_attention", "short_conv"}
                or contract.get("hidden_size") is None
            ):
                raise Lfm2CpcfCachedEvidenceError(
                    f"Layer topology entry {index} is not canonical LFM2."
                )
        widths = {
            _positive_int(
                entry["contract"]["hidden_size"],
                f"layer {index} hidden_size",
            )
            for index, entry in enumerate(self.topology)
        }
        if len(widths) != 1:
            raise Lfm2CpcfCachedEvidenceError(
                "This LFM2 CPCF adapter requires one residual width."
            )
        self.source_width = next(iter(widths))
        ordered = corpus.get("ordered_batch_input_sha256")
        if not isinstance(ordered, Sequence) or isinstance(
            ordered,
            (str, bytes),
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Cache corpus lacks ordered batch identities."
            )
        batch_count = getattr(cache, "batch_count", None)
        if (
            isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count != len(ordered)
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "Cache batch count differs from its sealed corpus."
            )
        self.ordered_input_sha256 = tuple(str(value) for value in ordered)
        self.split = _build_split(
            self.cache_key,
            self.ordered_input_sha256,
            self.config.validation_fraction,
        )
        (
            self.missing_cache_fields_for_full_compilation,
            self.external_requirements_for_full_compilation,
        ) = _full_compilation_gaps(self.topology)

    def _build_boundary(
        self,
        layer_index: int,
    ) -> Lfm2CpcfBoundaryEvidence:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < len(self.topology)
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "layer_index is outside the sealed LFM2 topology."
            )
        torch = _torch()
        operator_kind = str(
            self.topology[layer_index]["contract"]["kind"]
        )
        split_indices = {
            "fit": self.split.fit_batch_indices,
            "validation": self.split.validation_batch_indices,
        }
        state_chunks: dict[str, list[Any]] = {
            "fit": [],
            "validation": [],
        }
        feature_chunks: dict[str, dict[str, list[Any]]] = {
            "fit": {},
            "validation": {},
        }
        row_ids: dict[str, list[str]] = {"fit": [], "validation": []}
        group_ids: dict[str, list[str]] = {"fit": [], "validation": []}
        contracts: dict[str, dict[str, Any]] = {}
        feature_roles: dict[str, tuple[str, float]] = {}
        for split_name, batch_indices in split_indices.items():
            for batch_index in batch_indices:
                batch = self.cache.load_batch(batch_index)
                expected_identity = self.ordered_input_sha256[batch_index]
                if getattr(batch, "input_sha256", None) != expected_identity:
                    raise Lfm2CpcfCachedEvidenceError(
                        f"Cache batch {batch_index} identity changed."
                    )
                tensors = getattr(batch, "tensors", None)
                record = getattr(batch, "record", None)
                if not isinstance(tensors, Mapping) or not isinstance(
                    record,
                    Mapping,
                ):
                    raise Lfm2CpcfCachedEvidenceError(
                        f"Cache batch {batch_index} is not a verified trace batch."
                    )
                for phase in _phase_names(record, batch_index):
                    state, specifications = _feature_specifications(
                        self.cache_key,
                        self.config,
                        tensors,
                        phase=phase,
                        layer_index=layer_index,
                        final_layer_index=len(self.topology) - 1,
                        operator_kind=operator_kind,
                    )
                    rows = int(state.shape[0])
                    if int(state.shape[1]) != self.source_width:
                        raise Lfm2CpcfCachedEvidenceError(
                            f"Layer {layer_index} residual width changed."
                        )
                    state_chunks[split_name].append(state)
                    group = f"batch.{batch_index:06d}/{phase}"
                    for row in range(rows):
                        row_ids[split_name].append(
                            f"{expected_identity}/{phase}/row.{row:04d}"
                        )
                        group_ids[split_name].append(group)
                    observed_names = tuple(sorted(specifications))
                    if contracts and set(observed_names) != set(contracts):
                        raise Lfm2CpcfCachedEvidenceError(
                            f"Layer {layer_index} feature fields changed by phase."
                        )
                    for (
                        name,
                        (_canonical_name, role, weight, values, contract),
                    ) in specifications.items():
                        if int(values.shape[0]) != rows:
                            raise Lfm2CpcfCachedEvidenceError(
                                f"Feature {name!r} is not aligned to state rows."
                            )
                        previous = contracts.get(name)
                        stable_contract = {
                            key: value
                            for key, value in _stable_feature_contract(
                                contract,
                                phase=phase,
                                layer_index=layer_index,
                                final_layer_index=len(self.topology) - 1,
                            ).items()
                            if key
                            not in {
                                "phase",
                                "sampled_key_rows",
                                "query_rows",
                                "resolved_key_history_field",
                            }
                        }
                        if previous is None:
                            contracts[name] = stable_contract
                            feature_roles[name] = (role, weight)
                        elif previous != stable_contract:
                            raise Lfm2CpcfCachedEvidenceError(
                                f"Feature {name!r} changed its width or transform."
                            )
                        feature_chunks[split_name].setdefault(
                            name,
                            [],
                        ).append(values)
        fit_states = torch.cat(state_chunks["fit"], dim=0).contiguous()
        validation_states = torch.cat(
            state_chunks["validation"],
            dim=0,
        ).contiguous()
        fit_features = tuple(
            CausalFeatureBlock(
                name=name,
                role=feature_roles[name][0],
                weight=feature_roles[name][1],
                values=torch.cat(
                    feature_chunks["fit"][name],
                    dim=0,
                ),
            )
            for name in sorted(contracts)
        )
        validation_features = tuple(
            CausalFeatureBlock(
                name=name,
                role=feature_roles[name][0],
                weight=feature_roles[name][1],
                values=torch.cat(
                    feature_chunks["validation"][name],
                    dim=0,
                ),
            )
            for name in sorted(contracts)
        )
        fit_weights = _balanced_phase_weights(group_ids["fit"])
        validation_weights = _balanced_phase_weights(
            group_ids["validation"]
        )
        evidence_body = {
            "schema": LFM2_CPCF_BOUNDARY_SCHEMA,
            "cache_key": self.cache_key,
            "split_sha256": self.split.split_sha256,
            "layer_index": layer_index,
            "operator_kind": operator_kind,
            "source_width": self.source_width,
            "fit_states_sha256": _tensor_sha256(fit_states),
            "validation_states_sha256": _tensor_sha256(validation_states),
            "fit_feature_sha256": {
                block.name: _tensor_sha256(block.values)
                for block in fit_features
            },
            "validation_feature_sha256": {
                block.name: _tensor_sha256(block.values)
                for block in validation_features
            },
            "fit_row_ids_sha256": _sha(row_ids["fit"]),
            "validation_row_ids_sha256": _sha(row_ids["validation"]),
            "feature_contracts": contracts,
            "phase_balanced_weights": True,
        }
        return Lfm2CpcfBoundaryEvidence(
            cache_key=self.cache_key,
            split_sha256=self.split.split_sha256,
            layer_index=layer_index,
            operator_kind=operator_kind,
            source_width=self.source_width,
            fit_states=fit_states,
            validation_states=validation_states,
            fit_features=fit_features,
            validation_features=validation_features,
            fit_weights=fit_weights,
            validation_weights=validation_weights,
            fit_row_ids=tuple(row_ids["fit"]),
            validation_row_ids=tuple(row_ids["validation"]),
            feature_contracts=contracts,
            evidence_sha256=_sha(evidence_body),
        )

    def boundary(self, layer_index: int) -> Lfm2CpcfBoundaryEvidence:
        """Build one layer's evidence without retaining other cache batches."""

        return self._build_boundary(layer_index)

    def iter_boundaries(self) -> Iterator[Lfm2CpcfBoundaryEvidence]:
        """Yield every layer boundary in canonical topology order."""

        for layer_index in range(len(self.topology)):
            yield self._build_boundary(layer_index)

    def audit(
        self,
        *,
        candidate_widths: Sequence[int] = (),
    ) -> Lfm2CpcfPredictiveRankAudit:
        """Measure the 90% predictive width and held-out retention per layer."""

        requested = tuple(candidate_widths)
        if isinstance(candidate_widths, (str, bytes)) or any(
            isinstance(width, bool)
            or not isinstance(width, int)
            or not 1 <= width <= self.source_width
            for width in requested
        ):
            raise Lfm2CpcfCachedEvidenceError(
                "candidate_widths must contain in-range positive integers."
            )
        if len(set(requested)) != len(requested):
            raise Lfm2CpcfCachedEvidenceError(
                "candidate_widths must not contain duplicates."
            )
        results: list[Lfm2CpcfBoundaryRankResult] = []
        for evidence in self.iter_boundaries():
            fitted = fit_predictive_boundary_map(
                evidence.fit_states,
                evidence.fit_features,
                target_width=evidence.source_width,
                validation_states=evidence.validation_states,
                validation_features=evidence.validation_features,
                fit_weights=evidence.fit_weights,
                validation_weights=evidence.validation_weights,
                state_reconstruction_weight=(
                    self.config.state_reconstruction_weight
                ),
                ridge=self.config.ridge,
                minimum_predictive_energy_fraction=(
                    self.config.predictive_energy_fraction
                ),
                minimum_validation_block_retention=0.0,
                # A feasibility audit must record negative held-out transfer
                # instead of aborting before it can identify the offending
                # boundary. Compilation gates keep the strict default.
                enforce_validation_block_floor=False,
            )
            minimum_width = int(
                fitted.report[
                    "minimum_width_at_predictive_energy_floor"
                ]
            )
            validation = _minimum_width_validation(
                fitted,
                evidence,
                minimum_width=minimum_width,
                ridge=self.config.ridge,
            )
            body = {
                "schema": LFM2_CPCF_RANK_AUDIT_SCHEMA,
                "scope": "layer_boundary",
                "cache_key": self.cache_key,
                "split_sha256": self.split.split_sha256,
                "evidence_sha256": evidence.evidence_sha256,
                "layer_index": evidence.layer_index,
                "operator_kind": evidence.operator_kind,
                "source_width": evidence.source_width,
                "predictive_energy_fraction": (
                    self.config.predictive_energy_fraction
                ),
                "minimum_width": minimum_width,
                "predictive_energy_floor": (
                    self.config.predictive_energy_fraction
                ),
                **validation,
            }
            results.append(
                Lfm2CpcfBoundaryRankResult(
                    layer_index=evidence.layer_index,
                    operator_kind=evidence.operator_kind,
                    source_width=evidence.source_width,
                    minimum_width_at_90_predictive_energy=minimum_width,
                    predictive_energy_floor=(
                        self.config.predictive_energy_fraction
                    ),
                    validation_state_retained_energy_fraction=float(
                        validation[
                            "validation_state_retained_energy_fraction"
                        ]
                    ),
                    validation_state_relative_rmse=float(
                        validation["validation_state_relative_rmse"]
                    ),
                    minimum_validation_feature_retention=float(
                        validation[
                            "minimum_validation_feature_retention"
                        ]
                    ),
                    mean_validation_feature_retention=float(
                        validation[
                            "mean_validation_feature_retention"
                        ]
                    ),
                    validation_blocks=validation["validation_blocks"],
                    report_sha256=_sha(body),
                )
            )
        candidate_reports = {
            width: {
                "feasible_at_predictive_energy_floor": all(
                    result.supports_width(width) for result in results
                ),
                "failing_layer_indices": [
                    result.layer_index
                    for result in results
                    if not result.supports_width(width)
                ],
                "residual_width_compression_ratio": (
                    self.source_width / width
                ),
                "parameter_compaction_inferred": False,
            }
            for width in requested
        }
        body = {
            "schema": LFM2_CPCF_RANK_AUDIT_SCHEMA,
            "cache_key": self.cache_key,
            "layer_topology_sha256": self.layer_topology_sha256,
            "split_sha256": self.split.split_sha256,
            "config": self.config.as_dict(),
            "boundary_reports": [
                result.report_sha256 for result in results
            ],
            "candidate_widths": {
                str(width): report
                for width, report in candidate_reports.items()
            },
            "missing_cache_fields_for_full_compilation": list(
                self.missing_cache_fields_for_full_compilation
            ),
            "external_requirements_for_full_compilation": list(
                self.external_requirements_for_full_compilation
            ),
            "operation_audit": {
                "parent_forward_count": 0,
                "parent_generation_count": 0,
                "parent_cache_regeneration_count": 0,
                "training_step_count": 0,
                "backward_call_count": 0,
                "optimizer_step_count": 0,
            },
            "parameter_compaction_claimed": False,
            "measurement_only": True,
            "installability_contract_evaluated": False,
            "parent_side_compiler_inputs_ready": False,
            "scope": "residual_width_predictive_rank_feasibility_only",
        }
        # Verify again after all streamed reads.  The production cache loader
        # also verifies every loaded trace/record, but this final check binds
        # the report to an unchanged manifest/completion pair.
        self.cache.verify()
        if getattr(self.cache, "cache_key", None) != self.cache_key:
            raise Lfm2CpcfCachedEvidenceError(
                "Parent cache identity changed during CPCF evidence extraction."
            )
        return Lfm2CpcfPredictiveRankAudit(
            cache_key=self.cache_key,
            layer_topology_sha256=self.layer_topology_sha256,
            split=self.split,
            config=self.config,
            boundaries=tuple(results),
            candidate_widths=candidate_reports,
            missing_cache_fields_for_full_compilation=(
                self.missing_cache_fields_for_full_compilation
            ),
            external_requirements_for_full_compilation=(
                self.external_requirements_for_full_compilation
            ),
            report_sha256=_sha(body),
        )


def build_lfm2_cpcf_boundary_evidence(
    cache: Any,
    *,
    layer_index: int,
    config: Lfm2CpcfEvidenceConfig | None = None,
) -> Lfm2CpcfBoundaryEvidence:
    """Adapt one boundary from a sealed cache without a parent execution."""

    return Lfm2CpcfCachedEvidenceAdapter(
        cache,
        config=config,
    ).boundary(layer_index)


def audit_lfm2_cpcf_predictive_rank(
    cache: Any,
    *,
    config: Lfm2CpcfEvidenceConfig | None = None,
    candidate_widths: Sequence[int] = (),
) -> Lfm2CpcfPredictiveRankAudit:
    """Audit all LFM2 boundaries from sealed cached evidence only."""

    return Lfm2CpcfCachedEvidenceAdapter(
        cache,
        config=config,
    ).audit(candidate_widths=candidate_widths)


__all__ = [
    "LFM2_CPCF_BOUNDARY_SCHEMA",
    "LFM2_CPCF_CACHE_ADAPTER_SCHEMA",
    "LFM2_CPCF_RANK_AUDIT_SCHEMA",
    "LFM2_CPCF_SPLIT_SCHEMA",
    "Lfm2CpcfBoundaryEvidence",
    "Lfm2CpcfBoundaryRankResult",
    "Lfm2CpcfCacheSplit",
    "Lfm2CpcfCachedEvidenceAdapter",
    "Lfm2CpcfCachedEvidenceError",
    "Lfm2CpcfEvidenceConfig",
    "Lfm2CpcfPredictiveRankAudit",
    "audit_lfm2_cpcf_predictive_rank",
    "build_lfm2_cpcf_boundary_evidence",
]
