"""Pure zero-gradient closed-loop folding transactions for Transformers LFM2.

This module deliberately does not run models, collect rollouts, or train
parameters.  It consumes already aligned circuit evidence and constructs one
complete fixed-shape layer transaction:

* full GQA groups are merged through real source medoids;
* complete SwiGLU neurons are merged through one shared gate/up assignment;
* the attention ``out_proj`` and MLP ``w2`` are fitted from *child* circuit
  designs to endpoint error-feedback targets;
* LFM2 short-convolution operators are copied exactly for this first lane; and
* every target state tensor is overwritten, making installation independent of
  the target's nominal blank initialization seed.

Endpoint error feedback is the important distinction from local tensor
matching.  For either residual branch the fitted target is::

    parent_branch_output - child_pre_branch_residual

Consequently the fitted child branch is asked to close the current residual
state error, not to numerically resemble a parent weight matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_activation_cluster import (
    GatedMLPActivationClusterPlan,
    GQAActivationClusterPlan,
    build_gated_mlp_activation_cluster_plan,
    build_gqa_activation_cluster_plan,
)
from neural_forge.hf_closed_form_fit import (
    FitGate,
    PriorLinearFitResult,
    fit_ridge_linear,
    gate_fit,
    measure_fit,
)
from neural_forge.hf_lfm2_canonical import (
    Lfm2AttentionContract,
    Lfm2LayerContract,
    Lfm2ShortConvChannelPlan,
    Lfm2ShortConvContract,
    build_lfm2_short_conv_channel_plan,
    inspect_lfm2_layer,
)


_SCHEMA = "neural-forge-lfm2-closed-loop-fold-v1"
_TRANSACTION_SCHEMA = "neural-forge-lfm2-closed-loop-fold-transaction-v1"
_ATTENTION_FOLDED_KEYS = frozenset(
    {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.out_proj.weight",
    }
)
_MLP_FOLDED_KEYS = frozenset(
    {
        "feed_forward.w1.weight",
        "feed_forward.w3.weight",
        "feed_forward.w2.weight",
    }
)


class Lfm2ClosedLoopFoldError(ValueError):
    """Base failure for an LFM2 closed-loop fold."""


class Lfm2ClosedLoopTopologyError(Lfm2ClosedLoopFoldError):
    """The source and target cannot prove a supported LFM2 transaction."""


class Lfm2ClosedLoopEvidenceError(Lfm2ClosedLoopFoldError):
    """Circuit evidence is missing, non-finite, or topologically inconsistent."""


class Lfm2ClosedLoopFitRejected(Lfm2ClosedLoopFoldError):
    """A closed-form endpoint fit failed its declared local gate."""


class Lfm2ClosedLoopInvariantError(RuntimeError):
    """A supposedly pure build, installation, or rollback changed forbidden state."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for LFM2 closed-loop folding.") from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Lfm2ClosedLoopFoldError(f"{name} must be a positive integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2ClosedLoopFoldError(f"{name} must be a positive integer.") from exc
    if result <= 0:
        raise Lfm2ClosedLoopFoldError(f"{name} must be a positive integer.")
    return int(result)


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2ClosedLoopFoldError(f"{name} must be a finite non-negative number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2ClosedLoopFoldError(
            f"{name} must be a finite non-negative number."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2ClosedLoopFoldError(f"{name} must be a finite non-negative number.")
    return result


def _cosine_threshold(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2ClosedLoopFoldError(f"{name} must be finite and in [-1, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2ClosedLoopFoldError(f"{name} must be finite and in [-1, 1].") from exc
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise Lfm2ClosedLoopFoldError(f"{name} must be finite and in [-1, 1].")
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise Lfm2ClosedLoopInvariantError("Cannot fingerprint a non-tensor value.")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_tensor_sha256(values[name])))
    return digest.hexdigest()


def _state_items(module: Any) -> tuple[tuple[str, Any], ...]:
    torch = _torch()
    state = module.state_dict()
    result = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise Lfm2ClosedLoopInvariantError(
                f"State value {name!r} must be a materialized tensor."
            )
        result.append((name, value.detach().to(device="cpu").contiguous().clone()))
    if not result:
        raise Lfm2ClosedLoopInvariantError("An LFM2 layer cannot have empty state.")
    return tuple(result)


def _state_mapping(items: tuple[tuple[str, Any], ...]) -> Mapping[str, Any]:
    return MappingProxyType({name: value for name, value in items})


def _state_spec(
    values: Mapping[str, Any],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (name, tuple(int(size) for size in values[name].shape), str(values[name].dtype))
        for name in sorted(values)
    )


def _gradient_signature(
    module: Any,
) -> tuple[tuple[str, int, bool, int | None, str | None], ...]:
    result = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        result.append(
            (
                name,
                id(parameter),
                bool(parameter.requires_grad),
                None if gradient is None else id(gradient),
                None if gradient is None else _tensor_sha256(gradient),
            )
        )
    return tuple(result)


def _parameter_ids(module: Any) -> tuple[tuple[str, int], ...]:
    return tuple((name, id(parameter)) for name, parameter in module.named_parameters())


def _clone_for_target(source: Any, target: Any, name: str) -> Any:
    torch = _torch()
    if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise Lfm2ClosedLoopTopologyError(f"{name} must be a tensor in both models.")
    if tuple(source.shape) != tuple(target.shape):
        raise Lfm2ClosedLoopTopologyError(
            f"{name} exact-copy shapes differ: {tuple(source.shape)} != "
            f"{tuple(target.shape)}."
        )
    if source.dtype != target.dtype:
        raise Lfm2ClosedLoopTopologyError(
            f"{name} exact-copy dtypes differ: {source.dtype} != {target.dtype}."
        )
    return source.detach().to(device="cpu").contiguous().clone()


def _candidate_tensor(value: Any, target: Any, name: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise Lfm2ClosedLoopInvariantError(f"{name} candidate must be a tensor.")
    if tuple(value.shape) != tuple(target.shape):
        raise Lfm2ClosedLoopTopologyError(
            f"{name} candidate shape {tuple(value.shape)} does not match target "
            f"{tuple(target.shape)}."
        )
    converted = value.detach().to(device="cpu", dtype=target.dtype).contiguous().clone()
    if converted.is_floating_point() and not bool(
        torch.isfinite(converted).all().item()
    ):
        raise Lfm2ClosedLoopInvariantError(f"{name} candidate is non-finite.")
    return converted


def _float64_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2ClosedLoopEvidenceError(
            f"{name} must be a non-empty real floating rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2ClosedLoopEvidenceError(f"{name} contains NaN or infinity.")
    return result


def _optional_validation_triplet(
    design: Any | None,
    parent_output: Any | None,
    child_residual: Any | None,
    *,
    design_name: str,
    hidden_size: int,
    design_ndim: int,
) -> tuple[Any | None, Any | None]:
    supplied = (
        design is not None,
        parent_output is not None,
        child_residual is not None,
    )
    if any(supplied) and not all(supplied):
        raise Lfm2ClosedLoopEvidenceError(
            "Validation child design, parent branch output, and child pre-branch "
            "residual must be supplied together."
        )
    if not any(supplied):
        return None, None
    validation_design = _float64_tensor(
        design,
        f"validation_{design_name}",
        ndim=design_ndim,
    )
    validation_parent = _float64_tensor(
        parent_output,
        "validation_parent_branch_output_rows",
        ndim=2,
    )
    validation_residual = _float64_tensor(
        child_residual,
        "validation_child_pre_branch_residual_rows",
        ndim=2,
    )
    if validation_parent.shape != validation_residual.shape:
        raise Lfm2ClosedLoopEvidenceError(
            "Validation parent output and child residual shapes differ."
        )
    if int(validation_parent.shape[1]) != hidden_size:
        raise Lfm2ClosedLoopEvidenceError(
            "Validation endpoint target width does not match hidden_size."
        )
    if int(validation_design.shape[0]) != int(validation_parent.shape[0]):
        raise Lfm2ClosedLoopEvidenceError(
            "Validation child design and endpoint target row counts differ."
        )
    return validation_design, validation_parent - validation_residual


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopFoldConfig:
    """Deterministic planning and endpoint-fit policy."""

    endpoint_ridge: float = 1e-6
    cluster_ridge: float = 1e-8
    cluster_max_iterations: int = 16
    cluster_distance_chunk_size: int = 512
    max_endpoint_relative_rmse: float = 1.0
    min_endpoint_cosine_similarity: float = 0.0
    max_endpoint_prior_delta_ratio: float | None = None
    max_primal_features: int = 1024
    max_dual_rows: int = 1024
    max_correction_anchors: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "endpoint_ridge",
            _nonnegative_float(self.endpoint_ridge, "endpoint_ridge"),
        )
        object.__setattr__(
            self,
            "cluster_ridge",
            _nonnegative_float(self.cluster_ridge, "cluster_ridge"),
        )
        object.__setattr__(
            self,
            "cluster_max_iterations",
            _positive_int(self.cluster_max_iterations, "cluster_max_iterations"),
        )
        object.__setattr__(
            self,
            "cluster_distance_chunk_size",
            _positive_int(
                self.cluster_distance_chunk_size,
                "cluster_distance_chunk_size",
            ),
        )
        object.__setattr__(
            self,
            "max_endpoint_relative_rmse",
            _nonnegative_float(
                self.max_endpoint_relative_rmse,
                "max_endpoint_relative_rmse",
            ),
        )
        object.__setattr__(
            self,
            "min_endpoint_cosine_similarity",
            _cosine_threshold(
                self.min_endpoint_cosine_similarity,
                "min_endpoint_cosine_similarity",
            ),
        )
        if self.max_endpoint_prior_delta_ratio is not None:
            object.__setattr__(
                self,
                "max_endpoint_prior_delta_ratio",
                _nonnegative_float(
                    self.max_endpoint_prior_delta_ratio,
                    "max_endpoint_prior_delta_ratio",
                ),
            )
        object.__setattr__(
            self,
            "max_primal_features",
            _positive_int(self.max_primal_features, "max_primal_features"),
        )
        object.__setattr__(
            self,
            "max_dual_rows",
            _positive_int(self.max_dual_rows, "max_dual_rows"),
        )
        object.__setattr__(
            self,
            "max_correction_anchors",
            _positive_int(
                self.max_correction_anchors,
                "max_correction_anchors",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint_ridge": self.endpoint_ridge,
            "cluster_ridge": self.cluster_ridge,
            "cluster_max_iterations": self.cluster_max_iterations,
            "cluster_distance_chunk_size": self.cluster_distance_chunk_size,
            "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
            "min_endpoint_cosine_similarity": (self.min_endpoint_cosine_similarity),
            "max_endpoint_prior_delta_ratio": (self.max_endpoint_prior_delta_ratio),
            "max_primal_features": self.max_primal_features,
            "max_dual_rows": self.max_dual_rows,
            "max_correction_anchors": self.max_correction_anchors,
        }


@dataclass(frozen=True, slots=True)
class Lfm2AttentionFoldEvidence:
    """Rows used to select real GQA groups and fit the child attention endpoint.

    Parent Q/K/V/context tensors are row-major:

    * query/context: ``[rows, source_query_heads, head_dim]``
    * key/value: ``[rows, source_key_value_heads, head_dim]``

    ``child_context_rows`` must already be ordered in the target head order
    selected by the merge plan and has shape
    ``[fit_rows, target_query_heads, head_dim]``.
    """

    parent_query_rows: Any
    parent_key_rows: Any
    parent_value_rows: Any
    parent_context_rows: Any
    child_context_rows: Any
    parent_branch_output_rows: Any
    child_pre_branch_residual_rows: Any
    parent_contribution_rows: Any | None = None
    validation_child_context_rows: Any | None = None
    validation_parent_branch_output_rows: Any | None = None
    validation_child_pre_branch_residual_rows: Any | None = None


@dataclass(frozen=True, slots=True)
class Lfm2MLPFoldEvidence:
    """Rows used to merge real SwiGLU neurons and fit the child MLP endpoint."""

    parent_gate_rows: Any
    parent_up_rows: Any
    parent_post_swiglu_rows: Any
    child_post_swiglu_rows: Any
    parent_branch_output_rows: Any
    child_pre_branch_residual_rows: Any
    parent_contribution_rows: Any | None = None
    validation_child_post_swiglu_rows: Any | None = None
    validation_parent_branch_output_rows: Any | None = None
    validation_child_pre_branch_residual_rows: Any | None = None


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopFoldEvidence:
    """Complete trace evidence for one LFM2 layer transaction."""

    mlp: Lfm2MLPFoldEvidence
    attention: Lfm2AttentionFoldEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mlp, Lfm2MLPFoldEvidence):
            raise TypeError("mlp must be Lfm2MLPFoldEvidence.")
        if self.attention is not None and not isinstance(
            self.attention, Lfm2AttentionFoldEvidence
        ):
            raise TypeError("attention must be Lfm2AttentionFoldEvidence or None.")


def _metrics_dict(value: Any) -> dict[str, Any]:
    return {
        "rows": value.rows,
        "output_width": value.output_width,
        "rmse": value.rmse,
        "target_rms": value.target_rms,
        "relative_rmse": value.relative_rmse,
        "mean_cosine_similarity": value.mean_cosine_similarity,
        "finite": value.finite,
    }


@dataclass(frozen=True, slots=True)
class Lfm2EndpointFit:
    """One child-design-to-parent-endpoint closed-form fit."""

    branch: str
    fit: PriorLinearFitResult
    gate: FitGate
    solver: str
    correction_rank_bound: int
    correction_anchor_indices: tuple[int, ...]
    design_sha256: str
    endpoint_target_sha256: str
    validation_design_sha256: str | None
    validation_endpoint_target_sha256: str | None

    @property
    def accepted(self) -> bool:
        return self.gate.accepted

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "objective": (
                "child_branch_design_to_parent_branch_output_minus_"
                "child_pre_branch_residual"
            ),
            "endpoint_error_feedback": True,
            "fit_bias": False,
            "ridge": self.fit.ridge,
            "effective_ridge": self.fit.effective_ridge,
            "prior_norm": self.fit.prior_norm,
            "prior_delta_norm": self.fit.prior_delta_norm,
            "solver": self.solver,
            "correction_rank_bound": self.correction_rank_bound,
            "correction_anchor_count": len(self.correction_anchor_indices),
            "correction_anchor_indices": list(self.correction_anchor_indices),
            "train_metrics": _metrics_dict(self.fit.train_metrics),
            "validation_metrics": (
                None
                if self.fit.validation_metrics is None
                else _metrics_dict(self.fit.validation_metrics)
            ),
            "gate": {
                "accepted": self.gate.accepted,
                "reasons": list(self.gate.reasons),
                "metrics": _metrics_dict(self.gate.metrics),
            },
            "design_sha256": self.design_sha256,
            "endpoint_target_sha256": self.endpoint_target_sha256,
            "validation_design_sha256": self.validation_design_sha256,
            "validation_endpoint_target_sha256": (
                self.validation_endpoint_target_sha256
            ),
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2AttentionGroupMerge:
    """Complete real-medoid GQA producer merge plus fitted output consumer."""

    cluster_plan: GQAActivationClusterPlan
    query_weight: Any
    key_weight: Any
    value_weight: Any
    output_weight: Any
    endpoint_fit: Lfm2EndpointFit

    @property
    def source_query_heads(self) -> int:
        return self.cluster_plan.source_query_heads

    @property
    def source_key_value_heads(self) -> int:
        return self.cluster_plan.source_key_value_heads

    @property
    def target_query_heads(self) -> int:
        return self.cluster_plan.target_query_heads

    @property
    def target_key_value_heads(self) -> int:
        return self.cluster_plan.target_key_value_heads

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "coupled-gqa-group-merge",
            "source_query_heads": self.source_query_heads,
            "source_key_value_heads": self.source_key_value_heads,
            "target_query_heads": self.target_query_heads,
            "target_key_value_heads": self.target_key_value_heads,
            "head_dim": self.cluster_plan.head_dim,
            "source_query_heads_per_group": (
                self.cluster_plan.source_query_heads_per_group
            ),
            "target_query_heads_per_group": (
                self.cluster_plan.target_query_heads_per_group
            ),
            "group_medoids": list(self.cluster_plan.group_medoids),
            "group_assignment": list(self.cluster_plan.group_assignment),
            "query_head_medoids": list(self.cluster_plan.query_head_medoids),
            "query_head_assignment": list(self.cluster_plan.query_head_assignment),
            "full_source_coverage": self.cluster_plan.report.full_source_coverage,
            "complete_real_qkv_producers": True,
            "qkv_group_coupled": True,
            "producer_averaging": False,
            "endpoint_fit": self.endpoint_fit.as_dict(),
            "query_weight_sha256": _tensor_sha256(self.query_weight),
            "key_weight_sha256": _tensor_sha256(self.key_weight),
            "value_weight_sha256": _tensor_sha256(self.value_weight),
            "output_weight_sha256": _tensor_sha256(self.output_weight),
        }


@dataclass(frozen=True, slots=True)
class Lfm2SwiGLUNeuronMerge:
    """Complete real-medoid SwiGLU producer merge plus fitted ``w2``."""

    cluster_plan: GatedMLPActivationClusterPlan
    gate_weight: Any
    up_weight: Any
    down_weight: Any
    endpoint_fit: Lfm2EndpointFit

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "coupled-swiglu-neuron-merge",
            "source_neurons": self.cluster_plan.source_neurons,
            "target_neurons": self.cluster_plan.target_neurons,
            "medoid_source_neurons": list(self.cluster_plan.medoid_source_neurons),
            "source_to_target": list(self.cluster_plan.source_to_target),
            "group_sizes": list(self.cluster_plan.report.group_sizes),
            "full_source_coverage": self.cluster_plan.report.full_source_coverage,
            "real_source_gate_up_producers": True,
            "gate_up_indices_coupled": True,
            "producer_averaging": False,
            "endpoint_fit": self.endpoint_fit.as_dict(),
            "gate_weight_sha256": _tensor_sha256(self.gate_weight),
            "up_weight_sha256": _tensor_sha256(self.up_weight),
            "down_weight_sha256": _tensor_sha256(self.down_weight),
        }


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopFoldPlan:
    """A target-seed-independent complete state replacement for one LFM2 layer."""

    source_contract: Lfm2LayerContract
    target_contract: Lfm2LayerContract
    config: Lfm2ClosedLoopFoldConfig
    attention: Lfm2AttentionGroupMerge | None
    mlp: Lfm2SwiGLUNeuronMerge
    short_conv_plan: Lfm2ShortConvChannelPlan | None
    candidate_state_items: tuple[tuple[str, Any], ...]
    target_state_spec: tuple[tuple[str, tuple[int, ...], str], ...]
    source_state_sha256: str
    target_state_sha256_before: str
    candidate_state_sha256: str
    plan_sha256: str

    @property
    def zero_gradient(self) -> bool:
        return True

    @property
    def seed_erasure_by_construction(self) -> bool:
        return True

    @property
    def short_conv_preserved_exactly(self) -> bool:
        return self.target_contract.kind != "short_conv" or (
            self.short_conv_plan is not None
            and self.short_conv_plan.exactness.algebraically_exact
            and self.short_conv_plan.target_hidden_size
            == self.short_conv_plan.source_contract.hidden_size
        )

    def candidate_state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in self.candidate_state_items
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "source_contract": self.source_contract.as_dict(),
            "target_contract": self.target_contract.as_dict(),
            "config": self.config.as_dict(),
            "attention": None if self.attention is None else self.attention.as_dict(),
            "mlp": self.mlp.as_dict(),
            "short_conv": (
                None
                if self.short_conv_plan is None
                else {
                    "preserved_exactly": self.short_conv_preserved_exactly,
                    "plan": self.short_conv_plan.as_dict(),
                }
            ),
            "candidate_state_keys": [
                name for name, _value in self.candidate_state_items
            ],
            "complete_target_state_coverage": True,
            "source_state_sha256": self.source_state_sha256,
            "target_state_sha256_before": self.target_state_sha256_before,
            "candidate_state_sha256": self.candidate_state_sha256,
            "plan_sha256": self.plan_sha256,
            "seed_erasure_by_construction": self.seed_erasure_by_construction,
            "short_conv_preserved_exactly": self.short_conv_preserved_exactly,
            "zero_gradient": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_steps": 0,
            "parameter_gradients_written": False,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LayerSnapshot:
    """Exact state and runtime identity snapshot for rollback."""

    contract: Lfm2LayerContract
    state_items: tuple[tuple[str, Any], ...]
    state_spec: tuple[tuple[str, tuple[int, ...], str], ...]
    state_sha256: str
    parameter_ids: tuple[tuple[str, int], ...]
    gradient_signature: tuple[tuple[str, int, bool, int | None, str | None], ...]

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in self.state_items
            }
        )

    def restore(self, target_layer: Any) -> bool:
        return restore_lfm2_layer_snapshot(target_layer, self)


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopFoldTransactionReport:
    """Commit or rollback result for one installed plan."""

    accepted: bool
    rollback_exact: bool | None
    candidate_state_sha256: str
    final_state_sha256: str
    parameter_identities_preserved: bool
    gradient_state_preserved: bool
    seed_erasure_by_construction: bool
    short_conv_preserved_exactly: bool
    plan_sha256: str

    @property
    def zero_gradient(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _TRANSACTION_SCHEMA,
            "accepted": self.accepted,
            "rollback_exact": self.rollback_exact,
            "candidate_state_sha256": self.candidate_state_sha256,
            "final_state_sha256": self.final_state_sha256,
            "parameter_identities_preserved": self.parameter_identities_preserved,
            "gradient_state_preserved": self.gradient_state_preserved,
            "seed_erasure_by_construction": self.seed_erasure_by_construction,
            "short_conv_preserved_exactly": self.short_conv_preserved_exactly,
            "plan_sha256": self.plan_sha256,
            "zero_gradient": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_steps": 0,
            "parameter_gradients_written": False,
        }


class Lfm2ClosedLoopFoldTransaction:
    """An installed candidate that must be explicitly committed or rolled back."""

    def __init__(
        self,
        target_layer: Any,
        plan: Lfm2ClosedLoopFoldPlan,
        snapshot: Lfm2LayerSnapshot,
    ) -> None:
        self._target_layer = target_layer
        self.plan = plan
        self.snapshot = snapshot
        self._active = True
        self._report: Lfm2ClosedLoopFoldTransactionReport | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def report(self) -> Lfm2ClosedLoopFoldTransactionReport | None:
        return self._report

    def _runtime_flags(self) -> tuple[bool, bool]:
        parameter_ids_preserved = (
            _parameter_ids(self._target_layer) == self.snapshot.parameter_ids
        )
        gradients_preserved = (
            _gradient_signature(self._target_layer) == self.snapshot.gradient_signature
        )
        return parameter_ids_preserved, gradients_preserved

    def commit(self) -> Lfm2ClosedLoopFoldTransactionReport:
        if not self._active:
            raise Lfm2ClosedLoopInvariantError(
                "The LFM2 transaction is no longer active."
            )
        final_state = _state_mapping(_state_items(self._target_layer))
        final_sha = _state_sha256(final_state)
        if final_sha != self.plan.candidate_state_sha256:
            raise Lfm2ClosedLoopInvariantError(
                "The installed LFM2 candidate changed before commit."
            )
        parameter_ids_preserved, gradients_preserved = self._runtime_flags()
        if not parameter_ids_preserved or not gradients_preserved:
            raise Lfm2ClosedLoopInvariantError(
                "The LFM2 transaction changed parameter identities or gradients."
            )
        self._active = False
        self._report = Lfm2ClosedLoopFoldTransactionReport(
            accepted=True,
            rollback_exact=None,
            candidate_state_sha256=self.plan.candidate_state_sha256,
            final_state_sha256=final_sha,
            parameter_identities_preserved=True,
            gradient_state_preserved=True,
            seed_erasure_by_construction=self.plan.seed_erasure_by_construction,
            short_conv_preserved_exactly=self.plan.short_conv_preserved_exactly,
            plan_sha256=self.plan.plan_sha256,
        )
        return self._report

    def rollback(self) -> Lfm2ClosedLoopFoldTransactionReport:
        if not self._active:
            raise Lfm2ClosedLoopInvariantError(
                "The LFM2 transaction is no longer active."
            )
        rollback_exact = self.snapshot.restore(self._target_layer)
        parameter_ids_preserved, gradients_preserved = self._runtime_flags()
        final_state = _state_mapping(_state_items(self._target_layer))
        final_sha = _state_sha256(final_state)
        rollback_exact = bool(
            rollback_exact
            and final_sha == self.snapshot.state_sha256
            and parameter_ids_preserved
            and gradients_preserved
        )
        if not rollback_exact:
            raise Lfm2ClosedLoopInvariantError(
                "The rejected LFM2 candidate was not restored exactly."
            )
        self._active = False
        self._report = Lfm2ClosedLoopFoldTransactionReport(
            accepted=False,
            rollback_exact=True,
            candidate_state_sha256=self.plan.candidate_state_sha256,
            final_state_sha256=final_sha,
            parameter_identities_preserved=True,
            gradient_state_preserved=True,
            seed_erasure_by_construction=self.plan.seed_erasure_by_construction,
            short_conv_preserved_exactly=self.plan.short_conv_preserved_exactly,
            plan_sha256=self.plan.plan_sha256,
        )
        return self._report

    def __enter__(self) -> "Lfm2ClosedLoopFoldTransaction":
        if not self._active:
            raise Lfm2ClosedLoopInvariantError(
                "Cannot enter an inactive LFM2 transaction."
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._active:
            self.rollback()
        return False


def _validate_layer_pair(
    source: Lfm2LayerContract,
    target: Lfm2LayerContract,
    evidence: Lfm2ClosedLoopFoldEvidence,
) -> None:
    if source.kind != target.kind:
        raise Lfm2ClosedLoopTopologyError(
            "Source and target LFM2 layers must have the same operator kind."
        )
    if source.hidden_size != target.hidden_size:
        raise Lfm2ClosedLoopTopologyError(
            "This first LFM2 transaction preserves hidden_size exactly."
        )
    if target.ffn_intermediate_size > source.ffn_intermediate_size:
        raise Lfm2ClosedLoopTopologyError(
            "The target SwiGLU width cannot exceed the source width."
        )
    if source.kind == "full_attention":
        if evidence.attention is None:
            raise Lfm2ClosedLoopEvidenceError(
                "Attention evidence is required for a full-attention LFM2 layer."
            )
        if not isinstance(source.operator, Lfm2AttentionContract) or not isinstance(
            target.operator, Lfm2AttentionContract
        ):
            raise Lfm2ClosedLoopTopologyError(
                "Internal LFM2 attention contract mismatch."
            )
        source_attention = source.operator
        target_attention = target.operator
        if source_attention.head_dim != target_attention.head_dim:
            raise Lfm2ClosedLoopTopologyError(
                "Attention group folding must preserve runtime operator.head_dim."
            )
        source_ratio = source_attention.query_heads // source_attention.key_value_heads
        target_ratio = target_attention.query_heads // target_attention.key_value_heads
        if source_ratio != target_ratio:
            raise Lfm2ClosedLoopTopologyError(
                "This first attention transaction merges complete GQA groups and "
                "therefore preserves query heads per KV group."
            )
        if (
            target_attention.query_heads > source_attention.query_heads
            or target_attention.key_value_heads > source_attention.key_value_heads
        ):
            raise Lfm2ClosedLoopTopologyError(
                "The target attention topology cannot expand source head counts."
            )
    elif evidence.attention is not None:
        raise Lfm2ClosedLoopEvidenceError(
            "Short-conv LFM2 layers must not receive attention evidence."
        )


def _attention_contribution_rows(
    parent_context: Any,
    source_output_weight: Any,
    source_query_heads: int,
    head_dim: int,
) -> Any:
    torch = _torch()
    hidden_size = int(source_output_weight.shape[0])
    output_blocks = (
        source_output_weight.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(hidden_size, source_query_heads, head_dim)
    )
    return torch.einsum("rqd,hqd->rqh", parent_context, output_blocks)


def _fit_endpoint(
    *,
    branch: str,
    design: Any,
    parent_output: Any,
    child_residual: Any,
    prior_weight: Any,
    validation_design: Any | None,
    validation_target: Any | None,
    config: Lfm2ClosedLoopFoldConfig,
) -> Lfm2EndpointFit:
    if parent_output.shape != child_residual.shape:
        raise Lfm2ClosedLoopEvidenceError(
            f"{branch} parent output and child pre-branch residual shapes differ."
        )
    if int(design.shape[0]) != int(parent_output.shape[0]):
        raise Lfm2ClosedLoopEvidenceError(
            f"{branch} child design and endpoint target row counts differ."
        )
    endpoint_target = parent_output - child_residual
    fit, solver, rank_bound, anchors = _fit_prior_residual_correction(
        design,
        endpoint_target,
        prior_weight,
        validation_design=validation_design,
        validation_target=validation_target,
        config=config,
    )
    selected_metrics = fit.validation_metrics or fit.train_metrics
    decision = gate_fit(
        selected_metrics,
        max_relative_rmse=config.max_endpoint_relative_rmse,
        min_mean_cosine_similarity=config.min_endpoint_cosine_similarity,
    )
    result = Lfm2EndpointFit(
        branch=branch,
        fit=fit,
        gate=decision,
        solver=solver,
        correction_rank_bound=rank_bound,
        correction_anchor_indices=anchors,
        design_sha256=_tensor_sha256(design),
        endpoint_target_sha256=_tensor_sha256(endpoint_target),
        validation_design_sha256=(
            None if validation_design is None else _tensor_sha256(validation_design)
        ),
        validation_endpoint_target_sha256=(
            None if validation_target is None else _tensor_sha256(validation_target)
        ),
    )
    if not result.accepted:
        raise Lfm2ClosedLoopFitRejected(
            f"{branch} endpoint fit was rejected: {', '.join(decision.reasons)}"
        )
    return result


def _stable_linear_solve(system: Any, right: Any) -> Any:
    """Solve one bounded FP64 system with a deterministic least-squares fallback."""

    torch = _torch()
    try:
        solution = torch.linalg.solve(system, right)
    except RuntimeError:
        solution = torch.linalg.lstsq(system, right, driver="gelsd").solution
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2ClosedLoopInvariantError(
            "Endpoint residual correction produced non-finite coefficients."
        )
    return solution


def _correction_anchor_indices(
    design: Any,
    residual_target: Any,
    count: int,
) -> tuple[int, ...]:
    """Choose survivor columns with the largest residual-output correlation."""

    torch = _torch()
    features = int(design.shape[1])
    count = min(_positive_int(count, "correction anchor count"), features)
    # The correlation matrix is O(features * output_width), not O(features^2).
    correlation = design.transpose(0, 1).matmul(residual_target)
    scores = correlation.square().sum(dim=1)
    if not bool(torch.isfinite(scores).all().item()):
        raise Lfm2ClosedLoopInvariantError(
            "Endpoint correction anchor scores are non-finite."
        )
    ordered = sorted(
        range(features),
        key=lambda index: (-float(scores[index].item()), index),
    )
    return tuple(sorted(ordered[:count]))


def _fit_prior_residual_correction(
    design: Any,
    target: Any,
    prior_weight: Any,
    *,
    validation_design: Any | None,
    validation_target: Any | None,
    config: Lfm2ClosedLoopFoldConfig,
) -> tuple[PriorLinearFitResult, str, int, tuple[int, ...]]:
    """Fit endpoint error around a full-coverage cluster prior without huge Grams."""

    torch = _torch()
    x = _float64_tensor(design, "endpoint_design", ndim=2)
    y = _float64_tensor(target, "endpoint_target", ndim=2)
    prior = _float64_tensor(prior_weight, "endpoint_prior_weight", ndim=2)
    rows, features = (int(x.shape[0]), int(x.shape[1]))
    outputs = int(y.shape[1])
    if int(y.shape[0]) != rows:
        raise Lfm2ClosedLoopEvidenceError(
            "Endpoint design and target row counts differ."
        )
    if tuple(prior.shape) != (outputs, features):
        raise Lfm2ClosedLoopEvidenceError(
            "Endpoint prior weight does not match target/design widths."
        )
    residual_target = y - x.matmul(prior.transpose(0, 1))
    correction: Any
    anchors: tuple[int, ...] = ()
    ridge = config.endpoint_ridge
    maximum_ratio = config.max_endpoint_prior_delta_ratio
    if maximum_ratio == 0.0:
        correction = torch.zeros_like(prior)
        solver = "cluster-prior-without-endpoint-solve"
        rank_bound = 0
    elif features <= config.max_primal_features:
        correction_fit = fit_ridge_linear(
            x,
            residual_target,
            ridge=ridge,
            fit_bias=False,
        )
        correction = correction_fit.weight
        solver = "full-primal-residual-ridge"
        rank_bound = min(rows, features)
    elif rows < features and rows <= config.max_dual_rows:
        system = x.matmul(x.transpose(0, 1))
        if ridge > 0.0:
            system = system + ridge * torch.eye(rows, dtype=torch.float64)
        dual = _stable_linear_solve(system, residual_target)
        correction = dual.transpose(0, 1).matmul(x)
        solver = "full-dual-residual-ridge"
        rank_bound = rows
    else:
        anchors = _correction_anchor_indices(
            x,
            residual_target,
            config.max_correction_anchors,
        )
        anchor_index = torch.tensor(anchors, dtype=torch.long)
        anchor_design = x.index_select(1, anchor_index)
        anchor_fit = fit_ridge_linear(
            anchor_design,
            residual_target,
            ridge=ridge,
            fit_bias=False,
        )
        correction = torch.zeros_like(prior)
        correction.index_copy_(1, anchor_index, anchor_fit.weight)
        solver = "bounded-survivor-anchor-residual-ridge"
        rank_bound = min(rows, len(anchors))
    prior_norm = float(torch.linalg.vector_norm(prior).item())
    raw_delta_norm = float(torch.linalg.vector_norm(correction).item())
    update_scale = 1.0
    if maximum_ratio == 0.0:
        update_scale = 0.0
    elif maximum_ratio is not None and raw_delta_norm > 0.0:
        maximum_delta_norm = maximum_ratio * max(
            prior_norm,
            torch.finfo(torch.float64).tiny,
        )
        update_scale = min(1.0, maximum_delta_norm / raw_delta_norm)
        if update_scale < 1.0:
            correction = correction * update_scale
    weight = prior + correction
    train_metrics = measure_fit(x.matmul(weight.transpose(0, 1)), y)

    validation_metrics = None
    if (validation_design is None) != (validation_target is None):
        raise Lfm2ClosedLoopEvidenceError(
            "Validation endpoint design and target must be supplied together."
        )
    if validation_design is not None:
        validation_x = _float64_tensor(
            validation_design,
            "validation_endpoint_design",
            ndim=2,
        )
        validation_y = _float64_tensor(
            validation_target,
            "validation_endpoint_target",
            ndim=2,
        )
        if (
            int(validation_x.shape[1]) != features
            or int(validation_y.shape[1]) != outputs
            or int(validation_x.shape[0]) != int(validation_y.shape[0])
        ):
            raise Lfm2ClosedLoopEvidenceError(
                "Validation endpoint evidence does not match the fitted topology."
            )
        validation_metrics = measure_fit(
            validation_x.matmul(weight.transpose(0, 1)),
            validation_y,
        )
    delta_norm = float(torch.linalg.vector_norm(correction).item())
    trace_scale = float(x.square().sum().item()) / max(1, features)
    fit = PriorLinearFitResult(
        weight=weight.detach().contiguous(),
        bias=None,
        ridge=ridge,
        effective_ridge=ridge,
        trace_scale=trace_scale,
        prior_norm=prior_norm,
        prior_delta_norm=delta_norm,
        update_scale=update_scale,
        clipped=update_scale < 1.0,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
    )
    return fit, solver, rank_bound, anchors


def _build_attention_merge(
    source_layer: Any,
    target_layer: Any,
    source_contract: Lfm2LayerContract,
    target_contract: Lfm2LayerContract,
    evidence: Lfm2AttentionFoldEvidence,
    config: Lfm2ClosedLoopFoldConfig,
) -> Lfm2AttentionGroupMerge:
    source_attention = source_contract.operator
    target_attention = target_contract.operator
    if not isinstance(source_attention, Lfm2AttentionContract) or not isinstance(
        target_attention, Lfm2AttentionContract
    ):  # pragma: no cover - guarded by _validate_layer_pair
        raise Lfm2ClosedLoopTopologyError("LFM2 attention contract mismatch.")
    query = _float64_tensor(evidence.parent_query_rows, "parent_query_rows", ndim=3)
    key = _float64_tensor(evidence.parent_key_rows, "parent_key_rows", ndim=3)
    value = _float64_tensor(evidence.parent_value_rows, "parent_value_rows", ndim=3)
    context = _float64_tensor(
        evidence.parent_context_rows,
        "parent_context_rows",
        ndim=3,
    )
    expected_query = (source_attention.query_heads, source_attention.head_dim)
    expected_key_value = (
        source_attention.key_value_heads,
        source_attention.head_dim,
    )
    for name, tensor, expected in (
        ("parent_query_rows", query, expected_query),
        ("parent_context_rows", context, expected_query),
        ("parent_key_rows", key, expected_key_value),
        ("parent_value_rows", value, expected_key_value),
    ):
        if tuple(tensor.shape[1:]) != expected:
            raise Lfm2ClosedLoopEvidenceError(
                f"{name} must end with {expected}; received {tuple(tensor.shape[1:])}."
            )
    selection_rows = int(query.shape[0])
    if any(int(tensor.shape[0]) != selection_rows for tensor in (key, value, context)):
        raise Lfm2ClosedLoopEvidenceError(
            "Parent attention selection rows must be aligned."
        )
    source_output = source_layer.self_attn.out_proj.weight
    if evidence.parent_contribution_rows is None:
        contribution = _attention_contribution_rows(
            context,
            source_output,
            source_attention.query_heads,
            source_attention.head_dim,
        )
    else:
        contribution = _float64_tensor(
            evidence.parent_contribution_rows,
            "parent_contribution_rows",
            ndim=3,
        )
    if tuple(contribution.shape) != (
        selection_rows,
        source_attention.query_heads,
        source_contract.hidden_size,
    ):
        raise Lfm2ClosedLoopEvidenceError(
            "parent_contribution_rows must have shape "
            "[rows, source_query_heads, hidden_size]."
        )
    cluster = build_gqa_activation_cluster_plan(
        query.movedim(1, 0).contiguous(),
        key.movedim(1, 0).contiguous(),
        value.movedim(1, 0).contiguous(),
        context.movedim(1, 0).contiguous(),
        contribution.movedim(1, 0).contiguous(),
        source_query_heads=source_attention.query_heads,
        source_key_value_heads=source_attention.key_value_heads,
        target_query_heads=target_attention.query_heads,
        target_key_value_heads=target_attention.key_value_heads,
        source_head_dim=source_attention.head_dim,
        target_head_dim=target_attention.head_dim,
        ridge=config.cluster_ridge,
        max_iterations=config.cluster_max_iterations,
        distance_chunk_size=config.cluster_distance_chunk_size,
    )
    query_weight = cluster.select_query_producer(
        source_layer.self_attn.q_proj.weight,
        axis=0,
    )
    key_weight = cluster.select_key_producer(
        source_layer.self_attn.k_proj.weight,
        axis=0,
    )
    value_weight = cluster.select_value_producer(
        source_layer.self_attn.v_proj.weight,
        axis=0,
    )

    child_context = _float64_tensor(
        evidence.child_context_rows,
        "child_context_rows",
        ndim=3,
    )
    if tuple(child_context.shape[1:]) != (
        target_attention.query_heads,
        target_attention.head_dim,
    ):
        raise Lfm2ClosedLoopEvidenceError(
            "child_context_rows does not match target GQA topology."
        )
    parent_output = _float64_tensor(
        evidence.parent_branch_output_rows,
        "parent_branch_output_rows",
        ndim=2,
    )
    child_residual = _float64_tensor(
        evidence.child_pre_branch_residual_rows,
        "child_pre_branch_residual_rows",
        ndim=2,
    )
    if int(parent_output.shape[1]) != source_contract.hidden_size:
        raise Lfm2ClosedLoopEvidenceError(
            "Attention endpoint target width does not match hidden_size."
        )
    validation_context, validation_target = _optional_validation_triplet(
        evidence.validation_child_context_rows,
        evidence.validation_parent_branch_output_rows,
        evidence.validation_child_pre_branch_residual_rows,
        design_name="child_context_rows",
        hidden_size=source_contract.hidden_size,
        design_ndim=3,
    )
    flat_design = child_context.reshape(
        int(child_context.shape[0]),
        target_attention.query_heads * target_attention.head_dim,
    )
    flat_validation = None
    if validation_context is not None:
        if tuple(validation_context.shape[1:]) != (
            target_attention.query_heads,
            target_attention.head_dim,
        ):
            raise Lfm2ClosedLoopEvidenceError(
                "Validation child attention context does not match target topology."
            )
        flat_validation = validation_context.reshape(
            int(validation_context.shape[0]),
            target_attention.query_heads * target_attention.head_dim,
        )
    output_prior = cluster.aggregate_output_consumer(
        source_layer.self_attn.out_proj.weight,
        axis=1,
    )
    endpoint_fit = _fit_endpoint(
        branch="attention",
        design=flat_design,
        parent_output=parent_output,
        child_residual=child_residual,
        prior_weight=output_prior,
        validation_design=flat_validation,
        validation_target=validation_target,
        config=config,
    )
    output_weight = endpoint_fit.fit.weight
    expected_shapes = {
        "query": tuple(target_layer.self_attn.q_proj.weight.shape),
        "key": tuple(target_layer.self_attn.k_proj.weight.shape),
        "value": tuple(target_layer.self_attn.v_proj.weight.shape),
        "output": tuple(target_layer.self_attn.out_proj.weight.shape),
    }
    observed_shapes = {
        "query": tuple(query_weight.shape),
        "key": tuple(key_weight.shape),
        "value": tuple(value_weight.shape),
        "output": tuple(output_weight.shape),
    }
    if observed_shapes != expected_shapes:
        raise Lfm2ClosedLoopInvariantError(
            f"Folded attention shapes do not match target: {observed_shapes} != "
            f"{expected_shapes}."
        )
    return Lfm2AttentionGroupMerge(
        cluster_plan=cluster,
        query_weight=query_weight.detach().to(device="cpu").contiguous().clone(),
        key_weight=key_weight.detach().to(device="cpu").contiguous().clone(),
        value_weight=value_weight.detach().to(device="cpu").contiguous().clone(),
        output_weight=output_weight.detach().to(device="cpu").contiguous().clone(),
        endpoint_fit=endpoint_fit,
    )


def _build_mlp_merge(
    source_layer: Any,
    target_layer: Any,
    source_contract: Lfm2LayerContract,
    target_contract: Lfm2LayerContract,
    evidence: Lfm2MLPFoldEvidence,
    config: Lfm2ClosedLoopFoldConfig,
) -> Lfm2SwiGLUNeuronMerge:
    torch = _torch()
    gate = _float64_tensor(evidence.parent_gate_rows, "parent_gate_rows", ndim=2)
    up = _float64_tensor(evidence.parent_up_rows, "parent_up_rows", ndim=2)
    post = _float64_tensor(
        evidence.parent_post_swiglu_rows,
        "parent_post_swiglu_rows",
        ndim=2,
    )
    expected_shape = (int(gate.shape[0]), source_contract.ffn_intermediate_size)
    if (
        tuple(gate.shape) != expected_shape
        or tuple(up.shape) != expected_shape
        or tuple(post.shape) != expected_shape
    ):
        raise Lfm2ClosedLoopEvidenceError(
            "Parent gate, up, and post-SwiGLU rows must be aligned to the source "
            "intermediate width."
        )
    source_down = source_layer.feed_forward.w2.weight.detach().to(
        device="cpu", dtype=torch.float64
    )
    if evidence.parent_contribution_rows is None:
        # Activation clustering accepts a rank-2 scalar contribution signature.
        # Using realized activity times the source consumer-column norm avoids
        # materializing [rows, source_neurons, hidden_size], which would be
        # several GiB for the real 2560 x 1024 LFM2.5-230M MLP.
        contribution = post * torch.linalg.vector_norm(source_down, dim=0).unsqueeze(0)
    else:
        raw_contribution = evidence.parent_contribution_rows
        raw_ndim = getattr(raw_contribution, "ndim", None)
        if raw_ndim not in {2, 3}:
            raise Lfm2ClosedLoopEvidenceError(
                "MLP parent_contribution_rows must have rank 2 or 3."
            )
        contribution = _float64_tensor(
            raw_contribution,
            "parent_contribution_rows",
            ndim=int(raw_ndim),
        )
    if tuple(contribution.shape[:2]) != (
        int(gate.shape[0]),
        source_contract.ffn_intermediate_size,
    ):
        raise Lfm2ClosedLoopEvidenceError(
            "MLP parent_contribution_rows must begin with [rows, source_neurons]."
        )
    cluster = build_gated_mlp_activation_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=target_contract.ffn_intermediate_size,
        source_neurons=source_contract.ffn_intermediate_size,
        ridge=config.cluster_ridge,
        max_iterations=config.cluster_max_iterations,
        distance_chunk_size=config.cluster_distance_chunk_size,
    )
    gate_weight = cluster.select_gate_producer(
        source_layer.feed_forward.w1.weight,
        axis=0,
    )
    up_weight = cluster.select_up_producer(
        source_layer.feed_forward.w3.weight,
        axis=0,
    )
    child_post = _float64_tensor(
        evidence.child_post_swiglu_rows,
        "child_post_swiglu_rows",
        ndim=2,
    )
    if int(child_post.shape[1]) != target_contract.ffn_intermediate_size:
        raise Lfm2ClosedLoopEvidenceError(
            "child_post_swiglu_rows does not match target intermediate width."
        )
    parent_output = _float64_tensor(
        evidence.parent_branch_output_rows,
        "parent_branch_output_rows",
        ndim=2,
    )
    child_residual = _float64_tensor(
        evidence.child_pre_branch_residual_rows,
        "child_pre_branch_residual_rows",
        ndim=2,
    )
    if int(parent_output.shape[1]) != source_contract.hidden_size:
        raise Lfm2ClosedLoopEvidenceError(
            "MLP endpoint target width does not match hidden_size."
        )
    validation_post, validation_target = _optional_validation_triplet(
        evidence.validation_child_post_swiglu_rows,
        evidence.validation_parent_branch_output_rows,
        evidence.validation_child_pre_branch_residual_rows,
        design_name="child_post_swiglu_rows",
        hidden_size=source_contract.hidden_size,
        design_ndim=2,
    )
    if (
        validation_post is not None
        and int(validation_post.shape[1]) != target_contract.ffn_intermediate_size
    ):
        raise Lfm2ClosedLoopEvidenceError(
            "Validation child post-SwiGLU rows do not match target width."
        )
    down_prior = cluster.aggregate_down_consumer(
        source_layer.feed_forward.w2.weight,
        axis=1,
    )
    endpoint_fit = _fit_endpoint(
        branch="mlp",
        design=child_post,
        parent_output=parent_output,
        child_residual=child_residual,
        prior_weight=down_prior,
        validation_design=validation_post,
        validation_target=validation_target,
        config=config,
    )
    down_weight = endpoint_fit.fit.weight
    expected_shapes = {
        "gate": tuple(target_layer.feed_forward.w1.weight.shape),
        "up": tuple(target_layer.feed_forward.w3.weight.shape),
        "down": tuple(target_layer.feed_forward.w2.weight.shape),
    }
    observed_shapes = {
        "gate": tuple(gate_weight.shape),
        "up": tuple(up_weight.shape),
        "down": tuple(down_weight.shape),
    }
    if observed_shapes != expected_shapes:
        raise Lfm2ClosedLoopInvariantError(
            f"Folded MLP shapes do not match target: {observed_shapes} != "
            f"{expected_shapes}."
        )
    return Lfm2SwiGLUNeuronMerge(
        cluster_plan=cluster,
        gate_weight=gate_weight.detach().to(device="cpu").contiguous().clone(),
        up_weight=up_weight.detach().to(device="cpu").contiguous().clone(),
        down_weight=down_weight.detach().to(device="cpu").contiguous().clone(),
        endpoint_fit=endpoint_fit,
    )


def _build_candidate_state(
    source_layer: Any,
    target_layer: Any,
    *,
    attention: Lfm2AttentionGroupMerge | None,
    mlp: Lfm2SwiGLUNeuronMerge,
) -> tuple[
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, tuple[int, ...], str], ...],
]:
    source = source_layer.state_dict()
    target = target_layer.state_dict()
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise Lfm2ClosedLoopTopologyError(
            f"Source/target LFM2 state keys differ; missing={missing}, extra={extra}."
        )
    folded_keys = set(_MLP_FOLDED_KEYS)
    if attention is not None:
        folded_keys.update(_ATTENTION_FOLDED_KEYS)
    candidate: dict[str, Any] = {}
    for name in sorted(target):
        if source[name].dtype != target[name].dtype:
            raise Lfm2ClosedLoopTopologyError(
                f"Source/target state dtype differs for {name!r}."
            )
        if name not in folded_keys:
            candidate[name] = _clone_for_target(source[name], target[name], name)

    overrides = {
        "feed_forward.w1.weight": mlp.gate_weight,
        "feed_forward.w3.weight": mlp.up_weight,
        "feed_forward.w2.weight": mlp.down_weight,
    }
    if attention is not None:
        overrides.update(
            {
                "self_attn.q_proj.weight": attention.query_weight,
                "self_attn.k_proj.weight": attention.key_weight,
                "self_attn.v_proj.weight": attention.value_weight,
                "self_attn.out_proj.weight": attention.output_weight,
            }
        )
    for name, value in overrides.items():
        if name not in target:
            raise Lfm2ClosedLoopTopologyError(
                f"Target LFM2 state is missing folded tensor {name!r}."
            )
        candidate[name] = _candidate_tensor(value, target[name], name)
    if set(candidate) != set(target):
        missing = sorted(set(target) - set(candidate))
        raise Lfm2ClosedLoopInvariantError(
            f"The candidate does not overwrite every target state tensor: {missing}."
        )
    items = tuple((name, candidate[name]) for name in sorted(candidate))
    return items, _state_spec(target)


def build_lfm2_closed_loop_fold_plan(
    source_layer: Any,
    target_layer: Any,
    evidence: Lfm2ClosedLoopFoldEvidence,
    *,
    config: Lfm2ClosedLoopFoldConfig | None = None,
) -> Lfm2ClosedLoopFoldPlan:
    """Build one immutable, target-seed-independent LFM2 layer candidate."""

    if not isinstance(evidence, Lfm2ClosedLoopFoldEvidence):
        raise TypeError("evidence must be Lfm2ClosedLoopFoldEvidence.")
    selected = config or Lfm2ClosedLoopFoldConfig()
    if not isinstance(selected, Lfm2ClosedLoopFoldConfig):
        raise TypeError("config must be Lfm2ClosedLoopFoldConfig or None.")
    source_contract = inspect_lfm2_layer(source_layer)
    target_contract = inspect_lfm2_layer(target_layer)
    _validate_layer_pair(source_contract, target_contract, evidence)
    source_before_items = _state_items(source_layer)
    target_before_items = _state_items(target_layer)
    source_before = _state_mapping(source_before_items)
    target_before = _state_mapping(target_before_items)
    source_runtime = _gradient_signature(source_layer)
    target_runtime = _gradient_signature(target_layer)

    torch = _torch()
    with torch.no_grad():
        attention = None
        if source_contract.kind == "full_attention":
            assert evidence.attention is not None  # guarded above
            attention = _build_attention_merge(
                source_layer,
                target_layer,
                source_contract,
                target_contract,
                evidence.attention,
                selected,
            )
        mlp = _build_mlp_merge(
            source_layer,
            target_layer,
            source_contract,
            target_contract,
            evidence.mlp,
            selected,
        )
        short_conv_plan = None
        if source_contract.kind == "short_conv":
            if not isinstance(source_contract.operator, Lfm2ShortConvContract):
                raise Lfm2ClosedLoopTopologyError(
                    "Internal LFM2 short-conv contract mismatch."
                )
            short_conv_plan = build_lfm2_short_conv_channel_plan(
                source_layer,
                target_hidden_size=source_contract.hidden_size,
                mode="subset",
                selected_channels=tuple(range(source_contract.hidden_size)),
            )
            if not short_conv_plan.exactness.algebraically_exact:
                raise Lfm2ClosedLoopInvariantError(
                    "The first LFM2 transaction must preserve short-conv exactly."
                )
        candidate_items, target_spec = _build_candidate_state(
            source_layer,
            target_layer,
            attention=attention,
            mlp=mlp,
        )

    source_after = _state_mapping(_state_items(source_layer))
    target_after = _state_mapping(_state_items(target_layer))
    if (
        _state_sha256(source_after) != _state_sha256(source_before)
        or _state_sha256(target_after) != _state_sha256(target_before)
        or _gradient_signature(source_layer) != source_runtime
        or _gradient_signature(target_layer) != target_runtime
    ):
        raise Lfm2ClosedLoopInvariantError(
            "Building an LFM2 fold plan mutated a model or gradient state."
        )
    candidate = _state_mapping(candidate_items)
    source_sha = _state_sha256(source_before)
    target_sha = _state_sha256(target_before)
    candidate_sha = _state_sha256(candidate)
    metadata = {
        "schema": _SCHEMA,
        "source_contract": source_contract.as_dict(),
        "target_contract": target_contract.as_dict(),
        "config": selected.as_dict(),
        "attention_cluster": (
            None
            if attention is None
            else {
                "group_medoids": list(attention.cluster_plan.group_medoids),
                "group_assignment": list(attention.cluster_plan.group_assignment),
                "query_head_medoids": list(attention.cluster_plan.query_head_medoids),
                "endpoint_fit": attention.endpoint_fit.as_dict(),
            }
        ),
        "mlp_cluster": {
            "medoids": list(mlp.cluster_plan.medoid_source_neurons),
            "assignment": list(mlp.cluster_plan.source_to_target),
            "endpoint_fit": mlp.endpoint_fit.as_dict(),
        },
        "short_conv_plan": (
            None if short_conv_plan is None else short_conv_plan.as_dict()
        ),
        "candidate_state_sha256": candidate_sha,
        "target_state_spec": [
            [name, list(shape), dtype] for name, shape, dtype in target_spec
        ],
        "zero_gradient": True,
    }
    plan_sha = _canonical_sha256(metadata)
    return Lfm2ClosedLoopFoldPlan(
        source_contract=source_contract,
        target_contract=target_contract,
        config=selected,
        attention=attention,
        mlp=mlp,
        short_conv_plan=short_conv_plan,
        candidate_state_items=candidate_items,
        target_state_spec=target_spec,
        source_state_sha256=source_sha,
        target_state_sha256_before=target_sha,
        candidate_state_sha256=candidate_sha,
        plan_sha256=plan_sha,
    )


def snapshot_lfm2_layer(target_layer: Any) -> Lfm2LayerSnapshot:
    """Capture exact target state without changing parameter identities or grads."""

    contract = inspect_lfm2_layer(target_layer)
    items = _state_items(target_layer)
    state = _state_mapping(items)
    return Lfm2LayerSnapshot(
        contract=contract,
        state_items=items,
        state_spec=_state_spec(state),
        state_sha256=_state_sha256(state),
        parameter_ids=_parameter_ids(target_layer),
        gradient_signature=_gradient_signature(target_layer),
    )


def restore_lfm2_layer_snapshot(
    target_layer: Any,
    snapshot: Lfm2LayerSnapshot,
) -> bool:
    """Restore one snapshot by in-place copies and prove byte-exact rollback."""

    if not isinstance(snapshot, Lfm2LayerSnapshot):
        raise TypeError("snapshot must be Lfm2LayerSnapshot.")
    contract = inspect_lfm2_layer(target_layer)
    if contract.as_dict() != snapshot.contract.as_dict():
        raise Lfm2ClosedLoopTopologyError("Snapshot and target LFM2 topology differ.")
    current = target_layer.state_dict()
    if _state_spec(current) != snapshot.state_spec:
        raise Lfm2ClosedLoopTopologyError(
            "Snapshot and target LFM2 state specs differ."
        )
    if _parameter_ids(target_layer) != snapshot.parameter_ids:
        raise Lfm2ClosedLoopInvariantError(
            "Target parameter identities changed before rollback."
        )
    if _gradient_signature(target_layer) != snapshot.gradient_signature:
        raise Lfm2ClosedLoopInvariantError(
            "Target gradient state changed before rollback."
        )
    saved = _state_mapping(snapshot.state_items)
    torch = _torch()
    with torch.no_grad():
        for name in sorted(current):
            current[name].copy_(
                saved[name].to(device=current[name].device, dtype=current[name].dtype)
            )
    restored = _state_mapping(_state_items(target_layer))
    return bool(
        _state_sha256(restored) == snapshot.state_sha256
        and _parameter_ids(target_layer) == snapshot.parameter_ids
        and _gradient_signature(target_layer) == snapshot.gradient_signature
    )


def install_lfm2_closed_loop_fold_plan(
    target_layer: Any,
    plan: Lfm2ClosedLoopFoldPlan,
) -> Lfm2ClosedLoopFoldTransaction:
    """Install a candidate in place and return an explicit commit/rollback handle."""

    if not isinstance(plan, Lfm2ClosedLoopFoldPlan):
        raise TypeError("plan must be Lfm2ClosedLoopFoldPlan.")
    contract = inspect_lfm2_layer(target_layer)
    if contract.as_dict() != plan.target_contract.as_dict():
        raise Lfm2ClosedLoopTopologyError(
            "The plan target contract does not match this LFM2 layer."
        )
    current = target_layer.state_dict()
    if _state_spec(current) != plan.target_state_spec:
        raise Lfm2ClosedLoopTopologyError(
            "The plan target state spec does not match this LFM2 layer."
        )
    candidate = plan.candidate_state_dict()
    if _state_sha256(candidate) != plan.candidate_state_sha256:
        raise Lfm2ClosedLoopInvariantError(
            "The LFM2 plan candidate tensors changed after planning."
        )
    snapshot = snapshot_lfm2_layer(target_layer)
    torch = _torch()
    try:
        with torch.no_grad():
            for name in sorted(current):
                current[name].copy_(
                    candidate[name].to(
                        device=current[name].device,
                        dtype=current[name].dtype,
                    )
                )
        installed = _state_mapping(_state_items(target_layer))
        if _state_sha256(installed) != plan.candidate_state_sha256:
            raise Lfm2ClosedLoopInvariantError(
                "Installed LFM2 state does not match the planned candidate."
            )
        if (
            _parameter_ids(target_layer) != snapshot.parameter_ids
            or _gradient_signature(target_layer) != snapshot.gradient_signature
        ):
            raise Lfm2ClosedLoopInvariantError(
                "Installing the LFM2 candidate changed identities or gradients."
            )
    except Exception:
        snapshot.restore(target_layer)
        raise
    return Lfm2ClosedLoopFoldTransaction(target_layer, plan, snapshot)


def begin_lfm2_closed_loop_fold_transaction(
    source_layer: Any,
    target_layer: Any,
    evidence: Lfm2ClosedLoopFoldEvidence,
    *,
    config: Lfm2ClosedLoopFoldConfig | None = None,
) -> Lfm2ClosedLoopFoldTransaction:
    """Build and install one candidate without implicitly accepting it."""

    plan = build_lfm2_closed_loop_fold_plan(
        source_layer,
        target_layer,
        evidence,
        config=config,
    )
    return install_lfm2_closed_loop_fold_plan(target_layer, plan)


# Readable compatibility aliases for callers that use plan/install terminology.
build_lfm2_closed_loop_plan = build_lfm2_closed_loop_fold_plan
install_lfm2_closed_loop_fold = install_lfm2_closed_loop_fold_plan


__all__ = [
    "Lfm2AttentionFoldEvidence",
    "Lfm2AttentionGroupMerge",
    "Lfm2ClosedLoopEvidenceError",
    "Lfm2ClosedLoopFitRejected",
    "Lfm2ClosedLoopFoldConfig",
    "Lfm2ClosedLoopFoldError",
    "Lfm2ClosedLoopFoldEvidence",
    "Lfm2ClosedLoopFoldPlan",
    "Lfm2ClosedLoopFoldTransaction",
    "Lfm2ClosedLoopFoldTransactionReport",
    "Lfm2ClosedLoopInvariantError",
    "Lfm2ClosedLoopTopologyError",
    "Lfm2EndpointFit",
    "Lfm2LayerSnapshot",
    "Lfm2MLPFoldEvidence",
    "Lfm2SwiGLUNeuronMerge",
    "begin_lfm2_closed_loop_fold_transaction",
    "build_lfm2_closed_loop_fold_plan",
    "build_lfm2_closed_loop_plan",
    "install_lfm2_closed_loop_fold",
    "install_lfm2_closed_loop_fold_plan",
    "restore_lfm2_layer_snapshot",
    "snapshot_lfm2_layer",
]
