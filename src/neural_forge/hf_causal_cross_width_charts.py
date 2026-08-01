"""Rectangular residual boundary charts for cross-width causal folding.

The family-neutral branch kernels are already width-agnostic: the attention
kernel's Q/K/V fits are plain ridge maps from child rows of *any* width to
desired head targets, and ``fit_functional_swiglu`` constrains row counts
rather than input/target widths.  Width equality was enforced only by the
promotion preamble and the planner contract, so crossing the residual width is
a *target construction* problem, not a new solver problem.

This module builds the missing coordinate machinery:

* one orthonormal encoder ``E_b in R^{Hp x Hc}`` per residual-write boundary,
  selected by the family-neutral balanced controllability/observability POD of
  :mod:`neural_forge.dynamic_balanced_circuit_basis` — not activation PCA and
  not raw weight SVD;
* one cycle-anchored ridge decoder ``D_b in R^{Hc x Hp}``, the deterministic
  regularized-reciprocal-anchored solve the deployed moving-gauge reclosure
  already uses;
* the four measured compatibility defects (cycle, chart transition, absorption
  demand against the writer's representable rank, and norm commutation), each
  bounded by a sealed policy; and
* the load-bearing sequential branch target
  ``Y = h_{b+1} E_{b+1} - hhat_b_actual``, whose expansion carries the
  transported parent delta, the chart transition the ordinary writer must
  absorb, and the upstream-error correction that repairs the A0025 failure
  mode.

Everything is fold-only.  Chart selection reads sealed parent evidence and
already-materialized parent weights; it never runs the parent, never builds an
optimizer, and never touches a gradient.  The only model execution in this
module is the *child* recapture, which is what makes each branch fit sequential
rather than parent-coordinate-local.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_arch_layout import DecoderLayout, build_decoder_layout, resolve_module
from .hf_causal_circuit_evidence import HFCausalCircuitEvidence


CHART_SCHEMA = "neural-forge-residual-boundary-chart-v1"
CHART_SET_SCHEMA = "neural-forge-residual-boundary-chart-set-v1"
CROSS_WIDTH_BRANCH_FOLD_SCHEMA = "neural-forge-cross-width-branch-fold-v1"
REVERSE_SWEEP_SCHEMA = "neural-forge-cross-width-reverse-reclosure-sweep-v1"

#: Every residual-write boundary role this module charts.  The granularity
#: matches the sealed evidence: ``block_input`` opens the attention branch,
#: ``post_attention_residual`` opens the MLP branch, and the last layer's
#: ``block_output`` is the final-norm input.
ATTENTION_INPUT_BOUNDARY = "attention_in"
MLP_INPUT_BOUNDARY = "mlp_in"
TERMINAL_BOUNDARY_ID = "output.final_norm_in"
EMBEDDING_BOUNDARY_INDEX = 0


class CrossWidthResidualChartError(ValueError):
    """A residual boundary chart is outside its proven fold envelope."""


class CrossWidthResidualChartRankError(CrossWidthResidualChartError):
    """A chart or a writer cannot represent what the crossing demands."""


class CrossWidthResidualChartAuthenticationError(CrossWidthResidualChartError):
    """A chart set failed its seal, lineage, or tensor fingerprint check."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise CrossWidthResidualChartError(
            "PyTorch is required for residual boundary charts."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CrossWidthResidualChartError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CrossWidthResidualChartError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrossWidthResidualChartError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise CrossWidthResidualChartError(f"{name} must be finite and non-negative.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result <= 0.0:
        raise CrossWidthResidualChartError(f"{name} must be strictly positive.")
    return result


def _matrix(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise CrossWidthResidualChartError(f"{name} must be a nonempty real matrix.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise CrossWidthResidualChartError(f"{name} is not finite.")
    return result


def _sequences(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise CrossWidthResidualChartError(
            f"{name} must be a nonempty real [batch, sequence, width] tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise CrossWidthResidualChartError(f"{name} is not finite.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
            "bytes": tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex(),
        }
    )


def _masked_rows(values: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in values.shape[2:])
    return values.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _relative_frobenius(difference: Any, reference: Any) -> float:
    torch = _torch()
    numerator = float(torch.linalg.matrix_norm(difference, ord="fro").item())
    denominator = float(torch.linalg.matrix_norm(reference, ord="fro").item())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise CrossWidthResidualChartError(
            "A defect reference has no energy; the boundary evidence is empty."
        )
    return numerator / denominator


# ---------------------------------------------------------------------------
# Deterministic anchored ridge solves (the deployed moving-gauge decoder math)
# ---------------------------------------------------------------------------


def _ridge_scale(design: Any) -> Any:
    torch = _torch()
    spectral = torch.linalg.matrix_norm(design, ord=2)
    return max(float((spectral * spectral).item()), torch.finfo(torch.float64).tiny)


def _anchored_ridge(
    design: Any,
    target: Any,
    anchor: Any,
    *,
    ridge_ratio: float,
    absolute_ridge_floor: float,
) -> tuple[Any, dict[str, Any]]:
    """Solve ``min ||XW-Y||^2 + ridge ||W-anchor||^2`` in closed form."""

    torch = _torch()
    if (
        int(design.shape[0]) != int(target.shape[0])
        or int(design.shape[1]) != int(anchor.shape[0])
        or int(target.shape[1]) != int(anchor.shape[1])
    ):
        raise CrossWidthResidualChartError("Anchored ridge geometry is inconsistent.")
    scale = _ridge_scale(design)
    ridge = max(absolute_ridge_floor, ridge_ratio * scale)
    residual = target - design @ anchor
    if int(design.shape[1]) <= int(design.shape[0]):
        normal = design.T @ design
        delta = torch.linalg.solve(
            normal + ridge * torch.eye(int(normal.shape[0]), dtype=torch.float64),
            design.T @ residual,
        )
        solve_form = "primal"
    else:
        dual = design @ design.T
        delta = design.T @ torch.linalg.solve(
            dual + ridge * torch.eye(int(dual.shape[0]), dtype=torch.float64),
            residual,
        )
        solve_form = "dual"
    replacement = (anchor + delta).contiguous()
    if not bool(torch.isfinite(replacement).all().item()):
        raise CrossWidthResidualChartError("Anchored ridge replacement is non-finite.")
    return replacement, {
        "solve_form": solve_form,
        "rows": int(design.shape[0]),
        "input_width": int(design.shape[1]),
        "output_width": int(target.shape[1]),
        "ridge_ratio": float(ridge_ratio),
        "ridge_scale": float(scale),
        "effective_ridge": float(ridge),
        "closed_form_linear_solve": True,
        "autograd_used": False,
    }


def _regularized_reciprocal(
    encoder: Any,
    *,
    ridge_ratio: float,
    absolute_ridge_floor: float,
) -> tuple[Any, dict[str, Any]]:
    """Return a deterministic ridge reciprocal without ``pinv`` thresholds."""

    torch = _torch()
    source_width = int(encoder.shape[0])
    target_width = int(encoder.shape[1])
    return _anchored_ridge(
        encoder,
        torch.eye(source_width, dtype=torch.float64),
        torch.zeros(target_width, source_width, dtype=torch.float64),
        ridge_ratio=ridge_ratio,
        absolute_ridge_floor=absolute_ridge_floor,
    )


# ---------------------------------------------------------------------------
# Boundary identity
# ---------------------------------------------------------------------------


def residual_boundary_id(layer_index: int, role: str) -> str:
    """Name one residual-write boundary the way the evidence names its rows."""

    index = layer_index if isinstance(layer_index, int) else -1
    if isinstance(layer_index, bool) or index < 0:
        raise CrossWidthResidualChartError(
            "A residual boundary needs a non-negative layer index."
        )
    if role not in {ATTENTION_INPUT_BOUNDARY, MLP_INPUT_BOUNDARY}:
        raise CrossWidthResidualChartError(
            f"Unknown residual boundary role {role!r}."
        )
    return f"layer{index:03d}.{role}"


def residual_boundary_ids(layer_count: int) -> tuple[str, ...]:
    """Every residual-write boundary, in forward order, plus the terminal one."""

    count = _positive_int(layer_count, "layer_count")
    ordered: list[str] = []
    for index in range(count):
        ordered.append(residual_boundary_id(index, ATTENTION_INPUT_BOUNDARY))
        ordered.append(residual_boundary_id(index, MLP_INPUT_BOUNDARY))
    ordered.append(TERMINAL_BOUNDARY_ID)
    return tuple(ordered)


def interface_locked_boundary_ids(layer_count: int) -> tuple[str, ...]:
    """Boundary 0 (embedding output) and the terminal (final-norm input)."""

    ordered = residual_boundary_ids(layer_count)
    return (ordered[EMBEDDING_BOUNDARY_INDEX], TERMINAL_BOUNDARY_ID)


def _boundary_rows(evidence: HFCausalCircuitEvidence, boundary_id: str) -> Any:
    """Return the cached parent residual sequences at one boundary."""

    layers = evidence.layers
    if boundary_id == TERMINAL_BOUNDARY_ID:
        return layers[-1].block_output
    for index, layer in enumerate(layers):
        if boundary_id == residual_boundary_id(index, ATTENTION_INPUT_BOUNDARY):
            return layer.block_input
        if boundary_id == residual_boundary_id(index, MLP_INPUT_BOUNDARY):
            return layer.post_attention_residual
    raise CrossWidthResidualChartError(
        f"Boundary {boundary_id!r} is not a cached parent residual endpoint."
    )


def child_residual_topology_sha256(child: Any) -> str:
    """Seal the child geometry every chart in a set is bound to."""

    layout = child if isinstance(child, DecoderLayout) else build_decoder_layout(child)
    body = {
        "schema": CHART_SET_SCHEMA,
        "component": "child_residual_topology",
        "family": layout.family,
        "model_type": layout.model_type,
        "hidden_size": int(layout.hidden_size),
        "intermediate_size": int(layout.intermediate_size),
        "layer_count": len(layout.blocks),
        "vocabulary_size": int(layout.embeddings.vocab_size),
        "tie_word_embeddings": bool(layout.tie_word_embeddings),
        "position_kind": layout.position.kind,
        "rope_theta": layout.position.rope_theta,
        "rope_type": layout.position.rope_type,
        "max_positions": int(layout.position.max_positions),
        "final_norm_epsilon": float(layout.final_norm.epsilon),
        "blocks": [
            {
                "index": int(block.index),
                "attention_input_norm_epsilon": float(
                    block.attention_input_norm.epsilon
                ),
                "query_heads": int(block.attention.query_heads),
                "key_value_heads": int(block.attention.key_value_heads),
                "head_dim": int(block.attention.head_dim),
                "projection_style": block.attention.projection_style,
                "query_key_norm": bool(block.attention.query_norm is not None),
                "mlp_input_norm_epsilon": float(block.mlp_input_norm.epsilon),
                "intermediate_size": int(block.mlp.intermediate_size),
                "activation": block.mlp.activation_name,
                "residual_kind": block.residual.kind,
            }
            for block in layout.blocks
        ],
    }
    return canonical_sha256(body)


def _writer_capacity(layout: DecoderLayout, boundary_id: str) -> int | None:
    """Rank the branch that writes out of ``boundary_id`` can represent.

    The attention branch writes through ``W_o`` and the MLP branch through the
    down projection, so the absorption demand ``A_b`` is bounded by the child's
    residual width and by that writer's own inner width.  The terminal boundary
    has no writer at all, which is exactly why it is interface locked.
    """

    if boundary_id == TERMINAL_BOUNDARY_ID:
        return None
    for block in layout.blocks:
        if boundary_id == residual_boundary_id(block.index, ATTENTION_INPUT_BOUNDARY):
            return min(
                int(layout.hidden_size),
                int(block.attention.query_heads) * int(block.attention.head_dim),
            )
        if boundary_id == residual_boundary_id(block.index, MLP_INPUT_BOUNDARY):
            return min(int(layout.hidden_size), int(block.mlp.intermediate_size))
    raise CrossWidthResidualChartError(
        f"Boundary {boundary_id!r} has no child writer in this layout."
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidualBoundaryChartPolicy:
    """Fixed analytic bounds; no scalar here ever selects a child."""

    chart_ridge_ratio: float = 1e-8
    cycle_consistency_weight: float = 1e-2
    absolute_ridge_floor: float = 1e-12
    orthonormality_tolerance: float = 1e-9
    max_cycle_defect: float = 0.75
    max_norm_commutation_fluctuation: float = 0.25
    max_unabsorbed_transition_energy_fraction: float = 0.25
    structural_controllability_weight: float = 0.35
    cached_controllability_weight: float = 0.65
    gram_regularization: float = 1e-10
    rank_tolerance: float = 1e-12
    max_condition_number: float = 1e12
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        body: dict[str, Any] = {"schema": CHART_SET_SCHEMA, "component": "policy"}
        for name in (
            "chart_ridge_ratio",
            "cycle_consistency_weight",
            "orthonormality_tolerance",
            "max_cycle_defect",
            "max_norm_commutation_fluctuation",
            "max_unabsorbed_transition_energy_fraction",
        ):
            body[name] = _nonnegative_float(getattr(self, name), name)
        for name in (
            "absolute_ridge_floor",
            "structural_controllability_weight",
            "cached_controllability_weight",
            "gram_regularization",
            "rank_tolerance",
            "max_condition_number",
        ):
            body[name] = _positive_float(getattr(self, name), name)
        if body["max_condition_number"] < 1.0:
            raise CrossWidthResidualChartError(
                "max_condition_number must be at least one."
            )
        object.__setattr__(self, "policy_sha256", canonical_sha256(body))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CHART_SET_SCHEMA,
            "component": "policy",
            "chart_ridge_ratio": self.chart_ridge_ratio,
            "cycle_consistency_weight": self.cycle_consistency_weight,
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "orthonormality_tolerance": self.orthonormality_tolerance,
            "max_cycle_defect": self.max_cycle_defect,
            "max_norm_commutation_fluctuation": (
                self.max_norm_commutation_fluctuation
            ),
            "max_unabsorbed_transition_energy_fraction": (
                self.max_unabsorbed_transition_energy_fraction
            ),
            "structural_controllability_weight": (
                self.structural_controllability_weight
            ),
            "cached_controllability_weight": self.cached_controllability_weight,
            "gram_regularization": self.gram_regularization,
            "rank_tolerance": self.rank_tolerance,
            "max_condition_number": self.max_condition_number,
            "policy_sha256": self.policy_sha256,
        }


# ---------------------------------------------------------------------------
# Sealed artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidualBoundaryChart:
    """One sealed rectangular chart at one residual-write boundary."""

    boundary_id: str
    encoder: Any
    decoder: Any
    interface_locked: bool
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_id, str) or not self.boundary_id.strip():
            raise CrossWidthResidualChartError("boundary_id must be a nonempty string.")
        encoder = _matrix(self.encoder, f"{self.boundary_id} encoder")
        decoder = _matrix(self.decoder, f"{self.boundary_id} decoder")
        if not isinstance(self.interface_locked, bool):
            raise CrossWidthResidualChartError("interface_locked must be boolean.")
        if tuple(decoder.shape) != (int(encoder.shape[1]), int(encoder.shape[0])):
            raise CrossWidthResidualChartError(
                f"{self.boundary_id} decoder does not transpose the encoder shape."
            )
        if not isinstance(self.report, Mapping):
            raise CrossWidthResidualChartError("report must be a mapping.")
        object.__setattr__(self, "encoder", encoder)
        object.__setattr__(self, "decoder", decoder)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def source_width(self) -> int:
        return int(self.encoder.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.encoder.shape[1])

    @property
    def chart_sha256(self) -> str:
        value = self.report.get("chart_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise CrossWidthResidualChartError("Chart report seal is invalid.")
        return value

    def verify(self) -> "ResidualBoundaryChart":
        body = dict(self.report)
        seal = body.pop("chart_sha256", None)
        if (
            body.get("encoder_sha256") != _tensor_sha256(self.encoder)
            or body.get("decoder_sha256") != _tensor_sha256(self.decoder)
            or body.get("boundary_id") != self.boundary_id
            or body.get("interface_locked") is not self.interface_locked
            or canonical_sha256(body) != seal
        ):
            raise CrossWidthResidualChartAuthenticationError(
                f"Residual boundary chart {self.boundary_id!r} failed its seal."
            )
        return self

    def encode(self, rows: Any) -> Any:
        """Transport parent rows into this boundary's child coordinates."""

        torch = _torch()
        value = rows.detach().to(device="cpu", dtype=torch.float64)
        if int(value.shape[-1]) != self.source_width:
            raise CrossWidthResidualChartError(
                f"{self.boundary_id} encode received the wrong parent width."
            )
        return (value @ self.encoder).contiguous()

    def decode(self, rows: Any) -> Any:
        """Estimate parent rows from child rows through the anchored decoder."""

        torch = _torch()
        value = rows.detach().to(device="cpu", dtype=torch.float64)
        if int(value.shape[-1]) != self.target_width:
            raise CrossWidthResidualChartError(
                f"{self.boundary_id} decode received the wrong child width."
            )
        return (value @ self.decoder).contiguous()

    # ``fold_consumer_weight`` / ``source_width`` / ``target_width`` /
    # ``map_sha256`` make an interface-locked chart directly usable as the
    # dense vocabulary projector the token-interface operators already accept.
    @property
    def map_sha256(self) -> str:
        return self.chart_sha256

    def fold_consumer_weight(self, weight: Any) -> Any:
        value = _matrix(weight, f"{self.boundary_id} consumer weight")
        if int(value.shape[1]) != self.source_width:
            raise CrossWidthResidualChartError(
                f"{self.boundary_id} consumer weight does not match the parent width."
            )
        return (value @ self.encoder).contiguous()


@dataclass(frozen=True, slots=True)
class ResidualBoundaryChartSet:
    """One sealed ordered chart per residual-write boundary of one crossing."""

    parent_evidence_sha256: str
    child_topology_sha256: str
    charts: tuple[ResidualBoundaryChart, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        charts = tuple(self.charts)
        if not charts or any(
            not isinstance(value, ResidualBoundaryChart) for value in charts
        ):
            raise CrossWidthResidualChartError(
                "A chart set requires sealed ResidualBoundaryChart values."
            )
        ids = tuple(value.boundary_id for value in charts)
        if len(set(ids)) != len(ids):
            raise CrossWidthResidualChartError("Boundary ids must be unique.")
        widths = {(value.source_width, value.target_width) for value in charts}
        if len(widths) != 1:
            raise CrossWidthResidualChartError(
                "Every chart in one set shares one source/target width pair."
            )
        if not isinstance(self.report, Mapping):
            raise CrossWidthResidualChartError("report must be a mapping.")
        object.__setattr__(self, "charts", charts)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def source_width(self) -> int:
        return self.charts[0].source_width

    @property
    def target_width(self) -> int:
        return self.charts[0].target_width

    @property
    def boundary_ids(self) -> tuple[str, ...]:
        return tuple(value.boundary_id for value in self.charts)

    @property
    def set_sha256(self) -> str:
        value = self.report.get("set_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise CrossWidthResidualChartError("Chart set seal is invalid.")
        return value

    def chart(self, boundary_id: str) -> ResidualBoundaryChart:
        for value in self.charts:
            if value.boundary_id == boundary_id:
                return value
        raise CrossWidthResidualChartAuthenticationError(
            f"No chart is sealed for boundary {boundary_id!r}."
        )

    def verify(self) -> "ResidualBoundaryChartSet":
        """Reauthenticate every chart tensor, the lineage, and the set seal."""

        for chart in self.charts:
            chart.verify()
        body = dict(self.report)
        seal = body.pop("set_sha256", None)
        if (
            body.get("parent_evidence_sha256") != self.parent_evidence_sha256
            or body.get("child_topology_sha256") != self.child_topology_sha256
            or body.get("charts") != [value.chart_sha256 for value in self.charts]
            or body.get("boundary_ids") != list(self.boundary_ids)
            or canonical_sha256(body) != seal
        ):
            raise CrossWidthResidualChartAuthenticationError(
                "Residual boundary chart set failed its seal or lineage."
            )
        return self

    def authenticate(
        self,
        *,
        parent_evidence_sha256: str,
        child_topology_sha256: str,
        role: str,
    ) -> "ResidualBoundaryChartSet":
        """Fail closed unless this set is bound to exactly this crossing."""

        self.verify()
        if self.parent_evidence_sha256 != parent_evidence_sha256:
            raise CrossWidthResidualChartAuthenticationError(
                f"{role} chart set is bound to different parent evidence."
            )
        if self.child_topology_sha256 != child_topology_sha256:
            raise CrossWidthResidualChartAuthenticationError(
                f"{role} chart set is bound to a different child topology."
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


# ---------------------------------------------------------------------------
# Chart construction
# ---------------------------------------------------------------------------


def _residual_facing_operators(parent_model: Any, layout: DecoderLayout) -> tuple[
    tuple[Any, ...],
    dict[str, tuple[Any, ...]],
]:
    """Return (complete structural inventory, per-boundary consumer index).

    Reading already-materialized parent weights is not a parent execution: no
    forward, hook, trace, or gradient is involved.
    """

    from .dynamic_weight_complete_basis import (
        FeatureOrientation,
        ResidualFacingOperator,
    )

    if layout.embeddings.token_scale != 1.0:
        raise CrossWidthResidualChartError(
            "Chart selection requires unscaled token embeddings."
        )

    def weight(path: str) -> Any:
        return resolve_module(parent_model, path, family=layout.family).weight

    def consumer(tensor: Any, *, role: str, name: str) -> Any:
        return ResidualFacingOperator(
            role=role,
            name=name,
            tensor=tensor,
            orientation=FeatureOrientation.INPUT,
        )

    def producer(tensor: Any, *, role: str, name: str) -> Any:
        return ResidualFacingOperator(
            role=role,
            name=name,
            tensor=tensor,
            orientation=FeatureOrientation.OUTPUT,
        )

    structural: list[Any] = [
        consumer(
            weight(layout.embeddings.token_path),
            role="embedding_interface",
            name="input.token_embedding",
        ),
        consumer(
            weight(layout.lm_head_path),
            role="output_head_interface",
            name="output.lm_head",
        ),
        ResidualFacingOperator(
            role="normalization",
            name="output.final_norm",
            tensor=weight(layout.final_norm.path),
            orientation=FeatureOrientation.INPUT,
            diagonal=True,
        ),
    ]
    consumers_by_layer: dict[int, list[Any]] = {}
    for block in layout.blocks:
        if block.attention.projection_weight_layout != "output_input" or (
            block.mlp.projection_weight_layout != "output_input"
        ):
            raise CrossWidthResidualChartError(
                "Chart selection implements [output, input] projections only."
            )
        if (
            block.attention.query_path is None
            or block.attention.key_path is None
            or block.attention.value_path is None
            or block.mlp.gate_projection_path is None
            or block.mlp.up_projection_path is None
        ):
            raise CrossWidthResidualChartError(
                "Chart selection requires split Q/K/V and gate/up parents."
            )
        layer_consumers = [
            consumer(
                weight(str(block.attention.query_path)),
                role="qkv_input_sensitivity",
                name=f"layer{block.index:03d}.q_proj",
            ),
            consumer(
                weight(str(block.attention.key_path)),
                role="qkv_input_sensitivity",
                name=f"layer{block.index:03d}.k_proj",
            ),
            consumer(
                weight(str(block.attention.value_path)),
                role="qkv_input_sensitivity",
                name=f"layer{block.index:03d}.v_proj",
            ),
            consumer(
                weight(str(block.mlp.gate_projection_path)),
                role="mlp_input_sensitivity",
                name=f"layer{block.index:03d}.gate_proj",
            ),
            consumer(
                weight(str(block.mlp.up_projection_path)),
                role="mlp_input_sensitivity",
                name=f"layer{block.index:03d}.up_proj",
            ),
        ]
        consumers_by_layer[int(block.index)] = list(layer_consumers)
        structural.extend(layer_consumers)
        structural.extend(
            (
                producer(
                    weight(str(block.attention.output_path)),
                    role="operator_output_span",
                    name=f"layer{block.index:03d}.o_proj",
                ),
                producer(
                    weight(str(block.mlp.down_projection_path)),
                    role="mlp_output_span",
                    name=f"layer{block.index:03d}.down_proj",
                ),
                ResidualFacingOperator(
                    role="normalization",
                    name=f"layer{block.index:03d}.attention_input_norm",
                    tensor=weight(block.attention_input_norm.path),
                    orientation=FeatureOrientation.INPUT,
                    diagonal=True,
                ),
                ResidualFacingOperator(
                    role="normalization",
                    name=f"layer{block.index:03d}.mlp_input_norm",
                    tensor=weight(block.mlp_input_norm.path),
                    orientation=FeatureOrientation.INPUT,
                    diagonal=True,
                ),
            )
        )

    head_interface = (
        consumer(
            weight(layout.lm_head_path),
            role="output_head_interface",
            name="downstream.output_head",
        ),
        ResidualFacingOperator(
            role="normalization",
            name="downstream.final_norm",
            tensor=weight(layout.final_norm.path),
            orientation=FeatureOrientation.INPUT,
            diagonal=True,
        ),
    )
    downstream: dict[str, tuple[Any, ...]] = {}
    for block in layout.blocks:
        index = int(block.index)
        later = [
            value
            for other in layout.blocks
            if int(other.index) > index
            for value in consumers_by_layer[int(other.index)]
        ]
        attention_in = residual_boundary_id(index, ATTENTION_INPUT_BOUNDARY)
        mlp_in = residual_boundary_id(index, MLP_INPUT_BOUNDARY)
        downstream[attention_in] = (
            *consumers_by_layer[index],
            *later,
            *head_interface,
        )
        downstream[mlp_in] = (
            *consumers_by_layer[index][3:],
            *later,
            *head_interface,
        )
    downstream[TERMINAL_BOUNDARY_ID] = head_interface
    return tuple(structural), downstream


def _procrustes_frame_transport(encoder: Any, anchor: Any) -> tuple[Any, dict[str, Any]]:
    """Rotate one chart's frame onto its predecessor without moving its span.

    The balanced solve fixes each boundary's *subspace*; the orthonormal frame
    inside that subspace is canonical but arbitrary, so two nearly identical
    subspaces can be handed frames that differ by a large rotation.  That
    rotation would show up entirely as chart-transition absorption demand the
    branch writer has to realize through its own nonlinear operator, for no
    representational gain.  The deployed moving-gauge reclosure transports its
    encoder by orthogonal Procrustes for exactly this reason, and the same
    closed form applies here: ``R = U V^T`` from ``SVD(E^T E_anchor)`` minimizes
    ``||E R - E_anchor||_F`` over orthogonal ``R`` and leaves ``E R`` spanning
    the identical subspace with orthonormal columns.
    """

    torch = _torch()
    left, singular, right = torch.linalg.svd(encoder.T @ anchor, full_matrices=False)
    rotation = (left @ right).contiguous()
    transported = (encoder @ rotation).contiguous()
    before = float(torch.linalg.matrix_norm(encoder - anchor, ord="fro").item())
    after = float(torch.linalg.matrix_norm(transported - anchor, ord="fro").item())
    if not bool(torch.isfinite(transported).all().item()) or after > before + 1e-9:
        raise CrossWidthResidualChartError(
            "Orthogonal Procrustes chart transport did not reduce the frame "
            "mismatch it is defined to minimize."
        )
    return transported, {
        "kind": "orthogonal_procrustes_frame_transport",
        "rotation_sha256": _tensor_sha256(rotation),
        "frame_mismatch_before": before,
        "frame_mismatch_after": after,
        "aligned_singular_value_sum": float(singular.sum().item()),
        "subspace_preserved": True,
    }


def _measure_norm_commutation(rows: Any, encoder: Any) -> tuple[float, float, int]:
    """Row fluctuation of the captured-energy ratio ``||h E|| / ||h||``.

    RMSNorm does not commute with a rectangular chart.  For orthonormal ``E``
    the child RMS rescales by ``(||h E|| / ||h||) * sqrt(Hp / Hc)``; a constant
    ratio is a single scalar the fitted RMSNorm diagonal absorbs exactly, and
    the row-wise fluctuation is the part no diagonal can absorb.

    A boundary row with exactly zero energy carries no ratio at all — a padded
    embedding row is the ordinary case — so those rows are excluded and counted
    rather than being allowed to poison the statistic or to fail the crossing.
    """

    torch = _torch()
    parent_norm = torch.linalg.vector_norm(rows, dim=1)
    encoded_norm = torch.linalg.vector_norm(rows @ encoder, dim=1)
    live = parent_norm > 0.0
    measured = int(live.sum().item())
    if measured == 0:
        raise CrossWidthResidualChartError(
            "Every cached boundary row has zero energy; norm commutation is "
            "undefined at this boundary."
        )
    ratio = encoded_norm[live] / parent_norm[live]
    mean = float(ratio.mean().item())
    fluctuation = float(ratio.std(unbiased=False).item())
    if not math.isfinite(mean) or not math.isfinite(fluctuation):
        raise CrossWidthResidualChartError("Norm-commutation diagnostics are non-finite.")
    return mean, fluctuation, measured


def _absorption_split(demand: Any, capacity: int | None) -> tuple[float, float, int]:
    """Split ``A_b`` energy into the writer-representable part and the rest."""

    torch = _torch()
    singular = torch.linalg.svdvals(demand)
    total = float((singular * singular).sum().item())
    width = int(demand.shape[0])
    rank = width if capacity is None else min(width, int(capacity))
    absorbed = float((singular[:rank] * singular[:rank]).sum().item())
    if total <= 0.0:
        return 0.0, 0.0, rank
    return absorbed / total, 1.0 - absorbed / total, rank


def _seal_chart(
    *,
    boundary_id: str,
    encoder: Any,
    decoder: Any,
    interface_locked: bool,
    diagnostics: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> ResidualBoundaryChart:
    body = {
        "schema": CHART_SCHEMA,
        "boundary_id": boundary_id,
        "source_width": int(encoder.shape[0]),
        "target_width": int(encoder.shape[1]),
        "interface_locked": interface_locked,
        "encoder_sha256": _tensor_sha256(encoder),
        "decoder_sha256": _tensor_sha256(decoder),
        "diagnostics": dict(diagnostics),
        "lineage": dict(lineage),
        "fit_rows_only": True,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "autograd_used": False,
    }
    return ResidualBoundaryChart(
        boundary_id=boundary_id,
        encoder=encoder,
        decoder=decoder,
        interface_locked=interface_locked,
        report={**body, "chart_sha256": canonical_sha256(body)},
    )


def _seal_chart_set(
    *,
    parent_evidence_sha256: str,
    child_topology_sha256: str,
    charts: Sequence[ResidualBoundaryChart],
    policy: ResidualBoundaryChartPolicy,
    method: str,
    extra: Mapping[str, Any],
) -> ResidualBoundaryChartSet:
    ordered = tuple(charts)
    body = {
        "schema": CHART_SET_SCHEMA,
        "method": method,
        "parent_evidence_sha256": parent_evidence_sha256,
        "child_topology_sha256": child_topology_sha256,
        "source_width": ordered[0].source_width,
        "target_width": ordered[0].target_width,
        "boundary_ids": [value.boundary_id for value in ordered],
        "charts": [value.chart_sha256 for value in ordered],
        "interface_locked_boundary_ids": [
            value.boundary_id for value in ordered if value.interface_locked
        ],
        "policy": policy.as_dict(),
        "policy_sha256": policy.policy_sha256,
        "chart_reports": [dict(value.report) for value in ordered],
        "parent_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
        "cached_parent_evidence_only": True,
        **dict(extra),
    }
    return ResidualBoundaryChartSet(
        parent_evidence_sha256=parent_evidence_sha256,
        child_topology_sha256=child_topology_sha256,
        charts=ordered,
        report={**body, "set_sha256": canonical_sha256(body)},
    )


def _fit_decoder(
    encoder: Any,
    fit_rows: Any,
    policy: ResidualBoundaryChartPolicy,
) -> tuple[Any, dict[str, Any]]:
    """Cycle-anchored ridge decoder: ridge from encoded rows back to parent."""

    reciprocal, reciprocal_report = _regularized_reciprocal(
        encoder,
        ridge_ratio=policy.chart_ridge_ratio,
        absolute_ridge_floor=policy.absolute_ridge_floor,
    )
    decoder, decoder_report = _anchored_ridge(
        fit_rows @ encoder,
        fit_rows,
        reciprocal,
        ridge_ratio=(policy.chart_ridge_ratio + policy.cycle_consistency_weight),
        absolute_ridge_floor=policy.absolute_ridge_floor,
    )
    return decoder, {
        "regularized_reciprocal": reciprocal_report,
        "cycle_anchored_ridge": decoder_report,
        "cycle_consistency_anchor": "regularized_reciprocal_of_the_encoder",
    }


def _boundary_defects(
    *,
    boundary_id: str,
    encoder: Any,
    decoder: Any,
    fit_rows: Any,
    next_encoder: Any | None,
    previous_decoder: Any | None,
    capacity: int | None,
    policy: ResidualBoundaryChartPolicy,
) -> dict[str, Any]:
    """Measure every compatibility defect the design bounds, then bound them."""

    torch = _torch()
    target_width = int(encoder.shape[1])
    orthonormality = float(
        (encoder.T @ encoder - torch.eye(target_width, dtype=torch.float64))
        .abs()
        .max()
        .item()
    )
    if orthonormality > policy.orthonormality_tolerance:
        raise CrossWidthResidualChartError(
            f"Chart {boundary_id!r} encoder is not orthonormal: {orthonormality!r}."
        )
    cycle = _relative_frobenius((fit_rows @ encoder) @ decoder - fit_rows, fit_rows)
    if not math.isfinite(cycle) or cycle > policy.max_cycle_defect:
        raise CrossWidthResidualChartRankError(
            f"Chart {boundary_id!r} cycle defect {cycle!r} exceeds the policy "
            f"bound {policy.max_cycle_defect!r}; decode error would make chart "
            "transition absorption unreliable."
        )
    mean_ratio, fluctuation, measured_rows = _measure_norm_commutation(
        fit_rows, encoder
    )
    if fluctuation > policy.max_norm_commutation_fluctuation:
        raise CrossWidthResidualChartRankError(
            f"Chart {boundary_id!r} norm-commutation fluctuation {fluctuation!r} "
            f"exceeds the policy bound "
            f"{policy.max_norm_commutation_fluctuation!r}."
        )
    diagnostics: dict[str, Any] = {
        "encoder_orthonormality_max_abs_error": orthonormality,
        "cycle_defect": cycle,
        "max_cycle_defect": policy.max_cycle_defect,
        "norm_commutation_mean_ratio": mean_ratio,
        "norm_commutation_row_fluctuation": fluctuation,
        "norm_commutation_row_variance": fluctuation * fluctuation,
        "max_norm_commutation_row_fluctuation": (
            policy.max_norm_commutation_fluctuation
        ),
        "norm_commutation_measured_rows": measured_rows,
        "fit_rows": int(fit_rows.shape[0]),
        "writer_representable_rank": capacity,
    }
    if previous_decoder is not None:
        # ``T_b = E_{b+1}^T D_b^T - I_Hc`` measured on the *incoming* chart
        # pair, where this boundary plays ``b+1`` and ``previous_decoder`` is
        # ``D_b``.
        transition = encoder.T @ previous_decoder.T - torch.eye(
            target_width, dtype=torch.float64
        )
        diagnostics["incoming_transition_defect_frobenius"] = float(
            torch.linalg.matrix_norm(transition, ord="fro").item()
        )
        diagnostics["incoming_transition_defect_max_abs"] = float(
            transition.abs().max().item()
        )
    if next_encoder is None:
        diagnostics["absorption_demand"] = None
        return diagnostics
    demand = decoder @ (next_encoder - encoder)
    absorbed, unabsorbed, rank = _absorption_split(demand, capacity)
    if unabsorbed > policy.max_unabsorbed_transition_energy_fraction:
        raise CrossWidthResidualChartRankError(
            f"Chart {boundary_id!r} demands {unabsorbed!r} of its chart "
            "transition outside the writer's representable rank "
            f"{rank!r}; the policy bound is "
            f"{policy.max_unabsorbed_transition_energy_fraction!r}."
        )
    diagnostics["absorption_demand"] = {
        "sha256": _tensor_sha256(demand),
        "frobenius": float(torch.linalg.matrix_norm(demand, ord="fro").item()),
        "absorbed_energy_fraction": absorbed,
        "unabsorbed_energy_fraction": unabsorbed,
        "writer_representable_rank": rank,
        "max_unabsorbed_energy_fraction": (
            policy.max_unabsorbed_transition_energy_fraction
        ),
    }
    return diagnostics


def compile_residual_boundary_chart_set(
    evidence: HFCausalCircuitEvidence,
    parent_model: Any,
    child: Any,
    *,
    policy: ResidualBoundaryChartPolicy | None = None,
    interface_locked_encoders: Mapping[str, Any] | None = None,
) -> ResidualBoundaryChartSet:
    """Compile one sealed chart per residual-write boundary, fit rows only.

    ``parent_model`` supplies the residual-facing weights the balanced basis
    needs as its structural controllability and downstream observability
    lanes; only already-materialized tensors are read.  ``child`` is the child
    model or its :class:`DecoderLayout`; it fixes ``Hc`` and every writer
    capacity the absorption-demand bound is measured against.
    """

    torch = _torch()
    from .dynamic_balanced_circuit_basis import (
        BalancedCircuitBasisError,
        DownstreamObservability,
        EvidenceSplit,
        ResidualSnapshotBlock,
        build_balanced_circuit_basis,
    )

    if not isinstance(evidence, HFCausalCircuitEvidence):
        raise CrossWidthResidualChartError(
            "Chart selection requires a sealed HFCausalCircuitEvidence bundle."
        )
    evidence.verify()
    active_policy = policy or ResidualBoundaryChartPolicy()
    if not isinstance(active_policy, ResidualBoundaryChartPolicy):
        raise CrossWidthResidualChartError(
            "policy must be a ResidualBoundaryChartPolicy."
        )
    child_layout = (
        child if isinstance(child, DecoderLayout) else build_decoder_layout(child)
    )
    parent_layout = build_decoder_layout(parent_model)
    source_width = int(evidence.architecture.hidden_size)
    target_width = int(child_layout.hidden_size)
    if int(parent_layout.hidden_size) != source_width:
        raise CrossWidthResidualChartError(
            "The supplied parent model does not match the sealed evidence width."
        )
    if target_width > source_width:
        raise CrossWidthResidualChartRankError(
            "Residual width expansion is outside the v1 chart envelope; the "
            "balanced basis is a contraction construction."
        )
    if len(child_layout.blocks) != len(evidence.layers):
        raise CrossWidthResidualChartError(
            "Chart selection requires one child layer per cached parent layer."
        )
    boundaries = residual_boundary_ids(len(evidence.layers))
    locked = set(interface_locked_boundary_ids(len(evidence.layers)))
    supplied = dict(interface_locked_encoders or {})
    foreign = sorted(set(supplied) - locked)
    if foreign:
        raise CrossWidthResidualChartError(
            "Chart selection never modifies an interface-locked boundary and "
            "refuses supplied charts at ordinary boundaries: "
            + ", ".join(foreign)
            + "."
        )

    structural, downstream = _residual_facing_operators(parent_model, parent_layout)
    fit_mask = evidence.fit_token_mask
    validation_mask = evidence.validation_token_mask
    encoders: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    fit_rows_by_boundary: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    previous_boundary_id: str | None = None
    for boundary_id in boundaries:
        sequences = _sequences(
            _boundary_rows(evidence, boundary_id), f"{boundary_id} parent rows"
        )
        fit_rows = _masked_rows(sequences, fit_mask)
        heldout_rows = _masked_rows(sequences, validation_mask)
        fit_rows_by_boundary[boundary_id] = fit_rows
        provided = supplied.get(boundary_id)
        if provided is not None:
            encoder = _matrix(provided, f"{boundary_id} supplied encoder")
            if tuple(encoder.shape) != (source_width, target_width):
                raise CrossWidthResidualChartError(
                    f"Supplied interface chart {boundary_id!r} has the wrong shape."
                )
            error = float(
                (
                    encoder.T @ encoder
                    - torch.eye(target_width, dtype=torch.float64)
                )
                .abs()
                .max()
                .item()
            )
            if error > active_policy.orthonormality_tolerance:
                raise CrossWidthResidualChartError(
                    f"Supplied interface chart {boundary_id!r} is not an "
                    "orthogonal rotation of the interface subspace."
                )
            selection[boundary_id] = {
                "source": "interface_locked_supplied_chart",
                "encoder_sha256": _tensor_sha256(encoder),
            }
        else:
            try:
                basis = build_balanced_circuit_basis(
                    structural,
                    (
                        ResidualSnapshotBlock(
                            block_id=f"{boundary_id}.fit",
                            split=EvidenceSplit.FIT,
                            phase="cached_parent_circuit",
                            layer=boundary_id,
                            endpoint="residual",
                            rows=fit_rows,
                            record_ids=(f"{evidence.cache_key}:fit",),
                        ),
                    ),
                    (
                        ResidualSnapshotBlock(
                            block_id=f"{boundary_id}.heldout",
                            split=EvidenceSplit.HELDOUT,
                            phase="cached_parent_circuit",
                            layer=boundary_id,
                            endpoint="residual",
                            rows=heldout_rows,
                            record_ids=(f"{evidence.cache_key}:validation",),
                        ),
                    ),
                    tuple(
                        DownstreamObservability(
                            evidence_id=f"{boundary_id}->{operator.name}",
                            node_id=operator.name,
                            circuit_kind="residual_consumer",
                            operator=operator,
                        )
                        for operator in downstream[boundary_id]
                    ),
                    target_width,
                    structural_controllability_weight=(
                        active_policy.structural_controllability_weight
                    ),
                    cached_controllability_weight=(
                        active_policy.cached_controllability_weight
                    ),
                    gram_regularization=active_policy.gram_regularization,
                    rank_tolerance=active_policy.rank_tolerance,
                    max_condition_number=active_policy.max_condition_number,
                )
            except BalancedCircuitBasisError as exc:
                raise CrossWidthResidualChartRankError(
                    f"Balanced basis selection failed closed at {boundary_id!r}: "
                    f"{exc}"
                ) from exc
            encoder = basis.basis
            selection[boundary_id] = {
                "source": "regularized_balanced_pod",
                "canonical_encoder_sha256": _tensor_sha256(encoder),
                "encoder_sha256": _tensor_sha256(encoder),
                "balanced_basis_run_sha256": basis.report["run_sha256"],
                "balanced_basis_selection_sha256": basis.report["hashes"][
                    "selection_sha256"
                ],
                "heldout_retained_energy_fraction": basis.report["retention"][
                    "heldout_cached_residual_fraction"
                ],
                "fit_retained_energy_fraction": basis.report["retention"][
                    "fit_cached_residual_fraction"
                ],
                "downstream_observability_fraction": basis.report["retention"][
                    "downstream_observability_fraction"
                ],
            }
            anchor = encoders.get(previous_boundary_id) if previous_boundary_id else None
            if anchor is not None:
                encoder, transport_report = _procrustes_frame_transport(
                    encoder, anchor
                )
                selection[boundary_id]["frame_transport"] = transport_report
                selection[boundary_id]["encoder_sha256"] = _tensor_sha256(encoder)
            else:
                selection[boundary_id]["frame_transport"] = {
                    "kind": "anchor_boundary_keeps_its_canonical_frame",
                }
        encoders[boundary_id] = encoder
        previous_boundary_id = boundary_id
        decoder, decoder_report = _fit_decoder(encoder, fit_rows, active_policy)
        decoders[boundary_id] = decoder
        selection[boundary_id]["decoder"] = decoder_report

    charts: list[ResidualBoundaryChart] = []
    for index, boundary_id in enumerate(boundaries):
        next_id = boundaries[index + 1] if index + 1 < len(boundaries) else None
        previous_id = boundaries[index - 1] if index > 0 else None
        diagnostics = _boundary_defects(
            boundary_id=boundary_id,
            encoder=encoders[boundary_id],
            decoder=decoders[boundary_id],
            fit_rows=fit_rows_by_boundary[boundary_id],
            next_encoder=None if next_id is None else encoders[next_id],
            previous_decoder=(
                None if previous_id is None else decoders[previous_id]
            ),
            capacity=_writer_capacity(child_layout, boundary_id),
            policy=active_policy,
        )
        charts.append(
            _seal_chart(
                boundary_id=boundary_id,
                encoder=encoders[boundary_id],
                decoder=decoders[boundary_id],
                interface_locked=boundary_id in locked,
                diagnostics=diagnostics,
                lineage={
                    "cached_parent_evidence_sha256": evidence.cache_key,
                    "parent_graph_fingerprint": evidence.parent_graph_fingerprint,
                    "policy_sha256": active_policy.policy_sha256,
                    "selection": selection[boundary_id],
                },
            )
        )
    return _seal_chart_set(
        parent_evidence_sha256=evidence.cache_key,
        child_topology_sha256=child_residual_topology_sha256(child_layout),
        charts=charts,
        policy=active_policy,
        method="regularized-balanced-pod-encoder-cycle-anchored-ridge-decoder",
        extra={
            "parent_graph_fingerprint": evidence.parent_graph_fingerprint,
            "residual_width_expansion": False,
            "single_global_chart": False,
            "one_chart_per_residual_write_boundary": True,
        },
    )


def identity_residual_boundary_chart_set(
    evidence: HFCausalCircuitEvidence,
    child: Any,
    *,
    policy: ResidualBoundaryChartPolicy | None = None,
) -> ResidualBoundaryChartSet:
    """Width-preserving charts: every ``E_b`` and ``D_b`` is the identity.

    This is the control the cross-width machinery is measured against.  Every
    defect is exactly zero, the sequential target reduces to the parent's own
    residual difference, and the chart-transformed operator therefore has to
    reproduce the parent-coordinate operator it wraps.
    """

    torch = _torch()
    if not isinstance(evidence, HFCausalCircuitEvidence):
        raise CrossWidthResidualChartError(
            "Chart selection requires a sealed HFCausalCircuitEvidence bundle."
        )
    evidence.verify()
    active_policy = policy or ResidualBoundaryChartPolicy()
    child_layout = (
        child if isinstance(child, DecoderLayout) else build_decoder_layout(child)
    )
    width = int(evidence.architecture.hidden_size)
    if int(child_layout.hidden_size) != width:
        raise CrossWidthResidualChartError(
            "Identity charts require equal parent and child residual widths."
        )
    boundaries = residual_boundary_ids(len(evidence.layers))
    locked = set(interface_locked_boundary_ids(len(evidence.layers)))
    identity = torch.eye(width, dtype=torch.float64)
    fit_mask = evidence.fit_token_mask
    encoders = {boundary: identity.clone() for boundary in boundaries}
    charts: list[ResidualBoundaryChart] = []
    for index, boundary_id in enumerate(boundaries):
        next_id = boundaries[index + 1] if index + 1 < len(boundaries) else None
        previous_id = boundaries[index - 1] if index > 0 else None
        fit_rows = _masked_rows(
            _sequences(
                _boundary_rows(evidence, boundary_id), f"{boundary_id} parent rows"
            ),
            fit_mask,
        )
        diagnostics = _boundary_defects(
            boundary_id=boundary_id,
            encoder=encoders[boundary_id],
            decoder=identity.clone(),
            fit_rows=fit_rows,
            next_encoder=None if next_id is None else encoders[next_id],
            previous_decoder=None if previous_id is None else identity.clone(),
            capacity=_writer_capacity(child_layout, boundary_id),
            policy=active_policy,
        )
        charts.append(
            _seal_chart(
                boundary_id=boundary_id,
                encoder=encoders[boundary_id],
                decoder=identity.clone(),
                interface_locked=boundary_id in locked,
                diagnostics=diagnostics,
                lineage={
                    "cached_parent_evidence_sha256": evidence.cache_key,
                    "parent_graph_fingerprint": evidence.parent_graph_fingerprint,
                    "policy_sha256": active_policy.policy_sha256,
                    "selection": {"source": "width_preserving_identity_chart"},
                },
            )
        )
    return _seal_chart_set(
        parent_evidence_sha256=evidence.cache_key,
        child_topology_sha256=child_residual_topology_sha256(child_layout),
        charts=charts,
        policy=active_policy,
        method="width-preserving-identity-charts",
        extra={
            "parent_graph_fingerprint": evidence.parent_graph_fingerprint,
            "residual_width_expansion": False,
            "single_global_chart": False,
            "one_chart_per_residual_write_boundary": True,
            "width_preserving_control": True,
        },
    )


# ---------------------------------------------------------------------------
# The sequential branch target
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SequentialBranchTarget:
    """``Y = h_{b+1} E_{b+1} - hhat_b_actual`` plus its measured expansion."""

    input_boundary_id: str
    output_boundary_id: str
    input_state: Any
    normalized_target: Any
    endpoint_target: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def target_sha256(self) -> str:
        return str(self.report["endpoint_target_sha256"])


def build_sequential_branch_target(
    *,
    input_chart: ResidualBoundaryChart,
    output_chart: ResidualBoundaryChart,
    parent_input_state: Any,
    parent_output_state: Any,
    parent_branch_delta: Any,
    parent_normalized_input: Any,
    child_input_state: Any,
    fit_token_mask: Any,
) -> SequentialBranchTarget:
    """Build the one load-bearing cross-width branch target.

    The sequential form ``Y = h_{b+1} E_{b+1} - hhat_b_actual`` is the value
    that is fitted.  Its expansion

    ``Y = Delta E_{b+1} + h_b (E_{b+1} - E_b) - (hhat_b_actual - h_b E_b)``

    carries three separately measurable responsibilities: the parent's write
    transported into the downstream chart, the chart transition the ordinary
    writer must absorb because the child skip path is the identity, and the
    correction of accumulated upstream error.  Dropping that last term is
    exactly the A0025 failure mode, so it is measured here and reported.
    """

    torch = _torch()
    if not isinstance(input_chart, ResidualBoundaryChart) or not isinstance(
        output_chart, ResidualBoundaryChart
    ):
        raise CrossWidthResidualChartError(
            "Sequential branch targets require two sealed boundary charts."
        )
    input_chart.verify()
    output_chart.verify()
    if (
        input_chart.source_width != output_chart.source_width
        or input_chart.target_width != output_chart.target_width
    ):
        raise CrossWidthResidualChartError(
            "The two boundary charts of one branch must share their widths."
        )
    parent_input = _sequences(parent_input_state, "parent_input_state")
    parent_output = _sequences(parent_output_state, "parent_output_state")
    parent_delta = _sequences(parent_branch_delta, "parent_branch_delta")
    parent_normalized = _sequences(parent_normalized_input, "parent_normalized_input")
    child_state = _sequences(child_input_state, "child_input_state")
    if (
        int(parent_input.shape[-1]) != input_chart.source_width
        or tuple(parent_output.shape) != tuple(parent_input.shape)
        or tuple(parent_delta.shape) != tuple(parent_input.shape)
        or tuple(parent_normalized.shape) != tuple(parent_input.shape)
    ):
        raise CrossWidthResidualChartError(
            "Cached parent branch evidence does not share one residual shape."
        )
    if (
        tuple(child_state.shape[:2]) != tuple(parent_input.shape[:2])
        or int(child_state.shape[-1]) != input_chart.target_width
    ):
        raise CrossWidthResidualChartError(
            "The recaptured child state does not match the cached batch, "
            "sequence, and child residual width."
        )
    encoded_input = parent_input @ input_chart.encoder
    endpoint_target = parent_output @ output_chart.encoder - child_state
    transported_delta = parent_delta @ output_chart.encoder
    chart_transition = parent_input @ (output_chart.encoder - input_chart.encoder)
    upstream_error = child_state - encoded_input
    expansion = transported_delta + chart_transition - upstream_error
    expansion_error = float((endpoint_target - expansion).abs().max().item())
    reference = float(endpoint_target.abs().max().item())
    fit = fit_token_mask.reshape(-1).to(torch.bool)

    def energy(value: Any) -> float:
        return float(_masked_rows(value, fit).square().sum().item())

    total = energy(endpoint_target)
    report = {
        "schema": CROSS_WIDTH_BRANCH_FOLD_SCHEMA,
        "component": "sequential_branch_target",
        "equation": "Y = h_next @ E_next - hhat_actual",
        "expansion": (
            "Y = Delta @ E_next + h @ (E_next - E_b) - (hhat_actual - h @ E_b)"
        ),
        "input_boundary_id": input_chart.boundary_id,
        "output_boundary_id": output_chart.boundary_id,
        "input_chart_sha256": input_chart.chart_sha256,
        "output_chart_sha256": output_chart.chart_sha256,
        "sequential_child_replay": True,
        "parent_residual_coordinate_endpoints": False,
        "upstream_error_correction_included": True,
        "fit_rows": int(fit.sum().item()),
        "fit_transported_parent_delta_energy": energy(transported_delta),
        "fit_chart_transition_energy": energy(chart_transition),
        "fit_upstream_error_energy": energy(upstream_error),
        "fit_endpoint_target_energy": total,
        "fit_upstream_error_energy_ratio": (
            energy(upstream_error) / total if total > 0.0 else 0.0
        ),
        "sequential_expansion_max_abs_error": expansion_error,
        "sequential_expansion_reference_max_abs": reference,
        "endpoint_target_sha256": _tensor_sha256(endpoint_target),
        "child_input_state_sha256": _tensor_sha256(child_state),
    }
    return SequentialBranchTarget(
        input_boundary_id=input_chart.boundary_id,
        output_boundary_id=output_chart.boundary_id,
        input_state=child_state,
        normalized_target=(parent_normalized @ input_chart.encoder).contiguous(),
        endpoint_target=endpoint_target.contiguous(),
        report=report,
    )


# ---------------------------------------------------------------------------
# Sequential recapture of the live child
# ---------------------------------------------------------------------------


class LiveChildBoundaryRecapture:
    """Recapture the live child's residual state at every sealed boundary.

    The operator must never fall back to parent-coordinate inputs — that is
    precisely the A0025 failure — so this callable is a required collaborator
    and it fails closed rather than substituting anything.
    """

    def __init__(self, model_inputs: Mapping[str, Any] | Any) -> None:
        torch = _torch()
        inputs = (
            dict(model_inputs)
            if isinstance(model_inputs, Mapping)
            else {"input_ids": model_inputs}
        )
        if "input_ids" not in inputs:
            raise CrossWidthResidualChartError(
                "Child recapture requires child input_ids."
            )
        input_ids = torch.as_tensor(inputs["input_ids"])
        if input_ids.ndim != 2:
            raise CrossWidthResidualChartError("Child input_ids must be rank 2.")
        inputs["input_ids"] = input_ids
        inputs.setdefault("use_cache", False)
        self.model_inputs = MappingProxyType(inputs)
        self.recapture_calls = 0

    @property
    def inputs_sha256(self) -> str:
        torch = _torch()
        return canonical_sha256(
            {
                "schema": CROSS_WIDTH_BRANCH_FOLD_SCHEMA,
                "component": "child_recapture_inputs",
                "input_ids": [
                    int(value)
                    for value in torch.as_tensor(self.model_inputs["input_ids"])
                    .reshape(-1)
                    .tolist()
                ],
                "shape": [
                    int(size)
                    for size in torch.as_tensor(
                        self.model_inputs["input_ids"]
                    ).shape
                ],
            }
        )

    def __call__(self, child_model: Any, boundary_ids: Sequence[str]) -> dict[str, Any]:
        from .hf_transport_trace import TransportTraceError, collect_hf_transport_trace

        wanted = tuple(boundary_ids)
        if not wanted:
            raise CrossWidthResidualChartError(
                "Child recapture needs at least one boundary id."
            )
        try:
            trace = collect_hf_transport_trace(child_model, dict(self.model_inputs))
        except (TransportTraceError, RuntimeError, ValueError) as exc:
            raise CrossWidthResidualChartError(
                "Sequential child recapture is unavailable; the cross-width "
                "operator refuses to fall back to parent-coordinate inputs."
            ) from exc
        self.recapture_calls += 1
        states: dict[str, Any] = {}
        for block in trace.blocks:
            states[residual_boundary_id(int(block.index), ATTENTION_INPUT_BOUNDARY)] = (
                block.block_input
            )
            states[residual_boundary_id(int(block.index), MLP_INPUT_BOUNDARY)] = (
                block.post_attention_residual
            )
        states[TERMINAL_BOUNDARY_ID] = trace.blocks[-1].block_output
        missing = [value for value in wanted if value not in states]
        if missing:
            raise CrossWidthResidualChartError(
                "Sequential child recapture is missing boundaries: "
                + ", ".join(missing)
                + "."
            )
        return {
            value: _sequences(states[value], f"child {value} recapture")
            for value in wanted
        }


# ---------------------------------------------------------------------------
# Reverse reclosure sweep (hook)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReverseReclosureSweepPlan:
    """The one reverse sweep the design prescribes after the forward pass.

    Charts stay frozen; only writers are re-solved, back to front, against
    terminal-anchored targets.  This wave seals the schedule and reports that
    it has *not* run, so no artifact can silently claim a reconciliation that
    never happened.
    """

    terminal_boundary_id: str
    ordered_boundary_ids: tuple[str, ...]
    charts_frozen: bool
    executed: bool
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def sweep_sha256(self) -> str:
        return str(self.report["sweep_sha256"])


def plan_reverse_reclosure_sweep(
    chart_set: ResidualBoundaryChartSet,
) -> ReverseReclosureSweepPlan:
    """Seal the reverse writer-only sweep schedule for one chart set."""

    if not isinstance(chart_set, ResidualBoundaryChartSet):
        raise CrossWidthResidualChartError(
            "A reverse sweep plan requires a sealed chart set."
        )
    chart_set.verify()
    ordered = tuple(
        value
        for value in reversed(chart_set.boundary_ids)
        if value != TERMINAL_BOUNDARY_ID
    )
    body = {
        "schema": REVERSE_SWEEP_SCHEMA,
        "chart_set_sha256": chart_set.set_sha256,
        "terminal_boundary_id": TERMINAL_BOUNDARY_ID,
        "ordered_boundary_ids": list(ordered),
        "charts_frozen": True,
        "writers_only": True,
        "executed": False,
        "one_forward_analytic_sweep_then_one_reverse_reconciliation": True,
        "joint_global_optimization": False,
    }
    return ReverseReclosureSweepPlan(
        terminal_boundary_id=TERMINAL_BOUNDARY_ID,
        ordered_boundary_ids=ordered,
        charts_frozen=True,
        executed=False,
        report={**body, "sweep_sha256": canonical_sha256(body)},
    )


__all__ = [
    "ATTENTION_INPUT_BOUNDARY",
    "CHART_SCHEMA",
    "CHART_SET_SCHEMA",
    "CROSS_WIDTH_BRANCH_FOLD_SCHEMA",
    "MLP_INPUT_BOUNDARY",
    "REVERSE_SWEEP_SCHEMA",
    "TERMINAL_BOUNDARY_ID",
    "CrossWidthResidualChartAuthenticationError",
    "CrossWidthResidualChartError",
    "CrossWidthResidualChartRankError",
    "LiveChildBoundaryRecapture",
    "ResidualBoundaryChart",
    "ResidualBoundaryChartPolicy",
    "ResidualBoundaryChartSet",
    "ReverseReclosureSweepPlan",
    "SequentialBranchTarget",
    "build_sequential_branch_target",
    "child_residual_topology_sha256",
    "compile_residual_boundary_chart_set",
    "identity_residual_boundary_chart_set",
    "interface_locked_boundary_ids",
    "plan_reverse_reclosure_sweep",
    "residual_boundary_id",
    "residual_boundary_ids",
]
