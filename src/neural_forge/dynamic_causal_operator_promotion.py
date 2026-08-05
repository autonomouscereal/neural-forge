"""Executable fold operators for the family-neutral causal kernels.

The planner already names the holes these operators fill: a per-layer
``cross_family_closed_loop_layer_operator`` gap whenever no same-family layer
operator matches, and a ``token_interface_bridge`` method whose math is proven
but which had no registered executable.  This module promotes the existing
family-neutral kernels into ``FoldOperator`` implementations so both selections
become executable transactions.

Every operator is fold-only: one sealed parent evidence bundle in, closed-form
solves, one snapshot-backed atomic install out.  No parent forward, optimizer,
gradient, pruning, quantization, parameter sharing, or runtime sidecar is used,
and every unproven source/target crossing fails closed before any mutation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .dynamic_circuit_ir import CircuitKind, ModelCircuitGraph, canonical_sha256
from .dynamic_fold_executor import (
    DynamicFoldExecutionError,
    FoldEvidence,
    FoldExecutionContext,
)
from .dynamic_fold_planner import (
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
    CAUSAL_GATED_MLP_BRANCH_METHOD,
    CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
    CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
    CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
    CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
    NATIVE_RESIDUAL_CHART_INTERFACE_METHOD,
    FoldAction,
)
from .dynamic_native_circuit_evidence import (
    NativeCircuitEvidence,
    NativeResidualChart,
    node_module_path,
    owning_module_path,
)
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig, capture_model_state
from .hf_arch_layout import DecoderBlockLayout, DecoderLayout, build_decoder_layout
from .hf_causal_circuit_evidence import HFCausalCircuitEvidence
from .hf_causal_cross_width_charts import (
    ATTENTION_INPUT_BOUNDARY,
    CROSS_WIDTH_BRANCH_FOLD_SCHEMA,
    MLP_INPUT_BOUNDARY,
    TERMINAL_BOUNDARY_ID,
    CrossWidthResidualChartError,
    ResidualBoundaryChart,
    ResidualBoundaryChartSet,
    SequentialBranchTarget,
    build_sequential_branch_target,
    child_residual_topology_sha256,
    plan_reverse_reclosure_sweep,
    residual_boundary_id,
)
from .hf_causal_gqa_attention_fold import (
    HFCausalGQATopology,
    apply_rope,
    causal_attention_probabilities,
    compile_hf_causal_gqa_attention_fold,
)
from .hf_causal_kda_fold import (
    ChimeraKDAFoldPolicy,
    ChimeraKDAGeometry,
    ChimeraKDALayerEvidence,
    HFCausalKDAFoldError,
    chimera_kda_geometry,
    chimera_kda_parameters,
    compile_hf_causal_kda_fold,
    kda_effective_weights,
)
from .hf_causal_latent_kv_attention_fold import (
    ChimeraLatentKVFoldPolicy,
    ChimeraLatentKVGeometry,
    ChimeraLatentKVLayerEvidence,
    HFCausalLatentKVAttentionFoldError,
    chimera_latent_kv_geometry,
    chimera_latent_kv_parameters,
    compile_hf_causal_latent_kv_attention_fold,
    latent_kv_consistency,
)
from .hf_causal_sparse_latent_fold import (
    ChimeraSparseLatentFoldPolicy,
    ChimeraSparseLatentGeometry,
    ChimeraSummaryAttentionLayerEvidence,
    HFCausalSparseLatentFoldError,
    chimera_sparse_latent_geometry,
    compile_hf_causal_sparse_latent_fold,
    split_summary_branch_delta,
    summary_attention_probabilities,
    summary_block_mask,
    summary_pooling_matrix,
)
from .hf_causal_stable_latent_moe_fold import (
    ChimeraStableLatentMoEFoldPolicy,
    ChimeraStableLatentMoEGeometry,
    ChimeraStableLatentMoELayerEvidence,
    HFCausalStableLatentMoEFoldError,
    chimera_stable_latent_moe_geometry,
    chimera_stable_latent_moe_parameters,
    compile_hf_causal_stable_latent_moe_fold,
    stable_latent_moe_consistency,
)
from .hf_closed_form_causal_circuit_fold import (
    fit_functional_swiglu,
    fit_rmsnorm_diagonal,
)
from .hf_closed_form_fit import fit_diagonal_norm, fit_ridge_linear, normalize_rows
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    compile_strict_vocabulary_operators,
    load_strict_tokenizer_bridge_artifact,
)
from .hf_tokenizer_pair_bridge import (
    TokenizerPairBridgeArtifact,
    compile_pair_bridge_anchor_transport,
    compile_pair_bridge_length_normalized_transport,
    compile_pair_bridge_occurrence_weighted_transport,
    load_hf_tokenizer_pair_bridge_artifact,
)


SCHEMA = "neural-forge-dynamic-causal-operator-promotion-v1"

TOKEN_INTERFACE_BRIDGE_METHOD = "token_interface_bridge"

_GATED_ACTIVATIONS = frozenset({"silu", "swish"})

_ANCHOR_PAIR_TRANSPORT = "anchor-exact"
_LENGTH_NORMALIZED_PAIR_TRANSPORT = "length-normalized"
_OCCURRENCE_PAIR_TRANSPORT = "occurrence-weighted"
#: Take the byte-exact anchor row wherever one exists and the length-normalized
#: code average everywhere else.  Both agree on rows whose only incidence is a
#: single direct code, so the blend only ever *upgrades* a mixed row to its
#: exact anchor.
ANCHOR_PREFERRED_PAIR_TRANSPORT = "anchor-preferred"
_PAIR_TRANSPORTS = (
    ANCHOR_PREFERRED_PAIR_TRANSPORT,
    _ANCHOR_PAIR_TRANSPORT,
    _LENGTH_NORMALIZED_PAIR_TRANSPORT,
    _OCCURRENCE_PAIR_TRANSPORT,
)
#: A pair bridge does not reach every target row.  The builder returns exact
#: zeros there; installing them would be an unproven claim, so they are left at
#: the child's incumbent (blank) value and recorded.
_PAIR_MASKED_ROW_POLICY = "leave-uncovered-target-rows-at-incumbent-blank"
_UNCOVERED_ROW_INDEX_LIMIT = 64


class DynamicCausalOperatorPromotionError(DynamicFoldExecutionError):
    """A promoted causal operator is outside its proven fold envelope."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicCausalOperatorPromotionError(
            "PyTorch is required for the promoted causal fold operators."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicCausalOperatorPromotionError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DynamicCausalOperatorPromotionError(
            f"{name} must be non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise DynamicCausalOperatorPromotionError(
            f"{name} must be finite and non-negative."
        )
    return result


def _masked_rows(values: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in values.shape[2:])
    return values.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _rmsnorm(values: Any, weight: Any, epsilon: float) -> Any:
    torch = _torch()
    rows = values.detach().to(device="cpu", dtype=torch.float64)
    scale = (rows.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()
    return (rows * scale * weight.reshape(1, 1, -1)).contiguous()


def _relative_rmse(prediction: Any, target: Any) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _fold_only_operation_audit() -> dict[str, Any]:
    return {
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "parameter_sharing_operations": 0,
        "sidecar_parameters": 0,
        "cached_parent_evidence_only": True,
    }


def _sealed_causal_evidence(
    evidence: FoldEvidence,
    role: str,
) -> HFCausalCircuitEvidence:
    payload = evidence.payload
    if not isinstance(payload, HFCausalCircuitEvidence):
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires a sealed HFCausalCircuitEvidence payload."
        )
    if payload.cache_key != evidence.fingerprint:
        raise DynamicCausalOperatorPromotionError(
            f"{role} evidence payload and fingerprint differ."
        )
    return payload.verify()


def _single_parent(action: FoldAction, context: FoldExecutionContext, role: str) -> str:
    if len(action.sources) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} accepts exactly one parent source."
        )
    parent_name = action.sources[0].parent_name
    if parent_name not in context.parent_models:
        raise DynamicCausalOperatorPromotionError(
            f"{role} source parent is not loaded in the execution context."
        )
    return parent_name


def _source_layer_index(
    action: FoldAction,
    evidence: HFCausalCircuitEvidence,
    role: str,
) -> int:
    configuration = dict(action.configuration)
    value = configuration.get("source_layer_index")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= len(evidence.layers)
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} action does not bind one cached source layer index."
        )
    return value


def _child_block(
    context: FoldExecutionContext,
    action: FoldAction,
    role: str,
) -> tuple[DecoderLayout, DecoderBlockLayout]:
    configuration = dict(action.configuration)
    index = configuration.get("target_layer_index")
    layout = build_decoder_layout(context.child_model)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= layout.layer_count
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} action does not bind one target decoder layer."
        )
    return layout, layout.blocks[index]


def _require_rope_transport(
    layout: DecoderLayout,
    evidence: HFCausalCircuitEvidence,
    role: str,
) -> None:
    position = layout.position
    if (
        position.kind != "rope"
        or position.rope_theta is None
        or position.rope_variants
        or position.rope_type not in {None, "default"}
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} implements default single-theta RoPE targets only."
        )
    if evidence.architecture.rope_type != "default":
        raise DynamicCausalOperatorPromotionError(
            f"{role} implements default source RoPE only."
        )


def _require_shared_residual_width(
    layout: DecoderLayout,
    evidence: HFCausalCircuitEvidence,
    role: str,
) -> None:
    if layout.hidden_size != evidence.architecture.hidden_size:
        raise DynamicCausalOperatorPromotionError(
            f"{role} fits in one shared residual coordinate system; source and "
            "target hidden widths differ and no boundary-map operator is "
            "registered for this crossing."
        )


@dataclass(frozen=True, slots=True)
class CrossWidthBranchBinding:
    """Chart-transformed branch inputs for one authenticated width crossing.

    The wrapper computes this; the branch operators only substitute it for the
    three parent-coordinate sequences they would otherwise read, so the kernels
    themselves run unchanged in child coordinates.
    """

    target: SequentialBranchTarget
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.target, SequentialBranchTarget):
            raise DynamicCausalOperatorPromotionError(
                "A cross-width binding requires a SequentialBranchTarget."
            )
        if not isinstance(self.report, Mapping):
            raise DynamicCausalOperatorPromotionError(
                "A cross-width binding report must be a mapping."
            )
        object.__setattr__(self, "report", dict(self.report))


def _boundary_chart_required(action: FoldAction) -> bool:
    return bool(dict(action.configuration).get("boundary_chart_required") is True)


def _require_chart_bound_residual_widths(
    layout: DecoderLayout,
    evidence: HFCausalCircuitEvidence,
    action: FoldAction,
    binding: CrossWidthBranchBinding,
    role: str,
) -> None:
    """Authenticate the width crossing the supplied chart binding claims.

    The equal-width case is still allowed — that is the width-preserving
    control — but a real crossing must have been planned as one: an action
    that does not declare ``boundary_chart_required`` never gets to change the
    residual width just because an operator handed it a chart.
    """

    source_width = int(evidence.architecture.hidden_size)
    target_width = int(layout.hidden_size)
    if target_width > source_width:
        raise DynamicCausalOperatorPromotionError(
            f"{role} is contraction-only; residual width expansion is outside "
            "the boundary-chart envelope."
        )
    if target_width != source_width and not _boundary_chart_required(action):
        raise DynamicCausalOperatorPromotionError(
            f"{role} received a residual-width crossing that the planner did "
            "not mark boundary_chart_required."
        )
    state = binding.target.input_state
    endpoint = binding.target.endpoint_target
    normalized = binding.target.normalized_target
    if (
        int(state.shape[-1]) != target_width
        or int(endpoint.shape[-1]) != target_width
        or int(normalized.shape[-1]) != target_width
        or tuple(state.shape) != tuple(endpoint.shape)
        or tuple(state.shape) != tuple(normalized.shape)
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} chart binding does not match the target residual width."
        )


def _require_direct_rms_norm(block_norm: Any, role: str, name: str) -> None:
    if (
        block_norm.kind != "rms_norm"
        or block_norm.weight_kind != "direct_scale"
        or block_norm.has_bias
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires a bias-free direct-scale RMSNorm {name}."
        )


def _target_query_key_norm(attention: Any, role: str) -> str:
    """Classify the target's per-head Q/K normalization, or fail closed.

    ``identity`` and a bias-free direct-scale per-head RMS pair (the qwen3
    layout) are the two shapes the coupled kernel can compile into.  A
    unit-offset gain (gemma3) stores ``1 + w`` rather than ``w`` and would be
    silently mis-scaled by a direct-scale fit, so it is rejected here rather
    than being approximated.
    """

    query_norm = attention.query_norm
    key_norm = attention.key_norm
    if query_norm is None and key_norm is None:
        return "identity"
    if query_norm is None or key_norm is None:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires a target that normalizes both queries and keys "
            "per head, or neither."
        )
    for norm, name, placement in (
        (query_norm, "query", "query_head"),
        (key_norm, "key", "key_head"),
    ):
        if (
            norm.kind != "rms_norm"
            or norm.weight_kind != "direct_scale"
            or norm.has_bias
            or norm.placement != placement
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} installs bias-free direct-scale per-head RMS Q/K "
                f"normalization only; the target {name} norm is "
                f"{norm.kind!r}/{norm.weight_kind!r} at {norm.placement!r}."
            )
    if query_norm.epsilon != key_norm.epsilon:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires one shared target Q/K normalization epsilon."
        )
    return "rms_norm"


def _installed_paths(
    action: FoldAction,
    paths: Sequence[str],
    role: str,
) -> tuple[str, ...]:
    resolved = tuple(sorted(paths))
    if len(set(resolved)) != len(resolved):
        raise DynamicCausalOperatorPromotionError(
            f"{role} resolved a duplicated target parameter path."
        )
    if resolved != tuple(sorted(action.target_parameter_paths)):
        raise DynamicCausalOperatorPromotionError(
            f"{role} layout paths differ from the planned target parameters: "
            f"layout={list(resolved)}, planned={list(action.target_parameter_paths)}."
        )
    return resolved


def _install(
    context: FoldExecutionContext,
    values: Sequence[tuple[str, Any]],
) -> None:
    torch = _torch()
    with torch.no_grad():
        for path, value in values:
            parameter = context.child_model.get_parameter(path)
            if tuple(parameter.shape) != tuple(value.shape):
                raise DynamicCausalOperatorPromotionError(
                    f"Compiled tensor for {path!r} has the wrong target shape: "
                    f"{tuple(value.shape)} into {tuple(parameter.shape)}."
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _capture(
    context: FoldExecutionContext,
    paths: Sequence[str],
    snapshot_config: DynamicFoldSnapshotConfig,
) -> Any:
    return capture_model_state(
        tuple(
            (path, context.child_model.get_parameter(path)) for path in paths
        ),
        snapshot_config,
    )


def _live_parameters(
    context: FoldExecutionContext,
    paths: Sequence[str],
) -> dict[str, Any]:
    return {path: context.child_model.get_parameter(path) for path in paths}


class CausalBranchFoldTransaction:
    """One atomic snapshot-backed install of a family-neutral causal branch."""

    def __init__(
        self,
        *,
        method_id: str,
        context: FoldExecutionContext,
        snapshot: Any,
        installed_paths: tuple[str, ...],
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
        report: Mapping[str, Any],
    ) -> None:
        self.method_id = method_id
        self.context = context
        self.snapshot = snapshot
        self.installed_paths = installed_paths
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = candidate_state_fingerprint
        self.report = dict(report)
        self.active = True
        self.verified_noop = False

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicCausalOperatorPromotionError(
                "The causal branch transaction is no longer active."
            )
        if self.context.child_state_fingerprint != self.candidate_state_fingerprint:
            raise DynamicCausalOperatorPromotionError(
                "The causal branch candidate changed before commit."
            )
        self.snapshot.cleanup()
        self.active = False
        return {
            **self.report,
            "accepted": True,
            "rollback_exact": None,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": self.candidate_state_fingerprint,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicCausalOperatorPromotionError(
                "The causal branch transaction is no longer active."
            )
        self.snapshot.restore(
            _live_parameters(self.context, self.installed_paths)
        )
        restored = self.context.child_state_fingerprint
        self.snapshot.cleanup()
        self.active = False
        if restored != self.incumbent_state_fingerprint:
            raise DynamicCausalOperatorPromotionError(
                "The causal branch rollback did not restore the incumbent."
            )
        return {
            **self.report,
            "accepted": False,
            "rollback_exact": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": restored,
        }


class _CausalBranchOperator:
    """Shared fail-closed preamble for both promoted causal branch operators."""

    method_id = ""
    branch_id = ""

    def __init__(
        self,
        *,
        ridge: float = 1e-6,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        self.ridge = _nonnegative_float(ridge, "ridge")
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(self.snapshot_config, DynamicFoldSnapshotConfig):
            raise DynamicCausalOperatorPromotionError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": SCHEMA,
                "operator": type(self).__qualname__,
                "method_id": self.method_id,
                "branch": self.branch_id,
                "ridge": self.ridge,
                "snapshot_backend": self.snapshot_config.backend,
            }
        )

    def _bind(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
        binding: CrossWidthBranchBinding | None = None,
    ) -> tuple[
        HFCausalCircuitEvidence,
        Any,
        DecoderLayout,
        DecoderBlockLayout,
        int,
    ]:
        role = f"Causal {self.branch_id} branch operator"
        if action.method_id != self.method_id:
            raise DynamicCausalOperatorPromotionError(
                f"{role} received method {action.method_id!r}."
            )
        _single_parent(action, context, role)
        sealed = _sealed_causal_evidence(evidence, role)
        parent_graphs = {
            graph.fingerprint for graph in context.parent_graphs.values()
        }
        if sealed.parent_graph_fingerprint not in parent_graphs:
            raise DynamicCausalOperatorPromotionError(
                f"{role} evidence is bound to a different parent graph."
            )
        layout, block = _child_block(context, action, role)
        _require_rope_transport(layout, sealed, role)
        if binding is None:
            # Without an authenticated per-boundary chart set this operator
            # still fits in one shared residual coordinate system.
            _require_shared_residual_width(layout, sealed, role)
        else:
            _require_chart_bound_residual_widths(
                layout,
                sealed,
                action,
                binding,
                role,
            )
        index = _source_layer_index(action, sealed, role)
        return sealed, sealed.layers[index], layout, block, index


class CausalGQAAttentionBranchOperator(_CausalBranchOperator):
    """Install one coupled GQA/RoPE residual-attention branch from evidence."""

    method_id = CAUSAL_ATTENTION_BRANCH_METHOD
    branch_id = "attention"

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
        *,
        cross_width_binding: CrossWidthBranchBinding | None = None,
    ) -> CausalBranchFoldTransaction:
        role = "Causal attention branch operator"
        sealed, layer, layout, block, source_index = self._bind(
            action,
            context,
            evidence,
            cross_width_binding,
        )
        attention = block.attention
        _require_direct_rms_norm(block.attention_input_norm, role, "input norm")
        if (
            attention.query_path is None
            or attention.key_path is None
            or attention.value_path is None
            or attention.projection_style != "split_qkv"
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires a split Q/K/V target attention."
            )
        if attention.projection_bias:
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires a bias-free target attention."
            )
        target_query_key_norm = _target_query_key_norm(attention, role)
        if (
            attention.logit_softcap is not None
            or attention.sliding_window is not None
            or block.attention_output_norm is not None
            or block.residual.kind != "serial_pre_norm"
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires a pre-norm block without attention "
                "softcapping, sliding windows, or a post-attention norm."
            )
        if sealed.architecture.attention_projection_bias:
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires bias-free cached source projections."
            )
        source_topology = HFCausalGQATopology(
            query_heads=sealed.architecture.query_heads,
            key_value_heads=sealed.architecture.key_value_heads,
            head_dim=sealed.architecture.head_dim,
            rope_theta=sealed.architecture.rope_theta,
            attention_scaling=sealed.architecture.attention_scaling,
            query_key_norm=sealed.architecture.query_key_norm,
            rms_norm_epsilon=sealed.architecture.rms_norm_epsilon,
        )
        target_topology = HFCausalGQATopology(
            query_heads=attention.query_heads,
            key_value_heads=attention.key_value_heads,
            head_dim=attention.head_dim,
            rope_theta=float(layout.position.rope_theta),
            attention_scaling=(
                attention.scaling
                if attention.scaling is not None
                else attention.head_dim ** -0.5
            ),
            query_key_norm=target_query_key_norm,
            # The topology's epsilon is the per-head Q/K one.  An identity
            # target never reads it, so it keeps its historical value.
            rms_norm_epsilon=(
                attention.query_norm.epsilon
                if target_query_key_norm == "rms_norm"
                else block.attention_input_norm.epsilon
            ),
        )
        norm_path = block.attention_input_norm.path + ".weight"
        query_path = str(attention.query_path) + ".weight"
        key_path = str(attention.key_path) + ".weight"
        value_path = str(attention.value_path) + ".weight"
        output_path = str(attention.output_path) + ".weight"
        head_norm_paths: tuple[str, ...] = ()
        if target_query_key_norm == "rms_norm":
            head_norm_paths = (
                attention.query_norm.path + ".weight",
                attention.key_norm.path + ".weight",
            )
        paths = _installed_paths(
            action,
            (
                norm_path,
                query_path,
                key_path,
                value_path,
                output_path,
                *head_norm_paths,
            ),
            role,
        )
        fit = sealed.fit_token_mask
        validation = sealed.validation_token_mask
        # One seam, three substitutions.  With no chart binding these are the
        # cached parent sequences this operator has always read; with one they
        # are the same quantities in the child's boundary charts, so every
        # kernel below runs unchanged.
        branch_input = (
            layer.block_input
            if cross_width_binding is None
            else cross_width_binding.target.input_state
        )
        branch_normalized_target = (
            layer.attention_norm_output
            if cross_width_binding is None
            else cross_width_binding.target.normalized_target
        )
        branch_delta_target = (
            layer.attention_delta
            if cross_width_binding is None
            else cross_width_binding.target.endpoint_target
        )
        norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(branch_input, fit),
            _masked_rows(branch_normalized_target, fit),
            epsilon=block.attention_input_norm.epsilon,
            ridge=self.ridge,
        )
        normalized = _rmsnorm(
            branch_input,
            norm_fold.weight,
            block.attention_input_norm.epsilon,
        )
        attention_fold = compile_hf_causal_gqa_attention_fold(
            normalized,
            source_query_sequences=layer.query_projection_output,
            source_key_sequences=layer.key_projection_output,
            source_value_sequences=layer.value_projection_output,
            target_residual_delta_sequences=branch_delta_target,
            attention_mask=sealed.attention_mask,
            position_ids=sealed.position_ids,
            valid_token_mask=sealed.valid_token_mask,
            fit_token_mask=fit,
            validation_token_mask=validation,
            source_topology=source_topology,
            target_topology=target_topology,
            source_query_norm_weight=layer.query_norm_weight,
            source_key_norm_weight=layer.key_norm_weight,
            ridge=self.ridge,
        )
        delta, _scores, _probabilities = attention_fold.replay(
            normalized,
            attention_mask=sealed.attention_mask,
            position_ids=sealed.position_ids,
        )
        report = {
            **_fold_only_operation_audit(),
            "schema": SCHEMA,
            "kind": "family_neutral_coupled_gqa_rope_attention_branch",
            "method_id": self.method_id,
            "source_family": sealed.architecture.family,
            "target_family": layout.family,
            "source_layer_index": source_index,
            "cached_parent_evidence_sha256": sealed.cache_key,
            "input_norm_report_sha256": norm_fold.report["report_sha256"],
            "attention_report_sha256": attention_fold.report["report_sha256"],
            "installed_parameter_paths": list(paths),
            "validation_attention_delta_relative_rmse": _relative_rmse(
                _masked_rows(delta, validation),
                _masked_rows(branch_delta_target, validation),
            ),
            "parent_residual_coordinate_endpoints": cross_width_binding is None,
            "sequential_child_replay": cross_width_binding is not None,
            "numeric_weight_matching_objective": False,
            "zero_gradient": True,
        }
        if cross_width_binding is not None:
            report.update(dict(cross_width_binding.report))
        installs = [
            (norm_path, norm_fold.weight),
            (query_path, attention_fold.query_weight),
            (key_path, attention_fold.key_weight),
            (value_path, attention_fold.value_weight),
            (output_path, attention_fold.output_weight),
        ]
        if target_query_key_norm == "rms_norm":
            # An identity-Q/K target keeps the exact report it had before this
            # lane existed, so these keys are only present when they mean
            # something.
            report.update(
                {
                    "target_query_key_norm": target_query_key_norm,
                    "target_query_key_norm_epsilon": (
                        target_topology.rms_norm_epsilon
                    ),
                    "query_norm_report_sha256": attention_fold.report[
                        "query_norm_report_sha256"
                    ],
                    "key_norm_report_sha256": attention_fold.report[
                        "key_norm_report_sha256"
                    ],
                    "query_norm_gain_relative_rmse": attention_fold.report[
                        "query_norm_gain_relative_rmse"
                    ],
                    "key_norm_gain_relative_rmse": attention_fold.report[
                        "key_norm_gain_relative_rmse"
                    ],
                }
            )
            installs.extend(
                zip(
                    head_norm_paths,
                    (
                        attention_fold.query_norm_weight,
                        attention_fold.key_norm_weight,
                    ),
                    strict=True,
                )
            )
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            _install(context, installs)
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class CausalGatedMLPBranchOperator(_CausalBranchOperator):
    """Install one coupled SwiGLU residual-MLP branch from sealed evidence."""

    method_id = CAUSAL_GATED_MLP_BRANCH_METHOD
    branch_id = "gated_mlp"

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
        *,
        cross_width_binding: CrossWidthBranchBinding | None = None,
    ) -> CausalBranchFoldTransaction:
        role = "Causal gated-MLP branch operator"
        sealed, layer, layout, block, source_index = self._bind(
            action,
            context,
            evidence,
            cross_width_binding,
        )
        mlp = block.mlp
        _require_direct_rms_norm(block.mlp_input_norm, role, "input norm")
        if (
            mlp.gate_projection_path is None
            or mlp.up_projection_path is None
            or mlp.input_projection_path is not None
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires a split gate/up/down target MLP."
            )
        if mlp.activation_name not in _GATED_ACTIVATIONS:
            raise DynamicCausalOperatorPromotionError(
                f"{role} implements SiLU-gated targets only, not "
                f"{mlp.activation_name!r}."
            )
        if mlp.projection_bias or sealed.architecture.mlp_projection_bias:
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires bias-free source and target MLP projections."
            )
        if (
            block.mlp_output_norm is not None
            or block.residual.kind != "serial_pre_norm"
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires a pre-norm block without a post-MLP norm."
            )
        if mlp.intermediate_size > sealed.architecture.intermediate_size:
            raise DynamicCausalOperatorPromotionError(
                f"{role} is contraction-only; the target MLP is wider than the "
                "cached source MLP."
            )
        norm_path = block.mlp_input_norm.path + ".weight"
        gate_path = str(mlp.gate_projection_path) + ".weight"
        up_path = str(mlp.up_projection_path) + ".weight"
        down_path = str(mlp.down_projection_path) + ".weight"
        paths = _installed_paths(
            action,
            (norm_path, gate_path, up_path, down_path),
            role,
        )
        fit = sealed.fit_token_mask
        validation = sealed.validation_token_mask
        # The same seam as the attention branch: parent-coordinate sequences,
        # or the identical quantities in the child's boundary charts.
        branch_input = (
            layer.post_attention_residual
            if cross_width_binding is None
            else cross_width_binding.target.input_state
        )
        branch_normalized_target = (
            layer.mlp_norm_output
            if cross_width_binding is None
            else cross_width_binding.target.normalized_target
        )
        branch_delta_target = (
            layer.mlp_delta
            if cross_width_binding is None
            else cross_width_binding.target.endpoint_target
        )
        norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(branch_input, fit),
            _masked_rows(branch_normalized_target, fit),
            epsilon=block.mlp_input_norm.epsilon,
            ridge=self.ridge,
        )
        normalized = _rmsnorm(
            branch_input,
            norm_fold.weight,
            block.mlp_input_norm.epsilon,
        )
        mlp_fold = fit_functional_swiglu(
            _masked_rows(normalized, fit),
            _masked_rows(layer.mlp_gate_output, fit),
            _masked_rows(layer.mlp_up_output, fit),
            _masked_rows(branch_delta_target, fit),
            target_neurons=_positive_int(
                mlp.intermediate_size,
                "target intermediate_size",
            ),
            ridge=self.ridge,
        )
        report = {
            **_fold_only_operation_audit(),
            "schema": SCHEMA,
            "kind": "family_neutral_coupled_swiglu_residual_mlp_branch",
            "method_id": self.method_id,
            "source_family": sealed.architecture.family,
            "target_family": layout.family,
            "source_layer_index": source_index,
            "cached_parent_evidence_sha256": sealed.cache_key,
            "input_norm_report_sha256": norm_fold.report["report_sha256"],
            "swiglu_report_sha256": mlp_fold.report["report_sha256"],
            "installed_parameter_paths": list(paths),
            "validation_mlp_delta_relative_rmse": _relative_rmse(
                mlp_fold.predict(_masked_rows(normalized, validation)),
                _masked_rows(branch_delta_target, validation),
            ),
            "parent_residual_coordinate_endpoints": cross_width_binding is None,
            "sequential_child_replay": cross_width_binding is not None,
            "numeric_weight_matching_objective": False,
            "zero_gradient": True,
        }
        if cross_width_binding is not None:
            report.update(dict(cross_width_binding.report))
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            _install(
                context,
                (
                    (norm_path, norm_fold.weight),
                    (gate_path, mlp_fold.gate_weight),
                    (up_path, mlp_fold.up_weight),
                    (down_path, mlp_fold.down_weight),
                ),
            )
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


def _branch_boundary_ids(
    branch_id: str,
    layer_index: int,
    layer_count: int,
) -> tuple[str, str]:
    """Name the two residual boundaries one branch reads and writes."""

    if branch_id == "attention":
        return (
            residual_boundary_id(layer_index, ATTENTION_INPUT_BOUNDARY),
            residual_boundary_id(layer_index, MLP_INPUT_BOUNDARY),
        )
    if branch_id == "gated_mlp":
        output = (
            residual_boundary_id(layer_index + 1, ATTENTION_INPUT_BOUNDARY)
            if layer_index + 1 < layer_count
            else TERMINAL_BOUNDARY_ID
        )
        return (residual_boundary_id(layer_index, MLP_INPUT_BOUNDARY), output)
    raise DynamicCausalOperatorPromotionError(
        f"No residual boundary pair is defined for branch {branch_id!r}."
    )


def _chart_diagnostics(chart: ResidualBoundaryChart) -> dict[str, Any]:
    diagnostics = dict(chart.report["diagnostics"])
    absorption = diagnostics.get("absorption_demand")
    return {
        "chart_sha256": chart.chart_sha256,
        "interface_locked": chart.interface_locked,
        "cycle_defect": diagnostics["cycle_defect"],
        "norm_commutation_row_fluctuation": diagnostics[
            "norm_commutation_row_fluctuation"
        ],
        "norm_commutation_mean_ratio": diagnostics["norm_commutation_mean_ratio"],
        "incoming_transition_defect_frobenius": diagnostics.get(
            "incoming_transition_defect_frobenius"
        ),
        "absorption_demand_absorbed_energy_fraction": (
            None if absorption is None else absorption["absorbed_energy_fraction"]
        ),
        "absorption_demand_unabsorbed_energy_fraction": (
            None if absorption is None else absorption["unabsorbed_energy_fraction"]
        ),
        "writer_representable_rank": (
            None if absorption is None else absorption["writer_representable_rank"]
        ),
    }


class CrossWidthCausalBranchOperator:
    """Run one promoted causal branch operator across a residual width change.

    The branch kernels are already width-agnostic, so this wrapper adds no
    solver: it authenticates a sealed chart set against the crossing, recaptures
    the live child's state at the branch's input boundary, builds the sequential
    target ``Y = h_{b+1} E_{b+1} - hhat_b_actual``, and hands all three
    chart-transformed sequences to the operator it wraps.  Every fail-closed
    condition the design names is checked before any child mutation.
    """

    #: A zero write predicts nothing, so its relative RMSE against ``Y`` is
    #: exactly one.  A branch at or above that number has not earned its
    #: install, which is the design's endpoint fail-closed condition.
    no_branch_endpoint_relative_rmse_baseline = 1.0

    def __init__(
        self,
        branch_operator: _CausalBranchOperator,
        *,
        chart_set: ResidualBoundaryChartSet,
        recapture: Any,
        max_endpoint_relative_rmse: float | None = None,
    ) -> None:
        if not isinstance(branch_operator, _CausalBranchOperator):
            raise DynamicCausalOperatorPromotionError(
                "CrossWidthCausalBranchOperator wraps a promoted causal branch "
                "operator."
            )
        if not isinstance(chart_set, ResidualBoundaryChartSet):
            raise DynamicCausalOperatorPromotionError(
                "chart_set must be a sealed ResidualBoundaryChartSet."
            )
        if not callable(recapture):
            raise DynamicCausalOperatorPromotionError(
                "recapture must be a callable live-child boundary recapture."
            )
        bound = (
            self.no_branch_endpoint_relative_rmse_baseline
            if max_endpoint_relative_rmse is None
            else max_endpoint_relative_rmse
        )
        bound = _nonnegative_float(bound, "max_endpoint_relative_rmse")
        if bound > self.no_branch_endpoint_relative_rmse_baseline:
            raise DynamicCausalOperatorPromotionError(
                "max_endpoint_relative_rmse cannot exceed the no-branch "
                "baseline; a branch that loses to writing nothing is refused."
            )
        self.branch_operator = branch_operator
        self.chart_set = chart_set.verify()
        self.recapture = recapture
        self.max_endpoint_relative_rmse = bound
        self.reverse_sweep = plan_reverse_reclosure_sweep(self.chart_set)

    @property
    def _endpoint_report_key(self) -> str:
        return (
            "validation_attention_delta_relative_rmse"
            if self.branch_id == "attention"
            else "validation_mlp_delta_relative_rmse"
        )

    @property
    def method_id(self) -> str:
        return self.branch_operator.method_id

    @property
    def branch_id(self) -> str:
        return self.branch_operator.branch_id

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": SCHEMA,
                "operator": type(self).__qualname__,
                "method_id": self.method_id,
                "branch": self.branch_id,
                "wrapped_operator_resource_fingerprint": (
                    self.branch_operator.resource_fingerprint
                ),
                "chart_set_sha256": self.chart_set.set_sha256,
                "reverse_reclosure_sweep_sha256": self.reverse_sweep.sweep_sha256,
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "recapture": type(self.recapture).__qualname__,
            }
        )

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        role = f"Cross-width causal {self.branch_id} branch operator"
        if action.method_id != self.method_id:
            raise DynamicCausalOperatorPromotionError(
                f"{role} received method {action.method_id!r}."
            )
        sealed = _sealed_causal_evidence(evidence, role)
        layout, _block = _child_block(context, action, role)
        try:
            chart_set = self.chart_set.authenticate(
                parent_evidence_sha256=sealed.cache_key,
                child_topology_sha256=child_residual_topology_sha256(layout),
                role=role,
            )
        except CrossWidthResidualChartError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} chart set failed authentication: {exc}"
            ) from exc
        if chart_set.source_width != int(sealed.architecture.hidden_size) or (
            chart_set.target_width != int(layout.hidden_size)
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} chart set widths differ from the planned crossing."
            )
        source_index = _source_layer_index(action, sealed, role)
        target_index = int(dict(action.configuration)["target_layer_index"])
        if source_index != target_index or len(layout.blocks) != len(sealed.layers):
            raise DynamicCausalOperatorPromotionError(
                f"{role} implements one cached parent layer per child layer; "
                "depth-changing crossings need their own boundary evidence."
            )
        layer = sealed.layers[source_index]
        input_id, output_id = _branch_boundary_ids(
            self.branch_id,
            target_index,
            len(layout.blocks),
        )
        try:
            input_chart = chart_set.chart(input_id)
            output_chart = chart_set.chart(output_id)
            states = self.recapture(context.child_model, (input_id,))
        except CrossWidthResidualChartError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} could not bind both boundary charts and the sequential "
                f"child recapture: {exc}"
            ) from exc
        if not isinstance(states, Mapping) or input_id not in states:
            raise DynamicCausalOperatorPromotionError(
                f"{role} sequential recapture returned no state for {input_id!r}; "
                "falling back to parent-coordinate inputs is exactly the A0025 "
                "failure and is refused."
            )
        if self.branch_id == "attention":
            parent_input = layer.block_input
            parent_output = layer.post_attention_residual
            parent_delta = layer.attention_delta
            parent_normalized = layer.attention_norm_output
        else:
            parent_input = layer.post_attention_residual
            parent_output = layer.block_output
            parent_delta = layer.mlp_delta
            parent_normalized = layer.mlp_norm_output
        try:
            target = build_sequential_branch_target(
                input_chart=input_chart,
                output_chart=output_chart,
                parent_input_state=parent_input,
                parent_output_state=parent_output,
                parent_branch_delta=parent_delta,
                parent_normalized_input=parent_normalized,
                child_input_state=states[input_id],
                fit_token_mask=sealed.fit_token_mask,
            )
        except CrossWidthResidualChartError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} could not build the sequential branch target: {exc}"
            ) from exc
        binding = CrossWidthBranchBinding(
            target=target,
            report={
                "cross_width_schema": CROSS_WIDTH_BRANCH_FOLD_SCHEMA,
                "cross_width_residual_crossing": True,
                "parent_residual_width": chart_set.source_width,
                "child_residual_width": chart_set.target_width,
                "boundary_chart_set_sha256": chart_set.set_sha256,
                "boundary_chart_policy_sha256": chart_set.report["policy_sha256"],
                "input_boundary_id": input_id,
                "output_boundary_id": output_id,
                "input_boundary_chart": _chart_diagnostics(input_chart),
                "output_boundary_chart": _chart_diagnostics(output_chart),
                "sequential_branch_target": dict(target.report),
                "reverse_reclosure_sweep_sha256": self.reverse_sweep.sweep_sha256,
                "reverse_reclosure_sweep_executed": False,
                "no_branch_endpoint_relative_rmse_baseline": (
                    self.no_branch_endpoint_relative_rmse_baseline
                ),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "child_recapture_calls": int(
                    getattr(self.recapture, "recapture_calls", 0)
                ),
                "child_recapture_inputs_sha256": getattr(
                    self.recapture, "inputs_sha256", None
                ),
            },
        )
        transaction = self.branch_operator.prepare(
            action,
            context,
            evidence,
            cross_width_binding=binding,
        )
        endpoint = transaction.report.get(self._endpoint_report_key)
        if (
            not isinstance(endpoint, float)
            or not math.isfinite(endpoint)
            or endpoint >= self.max_endpoint_relative_rmse
        ):
            # Undo this branch before reporting: a write that loses to the
            # no-branch baseline is not an install, it is a regression.
            transaction.rollback()
            raise DynamicCausalOperatorPromotionError(
                f"{role} endpoint relative RMSE {endpoint!r} is at or above the "
                f"no-branch baseline bound {self.max_endpoint_relative_rmse!r} "
                f"at boundary {output_id!r}."
            )
        return transaction


@dataclass(frozen=True, slots=True)
class IdentityVocabularyProjector:
    """Exact same-width donor-row transport for a strict tokenizer bridge.

    The strict bridge is a donor row subset.  When the child keeps the parent's
    residual coordinates there is nothing to re-coordinate, so the projector is
    the identity and its map identity is a width seal rather than a matrix.
    """

    source_width: int
    target_width: int = field(init=False)
    map_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        width = _positive_int(self.source_width, "source_width")
        object.__setattr__(self, "target_width", width)
        object.__setattr__(
            self,
            "map_sha256",
            canonical_sha256(
                {
                    "schema": SCHEMA,
                    "kind": "identity_vocabulary_projector",
                    "width": width,
                }
            ),
        )

    def fold_consumer_weight(self, source_weight: Any) -> Any:
        torch = _torch()
        if (
            not isinstance(source_weight, torch.Tensor)
            or source_weight.ndim != 2
            or int(source_weight.shape[1]) != self.source_width
        ):
            raise DynamicCausalOperatorPromotionError(
                "Identity vocabulary transport requires matching row widths."
            )
        return source_weight.detach().to(dtype=torch.float64).contiguous()


class TokenInterfaceBridgeTransaction:
    """One atomic snapshot-backed install of a sealed token-row interface.

    Both bridge kinds share this transaction: the strict donor-row bridge and
    the independent-vocabulary pair bridge differ only in how the target rows
    are compiled, never in how they are installed, verified, or rolled back.
    """

    method_id = TOKEN_INTERFACE_BRIDGE_METHOD

    def __init__(
        self,
        *,
        context: FoldExecutionContext,
        snapshot: Any,
        installed_paths: tuple[str, ...],
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
        report: Mapping[str, Any],
    ) -> None:
        self.context = context
        self.snapshot = snapshot
        self.installed_paths = installed_paths
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = candidate_state_fingerprint
        self.report = dict(report)
        self.active = True
        self.verified_noop = False

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicCausalOperatorPromotionError(
                "The token-interface bridge transaction is no longer active."
            )
        if self.context.child_state_fingerprint != self.candidate_state_fingerprint:
            raise DynamicCausalOperatorPromotionError(
                "The token-interface bridge candidate changed before commit."
            )
        self.snapshot.cleanup()
        self.active = False
        return {
            **self.report,
            "accepted": True,
            "rollback_exact": None,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": self.candidate_state_fingerprint,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicCausalOperatorPromotionError(
                "The token-interface bridge transaction is no longer active."
            )
        self.snapshot.restore(
            _live_parameters(self.context, self.installed_paths)
        )
        restored = self.context.child_state_fingerprint
        self.snapshot.cleanup()
        self.active = False
        if restored != self.incumbent_state_fingerprint:
            raise DynamicCausalOperatorPromotionError(
                "The token-interface bridge rollback did not restore the "
                "incumbent."
            )
        return {
            **self.report,
            "accepted": False,
            "rollback_exact": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": restored,
        }


def _interface_nodes(
    graph: ModelCircuitGraph,
    node_ids: Sequence[str],
    role: str,
    owner: str,
) -> tuple[Any, Any]:
    nodes = tuple(graph.node(node_id) for node_id in node_ids)
    embeddings = tuple(
        node for node in nodes if node.kind == CircuitKind.TOKEN_EMBEDDING.value
    )
    heads = tuple(
        node for node in nodes if node.kind == CircuitKind.OUTPUT_HEAD.value
    )
    if len(nodes) != 2 or len(embeddings) != 1 or len(heads) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires exactly one {owner} token embedding and one "
            f"{owner} output head."
        )
    return embeddings[0], heads[0]


def _single_parameter_path(node: Any, role: str, name: str) -> str:
    paths = tuple(node.parameter_paths)
    if len(paths) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires exactly one learned {name} tensor."
        )
    return paths[0]


def _pair_transport(value: Any, name: str) -> str:
    if value not in _PAIR_TRANSPORTS:
        raise DynamicCausalOperatorPromotionError(
            f"{name} must be one of {', '.join(_PAIR_TRANSPORTS)}."
        )
    return str(value)


def _dense_vocabulary_projection(
    rows: Any,
    coordinates: Any,
    name: str,
) -> tuple[Any, str]:
    """Push source-width vocabulary evidence through a dense projector.

    The pair bridge deliberately stops at the source width, so re-coordinating
    stays where it already lives.  Only the ``DenseVocabularyProjector``
    protocol is accepted here; a ``CoordinateMap`` has to be wrapped by its
    owner rather than being re-implemented in the operator layer.
    """

    torch = _torch()
    source_width = _positive_int(
        getattr(coordinates, "source_width", None), f"{name}.source_width"
    )
    target_width = _positive_int(
        getattr(coordinates, "target_width", None), f"{name}.target_width"
    )
    map_sha256 = getattr(coordinates, "map_sha256", None)
    if (
        not isinstance(map_sha256, str)
        or len(map_sha256) != 64
        or any(character not in "0123456789abcdef" for character in map_sha256)
        or not callable(getattr(coordinates, "fold_consumer_weight", None))
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{name} must expose a sealed fold_consumer_weight projector."
        )
    if int(rows.shape[1]) != source_width:
        raise DynamicCausalOperatorPromotionError(
            f"{name} source width differs from the compiled evidence width."
        )
    try:
        result = coordinates.fold_consumer_weight(rows)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise DynamicCausalOperatorPromotionError(
            f"{name} rejected the compiled vocabulary evidence."
        ) from exc
    if (
        not isinstance(result, torch.Tensor)
        or result.ndim != 2
        or not torch.is_floating_point(result)
        or torch.is_complex(result)
        or tuple(result.shape) != (int(rows.shape[0]), target_width)
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{name} returned an invalid projected matrix."
        )
    projected = result.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(projected).all().item()):
        raise DynamicCausalOperatorPromotionError(
            f"{name} returned NaN or infinity."
        )
    return projected, map_sha256


def _install_only_covered_rows(
    incumbent: Any,
    compiled: Any,
    mask: Any,
    role: str,
) -> Any:
    """Blend compiled rows over the incumbent wherever the bridge has evidence.

    An uncovered row is exactly zero in the transport output, which is not a
    claim the pair bridge can make about that token.  The incumbent value is
    round-tripped through float64 and back by ``_install``; both directions are
    exact for the float32 blank targets this lane installs into.
    """

    torch = _torch()
    live = incumbent.detach().to(device="cpu", dtype=torch.float64)
    if tuple(live.shape) != tuple(compiled.shape):
        raise DynamicCausalOperatorPromotionError(
            f"{role} compiled rows do not match the target matrix shape."
        )
    rows = mask.reshape(-1).to(torch.bool)
    if int(rows.numel()) != int(live.shape[0]):
        raise DynamicCausalOperatorPromotionError(
            f"{role} evidence mask does not cover every target row."
        )
    if not bool(rows.any().item()):
        raise DynamicCausalOperatorPromotionError(
            f"{role} pair bridge covered no target row at all."
        )
    return torch.where(rows.unsqueeze(1), compiled, live).contiguous()


def _uncovered_row_record(name: str, mask: Any) -> dict[str, Any]:
    """Record which target rows kept their blank value, and prove the record."""

    torch = _torch()
    rows = mask.reshape(-1).to(torch.bool)
    uncovered = torch.nonzero(~rows, as_tuple=False).reshape(-1).tolist()
    listed = [int(value) for value in uncovered[:_UNCOVERED_ROW_INDEX_LIMIT]]
    return {
        f"{name}_uncovered_target_rows": len(uncovered),
        f"{name}_uncovered_target_row_indices": listed,
        f"{name}_uncovered_target_row_indices_truncated": (
            len(uncovered) > len(listed)
        ),
        f"{name}_evidence_mask_sha256": canonical_sha256(
            {
                "schema": SCHEMA,
                "kind": "pair_bridge_evidence_mask",
                "rows": int(rows.numel()),
                "covered": [
                    int(value)
                    for value in torch.nonzero(rows, as_tuple=False)
                    .reshape(-1)
                    .tolist()
                ],
            }
        ),
    }


class CrossWidthTokenHeadInterfaceOperator:
    """Install both interface-locked boundaries of one width crossing.

    Boundary 0 is the embedding write and the terminal boundary is the
    final-norm read that feeds the head, so one chart pair fixes the embedding
    rows, the head rows, and the final-norm gains.  Chart selection owns those
    two charts; this operator only consumes them, exactly as the design's
    interface-locked clause requires.  The tokenizer identity must be shared:
    an independent vocabulary is the bridge lane's problem, not this one's.
    """

    method_id = TOKEN_INTERFACE_BRIDGE_METHOD

    def __init__(
        self,
        *,
        chart_set: ResidualBoundaryChartSet,
        ridge: float = 1e-6,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        if not isinstance(chart_set, ResidualBoundaryChartSet):
            raise DynamicCausalOperatorPromotionError(
                "chart_set must be a sealed ResidualBoundaryChartSet."
            )
        self.chart_set = chart_set.verify()
        self.ridge = _nonnegative_float(ridge, "ridge")
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(self.snapshot_config, DynamicFoldSnapshotConfig):
            raise DynamicCausalOperatorPromotionError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": SCHEMA,
                "operator": type(self).__qualname__,
                "method_id": self.method_id,
                "chart_set_sha256": self.chart_set.set_sha256,
                "ridge": self.ridge,
                "snapshot_backend": self.snapshot_config.backend,
            }
        )

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> TokenInterfaceBridgeTransaction:
        role = "Cross-width token/head interface operator"
        if action.method_id != self.method_id:
            raise DynamicCausalOperatorPromotionError(
                f"{role} received method {action.method_id!r}."
            )
        if not _boundary_chart_required(action):
            raise DynamicCausalOperatorPromotionError(
                f"{role} only executes an action the planner marked "
                "boundary_chart_required."
            )
        parent_name = _single_parent(action, context, role)
        parent_graph = context.parent_graphs[parent_name]
        child_graph = context.child_graph
        if (
            parent_graph.tokenizer_fingerprint is None
            or parent_graph.tokenizer_fingerprint != child_graph.tokenizer_fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires one shared tokenizer identity; an independent "
                "vocabulary needs a sealed tokenizer bridge as well."
            )
        sealed = _sealed_causal_evidence(evidence, role)
        if sealed.parent_graph_fingerprint != parent_graph.fingerprint:
            raise DynamicCausalOperatorPromotionError(
                f"{role} evidence is bound to a different parent graph."
            )
        layout = build_decoder_layout(context.child_model)
        try:
            chart_set = self.chart_set.authenticate(
                parent_evidence_sha256=sealed.cache_key,
                child_topology_sha256=child_residual_topology_sha256(layout),
                role=role,
            )
            embedding_chart = chart_set.chart(
                residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY)
            )
            terminal_chart = chart_set.chart(TERMINAL_BOUNDARY_ID)
        except CrossWidthResidualChartError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} chart set failed authentication: {exc}"
            ) from exc
        if not embedding_chart.interface_locked or not (
            terminal_chart.interface_locked
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} requires both interface boundaries to be chart locked."
            )
        _require_direct_rms_norm(layout.final_norm, role, "final norm")
        target_embedding_node, target_head_node = _interface_nodes(
            child_graph,
            tuple(
                node_id
                for node_id in action.target_node_ids
                if child_graph.node(node_id).kind
                in {CircuitKind.TOKEN_EMBEDDING.value, CircuitKind.OUTPUT_HEAD.value}
            ),
            role,
            "child",
        )
        target_embedding_path = _single_parameter_path(
            target_embedding_node, role, "child embedding"
        )
        target_head_path = _single_parameter_path(
            target_head_node, role, "child output head"
        )
        final_norm_path = layout.final_norm.path + ".weight"
        tied = bool(
            {
                parameter.storage_group
                for parameter in target_embedding_node.parameters
            }
            & {parameter.storage_group for parameter in target_head_node.parameters}
        )
        paths = _installed_paths(
            action,
            (target_embedding_path, target_head_path, final_norm_path),
            role,
        )
        if tied:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot install one tied tensor from two different "
                "interface charts."
            )
        embedding_rows = embedding_chart.fold_consumer_weight(
            sealed.token_embedding_weight
        )
        head_rows = terminal_chart.fold_consumer_weight(sealed.lm_head_weight)
        fit = sealed.fit_token_mask
        norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(terminal_chart.encode(sealed.layers[-1].block_output), fit),
            _masked_rows(terminal_chart.encode(sealed.final_norm_output), fit),
            epsilon=layout.final_norm.epsilon,
            ridge=self.ridge,
        )
        report = {
            **_fold_only_operation_audit(),
            "schema": SCHEMA,
            "kind": "chart_projected_token_head_interface",
            "method_id": self.method_id,
            "cross_width_schema": CROSS_WIDTH_BRANCH_FOLD_SCHEMA,
            "cross_width_residual_crossing": True,
            "parent_residual_width": chart_set.source_width,
            "child_residual_width": chart_set.target_width,
            "boundary_chart_set_sha256": chart_set.set_sha256,
            "boundary_chart_policy_sha256": chart_set.report["policy_sha256"],
            "embedding_boundary_chart": _chart_diagnostics(embedding_chart),
            "terminal_boundary_chart": _chart_diagnostics(terminal_chart),
            "cached_parent_evidence_sha256": sealed.cache_key,
            "final_norm_report_sha256": norm_fold.report["report_sha256"],
            "final_norm_relative_rmse": norm_fold.report["relative_rmse"],
            "installed_parameter_paths": list(paths),
            "target_tied_storage": tied,
            "shared_tokenizer_identity": True,
            "interface_locked_boundaries_consumed_not_selected": True,
            "random_target_rows": 0,
            "zero_gradient": True,
        }
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            _install(
                context,
                (
                    (target_embedding_path, embedding_rows),
                    (target_head_path, head_rows),
                    (final_norm_path, norm_fold.weight),
                ),
            )
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return TokenInterfaceBridgeTransaction(
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )

class StrictTokenInterfaceBridgeOperator:
    """Execute the planner's ``token_interface_bridge`` selection.

    The bridge itself is production-proven strict donor-row transport: the
    sealed artifact fixes which donor rows every target row is, and this
    operator only authenticates that seal against the planned parent/child
    tokenizer identities and installs the resulting rows atomically.
    """

    method_id = TOKEN_INTERFACE_BRIDGE_METHOD

    def __init__(
        self,
        *,
        artifact: StrictTokenizerBridgeArtifact | None = None,
        bridge_directory: str | Path | None = None,
        embedding_coordinates: Any | None = None,
        lm_head_coordinates: Any | None = None,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        if (artifact is None) == (bridge_directory is None):
            raise DynamicCausalOperatorPromotionError(
                "Provide exactly one of artifact or bridge_directory."
            )
        if artifact is None:
            artifact = load_strict_tokenizer_bridge_artifact(bridge_directory)
        if not isinstance(artifact, StrictTokenizerBridgeArtifact):
            raise DynamicCausalOperatorPromotionError(
                "artifact must be a StrictTokenizerBridgeArtifact."
            )
        artifact.verify_files()
        self.artifact = artifact
        self.embedding_coordinates = embedding_coordinates
        self.lm_head_coordinates = lm_head_coordinates
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(self.snapshot_config, DynamicFoldSnapshotConfig):
            raise DynamicCausalOperatorPromotionError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": SCHEMA,
                "operator": type(self).__qualname__,
                "method_id": self.method_id,
                "bridge_artifact_sha256": self.artifact.artifact_sha256,
                "transport_profile_fingerprint": self.artifact.profile_fingerprint,
                "snapshot_backend": self.snapshot_config.backend,
            }
        )

    def _coordinates(self, width: int, name: str) -> Any:
        configured = (
            self.embedding_coordinates
            if name == "embedding"
            else self.lm_head_coordinates
        )
        if configured is not None:
            return configured
        return IdentityVocabularyProjector(source_width=width)

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> TokenInterfaceBridgeTransaction:
        del evidence  # The sealed bridge artifact is this operator's evidence.
        role = "Strict token-interface bridge operator"
        if action.method_id != self.method_id:
            raise DynamicCausalOperatorPromotionError(
                f"{role} received method {action.method_id!r}."
            )
        parent_name = _single_parent(action, context, role)
        parent_graph = context.parent_graphs[parent_name]
        child_graph = context.child_graph
        artifact = self.artifact
        artifact.verify_files()
        if artifact.donor_tokenizer_semantics_sha256 != (
            parent_graph.tokenizer_fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} sealed donor tokenizer differs from the planned "
                "parent tokenizer identity."
            )
        if artifact.target_tokenizer_semantics_sha256 != (
            child_graph.tokenizer_fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} sealed target tokenizer differs from the planned "
                "child tokenizer identity."
            )
        source_embedding_node, source_head_node = _interface_nodes(
            parent_graph,
            action.sources[0].node_ids,
            role,
            "parent",
        )
        target_embedding_node, target_head_node = _interface_nodes(
            child_graph,
            action.target_node_ids,
            role,
            "child",
        )
        source_embedding_path = _single_parameter_path(
            source_embedding_node, role, "parent embedding"
        )
        source_head_path = _single_parameter_path(
            source_head_node, role, "parent output head"
        )
        target_embedding_path = _single_parameter_path(
            target_embedding_node, role, "child embedding"
        )
        target_head_path = _single_parameter_path(
            target_head_node, role, "child output head"
        )
        tied = bool(
            {
                parameter.storage_group
                for parameter in target_embedding_node.parameters
            }
            & {parameter.storage_group for parameter in target_head_node.parameters}
        )
        paths = _installed_paths(
            action,
            (target_embedding_path, target_head_path),
            role,
        )
        parent_model = context.parent_models[parent_name]
        source_embedding = parent_model.get_parameter(source_embedding_path)
        source_head = parent_model.get_parameter(source_head_path)
        target_embedding = context.child_model.get_parameter(target_embedding_path)
        target_head = context.child_model.get_parameter(target_head_path)
        if (
            int(target_embedding.shape[0]) != artifact.target_vocabulary_size
            or int(target_head.shape[0]) != artifact.target_vocabulary_size
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} target rows differ from the sealed target vocabulary."
            )
        transport = compile_strict_vocabulary_operators(
            source_embedding,
            source_head,
            artifact,
            self._coordinates(int(source_embedding.shape[1]), "embedding"),
            self._coordinates(int(source_head.shape[1]), "lm_head"),
            require_tied_target=tied,
        )
        installs = [(target_embedding_path, transport.embedding_weight)]
        if not tied:
            installs.append((target_head_path, transport.lm_head_weight))
        report = {
            **_fold_only_operation_audit(),
            "schema": SCHEMA,
            "kind": "strict_donor_row_token_interface_bridge",
            "method_id": self.method_id,
            "bridge_artifact_sha256": artifact.artifact_sha256,
            "transport_profile_fingerprint": artifact.profile_fingerprint,
            "donor_tokenizer_semantics_sha256": (
                artifact.donor_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                artifact.target_tokenizer_semantics_sha256
            ),
            "transport_report_sha256": transport.report["report_sha256"],
            "donor_row_capacity": artifact.transport_profile.donor_row_capacity,
            "target_row_capacity": artifact.target_vocabulary_size,
            "installed_parameter_paths": list(paths),
            "target_tied_storage": tied,
            "random_target_rows": 0,
            "zero_gradient": True,
        }
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            _install(context, installs)
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return TokenInterfaceBridgeTransaction(
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class PairCodeTokenInterfaceBridgeOperator:
    """Execute ``token_interface_bridge`` from an independent-vocabulary pair.

    The strict operator's bridge is a donor-row subset, so every target row is
    some parent row and the transport is a total function.  A pair bridge holds
    two unrelated vocabularies instead: each row of one is a nonempty *code* of
    rows of the other, and nothing guarantees that every target row appears in
    any code.  The transport builder returns exact zeros for those rows, and
    zeros are a claim this operator has no evidence for, so uncovered rows are
    never written: they keep the child's incumbent (blank) value and are
    reported.  Everything else — seal authentication, tied storage, atomic
    snapshot install — is identical to the strict lane.
    """

    method_id = TOKEN_INTERFACE_BRIDGE_METHOD

    def __init__(
        self,
        *,
        artifact: TokenizerPairBridgeArtifact | None = None,
        bridge_directory: str | Path | None = None,
        embedding_transport: str = ANCHOR_PREFERRED_PAIR_TRANSPORT,
        lm_head_transport: str = ANCHOR_PREFERRED_PAIR_TRANSPORT,
        token_frequencies: Mapping[int, int] | Sequence[int] | None = None,
        embedding_coordinates: Any | None = None,
        lm_head_coordinates: Any | None = None,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        if (artifact is None) == (bridge_directory is None):
            raise DynamicCausalOperatorPromotionError(
                "Provide exactly one of artifact or bridge_directory."
            )
        if artifact is None:
            artifact = load_hf_tokenizer_pair_bridge_artifact(bridge_directory)
        if not isinstance(artifact, TokenizerPairBridgeArtifact):
            raise DynamicCausalOperatorPromotionError(
                "artifact must be a TokenizerPairBridgeArtifact."
            )
        artifact.verify_files()
        self.artifact = artifact
        self.embedding_transport = _pair_transport(
            embedding_transport, "embedding_transport"
        )
        self.lm_head_transport = _pair_transport(
            lm_head_transport, "lm_head_transport"
        )
        if token_frequencies is None and _OCCURRENCE_PAIR_TRANSPORT in {
            self.embedding_transport,
            self.lm_head_transport,
        }:
            raise DynamicCausalOperatorPromotionError(
                "Occurrence-weighted pair transport requires token_frequencies."
            )
        self.token_frequencies = token_frequencies
        self.embedding_coordinates = embedding_coordinates
        self.lm_head_coordinates = lm_head_coordinates
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(self.snapshot_config, DynamicFoldSnapshotConfig):
            raise DynamicCausalOperatorPromotionError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": SCHEMA,
                "operator": type(self).__qualname__,
                "method_id": self.method_id,
                "bridge_kind": self.artifact.bridge_kind,
                "bridge_artifact_sha256": self.artifact.artifact_sha256,
                "pair_profile_fingerprint": self.artifact.profile_fingerprint,
                "embedding_transport": self.embedding_transport,
                "lm_head_transport": self.lm_head_transport,
                "masked_row_policy": _PAIR_MASKED_ROW_POLICY,
                "snapshot_backend": self.snapshot_config.backend,
            }
        )

    def _coordinates(self, width: int, name: str) -> Any:
        configured = (
            self.embedding_coordinates
            if name == "embedding"
            else self.lm_head_coordinates
        )
        if configured is not None:
            return configured
        return IdentityVocabularyProjector(source_width=width)

    def _transport(
        self,
        source_weight: Any,
        weighting: str,
    ) -> tuple[Any, Any, tuple[Mapping[str, Any], ...]]:
        """Return ``(evidence, evidence_mask, transport reports)``."""

        torch = _torch()
        artifact = self.artifact
        if weighting == ANCHOR_PREFERRED_PAIR_TRANSPORT:
            anchor = compile_pair_bridge_anchor_transport(source_weight, artifact)
            averaged = compile_pair_bridge_length_normalized_transport(
                source_weight, artifact
            )
            anchored = anchor.evidence_mask.to(torch.bool)
            covered = averaged.evidence_mask.to(torch.bool)
            if bool((anchored & ~covered).any().item()):
                raise DynamicCausalOperatorPromotionError(
                    "An anchor-exact pair transport reached a row the "
                    "length-normalized transport did not; the sealed codes are "
                    "inconsistent."
                )
            evidence = torch.where(
                anchored.unsqueeze(1), anchor.evidence, averaged.evidence
            )
            return evidence, covered, (anchor.report, averaged.report)
        if weighting == _ANCHOR_PAIR_TRANSPORT:
            result = compile_pair_bridge_anchor_transport(source_weight, artifact)
        elif weighting == _LENGTH_NORMALIZED_PAIR_TRANSPORT:
            result = compile_pair_bridge_length_normalized_transport(
                source_weight, artifact
            )
        else:
            result = compile_pair_bridge_occurrence_weighted_transport(
                source_weight,
                artifact,
                self.token_frequencies,
            )
        return (
            result.evidence,
            result.evidence_mask.to(torch.bool),
            (result.report,),
        )

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> TokenInterfaceBridgeTransaction:
        del evidence  # The sealed pair bridge is this operator's evidence.
        torch = _torch()
        role = "Pair-code token-interface bridge operator"
        if action.method_id != self.method_id:
            raise DynamicCausalOperatorPromotionError(
                f"{role} received method {action.method_id!r}."
            )
        parent_name = _single_parent(action, context, role)
        parent_graph = context.parent_graphs[parent_name]
        child_graph = context.child_graph
        artifact = self.artifact
        artifact.verify_files()
        if artifact.parent_tokenizer_semantics_sha256 != (
            parent_graph.tokenizer_fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} sealed source tokenizer differs from the planned "
                "parent tokenizer identity."
            )
        if artifact.child_tokenizer_semantics_sha256 != (
            child_graph.tokenizer_fingerprint
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} sealed target tokenizer differs from the planned "
                "child tokenizer identity."
            )
        source_embedding_node, source_head_node = _interface_nodes(
            parent_graph,
            action.sources[0].node_ids,
            role,
            "parent",
        )
        target_embedding_node, target_head_node = _interface_nodes(
            child_graph,
            action.target_node_ids,
            role,
            "child",
        )
        source_embedding_path = _single_parameter_path(
            source_embedding_node, role, "parent embedding"
        )
        source_head_path = _single_parameter_path(
            source_head_node, role, "parent output head"
        )
        target_embedding_path = _single_parameter_path(
            target_embedding_node, role, "child embedding"
        )
        target_head_path = _single_parameter_path(
            target_head_node, role, "child output head"
        )
        tied = bool(
            {
                parameter.storage_group
                for parameter in target_embedding_node.parameters
            }
            & {parameter.storage_group for parameter in target_head_node.parameters}
        )
        paths = _installed_paths(
            action,
            (target_embedding_path, target_head_path),
            role,
        )
        profile = artifact.profile
        parent_model = context.parent_models[parent_name]
        source_embedding = parent_model.get_parameter(source_embedding_path)
        source_head = parent_model.get_parameter(source_head_path)
        target_embedding = context.child_model.get_parameter(target_embedding_path)
        target_head = context.child_model.get_parameter(target_head_path)
        if (
            int(source_embedding.shape[0]) != profile.source_row_capacity
            or int(source_head.shape[0]) != profile.source_row_capacity
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} parent rows differ from the sealed source row capacity."
            )
        if (
            int(target_embedding.shape[0]) != profile.target_row_capacity
            or int(target_head.shape[0]) != profile.target_row_capacity
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} target rows differ from the sealed target row capacity."
            )
        embedding_evidence, embedding_mask, embedding_reports = self._transport(
            source_embedding,
            self.embedding_transport,
        )
        embedding_rows, embedding_map_sha256 = _dense_vocabulary_projection(
            embedding_evidence,
            self._coordinates(int(source_embedding.shape[1]), "embedding"),
            f"{role} embedding_coordinates",
        )
        if tied:
            # One transport, one tensor.  Running the builder twice and
            # comparing is only bit-exact when the inputs and weighting match,
            # so the tie is enforced by construction instead.
            head_rows = embedding_rows
            head_mask = embedding_mask
            head_reports = embedding_reports
            head_map_sha256 = embedding_map_sha256
        else:
            head_evidence, head_mask, head_reports = self._transport(
                source_head,
                self.lm_head_transport,
            )
            head_rows, head_map_sha256 = _dense_vocabulary_projection(
                head_evidence,
                self._coordinates(int(source_head.shape[1]), "lm_head"),
                f"{role} lm_head_coordinates",
            )
        embedding_install = _install_only_covered_rows(
            target_embedding, embedding_rows, embedding_mask, role
        )
        installs = [(target_embedding_path, embedding_install)]
        if not tied:
            installs.append(
                (
                    target_head_path,
                    _install_only_covered_rows(
                        target_head, head_rows, head_mask, role
                    ),
                )
            )
        # ``covered_output_rows`` / ``uncovered_output_rows`` describe the token
        # embedding, which is the interface the pair bridge is defined against.
        # A caller that gives the head a different weighting reads its own
        # figures from the ``lm_head_uncovered_target_*`` keys below.
        covered = int(embedding_mask.sum().item())
        report = {
            **_fold_only_operation_audit(),
            "schema": SCHEMA,
            "kind": "independent_vocabulary_pair_code_token_interface_bridge",
            "method_id": self.method_id,
            "bridge_kind": artifact.bridge_kind,
            "bridge_artifact_sha256": artifact.artifact_sha256,
            "pair_profile_fingerprint": artifact.profile_fingerprint,
            "parent_tokenizer_semantics_sha256": (
                artifact.parent_tokenizer_semantics_sha256
            ),
            "child_tokenizer_semantics_sha256": (
                artifact.child_tokenizer_semantics_sha256
            ),
            "embedding_weighting": self.embedding_transport,
            "lm_head_weighting": (
                self.embedding_transport if tied else self.lm_head_transport
            ),
            "embedding_transport_report_sha256": [
                value["report_sha256"] for value in embedding_reports
            ],
            "lm_head_transport_report_sha256": [
                value["report_sha256"] for value in head_reports
            ],
            "embedding_coordinate_sha256": embedding_map_sha256,
            "lm_head_coordinate_sha256": head_map_sha256,
            "source_row_capacity": profile.source_row_capacity,
            "source_vocabulary_size": profile.source_vocab_size,
            "target_row_capacity": profile.target_row_capacity,
            "target_vocabulary_size": profile.target_vocab_size,
            "covered_output_rows": covered,
            "uncovered_output_rows": profile.target_row_capacity - covered,
            "masked_row_policy": _PAIR_MASKED_ROW_POLICY,
            "installed_parameter_paths": list(paths),
            "target_tied_storage": tied,
            "random_target_rows": 0,
            "zero_gradient": True,
            **_uncovered_row_record("embedding", embedding_mask),
        }
        if not tied:
            report.update(_uncovered_row_record("lm_head", head_mask))
        incumbent = context.child_state_fingerprint
        blank_rows = [(target_embedding_path, target_embedding.detach().clone())]
        if not tied:
            blank_rows.append((target_head_path, target_head.detach().clone()))
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            _install(context, installs)
            for (path, blank), mask in zip(
                blank_rows,
                (embedding_mask,) if tied else (embedding_mask, head_mask),
                strict=True,
            ):
                live = context.child_model.get_parameter(path).detach()
                if not bool(torch.equal(live[~mask], blank[~mask])):
                    raise DynamicCausalOperatorPromotionError(
                        f"{role} overwrote a target row the pair bridge does "
                        f"not cover in {path!r}."
                    )
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return TokenInterfaceBridgeTransaction(
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


# ---------------------------------------------------------------------------
# Native (non-HF) trunks: windowed NoPE attention, fused gate/up MLP, and the
# chart-projected global interfaces.
# ---------------------------------------------------------------------------

#: What the coupled GQA kernel currently accepts.  The window/NoPE policy
#: fields are the parallel kernel wave's contract
#: (``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 1.3): they
#: live on :class:`HFCausalGQATopology`, so this module binds to whichever of
#: them has actually landed and refuses to pretend about the rest.
_GQA_TOPOLOGY_FIELDS = frozenset(
    descriptor.name for descriptor in fields(HFCausalGQATopology)
)
KERNEL_SUPPORTS_ATTENTION_WINDOW = "attention_window" in _GQA_TOPOLOGY_FIELDS
KERNEL_SUPPORTS_POSITION_ENCODING = "position_encoding" in _GQA_TOPOLOGY_FIELDS

#: RoPE evaluated at position zero is the identity map, exactly: ``cos(0) == 1``
#: and ``sin(0) == 0`` in IEEE double, so ``x * 1 + rotate_half(x) * 0 == x``
#: bit for bit.  That is how a NoPE branch runs through the rotary kernel
#: without a NoPE flag, and the operator proves it numerically before it fits
#: anything rather than asserting it in a comment.
_NOPE_ZERO_POSITION_TRANSPORT = "nope_zero_position_identity_rope"
_NOPE_KERNEL_FLAG_TRANSPORT = "nope_kernel_position_encoding_flag"
_WINDOW_BINDING_KERNEL = "kernel_band_mask"
_WINDOW_BINDING_INACTIVE = "inactive_on_sealed_evidence"
_WINDOW_BINDING_ABSENT = "no_window_declared"

_NORM_EPSILON_ATTRIBUTES = ("eps", "variance_epsilon", "epsilon")


def _native_evidence(evidence: FoldEvidence, role: str) -> NativeCircuitEvidence:
    payload = evidence.payload
    if not isinstance(payload, NativeCircuitEvidence):
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires a sealed NativeCircuitEvidence payload."
        )
    if payload.cache_key != evidence.fingerprint:
        raise DynamicCausalOperatorPromotionError(
            f"{role} evidence payload and fingerprint differ."
        )
    return payload.verify()


def _graph_hidden_size(graph: ModelCircuitGraph, role: str, name: str) -> int:
    value = dict(graph.metadata).get("hidden_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires a declared {name} residual width."
        )
    return value


def _nodes_for_action(
    graph: ModelCircuitGraph,
    node_ids: Sequence[str],
    role: str,
) -> tuple[Any, ...]:
    index = {node.node_id: node for node in graph.nodes}
    missing = sorted(set(node_ids) - set(index))
    if missing:
        raise DynamicCausalOperatorPromotionError(
            f"{role} action names circuits the graph does not contain: {missing}."
        )
    return tuple(index[value] for value in sorted(node_ids, key=lambda v: index[v].order))


def _single_node(nodes: Sequence[Any], kind: str, role: str, name: str) -> Any:
    selected = tuple(node for node in nodes if node.kind == kind)
    if len(selected) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires exactly one {name} circuit in the bundle."
        )
    return selected[0]


def _role_parameter_path(node: Any, wanted: str, role: str) -> str:
    matches = tuple(
        parameter.path
        for parameter in node.parameters
        if wanted in str(parameter.role).lower()
    )
    if len(matches) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} could not resolve exactly one {wanted!r} tensor on "
            f"{node.node_id!r}."
        )
    return matches[0]


def _norm_epsilon(model: Any, module_path: str, role: str) -> float:
    module = model.get_submodule(module_path)
    for name in _NORM_EPSILON_ATTRIBUTES:
        value = getattr(module, name, None)
        if isinstance(value, float) and math.isfinite(value) and value > 0.0:
            return float(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return float(value)
    raise DynamicCausalOperatorPromotionError(
        f"{role} cannot read the normalization epsilon of {module_path!r}; a "
        "normalization whose epsilon is unknown is refused rather than guessed."
    )


def _fit_metrics(metrics: Any) -> dict[str, Any]:
    return {
        "rows": int(metrics.rows),
        "output_width": int(metrics.output_width),
        "relative_rmse": float(metrics.relative_rmse),
        "mean_cosine_similarity": float(metrics.mean_cosine_similarity),
    }


def _split_rows(
    evidence: NativeCircuitEvidence,
    rows: Any,
) -> tuple[Any, Any, Any, Any]:
    """Flatten one capture into ``(all, fit, selection, validation)`` rows."""

    flat = evidence.flat_rows(rows)
    masks = evidence.row_split(rows)
    if any(int(value.numel()) != int(flat.shape[0]) for value in masks):
        raise DynamicCausalOperatorPromotionError(
            "A captured row set does not match its fit/selection/validation split."
        )
    return (flat, *(flat[value].contiguous() for value in masks))


class _NativeChartOperator:
    """Shared fail-closed preamble for the native chart-bound fold operators.

    Every one of these operators reads two things the HF lane gets for free: a
    sealed parent bundle captured at module boundaries instead of through a
    ``DecoderLayout``, and one global residual chart instead of a per-boundary
    chart set.  Both are supplied by the caller and authenticated here.
    """

    method_id = ""
    branch_id = ""

    #: A branch that writes nothing scores exactly one against its own target,
    #: so a branch at or above that number has not earned its install.  This is
    #: the design's endpoint fail-closed condition, enforced here rather than
    #: left to the outer evaluator.
    no_branch_endpoint_relative_rmse_baseline = 1.0

    #: The frozen ridge ladder.  A branch solve whose columns are a nonlinear
    #: expansion of a narrow residual is badly conditioned by construction; the
    #: ridge that repairs it is chosen on the *selection* partition and the
    #: whole ladder is sealed in the report, so the number is auditable rather
    #: than tuned into the validation endpoint.
    ridge_ladder = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)

    def __init__(
        self,
        *,
        chart: NativeResidualChart,
        recapture: Any,
        ridge: float = 1e-6,
        ridge_ladder: Sequence[float] | None = None,
        max_endpoint_relative_rmse: float | None = None,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        if not isinstance(chart, NativeResidualChart):
            raise DynamicCausalOperatorPromotionError(
                "chart must be a NativeResidualChart."
            )
        if not callable(recapture):
            raise DynamicCausalOperatorPromotionError(
                "recapture must be a callable live-child recapture."
            )
        self.chart = chart.verify()
        self.recapture = recapture
        self.ridge = _nonnegative_float(ridge, "ridge")
        ladder = self.ridge_ladder if ridge_ladder is None else tuple(ridge_ladder)
        if not ladder:
            raise DynamicCausalOperatorPromotionError(
                "ridge_ladder must offer at least one candidate."
            )
        self.ridge_ladder = tuple(
            _nonnegative_float(value, "ridge ladder entry") for value in ladder
        )
        bound = (
            self.no_branch_endpoint_relative_rmse_baseline
            if max_endpoint_relative_rmse is None
            else max_endpoint_relative_rmse
        )
        bound = _nonnegative_float(bound, "max_endpoint_relative_rmse")
        if bound > self.no_branch_endpoint_relative_rmse_baseline:
            raise DynamicCausalOperatorPromotionError(
                "max_endpoint_relative_rmse cannot exceed the no-branch "
                "baseline; a branch that loses to writing nothing is refused."
            )
        self.max_endpoint_relative_rmse = bound
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(self.snapshot_config, DynamicFoldSnapshotConfig):
            raise DynamicCausalOperatorPromotionError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": SCHEMA,
                "operator": type(self).__qualname__,
                "method_id": self.method_id,
                "branch": self.branch_id,
                "ridge": self.ridge,
                "ridge_ladder": list(self.ridge_ladder),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "snapshot_backend": self.snapshot_config.backend,
                "residual_chart_sha256": self.chart.map_sha256,
                "recapture": type(self.recapture).__qualname__,
                "kernel_attention_window": KERNEL_SUPPORTS_ATTENTION_WINDOW,
                "kernel_position_encoding": KERNEL_SUPPORTS_POSITION_ENCODING,
            }
        )

    def _require_endpoint(self, value: Any, role: str, name: str) -> float:
        if (
            not isinstance(value, float)
            or not math.isfinite(value)
            or value >= self.max_endpoint_relative_rmse
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} {name} {value!r} is at or above the no-branch baseline "
                f"bound {self.max_endpoint_relative_rmse!r}; a write that loses "
                "to writing nothing is a regression, not an install."
            )
        return float(value)

    def _bind(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
        role: str,
    ) -> tuple[NativeCircuitEvidence, ModelCircuitGraph, tuple[Any, ...], tuple[Any, ...]]:
        if action.method_id != self.method_id:
            raise DynamicCausalOperatorPromotionError(
                f"{role} received method {action.method_id!r}."
            )
        parent_name = _single_parent(action, context, role)
        sealed = _native_evidence(evidence, role)
        parent_graph = context.parent_graphs.get(parent_name)
        if parent_graph is None:
            raise DynamicCausalOperatorPromotionError(
                f"{role} source parent graph is not loaded."
            )
        if sealed.graph_fingerprint != parent_graph.fingerprint:
            raise DynamicCausalOperatorPromotionError(
                f"{role} evidence is bound to a different parent graph."
            )
        child_graph = context.child_graph
        source_width = _graph_hidden_size(parent_graph, role, "source")
        target_width = _graph_hidden_size(child_graph, role, "target")
        if target_width > source_width:
            raise DynamicCausalOperatorPromotionError(
                f"{role} is contraction-only; residual width expansion is "
                "outside the chart envelope."
            )
        if target_width != source_width and not _boundary_chart_required(action):
            raise DynamicCausalOperatorPromotionError(
                f"{role} received a residual-width crossing that the planner "
                "did not mark boundary_chart_required."
            )
        if (
            self.chart.source_width != source_width
            or self.chart.target_width != target_width
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} chart maps {self.chart.source_width} -> "
                f"{self.chart.target_width}, not {source_width} -> "
                f"{target_width}."
            )
        source_nodes = _nodes_for_action(
            parent_graph,
            self._source_ids(action),
            role,
        )
        target_nodes = _nodes_for_action(child_graph, action.target_node_ids, role)
        return sealed, parent_graph, source_nodes, target_nodes

    @staticmethod
    def _source_ids(action: FoldAction) -> tuple[str, ...]:
        return tuple(
            node_id
            for reference in action.sources
            for node_id in reference.node_ids
        )

    def _recapture_child(
        self,
        context: FoldExecutionContext,
        role: str,
    ) -> NativeCircuitEvidence:
        live = self.recapture(context.child_model, context.child_graph)
        if not isinstance(live, NativeCircuitEvidence):
            raise DynamicCausalOperatorPromotionError(
                f"{role} recapture did not return a NativeCircuitEvidence bundle."
            )
        if live.graph_fingerprint != context.child_graph.fingerprint:
            raise DynamicCausalOperatorPromotionError(
                f"{role} recapture is bound to a different child graph."
            )
        return live.verify()

    # -- the shared depth-selector fold -------------------------------------

    def _depth_selector_installs(
        self,
        *,
        selector: Any,
        parent_model: Any,
        child_model: Any,
        sealed: NativeCircuitEvidence,
        live: NativeCircuitEvidence,
        role: str,
    ) -> tuple[tuple[tuple[str, Any], ...], dict[str, Any]]:
        """Compile one learned depth-source selector, or say why it is dead.

        ``block_attnres`` never invokes the first block's attention selector --
        no block has completed at that depth -- so its tensors are dead weight
        there.  Those are installed deterministically and *recorded* as
        uninvoked instead of being fitted against rows that do not exist, and
        the IR's own ``invoked_at_this_depth`` declaration has to agree with
        what the forward actually did.
        """

        torch = _torch()
        attributes = dict(selector.attributes)
        query_path = _role_parameter_path(selector, "query", role)
        norm_weight_path = _role_parameter_path(selector, "norm", role)
        norm_module = owning_module_path(norm_weight_path)
        declared = attributes.get("invoked_at_this_depth")
        invoked = sealed.has(norm_module)
        if declared is not None and bool(declared) != invoked:
            raise DynamicCausalOperatorPromotionError(
                f"{role} depth selector {selector.node_id!r} declares "
                f"invoked_at_this_depth={declared!r} but the sealed parent "
                f"forward says {invoked!r}."
            )
        parent_query = parent_model.get_parameter(query_path).detach().to(
            device="cpu", dtype=torch.float64
        )
        if not invoked:
            return (
                (
                    (
                        norm_weight_path,
                        torch.ones(self.chart.target_width, dtype=torch.float64),
                    ),
                    (
                        query_path,
                        self.chart.fold_producer_weight(
                            parent_query.reshape(-1, 1)
                        ).reshape(-1),
                    ),
                ),
                {
                    "depth_selector_node": selector.node_id,
                    "depth_selector_invoked": False,
                    "depth_selector_method": (
                        "unexercised_deterministic_chart_transport"
                    ),
                    "depth_source_count": attributes.get("depth_source_count"),
                },
            )
        if not live.has(norm_module):
            raise DynamicCausalOperatorPromotionError(
                f"{role} parent invoked depth selector {selector.node_id!r} but "
                "the live child did not."
            )
        epsilon = _norm_epsilon(child_model, norm_module, role)
        parent_values = sealed.input(norm_module)
        parent_keys = sealed.output(norm_module)
        child_values = live.input(norm_module)
        if tuple(parent_values.shape[:-1]) != tuple(child_values.shape[:-1]):
            raise DynamicCausalOperatorPromotionError(
                f"{role} parent and child depth-source stacks disagree in shape."
            )
        _all_child, child_fit, _child_selection, child_validation = _split_rows(
            live, child_values
        )
        target = self.chart.encode(parent_keys)
        _all_target, target_fit, _target_selection, target_validation = _split_rows(
            sealed, target
        )
        norm_fold = fit_diagonal_norm(
            child_fit,
            target_fit,
            kind="rms_norm",
            epsilon=epsilon,
            has_bias=False,
            ridge=self.ridge,
            validation_input_rows=child_validation,
            validation_target_rows=target_validation,
        )
        child_keys = normalize_rows(
            live.flat_rows(child_values), kind="rms_norm", epsilon=epsilon
        ) * norm_fold.weight.reshape(1, -1)
        parent_scores = (sealed.flat_rows(parent_keys) @ parent_query).reshape(-1, 1)
        fit_mask, _selection_mask, validation_mask = live.row_split(child_values)
        query_fold = fit_ridge_linear(
            child_keys[fit_mask].contiguous(),
            parent_scores[fit_mask].contiguous(),
            ridge=self.ridge,
            fit_bias=False,
            validation_input_rows=child_keys[validation_mask].contiguous(),
            validation_target_rows=parent_scores[validation_mask].contiguous(),
        )
        return (
            (
                (norm_weight_path, norm_fold.weight),
                (query_path, query_fold.weight.reshape(-1)),
            ),
            {
                "depth_selector_node": selector.node_id,
                "depth_selector_invoked": True,
                "depth_selector_method": (
                    "closed_form_diagonal_rms_norm_then_pseudo_query_ridge"
                ),
                "depth_source_count": attributes.get("depth_source_count"),
                "depth_selector_norm_fit": _fit_metrics(norm_fold.train_metrics),
                "depth_selector_query_fit": _fit_metrics(query_fold.train_metrics),
                "validation_depth_norm_relative_rmse": float(
                    norm_fold.validation_metrics.relative_rmse
                ),
                "validation_depth_score_relative_rmse": float(
                    query_fold.validation_metrics.relative_rmse
                ),
            },
        )


def _require_nope_attention(
    attributes: Mapping[str, Any],
    role: str,
    side: str,
) -> None:
    """Refuse anything the windowed-NoPE operator cannot honestly execute."""

    if attributes.get("position_policy") != "nope":
        raise DynamicCausalOperatorPromotionError(
            f"{role} installs NoPE attention only; the {side} operator declares "
            f"position_policy={attributes.get('position_policy')!r}."
        )
    for name in (
        "rope_theta",
        "rope_type",
        "rope_scaling",
        "rope_variants",
        "partial_rotary_factor",
        "rotary_dim",
    ):
        if attributes.get(name) not in (None, "", False):
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses the {side} operator: it declares {name!r}, "
                "which only a rotary implementation can honour."
            )
    if (
        attributes.get("projection_style") != "split_qkv"
        or attributes.get("projection_bias") is not False
        or attributes.get("causal") is not True
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires a bias-free causal split-Q/K/V/O {side} operator."
        )
    if (
        attributes.get("fold_kernel_family_required")
        != "causal_sliding_window_split_qkv_attention"
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} refuses the {side} operator: it requires the "
            f"{attributes.get('fold_kernel_family_required')!r} kernel family, "
            "which this operator does not implement."
        )


def _attention_window(attributes: Mapping[str, Any], role: str, side: str) -> int | None:
    window = attributes.get("sliding_window")
    if window is None:
        return None
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise DynamicCausalOperatorPromotionError(
            f"{role} {side} operator declares an unusable sliding_window "
            f"{window!r}."
        )
    return window


class NativeWindowedNoPEAttentionBranchOperator(_NativeChartOperator):
    """Install one windowed NoPE local-attention branch through the GQA kernel.

    The design contract is explicit that local attention needs no new kernel:
    it is the deployed coupled GQA operator class with the rotation removed and
    a band added (``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section
    1.3).  This operator holds that line rather than forking 850 audited lines:

    * **NoPE.**  If the kernel exposes ``position_encoding`` it is used.  If it
      does not, the branch is still run honestly, because RoPE evaluated at
      position zero *is* the identity map -- and the operator proves that
      numerically on the real head width before it fits anything.
    * **Window.**  If the kernel exposes ``attention_window`` the band is
      threaded into the mask.  If it does not, the branch is admitted only when
      the sealed evidence proves the band never binds (every cached sequence is
      no longer than the window), which makes the kernel's causal mask exactly
      the parent's masked operator on that evidence.  Any other case fails
      closed naming the missing kernel flag; it is never approximated by
      dropping the band.

    The envelope this leaves behind is stated in the report, not hidden: when
    the band is inactive on the evidence, sequences longer than the window are
    outside what was fitted.
    """

    method_id = CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    branch_id = "windowed_nope_attention"

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        torch = _torch()
        role = "Native windowed-NoPE attention branch operator"
        sealed, parent_graph, source_nodes, target_nodes = self._bind(
            action, context, evidence, role
        )
        source_attention = _single_node(
            source_nodes, CircuitKind.ATTENTION.value, role, "source attention"
        )
        target_attention = _single_node(
            target_nodes, CircuitKind.ATTENTION.value, role, "target attention"
        )
        source_attributes = dict(source_attention.attributes)
        target_attributes = dict(target_attention.attributes)
        _require_nope_attention(source_attributes, role, "source")
        _require_nope_attention(target_attributes, role, "target")
        source_window = _attention_window(source_attributes, role, "source")
        target_window = _attention_window(target_attributes, role, "target")
        if source_window != target_window:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses a window change: the source band is "
                f"{source_window!r} and the target band is {target_window!r}. "
                "A window is a contract constant, not a fold coordinate."
            )
        target_norm = _single_node(
            target_nodes, CircuitKind.NORM.value, role, "target input norm"
        )
        norm_module = node_module_path(target_norm)
        norm_path = _role_parameter_path(target_norm, "norm", role)
        query_path = _role_parameter_path(target_attention, "q_proj", role)
        key_path = _role_parameter_path(target_attention, "k_proj", role)
        value_path = _role_parameter_path(target_attention, "v_proj", role)
        output_path = _role_parameter_path(target_attention, "out", role)
        selectors = tuple(
            node
            for node in target_nodes
            if node.kind == CircuitKind.RESIDUAL_ADD.value and tuple(node.parameters)
        )
        if len(selectors) > 1:
            raise DynamicCausalOperatorPromotionError(
                f"{role} admits at most one learned depth selector per branch."
            )
        selector_paths = tuple(
            parameter.path for node in selectors for parameter in node.parameters
        )
        paths = _installed_paths(
            action,
            (norm_path, query_path, key_path, value_path, output_path, *selector_paths),
            role,
        )
        source_operator_module = node_module_path(source_attention)
        target_operator_module = node_module_path(target_attention)
        if source_operator_module != target_operator_module:
            raise DynamicCausalOperatorPromotionError(
                f"{role} source and target attention modules differ: "
                f"{source_operator_module!r} against {target_operator_module!r}."
            )
        valid, fit, selection, validation = sealed.token_masks()
        positions = torch.zeros(
            (sealed.batch_size, sealed.sequence_length), dtype=torch.long
        )
        window_binding = _WINDOW_BINDING_ABSENT
        policy: dict[str, Any] = {}
        if source_window is not None:
            if KERNEL_SUPPORTS_ATTENTION_WINDOW:
                policy["attention_window"] = source_window
                window_binding = _WINDOW_BINDING_KERNEL
            elif source_window >= sealed.sequence_length:
                window_binding = _WINDOW_BINDING_INACTIVE
            else:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} cannot install a band that binds on the sealed "
                    f"evidence: window {source_window} against sequence length "
                    f"{sealed.sequence_length}, and HFCausalGQATopology does "
                    "not expose the 'attention_window' policy field yet. "
                    "Folding without the band would fit a different operator."
                )
        if KERNEL_SUPPORTS_POSITION_ENCODING:
            policy["position_encoding"] = "nope"
            position_transport = _NOPE_KERNEL_FLAG_TRANSPORT
            identity_proven = None
        else:
            position_transport = _NOPE_ZERO_POSITION_TRANSPORT
        topologies = []
        for attributes, side in (
            (source_attributes, "source"),
            (target_attributes, "target"),
        ):
            head_dim = _positive_int(attributes.get("head_dim"), f"{side} head_dim")
            topologies.append(
                HFCausalGQATopology(
                    query_heads=_positive_int(
                        attributes.get("query_heads"), f"{side} query_heads"
                    ),
                    key_value_heads=_positive_int(
                        attributes.get("key_value_heads"), f"{side} key_value_heads"
                    ),
                    head_dim=head_dim,
                    # Only read when a rotation is actually applied; under the
                    # NoPE policy it is inert, and under the zero-position
                    # fallback the rotation is the identity for any theta.
                    rope_theta=10000.0,
                    attention_scaling=float(head_dim) ** -0.5,
                    query_key_norm="identity",
                    **policy,
                )
            )
        source_topology, target_topology = topologies
        if not KERNEL_SUPPORTS_POSITION_ENCODING:
            probe = 1.0 + torch.arange(
                sealed.sequence_length * target_topology.head_dim,
                dtype=torch.float64,
            ).reshape(1, 1, sealed.sequence_length, target_topology.head_dim)
            identity_proven = bool(
                torch.equal(
                    apply_rope(
                        probe, positions[:1], theta=target_topology.rope_theta
                    ),
                    probe,
                )
            )
            if not identity_proven:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} could not prove the zero-position rotary transport "
                    "is the identity, so it refuses to run a NoPE branch "
                    "through the rotary kernel."
                )
        parent_model = context.parent_models[action.sources[0].parent_name]
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            selector_report: dict[str, Any] = {"depth_selector_present": bool(selectors)}
            if selectors:
                live = self._recapture_child(context, role)
                installs, selector_report = self._depth_selector_installs(
                    selector=selectors[0],
                    parent_model=parent_model,
                    child_model=context.child_model,
                    sealed=sealed,
                    live=live,
                    role=role,
                )
                selector_report["depth_selector_present"] = True
                _install(context, installs)
            live = self._recapture_child(context, role)
            epsilon = _norm_epsilon(context.child_model, norm_module, role)
            branch_input = live.input(norm_module)
            branch_normalized_target = self.chart.encode(sealed.output(norm_module))
            norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(branch_input, fit),
                _masked_rows(branch_normalized_target, fit),
                epsilon=epsilon,
                ridge=self.ridge,
            )
            normalized = _rmsnorm(branch_input, norm_fold.weight, epsilon)
            branch_delta_target = self.chart.encode(
                sealed.output(source_operator_module)
            )
            source_rows = {
                f"source_{name}_sequences": sealed.output(
                    owning_module_path(
                        _role_parameter_path(source_attention, projection, role)
                    )
                )
                for name, projection in (
                    ("query", "q_proj"),
                    ("key", "k_proj"),
                    ("value", "v_proj"),
                )
            }
            attention_fold = None
            selected_ridge = None
            selection_endpoint = None
            ladder: list[dict[str, Any]] = []
            for candidate in self.ridge_ladder:
                fold = compile_hf_causal_gqa_attention_fold(
                    normalized,
                    **source_rows,
                    target_residual_delta_sequences=branch_delta_target,
                    attention_mask=valid,
                    position_ids=positions,
                    valid_token_mask=valid,
                    fit_token_mask=fit,
                    validation_token_mask=validation,
                    source_topology=source_topology,
                    target_topology=target_topology,
                    ridge=candidate,
                )
                replayed, _scores, _probabilities = fold.replay(
                    normalized,
                    attention_mask=valid,
                    position_ids=positions,
                )
                score = _relative_rmse(
                    _masked_rows(replayed, selection),
                    _masked_rows(branch_delta_target, selection),
                )
                ladder.append({"ridge": candidate, "selection_relative_rmse": score})
                if (
                    math.isfinite(score)
                    and (selection_endpoint is None or score < selection_endpoint)
                ):
                    attention_fold = fold
                    selected_ridge = candidate
                    selection_endpoint = score
            if attention_fold is None:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} found no finite branch solve on the selection "
                    "partition."
                )
            delta, _scores, _probabilities = attention_fold.replay(
                normalized,
                attention_mask=valid,
                position_ids=positions,
            )
            endpoint = self._require_endpoint(
                _relative_rmse(
                    _masked_rows(delta, validation),
                    _masked_rows(branch_delta_target, validation),
                ),
                role,
                "validation attention-delta relative RMSE",
            )
            report = {
                **_fold_only_operation_audit(),
                **selector_report,
                "schema": SCHEMA,
                "kind": "native_windowed_nope_gqa_attention_branch",
                "method_id": self.method_id,
                "source_family": parent_graph.family,
                "target_family": context.child_graph.family,
                "source_node_id": source_attention.node_id,
                "target_node_id": target_attention.node_id,
                "cached_parent_evidence_sha256": sealed.cache_key,
                "residual_chart_sha256": self.chart.map_sha256,
                "residual_chart_source_width": self.chart.source_width,
                "residual_chart_target_width": self.chart.target_width,
                "per_boundary_chart_set": False,
                "position_policy": "nope",
                "position_transport": position_transport,
                "nope_identity_transport_verified": identity_proven,
                "attention_window": source_window,
                "attention_window_binding": window_binding,
                "sealed_sequence_length": sealed.sequence_length,
                "attention_window_envelope_note": (
                    "the band never binds on the sealed evidence, so sequences "
                    "longer than the window are outside the fitted envelope"
                    if window_binding == _WINDOW_BINDING_INACTIVE
                    else "the installed band is the kernel's own mask"
                ),
                "input_norm_report_sha256": norm_fold.report["report_sha256"],
                "attention_report_sha256": attention_fold.report["report_sha256"],
                "installed_parameter_paths": list(paths),
                "child_recaptures": int(getattr(self.recapture, "calls", 0)),
                "selected_ridge": selected_ridge,
                "ridge_ladder": ladder,
                "selection_endpoint_relative_rmse": selection_endpoint,
                "no_branch_endpoint_relative_rmse_baseline": (
                    self.no_branch_endpoint_relative_rmse_baseline
                ),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "validation_attention_delta_relative_rmse": endpoint,
                "parent_residual_coordinate_endpoints": False,
                "sequential_child_replay": True,
                "numeric_weight_matching_objective": False,
                "zero_gradient": True,
            }
            _install(
                context,
                (
                    (norm_path, norm_fold.weight),
                    (query_path, attention_fold.query_weight),
                    (key_path, attention_fold.key_weight),
                    (value_path, attention_fold.value_weight),
                    (output_path, attention_fold.output_weight),
                ),
            )
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class NativeFusedGatedMLPBranchOperator(_NativeChartOperator):
    """Install one dense SwiGLU branch whose gate and up share a fused tensor.

    The split is real work, not bookkeeping: the parent's fused output is
    chunked into its gate and up halves, the coupled SwiGLU solve runs on those
    halves against the chart-projected residual write, and the compiled gate
    and up matrices are re-fused into the child's ``[2 * intermediate, hidden]``
    layout inside the same snapshot-backed transaction.  Nothing is installed
    unless the re-fused tensor matches the live child parameter exactly.
    """

    method_id = CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    branch_id = "fused_gate_up_mlp"

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        torch = _torch()
        role = "Native fused gate/up MLP branch operator"
        sealed, parent_graph, source_nodes, target_nodes = self._bind(
            action, context, evidence, role
        )
        source_mlp = _single_node(
            source_nodes, CircuitKind.GATED_MLP.value, role, "source gated MLP"
        )
        target_mlp = _single_node(
            target_nodes, CircuitKind.GATED_MLP.value, role, "target gated MLP"
        )
        for node, side in ((source_mlp, "source"), (target_mlp, "target")):
            attributes = dict(node.attributes)
            if (
                attributes.get("fused_gate_up_projection") is not True
                or attributes.get("gate_up_split") != "chunk_two_on_last_dimension"
                or attributes.get("projection_layout") != "fused_gate_up_then_down"
                or attributes.get("projection_bias") is not False
                or attributes.get("channel_mixer") != "dense"
            ):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} requires a dense fused gate/up {side} MLP that "
                    "declares its chunk-two split."
                )
            if str(attributes.get("activation", "")).lower() not in _GATED_ACTIVATIONS:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} implements SiLU-gated targets only, not "
                    f"{attributes.get('activation')!r}."
                )
        source_intermediate = _positive_int(
            dict(source_mlp.attributes).get("intermediate_size"),
            "source intermediate_size",
        )
        target_intermediate = _positive_int(
            dict(target_mlp.attributes).get("intermediate_size"),
            "target intermediate_size",
        )
        if target_intermediate > source_intermediate:
            raise DynamicCausalOperatorPromotionError(
                f"{role} is contraction-only; the target MLP is wider than the "
                "cached source MLP."
            )
        target_norm = _single_node(
            target_nodes, CircuitKind.NORM.value, role, "target input norm"
        )
        norm_module = node_module_path(target_norm)
        norm_path = _role_parameter_path(target_norm, "norm", role)
        up_path = _role_parameter_path(target_mlp, "up_weight", role)
        down_path = _role_parameter_path(target_mlp, "down_weight", role)
        selectors = tuple(
            node
            for node in target_nodes
            if node.kind == CircuitKind.RESIDUAL_ADD.value and tuple(node.parameters)
        )
        if len(selectors) > 1:
            raise DynamicCausalOperatorPromotionError(
                f"{role} admits at most one learned depth selector per branch."
            )
        selector_paths = tuple(
            parameter.path for node in selectors for parameter in node.parameters
        )
        paths = _installed_paths(
            action, (norm_path, up_path, down_path, *selector_paths), role
        )
        source_up_module = owning_module_path(
            _role_parameter_path(source_mlp, "up_weight", role)
        )
        source_operator_module = node_module_path(source_mlp)
        _valid, fit, selection, validation = sealed.token_masks()
        parent_model = context.parent_models[action.sources[0].parent_name]
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            selector_report: dict[str, Any] = {"depth_selector_present": bool(selectors)}
            if selectors:
                live = self._recapture_child(context, role)
                installs, selector_report = self._depth_selector_installs(
                    selector=selectors[0],
                    parent_model=parent_model,
                    child_model=context.child_model,
                    sealed=sealed,
                    live=live,
                    role=role,
                )
                selector_report["depth_selector_present"] = True
                _install(context, installs)
            live = self._recapture_child(context, role)
            epsilon = _norm_epsilon(context.child_model, norm_module, role)
            branch_input = live.input(norm_module)
            branch_normalized_target = self.chart.encode(sealed.output(norm_module))
            norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(branch_input, fit),
                _masked_rows(branch_normalized_target, fit),
                epsilon=epsilon,
                ridge=self.ridge,
            )
            normalized = _rmsnorm(branch_input, norm_fold.weight, epsilon)
            fused = sealed.output(source_up_module)
            if int(fused.shape[-1]) != 2 * source_intermediate:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} cached fused gate/up width {int(fused.shape[-1])} "
                    f"is not twice the declared intermediate size "
                    f"{source_intermediate}."
                )
            gate_rows, up_rows = fused.chunk(2, dim=-1)
            branch_delta_target = self.chart.encode(
                sealed.output(source_operator_module)
            )
            mlp_fold = None
            selected_ridge = None
            selection_endpoint = None
            ladder: list[dict[str, Any]] = []
            for candidate in self.ridge_ladder:
                fold = fit_functional_swiglu(
                    _masked_rows(normalized, fit),
                    _masked_rows(gate_rows, fit),
                    _masked_rows(up_rows, fit),
                    _masked_rows(branch_delta_target, fit),
                    target_neurons=target_intermediate,
                    ridge=candidate,
                )
                score = _relative_rmse(
                    fold.predict(_masked_rows(normalized, selection)),
                    _masked_rows(branch_delta_target, selection),
                )
                ladder.append({"ridge": candidate, "selection_relative_rmse": score})
                if (
                    math.isfinite(score)
                    and (selection_endpoint is None or score < selection_endpoint)
                ):
                    mlp_fold = fold
                    selected_ridge = candidate
                    selection_endpoint = score
            if mlp_fold is None:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} found no finite branch solve on the selection "
                    "partition."
                )
            endpoint = self._require_endpoint(
                _relative_rmse(
                    mlp_fold.predict(_masked_rows(normalized, validation)),
                    _masked_rows(branch_delta_target, validation),
                ),
                role,
                "validation MLP-delta relative RMSE",
            )
            refused = torch.cat(
                (mlp_fold.gate_weight, mlp_fold.up_weight), dim=0
            ).contiguous()
            live_up = context.child_model.get_parameter(up_path)
            if tuple(refused.shape) != tuple(live_up.shape):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} re-fused gate/up tensor has shape "
                    f"{tuple(int(size) for size in refused.shape)}, but the "
                    f"child parameter is "
                    f"{tuple(int(size) for size in live_up.shape)}."
                )
            report = {
                **_fold_only_operation_audit(),
                **selector_report,
                "schema": SCHEMA,
                "kind": "native_fused_gate_up_swiglu_mlp_branch",
                "method_id": self.method_id,
                "source_family": parent_graph.family,
                "target_family": context.child_graph.family,
                "source_node_id": source_mlp.node_id,
                "target_node_id": target_mlp.node_id,
                "cached_parent_evidence_sha256": sealed.cache_key,
                "residual_chart_sha256": self.chart.map_sha256,
                "per_boundary_chart_set": False,
                "fused_gate_up_split": "chunk_two_on_last_dimension",
                "fused_gate_up_refused_in_transaction": True,
                "source_intermediate_size": source_intermediate,
                "target_intermediate_size": target_intermediate,
                "input_norm_report_sha256": norm_fold.report["report_sha256"],
                "swiglu_report_sha256": mlp_fold.report["report_sha256"],
                "installed_parameter_paths": list(paths),
                "child_recaptures": int(getattr(self.recapture, "calls", 0)),
                "selected_ridge": selected_ridge,
                "ridge_ladder": ladder,
                "selection_endpoint_relative_rmse": selection_endpoint,
                "no_branch_endpoint_relative_rmse_baseline": (
                    self.no_branch_endpoint_relative_rmse_baseline
                ),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "validation_mlp_delta_relative_rmse": endpoint,
                "parent_residual_coordinate_endpoints": False,
                "sequential_child_replay": True,
                "numeric_weight_matching_objective": False,
                "zero_gradient": True,
            }
            _install(
                context,
                (
                    (norm_path, norm_fold.weight),
                    (up_path, refused),
                    (down_path, mlp_fold.down_weight),
                ),
            )
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


# ---------------------------------------------------------------------------
# chimera sequence-mixer families: the bounded gated delta recurrence (KDA) and
# the composite local-window + compressed-summary attention (CSA/HCA).
# ---------------------------------------------------------------------------

#: Where the parent rows of each chimera family come from.  Every one of these
#: is a *module boundary in the sealed one-forward bundle*, resolved from the
#: node's own parameter roles rather than from a hard-coded attribute name, so
#: an operator that cannot find one of them says which role is missing instead
#: of reaching into the parent.
_SEALED_ROW_SOURCE = "sealed_native_module_boundary_capture"
#: Rows that no module boundary produces and that are therefore *derived* from
#: captured rows and carried parent weights.  Naming them is the point: the
#: kernel's own unroll gate then measures the derivation against a boundary the
#: parent really did emit (the pre-norm recurrent output), so nothing here is
#: asserted.
_KDA_DERIVED_ROWS = (
    "retention_rows",
    "beta_rows",
    "gate_rows",
    "decay_logit_rows",
    "state_snapshots",
)
_SUMMARY_DERIVED_ROWS = (
    "raw_summaries",
    "summary_scores",
    "summary_probabilities",
    "local_probabilities",
)
#: gated-MLA rows no boundary emits: the key/value split of the readout, the
#: head-broadcast shared key block, the scores, the probabilities, the sigmoid
#: gate and the context.  The kernel's own reconstruction gate then rebuilds the
#: *captured* branch delta from exactly those, so none of them is asserted.
_LATENT_KV_DERIVED_ROWS = (
    "base_key_rows",
    "value_rows",
    "shared_key_broadcast",
    "attention_scores",
    "attention_probabilities",
    "output_gate_rows",
)
#: Stable LatentMoE rows no boundary emits: the routing itself.  The router's
#: *logits* are captured; the sigmoid scores, the biased top-k selection, the
#: renormalized mixture weights and the per-expert row sets are recovered from
#: them plus the carried balancing buffer, and the kernel's routing identity
#: then checks the recovery against captured per-expert row counts and the
#: captured accumulator.
_STABLE_LATENT_MOE_DERIVED_ROWS = (
    "router_scores",
    "expert_indices",
    "expert_mixture_weights",
    "expert_row_sets",
)


def _f64(value: Any) -> Any:
    torch = _torch()
    return value.detach().to(device="cpu", dtype=torch.float64)


def _row_positions(batch: int, sequence: int) -> Any:
    """Token positions for a kernel replay signature that still takes them.

    Under the NoPE policy the rotary transport is the identity, so these are
    inert; the kernel's signature keeps them because the same code path serves
    a rotating topology.
    """

    torch = _torch()
    return torch.arange(int(sequence)).reshape(1, -1).expand(int(batch), int(sequence))


def _exact_role_parameter_path(node: Any, wanted: str, role: str) -> str:
    """Resolve one tensor by the exact trailing component of its IR role.

    :func:`_role_parameter_path` matches a substring, which is right when the
    role names are disjoint.  A composite operator breaks that: a compressed
    summary node owns both ``local.out`` and ``out``, so ``"out"`` names two
    tensors.  This matches the role's last component exactly instead.
    """

    matches = tuple(
        parameter.path
        for parameter in node.parameters
        if str(parameter.role).rpartition(".")[2] == wanted
    )
    if len(matches) != 1:
        raise DynamicCausalOperatorPromotionError(
            f"{role} could not resolve exactly one {wanted!r} tensor on "
            f"{node.node_id!r}; found {len(matches)}."
        )
    return matches[0]


def _submodule(model: Any, module_path: str, role: str) -> Any:
    try:
        return model.get_submodule(module_path)
    except (AttributeError, KeyError) as exc:
        raise DynamicCausalOperatorPromotionError(
            f"{role} cannot resolve the module {module_path!r} the graph names."
        ) from exc


def _declared_matches_module(
    attributes: Mapping[str, Any],
    measured: Mapping[str, Any],
    role: str,
    side: str,
) -> dict[str, Any]:
    """Fail closed unless the IR's declared geometry is the live module's.

    The IR is what the planner contracted against and the live module is what
    the kernel will read.  If they disagree, the plan was approved for an
    operator that is not the one about to be folded, so the transaction is
    refused rather than silently trusting either source.
    """

    disagreements = {
        name: (attributes.get(name), value)
        for name, value in measured.items()
        if attributes.get(name) != value
    }
    if disagreements:
        raise DynamicCausalOperatorPromotionError(
            f"{role} refuses the {side} operator: the graph declares "
            + ", ".join(
                f"{name}={declared!r} but the live module reports {actual!r}"
                for name, (declared, actual) in sorted(disagreements.items())
            )
            + "."
        )
    return dict(measured)


class _ChimeraMixerBranchOperator(_NativeChartOperator):
    """Shared fail-closed preamble for the two chimera sequence-mixer folds.

    Both families install exactly the same shape of transaction: the branch's
    input normalization plus every tensor of one mixer module, fitted from the
    live child's own rows against the chart-projected parent write, inside one
    snapshot.  What differs is the kernel in the middle, so only that is left
    to the subclasses.

    Two things these operators do that the local/dense pair does not:

    * they **assemble a family evidence record** for their kernel out of the
      sealed module-boundary bundle, reading no parent forward at all.  Rows a
      module boundary really produced are used as captured; rows that no
      boundary produces are derived from captured rows and carried parent
      weights, named as derived in the receipt, and then checked by the
      kernel's own evidence-consistency gate against a boundary the parent did
      emit;
    * they **surface the kernel's refusals**.  Both kernels enforce the design's
      1.0 no-branch floor after a BF16 replay and refuse to emit a child on
      breach.  Each rung of the ridge ladder records either its selection
      endpoint or the kernel's refusal message, and a ladder on which every
      rung refused fails the transaction closed with those messages rather than
      installing anything.
    """

    #: Ridge-independent evidence-consistency gates run once, before the ladder,
    #: so a genuinely inconsistent cache fails closed with the kernel's own
    #: message instead of being reported as "no rung produced a finite solve".
    evidence_gate_stage = "before_ridge_ladder"

    def _branch_targets(
        self,
        target_nodes: Sequence[Any],
        action: FoldAction,
        role: str,
        operator_kind: str,
    ) -> tuple[Any, Any, str, str, str, tuple[Any, ...], tuple[str, ...]]:
        """Resolve the target norm, the target mixer and the installed paths."""

        target_operator = _single_node(
            target_nodes, operator_kind, role, "target sequence mixer"
        )
        target_norm = _single_node(
            target_nodes, CircuitKind.NORM.value, role, "target input norm"
        )
        norm_module = node_module_path(target_norm)
        norm_path = _role_parameter_path(target_norm, "norm", role)
        mixer_module = node_module_path(target_operator)
        selectors = tuple(
            node
            for node in target_nodes
            if node.kind == CircuitKind.RESIDUAL_ADD.value and tuple(node.parameters)
        )
        if len(selectors) > 1:
            raise DynamicCausalOperatorPromotionError(
                f"{role} admits at most one learned depth selector per branch."
            )
        selector_paths = tuple(
            parameter.path for node in selectors for parameter in node.parameters
        )
        mixer_paths = tuple(target_operator.parameter_paths)
        paths = _installed_paths(
            action, (norm_path, *mixer_paths, *selector_paths), role
        )
        return (
            target_operator,
            target_norm,
            norm_module,
            norm_path,
            mixer_module,
            selectors,
            paths,
        )

    def _child_installs(
        self,
        *,
        child_state: Mapping[str, Any],
        mixer_module: str,
        target_operator: Any,
        context: FoldExecutionContext,
        role: str,
    ) -> tuple[tuple[str, Any], ...]:
        """Map a kernel's chimera-named child state onto the IR's own paths.

        The kernels emit exactly the parameter names of the target module.  The
        graph independently says which tensors that node owns.  Requiring the
        two sets to be equal is the same assertion the kernel suites make
        (``set(emitted) == set(child.state_dict())``) and it is what stops a
        partially installed mixer from ever reaching the child.
        """

        installs = {
            f"{mixer_module}.{name}": value for name, value in child_state.items()
        }
        declared = set(target_operator.parameter_paths)
        if set(installs) != declared:
            raise DynamicCausalOperatorPromotionError(
                f"{role} kernel emitted {sorted(set(installs) - declared)!r} and "
                f"omitted {sorted(declared - set(installs))!r} against the "
                f"tensors {target_operator.node_id!r} owns."
            )
        for path, value in installs.items():
            live = context.child_model.get_parameter(path)
            if tuple(int(size) for size in value.shape) != tuple(
                int(size) for size in live.shape
            ):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} compiled {path!r} with shape "
                    f"{tuple(int(size) for size in value.shape)}, but the child "
                    f"parameter is {tuple(int(size) for size in live.shape)}."
                )
        return tuple(sorted(installs.items()))

    @staticmethod
    def _ladder_rung(
        ladder: list[dict[str, Any]],
        candidate: float,
        compile_rung: Any,
        score_rung: Any,
        refusals: tuple[type[Exception], ...],
    ) -> tuple[Any, float] | None:
        """Compile one ridge rung, recording a kernel refusal as a refusal."""

        try:
            fold = compile_rung(candidate)
        except refusals as exc:
            ladder.append({"ridge": candidate, "kernel_refusal": str(exc)})
            return None
        score = score_rung(fold)
        ladder.append({"ridge": candidate, "selection_relative_rmse": score})
        if not math.isfinite(score):
            return None
        return fold, score

    def _select_over_ridge_ladder(
        self,
        *,
        compile_rung: Any,
        score_rung: Any,
        refusals: tuple[type[Exception], ...],
        role: str,
    ) -> tuple[Any, float, float, list[dict[str, Any]]]:
        ladder: list[dict[str, Any]] = []
        selected = None
        selected_ridge = None
        selected_score = None
        for candidate in self.ridge_ladder:
            outcome = self._ladder_rung(
                ladder, candidate, compile_rung, score_rung, refusals
            )
            if outcome is None:
                continue
            fold, score = outcome
            if selected_score is None or score < selected_score:
                selected, selected_ridge, selected_score = fold, candidate, score
        if selected is None:
            raise DynamicCausalOperatorPromotionError(
                f"{role} found no installable branch solve on the selection "
                "partition; every ridge rung either refused or was not finite: "
                + "; ".join(
                    f"ridge {entry['ridge']}: "
                    + str(
                        entry.get("kernel_refusal")
                        or entry.get("selection_relative_rmse")
                    )
                    for entry in ladder
                )
            )
        return selected, selected_ridge, selected_score, ladder


class BoundedGatedDeltaRecurrenceLayerOperator(_ChimeraMixerBranchOperator):
    """Install one bounded gated delta-rule recurrence branch (chimera KDA).

    The executable is ``hf_causal_kda_fold.compile_hf_causal_kda_fold``: exact
    effective-weight unroll, barycentric head contraction, balanced-ranked
    channel-aligned selection, inverse-chain decay/beta/gate ridges, an
    installed-recurrence replay and an endpoint-coupled writer, all closed form
    (design section 1.1).  This operator's whole job is to hand that kernel a
    real evidence record built from the sealed one-forward bundle and to make
    its refusals planner-visible.

    Where the parent rows come from, stated exactly:

    * ``pre_conv_*`` are the captured outputs of the three input projections;
    * ``query_rows`` / ``key_rows`` / ``value_rows`` are the captured outputs of
      the three depthwise short-convolution modules -- the parent's real
      post-convolution, post-SiLU rows -- with only the parent's own L2
      normalization applied to the two key-side ones.  Fitting them from the
      carried taps instead would make the kernel's conv-carry gate compare a
      replay with itself;
    * ``recurrent_output_rows`` is the captured *input* of the headwise output
      norm, which is the only boundary at which ``o_t`` exists as a value;
    * ``retention`` / ``beta`` / ``gate`` / ``decay logits`` are derived from
      captured projection outputs and the carried decay parameters, because no
      module boundary emits them.  The kernel's unroll gate then reconstructs
      ``o_t`` from those derived rows and compares it against the captured
      ``recurrent_output_rows``; a wrong derivation cannot survive that;
    * ``state_snapshots`` are stepped with the parent's own static
      ``delta_rule_update`` over those rows.  That is a method call on cached
      tensors, not a module forward, so the fold still performs none.
    """

    method_id = CAUSAL_KDA_RECURRENT_BRANCH_METHOD
    branch_id = "bounded_gated_delta_recurrence"

    #: Chunk boundaries at which the parent and child recurrence states are
    #: compared.  The design's state-drift gate needs more than one boundary to
    #: measure growth against the horizon at all.
    state_snapshot_stride = 8

    def __init__(
        self,
        *,
        state_snapshot_stride: int | None = None,
        fold_policy: ChimeraKDAFoldPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if state_snapshot_stride is not None:
            self.state_snapshot_stride = _positive_int(
                state_snapshot_stride, "state_snapshot_stride"
            )
        # The kernel's own sealed thresholds -- unroll and conv-carry
        # tolerances, saturation and dispersion fractions, the state-drift cap.
        # They are the design's fail-closed conditions, so they belong to the
        # transaction and are sealed in its fingerprint; a caller may tighten
        # them but never reaches past them.
        self.fold_policy = fold_policy or ChimeraKDAFoldPolicy()
        if not isinstance(self.fold_policy, ChimeraKDAFoldPolicy):
            raise DynamicCausalOperatorPromotionError(
                "fold_policy must be a ChimeraKDAFoldPolicy."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "base": super().resource_fingerprint,
                "state_snapshot_stride": int(self.state_snapshot_stride),
                "fold_policy": self.fold_policy.as_dict(),
            }
        )

    def _geometry(
        self,
        model: Any,
        mixer_module: str,
        attributes: Mapping[str, Any],
        hidden_size: int,
        role: str,
        side: str,
    ) -> tuple[Any, ChimeraKDAGeometry]:
        module = _submodule(model, mixer_module, role)
        try:
            geometry = chimera_kda_geometry(module)
        except HFCausalKDAFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot read the {side} bounded gated delta recurrence "
                f"geometry: {exc}"
            ) from exc
        if geometry.hidden_size != hidden_size:
            raise DynamicCausalOperatorPromotionError(
                f"{role} {side} recurrence reads a residual width of "
                f"{geometry.hidden_size}, not the chart's {hidden_size}."
            )
        _declared_matches_module(
            attributes,
            {
                "heads": geometry.heads,
                "key_head_dim": geometry.key_head_dim,
                "value_head_dim": geometry.value_head_dim,
                "decay_rank": geometry.decay_rank,
                "internal_short_conv_kernel_size": geometry.conv_kernel_size,
                "gate_lower_bound": geometry.gate_lower_bound,
                "state_elements": geometry.heads
                * geometry.key_head_dim
                * geometry.value_head_dim,
            },
            role,
            side,
        )
        return module, geometry

    def _layer_evidence(
        self,
        *,
        node: Any,
        module: Any,
        geometry: ChimeraKDAGeometry,
        sealed: NativeCircuitEvidence,
        norm_module: str,
        mixer_module: str,
        layer_index: int,
        role: str,
    ) -> ChimeraKDALayerEvidence:
        torch = _torch()

        def rows(path: str) -> Any:
            if not sealed.has(path):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} needs the sealed module boundary {path!r}; the "
                    "parent bundle does not carry it."
                )
            capture = sealed.capture(path)
            if capture.call_count != 1:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} refuses {path!r}: it ran {capture.call_count} times "
                    "in the sealed forward, so 'the' cached rows are ambiguous."
                )
            return sealed.output(path)

        def module_of(wanted: str) -> str:
            return owning_module_path(
                _exact_role_parameter_path(node, wanted, role)
            )

        def heads(value: Any, width: int) -> Any:
            return value.reshape(
                int(value.shape[0]), int(value.shape[1]), geometry.heads, width
            )

        def normalize(value: Any) -> Any:
            return value / value.square().sum(dim=-1, keepdim=True).clamp_min(
                1e-30
            ).sqrt()

        pre_conv_query = rows(module_of("q_proj_weight"))
        pre_conv_key = rows(module_of("k_proj_weight"))
        pre_conv_value = rows(module_of("v_proj_weight"))
        # ``q_conv``/``k_conv``/``v_conv`` wrap their own ``Conv1d``; the wrapper
        # is the boundary whose output is post-convolution and post-SiLU in the
        # trunk's own row layout.
        query_rows = normalize(
            heads(
                rows(owning_module_path(module_of("q_conv_conv_weight"))),
                geometry.key_head_dim,
            )
        )
        key_rows = normalize(
            heads(
                rows(owning_module_path(module_of("k_conv_conv_weight"))),
                geometry.key_head_dim,
            )
        )
        value_rows = heads(
            rows(owning_module_path(module_of("v_conv_conv_weight"))),
            geometry.value_head_dim,
        )
        decay_bias = _f64(module.decay_bias).reshape(
            1, 1, geometry.heads, geometry.key_head_dim
        )
        decay_logits = heads(
            rows(module_of("decay_up_weight")), geometry.key_head_dim
        ) + decay_bias
        scale = _f64(module.decay_log_scale).exp().reshape(1, 1, geometry.heads, 1)
        retention = torch.exp(
            geometry.gate_lower_bound * torch.sigmoid(scale * decay_logits)
        )
        beta = torch.sigmoid(rows(module_of("write_strength_weight")))
        gate = torch.sigmoid(
            heads(rows(module_of("output_gate_weight")), geometry.value_head_dim)
        )
        recurrent = sealed.input(module_of("output_norm_weight"))
        if tuple(int(size) for size in recurrent.shape[2:]) != (
            geometry.heads,
            geometry.value_head_dim,
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} captured a headwise output-norm input of shape "
                f"{tuple(int(size) for size in recurrent.shape)}, which is not "
                "the recurrence output this geometry predicts."
            )
        delta = rows(mixer_module)
        block_input = sealed.input(norm_module)
        update = getattr(module, "delta_rule_update", None)
        if not callable(update):
            raise DynamicCausalOperatorPromotionError(
                f"{role} needs the parent's own delta_rule_update to trace its "
                "recurrence state; the module does not expose one."
            )
        batch, sequence = int(delta.shape[0]), int(delta.shape[1])
        state = torch.zeros(
            batch,
            geometry.heads,
            geometry.key_head_dim,
            geometry.value_head_dim,
            dtype=torch.float64,
        )
        snapshots = []
        positions = []
        stride = int(self.state_snapshot_stride)
        for position in range(sequence):
            state = update(
                state,
                key_rows[:, position],
                value_rows[:, position],
                retention[:, position],
                beta[:, position],
            )
            if position == sequence - 1 or (position + 1) % stride == 0:
                snapshots.append(state.clone())
                positions.append(position)
        try:
            return ChimeraKDALayerEvidence(
                index=int(layer_index),
                block_input=block_input,
                attention_norm_output=rows(norm_module),
                pre_conv_query=pre_conv_query,
                pre_conv_key=pre_conv_key,
                pre_conv_value=pre_conv_value,
                query_rows=query_rows,
                key_rows=key_rows,
                value_rows=value_rows,
                retention_rows=retention,
                decay_logit_rows=decay_logits,
                beta_rows=beta,
                recurrent_output_rows=recurrent,
                gate_rows=gate,
                attention_delta=delta,
                block_output=block_input + delta,
                state_snapshots=torch.stack(snapshots, dim=0),
                state_snapshot_positions=tuple(positions),
                geometry=geometry,
                metadata={
                    "schema": SCHEMA,
                    "component": "chimera_kda_evidence_from_sealed_module_boundaries",
                    "parent_forward_calls": 0,
                    "training_steps": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                    "captured_row_source": _SEALED_ROW_SOURCE,
                    "derived_rows": list(_KDA_DERIVED_ROWS),
                    "state_snapshot_stride": stride,
                },
            )
        except HFCausalKDAFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} could not assemble a KDA evidence record from the "
                f"sealed bundle: {exc}"
            ) from exc

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        role = "Bounded gated delta recurrence layer operator"
        sealed, parent_graph, source_nodes, target_nodes = self._bind(
            action, context, evidence, role
        )
        source_operator = _single_node(
            source_nodes,
            CircuitKind.RECURRENT_MIXER.value,
            role,
            "source recurrent mixer",
        )
        (
            target_operator,
            _target_norm,
            norm_module,
            norm_path,
            mixer_module,
            selectors,
            paths,
        ) = self._branch_targets(
            target_nodes, action, role, CircuitKind.RECURRENT_MIXER.value
        )
        source_mixer_module = node_module_path(source_operator)
        if source_mixer_module != mixer_module:
            raise DynamicCausalOperatorPromotionError(
                f"{role} source and target mixer modules differ: "
                f"{source_mixer_module!r} against {mixer_module!r}."
            )
        parent_model = context.parent_models[action.sources[0].parent_name]
        source_module, source_geometry = self._geometry(
            parent_model,
            mixer_module,
            dict(source_operator.attributes),
            self.chart.source_width,
            role,
            "source",
        )
        _target_module, target_geometry = self._geometry(
            context.child_model,
            mixer_module,
            dict(target_operator.attributes),
            self.chart.target_width,
            role,
            "target",
        )
        try:
            parameters = chimera_kda_parameters(
                source_module, geometry=source_geometry
            )
        except HFCausalKDAFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot carry the parent recurrence parameters: {exc}"
            ) from exc
        valid, fit, selection, validation = sealed.token_masks()
        layer_evidence = self._layer_evidence(
            node=source_operator,
            module=source_module,
            geometry=source_geometry,
            sealed=sealed,
            norm_module=norm_module,
            mixer_module=mixer_module,
            layer_index=int(target_operator.layer_index or 0),
            role=role,
        )
        policy = self.fold_policy
        try:
            # The evidence-consistency gate is ridge-independent, so it runs
            # once, here, and a breach fails closed with the kernel's own
            # measurement instead of being reported as a bad ridge.
            unroll = kda_effective_weights(
                layer_evidence,
                row_mask=valid,
                tolerance=policy.unroll_reconstruction_tolerance,
            )
        except HFCausalKDAFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses this branch: the sealed evidence is not "
                f"consistent with a bounded gated delta recurrence -- {exc}"
            ) from exc
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            selector_report: dict[str, Any] = {"depth_selector_present": bool(selectors)}
            if selectors:
                live = self._recapture_child(context, role)
                installs, selector_report = self._depth_selector_installs(
                    selector=selectors[0],
                    parent_model=parent_model,
                    child_model=context.child_model,
                    sealed=sealed,
                    live=live,
                    role=role,
                )
                selector_report["depth_selector_present"] = True
                _install(context, installs)
            live = self._recapture_child(context, role)
            epsilon = _norm_epsilon(context.child_model, norm_module, role)
            branch_input = live.input(norm_module)
            branch_normalized_target = self.chart.encode(sealed.output(norm_module))
            norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(branch_input, fit),
                _masked_rows(branch_normalized_target, fit),
                epsilon=epsilon,
                ridge=self.ridge,
            )
            normalized = _rmsnorm(branch_input, norm_fold.weight, epsilon)
            branch_delta_target = self.chart.encode(layer_evidence.attention_delta)

            def compile_rung(candidate: float) -> Any:
                return compile_hf_causal_kda_fold(
                    layer_evidence,
                    parameters,
                    target_geometry,
                    fit_token_mask=fit,
                    validation_token_mask=validation,
                    valid_token_mask=valid,
                    child_normalized_rows=normalized,
                    endpoint_target=branch_delta_target,
                    policy=replace(policy, ridge=candidate),
                    unroll=unroll,
                )

            def score_rung(fold: Any) -> float:
                replayed, _recurrent, _snapshots = fold.replay(normalized)
                return _relative_rmse(
                    _masked_rows(replayed, selection),
                    _masked_rows(branch_delta_target, selection),
                )

            (
                kda_fold,
                selected_ridge,
                selection_endpoint,
                ladder,
            ) = self._select_over_ridge_ladder(
                compile_rung=compile_rung,
                score_rung=score_rung,
                refusals=(HFCausalKDAFoldError,),
                role=role,
            )
            delta, _recurrent, _snapshots = kda_fold.replay(normalized)
            endpoint = self._require_endpoint(
                _relative_rmse(
                    _masked_rows(delta, validation),
                    _masked_rows(branch_delta_target, validation),
                ),
                role,
                "validation recurrence-delta relative RMSE",
            )
            receipt = dict(kda_fold.report)
            installs = self._child_installs(
                child_state=kda_fold.child_state_dict(),
                mixer_module=mixer_module,
                target_operator=target_operator,
                context=context,
                role=role,
            )
            report = {
                **_fold_only_operation_audit(),
                **selector_report,
                "schema": SCHEMA,
                "kind": "native_bounded_gated_delta_recurrence_branch",
                "method_id": self.method_id,
                "source_family": parent_graph.family,
                "target_family": context.child_graph.family,
                "source_node_id": source_operator.node_id,
                "target_node_id": target_operator.node_id,
                "cached_parent_evidence_sha256": sealed.cache_key,
                "residual_chart_sha256": self.chart.map_sha256,
                "residual_chart_source_width": self.chart.source_width,
                "residual_chart_target_width": self.chart.target_width,
                "per_boundary_chart_set": False,
                "position_policy": "nope",
                "source_geometry": source_geometry.as_dict(),
                "target_geometry": target_geometry.as_dict(),
                "captured_row_source": _SEALED_ROW_SOURCE,
                "derived_rows": list(_KDA_DERIVED_ROWS),
                "derived_row_consistency_gate": (
                    "kda_effective_weight_unroll_against_the_captured_"
                    "pre_norm_recurrent_output"
                ),
                "unroll_reconstruction_relative_rmse": unroll.relative_rmse,
                "unroll_report_sha256": unroll.diagnostics["report_sha256"],
                "conv_tap_carry_relative_rmse": receipt["conv_tap_carry_relative_rmse"],
                "state_snapshot_stride": int(self.state_snapshot_stride),
                "kernel_fold_policy": policy.as_dict(),
                "state_snapshot_positions": receipt["state_snapshot_positions"],
                "state_snapshot_relative_rmse": receipt["state_snapshot_relative_rmse"],
                "state_drift_growth_ratio": receipt["state_drift_growth_ratio"],
                "state_drift_gate_applicable": receipt["state_drift_gate_applicable"],
                "state_snapshot_source": (
                    "parent_delta_rule_update_stepped_over_sealed_rows"
                ),
                "input_norm_report_sha256": norm_fold.report["report_sha256"],
                "recurrence_report_sha256": receipt["report_sha256"],
                "installed_parameter_paths": list(paths),
                "child_recaptures": int(getattr(self.recapture, "calls", 0)),
                "selected_ridge": selected_ridge,
                "ridge_ladder": ladder,
                "kernel_refused_ridge_rungs": sum(
                    1 for entry in ladder if "kernel_refusal" in entry
                ),
                "selection_endpoint_relative_rmse": selection_endpoint,
                "no_branch_endpoint_relative_rmse_baseline": (
                    self.no_branch_endpoint_relative_rmse_baseline
                ),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "validation_recurrence_delta_relative_rmse": endpoint,
                "validation_residual_relative_rmse_bf16": receipt[
                    "validation_residual_relative_rmse_bf16"
                ],
                "parent_residual_coordinate_endpoints": False,
                "sequential_child_replay": True,
                "numeric_weight_matching_objective": False,
                "zero_gradient": True,
            }
            _install(context, (*installs, (norm_path, norm_fold.weight)))
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class CompressedSummaryAttentionLayerOperator(_ChimeraMixerBranchOperator):
    """Install one composite local-window plus compressed-summary branch.

    One executable covers ``csa`` and ``hca``:
    ``hf_causal_sparse_latent_fold.compile_hf_causal_sparse_latent_fold`` is
    parameterized by the summary stride, and the stride is read off the IR node
    rather than off a method id.  A stride change is refused by the kernel as a
    kind translation, and the planner refuses it first, so the two families can
    never be folded into one another by accident.

    The evidence record is assembled from the sealed one-forward bundle with no
    parent forward:

    * ``local_delta`` is the captured output of the nested local-attention
      module and ``summary_delta`` the captured output of the summary writer,
      so the kernel's split gate compares two independently captured values
      against the captured branch write rather than against their own sum;
    * the six projection row sets are the captured outputs of the six
      projections, and the summary key/value rows already carry the parent's
      prepended zero summary because they were captured downstream of it;
    * the pooled summaries and both probability tensors are recomputed with the
      kernel's own pooling and softmax helpers, which are the parent's operators
      and are asserted to be so in the kernel's suite.
    """

    method_id = CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD
    branch_id = "compressed_summary_attention"

    def __init__(
        self,
        *,
        fold_policy: ChimeraSparseLatentFoldPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # The kernel's own sealed thresholds: the sub-branch split tolerance,
        # the minimum summary probability overlap, and the alternation budget.
        self.fold_policy = fold_policy or ChimeraSparseLatentFoldPolicy()
        if not isinstance(self.fold_policy, ChimeraSparseLatentFoldPolicy):
            raise DynamicCausalOperatorPromotionError(
                "fold_policy must be a ChimeraSparseLatentFoldPolicy."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "base": super().resource_fingerprint,
                "fold_policy": self.fold_policy.as_dict(),
            }
        )

    def _transported_child_state(
        self,
        *,
        child_state: Mapping[str, Any],
        mixer_module: str,
        context: FoldExecutionContext,
        role: str,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Carry the sub-branch writers into the child's residual coordinates.

        The kernel fits both sub-branches against the parent's own sub-branch
        deltas, because that is where its split gate captured them, so both
        writers come back mapping into the *parent* residual width.  Each of
        them is the last linear map in its sub-branch, so

            ``context @ (E^T W)^T == (context @ W^T) @ E``

        holds identically: transporting the writer is the same thing as
        projecting the sub-branch's output, exactly, with no second fit.  Every
        other emitted tensor must already match the child parameter it names,
        and anything that matches neither shape fails closed rather than being
        reshaped into place.
        """

        transported: dict[str, Any] = {}
        folded: list[str] = []
        for name, value in child_state.items():
            path = f"{mixer_module}.{name}"
            try:
                live = context.child_model.get_parameter(path)
            except AttributeError as exc:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} kernel emitted {name!r}, which the child module "
                    "does not own."
                ) from exc
            shape = tuple(int(size) for size in value.shape)
            wanted = tuple(int(size) for size in live.shape)
            if shape == wanted:
                transported[name] = value
                continue
            if (
                len(shape) == 2
                and shape[0] == self.chart.source_width
                and wanted == (self.chart.target_width, shape[1])
            ):
                transported[name] = self.chart.fold_producer_weight(value)
                folded.append(name)
                continue
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot place the compiled {name!r} of shape {shape} "
                f"into the child parameter of shape {wanted}."
            )
        return transported, tuple(sorted(folded))

    def _geometry(
        self,
        model: Any,
        mixer_module: str,
        attributes: Mapping[str, Any],
        hidden_size: int,
        role: str,
        side: str,
    ) -> tuple[Any, ChimeraSparseLatentGeometry]:
        module = _submodule(model, mixer_module, role)
        family = attributes.get("mixer_family")
        if family not in {"csa", "hca"}:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses the {side} operator: it declares "
                f"mixer_family={family!r}, which is neither csa nor hca."
            )
        try:
            geometry = chimera_sparse_latent_geometry(module, family=str(family))
        except HFCausalSparseLatentFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot read the {side} compressed-summary geometry: "
                f"{exc}"
            ) from exc
        if geometry.hidden_size != hidden_size:
            raise DynamicCausalOperatorPromotionError(
                f"{role} {side} mixer reads a residual width of "
                f"{geometry.hidden_size}, not the chart's {hidden_size}."
            )
        _declared_matches_module(
            attributes,
            {
                "query_heads": geometry.query_heads,
                "key_value_heads": geometry.key_value_heads,
                "head_dim": geometry.head_dim,
                "local_window": geometry.local_window,
                "summary_stride": geometry.summary_stride,
                "mixer_family": geometry.family,
            },
            role,
            side,
        )
        return module, geometry

    def _layer_evidence(
        self,
        *,
        node: Any,
        geometry: ChimeraSparseLatentGeometry,
        sealed: NativeCircuitEvidence,
        norm_module: str,
        mixer_module: str,
        layer_index: int,
        valid: Any,
        role: str,
    ) -> ChimeraSummaryAttentionLayerEvidence:
        torch = _torch()

        def rows(path: str) -> Any:
            if not sealed.has(path):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} needs the sealed module boundary {path!r}; the "
                    "parent bundle does not carry it."
                )
            capture = sealed.capture(path)
            if capture.call_count != 1:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} refuses {path!r}: it ran {capture.call_count} times "
                    "in the sealed forward, so 'the' cached rows are ambiguous."
                )
            return sealed.output(path)

        def module_of(wanted: str) -> str:
            return owning_module_path(
                _exact_role_parameter_path(node, wanted, role)
            )

        local_module = owning_module_path(module_of("local_q_proj_weight"))
        normalized = rows(norm_module)
        batch, sequence = int(normalized.shape[0]), int(normalized.shape[1])
        local_delta = rows(local_module)
        summary_delta = rows(module_of("out_weight"))
        attention_delta = rows(mixer_module)
        pooling = summary_pooling_matrix(sequence, geometry.summary_stride)
        summaries = torch.matmul(pooling.reshape(1, *pooling.shape), normalized)
        summary_query = rows(module_of("summary_q_weight"))
        summary_key = rows(module_of("summary_k_weight"))
        summary_value = rows(module_of("summary_v_weight"))
        scores, probabilities = summary_attention_probabilities(
            summary_query,
            summary_key,
            block_mask=summary_block_mask(sequence, geometry.summary_stride),
            scaling=geometry.summary_scaling,
        )
        local_query = rows(module_of("local_q_proj_weight"))
        local_key = rows(module_of("local_k_proj_weight"))
        local_value = rows(module_of("local_v_proj_weight"))
        topology = geometry.local_topology()

        def per_head(flat: Any, count: int) -> Any:
            return (
                flat.reshape(batch, sequence, count, geometry.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        _local_scores, local_probabilities = causal_attention_probabilities(
            per_head(local_query, geometry.query_heads),
            per_head(local_key, geometry.key_value_heads),
            attention_mask=valid,
            scaling=topology.attention_scaling,
            attention_window=topology.attention_window,
        )
        block_input = sealed.input(norm_module)
        try:
            return ChimeraSummaryAttentionLayerEvidence(
                index=int(layer_index),
                block_input=block_input,
                attention_norm_output=normalized,
                local_delta=local_delta,
                summary_delta=summary_delta,
                attention_delta=attention_delta,
                block_output=block_input + attention_delta,
                local_query_rows=local_query,
                local_key_rows=local_key,
                local_value_rows=local_value,
                local_probabilities=local_probabilities,
                raw_summaries=summaries[:, 1:].contiguous(),
                summary_query_rows=summary_query,
                summary_key_rows=summary_key,
                summary_value_rows=summary_value,
                summary_scores=scores,
                summary_probabilities=probabilities,
                block_ids=tuple(
                    position // geometry.summary_stride for position in range(sequence)
                ),
                geometry=geometry,
                metadata={
                    "schema": SCHEMA,
                    "component": (
                        "chimera_sparse_latent_evidence_from_sealed_module_boundaries"
                    ),
                    "parent_forward_calls": 0,
                    "training_steps": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                    "captured_row_source": _SEALED_ROW_SOURCE,
                    "derived_rows": list(_SUMMARY_DERIVED_ROWS),
                },
            )
        except HFCausalSparseLatentFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} could not assemble a compressed-summary evidence "
                f"record from the sealed bundle: {exc}"
            ) from exc

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        role = "Compressed summary attention layer operator"
        sealed, parent_graph, source_nodes, target_nodes = self._bind(
            action, context, evidence, role
        )
        source_operator = _single_node(
            source_nodes, CircuitKind.ATTENTION.value, role, "source attention"
        )
        (
            target_operator,
            _target_norm,
            norm_module,
            norm_path,
            mixer_module,
            selectors,
            paths,
        ) = self._branch_targets(
            target_nodes, action, role, CircuitKind.ATTENTION.value
        )
        source_mixer_module = node_module_path(source_operator)
        if source_mixer_module != mixer_module:
            raise DynamicCausalOperatorPromotionError(
                f"{role} source and target mixer modules differ: "
                f"{source_mixer_module!r} against {mixer_module!r}."
            )
        parent_model = context.parent_models[action.sources[0].parent_name]
        _source_module, source_geometry = self._geometry(
            parent_model,
            mixer_module,
            dict(source_operator.attributes),
            self.chart.source_width,
            role,
            "source",
        )
        _target_module, target_geometry = self._geometry(
            context.child_model,
            mixer_module,
            dict(target_operator.attributes),
            self.chart.target_width,
            role,
            "target",
        )
        if (
            source_geometry.summary_stride != target_geometry.summary_stride
            or source_geometry.local_window != target_geometry.local_window
            or source_geometry.family != target_geometry.family
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses a stride, window or family change: "
                f"{source_geometry.as_dict()} against "
                f"{target_geometry.as_dict()}. Both are contract constants, not "
                "fold coordinates."
            )
        valid, fit, selection, validation = sealed.token_masks()
        layer_evidence = self._layer_evidence(
            node=source_operator,
            geometry=source_geometry,
            sealed=sealed,
            norm_module=norm_module,
            mixer_module=mixer_module,
            layer_index=int(target_operator.layer_index or 0),
            valid=valid,
            role=role,
        )
        policy = self.fold_policy
        try:
            # Ridge-independent, so the sub-branch split gate runs once and a
            # breach fails closed with the kernel's own measurement.
            split = split_summary_branch_delta(
                layer_evidence,
                row_mask=valid,
                tolerance=policy.split_reconstruction_tolerance,
            )
        except HFCausalSparseLatentFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses this branch: the captured local and summary "
                f"sub-branches do not reconstruct the sealed write -- {exc}"
            ) from exc
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            selector_report: dict[str, Any] = {"depth_selector_present": bool(selectors)}
            if selectors:
                live = self._recapture_child(context, role)
                installs, selector_report = self._depth_selector_installs(
                    selector=selectors[0],
                    parent_model=parent_model,
                    child_model=context.child_model,
                    sealed=sealed,
                    live=live,
                    role=role,
                )
                selector_report["depth_selector_present"] = True
                _install(context, installs)
            live = self._recapture_child(context, role)
            epsilon = _norm_epsilon(context.child_model, norm_module, role)
            branch_input = live.input(norm_module)
            branch_normalized_target = self.chart.encode(sealed.output(norm_module))
            norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(branch_input, fit),
                _masked_rows(branch_normalized_target, fit),
                epsilon=epsilon,
                ridge=self.ridge,
            )
            normalized = _rmsnorm(branch_input, norm_fold.weight, epsilon)
            branch_delta_target = self.chart.encode(layer_evidence.attention_delta)
            score_chart = (
                None
                if self.chart.source_width == self.chart.target_width
                else self.chart.encoder
            )

            def compile_rung(candidate: float) -> Any:
                # No ``endpoint_target``: the kernel's cross-width lane fits
                # both sub-branches against the *parent's own* sub-branch
                # deltas, which is where its split gate captured them.  The two
                # sub-branch writers are the last linear map in their
                # sub-branch, so transporting them through the chart afterwards
                # is exact rather than a second fit -- see
                # ``_transported_child_state``.
                return compile_hf_causal_sparse_latent_fold(
                    layer_evidence,
                    target_geometry,
                    attention_mask=valid,
                    fit_token_mask=fit,
                    validation_token_mask=validation,
                    valid_token_mask=valid,
                    child_normalized_rows=normalized,
                    score_chart=score_chart,
                    policy=replace(policy, ridge=candidate),
                )

            positions = _row_positions(sealed.batch_size, sealed.sequence_length)

            def replay(fold: Any) -> tuple[Any, Any, Any]:
                total, local, summary = fold.replay(
                    normalized, attention_mask=valid, position_ids=positions
                )
                return (
                    self.chart.encode(total),
                    self.chart.encode(local),
                    self.chart.encode(summary),
                )

            def score_rung(fold: Any) -> float:
                replayed, _local, _summary = replay(fold)
                return _relative_rmse(
                    _masked_rows(replayed, selection),
                    _masked_rows(branch_delta_target, selection),
                )

            (
                summary_fold,
                selected_ridge,
                selection_endpoint,
                ladder,
            ) = self._select_over_ridge_ladder(
                compile_rung=compile_rung,
                score_rung=score_rung,
                refusals=(HFCausalSparseLatentFoldError,),
                role=role,
            )
            delta, local_delta, summary_delta = replay(summary_fold)
            endpoint = self._require_endpoint(
                _relative_rmse(
                    _masked_rows(delta, validation),
                    _masked_rows(branch_delta_target, validation),
                ),
                role,
                "validation composite-delta relative RMSE",
            )
            receipt = dict(summary_fold.report)
            child_state, transported = self._transported_child_state(
                child_state=summary_fold.child_state_dict(),
                mixer_module=mixer_module,
                context=context,
                role=role,
            )
            installs = self._child_installs(
                child_state=child_state,
                mixer_module=mixer_module,
                target_operator=target_operator,
                context=context,
                role=role,
            )
            report = {
                **_fold_only_operation_audit(),
                **selector_report,
                "schema": SCHEMA,
                "kind": "native_compressed_summary_attention_branch",
                "method_id": self.method_id,
                "source_family": parent_graph.family,
                "target_family": context.child_graph.family,
                "source_node_id": source_operator.node_id,
                "target_node_id": target_operator.node_id,
                "cached_parent_evidence_sha256": sealed.cache_key,
                "residual_chart_sha256": self.chart.map_sha256,
                "residual_chart_source_width": self.chart.source_width,
                "residual_chart_target_width": self.chart.target_width,
                "per_boundary_chart_set": False,
                "position_policy": "nope",
                "mixer_family": target_geometry.family,
                "summary_stride": target_geometry.summary_stride,
                "local_window": target_geometry.local_window,
                "one_executable_for_csa_and_hca": True,
                "source_geometry": source_geometry.as_dict(),
                "target_geometry": target_geometry.as_dict(),
                "captured_row_source": _SEALED_ROW_SOURCE,
                "derived_rows": list(_SUMMARY_DERIVED_ROWS),
                "split_report_sha256": split.diagnostics["report_sha256"],
                "split_reconstruction_relative_rmse": split.diagnostics[
                    "split_reconstruction_relative_rmse"
                ],
                "local_energy_fraction": split.diagnostics["local_energy_fraction"],
                "summary_energy_fraction": split.diagnostics["summary_energy_fraction"],
                "score_chart_used": bool(score_chart is not None),
                "kernel_fold_policy": policy.as_dict(),
                "summary_probability_overlap": receipt["summary_probability_overlap"],
                "sub_branch_writers_transported_through_chart": list(transported),
                "sub_branch_writer_transport": (
                    "exact_producer_fold_of_the_last_linear_map_in_each_"
                    "sub_branch"
                ),
                "sub_branch_fit_coordinate_system": "parent_residual_width",
                "endpoint_gate_coordinate_system": "child_residual_width",
                "local_sub_branch_validation_relative_rmse": _relative_rmse(
                    _masked_rows(local_delta, validation),
                    _masked_rows(self.chart.encode(layer_evidence.local_delta), validation),
                ),
                "summary_sub_branch_validation_relative_rmse": _relative_rmse(
                    _masked_rows(summary_delta, validation),
                    _masked_rows(
                        self.chart.encode(layer_evidence.summary_delta), validation
                    ),
                ),
                "input_norm_report_sha256": norm_fold.report["report_sha256"],
                "summary_attention_report_sha256": receipt["report_sha256"],
                "local_fold_report_sha256": receipt["local_fold_report_sha256"],
                "installed_parameter_paths": list(paths),
                "child_recaptures": int(getattr(self.recapture, "calls", 0)),
                "selected_ridge": selected_ridge,
                "ridge_ladder": ladder,
                "kernel_refused_ridge_rungs": sum(
                    1 for entry in ladder if "kernel_refusal" in entry
                ),
                "selection_endpoint_relative_rmse": selection_endpoint,
                "no_branch_endpoint_relative_rmse_baseline": (
                    self.no_branch_endpoint_relative_rmse_baseline
                ),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "validation_composite_delta_relative_rmse": endpoint,
                "validation_residual_relative_rmse_bf16": receipt[
                    "validation_residual_relative_rmse_bf16"
                ],
                "parent_residual_coordinate_endpoints": False,
                "sequential_child_replay": True,
                "numeric_weight_matching_objective": False,
                "zero_gradient": True,
            }
            _install(context, (*installs, (norm_path, norm_fold.weight)))
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class CompressedLatentKVAttentionLayerOperator(_ChimeraMixerBranchOperator):
    """Install one gated compressed-latent-KV attention branch (chimera MLA).

    The executable is
    ``hf_causal_latent_kv_attention_fold.compile_hf_causal_latent_kv_attention_fold``:
    the readout identity and the branch reconstruction bind the sealed
    evidence, the latent bases are POD on observed rows, every consumer is a
    ridge with the child's own normalizations replayed, the probabilities are
    replayed before the writer is fitted, and the writer is endpoint-coupled
    (design section 1.3).

    One executable covers both query variants; the query latent size is a
    contract shape of the crossing, checked by the planner and again here.

    Where the parent rows come from, stated exactly -- every one of them is a
    sealed module boundary of the one forward, and none of them is a parent
    call:

    * ``compressed_kv_rows`` / ``normalized_latent_rows`` / ``kv_rows`` are the
      captured outputs of ``kv_down`` / ``kv_norm`` / ``kv_up``, so the kernel's
      readout identity compares two independently captured values through the
      carried weight rather than a replay against itself;
    * ``query_rows`` is the captured output of ``q_up`` (or ``q_proj``), with
      ``q_down`` and ``q_norm`` captured alongside it in the compressed variant;
    * ``gate_rows`` is the captured **pre-sigmoid** output of ``gate``, which is
      why this family needs no inverse-sigmoid clamp policy at all;
    * ``attention_delta`` is the captured mixer output, and the kernel's
      reconstruction gate rebuilds exactly that from the derived rows.

    Unlike the compressed-summary operator this one passes the endpoint target
    already chart-projected: an MLA score is an inner product in *head*
    coordinates, not in the residual width, so no score chart exists to take and
    the single writer is fitted straight into the child's coordinates.
    """

    method_id = CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD
    branch_id = "compressed_latent_attention"

    def __init__(
        self,
        *,
        fold_policy: ChimeraLatentKVFoldPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # The kernel's own sealed thresholds: the two evidence tolerances, the
        # probability-overlap floor and the gate-saturation cap.
        self.fold_policy = fold_policy or ChimeraLatentKVFoldPolicy()
        if not isinstance(self.fold_policy, ChimeraLatentKVFoldPolicy):
            raise DynamicCausalOperatorPromotionError(
                "fold_policy must be a ChimeraLatentKVFoldPolicy."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "base": super().resource_fingerprint,
                "fold_policy": self.fold_policy.as_dict(),
            }
        )

    def _geometry(
        self,
        model: Any,
        mixer_module: str,
        attributes: Mapping[str, Any],
        hidden_size: int,
        role: str,
        side: str,
    ) -> tuple[Any, ChimeraLatentKVGeometry]:
        module = _submodule(model, mixer_module, role)
        try:
            geometry = chimera_latent_kv_geometry(module)
        except HFCausalLatentKVAttentionFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot read the {side} compressed-latent-KV geometry: "
                f"{exc}"
            ) from exc
        if geometry.hidden_size != hidden_size:
            raise DynamicCausalOperatorPromotionError(
                f"{role} {side} mixer reads a residual width of "
                f"{geometry.hidden_size}, not the chart's {hidden_size}."
            )
        _declared_matches_module(
            attributes,
            {
                "query_heads": geometry.heads,
                "key_value_heads": geometry.heads,
                "head_dim": geometry.query_head_dim,
                "key_head_dim": geometry.key_head_dim,
                "value_head_dim": geometry.value_head_dim,
                "extra_head_dim": geometry.extra_head_dim,
                "kv_latent_size": geometry.kv_latent_size,
                "query_latent_size": geometry.query_latent_size,
            },
            role,
            side,
        )
        return module, geometry

    def _layer_evidence(
        self,
        *,
        node: Any,
        geometry: ChimeraLatentKVGeometry,
        sealed: NativeCircuitEvidence,
        norm_module: str,
        mixer_module: str,
        layer_index: int,
        role: str,
    ) -> ChimeraLatentKVLayerEvidence:
        def rows(path: str) -> Any:
            if not sealed.has(path):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} needs the sealed module boundary {path!r}; the "
                    "parent bundle does not carry it."
                )
            capture = sealed.capture(path)
            if capture.call_count != 1:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} refuses {path!r}: it ran {capture.call_count} times "
                    "in the sealed forward, so 'the' cached rows are ambiguous."
                )
            return sealed.output(path)

        def module_of(wanted: str) -> str:
            return owning_module_path(
                _exact_role_parameter_path(node, wanted, role)
            )

        block_input = sealed.input(norm_module)
        if geometry.compressed_query:
            query_latent = rows(module_of("q_down_weight"))
            normalized_query_latent = rows(module_of("q_norm_weight"))
            query_rows = rows(module_of("q_up_weight"))
        else:
            query_latent = None
            normalized_query_latent = None
            query_rows = rows(module_of("q_proj_weight"))
        try:
            return ChimeraLatentKVLayerEvidence(
                index=int(layer_index),
                block_input=block_input,
                attention_norm_output=rows(norm_module),
                query_rows=query_rows,
                compressed_kv_rows=rows(module_of("kv_down_weight")),
                normalized_latent_rows=rows(module_of("kv_norm_weight")),
                kv_rows=rows(module_of("kv_up_weight")),
                gate_rows=rows(module_of("gate_weight")),
                attention_delta=rows(mixer_module),
                block_output=block_input + rows(mixer_module),
                geometry=geometry,
                metadata={
                    "schema": SCHEMA,
                    "component": (
                        "chimera_latent_kv_evidence_from_sealed_module_boundaries"
                    ),
                    "parent_forward_calls": 0,
                    "training_steps": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                    "captured_row_source": _SEALED_ROW_SOURCE,
                    "derived_rows": list(_LATENT_KV_DERIVED_ROWS),
                },
                query_latent_rows=query_latent,
                normalized_query_latent_rows=normalized_query_latent,
            )
        except HFCausalLatentKVAttentionFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} could not assemble a compressed-latent-KV evidence "
                f"record from the sealed bundle: {exc}"
            ) from exc

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        role = "Compressed latent KV attention layer operator"
        sealed, parent_graph, source_nodes, target_nodes = self._bind(
            action, context, evidence, role
        )
        source_operator = _single_node(
            source_nodes, CircuitKind.ATTENTION.value, role, "source attention"
        )
        (
            target_operator,
            _target_norm,
            norm_module,
            norm_path,
            mixer_module,
            selectors,
            paths,
        ) = self._branch_targets(
            target_nodes, action, role, CircuitKind.ATTENTION.value
        )
        source_mixer_module = node_module_path(source_operator)
        if source_mixer_module != mixer_module:
            raise DynamicCausalOperatorPromotionError(
                f"{role} source and target mixer modules differ: "
                f"{source_mixer_module!r} against {mixer_module!r}."
            )
        parent_model = context.parent_models[action.sources[0].parent_name]
        source_module, source_geometry = self._geometry(
            parent_model,
            mixer_module,
            dict(source_operator.attributes),
            self.chart.source_width,
            role,
            "source",
        )
        _target_module, target_geometry = self._geometry(
            context.child_model,
            mixer_module,
            dict(target_operator.attributes),
            self.chart.target_width,
            role,
            "target",
        )
        if (
            source_geometry.extra_head_dim != target_geometry.extra_head_dim
            or source_geometry.compressed_query
            != target_geometry.compressed_query
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses a shared-key-channel or query-variant change: "
                f"{source_geometry.as_dict()} against "
                f"{target_geometry.as_dict()}. Both are contract shapes of the "
                "crossing, not fold coordinates."
            )
        try:
            parameters = chimera_latent_kv_parameters(
                source_module, geometry=source_geometry
            )
        except HFCausalLatentKVAttentionFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot carry the parent readout and writer: {exc}"
            ) from exc
        valid, fit, selection, validation = sealed.token_masks()
        layer_evidence = self._layer_evidence(
            node=source_operator,
            geometry=source_geometry,
            sealed=sealed,
            norm_module=norm_module,
            mixer_module=mixer_module,
            layer_index=int(target_operator.layer_index or 0),
            role=role,
        )
        policy = self.fold_policy
        try:
            # Ridge-independent, so both evidence gates run once and a breach
            # fails closed with the kernel's own measurement.
            gates = latent_kv_consistency(
                layer_evidence,
                parameters,
                attention_mask=valid,
                row_mask=valid,
                readout_tolerance=policy.readout_identity_tolerance,
                delta_tolerance=policy.delta_reconstruction_tolerance,
            )
        except HFCausalLatentKVAttentionFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses this branch: the sealed evidence is not "
                f"consistent with a gated compressed-latent-KV attention -- "
                f"{exc}"
            ) from exc
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            selector_report: dict[str, Any] = {"depth_selector_present": bool(selectors)}
            if selectors:
                live = self._recapture_child(context, role)
                installs, selector_report = self._depth_selector_installs(
                    selector=selectors[0],
                    parent_model=parent_model,
                    child_model=context.child_model,
                    sealed=sealed,
                    live=live,
                    role=role,
                )
                selector_report["depth_selector_present"] = True
                _install(context, installs)
            live = self._recapture_child(context, role)
            epsilon = _norm_epsilon(context.child_model, norm_module, role)
            branch_input = live.input(norm_module)
            branch_normalized_target = self.chart.encode(sealed.output(norm_module))
            norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(branch_input, fit),
                _masked_rows(branch_normalized_target, fit),
                epsilon=epsilon,
                ridge=self.ridge,
            )
            normalized = _rmsnorm(branch_input, norm_fold.weight, epsilon)
            branch_delta_target = self.chart.encode(layer_evidence.attention_delta)

            def compile_rung(candidate: float) -> Any:
                return compile_hf_causal_latent_kv_attention_fold(
                    layer_evidence,
                    parameters,
                    target_geometry,
                    attention_mask=valid,
                    fit_token_mask=fit,
                    validation_token_mask=validation,
                    valid_token_mask=valid,
                    child_normalized_rows=normalized,
                    endpoint_target=branch_delta_target,
                    policy=replace(policy, ridge=candidate),
                    consistency=gates,
                )

            def score_rung(fold: Any) -> float:
                replayed, _probabilities = fold.replay(
                    normalized, attention_mask=valid
                )
                return _relative_rmse(
                    _masked_rows(replayed, selection),
                    _masked_rows(branch_delta_target, selection),
                )

            (
                latent_fold,
                selected_ridge,
                selection_endpoint,
                ladder,
            ) = self._select_over_ridge_ladder(
                compile_rung=compile_rung,
                score_rung=score_rung,
                refusals=(HFCausalLatentKVAttentionFoldError,),
                role=role,
            )
            delta, _probabilities = latent_fold.replay(
                normalized, attention_mask=valid
            )
            endpoint = self._require_endpoint(
                _relative_rmse(
                    _masked_rows(delta, validation),
                    _masked_rows(branch_delta_target, validation),
                ),
                role,
                "validation latent-attention-delta relative RMSE",
            )
            receipt = dict(latent_fold.report)
            installs = self._child_installs(
                child_state=latent_fold.child_state_dict(),
                mixer_module=mixer_module,
                target_operator=target_operator,
                context=context,
                role=role,
            )
            report = {
                **_fold_only_operation_audit(),
                **selector_report,
                "schema": SCHEMA,
                "kind": "native_compressed_latent_kv_attention_branch",
                "method_id": self.method_id,
                "source_family": parent_graph.family,
                "target_family": context.child_graph.family,
                "source_node_id": source_operator.node_id,
                "target_node_id": target_operator.node_id,
                "cached_parent_evidence_sha256": sealed.cache_key,
                "residual_chart_sha256": self.chart.map_sha256,
                "residual_chart_source_width": self.chart.source_width,
                "residual_chart_target_width": self.chart.target_width,
                "per_boundary_chart_set": False,
                "position_policy": "nope",
                "one_executable_for_both_query_variants": True,
                "compressed_query_path": target_geometry.compressed_query,
                "source_geometry": source_geometry.as_dict(),
                "target_geometry": target_geometry.as_dict(),
                "captured_row_source": _SEALED_ROW_SOURCE,
                "derived_rows": list(_LATENT_KV_DERIVED_ROWS),
                "derived_row_consistency_gate": (
                    "latent_kv_attention_replay_against_the_captured_branch_delta"
                ),
                "consistency_report_sha256": gates.diagnostics["report_sha256"],
                "readout_identity_relative_rmse": gates.readout_relative_rmse,
                "delta_reconstruction_relative_rmse": gates.delta_relative_rmse,
                "shared_key_channel_score_energy_fraction": gates.diagnostics[
                    "shared_key_channel_score_energy_fraction"
                ],
                "kernel_fold_policy": policy.as_dict(),
                "attention_probability_overlap": receipt[
                    "attention_probability_overlap"
                ],
                "gate_saturated_row_fraction": receipt["gate_saturated_row_fraction"],
                "kv_latent_basis": receipt["kv_latent_basis"],
                "kv_latent_retained_energy_fraction": receipt[
                    "kv_latent_retained_energy_fraction"
                ],
                "score_chart_used": False,
                "score_chart_not_required": receipt["score_chart_not_required"],
                "input_norm_report_sha256": norm_fold.report["report_sha256"],
                "latent_kv_report_sha256": receipt["report_sha256"],
                "installed_parameter_paths": list(paths),
                "child_recaptures": int(getattr(self.recapture, "calls", 0)),
                "selected_ridge": selected_ridge,
                "ridge_ladder": ladder,
                "kernel_refused_ridge_rungs": sum(
                    1 for entry in ladder if "kernel_refusal" in entry
                ),
                "selection_endpoint_relative_rmse": selection_endpoint,
                "no_branch_endpoint_relative_rmse_baseline": (
                    self.no_branch_endpoint_relative_rmse_baseline
                ),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "validation_latent_attention_delta_relative_rmse": endpoint,
                "validation_residual_relative_rmse_bf16": receipt[
                    "validation_residual_relative_rmse_bf16"
                ],
                "parent_residual_coordinate_endpoints": False,
                "sequential_child_replay": True,
                "numeric_weight_matching_objective": False,
                "zero_gradient": True,
            }
            _install(context, (*installs, (norm_path, norm_fold.weight)))
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class StableLatentMoEChannelMixerLayerOperator(_ChimeraMixerBranchOperator):
    """Install one composite Stable LatentMoE channel-mixer branch.

    The executable is
    ``hf_causal_stable_latent_moe_fold.compile_hf_causal_stable_latent_moe_fold``:
    the routing is recovered from the captured router boundary and checked
    against the captured per-expert row counts, the routed and shared
    sub-branch deltas must reconstruct the sealed write, each routed expert is
    fitted on the rows the parent actually sent it, and both writers are
    endpoint-coupled under bounded alternation (design section 2.4).

    Two things are specific to this family.

    * **The per-expert boundaries are ragged.**  ``routed_experts.j`` is invoked
      once per forward with only the rows the router selected for it
      (``layers.py:436-441``), so its captured input and output are that row
      block and nothing else.  An expert the parent never selected produces no
      boundary at all, and this operator says so by name rather than folding an
      expert with no evidence.
    * **``router_bias`` is a buffer, not a parameter.**  It is not in the IR, so
      no transaction owns it and the snapshot/rollback envelope does not cover
      it.  It enters *selection* only, so the parent's and the child's must be
      equal or the child's selection boundary is somewhere else; that is checked
      here, before any mutation, and recorded in the receipt.
    """

    method_id = CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD
    branch_id = "stable_latent_moe_channel_mixer"

    def __init__(
        self,
        *,
        fold_policy: ChimeraStableLatentMoEFoldPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # The kernel's own sealed thresholds: the routing and split tolerances,
        # the router-agreement floor and the sparse expert row budget.
        self.fold_policy = fold_policy or ChimeraStableLatentMoEFoldPolicy()
        if not isinstance(self.fold_policy, ChimeraStableLatentMoEFoldPolicy):
            raise DynamicCausalOperatorPromotionError(
                "fold_policy must be a ChimeraStableLatentMoEFoldPolicy."
            )

    @property
    def resource_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "base": super().resource_fingerprint,
                "fold_policy": self.fold_policy.as_dict(),
            }
        )

    def _geometry(
        self,
        model: Any,
        mixer_module: str,
        attributes: Mapping[str, Any],
        hidden_size: int,
        role: str,
        side: str,
    ) -> tuple[Any, ChimeraStableLatentMoEGeometry]:
        module = _submodule(model, mixer_module, role)
        try:
            geometry = chimera_stable_latent_moe_geometry(module)
        except HFCausalStableLatentMoEFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot read the {side} Stable LatentMoE geometry: {exc}"
            ) from exc
        if geometry.hidden_size != hidden_size:
            raise DynamicCausalOperatorPromotionError(
                f"{role} {side} channel mixer reads a residual width of "
                f"{geometry.hidden_size}, not the chart's {hidden_size}."
            )
        _declared_matches_module(
            attributes,
            {
                "expert_count": geometry.expert_count,
                "experts_per_token": geometry.experts_per_token,
                "shared_expert_count": geometry.shared_expert_count,
                "expert_latent_size": geometry.latent_size,
                "routed_intermediate_size": geometry.routed_intermediate_size,
                "intermediate_size": geometry.intermediate_size,
                "activation": geometry.activation,
                "situ_gate_beta": geometry.situ_gate_beta,
                "situ_up_beta": geometry.situ_up_beta,
            },
            role,
            side,
        )
        return module, geometry

    def _layer_evidence(
        self,
        *,
        geometry: ChimeraStableLatentMoEGeometry,
        sealed: NativeCircuitEvidence,
        norm_module: str,
        mixer_module: str,
        layer_index: int,
        role: str,
    ) -> ChimeraStableLatentMoELayerEvidence:
        def capture(path: str, what: str) -> Any:
            if not sealed.has(path):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} needs the sealed module boundary {path!r} for the "
                    f"{what}; the parent bundle does not carry it. A routed "
                    "expert the parent never selected leaves no boundary at "
                    "all, and there is no evidence to fold it from."
                )
            record = sealed.capture(path)
            if record.call_count != 1:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} refuses {path!r}: it ran {record.call_count} times "
                    "in the sealed forward, so 'the' cached rows are ambiguous."
                )
            return record

        def rows(path: str, what: str) -> Any:
            return sealed.output(capture(path, what).module_path)

        def halves(path: str, what: str) -> tuple[Any, Any]:
            fused = rows(path, what)
            gate, up = fused.chunk(2, dim=-1)
            return gate.contiguous(), up.contiguous()

        expert_inputs = []
        expert_gates = []
        expert_ups = []
        expert_outputs = []
        for index in range(geometry.expert_count):
            expert_path = f"{mixer_module}.routed_experts.{index}"
            record = capture(expert_path, f"routed expert {index}")
            expert_inputs.append(sealed.input(record.module_path))
            expert_outputs.append(sealed.output(record.module_path))
            gate, up = halves(f"{expert_path}.up", f"routed expert {index} gate/up")
            expert_gates.append(gate)
            expert_ups.append(up)
        shared_gates = []
        shared_ups = []
        shared_deltas = []
        for index in range(geometry.shared_expert_count):
            shared_path = f"{mixer_module}.shared_experts.{index}"
            capture(shared_path, f"shared expert {index}")
            shared_deltas.append(rows(shared_path, f"shared expert {index}"))
            gate, up = halves(f"{shared_path}.up", f"shared expert {index} gate/up")
            shared_gates.append(gate)
            shared_ups.append(up)
        routed_norm_module = f"{mixer_module}.routed_norm"
        capture(routed_norm_module, "routed accumulator norm")
        try:
            return ChimeraStableLatentMoELayerEvidence(
                index=int(layer_index),
                block_input=sealed.input(norm_module),
                feed_forward_norm_output=rows(norm_module, "branch norm"),
                router_logit_rows=rows(f"{mixer_module}.router", "router"),
                latent_rows=rows(f"{mixer_module}.latent_down", "latent projection"),
                expert_input_rows=tuple(expert_inputs),
                expert_gate_rows=tuple(expert_gates),
                expert_up_rows=tuple(expert_ups),
                expert_output_rows=tuple(expert_outputs),
                routed_rows=sealed.input(routed_norm_module),
                normalized_routed_rows=sealed.output(routed_norm_module),
                routed_delta=rows(f"{mixer_module}.latent_up", "routed writer"),
                shared_gate_rows=tuple(shared_gates),
                shared_up_rows=tuple(shared_ups),
                shared_delta_rows=tuple(shared_deltas),
                feed_forward_delta=rows(mixer_module, "channel mixer"),
                geometry=geometry,
                metadata={
                    "schema": SCHEMA,
                    "component": (
                        "chimera_stable_latent_moe_evidence_from_sealed_module_"
                        "boundaries"
                    ),
                    "parent_forward_calls": 0,
                    "training_steps": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                    "captured_row_source": _SEALED_ROW_SOURCE,
                    "derived_rows": list(_STABLE_LATENT_MOE_DERIVED_ROWS),
                },
            )
        except HFCausalStableLatentMoEFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} could not assemble a Stable LatentMoE evidence record "
                f"from the sealed bundle: {exc}"
            ) from exc

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        torch = _torch()
        role = "Stable LatentMoE channel mixer layer operator"
        sealed, parent_graph, source_nodes, target_nodes = self._bind(
            action, context, evidence, role
        )
        source_operator = _single_node(
            source_nodes, CircuitKind.GATED_MLP.value, role, "source channel mixer"
        )
        (
            target_operator,
            _target_norm,
            norm_module,
            norm_path,
            mixer_module,
            selectors,
            paths,
        ) = self._branch_targets(
            target_nodes, action, role, CircuitKind.GATED_MLP.value
        )
        source_mixer_module = node_module_path(source_operator)
        if source_mixer_module != mixer_module:
            raise DynamicCausalOperatorPromotionError(
                f"{role} source and target channel-mixer modules differ: "
                f"{source_mixer_module!r} against {mixer_module!r}."
            )
        parent_model = context.parent_models[action.sources[0].parent_name]
        source_module, source_geometry = self._geometry(
            parent_model,
            mixer_module,
            dict(source_operator.attributes),
            self.chart.source_width,
            role,
            "source",
        )
        target_module, target_geometry = self._geometry(
            context.child_model,
            mixer_module,
            dict(target_operator.attributes),
            self.chart.target_width,
            role,
            "target",
        )
        constants = (
            "expert_count",
            "experts_per_token",
            "shared_expert_count",
            "activation",
            "situ_gate_beta",
            "situ_up_beta",
        )
        differing = [
            name
            for name in constants
            if getattr(source_geometry, name) != getattr(target_geometry, name)
        ]
        if differing:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses a change to {', '.join(sorted(differing))}: "
                f"{source_geometry.as_dict()} against "
                f"{target_geometry.as_dict()}. Every one of them is a contract "
                "constant, not a fold coordinate."
            )
        try:
            parameters = chimera_stable_latent_moe_parameters(
                source_module, geometry=source_geometry
            )
        except HFCausalStableLatentMoEFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot carry the parent router balancing buffer: {exc}"
            ) from exc
        child_bias = getattr(target_module, "router_bias", None)
        if child_bias is None or not bool(
            torch.equal(
                _f64(child_bias),
                parameters.router_bias,
            )
        ):
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses this branch: the parent and the child declare "
                f"different {mixer_module}.router_bias values. That buffer is "
                "not a learned tensor, so no fold transaction owns it, and it "
                "moves the child's expert-selection boundary; transporting it "
                "is the caller's responsibility and is not a fold."
            )
        valid, fit, selection, validation = sealed.token_masks()
        layer_evidence = self._layer_evidence(
            geometry=source_geometry,
            sealed=sealed,
            norm_module=norm_module,
            mixer_module=mixer_module,
            layer_index=int(target_operator.layer_index or 0),
            role=role,
        )
        policy = self.fold_policy
        try:
            # Ridge-independent, so the routing identity and the channel split
            # run once and a breach fails closed with the kernel's measurement.
            gates = stable_latent_moe_consistency(
                layer_evidence,
                parameters,
                row_mask=valid,
                routing_tolerance=policy.routing_identity_tolerance,
                split_tolerance=policy.split_reconstruction_tolerance,
            )
        except HFCausalStableLatentMoEFoldError as exc:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses this branch: the sealed evidence is not "
                f"consistent with a top-k routed Stable LatentMoE -- {exc}"
            ) from exc
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            selector_report: dict[str, Any] = {"depth_selector_present": bool(selectors)}
            if selectors:
                live = self._recapture_child(context, role)
                installs, selector_report = self._depth_selector_installs(
                    selector=selectors[0],
                    parent_model=parent_model,
                    child_model=context.child_model,
                    sealed=sealed,
                    live=live,
                    role=role,
                )
                selector_report["depth_selector_present"] = True
                _install(context, installs)
            live = self._recapture_child(context, role)
            epsilon = _norm_epsilon(context.child_model, norm_module, role)
            branch_input = live.input(norm_module)
            branch_normalized_target = self.chart.encode(sealed.output(norm_module))
            norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(branch_input, fit),
                _masked_rows(branch_normalized_target, fit),
                epsilon=epsilon,
                ridge=self.ridge,
            )
            normalized = _rmsnorm(branch_input, norm_fold.weight, epsilon)
            rows = int(sealed.batch_size * sealed.sequence_length)

            def project(value: Any) -> Any:
                return self.chart.encode(value).reshape(
                    rows, self.chart.target_width
                )

            branch_delta_target = project(layer_evidence.feed_forward_delta)

            def compile_rung(candidate: float) -> Any:
                return compile_hf_causal_stable_latent_moe_fold(
                    layer_evidence,
                    parameters,
                    target_geometry,
                    fit_token_mask=fit,
                    validation_token_mask=validation,
                    valid_token_mask=valid,
                    child_normalized_rows=normalized,
                    endpoint_target=branch_delta_target,
                    routed_endpoint_target=project(layer_evidence.routed_delta),
                    shared_endpoint_targets=tuple(
                        project(value)
                        for value in layer_evidence.shared_delta_rows
                    ),
                    policy=replace(policy, ridge=candidate),
                    consistency=gates,
                )

            def score_rung(fold: Any) -> float:
                replayed, _routed, _shared = fold.replay(normalized)
                return _relative_rmse(
                    replayed[selection.reshape(-1)],
                    branch_delta_target[selection.reshape(-1)],
                )

            (
                moe_fold,
                selected_ridge,
                selection_endpoint,
                ladder,
            ) = self._select_over_ridge_ladder(
                compile_rung=compile_rung,
                score_rung=score_rung,
                refusals=(HFCausalStableLatentMoEFoldError,),
                role=role,
            )
            delta, _routed, _shared = moe_fold.replay(normalized)
            endpoint = self._require_endpoint(
                _relative_rmse(
                    delta[validation.reshape(-1)],
                    branch_delta_target[validation.reshape(-1)],
                ),
                role,
                "validation channel-mixer-delta relative RMSE",
            )
            receipt = dict(moe_fold.report)
            installs = self._child_installs(
                child_state=moe_fold.child_state_dict(),
                mixer_module=mixer_module,
                target_operator=target_operator,
                context=context,
                role=role,
            )
            report = {
                **_fold_only_operation_audit(),
                **selector_report,
                "schema": SCHEMA,
                "kind": "native_stable_latent_moe_channel_mixer_branch",
                "method_id": self.method_id,
                "source_family": parent_graph.family,
                "target_family": context.child_graph.family,
                "source_node_id": source_operator.node_id,
                "target_node_id": target_operator.node_id,
                "cached_parent_evidence_sha256": sealed.cache_key,
                "residual_chart_sha256": self.chart.map_sha256,
                "residual_chart_source_width": self.chart.source_width,
                "residual_chart_target_width": self.chart.target_width,
                "per_boundary_chart_set": False,
                "source_geometry": source_geometry.as_dict(),
                "target_geometry": target_geometry.as_dict(),
                "captured_row_source": _SEALED_ROW_SOURCE,
                "derived_rows": list(_STABLE_LATENT_MOE_DERIVED_ROWS),
                "derived_row_consistency_gate": (
                    "recovered_routing_against_the_captured_expert_row_counts_"
                    "and_the_captured_routed_accumulator"
                ),
                "consistency_report_sha256": gates.diagnostics["report_sha256"],
                "routing_identity_relative_rmse": gates.routing_relative_rmse,
                "expert_input_relative_rmse": gates.expert_input_relative_rmse,
                "split_reconstruction_relative_rmse": gates.split_relative_rmse,
                "routed_energy_fraction": gates.diagnostics["routed_energy_fraction"],
                "shared_energy_fraction": gates.diagnostics["shared_energy_fraction"],
                "expert_row_counts": receipt["expert_row_counts"],
                "expert_fit_row_counts": receipt["expert_fit_row_counts"],
                "expected_expert_rows": receipt["expected_expert_rows"],
                "expert_rows_are_the_parents_own_assignment": receipt[
                    "expert_rows_are_the_parents_own_assignment"
                ],
                "router_top_k_agreement": receipt["router_top_k_agreement"],
                "router_bias_equality_required": True,
                "router_bias_is_a_non_gradient_buffer": True,
                "router_bias_transported_by_this_transaction": False,
                "latent_basis": receipt["latent_basis"],
                "latent_retained_energy_fraction": receipt[
                    "latent_retained_energy_fraction"
                ],
                "routed_sub_branch_validation_relative_rmse": receipt[
                    "routed_sub_branch_validation_relative_rmse"
                ],
                "shared_sub_branch_validation_relative_rmse": receipt[
                    "shared_sub_branch_validation_relative_rmse"
                ],
                "sub_branch_fit_coordinate_system": "child_residual_width",
                "endpoint_gate_coordinate_system": "child_residual_width",
                "alternation_objective_monotone": receipt[
                    "alternation_objective_monotone"
                ],
                "kernel_fold_policy": policy.as_dict(),
                "input_norm_report_sha256": norm_fold.report["report_sha256"],
                "stable_latent_moe_report_sha256": receipt["report_sha256"],
                "installed_parameter_paths": list(paths),
                "child_recaptures": int(getattr(self.recapture, "calls", 0)),
                "selected_ridge": selected_ridge,
                "ridge_ladder": ladder,
                "kernel_refused_ridge_rungs": sum(
                    1 for entry in ladder if "kernel_refusal" in entry
                ),
                "selection_endpoint_relative_rmse": selection_endpoint,
                "no_branch_endpoint_relative_rmse_baseline": (
                    self.no_branch_endpoint_relative_rmse_baseline
                ),
                "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
                "validation_channel_mixer_delta_relative_rmse": endpoint,
                "validation_residual_relative_rmse_bf16": receipt[
                    "validation_residual_relative_rmse_bf16"
                ],
                "parent_residual_coordinate_endpoints": False,
                "sequential_child_replay": True,
                "numeric_weight_matching_objective": False,
                "zero_gradient": True,
            }
            _install(context, (*installs, (norm_path, norm_fold.weight)))
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )


class NativeResidualChartInterfaceOperator(_NativeChartOperator):
    """Carry one global interface circuit across the residual chart.

    Four transports, chosen by what the circuit *is* rather than by its name:

    * a **row table** whose rows are residual vectors (token embeddings, typed
      media rows) is right-multiplied by the chart -- exact, because the input
      is one-hot and the chart is the only coordinate change;
    * a **producing projection** that writes the residual from an unchanged
      input (the encoder-free media boundary, the continuous position
      projection) is left-multiplied by the chart transpose -- also exact, for
      the same reason;
    * a **consuming head** that reads the residual is refit closed-form from
      the live child's own rows to the parent's own outputs, because the
      child's rows are not exactly the chart image of the parent's;
    * a **normalization gain** is refit as a diagonal against the
      chart-projected parent output, and a **depth selector** gets the shared
      selector fold.

    Anything else fails closed.  In particular a square interface tensor is
    refused: its orientation cannot be decided from shape.
    """

    method_id = NATIVE_RESIDUAL_CHART_INTERFACE_METHOD
    branch_id = "residual_chart_interface"

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        role = "Native residual-chart interface operator"
        sealed, parent_graph, source_nodes, target_nodes = self._bind(
            action, context, evidence, role
        )
        source_by_id = {node.node_id: node for node in source_nodes}
        if set(source_by_id) != {node.node_id for node in target_nodes}:
            raise DynamicCausalOperatorPromotionError(
                f"{role} source and target interface bundles are not the same "
                "circuits."
            )
        parent_model = context.parent_models[action.sources[0].parent_name]
        paths = _installed_paths(
            action,
            tuple(
                parameter.path for node in target_nodes for parameter in node.parameters
            ),
            role,
        )
        # An exact transport reads no child rows at all, so a bundle of only
        # exact transports does not pay for a forward.
        needs_live = any(
            node.kind
            in {
                CircuitKind.NORM.value,
                CircuitKind.OUTPUT_HEAD.value,
                CircuitKind.MODALITY_OUTPUT.value,
                CircuitKind.RESIDUAL_ADD.value,
            }
            for node in target_nodes
        )
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
            live = self._recapture_child(context, role) if needs_live else None
            installs: list[tuple[str, Any]] = []
            transports: list[dict[str, Any]] = []
            for target in target_nodes:
                source = source_by_id[target.node_id]
                node_installs, node_report = self._interface_installs(
                    source=source,
                    target=target,
                    parent_model=parent_model,
                    context=context,
                    sealed=sealed,
                    live=live,
                    role=role,
                )
                installs.extend(node_installs)
                transports.append(node_report)
            report = {
                **_fold_only_operation_audit(),
                "schema": SCHEMA,
                "kind": "native_residual_chart_interface_group",
                "method_id": self.method_id,
                "source_family": parent_graph.family,
                "target_family": context.child_graph.family,
                "cached_parent_evidence_sha256": sealed.cache_key,
                "residual_chart_sha256": self.chart.map_sha256,
                "residual_chart_source_width": self.chart.source_width,
                "residual_chart_target_width": self.chart.target_width,
                "per_boundary_chart_set": False,
                "installed_parameter_paths": list(paths),
                "child_recaptured": bool(needs_live),
                "child_recaptures": int(getattr(self.recapture, "calls", 0)),
                "interface_transports": transports,
                "exact_chart_transports": sum(
                    1 for value in transports if value.get("exact") is True
                ),
                "parent_residual_coordinate_endpoints": False,
                "sequential_child_replay": True,
                "numeric_weight_matching_objective": False,
                "zero_gradient": True,
            }
            _install(context, installs)
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_parameters(context, paths))
            finally:
                snapshot.cleanup()
            raise
        return CausalBranchFoldTransaction(
            method_id=self.method_id,
            context=context,
            snapshot=snapshot,
            installed_paths=paths,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            report=report,
        )

    def _interface_installs(
        self,
        *,
        source: Any,
        target: Any,
        parent_model: Any,
        context: FoldExecutionContext,
        sealed: NativeCircuitEvidence,
        live: NativeCircuitEvidence | None,
        role: str,
    ) -> tuple[tuple[tuple[str, Any], ...], dict[str, Any]]:
        torch = _torch()
        if live is None and target.kind in {
            CircuitKind.NORM.value,
            CircuitKind.OUTPUT_HEAD.value,
            CircuitKind.MODALITY_OUTPUT.value,
            CircuitKind.RESIDUAL_ADD.value,
        }:
            raise DynamicCausalOperatorPromotionError(
                f"{role} needs live child rows for {target.node_id!r} but none "
                "were recaptured."
            )
        if target.kind == CircuitKind.RESIDUAL_ADD.value:
            installs, selector_report = self._depth_selector_installs(
                selector=target,
                parent_model=parent_model,
                child_model=context.child_model,
                sealed=sealed,
                live=live,
                role=role,
            )
            return installs, {
                "node_id": target.node_id,
                "transport": "depth_source_selector",
                "exact": False,
                **selector_report,
            }
        source_parameters = tuple(source.parameters)
        target_parameters = tuple(target.parameters)
        if len(source_parameters) != 1 or len(target_parameters) != 1:
            raise DynamicCausalOperatorPromotionError(
                f"{role} interface circuit {target.node_id!r} does not own "
                "exactly one tensor."
            )
        target_path = target_parameters[0].path
        parent_weight = parent_model.get_parameter(source_parameters[0].path).detach()
        parent_weight = parent_weight.to(device="cpu", dtype=torch.float64)
        module_path = owning_module_path(target_path)
        source_shape = tuple(int(size) for size in parent_weight.shape)
        target_shape = tuple(
            int(size) for size in context.child_model.get_parameter(target_path).shape
        )
        if target.kind == CircuitKind.NORM.value:
            if source_shape != (self.chart.source_width,) or target_shape != (
                self.chart.target_width,
            ):
                raise DynamicCausalOperatorPromotionError(
                    f"{role} normalization {target.node_id!r} does not carry one "
                    "residual-width gain on each side."
                )
            epsilon = _norm_epsilon(context.child_model, module_path, role)
            child_rows = live.input(module_path)
            target_rows = self.chart.encode(sealed.output(module_path))
            _all_child, child_fit, _child_selection, child_validation = _split_rows(
                live, child_rows
            )
            (
                _all_target,
                target_fit,
                _target_selection,
                target_validation,
            ) = _split_rows(sealed, target_rows)
            fold = fit_diagonal_norm(
                child_fit,
                target_fit,
                kind="rms_norm",
                epsilon=epsilon,
                has_bias=False,
                ridge=self.ridge,
                validation_input_rows=child_validation,
                validation_target_rows=target_validation,
            )
            return (
                ((target_path, fold.weight),),
                {
                    "node_id": target.node_id,
                    "transport": "residual_norm_gain",
                    "exact": False,
                    "fit": _fit_metrics(fold.train_metrics),
                    "validation_relative_rmse": float(
                        fold.validation_metrics.relative_rmse
                    ),
                },
            )
        if len(source_shape) != 2 or source_shape[0] == source_shape[1]:
            raise DynamicCausalOperatorPromotionError(
                f"{role} refuses interface tensor {target_path!r}: a square or "
                "non-matrix interface has no shape-decidable residual side."
            )
        if target.kind in {
            CircuitKind.OUTPUT_HEAD.value,
            CircuitKind.MODALITY_OUTPUT.value,
        }:
            if source_shape[1] != self.chart.source_width:
                raise DynamicCausalOperatorPromotionError(
                    f"{role} head {target.node_id!r} does not read the residual."
                )
            child_rows = live.input(module_path)
            parent_rows = sealed.output(module_path)
            _all_child, child_fit, _child_selection, child_validation = _split_rows(
                live, child_rows
            )
            (
                _all_parent,
                parent_fit,
                _parent_selection,
                parent_validation,
            ) = _split_rows(sealed, parent_rows)
            fold = fit_ridge_linear(
                child_fit,
                parent_fit,
                ridge=self.ridge,
                fit_bias=False,
                validation_input_rows=child_validation,
                validation_target_rows=parent_validation,
            )
            return (
                ((target_path, fold.weight),),
                {
                    "node_id": target.node_id,
                    "transport": "residual_consumer_head",
                    "exact": False,
                    "fit": _fit_metrics(fold.train_metrics),
                    "validation_relative_rmse": float(
                        fold.validation_metrics.relative_rmse
                    ),
                },
            )
        if source_shape[1] == self.chart.source_width:
            value = self.chart.fold_consumer_weight(parent_weight)
            transport = "residual_row_table"
        elif source_shape[0] == self.chart.source_width:
            value = self.chart.fold_producer_weight(parent_weight)
            transport = "residual_producer_projection"
        else:
            raise DynamicCausalOperatorPromotionError(
                f"{role} interface tensor {target_path!r} has no residual side."
            )
        exactness = self._transport_exactness(
            transport=transport,
            module_path=module_path,
            parent_weight=parent_weight,
            value=value,
            sealed=sealed,
        )
        return (
            ((target_path, value),),
            {
                "node_id": target.node_id,
                "transport": transport,
                "exact": True,
                **exactness,
            },
        )

    def _transport_exactness(
        self,
        *,
        transport: str,
        module_path: str,
        parent_weight: Any,
        value: Any,
        sealed: NativeCircuitEvidence,
    ) -> dict[str, Any]:
        """Measure the claim ``exact`` instead of asserting it.

        A producing projection reads inputs both models share, so replaying the
        child map on the cached parent input must reproduce the chart-projected
        parent output to numerical noise.  A row table is fed one-hot, so its
        replay is the transported matrix itself and the residual is exactly
        zero by construction; that is stated rather than measured.
        """

        if transport != "residual_producer_projection" or not sealed.has(module_path):
            return {"exactness_check": "one_hot_input_transport_is_exact_by_construction"}
        capture = sealed.capture(module_path)
        if not capture.inputs or capture.inputs[0] is None:
            return {"exactness_check": "no_cached_input_rows"}
        inputs = sealed.flat_rows(capture.inputs[0])
        replayed = inputs @ value.transpose(0, 1)
        expected = self.chart.encode(sealed.flat_rows(capture.output))
        return {
            "exactness_check": "cached_input_replay_against_chart_projected_output",
            "transport_replay_relative_rmse": _relative_rmse(replayed, expected),
        }


__all__ = [
    "ANCHOR_PREFERRED_PAIR_TRANSPORT",
    "CAUSAL_ATTENTION_BRANCH_METHOD",
    "CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD",
    "CAUSAL_GATED_MLP_BRANCH_METHOD",
    "CAUSAL_KDA_RECURRENT_BRANCH_METHOD",
    "CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD",
    "CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD",
    "CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD",
    "CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD",
    "KERNEL_SUPPORTS_ATTENTION_WINDOW",
    "KERNEL_SUPPORTS_POSITION_ENCODING",
    "NATIVE_RESIDUAL_CHART_INTERFACE_METHOD",
    "SCHEMA",
    "TOKEN_INTERFACE_BRIDGE_METHOD",
    "BoundedGatedDeltaRecurrenceLayerOperator",
    "CausalBranchFoldTransaction",
    "CausalGQAAttentionBranchOperator",
    "CausalGatedMLPBranchOperator",
    "CompressedLatentKVAttentionLayerOperator",
    "CompressedSummaryAttentionLayerOperator",
    "CrossWidthBranchBinding",
    "CrossWidthCausalBranchOperator",
    "CrossWidthTokenHeadInterfaceOperator",
    "DynamicCausalOperatorPromotionError",
    "IdentityVocabularyProjector",
    "NativeFusedGatedMLPBranchOperator",
    "NativeResidualChartInterfaceOperator",
    "NativeWindowedNoPEAttentionBranchOperator",
    "PairCodeTokenInterfaceBridgeOperator",
    "StableLatentMoEChannelMixerLayerOperator",
    "StrictTokenInterfaceBridgeOperator",
    "TokenInterfaceBridgeTransaction",
]
