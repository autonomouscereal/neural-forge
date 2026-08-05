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
from dataclasses import dataclass, field, fields
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
    compile_hf_causal_gqa_attention_fold,
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
    "CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD",
    "KERNEL_SUPPORTS_ATTENTION_WINDOW",
    "KERNEL_SUPPORTS_POSITION_ENCODING",
    "NATIVE_RESIDUAL_CHART_INTERFACE_METHOD",
    "SCHEMA",
    "TOKEN_INTERFACE_BRIDGE_METHOD",
    "CausalBranchFoldTransaction",
    "CausalGQAAttentionBranchOperator",
    "CausalGatedMLPBranchOperator",
    "CrossWidthBranchBinding",
    "CrossWidthCausalBranchOperator",
    "CrossWidthTokenHeadInterfaceOperator",
    "DynamicCausalOperatorPromotionError",
    "IdentityVocabularyProjector",
    "NativeFusedGatedMLPBranchOperator",
    "NativeResidualChartInterfaceOperator",
    "NativeWindowedNoPEAttentionBranchOperator",
    "PairCodeTokenInterfaceBridgeOperator",
    "StrictTokenInterfaceBridgeOperator",
    "TokenInterfaceBridgeTransaction",
]
