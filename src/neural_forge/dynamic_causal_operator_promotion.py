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
from dataclasses import dataclass, field
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
    CAUSAL_GATED_MLP_BRANCH_METHOD,
    FoldAction,
)
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig, capture_model_state
from .hf_arch_layout import DecoderBlockLayout, DecoderLayout, build_decoder_layout
from .hf_causal_circuit_evidence import HFCausalCircuitEvidence
from .hf_causal_gqa_attention_fold import (
    HFCausalGQATopology,
    compile_hf_causal_gqa_attention_fold,
)
from .hf_closed_form_causal_circuit_fold import (
    fit_functional_swiglu,
    fit_rmsnorm_diagonal,
)
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    compile_strict_vocabulary_operators,
    load_strict_tokenizer_bridge_artifact,
)


SCHEMA = "neural-forge-dynamic-causal-operator-promotion-v1"

TOKEN_INTERFACE_BRIDGE_METHOD = "token_interface_bridge"

_GATED_ACTIVATIONS = frozenset({"silu", "swish"})


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


def _require_direct_rms_norm(block_norm: Any, role: str, name: str) -> None:
    if (
        block_norm.kind != "rms_norm"
        or block_norm.weight_kind != "direct_scale"
        or block_norm.has_bias
    ):
        raise DynamicCausalOperatorPromotionError(
            f"{role} requires a bias-free direct-scale RMSNorm {name}."
        )


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
        _require_shared_residual_width(layout, sealed, role)
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
    ) -> CausalBranchFoldTransaction:
        role = "Causal attention branch operator"
        sealed, layer, layout, block, source_index = self._bind(
            action,
            context,
            evidence,
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
        if attention.query_norm is not None or attention.key_norm is not None:
            raise DynamicCausalOperatorPromotionError(
                f"{role} cannot yet install a target with per-head Q/K "
                "normalization: the coupled kernel fits pre-RoPE target "
                "coordinates with identity Q/K semantics."
            )
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
            rms_norm_epsilon=block.attention_input_norm.epsilon,
        )
        norm_path = block.attention_input_norm.path + ".weight"
        query_path = str(attention.query_path) + ".weight"
        key_path = str(attention.key_path) + ".weight"
        value_path = str(attention.value_path) + ".weight"
        output_path = str(attention.output_path) + ".weight"
        paths = _installed_paths(
            action,
            (norm_path, query_path, key_path, value_path, output_path),
            role,
        )
        fit = sealed.fit_token_mask
        validation = sealed.validation_token_mask
        norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(layer.block_input, fit),
            _masked_rows(layer.attention_norm_output, fit),
            epsilon=block.attention_input_norm.epsilon,
            ridge=self.ridge,
        )
        normalized = _rmsnorm(
            layer.block_input,
            norm_fold.weight,
            block.attention_input_norm.epsilon,
        )
        attention_fold = compile_hf_causal_gqa_attention_fold(
            normalized,
            source_query_sequences=layer.query_projection_output,
            source_key_sequences=layer.key_projection_output,
            source_value_sequences=layer.value_projection_output,
            target_residual_delta_sequences=layer.attention_delta,
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
                _masked_rows(layer.attention_delta, validation),
            ),
            "parent_residual_coordinate_endpoints": True,
            "sequential_child_replay": False,
            "numeric_weight_matching_objective": False,
            "zero_gradient": True,
        }
        incumbent = context.child_state_fingerprint
        snapshot = _capture(context, paths, self.snapshot_config)
        try:
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


class CausalGatedMLPBranchOperator(_CausalBranchOperator):
    """Install one coupled SwiGLU residual-MLP branch from sealed evidence."""

    method_id = CAUSAL_GATED_MLP_BRANCH_METHOD
    branch_id = "gated_mlp"

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> CausalBranchFoldTransaction:
        role = "Causal gated-MLP branch operator"
        sealed, layer, layout, block, source_index = self._bind(
            action,
            context,
            evidence,
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
        norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(layer.post_attention_residual, fit),
            _masked_rows(layer.mlp_norm_output, fit),
            epsilon=block.mlp_input_norm.epsilon,
            ridge=self.ridge,
        )
        normalized = _rmsnorm(
            layer.post_attention_residual,
            norm_fold.weight,
            block.mlp_input_norm.epsilon,
        )
        mlp_fold = fit_functional_swiglu(
            _masked_rows(normalized, fit),
            _masked_rows(layer.mlp_gate_output, fit),
            _masked_rows(layer.mlp_up_output, fit),
            _masked_rows(layer.mlp_delta, fit),
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
                _masked_rows(layer.mlp_delta, validation),
            ),
            "parent_residual_coordinate_endpoints": True,
            "sequential_child_replay": False,
            "numeric_weight_matching_objective": False,
            "zero_gradient": True,
        }
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
    """One atomic snapshot-backed install of a strict token-row interface."""

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


__all__ = [
    "CAUSAL_ATTENTION_BRANCH_METHOD",
    "CAUSAL_GATED_MLP_BRANCH_METHOD",
    "SCHEMA",
    "TOKEN_INTERFACE_BRIDGE_METHOD",
    "CausalBranchFoldTransaction",
    "CausalGQAAttentionBranchOperator",
    "CausalGatedMLPBranchOperator",
    "DynamicCausalOperatorPromotionError",
    "IdentityVocabularyProjector",
    "StrictTokenInterfaceBridgeOperator",
    "TokenInterfaceBridgeTransaction",
]
