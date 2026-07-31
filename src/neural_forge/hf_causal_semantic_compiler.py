"""Production same-depth dense Llama semantic folding compiler.

The compiler composes the generic blank-target, semantic-stage, and
whole-model transaction ABIs.  It installs one complete standalone child from
one sealed parent evidence bundle.  Component diagnostics remain receipts
inside that one full-model operation.

Version one intentionally fails closed on depth composition, tokenizer
translation, non-Llama parents, non-default RoPE, projection biases, and Q/K
normalization crossings.  Those capabilities are not advertised until their
operators are independently implemented.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_closed_loop_fold import CompileStage
from .dynamic_fold_budget import (
    SEMANTIC_ATTENTION_QUERY_KEY,
    SEMANTIC_ATTENTION_WRITER,
    SEMANTIC_INTERFACE,
    SEMANTIC_MLP_PRODUCER,
    SEMANTIC_MLP_WRITER,
    SEMANTIC_NORMALIZATION,
    SEMANTIC_OTHER,
    SEMANTIC_STATE_MIXER_INTERNAL,
)
from .dynamic_parameter_frontier import HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY
from .dynamic_semantic_circuit_stage import (
    SemanticCircuitSelector,
    SemanticCircuitStageContract,
    SemanticCircuitStageRegistry,
    SemanticCircuitStageReport,
)
from .dynamic_semantic_fold import (
    DynamicSemanticFoldError,
    SemanticComponentFoldReceipt,
    SemanticWholeModelCompileRequest,
    SemanticWholeModelCompileResult,
    SemanticWholeModelFoldEvidence,
    SemanticWholeModelFoldResources,
    SemanticWholeModelTargetBinding,
)
from .dynamic_target_synthesis import DynamicTargetSynthesisResult
from .hf_arch_layout import DecoderLayout, build_decoder_layout, resolve_module
from .hf_causal_circuit_evidence import HFCausalCircuitEvidence
from .hf_closed_form_causal_circuit_fold import (
    CausalCircuitFoldIntegrationContract,
    CausalFeatureBlock,
    DenseParameterSite,
    FoldingOnlyOperationAudit,
    fit_functional_swiglu,
    fit_predictive_boundary_map,
    fit_rmsnorm_diagonal,
    seal_causal_circuit_fold_plan,
)
from .hf_causal_gqa_attention_fold import (
    HFCausalGQATopology,
    compile_hf_causal_gqa_attention_fold,
)


SCHEMA = "neural-forge-hf-causal-same-depth-semantic-compiler-v1"
HF_CAUSAL_SAME_DEPTH_SEMANTIC_METHOD = (
    "hf-causal-same-depth-semantic-fold-v1"
)
HF_CAUSAL_SAME_DEPTH_RESOLVER_ID = (
    "neural_forge.hf.causal_same_depth_semantic"
)


class HFCausalSemanticCompilerError(DynamicSemanticFoldError):
    """The target/evidence pair is outside the implemented fold envelope."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalSemanticCompilerError(
            "PyTorch is required for HF causal semantic folding."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HFCausalSemanticCompilerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalSemanticCompilerError(f"{name} must be non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise HFCausalSemanticCompilerError(
            f"{name} must be finite and non-negative."
        )
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _canonical_report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    decoded = json.loads(json.dumps(dict(body), allow_nan=False, sort_keys=True))
    decoded["report_sha256"] = canonical_sha256(decoded)
    return MappingProxyType(decoded)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _named_parameters(model: Any) -> dict[str, Any]:
    try:
        values = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError:  # pragma: no cover - old Torch compatibility
        values = dict(model.named_parameters(recurse=True))
    if not values:
        raise HFCausalSemanticCompilerError("The child exposes no parameters.")
    return values


def _decode_singleton_spans(
    target: DynamicTargetSynthesisResult,
    source_depth: int,
) -> tuple[tuple[int, ...], ...]:
    value = dict(target.request.configuration).get(
        HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY
    )
    if not isinstance(value, str) or not value:
        raise HFCausalSemanticCompilerError(
            "The target lacks authenticated ordered source spans."
        )
    try:
        spans = tuple(
            tuple(int(item) for item in group.split(","))
            for group in value.split(";")
        )
    except ValueError as exc:
        raise HFCausalSemanticCompilerError(
            "The target ordered source spans are malformed."
        ) from exc
    if (
        spans != tuple((index,) for index in range(source_depth))
        or len(target.graph.layer_indices) != source_depth
    ):
        raise HFCausalSemanticCompilerError(
            "Ordered depth composition is not implemented: every target layer "
            "must own exactly one same-index source layer."
        )
    return spans


def _encode_sequences(boundary: Any, values: Any) -> Any:
    batch, sequence, width = values.shape
    encoded = boundary.encode(values.reshape(int(batch * sequence), int(width)))
    return encoded.reshape(int(batch), int(sequence), boundary.target_width)


def _masked_rows(values: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in values.shape[2:])
    return values.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _relative_rmse(prediction: Any, target: Any) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _rmsnorm(values: Any, weight: Any, epsilon: float) -> Any:
    return values * (
        values.square().mean(dim=-1, keepdim=True) + epsilon
    ).rsqrt() * weight.reshape(1, 1, -1)


def _boundary_map(
    state: Any,
    features: Mapping[str, tuple[str, Any]],
    *,
    target_width: int,
    fit_mask: Any,
    validation_mask: Any,
    ridge: float,
) -> Any:
    fit_features = tuple(
        CausalFeatureBlock(
            name=name,
            role=role,
            values=_masked_rows(value, fit_mask),
        )
        for name, (role, value) in features.items()
    )
    validation_features = tuple(
        CausalFeatureBlock(
            name=name,
            role=role,
            values=_masked_rows(value, validation_mask),
        )
        for name, (role, value) in features.items()
    )
    return fit_predictive_boundary_map(
        _masked_rows(state, fit_mask),
        fit_features,
        target_width=target_width,
        validation_states=_masked_rows(state, validation_mask).clone(),
        validation_features=validation_features,
        ridge=ridge,
        minimum_predictive_energy_fraction=0.0,
        minimum_validation_block_retention=0.0,
        enforce_validation_block_floor=False,
    )


def _source_topology(evidence: HFCausalCircuitEvidence) -> HFCausalGQATopology:
    value = evidence.architecture
    return HFCausalGQATopology(
        query_heads=value.query_heads,
        key_value_heads=value.key_value_heads,
        head_dim=value.head_dim,
        rope_theta=value.rope_theta,
        attention_scaling=value.attention_scaling,
        query_key_norm=value.query_key_norm,
        rms_norm_epsilon=value.rms_norm_epsilon,
    )


def _target_topology(layout: DecoderLayout) -> HFCausalGQATopology:
    attention = layout.blocks[0].attention
    return HFCausalGQATopology(
        query_heads=attention.query_heads,
        key_value_heads=attention.key_value_heads,
        head_dim=attention.head_dim,
        rope_theta=float(layout.position.rope_theta),
        attention_scaling=(
            attention.scaling
            if attention.scaling is not None
            else attention.head_dim ** -0.5
        ),
        query_key_norm=(
            "identity"
            if attention.query_norm is None and attention.key_norm is None
            else "rms_norm"
        ),
        rms_norm_epsilon=layout.blocks[0].attention_input_norm.epsilon,
    )


@dataclass(slots=True)
class _CompileWorkspace:
    child_model: Any
    child_graph: Any
    target_layout: DecoderLayout
    evidence: HFCausalCircuitEvidence
    ridge: float
    input_boundary: Any
    post_attention_boundaries: tuple[Any, ...]
    output_boundaries: tuple[Any, ...]
    final_boundary: Any
    write_counts: dict[str, int] = field(default_factory=dict)
    component_reports: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    final_child_sequences: Any | None = None
    final_logits: Any | None = None

    def _install(self, request: Any, path: str, value: Any) -> None:
        torch = _torch()
        if path not in request.write_parameter_paths:
            raise HFCausalSemanticCompilerError(
                f"Stage {request.stage_id!r} cannot write {path!r}."
            )
        count = self.write_counts.get(path, 0)
        if count:
            raise HFCausalSemanticCompilerError(
                f"Target parameter {path!r} would be installed more than once."
            )
        parameter = _named_parameters(self.child_model).get(path)
        if parameter is None or tuple(parameter.shape) != tuple(value.shape):
            raise HFCausalSemanticCompilerError(
                f"Compiled tensor for {path!r} has the wrong target shape."
            )
        with torch.no_grad():
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        self.write_counts[path] = 1

    def install_interface(self, request: Any) -> SemanticCircuitStageReport:
        embedding_path = self.target_layout.embeddings.token_path + ".weight"
        head_path = self.target_layout.lm_head_path + ".weight"
        source_embedding = self.evidence.token_embedding_weight.to(_torch().float64)
        source_head = self.evidence.lm_head_weight.to(_torch().float64)
        embedding = source_embedding @ self.input_boundary.encoder
        head = source_head @ self.final_boundary.decoder.T
        self._install(request, embedding_path, embedding)
        self._install(request, head_path, head)
        body = {
            "schema": SCHEMA,
            "component": "same_tokenizer_dense_interface",
            "method": "boundary_encoder_embedding_and_decoder_covector_head",
            "vocabulary_rows": self.evidence.architecture.vocabulary_size,
            "embedding_weight_sha256": _tensor_sha256(embedding),
            "head_weight_sha256": _tensor_sha256(head),
            "tokenizer_bridge_used": False,
            "numeric_weight_matching_objective": False,
            "writes": 2,
        }
        report = _canonical_report(body)
        self.component_reports["token_interface"] = report
        return SemanticCircuitStageReport(
            partition_receipts=(("token_interface", report),),
            operation_audit=_stage_operation_audit(),
            stage_receipt={
                "component": "token_interface",
                "parameter_writes": 2,
                "report_sha256": report["report_sha256"],
            },
        )

    def install_body(self, request: Any) -> SemanticCircuitStageReport:
        torch = _torch()
        target_topology = _target_topology(self.target_layout)
        source_topology = _source_topology(self.evidence)
        token_weight = resolve_module(
            self.child_model,
            self.target_layout.embeddings.token_path,
            family="llama",
        ).weight.detach().to(device="cpu", dtype=torch.float64)
        child = token_weight.index_select(
            0, self.evidence.input_ids.reshape(-1).to(torch.long)
        ).reshape(*self.evidence.input_ids.shape, self.target_layout.hidden_size)
        layer_receipts = []
        for index, (target_block, source_layer) in enumerate(
            zip(self.target_layout.blocks, self.evidence.layers, strict=True)
        ):
            input_boundary = (
                self.input_boundary if index == 0 else self.output_boundaries[index - 1]
            )
            post_boundary = self.post_attention_boundaries[index]
            output_boundary = self.output_boundaries[index]
            desired_attention_norm = _encode_sequences(
                input_boundary, source_layer.attention_norm_output
            )
            attention_norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(child, self.evidence.fit_token_mask),
                _masked_rows(desired_attention_norm, self.evidence.fit_token_mask),
                epsilon=target_block.attention_input_norm.epsilon,
                ridge=self.ridge,
            )
            attention_norm_path = target_block.attention_input_norm.path + ".weight"
            self._install(request, attention_norm_path, attention_norm_fold.weight)
            normalized = _rmsnorm(
                child,
                attention_norm_fold.weight,
                target_block.attention_input_norm.epsilon,
            )
            desired_post_attention = _encode_sequences(
                post_boundary, source_layer.post_attention_residual
            )
            attention_fold = compile_hf_causal_gqa_attention_fold(
                normalized,
                source_query_sequences=source_layer.query_projection_output,
                source_key_sequences=source_layer.key_projection_output,
                source_value_sequences=source_layer.value_projection_output,
                target_residual_delta_sequences=desired_post_attention - child,
                attention_mask=self.evidence.attention_mask,
                position_ids=self.evidence.position_ids,
                valid_token_mask=self.evidence.valid_token_mask,
                fit_token_mask=self.evidence.fit_token_mask,
                validation_token_mask=self.evidence.validation_token_mask,
                source_topology=source_topology,
                target_topology=target_topology,
                source_query_norm_weight=source_layer.query_norm_weight,
                source_key_norm_weight=source_layer.key_norm_weight,
                ridge=self.ridge,
            )
            attention_paths = (
                target_block.attention.query_path,
                target_block.attention.key_path,
                target_block.attention.value_path,
                target_block.attention.output_path,
            )
            if any(path is None for path in attention_paths[:3]):
                raise HFCausalSemanticCompilerError(
                    "The target attention is not split Q/K/V."
                )
            for path, value in zip(
                attention_paths,
                (
                    attention_fold.query_weight,
                    attention_fold.key_weight,
                    attention_fold.value_weight,
                    attention_fold.output_weight,
                ),
                strict=True,
            ):
                self._install(request, str(path) + ".weight", value)
            attention_delta, _scores, _probabilities = attention_fold.replay(
                normalized,
                attention_mask=self.evidence.attention_mask,
                position_ids=self.evidence.position_ids,
            )
            post_attention = child + attention_delta
            desired_mlp_norm = _encode_sequences(
                post_boundary, source_layer.mlp_norm_output
            )
            mlp_norm_fold = fit_rmsnorm_diagonal(
                _masked_rows(post_attention, self.evidence.fit_token_mask),
                _masked_rows(desired_mlp_norm, self.evidence.fit_token_mask),
                epsilon=target_block.mlp_input_norm.epsilon,
                ridge=self.ridge,
            )
            mlp_norm_path = target_block.mlp_input_norm.path + ".weight"
            self._install(request, mlp_norm_path, mlp_norm_fold.weight)
            mlp_normalized = _rmsnorm(
                post_attention,
                mlp_norm_fold.weight,
                target_block.mlp_input_norm.epsilon,
            )
            desired_block_output = _encode_sequences(
                output_boundary, source_layer.block_output
            )
            mlp_fold = fit_functional_swiglu(
                _masked_rows(mlp_normalized, self.evidence.fit_token_mask),
                _masked_rows(source_layer.mlp_gate_output, self.evidence.fit_token_mask),
                _masked_rows(source_layer.mlp_up_output, self.evidence.fit_token_mask),
                _masked_rows(
                    desired_block_output - post_attention,
                    self.evidence.fit_token_mask,
                ),
                target_neurons=target_block.mlp.intermediate_size,
                ridge=self.ridge,
            )
            mlp_paths = (
                target_block.mlp.gate_projection_path,
                target_block.mlp.up_projection_path,
                target_block.mlp.down_projection_path,
            )
            if any(path is None for path in mlp_paths[:2]):
                raise HFCausalSemanticCompilerError("The target MLP is not SwiGLU.")
            for path, value in zip(
                mlp_paths,
                (mlp_fold.gate_weight, mlp_fold.up_weight, mlp_fold.down_weight),
                strict=True,
            ):
                self._install(request, str(path) + ".weight", value)
            flattened = mlp_normalized.reshape(-1, self.target_layout.hidden_size)
            mlp_delta = mlp_fold.predict(flattened).reshape_as(post_attention)
            child = post_attention + mlp_delta
            validation_prediction = _masked_rows(child, self.evidence.validation_token_mask)
            validation_target = _masked_rows(
                desired_block_output, self.evidence.validation_token_mask
            )
            layer_receipts.append(
                {
                    "layer_index": index,
                    "source_layer_index": source_layer.index,
                    "attention_norm_report_sha256": attention_norm_fold.report[
                        "report_sha256"
                    ],
                    "attention_report_sha256": attention_fold.report["report_sha256"],
                    "mlp_norm_report_sha256": mlp_norm_fold.report["report_sha256"],
                    "swiglu_report_sha256": mlp_fold.report["report_sha256"],
                    "validation_block_relative_rmse": _relative_rmse(
                        validation_prediction, validation_target
                    ),
                    "installed_parameter_count": 9,
                    "sequential_child_replay": True,
                }
            )
        desired_final_norm = _encode_sequences(
            self.final_boundary, self.evidence.final_norm_output
        )
        final_norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(child, self.evidence.fit_token_mask),
            _masked_rows(desired_final_norm, self.evidence.fit_token_mask),
            epsilon=self.target_layout.final_norm.epsilon,
            ridge=self.ridge,
        )
        final_norm_path = self.target_layout.final_norm.path + ".weight"
        self._install(request, final_norm_path, final_norm_fold.weight)
        final_sequences = _rmsnorm(
            child,
            final_norm_fold.weight,
            self.target_layout.final_norm.epsilon,
        )
        head_weight = resolve_module(
            self.child_model, self.target_layout.lm_head_path, family="llama"
        ).weight.detach().to(device="cpu", dtype=torch.float64)
        logits = final_sequences @ head_weight.T
        self.final_child_sequences = final_sequences
        self.final_logits = logits
        body = {
            "schema": SCHEMA,
            "component": "same_depth_causal_decoder_body",
            "method": "sequential_boundary_qk_softmax_vo_swiglu_replay",
            "layer_count": len(layer_receipts),
            "layers": layer_receipts,
            "final_norm_report_sha256": final_norm_fold.report["report_sha256"],
            "final_validation_logit_relative_rmse": _relative_rmse(
                _masked_rows(logits, self.evidence.validation_token_mask),
                _masked_rows(self.evidence.logits, self.evidence.validation_token_mask),
            ),
            "sequential_target_only_replay": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "parameter_writes": 9 * len(layer_receipts) + 1,
        }
        report = _canonical_report(body)
        self.component_reports["causal_decoder_body"] = report
        return SemanticCircuitStageReport(
            partition_receipts=(("causal_decoder_body", report),),
            operation_audit=_stage_operation_audit(),
            stage_receipt={
                "component": "causal_decoder_body",
                "layer_count": len(layer_receipts),
                "parameter_writes": 9 * len(layer_receipts) + 1,
                "report_sha256": report["report_sha256"],
            },
        )


class _TokenInterfaceStageCompiler:
    semantic_stage_resource_identity = canonical_sha256(
        {"schema": SCHEMA, "stage": "token_interface", "version": 1}
    )

    def compile_stage(self, request: Any, shared_workspace: Any, prior: Any) -> Any:
        if prior:
            raise HFCausalSemanticCompilerError(
                "The token-interface stage cannot have prior results."
            )
        return shared_workspace.install_interface(request)


class _CausalDecoderBodyStageCompiler:
    semantic_stage_resource_identity = canonical_sha256(
        {"schema": SCHEMA, "stage": "causal_decoder_body", "version": 1}
    )

    def compile_stage(self, request: Any, shared_workspace: Any, prior: Any) -> Any:
        if set(prior) != {"token_interface"}:
            raise HFCausalSemanticCompilerError(
                "The causal body requires the installed token interface."
            )
        return shared_workspace.install_body(request)


def _stage_operation_audit() -> dict[str, Any]:
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


def _semantic_operation_audit() -> dict[str, Any]:
    return {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "parameter_sharing_operations": 0,
        "sidecar_parameters": 0,
        "cached_parent_evidence_only": True,
        "complete_model_compaction": True,
        "component_diagnostics_have_acceptance_authority": False,
        "numeric_diagnostics_have_acceptance_authority": False,
    }


def _registry() -> SemanticCircuitStageRegistry:
    interface = SemanticCircuitSelector(parameter_semantics=(SEMANTIC_INTERFACE,))
    body = SemanticCircuitSelector(
        parameter_semantics=(
            SEMANTIC_STATE_MIXER_INTERNAL,
            SEMANTIC_ATTENTION_QUERY_KEY,
            SEMANTIC_ATTENTION_WRITER,
            SEMANTIC_MLP_PRODUCER,
            SEMANTIC_MLP_WRITER,
            SEMANTIC_NORMALIZATION,
            SEMANTIC_OTHER,
        )
    )
    registry = SemanticCircuitStageRegistry()
    token_compiler = _TokenInterfaceStageCompiler()
    registry.register(
        SemanticCircuitStageContract(
            stage_id="token_interface",
            stage_role=CompileStage.OUTPUT_INTERFACE,
            dependencies=(),
            required_selectors=(interface,),
            mutation_selectors=(interface,),
            receipt_partitions=(("token_interface", interface),),
            requires_effectful_write=True,
        ),
        token_compiler,
        resource_identity=token_compiler.semantic_stage_resource_identity,
        implementation_id="neural-forge.hf-causal.token-interface-stage",
        version=1,
    )
    body_compiler = _CausalDecoderBodyStageCompiler()
    registry.register(
        SemanticCircuitStageContract(
            stage_id="causal_decoder_body",
            stage_role=CompileStage.BODY_RECLOSURE,
            dependencies=("token_interface",),
            required_selectors=(interface, body),
            mutation_selectors=(body,),
            receipt_partitions=(("causal_decoder_body", body),),
            requires_effectful_write=True,
        ),
        body_compiler,
        resource_identity=body_compiler.semantic_stage_resource_identity,
        implementation_id="neural-forge.hf-causal.decoder-body-stage",
        version=1,
    )
    return registry


class HFCausalSameDepthSemanticCompiler:
    """Compile one complete high-ratio Llama child without optimization."""

    compiler_id = "neural-forge.hf-causal.same-depth-semantic-compiler"
    version = 1

    def __init__(self, *, ridge: float = 1e-6) -> None:
        self.ridge = _nonnegative_float(ridge, "ridge")
        self.resource_fingerprint = canonical_sha256(
            {"schema": SCHEMA, "compiler": self.compiler_id, "ridge": self.ridge}
        )

    def _validate(
        self,
        request: SemanticWholeModelCompileRequest,
        evidence: HFCausalCircuitEvidence,
        target_layout: DecoderLayout,
    ) -> None:
        architecture = evidence.architecture
        if architecture.family != "llama" or target_layout.family != "llama":
            raise HFCausalSemanticCompilerError(
                "Version one implements same-family Llama-to-Llama folding only."
            )
        _decode_singleton_spans(request.target, architecture.layer_count)
        if target_layout.layer_count != architecture.layer_count:
            raise HFCausalSemanticCompilerError(
                "Source and target depths must match in the same-depth compiler."
            )
        if architecture.query_key_norm != "identity" or any(
            block.attention.query_norm is not None
            or block.attention.key_norm is not None
            for block in target_layout.blocks
        ):
            raise HFCausalSemanticCompilerError(
                "Q/K normalization translation is not implemented by version one."
            )
        if architecture.rope_type != "default" or target_layout.position.rope_type not in {
            None,
            "default",
        }:
            raise HFCausalSemanticCompilerError(
                "Only default RoPE is implemented by version one."
            )
        if architecture.attention_projection_bias or architecture.mlp_projection_bias:
            raise HFCausalSemanticCompilerError(
                "Biased source projections are outside the first fold envelope."
            )
        if any(
            block.attention.projection_bias or block.mlp.projection_bias
            for block in target_layout.blocks
        ):
            raise HFCausalSemanticCompilerError(
                "The dense target must use bias-free attention and SwiGLU."
            )
        target_attention = target_layout.blocks[0].attention
        if (
            target_layout.hidden_size > architecture.hidden_size
            or target_layout.intermediate_size > architecture.intermediate_size
            or target_attention.query_heads > architecture.query_heads
            or target_attention.key_value_heads > architecture.key_value_heads
            or target_attention.head_dim > architecture.head_dim
        ):
            raise HFCausalSemanticCompilerError(
                "Width/head expansion is outside the contraction-only compiler."
            )
        if target_layout.embeddings.vocab_size != architecture.vocabulary_size:
            raise HFCausalSemanticCompilerError(
                "Version one requires identical source/target vocabulary rows."
            )
        if target_layout.tie_word_embeddings:
            raise HFCausalSemanticCompilerError(
                "The target must be dense and untied; parameter sharing is forbidden."
            )
        parent_graphs = {
            graph.fingerprint for graph in request.context.parent_graphs.values()
        }
        if evidence.parent_graph_fingerprint not in parent_graphs:
            raise HFCausalSemanticCompilerError(
                "Cached evidence is bound to a different parent graph."
            )

    def _workspace(
        self,
        request: SemanticWholeModelCompileRequest,
        evidence: HFCausalCircuitEvidence,
        target_layout: DecoderLayout,
    ) -> _CompileWorkspace:
        fit = evidence.fit_token_mask
        validation = evidence.validation_token_mask
        width = target_layout.hidden_size
        first = evidence.layers[0]
        input_boundary = _boundary_map(
            first.block_input,
            {
                "attention_delta": ("attention_writer", first.attention_delta),
                "mlp_delta": ("mlp_writer", first.mlp_delta),
                "block_output": ("future_residual", first.block_output),
            },
            target_width=width,
            fit_mask=fit,
            validation_mask=validation,
            ridge=self.ridge,
        )
        post_boundaries = []
        output_boundaries = []
        for index, layer in enumerate(evidence.layers):
            post_boundaries.append(
                _boundary_map(
                    layer.post_attention_residual,
                    {
                        "mlp_delta": ("mlp_writer", layer.mlp_delta),
                        "block_output": ("future_residual", layer.block_output),
                        "final_norm": ("terminal_state", evidence.final_norm_output),
                    },
                    target_width=width,
                    fit_mask=fit,
                    validation_mask=validation,
                    ridge=self.ridge,
                )
            )
            next_attention = (
                evidence.layers[index + 1].attention_delta
                if index + 1 < len(evidence.layers)
                else evidence.final_norm_output
            )
            output_boundaries.append(
                _boundary_map(
                    layer.block_output,
                    {
                        "next_attention": ("future_attention", next_attention),
                        "final_norm": ("terminal_state", evidence.final_norm_output),
                        "current_mlp_delta": ("mlp_writer", layer.mlp_delta),
                    },
                    target_width=width,
                    fit_mask=fit,
                    validation_mask=validation,
                    ridge=self.ridge,
                )
            )
        logits_width = min(evidence.architecture.hidden_size, evidence.architecture.vocabulary_size)
        final_boundary = _boundary_map(
            evidence.final_norm_output,
            {
                "terminal_logits": (
                    "output_action",
                    evidence.logits[..., :logits_width],
                ),
                "last_block": ("residual_reconstruction", evidence.layers[-1].block_output),
            },
            target_width=width,
            fit_mask=fit,
            validation_mask=validation,
            ridge=self.ridge,
        )
        return _CompileWorkspace(
            child_model=request.context.child_model,
            child_graph=request.context.child_graph,
            target_layout=target_layout,
            evidence=evidence,
            ridge=self.ridge,
            input_boundary=input_boundary,
            post_attention_boundaries=tuple(post_boundaries),
            output_boundaries=tuple(output_boundaries),
            final_boundary=final_boundary,
        )

    def compile(
        self,
        request: SemanticWholeModelCompileRequest,
    ) -> SemanticWholeModelCompileResult:
        if not isinstance(request, SemanticWholeModelCompileRequest):
            raise HFCausalSemanticCompilerError(
                "compile request must be SemanticWholeModelCompileRequest."
            )
        source = request.evidence.source
        if not isinstance(source, HFCausalCircuitEvidence):
            raise HFCausalSemanticCompilerError(
                "Semantic evidence source must be HFCausalCircuitEvidence."
            )
        evidence = source.verify()
        if request.evidence.root_fingerprint != evidence.cache_key:
            raise HFCausalSemanticCompilerError(
                "Semantic evidence root differs from the full-circuit cache."
            )
        target_layout = build_decoder_layout(request.context.child_model)
        self._validate(request, evidence, target_layout)
        workspace = self._workspace(request, evidence, target_layout)
        compilation = _registry().compile(
            child_model=request.context.child_model,
            child_graph=request.context.child_graph,
            shared_workspace=workspace,
        )
        parameters = _named_parameters(request.context.child_model)
        if set(workspace.write_counts) != set(parameters) or any(
            count != 1 for count in workspace.write_counts.values()
        ):
            raise HFCausalSemanticCompilerError(
                "The compiler did not install every dense target parameter exactly once."
            )
        sites_by_path = {
            parameter.path: parameter
            for node in request.context.child_graph.nodes
            for parameter in node.parameters
        }
        sites = tuple(
            DenseParameterSite(
                name=path,
                shape=tuple(int(size) for size in value.shape),
                semantic_role=sites_by_path[path].role,
                alias_group=sites_by_path[path].storage_group,
            )
            for path, value in sorted(parameters.items())
        )
        contract = CausalCircuitFoldIntegrationContract(
            parent_cache_sha256=evidence.cache_key,
            fit_split_sha256=evidence.tensor_manifest["fit_token_mask"],
            validation_split_sha256=evidence.tensor_manifest[
                "validation_token_mask"
            ],
            target_topology_sha256=request.target.topology_fingerprint,
            target_parameter_count=request.context.child_graph.unique_parameter_count,
            target_sites=sites,
            minimum_predictive_energy_fraction=0.0,
            minimum_validation_feature_retention=0.0,
        )
        plan = seal_causal_circuit_fold_plan(
            contract,
            parameters,
            component_reports=workspace.component_reports,
            operation_audit=FoldingOnlyOperationAudit(),
        )
        if workspace.final_logits is None:
            raise HFCausalSemanticCompilerError(
                "Sequential replay did not produce the final behavior packet."
            )
        valid = evidence.valid_token_mask
        parent_top_tokens = evidence.logits.argmax(dim=-1)[valid].tolist()
        child_top_tokens = workspace.final_logits.argmax(dim=-1)[valid].tolist()
        behavior_packet = {
            "input_token_ids": evidence.input_ids[valid].tolist(),
            "parent_top_token_ids": parent_top_tokens,
            "child_top_token_ids": child_top_tokens,
            "valid_token_count": int(valid.sum().item()),
            "tokenizer_decode_available": False,
            "manual_review_required": True,
            "intelligence_claim": False,
        }
        components = []
        for stage_result in compilation.stage_results:
            for partition in stage_result.partition_receipts:
                components.append(
                    SemanticComponentFoldReceipt(
                        component_id=partition.partition_id,
                        covered_parameter_paths=partition.parameter_paths,
                        receipt=_plain_json(partition.receipt),
                    )
                )
        return SemanticWholeModelCompileResult(
            covered_target_parameter_paths=compilation.manifest.learned_parameter_paths,
            component_receipts=tuple(components),
            whole_model_receipt={
                "schema": SCHEMA,
                "status": "complete-same-depth-llama-child-installed",
                "source_family": evidence.architecture.family,
                "target_family": target_layout.family,
                "source_parameter_count": sum(
                    graph.unique_parameter_count
                    for graph in request.context.parent_graphs.values()
                ),
                "target_parameter_count": request.context.child_graph.unique_parameter_count,
                "parameter_compaction_ratio": sum(
                    graph.unique_parameter_count
                    for graph in request.context.parent_graphs.values()
                )
                / request.context.child_graph.unique_parameter_count,
                "layer_count": target_layout.layer_count,
                "stage_compilation_sha256": compilation.result_fingerprint,
                "causal_fold_plan_sha256": plan.plan_sha256,
                "dense_parameter_audit_sha256": plan.parameter_audit.report_sha256,
                "every_target_parameter_installed_exactly_once": True,
                "sequential_target_only_replay": True,
                "behavior_review_packet": behavior_packet,
                "decoded_intelligence_review_is_outer_authority": True,
            },
            operation_audit=_semantic_operation_audit(),
        )


class HFCausalSameDepthResourceResolver:
    """Resolve sealed same-family evidence into the generic whole-model stack."""

    resolver_id = HF_CAUSAL_SAME_DEPTH_RESOLVER_ID
    version = 1
    priority = 450
    target_families = ("llama",)
    target_modalities = ("text",)
    method_ids = (HF_CAUSAL_SAME_DEPTH_SEMANTIC_METHOD,)

    def __init__(
        self,
        *,
        evidence_catalog: Sequence[HFCausalCircuitEvidence],
        final_evaluator: Any,
        model_saver: Callable[[Any, Path], None],
        model_loader: Callable[[Path], Any],
        fresh_reload_evaluator: Any,
        saver_id: str,
        loader_id: str,
        ridge: float = 1e-6,
        publication_recorder: Callable[[Any], None] | None = None,
    ) -> None:
        evidence = tuple(value.verify() for value in evidence_catalog)
        if not evidence or len({value.parent_graph_fingerprint for value in evidence}) != len(
            evidence
        ):
            raise HFCausalSemanticCompilerError(
                "Evidence catalog must contain unique parent graph bindings."
            )
        self._evidence = {
            value.parent_graph_fingerprint: value for value in evidence
        }
        self._compiler = HFCausalSameDepthSemanticCompiler(ridge=ridge)
        self._final_evaluator = final_evaluator
        self._model_saver = model_saver
        self._model_loader = model_loader
        self._fresh_reload_evaluator = fresh_reload_evaluator
        self._saver_id = str(saver_id)
        self._loader_id = str(loader_id)
        self._publication_recorder = publication_recorder
        self.resource_fingerprint = canonical_sha256(
            {
                "schema": SCHEMA,
                "resolver": self.resolver_id,
                "evidence": sorted(value.cache_key for value in evidence),
                "compiler": self._compiler.resource_fingerprint,
                "saver_id": self._saver_id,
                "loader_id": self._loader_id,
            }
        )

    def _select(self, target: DynamicTargetSynthesisResult) -> HFCausalCircuitEvidence:
        matches = [
            self._evidence[fingerprint]
            for _name, fingerprint in target.parent_graph_fingerprints
            if fingerprint in self._evidence
        ]
        if len(matches) != 1:
            raise HFCausalSemanticCompilerError(
                "Target must resolve exactly one sealed same-family evidence cache."
            )
        return matches[0].verify()

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            if (
                not isinstance(target, DynamicTargetSynthesisResult)
                or target.graph.family != "llama"
                or target.resource_profile.modalities != ("text",)
                or target.request.fold_method_id
                != HF_CAUSAL_SAME_DEPTH_SEMANTIC_METHOD
            ):
                return False
            evidence = self._select(target)
            layout = build_decoder_layout(target.model)
            _decode_singleton_spans(target, evidence.architecture.layer_count)
            return (
                evidence.architecture.family == "llama"
                and layout.layer_count == evidence.architecture.layer_count
                and layout.embeddings.vocab_size
                == evidence.architecture.vocabulary_size
            )
        except Exception:
            return False

    def resolve(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> SemanticWholeModelFoldResources:
        if not self.supports(target):
            raise HFCausalSemanticCompilerError(
                "Target is outside the same-depth Llama resolver envelope."
            )
        evidence = self._select(target)
        semantic_evidence = SemanticWholeModelFoldEvidence(
            root_fingerprint=evidence.cache_key,
            source=evidence,
        )
        binding = SemanticWholeModelTargetBinding(
            binding_id=(
                "hf-causal-same-depth:" + target.topology_fingerprint[:16]
            ),
            target_family="llama",
            target_modalities=("text",),
            target_adapter_id=target.adapter_id,
            target_adapter_version=target.adapter_version,
            method_id=HF_CAUSAL_SAME_DEPTH_SEMANTIC_METHOD,
            exact_target_unique_parameters=target.graph.unique_parameter_count,
            parent_graph_fingerprints=target.parent_graph_fingerprints,
            child_tokenizer_fingerprint=target.graph.tokenizer_fingerprint,
            target_configuration_sha256=canonical_sha256(
                dict(target.request.configuration)
            ),
            topology_fingerprint=target.topology_fingerprint,
        )
        return SemanticWholeModelFoldResources(
            binding=binding,
            evidence=semantic_evidence,
            compiler=self._compiler,
            final_evaluator=self._final_evaluator,
            model_saver=self._model_saver,
            model_loader=self._model_loader,
            fresh_reload_evaluator=self._fresh_reload_evaluator,
            saver_id=self._saver_id,
            loader_id=self._loader_id,
            stack_adapter_id=self.resolver_id,
            stack_adapter_version=self.version,
            priority=self.priority,
            publication_recorder=self._publication_recorder,
        )


__all__ = [
    "SCHEMA",
    "HF_CAUSAL_SAME_DEPTH_RESOLVER_ID",
    "HF_CAUSAL_SAME_DEPTH_SEMANTIC_METHOD",
    "HFCausalSameDepthResourceResolver",
    "HFCausalSameDepthSemanticCompiler",
    "HFCausalSemanticCompilerError",
]
