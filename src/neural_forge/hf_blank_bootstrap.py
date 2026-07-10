"""Complete forward-only bootstrap of a blank supported decoder target."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from neural_forge.hf_arch_layout import (
    DecoderLayout,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_closed_form_fit import (
    FitMetrics,
    fit_diagonal_norm,
    fit_ridge_linear,
    measure_fit,
)
from neural_forge.hf_feature_fold import (
    FeatureFoldMap,
    build_feature_fold_map,
)
from neural_forge.hf_operator_adapter import (
    AttentionOperators,
    LinearOperator,
    MLPOperators,
    read_attention_operators,
    read_mlp_operators,
    write_attention_operators,
    write_mlp_operators,
)
from neural_forge.hf_transport import (
    Disposition,
    GateDecision,
    RoleEvidence,
    TransportTransactionResult,
    run_transport_transaction,
)
from neural_forge.hf_transport_manifest import (
    DecoderTransportManifest,
    build_decoder_transport_manifest,
)
from neural_forge.hf_transport_math import (
    CoordinateMap,
    build_parent_residual_basis,
    encode_coordinates,
)
from neural_forge.hf_transport_trace import (
    DecoderBlockTrace,
    FlattenedTransportTrace,
    collect_hf_transport_trace,
)
from neural_forge.hf_trace_pooling import pool_hf_transport_trace


class BlankBootstrapError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BootstrapGateMetrics:
    final_state: FitMetrics
    logits: FitMetrics
    top1_agreement: float


@dataclass(frozen=True, slots=True)
class BlankBootstrapResult:
    transaction: TransportTransactionResult
    source_layout: DecoderLayout
    target_layout: DecoderLayout
    residual_coordinates: CoordinateMap
    donor_layer_spans: tuple[tuple[int, ...], ...]
    baseline_gate_metrics: BootstrapGateMetrics
    candidate_gate_metrics: BootstrapGateMetrics
    fit_reports: Mapping[str, Any]


def bootstrap_blank_decoder_models(
    source_model: Any,
    target_model: Any,
    calibration_inputs: Mapping[str, Any] | Any,
    *,
    validation_inputs: Mapping[str, Any] | Any | None = None,
    target_calibration_inputs: Mapping[str, Any] | Any | None = None,
    target_validation_inputs: Mapping[str, Any] | Any | None = None,
    trace_pooling: Literal["token_rows", "mean", "last_valid"] = "token_rows",
    ridge: float = 1e-5,
    min_final_state_relative_improvement: float = 0.0,
    min_logit_relative_improvement: float = 0.0,
    min_top1_agreement_gain: float = 0.0,
    fit_full_lm_head: bool = True,
    max_full_head_fit_elements: int = 10_000_000,
    max_coordinate_rows: int = 4096,
    vocab_projection_chunk_size: int = 2048,
) -> BlankBootstrapResult:
    """Compile source behavior into a blank GPT-2/Llama target without gradients.

    This first proof slice requires one shared token-ID space. Cross-tokenizer
    embedding/head construction is handled separately by ``hf_token_transport``.
    """

    torch = _require_torch()
    ridge = _nonnegative_float(ridge, "ridge")
    for name, value in (
        ("min_final_state_relative_improvement", min_final_state_relative_improvement),
        ("min_logit_relative_improvement", min_logit_relative_improvement),
        ("min_top1_agreement_gain", min_top1_agreement_gain),
    ):
        _nonnegative_float(value, name)
    if not isinstance(fit_full_lm_head, bool):
        raise BlankBootstrapError("fit_full_lm_head must be boolean.")
    if trace_pooling not in {"token_rows", "mean", "last_valid"}:
        raise BlankBootstrapError(
            "trace_pooling must be token_rows, mean, or last_valid."
        )
    if (
        isinstance(max_full_head_fit_elements, bool)
        or not isinstance(max_full_head_fit_elements, int)
        or max_full_head_fit_elements <= 0
    ):
        raise BlankBootstrapError("max_full_head_fit_elements must be positive.")
    for name, value in (
        ("max_coordinate_rows", max_coordinate_rows),
        ("vocab_projection_chunk_size", vocab_projection_chunk_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BlankBootstrapError(f"{name} must be a positive integer.")

    source_layout = build_decoder_layout(source_model)
    target_layout = build_decoder_layout(target_model)
    if max_coordinate_rows < target_layout.hidden_size:
        raise BlankBootstrapError(
            "max_coordinate_rows cannot be smaller than target hidden size."
        )
    manifest = build_decoder_transport_manifest(target_model, layout=target_layout)
    _validate_bootstrap_contract(
        source_model,
        target_model,
        source_layout,
        target_layout,
    )
    donor_layer_spans = _contiguous_depth_spans(
        source_layout.layer_count,
        target_layout.layer_count,
    )
    validation_values = (
        calibration_inputs if validation_inputs is None else validation_inputs
    )
    target_calibration_values = (
        calibration_inputs
        if target_calibration_inputs is None
        else target_calibration_inputs
    )
    if target_validation_inputs is not None:
        target_validation_values = target_validation_inputs
    elif target_calibration_inputs is None:
        target_validation_values = validation_values
    elif validation_inputs is None:
        target_validation_values = target_calibration_values
    else:
        raise BlankBootstrapError(
            "target_validation_inputs is required when separate source/target "
            "calibration inputs and a source validation split are supplied."
        )

    source_calibration = _trace_rows(
        source_model,
        calibration_inputs,
        trace_pooling,
    )
    source_validation = _trace_rows(
        source_model,
        validation_values,
        trace_pooling,
    )
    baseline_validation = _trace_rows(
        target_model,
        target_validation_values,
        trace_pooling,
    )
    _require_row_alignment(source_validation, baseline_validation, "validation")
    residual_rows = _residual_basis_rows(
        torch,
        source_calibration,
        max_coordinate_rows,
    )
    residual_coordinates = build_parent_residual_basis(
        residual_rows,
        target_layout.hidden_size,
        center=False,
    )
    desired_validation_final = encode_coordinates(
        source_validation.blocks[-1].block_output,
        residual_coordinates,
    )
    baseline_gate_metrics = _gate_metrics(
        baseline_validation,
        desired_validation_final,
        source_validation.logits,
    )
    fit_reports: dict[str, Any] = {}
    candidate_gate_metrics: BootstrapGateMetrics | None = None

    def mutate(coverage) -> None:
        _initialize_embedding_and_head(
            torch,
            source_model,
            target_model,
            source_layout,
            target_layout,
            residual_coordinates,
            source_calibration,
            target_calibration_values,
            trace_pooling,
            vocab_projection_chunk_size,
            fit_reports,
        )
        for target_index, source_span in enumerate(donor_layer_spans):
            _fit_target_block(
                torch,
                source_model,
                target_model,
                source_layout,
                target_layout,
                source_calibration,
                target_calibration_values,
                trace_pooling,
                target_index,
                source_span,
                residual_coordinates,
                ridge,
                max_coordinate_rows,
                fit_reports,
            )
        _fit_final_interface(
            torch,
            source_model,
            target_model,
            source_layout,
            target_layout,
            source_calibration,
            target_calibration_values,
            trace_pooling,
            residual_coordinates,
            ridge,
            fit_full_lm_head,
            max_full_head_fit_elements,
            fit_reports,
        )
        _record_complete_evidence(coverage, manifest, fit_reports)

    def gate(_coverage_report) -> GateDecision:
        nonlocal candidate_gate_metrics
        candidate_validation = _trace_rows(
            target_model,
            target_validation_values,
            trace_pooling,
        )
        candidate_gate_metrics = _gate_metrics(
            candidate_validation,
            desired_validation_final,
            source_validation.logits,
        )
        final_gain = _relative_improvement(
            baseline_gate_metrics.final_state.relative_rmse,
            candidate_gate_metrics.final_state.relative_rmse,
        )
        logit_gain = _relative_improvement(
            baseline_gate_metrics.logits.relative_rmse,
            candidate_gate_metrics.logits.relative_rmse,
        )
        top1_gain = (
            candidate_gate_metrics.top1_agreement - baseline_gate_metrics.top1_agreement
        )
        accepted = (
            final_gain >= float(min_final_state_relative_improvement)
            and logit_gain >= float(min_logit_relative_improvement)
            and top1_gain >= float(min_top1_agreement_gain)
        )
        return GateDecision(
            accepted=accepted,
            reason=(
                "heldout-source-anchors-improved"
                if accepted
                else "heldout-source-anchors-regressed"
            ),
            evidence={
                "final_state_relative_improvement": final_gain,
                "logit_relative_improvement": logit_gain,
                "top1_agreement_gain": top1_gain,
                "baseline": _gate_metrics_dict(baseline_gate_metrics),
                "candidate": _gate_metrics_dict(candidate_gate_metrics),
            },
        )

    transaction = run_transport_transaction(
        plan=manifest.plan,
        targets={"model": target_model},
        mutation=mutate,
        gate=gate,
    )
    if candidate_gate_metrics is None:  # pragma: no cover - gate contract
        raise RuntimeError("Blank bootstrap gate did not execute.")
    return BlankBootstrapResult(
        transaction=transaction,
        source_layout=source_layout,
        target_layout=target_layout,
        residual_coordinates=residual_coordinates,
        donor_layer_spans=donor_layer_spans,
        baseline_gate_metrics=baseline_gate_metrics,
        candidate_gate_metrics=candidate_gate_metrics,
        fit_reports=fit_reports,
    )


def _validate_bootstrap_contract(
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_layout: DecoderLayout,
) -> None:
    if target_layout.layer_count > source_layout.layer_count:
        raise BlankBootstrapError(
            "Compression bootstrap cannot have more target than source layers."
        )
    if target_layout.hidden_size > source_layout.hidden_size:
        raise BlankBootstrapError(
            "Compression bootstrap target hidden size cannot exceed source hidden size."
        )
    if target_layout.intermediate_size > source_layout.intermediate_size:
        raise BlankBootstrapError(
            "Compression bootstrap target MLP width cannot exceed source MLP width."
        )
    source_embedding = source_model.get_input_embeddings()
    target_embedding = target_model.get_input_embeddings()
    source_head = source_model.get_output_embeddings()
    target_head = target_model.get_output_embeddings()
    if any(
        item is None
        for item in (source_embedding, target_embedding, source_head, target_head)
    ):
        raise BlankBootstrapError("Both models must expose embeddings and LM heads.")
    source_vocab = int(source_embedding.weight.shape[0])
    target_vocab = int(target_embedding.weight.shape[0])
    if source_vocab != target_vocab or int(source_head.weight.shape[0]) != target_vocab:
        raise BlankBootstrapError(
            "This bootstrap slice requires one shared token-ID vocabulary size."
        )
    target_tied = _same_storage(target_embedding.weight, target_head.weight)
    source_tied = _same_storage(source_embedding.weight, source_head.weight)
    if target_tied and not source_tied:
        raise BlankBootstrapError(
            "An untied source embedding/head cannot compile into a tied blank target."
        )


def _trace_rows(
    model: Any,
    inputs: Mapping[str, Any] | Any,
    pooling: Literal["token_rows", "mean", "last_valid"],
) -> Any:
    trace = collect_hf_transport_trace(model, inputs)
    if pooling == "token_rows":
        return trace.flattened_valid_rows()
    return pool_hf_transport_trace(
        trace,
        pooling=pooling,
        logits_policy="last_valid",
    )


def _require_row_alignment(left: Any, right: Any, name: str) -> None:
    left_rows = int(left.blocks[0].block_input.shape[0])
    right_rows = int(right.blocks[0].block_input.shape[0])
    if left_rows != right_rows:
        raise BlankBootstrapError(
            f"Source and target {name} traces must have equal row counts; "
            f"got {left_rows} and {right_rows}."
        )


def _residual_basis_rows(
    torch: Any,
    trace: FlattenedTransportTrace,
    max_rows: int,
) -> Any:
    values = []
    for block in trace.blocks:
        values.extend([block.block_input, block.block_output])
    values.append(trace.final_norm_output)
    quotas = _row_quotas([int(value.shape[0]) for value in values], max_rows)
    sampled = [
        _evenly_sample_rows(torch, value, quota)
        for value, quota in zip(values, quotas, strict=True)
        if quota > 0
    ]
    return torch.cat(sampled, dim=0)


def _row_quotas(lengths: Sequence[int], max_rows: int) -> tuple[int, ...]:
    total = sum(lengths)
    selected = min(total, max_rows)
    if selected == total:
        return tuple(lengths)
    torch = _require_torch()
    indices = (
        torch.linspace(0, total - 1, steps=selected)
        .round()
        .to(torch.long)
    )
    quotas = []
    offset = 0
    for length in lengths:
        quotas.append(int(((indices >= offset) & (indices < offset + length)).sum()))
        offset += length
    return tuple(quotas)


def _evenly_sample_rows(torch: Any, rows: Any, count: int) -> Any:
    if count >= int(rows.shape[0]):
        return rows
    indices = (
        torch.linspace(0, int(rows.shape[0]) - 1, steps=count)
        .round()
        .to(torch.long)
    )
    return rows.index_select(0, indices)


def _sample_concatenated_rows(
    torch: Any,
    values: Sequence[Any],
    max_rows: int,
) -> Any:
    quotas = _row_quotas([int(value.shape[0]) for value in values], max_rows)
    sampled = [
        _evenly_sample_rows(torch, value, quota)
        for value, quota in zip(values, quotas, strict=True)
        if quota > 0
    ]
    return torch.cat(sampled, dim=0)


def _copy_projected_rows(
    torch: Any,
    source_weight: Any,
    target_weight: Any,
    coordinates: CoordinateMap,
    chunk_size: int,
) -> None:
    if int(source_weight.shape[0]) != int(target_weight.shape[0]):
        raise BlankBootstrapError(
            "Source and target vocabulary operators must have equal row counts."
        )
    expected = (int(source_weight.shape[0]), coordinates.target_dim)
    if tuple(target_weight.shape) != expected:
        raise BlankBootstrapError(
            f"Target vocabulary operator must have shape {expected}."
        )
    with torch.no_grad():
        for start in range(0, int(source_weight.shape[0]), chunk_size):
            end = min(int(source_weight.shape[0]), start + chunk_size)
            projected = encode_coordinates(source_weight[start:end], coordinates)
            target_weight[start:end].copy_(projected.to(target_weight))


def _initialize_embedding_and_head(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_layout: DecoderLayout,
    residual_coordinates: CoordinateMap,
    source_trace: FlattenedTransportTrace,
    target_calibration_inputs: Mapping[str, Any] | Any,
    trace_pooling: Literal["token_rows", "mean", "last_valid"],
    vocab_projection_chunk_size: int,
    reports: dict[str, Any],
) -> None:
    source_embedding = source_model.get_input_embeddings()
    target_embedding = target_model.get_input_embeddings()
    source_head = source_model.get_output_embeddings()
    target_head = target_model.get_output_embeddings()
    _copy_projected_rows(
        torch,
        source_embedding.weight,
        target_embedding.weight,
        residual_coordinates,
        vocab_projection_chunk_size,
    )
    _copy_projected_rows(
        torch,
        source_head.weight,
        target_head.weight,
        residual_coordinates,
        vocab_projection_chunk_size,
    )
    with torch.no_grad():
        source_bias = getattr(source_head, "bias", None)
        target_bias = getattr(target_head, "bias", None)
        if target_bias is not None:
            target_bias.zero_()
            if source_bias is not None:
                target_bias.copy_(source_bias.to(target_bias))
    reports["embeddings.token"] = {
        "method": "chunked-parent-residual-basis-projection",
        "shape": list(target_embedding.weight.shape),
        "chunk_rows": vocab_projection_chunk_size,
    }
    reports["lm_head"] = {
        "method": "chunked-parent-residual-basis-projection",
        "shape": list(target_head.weight.shape),
        "chunk_rows": vocab_projection_chunk_size,
    }
    if target_layout.position.kind == "learned_absolute":
        _initialize_learned_positions(
            torch,
            source_model,
            target_model,
            source_layout,
            target_layout,
            source_trace,
            target_calibration_inputs,
            trace_pooling,
            residual_coordinates,
            reports,
        )


def _initialize_learned_positions(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_layout: DecoderLayout,
    source_trace: FlattenedTransportTrace,
    target_calibration_inputs: Mapping[str, Any] | Any,
    trace_pooling: Literal["token_rows", "mean", "last_valid"],
    residual_coordinates: CoordinateMap,
    reports: dict[str, Any],
) -> None:
    assert target_layout.embeddings.position_path is not None
    target_position = resolve_module(
        target_model, target_layout.embeddings.position_path
    )
    with torch.no_grad():
        target_position.weight.zero_()
        if source_layout.position.kind == "learned_absolute":
            assert source_layout.embeddings.position_path is not None
            source_position = resolve_module(
                source_model,
                source_layout.embeddings.position_path,
            )
            rows = min(source_position.weight.shape[0], target_position.weight.shape[0])
            projected = encode_coordinates(
                source_position.weight[:rows],
                residual_coordinates,
            )
            target_position.weight[:rows].copy_(projected.to(target_position.weight))
            method = "source-position-basis-projection"
        else:
            if trace_pooling != "token_rows":
                raise BlankBootstrapError(
                    "A learned-absolute target cannot infer per-position rows from "
                    "prompt-pooled rotary source traces."
                )
            input_ids = _input_ids(target_calibration_inputs).detach().cpu()
            target_tokens = (
                target_model.get_input_embeddings().weight.detach().cpu()[input_ids]
            )
            desired = encode_coordinates(
                source_trace.blocks[0].block_input,
                residual_coordinates,
            )
            desired = desired.reshape(*input_ids.shape, -1)
            residual = desired - target_tokens.to(torch.float64)
            for position in range(
                min(input_ids.shape[1], target_position.weight.shape[0])
            ):
                target_position.weight[position].copy_(
                    residual[:, position].mean(dim=0).to(target_position.weight)
                )
            method = "calibration-position-residual-mean"
    reports["position.learned_absolute"] = {
        "method": method,
        "rows": int(target_position.weight.shape[0]),
    }


def _fit_target_block(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_layout: DecoderLayout,
    source_trace: FlattenedTransportTrace,
    target_calibration_inputs: Mapping[str, Any] | Any,
    trace_pooling: Literal["token_rows", "mean", "last_valid"],
    target_index: int,
    source_span: tuple[int, ...],
    residual_coordinates: CoordinateMap,
    ridge: float,
    max_coordinate_rows: int,
    reports: dict[str, Any],
) -> None:
    target_block_layout = target_layout.blocks[target_index]
    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    _require_row_alignment(source_trace, current, "calibration")
    target_block = current.blocks[target_index]
    source_blocks = [source_trace.blocks[index] for index in source_span]

    desired_attention_norm = _mean_projected(
        torch,
        source_blocks,
        "pre_attention_norm_output",
        residual_coordinates,
    )
    attention_norm_fit = fit_diagonal_norm(
        target_block.block_input,
        desired_attention_norm,
        kind=target_block_layout.attention_input_norm.kind,
        epsilon=target_block_layout.attention_input_norm.epsilon,
        has_bias=target_block_layout.attention_input_norm.has_bias,
        ridge=ridge,
    )
    _write_norm_fit(
        target_model,
        target_block_layout.attention_input_norm.path,
        attention_norm_fit,
    )
    reports[f"block.{target_index}.attention.norm"] = _fit_summary(attention_norm_fit)

    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    target_block = current.blocks[target_index]
    source_q_rows, source_k_rows, source_v_rows = (
        _effective_source_attention_projection_rows(
            torch,
            source_model,
            source_layout,
            source_blocks,
            source_span,
        )
    )
    q_coordinates, k_coordinates, v_coordinates = _attention_feature_coordinates(
        torch,
        source_q_rows,
        source_k_rows,
        source_v_rows,
        target_block_layout,
        max_coordinate_rows,
    )
    reports[f"block.{target_index}.attention.feature_folds"] = {
        "query": q_coordinates.report.as_dict(),
        "key": k_coordinates.report.as_dict(),
        "value": v_coordinates.report.as_dict(),
        "source_query_head_norms": [
            source_layout.blocks[index].attention.query_norm.path
            for index in source_span
            if source_layout.blocks[index].attention.query_norm is not None
        ],
        "source_key_head_norms": [
            source_layout.blocks[index].attention.key_norm.path
            for index in source_span
            if source_layout.blocks[index].attention.key_norm is not None
        ],
    }
    desired_q = _mean_encoded_rows(torch, source_q_rows, q_coordinates)
    desired_k = _mean_encoded_rows(torch, source_k_rows, k_coordinates)
    desired_v = _mean_encoded_rows(torch, source_v_rows, v_coordinates)
    q_fit = fit_ridge_linear(
        target_block.pre_attention_norm_output,
        desired_q,
        ridge=ridge,
        fit_bias=target_block_layout.attention.projection_bias,
    )
    k_fit = fit_ridge_linear(
        target_block.pre_attention_norm_output,
        desired_k,
        ridge=ridge,
        fit_bias=target_block_layout.attention.projection_bias,
    )
    v_fit = fit_ridge_linear(
        target_block.pre_attention_norm_output,
        desired_v,
        ridge=ridge,
        fit_bias=target_block_layout.attention.projection_bias,
    )
    target_attention = read_attention_operators(
        target_model,
        target_block_layout.attention,
    )
    write_attention_operators(
        target_model,
        target_block_layout.attention,
        AttentionOperators(
            query=_operator_from_fit(target_attention.query, q_fit),
            key=_operator_from_fit(target_attention.key, k_fit),
            value=_operator_from_fit(target_attention.value, v_fit),
            output=target_attention.output,
        ),
    )
    if target_block_layout.attention.projection_style == "fused_qkv":
        reports[f"block.{target_index}.attention.qkv"] = {
            "method": "joint-closed-form-qkv-producer-fit",
            "query": _fit_summary(q_fit),
            "key": _fit_summary(k_fit),
            "value": _fit_summary(v_fit),
        }
    else:
        for name, fit in (("query", q_fit), ("key", k_fit), ("value", v_fit)):
            reports[f"block.{target_index}.attention.{name}"] = _fit_summary(fit)

    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    target_block = current.blocks[target_index]
    desired_attention_delta = _sum_projected(
        torch,
        source_blocks,
        "attention_delta",
        residual_coordinates,
    )
    output_fit = fit_ridge_linear(
        target_block.attention_output_projection_input,
        desired_attention_delta,
        ridge=ridge,
        fit_bias=target_block_layout.attention.projection_bias,
    )
    target_attention = read_attention_operators(
        target_model,
        target_block_layout.attention,
    )
    write_attention_operators(
        target_model,
        target_block_layout.attention,
        AttentionOperators(
            query=target_attention.query,
            key=target_attention.key,
            value=target_attention.value,
            output=_operator_from_fit(target_attention.output, output_fit),
        ),
    )
    reports[f"block.{target_index}.attention.output"] = _fit_summary(output_fit)

    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    target_block = current.blocks[target_index]
    desired_mlp_norm = _mean_projected(
        torch,
        source_blocks,
        "pre_mlp_norm_output",
        residual_coordinates,
    )
    mlp_norm_fit = fit_diagonal_norm(
        target_block.post_attention_residual,
        desired_mlp_norm,
        kind=target_block_layout.mlp_input_norm.kind,
        epsilon=target_block_layout.mlp_input_norm.epsilon,
        has_bias=target_block_layout.mlp_input_norm.has_bias,
        ridge=ridge,
    )
    _write_norm_fit(
        target_model,
        target_block_layout.mlp_input_norm.path,
        mlp_norm_fit,
    )
    reports[f"block.{target_index}.mlp.norm"] = _fit_summary(mlp_norm_fit)

    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    target_block = current.blocks[target_index]
    mlp_feature_rows = _source_mlp_feature_rows(
        torch,
        source_blocks,
        max_coordinate_rows,
    )
    mlp_coordinates = build_feature_fold_map(
        mlp_feature_rows,
        target_block_layout.mlp.intermediate_size,
    )
    reports[f"block.{target_index}.mlp.feature_fold"] = (
        mlp_coordinates.report.as_dict()
    )
    target_mlp = read_mlp_operators(target_model, target_block_layout)
    if target_block_layout.mlp.kind == "dense":
        desired_input = _mean_source_mlp_producer(
            torch,
            source_blocks,
            "dense",
            mlp_coordinates,
        )
        input_fit = fit_ridge_linear(
            target_block.pre_mlp_norm_output,
            desired_input,
            ridge=ridge,
            fit_bias=target_block_layout.mlp.projection_bias,
        )
        assert target_mlp.input is not None
        target_mlp = MLPOperators(
            kind="dense",
            input=_operator_from_fit(target_mlp.input, input_fit),
            gate=None,
            up=None,
            down=target_mlp.down,
        )
        reports[f"block.{target_index}.mlp.input"] = _fit_summary(input_fit)
    else:
        desired_gate = _mean_source_mlp_producer(
            torch,
            source_blocks,
            "gate",
            mlp_coordinates,
        )
        desired_up = _mean_source_mlp_producer(
            torch,
            source_blocks,
            "up",
            mlp_coordinates,
        )
        gate_fit = fit_ridge_linear(
            target_block.pre_mlp_norm_output,
            desired_gate,
            ridge=ridge,
            fit_bias=target_block_layout.mlp.projection_bias,
        )
        up_fit = fit_ridge_linear(
            target_block.pre_mlp_norm_output,
            desired_up,
            ridge=ridge,
            fit_bias=target_block_layout.mlp.projection_bias,
        )
        assert target_mlp.gate is not None and target_mlp.up is not None
        target_mlp = MLPOperators(
            kind="gated",
            input=None,
            gate=_operator_from_fit(target_mlp.gate, gate_fit),
            up=_operator_from_fit(target_mlp.up, up_fit),
            down=target_mlp.down,
        )
        reports[f"block.{target_index}.mlp.gate"] = _fit_summary(gate_fit)
        reports[f"block.{target_index}.mlp.up"] = _fit_summary(up_fit)
    write_mlp_operators(target_model, target_block_layout, target_mlp)

    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    target_block = current.blocks[target_index]
    desired_endpoint = encode_coordinates(
        source_blocks[-1].block_output,
        residual_coordinates,
    )
    desired_mlp_delta = desired_endpoint - target_block.post_attention_residual
    down_fit = fit_ridge_linear(
        target_block.down_projection_input,
        desired_mlp_delta,
        ridge=ridge,
        fit_bias=target_block_layout.mlp.projection_bias,
    )
    target_mlp = read_mlp_operators(target_model, target_block_layout)
    target_mlp = MLPOperators(
        kind=target_mlp.kind,
        input=target_mlp.input,
        gate=target_mlp.gate,
        up=target_mlp.up,
        down=_operator_from_fit(target_mlp.down, down_fit),
    )
    write_mlp_operators(target_model, target_block_layout, target_mlp)
    reports[f"block.{target_index}.mlp.down"] = _fit_summary(down_fit)
    reports[f"block.{target_index}.span"] = {
        "source_layers": list(source_span),
        "ordered_contiguous": True,
    }


def _fit_final_interface(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_layout: DecoderLayout,
    source_trace: FlattenedTransportTrace,
    target_calibration_inputs: Mapping[str, Any] | Any,
    trace_pooling: Literal["token_rows", "mean", "last_valid"],
    residual_coordinates: CoordinateMap,
    ridge: float,
    fit_full_lm_head: bool,
    max_full_head_fit_elements: int,
    reports: dict[str, Any],
) -> None:
    del source_model, source_layout
    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    target_norm = resolve_module(target_model, target_layout.final_norm.path)
    norm_input = _norm_input_from_module_trace(current)
    desired_norm = encode_coordinates(
        source_trace.final_norm_output,
        residual_coordinates,
    )
    norm_fit = fit_diagonal_norm(
        norm_input,
        desired_norm,
        kind=target_layout.final_norm.kind,
        epsilon=target_layout.final_norm.epsilon,
        has_bias=target_layout.final_norm.has_bias,
        ridge=ridge,
    )
    _write_norm_fit(target_model, target_layout.final_norm.path, norm_fit)
    reports["decoder.final_norm"] = _fit_summary(norm_fit)

    current = _trace_rows(target_model, target_calibration_inputs, trace_pooling)
    head = target_model.get_output_embeddings()
    if fit_full_lm_head:
        elements = int(current.final_norm_output.shape[0]) * int(
            source_trace.logits.shape[1]
        )
        if elements > max_full_head_fit_elements:
            reports["lm_head"]["full_logit_fit"] = {
                "applied": False,
                "reason": "max_full_head_fit_elements",
                "elements": elements,
            }
        elif target_layout.tie_word_embeddings:
            reports["lm_head"]["full_logit_fit"] = {
                "applied": False,
                "reason": "target_head_is_tied",
                "elements": elements,
            }
        else:
            head_fit = fit_ridge_linear(
                current.final_norm_output,
                source_trace.logits,
                ridge=ridge,
                fit_bias=getattr(head, "bias", None) is not None,
            )
            with torch.no_grad():
                head.weight.copy_(head_fit.weight.to(head.weight))
                if head.bias is not None and head_fit.bias is not None:
                    head.bias.copy_(head_fit.bias.to(head.bias))
            reports["lm_head"]["full_logit_fit"] = {
                "applied": True,
                **_fit_summary(head_fit),
            }
    del target_norm


def _record_complete_evidence(
    coverage: Any,
    manifest: DecoderTransportManifest,
    reports: Mapping[str, Any],
) -> None:
    for role in manifest.plan.required_roles:
        if role.allowed_dispositions == (Disposition.TARGET_NATIVE,):
            coverage.record(
                RoleEvidence(
                    role=role.name,
                    disposition=Disposition.TARGET_NATIVE,
                    method="validated-target-native-operator",
                    target_tensors=role.target_tensors,
                    metadata={"description": role.description},
                )
            )
            continue
        evidence = reports.get(role.name)
        if evidence is None:
            raise BlankBootstrapError(
                f"No closed-form fit report exists for required role {role.name!r}."
            )
        coverage.record(
            RoleEvidence(
                role=role.name,
                disposition=Disposition.CLOSED_FORM_FITTED,
                method=str(evidence.get("method", "closed-form-parent-anchor-fit")),
                target_tensors=role.target_tensors,
                source_artifacts=(f"source-trace:{role.name}",),
                metadata=evidence,
            )
        )


def _attention_feature_coordinates(
    torch: Any,
    query_rows: Sequence[Any],
    key_rows: Sequence[Any],
    value_rows: Sequence[Any],
    target_block_layout: Any,
    max_coordinate_rows: int,
) -> tuple[FeatureFoldMap, FeatureFoldMap, FeatureFoldMap]:
    q_rows = _sample_concatenated_rows(
        torch,
        query_rows,
        max_coordinate_rows,
    )
    k_rows = _sample_concatenated_rows(
        torch,
        key_rows,
        max_coordinate_rows,
    )
    v_rows = _sample_concatenated_rows(
        torch,
        value_rows,
        max_coordinate_rows,
    )
    q_width = target_block_layout.attention.query_width
    k_width = target_block_layout.attention.key_value_width
    if q_rows.shape[1] == k_rows.shape[1] and q_width == k_width:
        shared = build_feature_fold_map(
            [q_rows, k_rows],
            q_width,
        )
        q_coordinates = shared
        k_coordinates = shared
    else:
        q_coordinates = build_feature_fold_map(
            q_rows,
            q_width,
        )
        k_coordinates = build_feature_fold_map(
            k_rows,
            k_width,
        )
    v_coordinates = build_feature_fold_map(v_rows, k_width)
    return q_coordinates, k_coordinates, v_coordinates


def _effective_source_attention_projection_rows(
    torch: Any,
    source_model: Any,
    source_layout: DecoderLayout,
    source_blocks: Sequence[DecoderBlockTrace],
    source_span: Sequence[int],
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    query_rows = []
    key_rows = []
    value_rows = []
    for source_index, block in zip(source_span, source_blocks, strict=True):
        attention = source_layout.blocks[source_index].attention
        query = block.query_projection_output
        key = block.key_projection_output
        if attention.query_norm is not None:
            query = _apply_head_norm(
                torch,
                query,
                source_model,
                attention.query_norm,
                attention.query_heads,
                attention.head_dim,
            )
        if attention.key_norm is not None:
            key = _apply_head_norm(
                torch,
                key,
                source_model,
                attention.key_norm,
                attention.key_value_heads,
                attention.head_dim,
            )
        query_rows.append(query)
        key_rows.append(key)
        value_rows.append(block.value_projection_output)
    return tuple(query_rows), tuple(key_rows), tuple(value_rows)


def _apply_head_norm(
    torch: Any,
    rows: Any,
    model: Any,
    layout: Any,
    head_count: int,
    head_dim: int,
) -> Any:
    if layout.kind != "rms_norm" or layout.has_bias:
        raise BlankBootstrapError(
            "Only bias-free per-head RMSNorm is supported in source attention."
        )
    module = resolve_module(model, layout.path)
    weight = module.weight.detach().to(device="cpu", dtype=torch.float64)
    if tuple(weight.shape) != (head_dim,):
        raise BlankBootstrapError(
            f"Per-head norm {layout.path!r} must have shape ({head_dim},)."
        )
    values = rows.detach().to(device="cpu", dtype=torch.float64)
    expected_width = head_count * head_dim
    if int(values.shape[1]) != expected_width:
        raise BlankBootstrapError(
            f"Per-head norm input must have width {expected_width}."
        )
    heads = values.reshape(values.shape[0], head_count, head_dim)
    normalized = heads * torch.rsqrt(
        heads.square().mean(dim=-1, keepdim=True) + float(layout.epsilon)
    )
    result = (normalized * weight).reshape(values.shape[0], expected_width)
    if not bool(torch.isfinite(result).all()):
        raise BlankBootstrapError("Per-head RMSNorm produced non-finite rows.")
    return result.contiguous()


def _source_mlp_feature_rows(
    torch: Any,
    source_blocks: Sequence[DecoderBlockTrace],
    max_rows: int,
) -> tuple[Any, ...]:
    fields = (
        "down_projection_input",
        "mlp_input_projection_output",
        "mlp_gate_projection_output",
        "mlp_up_projection_output",
    )
    matrices = []
    for field in fields:
        values = [getattr(block, field) for block in source_blocks]
        present = [value for value in values if value is not None]
        if present:
            matrices.append(
                _sample_concatenated_rows(torch, present, max_rows)
            )
    if not matrices:
        raise BlankBootstrapError("Source trace has no MLP feature rows to fold.")
    return tuple(matrices)


def _mean_source_mlp_producer(
    torch: Any,
    source_blocks: Sequence[DecoderBlockTrace],
    target_role: str,
    coordinates: Any,
) -> Any:
    projected = []
    for block in source_blocks:
        if target_role == "dense":
            source = (
                block.mlp_input_projection_output
                if block.mlp_input_projection_output is not None
                else block.mlp_gate_projection_output
            )
        elif target_role == "gate":
            source = (
                block.mlp_gate_projection_output
                if block.mlp_gate_projection_output is not None
                else block.mlp_input_projection_output
            )
        elif target_role == "up":
            source = (
                block.mlp_up_projection_output
                if block.mlp_up_projection_output is not None
                else block.mlp_input_projection_output
            )
        else:  # pragma: no cover - private call contract
            raise AssertionError(f"Unknown MLP producer role: {target_role}")
        if source is None:
            raise BlankBootstrapError(
                f"Source trace cannot provide MLP producer role {target_role!r}."
            )
        projected.append(_encode_rows(source, coordinates))
    return torch.stack(projected, dim=0).mean(dim=0)


def _mean_projected(
    torch: Any,
    blocks: Sequence[DecoderBlockTrace],
    field: str,
    coordinates: Any,
) -> Any:
    values = [_encode_rows(getattr(block, field), coordinates) for block in blocks]
    return torch.stack(values, dim=0).mean(dim=0)


def _mean_encoded_rows(
    torch: Any,
    rows: Sequence[Any],
    coordinates: Any,
) -> Any:
    values = [_encode_rows(value, coordinates) for value in rows]
    return torch.stack(values, dim=0).mean(dim=0)


def _sum_projected(
    torch: Any,
    blocks: Sequence[DecoderBlockTrace],
    field: str,
    coordinates: Any,
) -> Any:
    values = [_encode_rows(getattr(block, field), coordinates) for block in blocks]
    return torch.stack(values, dim=0).sum(dim=0)


def _encode_rows(rows: Any, coordinates: Any) -> Any:
    if isinstance(coordinates, FeatureFoldMap):
        return coordinates.encode(rows)
    return encode_coordinates(rows, coordinates)


def _operator_from_fit(operator: LinearOperator, fit: Any) -> LinearOperator:
    return LinearOperator(
        path=operator.path,
        weight=fit.weight,
        bias=fit.bias,
    )


def _write_norm_fit(model: Any, path: str, fit: Any) -> None:
    torch = _require_torch()
    module = resolve_module(model, path)
    with torch.no_grad():
        module.weight.copy_(fit.weight.to(module.weight))
        module_bias = getattr(module, "bias", None)
        if (module_bias is None) != (fit.bias is None):
            raise BlankBootstrapError(f"Norm bias contract changed for {path!r}.")
        if module_bias is not None:
            module_bias.copy_(fit.bias.to(module_bias))


def _norm_input_from_module_trace(trace: FlattenedTransportTrace) -> Any:
    return trace.blocks[-1].block_output


def _fit_summary(fit: Any) -> dict[str, Any]:
    metrics = fit.train_metrics
    return {
        "method": "closed-form-parent-anchor-fit",
        "ridge": float(fit.ridge),
        "rows": metrics.rows,
        "relative_rmse": metrics.relative_rmse,
        "mean_cosine_similarity": metrics.mean_cosine_similarity,
        "finite": metrics.finite,
    }


def _gate_metrics(
    trace: FlattenedTransportTrace,
    desired_final_state: Any,
    desired_logits: Any,
) -> BootstrapGateMetrics:
    final_metrics = measure_fit(trace.blocks[-1].block_output, desired_final_state)
    logit_metrics = measure_fit(trace.logits, desired_logits)
    top1 = _top1_agreement(trace.logits, desired_logits)
    return BootstrapGateMetrics(
        final_state=final_metrics,
        logits=logit_metrics,
        top1_agreement=top1,
    )


def _top1_agreement(left: Any, right: Any) -> float:
    torch = _require_torch()
    if tuple(left.shape) != tuple(right.shape):
        raise BlankBootstrapError("Top-1 logit tensors must have equal shapes.")
    return float((left.argmax(dim=-1) == right.argmax(dim=-1)).to(torch.float64).mean())


def _gate_metrics_dict(metrics: BootstrapGateMetrics) -> dict[str, Any]:
    return {
        "final_state_relative_rmse": metrics.final_state.relative_rmse,
        "final_state_mean_cosine_similarity": (
            metrics.final_state.mean_cosine_similarity
        ),
        "logit_relative_rmse": metrics.logits.relative_rmse,
        "logit_mean_cosine_similarity": metrics.logits.mean_cosine_similarity,
        "top1_agreement": metrics.top1_agreement,
    }


def _relative_improvement(before: float, after: float) -> float:
    denominator = max(abs(float(before)), 1e-12)
    return (float(before) - float(after)) / denominator


def _contiguous_depth_spans(
    source_layers: int,
    target_layers: int,
) -> tuple[tuple[int, ...], ...]:
    spans = []
    for target_index in range(target_layers):
        start = (target_index * source_layers) // target_layers
        end = ((target_index + 1) * source_layers) // target_layers
        span = tuple(range(start, end))
        if not span:
            raise BlankBootstrapError("Depth planning produced an empty source span.")
        spans.append(span)
    flattened = tuple(index for span in spans for index in span)
    if flattened != tuple(range(source_layers)):
        raise BlankBootstrapError(
            "Depth planning must cover every source layer exactly once."
        )
    return tuple(spans)


def _input_ids(inputs: Mapping[str, Any] | Any) -> Any:
    if isinstance(inputs, Mapping):
        value = inputs.get("input_ids")
    else:
        value = inputs
    if value is None:
        raise BlankBootstrapError("Calibration inputs must contain input_ids.")
    return value


def _same_storage(left: Any, right: Any) -> bool:
    if left is right:
        return True
    return bool(
        left.device == right.device
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and int(left.data_ptr()) == int(right.data_ptr())
    )


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BlankBootstrapError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BlankBootstrapError(f"{name} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise BlankBootstrapError(f"{name} must be finite and non-negative.")
    return result


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for blank-target bootstrap.") from exc
    return torch


__all__ = [
    "BlankBootstrapError",
    "BlankBootstrapResult",
    "BootstrapGateMetrics",
    "bootstrap_blank_decoder_models",
]
