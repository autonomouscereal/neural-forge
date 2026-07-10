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
from neural_forge.hf_attention_fold import build_gqa_attention_fold_maps
from neural_forge.hf_closed_form_fit import (
    FitMetrics,
    fit_diagonal_norm,
    fit_prior_preserving_ridge_linear,
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
from neural_forge.hf_targeted_trace import collect_hf_targeted_trace
from neural_forge.hf_token_align import StrictTokenTransportProfile
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
    transport_linear_operator,
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
    residual_coordinates: Any
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
    token_transport_profile: StrictTokenTransportProfile | None = None,
    cross_tokenizer_mode: bool = False,
    trace_pooling: Literal["token_rows", "mean", "last_valid"] = "token_rows",
    residual_map_method: Literal[
        "sparse_fold",
        "joint_sparse_fold",
        "pca",
    ] = "sparse_fold",
    ridge: float = 1e-5,
    max_operator_delta_ratio: float = 4.0,
    min_final_state_relative_improvement: float = 0.0,
    min_logit_relative_improvement: float = 0.0,
    min_top1_agreement_gain: float = 0.0,
    fit_full_lm_head: bool = True,
    max_full_head_fit_elements: int = 10_000_000,
    max_coordinate_rows: int = 4096,
    vocab_projection_chunk_size: int = 2048,
    vocab_row_norm_repair: Literal["none", "source_l2"] = "none",
) -> BlankBootstrapResult:
    """Compile source behavior into a blank GPT-2/Llama target without gradients."""

    torch = _require_torch()
    ridge = _nonnegative_float(ridge, "ridge")
    max_operator_delta_ratio = _nonnegative_float(
        max_operator_delta_ratio,
        "max_operator_delta_ratio",
    )
    if max_operator_delta_ratio == 0.0:
        raise BlankBootstrapError("max_operator_delta_ratio must be positive.")
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
    if residual_map_method not in {"sparse_fold", "joint_sparse_fold", "pca"}:
        raise BlankBootstrapError(
            "residual_map_method must be sparse_fold, joint_sparse_fold, or pca."
        )
    if vocab_row_norm_repair not in {"none", "source_l2"}:
        raise BlankBootstrapError(
            "vocab_row_norm_repair must be none or source_l2."
        )
    if not isinstance(cross_tokenizer_mode, bool):
        raise BlankBootstrapError("cross_tokenizer_mode must be boolean.")
    if cross_tokenizer_mode:
        if token_transport_profile is None:
            raise BlankBootstrapError(
                "cross_tokenizer_mode requires a strict token transport profile."
            )
        if trace_pooling not in {"mean", "last_valid"}:
            raise BlankBootstrapError(
                "Cross-tokenizer bootstrap requires mean or last_valid trace pooling."
            )
        if target_calibration_inputs is None or target_validation_inputs is None:
            raise BlankBootstrapError(
                "Cross-tokenizer bootstrap requires independently tokenized target "
                "calibration and validation inputs."
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
        token_transport_profile,
    )
    joint_source_geometry = None
    if residual_map_method == "joint_sparse_fold":
        joint_source_geometry = _require_tied_source_vocabulary_geometry(
            torch,
            source_model,
            source_layout,
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
    baseline_calibration = _target_trace_rows(
        target_model,
        target_calibration_values,
        trace_pooling,
        block_index=target_layout.layer_count - 1,
    )
    _require_row_alignment(source_calibration, baseline_calibration, "calibration")
    baseline_validation = _target_trace_rows(
        target_model,
        target_validation_values,
        trace_pooling,
        block_index=target_layout.layer_count - 1,
        capture_final_norm=True,
        capture_logits=True,
    )
    _require_row_alignment(source_validation, baseline_validation, "validation")
    target_teacher_validation_logits = _project_teacher_logits_to_target(
        torch,
        source_validation.logits,
        token_transport_profile,
        target_layout.embeddings.vocab_size,
    )
    residual_rows = _residual_basis_rows(
        torch,
        source_calibration,
        max_coordinate_rows,
    )
    residual_map_report = None
    if residual_map_method == "sparse_fold":
        residual_coordinates = build_feature_fold_map(
            residual_rows,
            target_layout.hidden_size,
        )
    elif residual_map_method == "joint_sparse_fold":
        assert joint_source_geometry is not None
        residual_coordinates, residual_map_report = _build_joint_sparse_residual_map(
            torch,
            residual_rows,
            joint_source_geometry,
            target_layout.hidden_size,
            max_coordinate_rows,
            source_layout.layer_count,
        )
    else:
        residual_coordinates = build_parent_residual_basis(
            residual_rows,
            target_layout.hidden_size,
            center=False,
        )
    desired_validation_final = _encode_rows(
        source_validation.blocks[-1].block_output,
        residual_coordinates,
    )
    baseline_gate_metrics = _gate_metrics(
        baseline_validation,
        desired_validation_final,
        target_teacher_validation_logits,
    )
    fit_reports: dict[str, Any] = {
        "teacher_logits.target_space": _target_teacher_logit_contract(
            source_validation.logits,
            target_teacher_validation_logits,
            token_transport_profile,
        )
    }
    if residual_map_report is not None:
        fit_reports["residual.map"] = residual_map_report
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
            vocab_row_norm_repair,
            token_transport_profile,
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
                max_operator_delta_ratio,
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
            token_transport_profile,
            fit_reports,
        )
        _record_complete_evidence(coverage, manifest, fit_reports)

    def gate(_coverage_report) -> GateDecision:
        nonlocal candidate_gate_metrics
        candidate_validation = _target_trace_rows(
            target_model,
            target_validation_values,
            trace_pooling,
            block_index=target_layout.layer_count - 1,
            capture_final_norm=True,
            capture_logits=True,
        )
        candidate_gate_metrics = _gate_metrics(
            candidate_validation,
            desired_validation_final,
            target_teacher_validation_logits,
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
    token_transport_profile: StrictTokenTransportProfile | None,
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
    if token_transport_profile is None:
        vocabularies_match = source_vocab == target_vocab
    else:
        vocabularies_match = (
            token_transport_profile.donor_row_capacity == source_vocab
            and token_transport_profile.target_row_capacity == target_vocab
        )
    if (
        not vocabularies_match
        or int(source_head.weight.shape[0]) != source_vocab
        or int(target_head.weight.shape[0]) != target_vocab
    ):
        raise BlankBootstrapError(
            "Vocabulary operators do not satisfy the shared or strict token "
            "transport row contract."
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


def _target_trace_rows(
    model: Any,
    inputs: Mapping[str, Any] | Any,
    pooling: Literal["token_rows", "mean", "last_valid"],
    *,
    block_index: int,
    capture_final_norm: bool = False,
    capture_logits: bool = False,
) -> Any:
    mode = "flatten_valid" if pooling == "token_rows" else pooling
    return collect_hf_targeted_trace(
        model,
        inputs,
        block_index=block_index,
        mode=mode,
        capture_final_norm=capture_final_norm,
        capture_logits=capture_logits,
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


def _require_tied_source_vocabulary_geometry(
    torch: Any,
    source_model: Any,
    source_layout: DecoderLayout,
) -> tuple[Any, dict[str, Any]]:
    source_embedding = source_model.get_input_embeddings()
    source_head = source_model.get_output_embeddings()
    if source_embedding is None or source_head is None:
        raise BlankBootstrapError(
            "joint_sparse_fold requires source embeddings and an LM head."
        )
    embedding_weight = getattr(source_embedding, "weight", None)
    head_weight = getattr(source_head, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor) or not isinstance(
        head_weight,
        torch.Tensor,
    ):
        raise BlankBootstrapError(
            "joint_sparse_fold requires tensor embedding and LM-head weights."
        )
    if (
        embedding_weight.layout != torch.strided
        or embedding_weight.device.type == "meta"
        or not torch.is_floating_point(embedding_weight)
        or torch.is_complex(embedding_weight)
    ):
        raise BlankBootstrapError(
            "joint_sparse_fold requires a dense, real, non-meta source vocabulary weight."
        )
    if not source_layout.tie_word_embeddings or not _same_storage(
        embedding_weight,
        head_weight,
    ):
        raise BlankBootstrapError(
            "joint_sparse_fold requires a genuinely tied source embedding/LM head."
        )
    expected_shape = (
        int(source_layout.embeddings.vocab_size),
        int(source_layout.hidden_size),
    )
    if embedding_weight.ndim != 2 or tuple(embedding_weight.shape) != expected_shape:
        raise BlankBootstrapError(
            "Tied source vocabulary weight shape does not match the decoder layout; "
            f"expected {expected_shape}, got {tuple(embedding_weight.shape)}."
        )
    if tuple(head_weight.shape) != expected_shape:
        raise BlankBootstrapError(
            "Tied source LM-head shape does not match the decoder layout."
        )
    return embedding_weight, {
        "embedding_path": source_layout.embeddings.token_path,
        "lm_head_path": source_layout.lm_head_path,
        "weight_shape": list(expected_shape),
        "tied_by_storage": True,
    }


def _build_joint_sparse_residual_map(
    torch: Any,
    residual_rows: Any,
    source_geometry: tuple[Any, dict[str, Any]],
    target_width: int,
    max_rows: int,
    source_layer_count: int,
) -> tuple[FeatureFoldMap, dict[str, Any]]:
    source_weight, vocabulary_provenance = source_geometry
    if residual_rows.ndim != 2:
        raise BlankBootstrapError(
            "Joint residual activation rows must be two-dimensional."
        )
    source_width = int(source_weight.shape[1])
    if int(residual_rows.shape[1]) != source_width:
        raise BlankBootstrapError(
            "Joint residual activation and vocabulary widths must match."
        )
    if int(residual_rows.shape[0]) > max_rows:
        raise BlankBootstrapError(
            "Joint residual activation sampling exceeded max_coordinate_rows."
        )

    vocabulary_rows, sampling_report = _sample_vocabulary_geometry_rows(
        torch,
        source_weight,
        max_rows,
    )
    normalized_residuals, residual_report = _normalize_joint_basis_input(
        torch,
        residual_rows,
        source_width,
        name="parent_residual_activations",
    )
    normalized_vocabulary, vocabulary_report = _normalize_joint_basis_input(
        torch,
        vocabulary_rows,
        source_width,
        name="tied_source_embedding_lm_head",
    )
    coordinates = build_feature_fold_map(
        (normalized_residuals, normalized_vocabulary),
        target_width,
    )
    if coordinates.report is None or not coordinates.report.full_source_coverage:
        raise BlankBootstrapError(
            "Joint sparse residual fold did not prove complete source coverage."
        )

    residual_report.update(
        {
            "provenance": {
                "trace_fields": [
                    "block.*.block_input",
                    "block.*.block_output",
                    "final_norm_output",
                ],
                "source_layer_count": int(source_layer_count),
                "sampling": "deterministic-global-row-quota",
            },
            "row_bound": int(max_rows),
        }
    )
    vocabulary_report.update(
        {
            "provenance": vocabulary_provenance,
            "sampling": sampling_report,
            "row_bound": int(max_rows),
            "full_vocabulary_copy_materialized": (
                sampling_report["sampled_rows"]
                == sampling_report["available_rows"]
            ),
            "copy_bounded_by_max_rows": True,
        }
    )
    return coordinates, {
        "method": "joint-logit-aware-sparse-fold",
        "deterministic": True,
        "source_width": source_width,
        "target_width": int(target_width),
        "input_count": 2,
        "normalization": {
            "method": "equal-total-frobenius-energy",
            "target_energy_per_input": float(source_width),
            "prevents_row-count_or_raw-energy_dominance": True,
        },
        "inputs": {
            "parent_residual_activations": residual_report,
            "tied_source_embedding_lm_head": vocabulary_report,
        },
        "feature_fold": coordinates.report.as_dict(),
    }


def _sample_vocabulary_geometry_rows(
    torch: Any,
    source_weight: Any,
    max_rows: int,
) -> tuple[Any, dict[str, Any]]:
    available_rows = int(source_weight.shape[0])
    sampled_rows = min(available_rows, max_rows)
    if available_rows <= 0 or sampled_rows <= 0:
        raise BlankBootstrapError("Source vocabulary geometry cannot be empty.")
    if sampled_rows == 1:
        indices = torch.zeros(1, dtype=torch.long)
    else:
        positions = torch.arange(sampled_rows, dtype=torch.long)
        numerator = positions * (available_rows - 1)
        denominator = sampled_rows - 1
        indices = (2 * numerator + denominator) // (2 * denominator)
    if int(torch.unique(indices).numel()) != sampled_rows:
        raise BlankBootstrapError(
            "Deterministic vocabulary sampling produced duplicate row indices."
        )
    sampled = (
        source_weight.detach()
        .index_select(0, indices.to(device=source_weight.device))
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    expected_shape = (sampled_rows, int(source_weight.shape[1]))
    if tuple(sampled.shape) != expected_shape or not bool(torch.isfinite(sampled).all()):
        raise BlankBootstrapError(
            "Sampled source vocabulary geometry has an invalid shape or value."
        )
    return sampled, {
        "method": "integer-even-spacing-half-up-v1",
        "available_rows": available_rows,
        "sampled_rows": sampled_rows,
        "first_row": int(indices[0]),
        "last_row": int(indices[-1]),
        "index_sum": int(indices.sum()),
        "bounded": sampled_rows <= max_rows,
        "deterministic": True,
    }


def _normalize_joint_basis_input(
    torch: Any,
    matrix: Any,
    source_width: int,
    *,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise BlankBootstrapError(f"Joint basis input {name!r} must be a matrix.")
    if int(matrix.shape[0]) <= 0 or int(matrix.shape[1]) != source_width:
        raise BlankBootstrapError(
            f"Joint basis input {name!r} has an invalid shape."
        )
    values = matrix.detach().to(device="cpu", dtype=torch.float64).contiguous()
    norm = float(torch.linalg.vector_norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise BlankBootstrapError(
            f"Joint basis input {name!r} must have finite positive energy."
        )
    target_norm = math.sqrt(float(source_width))
    scale = target_norm / norm
    normalized = values * scale
    if not bool(torch.isfinite(normalized).all()):
        raise BlankBootstrapError(
            f"Joint basis input {name!r} normalization produced non-finite values."
        )
    normalized_norm = float(torch.linalg.vector_norm(normalized))
    return normalized.contiguous(), {
        "name": name,
        "shape": [int(value) for value in values.shape],
        "sampled_rows": int(values.shape[0]),
        "raw_frobenius_norm": norm,
        "raw_total_energy": norm * norm,
        "applied_scale": scale,
        "normalized_frobenius_norm": normalized_norm,
        "normalized_total_energy": normalized_norm * normalized_norm,
    }


def _row_quotas(lengths: Sequence[int], max_rows: int) -> tuple[int, ...]:
    total = sum(lengths)
    selected = min(total, max_rows)
    if selected == total:
        return tuple(lengths)
    torch = _require_torch()
    indices = torch.linspace(0, total - 1, steps=selected).round().to(torch.long)
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
        torch.linspace(0, int(rows.shape[0]) - 1, steps=count).round().to(torch.long)
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
    coordinates: Any,
    chunk_size: int,
    *,
    row_norm_repair: Literal["none", "source_l2"] = "none",
) -> dict[str, Any] | None:
    if int(source_weight.shape[0]) != int(target_weight.shape[0]):
        raise BlankBootstrapError(
            "Source and target vocabulary operators must have equal row counts."
        )
    target_dim = (
        coordinates.target_width
        if isinstance(coordinates, FeatureFoldMap)
        else coordinates.target_dim
    )
    expected = (int(source_weight.shape[0]), target_dim)
    if tuple(target_weight.shape) != expected:
        raise BlankBootstrapError(
            f"Target vocabulary operator must have shape {expected}."
        )
    if row_norm_repair not in {"none", "source_l2"}:
        raise BlankBootstrapError("row_norm_repair must be none or source_l2.")
    if row_norm_repair == "none":
        with torch.no_grad():
            for start in range(0, int(source_weight.shape[0]), chunk_size):
                end = min(int(source_weight.shape[0]), start + chunk_size)
                projected = _encode_rows(source_weight[start:end], coordinates)
                target_weight[start:end].copy_(projected.to(target_weight))
        return None

    repaired_rows = 0
    zero_source_rows = 0
    zero_projected_rows = 0
    zeroed_projected_rows = 0
    pre_ratio = _empty_ratio_summary()
    repair_ratio = _empty_ratio_summary()
    post_ratio = _empty_ratio_summary()
    with torch.no_grad():
        for start in range(0, int(source_weight.shape[0]), chunk_size):
            end = min(int(source_weight.shape[0]), start + chunk_size)
            source_rows = (
                source_weight[start:end]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            )
            if not bool(torch.isfinite(source_rows).all()):
                raise BlankBootstrapError(
                    "Source vocabulary projection chunk contains NaN or infinity."
                )
            projected = _encode_rows(source_rows, coordinates)
            source_norms = torch.linalg.vector_norm(source_rows, dim=1)
            projected_norms = torch.linalg.vector_norm(projected, dim=1)
            if not bool(torch.isfinite(source_norms).all()) or not bool(
                torch.isfinite(projected_norms).all()
            ):
                raise BlankBootstrapError(
                    "Vocabulary row-norm repair encountered a non-finite norm."
                )

            source_zero = source_norms == 0.0
            projected_zero = projected_norms == 0.0
            unrecoverable = (~source_zero) & projected_zero
            if bool(unrecoverable.any()):
                local_index = int(torch.nonzero(unrecoverable, as_tuple=False)[0])
                raise BlankBootstrapError(
                    "Cannot repair a nonzero source vocabulary row projected to zero "
                    f"at row {start + local_index}."
                )

            zero_source_rows += int(source_zero.sum())
            zero_projected_rows += int(projected_zero.sum())
            source_zero_nonzero_projection = source_zero & (~projected_zero)
            zeroed_projected_rows += int(source_zero_nonzero_projection.sum())
            projected[source_zero] = 0.0

            repairable = (~source_zero) & (~projected_zero)
            ratios = source_norms[repairable] / projected_norms[repairable]
            if not bool(torch.isfinite(ratios).all()):
                raise BlankBootstrapError(
                    "Vocabulary row-norm repair produced a non-finite scale ratio."
                )
            projected[repairable] *= ratios.unsqueeze(1)
            repaired_rows += int(repairable.sum())
            _update_ratio_summary(
                pre_ratio,
                projected_norms[repairable] / source_norms[repairable],
            )
            _update_ratio_summary(repair_ratio, ratios)
            if not bool(torch.isfinite(projected).all()):
                raise BlankBootstrapError(
                    "Vocabulary row-norm repair produced NaN or infinity."
                )

            target_weight[start:end].copy_(projected.to(target_weight))
            copied = (
                target_weight[start:end]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            )
            if not bool(torch.isfinite(copied).all()):
                raise BlankBootstrapError(
                    "Vocabulary row-norm repair overflowed the target dtype."
                )
            copied_norms = torch.linalg.vector_norm(copied, dim=1)
            if bool((copied_norms[source_zero] != 0.0).any()):
                raise BlankBootstrapError(
                    "A zero source vocabulary row did not remain zero after repair."
                )
            copied_nonzero = copied_norms[repairable]
            if bool((copied_nonzero == 0.0).any()):
                raise BlankBootstrapError(
                    "Vocabulary row-norm repair underflowed a nonzero target row."
                )
            _update_ratio_summary(
                post_ratio,
                copied_nonzero / source_norms[repairable],
            )
    return {
        "method": "source-l2-row-norm-repair",
        "total_rows": int(source_weight.shape[0]),
        "repaired_rows": repaired_rows,
        "zero_source_rows": zero_source_rows,
        "zero_projected_rows": zero_projected_rows,
        "zeroed_projected_rows": zeroed_projected_rows,
        "finite": True,
        "pre_repair_projected_to_source_ratio": _finalize_ratio_summary(pre_ratio),
        "applied_source_to_projected_ratio": _finalize_ratio_summary(repair_ratio),
        "post_copy_to_source_ratio": _finalize_ratio_summary(post_ratio),
    }


def _transport_source_rows(
    torch: Any,
    source_rows: Any,
    profile: StrictTokenTransportProfile | None,
) -> Any:
    if profile is None:
        return source_rows
    indices = torch.tensor(
        profile.target_to_donor_row_indices,
        device=source_rows.device,
        dtype=torch.long,
    )
    return source_rows.index_select(0, indices)


def _project_teacher_logits_to_target(
    torch: Any,
    source_logits: Any,
    profile: StrictTokenTransportProfile | None,
    target_vocab_size: int,
) -> Any:
    if source_logits.ndim != 2:
        raise BlankBootstrapError("Pooled teacher logits must be rank two.")
    if profile is None:
        projected = source_logits
    else:
        indices = torch.tensor(
            profile.target_to_donor_row_indices,
            device=source_logits.device,
            dtype=torch.long,
        )
        projected = source_logits.index_select(1, indices)
    if int(projected.shape[1]) != int(target_vocab_size):
        raise BlankBootstrapError(
            "Target-space teacher logits do not match the target vocabulary."
        )
    return projected


def _target_teacher_logit_contract(
    source_logits: Any,
    target_logits: Any,
    profile: StrictTokenTransportProfile | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "space": "target-tokenizer-columns",
        "projection": (
            "strict-target-row-to-donor-row-index-select"
            if profile is not None
            else "shared-token-id-identity"
        ),
        "source_shape": list(source_logits.shape),
        "target_shape": list(target_logits.shape),
        "alignment_fingerprint": (
            profile.alignment_fingerprint if profile is not None else None
        ),
        "transport_profile_fingerprint": (
            profile.fingerprint() if profile is not None else None
        ),
    }


def _empty_ratio_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "sum": 0.0,
        "minimum": math.inf,
        "maximum": -math.inf,
    }


def _update_ratio_summary(summary: dict[str, Any], values: Any) -> None:
    if int(values.numel()) == 0:
        return
    summary["count"] += int(values.numel())
    summary["sum"] += float(values.sum())
    summary["minimum"] = min(summary["minimum"], float(values.min()))
    summary["maximum"] = max(summary["maximum"], float(values.max()))


def _finalize_ratio_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    count = int(summary["count"])
    if count == 0:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "count": count,
        "minimum": float(summary["minimum"]),
        "maximum": float(summary["maximum"]),
        "mean": float(summary["sum"]) / count,
    }


def _initialize_embedding_and_head(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_layout: DecoderLayout,
    residual_coordinates: Any,
    source_trace: FlattenedTransportTrace,
    target_calibration_inputs: Mapping[str, Any] | Any,
    trace_pooling: Literal["token_rows", "mean", "last_valid"],
    vocab_projection_chunk_size: int,
    vocab_row_norm_repair: Literal["none", "source_l2"],
    token_transport_profile: StrictTokenTransportProfile | None,
    reports: dict[str, Any],
) -> None:
    source_embedding = source_model.get_input_embeddings()
    target_embedding = target_model.get_input_embeddings()
    source_head = source_model.get_output_embeddings()
    target_head = target_model.get_output_embeddings()
    source_embedding_rows = _transport_source_rows(
        torch,
        source_embedding.weight,
        token_transport_profile,
    )
    source_head_rows = _transport_source_rows(
        torch,
        source_head.weight,
        token_transport_profile,
    )
    embedding_repair = _copy_projected_rows(
        torch,
        source_embedding_rows,
        target_embedding.weight,
        residual_coordinates,
        vocab_projection_chunk_size,
        row_norm_repair=vocab_row_norm_repair,
    )
    head_repair = _copy_projected_rows(
        torch,
        source_head_rows,
        target_head.weight,
        residual_coordinates,
        vocab_projection_chunk_size,
        row_norm_repair=vocab_row_norm_repair,
    )
    with torch.no_grad():
        source_bias = getattr(source_head, "bias", None)
        target_bias = getattr(target_head, "bias", None)
        if target_bias is not None:
            target_bias.zero_()
            if source_bias is not None:
                transported_bias = _transport_source_rows(
                    torch,
                    source_bias,
                    token_transport_profile,
                )
                target_bias.copy_(transported_bias.to(target_bias))
    reports["embeddings.token"] = {
        "method": "chunked-parent-residual-basis-projection",
        "shape": list(target_embedding.weight.shape),
        "chunk_rows": vocab_projection_chunk_size,
        "row_transport": (
            "strict-index-select" if token_transport_profile is not None else "identity"
        ),
    }
    reports["lm_head"] = {
        "method": "chunked-parent-residual-basis-projection",
        "shape": list(target_head.weight.shape),
        "chunk_rows": vocab_projection_chunk_size,
        "row_transport": (
            "strict-index-select" if token_transport_profile is not None else "identity"
        ),
    }
    if embedding_repair is not None:
        target_tied = _same_storage(target_embedding.weight, target_head.weight)
        reports["embeddings.token"]["row_norm_repair"] = embedding_repair
        reports["embeddings.token"]["target_tied_storage"] = target_tied
        reports["lm_head"]["row_norm_repair"] = head_repair
        reports["lm_head"]["target_tied_storage"] = target_tied
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
    residual_coordinates: Any,
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
            projected = _encode_rows(
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
            desired = _encode_rows(
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
    residual_coordinates: Any,
    ridge: float,
    max_operator_delta_ratio: float,
    max_coordinate_rows: int,
    reports: dict[str, Any],
) -> None:
    target_block_layout = target_layout.blocks[target_index]

    def current_trace() -> Any:
        return _target_trace_rows(
            target_model,
            target_calibration_inputs,
            trace_pooling,
            block_index=target_index,
        )

    current = current_trace()
    _require_row_alignment(source_trace, current, "calibration")
    target_block = current.block
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

    current = current_trace()
    target_block = current.block
    target_attention_layout = target_block_layout.attention
    for role, target_norm in (
        ("query", target_attention_layout.query_norm),
        ("key", target_attention_layout.key_norm),
    ):
        if target_norm is None:
            continue
        reports[f"block.{target_index}.attention.{role}_norm"] = (
            _transport_target_head_norm(
                torch,
                source_model,
                target_model,
                source_layout,
                source_span,
                target_norm,
                target_attention_layout.head_dim,
                role=role,
            )
        )
    source_q_rows, source_k_rows, source_v_rows = (
        _effective_source_attention_projection_rows(
            torch,
            source_model,
            source_layout,
            source_blocks,
            source_span,
            target_attention_layout,
        )
    )
    context_rows = tuple(
        block.attention_output_projection_input for block in source_blocks
    )
    q_coordinates, k_coordinates, v_coordinates, context_coordinates = (
        _attention_feature_coordinates(
        torch,
        source_q_rows,
        source_k_rows,
        source_v_rows,
        context_rows,
        source_layout.blocks[source_span[-1]].attention,
        target_block_layout,
        max_coordinate_rows,
        use_coordinate_maps=isinstance(residual_coordinates, CoordinateMap),
        )
    )
    reports[f"block.{target_index}.attention.feature_folds"] = {
        "query": _coordinate_report(q_coordinates),
        "key": _coordinate_report(k_coordinates),
        "value": _coordinate_report(v_coordinates),
        "context": _coordinate_report(context_coordinates),
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
    attention_prior_report = _initialize_attention_structural_prior(
        torch,
        source_model,
        target_model,
        source_layout,
        target_block_layout,
        source_span,
        residual_coordinates,
        q_coordinates,
        k_coordinates,
        v_coordinates,
        context_coordinates,
    )
    reports[f"block.{target_index}.attention.structural_prior"] = (
        attention_prior_report
    )
    desired_q = _mean_encoded_rows(torch, source_q_rows, q_coordinates)
    desired_k = _mean_encoded_rows(torch, source_k_rows, k_coordinates)
    desired_v = _mean_encoded_rows(torch, source_v_rows, v_coordinates)
    target_attention = read_attention_operators(
        target_model,
        target_block_layout.attention,
    )
    q_fit = _fit_operator_with_prior(
        target_block.pre_attention_norm_output,
        desired_q,
        target_attention.query,
        ridge=ridge,
        preserve_prior=bool(attention_prior_report["applied"]),
        max_prior_delta_ratio=max_operator_delta_ratio,
    )
    k_fit = _fit_operator_with_prior(
        target_block.pre_attention_norm_output,
        desired_k,
        target_attention.key,
        ridge=ridge,
        preserve_prior=bool(attention_prior_report["applied"]),
        max_prior_delta_ratio=max_operator_delta_ratio,
    )
    v_fit = _fit_operator_with_prior(
        target_block.pre_attention_norm_output,
        desired_v,
        target_attention.value,
        ridge=ridge,
        preserve_prior=bool(attention_prior_report["applied"]),
        max_prior_delta_ratio=max_operator_delta_ratio,
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

    current = current_trace()
    target_block = current.block
    desired_attention_delta = _sum_projected(
        torch,
        source_blocks,
        "attention_delta",
        residual_coordinates,
    )
    target_attention = read_attention_operators(
        target_model,
        target_block_layout.attention,
    )
    output_fit = _fit_operator_with_prior(
        target_block.attention_output_projection_input,
        desired_attention_delta,
        target_attention.output,
        ridge=ridge,
        preserve_prior=bool(attention_prior_report["applied"]),
        max_prior_delta_ratio=max_operator_delta_ratio,
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

    current = current_trace()
    target_block = current.block
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

    current = current_trace()
    target_block = current.block
    mlp_feature_rows = _source_mlp_feature_rows(
        torch,
        source_blocks,
        max_coordinate_rows,
    )
    gated_triplet_compatible = (
        target_block_layout.mlp.kind == "gated"
        and all(
            source_layout.blocks[index].mlp.kind == "gated"
            for index in source_span
        )
    )
    if isinstance(residual_coordinates, CoordinateMap):
        mlp_coordinates = build_parent_residual_basis(
            _sample_concatenated_rows(
                torch,
                mlp_feature_rows,
                max_coordinate_rows,
            ),
            target_block_layout.mlp.intermediate_size,
            center=False,
        )
    else:
        mlp_coordinates = build_feature_fold_map(
            mlp_feature_rows,
            target_block_layout.mlp.intermediate_size,
            allow_sign_flip=not gated_triplet_compatible,
        )
    reports[f"block.{target_index}.mlp.feature_fold"] = _coordinate_report(
        mlp_coordinates
    )
    mlp_prior_report = _initialize_mlp_structural_prior(
        torch,
        source_model,
        target_model,
        source_layout,
        target_block_layout,
        source_span,
        residual_coordinates,
        mlp_coordinates,
    )
    reports[f"block.{target_index}.mlp.structural_prior"] = mlp_prior_report
    target_mlp = read_mlp_operators(target_model, target_block_layout)
    product_target_report = {
        "applied": False,
        "reason": "target-mlp-is-not-compatible-gated-swiglu",
    }
    desired_product = None
    if target_block_layout.mlp.kind == "dense":
        assert target_mlp.input is not None
        desired_input = _mean_source_mlp_producer(
            torch,
            source_blocks,
            "dense",
            mlp_coordinates,
        )
        input_fit = _fit_operator_with_prior(
            target_block.pre_mlp_norm_output,
            desired_input,
            target_mlp.input,
            ridge=ridge,
            preserve_prior=bool(mlp_prior_report["applied"]),
            max_prior_delta_ratio=max_operator_delta_ratio,
        )
        target_mlp = MLPOperators(
            kind="dense",
            input=_operator_from_fit(target_mlp.input, input_fit),
            gate=None,
            up=None,
            down=target_mlp.down,
        )
        reports[f"block.{target_index}.mlp.input"] = _fit_summary(input_fit)
    else:
        assert target_mlp.gate is not None and target_mlp.up is not None
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
        if (
            gated_triplet_compatible
            and target_block_layout.mlp.activation_name == "silu"
        ):
            desired_product = _sum_projected(
                torch,
                source_blocks,
                "down_projection_input",
                mlp_coordinates,
            )
            desired_up, product_target_report = _gated_up_target_for_product(
                torch,
                desired_gate,
                desired_product,
            )
        else:
            product_target_report = {
                "applied": False,
                "reason": (
                    "source-target-mlp-topology-mismatch"
                    if not gated_triplet_compatible
                    else "unsupported-gated-activation"
                ),
                "activation": target_block_layout.mlp.activation_name,
            }
        reports[f"block.{target_index}.mlp.product_target"] = (
            product_target_report
        )
        gate_fit = _fit_operator_with_prior(
            target_block.pre_mlp_norm_output,
            desired_gate,
            target_mlp.gate,
            ridge=ridge,
            preserve_prior=bool(mlp_prior_report["applied"]),
            max_prior_delta_ratio=max_operator_delta_ratio,
        )
        up_fit = _fit_operator_with_prior(
            target_block.pre_mlp_norm_output,
            desired_up,
            target_mlp.up,
            ridge=ridge,
            preserve_prior=bool(mlp_prior_report["applied"]),
            max_prior_delta_ratio=max_operator_delta_ratio,
        )
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

    current = current_trace()
    target_block = current.block
    if bool(product_target_report.get("applied")):
        assert desired_product is not None
        product_metrics = measure_fit(
            target_block.down_projection_input,
            desired_product,
        )
        reports[f"block.{target_index}.mlp.product_fit"] = {
            "method": "folded-source-product-anchor",
            "rows": product_metrics.rows,
            "relative_rmse": product_metrics.relative_rmse,
            "mean_cosine_similarity": product_metrics.mean_cosine_similarity,
            "finite": product_metrics.finite,
        }
    desired_endpoint = _encode_rows(
        source_blocks[-1].block_output,
        residual_coordinates,
    )
    desired_mlp_delta = desired_endpoint - target_block.post_attention_residual
    target_mlp = read_mlp_operators(target_model, target_block_layout)
    down_fit = _fit_operator_with_prior(
        target_block.down_projection_input,
        desired_mlp_delta,
        target_mlp.down,
        ridge=ridge,
        preserve_prior=bool(mlp_prior_report["applied"]),
        max_prior_delta_ratio=max_operator_delta_ratio,
    )
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
        "residual_composition": "ordered-source-span-endpoint",
        "attention_delta_target": "sum-encoded-source-attention-deltas",
        "mlp_delta_target": (
            "encoded-final-source-block-output-minus-target-post-attention-residual"
        ),
        "structural_prior_residual_consumers": "sum",
        "structural_prior_feature_producers": "mean",
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
    residual_coordinates: Any,
    ridge: float,
    fit_full_lm_head: bool,
    max_full_head_fit_elements: int,
    token_transport_profile: StrictTokenTransportProfile | None,
    reports: dict[str, Any],
) -> None:
    del source_model, source_layout
    last_block_index = target_layout.layer_count - 1
    current = _target_trace_rows(
        target_model,
        target_calibration_inputs,
        trace_pooling,
        block_index=last_block_index,
    )
    target_norm = resolve_module(target_model, target_layout.final_norm.path)
    norm_input = _norm_input_from_module_trace(current)
    desired_norm = _encode_rows(
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

    head = target_model.get_output_embeddings()
    if fit_full_lm_head:
        current = _target_trace_rows(
            target_model,
            target_calibration_inputs,
            trace_pooling,
            block_index=last_block_index,
            capture_final_norm=True,
        )
        target_teacher_logits = _project_teacher_logits_to_target(
            torch,
            source_trace.logits,
            token_transport_profile,
            target_layout.embeddings.vocab_size,
        )
        elements = int(current.final_norm_output.shape[0]) * int(
            target_teacher_logits.shape[1]
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
                target_teacher_logits,
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
    context_rows: Sequence[Any],
    source_attention_layout: Any,
    target_block_layout: Any,
    max_coordinate_rows: int,
    *,
    use_coordinate_maps: bool = False,
) -> tuple[Any, Any, Any, Any]:
    target_attention = target_block_layout.attention
    if use_coordinate_maps:
        q_rows = _sample_concatenated_rows(torch, query_rows, max_coordinate_rows)
        k_rows = _sample_concatenated_rows(torch, key_rows, max_coordinate_rows)
        v_rows = _sample_concatenated_rows(torch, value_rows, max_coordinate_rows)
        o_rows = _sample_concatenated_rows(torch, context_rows, max_coordinate_rows)
        return (
            build_parent_residual_basis(
                q_rows,
                target_attention.query_width,
                center=False,
            ),
            build_parent_residual_basis(
                k_rows,
                target_attention.key_value_width,
                center=False,
            ),
            build_parent_residual_basis(
                v_rows,
                target_attention.key_value_width,
                center=False,
            ),
            build_parent_residual_basis(
                o_rows,
                target_attention.query_width,
                center=False,
            ),
        )
    if (
        source_attention_layout.projection_style == "split_qkv"
        and target_attention.projection_style == "split_qkv"
    ):
        maps = build_gqa_attention_fold_maps(
            source_query_heads=source_attention_layout.query_heads,
            source_key_value_heads=source_attention_layout.key_value_heads,
            source_head_dim=source_attention_layout.head_dim,
            target_query_heads=target_attention.query_heads,
            target_key_value_heads=target_attention.key_value_heads,
            target_head_dim=target_attention.head_dim,
        )
        return maps.query, maps.key, maps.value, maps.context
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
    o_rows = _sample_concatenated_rows(
        torch,
        context_rows,
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
    context_coordinates = build_feature_fold_map(o_rows, q_width)
    return q_coordinates, k_coordinates, v_coordinates, context_coordinates


def _coordinate_report(coordinates: Any) -> dict[str, Any]:
    if isinstance(coordinates, FeatureFoldMap):
        return coordinates.report.as_dict()
    if not isinstance(coordinates, CoordinateMap):
        raise BlankBootstrapError("Unsupported feature-coordinate map.")
    return {
        "method": "pca-coordinate-map",
        "source_width": coordinates.source_dim,
        "target_width": coordinates.target_dim,
        "activation_row_count": coordinates.activation_row_count,
        "effective_rank": coordinates.effective_rank,
        "explained_variance_fraction": coordinates.explained_variance_fraction,
        "centered_for_fit": coordinates.centered_for_fit,
        "orthonormal_columns": (
            coordinates.diagnostics.target_identity_max_abs_error <= 1e-10
        ),
        "target_identity_max_abs_error": (
            coordinates.diagnostics.target_identity_max_abs_error
        ),
        "finite": coordinates.diagnostics.finite,
    }


def _effective_source_attention_projection_rows(
    torch: Any,
    source_model: Any,
    source_layout: DecoderLayout,
    source_blocks: Sequence[DecoderBlockTrace],
    source_span: Sequence[int],
    target_attention_layout: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    query_rows = []
    key_rows = []
    value_rows = []
    for source_index, block in zip(source_span, source_blocks, strict=True):
        attention = source_layout.blocks[source_index].attention
        query = block.query_projection_output
        key = block.key_projection_output
        if (
            target_attention_layout.query_norm is None
            and attention.query_norm is not None
        ):
            query = _apply_head_norm(
                torch,
                query,
                source_model,
                attention.query_norm,
                attention.query_heads,
                attention.head_dim,
            )
        if (
            target_attention_layout.key_norm is None
            and attention.key_norm is not None
        ):
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


def _transport_target_head_norm(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    source_span: Sequence[int],
    target_norm: Any,
    target_head_dim: int,
    *,
    role: Literal["query", "key"],
) -> dict[str, Any]:
    if not source_span:
        raise BlankBootstrapError("A head-norm transfer needs a nonempty source span.")
    _validate_head_norm_layout(target_norm, label="target")

    folded_scales = []
    source_norm_paths = []
    identity_source_layers = []
    dimension_group_sizes = []
    for source_index in source_span:
        source_attention = source_layout.blocks[source_index].attention
        source_norm = getattr(source_attention, f"{role}_norm")
        source_head_dim = int(source_attention.head_dim)
        if source_norm is None:
            effective_scale = torch.ones(source_head_dim, dtype=torch.float64)
            identity_source_layers.append(int(source_index))
        else:
            effective_scale = _read_effective_head_norm_scale(
                torch,
                source_model,
                source_norm,
                source_head_dim,
                label=f"source block {source_index} {role}",
            )
            source_norm_paths.append(source_norm.path)
        folded, group_sizes = _fold_head_norm_scale(
            torch,
            effective_scale,
            target_head_dim,
        )
        folded_scales.append(folded)
        dimension_group_sizes.append(list(group_sizes))

    transported = torch.stack(folded_scales, dim=0).mean(dim=0)
    if not bool(torch.isfinite(transported).all()):
        raise BlankBootstrapError("Head-norm scale transport produced non-finite values.")
    _write_effective_head_norm_scale(
        torch,
        target_model,
        target_norm,
        transported,
        target_head_dim,
    )
    return {
        "method": "deterministic-full-coverage-effective-rms-scale-mean",
        "role": role,
        "source_layers": [int(index) for index in source_span],
        "source_norm_paths": source_norm_paths,
        "identity_source_layers": identity_source_layers,
        "source_layer_count": len(source_span),
        "source_norm_count": len(source_norm_paths),
        "identity_source_count": len(identity_source_layers),
        "target_head_dim": int(target_head_dim),
        "dimension_group_sizes": dimension_group_sizes,
        "source_layer_coverage_fraction": 1.0,
        "full_source_coverage": True,
        "target_weight_kind": target_norm.weight_kind,
        "effective_scale_min": float(transported.min()),
        "effective_scale_max": float(transported.max()),
        "effective_scale_mean": float(transported.mean()),
        "finite": True,
    }


def _fold_head_norm_scale(
    torch: Any,
    effective_scale: Any,
    target_head_dim: int,
) -> tuple[Any, tuple[int, ...]]:
    values = effective_scale.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1 or int(values.numel()) <= 0:
        raise BlankBootstrapError("A per-head norm scale must be one-dimensional.")
    if not bool(torch.isfinite(values).all()):
        raise BlankBootstrapError("A per-head norm scale contains NaN or infinity.")
    source_head_dim = int(values.numel())
    if (
        isinstance(target_head_dim, bool)
        or not isinstance(target_head_dim, int)
        or target_head_dim <= 0
    ):
        raise BlankBootstrapError("Target head dimension must be a positive integer.")
    if source_head_dim < target_head_dim:
        raise BlankBootstrapError(
            "Head-norm expansion is outside the compression transfer operator."
        )
    if source_head_dim % 2 or target_head_dim % 2:
        raise BlankBootstrapError(
            "Per-head norm transfer requires even RoPE head dimensions."
        )

    source_half = source_head_dim // 2
    target_half = target_head_dim // 2
    assignment = torch.empty(source_head_dim, dtype=torch.long)
    for source_channel in range(source_head_dim):
        half = 0 if source_channel < source_half else 1
        source_pair = source_channel % source_half
        target_pair = min(
            target_half - 1,
            source_pair * target_half // source_half,
        )
        assignment[source_channel] = target_pair + half * target_half

    counts = torch.bincount(assignment, minlength=target_head_dim)
    if bool((counts <= 0).any()) or int(counts.sum()) != source_head_dim:
        raise BlankBootstrapError(
            "Head-norm dimension transfer did not cover every source channel."
        )
    folded = torch.zeros(target_head_dim, dtype=torch.float64)
    folded.index_add_(0, assignment, values)
    folded = folded / counts.to(torch.float64)
    if not bool(torch.isfinite(folded).all()):
        raise BlankBootstrapError("Head-norm dimension mean produced non-finite values.")
    return folded.contiguous(), tuple(int(value) for value in counts.tolist())


def _validate_head_norm_layout(layout: Any, *, label: str) -> None:
    if layout.kind != "rms_norm":
        raise BlankBootstrapError(
            f"Only per-head RMSNorm is supported for {label} attention norms."
        )
    if layout.has_bias:
        raise BlankBootstrapError(
            f"Biased per-head normalization is unsupported for {label} attention norms."
        )
    if layout.weight_kind not in {"direct_scale", "unit_offset"}:
        raise BlankBootstrapError(
            f"Unsupported effective-scale topology {layout.weight_kind!r} for "
            f"{label} attention norm."
        )


def _read_effective_head_norm_scale(
    torch: Any,
    model: Any,
    layout: Any,
    head_dim: int,
    *,
    label: str,
) -> Any:
    _validate_head_norm_layout(layout, label=label)
    module = resolve_module(model, layout.path)
    module_bias = getattr(module, "bias", None)
    if module_bias is not None:
        raise BlankBootstrapError(
            f"Per-head norm {layout.path!r} has an unsupported bias parameter."
        )
    weight = module.weight.detach().to(device="cpu", dtype=torch.float64)
    if tuple(weight.shape) != (head_dim,):
        raise BlankBootstrapError(
            f"Per-head norm {layout.path!r} must have shape ({head_dim},)."
        )
    effective = weight if layout.weight_kind == "direct_scale" else weight + 1.0
    if not bool(torch.isfinite(effective).all()):
        raise BlankBootstrapError(
            f"Per-head norm {layout.path!r} has a non-finite effective scale."
        )
    return effective.contiguous()


def _write_effective_head_norm_scale(
    torch: Any,
    model: Any,
    layout: Any,
    effective_scale: Any,
    head_dim: int,
) -> None:
    _validate_head_norm_layout(layout, label="target")
    module = resolve_module(model, layout.path)
    module_bias = getattr(module, "bias", None)
    if module_bias is not None:
        raise BlankBootstrapError(
            f"Per-head norm {layout.path!r} has an unsupported bias parameter."
        )
    if tuple(module.weight.shape) != (head_dim,):
        raise BlankBootstrapError(
            f"Per-head norm {layout.path!r} must have shape ({head_dim},)."
        )
    values = effective_scale.detach().to(device="cpu", dtype=torch.float64)
    if tuple(values.shape) != (head_dim,) or not bool(torch.isfinite(values).all()):
        raise BlankBootstrapError(
            "Target per-head norm effective scale has an invalid shape or value."
        )
    stored = values if layout.weight_kind == "direct_scale" else values - 1.0
    with torch.no_grad():
        module.weight.copy_(stored.to(module.weight))


def _apply_head_norm(
    torch: Any,
    rows: Any,
    model: Any,
    layout: Any,
    head_count: int,
    head_dim: int,
) -> Any:
    weight = _read_effective_head_norm_scale(
        torch,
        model,
        layout,
        head_dim,
        label="source",
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
            matrices.append(_sample_concatenated_rows(torch, present, max_rows))
    if not matrices:
        raise BlankBootstrapError("Source trace has no MLP feature rows to fold.")
    return tuple(matrices)


def _transport_operator_prior(
    torch: Any,
    source: LinearOperator,
    target: LinearOperator,
    output_coordinates: Any,
    input_coordinates: Any,
) -> LinearOperator:
    if isinstance(input_coordinates, CoordinateMap) and isinstance(
        output_coordinates,
        CoordinateMap,
    ):
        transported = transport_linear_operator(
            source.weight,
            input_coordinates,
            output_coordinates,
            source_bias=source.bias,
        )
        bias = transported.bias
        if target.bias is not None and bias is None:
            bias = torch.zeros(transported.weight.shape[0], dtype=torch.float64)
        return LinearOperator(
            path=target.path,
            weight=transported.weight,
            bias=bias if target.bias is not None else None,
        )
    if not isinstance(input_coordinates, FeatureFoldMap) or not isinstance(
        output_coordinates,
        FeatureFoldMap,
    ):
        raise BlankBootstrapError(
            "Structural prior coordinate maps must use one coherent map family."
        )
    folded_input = input_coordinates.fold_consumer_weight(source.weight)
    weight = output_coordinates.fold_producer_weight(folded_input)
    bias = None
    if target.bias is not None:
        if source.bias is None:
            bias = torch.zeros(weight.shape[0], dtype=torch.float64)
        else:
            bias = output_coordinates.fold_producer_weight(
                source.bias.reshape(-1, 1)
            ).reshape(-1)
    return LinearOperator(path=target.path, weight=weight, bias=bias)


def _mean_operator_priors(
    torch: Any,
    target: LinearOperator,
    values: Sequence[LinearOperator],
) -> LinearOperator:
    if not values:
        raise BlankBootstrapError("A structural prior needs at least one donor operator.")
    weight = torch.stack([value.weight for value in values], dim=0).mean(dim=0)
    bias = None
    if target.bias is not None:
        if any(value.bias is None for value in values):
            raise BlankBootstrapError("Structural prior bias topology is inconsistent.")
        bias = torch.stack([value.bias for value in values], dim=0).mean(dim=0)
    return LinearOperator(path=target.path, weight=weight, bias=bias)


def _sum_operator_priors(
    torch: Any,
    target: LinearOperator,
    values: Sequence[LinearOperator],
) -> LinearOperator:
    if not values:
        raise BlankBootstrapError("A structural prior needs at least one donor operator.")
    weight = torch.stack([value.weight for value in values], dim=0).sum(dim=0)
    bias = None
    if target.bias is not None:
        if any(value.bias is None for value in values):
            raise BlankBootstrapError("Structural prior bias topology is inconsistent.")
        bias = torch.stack([value.bias for value in values], dim=0).sum(dim=0)
    return LinearOperator(path=target.path, weight=weight, bias=bias)


def _initialize_attention_structural_prior(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_block_layout: Any,
    source_span: Sequence[int],
    residual_coordinates: Any,
    query_coordinates: Any,
    key_coordinates: Any,
    value_coordinates: Any,
    context_coordinates: Any,
) -> dict[str, Any]:
    if not isinstance(residual_coordinates, (FeatureFoldMap, CoordinateMap)):
        return {"applied": False, "reason": "unsupported-residual-coordinate-map"}
    source_projection_styles = {
        source_layout.blocks[index].attention.projection_style
        for index in source_span
    }
    if source_projection_styles != {
        target_block_layout.attention.projection_style
    }:
        return {
            "applied": False,
            "reason": "source-target-attention-projection-topology-mismatch",
            "source_projection_styles": sorted(source_projection_styles),
            "target_projection_style": (
                target_block_layout.attention.projection_style
            ),
        }
    target = read_attention_operators(
        target_model,
        target_block_layout.attention,
    )
    transported = []
    for source_index in source_span:
        source = read_attention_operators(
            source_model,
            source_layout.blocks[source_index].attention,
        )
        transported.append(
            AttentionOperators(
                query=_transport_operator_prior(
                    torch,
                    source.query,
                    target.query,
                    query_coordinates,
                    residual_coordinates,
                ),
                key=_transport_operator_prior(
                    torch,
                    source.key,
                    target.key,
                    key_coordinates,
                    residual_coordinates,
                ),
                value=_transport_operator_prior(
                    torch,
                    source.value,
                    target.value,
                    value_coordinates,
                    residual_coordinates,
                ),
                output=_transport_operator_prior(
                    torch,
                    source.output,
                    target.output,
                    residual_coordinates,
                    context_coordinates,
                ),
            )
        )
    merged = AttentionOperators(
        query=_mean_operator_priors(
            torch,
            target.query,
            [value.query for value in transported],
        ),
        key=_mean_operator_priors(
            torch,
            target.key,
            [value.key for value in transported],
        ),
        value=_mean_operator_priors(
            torch,
            target.value,
            [value.value for value in transported],
        ),
        output=_sum_operator_priors(
            torch,
            target.output,
            [value.output for value in transported],
        ),
    )
    write_attention_operators(
        target_model,
        target_block_layout.attention,
        merged,
    )
    return {
        "applied": True,
        "method": "full-coverage-source-operator-transport-residual-sum",
        "coordinate_method": (
            "pca-coordinate-formula"
            if isinstance(residual_coordinates, CoordinateMap)
            else "sparse-feature-fold"
        ),
        "producer_aggregation": "mean",
        "residual_consumer_aggregation": "sum",
        "source_layers": list(source_span),
        "query_weight_norm": float(torch.linalg.vector_norm(merged.query.weight)),
        "key_weight_norm": float(torch.linalg.vector_norm(merged.key.weight)),
        "value_weight_norm": float(torch.linalg.vector_norm(merged.value.weight)),
        "output_weight_norm": float(torch.linalg.vector_norm(merged.output.weight)),
    }


def _initialize_mlp_structural_prior(
    torch: Any,
    source_model: Any,
    target_model: Any,
    source_layout: DecoderLayout,
    target_block_layout: Any,
    source_span: Sequence[int],
    residual_coordinates: Any,
    mlp_coordinates: Any,
) -> dict[str, Any]:
    if not isinstance(residual_coordinates, (FeatureFoldMap, CoordinateMap)):
        return {"applied": False, "reason": "unsupported-residual-coordinate-map"}
    target = read_mlp_operators(target_model, target_block_layout)
    sources = [
        read_mlp_operators(source_model, source_layout.blocks[index])
        for index in source_span
    ]
    if any(source.kind != target.kind for source in sources):
        return {"applied": False, "reason": "source-target-mlp-topology-mismatch"}

    if target.kind == "dense":
        assert target.input is not None
        source_inputs = [source.input for source in sources]
        assert all(value is not None for value in source_inputs)
        input_priors = [
            _transport_operator_prior(
                torch,
                source,
                target.input,
                mlp_coordinates,
                residual_coordinates,
            )
            for source in source_inputs
        ]
        merged = MLPOperators(
            kind="dense",
            input=_mean_operator_priors(torch, target.input, input_priors),
            gate=None,
            up=None,
            down=_sum_operator_priors(
                torch,
                target.down,
                [
                    _transport_operator_prior(
                        torch,
                        source.down,
                        target.down,
                        residual_coordinates,
                        mlp_coordinates,
                    )
                    for source in sources
                ],
            ),
        )
    else:
        assert target.gate is not None and target.up is not None
        assert all(source.gate is not None and source.up is not None for source in sources)
        merged = MLPOperators(
            kind="gated",
            input=None,
            gate=_mean_operator_priors(
                torch,
                target.gate,
                [
                    _transport_operator_prior(
                        torch,
                        source.gate,
                        target.gate,
                        mlp_coordinates,
                        residual_coordinates,
                    )
                    for source in sources
                ],
            ),
            up=_mean_operator_priors(
                torch,
                target.up,
                [
                    _transport_operator_prior(
                        torch,
                        source.up,
                        target.up,
                        mlp_coordinates,
                        residual_coordinates,
                    )
                    for source in sources
                ],
            ),
            down=_sum_operator_priors(
                torch,
                target.down,
                [
                    _transport_operator_prior(
                        torch,
                        source.down,
                        target.down,
                        residual_coordinates,
                        mlp_coordinates,
                    )
                    for source in sources
                ],
            ),
        )
    write_mlp_operators(target_model, target_block_layout, merged)
    return {
        "applied": True,
        "method": "full-coverage-source-operator-transport-residual-sum",
        "coordinate_method": (
            "pca-coordinate-formula"
            if isinstance(residual_coordinates, CoordinateMap)
            else "sparse-feature-fold"
        ),
        "producer_aggregation": "mean",
        "residual_consumer_aggregation": "sum",
        "source_layers": list(source_span),
        "down_weight_norm": float(torch.linalg.vector_norm(merged.down.weight)),
    }


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


def _gated_up_target_for_product(
    torch: Any,
    desired_gate_rows: Any,
    desired_product_rows: Any,
) -> tuple[Any, dict[str, Any]]:
    gate, product = (
        desired_gate_rows.detach().to(device="cpu", dtype=torch.float64),
        desired_product_rows.detach().to(device="cpu", dtype=torch.float64),
    )
    if gate.shape != product.shape or gate.ndim != 2:
        raise BlankBootstrapError(
            "Gated product anchors must be aligned two-dimensional matrices."
        )
    gate_activation = torch.nn.functional.silu(gate)
    channel_rms = torch.sqrt(gate_activation.square().mean(dim=0))
    denominator_floor = torch.clamp(channel_rms * 0.05, min=1e-4)
    signs = torch.where(
        gate_activation < 0.0,
        -torch.ones_like(gate_activation),
        torch.ones_like(gate_activation),
    )
    stabilized = torch.where(
        gate_activation.abs() >= denominator_floor.unsqueeze(0),
        gate_activation,
        signs * denominator_floor.unsqueeze(0),
    )
    raw_target = product / stabilized
    target_rms = torch.sqrt(raw_target.square().mean(dim=0))
    clip_bound = torch.clamp(target_rms * 8.0, min=1e-3)
    target = torch.maximum(
        torch.minimum(raw_target, clip_bound.unsqueeze(0)),
        -clip_bound.unsqueeze(0),
    )
    if not bool(torch.isfinite(target).all()):
        raise BlankBootstrapError("Gated product target contains NaN or infinity.")
    stabilized_fraction = float(
        (gate_activation.abs() < denominator_floor.unsqueeze(0))
        .to(torch.float64)
        .mean()
    )
    clipped_fraction = float(
        (raw_target.abs() > clip_bound.unsqueeze(0)).to(torch.float64).mean()
    )
    return target.contiguous(), {
        "applied": True,
        "method": "alternating-closed-form-swiglu-product-anchor",
        "rows": int(target.shape[0]),
        "width": int(target.shape[1]),
        "denominator_stabilized_fraction": stabilized_fraction,
        "target_clipped_fraction": clipped_fraction,
        "gate_activation_rms": float(
            torch.sqrt(gate_activation.square().mean())
        ),
        "desired_product_rms": float(torch.sqrt(product.square().mean())),
        "up_target_rms": float(torch.sqrt(target.square().mean())),
    }


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


def _fit_operator_with_prior(
    input_rows: Any,
    target_rows: Any,
    prior: LinearOperator,
    *,
    ridge: float,
    preserve_prior: bool,
    max_prior_delta_ratio: float,
) -> Any:
    if not isinstance(prior, LinearOperator):
        raise BlankBootstrapError("A linear fit requires a target operator prior.")
    if not preserve_prior:
        return fit_ridge_linear(
            input_rows,
            target_rows,
            ridge=ridge,
            fit_bias=prior.bias is not None,
        )
    return fit_prior_preserving_ridge_linear(
        input_rows,
        target_rows,
        prior.weight,
        prior_bias=prior.bias,
        ridge=ridge,
        fit_bias=prior.bias is not None,
        max_prior_delta_ratio=max_prior_delta_ratio,
    )


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
    report = {
        "method": "closed-form-parent-anchor-fit",
        "ridge": float(fit.ridge),
        "rows": metrics.rows,
        "relative_rmse": metrics.relative_rmse,
        "mean_cosine_similarity": metrics.mean_cosine_similarity,
        "finite": metrics.finite,
    }
    if hasattr(fit, "effective_ridge"):
        report.update(
            {
                "method": "trace-scaled-prior-preserving-parent-anchor-fit",
                "effective_ridge": float(fit.effective_ridge),
                "trace_scale": float(fit.trace_scale),
                "prior_norm": float(fit.prior_norm),
                "prior_delta_norm": float(fit.prior_delta_norm),
                "update_scale": float(fit.update_scale),
                "clipped": bool(fit.clipped),
            }
        )
    return report


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
