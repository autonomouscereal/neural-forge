#!/usr/bin/env python3
"""Run measured, zero-training 15%/20% variable-width LFM2 folds.

The local LFM2 parent and tokenizer are loaded only from explicit directories.
The sixteen sealed recovery trajectories are reused as forced-decode circuit
evidence; the twelve manual cases never influence allocation or correction and
form the binding whole-output promotion gate.

For each layer, held-out residual sensitivity is measured at a small width
frontier.  An exact-budget allocator then chooses fourteen unequal widths.
After planting real medoid producers, ``w2`` is corrected sequentially from
fresh child residual traces.  A candidate is retained only when both the
64-token residual-path gate and the strict whole-output gate pass.  No
checkpoint bytes are written before that decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from experiments import run_lfm2_long_horizon_repair as horizon
from experiments import run_lfm2_residual_attention_fold as residual
from experiments.run_lfm2_dynamic_catalog import fingerprint_local_tokenizer
from neural_forge.dynamic_lfm2_behavior import (
    Lfm2CachedWholeOutputBaseline,
    Lfm2WholeOutputBehaviorEvaluator,
    load_lfm2_cached_whole_output_baseline,
)
from neural_forge.hf_closed_form_fit import (
    fit_prior_preserving_ridge_linear,
    measure_fit,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_variable_width_fold import (
    Lfm2LayerWidthCurve,
    Lfm2VariableWidthCorrectionContext,
    Lfm2VariableWidthResidualCorrection,
    Lfm2WidthCostPoint,
    install_lfm2_variable_width_fold_plan,
    plan_lfm2_variable_width_fold,
    seal_lfm2_variable_width_evidence,
)
from neural_forge.hf_lfm2_long_horizon_anchors import (
    tag_lfm2_long_horizon_rows,
)


RUN_SCHEMA = "neural-forge-lfm2-variable-width-run-v1"
CURVE_SCHEMA = "neural-forge-lfm2-measured-width-curves-v1"
CHECKPOINT_SCHEMA = "neural-forge-lfm2-variable-width-checkpoint-v1"
TARGET_ATTEMPT_SCHEMA = "neural-forge-lfm2-variable-width-target-attempt-v1"
EXPECTED_PARENT_PARAMETERS = 229_693_184
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_SOURCE_WIDTH = 2560
EXPECTED_LAYER_COUNT = 14
DEFAULT_CURVE_WIDTHS = (2560, 2304, 2048, 1792, 1536, 1280, 1024, 768)
TARGET_FRACTIONS = (0.15, 0.20)
CORRECTION_RIDGES = (1e-8, 1e-6, 1e-4)

# These make each exact budget feasible.  They are measured like every other
# curve point and are never installed directly unless the measured allocator
# independently chooses them.
PROVISIONAL_FEASIBILITY_WIDTHS = {
    0.15: (
        1100,
        1100,
        1550,
        1550,
        1550,
        1550,
        2400,
        1650,
        2400,
        1650,
        1650,
        2400,
        1650,
        2425,
    ),
    0.20: (
        900,
        900,
        1250,
        1250,
        1250,
        1250,
        2100,
        1371,
        2100,
        1371,
        1372,
        2100,
        1372,
        2300,
    ),
}


class Lfm2VariableWidthRunError(RuntimeError):
    """The measured local-only run failed a binding or invariant."""


def _progress(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, **fields},
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


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
        raise Lfm2VariableWidthRunError(
            "Run provenance must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2VariableWidthRunError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _safe_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2VariableWidthRunError(f"{name} must be a local file.")
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2VariableWidthRunError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


def _safe_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2VariableWidthRunError(f"{name} must be a local directory.")
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2VariableWidthRunError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_target_attempt(
    target_root: Path,
    target: Mapping[str, Any],
    *,
    parent_identity_sha256: str,
    parent_cache_key: str,
    parent_trace_rows_sha256: str,
    evidence_sha256: str,
) -> Path:
    body = {
        "schema": TARGET_ATTEMPT_SCHEMA,
        "parent_identity_sha256": _digest(
            parent_identity_sha256,
            "parent_identity_sha256",
        ),
        "parent_cache_key": _digest(parent_cache_key, "parent_cache_key"),
        "parent_trace_rows_sha256": _digest(
            parent_trace_rows_sha256,
            "parent_trace_rows_sha256",
        ),
        "evidence_sha256": _digest(evidence_sha256, "evidence_sha256"),
        "target": dict(target),
    }
    document = {**body, "target_attempt_sha256": _json_sha256(body)}
    path = target_root / "attempt-report.json"
    _atomic_write_json(path, document)
    return path


def _strict_fraction(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2VariableWidthRunError(
            "Target fractions must be numeric."
        ) from exc
    if result not in TARGET_FRACTIONS:
        raise Lfm2VariableWidthRunError(
            "This measured frontier is fixed to 0.15 and/or 0.20."
        )
    return result


def target_parameter_budget(
    fraction: float,
    *,
    parent_parameters: int = EXPECTED_PARENT_PARAMETERS,
    hidden_size: int = EXPECTED_HIDDEN_SIZE,
) -> dict[str, Any]:
    """Return the nearest physically realizable coupled-neuron target."""

    selected = _strict_fraction(fraction)
    if parent_parameters <= 0 or hidden_size <= 0:
        raise Lfm2VariableWidthRunError(
            "Parent parameters and hidden size must be positive."
        )
    unit_savings = 3 * hidden_size
    removed_units = round(parent_parameters * selected / unit_savings)
    parameters_saved = removed_units * unit_savings
    return {
        "requested_fraction": selected,
        "unit_parameter_savings": unit_savings,
        "removed_neurons": removed_units,
        "parameters_saved": parameters_saved,
        "target_unique_parameters": parent_parameters - parameters_saved,
        "realized_fraction": parameters_saved / parent_parameters,
        "target_total_intermediate_width": (
            EXPECTED_LAYER_COUNT * EXPECTED_SOURCE_WIDTH - removed_units
        ),
    }


def candidate_widths_for_layer(
    layer_index: int,
    *,
    curve_widths: Sequence[int] = DEFAULT_CURVE_WIDTHS,
) -> tuple[int, ...]:
    """Measured frontier plus exact-budget feasibility anchors."""

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < EXPECTED_LAYER_COUNT
    ):
        raise Lfm2VariableWidthRunError("layer_index must be in 0..13.")
    values = set(int(value) for value in curve_widths)
    values.update(
        widths[layer_index]
        for widths in PROVISIONAL_FEASIBILITY_WIDTHS.values()
    )
    if EXPECTED_SOURCE_WIDTH not in values or any(
        value <= 0 or value > EXPECTED_SOURCE_WIDTH for value in values
    ):
        raise Lfm2VariableWidthRunError(
            "Curve widths must be positive contractions containing 2560."
        )
    return tuple(sorted(values, reverse=True))


def maximum_curve_step(
    *,
    curve_widths: Sequence[int] = DEFAULT_CURVE_WIDTHS,
) -> int:
    return max(
        left - right
        for layer_index in range(EXPECTED_LAYER_COUNT)
        for left, right in zip(
            candidate_widths_for_layer(
                layer_index,
                curve_widths=curve_widths,
            ),
            candidate_widths_for_layer(
                layer_index,
                curve_widths=curve_widths,
            )[1:],
        )
    )


def _index_rows(value: Any, indices: Sequence[int]) -> Any:
    index = torch.tensor(tuple(indices), dtype=torch.long, device=value.device)
    return value.index_select(0, index).detach().to(device="cpu").contiguous()


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


def build_measured_width_curves(
    parent: Any,
    parent_rows: Any,
    *,
    selection_indices: Sequence[int],
    validation_indices: Sequence[int],
    curve_widths: Sequence[int] = DEFAULT_CURVE_WIDTHS,
) -> tuple[
    tuple[Lfm2LayerWidthCurve, ...],
    tuple[Mapping[str, Any], ...],
    Mapping[int, Mapping[str, Any]],
]:
    """Measure activation/contribution residual cost without model training."""

    layers = tuple(parent.model.layers)
    if len(layers) != EXPECTED_LAYER_COUNT:
        raise Lfm2VariableWidthRunError(
            "The measured run requires fourteen LFM2 layers."
        )
    selection = tuple(selection_indices)
    validation = tuple(validation_indices)
    if not selection or not validation or set(selection) & set(validation):
        raise Lfm2VariableWidthRunError(
            "Curve selection and validation rows must be nonempty/disjoint."
        )
    curves = []
    reports = []
    evidence_rows = {}
    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            gate = _index_rows(
                parent_rows.field(layer_index, "mlp.gate"),
                selection,
            )
            up = _index_rows(
                parent_rows.field(layer_index, "mlp.up"),
                selection,
            )
            selection_post = _index_rows(
                parent_rows.field(layer_index, "mlp.post_swiglu"),
                selection,
            ).to(torch.float64)
            validation_post = _index_rows(
                parent_rows.field(layer_index, "mlp.post_swiglu"),
                validation,
            ).to(torch.float64)
            parent_branch = (
                _index_rows(
                    parent_rows.field(layer_index, "residual.post_layer"),
                    validation,
                )
                - _index_rows(
                    parent_rows.field(
                        layer_index,
                        "residual.post_operator",
                    ),
                    validation,
                )
            ).to(torch.float64)
            down = layer.feed_forward.w2.weight.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            importance = (
                selection_post.square().mean(dim=0).sqrt()
                * torch.linalg.vector_norm(down, dim=0)
            )
            ordered = tuple(
                sorted(
                    range(EXPECTED_SOURCE_WIDTH),
                    key=lambda index: (
                        -float(importance[index].item()),
                        index,
                    ),
                )
            )
            points = []
            measurements = []
            for width in candidate_widths_for_layer(
                layer_index,
                curve_widths=curve_widths,
            ):
                if width == EXPECTED_SOURCE_WIDTH:
                    relative_rmse = 0.0
                    mean_cosine = 1.0
                    metrics = None
                else:
                    kept = torch.tensor(
                        tuple(sorted(ordered[:width])),
                        dtype=torch.long,
                    )
                    prediction = validation_post.index_select(
                        1,
                        kept,
                    ).matmul(
                        down.index_select(1, kept).transpose(0, 1)
                    )
                    metrics = measure_fit(prediction, parent_branch)
                    relative_rmse = metrics.relative_rmse
                    mean_cosine = metrics.mean_cosine_similarity
                residual_cost = max(0.0, float(relative_rmse))
                trajectory_cost = residual_cost + max(
                    0.0,
                    1.0 - float(mean_cosine),
                )
                points.append(
                    Lfm2WidthCostPoint(
                        width=width,
                        whole_trajectory_cost=trajectory_cost,
                        residual_cost=residual_cost,
                    )
                )
                measurements.append(
                    {
                        "width": width,
                        "cost_source": (
                            "measured_parent_activation_contribution_residual"
                        ),
                        "validation_metrics": (
                            {
                                "rows": len(validation),
                                "output_width": EXPECTED_HIDDEN_SIZE,
                                "rmse": 0.0,
                                "target_rms": 0.0,
                                "relative_rmse": 0.0,
                                "mean_cosine_similarity": 1.0,
                                "finite": True,
                            }
                            if metrics is None
                            else _metrics_dict(metrics)
                        ),
                        "whole_trajectory_proxy_cost": trajectory_cost,
                        "residual_cost": residual_cost,
                    }
                )
            curves.append(
                Lfm2LayerWidthCurve(
                    layer_index=layer_index,
                    points=tuple(points),
                )
            )
            reports.append(
                {
                    "layer_index": layer_index,
                    "selection_row_count": len(selection),
                    "validation_row_count": len(validation),
                    "importance_ranking_sha256": _json_sha256(list(ordered)),
                    "measurements": measurements,
                }
            )
            evidence_rows[layer_index] = {
                "gate": gate,
                "up": up,
                "post_swiglu": selection_post.to(
                    dtype=parent_rows.field(
                        layer_index,
                        "mlp.post_swiglu",
                    ).dtype
                ),
            }
    return tuple(curves), tuple(reports), evidence_rows


def _tensorize_horizon_batches(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    result = []
    for row in rows:
        batch = {
            "input_ids": torch.tensor(
                row["input_ids"],
                dtype=torch.long,
            ).unsqueeze(0),
            "forced_decode_ids": torch.tensor(
                row["forced_decode_ids"],
                dtype=torch.long,
            ).unsqueeze(0),
        }
        if "attention_mask" in row:
            batch["attention_mask"] = torch.tensor(
                row["attention_mask"],
                dtype=torch.long,
            ).unsqueeze(0)
        result.append(batch)
    return tuple(result)


def _manual_baseline(
    baseline: Lfm2CachedWholeOutputBaseline,
    manual_case_ids: Sequence[str],
) -> Lfm2CachedWholeOutputBaseline:
    selected = set(manual_case_ids)
    cases = tuple(case for case in baseline.cases if case.case_id in selected)
    contracts = tuple(
        value for value in baseline.contracts if value[0] in selected
    )
    if len(cases) != len(selected) or len(contracts) != len(selected):
        raise Lfm2VariableWidthRunError(
            "The sealed manual promotion cases are absent from the baseline."
        )
    return replace(baseline, cases=cases, contracts=contracts)


@dataclass(frozen=True, slots=True)
class SequentialCorrectionConfig:
    child_cache_root: Path
    batches: tuple[Mapping[str, Any], ...]
    capture_config: Lfm2CircuitTraceCaptureConfig
    parent_rows: Any
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    ridges: tuple[float, ...] = CORRECTION_RIDGES
    max_prior_delta_ratio: float = 1.0


class SequentialW2Corrector:
    """Recapture the complete child and accept only validation-improving w2."""

    def __init__(self, config: SequentialCorrectionConfig) -> None:
        self.config = config
        self.reports: list[dict[str, Any]] = []

    def __call__(
        self,
        context: Lfm2VariableWidthCorrectionContext,
    ) -> Lfm2VariableWidthResidualCorrection | None:
        cache, rows = residual.build_child_trace_cache(
            self.config.child_cache_root,
            context.child,
            self.config.batches,
            capture_config=self.config.capture_config,
        )
        self.config.parent_rows.assert_aligned(rows)
        layer_index = context.layer_index
        fit = self.config.fit_indices
        validation = self.config.validation_indices
        fit_design = _index_rows(
            rows.field(layer_index, "mlp.post_swiglu"),
            fit,
        ).to(torch.float64)
        fit_target = (
            _index_rows(
                self.config.parent_rows.field(
                    layer_index,
                    "residual.post_layer",
                ),
                fit,
            )
            - _index_rows(
                rows.field(layer_index, "residual.post_operator"),
                fit,
            )
        ).to(torch.float64)
        validation_design = _index_rows(
            rows.field(layer_index, "mlp.post_swiglu"),
            validation,
        ).to(torch.float64)
        validation_target = (
            _index_rows(
                self.config.parent_rows.field(
                    layer_index,
                    "residual.post_layer",
                ),
                validation,
            )
            - _index_rows(
                rows.field(layer_index, "residual.post_operator"),
                validation,
            )
        ).to(torch.float64)
        prior = context.child.model.layers[
            layer_index
        ].feed_forward.w2.weight.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        prior_metrics = measure_fit(
            validation_design.matmul(prior.transpose(0, 1)),
            validation_target,
        )
        attempts = []
        candidates = []
        for ridge in self.config.ridges:
            result = fit_prior_preserving_ridge_linear(
                fit_design,
                fit_target,
                prior,
                ridge=ridge,
                fit_bias=False,
                max_prior_delta_ratio=self.config.max_prior_delta_ratio,
                validation_input_rows=validation_design,
                validation_target_rows=validation_target,
            )
            metrics = result.validation_metrics or result.train_metrics
            attempts.append(
                {
                    "ridge": ridge,
                    "train_metrics": _metrics_dict(result.train_metrics),
                    "validation_metrics": _metrics_dict(metrics),
                    "prior_delta_norm": result.prior_delta_norm,
                    "prior_norm": result.prior_norm,
                    "update_scale": result.update_scale,
                    "clipped": result.clipped,
                }
            )
            candidates.append((metrics, ridge, result))
        metrics, ridge, selected = min(
            candidates,
            key=lambda value: (
                value[0].relative_rmse,
                -value[0].mean_cosine_similarity,
                value[1],
            ),
        )
        accepted = bool(
            metrics.finite
            and metrics.relative_rmse <= prior_metrics.relative_rmse
            and metrics.mean_cosine_similarity
            >= prior_metrics.mean_cosine_similarity - 0.01
        )
        evidence_body = {
            "schema": "neural-forge-lfm2-variable-width-w2-correction-v1",
            "plan_sha256": context.plan_sha256,
            "layer_index": layer_index,
            "child_cache_key": cache.cache_key,
            "child_trace_rows_sha256": rows.trace_rows_sha256,
            "parent_trace_rows_sha256": (
                self.config.parent_rows.trace_rows_sha256
            ),
            "fit_indices_sha256": _json_sha256(list(fit)),
            "validation_indices_sha256": _json_sha256(list(validation)),
            "selected_ridge": ridge,
            "accepted": accepted,
        }
        evidence_sha256 = _json_sha256(evidence_body)
        self.reports.append(
            {
                **evidence_body,
                "evidence_sha256": evidence_sha256,
                "prior_validation_metrics": _metrics_dict(prior_metrics),
                "attempts": attempts,
                "selected_validation_metrics": _metrics_dict(metrics),
            }
        )
        if not accepted:
            return None
        return Lfm2VariableWidthResidualCorrection(
            down_weight=selected.weight,
            evidence_sha256=evidence_sha256,
            method_id=(
                "sequential_child_recapture_prior_preserving_ridge_w2"
            ),
        )


def _strict_behavior_evaluator(
    baseline: Lfm2CachedWholeOutputBaseline,
    tokenizer: Any,
) -> Lfm2WholeOutputBehaviorEvaluator:
    return Lfm2WholeOutputBehaviorEvaluator(
        baseline=baseline,
        tokenizer=tokenizer,
        maximum_prompt_tokens=512,
        fragment_minimum_characters=12,
        maximum_new_empty_responses=0,
        maximum_new_fragment_responses=0,
        minimum_parent_nonempty_retention=0.90,
        minimum_task_score_retention=0.90,
        maximum_parent_task_score_regressions=1,
        maximum_new_generation_limit_saturations=1,
        maximum_repeated_ngram_share_regressions=1,
        maximum_repeated_bigram_share_delta=0.05,
    )


def _publish_passed_checkpoint(
    child: Any,
    destination: Path,
    *,
    parent_model: Path,
    fraction: float,
    plan: Any,
    report: Any,
) -> dict[str, Any]:
    """Serialize only an already-promoted custom nonuniform in-memory child."""

    from safetensors.torch import save_file

    if destination.exists():
        raise Lfm2VariableWidthRunError(
            f"Refusing to replace checkpoint destination: {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        weights = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in child.state_dict().items()
        }
        weights_path = stage / "model.safetensors"
        save_file(weights, str(weights_path))
        body = {
            "schema": CHECKPOINT_SCHEMA,
            "base_model_directory": str(parent_model),
            "fraction": fraction,
            "target_widths": list(plan.target_widths),
            "plan_sha256": plan.plan_sha256,
            "candidate_runtime_sha256": report.candidate_runtime_sha256,
            "unique_parameters": report.unique_parameters_after,
            "weights_file": weights_path.name,
            "weights_sha256": _file_sha256(weights_path),
            "custom_variable_width_loader_required": True,
            "standard_hf_config_is_not_sufficient": True,
            "strict_promotion_passed_before_serialization": True,
        }
        manifest = {**body, "manifest_sha256": _json_sha256(body)}
        _atomic_write_json(stage / "manifest.json", manifest)
        os.rename(stage, destination)
        return {**manifest, "path": str(destination)}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@dataclass(frozen=True, slots=True)
class VariableWidthRunConfig:
    parent_model: Path
    cached_parent_outputs: Path
    cached_parent_outputs_sha256: str
    parent_horizon_cache_root: Path
    work_root: Path
    report_path: Path
    checkpoint_root: Path | None
    target_fractions: tuple[float, ...] = TARGET_FRACTIONS
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    publish_strict_passes: bool = False
    sample_last_tokens: int = 8
    max_capture_bytes: int = 512 * 1024 * 1024
    capture_logit_rtol: float = 0.20
    capture_logit_atol: float = 1.0
    cluster_ridge: float = 1e-8
    cluster_max_iterations: int = 8
    cluster_distance_chunk_size: int = 512
    max_prior_delta_ratio: float = 1.0
    keep_child_caches: bool = False


def execute_variable_width_run(
    config: VariableWidthRunConfig,
) -> dict[str, Any]:
    """Measure, allocate, fold, correct, and strictly gate 15%/20% targets."""

    parent_directory = _safe_directory(config.parent_model, "parent_model")
    envelope_path = _safe_file(
        config.cached_parent_outputs,
        "cached_parent_outputs",
    )
    envelope_file_sha = _file_sha256(envelope_path)
    if envelope_file_sha != _digest(
        config.cached_parent_outputs_sha256,
        "cached_parent_outputs_sha256",
    ):
        raise Lfm2VariableWidthRunError(
            "The cached parent-output envelope changed."
        )
    if config.report_path.exists():
        raise Lfm2VariableWidthRunError(
            f"Refusing to replace run report: {config.report_path}."
        )
    if config.publish_strict_passes and config.checkpoint_root is None:
        raise Lfm2VariableWidthRunError(
            "checkpoint_root is required when publishing strict passes."
        )
    fractions = tuple(_strict_fraction(value) for value in config.target_fractions)
    if not fractions or len(set(fractions)) != len(fractions):
        raise Lfm2VariableWidthRunError(
            "target_fractions must be a nonempty unique subset."
        )
    config.parent_horizon_cache_root.mkdir(parents=True, exist_ok=True)
    config.work_root.mkdir(parents=True, exist_ok=True)

    dtype = residual._torch_dtype(config.dtype)
    device = torch.device(config.device)
    tokenizer = residual.load_local_tokenizer(parent_directory)
    tokenizer_fingerprint = fingerprint_local_tokenizer(parent_directory)
    envelope = horizon.audit_sealed_horizon_envelope(
        envelope_path,
        expected_sha256=envelope_file_sha,
        token_text_decoder=lambda token_ids: horizon._decode_token_pieces(
            tokenizer,
            token_ids,
        ),
        tokenizer_eos_token_ids=horizon._tokenizer_eos_ids(tokenizer),
    )
    if not envelope.ready:
        body = {
            "schema": RUN_SCHEMA,
            "status": "data_contract_blocked",
            "sealed_envelope": envelope.as_dict(),
            "models_loaded": False,
            "checkpoint_published": False,
            "training_steps": 0,
        }
        document = {**body, "report_sha256": _json_sha256(body)}
        _atomic_write_json(config.report_path, document)
        return document
    if envelope.tokenizer_fingerprint != tokenizer_fingerprint:
        raise Lfm2VariableWidthRunError(
            "The local tokenizer differs from the sealed envelope."
        )
    _progress(
        "sealed_envelope_ready",
        recovery_cases=len(envelope.recovery_case_ids),
        promotion_cases=len(envelope.manual_case_ids),
    )
    batch_records = horizon.build_horizon_batch_records(envelope)
    batches = _tensorize_horizon_batches(batch_records)
    parent_capture_config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=config.sample_last_tokens,
        max_decode_steps=horizon.MAX_DECODE_STEPS,
        max_capture_bytes=config.max_capture_bytes,
        logit_rtol=config.capture_logit_rtol,
        logit_atol=config.capture_logit_atol,
        require_top1_agreement=False,
    )
    child_capture_config = replace(
        parent_capture_config,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    parent = residual.load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256
    parent_parameters = residual.unique_parameter_count(parent)
    if (
        parent_identity != envelope.parent_state_fingerprint
        or parent_parameters != EXPECTED_PARENT_PARAMETERS
    ):
        raise Lfm2VariableWidthRunError(
            "The loaded parent differs from the sealed LFM2.5-230M contract."
        )
    cache_keys_before = {
        path.name
        for path in config.parent_horizon_cache_root.iterdir()
        if path.is_dir()
    }
    parent_cache = build_lfm2_circuit_trace_cache(
        config.parent_horizon_cache_root,
        parent,
        batches,
        config=parent_capture_config,
        input_device=residual._model_device(parent),
    )
    parent_cache_reused = parent_cache.cache_key in cache_keys_before
    _progress(
        "parent_horizon_cache_ready",
        cache_key=parent_cache.cache_key,
        reused=parent_cache_reused,
    )
    parent_rows = residual.extract_trace_rows(parent_cache)
    anchor_envelope = horizon.build_anchor_case_envelope(envelope)
    descriptors = horizon.trace_path_descriptors_from_paths(
        parent_rows.paths,
        envelope,
    )
    tagging = tag_lfm2_long_horizon_rows(anchor_envelope, descriptors)
    anchor_plan = horizon.build_repair_anchor_plan(
        anchor_envelope,
        descriptors,
        tagging=tagging,
    )
    profile = horizon.build_evidence_profiles(anchor_plan, tagging)[0]
    curves, curve_reports, evidence_rows = build_measured_width_curves(
        parent,
        parent_rows,
        selection_indices=profile.selection_indices,
        validation_indices=profile.validation_indices,
    )
    _progress(
        "measured_width_curves_ready",
        layer_count=len(curves),
        curve_point_count=sum(len(curve.points) for curve in curves),
    )
    evidence = seal_lfm2_variable_width_evidence(
        parent,
        evidence_rows,
        curves,
        cache_key=parent_cache.cache_key,
        heldout_corpus_sha256=parent_cache.corpus_fingerprint,
        evidence_builder_id=(
            "lfm2.long_horizon.measured_residual_width_curves.v1"
        ),
    )
    baseline = load_lfm2_cached_whole_output_baseline(
        envelope_path,
        expected_sha256=envelope_file_sha,
        expected_parent_cache_key=envelope.parent_cache_key,
        expected_parent_state_fingerprint=parent_identity,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_generation_limit=horizon.MAX_DECODE_STEPS,
        fragment_minimum_characters=12,
    )
    promotion_baseline = _manual_baseline(
        baseline,
        envelope.manual_case_ids,
    )
    targets = []
    published_any = False
    curve_step = maximum_curve_step()

    for fraction in fractions:
        budget = target_parameter_budget(fraction)
        _progress(
            "target_started",
            fraction=fraction,
            parameters_saved=budget["parameters_saved"],
        )
        target_root = config.work_root / f"target-{int(fraction * 100)}pct"
        child_cache_root = target_root / "child-circuit-cache"
        child_cache_root.mkdir(parents=True, exist_ok=True)
        transaction = None
        child = None
        target_result = None
        try:
            plan = plan_lfm2_variable_width_fold(
                parent,
                evidence,
                target_parameters_saved=budget["parameters_saved"],
                minimum_width=min(DEFAULT_CURVE_WIDTHS),
                maximum_width_step=curve_step,
                whole_trajectory_weight=1.0,
                residual_weight=1.0,
                cluster_ridge=config.cluster_ridge,
                cluster_max_iterations=config.cluster_max_iterations,
                cluster_distance_chunk_size=(
                    config.cluster_distance_chunk_size
                ),
            )
            _progress(
                "target_allocation_ready",
                fraction=fraction,
                target_widths=list(plan.target_widths),
                plan_sha256=plan.plan_sha256,
            )
            child = residual.load_local_lfm2_model(
                parent_directory,
                device=device,
                dtype=dtype,
            )
            corrector = SequentialW2Corrector(
                SequentialCorrectionConfig(
                    child_cache_root=child_cache_root,
                    batches=batches,
                    capture_config=child_capture_config,
                    parent_rows=parent_rows,
                    fit_indices=tuple(profile.fit_indices),
                    validation_indices=tuple(profile.validation_indices),
                    max_prior_delta_ratio=config.max_prior_delta_ratio,
                )
            )
            transaction = install_lfm2_variable_width_fold_plan(
                parent,
                child,
                evidence,
                plan,
                residual_corrector=corrector,
            )
            final_cache, child_rows = residual.build_child_trace_cache(
                child_cache_root,
                child,
                batches,
                capture_config=child_capture_config,
            )
            trajectory = residual.evaluate_trajectory_gate(
                parent_cache,
                final_cache,
                child,
                parent_rows,
                child_rows,
                config=residual.TrajectoryGateConfig(
                    min_mean_residual_cosine=0.95,
                    logit_rtol=config.capture_logit_rtol,
                    logit_atol=config.capture_logit_atol,
                ),
            )
            evaluator = _strict_behavior_evaluator(
                promotion_baseline,
                tokenizer,
            )
            whole_output = evaluator.evaluate_model(
                child,
                phase=(
                    f"variable_width_{int(fraction * 100)}pct_"
                    "manual_only_strict"
                ),
            )
            strict_pass = bool(
                trajectory["passed"] and whole_output.accepted
            )
            checkpoint = None
            if strict_pass:
                transaction.commit()
                if config.publish_strict_passes:
                    assert config.checkpoint_root is not None
                    checkpoint = _publish_passed_checkpoint(
                        child,
                        config.checkpoint_root
                        / f"lfm2-variable-{int(fraction * 100)}pct",
                        parent_model=parent_directory,
                        fraction=fraction,
                        plan=plan,
                        report=transaction.report,
                    )
                    published_any = True
            else:
                transaction.rollback()
            _progress(
                "target_finished",
                fraction=fraction,
                status="accepted" if strict_pass else "rejected",
                trajectory_passed=trajectory["passed"],
                whole_output_accepted=whole_output.accepted,
                checkpoint_published=checkpoint is not None,
            )
            target_result = {
                "fraction": fraction,
                "status": "accepted" if strict_pass else "rejected",
                "budget": budget,
                "allocation": plan.allocation.as_dict(),
                "target_widths": list(plan.target_widths),
                "matches_provisional_feasibility_vector": (
                    plan.target_widths
                    == PROVISIONAL_FEASIBILITY_WIDTHS[fraction]
                ),
                "provisional_vector_was_directly_forced": False,
                "plan_sha256": plan.plan_sha256,
                "fold_report": transaction.report.as_dict(),
                "sequential_w2_corrections": corrector.reports,
                "trajectory_gate": trajectory,
                "whole_output_gate": whole_output.as_dict(),
                "whole_output_events": evaluator.events,
                "promotion_case_ids": list(envelope.manual_case_ids),
                "development_case_ids": list(envelope.recovery_case_ids),
                "checkpoint": checkpoint,
                "checkpoint_published": checkpoint is not None,
            }
        except Exception as exc:
            if transaction is not None and transaction.active:
                transaction.rollback()
            target_result = {
                "fraction": fraction,
                "status": "failed",
                "budget": budget,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "checkpoint_published": False,
            }
            _progress(
                "target_failed",
                fraction=fraction,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        finally:
            if child is not None:
                del child
            if not config.keep_child_caches and child_cache_root.exists():
                shutil.rmtree(child_cache_root)
        if target_result is None:
            raise Lfm2VariableWidthRunError(
                f"Target {fraction} produced no attempt result."
            )
        targets.append(target_result)
        attempt_path = _write_target_attempt(
            target_root,
            target_result,
            parent_identity_sha256=parent_identity,
            parent_cache_key=parent_cache.cache_key,
            parent_trace_rows_sha256=parent_rows.trace_rows_sha256,
            evidence_sha256=evidence.evidence_sha256,
        )
        _progress(
            "target_attempt_preserved",
            fraction=fraction,
            attempt_path=str(attempt_path),
        )

    curve_body = {
        "schema": CURVE_SCHEMA,
        "parent_cache_key": parent_cache.cache_key,
        "parent_trace_rows_sha256": parent_rows.trace_rows_sha256,
        "anchor_plan_sha256": anchor_plan.plan_sha256,
        "profile": profile.as_dict(),
        "curve_reports": list(curve_reports),
        "provisional_vectors_are_feasibility_points_only": True,
        "allocator_uses_measured_costs": True,
    }
    body = {
        "schema": RUN_SCHEMA,
        "status": (
            "accepted"
            if any(target["status"] == "accepted" for target in targets)
            else "rejected"
        ),
        "local_model_io": {
            "parent_model": str(parent_directory),
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ[
                    "TRANSFORMERS_OFFLINE"
                ],
            },
            "hugging_face_hub_loader_used": False,
            "network_model_resolution_allowed": False,
        },
        "parent": {
            "identity_sha256": parent_identity,
            "unique_parameters": parent_parameters,
            "horizon_cache_key": parent_cache.cache_key,
            "horizon_cache_reused": parent_cache_reused,
            "new_parent_response_generation_calls": 0,
            "new_parent_trace_forwards": 0 if parent_cache_reused else 1,
        },
        "sealed_envelope": envelope.as_dict(),
        "measured_curves": {
            **curve_body,
            "curves_sha256": _json_sha256(curve_body),
        },
        "evidence_sha256": evidence.evidence_sha256,
        "targets": targets,
        "strict_promotion": {
            "whole_output_case_scope": "manual-only-12",
            "minimum_nonempty_retention": 0.90,
            "minimum_task_score_retention": 0.90,
            "maximum_task_regressions": 1,
            "maximum_new_saturations": 1,
            "maximum_repetition_regressions": 1,
            "minimum_mean_residual_cosine": 0.95,
            "exact_token_match_is_diagnostic_only": True,
        },
        "trace_capture": {
            "parent_decode_trace_mode": "cached_verified",
            "parent_cache_reused": parent_cache_reused,
            "child_decode_trace_mode": FULL_PREFIX_CANONICAL_TRACE_MODE,
            "child_cached_logits_accepted": False,
            "capture_logit_rtol": config.capture_logit_rtol,
            "capture_logit_atol": config.capture_logit_atol,
            "parity_tolerances_relaxed": False,
        },
        "checkpoint_published": published_any,
        "training": {
            "training_steps": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
        },
    }
    document = {**body, "report_sha256": _json_sha256(body)}
    _atomic_write_json(config.report_path, document)
    return document


def _parse_fractions(value: str) -> tuple[float, ...]:
    try:
        result = tuple(_strict_fraction(item) for item in value.split(","))
    except Lfm2VariableWidthRunError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "target fractions must be unique comma-separated values"
        )
    return result


def _parse_ridges(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "correction ridges must be comma-separated floats"
        ) from exc
    if not result or any(not math.isfinite(item) or item < 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            "correction ridges must be finite and non-negative"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measured zero-training variable-width LFM2 folds at real 15% "
            "and 20% physical parameter removal."
        )
    )
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--cached-parent-outputs", required=True, type=Path)
    parser.add_argument("--cached-parent-outputs-sha256", required=True)
    parser.add_argument(
        "--parent-horizon-cache-root",
        required=True,
        type=Path,
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--target-fractions",
        type=_parse_fractions,
        default=TARGET_FRACTIONS,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--publish-strict-passes", action="store_true")
    parser.add_argument("--sample-last-tokens", type=int, default=8)
    parser.add_argument(
        "--max-capture-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--capture-logit-rtol", type=float, default=0.20)
    parser.add_argument("--capture-logit-atol", type=float, default=1.0)
    parser.add_argument("--cluster-ridge", type=float, default=1e-8)
    parser.add_argument("--cluster-max-iterations", type=int, default=8)
    parser.add_argument(
        "--cluster-distance-chunk-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--correction-ridges",
        type=_parse_ridges,
        default=CORRECTION_RIDGES,
    )
    parser.add_argument("--max-prior-delta-ratio", type=float, default=1.0)
    parser.add_argument("--keep-child-caches", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> VariableWidthRunConfig:
    # The fixed ridge grid is intentionally parsed/audited at the CLI even
    # though this first runner keeps its authenticated default.
    if tuple(args.correction_ridges) != CORRECTION_RIDGES:
        raise Lfm2VariableWidthRunError(
            "This first real run uses the fixed correction ridge grid "
            "1e-8,1e-6,1e-4."
        )
    return VariableWidthRunConfig(
        parent_model=args.parent_model,
        cached_parent_outputs=args.cached_parent_outputs,
        cached_parent_outputs_sha256=args.cached_parent_outputs_sha256,
        parent_horizon_cache_root=args.parent_horizon_cache_root,
        work_root=args.work_root,
        report_path=args.report_path,
        checkpoint_root=args.checkpoint_root,
        target_fractions=tuple(args.target_fractions),
        device=args.device,
        dtype=args.dtype,
        publish_strict_passes=args.publish_strict_passes,
        sample_last_tokens=args.sample_last_tokens,
        max_capture_bytes=args.max_capture_bytes,
        capture_logit_rtol=args.capture_logit_rtol,
        capture_logit_atol=args.capture_logit_atol,
        cluster_ridge=args.cluster_ridge,
        cluster_max_iterations=args.cluster_max_iterations,
        cluster_distance_chunk_size=args.cluster_distance_chunk_size,
        max_prior_delta_ratio=args.max_prior_delta_ratio,
        keep_child_caches=args.keep_child_caches,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = execute_variable_width_run(_config_from_args(args))
    print(
        json.dumps(
            {
                "status": report["status"],
                "targets": [
                    {
                        "fraction": target["fraction"],
                        "status": target["status"],
                        "target_widths": target.get("target_widths"),
                    }
                    for target in report.get("targets", ())
                ],
                "checkpoint_published": report["checkpoint_published"],
                "report_sha256": report["report_sha256"],
                "report_path": str(args.report_path),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
