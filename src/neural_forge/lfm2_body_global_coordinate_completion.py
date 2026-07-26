"""Pure body-global residual coordinate completion for the LFM2 frontier.

The family-facing contract in this module is intentionally small.  It consumes
only cached FIT-partition residual rows, an authenticated interface encoder,
and immutable residual-facing parent operators.  It first constructs the
prior-centred least-squares target-to-parent decoder

``pinv(E) + pinv(X) @ (Y - X @ pinv(E))``

where ``E`` is the parent-to-child producer encoder.  The fitted action of that
raw decoder is then passed to the architecture-neutral analytic coordinate
completion frontier.  Completion may change only directions not identified by
``X``; the selected result is therefore full rank without changing its cached
FIT-row predictions.

No model is loaded or executed here.  The compiler performs no trace capture,
training, optimization, pruning, quantization, random initialization, or
filesystem writes.  Numeric diagnostics are evidence for later decoded-output
review, never semantic quality gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from neural_forge.analytic_coordinate_completion import (
    AnalyticCoordinateCompletionCandidate,
    AnalyticCoordinateCompletionError,
    AnalyticCoordinateCompletionFrontier,
    analytic_coordinate_completion_evidence_sha256,
    compile_analytic_coordinate_completion_frontier,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_weight_complete_basis import ResidualFacingOperator


LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-body-global-coordinate-completion-evidence-v1"
)
LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-body-global-coordinate-completion-receipt-v1"
)
LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES = ("bridge-procrustes-isometric",)

_PSEUDOINVERSE_RELATIVE_TOLERANCE = 1e-12
_COMPLETION_RANK_RELATIVE_TOLERANCE = 1e-12
_COMPLETION_DEGENERACY_RELATIVE_TOLERANCE = 1e-11
_NUMERICAL_LOCK = threading.RLock()
_HEX_DIGITS = frozenset("0123456789abcdef")


class Lfm2BodyGlobalCoordinateCompletionError(AnalyticCoordinateCompletionError):
    """Cached body-global evidence or recipe selection is invalid."""


@dataclass(frozen=True, slots=True)
class Lfm2BodyGlobalCoordinateCompletion:
    """One selected raw decoder, its full frontier, and sealed receipt."""

    raw_decoder: torch.Tensor
    frontier: AnalyticCoordinateCompletionFrontier
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        decoder = _matrix(self.raw_decoder, "raw_decoder")
        if not isinstance(self.frontier, AnalyticCoordinateCompletionFrontier):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "frontier must be an AnalyticCoordinateCompletionFrontier."
            )
        if not isinstance(self.receipt, Mapping):
            raise Lfm2BodyGlobalCoordinateCompletionError("receipt must be a mapping.")
        receipt = dict(self.receipt)
        if receipt.get("schema") != (
            LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA
        ):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "receipt schema is not the body-global completion schema."
            )
        if receipt.get("raw_decoder_sha256") != _tensor_sha256(decoder):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "receipt does not authenticate raw_decoder."
            )
        expected_rank = min(map(int, decoder.shape))
        if (
            receipt.get("coordinate_map_sha256") != _tensor_sha256(decoder)
            or receipt.get("public_coordinate_map_sha256")
            != _authenticated_coordinate_map_sha256(decoder)
            or receipt.get("coordinate_map_shape")
            != [int(size) for size in decoder.shape]
            or receipt.get("required_full_rank") != expected_rank
            or receipt.get("structural_rank") != expected_rank
            or receipt.get("materialized_rank") != expected_rank
            or receipt.get("full_rank_materialized") is not True
        ):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "receipt does not authenticate the public full-rank coordinate "
                "map contract."
            )
        if receipt.get("analytic_frontier_result_sha256") != (
            self.frontier.report.get("result_sha256")
        ):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "receipt does not authenticate the analytic frontier."
            )
        try:
            selected_candidate = self.frontier.candidate(
                str(receipt.get("materialized_candidate_recipe"))
            )
        except AnalyticCoordinateCompletionError as exc:
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "receipt names no materialized analytic candidate."
            ) from exc
        if (
            selected_candidate.report.get("result_sha256")
            != receipt.get("selected_candidate_result_sha256")
            or selected_candidate.report.get("coordinate_map_sha256")
            != receipt.get("coordinate_map_sha256")
            or not torch.equal(selected_candidate.coordinate_map, decoder)
        ):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "receipt does not bind the selected analytic candidate."
            )
        expected_result = receipt.get("result_sha256")
        unsealed = {
            key: value for key, value in receipt.items() if key != "result_sha256"
        }
        if expected_result != canonical_sha256(unsealed):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "receipt result seal is invalid."
            )
        object.__setattr__(self, "raw_decoder", decoder)
        object.__setattr__(self, "receipt", MappingProxyType(receipt))

    @property
    def target_width(self) -> int:
        return int(self.raw_decoder.shape[0])

    @property
    def source_width(self) -> int:
        return int(self.raw_decoder.shape[1])

    @property
    def recipe(self) -> str:
        return str(self.receipt["requested_recipe"])

    def decode(self, child_residual_rows: torch.Tensor) -> torch.Tensor:
        """Apply the selected raw target-to-parent coordinate decoder."""

        rows = _floating_tensor(child_residual_rows, "child_residual_rows")
        if int(rows.shape[-1]) != self.target_width:
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "child_residual_rows final axis does not match target_width."
            )
        return (rows @ self.raw_decoder).contiguous()

    def materialize(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: Any = "cpu",
    ) -> torch.Tensor:
        """Return a detached dense copy for a family-specific installer."""

        if not isinstance(dtype, torch.dtype):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "dtype must be a torch dtype."
            )
        result = self.raw_decoder.to(device=device, dtype=dtype).contiguous()
        if not bool(torch.isfinite(result).all().item()):
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "Materialized raw decoder is not finite."
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.receipt)


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _floating_tensor(value: Any, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.ndim < 1
        or any(int(size) <= 0 for size in value.shape)
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            f"{label} must be one materialized dense real floating tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            f"{label} contains non-finite values."
        )
    return result


def _matrix(value: Any, label: str) -> torch.Tensor:
    result = _floating_tensor(value, label)
    if result.ndim != 2:
        raise Lfm2BodyGlobalCoordinateCompletionError(f"{label} must be rank two.")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _authenticated_coordinate_map_sha256(value: torch.Tensor) -> str:
    """Return the public semantic-inventory digest for an installed map."""

    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _recipe(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "recipe must be one explicit non-empty recipe name."
        )
    result = value.strip()
    if result not in LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Unsupported body-global completion recipe: " + result
        )
    return result


def _operator_snapshots(
    parent_operators: Sequence[ResidualFacingOperator],
) -> tuple[ResidualFacingOperator, ...]:
    values = tuple(parent_operators)
    if not values or any(
        not isinstance(value, ResidualFacingOperator) for value in values
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "parent_operators must contain ResidualFacingOperator values."
        )
    snapshots: list[ResidualFacingOperator] = []
    for value in values:
        try:
            snapshot = ResidualFacingOperator(
                role=value.role,
                name=value.name,
                tensor=value.tensor,
                orientation=value.orientation,
                feature_axis=value.feature_axis,
                gram_weight=value.gram_weight,
                diagonal=value.diagonal,
            )
        except ValueError as exc:
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "A parent residual-facing operator is no longer valid."
            ) from exc
        if snapshot.operator_sha256 != value.operator_sha256:
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "A parent residual-facing operator changed after authentication."
            )
        snapshots.append(snapshot)
    ordered = tuple(
        sorted(
            snapshots,
            key=lambda item: (
                item.role,
                item.name,
                item.orientation.value,
                item.feature_axis,
                item.operator_sha256,
            ),
        )
    )
    if len({value.operator_sha256 for value in ordered}) != len(ordered):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Duplicate parent residual-facing operators are not allowed."
        )
    if len({value.source_width for value in ordered}) != 1:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Every parent operator must share one source width."
        )
    return ordered


def _validate_shapes(
    child_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    encoder: torch.Tensor,
    operators: Sequence[ResidualFacingOperator],
) -> tuple[int, int, int]:
    row_count, target_width = map(int, child_rows.shape)
    parent_row_count, source_width = map(int, parent_rows.shape)
    if parent_row_count != row_count:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "FIT child and parent residual rows must have the same row count."
        )
    if tuple(encoder.shape) != (source_width, target_width):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "producer_encoder must have shape [source_width, target_width]."
        )
    if operators[0].source_width != source_width:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Parent operator width must match parent residual source width."
        )
    return row_count, target_width, source_width


def _evidence_sha256_from_snapshots(
    child_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    encoder: torch.Tensor,
    operators: Sequence[ResidualFacingOperator],
    *,
    recipe: str,
) -> str:
    body = {
        "schema": LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_EVIDENCE_SCHEMA,
        "fit_child_residual_rows_sha256": _tensor_sha256(child_rows),
        "fit_parent_residual_rows_sha256": _tensor_sha256(parent_rows),
        "fit_child_residual_rows_public_sha256": (
            _authenticated_coordinate_map_sha256(child_rows)
        ),
        "fit_parent_residual_rows_public_sha256": (
            _authenticated_coordinate_map_sha256(parent_rows)
        ),
        "producer_encoder_sha256": _tensor_sha256(encoder),
        "parent_operator_sha256s": [value.operator_sha256 for value in operators],
        "requested_recipe": recipe,
        "fit_algorithm": (
            "producer-encoder-pseudoinverse-prior-plus-"
            "prior-centred-minimum-norm-trace-correction"
        ),
        "pseudoinverse_relative_tolerance": (_PSEUDOINVERSE_RELATIVE_TOLERANCE),
        "completion_rank_relative_tolerance": (_COMPLETION_RANK_RELATIVE_TOLERANCE),
        "completion_degeneracy_relative_tolerance": (
            _COMPLETION_DEGENERACY_RELATIVE_TOLERANCE
        ),
        "fit_partition_only": True,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "trace_capture_calls": 0,
    }
    return canonical_sha256(body)


def lfm2_body_global_coordinate_completion_evidence_sha256(
    fit_child_residual_rows: torch.Tensor,
    fit_parent_residual_rows: torch.Tensor,
    producer_encoder: torch.Tensor,
    parent_operators: Sequence[ResidualFacingOperator],
    *,
    recipe: str,
) -> str:
    """Seal every cached input consumed by one body-global compilation."""

    selected_recipe = _recipe(recipe)
    child_rows = _matrix(
        fit_child_residual_rows,
        "fit_child_residual_rows",
    )
    parent_rows = _matrix(
        fit_parent_residual_rows,
        "fit_parent_residual_rows",
    )
    encoder = _matrix(producer_encoder, "producer_encoder")
    operators = _operator_snapshots(parent_operators)
    _validate_shapes(child_rows, parent_rows, encoder, operators)
    return _evidence_sha256_from_snapshots(
        child_rows,
        parent_rows,
        encoder,
        operators,
        recipe=selected_recipe,
    )


def _rank(value: torch.Tensor) -> int:
    singular_values = torch.linalg.svdvals(value)
    if singular_values.numel() == 0:
        return 0
    maximum = float(singular_values.max().item())
    if maximum == 0.0:
        return 0
    threshold = maximum * max(
        _PSEUDOINVERSE_RELATIVE_TOLERANCE,
        torch.finfo(torch.float64).eps * max(map(int, value.shape)),
    )
    return int((singular_values > threshold).sum().item())


def _fit_raw_decoder(
    child_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    encoder: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    prior = torch.linalg.pinv(
        encoder,
        atol=0.0,
        rtol=_PSEUDOINVERSE_RELATIVE_TOLERANCE,
        hermitian=False,
    ).contiguous()
    trace_pseudoinverse = torch.linalg.pinv(
        child_rows,
        atol=0.0,
        rtol=_PSEUDOINVERSE_RELATIVE_TOLERANCE,
        hermitian=False,
    ).contiguous()
    prior_prediction = child_rows @ prior
    correction = trace_pseudoinverse @ (parent_rows - prior_prediction)
    fitted = (prior + correction).contiguous()
    fitted_prediction = child_rows @ fitted
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (
            prior,
            trace_pseudoinverse,
            correction,
            fitted,
            prior_prediction,
            fitted_prediction,
        )
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "The FP64 body-global raw decoder fit produced non-finite values."
        )
    prior_residual = parent_rows - prior_prediction
    fitted_residual = parent_rows - fitted_prediction
    prior_l2 = float(torch.linalg.vector_norm(prior_residual).item())
    fitted_l2 = float(torch.linalg.vector_norm(fitted_residual).item())
    if not math.isfinite(prior_l2) or not math.isfinite(fitted_l2):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "The FP64 body-global fit residual norm overflowed."
        )
    report = {
        "algorithm": (
            "producer-encoder-pseudoinverse-prior-plus-"
            "prior-centred-minimum-norm-trace-correction"
        ),
        "producer_decoder_prior_sha256": _tensor_sha256(prior),
        "trace_pseudoinverse_sha256": _tensor_sha256(trace_pseudoinverse),
        "trace_supported_correction_sha256": _tensor_sha256(correction),
        "trace_supported_fitted_map_sha256": _tensor_sha256(fitted),
        "fit_prediction_sha256": _tensor_sha256(fitted_prediction),
        "child_trace_rank": _rank(child_rows),
        "parent_fit_row_rank": _rank(parent_rows),
        "producer_encoder_rank": _rank(encoder),
        "producer_decoder_prior_rank": _rank(prior),
        "trace_supported_fitted_map_rank": _rank(fitted),
        "prior_fit_residual_l2": prior_l2,
        "trace_supported_fit_residual_l2": fitted_l2,
        "trace_supported_fit_residual_max_abs": float(
            fitted_residual.abs().max().item()
        ),
        "trace_fit_residual_is_navigation_only": True,
        "trace_fit_residual_quality_gate_applied": False,
    }
    return prior, fitted, report


def _resolve_candidate(
    frontier: AnalyticCoordinateCompletionFrontier,
    requested_recipe: str,
) -> tuple[AnalyticCoordinateCompletionCandidate, str]:
    try:
        return frontier.candidate(requested_recipe), "requested-recipe-materialized"
    except KeyError:
        pass
    if int(frontier.report.get("completion_rank", -1)) == 0:
        candidate = frontier.candidate("identified-map-already-full-rank")
        return candidate, "requested-recipe-degenerated-to-no-completion"
    duplicate_records = frontier.report.get(
        "duplicate_recipe_materializations",
        (),
    )
    for record in duplicate_records:
        if (
            isinstance(record, Mapping)
            and record.get("recipe") == requested_recipe
            and isinstance(record.get("duplicates"), str)
        ):
            candidate = frontier.candidate(str(record["duplicates"]))
            return candidate, "requested-recipe-materialized-as-identical-alias"
    rejected_records = frontier.report.get(
        "rejected_optional_recipe_materializations",
        (),
    )
    for record in rejected_records:
        if isinstance(record, Mapping) and record.get("recipe") == requested_recipe:
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "Requested body-global recipe failed numerical full-rank "
                "materialization; no hidden fallback was selected."
            )
    raise Lfm2BodyGlobalCoordinateCompletionError(
        "Requested body-global recipe is absent from the analytic frontier."
    )


def compile_lfm2_body_global_coordinate_completion(
    fit_child_residual_rows: torch.Tensor,
    fit_parent_residual_rows: torch.Tensor,
    producer_encoder: torch.Tensor,
    parent_operators: Sequence[ResidualFacingOperator],
    *,
    recipe: str,
    expected_evidence_sha256: str,
) -> Lfm2BodyGlobalCoordinateCompletion:
    """Compile one authenticated full-rank raw body decoder from cached FIT rows."""

    selected_recipe = _recipe(recipe)
    expected_evidence = _digest(
        expected_evidence_sha256,
        "expected_evidence_sha256",
    )
    # Take one owning snapshot of every caller-controlled tensor.  The same
    # snapshots are authenticated, fitted, and passed into the analytic solve.
    child_rows = _matrix(
        fit_child_residual_rows,
        "fit_child_residual_rows",
    )
    parent_rows = _matrix(
        fit_parent_residual_rows,
        "fit_parent_residual_rows",
    )
    encoder = _matrix(producer_encoder, "producer_encoder")
    operators = _operator_snapshots(parent_operators)
    row_count, target_width, source_width = _validate_shapes(
        child_rows,
        parent_rows,
        encoder,
        operators,
    )
    actual_evidence = _evidence_sha256_from_snapshots(
        child_rows,
        parent_rows,
        encoder,
        operators,
        recipe=selected_recipe,
    )
    if actual_evidence != expected_evidence:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Cached body-global coordinate evidence authentication failed."
        )

    with _NUMERICAL_LOCK:
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.no_grad():
                prior, fitted, fit_report = _fit_raw_decoder(
                    child_rows,
                    parent_rows,
                    encoder,
                )
        finally:
            torch.set_num_threads(original_threads)

    analytic_evidence = analytic_coordinate_completion_evidence_sha256(
        child_rows,
        fitted,
        operators,
        bridge_prior=prior,
    )
    frontier = compile_analytic_coordinate_completion_frontier(
        child_rows,
        fitted,
        operators,
        expected_evidence_sha256=analytic_evidence,
        bridge_prior=prior,
        rank_relative_tolerance=(_COMPLETION_RANK_RELATIVE_TOLERANCE),
        degeneracy_relative_tolerance=(_COMPLETION_DEGENERACY_RELATIVE_TOLERANCE),
    )
    candidate, selection_resolution = _resolve_candidate(
        frontier,
        selected_recipe,
    )
    raw_decoder = candidate.coordinate_map.detach().clone().contiguous()
    trace_reference = child_rows @ fitted
    trace_selected = child_rows @ raw_decoder
    trace_delta = trace_selected - trace_reference
    maximum_fit_action_change = float(trace_delta.abs().max().item())
    fit_action_roundoff_bound = float(
        candidate.report["trace_prediction_fp64_roundoff_bound"]
    )
    if maximum_fit_action_change > fit_action_roundoff_bound:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Selected raw decoder changed the authenticated FIT-row action."
        )
    body: dict[str, Any] = {
        "schema": LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA,
        "status": "body-global-full-rank-raw-decoder-materialized",
        "input_evidence_sha256": actual_evidence,
        "analytic_evidence_sha256": analytic_evidence,
        "analytic_frontier_result_sha256": frontier.report["result_sha256"],
        "selected_candidate_result_sha256": candidate.report["result_sha256"],
        "requested_recipe": selected_recipe,
        "recipe": selected_recipe,
        "materialized_candidate_recipe": candidate.recipe,
        "recipe_selection_resolution": selection_resolution,
        "fit_row_count": row_count,
        "target_width": target_width,
        "source_width": source_width,
        "required_full_rank": min(target_width, source_width),
        "structural_rank": min(target_width, source_width),
        "materialized_rank": int(candidate.report["materialized_rank"]),
        "full_rank_materialized": bool(candidate.report["full_rank_materialized"]),
        "fit_child_residual_rows_sha256": _tensor_sha256(child_rows),
        "fit_parent_residual_rows_sha256": _tensor_sha256(parent_rows),
        "producer_encoder_sha256": _tensor_sha256(encoder),
        "parent_operator_sha256s": [value.operator_sha256 for value in operators],
        "parent_operator_manifest_sha256": canonical_sha256(
            [value.operator_sha256 for value in operators]
        ),
        **fit_report,
        "raw_decoder_sha256": _tensor_sha256(raw_decoder),
        "coordinate_map_sha256": _tensor_sha256(raw_decoder),
        "public_coordinate_map_sha256": (
            _authenticated_coordinate_map_sha256(raw_decoder)
        ),
        "coordinate_map_shape": [target_width, source_width],
        "evidence_sha256": actual_evidence,
        "selected_fit_prediction_sha256": _tensor_sha256(trace_selected),
        "maximum_fit_action_change": maximum_fit_action_change,
        "fit_action_fp64_roundoff_bound": fit_action_roundoff_bound,
        "fit_action_preserved_to_fp64_roundoff": True,
        "fit_partition_only": True,
        "numeric_diagnostics_are_navigation_only": True,
        "semantic_acceptance_requires_decoded_behavior_review": True,
        "quality_score_gate_applied": False,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "trace_capture_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "random_completion_used": False,
        "filesystem_writes": 0,
    }
    receipt = {**body, "result_sha256": canonical_sha256(body)}
    return Lfm2BodyGlobalCoordinateCompletion(
        raw_decoder=raw_decoder,
        frontier=frontier,
        receipt=receipt,
    )


__all__ = [
    "LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_EVIDENCE_SCHEMA",
    "LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA",
    "LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES",
    "Lfm2BodyGlobalCoordinateCompletion",
    "Lfm2BodyGlobalCoordinateCompletionError",
    "compile_lfm2_body_global_coordinate_completion",
    "lfm2_body_global_coordinate_completion_evidence_sha256",
]
