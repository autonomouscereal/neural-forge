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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
from neural_forge.analytic_execution_profile import (
    AnalyticExecutionProfile,
    AnalyticExecutionProfileError,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_weight_complete_basis import (
    ResidualFacingOperator,
    RoleBalancedOperatorGeometry,
    WeightCompleteBasisError,
)


LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-body-global-coordinate-completion-evidence-v1"
)
LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-body-global-coordinate-completion-receipt-v1"
)
LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-body-global-coordinate-completion-evidence-v2"
)
LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-body-global-coordinate-completion-receipt-v2"
)
LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES = ("bridge-procrustes-isometric",)

_PSEUDOINVERSE_RELATIVE_TOLERANCE = 1e-12
_COMPLETION_RANK_RELATIVE_TOLERANCE = 1e-12
_COMPLETION_DEGENERACY_RELATIVE_TOLERANCE = 1e-11
_NUMERICAL_LOCK = threading.RLock()
_HEX_DIGITS = frozenset("0123456789abcdef")
_RECEIPT_SCHEMAS = frozenset(
    {
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA,
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA,
    }
)
_GEOMETRY_RECEIPT_REQUIRED_FIELDS = frozenset(
    {
        "fit_child_residual_rows_public_sha256",
        "fit_parent_residual_rows_public_sha256",
        "parent_geometry_sha256",
        "parent_geometry_result_sha256",
        "parent_operator_sha256s",
        "parent_operator_manifest_sha256",
        "parent_raw_gram_sha256",
        "parent_balanced_gram_sha256",
        "preaggregated_parent_geometry",
        "parent_operator_reconstruction_calls",
        "least_squares_execution_profile",
        "least_squares_execution_profile_sha256",
        "least_squares_requested_device",
        "least_squares_resolved_device",
        "least_squares_compute_dtype",
        "least_squares_pseudoinverse_count",
        "least_squares_normal_equations_used",
        "least_squares_deterministic_settings_applied",
        "least_squares_deterministic_settings_restored",
        "least_squares_canonical_report_device",
    }
)


class Lfm2BodyGlobalCoordinateCompletionError(AnalyticCoordinateCompletionError):
    """Cached body-global evidence or recipe selection is invalid."""


def _validate_geometry_receipt_contract(
    receipt: Mapping[str, Any],
    frontier: AnalyticCoordinateCompletionFrontier,
) -> None:
    missing = sorted(_GEOMETRY_RECEIPT_REQUIRED_FIELDS.difference(receipt))
    if missing:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt v2 fields are incomplete: " + ", ".join(missing)
        )

    geometry_sha256 = _digest(
        receipt.get("parent_geometry_sha256"),
        "parent_geometry_sha256",
    )
    geometry_result_sha256 = _digest(
        receipt.get("parent_geometry_result_sha256"),
        "parent_geometry_result_sha256",
    )
    raw_gram_sha256 = _digest(
        receipt.get("parent_raw_gram_sha256"),
        "parent_raw_gram_sha256",
    )
    balanced_gram_sha256 = _digest(
        receipt.get("parent_balanced_gram_sha256"),
        "parent_balanced_gram_sha256",
    )
    child_public_sha256 = _digest(
        receipt.get("fit_child_residual_rows_public_sha256"),
        "fit_child_residual_rows_public_sha256",
    )
    parent_public_sha256 = _digest(
        receipt.get("fit_parent_residual_rows_public_sha256"),
        "fit_parent_residual_rows_public_sha256",
    )
    operator_sha256s = receipt.get("parent_operator_sha256s")
    if (
        not isinstance(operator_sha256s, list)
        or not operator_sha256s
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in _HEX_DIGITS for character in value)
            for value in operator_sha256s
        )
        or len(set(operator_sha256s)) != len(operator_sha256s)
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt parent operator hash order is invalid."
        )
    operator_manifest_sha256 = _digest(
        receipt.get("parent_operator_manifest_sha256"),
        "parent_operator_manifest_sha256",
    )
    if (
        geometry_result_sha256 != geometry_sha256
        or operator_manifest_sha256 != canonical_sha256(operator_sha256s)
        or receipt.get("preaggregated_parent_geometry") is not True
        or type(receipt.get("parent_operator_reconstruction_calls")) is not int
        or receipt.get("parent_operator_reconstruction_calls") != 0
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt v2 source bindings are contradictory."
        )

    try:
        profile = AnalyticExecutionProfile(
            receipt.get("least_squares_execution_profile")
        )
    except AnalyticExecutionProfileError as exc:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt has no valid least-squares execution profile."
        ) from exc
    profile_sha256 = _digest(
        receipt.get("least_squares_execution_profile_sha256"),
        "least_squares_execution_profile_sha256",
    )
    if (
        profile_sha256 != profile.result_sha256
        or receipt.get("least_squares_requested_device") != profile.requested_device
        or receipt.get("least_squares_resolved_device") != profile.resolved_device
        or receipt.get("least_squares_compute_dtype") != "float64"
        or type(receipt.get("least_squares_pseudoinverse_count")) is not int
        or receipt.get("least_squares_pseudoinverse_count") != 2
        or receipt.get("least_squares_normal_equations_used") is not False
        or receipt.get("least_squares_deterministic_settings_applied") is not True
        or receipt.get("least_squares_deterministic_settings_restored") is not True
        or receipt.get("least_squares_canonical_report_device") != "cpu"
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt least-squares profile bindings are contradictory."
        )

    frontier_geometry = frontier.report.get("parent_geometry")
    frontier_operator_reports = (
        frontier_geometry.get("operators")
        if isinstance(frontier_geometry, Mapping)
        else None
    )
    if not isinstance(frontier_operator_reports, Sequence) or isinstance(
        frontier_operator_reports,
        str | bytes,
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt is not bound to its analytic frontier."
        )
    frontier_operator_sha256s = [
        value.get("operator_sha256") if isinstance(value, Mapping) else None
        for value in frontier_operator_reports
    ]
    if (
        frontier_geometry.get("operator_count") != len(operator_sha256s)
        or frontier_operator_sha256s != operator_sha256s
        or frontier_geometry.get("raw_joint_gram_sha256") != raw_gram_sha256
        or frontier_geometry.get("balanced_structural_gram_sha256")
        != balanced_gram_sha256
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt contradicts its analytic frontier geometry."
        )

    expected_evidence = canonical_sha256(
        {
            "schema": (LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_EVIDENCE_SCHEMA),
            "fit_child_residual_rows_sha256": _digest(
                receipt.get("fit_child_residual_rows_sha256"),
                "fit_child_residual_rows_sha256",
            ),
            "fit_parent_residual_rows_sha256": _digest(
                receipt.get("fit_parent_residual_rows_sha256"),
                "fit_parent_residual_rows_sha256",
            ),
            "fit_child_residual_rows_public_sha256": child_public_sha256,
            "fit_parent_residual_rows_public_sha256": parent_public_sha256,
            "producer_encoder_sha256": _digest(
                receipt.get("producer_encoder_sha256"),
                "producer_encoder_sha256",
            ),
            "parent_geometry_sha256": geometry_sha256,
            "parent_operator_sha256s": operator_sha256s,
            "parent_raw_gram_sha256": raw_gram_sha256,
            "parent_balanced_gram_sha256": balanced_gram_sha256,
            "least_squares_execution_profile_sha256": profile_sha256,
            "requested_recipe": receipt.get("requested_recipe"),
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
            "preaggregated_parent_geometry": True,
            "parent_operator_reconstruction_calls": 0,
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "trace_capture_calls": 0,
        }
    )
    if (
        receipt.get("input_evidence_sha256") != expected_evidence
        or receipt.get("evidence_sha256") != expected_evidence
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "geometry receipt v2 evidence bindings are contradictory."
        )


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
        schema = receipt.get("schema")
        if schema not in _RECEIPT_SCHEMAS:
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
        if schema == LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA:
            _validate_geometry_receipt_contract(receipt, self.frontier)
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


def _parent_geometry(
    value: Any,
) -> RoleBalancedOperatorGeometry:
    if not isinstance(value, RoleBalancedOperatorGeometry):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "parent_geometry must be a RoleBalancedOperatorGeometry."
        )
    try:
        snapshot = RoleBalancedOperatorGeometry(
            raw_gram=value.raw_gram,
            balanced_gram=value.balanced_gram,
            report=value.report,
        )
    except (RuntimeError, ValueError, WeightCompleteBasisError) as exc:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "parent_geometry authentication failed."
        ) from exc
    return snapshot


def _least_squares_profile(value: Any) -> AnalyticExecutionProfile:
    if not isinstance(value, AnalyticExecutionProfile):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "least_squares_execution_profile must be an AnalyticExecutionProfile."
        )
    try:
        return AnalyticExecutionProfile(value.receipt)
    except AnalyticExecutionProfileError as exc:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "least_squares_execution_profile authentication failed."
        ) from exc


def _geometry_hashes(
    geometry: RoleBalancedOperatorGeometry,
) -> tuple[str, str]:
    report = geometry.report
    hashes = report.get("hashes")
    if not isinstance(hashes, Mapping):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "parent_geometry has no authenticated Gram hashes."
        )
    return (
        _digest(hashes.get("raw_gram_sha256"), "parent raw Gram SHA-256"),
        _digest(
            hashes.get("balanced_gram_sha256"),
            "parent balanced Gram SHA-256",
        ),
    )


def _coordinate_source(
    parent_operators: Sequence[ResidualFacingOperator] | None,
    parent_geometry: RoleBalancedOperatorGeometry | None,
    least_squares_execution_profile: AnalyticExecutionProfile | None,
) -> tuple[
    tuple[ResidualFacingOperator, ...] | None,
    RoleBalancedOperatorGeometry | None,
    AnalyticExecutionProfile | None,
]:
    has_operators = parent_operators is not None
    has_geometry = parent_geometry is not None
    if has_operators == has_geometry:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Exactly one of parent_operators or parent_geometry is required."
        )
    if has_operators:
        if least_squares_execution_profile is not None:
            raise Lfm2BodyGlobalCoordinateCompletionError(
                "The legacy parent_operators path does not accept an execution profile."
            )
        return _operator_snapshots(parent_operators), None, None
    if least_squares_execution_profile is None:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "parent_geometry requires least_squares_execution_profile."
        )
    return (
        None,
        _parent_geometry(parent_geometry),
        _least_squares_profile(least_squares_execution_profile),
    )


def _validate_shapes(
    child_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    encoder: torch.Tensor,
    parent_source_width: int,
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
    if parent_source_width != source_width:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Parent geometry width must match parent residual source width."
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


def _geometry_evidence_sha256_from_snapshots(
    child_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    encoder: torch.Tensor,
    geometry: RoleBalancedOperatorGeometry,
    profile: AnalyticExecutionProfile,
    *,
    recipe: str,
) -> str:
    raw_gram_sha256, balanced_gram_sha256 = _geometry_hashes(geometry)
    body = {
        "schema": (LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_EVIDENCE_SCHEMA),
        "fit_child_residual_rows_sha256": _tensor_sha256(child_rows),
        "fit_parent_residual_rows_sha256": _tensor_sha256(parent_rows),
        "fit_child_residual_rows_public_sha256": (
            _authenticated_coordinate_map_sha256(child_rows)
        ),
        "fit_parent_residual_rows_public_sha256": (
            _authenticated_coordinate_map_sha256(parent_rows)
        ),
        "producer_encoder_sha256": _tensor_sha256(encoder),
        "parent_geometry_sha256": geometry.geometry_sha256,
        "parent_operator_sha256s": list(geometry.ordered_operator_sha256s),
        "parent_raw_gram_sha256": raw_gram_sha256,
        "parent_balanced_gram_sha256": balanced_gram_sha256,
        "least_squares_execution_profile_sha256": profile.result_sha256,
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
        "preaggregated_parent_geometry": True,
        "parent_operator_reconstruction_calls": 0,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "trace_capture_calls": 0,
    }
    return canonical_sha256(body)


def lfm2_body_global_coordinate_completion_evidence_sha256(
    fit_child_residual_rows: torch.Tensor,
    fit_parent_residual_rows: torch.Tensor,
    producer_encoder: torch.Tensor,
    parent_operators: Sequence[ResidualFacingOperator] | None = None,
    *,
    parent_geometry: RoleBalancedOperatorGeometry | None = None,
    least_squares_execution_profile: AnalyticExecutionProfile | None = None,
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
    operators, geometry, profile = _coordinate_source(
        parent_operators,
        parent_geometry,
        least_squares_execution_profile,
    )
    source_width = (
        operators[0].source_width if operators is not None else geometry.source_width
    )
    _validate_shapes(child_rows, parent_rows, encoder, source_width)
    if operators is not None:
        return _evidence_sha256_from_snapshots(
            child_rows,
            parent_rows,
            encoder,
            operators,
            recipe=selected_recipe,
        )
    assert geometry is not None
    assert profile is not None
    return _geometry_evidence_sha256_from_snapshots(
        child_rows,
        parent_rows,
        encoder,
        geometry,
        profile,
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


@contextmanager
def _deterministic_execution_settings(
    profile: AnalyticExecutionProfile,
) -> Iterator[None]:
    """Apply and restore the process-global settings bound by ``profile``."""

    use_cpu = profile.torch_device.type == "cpu"
    original_threads = torch.get_num_threads() if use_cpu else None
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        if original_threads is not None:
            torch.set_num_threads(int(profile.receipt["cpu_thread_count"]))
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        try:
            torch.backends.cudnn.benchmark = original_cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
        finally:
            try:
                torch.use_deterministic_algorithms(
                    original_deterministic,
                    warn_only=original_warn_only,
                )
            finally:
                if original_threads is not None:
                    torch.set_num_threads(original_threads)


def _fit_raw_decoder_with_profile(
    child_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    encoder: torch.Tensor,
    profile: AnalyticExecutionProfile,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run the unchanged two-pseudoinverse fit on the resolved FP64 device."""

    device = profile.torch_device
    child_device = child_rows.to(device=device, dtype=torch.float64)
    parent_device = parent_rows.to(device=device, dtype=torch.float64)
    encoder_device = encoder.to(device=device, dtype=torch.float64)
    prior_device = torch.linalg.pinv(
        encoder_device,
        atol=0.0,
        rtol=_PSEUDOINVERSE_RELATIVE_TOLERANCE,
        hermitian=False,
    ).contiguous()
    trace_pseudoinverse_device = torch.linalg.pinv(
        child_device,
        atol=0.0,
        rtol=_PSEUDOINVERSE_RELATIVE_TOLERANCE,
        hermitian=False,
    ).contiguous()
    prior_prediction_device = child_device @ prior_device
    correction_device = trace_pseudoinverse_device @ (
        parent_device - prior_prediction_device
    )
    fitted_device = (prior_device + correction_device).contiguous()
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (
            prior_device,
            trace_pseudoinverse_device,
            correction_device,
            fitted_device,
        )
    ):
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "The FP64 body-global raw decoder fit produced non-finite values."
        )

    prior = prior_device.to(device="cpu", dtype=torch.float64).contiguous()
    fitted = fitted_device.to(device="cpu", dtype=torch.float64).contiguous()
    trace_pseudoinverse = trace_pseudoinverse_device.to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()

    # Every installed-map diagnostic is deliberately recomputed from the
    # canonical CPU FP64 materialization, not from device-local intermediates.
    correction = (fitted - prior).contiguous()
    prior_prediction = child_rows @ prior
    fitted_prediction = child_rows @ fitted
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
        "least_squares_execution_profile": profile.as_dict(),
        "least_squares_execution_profile_sha256": profile.result_sha256,
        "least_squares_requested_device": profile.requested_device,
        "least_squares_resolved_device": profile.resolved_device,
        "least_squares_compute_dtype": "float64",
        "least_squares_pseudoinverse_count": 2,
        "least_squares_normal_equations_used": False,
        "least_squares_deterministic_settings_applied": True,
        "least_squares_deterministic_settings_restored": True,
        "least_squares_canonical_report_device": "cpu",
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
    parent_operators: Sequence[ResidualFacingOperator] | None = None,
    *,
    parent_geometry: RoleBalancedOperatorGeometry | None = None,
    least_squares_execution_profile: AnalyticExecutionProfile | None = None,
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
    operators, geometry, profile = _coordinate_source(
        parent_operators,
        parent_geometry,
        least_squares_execution_profile,
    )
    if operators is not None:
        parent_source_width = operators[0].source_width
    else:
        assert geometry is not None
        parent_source_width = geometry.source_width
    row_count, target_width, source_width = _validate_shapes(
        child_rows,
        parent_rows,
        encoder,
        parent_source_width,
    )
    if operators is not None:
        actual_evidence = _evidence_sha256_from_snapshots(
            child_rows,
            parent_rows,
            encoder,
            operators,
            recipe=selected_recipe,
        )
    else:
        assert geometry is not None
        assert profile is not None
        actual_evidence = _geometry_evidence_sha256_from_snapshots(
            child_rows,
            parent_rows,
            encoder,
            geometry,
            profile,
            recipe=selected_recipe,
        )
    if actual_evidence != expected_evidence:
        raise Lfm2BodyGlobalCoordinateCompletionError(
            "Cached body-global coordinate evidence authentication failed."
        )

    if operators is not None:
        # Preserve the historical CPU-v1 arithmetic and receipt byte for byte.
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
    else:
        assert profile is not None
        with _NUMERICAL_LOCK:
            with _deterministic_execution_settings(profile):
                with torch.no_grad():
                    prior, fitted, fit_report = _fit_raw_decoder_with_profile(
                        child_rows,
                        parent_rows,
                        encoder,
                        profile,
                    )

    if operators is not None:
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
    else:
        assert geometry is not None
        analytic_evidence = analytic_coordinate_completion_evidence_sha256(
            child_rows,
            fitted,
            parent_geometry=geometry,
            bridge_prior=prior,
        )
        frontier = compile_analytic_coordinate_completion_frontier(
            child_rows,
            fitted,
            expected_evidence_sha256=analytic_evidence,
            parent_geometry=geometry,
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
    if operators is not None:
        source_receipt: dict[str, Any] = {
            "parent_operator_sha256s": [value.operator_sha256 for value in operators],
            "parent_operator_manifest_sha256": canonical_sha256(
                [value.operator_sha256 for value in operators]
            ),
        }
        receipt_schema = LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA
    else:
        assert geometry is not None
        raw_gram_sha256, balanced_gram_sha256 = _geometry_hashes(geometry)
        operator_sha256s = list(geometry.ordered_operator_sha256s)
        source_receipt = {
            "fit_child_residual_rows_public_sha256": (
                _authenticated_coordinate_map_sha256(child_rows)
            ),
            "fit_parent_residual_rows_public_sha256": (
                _authenticated_coordinate_map_sha256(parent_rows)
            ),
            "parent_geometry_sha256": geometry.geometry_sha256,
            "parent_geometry_result_sha256": geometry.geometry_sha256,
            "parent_operator_sha256s": operator_sha256s,
            "parent_operator_manifest_sha256": canonical_sha256(operator_sha256s),
            "parent_raw_gram_sha256": raw_gram_sha256,
            "parent_balanced_gram_sha256": balanced_gram_sha256,
            "preaggregated_parent_geometry": True,
            "parent_operator_reconstruction_calls": 0,
        }
        receipt_schema = LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA
    body: dict[str, Any] = {
        "schema": receipt_schema,
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
        **source_receipt,
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
    "LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_EVIDENCE_SCHEMA",
    "LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA",
    "LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA",
    "LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES",
    "Lfm2BodyGlobalCoordinateCompletion",
    "Lfm2BodyGlobalCoordinateCompletionError",
    "compile_lfm2_body_global_coordinate_completion",
    "lfm2_body_global_coordinate_completion_evidence_sha256",
]
