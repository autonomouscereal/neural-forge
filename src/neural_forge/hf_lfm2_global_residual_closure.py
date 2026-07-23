"""Global causal residual-closure folding for already compressed LFM2 children.

This module is the standalone numerical core for a whole-stack correction that
is deliberately different from another local layer refit.  Every residual
writer is frozen except for one dense additive correction

``delta_writer[site] = output_basis @ coefficient_block[site]``.

The output basis is built once from three authenticated sources: mapped FIT
residual defects, source-writer columns, and terminal tied-table directions.
Its rank is derived from target topology (``2 * head_dimension``; rank 128 for
the canonical LFM2 child).  A caller-supplied analytic whole-stack JVP/VJP then
couples all writer blocks to every downstream residual boundary.  The core
solves the resulting global ridge system with a fixed matrix-free PCG schedule
and materializes ordinary dense writer tensors only after the solve.

The callable linearization boundary is intentional.  The semantic layer-stack
adapter owns cached attention, short-convolution, and SwiGLU primal state; this
module owns authenticated evidence, block parameterization, weighted normal
equations, deterministic PCG, adjoint/causality audits, and dense publication.

This is folding, not training.  There is no autograd graph, backward call,
optimizer, parameter pruning, quantization, quality gate, or recipe selection.
Numeric diagnostics are recorded for manual review and never accept, reject,
delete, or choose a child.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_fold_executor import FoldOnlyGuardReport, run_fold_only_callback


LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA = "neural-forge-lfm2-global-residual-closure-v1"
LFM2_GLOBAL_RESIDUAL_LINEARIZATION_SCHEMA = (
    "neural-forge-lfm2-global-residual-linearization-v1"
)
_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_global_residual_closure."
    "rank-two-head-global-causal-matrix-free-pcg.v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WRITER_KINDS = frozenset(
    {
        "attention_input_transport",
        "attention_output",
        "short_conv_input_transport",
        "short_conv_output",
        "mlp_input_transport",
        "mlp_down",
    }
)
_BOUNDARY_KINDS = frozenset(
    {"post_operator_residual", "post_layer_residual", "terminal_observable"}
)
_ZERO_EXECUTION = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "generation_calls": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2GlobalResidualClosureError(ValueError):
    """Global residual-closure evidence or geometry is invalid."""


class Lfm2GlobalResidualClosureAuthenticationError(Lfm2GlobalResidualClosureError):
    """Global residual-closure evidence failed authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for global residual-closure folding."
        ) from exc
    return torch


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
        raise Lfm2GlobalResidualClosureError(
            "Global residual-closure metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _sha256(detached)})


def verify_lfm2_global_residual_closure_report(
    report: Mapping[str, Any],
) -> bool:
    """Verify a canonical report emitted by this module."""

    if not isinstance(report, Mapping):
        return False
    try:
        body = dict(report)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _sha256(body)
    except (TypeError, ValueError):
        return False


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2GlobalResidualClosureAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def fingerprint_lfm2_global_residual_tensor(value: Any) -> str:
    """Return a stable dtype/shape/raw-byte fingerprint for one tensor."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2GlobalResidualClosureError(
            "Only materialized dense tensors can be authenticated."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _plain_finite_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or (dimensions is not None and value.ndim != dimensions)
        or not bool(torch.isfinite(value).all().item())
    ):
        rank = "" if dimensions is None else f" rank-{dimensions}"
        raise Lfm2GlobalResidualClosureError(
            f"{name} must be a finite nonempty dense real{rank} tensor "
            "that does not require gradients."
        )
    return value.detach()


def _authenticate_tensor(value: Any, expected: Any, name: str) -> str:
    required = _require_digest(expected, f"{name}_sha256")
    observed = fingerprint_lfm2_global_residual_tensor(value)
    if observed != required:
        raise Lfm2GlobalResidualClosureAuthenticationError(
            f"{name} differs from its authenticated SHA-256 digest."
        )
    return observed


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2GlobalResidualClosureError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise Lfm2GlobalResidualClosureError(f"{name} must not exceed {maximum}.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2GlobalResidualClosureError(f"{name} must be a nonnegative integer.")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2GlobalResidualClosureError(
            f"{name} must be a finite nonnegative number."
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2GlobalResidualClosureError(
            f"{name} must be a finite nonnegative number."
        )
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise Lfm2GlobalResidualClosureError(
            f"{name} must be a finite positive number."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualWriterSite:
    """One residual-writing matrix in causal stack order."""

    site_id: str
    layer_index: int
    causal_order: int
    writer_kind: str
    input_width: int

    def __post_init__(self) -> None:
        site_id = str(self.site_id).strip()
        if not site_id:
            raise Lfm2GlobalResidualClosureError("site_id cannot be empty.")
        writer_kind = str(self.writer_kind).strip().lower()
        if writer_kind not in _WRITER_KINDS:
            raise Lfm2GlobalResidualClosureError(
                f"writer_kind must be one of {sorted(_WRITER_KINDS)}."
            )
        object.__setattr__(self, "site_id", site_id)
        object.__setattr__(
            self,
            "layer_index",
            _nonnegative_int(self.layer_index, "layer_index"),
        )
        object.__setattr__(
            self,
            "causal_order",
            _nonnegative_int(self.causal_order, "causal_order"),
        )
        object.__setattr__(self, "writer_kind", writer_kind)
        object.__setattr__(
            self,
            "input_width",
            _positive_int(self.input_width, "input_width"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "layer_index": self.layer_index,
            "causal_order": self.causal_order,
            "writer_kind": self.writer_kind,
            "input_width": self.input_width,
        }


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualWriterTopology:
    """Authenticated writer inventory and topology-derived basis rank."""

    hidden_size: int
    head_dimension: int
    sites: tuple[Lfm2GlobalResidualWriterSite, ...]
    topology_sha256: str = ""

    def __post_init__(self) -> None:
        hidden = _positive_int(self.hidden_size, "hidden_size")
        head = _positive_int(self.head_dimension, "head_dimension")
        site_values = tuple(self.sites)
        if not site_values or not all(
            isinstance(site, Lfm2GlobalResidualWriterSite) for site in site_values
        ):
            raise Lfm2GlobalResidualClosureError(
                "sites must contain at least one writer site."
            )
        if (
            tuple(sorted(site_values, key=lambda site: site.causal_order))
            != site_values
        ):
            raise Lfm2GlobalResidualClosureError(
                "writer sites must be supplied in increasing causal order."
            )
        if len({site.site_id for site in site_values}) != len(site_values):
            raise Lfm2GlobalResidualClosureError("writer site ids must be unique.")
        if len({site.causal_order for site in site_values}) != len(site_values):
            raise Lfm2GlobalResidualClosureError(
                "writer causal_order values must be unique."
            )
        object.__setattr__(self, "hidden_size", hidden)
        object.__setattr__(self, "head_dimension", head)
        object.__setattr__(self, "sites", site_values)
        expected = _sha256(self.authentication_body())
        supplied = str(self.topology_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "writer topology_sha256 is incorrect."
            )
        object.__setattr__(self, "topology_sha256", expected)

    @property
    def topology_basis_rank(self) -> int:
        """Return ``2 * head_dimension`` before hidden-width clipping."""

        return 2 * self.head_dimension

    @property
    def effective_basis_rank(self) -> int:
        return min(self.hidden_size, self.topology_basis_rank)

    @property
    def coefficient_coordinate_count(self) -> int:
        return self.effective_basis_rank * sum(site.input_width for site in self.sites)

    @property
    def dense_writer_coordinate_count(self) -> int:
        return self.hidden_size * sum(site.input_width for site in self.sites)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
            "kind": "writer_topology",
            "hidden_size": self.hidden_size,
            "head_dimension": self.head_dimension,
            "topology_basis_rank_formula": "2*head_dimension",
            "topology_basis_rank": self.topology_basis_rank,
            "effective_basis_rank": self.effective_basis_rank,
            "sites": [site.as_dict() for site in self.sites],
            "coefficient_coordinate_count": self.coefficient_coordinate_count,
            "dense_writer_coordinate_count": self.dense_writer_coordinate_count,
        }

    def verify_authentication(self) -> bool:
        return self.topology_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "topology_sha256": self.topology_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualClosurePolicy:
    """Authenticated fixed-schedule solve policy with no quality gate."""

    condition_cap: float = 10_000.0
    pcg_iterations: int = 24
    absolute_ridge_floor: float = 1e-12
    boundary_chunk_rows: int = 256
    maximum_boundary_workspace_bytes: int = 512 * 1024 * 1024
    backend: str = "auto"
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        cap = _finite_positive(self.condition_cap, "condition_cap")
        if cap not in {1_000.0, 10_000.0}:
            raise Lfm2GlobalResidualClosureError("condition_cap must be 1000 or 10000.")
        iterations = _positive_int(
            self.pcg_iterations,
            "pcg_iterations",
            maximum=256,
        )
        ridge = _finite_positive(
            self.absolute_ridge_floor,
            "absolute_ridge_floor",
        )
        chunk_rows = _positive_int(
            self.boundary_chunk_rows,
            "boundary_chunk_rows",
            maximum=1_048_576,
        )
        workspace = _positive_int(
            self.maximum_boundary_workspace_bytes,
            "maximum_boundary_workspace_bytes",
        )
        backend = str(self.backend).strip().lower()
        if backend not in {"auto", "cpu", "cuda"}:
            raise Lfm2GlobalResidualClosureError("backend must be auto, cpu, or cuda.")
        object.__setattr__(self, "condition_cap", cap)
        object.__setattr__(self, "pcg_iterations", iterations)
        object.__setattr__(self, "absolute_ridge_floor", ridge)
        object.__setattr__(self, "boundary_chunk_rows", chunk_rows)
        object.__setattr__(self, "maximum_boundary_workspace_bytes", workspace)
        object.__setattr__(self, "backend", backend)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "global closure policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
            "kind": "global_residual_closure_policy",
            "condition_cap": self.condition_cap,
            "pcg_iterations": self.pcg_iterations,
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "boundary_chunk_rows": self.boundary_chunk_rows,
            "maximum_boundary_workspace_bytes": (self.maximum_boundary_workspace_bytes),
            "boundary_archive_storage_device": "cpu",
            "full_boundary_compute_staging": False,
            "backend": self.backend,
            "compute_dtype": "float64",
            "fixed_iteration_schedule": True,
            "early_stopping": False,
            "quality_gate": False,
            "numeric_diagnostics_used_for_semantic_decision": False,
            "numeric_diagnostics_used_for_auto_veto": False,
            "execution": {
                **_ZERO_EXECUTION,
                "autograd_graphs": 0,
                "closed_form_iterative_linear_solve": True,
            },
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


def estimate_lfm2_global_residual_streaming_memory(
    *,
    hidden_size: int,
    boundary_row_counts: Sequence[int],
    requested_chunk_rows: int,
    maximum_workspace_bytes: int,
) -> Mapping[str, Any]:
    """Estimate real geometry without allocating the boundary bank."""

    hidden = _positive_int(hidden_size, "hidden_size")
    row_counts = tuple(
        _positive_int(value, f"boundary_row_counts[{index}]")
        for index, value in enumerate(boundary_row_counts)
    )
    if not row_counts:
        raise Lfm2GlobalResidualClosureError("boundary_row_counts cannot be empty.")
    requested = _positive_int(requested_chunk_rows, "requested_chunk_rows")
    budget = _positive_int(maximum_workspace_bytes, "maximum_workspace_bytes")
    fp64_bytes = 8
    archive_bytes = sum(
        rows * (hidden * fp64_bytes + fp64_bytes) for rows in row_counts
    )
    bundle_effect_archive_bytes = sum(rows * hidden * fp64_bytes for rows in row_counts)
    unstreamed_triple_copy = sum(rows * hidden * fp64_bytes * 3 for rows in row_counts)
    bytes_per_chunk_row = hidden * fp64_bytes * 3 + fp64_bytes
    budget_rows = budget // bytes_per_chunk_row
    if budget_rows < 1:
        raise Lfm2GlobalResidualClosureError(
            "maximum_workspace_bytes cannot hold one hidden row."
        )
    effective_rows = min(requested, budget_rows)
    peak_workspace = effective_rows * bytes_per_chunk_row
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "cpu_boundary_streaming_memory_estimate",
        "hidden_size": hidden,
        "boundary_row_counts": list(row_counts),
        "total_boundary_rows": sum(row_counts),
        "floating_dtype": "float64",
        "boundary_archive_storage_device": "cpu",
        "boundary_archive_bytes": archive_bytes,
        "whole_stack_bundle_output_storage_device": "cpu",
        "whole_stack_bundle_effect_archive_bytes": (bundle_effect_archive_bytes),
        "whole_stack_bundle_peak_additional_cpu_bytes": (bundle_effect_archive_bytes),
        "whole_stack_bundle_weighted_effect_copy_materialized": False,
        "unstreamed_defect_effect_weighted_bytes": unstreamed_triple_copy,
        "requested_chunk_rows": requested,
        "effective_chunk_rows": effective_rows,
        "bytes_per_chunk_row_upper_bound": bytes_per_chunk_row,
        "estimated_peak_boundary_workspace_bytes": peak_workspace,
        "maximum_workspace_bytes": budget,
        "workspace_budget_passed": peak_workspace <= budget,
        "full_boundary_compute_staging": False,
        "defect_effect_weighted_triple_copy_on_gpu": False,
    }
    return _seal_report(body)


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualBasis:
    """Rank-limited dense output basis and deterministic QDEIM coordinates."""

    output_basis: Any
    qdeim_coordinates: tuple[int, ...]
    evidence_eigenvalues: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        basis = (
            _plain_finite_tensor(
                self.output_basis,
                "output_basis",
                dimensions=2,
            )
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        spectrum = (
            _plain_finite_tensor(
                self.evidence_eigenvalues,
                "evidence_eigenvalues",
                dimensions=1,
            )
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        coordinates = tuple(int(value) for value in self.qdeim_coordinates)
        object.__setattr__(self, "output_basis", basis)
        object.__setattr__(self, "evidence_eigenvalues", spectrum)
        object.__setattr__(self, "qdeim_coordinates", coordinates)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def hidden_size(self) -> int:
        return int(self.output_basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.output_basis.shape[1])

    @property
    def basis_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_global_residual_closure_report(self.report)
            and self.report.get("output_basis_sha256")
            == fingerprint_lfm2_global_residual_tensor(self.output_basis)
            and self.report.get("evidence_eigenvalues_sha256")
            == fingerprint_lfm2_global_residual_tensor(self.evidence_eigenvalues)
            and tuple(self.report.get("qdeim_coordinates", ()))
            == self.qdeim_coordinates
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _authenticated_tensor_mapping(
    values: Mapping[str, Any],
    expected_sha256: Mapping[str, str],
    name: str,
) -> tuple[tuple[str, Any, str], ...]:
    if not isinstance(values, Mapping):
        raise Lfm2GlobalResidualClosureError(f"{name} must be a mapping.")
    if not isinstance(expected_sha256, Mapping) or set(values) != set(expected_sha256):
        raise Lfm2GlobalResidualClosureAuthenticationError(
            f"{name} and {name}_sha256 must have identical keys."
        )
    result = []
    for block_id in sorted(values):
        clean_id = str(block_id).strip()
        if not clean_id or clean_id != block_id:
            raise Lfm2GlobalResidualClosureError(
                f"{name} keys must be nonempty canonical strings."
            )
        tensor = _plain_finite_tensor(values[block_id], f"{name}.{block_id}")
        digest = _authenticate_tensor(
            tensor,
            expected_sha256[block_id],
            f"{name}.{block_id}",
        )
        result.append((block_id, tensor, digest))
    return tuple(result)


def _canonicalize_column_signs(value: Any) -> Any:
    torch = _torch()
    result = value.clone()
    for column_index in range(int(result.shape[1])):
        column = result[:, column_index]
        pivot = int(torch.argmax(column.abs()).item())
        if float(column[pivot].item()) < 0.0:
            result[:, column_index] = -column
    return result


def _complete_orthonormal_columns(
    initial: Any,
    *,
    hidden_size: int,
    rank: int,
) -> tuple[Any, int]:
    """Complete evidence columns with deterministic canonical residual axes."""

    torch = _torch()
    columns = [initial[:, index].clone() for index in range(int(initial.shape[1]))]
    evidence_columns = len(columns)
    epsilon = torch.finfo(torch.float64).eps
    for axis_index in range(hidden_size):
        if len(columns) == rank:
            break
        candidate = torch.zeros((hidden_size,), dtype=torch.float64)
        candidate[axis_index] = 1.0
        for _pass in range(2):
            for column in columns:
                candidate = candidate - torch.dot(column, candidate) * column
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm > 64.0 * epsilon:
            columns.append(candidate / norm)
    if len(columns) != rank:
        raise Lfm2GlobalResidualClosureError(
            "Could not complete the topology-derived output basis."
        )
    return torch.stack(columns, dim=1).contiguous(), rank - evidence_columns


def _deterministic_qdeim_coordinates(basis: Any) -> tuple[int, ...]:
    """Select QDEIM coordinates with deterministic pivoted modified GS."""

    torch = _torch()
    residual = basis.transpose(0, 1).clone()
    rank, hidden = map(int, residual.shape)
    selected: list[int] = []
    tiny = torch.finfo(torch.float64).eps
    for _index in range(rank):
        squared_norms = residual.square().sum(dim=0)
        if selected:
            squared_norms[torch.tensor(selected, dtype=torch.long)] = -1.0
        pivot = int(torch.argmax(squared_norms).item())
        norm = math.sqrt(max(0.0, float(squared_norms[pivot].item())))
        if norm <= tiny:
            for fallback in range(hidden):
                if fallback not in selected:
                    pivot = fallback
                    break
            vector = residual[:, pivot]
            norm = float(torch.linalg.vector_norm(vector).item())
        if norm <= tiny:
            raise Lfm2GlobalResidualClosureError(
                "QDEIM coordinate selection encountered a zero residual pivot."
            )
        selected.append(pivot)
        direction = residual[:, pivot] / norm
        residual = residual - direction.unsqueeze(1) * (
            direction.unsqueeze(0) @ residual
        )
    return tuple(selected)


def build_lfm2_global_residual_basis(
    topology: Lfm2GlobalResidualWriterTopology,
    *,
    mapped_fit_residual_defects: Mapping[str, Any],
    mapped_fit_residual_defects_sha256: Mapping[str, str],
    source_writer_weights: Mapping[str, Any],
    source_writer_weights_sha256: Mapping[str, str],
    terminal_tied_table_directions: Mapping[str, Any],
    terminal_tied_table_directions_sha256: Mapping[str, str],
) -> Lfm2GlobalResidualBasis:
    """Build the deterministic topology-rank mixed global output basis.

    Each source block is Frobenius-normalized; blocks receive equal mass within
    a source group and the three source groups receive equal total mass.  The
    balanced covariance is accumulated directly in FP64, so no potentially
    enormous mixed row bank is materialized.  Positive-evidence POD columns
    are followed, if necessary, by deterministic canonical complement axes;
    insufficient empirical rank therefore remains diagnostic rather than an
    automatic quality veto.
    """

    torch = _torch()
    if not isinstance(topology, Lfm2GlobalResidualWriterTopology) or not (
        topology.verify_authentication()
    ):
        raise Lfm2GlobalResidualClosureAuthenticationError(
            "topology failed authentication."
        )
    hidden = topology.hidden_size
    rank = topology.effective_basis_rank
    groups = (
        (
            "mapped_fit_residual_defects",
            _authenticated_tensor_mapping(
                mapped_fit_residual_defects,
                mapped_fit_residual_defects_sha256,
                "mapped_fit_residual_defects",
            ),
            "rows_last_dimension_hidden",
        ),
        (
            "source_writer_weights",
            _authenticated_tensor_mapping(
                source_writer_weights,
                source_writer_weights_sha256,
                "source_writer_weights",
            ),
            "hidden_by_input_columns_transposed_to_rows",
        ),
        (
            "terminal_tied_table_directions",
            _authenticated_tensor_mapping(
                terminal_tied_table_directions,
                terminal_tied_table_directions_sha256,
                "terminal_tied_table_directions",
            ),
            "rows_last_dimension_hidden",
        ),
    )
    prepared_groups = []
    for group_name, blocks, orientation in groups:
        prepared = []
        for block_id, tensor, digest in blocks:
            value = tensor.detach().to(device="cpu", dtype=torch.float64)
            if group_name == "source_writer_weights":
                if value.ndim != 2 or int(value.shape[0]) != hidden:
                    raise Lfm2GlobalResidualClosureError(
                        f"{group_name}.{block_id} must have shape [hidden,input]."
                    )
                rows = value.transpose(0, 1).contiguous()
            else:
                if value.ndim < 1 or int(value.shape[-1]) != hidden:
                    raise Lfm2GlobalResidualClosureError(
                        f"{group_name}.{block_id} must end in hidden_size."
                    )
                rows = value.reshape(-1, hidden).contiguous()
            frobenius = float(torch.linalg.vector_norm(rows).item())
            prepared.append((block_id, rows, digest, frobenius))
        nonzero = [item for item in prepared if item[3] > 0.0]
        prepared_groups.append((group_name, orientation, prepared, nonzero))

    covariance = torch.zeros((hidden, hidden), dtype=torch.float64)
    inventory = []
    group_reports = []
    active_group_count = sum(bool(group[3]) for group in prepared_groups)
    for group_name, orientation, prepared, nonzero in prepared_groups:
        block_scale = (
            0.0 if not nonzero else 1.0 / math.sqrt(active_group_count * len(nonzero))
        )
        for block_id, rows, digest, frobenius in prepared:
            contributes = active_group_count > 0 and frobenius > 0.0
            if contributes:
                balanced = rows * (block_scale / frobenius)
                covariance.add_(balanced.transpose(0, 1) @ balanced)
            inventory.append(
                {
                    "group": group_name,
                    "block_id": block_id,
                    "source_sha256": digest,
                    "source_shape": [int(size) for size in rows.shape],
                    "source_orientation": orientation,
                    "frobenius_norm": frobenius,
                    "contributed_to_covariance": contributes,
                }
            )
        group_reports.append(
            {
                "group": group_name,
                "block_count": len(prepared),
                "nonzero_block_count": len(nonzero),
                "group_skipped_as_zero": not bool(nonzero),
                "balanced_total_frobenius_squared_mass": (
                    0.0 if not nonzero else 1.0 / active_group_count
                ),
            }
        )
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.arange(hidden - 1, -1, -1, dtype=torch.long)
    eigenvalues = eigenvalues[order].clamp(min=0.0).contiguous()
    eigenvectors = eigenvectors[:, order].contiguous()
    largest = float(eigenvalues[0].item())
    tolerance = torch.finfo(torch.float64).eps * hidden * max(largest, 1.0)
    evidence_rank = int((eigenvalues > tolerance).sum().item())
    evidence_columns = min(rank, evidence_rank)
    initial = _canonicalize_column_signs(eigenvectors[:, :evidence_columns])
    basis, completion_columns = _complete_orthonormal_columns(
        initial,
        hidden_size=hidden,
        rank=rank,
    )
    basis = _canonicalize_column_signs(basis)
    gram = basis.transpose(0, 1) @ basis
    orthogonality_error = float(
        (gram - torch.eye(rank, dtype=torch.float64)).abs().max().item()
    )
    qdeim_coordinates = _deterministic_qdeim_coordinates(basis)
    interpolation = basis[
        torch.tensor(qdeim_coordinates, dtype=torch.long),
        :,
    ]
    interpolation_condition = float(torch.linalg.cond(interpolation).item())
    retained_eigenvalues = eigenvalues[:rank].clone()
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "topology_rank_mixed_output_basis",
        "implementation": _IMPLEMENTATION,
        "topology_sha256": topology.topology_sha256,
        "hidden_size": hidden,
        "head_dimension": topology.head_dimension,
        "topology_rank_formula": "2*head_dimension",
        "topology_rank_before_hidden_clip": topology.topology_basis_rank,
        "basis_rank": rank,
        "canonical_lfm2_rank_128": topology.topology_basis_rank == 128,
        "basis_construction": (
            "equal-group-equal-block-frobenius-balanced-fp64-pod-"
            "with-deterministic-canonical-completion"
        ),
        "mixed_source_groups": [value[0] for value in groups],
        "all_three_source_groups_present": all(
            bool(group[2]) for group in prepared_groups
        ),
        "nonzero_source_group_count": active_group_count,
        "zero_source_groups_skipped": [
            group_name
            for group_name, _orientation, _prepared, nonzero in prepared_groups
            if not nonzero
        ],
        "all_zero_evidence_uses_canonical_completion": active_group_count == 0,
        "group_reports": group_reports,
        "source_inventory": inventory,
        "source_inventory_sha256": _sha256(inventory),
        "balanced_covariance_materialized": True,
        "mixed_row_bank_materialized": False,
        "evidence_rank": evidence_rank,
        "evidence_rank_tolerance": tolerance,
        "evidence_pod_columns": evidence_columns,
        "deterministic_completion_columns": completion_columns,
        "orthogonality_maximum_absolute_error": orthogonality_error,
        "qdeim_method": "deterministic-column-pivoted-modified-gram-schmidt",
        "qdeim_coordinates": list(qdeim_coordinates),
        "qdeim_interpolation_condition_number": interpolation_condition,
        "output_basis_sha256": fingerprint_lfm2_global_residual_tensor(basis),
        "evidence_eigenvalues_sha256": (
            fingerprint_lfm2_global_residual_tensor(retained_eigenvalues)
        ),
        "fit_only": True,
        "numeric_diagnostics_used_for_semantic_decision": False,
        "numeric_diagnostics_used_for_auto_veto": False,
        "execution": {
            **_ZERO_EXECUTION,
            "autograd_graphs": 0,
            "fp64_covariance_eigendecompositions": 1,
        },
    }
    return Lfm2GlobalResidualBasis(
        output_basis=basis,
        qdeim_coordinates=qdeim_coordinates,
        evidence_eigenvalues=retained_eigenvalues,
        report=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualBoundary:
    """One authenticated FIT residual defect with normalized row mass."""

    boundary_id: str
    layer_index: int
    causal_order: int
    boundary_kind: str
    defect: Any
    row_mass: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        defect = (
            _plain_finite_tensor(self.defect, "boundary.defect")
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        mass = (
            _plain_finite_tensor(self.row_mass, "boundary.row_mass")
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        object.__setattr__(self, "defect", defect)
        object.__setattr__(self, "row_mass", mass)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_global_residual_closure_report(self.report)
            and self.report.get("boundary_id") == self.boundary_id
            and self.report.get("layer_index") == self.layer_index
            and self.report.get("causal_order") == self.causal_order
            and self.report.get("boundary_kind") == self.boundary_kind
            and self.report.get("defect_sha256")
            == fingerprint_lfm2_global_residual_tensor(self.defect)
            and self.report.get("normalized_row_mass_sha256")
            == fingerprint_lfm2_global_residual_tensor(self.row_mass)
        )


def seal_lfm2_global_residual_boundary(
    *,
    boundary_id: str,
    layer_index: int,
    causal_order: int,
    boundary_kind: str,
    defect: Any,
    defect_sha256: str,
    row_mass: Any,
    row_mass_sha256: str,
) -> Lfm2GlobalResidualBoundary:
    """Authenticate and seal one FIT boundary without a quality decision."""

    torch = _torch()
    clean_id = str(boundary_id).strip()
    if not clean_id:
        raise Lfm2GlobalResidualClosureError("boundary_id cannot be empty.")
    layer = _nonnegative_int(layer_index, "layer_index")
    order = _nonnegative_int(causal_order, "causal_order")
    kind = str(boundary_kind).strip().lower()
    if kind not in _BOUNDARY_KINDS:
        raise Lfm2GlobalResidualClosureError(
            f"boundary_kind must be one of {sorted(_BOUNDARY_KINDS)}."
        )
    defect_value = _plain_finite_tensor(defect, "defect")
    if defect_value.ndim < 2:
        raise Lfm2GlobalResidualClosureError(
            "defect must have at least one row dimension and one hidden dimension."
        )
    mass_value = _plain_finite_tensor(row_mass, "row_mass")
    if tuple(mass_value.shape) != tuple(defect_value.shape[:-1]):
        raise Lfm2GlobalResidualClosureError(
            "row_mass must match all non-hidden defect dimensions."
        )
    if bool((mass_value < 0).any().item()):
        raise Lfm2GlobalResidualClosureError("row_mass cannot be negative.")
    total_mass = float(mass_value.to(dtype=torch.float64).sum().item())
    if total_mass <= 0.0:
        raise Lfm2GlobalResidualClosureError("row_mass must have positive mass.")
    defect_source_sha = _authenticate_tensor(defect_value, defect_sha256, "defect")
    mass_source_sha = _authenticate_tensor(
        mass_value,
        row_mass_sha256,
        "row_mass",
    )
    normalized_mass = (
        mass_value.detach().to(device="cpu", dtype=torch.float64) / total_mass
    ).contiguous()
    archived_defect = (
        defect_value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    )
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "fit_residual_boundary",
        "boundary_id": clean_id,
        "layer_index": layer,
        "causal_order": order,
        "boundary_kind": kind,
        "defect_shape": [int(size) for size in archived_defect.shape],
        "source_defect_sha256": defect_source_sha,
        "source_row_mass_sha256": mass_source_sha,
        "source_row_mass_sum": total_mass,
        "defect_sha256": fingerprint_lfm2_global_residual_tensor(archived_defect),
        "normalized_row_mass_sha256": (
            fingerprint_lfm2_global_residual_tensor(normalized_mass)
        ),
        "normalized_row_mass_sum": float(normalized_mass.sum().item()),
        "split": "fit",
        "quality_gate": False,
    }
    return Lfm2GlobalResidualBoundary(
        boundary_id=clean_id,
        layer_index=layer,
        causal_order=order,
        boundary_kind=kind,
        defect=archived_defect,
        row_mass=normalized_mass,
        report=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualBoundaryBundle:
    """Causally ordered boundary inventory with equal boundary mass."""

    boundaries: tuple[Lfm2GlobalResidualBoundary, ...]
    bundle_sha256: str = ""

    def __post_init__(self) -> None:
        boundaries = tuple(self.boundaries)
        if not boundaries or not all(
            isinstance(boundary, Lfm2GlobalResidualBoundary)
            and boundary.verify_authentication()
            for boundary in boundaries
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "Every boundary must be authenticated."
            )
        if (
            tuple(sorted(boundaries, key=lambda boundary: boundary.causal_order))
            != boundaries
        ):
            raise Lfm2GlobalResidualClosureError(
                "boundaries must be supplied in increasing causal order."
            )
        if len({boundary.boundary_id for boundary in boundaries}) != len(boundaries):
            raise Lfm2GlobalResidualClosureError("boundary ids must be unique.")
        if len({boundary.causal_order for boundary in boundaries}) != len(boundaries):
            raise Lfm2GlobalResidualClosureError(
                "boundary causal_order values must be unique."
            )
        object.__setattr__(self, "boundaries", boundaries)
        expected = _sha256(self.authentication_body())
        supplied = str(self.bundle_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "boundary bundle_sha256 is incorrect."
            )
        object.__setattr__(self, "bundle_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
            "kind": "equal_mass_fit_boundary_bundle",
            "boundary_count": len(self.boundaries),
            "boundary_report_sha256": [
                boundary.report["report_sha256"] for boundary in self.boundaries
            ],
            "boundary_ids": [boundary.boundary_id for boundary in self.boundaries],
            "equal_total_mass_per_boundary": True,
            "coordinate_average_per_boundary": True,
            "fit_only": True,
        }

    def verify_authentication(self) -> bool:
        return all(
            boundary.verify_authentication() for boundary in self.boundaries
        ) and self.bundle_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "bundle_sha256": self.bundle_sha256,
        }


def seal_lfm2_global_residual_linearization_receipt(
    topology: Lfm2GlobalResidualWriterTopology,
    boundaries: Lfm2GlobalResidualBoundaryBundle,
    *,
    kernel_id: str,
    source_sha256: Mapping[str, str],
    analytic_math_components: Sequence[str],
    operator_squared_norm_upper_bound: float,
    block_squared_norm_upper_bounds: Sequence[float],
    operator_squared_conditioning_scale: float | None = None,
    block_squared_conditioning_scales: Sequence[float] | None = None,
    conditioning_scale_method: str = "certified_norm_bound_fallback",
) -> Mapping[str, Any]:
    """Seal the adapter-owned analytic JVP/VJP contract consumed by the core.

    Certified norm bounds and useful numerical conditioning scales are separate
    quantities.  A deep residual stack can have a mathematically valid but
    extremely loose product bound; using that product as a ridge scale can
    erase the correction.  Callers may therefore supply deterministic scales
    in the weighted normal-operator units used by the solve.  Those scales tune
    ridge/preconditioning only and never admit, veto, or publish a candidate.
    """

    if not topology.verify_authentication() or not boundaries.verify_authentication():
        raise Lfm2GlobalResidualClosureAuthenticationError(
            "linearization inputs failed authentication."
        )
    clean_kernel = str(kernel_id).strip()
    if not clean_kernel:
        raise Lfm2GlobalResidualClosureError("kernel_id cannot be empty.")
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise Lfm2GlobalResidualClosureError("source_sha256 cannot be empty.")
    authenticated_sources = {
        str(name): _require_digest(digest, f"source_sha256.{name}")
        for name, digest in sorted(source_sha256.items())
    }
    components = tuple(str(value).strip() for value in analytic_math_components)
    if not components or any(not value for value in components):
        raise Lfm2GlobalResidualClosureError(
            "analytic_math_components cannot be empty."
        )
    operator_bound = _finite_nonnegative(
        operator_squared_norm_upper_bound,
        "operator_squared_norm_upper_bound",
    )
    block_bounds = tuple(
        _finite_nonnegative(value, f"block_squared_norm_upper_bounds[{index}]")
        for index, value in enumerate(block_squared_norm_upper_bounds)
    )
    if len(block_bounds) != len(topology.sites):
        raise Lfm2GlobalResidualClosureError(
            "block_squared_norm_upper_bounds must have one value per writer site."
        )
    if any(value > operator_bound for value in block_bounds):
        raise Lfm2GlobalResidualClosureError(
            "A block norm bound cannot exceed the global operator bound."
        )
    if (operator_squared_conditioning_scale is None) != (
        block_squared_conditioning_scales is None
    ):
        raise Lfm2GlobalResidualClosureError(
            "Global and per-block conditioning scales must be supplied together."
        )
    if operator_squared_conditioning_scale is None:
        operator_scale = operator_bound
        block_scales = block_bounds
        scale_method = "certified_norm_bound_fallback"
        certified_scale = True
    else:
        operator_scale = _finite_nonnegative(
            operator_squared_conditioning_scale,
            "operator_squared_conditioning_scale",
        )
        block_scales = tuple(
            _finite_nonnegative(
                value,
                f"block_squared_conditioning_scales[{index}]",
            )
            for index, value in enumerate(block_squared_conditioning_scales or ())
        )
        if len(block_scales) != len(topology.sites):
            raise Lfm2GlobalResidualClosureError(
                "block_squared_conditioning_scales must have one value per "
                "writer site."
            )
        scale_method = str(conditioning_scale_method).strip()
        if not scale_method or scale_method == "certified_norm_bound_fallback":
            raise Lfm2GlobalResidualClosureError(
                "An explicit conditioning scale requires a descriptive non-bound "
                "method."
            )
        certified_scale = False
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_LINEARIZATION_SCHEMA,
        "kind": "analytic_global_causal_writer_to_boundary_linearization",
        "kernel_id": clean_kernel,
        "topology_sha256": topology.topology_sha256,
        "boundary_bundle_sha256": boundaries.bundle_sha256,
        "source_sha256": authenticated_sources,
        "source_bundle_sha256": _sha256(authenticated_sources),
        "analytic_math_components": list(components),
        "parameterization": "delta_writer=output_basis@coefficient_block",
        "block_lower_triangular_causal_operator": True,
        "analytic_jvp": True,
        "analytic_vjp": True,
        "operator_squared_norm_upper_bound": operator_bound,
        "block_squared_norm_upper_bounds": list(block_bounds),
        "operator_squared_conditioning_scale": operator_scale,
        "block_squared_conditioning_scales": list(block_scales),
        "conditioning_scale_method": scale_method,
        "conditioning_scale_is_certified_norm_bound": certified_scale,
        "conditioning_scale_has_quality_gate_authority": False,
        "normal_design_materialized": False,
        "gram_materialized": False,
        "execution": {
            **_ZERO_EXECUTION,
            "autograd_graphs": 0,
        },
    }
    return _seal_report(body)


def _plain_kernel_state_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or isinstance(value, torch.nn.Parameter)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
        or value.requires_grad
        or value.is_complex()
        or (
            torch.is_floating_point(value)
            and not bool(torch.isfinite(value).all().item())
        )
    ):
        raise Lfm2GlobalResidualClosureError(
            f"{name} must be a finite materialized tensor-only kernel state "
            "value that is not a Parameter and does not require gradients."
        )
    return value.detach()


def _tensor_storage_identity(value: Any) -> tuple[str, int]:
    """Return a process-local storage identity for alias rejection."""

    return (str(value.device), int(value.untyped_storage().data_ptr()))


BlockSequence = tuple[Any, ...]


class Lfm2GlobalResidualLinearizationKernel:
    """Immutable authenticated base for tensor-only JVP/VJP kernels.

    Subclasses must declare ``__slots__ = ()`` and implement only the two
    chunk ``_tensor_only`` methods.  They may additionally implement both
    bundle methods.  A bundle implementation traverses the whole causal stack
    once, archives every boundary effect directly on CPU, and performs one
    exact reverse traversal for all boundary cotangents.  It must not retain a
    mutable cache between calls.

    The base clones every supplied state tensor to CPU, converts floating
    primal/parameter state to FP64, freezes all object attributes, and owns all
    public call paths.  The closure operator calls the base methods unbound, so
    a subclass cannot bypass state, basis, or input mutation checks by
    overriding a public method.
    """

    __slots__ = (
        "topology",
        "boundaries",
        "linearization_receipt",
        "transpose_contract",
        "transpose_contract_sha256",
        "authentication_sha256",
        "_tensor_state",
        "_tensor_inventory",
        "_sealed",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("__slots__") not in {(), ""}:
            raise TypeError(
                "Global residual linearization kernel subclasses must declare "
                "__slots__ = () so model handles and mutable captured state "
                "cannot be attached to an instance."
            )
        forbidden_public_overrides = sorted(
            name
            for name in (
                "run_jvp_chunk",
                "run_vjp_chunk",
                "run_jvp_bundle",
                "run_vjp_bundle",
            )
            if name in cls.__dict__
        )
        if forbidden_public_overrides:
            raise TypeError(
                "Global residual linearization kernels cannot override the "
                "authenticated public call path: "
                + ", ".join(forbidden_public_overrides)
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise Lfm2GlobalResidualClosureError(
                "The authenticated tensor-only linearization kernel is immutable."
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        topology: Lfm2GlobalResidualWriterTopology,
        boundaries: Lfm2GlobalResidualBoundaryBundle,
        *,
        tensor_state: Mapping[str, Any],
        tensor_state_sha256: Mapping[str, str],
        linearization_receipt: Mapping[str, Any],
        transpose_contract_body: Mapping[str, Any],
    ) -> None:
        torch = _torch()
        object.__setattr__(self, "_sealed", False)
        if (
            not topology.verify_authentication()
            or not boundaries.verify_authentication()
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "kernel topology or boundary bundle failed authentication."
            )
        if not isinstance(tensor_state, Mapping) or not tensor_state:
            raise Lfm2GlobalResidualClosureError(
                "tensor_state must be a nonempty tensor mapping."
            )
        if not isinstance(tensor_state_sha256, Mapping) or set(tensor_state) != set(
            tensor_state_sha256
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "tensor_state and tensor_state_sha256 must have identical keys."
            )
        cloned: dict[str, Any] = {}
        inventory = []
        for state_name in sorted(tensor_state):
            clean_name = str(state_name).strip()
            if not clean_name or clean_name != state_name:
                raise Lfm2GlobalResidualClosureError(
                    "tensor_state keys must be nonempty canonical strings."
                )
            source = _plain_kernel_state_tensor(
                tensor_state[state_name],
                f"tensor_state.{state_name}",
            )
            source_sha = _authenticate_tensor(
                source,
                tensor_state_sha256[state_name],
                f"tensor_state.{state_name}",
            )
            target_dtype = (
                torch.float64 if torch.is_floating_point(source) else source.dtype
            )
            archived = source.to(device="cpu", dtype=target_dtype).contiguous().clone()
            internal_sha = fingerprint_lfm2_global_residual_tensor(archived)
            cloned[state_name] = archived
            inventory.append(
                {
                    "name": state_name,
                    "source_sha256": source_sha,
                    "internal_sha256": internal_sha,
                    "internal_shape": [int(size) for size in archived.shape],
                    "internal_dtype": _dtype_name(archived),
                    "sealed_tensor_version": int(archived._version),
                    "storage_device": "cpu",
                }
            )
        receipt = MappingProxyType(dict(linearization_receipt))
        if (
            not verify_lfm2_global_residual_closure_report(receipt)
            or receipt.get("schema") != LFM2_GLOBAL_RESIDUAL_LINEARIZATION_SCHEMA
            or receipt.get("topology_sha256") != topology.topology_sha256
            or receipt.get("boundary_bundle_sha256") != boundaries.bundle_sha256
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "kernel linearization receipt failed exact binding authentication."
            )
        if not isinstance(transpose_contract_body, Mapping) or not (
            transpose_contract_body
        ):
            raise Lfm2GlobalResidualClosureError(
                "transpose_contract_body must be a nonempty mapping."
            )
        contract_body = {
            "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
            "kind": "manual_exact_transpose_construction_contract",
            "topology_sha256": topology.topology_sha256,
            "boundary_bundle_sha256": boundaries.bundle_sha256,
            "linearization_report_sha256": receipt["report_sha256"],
            "construction": dict(transpose_contract_body),
            "vjp_is_reverse_composition_of_jvp": True,
            "autograd_used": False,
        }
        contract_sha = _sha256(contract_body)
        contract = MappingProxyType(
            {**contract_body, "transpose_contract_sha256": contract_sha}
        )
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "linearization_receipt", receipt)
        object.__setattr__(self, "transpose_contract", contract)
        object.__setattr__(self, "transpose_contract_sha256", contract_sha)
        object.__setattr__(self, "_tensor_state", MappingProxyType(cloned))
        object.__setattr__(self, "_tensor_inventory", tuple(inventory))
        # A fast path is valid only as an exact JVP/VJP pair.  Reject a partial
        # implementation before sealing the kernel rather than silently mixing
        # a bundle forward with repeated chunk reverse traversals.
        _ = self.supports_bundle_linearization
        authentication_sha = self.recompute_authentication_sha256()
        object.__setattr__(self, "authentication_sha256", authentication_sha)
        object.__setattr__(self, "_sealed", True)

    @property
    def state_storage_device(self) -> str:
        return "cpu"

    @property
    def tensor_state_names(self) -> tuple[str, ...]:
        return tuple(value["name"] for value in self._tensor_inventory)

    @property
    def supports_bundle_linearization(self) -> bool:
        """Whether this kernel supplies one whole-stack JVP/VJP pair."""

        kernel_type = type(self)
        jvp = (
            kernel_type._jvp_bundle_tensor_only
            is not Lfm2GlobalResidualLinearizationKernel._jvp_bundle_tensor_only
        )
        vjp = (
            kernel_type._vjp_bundle_tensor_only
            is not Lfm2GlobalResidualLinearizationKernel._vjp_bundle_tensor_only
        )
        if jvp != vjp:
            raise Lfm2GlobalResidualClosureError(
                "A whole-stack kernel must implement both bundle JVP and "
                "bundle VJP, or neither."
            )
        return jvp

    def _current_tensor_inventory(
        self, *, force_raw: bool = False
    ) -> list[dict[str, Any]]:
        current = []
        for sealed in self._tensor_inventory:
            tensor = self._tensor_state[sealed["name"]]
            version = int(tensor._version)
            digest = (
                fingerprint_lfm2_global_residual_tensor(tensor)
                if force_raw or version != sealed["sealed_tensor_version"]
                else sealed["internal_sha256"]
            )
            current.append(
                {
                    **sealed,
                    "current_internal_sha256": digest,
                    "current_tensor_version": version,
                }
            )
        return current

    def authentication_body(self, *, force_raw: bool = False) -> dict[str, Any]:
        inventory = self._current_tensor_inventory(force_raw=force_raw)
        return {
            "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
            "kind": "immutable_tensor_only_linearization_kernel",
            "implementation_type": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "topology_sha256": self.topology.topology_sha256,
            "boundary_bundle_sha256": self.boundaries.bundle_sha256,
            "linearization_report_sha256": self.linearization_receipt["report_sha256"],
            "transpose_contract_sha256": self.transpose_contract_sha256,
            "tensor_inventory": inventory,
            "tensor_state_cloned": True,
            "tensor_state_storage_device": "cpu",
            "floating_tensor_state_dtype": "float64",
            "model_handles_present": False,
            "whole_stack_bundle_linearization": (self.supports_bundle_linearization),
            "bundle_output_storage_device": (
                "cpu" if self.supports_bundle_linearization else None
            ),
            "bundle_mutable_cache_allowed": False,
        }

    def recompute_authentication_sha256(self, *, force_raw: bool = False) -> str:
        return _sha256(self.authentication_body(force_raw=force_raw))

    def verify_authentication(self, *, force_raw: bool = False) -> bool:
        return self.authentication_sha256 == self.recompute_authentication_sha256(
            force_raw=force_raw
        )

    def assert_authenticated(self, phase: str, *, force_raw: bool = False) -> str:
        observed = self.recompute_authentication_sha256(force_raw=force_raw)
        if observed != self.authentication_sha256:
            raise Lfm2GlobalResidualClosureAuthenticationError(
                f"Tensor-only linearization kernel mutated during {phase}."
            )
        return observed

    def _jvp_chunk_tensor_only(
        self,
        state: Mapping[str, Any],
        coefficient_blocks: BlockSequence,
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> Any:
        raise NotImplementedError

    def _vjp_chunk_tensor_only(
        self,
        state: Mapping[str, Any],
        boundary_cotangent: Any,
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> BlockSequence:
        raise NotImplementedError

    def _jvp_bundle_tensor_only(
        self,
        state: Mapping[str, Any],
        coefficient_blocks: BlockSequence,
        output_basis: Any,
        *,
        maximum_workspace_bytes: int,
    ) -> BlockSequence:
        """Optional whole-stack forward traversal implemented by subclasses.

        The returned tensors must be flattened, contiguous CPU FP64 boundary
        archives.  Their ownership transfers to the closure core; they must
        not alias kernel state, the basis, or coefficient inputs.
        """

        del state, coefficient_blocks, output_basis, maximum_workspace_bytes
        raise NotImplementedError

    def _vjp_bundle_tensor_only(
        self,
        state: Mapping[str, Any],
        boundary_cotangents_cpu: BlockSequence,
        output_basis: Any,
        *,
        boundary_row_scales_cpu: BlockSequence | None,
        maximum_workspace_bytes: int,
    ) -> BlockSequence:
        """Optional exact whole-stack reverse traversal.

        Cotangents and optional row scales remain CPU archives.  A concrete
        kernel stages only its current batch on the compute device and must not
        mutate or retain any supplied tensor.
        """

        del (
            state,
            boundary_cotangents_cpu,
            output_basis,
            boundary_row_scales_cpu,
            maximum_workspace_bytes,
        )
        raise NotImplementedError

    def run_jvp_chunk(
        self,
        coefficient_blocks: BlockSequence,
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> Any:
        self.assert_authenticated("JVP preflight")
        basis = output_basis.detach().clone()
        coefficients = tuple(value.detach().clone() for value in coefficient_blocks)
        basis_sha = fingerprint_lfm2_global_residual_tensor(basis)
        basis_version = int(basis._version)
        coefficient_versions = tuple(int(value._version) for value in coefficients)
        result = self._jvp_chunk_tensor_only(
            self._tensor_state,
            coefficients,
            basis,
            boundary_index=boundary_index,
            row_start=row_start,
            row_stop=row_stop,
        )
        self.assert_authenticated("JVP completion")
        if (
            int(basis._version) != basis_version
            or fingerprint_lfm2_global_residual_tensor(basis) != basis_sha
            or tuple(int(value._version) for value in coefficients)
            != coefficient_versions
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "Tensor-only JVP attempted to mutate cloned basis or coefficients."
            )
        value = _plain_finite_tensor(result, "kernel.jvp_chunk", dimensions=2)
        expected_rows = row_stop - row_start
        if tuple(value.shape) != (expected_rows, self.topology.hidden_size):
            raise Lfm2GlobalResidualClosureError(
                "Tensor-only JVP returned incorrect chunk geometry."
            )
        return value.detach().clone()

    def run_vjp_chunk(
        self,
        boundary_cotangent: Any,
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> BlockSequence:
        self.assert_authenticated("VJP preflight")
        basis = output_basis.detach().clone()
        cotangent = boundary_cotangent.detach().clone()
        basis_sha = fingerprint_lfm2_global_residual_tensor(basis)
        basis_version = int(basis._version)
        cotangent_version = int(cotangent._version)
        result = tuple(
            self._vjp_chunk_tensor_only(
                self._tensor_state,
                cotangent,
                basis,
                boundary_index=boundary_index,
                row_start=row_start,
                row_stop=row_stop,
            )
        )
        self.assert_authenticated("VJP completion")
        if (
            int(basis._version) != basis_version
            or fingerprint_lfm2_global_residual_tensor(basis) != basis_sha
            or int(cotangent._version) != cotangent_version
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "Tensor-only VJP attempted to mutate cloned basis or cotangent."
            )
        if len(result) != len(self.topology.sites):
            raise Lfm2GlobalResidualClosureError(
                "Tensor-only VJP must return one block per writer site."
            )
        archived = []
        rank = int(basis.shape[1])
        for site, value in zip(self.topology.sites, result, strict=True):
            block = _plain_finite_tensor(
                value,
                f"kernel.vjp_chunk.{site.site_id}",
                dimensions=2,
            )
            if tuple(block.shape) != (rank, site.input_width):
                raise Lfm2GlobalResidualClosureError(
                    f"Tensor-only VJP block {site.site_id} has wrong geometry."
                )
            archived.append(block.detach().clone())
        return tuple(archived)

    def run_jvp_bundle(
        self,
        coefficient_blocks: BlockSequence,
        output_basis: Any,
        *,
        maximum_workspace_bytes: int,
    ) -> BlockSequence:
        """Run one authenticated whole-stack JVP into owned CPU archives."""

        torch = _torch()
        if not self.supports_bundle_linearization:
            raise Lfm2GlobalResidualClosureError(
                "This tensor-only kernel has no whole-stack bundle fast path."
            )
        workspace = _positive_int(
            maximum_workspace_bytes,
            "maximum_workspace_bytes",
        )
        self.assert_authenticated("bundle JVP preflight")
        source_basis = _plain_finite_tensor(
            output_basis,
            "bundle.output_basis",
            dimensions=2,
        )
        if (
            tuple(source_basis.shape)
            != (
                self.topology.hidden_size,
                self.topology.effective_basis_rank,
            )
            or source_basis.dtype != torch.float64
        ):
            raise Lfm2GlobalResidualClosureError(
                "Bundle output_basis must have topology geometry and FP64 dtype."
            )
        source_coefficients = tuple(coefficient_blocks)
        if len(source_coefficients) != len(self.topology.sites):
            raise Lfm2GlobalResidualClosureError(
                "Bundle JVP requires one coefficient block per writer site."
            )
        basis = source_basis.detach().clone()
        coefficients = []
        for site, value in zip(
            self.topology.sites,
            source_coefficients,
            strict=True,
        ):
            block = _plain_finite_tensor(
                value,
                f"bundle.coefficient_blocks.{site.site_id}",
                dimensions=2,
            )
            if (
                tuple(block.shape)
                != (self.topology.effective_basis_rank, site.input_width)
                or block.dtype != torch.float64
                or block.device != basis.device
            ):
                raise Lfm2GlobalResidualClosureError(
                    f"Bundle coefficient block {site.site_id} has incorrect "
                    "geometry, dtype, or device."
                )
            coefficients.append(block.detach().clone())
        coefficient_tuple = tuple(coefficients)
        basis_sha = fingerprint_lfm2_global_residual_tensor(basis)
        coefficient_sha = tuple(
            fingerprint_lfm2_global_residual_tensor(value)
            for value in coefficient_tuple
        )
        basis_version = int(basis._version)
        coefficient_versions = tuple(int(value._version) for value in coefficient_tuple)
        raw_result = self._jvp_bundle_tensor_only(
            self._tensor_state,
            coefficient_tuple,
            basis,
            maximum_workspace_bytes=workspace,
        )
        self.assert_authenticated("bundle JVP completion")
        if (
            int(basis._version) != basis_version
            or fingerprint_lfm2_global_residual_tensor(basis) != basis_sha
            or tuple(int(value._version) for value in coefficient_tuple)
            != coefficient_versions
            or tuple(
                fingerprint_lfm2_global_residual_tensor(value)
                for value in coefficient_tuple
            )
            != coefficient_sha
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "Tensor-only bundle JVP attempted to mutate cloned basis or "
                "coefficients."
            )
        try:
            result = tuple(raw_result)
        except TypeError as exc:
            raise Lfm2GlobalResidualClosureError(
                "Bundle JVP must return one CPU tensor per boundary."
            ) from exc
        if len(result) != len(self.boundaries.boundaries):
            raise Lfm2GlobalResidualClosureError(
                "Bundle JVP must return one CPU tensor per boundary."
            )
        forbidden_storage = {
            _tensor_storage_identity(value)
            for value in (
                *self._tensor_state.values(),
                basis,
                *coefficient_tuple,
            )
        }
        archived = []
        output_storage: set[tuple[str, int]] = set()
        for boundary, value in zip(
            self.boundaries.boundaries,
            result,
            strict=True,
        ):
            output = _plain_finite_tensor(
                value,
                f"kernel.jvp_bundle.{boundary.boundary_id}",
                dimensions=2,
            )
            expected_rows = boundary.defect.numel() // self.topology.hidden_size
            if (
                tuple(output.shape) != (expected_rows, self.topology.hidden_size)
                or output.device.type != "cpu"
                or output.dtype != torch.float64
                or not output.is_contiguous()
            ):
                raise Lfm2GlobalResidualClosureError(
                    f"Bundle JVP boundary {boundary.boundary_id} must be a "
                    "contiguous flattened CPU FP64 archive with exact geometry."
                )
            storage = _tensor_storage_identity(output)
            if storage in forbidden_storage:
                raise Lfm2GlobalResidualClosureAuthenticationError(
                    "Bundle JVP output aliases authenticated state, basis, or "
                    "coefficient storage."
                )
            if storage in output_storage:
                raise Lfm2GlobalResidualClosureAuthenticationError(
                    "Bundle JVP boundary outputs must own mutually independent storage."
                )
            output_storage.add(storage)
            # No clone: ownership of this fresh CPU archive transfers from the
            # immutable kernel call to the closure core.
            archived.append(output.detach())
        return tuple(archived)

    def run_vjp_bundle(
        self,
        boundary_cotangents_cpu: BlockSequence,
        output_basis: Any,
        *,
        boundary_row_scales_cpu: BlockSequence | None = None,
        maximum_workspace_bytes: int,
    ) -> BlockSequence:
        """Run one authenticated whole-stack VJP over CPU boundary archives."""

        torch = _torch()
        if not self.supports_bundle_linearization:
            raise Lfm2GlobalResidualClosureError(
                "This tensor-only kernel has no whole-stack bundle fast path."
            )
        workspace = _positive_int(
            maximum_workspace_bytes,
            "maximum_workspace_bytes",
        )
        self.assert_authenticated("bundle VJP preflight")
        source_basis = _plain_finite_tensor(
            output_basis,
            "bundle.output_basis",
            dimensions=2,
        )
        if (
            tuple(source_basis.shape)
            != (
                self.topology.hidden_size,
                self.topology.effective_basis_rank,
            )
            or source_basis.dtype != torch.float64
        ):
            raise Lfm2GlobalResidualClosureError(
                "Bundle output_basis must have topology geometry and FP64 dtype."
            )
        basis = source_basis.detach().clone()
        cotangent_values = tuple(boundary_cotangents_cpu)
        if len(cotangent_values) != len(self.boundaries.boundaries):
            raise Lfm2GlobalResidualClosureError(
                "Bundle VJP requires one CPU cotangent archive per boundary."
            )
        cotangents = []
        for boundary, value in zip(
            self.boundaries.boundaries,
            cotangent_values,
            strict=True,
        ):
            cotangent = _plain_finite_tensor(
                value,
                f"bundle.boundary_cotangents.{boundary.boundary_id}",
                dimensions=2,
            )
            expected_rows = boundary.defect.numel() // self.topology.hidden_size
            if (
                tuple(cotangent.shape) != (expected_rows, self.topology.hidden_size)
                or cotangent.device.type != "cpu"
                or cotangent.dtype != torch.float64
                or not cotangent.is_contiguous()
            ):
                raise Lfm2GlobalResidualClosureError(
                    f"Bundle VJP cotangent {boundary.boundary_id} must be a "
                    "contiguous flattened CPU FP64 archive with exact geometry."
                )
            # Bundle cotangents can be persistent operator evidence (the RHS)
            # rather than disposable effects.  Clone before crossing the
            # subclass boundary so a rejected implementation cannot corrupt
            # the caller's archive.
            cotangents.append(cotangent.detach().clone())
        cotangent_tuple = tuple(cotangents)
        scales: BlockSequence | None
        if boundary_row_scales_cpu is None:
            scales = None
        else:
            scale_values = tuple(boundary_row_scales_cpu)
            if len(scale_values) != len(self.boundaries.boundaries):
                raise Lfm2GlobalResidualClosureError(
                    "Bundle VJP row scales require one CPU vector per boundary."
                )
            validated_scales = []
            for boundary, cotangent, value in zip(
                self.boundaries.boundaries,
                cotangent_tuple,
                scale_values,
                strict=True,
            ):
                scale = _plain_finite_tensor(
                    value,
                    f"bundle.boundary_row_scales.{boundary.boundary_id}",
                    dimensions=1,
                )
                if (
                    tuple(scale.shape) != (int(cotangent.shape[0]),)
                    or scale.device.type != "cpu"
                    or scale.dtype != torch.float64
                    or not scale.is_contiguous()
                ):
                    raise Lfm2GlobalResidualClosureError(
                        f"Bundle VJP row scale {boundary.boundary_id} must be a "
                        "contiguous CPU FP64 vector with exact geometry."
                    )
                validated_scales.append(scale.detach().clone())
            scales = tuple(validated_scales)
        state_storage = {
            _tensor_storage_identity(value) for value in self._tensor_state.values()
        }
        if any(
            _tensor_storage_identity(value) in state_storage
            for value in (
                *cotangent_tuple,
                *(() if scales is None else scales),
            )
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "Bundle VJP inputs cannot alias authenticated kernel state."
            )
        basis_sha = fingerprint_lfm2_global_residual_tensor(basis)
        basis_version = int(basis._version)
        cotangent_versions = tuple(int(value._version) for value in cotangent_tuple)
        cotangent_storage = tuple(
            _tensor_storage_identity(value) for value in cotangent_tuple
        )
        cotangent_sha = tuple(
            fingerprint_lfm2_global_residual_tensor(value) for value in cotangent_tuple
        )
        scale_versions = (
            () if scales is None else tuple(int(value._version) for value in scales)
        )
        scale_storage = (
            ()
            if scales is None
            else tuple(_tensor_storage_identity(value) for value in scales)
        )
        scale_sha = (
            ()
            if scales is None
            else tuple(
                fingerprint_lfm2_global_residual_tensor(value) for value in scales
            )
        )
        raw_result = self._vjp_bundle_tensor_only(
            self._tensor_state,
            cotangent_tuple,
            basis,
            boundary_row_scales_cpu=scales,
            maximum_workspace_bytes=workspace,
        )
        self.assert_authenticated("bundle VJP completion")
        if (
            int(basis._version) != basis_version
            or fingerprint_lfm2_global_residual_tensor(basis) != basis_sha
            or tuple(int(value._version) for value in cotangent_tuple)
            != cotangent_versions
            or tuple(_tensor_storage_identity(value) for value in cotangent_tuple)
            != cotangent_storage
            or tuple(
                fingerprint_lfm2_global_residual_tensor(value)
                for value in cotangent_tuple
            )
            != cotangent_sha
            or (
                () if scales is None else tuple(int(value._version) for value in scales)
            )
            != scale_versions
            or (
                ()
                if scales is None
                else tuple(_tensor_storage_identity(value) for value in scales)
            )
            != scale_storage
            or (
                ()
                if scales is None
                else tuple(
                    fingerprint_lfm2_global_residual_tensor(value) for value in scales
                )
            )
            != scale_sha
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "Tensor-only bundle VJP attempted to mutate a basis, cotangent, "
                "or row-scale input."
            )
        try:
            result = tuple(raw_result)
        except TypeError as exc:
            raise Lfm2GlobalResidualClosureError(
                "Bundle VJP must return one block per writer site."
            ) from exc
        if len(result) != len(self.topology.sites):
            raise Lfm2GlobalResidualClosureError(
                "Bundle VJP must return one block per writer site."
            )
        archived = []
        rank = int(basis.shape[1])
        forbidden_result_storage = {
            _tensor_storage_identity(value)
            for value in (
                *self._tensor_state.values(),
                basis,
                *cotangent_tuple,
                *(() if scales is None else scales),
            )
        }
        result_storage: set[tuple[str, int]] = set()
        for site, value in zip(self.topology.sites, result, strict=True):
            block = _plain_finite_tensor(
                value,
                f"kernel.vjp_bundle.{site.site_id}",
                dimensions=2,
            )
            if (
                tuple(block.shape) != (rank, site.input_width)
                or block.dtype != torch.float64
            ):
                raise Lfm2GlobalResidualClosureError(
                    f"Bundle VJP block {site.site_id} has wrong geometry or dtype."
                )
            storage = _tensor_storage_identity(block)
            if storage in forbidden_result_storage:
                raise Lfm2GlobalResidualClosureAuthenticationError(
                    "Bundle VJP result aliases authenticated state or an input archive."
                )
            if storage in result_storage:
                raise Lfm2GlobalResidualClosureAuthenticationError(
                    "Bundle VJP coefficient blocks must own mutually independent "
                    "storage."
                )
            result_storage.add(storage)
            archived.append(block.detach().clone())
        return tuple(archived)


def apply_lfm2_global_residual_writer_block(
    producer_rows: Any,
    output_basis: Any,
    coefficient_block: Any,
) -> Any:
    """Apply one ``U @ C`` writer correction without materializing ``U @ C``."""

    rows = _plain_finite_tensor(producer_rows, "producer_rows")
    basis = _plain_finite_tensor(output_basis, "output_basis", dimensions=2)
    coefficients = _plain_finite_tensor(
        coefficient_block,
        "coefficient_block",
        dimensions=2,
    )
    if int(rows.shape[-1]) != int(coefficients.shape[1]):
        raise Lfm2GlobalResidualClosureError(
            "producer input width differs from coefficient_block."
        )
    if int(basis.shape[1]) != int(coefficients.shape[0]):
        raise Lfm2GlobalResidualClosureError(
            "output basis rank differs from coefficient_block."
        )
    if not (
        rows.device == basis.device == coefficients.device
        and rows.dtype == basis.dtype == coefficients.dtype
    ):
        raise Lfm2GlobalResidualClosureError(
            "writer-block operands must share one device and dtype."
        )
    return (rows @ coefficients.transpose(0, 1)) @ basis.transpose(0, 1)


def adjoint_lfm2_global_residual_writer_block(
    producer_rows: Any,
    output_basis: Any,
    output_cotangent: Any,
) -> Any:
    """Exact coefficient-space transpose of one writer-block application."""

    rows = _plain_finite_tensor(producer_rows, "producer_rows")
    basis = _plain_finite_tensor(output_basis, "output_basis", dimensions=2)
    cotangent = _plain_finite_tensor(output_cotangent, "output_cotangent")
    if tuple(rows.shape[:-1]) != tuple(cotangent.shape[:-1]):
        raise Lfm2GlobalResidualClosureError(
            "producer rows and output cotangent must share row geometry."
        )
    if int(cotangent.shape[-1]) != int(basis.shape[0]):
        raise Lfm2GlobalResidualClosureError(
            "output cotangent hidden width differs from output_basis."
        )
    if not (
        rows.device == basis.device == cotangent.device
        and rows.dtype == basis.dtype == cotangent.dtype
    ):
        raise Lfm2GlobalResidualClosureError(
            "writer-block adjoint operands must share one device and dtype."
        )
    row_width = int(rows.shape[-1])
    hidden = int(cotangent.shape[-1])
    row_count = rows.numel() // row_width
    coordinate_cotangent = cotangent.reshape(row_count, hidden) @ basis
    return coordinate_cotangent.transpose(0, 1) @ rows.reshape(row_count, row_width)


def _resolve_device(policy: Lfm2GlobalResidualClosurePolicy) -> Any:
    torch = _torch()
    cuda_available = bool(torch.cuda.is_available())
    if policy.backend == "cuda" and not cuda_available:
        raise Lfm2GlobalResidualClosureError(
            "backend='cuda' was requested but CUDA is unavailable."
        )
    if policy.backend == "cuda" or (policy.backend == "auto" and cuda_available):
        return torch.device("cuda")
    return torch.device("cpu")


class Lfm2GlobalResidualClosureOperator:
    """Authenticated, CPU-streamed whole-stack writer-to-boundary operator."""

    def __init__(
        self,
        topology: Lfm2GlobalResidualWriterTopology,
        basis: Lfm2GlobalResidualBasis,
        boundaries: Lfm2GlobalResidualBoundaryBundle,
        *,
        kernel: Lfm2GlobalResidualLinearizationKernel,
        policy: Lfm2GlobalResidualClosurePolicy | None = None,
    ) -> None:
        torch = _torch()
        self.topology = topology
        self.basis = basis
        self.boundaries = boundaries
        self.policy = policy or Lfm2GlobalResidualClosurePolicy()
        if not topology.verify_authentication():
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "operator topology failed authentication."
            )
        if not basis.verify_authentication():
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "operator basis failed authentication."
            )
        if not boundaries.verify_authentication():
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "operator boundaries failed authentication."
            )
        if not self.policy.verify_authentication():
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "operator policy failed authentication."
            )
        if basis.hidden_size != topology.hidden_size or (
            basis.rank != topology.effective_basis_rank
        ):
            raise Lfm2GlobalResidualClosureError(
                "basis geometry differs from writer topology."
            )
        if any(
            int(boundary.defect.shape[-1]) != topology.hidden_size
            for boundary in boundaries.boundaries
        ):
            raise Lfm2GlobalResidualClosureError(
                "Every boundary defect must use the topology hidden width."
            )
        if not isinstance(kernel, Lfm2GlobalResidualLinearizationKernel):
            raise TypeError(
                "kernel must derive from Lfm2GlobalResidualLinearizationKernel."
            )
        if not kernel.verify_authentication(force_raw=True):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "tensor-only linearization kernel failed authentication."
            )
        receipt = kernel.linearization_receipt
        if (
            not verify_lfm2_global_residual_closure_report(receipt)
            or receipt.get("schema") != LFM2_GLOBAL_RESIDUAL_LINEARIZATION_SCHEMA
            or receipt.get("topology_sha256") != topology.topology_sha256
            or receipt.get("boundary_bundle_sha256") != boundaries.bundle_sha256
            or receipt.get("analytic_jvp") is not True
            or receipt.get("analytic_vjp") is not True
            or receipt.get("block_lower_triangular_causal_operator") is not True
            or receipt.get("normal_design_materialized") is not False
            or receipt.get("gram_materialized") is not False
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "linearization receipt does not bind the declared analytic problem."
            )
        block_bounds = tuple(receipt.get("block_squared_norm_upper_bounds", ()))
        block_scales = tuple(
            receipt.get("block_squared_conditioning_scales", block_bounds)
        )
        if (
            len(block_bounds) != len(topology.sites)
            or len(block_scales) != len(topology.sites)
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "linearization receipt has the wrong block-bound or conditioning-"
                "scale inventory."
            )
        if (
            kernel.topology.topology_sha256 != topology.topology_sha256
            or kernel.boundaries.bundle_sha256 != boundaries.bundle_sha256
            or kernel.state_storage_device != "cpu"
        ):
            raise Lfm2GlobalResidualClosureAuthenticationError(
                "tensor-only kernel does not bind the exact CPU-streamed problem."
            )
        self.kernel = kernel
        self.linearization_receipt = receipt
        self.device = _resolve_device(self.policy)
        self.dtype = torch.float64
        self._basis = basis.output_basis.to(
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        self._defects_cpu = tuple(
            boundary.defect.reshape(-1, topology.hidden_size)
            .to(device="cpu", dtype=self.dtype)
            .contiguous()
            for boundary in boundaries.boundaries
        )
        boundary_count = len(self._defects_cpu)
        self._weights_cpu = tuple(
            (
                boundary.row_mass.reshape(-1).to(device="cpu", dtype=self.dtype)
                / (boundary_count * topology.hidden_size)
            ).contiguous()
            for boundary in boundaries.boundaries
        )
        self.streaming_memory_report = estimate_lfm2_global_residual_streaming_memory(
            hidden_size=topology.hidden_size,
            boundary_row_counts=tuple(
                int(value.shape[0]) for value in self._defects_cpu
            ),
            requested_chunk_rows=self.policy.boundary_chunk_rows,
            maximum_workspace_bytes=(self.policy.maximum_boundary_workspace_bytes),
        )
        self.boundary_archive_bytes = int(
            self.streaming_memory_report["boundary_archive_bytes"]
        )
        self.boundary_chunk_rows = int(
            self.streaming_memory_report["effective_chunk_rows"]
        )
        self.estimated_peak_boundary_workspace_bytes = int(
            self.streaming_memory_report["estimated_peak_boundary_workspace_bytes"]
        )
        operator_bound = _finite_nonnegative(
            receipt["operator_squared_norm_upper_bound"],
            "linearization.operator_squared_norm_upper_bound",
        )
        operator_scale = _finite_nonnegative(
            receipt.get("operator_squared_conditioning_scale", operator_bound),
            "linearization.operator_squared_conditioning_scale",
        )
        self.operator_squared_conditioning_scale = operator_scale
        self.block_squared_conditioning_scales = tuple(
            _finite_nonnegative(value, "block_conditioning_scale")
            for value in block_scales
        )
        self.ridge = max(
            self.policy.absolute_ridge_floor,
            operator_scale / (self.policy.condition_cap - 1.0),
        )
        self.regularized_condition_upper_bound = (
            operator_bound + self.ridge
        ) / self.ridge
        self.regularization_scale_ratio = (
            operator_scale + self.ridge
        ) / self.ridge
        block_preconditioners = tuple(
            self.ridge + value
            for value in self.block_squared_conditioning_scales
        )
        self._block_preconditioner_tensor = torch.tensor(
            block_preconditioners,
            device=self.device,
            dtype=self.dtype,
        )
        self.jvp_calls = 0
        self.vjp_calls = 0
        self.jvp_bundle_calls = 0
        self.vjp_bundle_calls = 0
        self.jvp_chunk_calls = 0
        self.vjp_chunk_calls = 0
        self.normal_calls = 0
        self._cheap_tensor_authentication_checks = 0
        self._raw_tensor_authentication_checks = 0
        self._guard_report_sha256s: list[str] = []
        self._guard_phase_counts: dict[str, int] = {}
        self._guard_counter_totals = {
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "protected_model_count": 0,
        }
        self._operator_tensor_inventory = tuple(
            {
                "name": name,
                "sha256": fingerprint_lfm2_global_residual_tensor(value),
                "sealed_tensor_version": int(value._version),
                "shape": [int(size) for size in value.shape],
                "dtype": _dtype_name(value),
                "device": value.device.type,
            }
            for name, value in self._operator_tensors().items()
        )
        self.internal_tensor_state_sha256 = (
            self.recompute_internal_tensor_state_sha256()
        )

    @property
    def output_basis(self) -> Any:
        return self._basis.detach().clone()

    def _operator_tensors(self) -> dict[str, Any]:
        values = {"output_basis": self._basis}
        values.update(
            {
                f"boundary_defect.{index}": value
                for index, value in enumerate(self._defects_cpu)
            }
        )
        values.update(
            {
                f"boundary_weight.{index}": value
                for index, value in enumerate(self._weights_cpu)
            }
        )
        values["block_preconditioners"] = self._block_preconditioner_tensor
        return values

    def recompute_internal_tensor_state_sha256(
        self,
        *,
        force_raw: bool = False,
    ) -> str:
        current_tensors = self._operator_tensors()
        inventory = []
        for sealed in self._operator_tensor_inventory:
            value = current_tensors[sealed["name"]]
            version = int(value._version)
            digest = (
                fingerprint_lfm2_global_residual_tensor(value)
                if force_raw or version != sealed["sealed_tensor_version"]
                else sealed["sha256"]
            )
            inventory.append(
                {
                    **sealed,
                    "current_sha256": digest,
                    "current_tensor_version": version,
                }
            )
        return _sha256(
            {
                "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
                "kind": "closure_operator_internal_tensor_state",
                "inventory": inventory,
            }
        )

    def assert_internal_tensor_authentication(
        self,
        phase: str,
        *,
        force_raw: bool = False,
    ) -> str:
        if force_raw:
            self._raw_tensor_authentication_checks += 1
        else:
            self._cheap_tensor_authentication_checks += 1
        observed = self.recompute_internal_tensor_state_sha256(force_raw=force_raw)
        if observed != self.internal_tensor_state_sha256:
            raise Lfm2GlobalResidualClosureAuthenticationError(
                f"Closure operator tensor state mutated during {phase}."
            )
        return observed

    def _assert_kernel_authentication(
        self,
        phase: str,
        *,
        force_raw: bool = False,
    ) -> str:
        if force_raw:
            self._raw_tensor_authentication_checks += 1
        else:
            self._cheap_tensor_authentication_checks += 1
        return self.kernel.assert_authenticated(phase, force_raw=force_raw)

    @property
    def operator_sha256(self) -> str:
        return _sha256(
            {
                "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
                "kind": "matrix_free_operator_binding",
                "topology_sha256": self.topology.topology_sha256,
                "basis_sha256": self.basis.basis_sha256,
                "boundary_bundle_sha256": self.boundaries.bundle_sha256,
                "linearization_report_sha256": self.linearization_receipt[
                    "report_sha256"
                ],
                "kernel_authentication_sha256": self.kernel.authentication_sha256,
                "transpose_contract_sha256": (self.kernel.transpose_contract_sha256),
                "internal_tensor_state_sha256": (self.internal_tensor_state_sha256),
                "policy_sha256": self.policy.authentication_sha256,
                "backend": self.device.type,
                "compute_dtype": "float64",
                "boundary_archive_storage_device": "cpu",
                "whole_stack_bundle_linearization": (
                    self.kernel.supports_bundle_linearization
                ),
            }
        )

    def verify_authentication(self) -> bool:
        return (
            self.topology.verify_authentication()
            and self.basis.verify_authentication()
            and self.boundaries.verify_authentication()
            and self.policy.verify_authentication()
            and self.kernel.verify_authentication()
            and self.internal_tensor_state_sha256
            == self.recompute_internal_tensor_state_sha256()
            and verify_lfm2_global_residual_closure_report(self.linearization_receipt)
        )

    def zero_coefficients(self) -> BlockSequence:
        return tuple(
            _torch().zeros(
                (self.basis.rank, site.input_width),
                device=self.device,
                dtype=self.dtype,
            )
            for site in self.topology.sites
        )

    def _coerce_coefficients(self, values: Sequence[Any]) -> BlockSequence:
        blocks = tuple(values)
        if len(blocks) != len(self.topology.sites):
            raise Lfm2GlobalResidualClosureError(
                "coefficient blocks must have one value per writer site."
            )
        result = []
        for site, value in zip(self.topology.sites, blocks, strict=True):
            block = _plain_finite_tensor(
                value,
                f"coefficient_blocks.{site.site_id}",
                dimensions=2,
            )
            if tuple(block.shape) != (self.basis.rank, site.input_width):
                raise Lfm2GlobalResidualClosureError(
                    f"coefficient block {site.site_id} has incorrect geometry."
                )
            result.append(block.to(device=self.device, dtype=self.dtype).contiguous())
        return tuple(result)

    def _coerce_boundary_values_cpu(self, values: Sequence[Any]) -> BlockSequence:
        blocks = tuple(values)
        if len(blocks) != len(self.boundaries.boundaries):
            raise Lfm2GlobalResidualClosureError(
                "boundary values must have one tensor per boundary."
            )
        result = []
        for boundary, value in zip(
            self.boundaries.boundaries,
            blocks,
            strict=True,
        ):
            block = _plain_finite_tensor(
                value,
                f"boundary_values.{boundary.boundary_id}",
            )
            if tuple(block.shape) != tuple(boundary.defect.shape):
                raise Lfm2GlobalResidualClosureError(
                    f"boundary value {boundary.boundary_id} has incorrect geometry."
                )
            result.append(
                block.to(device="cpu", dtype=self.dtype)
                .reshape(-1, self.topology.hidden_size)
                .contiguous()
            )
        return tuple(result)

    def _record_guard(self, report: FoldOnlyGuardReport) -> None:
        body = report.as_dict()
        if body.get("clean") is not True:
            raise Lfm2GlobalResidualClosureError(
                "A tensor-only kernel call did not produce a clean fold-only guard."
            )
        self._guard_report_sha256s.append(_sha256(body))
        self._guard_phase_counts[report.phase] = (
            self._guard_phase_counts.get(report.phase, 0) + 1
        )
        for name in self._guard_counter_totals:
            self._guard_counter_totals[name] += int(getattr(report, name))

    def fold_only_guard_audit(self) -> dict[str, Any]:
        counters = dict(self._guard_counter_totals)
        forbidden_names = (
            "optimizer_constructions",
            "optimizer_steps",
            "backward_calls",
            "autograd_grad_calls",
            "pruning_operations",
            "quantization_operations",
        )
        return {
            "kernel_call_count": len(self._guard_report_sha256s),
            "guard_report_bundle_sha256": _sha256(self._guard_report_sha256s),
            "guard_phase_counts": dict(sorted(self._guard_phase_counts.items())),
            "observed_counters": counters,
            "all_guard_reports_clean": all(
                counters[name] == 0 for name in forbidden_names
            ),
            "hardcoded_zero_counter_claims_used": False,
        }

    def _run_guarded_kernel_operation(
        self,
        callback: Callable[[], Any],
        *,
        phase: str,
    ) -> Any:
        self.assert_internal_tensor_authentication(f"{phase} preflight")
        result, guard = run_fold_only_callback(
            callback,
            phase=phase,
            protected_models=(),
            freeze_parameters=False,
        )
        self._record_guard(guard)
        self.assert_internal_tensor_authentication(f"{phase} completion")
        return result

    def _jvp_chunk(
        self,
        coefficients: BlockSequence,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> Any:
        result = Lfm2GlobalResidualLinearizationKernel.run_jvp_chunk(
            self.kernel,
            coefficients,
            self._basis,
            boundary_index=boundary_index,
            row_start=row_start,
            row_stop=row_stop,
        )
        self._cheap_tensor_authentication_checks += 2
        value = _plain_finite_tensor(result, "guarded_jvp_chunk", dimensions=2)
        self.jvp_calls += 1
        self.jvp_chunk_calls += 1
        return value.to(device=self.device, dtype=self.dtype).contiguous()

    def _vjp_chunk(
        self,
        cotangent: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> BlockSequence:
        result = Lfm2GlobalResidualLinearizationKernel.run_vjp_chunk(
            self.kernel,
            cotangent,
            self._basis,
            boundary_index=boundary_index,
            row_start=row_start,
            row_stop=row_stop,
        )
        self._cheap_tensor_authentication_checks += 2
        self.vjp_calls += 1
        self.vjp_chunk_calls += 1
        return self._coerce_coefficients(result)

    def _jvp_bundle(
        self,
        coefficients: BlockSequence,
    ) -> BlockSequence:
        result = Lfm2GlobalResidualLinearizationKernel.run_jvp_bundle(
            self.kernel,
            coefficients,
            self._basis,
            maximum_workspace_bytes=(self.policy.maximum_boundary_workspace_bytes),
        )
        self._cheap_tensor_authentication_checks += 2
        self.assert_internal_tensor_authentication(
            "operator bundle JVP completion",
        )
        self.jvp_calls += 1
        self.jvp_bundle_calls += 1
        return tuple(result)

    def _vjp_bundle(
        self,
        cotangents_cpu: BlockSequence,
        *,
        row_scales_cpu: BlockSequence | None = None,
    ) -> BlockSequence:
        result = Lfm2GlobalResidualLinearizationKernel.run_vjp_bundle(
            self.kernel,
            cotangents_cpu,
            self._basis,
            boundary_row_scales_cpu=row_scales_cpu,
            maximum_workspace_bytes=(self.policy.maximum_boundary_workspace_bytes),
        )
        self._cheap_tensor_authentication_checks += 2
        self.assert_internal_tensor_authentication(
            "operator bundle VJP completion",
        )
        self.vjp_calls += 1
        self.vjp_bundle_calls += 1
        return self._coerce_coefficients(result)

    def _chunk_ranges(self, boundary_index: int) -> tuple[tuple[int, int], ...]:
        rows = int(self._defects_cpu[boundary_index].shape[0])
        return tuple(
            (start, min(rows, start + self.boundary_chunk_rows))
            for start in range(0, rows, self.boundary_chunk_rows)
        )

    def jvp(self, coefficient_blocks: Sequence[Any]) -> BlockSequence:
        """Materialize full effects on CPU, preferring one stack traversal."""

        coefficients = self._coerce_coefficients(coefficient_blocks)

        def operation() -> BlockSequence:
            if self.kernel.supports_bundle_linearization:
                bundled = self._jvp_bundle(coefficients)
                return tuple(
                    value.reshape(boundary.defect.shape)
                    for boundary, value in zip(
                        self.boundaries.boundaries,
                        bundled,
                        strict=True,
                    )
                )
            archived = []
            for boundary_index, boundary in enumerate(self.boundaries.boundaries):
                output = _torch().empty_like(self._defects_cpu[boundary_index])
                for row_start, row_stop in self._chunk_ranges(boundary_index):
                    output[row_start:row_stop].copy_(
                        self._jvp_chunk(
                            coefficients,
                            boundary_index=boundary_index,
                            row_start=row_start,
                            row_stop=row_stop,
                        ).to(device="cpu")
                    )
                archived.append(output.reshape(boundary.defect.shape))
            return tuple(archived)

        return self._run_guarded_kernel_operation(
            operation,
            phase="global-residual-closure-full-jvp",
        )

    def vjp(self, boundary_cotangents: Sequence[Any]) -> BlockSequence:
        cotangents = self._coerce_boundary_values_cpu(boundary_cotangents)

        def operation() -> BlockSequence:
            if self.kernel.supports_bundle_linearization:
                return self._vjp_bundle(cotangents)
            accumulated = list(self.zero_coefficients())
            for boundary_index, boundary_value in enumerate(cotangents):
                for row_start, row_stop in self._chunk_ranges(boundary_index):
                    contribution = self._vjp_chunk(
                        boundary_value[row_start:row_stop].to(device=self.device),
                        boundary_index=boundary_index,
                        row_start=row_start,
                        row_stop=row_stop,
                    )
                    accumulated = [
                        value + delta
                        for value, delta in zip(
                            accumulated,
                            contribution,
                            strict=True,
                        )
                    ]
            return tuple(accumulated)

        return self._run_guarded_kernel_operation(
            operation,
            phase="global-residual-closure-full-vjp",
        )

    def right_hand_side(self) -> BlockSequence:
        def operation() -> BlockSequence:
            if self.kernel.supports_bundle_linearization:
                # Row weights remain their own small CPU vectors.  The VJP
                # stages and applies them per batch, so no full weighted-defect
                # archive is ever materialized.
                return self._vjp_bundle(
                    self._defects_cpu,
                    row_scales_cpu=self._weights_cpu,
                )
            accumulated = list(self.zero_coefficients())
            for boundary_index in range(len(self._defects_cpu)):
                for row_start, row_stop in self._chunk_ranges(boundary_index):
                    cotangent = (
                        self._defects_cpu[boundary_index][row_start:row_stop]
                        .to(device=self.device)
                        .clone()
                    )
                    weight = self._weights_cpu[boundary_index][row_start:row_stop].to(
                        device=self.device
                    )
                    cotangent.mul_(weight.unsqueeze(-1))
                    contribution = self._vjp_chunk(
                        cotangent,
                        boundary_index=boundary_index,
                        row_start=row_start,
                        row_stop=row_stop,
                    )
                    accumulated = [
                        value + delta
                        for value, delta in zip(
                            accumulated,
                            contribution,
                            strict=True,
                        )
                    ]
            return tuple(accumulated)

        return self._run_guarded_kernel_operation(
            operation,
            phase="global-residual-closure-right-hand-side",
        )

    def normal(self, coefficient_blocks: Sequence[Any]) -> BlockSequence:
        coefficients = self._coerce_coefficients(coefficient_blocks)

        def operation() -> BlockSequence:
            if self.kernel.supports_bundle_linearization:
                # Exactly one whole-stack forward traversal and one exact
                # reverse traversal.  Effects live in CPU RAM; row weights are
                # supplied separately and applied only to each staged batch.
                effects_cpu = self._jvp_bundle(coefficients)
                accumulated = list(
                    self._vjp_bundle(
                        effects_cpu,
                        row_scales_cpu=self._weights_cpu,
                    )
                )
            else:
                accumulated = list(self.zero_coefficients())
                for boundary_index in range(len(self._defects_cpu)):
                    for row_start, row_stop in self._chunk_ranges(boundary_index):
                        effect = self._jvp_chunk(
                            coefficients,
                            boundary_index=boundary_index,
                            row_start=row_start,
                            row_stop=row_stop,
                        )
                        weight = self._weights_cpu[boundary_index][
                            row_start:row_stop
                        ].to(device=self.device)
                        effect.mul_(weight.unsqueeze(-1))
                        contribution = self._vjp_chunk(
                            effect,
                            boundary_index=boundary_index,
                            row_start=row_start,
                            row_stop=row_stop,
                        )
                        accumulated = [
                            value + delta
                            for value, delta in zip(
                                accumulated,
                                contribution,
                                strict=True,
                            )
                        ]
            self.normal_calls += 1
            return tuple(
                gradient + self.ridge * coefficient
                for gradient, coefficient in zip(
                    accumulated,
                    coefficients,
                    strict=True,
                )
            )

        return self._run_guarded_kernel_operation(
            operation,
            phase=f"global-residual-closure-normal-{self.normal_calls}",
        )

    def precondition(self, values: Sequence[Any]) -> BlockSequence:
        blocks = self._coerce_coefficients(values)
        return tuple(
            block / self._block_preconditioner_tensor[index]
            for index, block in enumerate(
                blocks,
            )
        )

    def weighted_objective_for_coefficients(
        self,
        coefficient_blocks: Sequence[Any] | None,
    ) -> float:
        coefficients = (
            None
            if coefficient_blocks is None
            else self._coerce_coefficients(coefficient_blocks)
        )

        def operation() -> float:
            total = 0.0
            bundled_effects = (
                self._jvp_bundle(coefficients)
                if coefficients is not None
                and self.kernel.supports_bundle_linearization
                else None
            )
            for boundary_index in range(len(self._defects_cpu)):
                for row_start, row_stop in self._chunk_ranges(boundary_index):
                    defect = self._defects_cpu[boundary_index][row_start:row_stop]
                    if coefficients is None:
                        residual = defect
                    elif bundled_effects is not None:
                        residual = (
                            bundled_effects[boundary_index][row_start:row_stop] - defect
                        )
                    else:
                        effect = self._jvp_chunk(
                            coefficients,
                            boundary_index=boundary_index,
                            row_start=row_start,
                            row_stop=row_stop,
                        ).to(device="cpu")
                        residual = effect - defect
                    weight = self._weights_cpu[boundary_index][row_start:row_stop]
                    total += float(
                        (residual.square() * weight.unsqueeze(-1)).sum().item()
                    )
            return total

        if coefficients is None:
            return operation()
        return self._run_guarded_kernel_operation(
            operation,
            phase="global-residual-closure-weighted-objective",
        )


def _blocks_dot(left: Sequence[Any], right: Sequence[Any]) -> Any:
    torch = _torch()
    terms = [
        (left_value.reshape(-1) * right_value.reshape(-1)).sum()
        for left_value, right_value in zip(left, right, strict=True)
    ]
    if not terms:
        raise Lfm2GlobalResidualClosureError("block vector cannot be empty.")
    return torch.stack(terms).sum()


def _blocks_norm(values: Sequence[Any]) -> float:
    return math.sqrt(max(0.0, float(_blocks_dot(values, values).item())))


def _blocks_add_scaled(
    left: Sequence[Any],
    right: Sequence[Any],
    scale: Any,
) -> BlockSequence:
    return tuple(
        left_value + scale * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


@dataclass(frozen=True, slots=True)
class _PcgResult:
    coefficients: BlockSequence
    residual_norms: tuple[float, ...]
    relative_residual_norms: tuple[float, ...]
    breakdown_iterations: tuple[int, ...]


def _fixed_pcg(
    operator: Lfm2GlobalResidualClosureOperator,
    right_hand_side: Sequence[Any],
) -> _PcgResult:
    torch = _torch()
    solution = operator.zero_coefficients()
    residual = operator._coerce_coefficients(right_hand_side)
    preconditioned = operator.precondition(residual)
    direction = tuple(value.clone() for value in preconditioned)
    residual_preconditioned = _blocks_dot(residual, preconditioned)
    initial_norm = _blocks_norm(residual)
    residual_norms = [initial_norm]
    relative_norms = [0.0 if initial_norm == 0.0 else 1.0]
    breakdowns = []
    active = initial_norm != 0.0
    tiny = torch.finfo(operator.dtype).tiny
    for iteration in range(operator.policy.pcg_iterations):
        if active:
            normal_direction = operator.normal(direction)
            curvature = _blocks_dot(direction, normal_direction)
            curvature_value = float(curvature.item())
            if not math.isfinite(curvature_value) or curvature_value <= tiny:
                active = False
                breakdowns.append(iteration)
            else:
                alpha = residual_preconditioned / curvature
                solution = _blocks_add_scaled(solution, direction, alpha)
                residual = _blocks_add_scaled(
                    residual,
                    normal_direction,
                    -alpha,
                )
                next_preconditioned = operator.precondition(residual)
                next_residual_preconditioned = _blocks_dot(
                    residual,
                    next_preconditioned,
                )
                previous_value = float(residual_preconditioned.item())
                beta = (
                    torch.zeros((), device=operator.device, dtype=operator.dtype)
                    if (
                        not math.isfinite(previous_value) or abs(previous_value) <= tiny
                    )
                    else next_residual_preconditioned / residual_preconditioned
                )
                direction = _blocks_add_scaled(
                    next_preconditioned,
                    direction,
                    beta,
                )
                residual_preconditioned = next_residual_preconditioned
        norm = _blocks_norm(residual)
        residual_norms.append(norm)
        relative_norms.append(0.0 if initial_norm == 0.0 else norm / initial_norm)
    if any(not bool(torch.isfinite(value).all().item()) for value in solution):
        raise Lfm2GlobalResidualClosureError(
            "global residual-closure PCG produced NaN or infinity."
        )
    return _PcgResult(
        coefficients=tuple(solution),
        residual_norms=tuple(residual_norms),
        relative_residual_norms=tuple(relative_norms),
        breakdown_iterations=tuple(breakdowns),
    )


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualClosureResult:
    """Published coefficient blocks and standalone dense writer corrections."""

    coefficient_blocks: BlockSequence
    dense_writer_corrections: BlockSequence
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        coefficients = tuple(
            _plain_finite_tensor(value, "result.coefficient_block", dimensions=2)
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
            for value in self.coefficient_blocks
        )
        dense = tuple(
            _plain_finite_tensor(value, "result.dense_writer", dimensions=2)
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
            for value in self.dense_writer_corrections
        )
        object.__setattr__(self, "coefficient_blocks", coefficients)
        object.__setattr__(self, "dense_writer_corrections", dense)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_global_residual_closure_report(self.report)
            and self.report.get("coefficient_block_sha256")
            == [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in self.coefficient_blocks
            ]
            and self.report.get("dense_writer_correction_sha256")
            == [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in self.dense_writer_corrections
            ]
        )


def solve_lfm2_global_residual_closure(
    operator: Lfm2GlobalResidualClosureOperator,
) -> Lfm2GlobalResidualClosureResult:
    """Solve and densely materialize one authenticated global closure fold."""

    if not isinstance(operator, Lfm2GlobalResidualClosureOperator):
        raise TypeError("operator must be Lfm2GlobalResidualClosureOperator.")
    operator._assert_kernel_authentication("solve preflight", force_raw=True)
    operator.assert_internal_tensor_authentication(
        "solve preflight",
        force_raw=True,
    )
    if not operator.verify_authentication():
        raise Lfm2GlobalResidualClosureAuthenticationError(
            "global closure operator failed authentication."
        )
    transpose_contract = establish_lfm2_global_residual_transpose_contract(operator)
    right = operator.right_hand_side()
    pcg = _fixed_pcg(operator, right)
    basis = operator.output_basis
    coefficient_blocks = tuple(
        value.detach().to(device="cpu", dtype=_torch().float64).contiguous()
        for value in pcg.coefficients
    )
    dense_corrections = tuple(
        (basis @ value).detach().to(device="cpu", dtype=_torch().float64).contiguous()
        for value in pcg.coefficients
    )
    before = operator.weighted_objective_for_coefficients(None)
    after = operator.weighted_objective_for_coefficients(pcg.coefficients)
    operator._assert_kernel_authentication("result publication", force_raw=True)
    operator.assert_internal_tensor_authentication(
        "result publication",
        force_raw=True,
    )
    guard_audit = operator.fold_only_guard_audit()
    observed_guard = guard_audit["observed_counters"]
    operator_bound = float(
        operator.linearization_receipt["operator_squared_norm_upper_bound"]
    )
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "global_causal_residual_closure_result",
        "implementation": _IMPLEMENTATION,
        "operator_sha256": operator.operator_sha256,
        "topology_sha256": operator.topology.topology_sha256,
        "basis_sha256": operator.basis.basis_sha256,
        "boundary_bundle_sha256": operator.boundaries.bundle_sha256,
        "linearization_report_sha256": operator.linearization_receipt["report_sha256"],
        "kernel_authentication_sha256": operator.kernel.authentication_sha256,
        "transpose_contract_sha256": transpose_contract["report_sha256"],
        "manual_transpose_construction_sha256": (
            operator.kernel.transpose_contract_sha256
        ),
        "operator_internal_tensor_state_sha256": (
            operator.internal_tensor_state_sha256
        ),
        "policy_sha256": operator.policy.authentication_sha256,
        "site_ids": [site.site_id for site in operator.topology.sites],
        "basis_rank": operator.basis.rank,
        "topology_rank_formula": "2*head_dimension",
        "coefficient_coordinate_count": (
            operator.topology.coefficient_coordinate_count
        ),
        "dense_writer_coordinate_count": (
            operator.topology.dense_writer_coordinate_count
        ),
        "parameterization": "delta_writer=output_basis@coefficient_block",
        "coefficient_block_sha256": [
            fingerprint_lfm2_global_residual_tensor(value)
            for value in coefficient_blocks
        ],
        "dense_writer_correction_sha256": [
            fingerprint_lfm2_global_residual_tensor(value)
            for value in dense_corrections
        ],
        "standalone_dense_writer_corrections_materialized": True,
        "shared_or_sparse_runtime_sidecar_required": False,
        "streaming": {
            "memory_estimate_report_sha256": (
                operator.streaming_memory_report["report_sha256"]
            ),
            "boundary_archive_storage_device": "cpu",
            "boundary_archive_bytes": operator.boundary_archive_bytes,
            "compute_device": operator.device.type,
            "boundary_chunk_rows": operator.boundary_chunk_rows,
            "estimated_peak_boundary_workspace_bytes": (
                operator.estimated_peak_boundary_workspace_bytes
            ),
            "maximum_boundary_workspace_bytes": (
                operator.policy.maximum_boundary_workspace_bytes
            ),
            "workspace_budget_passed": (
                operator.estimated_peak_boundary_workspace_bytes
                <= operator.policy.maximum_boundary_workspace_bytes
            ),
            "full_boundary_compute_staging": False,
            "defect_effect_weighted_triple_copy_on_gpu": False,
            "whole_stack_bundle_linearization": (
                operator.kernel.supports_bundle_linearization
            ),
            "whole_stack_bundle_output_storage_device": "cpu",
            "weighted_effect_archive_materialized": False,
            "whole_stack_bundle_effect_archive_bytes": (
                operator.streaming_memory_report[
                    "whole_stack_bundle_effect_archive_bytes"
                ]
            ),
        },
        "solve": {
            "method": "fixed-schedule-block-preconditioned-conjugate-gradient",
            "backend": operator.device.type,
            "compute_dtype": "float64",
            "fixed_iteration_count": operator.policy.pcg_iterations,
            "early_stopping": False,
            "normal_design_materialized": False,
            "gram_materialized": False,
            "autograd_used": False,
            "residual_norms": list(pcg.residual_norms),
            "relative_residual_norms": list(pcg.relative_residual_norms),
            "breakdown_iterations": list(pcg.breakdown_iterations),
            "jvp_calls": operator.jvp_calls,
            "vjp_calls": operator.vjp_calls,
            "jvp_bundle_calls": operator.jvp_bundle_calls,
            "vjp_bundle_calls": operator.vjp_bundle_calls,
            "jvp_chunk_calls": operator.jvp_chunk_calls,
            "vjp_chunk_calls": operator.vjp_chunk_calls,
            "normal_operator_calls": operator.normal_calls,
        },
        "conditioning": {
            "condition_cap": operator.policy.condition_cap,
            "operator_squared_norm_upper_bound": operator_bound,
            "operator_squared_conditioning_scale": (
                operator.operator_squared_conditioning_scale
            ),
            "block_squared_conditioning_scales": list(
                operator.block_squared_conditioning_scales
            ),
            "conditioning_scale_method": operator.linearization_receipt.get(
                "conditioning_scale_method",
                "certified_norm_bound_fallback",
            ),
            "conditioning_scale_is_certified_norm_bound": (
                operator.linearization_receipt.get(
                    "conditioning_scale_is_certified_norm_bound",
                    True,
                )
            ),
            "effective_ridge": operator.ridge,
            "ridge_formula": (
                "max(floor,operator_squared_conditioning_scale/"
                "(condition_cap-1))"
            ),
            "regularization_scale_ratio": operator.regularization_scale_ratio,
            "regularization_scale_ratio_target_passed": (
                operator.regularization_scale_ratio
                <= operator.policy.condition_cap * (1.0 + 1e-12)
            ),
            "regularized_condition_upper_bound": (
                operator.regularized_condition_upper_bound
            ),
            "condition_cap_passed": (
                operator.regularized_condition_upper_bound
                <= operator.policy.condition_cap * (1.0 + 1e-12)
            ),
            "certified_bound_condition_cap_is_diagnostic_only": True,
            "conditioning_scale_has_quality_gate_authority": False,
            "block_preconditioner": (
                "authenticated-block-conditioning-scale-plus-ridge"
            ),
        },
        "weighted_fit_objective_before": before,
        "weighted_fit_objective_after": after,
        "weighted_fit_objective_improved_diagnostic": after <= before,
        "quality_gate": False,
        "numeric_diagnostics_used_for_semantic_decision": False,
        "numeric_diagnostics_used_for_auto_veto": False,
        "fit_only": True,
        "authentication_checks": {
            "cheap_tensor_version_checks": (
                operator._cheap_tensor_authentication_checks
            ),
            "force_raw_tensor_hash_checks": (
                operator._raw_tensor_authentication_checks
            ),
            "force_raw_schedule": ["solve_preflight", "result_publication"],
            "force_raw_subjects_per_scheduled_point": [
                "kernel_tensor_archive",
                "operator_tensor_archive",
            ],
            "force_raw_hash_per_bundle_call": False,
            "bundle_input_clone_hash_and_version_checks": True,
        },
        "fold_only_guard": guard_audit,
        "execution": {
            "optimizer_constructions": observed_guard["optimizer_constructions"],
            "optimizer_steps": observed_guard["optimizer_steps"],
            "backward_calls": observed_guard["backward_calls"],
            "autograd_grad_calls": observed_guard["autograd_grad_calls"],
            "pruning_operations": observed_guard["pruning_operations"],
            "quantization_operations": observed_guard["quantization_operations"],
            "guard_reports_clean": guard_audit["all_guard_reports_clean"],
            "kernel_tensor_only_no_model_handles": True,
            "fixed_iterative_linear_solve": True,
        },
    }
    return Lfm2GlobalResidualClosureResult(
        coefficient_blocks=coefficient_blocks,
        dense_writer_corrections=dense_corrections,
        report=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2GlobalResidualCorrectedWriterState:
    """Ordinary dense target writer weights after transactional publication."""

    site_ids: tuple[str, ...]
    writer_weights: BlockSequence
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        weights = tuple(
            _plain_finite_tensor(value, "corrected_writer", dimensions=2)
            .to(device="cpu")
            .contiguous()
            .clone()
            for value in self.writer_weights
        )
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        object.__setattr__(self, "writer_weights", weights)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_global_residual_closure_report(self.report)
            and list(self.site_ids) == self.report.get("site_ids")
            and self.report.get("corrected_writer_sha256")
            == [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in self.writer_weights
            ]
        )


def materialize_lfm2_global_residual_corrected_writers(
    topology: Lfm2GlobalResidualWriterTopology,
    result: Lfm2GlobalResidualClosureResult,
    *,
    seed_writer_weights: Mapping[str, Any],
    seed_writer_weights_sha256: Mapping[str, str],
) -> Lfm2GlobalResidualCorrectedWriterState:
    """Add solved corrections to authenticated seeds without mutating inputs."""

    torch = _torch()
    if not topology.verify_authentication() or not result.verify_authentication():
        raise Lfm2GlobalResidualClosureAuthenticationError(
            "corrected-writer inputs failed authentication."
        )
    expected_ids = {site.site_id for site in topology.sites}
    if (
        set(seed_writer_weights) != expected_ids
        or set(seed_writer_weights_sha256) != expected_ids
    ):
        raise Lfm2GlobalResidualClosureAuthenticationError(
            "seed writer mappings must exactly match topology site ids."
        )
    if result.report.get("topology_sha256") != topology.topology_sha256:
        raise Lfm2GlobalResidualClosureAuthenticationError(
            "result names a different writer topology."
        )
    corrected = []
    seed_hashes = []
    for site, correction in zip(
        topology.sites,
        result.dense_writer_corrections,
        strict=True,
    ):
        seed = _plain_finite_tensor(
            seed_writer_weights[site.site_id],
            f"seed_writer_weights.{site.site_id}",
            dimensions=2,
        )
        digest = _authenticate_tensor(
            seed,
            seed_writer_weights_sha256[site.site_id],
            f"seed_writer_weights.{site.site_id}",
        )
        if tuple(seed.shape) != (topology.hidden_size, site.input_width):
            raise Lfm2GlobalResidualClosureError(
                f"Seed writer {site.site_id} has incorrect geometry."
            )
        updated = (
            (seed.detach().to(device="cpu", dtype=torch.float64) + correction)
            .to(dtype=seed.dtype)
            .contiguous()
        )
        corrected.append(updated)
        seed_hashes.append(digest)
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "standalone_dense_corrected_writer_state",
        "topology_sha256": topology.topology_sha256,
        "closure_result_sha256": result.report["report_sha256"],
        "site_ids": [site.site_id for site in topology.sites],
        "seed_writer_sha256": seed_hashes,
        "corrected_writer_sha256": [
            fingerprint_lfm2_global_residual_tensor(value) for value in corrected
        ],
        "all_tensors_dense": True,
        "target_dtype_preserved": True,
        "seed_tensors_mutated": False,
        "runtime_sidecar_required": False,
        "execution": {**_ZERO_EXECUTION, "autograd_graphs": 0},
    }
    return Lfm2GlobalResidualCorrectedWriterState(
        site_ids=tuple(site.site_id for site in topology.sites),
        writer_weights=tuple(corrected),
        report=_seal_report(body),
    )


def _deterministic_probe(shape: Sequence[int], offset: int, device: Any) -> Any:
    torch = _torch()
    count = math.prod(int(value) for value in shape)
    index = torch.arange(count, device=device, dtype=torch.int64)
    values = ((index * 17 + offset * 13) % 37).to(torch.float64) - 18.0
    norm = torch.linalg.vector_norm(values)
    if float(norm.item()) == 0.0:
        values[0] = 1.0
        norm = torch.linalg.vector_norm(values)
    return (values / norm).reshape(tuple(int(value) for value in shape))


def audit_lfm2_global_residual_closure_adjoint(
    operator: Lfm2GlobalResidualClosureOperator,
    *,
    probe_count: int = 2,
    relative_tolerance: float = 1e-9,
) -> Mapping[str, Any]:
    """Stream ``<Jc,y> == <c,J^T y>`` probes through guarded chunks."""

    probes = _positive_int(probe_count, "probe_count", maximum=8)
    tolerance = _finite_positive(relative_tolerance, "relative_tolerance")

    def operation() -> list[dict[str, Any]]:
        reports = []
        for probe_index in range(probes):
            coefficients = tuple(
                _deterministic_probe(
                    (operator.basis.rank, site.input_width),
                    101 * probe_index + site_index,
                    operator.device,
                )
                for site_index, site in enumerate(operator.topology.sites)
            )
            if operator.kernel.supports_bundle_linearization:
                # Audit the actual fast path used by PCG.  Effects and probe
                # cotangents trade places boundary by boundary in CPU RAM, so
                # the audit never holds two complete GPU boundary banks.
                effects: list[Any | None] = list(operator._jvp_bundle(coefficients))
                cotangents_cpu = []
                left = 0.0
                for boundary_index, effect_value in enumerate(effects):
                    if effect_value is None:  # pragma: no cover - local invariant
                        raise AssertionError("missing bundle JVP boundary")
                    cotangent_cpu = _torch().empty_like(effect_value)
                    for chunk_index, (row_start, row_stop) in enumerate(
                        operator._chunk_ranges(boundary_index)
                    ):
                        cotangent_cpu[row_start:row_stop].copy_(
                            _deterministic_probe(
                                (
                                    row_stop - row_start,
                                    operator.topology.hidden_size,
                                ),
                                1009 * probe_index + 211 * boundary_index + chunk_index,
                                _torch().device("cpu"),
                            )
                        )
                        left += float(
                            (
                                effect_value[row_start:row_stop]
                                * cotangent_cpu[row_start:row_stop]
                            )
                            .sum()
                            .item()
                        )
                    cotangents_cpu.append(cotangent_cpu)
                    effects[boundary_index] = None
                transpose = list(operator._vjp_bundle(tuple(cotangents_cpu)))
            else:
                transpose = list(operator.zero_coefficients())
                left = 0.0
                for boundary_index in range(len(operator.boundaries.boundaries)):
                    for chunk_index, (row_start, row_stop) in enumerate(
                        operator._chunk_ranges(boundary_index)
                    ):
                        cotangent = _deterministic_probe(
                            (row_stop - row_start, operator.topology.hidden_size),
                            1009 * probe_index + 211 * boundary_index + chunk_index,
                            operator.device,
                        )
                        effect = operator._jvp_chunk(
                            coefficients,
                            boundary_index=boundary_index,
                            row_start=row_start,
                            row_stop=row_stop,
                        )
                        left += float((effect * cotangent).sum().item())
                        contribution = operator._vjp_chunk(
                            cotangent,
                            boundary_index=boundary_index,
                            row_start=row_start,
                            row_stop=row_stop,
                        )
                        transpose = [
                            value + delta
                            for value, delta in zip(
                                transpose,
                                contribution,
                                strict=True,
                            )
                        ]
            right = float(_blocks_dot(coefficients, transpose).item())
            absolute = abs(left - right)
            relative = absolute / max(abs(left), abs(right), 1e-30)
            reports.append(
                {
                    "probe_index": probe_index,
                    "jvp_inner_product": left,
                    "vjp_inner_product": right,
                    "absolute_error": absolute,
                    "relative_error": relative,
                    "within_declared_tolerance": relative <= tolerance,
                }
            )
        return reports

    reports = operator._run_guarded_kernel_operation(
        operation,
        phase="global-residual-closure-adjoint-contract-audit",
    )
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "analytic_jvp_vjp_adjoint_audit",
        "operator_sha256": operator.operator_sha256,
        "kernel_authentication_sha256": operator.kernel.authentication_sha256,
        "transpose_contract_sha256": (operator.kernel.transpose_contract_sha256),
        "probe_generation": "deterministic-integer-lattice-normalized-v1",
        "probe_count": probes,
        "relative_tolerance": tolerance,
        "probes": reports,
        "maximum_relative_error": max(value["relative_error"] for value in reports),
        "all_probes_within_declared_tolerance": all(
            value["within_declared_tolerance"] for value in reports
        ),
        "diagnostic_only": True,
        "used_to_auto_veto_publication": False,
        "fold_only_guard": operator.fold_only_guard_audit(),
    }
    return _seal_report(body)


def establish_lfm2_global_residual_transpose_contract(
    operator: Lfm2GlobalResidualClosureOperator,
    *,
    probe_count: int = 2,
    relative_tolerance: float = 1e-9,
) -> Mapping[str, Any]:
    """Fail closed unless the authenticated manual VJP acts as ``J.T``.

    This is a structural mathematical precondition for naming the subsequent
    operator ``J.T W J + ridge I``.  It is not a behavior, intelligence,
    objective-improvement, or candidate-selection gate.
    """

    operator._assert_kernel_authentication("transpose contract preflight")
    audit = audit_lfm2_global_residual_closure_adjoint(
        operator,
        probe_count=probe_count,
        relative_tolerance=relative_tolerance,
    )
    if audit["all_probes_within_declared_tolerance"] is not True:
        raise Lfm2GlobalResidualClosureError(
            "The authenticated VJP failed the structural transpose proof; "
            "PCG cannot be represented as a ridge solve."
        )
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "established_global_residual_transpose_contract",
        "operator_sha256": operator.operator_sha256,
        "kernel_authentication_sha256": operator.kernel.authentication_sha256,
        "manual_transpose_contract_sha256": (operator.kernel.transpose_contract_sha256),
        "dynamic_adjoint_audit_sha256": audit["report_sha256"],
        "structural_contract_established": True,
        "required_before_pcg": True,
        "semantic_or_behavior_quality_gate": False,
        "objective_improvement_gate": False,
        "fold_only_guard": operator.fold_only_guard_audit(),
    }
    return _seal_report(body)


def audit_lfm2_global_residual_closure_causality(
    operator: Lfm2GlobalResidualClosureOperator,
    *,
    absolute_tolerance: float = 1e-12,
) -> Mapping[str, Any]:
    """Probe that no writer changes a causally earlier boundary."""

    tolerance = _finite_positive(absolute_tolerance, "absolute_tolerance")

    def operation() -> list[dict[str, Any]]:
        site_reports = []
        for site_index, site in enumerate(operator.topology.sites):
            coefficients = list(operator.zero_coefficients())
            coefficients[site_index] = _deterministic_probe(
                coefficients[site_index].shape,
                4099 + site_index,
                operator.device,
            )
            earlier = []
            for boundary_index, boundary in enumerate(operator.boundaries.boundaries):
                if boundary.causal_order >= site.causal_order:
                    continue
                maximum = 0.0
                for row_start, row_stop in operator._chunk_ranges(boundary_index):
                    effect = operator._jvp_chunk(
                        tuple(coefficients),
                        boundary_index=boundary_index,
                        row_start=row_start,
                        row_stop=row_stop,
                    )
                    maximum = max(
                        maximum,
                        float(effect.abs().max().item()),
                    )
                earlier.append(maximum)
            maximum = max(earlier, default=0.0)
            site_reports.append(
                {
                    "site_id": site.site_id,
                    "causal_order": site.causal_order,
                    "earlier_boundary_count": len(earlier),
                    "maximum_absolute_earlier_effect": maximum,
                    "within_declared_tolerance": maximum <= tolerance,
                }
            )
        return site_reports

    site_reports = operator._run_guarded_kernel_operation(
        operation,
        phase="global-residual-closure-causality-audit",
    )
    body = {
        "schema": LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA,
        "kind": "block_lower_triangular_causality_audit",
        "operator_sha256": operator.operator_sha256,
        "absolute_tolerance": tolerance,
        "sites": site_reports,
        "all_sites_within_declared_tolerance": all(
            value["within_declared_tolerance"] for value in site_reports
        ),
        "diagnostic_only": True,
        "used_to_auto_veto_publication": False,
        "fold_only_guard": operator.fold_only_guard_audit(),
    }
    return _seal_report(body)


__all__ = [
    "LFM2_GLOBAL_RESIDUAL_CLOSURE_SCHEMA",
    "LFM2_GLOBAL_RESIDUAL_LINEARIZATION_SCHEMA",
    "Lfm2GlobalResidualBasis",
    "Lfm2GlobalResidualBoundary",
    "Lfm2GlobalResidualBoundaryBundle",
    "Lfm2GlobalResidualClosureAuthenticationError",
    "Lfm2GlobalResidualClosureError",
    "Lfm2GlobalResidualClosureOperator",
    "Lfm2GlobalResidualClosurePolicy",
    "Lfm2GlobalResidualClosureResult",
    "Lfm2GlobalResidualCorrectedWriterState",
    "Lfm2GlobalResidualLinearizationKernel",
    "Lfm2GlobalResidualWriterSite",
    "Lfm2GlobalResidualWriterTopology",
    "adjoint_lfm2_global_residual_writer_block",
    "apply_lfm2_global_residual_writer_block",
    "audit_lfm2_global_residual_closure_adjoint",
    "audit_lfm2_global_residual_closure_causality",
    "build_lfm2_global_residual_basis",
    "establish_lfm2_global_residual_transpose_contract",
    "estimate_lfm2_global_residual_streaming_memory",
    "fingerprint_lfm2_global_residual_tensor",
    "materialize_lfm2_global_residual_corrected_writers",
    "seal_lfm2_global_residual_boundary",
    "seal_lfm2_global_residual_linearization_receipt",
    "solve_lfm2_global_residual_closure",
    "verify_lfm2_global_residual_closure_report",
]
