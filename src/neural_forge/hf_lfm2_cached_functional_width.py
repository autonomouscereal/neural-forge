"""Authenticated cached-functional evidence for an LFM2 hidden-width frontier.

The heterogeneous physical-layout solver can already trade MLP, attention,
and ShortConv capacity at one fixed residual width.  This module supplies the
missing comparable evidence term for choosing *between* residual widths.

It deliberately does not rank widths by parameter count or by raw residual
energy.  Four causal roles are sampled from the sealed parent cache:

* the residual context entering each operator;
* the operator's residual write;
* the nonlinear SwiGLU residual write; and
* the final residual state after the complete source stack.

Every sampled row is normalized before it contributes to a role Gram, and
every role Gram is normalized to unit trace before the roles are averaged.
Consequently a large-magnitude role cannot drown out a small but functionally
distinct one.  One eigenspectrum then gives a directly comparable best-rank-H
direction-retention loss for every legal target width without model execution,
training, backpropagation, pruning, quantization, sparsity, or parameter
sharing.
"""

from __future__ import annotations

import hashlib
import math
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

from .dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)
from .hf_lfm2_heterogeneous_layout_selection import (
    discover_lfm2_source_layers,
)
from .hf_lfm2_retained_trace_rows import (
    Lfm2RetainedTraceRowAdapter,
    Lfm2RetainedTraceRowsError,
)


LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA = (
    "neural-forge-lfm2-cached-functional-hidden-width-v2"
)
LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA = (
    "neural-forge-lfm2-functional-hidden-width-score-v1"
)
LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY = (
    "sealed-cache-role-balanced-direction-round-robin-batched-gram-v2"
)
LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA = (
    "neural-forge-lfm2-deterministic-role-batched-gram-v1"
)
LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE = 64
LFM2_FUNCTIONAL_WIDTH_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-functional-width-progress-v1"
)
LFM2_FUNCTIONAL_WIDTH_MAXIMUM_PROGRESS_EVENTS = 16

_ROLES = (
    "nonlinear_mlp_write",
    "operator_write",
    "residual_context",
    "terminal_context",
)
_TORCH_NUMERIC_RUNTIME_LOCK = threading.RLock()


class Lfm2CachedFunctionalWidthError(RuntimeError):
    """Cached evidence cannot authenticate a residual-width comparison."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedFunctionalWidthError(f"{name} must be a positive integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedFunctionalWidthError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _fraction(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < -1e-12
        or float(value) > 1.0 + 1e-12
    ):
        raise Lfm2CachedFunctionalWidthError(f"{name} must be a finite fraction.")
    return min(1.0, max(0.0, float(value)))


def _gram_accumulation_contract() -> dict[str, Any]:
    return {
        "schema": LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA,
        "method": "deterministic-role-buffered-row-batch-addmm",
        "calculation_device": "cpu",
        "calculation_dtype": "torch.float64",
        "row_batch_size": LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
        "canonical_row_order": (
            "batch-index-then-location-key-then-selected-row-index-v1"
        ),
        "intraop_threads": 1,
        "bounded_pending_role_buffers": True,
    }


def _emit_progress(
    progress: Callable[[str, Mapping[str, Any]], None] | None,
    stage: str,
    **details: Any,
) -> None:
    if progress is None:
        return
    progress(
        stage,
        MappingProxyType(
            {
                "functional_width_progress_schema": (
                    LFM2_FUNCTIONAL_WIDTH_PROGRESS_SCHEMA
                ),
                "sealed_parent_cache_only": True,
                "parent_model_calls": 0,
                "child_model_calls": 0,
                "trace_regeneration_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                **details,
            }
        ),
    )


def _role_row_progress(
    prefix: str,
    rows_by_role: Mapping[str, Any],
) -> dict[str, int]:
    return {
        f"{prefix}_{role}_rows": int(rows_by_role[role])
        for role in _ROLES
    }


def _gram_diagnostics_progress(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        name: diagnostics[name]
        for name in (
            "gram_accumulation_schema",
            "row_batch_size",
            "row_outer_product_equivalent_count",
            "previous_location_matmul_count",
            "batched_matmul_count",
            "location_matmul_count_reduction",
            "peak_pending_rows",
            "peak_pending_bytes",
            "pending_row_scratch_bound",
            "pending_byte_scratch_bound",
            "maximum_flush_rows",
            "maximum_flush_bytes",
        )
    }
    result.update(
        _role_row_progress(
            "sampled",
            diagnostics["sampled_rows_by_role"],
        )
    )
    result.update(
        {
            f"batched_{role}_matmul_count": int(
                diagnostics["batched_matmul_count_by_role"][role]
            )
            for role in _ROLES
        }
    )
    return result


@contextmanager
def _single_threaded_torch_cpu_math(torch: Any) -> Iterator[None]:
    """Serialize the process-global intra-op thread override and restore it."""

    with _TORCH_NUMERIC_RUNTIME_LOCK:
        original_threads = int(torch.get_num_threads())
        try:
            if original_threads != 1:
                torch.set_num_threads(1)
            yield
        finally:
            if original_threads != 1:
                torch.set_num_threads(original_threads)


class _DeterministicRoleBatchedGramAccumulator:
    """Bounded FP64 role-Gram accumulator with a canonical row order."""

    def __init__(self, source_hidden_size: int, *, row_batch_size: int) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Functional width evidence requires PyTorch.") from exc
        self._torch = torch
        self.source_hidden_size = _positive_int(
            source_hidden_size,
            "source_hidden_size",
        )
        self.row_batch_size = _positive_int(row_batch_size, "row_batch_size")
        self.grams = {
            role: torch.zeros(
                (self.source_hidden_size, self.source_hidden_size),
                dtype=torch.float64,
            )
            for role in _ROLES
        }
        self.rows = {role: 0 for role in _ROLES}
        self.location_matmuls_replaced = {role: 0 for role in _ROLES}
        self.batched_matmuls = {role: 0 for role in _ROLES}
        self._pending = {role: [] for role in _ROLES}
        self._pending_rows = {role: 0 for role in _ROLES}
        self._peak_pending_rows = 0
        self._maximum_flush_rows = 0
        self._finalized = False

    def _record_pending_peak(self) -> None:
        self._peak_pending_rows = max(
            self._peak_pending_rows,
            sum(self._pending_rows.values()),
        )

    def _flush(self, role: str) -> None:
        pending_rows = self._pending_rows[role]
        if pending_rows <= 0:
            return
        chunks = self._pending[role]
        batch = chunks[0] if len(chunks) == 1 else self._torch.cat(chunks, dim=0)
        if tuple(batch.shape) != (pending_rows, self.source_hidden_size):
            raise Lfm2CachedFunctionalWidthError(
                "Functional Gram row buffer changed shape before accumulation."
            )
        self._pending[role] = []
        self._pending_rows[role] = 0
        self.grams[role].addmm_(batch.T, batch)
        self.batched_matmuls[role] += 1
        self._maximum_flush_rows = max(self._maximum_flush_rows, pending_rows)

    def add(self, role: str, normalized_rows: Any) -> None:
        if self._finalized:
            raise Lfm2CachedFunctionalWidthError(
                "Functional Gram accumulator is already finalized."
            )
        if role not in self.grams:
            raise Lfm2CachedFunctionalWidthError(
                f"Functional Gram role {role!r} is not canonical."
            )
        rows = normalized_rows
        if (
            not isinstance(rows, self._torch.Tensor)
            or rows.device.type != "cpu"
            or rows.dtype != self._torch.float64
            or rows.ndim != 2
            or int(rows.shape[1]) != self.source_hidden_size
        ):
            raise Lfm2CachedFunctionalWidthError(
                "Functional Gram input must be a CPU FP64 source-width matrix."
            )
        row_count = int(rows.shape[0])
        if row_count <= 0:
            return
        self.rows[role] += row_count
        self.location_matmuls_replaced[role] += 1
        offset = 0
        while offset < row_count:
            available = self.row_batch_size - self._pending_rows[role]
            take = min(available, row_count - offset)
            self._pending[role].append(rows[offset : offset + take])
            self._pending_rows[role] += take
            offset += take
            self._record_pending_peak()
            if self._pending_rows[role] == self.row_batch_size:
                self._flush(role)

    def diagnostics(self) -> Mapping[str, Any]:
        scalar_bytes = self._torch.empty((), dtype=self._torch.float64).element_size()
        row_outer_product_count = sum(self.rows.values())
        location_matmul_count = sum(self.location_matmuls_replaced.values())
        batched_matmul_count = sum(self.batched_matmuls.values())
        scratch_bound_rows = len(_ROLES) * self.row_batch_size
        diagnostics = MappingProxyType(
            {
                "gram_accumulation_schema": (
                    LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA
                ),
                "row_batch_size": self.row_batch_size,
                "sampled_rows_by_role": dict(self.rows),
                "row_outer_product_equivalent_count": row_outer_product_count,
                "previous_location_matmul_count": location_matmul_count,
                "batched_matmul_count": batched_matmul_count,
                "batched_matmul_count_by_role": dict(self.batched_matmuls),
                "location_matmul_count_reduction": (
                    location_matmul_count - batched_matmul_count
                ),
                "peak_pending_rows": self._peak_pending_rows,
                "peak_pending_bytes": (
                    self._peak_pending_rows * self.source_hidden_size * scalar_bytes
                ),
                "pending_row_scratch_bound": scratch_bound_rows,
                "pending_byte_scratch_bound": (
                    scratch_bound_rows * self.source_hidden_size * scalar_bytes
                ),
                "maximum_flush_rows": self._maximum_flush_rows,
                "maximum_flush_bytes": (
                    self._maximum_flush_rows
                    * self.source_hidden_size
                    * scalar_bytes
                ),
            }
        )
        return diagnostics

    def finalize(self) -> tuple[dict[str, Any], Mapping[str, Any]]:
        if self._finalized:
            raise Lfm2CachedFunctionalWidthError(
                "Functional Gram accumulator is already finalized."
            )
        for role in _ROLES:
            self._flush(role)
        self._finalized = True
        diagnostics = self.diagnostics()
        return self.grams, diagnostics


def _tensor_sha256(value: Any) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Functional width evidence requires PyTorch.") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a tensor.")
    cpu = value.detach().resolve_conj().resolve_neg().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(canonical_sha256(list(cpu.shape)).encode("ascii"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _ordered_parent_phases(record: Mapping[str, Any]) -> tuple[str, ...]:
    capture = record.get("capture")
    diagnostics = (
        capture.get("phase_diagnostics") if isinstance(capture, Mapping) else None
    )
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise Lfm2CachedFunctionalWidthError(
            "Parent cache batch has no authenticated causal phases."
        )

    def key(phase: Any) -> tuple[int, int]:
        if phase in {"prefill", "cached_prefill"}:
            return (0, 0)
        if isinstance(phase, str) and phase.startswith("decode."):
            suffix = phase.removeprefix("decode.")
            if suffix.isdigit():
                return (1, int(suffix))
        raise Lfm2CachedFunctionalWidthError(
            f"Parent cache contains an unknown causal phase {phase!r}."
        )

    phases = tuple(sorted(diagnostics, key=key))
    if len(phases) != len(set(phases)):
        raise Lfm2CachedFunctionalWidthError(
            "Parent cache causal phases are not unique."
        )
    return phases


@dataclass(frozen=True, slots=True)
class Lfm2FunctionalWidthScore:
    """One cache-derived, directly comparable residual-width loss."""

    hidden_size: int
    role_balanced_functional_retention: float
    role_balanced_functional_loss: float
    functional_eigenspectrum_sha256: str
    selection_evidence_sha256: str
    score_sha256: str

    def __post_init__(self) -> None:
        hidden = _positive_int(self.hidden_size, "hidden_size")
        retained = _fraction(
            self.role_balanced_functional_retention,
            "role_balanced_functional_retention",
        )
        loss = _fraction(
            self.role_balanced_functional_loss,
            "role_balanced_functional_loss",
        )
        if not math.isclose(retained + loss, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise Lfm2CachedFunctionalWidthError(
                "Functional width retention and loss are inconsistent."
            )
        _digest(
            self.functional_eigenspectrum_sha256,
            "functional_eigenspectrum_sha256",
        )
        _digest(self.selection_evidence_sha256, "selection_evidence_sha256")
        body = {
            "schema": LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA,
            "hidden_size": hidden,
            "role_balanced_functional_retention": retained,
            "role_balanced_functional_loss": loss,
            "functional_eigenspectrum_sha256": (self.functional_eigenspectrum_sha256),
            "selection_evidence_sha256": self.selection_evidence_sha256,
        }
        if _digest(self.score_sha256, "score_sha256") != canonical_sha256(body):
            raise Lfm2CachedFunctionalWidthError(
                "Functional width score seal is invalid."
            )

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        retained: float,
        eigenspectrum_sha256: str,
        selection_evidence_sha256: str,
    ) -> Lfm2FunctionalWidthScore:
        retained = _fraction(retained, "retained")
        body = {
            "schema": LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA,
            "hidden_size": _positive_int(hidden_size, "hidden_size"),
            "role_balanced_functional_retention": retained,
            "role_balanced_functional_loss": 1.0 - retained,
            "functional_eigenspectrum_sha256": _digest(
                eigenspectrum_sha256,
                "eigenspectrum_sha256",
            ),
            "selection_evidence_sha256": _digest(
                selection_evidence_sha256,
                "selection_evidence_sha256",
            ),
        }
        return cls(
            hidden_size=body["hidden_size"],
            role_balanced_functional_retention=retained,
            role_balanced_functional_loss=1.0 - retained,
            functional_eigenspectrum_sha256=eigenspectrum_sha256,
            selection_evidence_sha256=selection_evidence_sha256,
            score_sha256=canonical_sha256(body),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA,
            "hidden_size": self.hidden_size,
            "role_balanced_functional_retention": (
                self.role_balanced_functional_retention
            ),
            "role_balanced_functional_loss": self.role_balanced_functional_loss,
            "functional_eigenspectrum_sha256": (self.functional_eigenspectrum_sha256),
            "selection_evidence_sha256": self.selection_evidence_sha256,
            "score_sha256": self.score_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CachedFunctionalWidthEvidence:
    """Sealed role-balanced functional spectrum and requested width scores."""

    source_graph_fingerprint: str
    parent_cache_key: str
    selective_session_sha256: str
    source_hidden_size: int
    maximum_rows_per_role: int
    sampled_rows_by_role: tuple[tuple[str, int], ...]
    candidate_scores: tuple[Lfm2FunctionalWidthScore, ...]
    role_gram_sha256: tuple[tuple[str, str], ...]
    functional_eigenspectrum_sha256: str
    selection_manifest_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _digest(self.source_graph_fingerprint, "source_graph_fingerprint")
        _digest(self.parent_cache_key, "parent_cache_key")
        _digest(self.selective_session_sha256, "selective_session_sha256")
        source = _positive_int(self.source_hidden_size, "source_hidden_size")
        _positive_int(self.maximum_rows_per_role, "maximum_rows_per_role")
        rows = tuple(self.sampled_rows_by_role)
        if tuple(name for name, _count in rows) != _ROLES or any(
            _positive_int(count, f"{name} sampled rows") <= 0 for name, count in rows
        ):
            raise Lfm2CachedFunctionalWidthError(
                "Functional evidence must cover every canonical role."
            )
        scores = tuple(self.candidate_scores)
        widths = tuple(value.hidden_size for value in scores)
        if (
            not scores
            or widths != tuple(sorted(set(widths)))
            or widths[-1] > source
            or any(
                value.functional_eigenspectrum_sha256
                != self.functional_eigenspectrum_sha256
                for value in scores
            )
        ):
            raise Lfm2CachedFunctionalWidthError(
                "Functional width scores are not canonical for the source width."
            )
        gram_hashes = tuple(self.role_gram_sha256)
        if tuple(name for name, _sha in gram_hashes) != _ROLES:
            raise Lfm2CachedFunctionalWidthError(
                "Role Gram hashes are incomplete or unordered."
            )
        for name, value in gram_hashes:
            _digest(value, f"{name} Gram SHA-256")
        _digest(
            self.functional_eigenspectrum_sha256,
            "functional_eigenspectrum_sha256",
        )
        _digest(self.selection_manifest_sha256, "selection_manifest_sha256")
        if canonical_sha256(self._body()) != _digest(
            self.evidence_sha256,
            "evidence_sha256",
        ):
            raise Lfm2CachedFunctionalWidthError(
                "Cached functional-width evidence seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "parent_cache_key": self.parent_cache_key,
            "selective_session_sha256": self.selective_session_sha256,
            "source_hidden_size": self.source_hidden_size,
            "sampling": {
                "policy": LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY,
                "maximum_rows_per_role": self.maximum_rows_per_role,
                "sampled_rows_by_role": dict(self.sampled_rows_by_role),
                "selection_manifest_sha256": self.selection_manifest_sha256,
                "row_l2_normalized_before_gram": True,
                "role_grams_normalized_to_unit_trace": True,
                "roles_weighted_equally": True,
                "raw_residual_energy_used_as_width_score": False,
                "gram_accumulation": _gram_accumulation_contract(),
            },
            "role_gram_sha256": dict(self.role_gram_sha256),
            "functional_eigenspectrum_sha256": (self.functional_eigenspectrum_sha256),
            "candidate_scores": [value.as_dict() for value in self.candidate_scores],
            "selection_contract": {
                "sealed_parent_cache_only": True,
                "parent_model_calls": 0,
                "child_model_calls": 0,
                "trace_regeneration_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "sparse_parameters": 0,
                "shared_parameters": 0,
                "parameter_count_is_not_a_quality_score": True,
                "nonlinear_mlp_write_is_a_scored_role": True,
                "whole_stack_terminal_context_is_a_scored_role": True,
            },
        }

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {**self._body(), "evidence_sha256": self.evidence_sha256}
        )

    def score_for(self, hidden_size: int) -> Lfm2FunctionalWidthScore:
        hidden = _positive_int(hidden_size, "hidden_size")
        for score in self.candidate_scores:
            if score.hidden_size == hidden:
                return score
        raise Lfm2CachedFunctionalWidthError(
            f"No authenticated functional score exists for hidden width {hidden}."
        )


@dataclass(frozen=True, slots=True)
class _FunctionalLocation:
    role: str
    batch_index: int
    phase: str
    layer_index: int
    row_count: int
    tensor_names: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.role}|{self.batch_index:06d}|{self.phase}|{self.layer_index:06d}"


def enumerate_lfm2_global_hidden_sizes(
    source_graph: ModelCircuitGraph,
    *,
    exact_target_unique_parameters: int,
    target_vocabulary_size: int,
    minimum_hidden_size: int = 32,
    hidden_size_multiple: int = 8,
) -> tuple[int, ...]:
    """Enumerate every catalog-legal residual width with boundary headroom.

    This mirrors the registered LFM2 dense catalog's dynamic hidden-axis rule;
    there is no H512, H640, H784, or other architecture-specific shortlist.
    A width is only a candidate here.  The exact heterogeneous solver remains
    responsible for proving that its complete full-depth child is attainable.
    """

    if not isinstance(source_graph, ModelCircuitGraph) or source_graph.family != "lfm2":
        raise Lfm2CachedFunctionalWidthError(
            "A global LFM2 width frontier requires an inspected LFM2 graph."
        )
    target = _positive_int(
        exact_target_unique_parameters,
        "exact_target_unique_parameters",
    )
    vocab = _positive_int(target_vocabulary_size, "target_vocabulary_size")
    minimum = _positive_int(minimum_hidden_size, "minimum_hidden_size")
    multiple = _positive_int(hidden_size_multiple, "hidden_size_multiple")
    if minimum % multiple:
        raise Lfm2CachedFunctionalWidthError(
            "minimum_hidden_size must be a hidden_size_multiple."
        )
    metadata = dict(source_graph.metadata)
    source_hidden = _positive_int(
        metadata.get("hidden_size"),
        "source graph hidden_size",
    )
    lower = min(minimum, source_hidden)
    widths = list(range(lower, source_hidden + 1, multiple))
    if not widths or widths[-1] != source_hidden:
        widths.append(source_hidden)
    # The heterogeneous target uses an intentionally untied token interface.
    # Leave at least one parameter beyond embedding/head/final-norm; complete
    # layer feasibility is certified later by the exact solver.
    result = tuple(
        hidden for hidden in sorted(set(widths)) if 2 * vocab * hidden + hidden < target
    )
    if not result:
        raise Lfm2CachedFunctionalWidthError(
            "No legal residual width leaves room for a full-depth exact child."
        )
    return result


def _descriptor_shape(
    reference: Any,
    name: str,
    *,
    source_hidden_size: int,
) -> tuple[int, ...]:
    inventory = reference.record.get("tensors")
    descriptor = inventory.get(name) if isinstance(inventory, Mapping) else None
    shape = descriptor.get("shape") if isinstance(descriptor, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
        or shape[-1] != source_hidden_size
    ):
        raise Lfm2CachedFunctionalWidthError(
            f"Cached tensor {name!r} is not a source-width floating endpoint."
        )
    return tuple(shape)


def _location_row_allocations(
    locations: Sequence[_FunctionalLocation],
    *,
    maximum_rows: int,
) -> dict[str, tuple[int, ...]]:
    """Allocate a bounded deterministic round-robin sample over locations."""

    ordered = sorted(
        locations,
        key=lambda value: (
            hashlib.sha256(value.key.encode("ascii")).hexdigest(),
            value.key,
        ),
    )
    if not ordered:
        raise Lfm2CachedFunctionalWidthError(
            "A functional role has no authenticated cache locations."
        )
    counts = {value.key: 0 for value in ordered}
    remaining = min(maximum_rows, sum(value.row_count for value in ordered))
    active = list(ordered)
    while remaining and active:
        next_active = []
        for location in active:
            if remaining <= 0:
                break
            count = counts[location.key]
            if count < location.row_count:
                counts[location.key] = count + 1
                remaining -= 1
            if counts[location.key] < location.row_count:
                next_active.append(location)
        active = next_active

    result: dict[str, tuple[int, ...]] = {}
    by_key = {value.key: value for value in ordered}
    for key, take in counts.items():
        if take <= 0:
            continue
        row_count = by_key[key].row_count
        seed = int(hashlib.sha256(key.encode("ascii")).hexdigest(), 16)
        start = seed % row_count
        step = (
            1 if row_count == 1 else 1 + (seed // max(1, row_count)) % (row_count - 1)
        )
        while math.gcd(step, row_count) != 1:
            step = 1 if step + 1 >= row_count else step + 1
        result[key] = tuple((start + index * step) % row_count for index in range(take))
    return result


def _selected_role_rows(
    location: _FunctionalLocation,
    tensors: Mapping[str, Any],
    indices: Sequence[int],
    *,
    source_hidden_size: int,
) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Functional width evidence requires PyTorch.") from exc

    def rows(name: str) -> Any:
        value = tensors[name]
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim < 2
            or int(value.shape[-1]) != source_hidden_size
        ):
            raise Lfm2CachedFunctionalWidthError(
                f"Selected cache tensor {name!r} is not a floating residual endpoint."
            )
        result = (
            value.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(
                -1,
                source_hidden_size,
            )
        )
        if not bool(torch.isfinite(result).all().item()):
            raise Lfm2CachedFunctionalWidthError(
                f"Selected cache tensor {name!r} contains non-finite values."
            )
        return result

    if location.role in {"residual_context", "terminal_context"}:
        result = rows(location.tensor_names[0])
    else:
        left = rows(location.tensor_names[0])
        right = rows(location.tensor_names[1])
        if tuple(left.shape) != tuple(right.shape):
            raise Lfm2CachedFunctionalWidthError(
                "Cached residual write endpoints changed shape."
            )
        result = left - right
    if int(result.shape[0]) != location.row_count:
        raise Lfm2CachedFunctionalWidthError(
            "Selected cache tensor row count differs from its sealed descriptor."
        )
    return result[list(indices)]


def build_lfm2_cached_functional_width_evidence(
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    candidate_hidden_sizes: Sequence[int],
    *,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
    retained_parent_rows: Any | None = None,
    maximum_rows_per_role: int = 2048,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Lfm2CachedFunctionalWidthEvidence:
    """Build one bounded cache-only functional spectrum for every target H.

    With ``retained_parent_rows``, B2 reads every functional endpoint from the
    already materialized row index.  The selective session continues to bind
    immutable metadata and its evidence SHA, but performs zero tensor loads.
    """

    if not isinstance(source_graph, ModelCircuitGraph) or source_graph.family != "lfm2":
        raise TypeError("source_graph must be an inspected LFM2 ModelCircuitGraph.")
    if not isinstance(parent_cache, Lfm2CircuitTraceCache):
        raise TypeError("parent_cache must be Lfm2CircuitTraceCache.")
    if selective_session is not None and not isinstance(
        selective_session,
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise TypeError(
            "selective_session must be an authenticated selective session or None."
        )
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None.")
    row_budget = _positive_int(maximum_rows_per_role, "maximum_rows_per_role")
    source_hidden = _positive_int(
        dict(source_graph.metadata).get("hidden_size"),
        "source graph hidden_size",
    )
    widths = tuple(
        sorted(
            {
                _positive_int(value, "candidate_hidden_size")
                for value in candidate_hidden_sizes
            }
        )
    )
    if not widths or widths[-1] > source_hidden:
        raise Lfm2CachedFunctionalWidthError(
            "Candidate hidden widths must be nonempty and no wider than the source."
        )
    discovered = discover_lfm2_source_layers(source_graph)
    session_context = (
        parent_cache.selective_authenticated_session()
        if selective_session is None
        else nullcontext(selective_session)
    )
    with session_context as session:
        session.verify()
        if (
            session.path != parent_cache.path
            or session.cache_key != parent_cache.cache_key
            or session.batch_count != parent_cache.batch_count
            or session.manifest != parent_cache.manifest
        ):
            raise Lfm2CachedFunctionalWidthError(
                "Selective session differs from the parent cache."
            )
        topology = session.manifest.get("layer_topology")
        topology_layers = (
            topology.get("layers") if isinstance(topology, Mapping) else None
        )
        expected_kinds = tuple(
            "short_conv" if value.layer_type == "conv" else "full_attention"
            for value in discovered
        )
        if (
            not isinstance(topology_layers, list)
            or tuple(
                value.get("contract", {}).get("kind")
                if isinstance(value, Mapping)
                else None
                for value in topology_layers
            )
            != expected_kinds
        ):
            raise Lfm2CachedFunctionalWidthError(
                "Parent cache topology differs from the inspected source graph."
            )

        _emit_progress(
            progress,
            "functional_width_selection_started",
            source_hidden_size=source_hidden,
            maximum_rows_per_role=row_budget,
            canonical_role_count=len(_ROLES),
            cache_batch_count=session.batch_count,
            source_layer_count=len(discovered),
            retained_parent_rows=retained_parent_rows is not None,
        )

        retained_adapter = None
        if retained_parent_rows is not None:
            try:
                retained_adapter = Lfm2RetainedTraceRowAdapter.build(
                    retained_parent_rows,
                    cache_key=parent_cache.cache_key,
                    corpus_fingerprint=parent_cache.corpus_fingerprint,
                    batch_references=session.batch_references,
                    phases_by_batch=tuple(
                        _ordered_parent_phases(reference.record)
                        for reference in session.batch_references
                    ),
                    expected_layer_kinds=expected_kinds,
                )
            except Lfm2RetainedTraceRowsError as exc:
                raise Lfm2CachedFunctionalWidthError(
                    f"Retained functional rows failed authentication: {exc}"
                ) from exc

        locations: list[_FunctionalLocation] = []
        final_layer = len(discovered) - 1
        for expected_index, reference in enumerate(session.batch_references):
            if reference.index != expected_index:
                raise Lfm2CachedFunctionalWidthError(
                    "Selective cache references are not in canonical order."
                )
            available = frozenset(reference.ordered_tensor_names)
            for phase in _ordered_parent_phases(reference.record):
                for layer in discovered:
                    prefix = f"{phase}.layer.{layer.ordinal:03d}.residual."
                    pre = prefix + "pre"
                    post_operator = prefix + "post_operator"
                    post_layer = prefix + "post_layer"
                    names = (pre, post_operator, post_layer)
                    if any(name not in available for name in names):
                        raise Lfm2CachedFunctionalWidthError(
                            "Parent cache lacks a complete residual boundary triple "
                            f"for layer {layer.ordinal}, phase {phase}."
                        )
                    shapes = tuple(
                        _descriptor_shape(
                            reference,
                            name,
                            source_hidden_size=source_hidden,
                        )
                        for name in names
                    )
                    if len(set(shapes)) != 1:
                        raise Lfm2CachedFunctionalWidthError(
                            "Cached residual boundary shapes changed inside a layer."
                        )
                    row_count = math.prod(shapes[0][:-1])
                    base = {
                        "batch_index": reference.index,
                        "phase": phase,
                        "layer_index": layer.ordinal,
                        "row_count": row_count,
                    }
                    locations.extend(
                        (
                            _FunctionalLocation(
                                role="residual_context",
                                tensor_names=(pre,),
                                **base,
                            ),
                            _FunctionalLocation(
                                role="operator_write",
                                tensor_names=(post_operator, pre),
                                **base,
                            ),
                            _FunctionalLocation(
                                role="nonlinear_mlp_write",
                                tensor_names=(post_layer, post_operator),
                                **base,
                            ),
                        )
                    )
                    if layer.ordinal == final_layer:
                        locations.append(
                            _FunctionalLocation(
                                role="terminal_context",
                                tensor_names=(post_layer,),
                                **base,
                            )
                        )

        allocations: dict[str, tuple[int, ...]] = {}
        for role in _ROLES:
            allocations.update(
                _location_row_allocations(
                    tuple(value for value in locations if value.role == role),
                    maximum_rows=row_budget,
                )
            )
        selected_locations = tuple(
            value for value in locations if value.key in allocations
        )
        by_batch: dict[int, list[_FunctionalLocation]] = {}
        for location in selected_locations:
            by_batch.setdefault(location.batch_index, []).append(location)

        selected_rows_by_role = {
            role: sum(
                len(allocations[location.key])
                for location in selected_locations
                if location.role == role
            )
            for role in _ROLES
        }
        selected_tensor_name_count = sum(
            len(
                {
                    name
                    for location in batch_locations
                    for name in location.tensor_names
                }
            )
            for batch_locations in by_batch.values()
        )
        _emit_progress(
            progress,
            "functional_width_selection_completed",
            enumerated_location_count=len(locations),
            selected_location_count=len(selected_locations),
            selected_cache_batch_count=len(by_batch),
            selected_tensor_name_count=selected_tensor_name_count,
            retained_parent_rows=retained_parent_rows is not None,
            **_role_row_progress("selected", selected_rows_by_role),
        )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Functional width evidence requires PyTorch.") from exc
        accumulator = _DeterministicRoleBatchedGramAccumulator(
            source_hidden,
            row_batch_size=LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
        )
        selection_manifest = []
        ordered_batch_indices = tuple(sorted(by_batch))
        progress_stride = max(
            1,
            math.ceil(
                len(ordered_batch_indices)
                / LFM2_FUNCTIONAL_WIDTH_MAXIMUM_PROGRESS_EVENTS
            ),
        )
        _emit_progress(
            progress,
            "functional_width_gram_accumulation_started",
            selected_cache_batch_count=len(ordered_batch_indices),
            selected_location_count=len(selected_locations),
            gram_accumulation_schema=(
                LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA
            ),
            gram_accumulation_method=(
                "deterministic-role-buffered-row-batch-addmm"
            ),
            gram_row_batch_size=LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
            gram_accumulation_threads=1,
        )
        with _single_threaded_torch_cpu_math(torch):
            for completed_batch_count, batch_index in enumerate(
                ordered_batch_indices,
                start=1,
            ):
                batch_locations = sorted(
                    by_batch[batch_index],
                    key=lambda value: value.key,
                )
                selected_names = tuple(
                    sorted(
                        {
                            name
                            for location in batch_locations
                            for name in location.tensor_names
                        }
                    )
                )
                payload = None
                try:
                    if retained_adapter is None:
                        payload = session.load_selected_tensors(
                            batch_index,
                            selected_names,
                        )
                        tensors = payload.tensors
                    else:
                        reference = session.batch_references[batch_index]
                        try:
                            tensors = retained_adapter.selected_tensors(
                                reference,
                                selected_names,
                            )
                        except Lfm2RetainedTraceRowsError as exc:
                            raise Lfm2CachedFunctionalWidthError(
                                "Retained functional tensor selection failed "
                                f"authentication: {exc}"
                            ) from exc
                    for location in batch_locations:
                        indices = allocations[location.key]
                        rows = _selected_role_rows(
                            location,
                            tensors,
                            indices,
                            source_hidden_size=source_hidden,
                        )
                        norms = torch.linalg.vector_norm(rows, dim=1)
                        nonzero = norms > torch.finfo(torch.float64).tiny
                        normalized = rows[nonzero] / norms[nonzero, None]
                        if normalized.numel():
                            accumulator.add(location.role, normalized)
                        selection_manifest.append(
                            {
                                "location": location.key,
                                "tensor_names": list(location.tensor_names),
                                "sealed_row_count": location.row_count,
                                "selected_row_indices": list(indices),
                                "nonzero_selected_rows": int(nonzero.sum().item()),
                            }
                        )
                finally:
                    if payload is not None:
                        session.release_selected_tensors(payload)
                if (
                    completed_batch_count % progress_stride == 0
                    or completed_batch_count == len(ordered_batch_indices)
                ):
                    snapshot = accumulator.diagnostics()
                    _emit_progress(
                        progress,
                        "functional_width_gram_batch_progress",
                        completed_batch_count=completed_batch_count,
                        total_batch_count=len(ordered_batch_indices),
                        maximum_progress_event_count=(
                            LFM2_FUNCTIONAL_WIDTH_MAXIMUM_PROGRESS_EVENTS
                        ),
                        previous_location_matmul_count=snapshot[
                            "previous_location_matmul_count"
                        ],
                        batched_matmul_count=snapshot["batched_matmul_count"],
                        peak_pending_bytes=snapshot["peak_pending_bytes"],
                        **_role_row_progress(
                            "sampled",
                            snapshot["sampled_rows_by_role"],
                        ),
                    )
            role_grams, gram_diagnostics = accumulator.finalize()
        role_rows = dict(accumulator.rows)
        _emit_progress(
            progress,
            "functional_width_gram_accumulation_completed",
            **_gram_diagnostics_progress(gram_diagnostics),
        )
        session.verify()
        session_sha256 = session.evidence_sha256

    role_gram_hashes = []
    balanced = None
    for role in _ROLES:
        gram = role_grams[role]
        trace = torch.trace(gram)
        if (
            role_rows[role] <= 0
            or not bool(torch.isfinite(trace).item())
            or float(trace.item()) <= 0.0
        ):
            raise Lfm2CachedFunctionalWidthError(
                f"Cached functional role {role!r} has no nonzero directions."
            )
        gram.div_(trace)
        role_gram_hashes.append((role, _tensor_sha256(gram)))
        balanced = gram.clone() if balanced is None else balanced.add(gram)
    assert balanced is not None
    balanced.div_(len(_ROLES))
    _emit_progress(
        progress,
        "functional_width_eigensolve_started",
        source_hidden_size=source_hidden,
        role_gram_count=len(role_gram_hashes),
        eigensolver_threads=1,
        **_role_row_progress("sampled", role_rows),
    )
    with _single_threaded_torch_cpu_math(torch):
        eigenvalues = (
            torch.linalg.eigvalsh(balanced).clamp_min(0.0).flip(0).contiguous()
        )
    total = float(eigenvalues.sum().item())
    if not math.isfinite(total) or total <= 0.0:
        raise Lfm2CachedFunctionalWidthError(
            "Role-balanced functional Gram has no finite spectrum."
        )
    eigenspectrum_sha256 = _tensor_sha256(eigenvalues)
    _emit_progress(
        progress,
        "functional_width_eigensolve_completed",
        eigenvalue_count=int(eigenvalues.numel()),
        positive_spectrum_total=total,
        functional_eigenspectrum_sha256=eigenspectrum_sha256,
    )
    selection_manifest_sha256 = canonical_sha256(selection_manifest)
    common_selection = canonical_sha256(
        {
            "schema": LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA,
            "source_graph_fingerprint": source_graph.fingerprint,
            "parent_cache_key": parent_cache.cache_key,
            "selective_session_sha256": session_sha256,
            "sampling_policy": LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY,
            "gram_accumulation": _gram_accumulation_contract(),
            "selection_manifest_sha256": selection_manifest_sha256,
            "role_gram_sha256": dict(role_gram_hashes),
            "functional_eigenspectrum_sha256": eigenspectrum_sha256,
        }
    )
    cumulative = eigenvalues.cumsum(dim=0)
    scores = tuple(
        Lfm2FunctionalWidthScore.build(
            hidden_size=hidden,
            retained=min(1.0, float(cumulative[hidden - 1].item()) / total),
            eigenspectrum_sha256=eigenspectrum_sha256,
            selection_evidence_sha256=canonical_sha256(
                {
                    "common_selection_evidence_sha256": common_selection,
                    "hidden_size": hidden,
                    "rank_H_role_balanced_best_subspace": True,
                }
            ),
        )
        for hidden in widths
    )
    provisional = object.__new__(Lfm2CachedFunctionalWidthEvidence)
    for name, value in {
        "source_graph_fingerprint": source_graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "selective_session_sha256": session_sha256,
        "source_hidden_size": source_hidden,
        "maximum_rows_per_role": row_budget,
        "sampled_rows_by_role": tuple((role, role_rows[role]) for role in _ROLES),
        "candidate_scores": scores,
        "role_gram_sha256": tuple(role_gram_hashes),
        "functional_eigenspectrum_sha256": eigenspectrum_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "evidence_sha256": "0" * 64,
    }.items():
        object.__setattr__(provisional, name, value)
    evidence_sha256 = canonical_sha256(provisional._body())
    return Lfm2CachedFunctionalWidthEvidence(
        source_graph_fingerprint=source_graph.fingerprint,
        parent_cache_key=parent_cache.cache_key,
        selective_session_sha256=session_sha256,
        source_hidden_size=source_hidden,
        maximum_rows_per_role=row_budget,
        sampled_rows_by_role=tuple((role, role_rows[role]) for role in _ROLES),
        candidate_scores=scores,
        role_gram_sha256=tuple(role_gram_hashes),
        functional_eigenspectrum_sha256=eigenspectrum_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
        evidence_sha256=evidence_sha256,
    )


__all__ = [
    "LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA",
    "LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA",
    "LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE",
    "LFM2_FUNCTIONAL_WIDTH_MAXIMUM_PROGRESS_EVENTS",
    "LFM2_FUNCTIONAL_WIDTH_PROGRESS_SCHEMA",
    "LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY",
    "Lfm2CachedFunctionalWidthError",
    "Lfm2CachedFunctionalWidthEvidence",
    "Lfm2FunctionalWidthScore",
    "build_lfm2_cached_functional_width_evidence",
    "enumerate_lfm2_global_hidden_sizes",
]
