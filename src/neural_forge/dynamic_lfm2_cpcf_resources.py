"""Local-only production resource wiring for reviewed LFM2 CPCF reruns.

The first severe CPCF run is captured and reviewed outside the production
catalog.  This module reconstructs that exact run from its authenticated
receipt and local artifacts, then returns the one exact production bundle
that may rerun it.  It deliberately has no cache-generation, parent-execution,
optimization, pruning, quantization, or network path.

The cache is opened metadata-first.  The installability adapter owns cache
verification and the resulting target phases are materialized once.  A
separate immutable coverage binding proves that those phases cover every
trace record and every required prefill/decode phase; it does not rematerialize
the cache through a second installability audit.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import (
    ModelCircuitGraph,
    canonical_json_bytes,
    canonical_sha256,
)
from .dynamic_fold_executor import fingerprint_model_state
from .dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from .dynamic_lfm2_cpcf_adapter import (
    Lfm2CpcfExternalBehaviorGateContract,
    Lfm2CpcfStackResources,
)
from .dynamic_lfm2_cpcf_production import (
    build_lfm2_cpcf_exact_production_bundle,
)
from .dynamic_lfm2_cpcf_reviewed_gate import (
    CPCF_RUN_SCHEMA,
    ORDERED_PHASE_BINDING_SCHEMA,
    Lfm2CpcfReviewedPromotionArtifacts,
    Lfm2CpcfReviewedPromotionEvidence,
    Lfm2CpcfReviewedPromotionGate,
    authenticate_lfm2_cpcf_reviewed_promotion_evidence,
    load_lfm2_cpcf_reviewed_promotion_evidence,
)
from .dynamic_lfm2_targets import Lfm2Topology
from .dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from .dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from .dynamic_production_catalog import ProductionFoldBundle
from .dynamic_target_synthesis import (
    BudgetedDynamicTargetPlan,
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisResult,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from .hf_lfm2_cpcf_installability import Lfm2CpcfInstallabilityAdapter
from .hf_lfm2_cpcf_blank_target import LFM2_CPCF_BLANK_TARGET_SCHEMA
from .hf_lfm2_cpcf_layer_compiler import (
    parse_lfm2_cpcf_attention_sequential_transport_policy,
)
from .hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfTargetPhase,
    build_lfm2_cpcf_target_phases,
)
from .hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    Lfm2CpcfModelFoldPolicy,
)
from .hf_lfm2_cpcf_short_conv_fold import Lfm2CpcfShortConvGatePolicy
from .hf_lfm2_cpcf_short_conv_v4_policy import (
    parse_lfm2_cpcf_short_conv_v4_solver_policy,
)
from .hf_lfm2_direct_local import (
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


SCHEMA = "neural-forge-lfm2-cpcf-production-resources-v1"
COVERAGE_SCHEMA = "neural-forge-lfm2-cpcf-cache-phase-coverage-v1"
RECORD_COVERAGE_SCHEMA = "neural-forge-lfm2-cpcf-cache-record-phase-coverage-v1"
TRAJECTORY_SCHEMA = "neural-forge-lfm2-bridge-trajectory-text-v1"


class Lfm2CpcfResourceError(RuntimeError):
    """Local CPCF artifacts cannot be bound into one reviewed production lane."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfResourceError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfResourceError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfResourceError(f"{name} must be a nonnegative integer.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfResourceError(f"{name} must be nonempty text.")
    return value.strip()


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CpcfResourceError(f"{name} must be a local directory.")
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfResourceError(f"{name} does not exist.") from exc
    if not path.is_dir() or path.is_symlink():
        raise Lfm2CpcfResourceError(f"{name} must be a nonsymlink local directory.")
    return path


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CpcfResourceError(f"{name} must be a local file.")
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfResourceError(f"{name} does not exist.") from exc
    if not path.is_file() or path.is_symlink():
        raise Lfm2CpcfResourceError(f"{name} must be a nonsymlink local file.")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfResourceError(f"{name} must be a mapping.")
    return dict(value)


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    source = _mapping(value, name)
    try:
        return json.loads(canonical_json_bytes(source).decode("ascii"))
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfResourceError(
            f"{name} must contain canonical JSON values."
        ) from exc


def _read_trajectory_rows(
    path: Path,
) -> tuple[Mapping[str, str], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfResourceError(
            "Trajectory corpus is not valid local UTF-8 JSON."
        ) from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, Mapping) for row in value)
    ):
        raise Lfm2CpcfResourceError(
            "Trajectory corpus must be a nonempty list of objects."
        )
    rows: list[Mapping[str, str]] = []
    identifiers: set[str] = set()
    splits: set[str] = set()
    expected_fields = {
        "schema",
        "id",
        "split",
        "input_text",
        "forced_decode_text",
    }
    for index, raw_row in enumerate(value):
        row = dict(raw_row)
        identifier = row.get("id")
        split = row.get("split")
        prompt = row.get("input_text")
        forced = row.get("forced_decode_text")
        if (
            set(row) != expected_fields
            or row.get("schema") != TRAJECTORY_SCHEMA
            or not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or split not in {"selection", "fit", "validation"}
            or not isinstance(prompt, str)
            or not prompt
            or not isinstance(forced, str)
            or not forced
        ):
            raise Lfm2CpcfResourceError(
                f"Trajectory row {index} is incomplete, unknown, or duplicated."
            )
        identifiers.add(identifier)
        splits.add(str(split))
        rows.append(
            MappingProxyType(
                {
                    "id": identifier,
                    "split": str(split),
                    "input_text": prompt,
                    "forced_decode_text": forced,
                }
            )
        )
    if splits != {"selection", "fit", "validation"}:
        raise Lfm2CpcfResourceError(
            "Trajectory corpus must contain selection, fit, and validation."
        )
    return tuple(rows)


def _token_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    )
    if not isinstance(encoded, Mapping):
        raise Lfm2CpcfResourceError(
            "Direct local tokenizer returned a non-mapping encoding."
        )
    values = encoded.get("input_ids")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
    ):
        raise Lfm2CpcfResourceError(
            "Direct local tokenizer returned invalid token IDs."
        )
    return tuple(values)


def _build_trace_adapter(
    *,
    rows: Sequence[Mapping[str, str]],
    target_tokenizer: Any,
    bridge: StrictTokenizerBridgeArtifact,
    cache: Lfm2CircuitTraceCache,
) -> Lfm2StrictTokenizerBridgeTraceAdapter:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2CpcfResourceError(
            "PyTorch is required for CPCF trace construction."
        ) from exc
    manifest = cache.manifest
    capture = _mapping(manifest.get("capture"), "cache capture")
    corpus = _mapping(manifest.get("corpus"), "cache corpus")
    maximum_decode_steps = _positive_int(
        capture.get("max_decode_steps"),
        "cache max_decode_steps",
    )
    records = []
    for row in rows:
        prompt = _token_ids(
            target_tokenizer,
            row["input_text"],
            add_special_tokens=True,
        )
        forced = _token_ids(
            target_tokenizer,
            row["forced_decode_text"],
            add_special_tokens=False,
        )
        if len(forced) > maximum_decode_steps:
            raise Lfm2CpcfResourceError(
                f"Trajectory {row['id']!r} exceeds the sealed decode limit."
            )
        records.append(
            Lfm2TargetTokenizedRecord(
                record_id=row["id"],
                split=row["split"],
                input_ids=torch.tensor([prompt], dtype=torch.long),
                attention_mask=torch.ones((1, len(prompt)), dtype=torch.long),
                forced_decode_ids=torch.tensor([forced], dtype=torch.long),
            )
        )
    trace = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    ordered = corpus.get("ordered_batch_input_sha256")
    observed = tuple(record.donor_cache_input_sha256 for record in trace.records)
    if (
        not isinstance(ordered, list)
        or tuple(ordered) != observed
        or len(observed) != cache.batch_count
    ):
        raise Lfm2CpcfResourceError(
            "Target tokenization does not reproduce every sealed donor-cache "
            "input in order."
        )
    return trace.verify()


@dataclass(frozen=True, slots=True)
class Lfm2CpcfRecordPhaseCoverage:
    """Exact phase coverage for one trace-record/cache-batch identity."""

    ordinal: int
    record_id: str
    donor_input_sha256: str
    target_alignment_sha256: str
    forced_decode_steps: int
    layer_count: int
    phase_names: tuple[str, ...]
    parent_evidence_sha256: tuple[str, ...]
    target_phase_sha256: tuple[str, ...]
    record_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        ordinal = _nonnegative_int(self.ordinal, "ordinal")
        record_id = _text(self.record_id, "record_id")
        decode_steps = _positive_int(
            self.forced_decode_steps,
            "forced_decode_steps",
        )
        layer_count = _positive_int(self.layer_count, "layer_count")
        phases = tuple(self.phase_names)
        expected = (
            "prefill",
            *(f"decode.{index:03d}" for index in range(decode_steps)),
        )
        parent_seals = tuple(self.parent_evidence_sha256)
        target_seals = tuple(self.target_phase_sha256)
        if (
            phases != expected
            or len(parent_seals) != len(expected)
            or len(target_seals) != len(expected)
        ):
            raise Lfm2CpcfResourceError(
                f"Record {record_id!r} lacks its exact prefill/decode phase coverage."
            )
        for index, value in enumerate(parent_seals):
            _digest(value, f"parent evidence seal {ordinal}:{index}")
        for index, value in enumerate(target_seals):
            _digest(value, f"target phase seal {ordinal}:{index}")
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(
            self,
            "donor_input_sha256",
            _digest(self.donor_input_sha256, "donor_input_sha256"),
        )
        object.__setattr__(
            self,
            "target_alignment_sha256",
            _digest(
                self.target_alignment_sha256,
                "target_alignment_sha256",
            ),
        )
        object.__setattr__(self, "forced_decode_steps", decode_steps)
        object.__setattr__(self, "layer_count", layer_count)
        object.__setattr__(self, "phase_names", phases)
        object.__setattr__(
            self,
            "parent_evidence_sha256",
            parent_seals,
        )
        object.__setattr__(self, "target_phase_sha256", target_seals)
        object.__setattr__(
            self,
            "record_fingerprint",
            canonical_sha256(self.body()),
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": RECORD_COVERAGE_SCHEMA,
            "ordinal": self.ordinal,
            "record_id": self.record_id,
            "donor_input_sha256": self.donor_input_sha256,
            "target_alignment_sha256": self.target_alignment_sha256,
            "forced_decode_steps": self.forced_decode_steps,
            "layer_count": self.layer_count,
            "phase_names": list(self.phase_names),
            "parent_evidence_sha256": list(self.parent_evidence_sha256),
            "target_phase_sha256": list(self.target_phase_sha256),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.body(), "record_fingerprint": self.record_fingerprint}


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCachePhaseCoverageBinding:
    """Immutable cache/trace/phase coverage root constructed without re-audit."""

    cache_manifest_sha256: str
    cache_key: str
    cache_parent_fingerprint: str
    cache_corpus_fingerprint: str
    cache_batch_count: int
    parent_graph_fingerprint: str
    parent_layer_count: int
    trace_adapter_sha256: str
    trajectory_corpus_sha256: str
    records: tuple[Lfm2CpcfRecordPhaseCoverage, ...]
    binding_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "cache_manifest_sha256",
            "cache_key",
            "cache_parent_fingerprint",
            "cache_corpus_fingerprint",
            "parent_graph_fingerprint",
            "trace_adapter_sha256",
            "trajectory_corpus_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        batch_count = _positive_int(self.cache_batch_count, "cache_batch_count")
        layer_count = _positive_int(self.parent_layer_count, "parent_layer_count")
        records = tuple(self.records)
        if (
            len(records) != batch_count
            or any(
                not isinstance(record, Lfm2CpcfRecordPhaseCoverage)
                for record in records
            )
            or tuple(record.ordinal for record in records) != tuple(range(batch_count))
            or any(record.layer_count != layer_count for record in records)
        ):
            raise Lfm2CpcfResourceError(
                "Cache coverage records do not cover every ordinal with the "
                "full inspected parent depth."
            )
        object.__setattr__(self, "cache_batch_count", batch_count)
        object.__setattr__(self, "parent_layer_count", layer_count)
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "binding_fingerprint",
            canonical_sha256(self.body()),
        )

    @property
    def phase_count(self) -> int:
        return sum(len(record.phase_names) for record in self.records)

    def body(self) -> dict[str, Any]:
        return {
            "schema": COVERAGE_SCHEMA,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cache_key": self.cache_key,
            "cache_parent_fingerprint": self.cache_parent_fingerprint,
            "cache_corpus_fingerprint": self.cache_corpus_fingerprint,
            "cache_batch_count": self.cache_batch_count,
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "parent_layer_count": self.parent_layer_count,
            "trace_adapter_sha256": self.trace_adapter_sha256,
            "trajectory_corpus_sha256": self.trajectory_corpus_sha256,
            "trace_record_ordinals": [record.ordinal for record in self.records],
            "trace_record_count": len(self.records),
            "phase_count": self.phase_count,
            "records": [record.as_dict() for record in self.records],
            "installability_audit_rematerialized": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.body(), "binding_fingerprint": self.binding_fingerprint}


def build_lfm2_cpcf_cache_phase_coverage_binding(
    *,
    cache: Lfm2CircuitTraceCache,
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    phases: Sequence[Lfm2CpcfTargetPhase],
    parent_graph: ModelCircuitGraph,
    trajectory_corpus_sha256: str,
) -> Lfm2CpcfCachePhaseCoverageBinding:
    """Prove complete phase coverage from the already materialized phases."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(
        trace_adapter,
        Lfm2StrictTokenizerBridgeTraceAdapter,
    ):
        raise TypeError("trace_adapter must be Lfm2StrictTokenizerBridgeTraceAdapter.")
    if not isinstance(parent_graph, ModelCircuitGraph):
        raise TypeError("parent_graph must be ModelCircuitGraph.")
    trace = trace_adapter.verify()
    ordered_phases = tuple(phases)
    if not ordered_phases or any(
        not isinstance(phase, Lfm2CpcfTargetPhase) for phase in ordered_phases
    ):
        raise Lfm2CpcfResourceError(
            "phases must contain materialized CPCF target phases."
        )
    manifest = cache.manifest
    manifest_sha256 = _digest(
        manifest.get("manifest_sha256"),
        "cache manifest SHA-256",
    )
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256", None)
    if canonical_sha256(manifest_body) != manifest_sha256:
        raise Lfm2CpcfResourceError("Cache manifest canonical seal is invalid.")
    expected_layer_indices = tuple(parent_graph.layer_indices)
    layer_count = len(expected_layer_indices)
    _positive_int(layer_count, "parent graph layer count")
    if trace.record_count != cache.batch_count:
        raise Lfm2CpcfResourceError(
            "Trace record count differs from the sealed cache batch count."
        )
    expected_global: list[tuple[int, str]] = []
    records = []
    for ordinal, record in enumerate(trace.records):
        if record.ordinal != ordinal:
            raise Lfm2CpcfResourceError(
                "Trace record ordinals are incomplete or reordered."
            )
        expected_names = (
            "prefill",
            *(f"decode.{index:03d}" for index in range(record.forced_decode_steps)),
        )
        expected_global.extend((ordinal, phase_name) for phase_name in expected_names)
        selected = tuple(
            phase for phase in ordered_phases if phase.batch_index == ordinal
        )
        if tuple(phase.phase for phase in selected) != expected_names:
            raise Lfm2CpcfResourceError(
                f"Cache batch {ordinal} does not contain exactly its trace-"
                "required prefill/decode phases."
            )
        if any(
            phase.record_id != record.record_id
            or phase.split != record.split
            or phase.donor_input_sha256 != record.donor_cache_input_sha256
            or phase.target_alignment_sha256 != record.alignment_sha256
            or phase.parent_evidence.cache_key != cache.cache_key
            or phase.parent_evidence.batch_index != ordinal
            or phase.parent_evidence.phase != phase.phase
            or len(phase.parent_evidence.layers) != layer_count
            or tuple(layer.layer_index for layer in phase.parent_evidence.layers)
            != expected_layer_indices
            for phase in selected
        ):
            raise Lfm2CpcfResourceError(
                f"Cache batch {ordinal} phase evidence differs from its trace, "
                "cache, or full inspected parent depth."
            )
        records.append(
            Lfm2CpcfRecordPhaseCoverage(
                ordinal=ordinal,
                record_id=record.record_id,
                donor_input_sha256=record.donor_cache_input_sha256,
                target_alignment_sha256=record.alignment_sha256,
                forced_decode_steps=record.forced_decode_steps,
                layer_count=layer_count,
                phase_names=expected_names,
                parent_evidence_sha256=tuple(
                    phase.parent_evidence.evidence_sha256 for phase in selected
                ),
                target_phase_sha256=tuple(phase.phase_sha256 for phase in selected),
            )
        )
    if tuple(phase.identity for phase in ordered_phases) != tuple(expected_global):
        raise Lfm2CpcfResourceError(
            "Materialized CPCF phases contain an omission, reordering, or "
            "cross-record insertion."
        )
    return Lfm2CpcfCachePhaseCoverageBinding(
        cache_manifest_sha256=manifest_sha256,
        cache_key=cache.cache_key,
        cache_parent_fingerprint=cache.parent_fingerprint,
        cache_corpus_fingerprint=cache.corpus_fingerprint,
        cache_batch_count=cache.batch_count,
        parent_graph_fingerprint=parent_graph.fingerprint,
        parent_layer_count=layer_count,
        trace_adapter_sha256=trace.evidence_sha256,
        trajectory_corpus_sha256=trajectory_corpus_sha256,
        records=tuple(records),
    )


def _budget_from_receipt(value: Any) -> DenseParameterBudgetRequest:
    payload = _canonical_mapping(value, "receipt requested budget")
    expected_keys = {
        "schema",
        "kind",
        "target_family",
        "target_modalities",
        "target_unique_parameters",
        "compression_ratio",
        "target_vocabulary_size",
        "match_policy",
        "maximum_candidates",
    }
    if set(payload) != expected_keys or payload.get("kind") != "budget_request":
        raise Lfm2CpcfResourceError(
            "Receipt requested budget has unknown or incomplete fields."
        )
    ratio_value = payload.get("compression_ratio")
    ratio: str | None
    if ratio_value is None:
        ratio = None
    else:
        ratio_payload = _mapping(
            ratio_value,
            "receipt compression ratio",
        )
        if set(ratio_payload) != {"numerator", "denominator"}:
            raise Lfm2CpcfResourceError("Receipt compression ratio is incomplete.")
        numerator = _positive_int(
            ratio_payload.get("numerator"),
            "compression ratio numerator",
        )
        denominator = _positive_int(
            ratio_payload.get("denominator"),
            "compression ratio denominator",
        )
        ratio = f"{numerator}/{denominator}"
    budget = DenseParameterBudgetRequest(
        target_family=payload.get("target_family"),
        target_modalities=tuple(payload.get("target_modalities", ())),
        target_unique_parameters=payload.get("target_unique_parameters"),
        compression_ratio=ratio,
        target_vocabulary_size=payload.get("target_vocabulary_size"),
        match_policy=payload.get("match_policy"),
        maximum_candidates=payload.get("maximum_candidates"),
    )
    if canonical_json_bytes(budget.as_dict()) != canonical_json_bytes(payload):
        raise Lfm2CpcfResourceError(
            "Receipt requested budget does not round-trip through the typed "
            "budget contract."
        )
    return budget


def _policy_from_receipt(document: Mapping[str, Any]) -> Lfm2CpcfModelFoldPolicy:
    compiled = _mapping(
        document.get("compiled_model_fold"),
        "compiled model-fold receipt",
    )
    result_sha256 = _digest(
        compiled.get("result_sha256"),
        "compiled model-fold result SHA-256",
    )
    body = dict(compiled)
    body.pop("result_sha256", None)
    if (
        body.get("schema") != LFM2_CPCF_MODEL_FOLD_SCHEMA
        or canonical_sha256(body) != result_sha256
    ):
        raise Lfm2CpcfResourceError(
            "Compiled model-fold receipt schema or seal is invalid."
        )
    payload = _canonical_mapping(body.get("policy"), "CPCF fold policy")
    expected_keys = {
        "ridge",
        "state_reconstruction_weight",
        "minimum_predictive_energy_fraction",
        "minimum_validation_block_retention",
        "enforce_validation_block_floor",
        "short_conv_gate_policy",
        "promotion_compression_ratio",
        "require_promotion_ratio",
    }
    optional_v4_key = "short_conv_v4_solver_policy"
    optional_attention_key = "attention_sequential_transport_policy"
    if (
        not expected_keys.issubset(payload)
        or not set(payload).issubset(
            {
                *expected_keys,
                optional_v4_key,
                optional_attention_key,
            }
        )
    ):
        raise Lfm2CpcfResourceError(
            "CPCF fold policy has unknown or incomplete fields."
        )
    v4_payload = payload.pop(optional_v4_key, None)
    attention_payload = payload.pop(optional_attention_key, None)
    short_payload = _canonical_mapping(
        payload.pop("short_conv_gate_policy"),
        "short-convolution gate policy",
    )
    short_keys = {
        "maximum_fit_endpoint_relative_rmse",
        "maximum_validation_endpoint_relative_rmse",
        "minimum_fit_endpoint_cosine_similarity",
        "minimum_validation_endpoint_cosine_similarity",
        "enforce",
    }
    if set(short_payload) != short_keys:
        raise Lfm2CpcfResourceError(
            "Short-convolution gate policy has unknown or incomplete fields."
        )
    constructor = {
        **payload,
        "short_conv_gate_policy": Lfm2CpcfShortConvGatePolicy(**short_payload),
    }
    if v4_payload is not None:
        try:
            constructor[optional_v4_key] = parse_lfm2_cpcf_short_conv_v4_solver_policy(
                v4_payload
            )
        except (TypeError, ValueError) as exc:
            raise Lfm2CpcfResourceError(
                "Short-convolution v4 solver policy is invalid."
            ) from exc
    if attention_payload is not None:
        try:
            constructor[optional_attention_key] = (
                parse_lfm2_cpcf_attention_sequential_transport_policy(
                    attention_payload
                )
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2CpcfResourceError(
                "Attention sequential transport policy is invalid."
            ) from exc
    policy = Lfm2CpcfModelFoldPolicy(**constructor)
    expected_policy = _mapping(body.get("policy"), "CPCF fold policy")
    if canonical_json_bytes(policy.as_dict()) != canonical_json_bytes(expected_policy):
        raise Lfm2CpcfResourceError("CPCF fold policy does not round-trip exactly.")
    return policy


def _mapped_source_tied_table(
    parent_model: Any,
    bridge: StrictTokenizerBridgeArtifact,
) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2CpcfResourceError(
            "PyTorch is required for the CPCF tied interface."
        ) from exc
    get_input = getattr(parent_model, "get_input_embeddings", None)
    get_output = getattr(parent_model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise Lfm2CpcfResourceError(
            "Direct local parent exposes no embedding interfaces."
        )
    input_weight = getattr(get_input(), "weight", None)
    output_weight = getattr(get_output(), "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or input_weight.ndim != 2
        or int(input_weight.shape[0]) != bridge.transport_profile.donor_row_capacity
    ):
        raise Lfm2CpcfResourceError(
            "CPCF requires one native tied parent table matching the strict "
            "bridge donor-row capacity."
        )
    mapping = bridge.transport_profile.target_to_donor_row_indices
    index = torch.tensor(
        mapping,
        device=input_weight.device,
        dtype=torch.long,
    )
    if int(index.max().item()) >= int(input_weight.shape[0]):
        raise Lfm2CpcfResourceError(
            "Strict bridge references a row outside the parent tied table."
        )
    return input_weight.detach().index_select(0, index).to(device="cpu").contiguous()


def _verify_receipt_phase_and_policy_binding(
    document: Mapping[str, Any],
    *,
    evidence: Lfm2CpcfReviewedPromotionEvidence,
    phases: Sequence[Lfm2CpcfTargetPhase],
    policy: Lfm2CpcfModelFoldPolicy,
    blank_report_sha256: str,
) -> None:
    compiled = _mapping(
        document.get("compiled_model_fold"),
        "compiled model-fold receipt",
    )
    observed_phases = tuple(
        _digest(value, "receipt target phase SHA-256")
        for value in compiled.get("phase_sha256", ())
    )
    expected_phases = tuple(phase.phase_sha256 for phase in phases)
    policy_sha256 = canonical_sha256(policy.as_dict())
    phase_binding_sha256 = canonical_sha256(
        {
            "schema": ORDERED_PHASE_BINDING_SCHEMA,
            "phase_count": len(expected_phases),
            "ordered_phase_sha256": list(expected_phases),
        }
    )
    parameter_receipt = _mapping(
        compiled.get("parameter_receipt"),
        "compiled parameter receipt",
    )
    if (
        compiled.get("phase_count") != len(expected_phases)
        or observed_phases != expected_phases
        or compiled.get("blank_target_report_sha256") != blank_report_sha256
        or canonical_json_bytes(compiled.get("policy"))
        != canonical_json_bytes(policy.as_dict())
        or compiled.get("result_sha256") != evidence.compiled_model_fold_result_sha256
        or canonical_json_bytes(evidence.compiled_model_fold_policy)
        != canonical_json_bytes(policy.as_dict())
        or evidence.compiled_model_fold_policy_sha256 != policy_sha256
        or evidence.compiled_model_fold_phase_count != len(expected_phases)
        or evidence.compiled_model_fold_phase_sha256 != expected_phases
        or evidence.compiled_model_fold_phase_binding_sha256 != phase_binding_sha256
        or evidence.compiled_blank_target_report_sha256 != blank_report_sha256
        or parameter_receipt.get("receipt_sha256")
        != evidence.compiled_parameter_receipt_sha256
    ):
        raise Lfm2CpcfResourceError(
            "Reconstructed phases, blank report, policy, or parameter receipt "
            "differ from the loader-authenticated reviewed CPCF run."
        )


def _plan_from_reviewed_receipt(
    *,
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    bridge: StrictTokenizerBridgeArtifact,
    evidence: Lfm2CpcfReviewedPromotionEvidence,
    parent_name: str,
    child_name: str,
    blank_seed: int,
    minimum_hidden_size: int,
    target_adapter: Lfm2TargetSynthesisAdapter,
) -> tuple[BudgetedDynamicTargetPlan, DynamicTargetSynthesisResult]:
    source_name = _text(parent_name, "parent_name")
    budget = _budget_from_receipt(evidence.target_requested_budget)
    if budget.target_vocabulary_size != bridge.transport_profile.target_row_capacity:
        raise Lfm2CpcfResourceError(
            "Receipt target vocabulary differs from the strict bridge."
        )
    catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=_positive_int(
                    minimum_hidden_size,
                    "minimum_hidden_size",
                ),
                tokenizer_bridge_artifact=bridge,
            ),
        )
    )
    plan = plan_budgeted_dynamic_target(
        parent_graph,
        budget,
        child_name=_text(child_name, "child_name"),
        selected_candidate_rank=evidence.selected_target_candidate_rank,
        tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        adapter_id=target_adapter.adapter_id,
        fold_method_id=None,
        blank_seed=_nonnegative_int(blank_seed, "blank_seed"),
        catalogs=catalogs,
    )
    candidate = plan.frontier.candidates[
        plan.target_certificate.selected_candidate_rank
    ]
    if (
        plan.frontier.fingerprint != evidence.target_frontier_fingerprint
        or plan.target_certificate.fingerprint != evidence.target_certificate_sha256
        or candidate.fingerprint != evidence.selected_target_candidate_fingerprint
        or plan.target_certificate.parent_graph_fingerprint
        != evidence.blank_target_parent_graph_fingerprint
        or plan.target_request.exact_unique_parameters
        != evidence.exact_target_unique_parameters
    ):
        raise Lfm2CpcfResourceError(
            "Reconstructed target plan differs from reviewed run provenance. "
            "Legacy production-v3 receipts do not serialize child_name, "
            "blank_seed, or catalog minimum_hidden_size; supply their exact "
            "original values to this factory."
        )
    synthesis = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize_planned(
        plan=plan,
        parent_models={source_name: parent_model},
        parent_tokenizer_fingerprints={
            source_name: bridge.donor_tokenizer_semantics_sha256
        },
    )
    return plan, synthesis


@dataclass(frozen=True, slots=True)
class Lfm2CpcfProductionWiring:
    """One fully authenticated local CPCF production registration."""

    parent_model: Any
    parent_graph: ModelCircuitGraph
    donor_tokenizer: Any
    target_tokenizer: Any
    tokenizer_bridge: StrictTokenizerBridgeArtifact
    cache: Lfm2CircuitTraceCache
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter
    coverage_binding: Lfm2CpcfCachePhaseCoverageBinding
    reviewed_evidence: Lfm2CpcfReviewedPromotionEvidence
    reviewed_gate: Lfm2CpcfReviewedPromotionGate
    target_plan: BudgetedDynamicTargetPlan
    verified_blank_target: DynamicTargetSynthesisResult
    target_adapter: Lfm2TargetSynthesisAdapter
    stack_resources: Lfm2CpcfStackResources
    checkpoint_io: Lfm2DirectLocalCheckpointIO
    production_bundle: ProductionFoldBundle
    wiring_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "parent_graph_fingerprint": self.parent_graph.fingerprint,
            "cache_key": self.cache.cache_key,
            "trace_adapter_sha256": self.trace_adapter.evidence_sha256,
            "coverage_binding_fingerprint": (self.coverage_binding.binding_fingerprint),
            "reviewed_evidence_fingerprint": (
                self.reviewed_evidence.full_reviewed_evidence_fingerprint
            ),
            "target_plan_fingerprint": self.target_plan.fingerprint,
            "verified_blank_target_fingerprint": (
                self.verified_blank_target.fingerprint
            ),
            "stack_resource_fingerprint": (self.stack_resources.resource_fingerprint),
            "checkpoint_io_fingerprint": (self.checkpoint_io.resource_fingerprint),
            "production_bundle_fingerprint": (self.production_bundle.fingerprint),
            "wiring_fingerprint": self.wiring_fingerprint,
        }


def build_lfm2_cpcf_production_resources(
    *,
    parent_model_directory: str | os.PathLike[str],
    donor_tokenizer_directory: str | os.PathLike[str],
    target_tokenizer_directory: str | os.PathLike[str],
    bridge_directory: str | os.PathLike[str],
    trace_cache_directory: str | os.PathLike[str],
    trajectory_corpus: str | os.PathLike[str],
    reviewed_artifacts: Lfm2CpcfReviewedPromotionArtifacts,
    expected_full_reviewed_evidence_fingerprint: str,
    expected_cpcf_run_receipt_sha256: str,
    expected_cpcf_run_receipt_document_sha256: str,
    expected_bridge_artifact_sha256: str,
    expected_cache_key: str,
    expected_cache_manifest_sha256: str,
    expected_trajectory_corpus_sha256: str,
    expected_parent_graph_fingerprint: str,
    expected_exact_target_unique_parameters: int,
    device: Any,
    dtype: Any,
    parent_name: str = "lfm2-parent",
    child_name: str = "lfm2-cpcf-folded-child",
    blank_seed: int = 0,
    minimum_hidden_size: int = 32,
    snapshot_config: Any | None = None,
) -> Lfm2CpcfProductionWiring:
    """Build one exact local reviewed-rerun bundle without executing a parent."""

    expected_full = _digest(
        expected_full_reviewed_evidence_fingerprint,
        "expected full reviewed-evidence fingerprint",
    )
    expected_receipt = _digest(
        expected_cpcf_run_receipt_sha256,
        "expected CPCF run receipt SHA-256",
    )
    expected_receipt_document = _digest(
        expected_cpcf_run_receipt_document_sha256,
        "expected CPCF run receipt document SHA-256",
    )
    expected_bridge = _digest(
        expected_bridge_artifact_sha256,
        "expected tokenizer bridge artifact SHA-256",
    )
    expected_cache = _digest(expected_cache_key, "expected cache key")
    expected_cache_manifest = _digest(
        expected_cache_manifest_sha256,
        "expected cache manifest SHA-256",
    )
    expected_trajectory = _digest(
        expected_trajectory_corpus_sha256,
        "expected trajectory corpus SHA-256",
    )
    expected_graph = _digest(
        expected_parent_graph_fingerprint,
        "expected parent graph fingerprint",
    )
    expected_parameters = _positive_int(
        expected_exact_target_unique_parameters,
        "expected exact target unique parameters",
    )
    parent_name = _text(parent_name, "parent_name")

    parent_directory = _local_directory(
        parent_model_directory,
        "parent_model_directory",
    )
    donor_directory = _local_directory(
        donor_tokenizer_directory,
        "donor_tokenizer_directory",
    )
    target_directory = _local_directory(
        target_tokenizer_directory,
        "target_tokenizer_directory",
    )
    bridge_root = _local_directory(bridge_directory, "bridge_directory")
    cache_root = _local_directory(
        trace_cache_directory,
        "trace_cache_directory",
    )
    trajectory_path = _local_file(trajectory_corpus, "trajectory_corpus")

    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(reviewed_artifacts)
    authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)
    if (
        evidence.full_reviewed_evidence_fingerprint != expected_full
        or evidence.cpcf_run_receipt_sha256 != expected_receipt
        or evidence.cpcf_run_receipt_document_sha256 != expected_receipt_document
        or evidence.bridge_artifact_sha256 != expected_bridge
        or evidence.parent_cache_key != expected_cache
        or evidence.blank_target_parent_graph_fingerprint != expected_graph
        or evidence.exact_target_unique_parameters != expected_parameters
    ):
        raise Lfm2CpcfResourceError(
            "Reviewed evidence differs from an independently supplied production pin."
        )
    receipt = reviewed_artifacts.cpcf_run_receipt.load("CPCF run receipt")
    if (
        receipt.get("schema") != CPCF_RUN_SCHEMA
        or receipt.get("receipt_sha256") != expected_receipt
        or reviewed_artifacts.cpcf_run_receipt.document_sha256
        != expected_receipt_document
        or canonical_sha256(receipt) != expected_receipt_document
    ):
        raise Lfm2CpcfResourceError(
            "Reviewed CPCF receipt differs from its independent seal."
        )

    donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_directory)
    target_tokenizer = load_lfm2_direct_local_tokenizer(target_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_root,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=expected_bridge,
    )
    bridge.verify_files()
    if (
        fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bridge.target_tokenizer_semantics_sha256
        or evidence.bridge_profile_sha256 != bridge.profile_fingerprint
        or evidence.target_tokenizer_semantics_sha256
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfResourceError(
            "Direct local tokenizers, reviewed evidence, and strict bridge "
            "describe different token domains."
        )

    parent = load_lfm2_direct_local_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    parent.requires_grad_(False)
    parent.eval()
    parent_model_identity = fingerprint_lfm2_parent(parent).identity_sha256
    parent_state = fingerprint_model_state(parent)
    parent_graph = inspect_model_circuits(
        parent,
        model_name=parent_name,
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    if (
        parent_model_identity != evidence.cache_live_parent_fingerprint
        or parent_state != evidence.live_parent_state_fingerprint
        or parent_graph.fingerprint != expected_graph
        or parent_graph.family != "lfm2"
    ):
        raise Lfm2CpcfResourceError(
            "Direct local parent differs from the separately authenticated "
            "cache-model identity, dynamic state, or inspected graph."
        )

    cache = open_lfm2_circuit_trace_cache(cache_root, verify=False)
    cache_manifest = cache.manifest
    if (
        cache.cache_key != expected_cache
        or cache.parent_fingerprint != parent_model_identity
        or cache_manifest.get("manifest_sha256") != expected_cache_manifest
    ):
        raise Lfm2CpcfResourceError(
            "Sealed cache differs from the pinned key, manifest, or direct "
            "local parent."
        )
    trajectory_sha256 = _file_sha256(trajectory_path)
    if (
        trajectory_sha256 != expected_trajectory
        or trajectory_sha256 != evidence.trajectory_corpus_sha256
    ):
        raise Lfm2CpcfResourceError(
            "Trajectory corpus differs from the reviewed fold receipt."
        )
    rows = _read_trajectory_rows(trajectory_path)
    trace = _build_trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache=cache,
    )
    if trace.evidence_sha256 != evidence.trace_adapter_sha256:
        raise Lfm2CpcfResourceError(
            "Reconstructed strict trace differs from reviewed run evidence."
        )
    installability = Lfm2CpcfInstallabilityAdapter(cache)
    phases = build_lfm2_cpcf_target_phases(
        trace,
        installability.iter_compiler_inputs(),
    )
    coverage = build_lfm2_cpcf_cache_phase_coverage_binding(
        cache=cache,
        trace_adapter=trace,
        phases=phases,
        parent_graph=parent_graph,
        trajectory_corpus_sha256=trajectory_sha256,
    )
    policy = _policy_from_receipt(receipt)

    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=expected_cache,
        device=device,
        dtype=dtype,
        tokenizer_bridge_artifact=bridge,
    )
    target_plan, verified_blank = _plan_from_reviewed_receipt(
        parent_model=parent,
        parent_graph=parent_graph,
        bridge=bridge,
        evidence=evidence,
        parent_name=parent_name,
        child_name=child_name,
        blank_seed=blank_seed,
        minimum_hidden_size=minimum_hidden_size,
        target_adapter=target_adapter,
    )
    if (
        verified_blank.topology_fingerprint != evidence.target_topology_fingerprint
        or verified_blank.blank_state_fingerprint
        != evidence.blank_target_state_fingerprint
        or verified_blank.resource_profile.unique_parameters != expected_parameters
    ):
        raise Lfm2CpcfResourceError(
            "Fresh compiler-owned blank differs from the reviewed target "
            "topology, state, or exact parameter count."
        )
    blank_report = _mapping(
        receipt.get("blank_target"),
        "receipt blank target",
    )
    blank_report_sha256 = _digest(
        blank_report.get("report_sha256"),
        "blank target report SHA-256",
    )
    blank_body = dict(blank_report)
    blank_body.pop("report_sha256", None)
    blank_bindings = {
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "parent_state_fingerprint": parent_state,
        "parent_unique_parameters": parent_graph.unique_parameter_count,
        "target_unique_parameters": expected_parameters,
        "frontier_fingerprint": target_plan.frontier.fingerprint,
        "target_certificate_sha256": (target_plan.target_certificate.fingerprint),
        "selected_candidate_fingerprint": (
            evidence.selected_target_candidate_fingerprint
        ),
        "selected_candidate_rank": evidence.selected_target_candidate_rank,
        "target_topology_fingerprint": (verified_blank.topology_fingerprint),
        "blank_state_fingerprint": verified_blank.blank_state_fingerprint,
    }
    if (
        blank_body.get("schema") != LFM2_CPCF_BLANK_TARGET_SCHEMA
        or canonical_sha256(blank_body) != blank_report_sha256
        or any(blank_body.get(name) != value for name, value in blank_bindings.items())
        or canonical_json_bytes(blank_body.get("requested_budget"))
        != canonical_json_bytes(dict(evidence.target_requested_budget))
    ):
        raise Lfm2CpcfResourceError(
            "Receipt blank-target report seal or exact plan/state binding is invalid."
        )
    _verify_receipt_phase_and_policy_binding(
        receipt,
        evidence=evidence,
        phases=phases,
        policy=policy,
        blank_report_sha256=blank_report_sha256,
    )

    reviewed = dict(evidence.reviewed_evidence)
    external_contract = Lfm2CpcfExternalBehaviorGateContract(
        independent_holdout_manifest_sha256=evidence.manifest.manifest_sha256,
        independent_parent_envelope_sha256=_digest(
            reviewed.get("parent_envelope_sha256"),
            "reviewed parent envelope SHA-256",
        ),
        parent_relative_manual_packet_sha256=_digest(
            reviewed.get("parent_relative_manual_packet_sha256"),
            "reviewed parent-relative packet SHA-256",
        ),
        independent_holdout_case_count=len(evidence.manifest.cases),
    )
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=target_tokenizer,
        tokenizer_directory=target_directory,
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=device,
        dtype=dtype,
    )
    source_tied_table = _mapped_source_tied_table(parent, bridge)
    resources = Lfm2CpcfStackResources(
        phases=phases,
        trace_adapter=trace,
        coverage_binding=coverage,
        parent_graph_fingerprint=parent_graph.fingerprint,
        source_tied_table=source_tied_table,
        policy=policy,
        external_behavior_contract=external_contract,
        expected_full_reviewed_evidence_fingerprint=expected_full,
        final_evaluator=gate,
        model_saver=checkpoint_io.save,
        model_loader=checkpoint_io.load,
        fresh_reload_evaluator=gate,
        saver_id=checkpoint_io.saver_id,
        loader_id=checkpoint_io.loader_id,
        snapshot_config=snapshot_config,
    )
    bundle = build_lfm2_cpcf_exact_production_bundle(
        target_plan=target_plan,
        target_adapter=target_adapter,
        resources=resources,
    )
    if (
        fingerprint_lfm2_parent(parent).identity_sha256 != parent_model_identity
        or fingerprint_model_state(parent) != parent_state
    ):
        raise Lfm2CpcfResourceError(
            "Parent changed while local production resources were wired."
        )
    wiring_body = {
        "schema": SCHEMA,
        "cpcf_run_receipt_sha256": expected_receipt,
        "cpcf_run_receipt_document_sha256": expected_receipt_document,
        "full_reviewed_evidence_fingerprint": expected_full,
        "bridge_artifact_sha256": expected_bridge,
        "cache_key": expected_cache,
        "cache_manifest_sha256": expected_cache_manifest,
        "parent_model_fingerprint": parent_model_identity,
        "parent_state_fingerprint": parent_state,
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "exact_target_unique_parameters": expected_parameters,
        "trajectory_corpus_sha256": expected_trajectory,
        "trace_adapter_sha256": trace.evidence_sha256,
        "coverage_binding_fingerprint": coverage.binding_fingerprint,
        "policy_sha256": canonical_sha256(policy.as_dict()),
        "target_plan_fingerprint": target_plan.fingerprint,
        "verified_blank_target_fingerprint": verified_blank.fingerprint,
        "stack_resource_fingerprint": resources.resource_fingerprint,
        "checkpoint_io_fingerprint": checkpoint_io.resource_fingerprint,
        "production_bundle_fingerprint": bundle.fingerprint,
        "cache_regenerated": False,
        "parent_executed": False,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "network_allowed": False,
    }
    wiring_fingerprint = canonical_sha256(wiring_body)
    return Lfm2CpcfProductionWiring(
        parent_model=parent,
        parent_graph=parent_graph,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        tokenizer_bridge=bridge,
        cache=cache,
        trace_adapter=trace,
        coverage_binding=coverage,
        reviewed_evidence=evidence,
        reviewed_gate=gate,
        target_plan=target_plan,
        verified_blank_target=verified_blank,
        target_adapter=target_adapter,
        stack_resources=resources,
        checkpoint_io=checkpoint_io,
        production_bundle=bundle,
        wiring_fingerprint=wiring_fingerprint,
    )


__all__ = [
    "COVERAGE_SCHEMA",
    "Lfm2CpcfCachePhaseCoverageBinding",
    "Lfm2CpcfProductionWiring",
    "Lfm2CpcfRecordPhaseCoverage",
    "Lfm2CpcfResourceError",
    "RECORD_COVERAGE_SCHEMA",
    "SCHEMA",
    "build_lfm2_cpcf_cache_phase_coverage_binding",
    "build_lfm2_cpcf_production_resources",
]
