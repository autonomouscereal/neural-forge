"""Zero-training native-vision folding for the validated NEO architecture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_behavior_gate import (
    BehaviorCallableIdentity,
    BehaviorInferenceExecutionContract,
    TaskScoreCoveragePolicy,
    build_behavior_gate_binding,
    execution_contract_satisfied,
    expose_behavior_gate_binding,
    identify_behavior_callable,
    observe_model_execution,
)
from .dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvidence,
    FoldEvaluationContext,
    FoldExecutionContext,
    FoldOperatorRegistry,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_fold_snapshot import capture_model_state
from .dynamic_modality_fold_selection import (
    ModalityFoldSelectionVocabulary,
    normalize_sparse_anchor_selection,
    selection_from_statistics,
    sparse_anchor_regression,
    sparse_compensation_terms,
    survivors_for,
)
from .dynamic_neo_adapter import NeoNativeVisionTargetSynthesisAdapter
from .dynamic_target_synthesis import DynamicTargetSynthesisResult


METHOD_ID = "neo_native_vision_swiglu_model"
BEHAVIOR_SCHEMA = "neural-forge-neo-native-parent-record-v1"
FOLD_CACHE_SCHEMA = "neural-forge-neo-native-fold-cache-v1"
RECEIPT_SCHEMA = "neural-forge-neo-native-dynamic-fold-receipt-v1"
STACK_ADAPTER_ID = "neural_forge.neo.native_vision.catalog"
PROBE_SCHEMA = "neural-forge-native-vision-probe-v1"
IMAGE_HASH_BASIS = "sha256('RGB:<width>x<height>\\n' + decoded RGB bytes)"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DIRECT_WORD_RE = re.compile(r"\s*([A-Za-z]+)\s*")
_DIRECT_DIGIT_RE = re.compile(r"\s*([0-9]+)\s*")
_DIRECT_OCR_RE = re.compile(r"\s*([0-9]{3})\s*")
_COLOR_ANSWERS = frozenset({"red", "green", "blue", "yellow", "purple", "orange"})
_SPATIAL_ANSWERS = frozenset({"left", "right", "above", "below"})
_CATEGORIES = ("color", "count", "spatial", "ocr")
_SPLITS = ("calibration", "heldout")
_STATISTIC_NAMES = (
    "mean_abs",
    "mean_square",
    "sketch",
    "down_norm",
    "residual_endpoint_mean_square_l2",
)
NEO_INFERENCE_EXECUTION_CONTRACT = BehaviorInferenceExecutionContract(
    entrypoint="chat",
    required_module_path="language_model",
)


class DynamicNeoFoldError(RuntimeError):
    """Native NEO evidence, folding, behavior, or publication is invalid."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DynamicNeoFoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicNeoFoldError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicNeoFoldError(f"{name} must be a non-negative integer.")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicNeoFoldError(f"{name} must be finite.")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicNeoFoldError(f"{name} must be finite.")
    return result


# The NEO lane's binding of the shared modality-neutral selection numerics.
# The labels reproduce this lane's own message wording exactly; the validators
# are this lane's own, so the NEO-specific ``_finite`` message survives.
_SELECTION_VOCABULARY = ModalityFoldSelectionVocabulary(
    error_type=DynamicNeoFoldError,
    subject_label="NEO",
    selection_label="NEO",
    nonnegative_int=_nonnegative_int,
    positive_int=_positive_int,
    finite=_finite,
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
        raise DynamicNeoFoldError("Payload is not canonical JSON.") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _decoded_rgb_sha256(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise DynamicNeoFoldError(
            "Pillow is required to authenticate NEO probe images."
        ) from exc
    try:
        with Image.open(path) as source:
            canonical = source.convert("RGB")
            pixels = canonical.tobytes()
            width = canonical.width
            height = canonical.height
    except (OSError, ValueError) as exc:
        raise DynamicNeoFoldError(
            f"NEO probe image cannot be decoded: {path}."
        ) from exc
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}x{height}\n".encode("ascii"))
    digest.update(pixels)
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_descriptor(value: Any) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").contiguous()
    return {
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype),
        "sha256": _tensor_sha256(tensor),
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json(dict(payload)) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_direct_answer(category: str, raw_output: Any) -> str | None:
    if not isinstance(raw_output, str):
        return None
    if category in {"color", "spatial"}:
        match = _DIRECT_WORD_RE.fullmatch(raw_output)
        if match is None:
            return None
        normalized = match.group(1).lower()
        allowed = _COLOR_ANSWERS if category == "color" else _SPATIAL_ANSWERS
        return normalized if normalized in allowed else None
    if category == "count":
        match = _DIRECT_DIGIT_RE.fullmatch(raw_output)
        if match is None or match.group(1) not in "12345":
            return None
        return match.group(1)
    if category == "ocr":
        match = _DIRECT_OCR_RE.fullmatch(raw_output)
        return None if match is None else match.group(1)
    raise DynamicNeoFoldError(f"Unsupported NEO category {category!r}.")


@dataclass(frozen=True, slots=True)
class NeoVisionBehaviorRecord:
    index: int
    probe_id: str
    split: str
    category: str
    prompt: str
    expected: str
    image_path: str
    image_sha256: str
    manifest_record_sha256: str
    parent_raw_output: str
    parent_normalized_prediction: str | None
    parent_direct_answer: bool
    parent_correct: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _nonnegative_int(self.index, "index"))
        for name in ("probe_id", "prompt", "expected", "image_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DynamicNeoFoldError(f"{name} must be nonempty.")
        if self.split not in _SPLITS:
            raise DynamicNeoFoldError("NEO behavior split is invalid.")
        if self.category not in _CATEGORIES:
            raise DynamicNeoFoldError("NEO behavior category is invalid.")
        _digest(self.image_sha256, "image_sha256")
        _digest(self.manifest_record_sha256, "manifest_record_sha256")
        if not isinstance(self.parent_raw_output, str):
            raise DynamicNeoFoldError("parent_raw_output must be text.")
        if self.parent_normalized_prediction is not None and not isinstance(
            self.parent_normalized_prediction, str
        ):
            raise DynamicNeoFoldError(
                "parent_normalized_prediction must be text or None."
            )
        if not isinstance(self.parent_direct_answer, bool) or not isinstance(
            self.parent_correct, bool
        ):
            raise DynamicNeoFoldError("Parent direct/correct fields must be boolean.")

    def identity(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "probe_id": self.probe_id,
            "split": self.split,
            "category": self.category,
            "prompt": self.prompt,
            "expected": self.expected,
            "image_sha256": self.image_sha256,
            "manifest_record_sha256": self.manifest_record_sha256,
            "parent_raw_output_sha256": hashlib.sha256(
                self.parent_raw_output.encode("utf-8")
            ).hexdigest(),
            "parent_normalized_prediction": self.parent_normalized_prediction,
            "parent_direct_answer": self.parent_direct_answer,
            "parent_correct": self.parent_correct,
        }


@dataclass(frozen=True, slots=True)
class NeoParentBehaviorCache:
    manifest_path: str
    outputs_path: str
    manifest_sha256: str
    outputs_sha256: str
    records: tuple[NeoVisionBehaviorRecord, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        _digest(self.manifest_sha256, "manifest_sha256")
        _digest(self.outputs_sha256, "outputs_sha256")
        _digest(self.fingerprint, "behavior cache fingerprint")
        if not self.records:
            raise DynamicNeoFoldError("NEO behavior cache must contain records.")
        if tuple(record.index for record in self.records) != tuple(
            range(len(self.records))
        ):
            raise DynamicNeoFoldError(
                "NEO behavior records must preserve exact manifest order."
            )
        split_counts = {
            split: sum(record.split == split for record in self.records)
            for split in _SPLITS
        }
        if any(count <= 0 for count in split_counts.values()):
            raise DynamicNeoFoldError(
                "NEO behavior cache must contain calibration and held-out records."
            )

    @property
    def calibration(self) -> tuple[NeoVisionBehaviorRecord, ...]:
        return tuple(record for record in self.records if record.split == "calibration")

    @property
    def heldout(self) -> tuple[NeoVisionBehaviorRecord, ...]:
        return tuple(record for record in self.records if record.split == "heldout")

    def verify(self) -> None:
        manifest = Path(self.manifest_path)
        outputs = Path(self.outputs_path)
        if not manifest.is_file() or not outputs.is_file():
            raise DynamicNeoFoldError("NEO behavior cache source files are missing.")
        if (
            _file_sha256(manifest) != self.manifest_sha256
            or _file_sha256(outputs) != self.outputs_sha256
        ):
            raise DynamicNeoFoldError(
                "NEO behavior cache source files changed after authentication."
            )
        expected = canonical_sha256(
            {
                "schema": BEHAVIOR_SCHEMA,
                "manifest_sha256": self.manifest_sha256,
                "outputs_sha256": self.outputs_sha256,
                "records": [record.identity() for record in self.records],
            }
        )
        if expected != self.fingerprint:
            raise DynamicNeoFoldError("NEO behavior cache fingerprint is invalid.")


def load_neo_parent_behavior_cache(
    manifest_path: str | os.PathLike[str],
    outputs_path: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str | None = None,
    expected_outputs_sha256: str | None = None,
) -> NeoParentBehaviorCache:
    manifest = Path(manifest_path).expanduser().resolve()
    outputs = Path(outputs_path).expanduser().resolve()
    if not manifest.is_file() or not outputs.is_file():
        raise DynamicNeoFoldError("NEO manifest/output cache paths must be files.")
    manifest_sha = _file_sha256(manifest)
    outputs_sha = _file_sha256(outputs)
    if expected_manifest_sha256 is not None and manifest_sha != _digest(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    ):
        raise DynamicNeoFoldError("NEO manifest differs from the pinned digest.")
    if expected_outputs_sha256 is not None and outputs_sha != _digest(
        expected_outputs_sha256,
        "expected_outputs_sha256",
    ):
        raise DynamicNeoFoldError("NEO parent outputs differ from the pinned digest.")
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicNeoFoldError("NEO manifest is invalid JSON.") from exc
    if (
        not isinstance(manifest_payload, Mapping)
        or manifest_payload.get("schema") != PROBE_SCHEMA
        or manifest_payload.get("image_hash_basis") != IMAGE_HASH_BASIS
    ):
        raise DynamicNeoFoldError(
            "NEO manifest does not use the authenticated native-vision schema "
            "and decoded-RGB image hash basis."
        )
    source_records = manifest_payload.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise DynamicNeoFoldError("NEO manifest records are missing.")
    raw_lines = outputs.read_bytes()
    if raw_lines and not raw_lines.endswith(b"\n"):
        raise DynamicNeoFoldError("NEO parent output cache is truncated.")
    lines = raw_lines.splitlines()
    if len(lines) != len(source_records):
        raise DynamicNeoFoldError("NEO parent output count differs from the manifest.")
    records = []
    for index, (source, line) in enumerate(zip(source_records, lines, strict=True)):
        if not isinstance(source, Mapping):
            raise DynamicNeoFoldError("NEO manifest record is not an object.")
        try:
            cached = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicNeoFoldError(
                f"NEO output line {index + 1} is invalid."
            ) from exc
        if not isinstance(cached, Mapping):
            raise DynamicNeoFoldError("NEO output record is not an object.")
        record_sha = canonical_sha256(dict(source))
        raw_output = cached.get("raw_output")
        normalized = _normalize_direct_answer(source.get("category"), raw_output)
        image = (manifest.parent / str(source.get("image"))).resolve()
        try:
            image.relative_to(manifest.parent.resolve())
        except ValueError as exc:
            raise DynamicNeoFoldError(
                "NEO manifest image escapes the manifest directory."
            ) from exc
        image_sha = source.get("image_sha256")
        if not image.is_file() or _decoded_rgb_sha256(image) != image_sha:
            raise DynamicNeoFoldError(
                f"NEO image authentication failed for record {index}."
            )
        expected = source.get("answer")
        required = {
            "schema": BEHAVIOR_SCHEMA,
            "index": index,
            "probe_id": source.get("probe_id"),
            "record_sha256": record_sha,
            "category": source.get("category"),
            "split": source.get("split"),
            "expected": expected,
            "normalized_prediction": normalized,
            "direct_answer": normalized is not None,
            "correct": normalized == expected,
        }
        if any(cached.get(key) != value for key, value in required.items()):
            raise DynamicNeoFoldError(
                f"NEO cached parent record {index} failed authentication."
            )
        if (
            cached.get("raw_output_sha256")
            != hashlib.sha256(str(raw_output).encode("utf-8")).hexdigest()
        ):
            raise DynamicNeoFoldError(
                f"NEO cached parent output hash failed for record {index}."
            )
        records.append(
            NeoVisionBehaviorRecord(
                index=index,
                probe_id=str(source["probe_id"]),
                split=str(source["split"]),
                category=str(source["category"]),
                prompt=str(source["prompt"]),
                expected=str(expected),
                image_path=str(image),
                image_sha256=str(image_sha),
                manifest_record_sha256=record_sha,
                parent_raw_output=str(raw_output),
                parent_normalized_prediction=normalized,
                parent_direct_answer=normalized is not None,
                parent_correct=normalized == expected,
            )
        )
    fingerprint = canonical_sha256(
        {
            "schema": BEHAVIOR_SCHEMA,
            "manifest_sha256": manifest_sha,
            "outputs_sha256": outputs_sha,
            "records": [record.identity() for record in records],
        }
    )
    result = NeoParentBehaviorCache(
        manifest_path=str(manifest),
        outputs_path=str(outputs),
        manifest_sha256=manifest_sha,
        outputs_sha256=outputs_sha,
        records=tuple(records),
        fingerprint=fingerprint,
    )
    result.verify()
    return result


@dataclass(frozen=True, slots=True)
class NeoLayerFoldSelection:
    layer_index: int
    source_neurons: int
    target_neurons: int
    removed_neuron: int
    anchor_neuron: int
    anchor_target_index: int
    compensation_beta: float
    activation_correlation: float
    relative_reconstruction_error: float
    residual_endpoint_relative_rmse: float
    removed_importance: float
    trace_rows: int
    candidate_count: int
    anchor_neurons: tuple[int, ...] = ()
    anchor_target_indices: tuple[int, ...] = ()
    compensation_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        normalize_sparse_anchor_selection(self, vocabulary=_SELECTION_VOCABULARY)

    @property
    def survivors(self) -> tuple[int, ...]:
        return survivors_for(self.source_neurons, self.removed_neuron)

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_neurons)

    @property
    def compensation_terms(self) -> tuple[tuple[int, int, float], ...]:
        return sparse_compensation_terms(
            self.anchor_neurons,
            self.anchor_target_indices,
            self.compensation_coefficients,
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "layer_index": self.layer_index,
            "source_neurons": self.source_neurons,
            "target_neurons": self.target_neurons,
            "removed_neuron": self.removed_neuron,
            "anchor_neuron": self.anchor_neuron,
            "anchor_target_index": self.anchor_target_index,
            "compensation_beta": self.compensation_beta,
            "activation_correlation": self.activation_correlation,
            "relative_reconstruction_error": self.relative_reconstruction_error,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "removed_importance": self.removed_importance,
            "trace_rows": self.trace_rows,
            "candidate_count": self.candidate_count,
            "full_source_coverage": True,
            "attention_copied_exactly": True,
            "zero_gradient": True,
        }
        if self.anchor_count > 1:
            result.update(
                {
                    "anchor_neurons": list(self.anchor_neurons),
                    "anchor_target_indices": list(self.anchor_target_indices),
                    "compensation_coefficients": list(self.compensation_coefficients),
                }
            )
        return result


def _sparse_anchor_regression(
    sketch: Any,
    *,
    removed: int,
    anchor_count: int,
    column_norms: Any,
    epsilon: Any,
) -> tuple[tuple[int, ...], tuple[float, ...], float, float]:
    """Fit one removed activation from a small survivor anchor set."""

    return sparse_anchor_regression(
        sketch,
        removed=removed,
        anchor_count=anchor_count,
        column_norms=column_norms,
        epsilon=epsilon,
        vocabulary=_SELECTION_VOCABULARY,
    )


def _selection_from_statistics(
    layer_index: int,
    statistics: Mapping[str, Any],
    *,
    trace_rows: int,
    candidate_count: int,
    anchor_count: int = 1,
) -> NeoLayerFoldSelection:
    return selection_from_statistics(
        layer_index,
        statistics,
        trace_rows=trace_rows,
        candidate_count=candidate_count,
        anchor_count=anchor_count,
        vocabulary=_SELECTION_VOCABULARY,
        selection_factory=NeoLayerFoldSelection,
    )


class NeoNativeVisionFoldCache:
    """Immutable selected fold evidence backed by sealed trace statistics."""

    def __init__(
        self,
        *,
        parent_graph_fingerprint: str,
        parent_state_fingerprint: str,
        behavior_cache: NeoParentBehaviorCache,
        selections: Sequence[NeoLayerFoldSelection],
        statistics: Mapping[str, Any],
        sketch_dim: int,
        candidate_count: int,
        sample_runner_calls: int,
        cache_key: str | None = None,
        directory: str | None = None,
        source_cache_key: str | None = None,
    ) -> None:
        self.parent_graph_fingerprint = _digest(
            parent_graph_fingerprint,
            "parent_graph_fingerprint",
        )
        self.parent_state_fingerprint = _digest(
            parent_state_fingerprint,
            "parent_state_fingerprint",
        )
        if not isinstance(behavior_cache, NeoParentBehaviorCache):
            raise DynamicNeoFoldError("behavior_cache must be NeoParentBehaviorCache.")
        behavior_cache.verify()
        self.behavior_cache = behavior_cache
        self.selections = tuple(selections)
        if not self.selections or tuple(
            selection.layer_index for selection in self.selections
        ) != tuple(range(len(self.selections))):
            raise DynamicNeoFoldError(
                "NEO fold selections must cover contiguous decoder layers."
            )
        anchor_counts = {selection.anchor_count for selection in self.selections}
        if len(anchor_counts) != 1:
            raise DynamicNeoFoldError(
                "NEO fold selections use inconsistent sparse anchor counts."
            )
        self.anchor_count = next(iter(anchor_counts))
        self.statistics = MappingProxyType(
            {
                str(name): value.detach().to(device="cpu").contiguous()
                for name, value in statistics.items()
            }
        )
        self.sketch_dim = _positive_int(sketch_dim, "sketch_dim")
        self.candidate_count = _positive_int(
            candidate_count,
            "candidate_count",
        )
        self.sample_runner_calls = _positive_int(
            sample_runner_calls,
            "sample_runner_calls",
        )
        self.directory = directory
        self.source_cache_key = (
            None
            if source_cache_key is None
            else _digest(source_cache_key, "source_cache_key")
        )
        if self.source_cache_key is not None and self.source_cache_key == cache_key:
            raise DynamicNeoFoldError(
                "A derived NEO cache cannot cite itself as its source."
            )
        self.cache_key = (
            self._computed_cache_key()
            if cache_key is None
            else _digest(cache_key, "cache_key")
        )
        self.verify()

    @property
    def fingerprint(self) -> str:
        return self.cache_key

    @staticmethod
    def statistic_key(layer_index: int, name: str) -> str:
        if name not in _STATISTIC_NAMES:
            raise DynamicNeoFoldError("Unknown NEO statistic name.")
        return f"layer_{layer_index:03d}_{name}"

    def layer_statistics(self, layer_index: int) -> Mapping[str, Any]:
        index = _nonnegative_int(layer_index, "layer_index")
        if index >= len(self.selections):
            raise DynamicNeoFoldError("NEO cache layer index is out of range.")
        return MappingProxyType(
            {
                name: self.statistics[self.statistic_key(index, name)]
                for name in _STATISTIC_NAMES
            }
        )

    def selection(self, layer_index: int) -> NeoLayerFoldSelection:
        index = _nonnegative_int(layer_index, "layer_index")
        try:
            return self.selections[index]
        except IndexError as exc:
            raise DynamicNeoFoldError("NEO cache layer index is out of range.") from exc

    def _body(self) -> dict[str, Any]:
        body = {
            "schema": FOLD_CACHE_SCHEMA,
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "parent_state_fingerprint": self.parent_state_fingerprint,
            "behavior_cache_fingerprint": self.behavior_cache.fingerprint,
            "behavior_manifest_sha256": self.behavior_cache.manifest_sha256,
            "parent_outputs_sha256": self.behavior_cache.outputs_sha256,
            "calibration_probe_ids": [
                record.probe_id for record in self.behavior_cache.calibration
            ],
            "heldout_probe_ids": [
                record.probe_id for record in self.behavior_cache.heldout
            ],
            "selection_used_splits": ["calibration"],
            "selection_excluded_splits": ["heldout"],
            "attention_conditioning": (
                "post_native_spatial_attention_residual_then_swiglu"
            ),
            "residual_endpoint_source": (
                "decoder_layer_post_attention_post_mlp_hidden_state"
            ),
            "sketch_dim": self.sketch_dim,
            "candidate_count": self.candidate_count,
            "sample_runner_calls": self.sample_runner_calls,
            "parent_outputs_regenerated": False,
            "selections": [selection.as_dict() for selection in self.selections],
            "statistics": {
                name: _tensor_descriptor(value)
                for name, value in sorted(self.statistics.items())
            },
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
        if self.anchor_count > 1:
            body["anchor_count"] = self.anchor_count
            body["selection_policy"] = (
                "heldout_blind_low_importance_sparse_anchor_regression"
            )
        if self.source_cache_key is not None:
            body["source_cache_key"] = self.source_cache_key
            body["statistics_reused_without_parent_forward"] = True
        return body

    def _computed_cache_key(self) -> str:
        return canonical_sha256(self._body())

    def verify(
        self,
        *,
        parent_model: Any | None = None,
        parent_graph: ModelCircuitGraph | None = None,
    ) -> None:
        self.behavior_cache.verify()
        expected_names = {
            self.statistic_key(index, name)
            for index in range(len(self.selections))
            for name in _STATISTIC_NAMES
        }
        if set(self.statistics) != expected_names:
            raise DynamicNeoFoldError(
                "NEO cache statistics do not cover every selected layer."
            )
        for selection in self.selections:
            rebuilt = _selection_from_statistics(
                selection.layer_index,
                self.layer_statistics(selection.layer_index),
                trace_rows=selection.trace_rows,
                candidate_count=self.candidate_count,
                anchor_count=self.anchor_count,
            )
            if rebuilt != selection:
                raise DynamicNeoFoldError(
                    f"NEO layer {selection.layer_index} selection is not "
                    "reproducible from sealed statistics."
                )
        if self._computed_cache_key() != self.cache_key:
            raise DynamicNeoFoldError("NEO fold cache seal is invalid.")
        if parent_graph is not None and (
            parent_graph.fingerprint != self.parent_graph_fingerprint
            or parent_graph.family != "neo_chat"
            or len(parent_graph.layer_indices) != len(self.selections)
        ):
            raise DynamicNeoFoldError(
                "Loaded NEO parent graph differs from the fold cache."
            )
        if parent_model is not None and (
            fingerprint_model_state(parent_model) != self.parent_state_fingerprint
        ):
            raise DynamicNeoFoldError(
                "Loaded NEO parent state differs from the fold cache."
            )

    def save(self, destination: str | os.PathLike[str]) -> Path:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise DynamicNeoFoldError(
                "safetensors is required to publish NEO fold evidence."
            ) from exc
        destination_path = Path(destination).expanduser().absolute()
        if destination_path.exists():
            raise DynamicNeoFoldError(
                f"NEO fold cache destination exists: {destination_path}."
            )
        parent = destination_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=".neo-fold-cache-stage-", dir=parent)
        ).absolute()
        try:
            statistics_path = stage / "statistics.safetensors"
            save_file(dict(self.statistics), str(statistics_path))
            body = self._body()
            body["statistics_file"] = statistics_path.name
            body["statistics_file_sha256"] = _file_sha256(statistics_path)
            body["cache_key"] = self.cache_key
            manifest_sha256 = canonical_sha256(body)
            _write_exclusive_json(
                stage / "manifest.json",
                {**body, "manifest_sha256": manifest_sha256},
            )
            stage.rename(destination_path)
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        return destination_path

    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        *,
        behavior_cache: NeoParentBehaviorCache,
    ) -> "NeoNativeVisionFoldCache":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise DynamicNeoFoldError(
                "safetensors is required to open NEO fold evidence."
            ) from exc
        root = Path(directory).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DynamicNeoFoldError("NEO fold cache manifest is missing.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicNeoFoldError("NEO fold cache manifest is invalid.") from exc
        manifest_sha256 = manifest.pop("manifest_sha256", None)
        if canonical_sha256(manifest) != manifest_sha256:
            raise DynamicNeoFoldError("NEO fold cache manifest seal is invalid.")
        cache_key = manifest.get("cache_key")
        _digest(cache_key, "cache_key")
        if (
            manifest.get("schema") != FOLD_CACHE_SCHEMA
            or manifest.get("behavior_cache_fingerprint") != behavior_cache.fingerprint
        ):
            raise DynamicNeoFoldError(
                "NEO fold cache is bound to different behavior evidence."
            )
        statistics_path = root / str(manifest.get("statistics_file"))
        if not statistics_path.is_file() or _file_sha256(
            statistics_path
        ) != manifest.get("statistics_file_sha256"):
            raise DynamicNeoFoldError("NEO fold statistics file is invalid.")
        statistics = load_file(str(statistics_path), device="cpu")
        descriptors = manifest.get("statistics")
        if not isinstance(descriptors, Mapping) or set(descriptors) != set(statistics):
            raise DynamicNeoFoldError("NEO fold statistic descriptors are incomplete.")
        if any(
            _tensor_descriptor(statistics[name]) != descriptors[name]
            for name in statistics
        ):
            raise DynamicNeoFoldError("NEO fold statistic tensor seal is invalid.")
        selections = tuple(
            NeoLayerFoldSelection(
                **{
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "full_source_coverage",
                        "attention_copied_exactly",
                        "zero_gradient",
                    }
                }
            )
            for item in manifest["selections"]
        )
        result = cls(
            parent_graph_fingerprint=manifest["parent_graph_fingerprint"],
            parent_state_fingerprint=manifest["parent_state_fingerprint"],
            behavior_cache=behavior_cache,
            selections=selections,
            statistics=statistics,
            sketch_dim=manifest["sketch_dim"],
            candidate_count=manifest["candidate_count"],
            sample_runner_calls=manifest["sample_runner_calls"],
            cache_key=cache_key,
            directory=str(root),
            source_cache_key=manifest.get("source_cache_key"),
        )
        # The in-memory seal deliberately excludes storage-container fields.
        if result.cache_key != result._computed_cache_key():
            raise DynamicNeoFoldError(
                "NEO fold cache logical seal differs after reload."
            )
        return result


def derive_neo_native_vision_fold_cache(
    source_cache: NeoNativeVisionFoldCache,
    *,
    destination: str | os.PathLike[str],
    candidate_count: int,
    anchor_count: int,
) -> NeoNativeVisionFoldCache:
    """Derive a sealed heldout-blind portfolio arm without new parent traces."""

    if not isinstance(source_cache, NeoNativeVisionFoldCache):
        raise DynamicNeoFoldError("source_cache must be NeoNativeVisionFoldCache.")
    source_cache.verify()
    candidate_count = _positive_int(candidate_count, "candidate_count")
    anchor_count = _positive_int(anchor_count, "anchor_count")
    selections = tuple(
        _selection_from_statistics(
            layer_index,
            source_cache.layer_statistics(layer_index),
            trace_rows=source_cache.selection(layer_index).trace_rows,
            candidate_count=candidate_count,
            anchor_count=anchor_count,
        )
        for layer_index in range(len(source_cache.selections))
    )
    derived = NeoNativeVisionFoldCache(
        parent_graph_fingerprint=source_cache.parent_graph_fingerprint,
        parent_state_fingerprint=source_cache.parent_state_fingerprint,
        behavior_cache=source_cache.behavior_cache,
        selections=selections,
        statistics=source_cache.statistics,
        sketch_dim=source_cache.sketch_dim,
        candidate_count=candidate_count,
        sample_runner_calls=source_cache.sample_runner_calls,
        source_cache_key=source_cache.cache_key,
    )
    derived.save(destination)
    return NeoNativeVisionFoldCache.open(
        destination,
        behavior_cache=source_cache.behavior_cache,
    )


class _TraceAccumulator:
    def __init__(self, layer_index: int, sketch_dim: int) -> None:
        self.layer_index = layer_index
        self.sketch_dim = sketch_dim
        self.sum_abs: Any | None = None
        self.sum_square: Any | None = None
        self.sketch: Any | None = None
        self.rows = 0
        self.residual_endpoint_sum_square_l2: Any | None = None
        self.residual_endpoint_rows = 0

    def add(self, value: Any) -> None:
        import torch

        rows = value.detach().reshape(-1, value.shape[-1]).to(dtype=torch.float32)
        if rows.numel() == 0:
            return
        width = int(rows.shape[1])
        if self.sum_abs is None:
            self.sum_abs = torch.zeros(width, device=rows.device, dtype=torch.float32)
            self.sum_square = torch.zeros(
                width,
                device=rows.device,
                dtype=torch.float32,
            )
            self.sketch = torch.zeros(
                self.sketch_dim,
                width,
                device=rows.device,
                dtype=torch.float32,
            )
        elif int(self.sum_abs.shape[0]) != width:
            raise DynamicNeoFoldError(
                "NEO trace width changed within one decoder layer."
            )
        values = rows
        self.sum_abs.add_(values.abs().sum(dim=0))
        self.sum_square.add_((values * values).sum(dim=0))
        row_ids = torch.arange(
            self.rows,
            self.rows + int(values.shape[0]),
            device=values.device,
            dtype=torch.int64,
        )
        buckets = row_ids.remainder(self.sketch_dim)
        parity = (
            row_ids * 1_103_515_245 + (self.layer_index + 1) * 12_345
        ).bitwise_and(1)
        signs = torch.where(
            parity == 0,
            torch.ones_like(row_ids, dtype=torch.float32),
            -torch.ones_like(row_ids, dtype=torch.float32),
        )
        self.sketch.index_add_(0, buckets, values * signs.unsqueeze(1))
        self.rows += int(values.shape[0])

    def add_residual_endpoint(self, value: Any) -> None:
        import torch

        if isinstance(value, (tuple, list)):
            if not value:
                raise DynamicNeoFoldError(
                    "NEO decoder layer returned an empty endpoint tuple."
                )
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise DynamicNeoFoldError(
                "NEO decoder layer returned a non-tensor residual endpoint."
            )
        rows = value.detach().reshape(-1, value.shape[-1]).to(dtype=torch.float32)
        if rows.numel() == 0:
            return
        endpoint_energy = (rows * rows).sum()
        if self.residual_endpoint_sum_square_l2 is None:
            self.residual_endpoint_sum_square_l2 = torch.zeros(
                (),
                device=rows.device,
                dtype=torch.float32,
            )
        self.residual_endpoint_sum_square_l2.add_(endpoint_energy)
        self.residual_endpoint_rows += int(rows.shape[0])

    def finalize(self, down_weight: Any) -> Mapping[str, Any]:
        import torch

        if (
            self.rows <= 0
            or self.sum_abs is None
            or self.sum_square is None
            or self.sketch is None
            or self.residual_endpoint_rows <= 0
            or self.residual_endpoint_sum_square_l2 is None
            or float(self.residual_endpoint_sum_square_l2.item()) <= 0.0
        ):
            raise DynamicNeoFoldError(
                f"NEO layer {self.layer_index} captured incomplete activation/"
                "residual endpoint rows."
            )
        down = down_weight.detach()
        if down.ndim != 2 or int(down.shape[1]) != int(self.sum_abs.shape[0]):
            raise DynamicNeoFoldError(
                "NEO down projection differs from captured trace width."
            )
        return MappingProxyType(
            {
                "mean_abs": (self.sum_abs / self.rows)
                .to(device="cpu", dtype=torch.float32)
                .contiguous(),
                "mean_square": (self.sum_square / self.rows)
                .to(device="cpu", dtype=torch.float32)
                .contiguous(),
                "sketch": self.sketch.to(
                    device="cpu",
                    dtype=torch.float32,
                ).contiguous(),
                "down_norm": torch.linalg.vector_norm(
                    down.to(dtype=torch.float32),
                    dim=0,
                )
                .to(device="cpu")
                .contiguous(),
                "residual_endpoint_mean_square_l2": torch.tensor(
                    float(self.residual_endpoint_sum_square_l2.item())
                    / self.residual_endpoint_rows,
                    dtype=torch.float32,
                ),
            }
        )


def _neo_decoder_layers(model: Any) -> tuple[Any, ...]:
    try:
        layers = tuple(model.language_model.model.layers)
    except (AttributeError, TypeError) as exc:
        raise DynamicNeoFoldError(
            "NEO model does not expose language_model.model.layers."
        ) from exc
    if not layers:
        raise DynamicNeoFoldError("NEO decoder has no layers.")
    return layers


def build_neo_native_vision_fold_cache(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    behavior_cache: NeoParentBehaviorCache,
    *,
    trace_runner: Callable[[Any, NeoVisionBehaviorRecord], Any],
    destination: str | os.PathLike[str],
    sketch_dim: int = 64,
    candidate_count: int = 8,
) -> NeoNativeVisionFoldCache:
    """Capture image-conditioned calibration traces once and seal selections."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise DynamicNeoFoldError("PyTorch is required for NEO trace capture.") from exc
    if parent_graph.family != "neo_chat":
        raise DynamicNeoFoldError("NEO trace capture requires a neo_chat graph.")
    if not callable(trace_runner):
        raise DynamicNeoFoldError("trace_runner must be callable.")
    behavior_cache.verify()
    sketch_dim = _positive_int(sketch_dim, "sketch_dim")
    candidate_count = _positive_int(candidate_count, "candidate_count")
    layers = _neo_decoder_layers(parent_model)
    if len(layers) != len(parent_graph.layer_indices):
        raise DynamicNeoFoldError("NEO parent graph and runtime decoder depth differ.")
    if any(parameter.grad is not None for parameter in parent_model.parameters()):
        raise DynamicNeoFoldError("NEO trace capture requires a zero-gradient parent.")
    state_before = fingerprint_model_state(parent_model)
    runtime_before = fingerprint_model_runtime_state(parent_model)
    module_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    accumulators = [
        _TraceAccumulator(index, sketch_dim) for index in range(len(layers))
    ]
    handles = []
    for index, layer in enumerate(layers):
        down = getattr(getattr(layer, "mlp", None), "down_proj", None)
        if down is None or not callable(
            getattr(down, "register_forward_pre_hook", None)
        ):
            raise DynamicNeoFoldError(
                f"NEO layer {index} lacks an observable MLP down projection."
            )

        def capture(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer_index: int = index,
        ) -> None:
            if not args:
                raise DynamicNeoFoldError("NEO down projection hook saw no input.")
            accumulators[layer_index].add(args[0])

        handles.append(down.register_forward_pre_hook(capture))

        def capture_residual_endpoint(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer_index: int = index,
        ) -> None:
            accumulators[layer_index].add_residual_endpoint(output)

        handles.append(layer.register_forward_hook(capture_residual_endpoint))
    calls = 0
    try:
        parent_model.eval()
        with torch.no_grad():
            for record in behavior_cache.calibration:
                trace_runner(parent_model, record)
                calls += 1
    finally:
        for handle in handles:
            handle.remove()
        for module, training in module_modes:
            module.train(training)
    if (
        fingerprint_model_state(parent_model) != state_before
        or fingerprint_model_runtime_state(parent_model) != runtime_before
    ):
        raise DynamicNeoFoldError("NEO trace capture mutated the parent model.")
    statistics: dict[str, Any] = {}
    selections = []
    for index, (layer, accumulator) in enumerate(
        zip(layers, accumulators, strict=True)
    ):
        values = accumulator.finalize(layer.mlp.down_proj.weight)
        for name, tensor in values.items():
            statistics[NeoNativeVisionFoldCache.statistic_key(index, name)] = tensor
        selections.append(
            _selection_from_statistics(
                index,
                values,
                trace_rows=accumulator.rows,
                candidate_count=candidate_count,
            )
        )
    cache = NeoNativeVisionFoldCache(
        parent_graph_fingerprint=parent_graph.fingerprint,
        parent_state_fingerprint=state_before,
        behavior_cache=behavior_cache,
        selections=tuple(selections),
        statistics=statistics,
        sketch_dim=sketch_dim,
        candidate_count=candidate_count,
        sample_runner_calls=calls,
    )
    path = cache.save(destination)
    return NeoNativeVisionFoldCache.open(
        path,
        behavior_cache=behavior_cache,
    )


class NeoNativeVisionEvidenceProvider:
    """Expose one immutable calibration cache without any live parent forward."""

    def __init__(self, cache: NeoNativeVisionFoldCache) -> None:
        if not isinstance(cache, NeoNativeVisionFoldCache):
            raise DynamicNeoFoldError("cache must be NeoNativeVisionFoldCache.")
        cache.verify()
        self.cache = cache
        self.prepare_count = 0
        self.refresh_count = 0
        self.restore_count = 0

    @property
    def base_fingerprint(self) -> str:
        return self.cache.cache_key

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        del context
        self.prepare_count += 1
        self.cache.verify()
        if action.method_id != METHOD_ID:
            raise DynamicNeoFoldError(
                "NEO evidence provider received a different fold method."
            )
        return FoldEvidence(
            self.cache.cache_key,
            payload=self.cache,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=0,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        self.cache.verify()
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del action_ids
        if context.child_state_fingerprint != _digest(
            state_fingerprint,
            "restored child state fingerprint",
        ):
            raise DynamicNeoFoldError(
                "NEO evidence restore observed an inexact child rollback."
            )
        self.cache.verify()
        self.restore_count += 1


class NeoNativeVisionFoldOnlyInitializer:
    """Authenticate cache/model binding without mutating the seeded blank."""

    def __init__(self, cache: NeoNativeVisionFoldCache) -> None:
        self.cache = cache

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        if len(plan.actions) != 1 or plan.actions[0].method_id != METHOD_ID:
            raise DynamicNeoFoldError(
                "NEO fold-only initialization requires one atomic model action."
            )
        if len(context.parent_models) != 1:
            raise DynamicNeoFoldError("NEO fold cache is bound to exactly one parent.")
        parent_name = next(iter(context.parent_models))
        parent = context.parent_models[parent_name]
        parent_graph = context.parent_graphs[parent_name]
        self.cache.verify(parent_model=parent, parent_graph=parent_graph)
        if plan.evidence_fingerprint != self.cache.cache_key:
            raise DynamicNeoFoldError(
                "NEO plan evidence differs from the authenticated cache."
            )
        return {
            "kind": "neo_native_vision_fold_only",
            "child_state_fingerprint": context.child_state_fingerprint,
            "cache_key": self.cache.cache_key,
            "calibration_samples": len(self.cache.behavior_cache.calibration),
            "heldout_samples_used_for_selection": 0,
            "parent_outputs_regenerated": False,
            "parent_forward_count": 0,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def _live_state(model: Any) -> Mapping[str, Any]:
    try:
        values = model.state_dict(keep_vars=True)
    except TypeError:
        values = model.state_dict()
    return values


class NeoNativeVisionFoldTransaction:
    method_id = METHOD_ID

    def __init__(
        self,
        *,
        context: FoldExecutionContext,
        snapshot: Any,
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
        cache_key: str,
        changed_paths: tuple[str, ...],
        selections: tuple[NeoLayerFoldSelection, ...],
    ) -> None:
        self.context = context
        self.snapshot = snapshot
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = candidate_state_fingerprint
        self.cache_key = cache_key
        self.changed_paths = changed_paths
        self.selections = selections
        self.active = True
        self.verified_noop = False

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicNeoFoldError("NEO transaction is no longer active.")
        if self.context.child_state_fingerprint != self.candidate_state_fingerprint:
            raise DynamicNeoFoldError(
                "NEO candidate changed before transaction commit."
            )
        self.snapshot.cleanup()
        self.active = False
        return {
            "kind": "neo_native_vision_atomic_model_fold",
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": self.candidate_state_fingerprint,
            "cache_key": self.cache_key,
            "folded_layers": len(self.selections),
            "removed_neurons": [
                selection.removed_neuron for selection in self.selections
            ],
            "anchor_neurons": [
                selection.anchor_neuron for selection in self.selections
            ],
            "anchor_count": self.selections[0].anchor_count,
            "sparse_anchor_neuron_sets": [
                list(selection.anchor_neurons) for selection in self.selections
            ],
            "changed_paths": list(self.changed_paths),
            "attention_copied_exactly": True,
            "vision_tower_copied_exactly": True,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicNeoFoldError("NEO transaction is no longer active.")
        self.snapshot.restore(_live_state(self.context.child_model))
        restored = self.context.child_state_fingerprint
        exact = restored == self.incumbent_state_fingerprint
        self.snapshot.cleanup()
        self.active = False
        if not exact:
            raise DynamicNeoFoldError(
                "NEO transaction rollback did not restore the incumbent."
            )
        return {
            "kind": "neo_native_vision_atomic_model_fold",
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": restored,
            "cache_key": self.cache_key,
            "rollback_exact": True,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
        }


class NeoNativeVisionSwiGLUFoldOperator:
    """Install all parent state plus one compensated MLP removal per layer."""

    method_id = METHOD_ID

    def __init__(
        self,
        *,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(self.snapshot_config, DynamicFoldSnapshotConfig):
            raise DynamicNeoFoldError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> NeoNativeVisionFoldTransaction:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise DynamicNeoFoldError("PyTorch is required for NEO folding.") from exc
        if action.method_id != self.method_id or len(action.sources) != 1:
            raise DynamicNeoFoldError(
                "NEO operator requires one atomic native-vision source action."
            )
        if not isinstance(evidence.payload, NeoNativeVisionFoldCache):
            raise DynamicNeoFoldError("NEO operator received the wrong evidence.")
        cache = evidence.payload
        parent_name = action.sources[0].parent_name
        if set(context.parent_models) != {parent_name}:
            raise DynamicNeoFoldError(
                "NEO operator source differs from the loaded parent."
            )
        parent = context.parent_models[parent_name]
        parent_graph = context.parent_graphs[parent_name]
        cache.verify(parent_model=parent, parent_graph=parent_graph)
        if cache.cache_key != evidence.fingerprint:
            raise DynamicNeoFoldError("NEO evidence payload and fingerprint differ.")
        if set(action.target_parameter_paths) != set(
            context.child_graph.parameter_paths
        ):
            raise DynamicNeoFoldError(
                "NEO atomic action does not own every child parameter."
            )
        source_layers = _neo_decoder_layers(parent)
        target_layers = _neo_decoder_layers(context.child_model)
        if len(source_layers) != len(target_layers) or len(source_layers) != len(
            cache.selections
        ):
            raise DynamicNeoFoldError(
                "NEO parent, child, and cache decoder depths differ."
            )
        incumbent = context.child_state_fingerprint
        child_state = _live_state(context.child_model)
        snapshot = capture_model_state(
            tuple(child_state.items()),
            self.snapshot_config,
        )
        source_state = parent.state_dict()
        if set(source_state) != set(child_state):
            snapshot.cleanup()
            raise DynamicNeoFoldError("NEO parent and child state keys differ.")
        overrides: dict[str, Any] = {}
        for selection, _source_layer, _target_layer in zip(
            cache.selections,
            source_layers,
            target_layers,
            strict=True,
        ):
            prefix = f"language_model.model.layers.{selection.layer_index}.mlp"
            gate_key = f"{prefix}.gate_proj.weight"
            up_key = f"{prefix}.up_proj.weight"
            down_key = f"{prefix}.down_proj.weight"
            for key in (gate_key, up_key, down_key):
                if key not in source_state or key not in child_state:
                    snapshot.cleanup()
                    raise DynamicNeoFoldError(
                        f"NEO state is missing folded tensor {key!r}."
                    )
            source_gate = source_state[gate_key]
            source_up = source_state[up_key]
            source_down = source_state[down_key]
            target_gate = child_state[gate_key]
            target_up = child_state[up_key]
            target_down = child_state[down_key]
            if (
                tuple(source_gate.shape)
                != (selection.source_neurons, source_gate.shape[1])
                or tuple(source_up.shape) != tuple(source_gate.shape)
                or tuple(source_down.shape)[1] != selection.source_neurons
                or tuple(target_gate.shape)
                != (selection.target_neurons, source_gate.shape[1])
                or tuple(target_up.shape) != tuple(target_gate.shape)
                or tuple(target_down.shape)
                != (source_down.shape[0], selection.target_neurons)
            ):
                snapshot.cleanup()
                raise DynamicNeoFoldError(
                    f"NEO layer {selection.layer_index} MLP shapes differ from "
                    "the cached one-neuron target."
                )
            survivors = torch.tensor(
                selection.survivors,
                device=source_gate.device,
                dtype=torch.long,
            )
            folded_gate = source_gate.index_select(0, survivors)
            folded_up = source_up.index_select(0, survivors)
            folded_down = source_down.index_select(1, survivors).clone()
            for (
                _anchor_neuron,
                anchor_target_index,
                coefficient,
            ) in selection.compensation_terms:
                folded_down[:, anchor_target_index].add_(
                    source_down[:, selection.removed_neuron],
                    alpha=coefficient,
                )
            overrides[gate_key] = folded_gate
            overrides[up_key] = folded_up
            overrides[down_key] = folded_down
        try:
            with torch.no_grad():
                for name, target in child_state.items():
                    source = overrides.get(name, source_state[name])
                    if tuple(source.shape) != tuple(target.shape):
                        raise DynamicNeoFoldError(
                            f"NEO source/target state shape differs for {name!r}."
                        )
                    if source.dtype != target.dtype:
                        raise DynamicNeoFoldError(
                            f"NEO source/target dtype differs for {name!r}."
                        )
                    target.copy_(source.to(device=target.device))
            candidate = context.child_state_fingerprint
        except BaseException:
            try:
                snapshot.restore(_live_state(context.child_model))
            finally:
                snapshot.cleanup()
            raise
        changed_paths = tuple(
            sorted(
                name for name in child_state if name in action.target_parameter_paths
            )
        )
        if candidate == incumbent:
            snapshot.restore(_live_state(context.child_model))
            snapshot.cleanup()
            raise DynamicNeoFoldError(
                "NEO fold installed no parent-derived candidate state."
            )
        return NeoNativeVisionFoldTransaction(
            context=context,
            snapshot=snapshot,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            cache_key=cache.cache_key,
            changed_paths=changed_paths,
            selections=cache.selections,
        )


class NeoNativeVisionFoldAdmissionEvaluator:
    """Admit the structurally complete cached transaction; behavior is final."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        if (
            context.action.method_id != METHOD_ID
            or not isinstance(
                context.transaction,
                NeoNativeVisionFoldTransaction,
            )
            or not isinstance(
                context.evidence.payload,
                NeoNativeVisionFoldCache,
            )
        ):
            return FoldDecision.reject("NEO transaction/evidence type mismatch.")
        transaction = context.transaction
        cache = context.evidence.payload
        if (
            transaction.candidate_state_fingerprint
            != context.candidate_state_fingerprint
            or transaction.cache_key != cache.cache_key
            or len(transaction.selections) != len(cache.selections)
        ):
            return FoldDecision.reject("NEO cached transaction binding mismatch.")
        return FoldDecision.accept(
            "Native NEO state is fully parent-derived with exact spatial "
            "attention copy and one cached coupled-SwiGLU merge per layer; "
            "whole-image behavior remains mandatory.",
            {
                "folded_layers": len(cache.selections),
                "removed_parameters": (
                    len(cache.selections)
                    * 3
                    * int(
                        context.execution_context.child_model.config.llm_config.hidden_size
                    )
                ),
                "parent_forward_count": context.evidence.parent_forward_count,
                "heldout_selection_count": 0,
                "attention_copied_exactly": True,
            },
        )


@dataclass(frozen=True, slots=True)
class NeoVisionBehaviorGateConfig:
    require_all_parent_correct_retained: bool = True
    require_no_split_accuracy_regression: bool = True
    require_no_split_category_regression: bool = True
    require_no_direct_answer_regression: bool = True
    require_parent_correct_task_coverage: bool = True
    required_splits: tuple[str, ...] = _SPLITS
    required_categories: tuple[str, ...] = _CATEGORIES
    require_split_category_parent_correct_coverage: bool = True

    def __post_init__(self) -> None:
        for name in (
            "require_all_parent_correct_retained",
            "require_no_split_accuracy_regression",
            "require_no_split_category_regression",
            "require_no_direct_answer_regression",
            "require_parent_correct_task_coverage",
            "require_split_category_parent_correct_coverage",
        ):
            if not isinstance(getattr(self, name), bool):
                raise DynamicNeoFoldError(f"{name} must be boolean.")
        for name, allowed in (
            ("required_splits", frozenset(_SPLITS)),
            ("required_categories", frozenset(_CATEGORIES)),
        ):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise DynamicNeoFoldError(f"{name} must be nonempty and unique.")
            if any(value not in allowed for value in values):
                raise DynamicNeoFoldError(f"{name} contains an unsupported value.")
            object.__setattr__(self, name, values)

    @property
    def coverage_policy(self) -> TaskScoreCoveragePolicy:
        if not self.require_parent_correct_task_coverage:
            return TaskScoreCoveragePolicy()
        return TaskScoreCoveragePolicy.production(
            required_splits=self.required_splits,
            required_categories=self.required_categories,
            require_split_category_cross_product=(
                self.require_split_category_parent_correct_coverage
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "require_all_parent_correct_retained": (
                self.require_all_parent_correct_retained
            ),
            "require_no_split_accuracy_regression": (
                self.require_no_split_accuracy_regression
            ),
            "require_no_split_category_regression": (
                self.require_no_split_category_regression
            ),
            "require_no_direct_answer_regression": (
                self.require_no_direct_answer_regression
            ),
            "require_parent_correct_task_coverage": (
                self.require_parent_correct_task_coverage
            ),
            "required_splits": list(self.required_splits),
            "required_categories": list(self.required_categories),
            "require_split_category_parent_correct_coverage": (
                self.require_split_category_parent_correct_coverage
            ),
        }


class NeoNativeVisionBehaviorGate:
    """Parent-relative whole-image gate using cached outputs as canonical."""

    def __init__(
        self,
        behavior_cache: NeoParentBehaviorCache,
        *,
        child_runner: Callable[[Any, NeoVisionBehaviorRecord], str],
        config: NeoVisionBehaviorGateConfig | None = None,
        child_runner_identity: BehaviorCallableIdentity | None = None,
    ) -> None:
        if not isinstance(behavior_cache, NeoParentBehaviorCache):
            raise DynamicNeoFoldError("behavior_cache must be NeoParentBehaviorCache.")
        if not callable(child_runner):
            raise DynamicNeoFoldError("child_runner must be callable.")
        self.behavior_cache = behavior_cache
        self.child_runner = child_runner
        self.config = config or NeoVisionBehaviorGateConfig()
        if not isinstance(self.config, NeoVisionBehaviorGateConfig):
            raise DynamicNeoFoldError("config must be NeoVisionBehaviorGateConfig.")
        self.last_report: Mapping[str, Any] | None = None
        self.evaluation_count = 0
        if child_runner_identity is None:
            child_runner_identity = identify_behavior_callable(
                child_runner,
                implementation_id=(
                    "neural_forge.neo.native_vision.child_record_runner"
                ),
                version=1,
            )
        if not isinstance(child_runner_identity, BehaviorCallableIdentity):
            raise DynamicNeoFoldError(
                "child_runner_identity must be BehaviorCallableIdentity."
            )
        scorer_identity = identify_behavior_callable(
            _normalize_direct_answer,
            implementation_id="neural_forge.neo.native_vision.direct_answer_scorer",
            version=1,
        )
        binding = build_behavior_gate_binding(
            gate_id=("neural_forge.neo.native_vision_parent_relative_behavior"),
            gate_version=2,
            baseline_fingerprint=behavior_cache.fingerprint,
            threshold_config=self.config.as_dict(),
            cases=[record.identity() for record in behavior_cache.records],
            baseline={
                "behavior_cache_fingerprint": behavior_cache.fingerprint,
                "manifest_sha256": behavior_cache.manifest_sha256,
                "parent_outputs_sha256": behavior_cache.outputs_sha256,
                "record_count": len(behavior_cache.records),
            },
            generator_identity=child_runner_identity,
            scorer_identity=scorer_identity,
            coverage_policy=self.config.coverage_policy.as_dict(),
            inference_execution_contract=NEO_INFERENCE_EXECUTION_CONTRACT,
        )
        expose_behavior_gate_binding(self, binding)

    @staticmethod
    def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        child_correct = sum(row["child_correct"] for row in rows)
        parent_correct = sum(row["parent_correct"] for row in rows)
        child_direct = sum(row["child_direct"] for row in rows)
        parent_direct = sum(row["parent_direct"] for row in rows)
        retained = sum(row["parent_correct"] and row["child_correct"] for row in rows)
        parent_correct_total = sum(row["parent_correct"] for row in rows)
        exact_parent_output = sum(row["exact_parent_output"] for row in rows)
        return {
            "total": total,
            "parent_correct": parent_correct,
            "child_correct": child_correct,
            "parent_direct": parent_direct,
            "child_direct": child_direct,
            "parent_correct_retained": retained,
            "parent_correct_total": parent_correct_total,
            "parent_correct_retention": (
                retained / parent_correct_total if parent_correct_total else 1.0
            ),
            "exact_parent_output": exact_parent_output,
            "exact_parent_output_rate": (exact_parent_output / total if total else 1.0),
        }

    def evaluate_model(self, model: Any) -> FoldDecision:
        self.behavior_cache.verify()
        coverage_policy = self.config.coverage_policy
        parent_correct_by_split = {
            split: sum(
                record.parent_correct and record.split == split
                for record in self.behavior_cache.records
            )
            for split in self.config.required_splits
        }
        parent_correct_by_category = {
            category: sum(
                record.parent_correct and record.category == category
                for record in self.behavior_cache.records
            )
            for category in self.config.required_categories
        }
        missing_split_coverage = tuple(
            split for split, count in parent_correct_by_split.items() if count <= 0
        )
        missing_category_coverage = tuple(
            category
            for category, count in parent_correct_by_category.items()
            if count <= 0
        )
        missing_cross_coverage = ()
        if coverage_policy.require_split_category_cross_product:
            missing_cross_coverage = tuple(
                f"{split}:{category}"
                for split in coverage_policy.required_splits
                for category in coverage_policy.required_categories
                if not any(
                    record.parent_correct
                    and record.split == split
                    and record.category == category
                    for record in self.behavior_cache.records
                )
            )
        coverage_failures = []
        if self.config.require_parent_correct_task_coverage:
            coverage_failures.extend(
                f"{split}:parent_correct_coverage" for split in missing_split_coverage
            )
            coverage_failures.extend(
                f"{category}:parent_correct_coverage"
                for category in missing_category_coverage
            )
            coverage_failures.extend(
                f"{value}:parent_correct_coverage" for value in missing_cross_coverage
            )
        if coverage_failures:
            report = {
                "schema": "neural-forge-neo-native-behavior-gate-v2",
                "behavior_cache_fingerprint": self.behavior_cache.fingerprint,
                "behavior_gate_manifest_fingerprint": (
                    self.behavior_gate_manifest_fingerprint
                ),
                "parent_outputs_sha256": self.behavior_cache.outputs_sha256,
                "coverage_policy": coverage_policy.as_dict(),
                "parent_correct_by_split": parent_correct_by_split,
                "parent_correct_by_category": parent_correct_by_category,
                "failures": sorted(set(coverage_failures)),
                "rows": [],
                "parent_forward_count": 0,
                "child_forward_suite_count": 0,
                "exact_parent_output_is_diagnostic_only": True,
            }
            self.last_report = MappingProxyType(report)
            self.evaluation_count += 1
            return FoldDecision.reject(
                "Native NEO task-score parent-correct coverage is incomplete: "
                + ", ".join(sorted(set(coverage_failures))),
                {
                    "heldout_parent_correct": parent_correct_by_split.get("heldout", 0),
                    "missing_required_split_count": len(missing_split_coverage),
                    "missing_required_category_count": len(missing_category_coverage),
                    "missing_required_split_category_count": len(
                        missing_cross_coverage
                    ),
                    "parent_forward_count": 0,
                    "child_outputs_scored": 0,
                },
            )
        rows = []
        with observe_model_execution(
            model,
            contract=NEO_INFERENCE_EXECUTION_CONTRACT,
        ) as execution_counts:
            for record in self.behavior_cache.records:
                raw_output = self.child_runner(model, record)
                if not isinstance(raw_output, str):
                    raise DynamicNeoFoldError(
                        f"Child runner returned non-text for {record.probe_id}."
                    )
                normalized = _normalize_direct_answer(record.category, raw_output)
                rows.append(
                    {
                        "probe_id": record.probe_id,
                        "split": record.split,
                        "category": record.category,
                        "expected": record.expected,
                        "parent_correct": record.parent_correct,
                        "parent_direct": record.parent_direct_answer,
                        "child_raw_output": raw_output,
                        "child_normalized_prediction": normalized,
                        "child_direct": normalized is not None,
                        "child_correct": normalized == record.expected,
                        "exact_parent_output": raw_output == record.parent_raw_output,
                    }
                )
        if not execution_contract_satisfied(execution_counts):
            raise DynamicNeoFoldError(
                "Native NEO child runner returned without executing the "
                "declared chat entrypoint and language_model root."
            )
        by_split = {
            split: self._metrics(tuple(row for row in rows if row["split"] == split))
            for split in _SPLITS
        }
        by_split_category = {
            split: {
                category: self._metrics(
                    tuple(
                        row
                        for row in rows
                        if row["split"] == split and row["category"] == category
                    )
                )
                for category in _CATEGORIES
            }
            for split in _SPLITS
        }
        failures = []
        if self.config.require_all_parent_correct_retained:
            for split, metrics in by_split.items():
                if (
                    metrics["parent_correct_retained"]
                    != metrics["parent_correct_total"]
                ):
                    failures.append(f"{split}:parent_correct_retention")
        if self.config.require_no_split_accuracy_regression:
            for split, metrics in by_split.items():
                if metrics["child_correct"] < metrics["parent_correct"]:
                    failures.append(f"{split}:accuracy")
        if self.config.require_no_direct_answer_regression:
            for split, metrics in by_split.items():
                if metrics["child_direct"] < metrics["parent_direct"]:
                    failures.append(f"{split}:direct_answers")
        if self.config.require_no_split_category_regression:
            for split, categories in by_split_category.items():
                for category, metrics in categories.items():
                    if metrics["child_correct"] < metrics["parent_correct"]:
                        failures.append(f"{split}:{category}:accuracy")
                    if (
                        self.config.require_no_direct_answer_regression
                        and metrics["child_direct"] < metrics["parent_direct"]
                    ):
                        failures.append(f"{split}:{category}:direct_answers")
        report = {
            "schema": "neural-forge-neo-native-behavior-gate-v2",
            "behavior_cache_fingerprint": self.behavior_cache.fingerprint,
            "behavior_gate_manifest_fingerprint": (
                self.behavior_gate_manifest_fingerprint
            ),
            "parent_outputs_sha256": self.behavior_cache.outputs_sha256,
            "coverage_policy": coverage_policy.as_dict(),
            "parent_correct_by_split": parent_correct_by_split,
            "parent_correct_by_category": parent_correct_by_category,
            "by_split": by_split,
            "by_split_category": by_split_category,
            "failures": sorted(set(failures)),
            "rows": rows,
            "parent_forward_count": 0,
            "child_forward_suite_count": len(rows),
            "observed_child_module_forward_calls": (
                execution_counts["module_forward_calls"]
            ),
            "observed_child_root_forward_calls": (
                execution_counts["root_forward_calls"]
            ),
            "exact_parent_output_is_diagnostic_only": True,
        }
        self.last_report = MappingProxyType(report)
        self.evaluation_count += 1
        flat_metrics = {
            "calibration_parent_correct": by_split["calibration"]["parent_correct"],
            "calibration_child_correct": by_split["calibration"]["child_correct"],
            "calibration_parent_correct_retained": by_split["calibration"][
                "parent_correct_retained"
            ],
            "heldout_parent_correct": by_split["heldout"]["parent_correct"],
            "heldout_child_correct": by_split["heldout"]["child_correct"],
            "heldout_parent_correct_retained": by_split["heldout"][
                "parent_correct_retained"
            ],
            "heldout_exact_parent_output_rate": by_split["heldout"][
                "exact_parent_output_rate"
            ],
            "parent_forward_count": 0,
            "child_outputs_scored": len(rows),
            "observed_child_module_forward_calls": (
                execution_counts["module_forward_calls"]
            ),
            "observed_child_root_forward_calls": (
                execution_counts["root_forward_calls"]
            ),
        }
        if failures:
            return FoldDecision.reject(
                "Native NEO whole-image behavior regressed: "
                + ", ".join(sorted(set(failures))),
                flat_metrics,
            )
        return FoldDecision.accept(
            "Native NEO child retained every cached parent-correct visual answer "
            "without split/category/direct-answer regression.",
            flat_metrics,
        )

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        return self.evaluate_model(context.execution_context.child_model)

    def __call__(self, model: Any) -> FoldDecision:
        return self.evaluate_model(model)


@dataclass(frozen=True, slots=True)
class NeoNativeVisionStackResources:
    cache: NeoNativeVisionFoldCache
    behavior_gate: NeoNativeVisionBehaviorGate
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    saver_id: str
    loader_id: str
    snapshot_config: DynamicFoldSnapshotConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cache, NeoNativeVisionFoldCache):
            raise DynamicNeoFoldError("cache must be NeoNativeVisionFoldCache.")
        if not isinstance(self.behavior_gate, NeoNativeVisionBehaviorGate):
            raise DynamicNeoFoldError(
                "behavior_gate must be NeoNativeVisionBehaviorGate."
            )
        if (
            self.behavior_gate.behavior_cache.fingerprint
            != self.cache.behavior_cache.fingerprint
        ):
            raise DynamicNeoFoldError(
                "NEO behavior gate and trace cache use different parent outputs."
            )
        for name in ("model_saver", "model_loader"):
            if not callable(getattr(self, name)):
                raise DynamicNeoFoldError(f"{name} must be callable.")
        for name in ("saver_id", "loader_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DynamicNeoFoldError(f"{name} must be nonempty.")
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise DynamicNeoFoldError(
                "snapshot_config must be DynamicFoldSnapshotConfig or None."
            )


class NeoNativeVisionFoldStackAdapter:
    """Catalog adapter for one authenticated native-vision cache and target."""

    adapter_id = STACK_ADAPTER_ID
    version = 1
    priority = 250
    target_families = ("neo_chat",)
    target_modalities = ("image", "text")
    method_ids = (METHOD_ID,)

    def __init__(self, resources: NeoNativeVisionStackResources) -> None:
        if not isinstance(resources, NeoNativeVisionStackResources):
            raise DynamicNeoFoldError(
                "resources must be NeoNativeVisionStackResources."
            )
        self.resources = resources
        self.adapter_id = f"{STACK_ADAPTER_ID}.{resources.cache.cache_key[:16]}"

    def _validate(self, target: DynamicTargetSynthesisResult) -> None:
        if not isinstance(target, DynamicTargetSynthesisResult):
            raise DynamicNeoFoldError(
                "NEO stack requires DynamicTargetSynthesisResult."
            )
        if (
            target.adapter_id != NeoNativeVisionTargetSynthesisAdapter.adapter_id
            or target.adapter_version != NeoNativeVisionTargetSynthesisAdapter.version
            or target.graph.family != "neo_chat"
            or target.resource_profile.modalities != ("image", "text")
        ):
            raise DynamicNeoFoldError("Synthesized target is not a native NEO child.")
        requested_cache_key = dict(target.request.configuration).get("fold_cache_key")
        if (
            requested_cache_key is not None
            and requested_cache_key != self.resources.cache.cache_key
        ):
            raise DynamicNeoFoldError(
                "Synthesized NEO target requests a different fold cache."
            )
        parent_graphs = dict(target.parent_graph_fingerprints)
        if len(parent_graphs) != 1 or next(iter(parent_graphs.values())) != (
            self.resources.cache.parent_graph_fingerprint
        ):
            raise DynamicNeoFoldError(
                "Synthesized NEO target parent graph differs from the cache."
            )
        source_widths = {
            selection.source_neurons for selection in self.resources.cache.selections
        }
        target_widths = {
            selection.target_neurons for selection in self.resources.cache.selections
        }
        if len(source_widths) != 1 or len(target_widths) != 1:
            raise DynamicNeoFoldError(
                "NEO fold cache does not describe one uniform decoder width."
            )
        source_intermediate = next(iter(source_widths))
        target_intermediate = next(iter(target_widths))
        report = dict(target.adapter_report)
        hidden_size = report.get("decoder_hidden_size")
        layer_count = len(self.resources.cache.selections)
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
        ):
            raise DynamicNeoFoldError(
                "NEO target report lacks a valid decoder hidden size."
            )
        expected_delta = (
            layer_count * 3 * hidden_size * (source_intermediate - target_intermediate)
        )
        if (
            report.get("source_intermediate_size") != source_intermediate
            or report.get("target_intermediate_size") != target_intermediate
            or report.get("decoder_layer_count") != layer_count
            or report.get("exact_parameter_delta") != expected_delta
            or report.get("hub_loader_used") is not False
        ):
            raise DynamicNeoFoldError(
                "NEO target does not match the authenticated cached fold shape."
            )
        if (
            target.resource_profile.unique_parameters
            != target.request.exact_unique_parameters
            or len(target.graph.layer_indices) != len(self.resources.cache.selections)
        ):
            raise DynamicNeoFoldError(
                "NEO target budget/depth differs from cached fold evidence."
            )

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            self._validate(target)
        except (DynamicNeoFoldError, TypeError, ValueError):
            return False
        return True

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        self._validate(target)
        cache = self.resources.cache
        cache.verify()
        snapshot = (
            DynamicFoldSnapshotConfig.memory()
            if self.resources.snapshot_config is None
            else self.resources.snapshot_config
        )
        return DynamicFoldSystemStack(
            operators=FoldOperatorRegistry(
                (
                    NeoNativeVisionSwiGLUFoldOperator(
                        snapshot_config=snapshot,
                    ),
                )
            ),
            evidence_provider=NeoNativeVisionEvidenceProvider(cache),
            evaluator=NeoNativeVisionFoldAdmissionEvaluator(),
            final_evaluator=self.resources.behavior_gate,
            model_saver=self.resources.model_saver,
            model_loader=self.resources.model_loader,
            fresh_reload_evaluator=self.resources.behavior_gate,
            saver_id=self.resources.saver_id,
            loader_id=self.resources.loader_id,
            behavior_gate_id=self.resources.behavior_gate.behavior_gate_id,
            behavior_gate_version=(self.resources.behavior_gate.behavior_gate_version),
            behavior_baseline_fingerprint=(
                self.resources.behavior_gate.behavior_baseline_fingerprint
            ),
            initializer=NeoNativeVisionFoldOnlyInitializer(cache),
            snapshot_config=snapshot,
            evidence_fingerprint=cache.cache_key,
        )


def write_neo_dynamic_fold_receipt(
    result: Any,
    cache: NeoNativeVisionFoldCache,
    behavior_gate: NeoNativeVisionBehaviorGate,
    *,
    destination: str | os.PathLike[str],
    cache_reused: bool = False,
    search_result: Any | None = None,
) -> Mapping[str, Any]:
    """Write one self-hashed receipt after standalone reload acceptance."""

    if not bool(getattr(result, "complete", False)):
        raise DynamicNeoFoldError(
            "Only a complete published NEO fold can receive a receipt."
        )
    if not isinstance(cache_reused, bool):
        raise DynamicNeoFoldError("cache_reused must be boolean.")
    artifact = result.published.artifact
    run = result.published.compilation.run
    if behavior_gate.last_report is None:
        raise DynamicNeoFoldError(
            "NEO receipt requires a completed whole-image behavior gate."
        )
    body = {
        "schema": RECEIPT_SCHEMA,
        "target_fingerprint": result.target.fingerprint,
        "plan_fingerprint": result.published.compilation.plan.fingerprint,
        "run_fingerprint": run.fingerprint,
        "artifact_certificate_sha256": artifact.certificate_sha256,
        "artifact_tree_sha256": artifact.artifact_tree_sha256,
        "artifact_file_count": artifact.artifact_file_count,
        "artifact_total_bytes": artifact.artifact_total_bytes,
        "child_graph_fingerprint": artifact.child_graph_fingerprint,
        "child_state_fingerprint": artifact.child_state_fingerprint,
        "exact_unique_parameters": artifact.exact_unique_parameters,
        "cache_key": cache.cache_key,
        "source_cache_key": cache.source_cache_key,
        "fold_anchor_count": cache.anchor_count,
        "fold_candidate_count": cache.candidate_count,
        "behavior_cache_fingerprint": cache.behavior_cache.fingerprint,
        "behavior_gate_manifest_fingerprint": (
            behavior_gate.behavior_gate_manifest_fingerprint
        ),
        "behavior_manifest_sha256": cache.behavior_cache.manifest_sha256,
        "parent_outputs_sha256": cache.behavior_cache.outputs_sha256,
        "calibration_samples": len(cache.behavior_cache.calibration),
        "heldout_samples": len(cache.behavior_cache.heldout),
        "heldout_samples_used_for_selection": 0,
        "fold_cache_reused_this_run": cache_reused,
        "sealed_trace_capture_calls": cache.sample_runner_calls,
        "trace_capture_calls_this_run": (
            0 if cache_reused else cache.sample_runner_calls
        ),
        "parent_outputs_regenerated_after_cache": False,
        "parent_forward_count_during_fold_and_gates": 0,
        "fresh_reload_accepted": artifact.fresh_reload_decision.accepted,
        "behavior_report": dict(behavior_gate.last_report),
        "behavior_report_sha256": canonical_sha256(dict(behavior_gate.last_report)),
        "optimizer_constructions": (
            run.optimizer_constructions + artifact.optimizer_constructions
        ),
        "optimizer_steps": run.optimizer_steps + artifact.optimizer_steps,
        "backward_calls": run.backward_calls + artifact.backward_calls,
        "autograd_grad_calls": (run.autograd_grad_calls + artifact.autograd_grad_calls),
        "zero_training": True,
        "standalone_fresh_reload": True,
        "compilation_optimizer_constructions": (
            artifact.compilation_optimizer_constructions
        ),
        "compilation_optimizer_steps": artifact.compilation_optimizer_steps,
        "compilation_backward_calls": artifact.compilation_backward_calls,
        "compilation_autograd_grad_calls": (artifact.compilation_autograd_grad_calls),
        "fresh_reload_optimizer_constructions": (artifact.optimizer_constructions),
        "fresh_reload_optimizer_steps": artifact.optimizer_steps,
        "fresh_reload_backward_calls": artifact.backward_calls,
        "fresh_reload_autograd_grad_calls": artifact.autograd_grad_calls,
    }
    if search_result is not None:
        attempts = getattr(search_result, "attempts", None)
        selected_candidate_id = getattr(
            search_result,
            "selected_candidate_id",
            None,
        )
        selected_destination = getattr(
            search_result,
            "selected_destination",
            None,
        )
        portfolio_id = getattr(search_result, "portfolio_id", None)
        portfolio_version = getattr(search_result, "portfolio_version", None)
        fingerprint = getattr(search_result, "fingerprint", None)
        if (
            not isinstance(attempts, tuple)
            or not isinstance(selected_candidate_id, str)
            or not isinstance(selected_destination, str)
            or not isinstance(portfolio_id, str)
            or isinstance(portfolio_version, bool)
            or not isinstance(portfolio_version, int)
            or not isinstance(fingerprint, str)
        ):
            raise DynamicNeoFoldError(
                "NEO catalog receipt received an incomplete search result."
            )
        body["production_catalog"] = {
            "portfolio_id": portfolio_id,
            "portfolio_version": portfolio_version,
            "search_result_fingerprint": fingerprint,
            "selected_candidate_id": selected_candidate_id,
            "selected_destination": selected_destination,
            "stop_after_first_accept": True,
            "attempts": [attempt.as_dict() for attempt in attempts],
        }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    path = Path(destination).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive_json(path, receipt)
    return MappingProxyType(receipt)


__all__ = [
    "BEHAVIOR_SCHEMA",
    "DynamicNeoFoldError",
    "FOLD_CACHE_SCHEMA",
    "METHOD_ID",
    "NeoLayerFoldSelection",
    "NeoNativeVisionBehaviorGate",
    "NeoNativeVisionEvidenceProvider",
    "NeoNativeVisionFoldAdmissionEvaluator",
    "NeoNativeVisionFoldCache",
    "NeoNativeVisionFoldOnlyInitializer",
    "NeoNativeVisionFoldStackAdapter",
    "NeoNativeVisionSwiGLUFoldOperator",
    "NeoNativeVisionStackResources",
    "NeoParentBehaviorCache",
    "NeoVisionBehaviorGateConfig",
    "NeoVisionBehaviorRecord",
    "RECEIPT_SCHEMA",
    "STACK_ADAPTER_ID",
    "build_neo_native_vision_fold_cache",
    "derive_neo_native_vision_fold_cache",
    "load_neo_parent_behavior_cache",
    "write_neo_dynamic_fold_receipt",
]
