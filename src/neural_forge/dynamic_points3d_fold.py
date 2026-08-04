"""Zero-training native-points3d folding foundation, model-agnostic by design.

This is the points3d mirror of :mod:`neural_forge.dynamic_video_fold`, which is
itself the video mirror of :mod:`neural_forge.dynamic_audio_fold` and of the
validated NEO native-vision lane.  The one deliberate difference every lane
shares is preserved here: nothing in this module names a parent architecture.
The parent enters through caller-supplied ``trace_runner``, ``child_runner``,
and ``layer_resolver`` callables, so choosing a real geometry-capable parent
model stays a later, separable decision.

What this module provides
-------------------------
* :class:`Points3dBehaviorRecord` / :class:`Points3dParentBehaviorCache` and
  :func:`load_points3d_parent_behavior_cache` -- authenticated cached parent
  answers over a sealed points3d probe manifest, with every cloud re-decoded
  and re-hashed under :data:`POINTS3D_HASH_BASIS` before a record is trusted.
* :class:`NativePoints3dBehaviorGate` -- parent-relative whole-cloud review.
  All parent-correct answers must be retained, no split or split/category
  regression is allowed, the declared inference execution contract must be
  observed, and the parent is never forwarded live.
* :func:`build_native_points3d_fold_trace_cache` -- points3d-conditioned
  calibration evidence.  The hook pattern (MLP down-projection input plus
  decoder-layer residual endpoint) is modality-agnostic; only the conditioning
  stimulus differs from the video, audio, and vision lanes.
* :class:`NativePoints3dFoldCache` and the operator, transaction, evidence
  provider, stack adapter, and receipt writer that consume it -- the same
  decoder-MLP width-reduction scope the video lane folds, with the learned
  point-set front end copied exactly.

Output-side review of *emitted* geometry lives in
:mod:`neural_forge.dynamic_points3d_output`, not here.  This module reviews a
model whose media is on the input side and whose answers are text, exactly like
every lane before it.
"""

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

from .dynamic_points3d_adapter import NativePoints3dTargetSynthesisAdapter
from .dynamic_behavior_gate import (
    POINTS3D_HASH_BASIS,
    BehaviorCallableIdentity,
    BehaviorInferenceExecutionContract,
    MediaReference,
    MultimodalWholeOutputCase,
    TaskScoreCoveragePolicy,
    build_multimodal_behavior_gate_binding,
    execution_contract_satisfied,
    expose_behavior_gate_binding,
    identify_behavior_callable,
    observe_model_execution,
)
from .dynamic_circuit_ir import MODALITY_POINTS3D, ModelCircuitGraph, canonical_sha256
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluationContext,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig, capture_model_state
from .dynamic_modality_fold_selection import (
    ModalityFoldSelectionVocabulary,
    normalize_sparse_anchor_selection,
    selection_from_statistics,
    sparse_anchor_regression,
    sparse_compensation_terms,
    survivors_for,
)
from .dynamic_target_synthesis import DynamicTargetSynthesisResult
from .native_points3d_probe import (
    CATEGORIES,
    CLOSED_ANSWERS,
    CODEC_ID,
    SPLITS,
    NativePoints3dProbeError,
    authenticate_native_points3d_probe,
)


METHOD_ID = "native_points3d_swiglu_model"
BEHAVIOR_SCHEMA = "neural-forge-native-points3d-parent-record-v1"
TRACE_CACHE_SCHEMA = "neural-forge-native-points3d-trace-cache-v1"
FOLD_CACHE_SCHEMA = "neural-forge-native-points3d-fold-cache-v1"
RECEIPT_SCHEMA = "neural-forge-native-points3d-dynamic-fold-receipt-v1"
STACK_ADAPTER_ID = "neural_forge.points3d.native_points3d.catalog"
GATE_REPORT_SCHEMA = "neural-forge-native-points3d-behavior-gate-v1"
GATE_ID = "neural_forge.points3d.native_points3d_parent_relative_behavior"
GATE_VERSION = 1
STATISTICS_BLOB = "statistics.bin"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DIRECT_WORD_RE = re.compile(r"\s*([A-Za-z]+)\s*")
_DIRECT_DIGIT_RE = re.compile(r"\s*([0-9]+)\s*")
_STATISTIC_NAMES = (
    "mean_abs",
    "mean_square",
    "sketch",
    "down_norm",
    "residual_endpoint_mean_square_l2",
)
POINTS3D_INFERENCE_EXECUTION_CONTRACT = (
    BehaviorInferenceExecutionContract.top_level_forward()
)


class DynamicPoints3dFoldError(RuntimeError):
    """Native points3d evidence, behavior, or trace capture is invalid."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DynamicPoints3dFoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicPoints3dFoldError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicPoints3dFoldError(f"{name} must be a non-negative integer.")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicPoints3dFoldError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicPoints3dFoldError(f"{name} must be finite.")
    return result


# The native points3d lane's binding of the shared modality-neutral selection
# numerics.  The labels reproduce this lane's own message wording exactly; the
# validators are this lane's own, so its ``_finite`` message survives verbatim.
_SELECTION_VOCABULARY = ModalityFoldSelectionVocabulary(
    error_type=DynamicPoints3dFoldError,
    subject_label="Native points3d",
    selection_label="native points3d",
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
        raise DynamicPoints3dFoldError("Payload is not canonical JSON.") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json(dict(payload)) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_float32_bytes(value: Any) -> bytes:
    import torch

    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return tensor.numpy().astype("<f4", copy=False).tobytes()


def _tensor_descriptor(value: Any) -> dict[str, Any]:
    payload = _canonical_float32_bytes(value)
    return {
        "shape": [int(size) for size in value.shape],
        "dtype": "float32",
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def normalize_points3d_answer(category: str, raw_output: Any) -> str | None:
    """Map one raw child or parent answer onto its closed points3d answer set.

    Anything outside the closed set for the category -- prose, an explanation,
    a refusal, a second sentence -- normalizes to ``None`` and counts as a
    non-direct answer, exactly as in the video, audio, and NEO vision lanes.
    """

    if category not in CLOSED_ANSWERS:
        raise DynamicPoints3dFoldError(f"Unsupported points3d category {category!r}.")
    if not isinstance(raw_output, str):
        return None
    allowed = CLOSED_ANSWERS[category]
    if category == "cluster_count":
        match = _DIRECT_DIGIT_RE.fullmatch(raw_output)
        if match is None:
            return None
        return match.group(1) if match.group(1) in allowed else None
    match = _DIRECT_WORD_RE.fullmatch(raw_output)
    if match is None:
        return None
    normalized = match.group(1).lower()
    return normalized if normalized in allowed else None


@dataclass(frozen=True, slots=True)
class Points3dBehaviorRecord:
    """One authenticated probe and the cached parent answer bound to it."""

    index: int
    probe_id: str
    split: str
    category: str
    prompt: str
    expected: str
    points_path: str
    points_sha256: str
    container_sha256: str
    container_bytes: int
    point_count: int
    components: int
    dtype: str
    scale: tuple[float, float, float]
    zero_point: tuple[int, int, int]
    manifest_record_sha256: str
    parent_raw_output: str
    parent_normalized_prediction: str | None
    parent_direct_answer: bool
    parent_correct: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _nonnegative_int(self.index, "index"))
        for name in ("probe_id", "prompt", "expected", "points_path", "dtype"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DynamicPoints3dFoldError(f"{name} must be nonempty.")
        if self.split not in SPLITS:
            raise DynamicPoints3dFoldError("Points3d behavior split is invalid.")
        if self.category not in CATEGORIES:
            raise DynamicPoints3dFoldError("Points3d behavior category is invalid.")
        if self.expected not in CLOSED_ANSWERS[self.category]:
            raise DynamicPoints3dFoldError(
                "Points3d expected answer escapes its closed answer set."
            )
        _digest(self.points_sha256, "points_sha256")
        _digest(self.container_sha256, "container_sha256")
        _digest(self.manifest_record_sha256, "manifest_record_sha256")
        for name in ("container_bytes", "point_count", "components"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        scale = tuple(_finite(value, "scale") for value in self.scale)
        if len(scale) != self.components or any(value <= 0.0 for value in scale):
            raise DynamicPoints3dFoldError(
                "Points3d scale must carry one positive value per component."
            )
        object.__setattr__(self, "scale", scale)
        zero_point = tuple(self.zero_point)
        if len(zero_point) != self.components or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in zero_point
        ):
            raise DynamicPoints3dFoldError(
                "Points3d zero_point must carry one integer per component."
            )
        object.__setattr__(self, "zero_point", zero_point)
        if not isinstance(self.parent_raw_output, str):
            raise DynamicPoints3dFoldError("parent_raw_output must be text.")
        if self.parent_normalized_prediction is not None and not isinstance(
            self.parent_normalized_prediction, str
        ):
            raise DynamicPoints3dFoldError(
                "parent_normalized_prediction must be text or None."
            )
        if not isinstance(self.parent_direct_answer, bool) or not isinstance(
            self.parent_correct, bool
        ):
            raise DynamicPoints3dFoldError(
                "Parent direct/correct fields must be boolean."
            )

    @property
    def media(self) -> MediaReference:
        """Bind the decoded quantized cloud, not its container, to this record."""

        return MediaReference(
            modality=MODALITY_POINTS3D,
            content_sha256=self.points_sha256,
            uri=self.points_path,
            byte_count=self.container_bytes,
            codec=CODEC_ID,
            hash_basis=POINTS3D_HASH_BASIS,
            container_sha256=self.container_sha256,
            attributes={
                "components": self.components,
                "dtype": self.dtype,
                "point_count": self.point_count,
                "scale_x": self.scale[0],
                "scale_y": self.scale[1],
                "scale_z": self.scale[2],
                "zero_x": self.zero_point[0],
                "zero_y": self.zero_point[1],
                "zero_z": self.zero_point[2],
            },
        )

    def as_case(self) -> MultimodalWholeOutputCase:
        """Express the record in the shared media-capable review schema."""

        return MultimodalWholeOutputCase(
            case_id=self.probe_id,
            prompt=self.prompt,
            parent_output=self.parent_raw_output,
            inputs=(self.media,),
            expected=self.expected,
            category=self.category,
            critical_nonempty=self.parent_correct,
            split=self.split,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "probe_id": self.probe_id,
            "split": self.split,
            "category": self.category,
            "prompt": self.prompt,
            "expected": self.expected,
            "points_sha256": self.points_sha256,
            "container_sha256": self.container_sha256,
            "manifest_record_sha256": self.manifest_record_sha256,
            "parent_raw_output_sha256": hashlib.sha256(
                self.parent_raw_output.encode("utf-8")
            ).hexdigest(),
            "parent_normalized_prediction": self.parent_normalized_prediction,
            "parent_direct_answer": self.parent_direct_answer,
            "parent_correct": self.parent_correct,
        }


@dataclass(frozen=True, slots=True)
class Points3dParentBehaviorCache:
    """Immutable cached parent answers for one sealed points3d probe suite."""

    manifest_path: str
    outputs_path: str
    manifest_sha256: str
    outputs_sha256: str
    records: tuple[Points3dBehaviorRecord, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        _digest(self.manifest_sha256, "manifest_sha256")
        _digest(self.outputs_sha256, "outputs_sha256")
        _digest(self.fingerprint, "behavior cache fingerprint")
        if not self.records:
            raise DynamicPoints3dFoldError(
                "Points3d behavior cache must contain records."
            )
        if tuple(record.index for record in self.records) != tuple(
            range(len(self.records))
        ):
            raise DynamicPoints3dFoldError(
                "Points3d behavior records must preserve exact manifest order."
            )
        split_counts = {
            split: sum(record.split == split for record in self.records)
            for split in SPLITS
        }
        if any(count <= 0 for count in split_counts.values()):
            raise DynamicPoints3dFoldError(
                "Points3d behavior cache must contain calibration and held-out "
                "records."
            )

    @property
    def calibration(self) -> tuple[Points3dBehaviorRecord, ...]:
        return tuple(record for record in self.records if record.split == "calibration")

    @property
    def heldout(self) -> tuple[Points3dBehaviorRecord, ...]:
        return tuple(record for record in self.records if record.split == "heldout")

    def cases(self) -> tuple[MultimodalWholeOutputCase, ...]:
        return tuple(record.as_case() for record in self.records)

    def verify(self) -> None:
        """Re-authenticate the sealed sources this cache was derived from."""

        manifest = Path(self.manifest_path)
        outputs = Path(self.outputs_path)
        if not manifest.is_file() or not outputs.is_file():
            raise DynamicPoints3dFoldError(
                "Points3d behavior cache sources are missing."
            )
        if (
            _file_sha256(manifest) != self.manifest_sha256
            or _file_sha256(outputs) != self.outputs_sha256
        ):
            raise DynamicPoints3dFoldError(
                "Points3d behavior cache sources changed after authentication."
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
            raise DynamicPoints3dFoldError(
                "Points3d behavior cache fingerprint is invalid."
            )


def points3d_parent_output_record(
    probe: Any,
    raw_output: str,
) -> dict[str, Any]:
    """Build one canonical parent-output cache line for an authenticated probe."""

    if not isinstance(raw_output, str):
        raise DynamicPoints3dFoldError("Parent raw output must be text.")
    normalized = normalize_points3d_answer(probe.category, raw_output)
    return {
        "schema": BEHAVIOR_SCHEMA,
        "index": probe.index,
        "probe_id": probe.probe_id,
        "record_sha256": probe.record_sha256,
        "category": probe.category,
        "split": probe.split,
        "expected": probe.answer,
        "points_sha256": probe.points_sha256,
        "normalized_prediction": normalized,
        "direct_answer": normalized is not None,
        "correct": normalized == probe.answer,
        "raw_output": raw_output,
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
    }


def write_points3d_parent_output_cache(
    manifest_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    parent_runner: Callable[[Any], str],
) -> Path:
    """Run a parent once per probe and seal its answers as a JSONL cache.

    This is the only place in the points3d lane that touches a parent model for
    behavior.  Everything downstream -- gating, retention, promotion --
    replays this file and performs zero parent forwards.
    """

    if not callable(parent_runner):
        raise DynamicPoints3dFoldError("parent_runner must be callable.")
    _manifest, probes = authenticate_native_points3d_probe(manifest_path)
    target = Path(destination).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for probe in probes:
        raw_output = parent_runner(probe)
        lines.append(_canonical_json(points3d_parent_output_record(probe, raw_output)))
    with target.open("xb") as handle:
        for line in lines:
            handle.write(line + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def load_points3d_parent_behavior_cache(
    manifest_path: str | os.PathLike[str],
    outputs_path: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str | None = None,
    expected_outputs_sha256: str | None = None,
) -> Points3dParentBehaviorCache:
    """Authenticate a sealed probe manifest and its cached parent answers."""

    manifest = Path(manifest_path).expanduser().resolve()
    outputs = Path(outputs_path).expanduser().resolve()
    if not manifest.is_file() or not outputs.is_file():
        raise DynamicPoints3dFoldError(
            "Points3d manifest/output cache paths must be files."
        )
    manifest_sha = _file_sha256(manifest)
    outputs_sha = _file_sha256(outputs)
    if expected_manifest_sha256 is not None and manifest_sha != _digest(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    ):
        raise DynamicPoints3dFoldError(
            "Points3d manifest differs from the pinned digest."
        )
    if expected_outputs_sha256 is not None and outputs_sha != _digest(
        expected_outputs_sha256,
        "expected_outputs_sha256",
    ):
        raise DynamicPoints3dFoldError(
            "Points3d parent outputs differ from the pinned digest."
        )
    try:
        _payload, probes = authenticate_native_points3d_probe(manifest)
    except NativePoints3dProbeError as exc:
        raise DynamicPoints3dFoldError(
            f"Points3d probe manifest authentication failed: {exc}"
        ) from exc
    raw_lines = outputs.read_bytes()
    if raw_lines and not raw_lines.endswith(b"\n"):
        raise DynamicPoints3dFoldError("Points3d parent output cache is truncated.")
    lines = raw_lines.splitlines()
    if len(lines) != len(probes):
        raise DynamicPoints3dFoldError(
            "Points3d parent output count differs from the manifest."
        )
    records = []
    for index, (probe, line) in enumerate(zip(probes, lines, strict=True)):
        try:
            cached = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicPoints3dFoldError(
                f"Points3d output line {index + 1} is invalid."
            ) from exc
        if not isinstance(cached, Mapping):
            raise DynamicPoints3dFoldError("Points3d output record is not an object.")
        raw_output = cached.get("raw_output")
        normalized = normalize_points3d_answer(probe.category, raw_output)
        required = {
            "schema": BEHAVIOR_SCHEMA,
            "index": index,
            "probe_id": probe.probe_id,
            "record_sha256": probe.record_sha256,
            "category": probe.category,
            "split": probe.split,
            "expected": probe.answer,
            "points_sha256": probe.points_sha256,
            "normalized_prediction": normalized,
            "direct_answer": normalized is not None,
            "correct": normalized == probe.answer,
        }
        if any(cached.get(key) != value for key, value in required.items()):
            raise DynamicPoints3dFoldError(
                f"Points3d cached parent record {index} failed authentication."
            )
        if (
            cached.get("raw_output_sha256")
            != hashlib.sha256(str(raw_output).encode("utf-8")).hexdigest()
        ):
            raise DynamicPoints3dFoldError(
                f"Points3d cached parent output hash failed for record {index}."
            )
        records.append(
            Points3dBehaviorRecord(
                index=index,
                probe_id=probe.probe_id,
                split=probe.split,
                category=probe.category,
                prompt=probe.prompt,
                expected=probe.answer,
                points_path=probe.path,
                points_sha256=probe.points_sha256,
                container_sha256=probe.container_sha256,
                container_bytes=probe.container_bytes,
                point_count=probe.point_count,
                components=probe.components,
                dtype=probe.dtype,
                scale=probe.scale,
                zero_point=probe.zero_point,
                manifest_record_sha256=probe.record_sha256,
                parent_raw_output=str(raw_output),
                parent_normalized_prediction=normalized,
                parent_direct_answer=normalized is not None,
                parent_correct=normalized == probe.answer,
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
    result = Points3dParentBehaviorCache(
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
class Points3dBehaviorGateConfig:
    """Parent-relative acceptance thresholds for whole-cloud points3d review."""

    require_all_parent_correct_retained: bool = True
    require_no_split_accuracy_regression: bool = True
    require_no_split_category_regression: bool = True
    require_no_direct_answer_regression: bool = True
    require_parent_correct_task_coverage: bool = True
    required_splits: tuple[str, ...] = SPLITS
    required_categories: tuple[str, ...] = CATEGORIES
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
                raise DynamicPoints3dFoldError(f"{name} must be boolean.")
        for name, allowed in (
            ("required_splits", frozenset(SPLITS)),
            ("required_categories", frozenset(CATEGORIES)),
        ):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise DynamicPoints3dFoldError(f"{name} must be nonempty and unique.")
            if any(value not in allowed for value in values):
                raise DynamicPoints3dFoldError(
                    f"{name} contains an unsupported value."
                )
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


class NativePoints3dBehaviorGate:
    """Parent-relative whole-cloud gate using cached answers as canonical."""

    def __init__(
        self,
        behavior_cache: Points3dParentBehaviorCache,
        *,
        child_runner: Callable[[Any, Points3dBehaviorRecord], str],
        config: Points3dBehaviorGateConfig | None = None,
        child_runner_identity: BehaviorCallableIdentity | None = None,
        inference_execution_contract: (
            BehaviorInferenceExecutionContract | None
        ) = None,
    ) -> None:
        if not isinstance(behavior_cache, Points3dParentBehaviorCache):
            raise DynamicPoints3dFoldError(
                "behavior_cache must be Points3dParentBehaviorCache."
            )
        if not callable(child_runner):
            raise DynamicPoints3dFoldError("child_runner must be callable.")
        self.behavior_cache = behavior_cache
        self.child_runner = child_runner
        self.config = config or Points3dBehaviorGateConfig()
        if not isinstance(self.config, Points3dBehaviorGateConfig):
            raise DynamicPoints3dFoldError(
                "config must be Points3dBehaviorGateConfig."
            )
        self.execution_contract = (
            POINTS3D_INFERENCE_EXECUTION_CONTRACT
            if inference_execution_contract is None
            else inference_execution_contract
        )
        if not isinstance(
            self.execution_contract,
            BehaviorInferenceExecutionContract,
        ):
            raise DynamicPoints3dFoldError(
                "inference_execution_contract must be "
                "BehaviorInferenceExecutionContract or None."
            )
        self.last_report: Mapping[str, Any] | None = None
        self.evaluation_count = 0
        if child_runner_identity is None:
            child_runner_identity = identify_behavior_callable(
                child_runner,
                implementation_id=(
                    "neural_forge.points3d.native_points3d.child_record_runner"
                ),
                version=1,
            )
        if not isinstance(child_runner_identity, BehaviorCallableIdentity):
            raise DynamicPoints3dFoldError(
                "child_runner_identity must be BehaviorCallableIdentity."
            )
        scorer_identity = identify_behavior_callable(
            normalize_points3d_answer,
            implementation_id=(
                "neural_forge.points3d.native_points3d.closed_answer_scorer"
            ),
            version=1,
        )
        binding = build_multimodal_behavior_gate_binding(
            gate_id=GATE_ID,
            gate_version=GATE_VERSION,
            cases=behavior_cache.cases(),
            threshold_config=self.config.as_dict(),
            generator_identity=child_runner_identity,
            scorer_identity=scorer_identity,
            coverage_policy=self.config.coverage_policy.as_dict(),
            baseline_fingerprint=behavior_cache.fingerprint,
            baseline={
                "behavior_cache_fingerprint": behavior_cache.fingerprint,
                "manifest_sha256": behavior_cache.manifest_sha256,
                "parent_outputs_sha256": behavior_cache.outputs_sha256,
                "record_count": len(behavior_cache.records),
            },
            inference_execution_contract=self.execution_contract,
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
        exact_parent_output = sum(row["exact_parent_output"] for row in rows)
        return {
            "total": total,
            "parent_correct": parent_correct,
            "child_correct": child_correct,
            "parent_direct": parent_direct,
            "child_direct": child_direct,
            "parent_correct_retained": retained,
            "parent_correct_total": parent_correct,
            "parent_correct_retention": (
                retained / parent_correct if parent_correct else 1.0
            ),
            "exact_parent_output": exact_parent_output,
            "exact_parent_output_rate": (exact_parent_output / total if total else 1.0),
        }

    def _coverage_counts(
        self,
    ) -> tuple[dict[str, int], dict[str, int], tuple[str, ...]]:
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
        missing_cross: tuple[str, ...] = ()
        if self.config.coverage_policy.require_split_category_cross_product:
            missing_cross = tuple(
                f"{split}:{category}"
                for split in self.config.required_splits
                for category in self.config.required_categories
                if not any(
                    record.parent_correct
                    and record.split == split
                    and record.category == category
                    for record in self.behavior_cache.records
                )
            )
        return parent_correct_by_split, parent_correct_by_category, missing_cross

    def evaluate_model(self, model: Any) -> FoldDecision:
        self.behavior_cache.verify()
        coverage_policy = self.config.coverage_policy
        (
            parent_correct_by_split,
            parent_correct_by_category,
            missing_cross_coverage,
        ) = self._coverage_counts()
        missing_split_coverage = tuple(
            split for split, count in parent_correct_by_split.items() if count <= 0
        )
        missing_category_coverage = tuple(
            category
            for category, count in parent_correct_by_category.items()
            if count <= 0
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
                "schema": GATE_REPORT_SCHEMA,
                "modality": MODALITY_POINTS3D,
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
                "Native points3d task-score parent-correct coverage is "
                "incomplete: " + ", ".join(sorted(set(coverage_failures))),
                {
                    "heldout_parent_correct": parent_correct_by_split.get(
                        "heldout",
                        0,
                    ),
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
            contract=self.execution_contract,
        ) as execution_counts:
            for record in self.behavior_cache.records:
                raw_output = self.child_runner(model, record)
                if not isinstance(raw_output, str):
                    raise DynamicPoints3dFoldError(
                        f"Child runner returned non-text for {record.probe_id}."
                    )
                normalized = normalize_points3d_answer(record.category, raw_output)
                rows.append(
                    {
                        "probe_id": record.probe_id,
                        "split": record.split,
                        "category": record.category,
                        "expected": record.expected,
                        "points_sha256": record.points_sha256,
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
            raise DynamicPoints3dFoldError(
                "Native points3d child runner returned without executing the "
                "declared inference entrypoint and module root."
            )
        by_split = {
            split: self._metrics(tuple(row for row in rows if row["split"] == split))
            for split in SPLITS
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
                for category in CATEGORIES
            }
            for split in SPLITS
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
            "schema": GATE_REPORT_SCHEMA,
            "modality": MODALITY_POINTS3D,
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
                "Native points3d whole-cloud behavior regressed: "
                + ", ".join(sorted(set(failures))),
                flat_metrics,
            )
        return FoldDecision.accept(
            "Native points3d child retained every cached parent-correct answer "
            "without split/category/direct-answer regression.",
            flat_metrics,
        )

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        return self.evaluate_model(context.execution_context.child_model)

    def __call__(self, model: Any) -> FoldDecision:
        return self.evaluate_model(model)


class _Points3dTraceAccumulator:
    """Modality-agnostic activation and residual-endpoint accumulator."""

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
            raise DynamicPoints3dFoldError(
                f"Points3d layer {self.layer_index} observed inconsistent widths."
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
                raise DynamicPoints3dFoldError(
                    "Decoder layer returned an empty endpoint tuple."
                )
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise DynamicPoints3dFoldError(
                "Decoder layer returned a non-tensor residual endpoint."
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
            raise DynamicPoints3dFoldError(
                f"Points3d layer {self.layer_index} captured incomplete "
                "activation/residual endpoint rows."
            )
        down = down_weight.detach()
        if down.ndim != 2 or int(down.shape[1]) != int(self.sum_abs.shape[0]):
            raise DynamicPoints3dFoldError(
                "Down projection differs from captured trace width."
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


def default_layer_resolver(model: Any) -> tuple[Any, ...]:
    """Resolve decoder layers for the common ``model.model.layers`` layout."""

    for path in ("model.layers", "language_model.model.layers", "layers"):
        current: Any = model
        for attribute in path.split("."):
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None:
            try:
                layers = tuple(current)
            except TypeError:
                continue
            if layers:
                return layers
    raise DynamicPoints3dFoldError(
        "Could not resolve decoder layers; supply an explicit layer_resolver."
    )


def default_projection_resolver(layer: Any, index: int) -> Any:
    """Resolve the observable MLP down projection of one decoder layer."""

    projection = getattr(getattr(layer, "mlp", None), "down_proj", None)
    if projection is None or not callable(
        getattr(projection, "register_forward_pre_hook", None)
    ):
        raise DynamicPoints3dFoldError(
            f"Layer {index} lacks an observable MLP down projection; supply an "
            "explicit projection_resolver."
        )
    return projection


class Points3dTraceStatisticsCache:
    """Sealed points3d-conditioned calibration evidence for one parent state.

    The cache stores statistics only.  It intentionally contains no fold
    selection: a selection rule belongs to the fold method that consumes this
    evidence, and sealing one here would freeze a policy this lane has not
    validated yet.
    """

    def __init__(
        self,
        *,
        parent_graph_fingerprint: str,
        parent_state_fingerprint: str,
        behavior_cache: Points3dParentBehaviorCache,
        statistics: Mapping[str, Any],
        sketch_dim: int,
        layer_count: int,
        trace_rows: Sequence[int],
        sample_runner_calls: int,
        cache_key: str | None = None,
        directory: str | None = None,
    ) -> None:
        import torch

        self.parent_graph_fingerprint = _digest(
            parent_graph_fingerprint,
            "parent_graph_fingerprint",
        )
        self.parent_state_fingerprint = _digest(
            parent_state_fingerprint,
            "parent_state_fingerprint",
        )
        if not isinstance(behavior_cache, Points3dParentBehaviorCache):
            raise DynamicPoints3dFoldError(
                "behavior_cache must be Points3dParentBehaviorCache."
            )
        behavior_cache.verify()
        self.behavior_cache = behavior_cache
        self.layer_count = _positive_int(layer_count, "layer_count")
        self.sketch_dim = _positive_int(sketch_dim, "sketch_dim")
        self.sample_runner_calls = _positive_int(
            sample_runner_calls,
            "sample_runner_calls",
        )
        rows = tuple(_positive_int(value, "trace_rows") for value in trace_rows)
        if len(rows) != self.layer_count:
            raise DynamicPoints3dFoldError(
                "trace_rows must contain one row count per observed layer."
            )
        self.trace_rows = rows
        self.statistics = MappingProxyType(
            {
                str(name): value.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                for name, value in statistics.items()
            }
        )
        self.directory = directory
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
            raise DynamicPoints3dFoldError("Unknown points3d statistic name.")
        return f"layer_{layer_index:03d}_{name}"

    def layer_statistics(self, layer_index: int) -> Mapping[str, Any]:
        index = _nonnegative_int(layer_index, "layer_index")
        if index >= self.layer_count:
            raise DynamicPoints3dFoldError(
                "Points3d cache layer index is out of range."
            )
        return MappingProxyType(
            {
                name: self.statistics[self.statistic_key(index, name)]
                for name in _STATISTIC_NAMES
            }
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": TRACE_CACHE_SCHEMA,
            "modality": MODALITY_POINTS3D,
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
            "evidence_used_splits": ["calibration"],
            "evidence_excluded_splits": ["heldout"],
            "conditioning": "points3d_conditioned_decoder_mlp_down_projection_input",
            "residual_endpoint_source": "decoder_layer_output_hidden_state",
            "layer_count": self.layer_count,
            "sketch_dim": self.sketch_dim,
            "trace_rows": list(self.trace_rows),
            "sample_runner_calls": self.sample_runner_calls,
            "statistic_names": list(_STATISTIC_NAMES),
            "statistics": {
                name: _tensor_descriptor(value)
                for name, value in sorted(self.statistics.items())
            },
            "fold_selection_included": False,
            "parent_outputs_regenerated": False,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }

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
            for index in range(self.layer_count)
            for name in _STATISTIC_NAMES
        }
        if set(self.statistics) != expected_names:
            raise DynamicPoints3dFoldError(
                "Points3d cache statistics do not cover every observed layer."
            )
        if self._computed_cache_key() != self.cache_key:
            raise DynamicPoints3dFoldError(
                "Points3d trace cache seal is invalid."
            )
        if parent_graph is not None and (
            parent_graph.fingerprint != self.parent_graph_fingerprint
            or len(parent_graph.layer_indices) != self.layer_count
        ):
            raise DynamicPoints3dFoldError(
                "Loaded parent graph differs from the points3d trace cache."
            )
        if parent_model is not None and (
            fingerprint_model_state(parent_model) != self.parent_state_fingerprint
        ):
            raise DynamicPoints3dFoldError(
                "Loaded parent state differs from the points3d trace cache."
            )

    def save(self, destination: str | os.PathLike[str]) -> Path:
        """Seal statistics as canonical little-endian float32 plus a manifest."""

        destination_path = Path(destination).expanduser().absolute()
        if destination_path.exists():
            raise DynamicPoints3dFoldError(
                f"Points3d trace cache destination exists: {destination_path}."
            )
        parent = destination_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=".points3d-trace-cache-stage-", dir=parent)
        ).absolute()
        try:
            blob_path = stage / STATISTICS_BLOB
            layout: dict[str, dict[str, Any]] = {}
            offset = 0
            with blob_path.open("xb") as handle:
                for name in sorted(self.statistics):
                    payload = _canonical_float32_bytes(self.statistics[name])
                    handle.write(payload)
                    layout[name] = {
                        "offset": offset,
                        "byte_count": len(payload),
                    }
                    offset += len(payload)
                handle.flush()
                os.fsync(handle.fileno())
            body = self._body()
            body["statistics_file"] = blob_path.name
            body["statistics_file_sha256"] = _file_sha256(blob_path)
            body["statistics_layout"] = layout
            body["statistics_byte_order"] = "little"
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
        behavior_cache: Points3dParentBehaviorCache,
    ) -> Points3dTraceStatisticsCache:
        import numpy as np
        import torch

        root = Path(directory).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DynamicPoints3dFoldError(
                "Points3d trace cache manifest is missing."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicPoints3dFoldError(
                "Points3d trace cache manifest is invalid."
            ) from exc
        manifest_sha256 = manifest.pop("manifest_sha256", None)
        if canonical_sha256(manifest) != manifest_sha256:
            raise DynamicPoints3dFoldError(
                "Points3d trace cache manifest seal is invalid."
            )
        cache_key = manifest.get("cache_key")
        _digest(cache_key, "cache_key")
        if (
            manifest.get("schema") != TRACE_CACHE_SCHEMA
            or manifest.get("behavior_cache_fingerprint") != behavior_cache.fingerprint
        ):
            raise DynamicPoints3dFoldError(
                "Points3d trace cache is bound to different behavior evidence."
            )
        blob_path = root / str(manifest.get("statistics_file"))
        if not blob_path.is_file() or _file_sha256(blob_path) != manifest.get(
            "statistics_file_sha256"
        ):
            raise DynamicPoints3dFoldError(
                "Points3d trace statistics file is invalid."
            )
        descriptors = manifest.get("statistics")
        layout = manifest.get("statistics_layout")
        if (
            not isinstance(descriptors, Mapping)
            or not isinstance(layout, Mapping)
            or set(descriptors) != set(layout)
        ):
            raise DynamicPoints3dFoldError(
                "Points3d trace statistic layout is incomplete."
            )
        blob = blob_path.read_bytes()
        statistics: dict[str, Any] = {}
        for name, descriptor in descriptors.items():
            placement = layout[name]
            start = int(placement["offset"])
            stop = start + int(placement["byte_count"])
            payload = blob[start:stop]
            if (
                len(payload) != int(descriptor["byte_count"])
                or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
            ):
                raise DynamicPoints3dFoldError(
                    f"Points3d trace statistic {name} failed its seal."
                )
            array = np.frombuffer(payload, dtype="<f4").reshape(
                tuple(int(size) for size in descriptor["shape"])
            )
            # np.ascontiguousarray would promote a sealed scalar statistic to
            # rank one and silently change the reloaded seal.
            statistics[name] = torch.from_numpy(
                np.array(array, dtype=np.float32, copy=True)
            )
        result = cls(
            parent_graph_fingerprint=manifest["parent_graph_fingerprint"],
            parent_state_fingerprint=manifest["parent_state_fingerprint"],
            behavior_cache=behavior_cache,
            statistics=statistics,
            sketch_dim=manifest["sketch_dim"],
            layer_count=manifest["layer_count"],
            trace_rows=manifest["trace_rows"],
            sample_runner_calls=manifest["sample_runner_calls"],
            cache_key=cache_key,
            directory=str(root),
        )
        # The in-memory seal deliberately excludes storage-container fields.
        if result.cache_key != result._computed_cache_key():
            raise DynamicPoints3dFoldError(
                "Points3d trace cache logical seal differs after reload."
            )
        return result


def build_native_points3d_fold_trace_cache(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    behavior_cache: Points3dParentBehaviorCache,
    *,
    trace_runner: Callable[[Any, Points3dBehaviorRecord], Any],
    destination: str | os.PathLike[str],
    layer_resolver: Callable[[Any], Sequence[Any]] = default_layer_resolver,
    projection_resolver: Callable[[Any, int], Any] = default_projection_resolver,
    sketch_dim: int = 64,
) -> Points3dTraceStatisticsCache:
    """Capture points3d-conditioned calibration traces once and seal them.

    The hook pattern is the modality-agnostic half of NEO's vision cache
    builder: a forward pre-hook on each layer's MLP down projection records the
    activation rows a fold would have to reproduce, and a forward hook on the
    layer records the residual endpoint energy those rows feed.  Only the
    stimulus is geometry, and only through ``trace_runner``, which the caller
    owns.

    Held-out probes are never traced, the parent is asserted to be
    zero-gradient before capture, and both the parameter and runtime state
    fingerprints must be unchanged afterwards.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is an extra
        raise DynamicPoints3dFoldError(
            "PyTorch is required for points3d trace capture."
        ) from exc
    if not isinstance(parent_graph, ModelCircuitGraph):
        raise DynamicPoints3dFoldError("parent_graph must be ModelCircuitGraph.")
    for name, value in (
        ("trace_runner", trace_runner),
        ("layer_resolver", layer_resolver),
        ("projection_resolver", projection_resolver),
    ):
        if not callable(value):
            raise DynamicPoints3dFoldError(f"{name} must be callable.")
    behavior_cache.verify()
    sketch_dim = _positive_int(sketch_dim, "sketch_dim")
    layers = tuple(layer_resolver(parent_model))
    if not layers:
        raise DynamicPoints3dFoldError(
            "Points3d trace capture found no decoder layers."
        )
    if len(layers) != len(parent_graph.layer_indices):
        raise DynamicPoints3dFoldError("Parent graph and runtime decoder depth differ.")
    if any(parameter.grad is not None for parameter in parent_model.parameters()):
        raise DynamicPoints3dFoldError(
            "Points3d trace capture requires a zero-gradient parent."
        )
    state_before = fingerprint_model_state(parent_model)
    runtime_before = fingerprint_model_runtime_state(parent_model)
    module_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    accumulators = [
        _Points3dTraceAccumulator(index, sketch_dim) for index in range(len(layers))
    ]
    projections = tuple(
        projection_resolver(layer, index) for index, layer in enumerate(layers)
    )
    handles = []
    try:
        for index, (layer, projection) in enumerate(
            zip(layers, projections, strict=True)
        ):
            if not callable(getattr(projection, "register_forward_pre_hook", None)):
                raise DynamicPoints3dFoldError(
                    f"Layer {index} projection cannot be observed."
                )

            def capture(
                _module: Any,
                args: tuple[Any, ...],
                *,
                layer_index: int = index,
            ) -> None:
                if not args:
                    raise DynamicPoints3dFoldError(
                        "Down projection hook saw no input."
                    )
                accumulators[layer_index].add(args[0])

            handles.append(projection.register_forward_pre_hook(capture))

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
        raise DynamicPoints3dFoldError(
            "Points3d trace capture mutated the parent model."
        )
    statistics: dict[str, Any] = {}
    trace_rows = []
    for index, (projection, accumulator) in enumerate(
        zip(projections, accumulators, strict=True)
    ):
        weight = getattr(projection, "weight", None)
        if weight is None:
            raise DynamicPoints3dFoldError(
                f"Layer {index} projection exposes no weight tensor."
            )
        values = accumulator.finalize(weight)
        for name, tensor in values.items():
            statistics[Points3dTraceStatisticsCache.statistic_key(index, name)] = tensor
        trace_rows.append(accumulator.rows)
    cache = Points3dTraceStatisticsCache(
        parent_graph_fingerprint=parent_graph.fingerprint,
        parent_state_fingerprint=state_before,
        behavior_cache=behavior_cache,
        statistics=statistics,
        sketch_dim=sketch_dim,
        layer_count=len(layers),
        trace_rows=trace_rows,
        sample_runner_calls=calls,
    )
    path = cache.save(destination)
    return Points3dTraceStatisticsCache.open(path, behavior_cache=behavior_cache)


@dataclass(frozen=True, slots=True)
class NativePoints3dLayerFoldSelection:
    """One removed decoder-MLP neuron and its compensating survivor anchors."""

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
            "points3d_front_end_copied_exactly": True,
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
) -> NativePoints3dLayerFoldSelection:
    """Choose the least-damaging removable neuron from sealed points3d evidence."""

    return selection_from_statistics(
        layer_index,
        statistics,
        trace_rows=trace_rows,
        candidate_count=candidate_count,
        anchor_count=anchor_count,
        vocabulary=_SELECTION_VOCABULARY,
        selection_factory=NativePoints3dLayerFoldSelection,
    )


class NativePoints3dFoldCache:
    """Sealed fold selections derived from one sealed points3d statistics cache.

    :class:`Points3dTraceStatisticsCache` deliberately carries no selection: it
    is evidence.  This cache is the *policy* artifact that a fold method
    consumes.  It copies the sealed statistics verbatim, adds one reproducible
    selection per decoder layer, and cites the statistics cache it was derived
    from, so the evidence seal never changes when a selection policy does.
    """

    def __init__(
        self,
        *,
        parent_graph_fingerprint: str,
        parent_state_fingerprint: str,
        behavior_cache: Points3dParentBehaviorCache,
        selections: Sequence[NativePoints3dLayerFoldSelection],
        statistics: Mapping[str, Any],
        sketch_dim: int,
        candidate_count: int,
        sample_runner_calls: int,
        source_cache_key: str,
        cache_key: str | None = None,
        directory: str | None = None,
    ) -> None:
        import torch

        self.parent_graph_fingerprint = _digest(
            parent_graph_fingerprint,
            "parent_graph_fingerprint",
        )
        self.parent_state_fingerprint = _digest(
            parent_state_fingerprint,
            "parent_state_fingerprint",
        )
        if not isinstance(behavior_cache, Points3dParentBehaviorCache):
            raise DynamicPoints3dFoldError(
                "behavior_cache must be Points3dParentBehaviorCache."
            )
        behavior_cache.verify()
        self.behavior_cache = behavior_cache
        self.selections = tuple(selections)
        if not self.selections or tuple(
            selection.layer_index for selection in self.selections
        ) != tuple(range(len(self.selections))):
            raise DynamicPoints3dFoldError(
                "Native points3d fold selections must cover contiguous decoder "
                "layers."
            )
        anchor_counts = {selection.anchor_count for selection in self.selections}
        if len(anchor_counts) != 1:
            raise DynamicPoints3dFoldError(
                "Native points3d fold selections use inconsistent sparse anchor "
                "counts."
            )
        self.anchor_count = next(iter(anchor_counts))
        self.statistics = MappingProxyType(
            {
                str(name): value.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                for name, value in statistics.items()
            }
        )
        self.sketch_dim = _positive_int(sketch_dim, "sketch_dim")
        self.candidate_count = _positive_int(candidate_count, "candidate_count")
        self.sample_runner_calls = _positive_int(
            sample_runner_calls,
            "sample_runner_calls",
        )
        self.source_cache_key = _digest(source_cache_key, "source_cache_key")
        self.directory = directory
        if cache_key is not None and self.source_cache_key == cache_key:
            raise DynamicPoints3dFoldError(
                "A derived native points3d cache cannot cite itself as its source."
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

    @property
    def layer_count(self) -> int:
        return len(self.selections)

    @staticmethod
    def statistic_key(layer_index: int, name: str) -> str:
        if name not in _STATISTIC_NAMES:
            raise DynamicPoints3dFoldError("Unknown points3d statistic name.")
        return f"layer_{layer_index:03d}_{name}"

    def layer_statistics(self, layer_index: int) -> Mapping[str, Any]:
        index = _nonnegative_int(layer_index, "layer_index")
        if index >= len(self.selections):
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache layer index is out of range."
            )
        return MappingProxyType(
            {
                name: self.statistics[self.statistic_key(index, name)]
                for name in _STATISTIC_NAMES
            }
        )

    def selection(self, layer_index: int) -> NativePoints3dLayerFoldSelection:
        index = _nonnegative_int(layer_index, "layer_index")
        try:
            return self.selections[index]
        except IndexError as exc:
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache layer index is out of range."
            ) from exc

    def _body(self) -> dict[str, Any]:
        return {
            "schema": FOLD_CACHE_SCHEMA,
            "modality": MODALITY_POINTS3D,
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
            "conditioning": "points3d_conditioned_decoder_mlp_down_projection_input",
            "residual_endpoint_source": "decoder_layer_output_hidden_state",
            "selection_policy": (
                "heldout_blind_low_importance_sparse_anchor_regression"
            ),
            "anchor_count": self.anchor_count,
            "sketch_dim": self.sketch_dim,
            "candidate_count": self.candidate_count,
            "sample_runner_calls": self.sample_runner_calls,
            "source_cache_key": self.source_cache_key,
            "statistics_reused_without_parent_forward": True,
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
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache statistics do not cover every "
                "selected layer."
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
                raise DynamicPoints3dFoldError(
                    f"Native points3d layer {selection.layer_index} selection is "
                    "not reproducible from sealed statistics."
                )
        if self._computed_cache_key() != self.cache_key:
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache seal is invalid."
            )
        if parent_graph is not None and (
            parent_graph.fingerprint != self.parent_graph_fingerprint
            or len(parent_graph.layer_indices) != len(self.selections)
        ):
            raise DynamicPoints3dFoldError(
                "Loaded parent graph differs from the native points3d fold cache."
            )
        if parent_model is not None and (
            fingerprint_model_state(parent_model) != self.parent_state_fingerprint
        ):
            raise DynamicPoints3dFoldError(
                "Loaded parent state differs from the native points3d fold cache."
            )

    def save(self, destination: str | os.PathLike[str]) -> Path:
        """Seal selections and statistics as canonical little-endian float32."""

        destination_path = Path(destination).expanduser().absolute()
        if destination_path.exists():
            raise DynamicPoints3dFoldError(
                f"Native points3d fold cache destination exists: "
                f"{destination_path}."
            )
        parent = destination_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=".points3d-fold-cache-stage-", dir=parent)
        ).absolute()
        try:
            blob_path = stage / STATISTICS_BLOB
            layout: dict[str, dict[str, Any]] = {}
            offset = 0
            with blob_path.open("xb") as handle:
                for name in sorted(self.statistics):
                    payload = _canonical_float32_bytes(self.statistics[name])
                    handle.write(payload)
                    layout[name] = {
                        "offset": offset,
                        "byte_count": len(payload),
                    }
                    offset += len(payload)
                handle.flush()
                os.fsync(handle.fileno())
            body = self._body()
            body["statistics_file"] = blob_path.name
            body["statistics_file_sha256"] = _file_sha256(blob_path)
            body["statistics_layout"] = layout
            body["statistics_byte_order"] = "little"
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
        behavior_cache: Points3dParentBehaviorCache,
    ) -> NativePoints3dFoldCache:
        import numpy as np
        import torch

        root = Path(directory).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache manifest is missing."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache manifest is invalid."
            ) from exc
        manifest_sha256 = manifest.pop("manifest_sha256", None)
        if canonical_sha256(manifest) != manifest_sha256:
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache manifest seal is invalid."
            )
        cache_key = manifest.get("cache_key")
        _digest(cache_key, "cache_key")
        if (
            manifest.get("schema") != FOLD_CACHE_SCHEMA
            or manifest.get("behavior_cache_fingerprint") != behavior_cache.fingerprint
        ):
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache is bound to different behavior "
                "evidence."
            )
        blob_path = root / str(manifest.get("statistics_file"))
        if not blob_path.is_file() or _file_sha256(blob_path) != manifest.get(
            "statistics_file_sha256"
        ):
            raise DynamicPoints3dFoldError(
                "Native points3d fold statistics file is invalid."
            )
        descriptors = manifest.get("statistics")
        layout = manifest.get("statistics_layout")
        if (
            not isinstance(descriptors, Mapping)
            or not isinstance(layout, Mapping)
            or set(descriptors) != set(layout)
        ):
            raise DynamicPoints3dFoldError(
                "Native points3d fold statistic layout is incomplete."
            )
        blob = blob_path.read_bytes()
        statistics: dict[str, Any] = {}
        for name, descriptor in descriptors.items():
            placement = layout[name]
            start = int(placement["offset"])
            stop = start + int(placement["byte_count"])
            payload = blob[start:stop]
            if (
                len(payload) != int(descriptor["byte_count"])
                or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
            ):
                raise DynamicPoints3dFoldError(
                    f"Native points3d fold statistic {name} failed its seal."
                )
            array = np.frombuffer(payload, dtype="<f4").reshape(
                tuple(int(size) for size in descriptor["shape"])
            )
            # np.ascontiguousarray would promote a sealed scalar statistic to
            # rank one and silently change the reloaded seal.
            statistics[name] = torch.from_numpy(
                np.array(array, dtype=np.float32, copy=True)
            )
        selections = tuple(
            NativePoints3dLayerFoldSelection(
                **{
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "full_source_coverage",
                        "attention_copied_exactly",
                        "points3d_front_end_copied_exactly",
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
            source_cache_key=manifest["source_cache_key"],
            cache_key=cache_key,
            directory=str(root),
        )
        # The in-memory seal deliberately excludes storage-container fields.
        if result.cache_key != result._computed_cache_key():
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache logical seal differs after reload."
            )
        return result


def derive_native_points3d_fold_cache(
    statistics_cache: Points3dTraceStatisticsCache,
    *,
    destination: str | os.PathLike[str],
    candidate_count: int,
    anchor_count: int = 1,
) -> NativePoints3dFoldCache:
    """Select folds from sealed points3d evidence without a new parent forward."""

    if not isinstance(statistics_cache, Points3dTraceStatisticsCache):
        raise DynamicPoints3dFoldError(
            "statistics_cache must be Points3dTraceStatisticsCache."
        )
    statistics_cache.verify()
    candidate_count = _positive_int(candidate_count, "candidate_count")
    anchor_count = _positive_int(anchor_count, "anchor_count")
    selections = tuple(
        _selection_from_statistics(
            layer_index,
            statistics_cache.layer_statistics(layer_index),
            trace_rows=statistics_cache.trace_rows[layer_index],
            candidate_count=candidate_count,
            anchor_count=anchor_count,
        )
        for layer_index in range(statistics_cache.layer_count)
    )
    derived = NativePoints3dFoldCache(
        parent_graph_fingerprint=statistics_cache.parent_graph_fingerprint,
        parent_state_fingerprint=statistics_cache.parent_state_fingerprint,
        behavior_cache=statistics_cache.behavior_cache,
        selections=selections,
        statistics=statistics_cache.statistics,
        sketch_dim=statistics_cache.sketch_dim,
        candidate_count=candidate_count,
        sample_runner_calls=statistics_cache.sample_runner_calls,
        source_cache_key=statistics_cache.cache_key,
    )
    derived.save(destination)
    return NativePoints3dFoldCache.open(
        destination,
        behavior_cache=statistics_cache.behavior_cache,
    )


class NativePoints3dEvidenceProvider:
    """Expose one immutable fold cache without any live parent forward."""

    def __init__(self, cache: NativePoints3dFoldCache) -> None:
        if not isinstance(cache, NativePoints3dFoldCache):
            raise DynamicPoints3dFoldError("cache must be NativePoints3dFoldCache.")
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
            raise DynamicPoints3dFoldError(
                "Native points3d evidence provider received a different fold "
                "method."
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
            raise DynamicPoints3dFoldError(
                "Native points3d evidence restore observed an inexact child "
                "rollback."
            )
        self.cache.verify()
        self.restore_count += 1


class NativePoints3dFoldOnlyInitializer:
    """Authenticate cache/model binding without mutating the seeded blank."""

    def __init__(self, cache: NativePoints3dFoldCache) -> None:
        self.cache = cache

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        if len(plan.actions) != 1 or plan.actions[0].method_id != METHOD_ID:
            raise DynamicPoints3dFoldError(
                "Native points3d fold-only initialization requires one atomic "
                "model action."
            )
        if len(context.parent_models) != 1:
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache is bound to exactly one parent."
            )
        parent_name = next(iter(context.parent_models))
        parent = context.parent_models[parent_name]
        parent_graph = context.parent_graphs[parent_name]
        self.cache.verify(parent_model=parent, parent_graph=parent_graph)
        if plan.evidence_fingerprint != self.cache.cache_key:
            raise DynamicPoints3dFoldError(
                "Native points3d plan evidence differs from the authenticated "
                "cache."
            )
        return {
            "kind": "native_points3d_fold_only",
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


def _native_points3d_decoder_layers(model: Any) -> tuple[Any, ...]:
    layers = getattr(
        getattr(getattr(model, "language_model", None), "model", None),
        "layers",
        None,
    )
    if layers is None:
        raise DynamicPoints3dFoldError(
            "Native points3d model is missing language_model.model.layers."
        )
    return tuple(layers)


class NativePoints3dFoldTransaction:
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
        selections: tuple[NativePoints3dLayerFoldSelection, ...],
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
            raise DynamicPoints3dFoldError(
                "Native points3d transaction is no longer active."
            )
        if self.context.child_state_fingerprint != self.candidate_state_fingerprint:
            raise DynamicPoints3dFoldError(
                "Native points3d candidate changed before transaction commit."
            )
        self.snapshot.cleanup()
        self.active = False
        return {
            "kind": "native_points3d_atomic_model_fold",
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
            "points3d_front_end_copied_exactly": True,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicPoints3dFoldError(
                "Native points3d transaction is no longer active."
            )
        self.snapshot.restore(_live_state(self.context.child_model))
        restored = self.context.child_state_fingerprint
        exact = restored == self.incumbent_state_fingerprint
        self.snapshot.cleanup()
        self.active = False
        if not exact:
            raise DynamicPoints3dFoldError(
                "Native points3d transaction rollback did not restore the "
                "incumbent."
            )
        return {
            "kind": "native_points3d_atomic_model_fold",
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": restored,
            "cache_key": self.cache_key,
            "rollback_exact": True,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
        }


class NativePoints3dSwiGLUFoldOperator:
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
            raise DynamicPoints3dFoldError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> NativePoints3dFoldTransaction:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is an extra
            raise DynamicPoints3dFoldError(
                "PyTorch is required for native points3d folding."
            ) from exc
        if action.method_id != self.method_id or len(action.sources) != 1:
            raise DynamicPoints3dFoldError(
                "Native points3d operator requires one atomic source action."
            )
        if not isinstance(evidence.payload, NativePoints3dFoldCache):
            raise DynamicPoints3dFoldError(
                "Native points3d operator received the wrong evidence."
            )
        cache = evidence.payload
        parent_name = action.sources[0].parent_name
        if set(context.parent_models) != {parent_name}:
            raise DynamicPoints3dFoldError(
                "Native points3d operator source differs from the loaded parent."
            )
        parent = context.parent_models[parent_name]
        parent_graph = context.parent_graphs[parent_name]
        cache.verify(parent_model=parent, parent_graph=parent_graph)
        if cache.cache_key != evidence.fingerprint:
            raise DynamicPoints3dFoldError(
                "Native points3d evidence payload and fingerprint differ."
            )
        if set(action.target_parameter_paths) != set(
            context.child_graph.parameter_paths
        ):
            raise DynamicPoints3dFoldError(
                "Native points3d atomic action does not own every child parameter."
            )
        source_layers = _native_points3d_decoder_layers(parent)
        target_layers = _native_points3d_decoder_layers(context.child_model)
        if len(source_layers) != len(target_layers) or len(source_layers) != len(
            cache.selections
        ):
            raise DynamicPoints3dFoldError(
                "Native points3d parent, child, and cache decoder depths differ."
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
            raise DynamicPoints3dFoldError(
                "Native points3d parent and child state keys differ."
            )
        overrides: dict[str, Any] = {}
        for selection in cache.selections:
            prefix = f"language_model.model.layers.{selection.layer_index}.mlp"
            gate_key = f"{prefix}.gate_proj.weight"
            up_key = f"{prefix}.up_proj.weight"
            down_key = f"{prefix}.down_proj.weight"
            for key in (gate_key, up_key, down_key):
                if key not in source_state or key not in child_state:
                    snapshot.cleanup()
                    raise DynamicPoints3dFoldError(
                        f"Native points3d state is missing folded tensor {key!r}."
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
                raise DynamicPoints3dFoldError(
                    f"Native points3d layer {selection.layer_index} MLP shapes "
                    "differ from the cached one-neuron target."
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
                        raise DynamicPoints3dFoldError(
                            "Native points3d source/target state shape differs "
                            f"for {name!r}."
                        )
                    if source.dtype != target.dtype:
                        raise DynamicPoints3dFoldError(
                            "Native points3d source/target dtype differs for "
                            f"{name!r}."
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
            raise DynamicPoints3dFoldError(
                "Native points3d fold installed no parent-derived candidate state."
            )
        return NativePoints3dFoldTransaction(
            context=context,
            snapshot=snapshot,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            cache_key=cache.cache_key,
            changed_paths=changed_paths,
            selections=cache.selections,
        )


class NativePoints3dFoldAdmissionEvaluator:
    """Admit the structurally complete cached transaction; behavior is final."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        if (
            context.action.method_id != METHOD_ID
            or not isinstance(context.transaction, NativePoints3dFoldTransaction)
            or not isinstance(context.evidence.payload, NativePoints3dFoldCache)
        ):
            return FoldDecision.reject(
                "Native points3d transaction/evidence type mismatch."
            )
        transaction = context.transaction
        cache = context.evidence.payload
        if (
            transaction.candidate_state_fingerprint
            != context.candidate_state_fingerprint
            or transaction.cache_key != cache.cache_key
            or len(transaction.selections) != len(cache.selections)
        ):
            return FoldDecision.reject(
                "Native points3d cached transaction binding mismatch."
            )
        hidden_size = int(
            context.execution_context.child_model.config.decoder_config.hidden_size
        )
        return FoldDecision.accept(
            "Native points3d state is fully parent-derived with an exact "
            "quantized-point front-end and attention copy plus one cached "
            "coupled-SwiGLU merge per layer; whole-cloud behavior remains "
            "mandatory.",
            {
                "folded_layers": len(cache.selections),
                "removed_parameters": len(cache.selections) * 3 * hidden_size,
                "parent_forward_count": context.evidence.parent_forward_count,
                "heldout_selection_count": 0,
                "points3d_front_end_copied_exactly": True,
            },
        )


@dataclass(frozen=True, slots=True)
class NativePoints3dStackResources:
    cache: NativePoints3dFoldCache
    behavior_gate: Any
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    saver_id: str
    loader_id: str
    snapshot_config: DynamicFoldSnapshotConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cache, NativePoints3dFoldCache):
            raise DynamicPoints3dFoldError("cache must be NativePoints3dFoldCache.")
        # The gate is duck-typed rather than pinned to the input-side class so
        # the output-side gate in ``dynamic_points3d_output`` can drive the same
        # published stack.  Both expose the same behavior-cache binding, the
        # same evaluator surface, and the same sealed manifest fields.
        gate_cache = getattr(self.behavior_gate, "behavior_cache", None)
        if not isinstance(gate_cache, Points3dParentBehaviorCache) or not callable(
            getattr(self.behavior_gate, "evaluate", None)
        ):
            raise DynamicPoints3dFoldError(
                "behavior_gate must expose a Points3dParentBehaviorCache and an "
                "evaluate() entry point."
            )
        if gate_cache.fingerprint != self.cache.behavior_cache.fingerprint:
            raise DynamicPoints3dFoldError(
                "Native points3d behavior gate and fold cache use different "
                "parent outputs."
            )
        for name in ("model_saver", "model_loader"):
            if not callable(getattr(self, name)):
                raise DynamicPoints3dFoldError(f"{name} must be callable.")
        for name in ("saver_id", "loader_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DynamicPoints3dFoldError(f"{name} must be nonempty.")
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise DynamicPoints3dFoldError(
                "snapshot_config must be DynamicFoldSnapshotConfig or None."
            )


class NativePoints3dFoldStackAdapter:
    """Catalog adapter for one authenticated native-points3d cache and target."""

    adapter_id = STACK_ADAPTER_ID
    version = 1
    priority = 240
    target_families = ("native_points3d_chat",)
    # Canonical sorted order, exactly as DynamicTargetRequest normalizes it.
    target_modalities = ("points3d", "text")
    method_ids = (METHOD_ID,)

    def __init__(self, resources: NativePoints3dStackResources) -> None:
        if not isinstance(resources, NativePoints3dStackResources):
            raise DynamicPoints3dFoldError(
                "resources must be NativePoints3dStackResources."
            )
        self.resources = resources
        self.adapter_id = f"{STACK_ADAPTER_ID}.{resources.cache.cache_key[:16]}"

    def _validate(self, target: DynamicTargetSynthesisResult) -> None:
        if not isinstance(target, DynamicTargetSynthesisResult):
            raise DynamicPoints3dFoldError(
                "Native points3d stack requires DynamicTargetSynthesisResult."
            )
        if (
            target.adapter_id != NativePoints3dTargetSynthesisAdapter.adapter_id
            or target.adapter_version != NativePoints3dTargetSynthesisAdapter.version
            or target.graph.family != "native_points3d_chat"
            or target.resource_profile.modalities != ("points3d", "text")
        ):
            raise DynamicPoints3dFoldError(
                "Synthesized target is not a native points3d child."
            )
        requested_cache_key = dict(target.request.configuration).get("fold_cache_key")
        if (
            requested_cache_key is not None
            and requested_cache_key != self.resources.cache.cache_key
        ):
            raise DynamicPoints3dFoldError(
                "Synthesized native points3d target requests a different fold "
                "cache."
            )
        parent_graphs = dict(target.parent_graph_fingerprints)
        if len(parent_graphs) != 1 or next(iter(parent_graphs.values())) != (
            self.resources.cache.parent_graph_fingerprint
        ):
            raise DynamicPoints3dFoldError(
                "Synthesized native points3d target parent graph differs from "
                "the cache."
            )
        source_widths = {
            selection.source_neurons for selection in self.resources.cache.selections
        }
        target_widths = {
            selection.target_neurons for selection in self.resources.cache.selections
        }
        if len(source_widths) != 1 or len(target_widths) != 1:
            raise DynamicPoints3dFoldError(
                "Native points3d fold cache does not describe one uniform "
                "decoder width."
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
            raise DynamicPoints3dFoldError(
                "Native points3d target report lacks a valid decoder hidden size."
            )
        expected_delta = (
            layer_count * 3 * hidden_size * (source_intermediate - target_intermediate)
        )
        if (
            report.get("source_intermediate_size") != source_intermediate
            or report.get("target_intermediate_size") != target_intermediate
            or report.get("decoder_layer_count") != layer_count
            or report.get("exact_parameter_delta") != expected_delta
            or report.get("points3d_front_end_copied_exactly") is not True
            or report.get("hub_loader_used") is not False
        ):
            raise DynamicPoints3dFoldError(
                "Native points3d target does not match the authenticated cached "
                "fold shape."
            )
        if (
            target.resource_profile.unique_parameters
            != target.request.exact_unique_parameters
            or len(target.graph.layer_indices) != len(self.resources.cache.selections)
        ):
            raise DynamicPoints3dFoldError(
                "Native points3d target budget/depth differs from cached fold "
                "evidence."
            )

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            self._validate(target)
        except (DynamicPoints3dFoldError, TypeError, ValueError):
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
                    NativePoints3dSwiGLUFoldOperator(
                        snapshot_config=snapshot,
                    ),
                )
            ),
            evidence_provider=NativePoints3dEvidenceProvider(cache),
            evaluator=NativePoints3dFoldAdmissionEvaluator(),
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
            initializer=NativePoints3dFoldOnlyInitializer(cache),
            snapshot_config=snapshot,
            evidence_fingerprint=cache.cache_key,
        )


def write_native_points3d_dynamic_fold_receipt(
    result: Any,
    cache: NativePoints3dFoldCache,
    behavior_gate: Any,
    *,
    destination: str | os.PathLike[str],
    cache_reused: bool = False,
) -> Mapping[str, Any]:
    """Write one self-hashed receipt after standalone reload acceptance."""

    if not bool(getattr(result, "complete", False)):
        raise DynamicPoints3dFoldError(
            "Only a complete published native points3d fold can receive a receipt."
        )
    if not isinstance(cache_reused, bool):
        raise DynamicPoints3dFoldError("cache_reused must be boolean.")
    artifact = result.published.artifact
    run = result.published.compilation.run
    last_report = getattr(behavior_gate, "last_report", None)
    if last_report is None:
        raise DynamicPoints3dFoldError(
            "Native points3d receipt requires a completed whole-cloud behavior "
            "gate."
        )
    body = {
        "schema": RECEIPT_SCHEMA,
        "modality": MODALITY_POINTS3D,
        "method_id": METHOD_ID,
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
        "fresh_reload_parameters_frozen": artifact.fresh_reload_parameters_frozen,
        "behavior_report": dict(last_report),
        "behavior_report_sha256": canonical_sha256(dict(last_report)),
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
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    path = Path(destination).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive_json(path, receipt)
    return MappingProxyType(receipt)


def points3d_lane_status() -> dict[str, Any]:
    """State what the points3d lane provides and what a later wave must wire."""

    return {
        "schema": "neural-forge-native-points3d-lane-status-v1",
        "modality": MODALITY_POINTS3D,
        "method_id": METHOD_ID,
        "provided": [
            "deterministic sealed quantized XYZ probe suite",
            "authenticated cached parent answers with decoded re-hashing",
            "parent-relative whole-cloud behavior gate with zero parent forwards",
            "points3d-conditioned trace evidence capture and sealing",
            "a heldout-blind sparse-anchor fold selection sealed apart from "
            "the evidence it is derived from",
            "a registered experimental fold method, operator, transaction, and "
            "catalog stack adapter",
            "an inspector that emits MODALITY_INPUT points3d nodes",
            "an optional MODALITY_OUTPUT centroid head reviewed by "
            "neural_forge.dynamic_points3d_output",
        ],
        "not_provided": [
            "a production geometry-capable parent model choice",
            "generative quality on the emitted-output side; the output gate "
            "proves the retention contract, not synthesis",
            "promotion of the fold method beyond experimental status",
        ],
        "model_agnostic_seams": [
            "trace_runner",
            "child_runner",
            "parent_runner",
            "layer_resolver",
            "projection_resolver",
            "inference_execution_contract",
        ],
        "parent_forwards_during_gating": 0,
    }


__all__ = [
    "BEHAVIOR_SCHEMA",
    "FOLD_CACHE_SCHEMA",
    "GATE_ID",
    "GATE_REPORT_SCHEMA",
    "GATE_VERSION",
    "METHOD_ID",
    "POINTS3D_INFERENCE_EXECUTION_CONTRACT",
    "RECEIPT_SCHEMA",
    "STACK_ADAPTER_ID",
    "TRACE_CACHE_SCHEMA",
    "DynamicPoints3dFoldError",
    "NativePoints3dBehaviorGate",
    "NativePoints3dEvidenceProvider",
    "NativePoints3dFoldAdmissionEvaluator",
    "NativePoints3dFoldCache",
    "NativePoints3dFoldOnlyInitializer",
    "NativePoints3dFoldStackAdapter",
    "NativePoints3dFoldTransaction",
    "NativePoints3dLayerFoldSelection",
    "NativePoints3dStackResources",
    "NativePoints3dSwiGLUFoldOperator",
    "Points3dBehaviorGateConfig",
    "Points3dBehaviorRecord",
    "Points3dParentBehaviorCache",
    "Points3dTraceStatisticsCache",
    "build_native_points3d_fold_trace_cache",
    "default_layer_resolver",
    "default_projection_resolver",
    "derive_native_points3d_fold_cache",
    "load_points3d_parent_behavior_cache",
    "normalize_points3d_answer",
    "points3d_lane_status",
    "points3d_parent_output_record",
    "write_native_points3d_dynamic_fold_receipt",
    "write_points3d_parent_output_cache",
]
