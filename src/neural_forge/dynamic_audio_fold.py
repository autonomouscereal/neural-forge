"""Zero-training native-audio folding foundation, model-agnostic by design.

This is the audio mirror of the validated NEO native-vision lane in
:mod:`neural_forge.dynamic_neo_fold`, with one deliberate difference: nothing
here names a parent architecture.  The parent enters through caller-supplied
``trace_runner``, ``child_runner``, and ``layer_resolver`` callables exactly the
way NEO's cache builder and gate take their runners, so choosing a real
audio-capable parent model is a later, separable decision.

What this module provides
-------------------------
* :class:`AudioBehaviorRecord` / :class:`AudioParentBehaviorCache` and
  :func:`load_audio_parent_behavior_cache` -- authenticated cached parent
  answers over a sealed audio probe manifest, with every waveform re-decoded
  and re-hashed under :data:`AUDIO_HASH_BASIS` before a record is trusted.
* :class:`NativeAudioBehaviorGate` -- parent-relative whole-clip review.  All
  parent-correct answers must be retained, no split or split/category
  regression is allowed, the declared inference execution contract must be
  observed, and the parent is never forwarded live.
* :func:`build_native_audio_fold_trace_cache` -- audio-conditioned calibration
  evidence.  The hook pattern (MLP down-projection input plus decoder-layer
  residual endpoint) is modality-agnostic; only the conditioning stimulus
  differs from the vision lane.

What this module deliberately does not provide is a fold selection rule, an
operator, or a registered method id.  This is evidence and review machinery;
the folding mathematics arrives with the method that consumes it.
"""

from __future__ import annotations

import hashlib
import json
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
    AUDIO_HASH_BASIS,
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
from .dynamic_circuit_ir import MODALITY_AUDIO, ModelCircuitGraph, canonical_sha256
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluationContext,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .native_audio_probe import (
    CATEGORIES,
    CLOSED_ANSWERS,
    CODEC_ID,
    SPLITS,
    NativeAudioProbeError,
    authenticate_native_audio_probe,
)


BEHAVIOR_SCHEMA = "neural-forge-native-audio-parent-record-v1"
TRACE_CACHE_SCHEMA = "neural-forge-native-audio-trace-cache-v1"
GATE_REPORT_SCHEMA = "neural-forge-native-audio-behavior-gate-v1"
GATE_ID = "neural_forge.audio.native_audio_parent_relative_behavior"
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
AUDIO_INFERENCE_EXECUTION_CONTRACT = (
    BehaviorInferenceExecutionContract.top_level_forward()
)


class DynamicAudioFoldError(RuntimeError):
    """Native audio evidence, behavior, or trace capture is invalid."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DynamicAudioFoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicAudioFoldError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicAudioFoldError(f"{name} must be a non-negative integer.")
    return value


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
        raise DynamicAudioFoldError("Payload is not canonical JSON.") from exc


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


def normalize_audio_answer(category: str, raw_output: Any) -> str | None:
    """Map one raw child or parent answer onto its closed audio answer set.

    Anything outside the closed set for the category -- prose, an explanation,
    a refusal, a second sentence -- normalizes to ``None`` and counts as a
    non-direct answer, exactly as in the NEO vision lane.
    """

    if category not in CLOSED_ANSWERS:
        raise DynamicAudioFoldError(f"Unsupported audio category {category!r}.")
    if not isinstance(raw_output, str):
        return None
    allowed = CLOSED_ANSWERS[category]
    if category == "tone_count":
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
class AudioBehaviorRecord:
    """One authenticated probe and the cached parent answer bound to it."""

    index: int
    probe_id: str
    split: str
    category: str
    prompt: str
    expected: str
    audio_path: str
    audio_sha256: str
    container_sha256: str
    container_bytes: int
    frames: int
    sample_rate: int
    channels: int
    manifest_record_sha256: str
    parent_raw_output: str
    parent_normalized_prediction: str | None
    parent_direct_answer: bool
    parent_correct: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _nonnegative_int(self.index, "index"))
        for name in ("probe_id", "prompt", "expected", "audio_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DynamicAudioFoldError(f"{name} must be nonempty.")
        if self.split not in SPLITS:
            raise DynamicAudioFoldError("Audio behavior split is invalid.")
        if self.category not in CATEGORIES:
            raise DynamicAudioFoldError("Audio behavior category is invalid.")
        if self.expected not in CLOSED_ANSWERS[self.category]:
            raise DynamicAudioFoldError(
                "Audio expected answer escapes its closed answer set."
            )
        _digest(self.audio_sha256, "audio_sha256")
        _digest(self.container_sha256, "container_sha256")
        _digest(self.manifest_record_sha256, "manifest_record_sha256")
        for name in ("container_bytes", "frames"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("sample_rate", "channels"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if not isinstance(self.parent_raw_output, str):
            raise DynamicAudioFoldError("parent_raw_output must be text.")
        if self.parent_normalized_prediction is not None and not isinstance(
            self.parent_normalized_prediction, str
        ):
            raise DynamicAudioFoldError(
                "parent_normalized_prediction must be text or None."
            )
        if not isinstance(self.parent_direct_answer, bool) or not isinstance(
            self.parent_correct, bool
        ):
            raise DynamicAudioFoldError("Parent direct/correct fields must be boolean.")

    @property
    def media(self) -> MediaReference:
        """Bind the decoded waveform, not its container, to this record."""

        return MediaReference(
            modality=MODALITY_AUDIO,
            content_sha256=self.audio_sha256,
            uri=self.audio_path,
            byte_count=self.container_bytes,
            codec=CODEC_ID,
            hash_basis=AUDIO_HASH_BASIS,
            container_sha256=self.container_sha256,
            attributes={
                "channels": self.channels,
                "frames": self.frames,
                "sample_rate": self.sample_rate,
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
            "audio_sha256": self.audio_sha256,
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
class AudioParentBehaviorCache:
    """Immutable cached parent answers for one sealed audio probe suite."""

    manifest_path: str
    outputs_path: str
    manifest_sha256: str
    outputs_sha256: str
    records: tuple[AudioBehaviorRecord, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        _digest(self.manifest_sha256, "manifest_sha256")
        _digest(self.outputs_sha256, "outputs_sha256")
        _digest(self.fingerprint, "behavior cache fingerprint")
        if not self.records:
            raise DynamicAudioFoldError("Audio behavior cache must contain records.")
        if tuple(record.index for record in self.records) != tuple(
            range(len(self.records))
        ):
            raise DynamicAudioFoldError(
                "Audio behavior records must preserve exact manifest order."
            )
        split_counts = {
            split: sum(record.split == split for record in self.records)
            for split in SPLITS
        }
        if any(count <= 0 for count in split_counts.values()):
            raise DynamicAudioFoldError(
                "Audio behavior cache must contain calibration and held-out "
                "records."
            )

    @property
    def calibration(self) -> tuple[AudioBehaviorRecord, ...]:
        return tuple(record for record in self.records if record.split == "calibration")

    @property
    def heldout(self) -> tuple[AudioBehaviorRecord, ...]:
        return tuple(record for record in self.records if record.split == "heldout")

    def cases(self) -> tuple[MultimodalWholeOutputCase, ...]:
        return tuple(record.as_case() for record in self.records)

    def verify(self) -> None:
        """Re-authenticate the sealed sources this cache was derived from."""

        manifest = Path(self.manifest_path)
        outputs = Path(self.outputs_path)
        if not manifest.is_file() or not outputs.is_file():
            raise DynamicAudioFoldError("Audio behavior cache sources are missing.")
        if (
            _file_sha256(manifest) != self.manifest_sha256
            or _file_sha256(outputs) != self.outputs_sha256
        ):
            raise DynamicAudioFoldError(
                "Audio behavior cache sources changed after authentication."
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
            raise DynamicAudioFoldError("Audio behavior cache fingerprint is invalid.")


def audio_parent_output_record(
    probe: Any,
    raw_output: str,
) -> dict[str, Any]:
    """Build one canonical parent-output cache line for an authenticated probe."""

    if not isinstance(raw_output, str):
        raise DynamicAudioFoldError("Parent raw output must be text.")
    normalized = normalize_audio_answer(probe.category, raw_output)
    return {
        "schema": BEHAVIOR_SCHEMA,
        "index": probe.index,
        "probe_id": probe.probe_id,
        "record_sha256": probe.record_sha256,
        "category": probe.category,
        "split": probe.split,
        "expected": probe.answer,
        "audio_sha256": probe.audio_sha256,
        "normalized_prediction": normalized,
        "direct_answer": normalized is not None,
        "correct": normalized == probe.answer,
        "raw_output": raw_output,
        "raw_output_sha256": hashlib.sha256(
            raw_output.encode("utf-8")
        ).hexdigest(),
    }


def write_audio_parent_output_cache(
    manifest_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    parent_runner: Callable[[Any], str],
) -> Path:
    """Run a parent once per probe and seal its answers as a JSONL cache.

    This is the only place in the audio lane that touches a parent model for
    behavior.  Everything downstream -- gating, retention, promotion --
    replays this file and performs zero parent forwards.
    """

    if not callable(parent_runner):
        raise DynamicAudioFoldError("parent_runner must be callable.")
    _manifest, probes = authenticate_native_audio_probe(manifest_path)
    target = Path(destination).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for probe in probes:
        raw_output = parent_runner(probe)
        lines.append(_canonical_json(audio_parent_output_record(probe, raw_output)))
    with target.open("xb") as handle:
        for line in lines:
            handle.write(line + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def load_audio_parent_behavior_cache(
    manifest_path: str | os.PathLike[str],
    outputs_path: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str | None = None,
    expected_outputs_sha256: str | None = None,
) -> AudioParentBehaviorCache:
    """Authenticate a sealed probe manifest and its cached parent answers."""

    manifest = Path(manifest_path).expanduser().resolve()
    outputs = Path(outputs_path).expanduser().resolve()
    if not manifest.is_file() or not outputs.is_file():
        raise DynamicAudioFoldError("Audio manifest/output cache paths must be files.")
    manifest_sha = _file_sha256(manifest)
    outputs_sha = _file_sha256(outputs)
    if expected_manifest_sha256 is not None and manifest_sha != _digest(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    ):
        raise DynamicAudioFoldError("Audio manifest differs from the pinned digest.")
    if expected_outputs_sha256 is not None and outputs_sha != _digest(
        expected_outputs_sha256,
        "expected_outputs_sha256",
    ):
        raise DynamicAudioFoldError(
            "Audio parent outputs differ from the pinned digest."
        )
    try:
        _payload, probes = authenticate_native_audio_probe(manifest)
    except NativeAudioProbeError as exc:
        raise DynamicAudioFoldError(
            f"Audio probe manifest authentication failed: {exc}"
        ) from exc
    raw_lines = outputs.read_bytes()
    if raw_lines and not raw_lines.endswith(b"\n"):
        raise DynamicAudioFoldError("Audio parent output cache is truncated.")
    lines = raw_lines.splitlines()
    if len(lines) != len(probes):
        raise DynamicAudioFoldError(
            "Audio parent output count differs from the manifest."
        )
    records = []
    for index, (probe, line) in enumerate(zip(probes, lines, strict=True)):
        try:
            cached = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicAudioFoldError(
                f"Audio output line {index + 1} is invalid."
            ) from exc
        if not isinstance(cached, Mapping):
            raise DynamicAudioFoldError("Audio output record is not an object.")
        raw_output = cached.get("raw_output")
        normalized = normalize_audio_answer(probe.category, raw_output)
        required = {
            "schema": BEHAVIOR_SCHEMA,
            "index": index,
            "probe_id": probe.probe_id,
            "record_sha256": probe.record_sha256,
            "category": probe.category,
            "split": probe.split,
            "expected": probe.answer,
            "audio_sha256": probe.audio_sha256,
            "normalized_prediction": normalized,
            "direct_answer": normalized is not None,
            "correct": normalized == probe.answer,
        }
        if any(cached.get(key) != value for key, value in required.items()):
            raise DynamicAudioFoldError(
                f"Audio cached parent record {index} failed authentication."
            )
        if (
            cached.get("raw_output_sha256")
            != hashlib.sha256(str(raw_output).encode("utf-8")).hexdigest()
        ):
            raise DynamicAudioFoldError(
                f"Audio cached parent output hash failed for record {index}."
            )
        records.append(
            AudioBehaviorRecord(
                index=index,
                probe_id=probe.probe_id,
                split=probe.split,
                category=probe.category,
                prompt=probe.prompt,
                expected=probe.answer,
                audio_path=probe.path,
                audio_sha256=probe.audio_sha256,
                container_sha256=probe.container_sha256,
                container_bytes=probe.container_bytes,
                frames=probe.frames,
                sample_rate=probe.sample_rate,
                channels=probe.channels,
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
    result = AudioParentBehaviorCache(
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
class AudioBehaviorGateConfig:
    """Parent-relative acceptance thresholds for whole-clip audio review."""

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
                raise DynamicAudioFoldError(f"{name} must be boolean.")
        for name, allowed in (
            ("required_splits", frozenset(SPLITS)),
            ("required_categories", frozenset(CATEGORIES)),
        ):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise DynamicAudioFoldError(f"{name} must be nonempty and unique.")
            if any(value not in allowed for value in values):
                raise DynamicAudioFoldError(f"{name} contains an unsupported value.")
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


class NativeAudioBehaviorGate:
    """Parent-relative whole-clip gate using cached answers as canonical."""

    def __init__(
        self,
        behavior_cache: AudioParentBehaviorCache,
        *,
        child_runner: Callable[[Any, AudioBehaviorRecord], str],
        config: AudioBehaviorGateConfig | None = None,
        child_runner_identity: BehaviorCallableIdentity | None = None,
        inference_execution_contract: (
            BehaviorInferenceExecutionContract | None
        ) = None,
    ) -> None:
        if not isinstance(behavior_cache, AudioParentBehaviorCache):
            raise DynamicAudioFoldError(
                "behavior_cache must be AudioParentBehaviorCache."
            )
        if not callable(child_runner):
            raise DynamicAudioFoldError("child_runner must be callable.")
        self.behavior_cache = behavior_cache
        self.child_runner = child_runner
        self.config = config or AudioBehaviorGateConfig()
        if not isinstance(self.config, AudioBehaviorGateConfig):
            raise DynamicAudioFoldError("config must be AudioBehaviorGateConfig.")
        self.execution_contract = (
            AUDIO_INFERENCE_EXECUTION_CONTRACT
            if inference_execution_contract is None
            else inference_execution_contract
        )
        if not isinstance(
            self.execution_contract,
            BehaviorInferenceExecutionContract,
        ):
            raise DynamicAudioFoldError(
                "inference_execution_contract must be "
                "BehaviorInferenceExecutionContract or None."
            )
        self.last_report: Mapping[str, Any] | None = None
        self.evaluation_count = 0
        if child_runner_identity is None:
            child_runner_identity = identify_behavior_callable(
                child_runner,
                implementation_id="neural_forge.audio.native_audio.child_record_runner",
                version=1,
            )
        if not isinstance(child_runner_identity, BehaviorCallableIdentity):
            raise DynamicAudioFoldError(
                "child_runner_identity must be BehaviorCallableIdentity."
            )
        scorer_identity = identify_behavior_callable(
            normalize_audio_answer,
            implementation_id="neural_forge.audio.native_audio.closed_answer_scorer",
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

    def _coverage_counts(self) -> tuple[dict[str, int], dict[str, int], tuple[str, ...]]:
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
                "modality": MODALITY_AUDIO,
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
                "Native audio task-score parent-correct coverage is incomplete: "
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
            contract=self.execution_contract,
        ) as execution_counts:
            for record in self.behavior_cache.records:
                raw_output = self.child_runner(model, record)
                if not isinstance(raw_output, str):
                    raise DynamicAudioFoldError(
                        f"Child runner returned non-text for {record.probe_id}."
                    )
                normalized = normalize_audio_answer(record.category, raw_output)
                rows.append(
                    {
                        "probe_id": record.probe_id,
                        "split": record.split,
                        "category": record.category,
                        "expected": record.expected,
                        "audio_sha256": record.audio_sha256,
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
            raise DynamicAudioFoldError(
                "Native audio child runner returned without executing the "
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
            "modality": MODALITY_AUDIO,
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
                "Native audio whole-clip behavior regressed: "
                + ", ".join(sorted(set(failures))),
                flat_metrics,
            )
        return FoldDecision.accept(
            "Native audio child retained every cached parent-correct answer "
            "without split/category/direct-answer regression.",
            flat_metrics,
        )

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        return self.evaluate_model(context.execution_context.child_model)

    def __call__(self, model: Any) -> FoldDecision:
        return self.evaluate_model(model)


class _AudioTraceAccumulator:
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
            raise DynamicAudioFoldError(
                f"Audio layer {self.layer_index} observed inconsistent widths."
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
                raise DynamicAudioFoldError(
                    "Decoder layer returned an empty endpoint tuple."
                )
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise DynamicAudioFoldError(
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
            raise DynamicAudioFoldError(
                f"Audio layer {self.layer_index} captured incomplete activation/"
                "residual endpoint rows."
            )
        down = down_weight.detach()
        if down.ndim != 2 or int(down.shape[1]) != int(self.sum_abs.shape[0]):
            raise DynamicAudioFoldError(
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
    raise DynamicAudioFoldError(
        "Could not resolve decoder layers; supply an explicit layer_resolver."
    )


def default_projection_resolver(layer: Any, index: int) -> Any:
    """Resolve the observable MLP down projection of one decoder layer."""

    projection = getattr(getattr(layer, "mlp", None), "down_proj", None)
    if projection is None or not callable(
        getattr(projection, "register_forward_pre_hook", None)
    ):
        raise DynamicAudioFoldError(
            f"Layer {index} lacks an observable MLP down projection; supply an "
            "explicit projection_resolver."
        )
    return projection


class AudioTraceStatisticsCache:
    """Sealed audio-conditioned calibration evidence for one parent state.

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
        behavior_cache: AudioParentBehaviorCache,
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
        if not isinstance(behavior_cache, AudioParentBehaviorCache):
            raise DynamicAudioFoldError(
                "behavior_cache must be AudioParentBehaviorCache."
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
            raise DynamicAudioFoldError(
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
            raise DynamicAudioFoldError("Unknown audio statistic name.")
        return f"layer_{layer_index:03d}_{name}"

    def layer_statistics(self, layer_index: int) -> Mapping[str, Any]:
        index = _nonnegative_int(layer_index, "layer_index")
        if index >= self.layer_count:
            raise DynamicAudioFoldError("Audio cache layer index is out of range.")
        return MappingProxyType(
            {
                name: self.statistics[self.statistic_key(index, name)]
                for name in _STATISTIC_NAMES
            }
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": TRACE_CACHE_SCHEMA,
            "modality": MODALITY_AUDIO,
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
            "conditioning": "audio_conditioned_decoder_mlp_down_projection_input",
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
            raise DynamicAudioFoldError(
                "Audio cache statistics do not cover every observed layer."
            )
        if self._computed_cache_key() != self.cache_key:
            raise DynamicAudioFoldError("Audio trace cache seal is invalid.")
        if parent_graph is not None and (
            parent_graph.fingerprint != self.parent_graph_fingerprint
            or len(parent_graph.layer_indices) != self.layer_count
        ):
            raise DynamicAudioFoldError(
                "Loaded parent graph differs from the audio trace cache."
            )
        if parent_model is not None and (
            fingerprint_model_state(parent_model) != self.parent_state_fingerprint
        ):
            raise DynamicAudioFoldError(
                "Loaded parent state differs from the audio trace cache."
            )

    def save(self, destination: str | os.PathLike[str]) -> Path:
        """Seal statistics as canonical little-endian float32 plus a manifest."""

        destination_path = Path(destination).expanduser().absolute()
        if destination_path.exists():
            raise DynamicAudioFoldError(
                f"Audio trace cache destination exists: {destination_path}."
            )
        parent = destination_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=".audio-trace-cache-stage-", dir=parent)
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
        behavior_cache: AudioParentBehaviorCache,
    ) -> AudioTraceStatisticsCache:
        import numpy as np
        import torch

        root = Path(directory).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DynamicAudioFoldError("Audio trace cache manifest is missing.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicAudioFoldError(
                "Audio trace cache manifest is invalid."
            ) from exc
        manifest_sha256 = manifest.pop("manifest_sha256", None)
        if canonical_sha256(manifest) != manifest_sha256:
            raise DynamicAudioFoldError("Audio trace cache manifest seal is invalid.")
        cache_key = manifest.get("cache_key")
        _digest(cache_key, "cache_key")
        if (
            manifest.get("schema") != TRACE_CACHE_SCHEMA
            or manifest.get("behavior_cache_fingerprint") != behavior_cache.fingerprint
        ):
            raise DynamicAudioFoldError(
                "Audio trace cache is bound to different behavior evidence."
            )
        blob_path = root / str(manifest.get("statistics_file"))
        if not blob_path.is_file() or _file_sha256(blob_path) != manifest.get(
            "statistics_file_sha256"
        ):
            raise DynamicAudioFoldError("Audio trace statistics file is invalid.")
        descriptors = manifest.get("statistics")
        layout = manifest.get("statistics_layout")
        if (
            not isinstance(descriptors, Mapping)
            or not isinstance(layout, Mapping)
            or set(descriptors) != set(layout)
        ):
            raise DynamicAudioFoldError("Audio trace statistic layout is incomplete.")
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
                raise DynamicAudioFoldError(
                    f"Audio trace statistic {name} failed its seal."
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
            raise DynamicAudioFoldError(
                "Audio trace cache logical seal differs after reload."
            )
        return result


def build_native_audio_fold_trace_cache(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    behavior_cache: AudioParentBehaviorCache,
    *,
    trace_runner: Callable[[Any, AudioBehaviorRecord], Any],
    destination: str | os.PathLike[str],
    layer_resolver: Callable[[Any], Sequence[Any]] = default_layer_resolver,
    projection_resolver: Callable[[Any, int], Any] = default_projection_resolver,
    sketch_dim: int = 64,
) -> AudioTraceStatisticsCache:
    """Capture audio-conditioned calibration traces once and seal them.

    The hook pattern is the modality-agnostic half of NEO's vision cache
    builder: a forward pre-hook on each layer's MLP down projection records the
    activation rows a fold would have to reproduce, and a forward hook on the
    layer records the residual endpoint energy those rows feed.  Only the
    stimulus is audio, and only through ``trace_runner``, which the caller owns.

    Held-out probes are never traced, the parent is asserted to be
    zero-gradient before capture, and both the parameter and runtime state
    fingerprints must be unchanged afterwards.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is an extra
        raise DynamicAudioFoldError(
            "PyTorch is required for audio trace capture."
        ) from exc
    if not isinstance(parent_graph, ModelCircuitGraph):
        raise DynamicAudioFoldError("parent_graph must be ModelCircuitGraph.")
    for name, value in (
        ("trace_runner", trace_runner),
        ("layer_resolver", layer_resolver),
        ("projection_resolver", projection_resolver),
    ):
        if not callable(value):
            raise DynamicAudioFoldError(f"{name} must be callable.")
    behavior_cache.verify()
    sketch_dim = _positive_int(sketch_dim, "sketch_dim")
    layers = tuple(layer_resolver(parent_model))
    if not layers:
        raise DynamicAudioFoldError("Audio trace capture found no decoder layers.")
    if len(layers) != len(parent_graph.layer_indices):
        raise DynamicAudioFoldError(
            "Parent graph and runtime decoder depth differ."
        )
    if any(parameter.grad is not None for parameter in parent_model.parameters()):
        raise DynamicAudioFoldError(
            "Audio trace capture requires a zero-gradient parent."
        )
    state_before = fingerprint_model_state(parent_model)
    runtime_before = fingerprint_model_runtime_state(parent_model)
    module_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    accumulators = [
        _AudioTraceAccumulator(index, sketch_dim) for index in range(len(layers))
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
                raise DynamicAudioFoldError(
                    f"Layer {index} projection cannot be observed."
                )

            def capture(
                _module: Any,
                args: tuple[Any, ...],
                *,
                layer_index: int = index,
            ) -> None:
                if not args:
                    raise DynamicAudioFoldError(
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
        raise DynamicAudioFoldError("Audio trace capture mutated the parent model.")
    statistics: dict[str, Any] = {}
    trace_rows = []
    for index, (projection, accumulator) in enumerate(
        zip(projections, accumulators, strict=True)
    ):
        weight = getattr(projection, "weight", None)
        if weight is None:
            raise DynamicAudioFoldError(
                f"Layer {index} projection exposes no weight tensor."
            )
        values = accumulator.finalize(weight)
        for name, tensor in values.items():
            statistics[AudioTraceStatisticsCache.statistic_key(index, name)] = tensor
        trace_rows.append(accumulator.rows)
    cache = AudioTraceStatisticsCache(
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
    return AudioTraceStatisticsCache.open(path, behavior_cache=behavior_cache)


def audio_lane_status() -> dict[str, Any]:
    """State what the audio lane provides and what a later wave must wire."""

    return {
        "schema": "neural-forge-native-audio-lane-status-v1",
        "modality": MODALITY_AUDIO,
        "provided": [
            "deterministic sealed PCM16 probe suite",
            "authenticated cached parent answers with decoded re-hashing",
            "parent-relative whole-clip behavior gate with zero parent forwards",
            "audio-conditioned trace evidence capture and sealing",
        ],
        "not_provided": [
            "a fold selection rule or operator",
            "a registered fold method id",
            "an inspector that emits MODALITY_INPUT audio nodes",
            "a concrete audio-capable parent model choice",
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
    "AUDIO_INFERENCE_EXECUTION_CONTRACT",
    "BEHAVIOR_SCHEMA",
    "GATE_ID",
    "GATE_REPORT_SCHEMA",
    "GATE_VERSION",
    "TRACE_CACHE_SCHEMA",
    "AudioBehaviorGateConfig",
    "AudioBehaviorRecord",
    "AudioParentBehaviorCache",
    "AudioTraceStatisticsCache",
    "DynamicAudioFoldError",
    "NativeAudioBehaviorGate",
    "audio_lane_status",
    "audio_parent_output_record",
    "build_native_audio_fold_trace_cache",
    "default_layer_resolver",
    "default_projection_resolver",
    "load_audio_parent_behavior_cache",
    "normalize_audio_answer",
    "write_audio_parent_output_cache",
]
