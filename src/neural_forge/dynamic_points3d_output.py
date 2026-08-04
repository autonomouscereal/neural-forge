"""Output-side review for children that EMIT quantized geometry.

Every lane before this one ends in a text head: media enters on the input side
and the reviewed answer is a string.  This module is the other half.  It
reviews a child whose output is itself decoded media -- the quantized XYZ
centroid emitted by the ``CircuitKind.MODALITY_OUTPUT`` node that
:class:`neural_forge.dynamic_model_inspector.NativePoints3dCircuitInspector`
compiles when ``points_config.emit_centroid`` is declared.

The contract, and its four moving parts
---------------------------------------
1. **A sealed reference-output store.**
   :func:`write_points3d_parent_emission_cache` runs the parent once per probe,
   stores every emitted payload as its own ``NFPC1`` container, and seals the
   container digest, the byte count, and the decoded-content digest under
   :data:`POINTS3D_HASH_BASIS`.  Nothing downstream re-runs the parent.
2. **A declared, fingerprinted decode contract.**
   A raw emitted tensor is not reviewable; only a *decoded payload* is.
   :func:`quantize_emitted_points3d` is that decode step, and its
   :class:`BehaviorCallableIdentity` is sealed into the gate manifest as an
   explicit component, exactly as
   :class:`BehaviorInferenceExecutionContract` seals the inference entrypoint.
3. **Parent-relative retention on a closed answer set.**
   The emitted payload is compared by content digest, not by tolerance.  This
   is deliberate and it is the whole reason the first executable output proof
   uses a *quantized* target: on a closed lattice, "did the child emit the same
   geometry?" is decidable.  Open generation is explicitly out of scope and is
   routed to manual review (see :func:`emitted_output_contract_status`).
4. **Non-text regression notions.**
   ``fragment_like`` and ``eos_margin`` are text ideas.  The output-side
   regressions this gate can see are emptiness, decode failure, and
   content-retention loss, thresholded in
   :class:`Points3dEmittedOutputGateConfig`.

Honest limit
------------
This proves the output-side *contract*: sealing, decoding, content binding,
parent-relative retention, coverage, and the fresh-reload replay of all of it.
It proves nothing about generative quality.  A gate that passes here means the
child reproduced the parent's emitted payloads exactly on the reviewed probes,
not that either model emits good geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .dynamic_behavior_gate import (
    POINTS3D_HASH_BASIS,
    BehaviorCallableIdentity,
    BehaviorInferenceExecutionContract,
    MediaReference,
    MultimodalCandidateOutput,
    MultimodalWholeOutputCase,
    TaskScoreCoveragePolicy,
    build_multimodal_behavior_gate_binding,
    execution_contract_satisfied,
    expose_behavior_gate_binding,
    identify_behavior_callable,
    observe_model_execution,
    points3d_content_sha256,
)
from .dynamic_circuit_ir import (
    CircuitKind,
    MODALITY_POINTS3D,
    ModelCircuitGraph,
    canonical_sha256,
    emitted_media_interfaces,
)
from .dynamic_fold_executor import FoldDecision, FoldEvaluationContext
from .dynamic_points3d_fold import (
    Points3dBehaviorRecord,
    Points3dParentBehaviorCache,
)
from .native_points3d_probe import (
    CATEGORIES,
    CODEC_ID,
    COMPONENTS,
    POINT_DTYPE,
    PROBE_SCALE,
    PROBE_ZERO_POINT,
    SPLITS,
    NativePoints3dProbeError,
    authenticate_native_points3d_probe,
    read_points3d_cloud,
    write_points3d_cloud,
)


EMISSION_SCHEMA = "neural-forge-native-points3d-parent-emission-v1"
EMISSION_DECODER_SCHEMA = "neural-forge-emitted-media-decode-contract-v1"
OUTPUT_GATE_REPORT_SCHEMA = "neural-forge-native-points3d-emitted-output-gate-v1"
OUTPUT_GATE_ID = "neural_forge.points3d.native_points3d_parent_relative_emission"
OUTPUT_GATE_VERSION = 1
EMITTED_CODEC_ID = CODEC_ID
EMISSIONS_DIRECTORY = "emissions"
EMISSIONS_INDEX = "emissions.jsonl"
# Emitted child payloads are reviewed by content and never stored, so their
# references carry this non-filesystem scheme instead of a temp path.  Binding
# a rollout to an absolute path is exactly the reproducibility debt NOTES-W6B
# section 6 item 1 recorded against the existing receipts.
CHILD_EMISSION_URI_SCHEME = "memory://neural-forge/points3d-emission"

POINTS3D_EMISSION_EXECUTION_CONTRACT = (
    BehaviorInferenceExecutionContract.top_level_forward()
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INT16_MINIMUM = -(2**15)
_INT16_MAXIMUM = 2**15 - 1


class DynamicPoints3dOutputError(RuntimeError):
    """Emitted-geometry evidence, decoding, or review is invalid."""


class Points3dEmissionDecodeError(DynamicPoints3dOutputError):
    """One emitted payload could not be decoded onto the declared lattice.

    This is a *reviewable* condition rather than a crash: a child whose head
    emits NaN, the wrong rank, or values off the lattice has regressed in a way
    the gate must report and threshold, not swallow.
    """


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DynamicPoints3dOutputError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicPoints3dOutputError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicPoints3dOutputError(f"{name} must be a non-negative integer.")
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
        raise DynamicPoints3dOutputError("Payload is not canonical JSON.") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def quantize_emitted_points3d(
    values: Any,
    *,
    scale: Sequence[float] = PROBE_SCALE,
    zero_point: Sequence[int] = PROBE_ZERO_POINT,
) -> np.ndarray:
    """Decode one raw emitted head output onto the declared int16 lattice.

    This is the gate's decode contract and its identity is sealed into the
    manifest.  It is the exact inverse of
    :meth:`neural_forge.raw_modality_tokens.PointQuantization.dequantize`:
    ``q = round(x / scale) + zero_point``.  Rounding is NumPy's half-to-even,
    which is deterministic and platform-independent.

    Anything the lattice cannot represent -- a non-finite value, a wrong rank,
    a coordinate outside int16 after quantization -- raises
    :class:`Points3dEmissionDecodeError` so the caller can count it as a
    reviewable decode failure instead of sealing a silently clipped payload.
    """

    array = np.asarray(_detached(values), dtype=np.float64)
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or int(array.shape[1]) != COMPONENTS or array.shape[0] == 0:
        raise Points3dEmissionDecodeError(
            "An emitted points3d payload must decode to (points, 3) coordinates."
        )
    if not bool(np.isfinite(array).all()):
        raise Points3dEmissionDecodeError(
            "An emitted points3d payload contains non-finite coordinates."
        )
    axes = np.asarray([float(value) for value in scale], dtype=np.float64)
    origins = np.asarray([int(value) for value in zero_point], dtype=np.float64)
    if axes.shape != (COMPONENTS,) or origins.shape != (COMPONENTS,):
        raise DynamicPoints3dOutputError(
            "Emitted points3d quantization needs three scales and three zero "
            "points."
        )
    if not bool((axes > 0.0).all()):
        raise DynamicPoints3dOutputError(
            "Emitted points3d scale values must be positive."
        )
    quantized = np.rint(array / axes) + origins
    if not bool(
        ((quantized >= _INT16_MINIMUM) & (quantized <= _INT16_MAXIMUM)).all()
    ):
        raise Points3dEmissionDecodeError(
            "An emitted points3d coordinate falls outside the int16 lattice."
        )
    return np.ascontiguousarray(quantized.astype(np.int16))


def _detached(values: Any) -> Any:
    detach = getattr(values, "detach", None)
    if callable(detach):
        return detach().to(device="cpu").numpy()
    return values


def emitted_points3d_payload(points: Any) -> bytes:
    """Serialize a quantized emitted array into its canonical wire bytes."""

    array = np.asarray(points)
    if array.dtype != np.int16 or array.ndim != 2 or int(array.shape[1]) != COMPONENTS:
        raise DynamicPoints3dOutputError(
            "An emitted points3d payload must be an int16 (points, 3) array."
        )
    return np.ascontiguousarray(array.astype("<i2", copy=False)).tobytes()


def emitted_points3d_media(
    points: Any,
    *,
    uri: str,
    scale: Sequence[float] = PROBE_SCALE,
    zero_point: Sequence[int] = PROBE_ZERO_POINT,
    container_sha256: str | None = None,
    byte_count: int | None = None,
) -> MediaReference:
    """Bind one emitted quantized payload as a content-addressed reference."""

    array = np.asarray(points)
    payload = emitted_points3d_payload(array)
    point_count = _positive_int(int(array.shape[0]), "emitted point_count")
    axes = tuple(float(value) for value in scale)
    origins = tuple(int(value) for value in zero_point)
    return MediaReference(
        modality=MODALITY_POINTS3D,
        content_sha256=points3d_content_sha256(
            POINT_DTYPE,
            point_count,
            axes,
            origins,
            payload,
        ),
        uri=uri,
        byte_count=len(payload) if byte_count is None else byte_count,
        codec=EMITTED_CODEC_ID,
        hash_basis=POINTS3D_HASH_BASIS,
        container_sha256=container_sha256,
        attributes={
            "components": COMPONENTS,
            "dtype": POINT_DTYPE,
            "emitted": True,
            "point_count": point_count,
            "scale_x": axes[0],
            "scale_y": axes[1],
            "scale_z": axes[2],
            "zero_x": origins[0],
            "zero_y": origins[1],
            "zero_z": origins[2],
        },
    )


@dataclass(frozen=True, slots=True)
class Points3dEmissionRecord:
    """One authenticated probe and the cached parent *emission* bound to it."""

    behavior: Points3dBehaviorRecord
    emitted_path: str
    emitted_sha256: str
    emitted_container_sha256: str
    emitted_container_bytes: int
    emitted_point_count: int
    components: int
    dtype: str
    scale: tuple[float, float, float]
    zero_point: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.behavior, Points3dBehaviorRecord):
            raise DynamicPoints3dOutputError(
                "behavior must be Points3dBehaviorRecord."
            )
        for name in ("emitted_path", "dtype"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DynamicPoints3dOutputError(f"{name} must be nonempty.")
        _digest(self.emitted_sha256, "emitted_sha256")
        _digest(self.emitted_container_sha256, "emitted_container_sha256")
        for name in (
            "emitted_container_bytes",
            "emitted_point_count",
            "components",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.components != COMPONENTS:
            raise DynamicPoints3dOutputError(
                "An emitted points3d payload must carry three components."
            )
        object.__setattr__(
            self,
            "scale",
            tuple(float(value) for value in self.scale),
        )
        object.__setattr__(
            self,
            "zero_point",
            tuple(int(value) for value in self.zero_point),
        )

    @property
    def index(self) -> int:
        return self.behavior.index

    @property
    def probe_id(self) -> str:
        return self.behavior.probe_id

    @property
    def split(self) -> str:
        return self.behavior.split

    @property
    def category(self) -> str:
        return self.behavior.category

    @property
    def emitted_media(self) -> MediaReference:
        """Bind the decoded emitted payload, not its container, to this record."""

        return MediaReference(
            modality=MODALITY_POINTS3D,
            content_sha256=self.emitted_sha256,
            uri=self.emitted_path,
            byte_count=self.emitted_container_bytes,
            codec=EMITTED_CODEC_ID,
            hash_basis=POINTS3D_HASH_BASIS,
            container_sha256=self.emitted_container_sha256,
            attributes={
                "components": self.components,
                "dtype": self.dtype,
                "emitted": True,
                "point_count": self.emitted_point_count,
                "scale_x": self.scale[0],
                "scale_y": self.scale[1],
                "scale_z": self.scale[2],
                "zero_x": self.zero_point[0],
                "zero_y": self.zero_point[1],
                "zero_z": self.zero_point[2],
            },
        )

    def as_case(self) -> MultimodalWholeOutputCase:
        """Express the record with media on BOTH sides of the review.

        This is the shape the media-capable case schema was built for and that
        no lane could populate before: ``inputs`` carries the consumed cloud and
        ``parent_output`` carries the emitted cloud, so
        :func:`multimodal_case_coverage` reports a nonzero
        ``media_output_count`` for the first time.
        """

        emitted = self.emitted_media
        return MultimodalWholeOutputCase(
            case_id=self.probe_id,
            prompt=self.behavior.prompt,
            parent_output=emitted,
            inputs=(self.behavior.media,),
            expected=emitted,
            category=self.category,
            critical_nonempty=True,
            split=self.split,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "probe_id": self.probe_id,
            "split": self.split,
            "category": self.category,
            "input_points_sha256": self.behavior.points_sha256,
            "emitted_sha256": self.emitted_sha256,
            "emitted_container_sha256": self.emitted_container_sha256,
            "emitted_container_bytes": self.emitted_container_bytes,
            "emitted_point_count": self.emitted_point_count,
            "components": self.components,
            "dtype": self.dtype,
            "scale": list(self.scale),
            "zero_point": list(self.zero_point),
        }

    def verify(self) -> None:
        """Re-decode and re-hash the stored emission before trusting it."""

        path = Path(self.emitted_path)
        if not path.is_file():
            raise DynamicPoints3dOutputError(
                f"Sealed points3d emission is missing: {path}."
            )
        if (
            _file_sha256(path) != self.emitted_container_sha256
            or path.stat().st_size != self.emitted_container_bytes
        ):
            raise DynamicPoints3dOutputError(
                f"Sealed points3d emission container changed: {path}."
            )
        points, quantization = read_points3d_cloud(path)
        payload = emitted_points3d_payload(points)
        digest = points3d_content_sha256(
            POINT_DTYPE,
            int(points.shape[0]),
            quantization.scale,
            quantization.zero_point,
            payload,
        )
        if (
            digest != self.emitted_sha256
            or int(points.shape[0]) != self.emitted_point_count
            or tuple(quantization.scale) != self.scale
            or tuple(quantization.zero_point) != self.zero_point
        ):
            raise DynamicPoints3dOutputError(
                f"Decoded points3d emission authentication failed for "
                f"{self.probe_id}."
            )


@dataclass(frozen=True, slots=True)
class Points3dParentEmissionCache:
    """Immutable cached parent emissions for one sealed points3d probe suite."""

    behavior_cache: Points3dParentBehaviorCache
    directory: str
    index_sha256: str
    records: tuple[Points3dEmissionRecord, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.behavior_cache, Points3dParentBehaviorCache):
            raise DynamicPoints3dOutputError(
                "behavior_cache must be Points3dParentBehaviorCache."
            )
        _digest(self.index_sha256, "index_sha256")
        _digest(self.fingerprint, "emission cache fingerprint")
        if not self.records:
            raise DynamicPoints3dOutputError(
                "Points3d emission cache must contain records."
            )
        if tuple(record.index for record in self.records) != tuple(
            range(len(self.records))
        ):
            raise DynamicPoints3dOutputError(
                "Points3d emission records must preserve exact manifest order."
            )
        if len(self.records) != len(self.behavior_cache.records):
            raise DynamicPoints3dOutputError(
                "Points3d emission cache does not cover every reviewed probe."
            )
        split_counts = {
            split: sum(record.split == split for record in self.records)
            for split in SPLITS
        }
        if any(count <= 0 for count in split_counts.values()):
            raise DynamicPoints3dOutputError(
                "Points3d emission cache must contain calibration and held-out "
                "records."
            )

    @property
    def calibration(self) -> tuple[Points3dEmissionRecord, ...]:
        return tuple(record for record in self.records if record.split == "calibration")

    @property
    def heldout(self) -> tuple[Points3dEmissionRecord, ...]:
        return tuple(record for record in self.records if record.split == "heldout")

    def cases(self) -> tuple[MultimodalWholeOutputCase, ...]:
        return tuple(record.as_case() for record in self.records)

    def verify(self) -> None:
        """Re-authenticate the behavior cache, the index, and every emission."""

        self.behavior_cache.verify()
        index = Path(self.directory) / EMISSIONS_INDEX
        if not index.is_file() or _file_sha256(index) != self.index_sha256:
            raise DynamicPoints3dOutputError(
                "Points3d emission index is missing or changed after sealing."
            )
        for record in self.records:
            record.verify()
        expected = canonical_sha256(
            {
                "schema": EMISSION_SCHEMA,
                "behavior_cache_fingerprint": self.behavior_cache.fingerprint,
                "index_sha256": self.index_sha256,
                "records": [record.identity() for record in self.records],
            }
        )
        if expected != self.fingerprint:
            raise DynamicPoints3dOutputError(
                "Points3d emission cache fingerprint is invalid."
            )


def write_points3d_parent_emission_cache(
    manifest_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    parent_emitter: Callable[[Any], Any],
) -> Path:
    """Run a parent once per probe and seal its EMITTED geometry.

    This is the output-side twin of
    :func:`neural_forge.dynamic_points3d_fold.write_points3d_parent_output_cache`
    and the only place in the lane that touches a parent model for emission.
    Every emitted payload is stored as its own ``NFPC1`` container so the
    reference store is re-decodable, and the index seals both the container
    digest and the decoded-content digest.
    """

    if not callable(parent_emitter):
        raise DynamicPoints3dOutputError("parent_emitter must be callable.")
    try:
        _manifest, probes = authenticate_native_points3d_probe(manifest_path)
    except NativePoints3dProbeError as exc:
        raise DynamicPoints3dOutputError(
            f"Points3d probe manifest authentication failed: {exc}"
        ) from exc
    root = Path(destination).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    emissions = root / EMISSIONS_DIRECTORY
    emissions.mkdir()
    lines = []
    for probe in probes:
        emitted = np.asarray(parent_emitter(probe))
        if emitted.dtype != np.int16:
            raise DynamicPoints3dOutputError(
                "A parent emitter must return an already-quantized int16 array; "
                "call quantize_emitted_points3d on a raw head output first."
            )
        relative = Path(EMISSIONS_DIRECTORY) / f"{probe.probe_id}.nfp"
        path = write_points3d_cloud(
            root / relative,
            emitted,
            scale=probe.scale,
            zero_point=probe.zero_point,
        )
        payload = emitted_points3d_payload(emitted)
        lines.append(
            _canonical_json(
                {
                    "schema": EMISSION_SCHEMA,
                    "index": probe.index,
                    "probe_id": probe.probe_id,
                    "record_sha256": probe.record_sha256,
                    "category": probe.category,
                    "split": probe.split,
                    "input_points_sha256": probe.points_sha256,
                    "emitted_points": relative.as_posix(),
                    "emitted_sha256": points3d_content_sha256(
                        POINT_DTYPE,
                        int(emitted.shape[0]),
                        probe.scale,
                        probe.zero_point,
                        payload,
                    ),
                    "emitted_container_sha256": _file_sha256(path),
                    "emitted_container_bytes": path.stat().st_size,
                    "emitted_point_count": int(emitted.shape[0]),
                    "components": int(emitted.shape[1]),
                    "dtype": POINT_DTYPE,
                    "scale": list(probe.scale),
                    "zero_point": list(probe.zero_point),
                }
            )
        )
    index = root / EMISSIONS_INDEX
    with index.open("xb") as handle:
        for line in lines:
            handle.write(line + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return root


def load_points3d_parent_emission_cache(
    behavior_cache: Points3dParentBehaviorCache,
    directory: str | os.PathLike[str],
    *,
    expected_index_sha256: str | None = None,
) -> Points3dParentEmissionCache:
    """Authenticate a sealed emission store against its behavior cache."""

    if not isinstance(behavior_cache, Points3dParentBehaviorCache):
        raise DynamicPoints3dOutputError(
            "behavior_cache must be Points3dParentBehaviorCache."
        )
    behavior_cache.verify()
    root = Path(directory).expanduser().resolve()
    index = root / EMISSIONS_INDEX
    if not index.is_file():
        raise DynamicPoints3dOutputError("Points3d emission index is missing.")
    index_sha = _file_sha256(index)
    if expected_index_sha256 is not None and index_sha != _digest(
        expected_index_sha256,
        "expected_index_sha256",
    ):
        raise DynamicPoints3dOutputError(
            "Points3d emission index differs from the pinned digest."
        )
    raw_lines = index.read_bytes()
    if raw_lines and not raw_lines.endswith(b"\n"):
        raise DynamicPoints3dOutputError("Points3d emission index is truncated.")
    lines = raw_lines.splitlines()
    if len(lines) != len(behavior_cache.records):
        raise DynamicPoints3dOutputError(
            "Points3d emission count differs from the reviewed probe suite."
        )
    records: list[Points3dEmissionRecord] = []
    for position, (behavior, line) in enumerate(
        zip(behavior_cache.records, lines, strict=True)
    ):
        try:
            cached = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicPoints3dOutputError(
                f"Points3d emission line {position + 1} is invalid."
            ) from exc
        if not isinstance(cached, Mapping):
            raise DynamicPoints3dOutputError(
                "Points3d emission record is not an object."
            )
        required = {
            "schema": EMISSION_SCHEMA,
            "index": behavior.index,
            "probe_id": behavior.probe_id,
            "record_sha256": behavior.manifest_record_sha256,
            "category": behavior.category,
            "split": behavior.split,
            "input_points_sha256": behavior.points_sha256,
            "dtype": POINT_DTYPE,
            "components": COMPONENTS,
        }
        if any(cached.get(key) != value for key, value in required.items()):
            raise DynamicPoints3dOutputError(
                f"Points3d cached emission record {position} failed "
                "authentication."
            )
        emitted_path = (root / str(cached.get("emitted_points"))).resolve()
        try:
            emitted_path.relative_to(root)
        except ValueError as exc:
            raise DynamicPoints3dOutputError(
                "Points3d emission path escapes the emission directory."
            ) from exc
        record = Points3dEmissionRecord(
            behavior=behavior,
            emitted_path=str(emitted_path),
            emitted_sha256=str(cached.get("emitted_sha256")),
            emitted_container_sha256=str(cached.get("emitted_container_sha256")),
            emitted_container_bytes=int(cached.get("emitted_container_bytes", 0)),
            emitted_point_count=int(cached.get("emitted_point_count", 0)),
            components=int(cached.get("components", 0)),
            dtype=str(cached.get("dtype")),
            scale=tuple(cached.get("scale", ())),
            zero_point=tuple(cached.get("zero_point", ())),
        )
        record.verify()
        records.append(record)
    fingerprint = canonical_sha256(
        {
            "schema": EMISSION_SCHEMA,
            "behavior_cache_fingerprint": behavior_cache.fingerprint,
            "index_sha256": index_sha,
            "records": [record.identity() for record in records],
        }
    )
    result = Points3dParentEmissionCache(
        behavior_cache=behavior_cache,
        directory=str(root),
        index_sha256=index_sha,
        records=tuple(records),
        fingerprint=fingerprint,
    )
    result.verify()
    return result


@dataclass(frozen=True, slots=True)
class Points3dEmittedOutputGateConfig:
    """Acceptance thresholds for parent-relative EMITTED-media review.

    The names deliberately do not reuse the text gate's vocabulary.  There is
    no ``fragment_like`` and no ``eos_margin`` here: an emitted payload is
    either absent, undecodable, or a decoded payload that either does or does
    not reproduce the parent's content digest.
    """

    minimum_emitted_retention: float = 1.0
    maximum_decode_failures: int = 0
    maximum_empty_emissions: int = 0
    require_no_split_retention_regression: bool = True
    require_no_split_category_retention_regression: bool = True
    require_parent_emission_coverage: bool = True
    required_splits: tuple[str, ...] = SPLITS
    required_categories: tuple[str, ...] = CATEGORIES

    def __post_init__(self) -> None:
        retention = self.minimum_emitted_retention
        if isinstance(retention, bool) or not isinstance(retention, (int, float)):
            raise DynamicPoints3dOutputError(
                "minimum_emitted_retention must be a real number."
            )
        retention = float(retention)
        if not 0.0 <= retention <= 1.0:
            raise DynamicPoints3dOutputError(
                "minimum_emitted_retention must be between zero and one."
            )
        object.__setattr__(self, "minimum_emitted_retention", retention)
        for name in ("maximum_decode_failures", "maximum_empty_emissions"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        for name in (
            "require_no_split_retention_regression",
            "require_no_split_category_retention_regression",
            "require_parent_emission_coverage",
        ):
            if not isinstance(getattr(self, name), bool):
                raise DynamicPoints3dOutputError(f"{name} must be boolean.")
        for name, allowed in (
            ("required_splits", frozenset(SPLITS)),
            ("required_categories", frozenset(CATEGORIES)),
        ):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise DynamicPoints3dOutputError(
                    f"{name} must be nonempty and unique."
                )
            if any(value not in allowed for value in values):
                raise DynamicPoints3dOutputError(
                    f"{name} contains an unsupported value."
                )
            object.__setattr__(self, name, values)

    @property
    def coverage_policy(self) -> TaskScoreCoveragePolicy:
        if not self.require_parent_emission_coverage:
            return TaskScoreCoveragePolicy()
        return TaskScoreCoveragePolicy.production(
            required_splits=self.required_splits,
            required_categories=self.required_categories,
            require_split_category_cross_product=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_emitted_retention": self.minimum_emitted_retention,
            "maximum_decode_failures": self.maximum_decode_failures,
            "maximum_empty_emissions": self.maximum_empty_emissions,
            "require_no_split_retention_regression": (
                self.require_no_split_retention_regression
            ),
            "require_no_split_category_retention_regression": (
                self.require_no_split_category_retention_regression
            ),
            "require_parent_emission_coverage": (
                self.require_parent_emission_coverage
            ),
            "required_splits": list(self.required_splits),
            "required_categories": list(self.required_categories),
        }


def assert_graph_emits_points3d(graph: ModelCircuitGraph) -> tuple[str, ...]:
    """Fail closed unless one graph really carries a points3d output emission.

    A gate that reviews emitted media against a model with no
    ``MODALITY_OUTPUT`` node reviews nothing.  This is the compiler-side
    precondition of the output contract and it is checked against the IR, not
    against a config flag.
    """

    if not isinstance(graph, ModelCircuitGraph):
        raise DynamicPoints3dOutputError("graph must be ModelCircuitGraph.")
    emitting = tuple(
        interface.interface_id
        for interface in emitted_media_interfaces(graph)
        if interface.modality == MODALITY_POINTS3D
    )
    nodes = graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT)
    if not emitting or not nodes:
        raise DynamicPoints3dOutputError(
            "Emitted-output review requires a graph with a points3d "
            "MODALITY_OUTPUT node and a direction='output' points3d interface."
        )
    return emitting


class NativePoints3dEmittedOutputGate:
    """Parent-relative EMITTED-geometry gate using cached emissions as canonical.

    The evaluator surface, the sealed manifest fields, and the
    ``behavior_cache`` binding are identical to
    :class:`neural_forge.dynamic_points3d_fold.NativePoints3dBehaviorGate`, so
    this gate can drive the same published stack, the same final evaluation,
    and the same standalone fresh reload.  What differs is what is reviewed:
    the child's own emitted media rather than its text.
    """

    def __init__(
        self,
        emission_cache: Points3dParentEmissionCache,
        *,
        child_emitter: Callable[[Any, Points3dBehaviorRecord], Any],
        config: Points3dEmittedOutputGateConfig | None = None,
        child_emitter_identity: BehaviorCallableIdentity | None = None,
        inference_execution_contract: (
            BehaviorInferenceExecutionContract | None
        ) = None,
    ) -> None:
        if not isinstance(emission_cache, Points3dParentEmissionCache):
            raise DynamicPoints3dOutputError(
                "emission_cache must be Points3dParentEmissionCache."
            )
        if not callable(child_emitter):
            raise DynamicPoints3dOutputError("child_emitter must be callable.")
        self.emission_cache = emission_cache
        self.behavior_cache = emission_cache.behavior_cache
        self.child_emitter = child_emitter
        self.config = config or Points3dEmittedOutputGateConfig()
        if not isinstance(self.config, Points3dEmittedOutputGateConfig):
            raise DynamicPoints3dOutputError(
                "config must be Points3dEmittedOutputGateConfig."
            )
        self.execution_contract = (
            POINTS3D_EMISSION_EXECUTION_CONTRACT
            if inference_execution_contract is None
            else inference_execution_contract
        )
        if not isinstance(
            self.execution_contract,
            BehaviorInferenceExecutionContract,
        ):
            raise DynamicPoints3dOutputError(
                "inference_execution_contract must be "
                "BehaviorInferenceExecutionContract or None."
            )
        self.last_report: Mapping[str, Any] | None = None
        self.evaluation_count = 0
        if child_emitter_identity is None:
            child_emitter_identity = identify_behavior_callable(
                child_emitter,
                implementation_id=(
                    "neural_forge.points3d.native_points3d.child_emission_runner"
                ),
                version=1,
            )
        if not isinstance(child_emitter_identity, BehaviorCallableIdentity):
            raise DynamicPoints3dOutputError(
                "child_emitter_identity must be BehaviorCallableIdentity."
            )
        # The decode contract is sealed as its own manifest component.  Without
        # it "the child emitted geometry" is an unverifiable claim: two runs
        # could decode the same head output differently and both call it a
        # retained emission.
        self.decoder_identity = identify_behavior_callable(
            quantize_emitted_points3d,
            implementation_id=(
                "neural_forge.points3d.native_points3d.emitted_media_decoder"
            ),
            version=1,
        )
        decode_contract = {
            "schema": EMISSION_DECODER_SCHEMA,
            "modality": MODALITY_POINTS3D,
            "codec": EMITTED_CODEC_ID,
            "hash_basis": POINTS3D_HASH_BASIS,
            "decoder": self.decoder_identity.as_dict(),
            "comparison": "decoded_content_digest_exact_match",
            "tolerance": "none",
            "open_generation_reviewed_here": False,
        }
        binding = build_multimodal_behavior_gate_binding(
            gate_id=OUTPUT_GATE_ID,
            gate_version=OUTPUT_GATE_VERSION,
            cases=emission_cache.cases(),
            threshold_config=self.config.as_dict(),
            generator_identity=child_emitter_identity,
            scorer_identity=self.decoder_identity,
            coverage_policy=self.config.coverage_policy.as_dict(),
            baseline_fingerprint=emission_cache.fingerprint,
            baseline={
                "emission_cache_fingerprint": emission_cache.fingerprint,
                "behavior_cache_fingerprint": (
                    emission_cache.behavior_cache.fingerprint
                ),
                "manifest_sha256": emission_cache.behavior_cache.manifest_sha256,
                "emission_index_sha256": emission_cache.index_sha256,
                "record_count": len(emission_cache.records),
            },
            components=(decode_contract,),
            inference_execution_contract=self.execution_contract,
        )
        expose_behavior_gate_binding(self, binding)

    @staticmethod
    def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        retained = sum(row["retained"] for row in rows)
        decode_failures = sum(row["decode_failed"] for row in rows)
        empty = sum(row["child_empty"] for row in rows)
        return {
            "total": total,
            "parent_emitted": total,
            "child_emitted": total - decode_failures,
            "emitted_retained": retained,
            "emitted_retention": (retained / total if total else 1.0),
            "decode_failures": decode_failures,
            "empty_emissions": empty,
        }

    def _emission_coverage(self) -> tuple[dict[str, int], dict[str, int], tuple[str, ...]]:
        by_split = {
            split: sum(record.split == split for record in self.emission_cache.records)
            for split in self.config.required_splits
        }
        by_category = {
            category: sum(
                record.category == category for record in self.emission_cache.records
            )
            for category in self.config.required_categories
        }
        missing_cross = tuple(
            f"{split}:{category}"
            for split in self.config.required_splits
            for category in self.config.required_categories
            if not any(
                record.split == split and record.category == category
                for record in self.emission_cache.records
            )
        )
        return by_split, by_category, missing_cross

    def evaluate_model(self, model: Any) -> FoldDecision:
        self.emission_cache.verify()
        coverage_policy = self.config.coverage_policy
        by_split_counts, by_category_counts, missing_cross = self._emission_coverage()
        coverage_failures: list[str] = []
        if self.config.require_parent_emission_coverage:
            coverage_failures.extend(
                f"{split}:parent_emission_coverage"
                for split, count in by_split_counts.items()
                if count <= 0
            )
            coverage_failures.extend(
                f"{category}:parent_emission_coverage"
                for category, count in by_category_counts.items()
                if count <= 0
            )
            coverage_failures.extend(
                f"{value}:parent_emission_coverage" for value in missing_cross
            )
        if coverage_failures:
            report = {
                "schema": OUTPUT_GATE_REPORT_SCHEMA,
                "modality": MODALITY_POINTS3D,
                "review_side": "output",
                "emission_cache_fingerprint": self.emission_cache.fingerprint,
                "behavior_cache_fingerprint": self.behavior_cache.fingerprint,
                "behavior_gate_manifest_fingerprint": (
                    self.behavior_gate_manifest_fingerprint
                ),
                "emission_index_sha256": self.emission_cache.index_sha256,
                "coverage_policy": coverage_policy.as_dict(),
                "parent_emissions_by_split": by_split_counts,
                "parent_emissions_by_category": by_category_counts,
                "failures": sorted(set(coverage_failures)),
                "rows": [],
                "parent_forward_count": 0,
                "child_emission_count": 0,
                "emitted_media_hash_basis": POINTS3D_HASH_BASIS,
            }
            self.last_report = MappingProxyType(report)
            self.evaluation_count += 1
            return FoldDecision.reject(
                "Native points3d emitted-output coverage is incomplete: "
                + ", ".join(sorted(set(coverage_failures))),
                {
                    "parent_forward_count": 0,
                    "child_emissions_scored": 0,
                    "missing_required_split_category_count": len(missing_cross),
                },
            )
        rows: list[dict[str, Any]] = []
        with observe_model_execution(
            model,
            contract=self.execution_contract,
        ) as execution_counts:
            for record in self.emission_cache.records:
                raw = self.child_emitter(model, record.behavior)
                decode_failed = False
                candidate: MultimodalCandidateOutput | None = None
                digest: str | None = None
                point_count = 0
                byte_count = 0
                try:
                    decoded = quantize_emitted_points3d(
                        raw,
                        scale=record.scale,
                        zero_point=record.zero_point,
                    )
                    media = emitted_points3d_media(
                        decoded,
                        uri=f"{CHILD_EMISSION_URI_SCHEME}/{record.probe_id}",
                        scale=record.scale,
                        zero_point=record.zero_point,
                    )
                    candidate = MultimodalCandidateOutput(output=media)
                    digest = media.content_sha256
                    point_count = int(decoded.shape[0])
                    byte_count = media.byte_count
                except Points3dEmissionDecodeError:
                    decode_failed = True
                rows.append(
                    {
                        "probe_id": record.probe_id,
                        "split": record.split,
                        "category": record.category,
                        "input_points_sha256": record.behavior.points_sha256,
                        "parent_emitted_sha256": record.emitted_sha256,
                        "child_emitted_sha256": digest,
                        "child_emitted_point_count": point_count,
                        "child_emitted_byte_count": byte_count,
                        "decode_failed": decode_failed,
                        "child_empty": (
                            False if candidate is None else candidate.empty
                        ),
                        "child_output_is_media": (
                            False if candidate is None else candidate.is_media
                        ),
                        "retained": digest == record.emitted_sha256,
                    }
                )
        if not execution_contract_satisfied(execution_counts):
            raise DynamicPoints3dOutputError(
                "Native points3d child emitter returned without executing the "
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
        failures: list[str] = []
        if self.config.require_no_split_retention_regression:
            for split, metrics in by_split.items():
                if metrics["emitted_retention"] < self.config.minimum_emitted_retention:
                    failures.append(f"{split}:emitted_retention")
                if metrics["decode_failures"] > self.config.maximum_decode_failures:
                    failures.append(f"{split}:emitted_decode_failures")
                if metrics["empty_emissions"] > self.config.maximum_empty_emissions:
                    failures.append(f"{split}:emitted_empty")
        if self.config.require_no_split_category_retention_regression:
            for split, categories in by_split_category.items():
                for category, metrics in categories.items():
                    if (
                        metrics["emitted_retention"]
                        < self.config.minimum_emitted_retention
                    ):
                        failures.append(f"{split}:{category}:emitted_retention")
        total_rows = len(rows)
        retained_rows = sum(row["retained"] for row in rows)
        report = {
            "schema": OUTPUT_GATE_REPORT_SCHEMA,
            "modality": MODALITY_POINTS3D,
            "review_side": "output",
            "emission_cache_fingerprint": self.emission_cache.fingerprint,
            "behavior_cache_fingerprint": self.behavior_cache.fingerprint,
            "behavior_gate_manifest_fingerprint": (
                self.behavior_gate_manifest_fingerprint
            ),
            "emission_index_sha256": self.emission_cache.index_sha256,
            "coverage_policy": coverage_policy.as_dict(),
            "parent_emissions_by_split": by_split_counts,
            "parent_emissions_by_category": by_category_counts,
            "by_split": by_split,
            "by_split_category": by_split_category,
            "failures": sorted(set(failures)),
            "rows": rows,
            "parent_forward_count": 0,
            "child_emission_count": total_rows,
            "observed_child_module_forward_calls": (
                execution_counts["module_forward_calls"]
            ),
            "observed_child_root_forward_calls": (
                execution_counts["root_forward_calls"]
            ),
            "emitted_media_hash_basis": POINTS3D_HASH_BASIS,
            "emitted_media_decoder": self.decoder_identity.as_dict(),
            "generative_quality_reviewed": False,
        }
        self.last_report = MappingProxyType(report)
        self.evaluation_count += 1
        flat_metrics = {
            "calibration_emitted_retained": by_split["calibration"][
                "emitted_retained"
            ],
            "calibration_emitted_total": by_split["calibration"]["total"],
            "heldout_emitted_retained": by_split["heldout"]["emitted_retained"],
            "heldout_emitted_total": by_split["heldout"]["total"],
            "heldout_emitted_retention": by_split["heldout"]["emitted_retention"],
            "emitted_retained": retained_rows,
            "emitted_decode_failures": sum(row["decode_failed"] for row in rows),
            "emitted_empty": sum(row["child_empty"] for row in rows),
            "parent_forward_count": 0,
            "child_emissions_scored": total_rows,
            "observed_child_module_forward_calls": (
                execution_counts["module_forward_calls"]
            ),
            "observed_child_root_forward_calls": (
                execution_counts["root_forward_calls"]
            ),
        }
        if failures:
            return FoldDecision.reject(
                "Native points3d emitted geometry regressed: "
                + ", ".join(sorted(set(failures))),
                flat_metrics,
            )
        return FoldDecision.accept(
            "Native points3d child reproduced every cached parent emission "
            "exactly under the declared decode contract; generative quality is "
            "not reviewed.",
            flat_metrics,
        )

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        return self.evaluate_model(context.execution_context.child_model)

    def __call__(self, model: Any) -> FoldDecision:
        return self.evaluate_model(model)


def emitted_output_contract_status() -> dict[str, Any]:
    """State exactly what output-side review does and does not cover today."""

    return {
        "schema": "neural-forge-modality-output-contract-status-v1",
        "modality": MODALITY_POINTS3D,
        "gate_id": OUTPUT_GATE_ID,
        "emitted_media_hash_basis": POINTS3D_HASH_BASIS,
        "provided": [
            "an inspector arm that emits a real CircuitKind.MODALITY_OUTPUT "
            "node and a direction='output' non-text ModalityInterface",
            "a sealed parent emission store with container and decoded-content "
            "digests per emitted payload",
            "a declared, fingerprinted decode contract sealed into the gate "
            "manifest as its own component",
            "parent-relative emitted retention by exact decoded-content digest, "
            "with split and split/category coverage",
            "non-text regression notions: decode failure and empty emission",
            "the same evaluator surface as the input-side gate, so publication "
            "and standalone fresh reload replay emitted-output review too",
        ],
        "not_provided": [
            "any claim about generative quality",
            "open-ended generation review; only closed quantized answer sets "
            "are gateable by exact match",
            "perceptual or distributional emitted-media metrics",
            "an output-side FOLD operator; folding still touches decoder "
            "internals only, never the emitting head",
        ],
        "open_generation_policy": (
            "manual review; an open emitted payload has no closed reference to "
            "compare against, so it must not be auto-promoted"
        ),
    }


__all__ = [
    "CHILD_EMISSION_URI_SCHEME",
    "EMISSIONS_DIRECTORY",
    "EMISSIONS_INDEX",
    "EMISSION_DECODER_SCHEMA",
    "EMISSION_SCHEMA",
    "EMITTED_CODEC_ID",
    "OUTPUT_GATE_ID",
    "OUTPUT_GATE_REPORT_SCHEMA",
    "OUTPUT_GATE_VERSION",
    "POINTS3D_EMISSION_EXECUTION_CONTRACT",
    "DynamicPoints3dOutputError",
    "NativePoints3dEmittedOutputGate",
    "Points3dEmissionDecodeError",
    "Points3dEmissionRecord",
    "Points3dEmittedOutputGateConfig",
    "Points3dParentEmissionCache",
    "assert_graph_emits_points3d",
    "emitted_output_contract_status",
    "emitted_points3d_media",
    "emitted_points3d_payload",
    "load_points3d_parent_emission_cache",
    "quantize_emitted_points3d",
    "write_points3d_parent_emission_cache",
]
