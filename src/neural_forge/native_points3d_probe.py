"""Deterministic quantized point-set probes for native points3d transfer.

The generator is the points3d mirror of
:mod:`neural_forge.native_video_probe`: it synthesizes small, fully
deterministic integer point clouds with closed answer sets, seals every decoded
payload under :data:`POINTS3D_HASH_BASIS`, and writes one sealed manifest.

Determinism is structural rather than incidental, and is held to the video
lane's standard: every coordinate is a signed integer produced by integer
arithmetic on integer inputs, so **no floating-point value participates in
generating a point**.  Each probe draws its jitter from a string-seeded
:class:`random.Random`, so no probe depends on generation order.  The one float
anywhere in the lane is the declared per-axis quantization scale, which is a
fixed module constant, never computed, and sealed into both the manifest and
every stored container.

The quantization contract is not invented here.  It is the one
:class:`neural_forge.raw_modality_tokens.PointQuantization` already defines for
the raw-byte codec: a ``(point_count, 3)`` integer array plus a positive
per-axis ``scale`` and a signed-int32 per-axis ``zero_point``, dequantized as
``(points - zero_point) * scale``.  Probes are stored with that exact triple so
one array can be handed to :meth:`RawModalityCodec.encode_points3d` and to this
container without either re-describing what the integers mean.

Only the standard library and NumPy are used.  Clouds are stored in the
self-describing ``NFPC1`` raw container written here: eight bytes of magic plus
four little-endian ``uint32`` header fields, three little-endian ``float64``
axis scales, and three little-endian ``int32`` axis zero points, followed by
row-major little-endian XYZ triples.  There is no ``open3d``, ``trimesh``, or
``plyfile`` dependency, no compression, and no learned component of any kind --
the container carries its own geometry and quantization, so a stored cloud is
re-decodable without trusting the manifest that describes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .dynamic_behavior_gate import POINTS3D_HASH_BASIS, points3d_content_sha256
from .dynamic_circuit_ir import MODALITY_POINTS3D, canonical_sha256
from .raw_modality_tokens import PointQuantization


PROBE_SCHEMA = "neural-forge-native-points3d-probe-v1"
MODALITY = MODALITY_POINTS3D
CODEC_ID = "points3d/nfpc1-int16"

POINT_COUNT = 60
POINT_DTYPE = "int16"
COMPONENTS = 3
COORDINATE_LIMIT = 1024

# The declared quantization.  Deliberately non-trivial and per-axis so the
# lane exercises the codec contract rather than an identity mapping, and
# deliberately built from exactly representable binary fractions so the sealed
# ``repr`` in the hash header is stable everywhere.
PROBE_SCALE: tuple[float, float, float] = (0.25, 0.5, 0.125)
PROBE_ZERO_POINT: tuple[int, int, int] = (-8, 4, 0)
PROBE_QUANTIZATION = PointQuantization(
    scale=PROBE_SCALE,
    zero_point=PROBE_ZERO_POINT,
)

# Separations that make each closed answer decidable from the stored integers
# alone: a test re-derives the answer from the decoded cloud instead of
# trusting the label.
CLUSTER_EXTENT = 6
CLUSTER_GAP = 300
SPREAD_MINIMUM_EXTENT = 96

CATEGORIES = (
    "cluster_count",
    "dominant_axis",
    "shape_kind",
    "origin_side",
)
SPLITS = ("calibration", "heldout")
CLOSED_ANSWERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "cluster_count": ("1", "2", "3", "4", "5"),
        "dominant_axis": ("x", "y", "z"),
        "shape_kind": ("line", "plane", "cluster", "scatter"),
        "origin_side": ("negative", "positive", "straddle"),
    }
)
PROMPTS: Mapping[str, str] = MappingProxyType(
    {
        "cluster_count": (
            "How many separate clusters does this point cloud contain? Answer "
            "with one digit."
        ),
        "dominant_axis": (
            "Which axis does this point cloud span the most? Answer x, y, or z."
        ),
        "shape_kind": (
            "Is this point cloud a line, a plane, a cluster, or a scatter? "
            "Answer with one word."
        ),
        "origin_side": (
            "Do the points sit on the negative side of x = 0, the positive "
            "side, or both? Answer negative, positive, or straddle."
        ),
    }
)

_AXES = ("x", "y", "z")
_DTYPE_CODES: Mapping[str, int] = MappingProxyType(
    {
        "int8": 1,
        "uint8": 2,
        "int16": 3,
        "uint16": 4,
        "int32": 5,
    }
)
_CODE_DTYPES: Mapping[int, str] = MappingProxyType(
    {code: name for name, code in _DTYPE_CODES.items()}
)

_CONTAINER_MAGIC = b"NFPC1\x00\x00\x00"
_CONTAINER_HEADER = struct.Struct("<4I3d3i")
CONTAINER_HEADER_BYTES = len(_CONTAINER_MAGIC) + _CONTAINER_HEADER.size


class NativePoints3dProbeError(ValueError):
    """A probe request, manifest, or stored point cloud is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativePoints3dProbeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativePoints3dProbeError(f"{name} must be a non-negative integer.")
    return value


def _quantization(
    scale: Sequence[float] | None,
    zero_point: Sequence[int] | None,
) -> PointQuantization:
    """Normalize one caller-supplied quantization into the codec's contract."""

    axes = PROBE_SCALE if scale is None else tuple(scale)
    origins = PROBE_ZERO_POINT if zero_point is None else tuple(zero_point)
    if len(axes) != COMPONENTS or len(origins) != COMPONENTS:
        raise NativePoints3dProbeError(
            "Point quantization needs exactly three scales and three zero points."
        )
    for value in axes:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NativePoints3dProbeError("Point scale values must be numbers.")
        if not float(value) > 0.0:
            raise NativePoints3dProbeError("Point scale values must be positive.")
    for value in origins:
        if isinstance(value, bool) or not isinstance(value, int):
            raise NativePoints3dProbeError("Point zero_point values must be integers.")
    return PointQuantization(
        scale=tuple(float(value) for value in axes),
        zero_point=tuple(int(value) for value in origins),
    )


def _integer_points(values: Sequence[Sequence[int]]) -> Any:
    """Materialize integer XYZ triples, refusing anything out of lattice range."""

    rows = tuple(values)
    if len(rows) != POINT_COUNT:
        raise NativePoints3dProbeError(
            f"A points3d probe must contain exactly {POINT_COUNT} points."
        )
    cloud = np.empty((POINT_COUNT, COMPONENTS), dtype=np.int16)
    for index, row in enumerate(rows):
        triple = tuple(row)
        if len(triple) != COMPONENTS:
            raise NativePoints3dProbeError("Every probe point must be an XYZ triple.")
        for axis, coordinate in enumerate(triple):
            if isinstance(coordinate, bool) or not isinstance(coordinate, int):
                raise NativePoints3dProbeError(
                    "Probe coordinates must be Python integers."
                )
            if abs(coordinate) > COORDINATE_LIMIT:
                raise NativePoints3dProbeError(
                    "Probe coordinate escapes the declared integer lattice."
                )
            cloud[index, axis] = coordinate
    return cloud


def _spread(index: int, extent: int) -> int:
    """Integer-only linear spread hitting exactly ``0`` and ``extent``."""

    _nonnegative_int(index, "spread index")
    _nonnegative_int(extent, "spread extent")
    return (index * extent) // (POINT_COUNT - 1)


def _cluster_count_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    count = 1 + index % 5
    per_cluster = POINT_COUNT // count
    if per_cluster * count != POINT_COUNT:
        raise NativePoints3dProbeError("Cluster sizes must divide the point count.")
    base_y = rng.randrange(-200, 201)
    base_z = rng.randrange(-200, 201)
    rows: list[tuple[int, int, int]] = []
    for cluster in range(count):
        centre_x = (cluster - (count - 1) // 2) * CLUSTER_GAP + rng.randrange(-40, 41)
        for member in range(per_cluster):
            rows.append(
                (
                    centre_x + (member * CLUSTER_EXTENT) // max(1, per_cluster - 1),
                    base_y + (member * 13) % (CLUSTER_EXTENT + 1),
                    base_z + (member * 5) % (CLUSTER_EXTENT + 1),
                )
            )
    return _integer_points(rows), PROMPTS["cluster_count"], str(count)


def _dominant_axis_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    axis = _AXES[index % len(_AXES)]
    dominant = index % len(_AXES)
    dominant_extent = 400 + rng.randrange(0, 5) * 20
    minor_extents = [40 + rng.randrange(0, 5) * 10 for _axis in range(COMPONENTS)]
    minor_extents[dominant] = dominant_extent
    bases = [
        rng.randrange(-COORDINATE_LIMIT + extent, COORDINATE_LIMIT - extent + 1)
        for extent in minor_extents
    ]
    rows = []
    for point in range(POINT_COUNT):
        rows.append(
            (
                bases[0] + _spread((point * 7) % POINT_COUNT, minor_extents[0]),
                bases[1] + _spread((point * 11) % POINT_COUNT, minor_extents[1]),
                bases[2] + _spread((point * 19) % POINT_COUNT, minor_extents[2]),
            )
        )
    return _integer_points(rows), PROMPTS["dominant_axis"], axis


def _shape_kind_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    kind = CLOSED_ANSWERS["shape_kind"][index % 4]
    varying = (index // 4) % COMPONENTS
    bases = [rng.randrange(-300, 301) for _axis in range(COMPONENTS)]
    rows = []
    if kind == "line":
        extent = 240 + rng.randrange(0, 5) * 12
        for point in range(POINT_COUNT):
            triple = list(bases)
            triple[varying] = bases[varying] + _spread(point, extent)
            rows.append(tuple(triple))
    elif kind == "plane":
        constant = varying
        first, second = (axis for axis in range(COMPONENTS) if axis != constant)
        first_extent = 200 + rng.randrange(0, 5) * 8
        second_extent = 160 + rng.randrange(0, 5) * 8
        for point in range(POINT_COUNT):
            triple = list(bases)
            triple[first] = bases[first] + _spread(point, first_extent)
            triple[second] = bases[second] + _spread(
                (point * 17) % POINT_COUNT,
                second_extent,
            )
            rows.append(tuple(triple))
    elif kind == "cluster":
        for point in range(POINT_COUNT):
            rows.append(
                (
                    bases[0] + _spread(point, CLUSTER_EXTENT),
                    bases[1] + (point * 13) % (CLUSTER_EXTENT + 1),
                    bases[2] + (point * 5) % (CLUSTER_EXTENT + 1),
                )
            )
    else:
        extents = [
            SPREAD_MINIMUM_EXTENT + 8 + rng.randrange(0, 5) * 16
            for _axis in range(COMPONENTS)
        ]
        for point in range(POINT_COUNT):
            rows.append(
                (
                    bases[0] + _spread(point, extents[0]),
                    bases[1] + _spread((point * 17) % POINT_COUNT, extents[1]),
                    bases[2] + _spread((point * 23) % POINT_COUNT, extents[2]),
                )
            )
    return _integer_points(rows), PROMPTS["shape_kind"], kind


def _origin_side_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    side = CLOSED_ANSWERS["origin_side"][index % 3]
    extent = 300 + rng.randrange(0, 5) * 20
    if side == "negative":
        base_x = -(100 + extent + rng.randrange(0, 200))
    elif side == "positive":
        base_x = 100 + rng.randrange(0, 200)
    else:
        base_x = -(extent // 2) - rng.randrange(1, 40)
    base_y = rng.randrange(-300, 301)
    base_z = rng.randrange(-300, 301)
    rows = []
    for point in range(POINT_COUNT):
        rows.append(
            (
                base_x + _spread(point, extent),
                base_y + _spread((point * 11) % POINT_COUNT, 64),
                base_z + _spread((point * 19) % POINT_COUNT, 48),
            )
        )
    cloud = _integer_points(rows)
    minimum = int(cloud[:, 0].min())
    maximum = int(cloud[:, 0].max())
    observed = (
        "negative"
        if maximum < 0
        else "positive"
        if minimum > 0
        else "straddle"
    )
    if observed != side:
        raise NativePoints3dProbeError(
            "Origin-side probe geometry disagrees with its own label."
        )
    return cloud, PROMPTS["origin_side"], side


_BUILDERS = {
    "cluster_count": _cluster_count_probe,
    "dominant_axis": _dominant_axis_probe,
    "shape_kind": _shape_kind_probe,
    "origin_side": _origin_side_probe,
}


@dataclass(frozen=True, slots=True)
class Points3dProbeRecord:
    """One manifest row describing a stored, content-bound point cloud."""

    probe_id: str
    category: str
    points: str
    prompt: str
    answer: str
    split: str
    points_sha256: str
    container_sha256: str
    container_bytes: int
    point_count: int
    components: int
    dtype: str
    scale: tuple[float, float, float]
    zero_point: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class AuthenticatedPoints3dProbe:
    """One manifest row whose stored cloud was re-decoded and re-hashed."""

    index: int
    probe_id: str
    category: str
    split: str
    prompt: str
    answer: str
    path: str
    points_sha256: str
    container_sha256: str
    container_bytes: int
    point_count: int
    components: int
    dtype: str
    scale: tuple[float, float, float]
    zero_point: tuple[int, int, int]
    record_sha256: str

    @property
    def quantization(self) -> PointQuantization:
        return PointQuantization(scale=self.scale, zero_point=self.zero_point)


def canonical_points3d_sha256(
    points: Any,
    *,
    scale: Sequence[float] | None = None,
    zero_point: Sequence[int] | None = None,
) -> str:
    """Hash a decoded quantized XYZ array under :data:`POINTS3D_HASH_BASIS`."""

    array = np.asarray(points)
    if array.dtype != np.int16:
        raise NativePoints3dProbeError(
            "Canonical points3d hashing requires int16 quantized points."
        )
    if array.ndim != 2 or int(array.shape[1]) != COMPONENTS:
        raise NativePoints3dProbeError(
            "Canonical points3d hashing requires (point_count, 3) integers."
        )
    quantization = _quantization(scale, zero_point)
    payload = np.ascontiguousarray(array.astype("<i2", copy=False)).tobytes()
    return points3d_content_sha256(
        POINT_DTYPE,
        _positive_int(int(array.shape[0]), "point_count"),
        quantization.scale,
        quantization.zero_point,
        payload,
    )


def write_points3d_cloud(
    path: str | os.PathLike[str],
    points: Any,
    *,
    scale: Sequence[float] | None = None,
    zero_point: Sequence[int] | None = None,
) -> Path:
    """Write quantized XYZ points as one self-describing ``NFPC1`` container."""

    array = np.asarray(points)
    if array.dtype != np.int16:
        raise NativePoints3dProbeError("NFPC1 output requires int16 points.")
    if array.ndim != 2 or int(array.shape[1]) != COMPONENTS:
        raise NativePoints3dProbeError(
            "NFPC1 output requires (point_count, 3) points."
        )
    quantization = _quantization(scale, zero_point)
    point_count = _positive_int(int(array.shape[0]), "point_count")
    payload = np.ascontiguousarray(array.astype("<i2", copy=False)).tobytes()
    destination = Path(path)
    with destination.open("xb") as handle:
        handle.write(_CONTAINER_MAGIC)
        handle.write(
            _CONTAINER_HEADER.pack(
                point_count,
                COMPONENTS,
                _DTYPE_CODES[POINT_DTYPE],
                len(payload),
                *quantization.scale,
                *quantization.zero_point,
            )
        )
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def read_points3d_cloud(
    path: str | os.PathLike[str],
) -> tuple[Any, PointQuantization]:
    """Decode an ``NFPC1`` container into ``(points, quantization)``."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise NativePoints3dProbeError(
            f"Points3d probe cannot be read: {source}."
        ) from exc
    if len(raw) < CONTAINER_HEADER_BYTES or not raw.startswith(_CONTAINER_MAGIC):
        raise NativePoints3dProbeError(
            f"Points3d probe {source} is not an NFPC1 container."
        )
    (
        point_count,
        components,
        dtype_code,
        payload_bytes,
        scale_x,
        scale_y,
        scale_z,
        zero_x,
        zero_y,
        zero_z,
    ) = _CONTAINER_HEADER.unpack(raw[len(_CONTAINER_MAGIC) : CONTAINER_HEADER_BYTES])
    if components != COMPONENTS or _CODE_DTYPES.get(int(dtype_code)) != POINT_DTYPE:
        raise NativePoints3dProbeError(
            f"Points3d probe {source} is not int16 XYZ geometry."
        )
    _positive_int(int(point_count), "stored point_count")
    quantization = _quantization(
        (scale_x, scale_y, scale_z),
        (int(zero_x), int(zero_y), int(zero_z)),
    )
    expected = int(point_count) * COMPONENTS * 2
    payload = raw[CONTAINER_HEADER_BYTES:]
    if payload_bytes != expected or len(payload) != expected:
        raise NativePoints3dProbeError(
            f"Points3d probe {source} is truncated or padded."
        )
    return (
        np.frombuffer(payload, dtype="<i2")
        .reshape(int(point_count), COMPONENTS)
        .astype(np.int16, copy=False),
        quantization,
    )


def decoded_points3d_sha256(path: str | os.PathLike[str]) -> str:
    """Re-decode a stored cloud and rehash it under the canonical basis."""

    points, quantization = read_points3d_cloud(path)
    return canonical_points3d_sha256(
        points,
        scale=quantization.scale,
        zero_point=quantization.zero_point,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_native_points3d_probe(
    output_dir: str | os.PathLike[str],
    *,
    examples_per_category: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a sealed synthetic points3d probe suite under ``output_dir``."""

    _positive_int(examples_per_category, "examples_per_category")
    _nonnegative_int(seed, "seed")
    root = Path(output_dir)
    points_dir = root / "points"
    root.mkdir(parents=True, exist_ok=False)
    points_dir.mkdir()
    records: list[Points3dProbeRecord] = []
    heldout_per_category = max(1, examples_per_category // 4)
    heldout_start = examples_per_category - heldout_per_category
    for category, builder in _BUILDERS.items():
        for category_index in range(examples_per_category):
            rng = random.Random(
                f"neural-forge-points3d-probe:{seed}:{category}:{category_index}"
            )
            cloud, prompt, answer = builder(category_index, rng)
            if answer not in CLOSED_ANSWERS[category]:
                raise NativePoints3dProbeError(
                    f"Probe answer {answer!r} escapes the closed set for "
                    f"{category!r}."
                )
            probe_id = f"{category}_{category_index:04d}"
            relative = Path("points") / f"{probe_id}.nfp"
            path = write_points3d_cloud(root / relative, cloud)
            records.append(
                Points3dProbeRecord(
                    probe_id=probe_id,
                    category=category,
                    points=relative.as_posix(),
                    prompt=prompt,
                    answer=answer,
                    split=(
                        "heldout"
                        if category_index >= heldout_start
                        else "calibration"
                    ),
                    points_sha256=canonical_points3d_sha256(cloud),
                    container_sha256=_file_sha256(path),
                    container_bytes=path.stat().st_size,
                    # Derived from the synthesized array rather than restated
                    # from the module constants: a record can never describe a
                    # geometry the stored cloud does not have.
                    point_count=int(cloud.shape[0]),
                    components=int(cloud.shape[1]),
                    dtype=POINT_DTYPE,
                    scale=PROBE_SCALE,
                    zero_point=PROBE_ZERO_POINT,
                )
            )
    body = {
        "schema": PROBE_SCHEMA,
        "seed": seed,
        "modality": MODALITY,
        "codec": CODEC_ID,
        "points_format": (
            f"{POINT_DTYPE} quantized XYZ, {POINT_COUNT} points, raw NFPC1 "
            "container"
        ),
        "points_hash_basis": POINTS3D_HASH_BASIS,
        "point_count": POINT_COUNT,
        "components": COMPONENTS,
        "dtype": POINT_DTYPE,
        "scale": list(PROBE_SCALE),
        "zero_point": list(PROBE_ZERO_POINT),
        "coordinate_limit": COORDINATE_LIMIT,
        "cluster_extent": CLUSTER_EXTENT,
        "cluster_gap": CLUSTER_GAP,
        "spread_minimum_extent": SPREAD_MINIMUM_EXTENT,
        "examples_per_category": examples_per_category,
        "heldout_per_category": heldout_per_category,
        "total": len(records),
        "calibration_count": sum(item.split == "calibration" for item in records),
        "heldout_count": sum(item.split == "heldout" for item in records),
        "categories": list(CATEGORIES),
        "splits": list(SPLITS),
        "closed_answers": {
            category: list(values) for category, values in CLOSED_ANSWERS.items()
        },
        "synthesis": {
            "coordinates": "signed integers from integer lattice arithmetic",
            "clusters": "constant integer centre separation per cluster",
            "extents": "integer floor-divided linear spreads",
            "sides": "integer sign of the stored x extrema",
            "floating_point_state": False,
            "generation_order_dependent": False,
            "quantization_is_a_declared_constant": True,
        },
        "records": [asdict(item) for item in records],
    }
    manifest_seal = canonical_sha256(body)
    manifest = root / "manifest.json"
    manifest.write_bytes(
        (
            json.dumps(
                {**body, "manifest_seal": manifest_seal},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    payload = {**body, "manifest_seal": manifest_seal}
    payload["manifest"] = str(manifest)
    payload["manifest_sha256"] = _file_sha256(manifest)
    return payload


def authenticate_native_points3d_probe(
    manifest_path: str | os.PathLike[str],
) -> tuple[Mapping[str, Any], tuple[AuthenticatedPoints3dProbe, ...]]:
    """Verify a probe manifest seal and re-hash every stored point cloud."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise NativePoints3dProbeError(
            "Points3d probe manifest path must be a file."
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativePoints3dProbeError(
            "Points3d probe manifest is invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise NativePoints3dProbeError("Points3d probe manifest is not an object.")
    body = {key: value for key, value in payload.items() if key != "manifest_seal"}
    if canonical_sha256(body) != payload.get("manifest_seal"):
        raise NativePoints3dProbeError("Points3d probe manifest seal is invalid.")
    if payload.get("schema") != PROBE_SCHEMA:
        raise NativePoints3dProbeError(
            "Points3d probe manifest does not use the authenticated schema."
        )
    if payload.get("points_hash_basis") != POINTS3D_HASH_BASIS:
        raise NativePoints3dProbeError(
            "Points3d probe manifest does not use the canonical quantized-point "
            "hash basis."
        )
    if payload.get("scale") != list(PROBE_SCALE) or payload.get("zero_point") != list(
        PROBE_ZERO_POINT
    ):
        raise NativePoints3dProbeError(
            "Points3d probe manifest was generated against a different "
            "quantization; regenerate the probes against this one."
        )
    source_records = payload.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise NativePoints3dProbeError(
            "Points3d probe manifest records are missing."
        )
    root = manifest.parent.resolve()
    authenticated: list[AuthenticatedPoints3dProbe] = []
    for index, source in enumerate(source_records):
        if not isinstance(source, Mapping):
            raise NativePoints3dProbeError(
                "Points3d probe manifest record is not an object."
            )
        category = str(source.get("category"))
        split = str(source.get("split"))
        answer = str(source.get("answer"))
        if category not in CATEGORIES or split not in SPLITS:
            raise NativePoints3dProbeError(
                f"Points3d probe record {index} has an unknown category or split."
            )
        if answer not in CLOSED_ANSWERS[category]:
            raise NativePoints3dProbeError(
                f"Points3d probe record {index} answer escapes its closed set."
            )
        cloud_path = (root / str(source.get("points"))).resolve()
        try:
            cloud_path.relative_to(root)
        except ValueError as exc:
            raise NativePoints3dProbeError(
                "Points3d probe manifest path escapes the manifest directory."
            ) from exc
        if not cloud_path.is_file():
            raise NativePoints3dProbeError(
                f"Points3d probe file is missing: {cloud_path}."
            )
        if (
            _file_sha256(cloud_path) != source.get("container_sha256")
            or cloud_path.stat().st_size != source.get("container_bytes")
        ):
            raise NativePoints3dProbeError(
                f"Points3d probe container changed for record {index}."
            )
        points, quantization = read_points3d_cloud(cloud_path)
        decoded = canonical_points3d_sha256(
            points,
            scale=quantization.scale,
            zero_point=quantization.zero_point,
        )
        if (
            decoded != source.get("points_sha256")
            or int(points.shape[0]) != source.get("point_count")
            or int(points.shape[1]) != source.get("components")
            or source.get("dtype") != POINT_DTYPE
            or list(quantization.scale) != source.get("scale")
            or list(quantization.zero_point) != source.get("zero_point")
        ):
            raise NativePoints3dProbeError(
                f"Decoded points3d authentication failed for record {index}."
            )
        authenticated.append(
            AuthenticatedPoints3dProbe(
                index=index,
                probe_id=str(source["probe_id"]),
                category=category,
                split=split,
                prompt=str(source["prompt"]),
                answer=answer,
                path=str(cloud_path),
                points_sha256=decoded,
                container_sha256=str(source["container_sha256"]),
                container_bytes=int(source["container_bytes"]),
                point_count=int(points.shape[0]),
                components=int(points.shape[1]),
                dtype=POINT_DTYPE,
                scale=quantization.scale,
                zero_point=quantization.zero_point,
                record_sha256=canonical_sha256(dict(source)),
            )
        )
    return MappingProxyType(dict(payload)), tuple(authenticated)


__all__ = [
    "CATEGORIES",
    "CLOSED_ANSWERS",
    "CLUSTER_EXTENT",
    "CLUSTER_GAP",
    "CODEC_ID",
    "COMPONENTS",
    "CONTAINER_HEADER_BYTES",
    "COORDINATE_LIMIT",
    "MODALITY",
    "POINTS3D_HASH_BASIS",
    "POINT_COUNT",
    "POINT_DTYPE",
    "PROBE_QUANTIZATION",
    "PROBE_SCALE",
    "PROBE_SCHEMA",
    "PROBE_ZERO_POINT",
    "PROMPTS",
    "SPLITS",
    "SPREAD_MINIMUM_EXTENT",
    "AuthenticatedPoints3dProbe",
    "NativePoints3dProbeError",
    "Points3dProbeRecord",
    "authenticate_native_points3d_probe",
    "build_native_points3d_probe",
    "canonical_points3d_sha256",
    "decoded_points3d_sha256",
    "read_points3d_cloud",
    "write_points3d_cloud",
]
