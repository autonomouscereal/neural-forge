"""Deterministic raw-RGB24 clip probes for native video transfer.

The generator is the video mirror of
:mod:`neural_forge.native_audio_probe`: it synthesizes small, fully
deterministic moving-shape clips with closed answer sets, seals every decoded
payload under :data:`VIDEO_HASH_BASIS`, and writes one sealed manifest.

Determinism is structural rather than incidental, and stricter than the audio
lane's: every pixel is an integer palette entry written into an integer array
by integer slice arithmetic, so **no floating-point value is produced anywhere
during generation**.  Each probe draws its jitter from a string-seeded
:class:`random.Random`, so no probe depends on generation order.  The palette
is the only shared constant a clip's content depends on and its digest is
sealed into the manifest, so a suite generated against a different palette
fails authentication loudly instead of silently producing different frames
under the same seed.

Only the standard library and NumPy are used.  Clips are stored in the
self-describing ``NFVID1`` raw container written here: the decoder is eight
bytes of magic plus six little-endian ``uint32`` header fields followed by
row-major RGB24 frames in playback order.  There is no ``Pillow``, ``av``, or
``ffmpeg`` dependency, no compression, and no learned component of any kind --
the container carries its own geometry and frame rate, so a stored clip is
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
from typing import Any, Mapping

import numpy as np

from .dynamic_behavior_gate import VIDEO_HASH_BASIS, video_content_sha256
from .dynamic_circuit_ir import MODALITY_VIDEO, canonical_sha256


PROBE_SCHEMA = "neural-forge-native-video-probe-v1"
MODALITY = MODALITY_VIDEO
CODEC_ID = "video/nfvid1-rgb24"

FRAME_WIDTH = 32
FRAME_HEIGHT = 32
PROBE_FRAMES = 6
FPS = 8
CHANNELS = 3

CATEGORIES = (
    "motion_direction",
    "shape_count",
    "shape_color",
    "shape_event",
)
SPLITS = ("calibration", "heldout")
CLOSED_ANSWERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "motion_direction": ("left", "right", "up", "down", "none"),
        "shape_count": ("1", "2", "3"),
        "shape_color": ("red", "green", "blue", "yellow", "purple", "orange"),
        "shape_event": ("appears", "disappears", "persists"),
    }
)
PROMPTS: Mapping[str, str] = MappingProxyType(
    {
        "motion_direction": (
            "Which way does the shape move across this clip? Answer left, "
            "right, up, down, or none."
        ),
        "shape_count": (
            "How many squares are in this clip? Answer with one digit."
        ),
        "shape_color": (
            "What color is the shape in the final frame? Answer with one word."
        ),
        "shape_event": (
            "Does the shape appear, disappear, or stay for the whole clip? "
            "Answer appears, disappears, or persists."
        ),
    }
)

_BACKGROUND = (245, 245, 245)
_COLORS: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    {
        "red": (220, 35, 45),
        "green": (25, 165, 80),
        "blue": (35, 90, 220),
        "yellow": (235, 205, 35),
        "purple": (145, 65, 190),
        "orange": (235, 120, 30),
    }
)
_SHAPE_SIZE = 8
_COUNT_SLOTS = ((2, 2), (2, 18), (18, 2))

_CONTAINER_MAGIC = b"NFVID1\x00\x00"
_CONTAINER_HEADER = struct.Struct("<6I")
CONTAINER_HEADER_BYTES = len(_CONTAINER_MAGIC) + _CONTAINER_HEADER.size


class NativeVideoProbeError(ValueError):
    """A probe request, manifest, or stored clip is invalid."""


def _build_palette() -> np.ndarray:
    table = np.array(
        [_BACKGROUND, *(_COLORS[name] for name in sorted(_COLORS))],
        dtype=np.uint8,
    )
    table.setflags(write=False)
    return table


PALETTE = _build_palette()
PALETTE_SHA256 = hashlib.sha256(PALETTE.tobytes()).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeVideoProbeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativeVideoProbeError(f"{name} must be a non-negative integer.")
    return value


def _blank_clip() -> Any:
    clip = np.empty((PROBE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), dtype=np.uint8)
    for channel, level in enumerate(_BACKGROUND):
        clip[..., channel] = level
    return clip


def _draw_square(
    clip: Any,
    frame_index: int,
    top: int,
    left: int,
    color: tuple[int, int, int],
    *,
    size: int = _SHAPE_SIZE,
) -> None:
    """Paint one integer-addressed square into one frame, bounds-checked."""

    _nonnegative_int(frame_index, "frame index")
    _nonnegative_int(top, "square top")
    _nonnegative_int(left, "square left")
    _positive_int(size, "square size")
    if (
        frame_index >= PROBE_FRAMES
        or top + size > FRAME_HEIGHT
        or left + size > FRAME_WIDTH
    ):
        raise NativeVideoProbeError("Square layout escapes the clip geometry.")
    clip[frame_index, top : top + size, left : left + size] = color


def _motion_direction_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    direction = CLOSED_ANSWERS["motion_direction"][index % 5]
    color = _COLORS[tuple(_COLORS)[(index // 5) % len(_COLORS)]]
    step = 2 + rng.randrange(0, 3)
    travel = step * (PROBE_FRAMES - 1)
    limit = FRAME_WIDTH - _SHAPE_SIZE
    if travel > limit:
        raise NativeVideoProbeError("Motion probe travel overflows the clip.")
    origin = rng.randrange(0, limit - travel + 1)
    cross = rng.randrange(0, limit + 1)
    clip = _blank_clip()
    for frame_index in range(PROBE_FRAMES):
        offset = step * frame_index
        if direction == "right":
            top, left = cross, origin + offset
        elif direction == "left":
            top, left = cross, origin + travel - offset
        elif direction == "down":
            top, left = origin + offset, cross
        elif direction == "up":
            top, left = origin + travel - offset, cross
        else:
            top, left = cross, origin
        _draw_square(clip, frame_index, top, left, color)
    return clip, PROMPTS["motion_direction"], direction


def _shape_count_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    count = 1 + index % 3
    color = _COLORS[tuple(_COLORS)[(index // 3) % len(_COLORS)]]
    jitter = rng.randrange(0, 3)
    clip = _blank_clip()
    for frame_index in range(PROBE_FRAMES):
        for slot_top, slot_left in _COUNT_SLOTS[:count]:
            _draw_square(
                clip,
                frame_index,
                slot_top + jitter,
                slot_left + frame_index,
                color,
            )
    return clip, PROMPTS["shape_count"], str(count)


def _shape_color_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    names = tuple(CLOSED_ANSWERS["shape_color"])
    final_name = names[index % len(names)]
    variant = (index // len(names)) % 2
    opening_name = names[(index + 1 + rng.randrange(0, 2)) % len(names)]
    if opening_name == final_name:
        opening_name = names[(index + 2) % len(names)]
    switch = PROBE_FRAMES // 2
    step = 2 + rng.randrange(0, 2)
    travel = step * (PROBE_FRAMES - 1)
    limit = FRAME_WIDTH - _SHAPE_SIZE
    origin = rng.randrange(0, limit - travel + 1)
    cross = rng.randrange(0, limit + 1)
    clip = _blank_clip()
    for frame_index in range(PROBE_FRAMES):
        name = (
            final_name
            if variant == 0 or frame_index >= switch
            else opening_name
        )
        _draw_square(
            clip,
            frame_index,
            cross,
            origin + step * frame_index,
            _COLORS[name],
        )
    return clip, PROMPTS["shape_color"], final_name


def _shape_event_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    event = CLOSED_ANSWERS["shape_event"][index % 3]
    color = _COLORS[tuple(_COLORS)[(index // 3) % len(_COLORS)]]
    limit = FRAME_WIDTH - _SHAPE_SIZE
    top = rng.randrange(0, limit + 1)
    left = rng.randrange(0, limit + 1)
    switch = PROBE_FRAMES // 2
    clip = _blank_clip()
    for frame_index in range(PROBE_FRAMES):
        if event == "appears" and frame_index < switch:
            continue
        if event == "disappears" and frame_index >= switch:
            continue
        _draw_square(clip, frame_index, top, left, color)
    return clip, PROMPTS["shape_event"], event


_BUILDERS = {
    "motion_direction": _motion_direction_probe,
    "shape_count": _shape_count_probe,
    "shape_color": _shape_color_probe,
    "shape_event": _shape_event_probe,
}


@dataclass(frozen=True, slots=True)
class VideoProbeRecord:
    """One manifest row describing a stored, content-bound clip."""

    probe_id: str
    category: str
    video: str
    prompt: str
    answer: str
    split: str
    video_sha256: str
    container_sha256: str
    container_bytes: int
    frames: int
    width: int
    height: int
    fps: int


@dataclass(frozen=True, slots=True)
class AuthenticatedVideoProbe:
    """One manifest row whose stored clip was re-decoded and re-hashed."""

    index: int
    probe_id: str
    category: str
    split: str
    prompt: str
    answer: str
    path: str
    video_sha256: str
    container_sha256: str
    container_bytes: int
    frames: int
    width: int
    height: int
    fps: int
    record_sha256: str


def canonical_video_sha256(frames: Any, *, fps: int = FPS) -> str:
    """Hash decoded RGB24 frames under :data:`VIDEO_HASH_BASIS`."""

    array = np.asarray(frames)
    if array.dtype != np.uint8:
        raise NativeVideoProbeError("Canonical video hashing requires uint8 frames.")
    if array.ndim != 4 or int(array.shape[3]) != CHANNELS:
        raise NativeVideoProbeError(
            "Canonical video hashing requires (frames, height, width, 3) RGB."
        )
    payload = np.ascontiguousarray(array).tobytes()
    return video_content_sha256(
        _positive_int(int(array.shape[2]), "width"),
        _positive_int(int(array.shape[1]), "height"),
        _positive_int(fps, "fps"),
        _positive_int(int(array.shape[0]), "frames"),
        payload,
    )


def write_rgb24_video(
    path: str | os.PathLike[str],
    frames: Any,
    *,
    fps: int = FPS,
) -> Path:
    """Write RGB24 frames as one self-describing ``NFVID1`` container."""

    array = np.asarray(frames)
    if array.dtype != np.uint8:
        raise NativeVideoProbeError("RGB24 output requires uint8 frames.")
    if array.ndim != 4 or int(array.shape[3]) != CHANNELS:
        raise NativeVideoProbeError(
            "RGB24 output requires (frames, height, width, 3) frames."
        )
    frame_count = _positive_int(int(array.shape[0]), "frames")
    height = _positive_int(int(array.shape[1]), "height")
    width = _positive_int(int(array.shape[2]), "width")
    payload = np.ascontiguousarray(array).tobytes()
    destination = Path(path)
    with destination.open("xb") as handle:
        handle.write(_CONTAINER_MAGIC)
        handle.write(
            _CONTAINER_HEADER.pack(
                width,
                height,
                frame_count,
                _positive_int(fps, "fps"),
                CHANNELS,
                len(payload),
            )
        )
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def read_rgb24_video(path: str | os.PathLike[str]) -> tuple[Any, int]:
    """Decode an ``NFVID1`` container into ``(frames, fps)``."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise NativeVideoProbeError(f"Video probe cannot be read: {source}.") from exc
    if len(raw) < CONTAINER_HEADER_BYTES or not raw.startswith(_CONTAINER_MAGIC):
        raise NativeVideoProbeError(f"Video probe {source} is not an NFVID1 container.")
    (
        width,
        height,
        frame_count,
        fps,
        channels,
        payload_bytes,
    ) = _CONTAINER_HEADER.unpack(raw[len(_CONTAINER_MAGIC) : CONTAINER_HEADER_BYTES])
    if channels != CHANNELS:
        raise NativeVideoProbeError(f"Video probe {source} is not RGB24.")
    for value, name in (
        (width, "width"),
        (height, "height"),
        (frame_count, "frames"),
        (fps, "fps"),
    ):
        _positive_int(int(value), f"stored {name}")
    expected = frame_count * height * width * CHANNELS
    payload = raw[CONTAINER_HEADER_BYTES:]
    if payload_bytes != expected or len(payload) != expected:
        raise NativeVideoProbeError(f"Video probe {source} is truncated or padded.")
    return (
        np.frombuffer(payload, dtype=np.uint8).reshape(
            int(frame_count),
            int(height),
            int(width),
            CHANNELS,
        ),
        int(fps),
    )


def decoded_video_sha256(path: str | os.PathLike[str]) -> str:
    """Re-decode a stored clip and rehash it under the canonical basis."""

    frames, fps = read_rgb24_video(path)
    return canonical_video_sha256(frames, fps=fps)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_native_video_probe(
    output_dir: str | os.PathLike[str],
    *,
    examples_per_category: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a sealed synthetic video probe suite under ``output_dir``."""

    _positive_int(examples_per_category, "examples_per_category")
    _nonnegative_int(seed, "seed")
    root = Path(output_dir)
    video_dir = root / "video"
    root.mkdir(parents=True, exist_ok=False)
    video_dir.mkdir()
    records: list[VideoProbeRecord] = []
    heldout_per_category = max(1, examples_per_category // 4)
    heldout_start = examples_per_category - heldout_per_category
    for category, builder in _BUILDERS.items():
        for category_index in range(examples_per_category):
            rng = random.Random(f"neural-forge-video-probe:{seed}:{category}:"
                                f"{category_index}")
            clip, prompt, answer = builder(category_index, rng)
            if answer not in CLOSED_ANSWERS[category]:
                raise NativeVideoProbeError(
                    f"Probe answer {answer!r} escapes the closed set for "
                    f"{category!r}."
                )
            probe_id = f"{category}_{category_index:04d}"
            relative = Path("video") / f"{probe_id}.nfv"
            path = write_rgb24_video(root / relative, clip)
            records.append(
                VideoProbeRecord(
                    probe_id=probe_id,
                    category=category,
                    video=relative.as_posix(),
                    prompt=prompt,
                    answer=answer,
                    split=(
                        "heldout"
                        if category_index >= heldout_start
                        else "calibration"
                    ),
                    video_sha256=canonical_video_sha256(clip),
                    container_sha256=_file_sha256(path),
                    container_bytes=path.stat().st_size,
                    # Derived from the synthesized array rather than restated
                    # from the module constants: a record can never describe a
                    # geometry the stored clip does not have.
                    frames=int(clip.shape[0]),
                    height=int(clip.shape[1]),
                    width=int(clip.shape[2]),
                    fps=FPS,
                )
            )
    body = {
        "schema": PROBE_SCHEMA,
        "seed": seed,
        "modality": MODALITY,
        "codec": CODEC_ID,
        "video_format": (
            f"RGB24 {FRAME_WIDTH}x{FRAME_HEIGHT} at {FPS} fps, {PROBE_FRAMES} "
            "frames, raw NFVID1 container"
        ),
        "video_hash_basis": VIDEO_HASH_BASIS,
        "palette_sha256": PALETTE_SHA256,
        "palette_size": int(PALETTE.shape[0]),
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "frames": PROBE_FRAMES,
        "fps": FPS,
        "channels": CHANNELS,
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
            "shapes": "integer palette squares written by integer slice bounds",
            "motion": "constant integer pixel step per frame",
            "events": "frame-index thresholded presence",
            "color_changes": "palette swap at the midpoint frame",
            "floating_point_state": False,
            "generation_order_dependent": False,
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


def authenticate_native_video_probe(
    manifest_path: str | os.PathLike[str],
) -> tuple[Mapping[str, Any], tuple[AuthenticatedVideoProbe, ...]]:
    """Verify a probe manifest seal and re-hash every stored clip."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise NativeVideoProbeError("Video probe manifest path must be a file.")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeVideoProbeError("Video probe manifest is invalid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise NativeVideoProbeError("Video probe manifest is not an object.")
    body = {key: value for key, value in payload.items() if key != "manifest_seal"}
    if canonical_sha256(body) != payload.get("manifest_seal"):
        raise NativeVideoProbeError("Video probe manifest seal is invalid.")
    if payload.get("schema") != PROBE_SCHEMA:
        raise NativeVideoProbeError(
            "Video probe manifest does not use the authenticated schema."
        )
    if payload.get("video_hash_basis") != VIDEO_HASH_BASIS:
        raise NativeVideoProbeError(
            "Video probe manifest does not use the canonical RGB24 hash basis."
        )
    if payload.get("palette_sha256") != PALETTE_SHA256:
        raise NativeVideoProbeError(
            "Video probe manifest was generated from a different palette; "
            "regenerate the probes against this palette."
        )
    source_records = payload.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise NativeVideoProbeError("Video probe manifest records are missing.")
    root = manifest.parent.resolve()
    authenticated: list[AuthenticatedVideoProbe] = []
    for index, source in enumerate(source_records):
        if not isinstance(source, Mapping):
            raise NativeVideoProbeError("Video probe manifest record is not an object.")
        category = str(source.get("category"))
        split = str(source.get("split"))
        answer = str(source.get("answer"))
        if category not in CATEGORIES or split not in SPLITS:
            raise NativeVideoProbeError(
                f"Video probe record {index} has an unknown category or split."
            )
        if answer not in CLOSED_ANSWERS[category]:
            raise NativeVideoProbeError(
                f"Video probe record {index} answer escapes its closed set."
            )
        video = (root / str(source.get("video"))).resolve()
        try:
            video.relative_to(root)
        except ValueError as exc:
            raise NativeVideoProbeError(
                "Video probe manifest path escapes the manifest directory."
            ) from exc
        if not video.is_file():
            raise NativeVideoProbeError(f"Video probe file is missing: {video}.")
        if (
            _file_sha256(video) != source.get("container_sha256")
            or video.stat().st_size != source.get("container_bytes")
        ):
            raise NativeVideoProbeError(
                f"Video probe container changed for record {index}."
            )
        frames, fps = read_rgb24_video(video)
        decoded = canonical_video_sha256(frames, fps=fps)
        if (
            decoded != source.get("video_sha256")
            or fps != source.get("fps")
            or int(frames.shape[0]) != source.get("frames")
            or int(frames.shape[1]) != source.get("height")
            or int(frames.shape[2]) != source.get("width")
        ):
            raise NativeVideoProbeError(
                f"Decoded video authentication failed for record {index}."
            )
        authenticated.append(
            AuthenticatedVideoProbe(
                index=index,
                probe_id=str(source["probe_id"]),
                category=category,
                split=split,
                prompt=str(source["prompt"]),
                answer=answer,
                path=str(video),
                video_sha256=decoded,
                container_sha256=str(source["container_sha256"]),
                container_bytes=int(source["container_bytes"]),
                frames=int(frames.shape[0]),
                width=int(frames.shape[2]),
                height=int(frames.shape[1]),
                fps=int(fps),
                record_sha256=canonical_sha256(dict(source)),
            )
        )
    return MappingProxyType(dict(payload)), tuple(authenticated)


__all__ = [
    "CATEGORIES",
    "CHANNELS",
    "CLOSED_ANSWERS",
    "CODEC_ID",
    "CONTAINER_HEADER_BYTES",
    "FPS",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "MODALITY",
    "PALETTE_SHA256",
    "PROBE_FRAMES",
    "PROBE_SCHEMA",
    "PROMPTS",
    "SPLITS",
    "VIDEO_HASH_BASIS",
    "AuthenticatedVideoProbe",
    "NativeVideoProbeError",
    "VideoProbeRecord",
    "authenticate_native_video_probe",
    "build_native_video_probe",
    "canonical_video_sha256",
    "decoded_video_sha256",
    "read_rgb24_video",
    "write_rgb24_video",
]
