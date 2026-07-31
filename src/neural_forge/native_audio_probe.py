"""Deterministic raw-PCM16 capability probes for native audio transfer.

The generator is the audio mirror of
:mod:`neural_forge.native_vision_probe`: it synthesizes small, fully
deterministic waveforms with closed answer sets, seals every decoded payload
under :data:`AUDIO_HASH_BASIS`, and writes one sealed manifest.

Determinism is structural rather than incidental.  Tones come from an integer
phase accumulator indexing a fixed int16 wavetable, chords are integer sums of
those tones, noise comes from a seeded PCG64 integer stream, and every probe
draws its jitter from a string-seeded :class:`random.Random`.  No probe depends
on generation order.  The wavetable is the only value derived from ``math.sin``
and its digest is sealed into the manifest, so a platform whose libm rounds a
table entry differently fails authentication loudly instead of silently
producing different audio under the same seed.

Only the standard library and NumPy are used: audio is synthesized in NumPy and
stored as RIFF/WAVE through :mod:`wave`.  There is no ``soundfile`` or
``librosa`` dependency and no learned component of any kind.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .dynamic_behavior_gate import AUDIO_HASH_BASIS, audio_content_sha256
from .dynamic_circuit_ir import MODALITY_AUDIO, canonical_sha256


PROBE_SCHEMA = "neural-forge-native-audio-probe-v1"
MODALITY = MODALITY_AUDIO
CODEC_ID = "audio/wav-pcm16-le"

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
PROBE_FRAMES = 8_000

CATEGORIES = ("tone_count", "pitch_direction", "sound_type")
SPLITS = ("calibration", "heldout")
CLOSED_ANSWERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "tone_count": ("1", "2", "3", "4", "5"),
        "pitch_direction": ("rising", "falling", "constant"),
        "sound_type": ("silence", "tone", "noise"),
    }
)
PROMPTS: Mapping[str, str] = MappingProxyType(
    {
        "tone_count": (
            "How many separate tone bursts are in this clip? Answer with one "
            "digit."
        ),
        "pitch_direction": (
            "Does the pitch rise, fall, or stay constant? Answer rising, "
            "falling, or constant."
        ),
        "sound_type": (
            "Is this clip silence, a tone, or noise? Answer silence, tone, or "
            "noise."
        ),
    }
)

_WAVETABLE_SIZE = 1024
_PHASE_BITS = 16
_PHASE_MODULUS = _WAVETABLE_SIZE << _PHASE_BITS
_FULL_SCALE = 32_767
_TONE_AMPLITUDE = 11_000
_NOISE_AMPLITUDE = 9_000
_TONE_FREQUENCIES = (220, 294, 349, 440, 523, 660)
_LADDER = (196, 247, 294, 349, 440, 523, 622, 740)
_CHORD_INTERVALS = ((0, 2, 4), (0, 3, 5), (1, 3, 6))


class NativeAudioProbeError(ValueError):
    """A probe request, manifest, or stored waveform is invalid."""


def _build_wavetable() -> np.ndarray:
    table = np.array(
        [
            int(round(_FULL_SCALE * math.sin(2.0 * math.pi * index / _WAVETABLE_SIZE)))
            for index in range(_WAVETABLE_SIZE)
        ],
        dtype=np.int64,
    )
    table.setflags(write=False)
    return table


WAVETABLE = _build_wavetable()
WAVETABLE_SHA256 = hashlib.sha256(WAVETABLE.astype("<i4").tobytes()).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeAudioProbeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativeAudioProbeError(f"{name} must be a non-negative integer.")
    return value


def _stream_seed(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _tone(frames: int, frequency_hz: int, amplitude: int, *, phase: int = 0) -> Any:
    """Synthesize one integer-phase sine burst with no floating-point state."""

    count = _nonnegative_int(frames, "tone frames")
    increment = (
        _positive_int(frequency_hz, "tone frequency") * _PHASE_MODULUS
    ) // SAMPLE_RATE
    positions = np.arange(count, dtype=np.int64)
    phases = (int(phase) + increment * positions) % _PHASE_MODULUS
    values = WAVETABLE[phases >> _PHASE_BITS]
    return (values * _positive_int(amplitude, "tone amplitude")) // _FULL_SCALE


def _chord(frames: int, frequencies: tuple[int, ...], amplitude: int) -> Any:
    if not frequencies:
        raise NativeAudioProbeError("A chord needs at least one frequency.")
    total = np.zeros(_nonnegative_int(frames, "chord frames"), dtype=np.int64)
    share = max(1, amplitude // len(frequencies))
    for offset, frequency in enumerate(frequencies):
        total += _tone(
            frames,
            frequency,
            share,
            phase=(offset * _PHASE_MODULUS) // 8,
        )
    return total


def _noise(frames: int, amplitude: int, material: str) -> Any:
    generator = np.random.Generator(np.random.PCG64(_stream_seed(material)))
    return generator.integers(
        -_positive_int(amplitude, "noise amplitude"),
        amplitude + 1,
        size=_nonnegative_int(frames, "noise frames"),
        dtype=np.int64,
    )


def _finalize(samples: Any) -> Any:
    return np.clip(samples, -32_768, _FULL_SCALE).astype(np.int16)


def _tone_count_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    count = 1 + index % 5
    frequency = _TONE_FREQUENCIES[(index // 5) % len(_TONE_FREQUENCIES)]
    burst = 640 + 40 * rng.randrange(0, 4)
    gap = 420 + 40 * rng.randrange(0, 4)
    cursor = 200 + 40 * rng.randrange(0, 4)
    samples = np.zeros(PROBE_FRAMES, dtype=np.int64)
    for _burst_index in range(count):
        end = cursor + burst
        if end > PROBE_FRAMES:
            raise NativeAudioProbeError("Tone-count probe layout overflows the clip.")
        samples[cursor:end] = _tone(burst, frequency, _TONE_AMPLITUDE)
        cursor = end + gap
    return _finalize(samples), PROMPTS["tone_count"], str(count)


def _pitch_direction_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    direction = CLOSED_ANSWERS["pitch_direction"][index % 3]
    step = 1 + rng.randrange(0, 2)
    start = (index // 3) % (len(_LADDER) - 2 * step)
    ladder = tuple(_LADDER[start + position * step] for position in range(3))
    if direction == "falling":
        ladder = tuple(reversed(ladder))
    elif direction == "constant":
        ladder = (_LADDER[start],) * 3
    segment = 2_200 + 40 * rng.randrange(0, 4)
    gap = 120 + 20 * rng.randrange(0, 4)
    cursor = 100 + 20 * rng.randrange(0, 4)
    samples = np.zeros(PROBE_FRAMES, dtype=np.int64)
    for frequency in ladder:
        end = cursor + segment
        if end > PROBE_FRAMES:
            raise NativeAudioProbeError(
                "Pitch-direction probe layout overflows the clip."
            )
        samples[cursor:end] = _tone(segment, frequency, _TONE_AMPLITUDE)
        cursor = end + gap
    return _finalize(samples), PROMPTS["pitch_direction"], direction


def _sound_type_probe(index: int, rng: random.Random) -> tuple[Any, str, str]:
    kind = CLOSED_ANSWERS["sound_type"][index % 3]
    samples = np.zeros(PROBE_FRAMES, dtype=np.int64)
    if kind == "tone":
        variant = (index // 3) % 2
        onset = 400 + 40 * rng.randrange(0, 4)
        length = PROBE_FRAMES - 2 * onset
        if variant == 0:
            frequency = _TONE_FREQUENCIES[(index // 6) % len(_TONE_FREQUENCIES)]
            samples[onset : onset + length] = _tone(
                length,
                frequency,
                _TONE_AMPLITUDE,
            )
        else:
            interval = _CHORD_INTERVALS[(index // 6) % len(_CHORD_INTERVALS)]
            frequencies = tuple(_LADDER[position] for position in interval)
            samples[onset : onset + length] = _chord(
                length,
                frequencies,
                _TONE_AMPLITUDE,
            )
    elif kind == "noise":
        onset = 300 + 40 * rng.randrange(0, 4)
        length = PROBE_FRAMES - 2 * onset
        samples[onset : onset + length] = _noise(
            length,
            _NOISE_AMPLITUDE,
            f"sound_type:{index}:noise",
        )
    return _finalize(samples), PROMPTS["sound_type"], kind


_BUILDERS = {
    "tone_count": _tone_count_probe,
    "pitch_direction": _pitch_direction_probe,
    "sound_type": _sound_type_probe,
}


@dataclass(frozen=True, slots=True)
class AudioProbeRecord:
    """One manifest row describing a stored, content-bound waveform."""

    probe_id: str
    category: str
    audio: str
    prompt: str
    answer: str
    split: str
    audio_sha256: str
    container_sha256: str
    container_bytes: int
    frames: int
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class AuthenticatedAudioProbe:
    """One manifest row whose stored waveform was re-decoded and re-hashed."""

    index: int
    probe_id: str
    category: str
    split: str
    prompt: str
    answer: str
    path: str
    audio_sha256: str
    container_sha256: str
    container_bytes: int
    frames: int
    sample_rate: int
    channels: int
    record_sha256: str


def canonical_audio_sha256(
    samples: Any,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> str:
    """Hash decoded PCM16 samples under :data:`AUDIO_HASH_BASIS`."""

    array = np.asarray(samples)
    if array.dtype != np.int16:
        raise NativeAudioProbeError("Canonical audio hashing requires int16 samples.")
    flat = np.ascontiguousarray(array.reshape(-1))
    channel_count = _positive_int(channels, "channels")
    if flat.size % channel_count:
        raise NativeAudioProbeError("Sample count is not a whole number of frames.")
    return audio_content_sha256(
        _positive_int(sample_rate, "sample_rate"),
        channel_count,
        flat.size // channel_count,
        flat.astype("<i2", copy=False).tobytes(),
    )


def write_pcm16_wav(
    path: str | os.PathLike[str],
    samples: Any,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> Path:
    """Write interleaved int16 samples as little-endian RIFF/WAVE."""

    array = np.asarray(samples)
    if array.dtype != np.int16:
        raise NativeAudioProbeError("PCM16 output requires int16 samples.")
    destination = Path(path)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(_positive_int(channels, "channels"))
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(_positive_int(sample_rate, "sample_rate"))
        handle.writeframes(np.ascontiguousarray(array.reshape(-1)).tobytes())
    return destination


def read_pcm16_wav(path: str | os.PathLike[str]) -> tuple[Any, int, int]:
    """Decode a PCM16 WAV into ``(samples, sample_rate, channels)``."""

    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            if handle.getsampwidth() != SAMPLE_WIDTH_BYTES:
                raise NativeAudioProbeError(
                    f"Audio probe {source} is not 16-bit PCM."
                )
            if handle.getcomptype() != "NONE":
                raise NativeAudioProbeError(f"Audio probe {source} is compressed.")
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (OSError, wave.Error, EOFError) as exc:
        raise NativeAudioProbeError(f"Audio probe cannot be decoded: {source}.") from exc
    if len(raw) != frames * channels * SAMPLE_WIDTH_BYTES:
        raise NativeAudioProbeError(f"Audio probe {source} is truncated.")
    return np.frombuffer(raw, dtype=np.int16), sample_rate, channels


def decoded_audio_sha256(path: str | os.PathLike[str]) -> str:
    """Re-decode a stored waveform and rehash it under the canonical basis."""

    samples, sample_rate, channels = read_pcm16_wav(path)
    return canonical_audio_sha256(
        samples,
        sample_rate=sample_rate,
        channels=channels,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_native_audio_probe(
    output_dir: str | os.PathLike[str],
    *,
    examples_per_category: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a sealed synthetic audio probe suite under ``output_dir``."""

    _positive_int(examples_per_category, "examples_per_category")
    _nonnegative_int(seed, "seed")
    root = Path(output_dir)
    audio_dir = root / "audio"
    root.mkdir(parents=True, exist_ok=False)
    audio_dir.mkdir()
    records: list[AudioProbeRecord] = []
    heldout_per_category = max(1, examples_per_category // 4)
    heldout_start = examples_per_category - heldout_per_category
    for category, builder in _BUILDERS.items():
        for category_index in range(examples_per_category):
            rng = random.Random(f"neural-forge-audio-probe:{seed}:{category}:"
                                f"{category_index}")
            samples, prompt, answer = builder(category_index, rng)
            if answer not in CLOSED_ANSWERS[category]:
                raise NativeAudioProbeError(
                    f"Probe answer {answer!r} escapes the closed set for "
                    f"{category!r}."
                )
            probe_id = f"{category}_{category_index:04d}"
            relative = Path("audio") / f"{probe_id}.wav"
            path = write_pcm16_wav(root / relative, samples)
            records.append(
                AudioProbeRecord(
                    probe_id=probe_id,
                    category=category,
                    audio=relative.as_posix(),
                    prompt=prompt,
                    answer=answer,
                    split=(
                        "heldout"
                        if category_index >= heldout_start
                        else "calibration"
                    ),
                    audio_sha256=canonical_audio_sha256(samples),
                    container_sha256=_file_sha256(path),
                    container_bytes=path.stat().st_size,
                    frames=int(np.asarray(samples).reshape(-1).size // CHANNELS),
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                )
            )
    body = {
        "schema": PROBE_SCHEMA,
        "seed": seed,
        "modality": MODALITY,
        "codec": CODEC_ID,
        "audio_format": (
            f"PCM16 mono {SAMPLE_RATE} Hz little-endian RIFF/WAVE, "
            f"{PROBE_FRAMES} frames"
        ),
        "audio_hash_basis": AUDIO_HASH_BASIS,
        "wavetable_sha256": WAVETABLE_SHA256,
        "wavetable_size": _WAVETABLE_SIZE,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "frames": PROBE_FRAMES,
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
            "tones": "integer phase accumulator over a sealed int16 wavetable",
            "chords": "integer sums of phase-offset wavetable tones",
            "noise": "seeded PCG64 integer stream",
            "silence": "exact zero samples",
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


def authenticate_native_audio_probe(
    manifest_path: str | os.PathLike[str],
) -> tuple[Mapping[str, Any], tuple[AuthenticatedAudioProbe, ...]]:
    """Verify a probe manifest seal and re-hash every stored waveform."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise NativeAudioProbeError("Audio probe manifest path must be a file.")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeAudioProbeError("Audio probe manifest is invalid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise NativeAudioProbeError("Audio probe manifest is not an object.")
    body = {key: value for key, value in payload.items() if key != "manifest_seal"}
    if canonical_sha256(body) != payload.get("manifest_seal"):
        raise NativeAudioProbeError("Audio probe manifest seal is invalid.")
    if payload.get("schema") != PROBE_SCHEMA:
        raise NativeAudioProbeError(
            "Audio probe manifest does not use the authenticated schema."
        )
    if payload.get("audio_hash_basis") != AUDIO_HASH_BASIS:
        raise NativeAudioProbeError(
            "Audio probe manifest does not use the canonical PCM16 hash basis."
        )
    if payload.get("wavetable_sha256") != WAVETABLE_SHA256:
        raise NativeAudioProbeError(
            "Audio probe manifest was generated from a different sine "
            "wavetable; regenerate the probes on this platform."
        )
    source_records = payload.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise NativeAudioProbeError("Audio probe manifest records are missing.")
    root = manifest.parent.resolve()
    authenticated: list[AuthenticatedAudioProbe] = []
    for index, source in enumerate(source_records):
        if not isinstance(source, Mapping):
            raise NativeAudioProbeError("Audio probe manifest record is not an object.")
        category = str(source.get("category"))
        split = str(source.get("split"))
        answer = str(source.get("answer"))
        if category not in CATEGORIES or split not in SPLITS:
            raise NativeAudioProbeError(
                f"Audio probe record {index} has an unknown category or split."
            )
        if answer not in CLOSED_ANSWERS[category]:
            raise NativeAudioProbeError(
                f"Audio probe record {index} answer escapes its closed set."
            )
        audio = (root / str(source.get("audio"))).resolve()
        try:
            audio.relative_to(root)
        except ValueError as exc:
            raise NativeAudioProbeError(
                "Audio probe manifest path escapes the manifest directory."
            ) from exc
        if not audio.is_file():
            raise NativeAudioProbeError(f"Audio probe file is missing: {audio}.")
        if (
            _file_sha256(audio) != source.get("container_sha256")
            or audio.stat().st_size != source.get("container_bytes")
        ):
            raise NativeAudioProbeError(
                f"Audio probe container changed for record {index}."
            )
        samples, sample_rate, channels = read_pcm16_wav(audio)
        decoded = canonical_audio_sha256(
            samples,
            sample_rate=sample_rate,
            channels=channels,
        )
        if (
            decoded != source.get("audio_sha256")
            or sample_rate != source.get("sample_rate")
            or channels != source.get("channels")
            or samples.size // channels != source.get("frames")
        ):
            raise NativeAudioProbeError(
                f"Decoded audio authentication failed for record {index}."
            )
        authenticated.append(
            AuthenticatedAudioProbe(
                index=index,
                probe_id=str(source["probe_id"]),
                category=category,
                split=split,
                prompt=str(source["prompt"]),
                answer=answer,
                path=str(audio),
                audio_sha256=decoded,
                container_sha256=str(source["container_sha256"]),
                container_bytes=int(source["container_bytes"]),
                frames=int(source["frames"]),
                sample_rate=int(sample_rate),
                channels=int(channels),
                record_sha256=canonical_sha256(dict(source)),
            )
        )
    return MappingProxyType(dict(payload)), tuple(authenticated)


__all__ = [
    "AUDIO_HASH_BASIS",
    "CATEGORIES",
    "CLOSED_ANSWERS",
    "CODEC_ID",
    "CHANNELS",
    "MODALITY",
    "PROBE_FRAMES",
    "PROBE_SCHEMA",
    "PROMPTS",
    "SAMPLE_RATE",
    "SPLITS",
    "WAVETABLE_SHA256",
    "AudioProbeRecord",
    "AuthenticatedAudioProbe",
    "NativeAudioProbeError",
    "authenticate_native_audio_probe",
    "build_native_audio_probe",
    "canonical_audio_sha256",
    "decoded_audio_sha256",
    "read_pcm16_wav",
    "write_pcm16_wav",
]
