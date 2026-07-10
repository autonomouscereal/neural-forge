"""Deterministic raw-RGB capability probes for native multimodal transfer."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROBE_SCHEMA = "neural-forge-native-vision-probe-v1"
_COLORS = {
    "red": (220, 35, 45),
    "green": (25, 165, 80),
    "blue": (35, 90, 220),
    "yellow": (235, 205, 35),
    "purple": (145, 65, 190),
    "orange": (235, 120, 30),
}


@dataclass(frozen=True, slots=True)
class VisionProbeRecord:
    probe_id: str
    category: str
    image: str
    prompt: str
    answer: str
    split: str
    image_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_image_sha256(image: Any) -> str:
    canonical = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{canonical.width}x{canonical.height}\n".encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _color_probe(index: int, rng: random.Random):
    from PIL import Image, ImageDraw

    name = tuple(_COLORS)[index % len(_COLORS)]
    image = Image.new("RGB", (64, 64), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    inset = 5 + rng.randrange(0, 8)
    draw.rectangle((inset, inset, 63 - inset, 63 - inset), fill=_COLORS[name])
    return image, "What is the dominant colored shape? Answer with one word.", name


def _count_probe(index: int, rng: random.Random):
    from PIL import Image, ImageDraw

    count = 1 + index % 5
    color_name = tuple(_COLORS)[(index // 5) % len(_COLORS)]
    image = Image.new("RGB", (64, 64), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    positions = [(12, 12), (32, 12), (52, 12), (22, 40), (44, 42)]
    radius = 6 + rng.randrange(0, 3)
    for x, y in positions[:count]:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=_COLORS[color_name],
            outline=(20, 20, 20),
        )
    prompt = f"How many {color_name} circles are shown? Answer with one digit."
    return image, prompt, str(count)


def _spatial_probe(index: int, rng: random.Random):
    from PIL import Image, ImageDraw

    relations = ("left", "right", "above", "below")
    relation = relations[index % len(relations)]
    jitter = rng.randrange(-2, 3)
    positions = {
        "left": ((15, 32 + jitter), (49, 32)),
        "right": ((49, 32 + jitter), (15, 32)),
        "above": ((32 + jitter, 15), (32, 49)),
        "below": ((32 + jitter, 49), (32, 15)),
    }
    red, blue = positions[relation]
    image = Image.new("RGB", (64, 64), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    for (x, y), color in ((red, _COLORS["red"]), (blue, _COLORS["blue"])):
        draw.rectangle((x - 7, y - 7, x + 7, y + 7), fill=color)
    return (
        image,
        "Where is the red square relative to the blue square? Answer left, right, above, or below.",
        relation,
    )


def _ocr_probe(index: int, rng: random.Random):
    from PIL import Image, ImageDraw

    code = f"{(index * 37 + 104 + rng.randrange(0, 7)) % 1000:03d}"
    image = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    segment_names = {
        "0": "abcdef",
        "1": "bc",
        "2": "abdeg",
        "3": "abcdg",
        "4": "bcfg",
        "5": "acdfg",
        "6": "acdefg",
        "7": "abc",
        "8": "abcdefg",
        "9": "abcdfg",
    }
    segments = {
        "a": (3, 0, 10, 2),
        "b": (11, 3, 13, 13),
        "c": (11, 17, 13, 27),
        "d": (3, 28, 10, 30),
        "e": (0, 17, 2, 27),
        "f": (0, 3, 2, 13),
        "g": (3, 14, 10, 16),
    }
    for digit_index, digit in enumerate(code):
        x_offset = 7 + digit_index * 18
        y_offset = 16
        for segment in segment_names[digit]:
            left, top, right, bottom = segments[segment]
            draw.rectangle(
                (
                    x_offset + left,
                    y_offset + top,
                    x_offset + right,
                    y_offset + bottom,
                ),
                fill=(15, 15, 15),
            )
    return image, "Read the three-digit code. Answer with only the digits.", code


_BUILDERS = {
    "color": _color_probe,
    "count": _count_probe,
    "spatial": _spatial_probe,
    "ocr": _ocr_probe,
}


def build_native_vision_probe(
    output_dir: str | Path,
    *,
    examples_per_category: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    if isinstance(examples_per_category, bool) or examples_per_category <= 0:
        raise ValueError("examples_per_category must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    root = Path(output_dir)
    images = root / "images"
    root.mkdir(parents=True, exist_ok=False)
    images.mkdir()
    rng = random.Random(seed)
    records: list[VisionProbeRecord] = []
    heldout_per_category = max(1, examples_per_category // 4)
    heldout_start = examples_per_category - heldout_per_category
    for category, builder in _BUILDERS.items():
        for category_index in range(examples_per_category):
            image, prompt, answer = builder(category_index, rng)
            probe_id = f"{category}_{category_index:04d}"
            relative = Path("images") / f"{probe_id}.png"
            path = root / relative
            image.save(path, format="PNG", optimize=False)
            split = "heldout" if category_index >= heldout_start else "calibration"
            records.append(
                VisionProbeRecord(
                    probe_id=probe_id,
                    category=category,
                    image=relative.as_posix(),
                    prompt=prompt,
                    answer=answer,
                    split=split,
                    image_sha256=canonical_image_sha256(image),
                )
            )
    payload = {
        "schema": PROBE_SCHEMA,
        "seed": seed,
        "image_format": "RGB uint8 64x64 PNG",
        "image_hash_basis": "sha256('RGB:<width>x<height>\\n' + decoded RGB bytes)",
        "examples_per_category": examples_per_category,
        "heldout_per_category": heldout_per_category,
        "total": len(records),
        "calibration_count": sum(item.split == "calibration" for item in records),
        "heldout_count": sum(item.split == "heldout" for item in records),
        "categories": list(_BUILDERS),
        "records": [asdict(item) for item in records],
    }
    manifest = root / "manifest.json"
    manifest.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    payload["manifest"] = str(manifest)
    payload["manifest_sha256"] = _sha256(manifest)
    return payload


__all__ = [
    "PROBE_SCHEMA",
    "VisionProbeRecord",
    "build_native_vision_probe",
    "canonical_image_sha256",
]
