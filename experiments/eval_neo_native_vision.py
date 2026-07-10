#!/usr/bin/env python3
"""Evaluate and immutably cache the NEO1.5 raw-RGB parent baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROBE_SCHEMA = "neural-forge-native-vision-probe-v1"
IMAGE_HASH_BASIS = "sha256('RGB:<width>x<height>\\n' + decoded RGB bytes)"
RUN_SCHEMA = "neural-forge-neo-native-parent-run-v1"
CACHE_SCHEMA = "neural-forge-neo-native-parent-record-v1"
REPORT_SCHEMA = "neural-forge-neo-native-parent-report-v1"
MODEL_ID = "Paranioar/NEO1_5-2B-SFT"

CATEGORIES = ("color", "count", "spatial", "ocr")
SPLITS = ("calibration", "heldout")
COLOR_ANSWERS = frozenset(("red", "green", "blue", "yellow", "purple", "orange"))
SPATIAL_ANSWERS = frozenset(("left", "right", "above", "below"))
SPECIAL_TOKEN_IDS = {
    "<IMG_CONTEXT>": 151669,
    "<img>": 151670,
    "</img>": 151671,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 0.5
RESIZE_FACTOR = 32
MIN_PIXELS = 65_536
MAX_PIXELS = 4_194_304
SOURCE_WIDTH = 64
SOURCE_HEIGHT = 64
EXPECTED_GRID = (16, 16)
EXPECTED_PATCH_SHAPE = (256, 768)

PREPROCESS_SPEC = {
    "name": "neo1.5-load_image_native",
    "source_contract": "VLMEvalKit_ov/utils.py:load_image_native",
    "source_mode": "RGB",
    "source_size": [SOURCE_WIDTH, SOURCE_HEIGHT],
    "resize_factor": RESIZE_FACTOR,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
    "resize_resample": "Pillow Image.resize default",
    "normalization_mean": list(IMAGENET_MEAN),
    "normalization_std": list(IMAGENET_STD),
    "patch_size": PATCH_SIZE,
    "downsample_ratio": DOWNSAMPLE_RATIO,
    "flatten_order": "CHW->[grid_h,patch_h,grid_w,patch_w]->[grid_h,grid_w,C,patch_h,patch_w]",
    "expected_resized_size": [256, 256],
    "expected_grid_hw": list(EXPECTED_GRID),
    "expected_patch_shape": list(EXPECTED_PATCH_SHAPE),
}

_MANIFEST_KEYS = frozenset(
    (
        "schema",
        "seed",
        "image_format",
        "image_hash_basis",
        "examples_per_category",
        "heldout_per_category",
        "total",
        "calibration_count",
        "heldout_count",
        "categories",
        "records",
    )
)
_RECORD_KEYS = frozenset(
    ("probe_id", "category", "image", "prompt", "answer", "split", "image_sha256")
)
_CACHE_KEYS = frozenset(
    (
        "schema",
        "inference_fingerprint",
        "index",
        "probe_id",
        "record_sha256",
        "category",
        "split",
        "expected",
        "raw_output",
        "raw_output_sha256",
        "normalized_prediction",
        "direct_answer",
        "correct",
        "pixel_values_sha256",
        "grid_hw",
        "elapsed_seconds",
        "completed_utc",
    )
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIRECT_WORD_RE = re.compile(r"^\s*[\"'`]?([A-Za-z]+)[\"'`]?[.!]?\s*$")
_DIRECT_DIGIT_RE = re.compile(r"^\s*[\"'`]?([0-9])[\"'`]?[.!]?\s*$")
_DIRECT_OCR_RE = re.compile(r"^\s*[\"'`]?([0-9]{3})[\"'`]?[.!]?\s*$")


class EvaluationContractError(RuntimeError):
    """Raised when an input, cache, or runtime violates the frozen protocol."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise EvaluationContractError(f"{label} must be a SHA-256 string.")
    normalized = value.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise EvaluationContractError(f"{label} must contain exactly 64 hexadecimal characters.")
    return normalized


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, encoded)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"Cannot read valid UTF-8 JSON from {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{label} must contain one JSON object.")
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationContractError(f"{label} must be an integer >= {minimum}.")
    return value


def canonical_decoded_rgb_sha256(image: Any) -> str:
    """Hash decoded pixels using the probe's Pillow/zlib-independent contract."""

    canonical = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{canonical.width}x{canonical.height}\n".encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _verify_answer(category: str, answer: Any, label: str) -> str:
    if not isinstance(answer, str):
        raise EvaluationContractError(f"{label} answer must be a string.")
    if category == "color" and answer not in COLOR_ANSWERS:
        raise EvaluationContractError(f"{label} has invalid color answer {answer!r}.")
    if category == "count" and re.fullmatch(r"[1-5]", answer) is None:
        raise EvaluationContractError(f"{label} count answer must be one digit from 1 through 5.")
    if category == "spatial" and answer not in SPATIAL_ANSWERS:
        raise EvaluationContractError(f"{label} has invalid spatial answer {answer!r}.")
    if category == "ocr" and re.fullmatch(r"[0-9]{3}", answer) is None:
        raise EvaluationContractError(f"{label} OCR answer must be exactly three digits.")
    return answer


def _resolved_image_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EvaluationContractError(f"{label} image path must be a nonempty string.")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise EvaluationContractError(f"{label} image path must stay below the manifest directory.")
    candidate = (root / candidate_relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EvaluationContractError(f"{label} image resolves outside the manifest directory.") from exc
    if not candidate.is_file():
        raise EvaluationContractError(f"{label} image is not a regular file: {candidate}")
    return candidate


def verify_manifest(
    manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate the exact probe schema and every decoded RGB image hash."""

    path = Path(manifest_path).resolve(strict=True)
    expected_digest = _normalize_sha256(expected_manifest_sha256, "manifest SHA-256")
    actual_digest = _file_sha256(path)
    if actual_digest != expected_digest:
        raise EvaluationContractError(
            f"Manifest SHA-256 mismatch: expected {expected_digest}, got {actual_digest}."
        )
    payload = _read_json_object(path, "probe manifest")
    if frozenset(payload) != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS.difference(payload))
        extra = sorted(set(payload).difference(_MANIFEST_KEYS))
        raise EvaluationContractError(
            f"Probe manifest fields do not match the frozen schema; missing={missing}, extra={extra}."
        )
    if payload["schema"] != PROBE_SCHEMA:
        raise EvaluationContractError(
            f"Unsupported probe schema {payload['schema']!r}; expected {PROBE_SCHEMA!r}."
        )
    if payload["image_hash_basis"] != IMAGE_HASH_BASIS:
        raise EvaluationContractError(
            "Unsupported image_hash_basis; encoded-PNG and unspecified hashes are not accepted."
        )
    if payload["image_format"] != "RGB uint8 64x64 PNG":
        raise EvaluationContractError(f"Unsupported image_format {payload['image_format']!r}.")
    _plain_int(payload["seed"], "manifest seed")
    examples_per_category = _plain_int(
        payload["examples_per_category"], "examples_per_category", minimum=1
    )
    heldout_per_category = _plain_int(
        payload["heldout_per_category"], "heldout_per_category", minimum=1
    )
    expected_heldout = max(1, examples_per_category // 4)
    if heldout_per_category != expected_heldout:
        raise EvaluationContractError(
            f"heldout_per_category must be {expected_heldout}, got {heldout_per_category}."
        )
    if payload["categories"] != list(CATEGORIES):
        raise EvaluationContractError(
            f"Manifest categories must be exactly {list(CATEGORIES)!r} in that order."
        )
    total = _plain_int(payload["total"], "manifest total", minimum=1)
    if total != examples_per_category * len(CATEGORIES):
        raise EvaluationContractError("Manifest total does not match category cardinality.")
    records = payload["records"]
    if not isinstance(records, list) or len(records) != total:
        raise EvaluationContractError("Manifest records length does not match total.")

    from PIL import Image, UnidentifiedImageError

    verified_records: list[dict[str, Any]] = []
    split_counts = {split: 0 for split in SPLITS}
    seen_ids: set[str] = set()
    root = path.parent.resolve()
    heldout_start = examples_per_category - heldout_per_category
    for index, source_record in enumerate(records):
        label = f"record[{index}]"
        if not isinstance(source_record, dict) or frozenset(source_record) != _RECORD_KEYS:
            raise EvaluationContractError(f"{label} fields do not match the frozen record schema.")
        record = dict(source_record)
        category_index, item_index = divmod(index, examples_per_category)
        expected_category = CATEGORIES[category_index]
        expected_id = f"{expected_category}_{item_index:04d}"
        if record["category"] != expected_category or record["probe_id"] != expected_id:
            raise EvaluationContractError(
                f"{label} ordering/id mismatch; expected {expected_id!r}."
            )
        if record["probe_id"] in seen_ids:
            raise EvaluationContractError(f"Duplicate probe_id {record['probe_id']!r}.")
        seen_ids.add(record["probe_id"])
        expected_split = "heldout" if item_index >= heldout_start else "calibration"
        if record["split"] != expected_split:
            raise EvaluationContractError(
                f"{label} split mismatch; expected {expected_split!r}."
            )
        split_counts[expected_split] += 1
        if record["image"] != f"images/{expected_id}.png":
            raise EvaluationContractError(f"{label} image path does not match its probe_id.")
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise EvaluationContractError(f"{label} prompt must be nonempty.")
        _verify_answer(expected_category, record["answer"], label)
        expected_image_digest = _normalize_sha256(record["image_sha256"], f"{label} image SHA-256")
        image_path = _resolved_image_path(root, record["image"], label)
        try:
            with Image.open(image_path) as image:
                image.load()
                if image.format != "PNG":
                    raise EvaluationContractError(f"{label} image must decode as PNG.")
                if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                    raise EvaluationContractError(
                        f"{label} image must be {SOURCE_WIDTH}x{SOURCE_HEIGHT}, got {image.size}."
                    )
                actual_image_digest = canonical_decoded_rgb_sha256(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise EvaluationContractError(f"{label} image cannot be decoded: {image_path}") from exc
        if actual_image_digest != expected_image_digest:
            raise EvaluationContractError(
                f"{label} decoded RGB SHA-256 mismatch: expected {expected_image_digest}, "
                f"got {actual_image_digest}."
            )
        record["image_sha256"] = expected_image_digest
        verified_records.append(record)

    calibration_count = _plain_int(payload["calibration_count"], "calibration_count")
    heldout_count = _plain_int(payload["heldout_count"], "heldout_count")
    if split_counts != {"calibration": calibration_count, "heldout": heldout_count}:
        raise EvaluationContractError("Manifest split counts do not match its records.")
    if calibration_count + heldout_count != total:
        raise EvaluationContractError("Manifest split counts do not sum to total.")
    return {
        "path": str(path),
        "sha256": actual_digest,
        "root": str(root),
        "payload": payload,
        "records": verified_records,
    }


def _config_at(config: Mapping[str, Any], path: Iterable[str]) -> Any:
    value: Any = config
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(value, Mapping) or key not in value:
            raise EvaluationContractError(f"Model config is missing {'.'.join(traversed)}.")
        value = value[key]
    return value


def _expect_config(config: Mapping[str, Any], path: tuple[str, ...], expected: Any) -> None:
    actual = _config_at(config, path)
    if actual != expected:
        raise EvaluationContractError(
            f"Model config {'.'.join(path)} mismatch: expected {expected!r}, got {actual!r}."
        )


def verify_model_snapshot(model_snapshot: str | Path) -> dict[str, Any]:
    """Fingerprint the complete local snapshot and pin NEO1.5 architecture invariants."""

    snapshot = Path(model_snapshot).resolve(strict=True)
    if not snapshot.is_dir():
        raise EvaluationContractError(f"Model snapshot is not a directory: {snapshot}")
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise EvaluationContractError("Local model snapshot has no config.json.")
    config = _read_json_object(config_path, "model config")
    expected_values = {
        ("model_type",): "neo_chat",
        ("architectures",): ["NEOChatModel"],
        ("auto_map", "AutoModel"): "modeling_neo_chat.NEOChatModel",
        ("template",): "neo1_0",
        ("patch_size",): PATCH_SIZE,
        ("downsample_ratio",): DOWNSAMPLE_RATIO,
        ("min_pixels",): MIN_PIXELS,
        ("eos_token_id",): 151645,
        ("pad_token_id",): 151643,
        ("llm_config", "model_type"): "qwen3",
        ("llm_config", "hidden_size"): 2048,
        ("llm_config", "intermediate_size"): 6144,
        ("llm_config", "num_hidden_layers"): 40,
        ("llm_config", "num_attention_heads"): 16,
        ("llm_config", "num_key_value_heads"): 8,
        ("llm_config", "vocab_size"): 151936,
        ("vision_config", "model_type"): "neo_vision",
        ("vision_config", "num_channels"): 3,
        ("vision_config", "patch_size"): PATCH_SIZE,
        ("vision_config", "downsample_ratio"): DOWNSAMPLE_RATIO,
        ("vision_config", "llm_hidden_size"): 2048,
    }
    for key_path, expected in expected_values.items():
        _expect_config(config, key_path, expected)

    files: list[dict[str, Any]] = []
    has_weights = False
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        digest = _file_sha256(path)
        size = path.stat().st_size
        files.append({"path": relative, "size": size, "sha256": digest})
        if relative.endswith((".safetensors", ".bin")):
            has_weights = True
    if not files or not has_weights:
        raise EvaluationContractError("Local model snapshot has no checkpoint weight file.")
    required_files = ("config.json", "modeling_neo_chat.py")
    relative_names = {item["path"] for item in files}
    missing = [name for name in required_files if name not in relative_names]
    if missing:
        raise EvaluationContractError(f"Local remote-code snapshot is incomplete; missing={missing}.")
    tree_sha256 = _canonical_sha256(files)
    revision = snapshot.name if re.fullmatch(r"[0-9a-f]{40,64}", snapshot.name) else None
    return {
        "path": str(snapshot),
        "model_id": MODEL_ID,
        "revision": revision,
        "tree_sha256": tree_sha256,
        "config_sha256": _file_sha256(config_path),
        "files": files,
        "config": config,
    }


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = RESIZE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Reproduce NEO's Qwen-derived native-resolution resize calculation."""

    if min(height, width, factor, min_pixels, max_pixels) <= 0 or min_pixels > max_pixels:
        raise EvaluationContractError("Invalid smart_resize dimensions or bounds.")
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise EvaluationContractError(
            f"Absolute image aspect ratio must be <= 200, got {aspect_ratio}."
        )
    resized_height = max(factor, round_by_factor(height, factor))
    resized_width = max(factor, round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, floor_by_factor(height / beta, factor))
        resized_width = max(factor, floor_by_factor(width / beta, factor))
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = ceil_by_factor(height * beta, factor)
        resized_width = ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def preprocess_pixel_values(
    pixel_values: Any,
    *,
    patch_size: int = PATCH_SIZE,
    torch_module: Any | None = None,
) -> tuple[Any, Any]:
    torch = torch_module or __import__("torch")
    if pixel_values.ndim != 3:
        raise EvaluationContractError("Normalized pixel_values must have CHW rank 3.")
    channels, height, width = (int(value) for value in pixel_values.shape)
    if channels != 3 or height % patch_size or width % patch_size:
        raise EvaluationContractError("Pixel tensor must be 3-channel and patch divisible.")
    grid_h = height // patch_size
    grid_w = width // patch_size
    flattened = (
        pixel_values.view(channels, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, channels * patch_size**2)
    )
    grid_hw = torch.tensor([[grid_h, grid_w]], dtype=torch.long)
    return flattened, grid_hw


def preprocess_neo_image(
    image_path: str | Path,
    *,
    torch_module: Any | None = None,
) -> tuple[Any, Any]:
    """Reproduce official NEO load_image_native without torchvision."""

    import numpy as np
    from PIL import Image

    torch = torch_module or __import__("torch")
    with Image.open(image_path) as source:
        source.load()
        image = source.convert("RGB")
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=RESIZE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    resized = image.resize((resized_width, resized_height))
    array = np.array(resized, dtype=np.uint8, copy=True)
    pixel_values = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    pixel_values.sub_(mean).div_(std)
    flattened, grid_hw = preprocess_pixel_values(
        pixel_values,
        patch_size=PATCH_SIZE,
        torch_module=torch,
    )
    if tuple(flattened.shape) != EXPECTED_PATCH_SHAPE:
        raise EvaluationContractError(
            f"NEO preprocessing shape mismatch: expected {EXPECTED_PATCH_SHAPE}, "
            f"got {tuple(flattened.shape)}."
        )
    if tuple(int(value) for value in grid_hw[0].tolist()) != EXPECTED_GRID:
        raise EvaluationContractError(
            f"NEO preprocessing grid mismatch: expected {EXPECTED_GRID}, got {grid_hw.tolist()}."
        )
    if flattened.dtype != torch.float32 or grid_hw.dtype != torch.long:
        raise EvaluationContractError("NEO preprocessing produced an unexpected tensor dtype.")
    return flattened, grid_hw


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}:{list(value.shape)}\n".encode("ascii")
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def normalize_direct_answer(category: str, raw_output: Any) -> str | None:
    """Normalize only a complete, direct response in the category's requested form."""

    if not isinstance(raw_output, str):
        return None
    if category in ("color", "spatial"):
        match = _DIRECT_WORD_RE.fullmatch(raw_output)
        if match is None:
            return None
        normalized = match.group(1).lower()
        allowed = COLOR_ANSWERS if category == "color" else SPATIAL_ANSWERS
        return normalized if normalized in allowed else None
    if category == "count":
        match = _DIRECT_DIGIT_RE.fullmatch(raw_output)
        if match is None or match.group(1) not in "12345":
            return None
        return match.group(1)
    if category == "ocr":
        match = _DIRECT_OCR_RE.fullmatch(raw_output)
        return None if match is None else match.group(1)
    raise EvaluationContractError(f"Unsupported answer category {category!r}.")


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_device(torch: Any, requested: str) -> tuple[Any, dict[str, Any]]:
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise EvaluationContractError(f"Invalid torch device {requested!r}.") from exc
    if device.type not in ("cpu", "cuda"):
        raise EvaluationContractError("NEO baseline supports explicit cpu or cuda devices only.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise EvaluationContractError("CUDA device requested but CUDA is unavailable.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise EvaluationContractError(f"CUDA device index {index} is unavailable.")
        device = torch.device(f"cuda:{index}")
        properties = torch.cuda.get_device_properties(index)
        identity = {
            "device": str(device),
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory": int(properties.total_memory),
        }
    else:
        identity = {
            "device": "cpu",
            "processor": platform.processor(),
            "machine": platform.machine(),
        }
    return device, identity


def _torch_dtype(torch: Any, name: str) -> Any:
    values = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    try:
        return values[name]
    except KeyError as exc:
        raise EvaluationContractError(f"Unsupported dtype {name!r}.") from exc


def load_neo_model(
    snapshot: Path,
    *,
    torch_module: Any,
    torch_dtype: Any,
    device: Any,
    auto_tokenizer_cls: Any | None = None,
    auto_model_cls: Any | None = None,
) -> tuple[Any, Any]:
    """Load only local remote-code/model artifacts, then place the full parent explicitly."""

    if auto_tokenizer_cls is None or auto_model_cls is None:
        from transformers import AutoModel, AutoTokenizer

        auto_tokenizer_cls = auto_tokenizer_cls or AutoTokenizer
        auto_model_cls = auto_model_cls or AutoModel
    local_path = str(snapshot.resolve(strict=True))
    tokenizer = auto_tokenizer_cls.from_pretrained(
        local_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    model = auto_model_cls.from_pretrained(
        local_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def _attribute_at(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for name in path:
        if isinstance(current, Mapping):
            if name not in current:
                raise EvaluationContractError(f"Loaded model is missing config {'.'.join(path)}.")
            current = current[name]
        elif hasattr(current, name):
            current = getattr(current, name)
        else:
            raise EvaluationContractError(f"Loaded model is missing config {'.'.join(path)}.")
    return current


def _validate_loaded_model(
    tokenizer: Any,
    model: Any,
    *,
    torch: Any,
    torch_dtype: Any,
    device: Any,
) -> dict[str, Any]:
    if not callable(getattr(model, "chat", None)):
        raise EvaluationContractError("Loaded parent does not expose NEO model.chat().")
    if not callable(getattr(tokenizer, "convert_tokens_to_ids", None)):
        raise EvaluationContractError("Loaded tokenizer cannot resolve NEO special tokens.")
    if getattr(model, "patch_size", None) != PATCH_SIZE:
        raise EvaluationContractError("Loaded parent patch_size does not match NEO preprocessing.")
    if getattr(model, "downsample_ratio", None) != DOWNSAMPLE_RATIO:
        raise EvaluationContractError("Loaded parent downsample_ratio does not match NEO preprocessing.")
    for path, expected in (
        (("config", "patch_size"), PATCH_SIZE),
        (("config", "downsample_ratio"), DOWNSAMPLE_RATIO),
        (("config", "vision_config", "patch_size"), PATCH_SIZE),
        (("config", "vision_config", "downsample_ratio"), DOWNSAMPLE_RATIO),
    ):
        actual = _attribute_at(model, path)
        if actual != expected:
            raise EvaluationContractError(
                f"Loaded model {'.'.join(path)} mismatch: expected {expected!r}, got {actual!r}."
            )
    token_ids: dict[str, int] = {}
    for token, expected in SPECIAL_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise EvaluationContractError(
                f"Tokenizer special token {token!r} mismatch: expected {expected}, got {actual!r}."
            )
        token_ids[token] = actual

    parameter_count = 0
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    saw_parameter = False
    for parameter in model.parameters():
        saw_parameter = True
        count = int(parameter.numel())
        parameter_count += count
        dtype_name = str(parameter.dtype)
        device_name = str(parameter.device)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + count
        device_counts[device_name] = device_counts.get(device_name, 0) + count
        if parameter.device.type == "meta":
            raise EvaluationContractError("Loaded parent still contains meta-device parameters.")
        if parameter.device != device:
            raise EvaluationContractError(
                f"Loaded parent parameter remains on {parameter.device}, expected {device}."
            )
        if parameter.is_floating_point() and parameter.dtype != torch_dtype:
            raise EvaluationContractError(
                f"Loaded floating parameter dtype {parameter.dtype} does not match {torch_dtype}."
            )
    if not saw_parameter or parameter_count <= 0:
        raise EvaluationContractError("Loaded parent exposes no parameters.")
    return {
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "parameter_count": parameter_count,
        "parameter_dtypes": dtype_counts,
        "parameter_devices": device_counts,
        "special_token_ids": token_ids,
    }


class _RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> "_RunLock":
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise EvaluationContractError(f"Another evaluator owns run lock {self.path}.") from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def _append_cache_record(cache_path: Path, record: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(record) + b"\n"
    with cache_path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_cache_record(
    cached: Any,
    *,
    index: int,
    source_record: Mapping[str, Any],
    inference_fingerprint: str,
) -> dict[str, Any]:
    if not isinstance(cached, dict) or frozenset(cached) != _CACHE_KEYS:
        raise EvaluationContractError(f"Cache line {index + 1} does not match the record schema.")
    if cached["schema"] != CACHE_SCHEMA or cached["inference_fingerprint"] != inference_fingerprint:
        raise EvaluationContractError(f"Cache line {index + 1} belongs to a different inference run.")
    if cached["index"] != index or cached["probe_id"] != source_record["probe_id"]:
        raise EvaluationContractError("Cache must be an exact prefix of manifest record order.")
    expected_record_sha = _canonical_sha256(source_record)
    if cached["record_sha256"] != expected_record_sha:
        raise EvaluationContractError(f"Cache record fingerprint mismatch for {source_record['probe_id']}.")
    for field in ("category", "split"):
        if cached[field] != source_record[field]:
            raise EvaluationContractError(f"Cached {field} mismatch for {source_record['probe_id']}.")
    if cached["expected"] != source_record["answer"]:
        raise EvaluationContractError(f"Cached expected answer mismatch for {source_record['probe_id']}.")
    raw_output = cached["raw_output"]
    if not isinstance(raw_output, str):
        raise EvaluationContractError(f"Cached output for {source_record['probe_id']} is not text.")
    if cached["raw_output_sha256"] != hashlib.sha256(raw_output.encode("utf-8")).hexdigest():
        raise EvaluationContractError(f"Cached raw output hash mismatch for {source_record['probe_id']}.")
    normalized = normalize_direct_answer(source_record["category"], raw_output)
    if cached["normalized_prediction"] != normalized:
        raise EvaluationContractError(f"Cached normalization mismatch for {source_record['probe_id']}.")
    direct = normalized is not None
    correct = normalized == source_record["answer"]
    if cached["direct_answer"] is not direct or cached["correct"] is not correct:
        raise EvaluationContractError(f"Cached score mismatch for {source_record['probe_id']}.")
    _normalize_sha256(cached["pixel_values_sha256"], "cached pixel tensor SHA-256")
    if cached["grid_hw"] != [list(EXPECTED_GRID)]:
        raise EvaluationContractError(f"Cached grid mismatch for {source_record['probe_id']}.")
    if not isinstance(cached["elapsed_seconds"], (int, float)) or cached["elapsed_seconds"] < 0:
        raise EvaluationContractError(f"Cached timing is invalid for {source_record['probe_id']}.")
    if not isinstance(cached["completed_utc"], str) or not cached["completed_utc"]:
        raise EvaluationContractError(f"Cached completion time is invalid for {source_record['probe_id']}.")
    return dict(cached)


def _load_cache(
    cache_path: Path,
    records: list[dict[str, Any]],
    inference_fingerprint: str,
) -> list[dict[str, Any]]:
    if not cache_path.exists():
        return []
    payload = cache_path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise EvaluationContractError("Cache JSONL has a truncated final line; refusing recomputation.")
    lines = payload.splitlines()
    if len(lines) > len(records):
        raise EvaluationContractError("Cache has more outputs than the manifest.")
    results: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            cached = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationContractError(f"Cache line {index + 1} is invalid JSON.") from exc
        results.append(
            _validate_cache_record(
                cached,
                index=index,
                source_record=records[index],
                inference_fingerprint=inference_fingerprint,
            )
        )
    return results


def _metric(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(items)
    total = len(values)
    correct = sum(item["correct"] is True for item in values)
    direct = sum(item["direct_answer"] is True for item in values)
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else None,
        "direct_answers": direct,
        "direct_answer_rate": (direct / total) if total else None,
    }


def _build_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _metric(entries),
        "by_split": {
            split: _metric(item for item in entries if item["split"] == split)
            for split in SPLITS
        },
        "by_category": {
            category: _metric(item for item in entries if item["category"] == category)
            for category in CATEGORIES
        },
        "by_split_category": {
            split: {
                category: _metric(
                    item
                    for item in entries
                    if item["split"] == split and item["category"] == category
                )
                for category in CATEGORIES
            }
            for split in SPLITS
        },
    }


def _manual_samples(entries: list[dict[str, Any]], per_group: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in SPLITS:
        for category in CATEGORIES:
            group = [
                item
                for item in entries
                if item["split"] == split and item["category"] == category
            ]
            group.sort(key=lambda item: (item["correct"] is True, item["index"]))
            for item in group[:per_group]:
                samples.append(
                    {
                        "probe_id": item["probe_id"],
                        "split": split,
                        "category": category,
                        "expected": item["expected"],
                        "raw_output": item["raw_output"],
                        "normalized_prediction": item["normalized_prediction"],
                        "direct_answer": item["direct_answer"],
                        "correct": item["correct"],
                    }
                )
    return samples


def _stable_runtime_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "model_class",
            "tokenizer_class",
            "parameter_count",
            "parameter_dtypes",
            "parameter_devices",
            "special_token_ids",
        )
    }


def evaluate_parent_baseline(
    *,
    manifest: str | Path,
    manifest_sha256: str,
    model_snapshot: str | Path,
    output_dir: str | Path,
    dtype: str = "bf16",
    device: str = "cuda:0",
    max_new_tokens: int = 16,
    seed: int = 0,
    manual_samples_per_group: int = 4,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Evaluate missing records only; existing valid parent outputs are immutable cache hits."""

    max_new_tokens = _plain_int(max_new_tokens, "max_new_tokens", minimum=1)
    seed = _plain_int(seed, "seed")
    manual_samples_per_group = _plain_int(
        manual_samples_per_group, "manual_samples_per_group", minimum=1
    )
    torch = torch_module or __import__("torch")
    torch_dtype = _torch_dtype(torch, dtype)
    resolved_device, device_identity = _resolve_device(torch, device)
    verified_manifest = verify_manifest(manifest, manifest_sha256)
    verified_snapshot = verify_model_snapshot(model_snapshot)
    script_path = Path(__file__).resolve(strict=True)
    evaluator_sha256 = _file_sha256(script_path)
    generation_config = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
    }
    inference_spec = {
        "protocol": CACHE_SCHEMA,
        "model_id": MODEL_ID,
        "manifest_sha256": verified_manifest["sha256"],
        "snapshot_tree_sha256": verified_snapshot["tree_sha256"],
        "evaluator_sha256": evaluator_sha256,
        "preprocess": PREPROCESS_SPEC,
        "generation_config": generation_config,
        "dtype": dtype,
        "device_identity": device_identity,
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "transformers": _package_version("transformers"),
            "pillow": _package_version("Pillow"),
            "numpy": _package_version("numpy"),
        },
    }
    inference_fingerprint = _canonical_sha256(inference_spec)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run_manifest.json"
    cache_path = output / "parent_outputs.jsonl"
    runtime_path = output / "model_runtime.json"
    progress_path = output / "progress.json"
    report_path = output / "report.json"
    failure_path = output / "failure.json"

    with _RunLock(output / ".eval.lock"):
        run_manifest = {
            "schema": RUN_SCHEMA,
            "created_utc": _utc_now(),
            "inference_fingerprint": inference_fingerprint,
            "inference_spec": inference_spec,
            "manifest": {
                "path": verified_manifest["path"],
                "sha256": verified_manifest["sha256"],
                "schema": PROBE_SCHEMA,
                "image_hash_basis": IMAGE_HASH_BASIS,
                "total": len(verified_manifest["records"]),
            },
            "snapshot": {
                key: verified_snapshot[key]
                for key in (
                    "path",
                    "model_id",
                    "revision",
                    "tree_sha256",
                    "config_sha256",
                    "files",
                )
            },
            "evaluator": {"path": str(script_path), "sha256": evaluator_sha256},
            "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        }
        if run_path.exists():
            existing_run = _read_json_object(run_path, "run manifest")
            if (
                existing_run.get("schema") != RUN_SCHEMA
                or existing_run.get("inference_fingerprint") != inference_fingerprint
                or existing_run.get("inference_spec") != inference_spec
            ):
                raise EvaluationContractError(
                    "Output directory belongs to a different inference contract; use a new directory."
                )
            run_manifest = existing_run
        else:
            if cache_path.exists() or runtime_path.exists() or report_path.exists():
                raise EvaluationContractError(
                    "Output directory contains orphaned evaluator artifacts without run_manifest.json."
                )
            _atomic_write_json(run_path, run_manifest)

        records = verified_manifest["records"]
        entries = _load_cache(cache_path, records, inference_fingerprint)
        cache_hits = len(entries)
        new_records = 0
        runtime_metadata: dict[str, Any] | None = None
        try:
            if len(entries) < len(records):
                loader = model_loader or load_neo_model
                tokenizer, model = loader(
                    Path(verified_snapshot["path"]),
                    torch_module=torch,
                    torch_dtype=torch_dtype,
                    device=resolved_device,
                )
                runtime_metadata = _validate_loaded_model(
                    tokenizer,
                    model,
                    torch=torch,
                    torch_dtype=torch_dtype,
                    device=resolved_device,
                )
                runtime_document = {
                    "schema": RUN_SCHEMA,
                    "inference_fingerprint": inference_fingerprint,
                    **runtime_metadata,
                }
                if runtime_path.exists():
                    existing_runtime = _read_json_object(runtime_path, "model runtime metadata")
                    if (
                        existing_runtime.get("schema") != RUN_SCHEMA
                        or existing_runtime.get("inference_fingerprint") != inference_fingerprint
                        or _stable_runtime_metadata(existing_runtime)
                        != _stable_runtime_metadata(runtime_document)
                    ):
                        raise EvaluationContractError(
                            "Reloaded model runtime does not match the partial cache's parent."
                        )
                    runtime_document = existing_runtime
                else:
                    _atomic_write_json(runtime_path, runtime_document)
                runtime_metadata = runtime_document

                torch.manual_seed(seed)
                if resolved_device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    if hasattr(torch.backends, "cudnn"):
                        torch.backends.cudnn.benchmark = False
                        torch.backends.cudnn.deterministic = True
                inference_context = (
                    torch.inference_mode() if callable(getattr(torch, "inference_mode", None)) else nullcontext()
                )
                for index in range(len(entries), len(records)):
                    source_record = records[index]
                    image_path = Path(verified_manifest["root"]) / source_record["image"]
                    pixel_values, grid_hw = preprocess_neo_image(
                        image_path,
                        torch_module=torch,
                    )
                    pixel_values_sha256 = _tensor_sha256(pixel_values)
                    record_seed = seed + index
                    torch.manual_seed(record_seed)
                    if resolved_device.type == "cuda":
                        torch.cuda.manual_seed_all(record_seed)
                    device_pixels = pixel_values.to(device=resolved_device, dtype=torch_dtype)
                    device_grid = grid_hw.to(device=resolved_device)
                    call_config = dict(generation_config)
                    started = time.perf_counter()
                    with inference_context:
                        raw_output = model.chat(
                            tokenizer,
                            pixel_values=device_pixels,
                            grid_hw=device_grid,
                            question=source_record["prompt"],
                            generation_config=call_config,
                            verbose=False,
                        )
                    if resolved_device.type == "cuda":
                        torch.cuda.synchronize(resolved_device)
                    elapsed = time.perf_counter() - started
                    if not isinstance(raw_output, str):
                        raise EvaluationContractError(
                            f"model.chat returned {type(raw_output).__name__} for {source_record['probe_id']}."
                        )
                    normalized = normalize_direct_answer(source_record["category"], raw_output)
                    cache_record = {
                        "schema": CACHE_SCHEMA,
                        "inference_fingerprint": inference_fingerprint,
                        "index": index,
                        "probe_id": source_record["probe_id"],
                        "record_sha256": _canonical_sha256(source_record),
                        "category": source_record["category"],
                        "split": source_record["split"],
                        "expected": source_record["answer"],
                        "raw_output": raw_output,
                        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
                        "normalized_prediction": normalized,
                        "direct_answer": normalized is not None,
                        "correct": normalized == source_record["answer"],
                        "pixel_values_sha256": pixel_values_sha256,
                        "grid_hw": grid_hw.tolist(),
                        "elapsed_seconds": elapsed,
                        "completed_utc": _utc_now(),
                    }
                    _append_cache_record(cache_path, cache_record)
                    entries.append(cache_record)
                    new_records += 1
                    _atomic_write_json(
                        progress_path,
                        {
                            "schema": REPORT_SCHEMA,
                            "inference_fingerprint": inference_fingerprint,
                            "status": "running",
                            "completed": len(entries),
                            "total": len(records),
                            "last_probe_id": source_record["probe_id"],
                            "updated_utc": _utc_now(),
                        },
                    )
                    if len(entries) == len(records) or len(entries) % 10 == 0:
                        print(
                            json.dumps(
                                {
                                    "completed": len(entries),
                                    "total": len(records),
                                    "last_probe_id": source_record["probe_id"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            else:
                if not runtime_path.exists():
                    raise EvaluationContractError(
                        "Complete output cache is missing model_runtime.json provenance."
                    )
                runtime_metadata = _read_json_object(runtime_path, "model runtime metadata")
                if (
                    runtime_metadata.get("schema") != RUN_SCHEMA
                    or runtime_metadata.get("inference_fingerprint") != inference_fingerprint
                ):
                    raise EvaluationContractError("Cached model runtime provenance is incompatible.")
                _stable_runtime_metadata(runtime_metadata)

            if len(entries) != len(records):
                raise EvaluationContractError("Evaluation ended without a complete manifest prefix.")
            cache_sha256 = _file_sha256(cache_path)
            metrics = _build_metrics(entries)
            report = {
                "schema": REPORT_SCHEMA,
                "status": "complete",
                "completed_utc": _utc_now(),
                "model_id": MODEL_ID,
                "inference_fingerprint": inference_fingerprint,
                "manifest": run_manifest["manifest"],
                "snapshot": {
                    key: run_manifest["snapshot"][key]
                    for key in (
                        "path",
                        "model_id",
                        "revision",
                        "tree_sha256",
                        "config_sha256",
                    )
                },
                "runtime": runtime_metadata,
                "preprocess": PREPROCESS_SPEC,
                "generation_config": generation_config,
                "cache": {
                    "path": str(cache_path),
                    "sha256": cache_sha256,
                    "records": len(entries),
                    "cache_hits_this_invocation": cache_hits,
                    "new_records_this_invocation": new_records,
                },
                "metrics": metrics,
                "manual_samples": _manual_samples(entries, manual_samples_per_group),
            }
            _atomic_write_json(report_path, report)
            report_sha256 = _file_sha256(report_path)
            _atomic_write_bytes(
                output / "report.sha256",
                f"{report_sha256}  report.json\n".encode("ascii"),
            )
            _atomic_write_json(
                progress_path,
                {
                    "schema": REPORT_SCHEMA,
                    "inference_fingerprint": inference_fingerprint,
                    "status": "complete",
                    "completed": len(entries),
                    "total": len(records),
                    "cache_sha256": cache_sha256,
                    "report_sha256": report_sha256,
                    "updated_utc": _utc_now(),
                },
            )
            if failure_path.exists():
                failure_path.unlink()
            return report
        except Exception as exc:
            _atomic_write_json(
                failure_path,
                {
                    "schema": REPORT_SCHEMA,
                    "inference_fingerprint": inference_fingerprint,
                    "status": "failed",
                    "completed": len(entries),
                    "total": len(verified_manifest["records"]),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_utc": _utc_now(),
                },
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--manifest-sha256",
        required=True,
        help="Expected SHA-256 printed when the probe manifest was built.",
    )
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manual-samples-per-group", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_parent_baseline(
            manifest=args.manifest,
            manifest_sha256=args.manifest_sha256,
            model_snapshot=args.model_snapshot,
            output_dir=args.output_dir,
            dtype=args.dtype,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            manual_samples_per_group=args.manual_samples_per_group,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_id": report["model_id"],
                "inference_fingerprint": report["inference_fingerprint"],
                "cache": report["cache"],
                "metrics": report["metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
