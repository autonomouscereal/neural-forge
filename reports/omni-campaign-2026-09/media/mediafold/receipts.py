"""Artifact receipts: hash what was written, never trust an exit code.

Every writer in this package routes through :func:`write_tensors`, which returns a
per-tensor sha256 table, and every script ends with :func:`write_receipt`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "chimera_omni.media_fold.receipt.v1"


def hash_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tensor(tensor: Any) -> str:
    """sha256 over (dtype, shape, raw bytes) of a torch tensor."""

    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def hash_state(state: dict[str, Any]) -> str:
    """One sha256 over a whole named tensor mapping, name order fixed."""

    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(hash_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def parameter_count(state: dict[str, Any]) -> int:
    """Storage counted in PARAMETERS. Tensors sharing storage count once."""

    seen: set[int] = set()
    total = 0
    for name in sorted(state):
        tensor = state[name]
        pointer = int(tensor.detach().data_ptr())
        if pointer in seen:
            continue
        seen.add(pointer)
        total += int(tensor.numel())
    return total


def write_tensors(state: dict[str, Any], path: str | os.PathLike[str]) -> dict[str, Any]:
    """Save a tensor mapping to safetensors and return its receipt block."""

    from safetensors.torch import save_file

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: value.detach().cpu().contiguous() for name, value in state.items()}
    save_file(payload, str(target))
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "file_sha256": hash_file(target),
        "state_sha256": hash_state(payload),
        "parameters": parameter_count(payload),
        "tensors": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hash_tensor(value),
            }
            for name, value in sorted(payload.items())
        },
    }


def read_tensors(path: str | os.PathLike[str], device: str = "cpu") -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path), device=device)


def runtime_block(device: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        block["torch"] = torch.__version__
        block["torch_cuda"] = torch.version.cuda
        block["cuda_available"] = bool(torch.cuda.is_available())
        if device and device.startswith("cuda") and torch.cuda.is_available():
            index = 0 if device == "cuda" else int(device.split(":")[1])
            properties = torch.cuda.get_device_properties(index)
            block["device_name"] = properties.name
            block["device_total_memory_bytes"] = int(properties.total_memory)
            block["device_capability"] = f"{properties.major}.{properties.minor}"
    except Exception as error:  # torch is optional for pure-numpy callers
        block["torch"] = f"unavailable: {type(error).__name__}"
    try:
        block["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used", "--format=csv,noheader"],
            text=True,
            timeout=30,
        ).strip().splitlines()
    except Exception:
        block["nvidia_smi"] = None
    return block


def write_receipt(path: str | os.PathLike[str], body: dict[str, Any]) -> str:
    """Write a receipt JSON and return its own sha256."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **body,
    }
    target.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return hash_file(target)


def source_sha256(module_paths: list[str]) -> dict[str, str]:
    """Hash the source files a run depended on so a result is reproducible."""

    table: dict[str, str] = {}
    for entry in module_paths:
        candidate = Path(entry)
        if candidate.is_file():
            table[str(candidate)] = hash_file(candidate)
    return table
