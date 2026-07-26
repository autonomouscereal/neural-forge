"""Immutable request capsules for the normalized-input decoder compiler.

The capsule is a diagnostic boundary, not a second compiler.  It stores the
exact tensor request immediately before
``compile_lfm2_normalized_input_decoder`` and can be authenticated without a
parent checkpoint, trace cache, target model, accelerator, or pickle loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    _atomic_rename_noreplace,
    _fsync_directory,
    _is_link_or_junction,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedNormalizedInputMlpFoldCase,
    lfm2_authenticated_tensor_sha256,
    lfm2_normalized_input_mlp_fit_evidence_sha256,
)


LFM2_NORMALIZED_INPUT_REPLAY_CAPSULE_SCHEMA = (
    "neural-forge-lfm2-normalized-input-replay-capsule-v1"
)
MANIFEST_FILENAME = "request.json"
TENSOR_FILENAME = "request.safetensors"

_CASE_TENSOR_FIELDS = (
    "target_input",
    "parent_normalized_input",
    "parent_gate",
    "parent_up",
    "parent_post_swiglu",
    "parent_post_mlp_residual_target_space",
    "target_pre_mlp_residual",
)
_INPUT_TENSOR_NAMES = (
    "residual_chart",
    "parent_ffn_norm_weight",
    "target_ffn_norm_weight",
    "raw_structural_decoder",
    "case_sample_weights",
)


class Lfm2NormalizedInputReplayCapsuleError(RuntimeError):
    """A normalized-input request capsule is unsafe or unauthenticated."""


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: Any, name: str) -> str:
    if not _valid_sha256(value):
        raise Lfm2NormalizedInputReplayCapsuleError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _owned_json_value(
    value: Any,
    *,
    name: str,
    active_containers: set[int],
) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Lfm2NormalizedInputReplayCapsuleError(
                f"{name} contains a non-finite float."
            )
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise Lfm2NormalizedInputReplayCapsuleError(
                f"{name} contains a recursive mapping."
            )
        active_containers.add(identity)
        try:
            result = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise Lfm2NormalizedInputReplayCapsuleError(
                        f"{name} contains a non-string mapping key."
                    )
                result[key] = _owned_json_value(
                    item,
                    name=f"{name}.{key}",
                    active_containers=active_containers,
                )
            return result
        finally:
            active_containers.remove(identity)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        identity = id(value)
        if identity in active_containers:
            raise Lfm2NormalizedInputReplayCapsuleError(
                f"{name} contains a recursive sequence."
            )
        active_containers.add(identity)
        try:
            return [
                _owned_json_value(
                    item,
                    name=f"{name}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    raise Lfm2NormalizedInputReplayCapsuleError(
        f"{name} contains unsupported value type {type(value).__name__}."
    )


def _json_owned(value: Any, name: str) -> Any:
    owned = _owned_json_value(
        value,
        name=name,
        active_containers=set(),
    )
    try:
        return json.loads(canonical_json_bytes(owned))
    except (TypeError, ValueError) as exc:
        raise Lfm2NormalizedInputReplayCapsuleError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _tensor_record(value: Any) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(size) for size in value.shape],
        "tensor_sha256": lfm2_authenticated_tensor_sha256(value),
    }


def _captured_tensor(value: Any, name: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise Lfm2NormalizedInputReplayCapsuleError(
            "PyTorch is required to capture normalized-input requests."
        ) from exc
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim == 0
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            f"{name} must be one non-empty materialized real floating tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2NormalizedInputReplayCapsuleError(f"{name} contains NaN or infinity.")
    return result


def _completion_coordinate_map_sha256(value: Any) -> str:
    """Match the dense content digest required by completion receipts."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise Lfm2NormalizedInputReplayCapsuleError(
            "PyTorch is required to authenticate normalized-input requests."
        ) from exc
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _authenticated_completion_receipt(
    value: Mapping[str, Any],
    *,
    claimed_sha256: str,
    raw_structural_decoder: Any,
) -> dict[str, Any]:
    receipt = _json_owned(value, "coordinate completion receipt")
    if not isinstance(receipt, dict):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "coordinate completion receipt must be a JSON mapping."
        )
    seal_fields = tuple(
        name for name in ("result_sha256", "receipt_sha256") if name in receipt
    )
    if len(seal_fields) != 1:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "coordinate completion receipt must contain exactly one stable seal."
        )
    seal_name = seal_fields[0]
    seal = receipt.pop(seal_name)
    expected = _digest(claimed_sha256, "coordinate completion receipt SHA-256")
    if seal != expected or canonical_sha256(receipt) != expected:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "coordinate completion receipt authentication failed."
        )
    if receipt.get("coordinate_map_shape") != [
        int(raw_structural_decoder.shape[0]),
        int(raw_structural_decoder.shape[1]),
    ] or receipt.get("coordinate_map_sha256") != (
        _completion_coordinate_map_sha256(raw_structural_decoder)
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "coordinate completion receipt does not bind the captured raw D0."
        )
    return {**receipt, seal_name: seal}


def _safe_root(value: str | os.PathLike[str], *, create: bool) -> Path:
    raw = os.fspath(value)
    if not raw or "://" in raw:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule paths must be non-empty local filesystem paths."
        )
    root = Path(raw).expanduser().absolute()
    if _is_link_or_junction(root):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule paths cannot be symlinks or junctions."
        )
    if create and not root.exists():
        parent = root.parent
        if (
            not parent.is_dir()
            or _is_link_or_junction(parent)
            or parent.resolve(strict=True) != parent
        ):
            raise Lfm2NormalizedInputReplayCapsuleError(
                "Replay capsule root parent is missing or unsafe."
            )
        try:
            root.mkdir()
        except FileExistsError:
            pass
    if not root.is_dir() or _is_link_or_junction(root):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule root is missing or unsafe."
        )
    resolved = root.resolve(strict=True)
    if resolved != root:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule path cannot traverse a link."
        )
    return resolved


def _safe_capsule_directory(value: str | os.PathLike[str]) -> Path:
    root = _safe_root(value, create=False)
    names = {entry.name for entry in root.iterdir()}
    if names != {MANIFEST_FILENAME, TENSOR_FILENAME}:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule must contain exactly request.json and request.safetensors."
        )
    return root


def prepare_lfm2_normalized_input_replay_capture_root(
    capture_root: str | os.PathLike[str],
) -> Path:
    """Create or authenticate one empty root before expensive body work."""

    root = _safe_root(capture_root, create=True)
    if any(root.iterdir()):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capture root must be empty at exact-fold entry."
        )
    return root


def _read_pinned_regular_file(path: Path, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise Lfm2NormalizedInputReplayCapsuleError(f"{name} is missing or unsafe.")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise Lfm2NormalizedInputReplayCapsuleError(f"{name} is not a regular file.")
    try:
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            payload = handle.read()
            opened_after = os.fstat(handle.fileno())
    except OSError as exc:
        raise Lfm2NormalizedInputReplayCapsuleError(f"{name} is unreadable.") from exc
    after = path.stat(follow_symlinks=False)
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_size,
            opened_before.st_mtime_ns,
        ),
        (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
        ),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if len(set(identities)) != 1 or len(payload) != before.st_size:
        raise Lfm2NormalizedInputReplayCapsuleError(
            f"{name} changed while it was read."
        )
    return payload


def _positive_weights(value: Any, *, case_count: int) -> tuple[float, ...]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise Lfm2NormalizedInputReplayCapsuleError(
            "PyTorch is required to authenticate normalized-input requests."
        ) from exc
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.device.type != "cpu"
        or tuple(value.shape) != (case_count,)
        or not bool(torch.isfinite(value).all().item())
        or bool((value <= 0.0).any().item())
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Captured case weights must be one positive finite CPU-FP64 vector."
        )
    return tuple(float(item) for item in value.tolist())


@dataclass(frozen=True, slots=True)
class Lfm2NormalizedInputReplayCapsule:
    """Authenticated compiler kwargs reconstructed from one safe capsule."""

    root: Path
    layer_index: int
    residual_chart: Any = field(repr=False, compare=False)
    residual_chart_sha256: str
    parent_ffn_norm_weight: Any = field(repr=False, compare=False)
    parent_ffn_norm_weight_sha256: str
    target_ffn_norm_weight: Any = field(repr=False, compare=False)
    target_ffn_norm_weight_sha256: str
    cases: tuple[Lfm2CachedNormalizedInputMlpFoldCase, ...] = field(
        repr=False,
        compare=False,
    )
    case_sample_weights: tuple[float, ...]
    normalized_decoder_ridge: float
    normalized_input_correction_rho: float
    raw_structural_decoder: Any = field(repr=False, compare=False)
    raw_structural_decoder_sha256: str
    coordinate_completion_receipt: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )
    coordinate_completion_receipt_sha256: str
    capsule_sha256: str
    manifest: Mapping[str, Any] = field(repr=False, compare=False)

    def compiler_kwargs(self) -> dict[str, Any]:
        return {
            "residual_chart": self.residual_chart,
            "residual_chart_sha256": self.residual_chart_sha256,
            "parent_ffn_norm_weight": self.parent_ffn_norm_weight,
            "parent_ffn_norm_weight_sha256": self.parent_ffn_norm_weight_sha256,
            "target_ffn_norm_weight": self.target_ffn_norm_weight,
            "target_ffn_norm_weight_sha256": self.target_ffn_norm_weight_sha256,
            "cases": self.cases,
            "case_sample_weights": self.case_sample_weights,
            "normalized_decoder_ridge": self.normalized_decoder_ridge,
            "normalized_input_correction_rho": (self.normalized_input_correction_rho),
            "raw_structural_decoder": self.raw_structural_decoder,
            "raw_structural_decoder_sha256": self.raw_structural_decoder_sha256,
            "coordinate_completion_receipt": (self.coordinate_completion_receipt),
            "coordinate_completion_receipt_sha256": (
                self.coordinate_completion_receipt_sha256
            ),
        }


def write_lfm2_normalized_input_replay_capsule(
    capture_root: str | os.PathLike[str],
    *,
    layer_index: int,
    residual_chart: Any,
    residual_chart_sha256: str,
    parent_ffn_norm_weight: Any,
    parent_ffn_norm_weight_sha256: str,
    target_ffn_norm_weight: Any,
    target_ffn_norm_weight_sha256: str,
    cases: Sequence[Lfm2CachedNormalizedInputMlpFoldCase],
    case_sample_weights: Sequence[float],
    normalized_decoder_ridge: float,
    normalized_input_correction_rho: float,
    raw_structural_decoder: Any,
    raw_structural_decoder_sha256: str,
    coordinate_completion_receipt: Mapping[str, Any],
    coordinate_completion_receipt_sha256: str,
) -> Lfm2NormalizedInputReplayCapsule:
    """Publish one exact layer request and refuse every overwrite."""

    try:
        import torch
        from safetensors.torch import save
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise Lfm2NormalizedInputReplayCapsuleError(
            "PyTorch and safetensors are required for replay capsules."
        ) from exc
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "layer_index must be a non-negative integer."
        )
    ordered_cases = tuple(cases)
    if not ordered_cases or not all(
        isinstance(item, Lfm2CachedNormalizedInputMlpFoldCase) for item in ordered_cases
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "cases must contain authenticated normalized-input case objects."
        )
    if len(case_sample_weights) != len(ordered_cases):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "case_sample_weights must contain one value per ordered case."
        )
    parsed_weights = []
    for value in case_sample_weights:
        if isinstance(value, bool):
            raise Lfm2NormalizedInputReplayCapsuleError(
                "case_sample_weights must be positive finite values."
            )
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0.0:
            raise Lfm2NormalizedInputReplayCapsuleError(
                "case_sample_weights must be positive finite values."
            )
        parsed_weights.append(weight)
    for value, name in (
        (normalized_decoder_ridge, "normalized_decoder_ridge"),
        (normalized_input_correction_rho, "normalized_input_correction_rho"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise Lfm2NormalizedInputReplayCapsuleError(
                f"{name} must be finite and non-negative."
            )

    tensors = {
        "residual_chart": _captured_tensor(residual_chart, "residual_chart"),
        "parent_ffn_norm_weight": _captured_tensor(
            parent_ffn_norm_weight,
            "parent_ffn_norm_weight",
        ),
        "target_ffn_norm_weight": _captured_tensor(
            target_ffn_norm_weight,
            "target_ffn_norm_weight",
        ),
        "raw_structural_decoder": _captured_tensor(
            raw_structural_decoder,
            "raw_structural_decoder",
        ),
        "case_sample_weights": torch.tensor(parsed_weights, dtype=torch.float64),
    }
    supplied_hashes = {
        "residual_chart": _digest(
            residual_chart_sha256,
            "residual_chart_sha256",
        ),
        "parent_ffn_norm_weight": _digest(
            parent_ffn_norm_weight_sha256,
            "parent_ffn_norm_weight_sha256",
        ),
        "target_ffn_norm_weight": _digest(
            target_ffn_norm_weight_sha256,
            "target_ffn_norm_weight_sha256",
        ),
        "raw_structural_decoder": _digest(
            raw_structural_decoder_sha256,
            "raw_structural_decoder_sha256",
        ),
    }
    for name, expected in supplied_hashes.items():
        if lfm2_authenticated_tensor_sha256(tensors[name]) != expected:
            raise Lfm2NormalizedInputReplayCapsuleError(
                f"{name} does not match its supplied authentication digest."
            )
    completion = _authenticated_completion_receipt(
        coordinate_completion_receipt,
        claimed_sha256=coordinate_completion_receipt_sha256,
        raw_structural_decoder=tensors["raw_structural_decoder"],
    )

    case_order = []
    captured_cases = []
    for case_index, case in enumerate(ordered_cases):
        tensor_names: dict[str, str | None] = {}
        captured_fields: dict[str, Any] = {}
        for field_name in _CASE_TENSOR_FIELDS:
            tensor_name = f"case.{case_index:04d}.{field_name}"
            captured = _captured_tensor(
                getattr(case, field_name),
                tensor_name,
            )
            tensors[tensor_name] = captured
            tensor_names[field_name] = tensor_name
            captured_fields[field_name] = captured
        contribution_name = None
        parent_contribution = None
        if case.parent_contribution is not None:
            contribution_name = f"case.{case_index:04d}.parent_contribution"
            parent_contribution = _captured_tensor(
                case.parent_contribution,
                contribution_name,
            )
            tensors[contribution_name] = parent_contribution
        tensor_names["parent_contribution"] = contribution_name
        captured_case = Lfm2CachedNormalizedInputMlpFoldCase(
            **captured_fields,
            parent_contribution=parent_contribution,
        )
        captured_cases.append(captured_case)
        case_order.append(
            {
                "case_index": case_index,
                "tensor_names": tensor_names,
            }
        )

    fit_evidence_sha256 = lfm2_normalized_input_mlp_fit_evidence_sha256(
        tuple(captured_cases),
        tuple(parsed_weights),
    )
    tensor_payload = save(dict(sorted(tensors.items())))
    tensor_records = {
        name: _tensor_record(value) for name, value in sorted(tensors.items())
    }
    body = {
        "schema": LFM2_NORMALIZED_INPUT_REPLAY_CAPSULE_SCHEMA,
        "status": "exact-production-compiler-request-captured",
        "layer_index": layer_index,
        "tensor_filename": TENSOR_FILENAME,
        "tensor_file_size": len(tensor_payload),
        "tensor_file_sha256": _file_sha256(tensor_payload),
        "tensor_keys": sorted(tensors),
        "tensor_records": tensor_records,
        "tensor_inventory_sha256": canonical_sha256(tensor_records),
        "input_tensor_names": {name: name for name in _INPUT_TENSOR_NAMES},
        "input_tensor_sha256": supplied_hashes,
        "case_order": case_order,
        "case_count": len(captured_cases),
        "case_sample_weights_sha256": lfm2_authenticated_tensor_sha256(
            tensors["case_sample_weights"]
        ),
        "fit_evidence_sha256": fit_evidence_sha256,
        "normalized_decoder_ridge": float(normalized_decoder_ridge),
        "normalized_input_correction_rho": float(normalized_input_correction_rho),
        "coordinate_completion_receipt": completion,
        "coordinate_completion_receipt_sha256": _digest(
            coordinate_completion_receipt_sha256,
            "coordinate_completion_receipt_sha256",
        ),
        "capture_boundary": (
            "immediately-before-compile_lfm2_normalized_input_decoder"
        ),
        "parent_checkpoint_required_for_replay": False,
        "trace_cache_required_for_replay": False,
        "target_model_required_for_replay": False,
        "gpu_required_for_replay": False,
        "pickle_used": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    manifest = {**body, "capsule_sha256": canonical_sha256(body)}

    root = _safe_root(capture_root, create=True)
    destination = root / f"layer-{layer_index:04d}"
    if os.path.lexists(destination):
        raise Lfm2NormalizedInputReplayCapsuleError(
            f"Replay capsule destination already exists: {destination}"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".normalized-input-layer-{layer_index:04d}-",
            dir=str(root),
        )
    ).absolute()
    stage_identity = (stage.lstat().st_dev, stage.lstat().st_ino)
    moved = False
    try:
        with (stage / TENSOR_FILENAME).open("xb") as handle:
            handle.write(tensor_payload)
            handle.flush()
            os.fsync(handle.fileno())
        with (stage / MANIFEST_FILENAME).open("xb") as handle:
            handle.write(canonical_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(stage)
        _atomic_rename_noreplace(stage, destination)
        moved = True
        _fsync_directory(root)
    except FileExistsError as exc:
        raise Lfm2NormalizedInputReplayCapsuleError(
            f"Replay capsule destination already exists: {destination}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule could not be atomically published."
        ) from exc
    finally:
        if not moved and os.path.lexists(stage):
            current = stage.lstat()
            if (
                (current.st_dev, current.st_ino) != stage_identity
                or not stat.S_ISDIR(current.st_mode)
                or stage.is_symlink()
                or stage.resolve(strict=True).parent != root
                or not stage.name.startswith(
                    f".normalized-input-layer-{layer_index:04d}-"
                )
            ):
                raise Lfm2NormalizedInputReplayCapsuleError(
                    "Refusing to remove unauthenticated replay-capsule staging."
                )
            shutil.rmtree(stage)
            _fsync_directory(root)
    return load_lfm2_normalized_input_replay_capsule(destination)


def load_lfm2_normalized_input_replay_capsule(
    capsule_directory: str | os.PathLike[str],
) -> Lfm2NormalizedInputReplayCapsule:
    """Safely load and authenticate one exact compiler request."""

    try:
        import torch
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise Lfm2NormalizedInputReplayCapsuleError(
            "PyTorch and safetensors are required for replay capsules."
        ) from exc
    root = _safe_capsule_directory(capsule_directory)
    root_stat = root.lstat()
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    manifest_payload = _read_pinned_regular_file(
        root / MANIFEST_FILENAME,
        MANIFEST_FILENAME,
    )
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule manifest is not valid JSON."
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest_payload != canonical_json_bytes(manifest) + b"\n"
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule manifest is not canonical JSON."
        )
    body = dict(manifest)
    capsule_sha256 = body.pop("capsule_sha256", None)
    layer_index = body.get("layer_index")
    if (
        body.get("schema") != LFM2_NORMALIZED_INPUT_REPLAY_CAPSULE_SCHEMA
        or _digest(capsule_sha256, "capsule_sha256") != canonical_sha256(body)
        or isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or root.name != f"layer-{layer_index:04d}"
        or body.get("tensor_filename") != TENSOR_FILENAME
        or body.get("capture_boundary")
        != "immediately-before-compile_lfm2_normalized_input_decoder"
        or body.get("parent_checkpoint_required_for_replay") is not False
        or body.get("trace_cache_required_for_replay") is not False
        or body.get("target_model_required_for_replay") is not False
        or body.get("gpu_required_for_replay") is not False
        or body.get("pickle_used") is not False
        or any(
            body.get(name) != 0
            for name in ("training_steps", "backward_calls", "optimizer_steps")
        )
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule manifest seal or execution contract is invalid."
        )

    tensor_payload = _read_pinned_regular_file(
        root / TENSOR_FILENAME,
        TENSOR_FILENAME,
    )
    if body.get("tensor_file_size") != len(tensor_payload) or body.get(
        "tensor_file_sha256"
    ) != _file_sha256(tensor_payload):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule safetensors bytes changed."
        )
    try:
        tensors = load(tensor_payload)
    except Exception as exc:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule safetensors are unreadable."
        ) from exc
    tensor_records = body.get("tensor_records")
    tensor_keys = body.get("tensor_keys")
    if (
        not isinstance(tensor_records, Mapping)
        or not isinstance(tensor_keys, list)
        or tensor_keys != sorted(tensors)
        or set(tensor_records) != set(tensors)
        or body.get("tensor_inventory_sha256") != canonical_sha256(tensor_records)
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule tensor inventory is incomplete or reordered."
        )
    for name, value in tensors.items():
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.ndim == 0
            or not value.is_floating_point()
            or value.is_complex()
            or not bool(torch.isfinite(value).all().item())
            or tensor_records.get(name) != _tensor_record(value)
        ):
            raise Lfm2NormalizedInputReplayCapsuleError(
                f"Replay capsule tensor {name!r} failed authentication."
            )

    input_names = body.get("input_tensor_names")
    input_hashes = body.get("input_tensor_sha256")
    if (
        input_names != {name: name for name in _INPUT_TENSOR_NAMES}
        or not isinstance(input_hashes, Mapping)
        or set(input_hashes)
        != {
            "residual_chart",
            "parent_ffn_norm_weight",
            "target_ffn_norm_weight",
            "raw_structural_decoder",
        }
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule compiler input bindings are invalid."
        )
    for name, expected in input_hashes.items():
        if _digest(expected, f"{name} SHA-256") != lfm2_authenticated_tensor_sha256(
            tensors[name]
        ):
            raise Lfm2NormalizedInputReplayCapsuleError(
                f"Replay capsule input {name!r} changed."
            )

    case_count = body.get("case_count")
    case_order = body.get("case_order")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count <= 0
        or not isinstance(case_order, list)
        or len(case_order) != case_count
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule case order is invalid."
        )
    cases = []
    used_case_tensor_names: set[str] = set()
    for expected_index, record in enumerate(case_order):
        if (
            not isinstance(record, Mapping)
            or record.get("case_index") != expected_index
        ):
            raise Lfm2NormalizedInputReplayCapsuleError(
                "Replay capsule case indices are not exact and ordered."
            )
        names = record.get("tensor_names")
        if not isinstance(names, Mapping) or set(names) != {
            *_CASE_TENSOR_FIELDS,
            "parent_contribution",
        }:
            raise Lfm2NormalizedInputReplayCapsuleError(
                "Replay capsule case tensor bindings are invalid."
            )
        kwargs = {}
        for field_name in _CASE_TENSOR_FIELDS:
            expected_name = f"case.{expected_index:04d}.{field_name}"
            if names.get(field_name) != expected_name or expected_name not in tensors:
                raise Lfm2NormalizedInputReplayCapsuleError(
                    "Replay capsule case tensor order or names changed."
                )
            kwargs[field_name] = tensors[expected_name]
            used_case_tensor_names.add(expected_name)
        contribution_name = names.get("parent_contribution")
        if contribution_name is not None:
            expected_name = f"case.{expected_index:04d}.parent_contribution"
            if contribution_name != expected_name or expected_name not in tensors:
                raise Lfm2NormalizedInputReplayCapsuleError(
                    "Replay capsule parent contribution binding changed."
                )
            kwargs["parent_contribution"] = tensors[expected_name]
            used_case_tensor_names.add(expected_name)
        else:
            kwargs["parent_contribution"] = None
        cases.append(Lfm2CachedNormalizedInputMlpFoldCase(**kwargs))
    if set(tensors) != set(_INPUT_TENSOR_NAMES) | used_case_tensor_names:
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule contains unbound case tensors."
        )

    weights = _positive_weights(
        tensors["case_sample_weights"],
        case_count=case_count,
    )
    if body.get("case_sample_weights_sha256") != (
        lfm2_authenticated_tensor_sha256(tensors["case_sample_weights"])
    ) or body.get("fit_evidence_sha256") != (
        lfm2_normalized_input_mlp_fit_evidence_sha256(tuple(cases), weights)
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule ordered FIT evidence authentication failed."
        )
    ridge = body.get("normalized_decoder_ridge")
    rho = body.get("normalized_input_correction_rho")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (ridge, rho)
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule decoder policy is invalid."
        )
    completion_sha256 = _digest(
        body.get("coordinate_completion_receipt_sha256"),
        "coordinate_completion_receipt_sha256",
    )
    completion = _authenticated_completion_receipt(
        body.get("coordinate_completion_receipt"),
        claimed_sha256=completion_sha256,
        raw_structural_decoder=tensors["raw_structural_decoder"],
    )
    rechecked_root = _safe_capsule_directory(root)
    rechecked_stat = rechecked_root.lstat()
    if (
        rechecked_root != root
        or (rechecked_stat.st_dev, rechecked_stat.st_ino) != root_identity
    ):
        raise Lfm2NormalizedInputReplayCapsuleError(
            "Replay capsule root changed during authentication."
        )

    return Lfm2NormalizedInputReplayCapsule(
        root=root,
        layer_index=layer_index,
        residual_chart=tensors["residual_chart"],
        residual_chart_sha256=input_hashes["residual_chart"],
        parent_ffn_norm_weight=tensors["parent_ffn_norm_weight"],
        parent_ffn_norm_weight_sha256=input_hashes["parent_ffn_norm_weight"],
        target_ffn_norm_weight=tensors["target_ffn_norm_weight"],
        target_ffn_norm_weight_sha256=input_hashes["target_ffn_norm_weight"],
        cases=tuple(cases),
        case_sample_weights=weights,
        normalized_decoder_ridge=float(ridge),
        normalized_input_correction_rho=float(rho),
        raw_structural_decoder=tensors["raw_structural_decoder"],
        raw_structural_decoder_sha256=input_hashes["raw_structural_decoder"],
        coordinate_completion_receipt=MappingProxyType(completion),
        coordinate_completion_receipt_sha256=completion_sha256,
        capsule_sha256=capsule_sha256,
        manifest=MappingProxyType(manifest),
    )


__all__ = [
    "LFM2_NORMALIZED_INPUT_REPLAY_CAPSULE_SCHEMA",
    "Lfm2NormalizedInputReplayCapsule",
    "Lfm2NormalizedInputReplayCapsuleError",
    "load_lfm2_normalized_input_replay_capsule",
    "prepare_lfm2_normalized_input_replay_capture_root",
    "write_lfm2_normalized_input_replay_capsule",
]
