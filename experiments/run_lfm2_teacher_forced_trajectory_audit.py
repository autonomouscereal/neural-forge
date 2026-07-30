#!/usr/bin/env python3
"""Run the cached fold-only LFM2 teacher-forced trajectory factorial audit.

This runner constructs only the authenticated child.  Parent prompt/continuation
tokens and parent hidden/logit references are selected from an immutable trace
cache; the exact parent tied table is read directly from local safetensors.
There is no parent construction, network access, trace recapture, training,
backpropagation, global-map solve, pruning, or quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_config,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_functional_output_refit import (  # noqa: E402
    authenticate_lfm2_functional_output_refit_result,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_separate_sparse_semantic_code_runtime,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (  # noqa: E402
    load_lfm2_sparse_semantic_codebook,
)
from neural_forge.hf_lfm2_teacher_forced_trajectory_audit import (  # noqa: E402
    LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA,
    Lfm2TeacherForcedTrajectoryAuditError,
    aggregate_teacher_forced_trajectory_audit,
    assert_text_span_roundtrip_only,
    authenticate_prompt_result,
    build_teacher_forced_trajectory_audit_request,
    resolve_floating_dtype,
    resolve_runtime_dtype_policy,
    run_teacher_forced_trajectory_batch,
    seal_parent_trajectory_batch,
    tensor_sha256,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)


RUNNER_SCHEMA = "neural-forge-lfm2-teacher-forced-trajectory-runner-v1"
BATCH_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-teacher-forced-trajectory-batch-checkpoint-v1"
)
COMPLETION_SCHEMA = (
    "neural-forge-lfm2-teacher-forced-trajectory-cache-completion-v1"
)
OUTPUT_COEFFICIENT_KEY = "output_semantic_code_coefficients"
PARENT_TIED_KEY = "model.embed_tokens.weight"
DIRECT_LOCAL_RUNTIME_ABI_SCHEMA = "neural-forge-lfm2-direct-local-runtime-abi-v1"


class Lfm2TeacherForcedTrajectoryRunnerError(RuntimeError):
    """Raised when direct-local audit orchestration fails closed."""


def _plain_json(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2TeacherForcedTrajectoryRunnerError(f"{name} must be positive.")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _installed_package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_source_record(identifier: str, value: Any) -> dict[str, Any]:
    is_path_value = isinstance(value, (str, os.PathLike))
    path_value = os.fspath(value) if is_path_value else None
    if path_value is None:
        try:
            path_value = inspect.getsourcefile(value)
        except TypeError:
            path_value = None
    module = None if is_path_value else inspect.getmodule(value)
    if path_value is None and module is not None:
        path_value = getattr(module, "__file__", None)
    if path_value is None:
        path_value = getattr(value, "__file__", None)
    if not isinstance(path_value, str):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            f"Runtime ABI source {identifier!r} has no local file."
        )
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            f"Runtime ABI source {identifier!r} is not a local file."
        )
    return {
        "identifier": identifier,
        "module": (
            None
            if is_path_value
            else getattr(value, "__module__", getattr(value, "__name__", None))
        ),
        "qualified_name": None if is_path_value else getattr(value, "__qualname__", None),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
    }


def _torch_backend_flag(value: Any, name: str) -> bool | int | float | str | None:
    observed = getattr(value, name, None)
    if callable(observed):
        try:
            observed = observed()
        except (RuntimeError, TypeError):
            return None
    if observed is None or isinstance(observed, (bool, int, float, str)):
        return observed
    return str(observed)


def build_direct_local_runtime_abi_receipt(
    checkpoint_directory: Path,
    *,
    runtime_device: str | torch.device,
    runtime_dtype: torch.dtype,
    provenance_device: str | torch.device,
    provenance_dtype: torch.dtype,
) -> dict[str, Any]:
    """Seal the child execution ABI without constructing the child model."""

    try:
        from transformers.models.lfm2 import configuration_lfm2, modeling_lfm2
    except ImportError as exc:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Concrete local Transformers LFM2 implementation is unavailable."
        ) from exc

    directory = _safe_input_directory(
        checkpoint_directory, "runtime ABI checkpoint directory"
    )
    config = load_lfm2_direct_local_config(directory)
    try:
        placement = torch.device(runtime_device)
        provenance_placement = torch.device(provenance_device)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Runtime ABI received an invalid device."
        ) from exc
    if runtime_dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    } or provenance_dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Runtime ABI dtypes must be explicit floating dtypes."
        )

    eager_attention = getattr(modeling_lfm2, "eager_attention_forward", None)
    if not callable(eager_attention):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Local LFM2 eager attention implementation is unavailable."
        )
    requested_attention = getattr(config, "_attn_implementation", None)
    registry = getattr(modeling_lfm2, "ALL_ATTENTION_FUNCTIONS", None)
    resolver = getattr(registry, "get_interface", None)
    if requested_attention in {None, "eager"}:
        attention_callable = eager_attention
    elif callable(resolver):
        attention_callable = resolver(requested_attention, eager_attention)
    else:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Configured LFM2 attention backend cannot be resolved pre-construction."
        )
    if not callable(attention_callable):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Selected LFM2 attention backend is not callable."
        )

    compiling = bool(modeling_lfm2.is_torchdynamo_compiling())
    if compiling:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Trajectory audit does not permit a torch-compiled runtime."
        )
    fast_conv_available = bool(
        getattr(modeling_lfm2, "is_fast_path_available", False)
    )
    selected_conv_backend = (
        "causal-conv1d-cuda-fast"
        if fast_conv_available and placement.type == "cuda"
        else "torch-depthwise-conv1d-slow"
    )
    conv_kernel_sources = []
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        kernel = getattr(modeling_lfm2, name, None)
        conv_kernel_sources.append(
            None if kernel is None else _runtime_source_record(name, kernel)
        )

    device_body: dict[str, Any] = {
        "requested": str(placement),
        "type": placement.type,
        "index": placement.index,
    }
    if placement.type == "cuda":
        if not torch.cuda.is_available():
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "CUDA runtime ABI was requested but CUDA is unavailable."
            )
        index = torch.cuda.current_device() if placement.index is None else placement.index
        properties = torch.cuda.get_device_properties(index)
        device_body.update(
            {
                "resolved_index": int(index),
                "name": str(properties.name),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(properties.total_memory),
            }
        )

    layer_types = tuple(str(value) for value in getattr(config, "layer_types", ()))
    if not layer_types:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Local LFM2 config has no explicit layer types."
        )
    source_records = [
        _runtime_source_record(
            "neural_forge.hf_lfm2_teacher_forced_trajectory_audit",
            SRC / "neural_forge" / "hf_lfm2_teacher_forced_trajectory_audit.py",
        ),
        _runtime_source_record(
            "neural_forge.hf_lfm2_direct_local",
            SRC / "neural_forge" / "hf_lfm2_direct_local.py",
        ),
        _runtime_source_record(
            "neural_forge.hf_lfm2_dynamic_blank_materializer",
            SRC / "neural_forge" / "hf_lfm2_dynamic_blank_materializer.py",
        ),
        _runtime_source_record(
            "neural_forge.hf_lfm2_dynamic_physical_layout",
            SRC / "neural_forge" / "hf_lfm2_dynamic_physical_layout.py",
        ),
        _runtime_source_record(
            "neural_forge.hf_lfm2_semantic_step_token_fold",
            SRC / "neural_forge" / "hf_lfm2_semantic_step_token_fold.py",
        ),
        _runtime_source_record("transformers.configuration_lfm2", configuration_lfm2),
        _runtime_source_record("transformers.modeling_lfm2", modeling_lfm2),
    ]
    attention_source = _runtime_source_record(
        "selected_attention_callable", attention_callable
    )
    body = {
        "schema": DIRECT_LOCAL_RUNTIME_ABI_SCHEMA,
        "python": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
        },
        "packages": {
            "torch": str(torch.__version__),
            "transformers": _installed_package_version("transformers"),
            "safetensors": _installed_package_version("safetensors"),
            "causal_conv1d": _installed_package_version("causal-conv1d"),
        },
        "torch_runtime": {
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": _torch_backend_flag(
                torch.backends.cuda.matmul, "allow_tf32"
            ),
            "cudnn_allow_tf32": _torch_backend_flag(
                torch.backends.cudnn, "allow_tf32"
            ),
            "flash_sdp_enabled": _torch_backend_flag(
                torch.backends.cuda, "flash_sdp_enabled"
            ),
            "mem_efficient_sdp_enabled": _torch_backend_flag(
                torch.backends.cuda, "mem_efficient_sdp_enabled"
            ),
            "math_sdp_enabled": _torch_backend_flag(
                torch.backends.cuda, "math_sdp_enabled"
            ),
        },
        "runtime_device": device_body,
        "runtime_dtype": str(runtime_dtype).removeprefix("torch."),
        "provenance_device": str(provenance_placement),
        "provenance_dtype": str(provenance_dtype).removeprefix("torch."),
        "local_sources": source_records,
        "topology_execution": {
            "model_type": getattr(config, "model_type", None),
            "layer_types_sha256": canonical_sha256(list(layer_types)),
            "attention_layer_count": sum(
                layer_type == "full_attention" for layer_type in layer_types
            ),
            "short_conv_layer_count": sum(
                layer_type != "full_attention" for layer_type in layer_types
            ),
            "attention_backend_requested": requested_attention,
            "attention_backend_selected": (
                getattr(attention_callable, "__qualname__", None)
                or getattr(attention_callable, "__name__", None)
            ),
            "attention_backend_source": attention_source,
            "short_conv_fast_path_available": fast_conv_available,
            "short_conv_backend_selected": selected_conv_backend,
            "short_conv_kernel_sources": conv_kernel_sources,
            "torchdynamo_compiling": compiling,
            "body_use_cache": False,
            "body_execution": "full-teacher-forced-donor-trajectory",
            "attention_mask": "2d-int64-ones",
        },
        "child_model_objects_constructed": 0,
        "network_access_performed": False,
    }
    return {**body, "abi_sha256": canonical_sha256(body)}


def _safe_input_directory(value: Path, name: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2TeacherForcedTrajectoryRunnerError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2TeacherForcedTrajectoryRunnerError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            f"{name} must be one canonical real directory."
        )
    return resolved


def _safe_input_file(value: Path, name: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2TeacherForcedTrajectoryRunnerError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2TeacherForcedTrajectoryRunnerError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_file():
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            f"{name} must be one canonical real file."
        )
    return resolved


def _output_directory(value: Path) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        if lexical.is_symlink() or not lexical.is_dir():
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "output directory must be one real directory."
            )
        return lexical.resolve(strict=True)
    lexical.mkdir(parents=True)
    return lexical.resolve(strict=True)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    file_path = _safe_input_file(path, name)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            f"{name} is not readable JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2TeacherForcedTrajectoryRunnerError(f"{name} must be an object.")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class StructuredProgress:
    """Ordered JSONL diagnostics mirrored to stdout and a local append log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sequence = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, value: Mapping[str, Any]) -> None:
        row = {"schema": RUNNER_SCHEMA, "sequence": self.sequence, **_plain_json(value)}
        self.sequence += 1
        payload = json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        print(payload, flush=True)
        with self.path.open("a", encoding="ascii", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()


def _checkpoint_inventory(directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Child checkpoint contains a link or non-file entry."
            )
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    body = {
        "schema": "neural-forge-local-checkpoint-inventory-v1",
        "files": files,
    }
    return {**body, "inventory_sha256": canonical_sha256(body)}


def _local_safetensor_path(directory: Path, tensor_key: str) -> Path:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file() and not index_path.is_symlink():
        index = _read_json(index_path, "parent safetensors index")
        weight_map = index.get("weight_map")
        name = weight_map.get(tensor_key) if isinstance(weight_map, Mapping) else None
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Parent safetensors index does not safely locate the tied table."
            )
        path = directory / name
    else:
        path = directory / "model.safetensors"
    if path.is_symlink() or not path.is_file() or path.parent != directory:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Parent tied-table safetensors file is missing or unsafe."
        )
    return path


def load_exact_parent_table(
    directory: Path,
    *,
    tensor_key: str,
    donor_row_count: int,
    hidden_size: int,
    expected_source_file_sha256: str,
    expected_table_sha256: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select only defined tied rows; never construct the parent model."""

    path = _local_safetensor_path(directory, tensor_key)
    source_sha = _file_sha256(path)
    if source_sha != _digest(
        expected_source_file_sha256, "expected parent source file SHA-256"
    ):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Parent tied-table source file differs from expectation."
        )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if tensor_key not in handle.keys():
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Parent tied-table tensor key is missing."
            )
        view = handle.get_slice(tensor_key)
        shape = tuple(int(size) for size in view.get_shape())
        if len(shape) != 2 or shape[0] < donor_row_count or shape[1] != hidden_size:
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Parent tied-table geometry differs from the requested defined rows."
            )
        table = view[:donor_row_count, :].to(dtype=torch.float32).contiguous()
    table_sha = tensor_sha256(table)
    if table_sha != _digest(expected_table_sha256, "expected parent table SHA-256"):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Selected exact parent table differs from expectation."
        )
    body = {
        "schema": "neural-forge-direct-local-exact-parent-table-v1",
        "source_file": path.name,
        "source_file_sha256": source_sha,
        "tensor_key": tensor_key,
        "source_shape": list(shape),
        "selected_shape": list(table.shape),
        "selected_dtype": "float32",
        "selected_table_sha256": table_sha,
        "parent_model_objects_constructed": 0,
        "parent_forward_calls": 0,
        "network_access_performed": False,
    }
    return table, {**body, "receipt_sha256": canonical_sha256(body)}


def _sparse_codebook_table_sha256(value: torch.Tensor) -> str:
    """Match the authenticated sparse-codebook table hash domain."""

    tensor = value.detach()
    digest = hashlib.sha256(b"neural-forge-lfm2-sparse-semantic-tensor-v1\0")
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    for start in range(0, int(tensor.shape[0]), 4096):
        block = tensor[start : min(start + 4096, int(tensor.shape[0]))]
        digest.update(
            block.to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _phase_names(forced_count: int) -> tuple[str, ...]:
    return ("prefill", *(f"decode.{index:03d}" for index in range(forced_count)))


def load_parent_trajectory_batch(
    *,
    session: Any,
    reference: Any,
    prompt_id: str,
    trace_parent_fingerprint: str,
    donor_row_count: int,
    hidden_size: int,
) -> Any:
    """Select, authenticate, align, and immediately release one trace batch."""

    record = reference.record
    capture = record.get("capture")
    if not isinstance(capture, Mapping):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Trace batch has no authenticated capture contract."
        )
    forced_count = _positive_int(capture.get("decode_steps"), "trace decode_steps")
    phases = _phase_names(forced_count)
    inventory = set(reference.ordered_tensor_names)
    selected = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
    }
    if "input.position_ids" in inventory:
        selected.add("input.position_ids")
    for phase in phases:
        selected.add(f"{phase}.output.lm_head_input")
        selected.add(f"{phase}.logits")
        if phase != "prefill":
            selected.add(f"{phase}.forced_token_ids")
    names = tuple(sorted(selected))
    if any(name not in inventory for name in names):
        missing = sorted(name for name in names if name not in inventory)
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            f"Trace batch is missing trajectory tensors: {missing}."
        )
    payload = session.load_selected_tensors(reference.index, names)
    try:
        tensors = payload.tensors
        prompt = tensors["input.input_ids"].detach().cpu().to(torch.int64)
        attention = tensors["input.attention_mask"].detach().cpu().to(torch.int64)
        forced = tensors["input.forced_decode_ids"].detach().cpu().to(torch.int64)
        if (
            prompt.ndim != 2
            or tuple(prompt.shape) != tuple(attention.shape)
            or int(prompt.shape[0]) != 1
            or forced.ndim != 2
            or tuple(forced.shape) != (1, forced_count)
            or not bool((attention == 1).all())
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Trace prompt/trajectory inputs are not canonical unpadded rows."
            )
        position = tensors.get("input.position_ids")
        position_vector = None if position is None else position[0].to(torch.int64)
        hidden_rows = []
        logit_rows = []
        for phase_index, phase in enumerate(phases):
            hidden = tensors[f"{phase}.output.lm_head_input"]
            logits = tensors[f"{phase}.logits"]
            if (
                hidden.ndim != 3
                or logits.ndim != 3
                or int(hidden.shape[0]) != 1
                or int(logits.shape[0]) != 1
                or int(hidden.shape[-1]) != hidden_size
                or int(logits.shape[-1]) < donor_row_count
            ):
                raise Lfm2TeacherForcedTrajectoryRunnerError(
                    f"Trace phase {phase!r} has invalid hidden/logit geometry."
                )
            hidden_rows.append(hidden[0, -1, :])
            logit_rows.append(logits[0, -1, :donor_row_count])
            if phase_index:
                phase_forced = tensors[f"{phase}.forced_token_ids"].to(torch.int64)
                if not torch.equal(phase_forced, forced[:, phase_index - 1 : phase_index]):
                    raise Lfm2TeacherForcedTrajectoryRunnerError(
                        f"Trace phase {phase!r} lost its forced-token identity."
                    )
        result = seal_parent_trajectory_batch(
            batch_index=int(reference.index),
            prompt_id=prompt_id,
            trace_cache_key=session.cache_key,
            trace_parent_fingerprint=trace_parent_fingerprint,
            trace_selective_session_sha256=session.evidence_sha256,
            trace_input_sha256=reference.input_sha256,
            trace_record_sha256=reference.record_sha256,
            selected_payload_sha256=payload.selection_sha256,
            prompt_token_ids=prompt[0],
            forced_token_ids=forced[0],
            parent_hidden_states=torch.stack(hidden_rows),
            parent_logits=torch.stack(logit_rows),
            position_ids=position_vector,
        )
    finally:
        session.release_selected_tensors(payload)
    return result


def _load_prompt_labels(path: Path | None, batch_count: int) -> tuple[str, ...]:
    if path is None:
        return tuple(f"batch-{index:06d}" for index in range(batch_count))
    value = _read_json(path, "prompt labels")
    rows = value.get("prompts", value.get("rows"))
    if not isinstance(rows, list) or len(rows) != batch_count:
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Prompt labels must contain exactly one row per trace batch."
        )
    labels = []
    for index, row in enumerate(rows):
        label = row.get("id") if isinstance(row, Mapping) else None
        if not isinstance(label, str) or not label:
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                f"Prompt label row {index} has no nonempty id."
            )
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise Lfm2TeacherForcedTrajectoryRunnerError("Prompt labels are duplicated.")
    return tuple(labels)


def _model_snapshot(model: Any) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    if any(parameter.grad is not None for parameter in parameters):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Child has gradients before the fold-only audit."
        )
    return {
        "ids": tuple(id(value) for value in parameters),
        "versions": tuple(int(value._version) for value in parameters),
        "numel": sum(int(value.numel()) for value in parameters),
        "module_ids": tuple(id(value) for value in model.modules()),
    }


def _verify_model_snapshot(model: Any, snapshot: Mapping[str, Any]) -> None:
    parameters = tuple(model.parameters())
    if (
        tuple(id(value) for value in parameters) != snapshot["ids"]
        or tuple(int(value._version) for value in parameters) != snapshot["versions"]
        or sum(int(value.numel()) for value in parameters) != snapshot["numel"]
        or tuple(id(value) for value in model.modules()) != snapshot["module_ids"]
        or any(parameter.grad is not None for parameter in parameters)
    ):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Child parameters/modules changed during the fold-only audit."
        )


def _batch_checkpoint(
    *, request_cache_key: str, reference: Any, prompt_result: Mapping[str, Any]
) -> dict[str, Any]:
    result = authenticate_prompt_result(prompt_result, expected_batch_index=reference.index)
    body = {
        "schema": BATCH_CHECKPOINT_SCHEMA,
        "request_cache_key": request_cache_key,
        "batch_index": int(reference.index),
        "trace_input_sha256": reference.input_sha256,
        "trace_record_sha256": reference.record_sha256,
        "prompt_result": result,
    }
    return {**body, "checkpoint_sha256": canonical_sha256(body)}


def _authenticate_batch_checkpoint(
    value: Mapping[str, Any], *, request_cache_key: str, reference: Any
) -> dict[str, Any]:
    checkpoint = _plain_json(value)
    supplied = checkpoint.pop("checkpoint_sha256", None)
    if (
        supplied != canonical_sha256(checkpoint)
        or checkpoint.get("schema") != BATCH_CHECKPOINT_SCHEMA
        or checkpoint.get("request_cache_key") != request_cache_key
        or checkpoint.get("batch_index") != reference.index
        or checkpoint.get("trace_input_sha256") != reference.input_sha256
        or checkpoint.get("trace_record_sha256") != reference.record_sha256
    ):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Trajectory batch checkpoint authentication failed."
        )
    checkpoint["prompt_result"] = authenticate_prompt_result(
        checkpoint["prompt_result"], expected_batch_index=reference.index
    )
    return {**checkpoint, "checkpoint_sha256": supplied}


def _cached_audit(cache_directory: Path, cache_key: str) -> dict[str, Any] | None:
    completion_path = cache_directory / "complete.json"
    audit_path = cache_directory / "audit.json"
    if not completion_path.exists() and not audit_path.exists():
        return None
    if not completion_path.is_file() or not audit_path.is_file():
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Trajectory cache has an incomplete final publication."
        )
    completion = _read_json(completion_path, "trajectory completion")
    supplied_completion = completion.pop("completion_sha256", None)
    audit = _read_json(audit_path, "trajectory audit")
    supplied_audit = audit.pop("audit_sha256", None)
    if (
        supplied_completion != canonical_sha256(completion)
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("request_cache_key") != cache_key
        or completion.get("audit_file_sha256") != _file_sha256(audit_path)
        or supplied_audit != canonical_sha256(audit)
        or audit.get("schema") != LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA
        or audit.get("request_cache_key") != cache_key
        or completion.get("audit_sha256") != supplied_audit
    ):
        raise Lfm2TeacherForcedTrajectoryRunnerError(
            "Trajectory final cache publication authentication failed."
        )
    return {**audit, "audit_sha256": supplied_audit}


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    output_root = _output_directory(args.output_directory)
    progress = StructuredProgress(output_root / "trajectory-audit-progress.jsonl")
    progress({"event": "preflight-start", "stage": "preflight"})
    try:
        donor_count = _positive_int(args.donor_row_count, "donor_row_count")
        target_count = _positive_int(args.target_row_count, "target_row_count")
        hidden_size = _positive_int(args.hidden_size, "hidden_size")
        parent_directory = _safe_input_directory(
            args.parent_model_directory, "parent model directory"
        )
        checkpoint_directory = _safe_input_directory(
            args.checkpoint_directory, "child checkpoint directory"
        )
        donor_tokenizer_directory = _safe_input_directory(
            args.donor_tokenizer_directory, "donor tokenizer directory"
        )
        target_tokenizer_directory = _safe_input_directory(
            args.target_tokenizer_directory, "target tokenizer directory"
        )
        bridge_directory = _safe_input_directory(
            args.bridge_directory, "bridge directory"
        )
        codebook_directory = _safe_input_directory(
            args.codebook_directory, "codebook directory"
        )
        trace_directory = _safe_input_directory(
            args.trace_cache_directory, "trace cache directory"
        )
        span_path = _safe_input_file(args.span_evidence_json, "span evidence")
        output_coefficients_path = _safe_input_file(
            args.output_coefficients_safetensors, "output coefficient tensor"
        )
        output_receipt_path = _safe_input_file(
            args.output_refit_receipt_json, "output refit receipt"
        )

        runtime_policy = resolve_runtime_dtype_policy(
            args.runtime_dtype, args.runtime_device
        )
        runtime_dtype = resolve_floating_dtype(
            runtime_policy["effective_dtype"], role="runtime", allow_low_precision=True
        )
        provenance_dtype = resolve_floating_dtype(
            args.provenance_dtype, role="provenance", allow_low_precision=True
        )
        metric_dtype = resolve_floating_dtype(
            args.metric_dtype, role="metric", allow_low_precision=False
        )
        score_dtype = resolve_floating_dtype(
            args.score_dtype, role="score", allow_low_precision=False
        )
        progress(
            {
                "event": "dtype-policy-resolved",
                "stage": "preflight",
                "runtime_policy": runtime_policy,
                "provenance_dtype": str(provenance_dtype).removeprefix("torch."),
                "metric_dtype": str(metric_dtype).removeprefix("torch."),
                "score_dtype": str(score_dtype).removeprefix("torch."),
            }
        )
        runtime_abi = build_direct_local_runtime_abi_receipt(
            checkpoint_directory,
            runtime_device=args.runtime_device,
            runtime_dtype=runtime_dtype,
            provenance_device=args.provenance_device,
            provenance_dtype=provenance_dtype,
        )
        progress(
            {
                "event": "direct-local-runtime-abi-sealed",
                "stage": "preflight",
                "runtime_abi": runtime_abi,
                "child_model_objects_constructed": 0,
            }
        )

        donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_tokenizer_directory)
        target_tokenizer = load_lfm2_direct_local_tokenizer(target_tokenizer_directory)
        bridge = load_strict_tokenizer_bridge_artifact(
            bridge_directory,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            expected_artifact_sha256=_digest(
                args.expected_bridge_artifact_sha256, "expected bridge artifact"
            ),
        )
        codebook = load_lfm2_sparse_semantic_codebook(
            codebook_directory,
            expected_artifact_sha256=_digest(
                args.expected_codebook_artifact_sha256, "expected codebook artifact"
            ),
        )
        codebook_receipt = codebook.as_dict()
        if (
            codebook_receipt.get("donor_row_count") != donor_count
            or codebook_receipt.get("target_row_count") != target_count
            or codebook_receipt.get("hidden_size") != hidden_size
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Authenticated codebook geometry differs from the request."
            )
        span_evidence = _read_json(span_path, "span evidence")
        if span_evidence.get("evidence_sha256") != _digest(
            args.expected_span_evidence_sha256, "expected span evidence"
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Semantic span evidence differs from expectation."
            )
        with safe_open(
            str(output_coefficients_path), framework="pt", device="cpu"
        ) as handle:
            if tuple(handle.keys()) != (args.output_coefficients_key,):
                raise Lfm2TeacherForcedTrajectoryRunnerError(
                    "Output coefficient safetensors inventory is not exact."
                )
            output_coefficients = handle.get_tensor(args.output_coefficients_key)
        output_result = authenticate_lfm2_functional_output_refit_result(
            output_semantic_code_coefficients=output_coefficients,
            receipt=_read_json(output_receipt_path, "output refit receipt"),
            expected_cache_key=_digest(
                args.expected_output_refit_cache_key, "expected output refit cache key"
            ),
        )
        if (
            output_result.result_sha256
            != _digest(args.expected_output_refit_result_sha256, "expected output result")
            or output_result.receipt.get("input_codebook_artifact_sha256")
            != codebook.artifact_sha256
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Output C_out result is not bound to the authenticated C_in codebook."
            )
        exact_table, exact_table_receipt = load_exact_parent_table(
            parent_directory,
            tensor_key=args.parent_tensor_key,
            donor_row_count=donor_count,
            hidden_size=hidden_size,
            expected_source_file_sha256=args.expected_parent_source_file_sha256,
            expected_table_sha256=args.expected_parent_table_sha256,
        )
        if _sparse_codebook_table_sha256(exact_table) != codebook_receipt.get(
            "donor_table_sha256"
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Exact parent table differs from the table sealed by C_in."
            )
        inventory = _checkpoint_inventory(checkpoint_directory)
        if inventory["inventory_sha256"] != _digest(
            args.expected_checkpoint_inventory_sha256,
            "expected checkpoint inventory",
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Child checkpoint inventory differs from expectation."
            )
        trace_cache = Lfm2CircuitTraceCache.open(trace_directory, verify=False)
        if (
            trace_cache.cache_key
            != _digest(args.expected_trace_cache_key, "expected trace cache key")
            or trace_cache.parent_fingerprint
            != _digest(
                args.expected_trace_parent_fingerprint,
                "expected trace parent fingerprint",
            )
            or trace_cache.batch_count
            != _positive_int(args.expected_trace_batch_count, "expected trace batches")
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Trace cache metadata differs from the expected parent evidence."
            )
        prompt_labels = _load_prompt_labels(args.prompt_labels_json, trace_cache.batch_count)
        artifact_bindings = {
            "implementation_sha256": canonical_sha256(
                {
                    "core_source_sha256": _file_sha256(
                        SRC
                        / "neural_forge"
                        / "hf_lfm2_teacher_forced_trajectory_audit.py"
                    ),
                    "runner_source_sha256": _file_sha256(Path(__file__).resolve()),
                    "algorithm": (
                        "four-lane-causal-full-donor-trajectory-factorial-v2-"
                        "same-run-exact-exact-hidden-contrast"
                    ),
                }
            ),
            "direct_local_runtime_abi": runtime_abi,
            "parent_table_sha256": tensor_sha256(exact_table),
            "parent_table_source_file_sha256": exact_table_receipt[
                "source_file_sha256"
            ],
            "trace_cache_key": trace_cache.cache_key,
            "trace_parent_fingerprint": trace_cache.parent_fingerprint,
            "checkpoint_inventory_sha256": inventory["inventory_sha256"],
            "expected_checkpoint_staged_state_sha256": _digest(
                args.expected_checkpoint_staged_state_sha256,
                "expected checkpoint staged state",
            ),
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "bridge_profile_sha256": bridge.profile_fingerprint,
            "donor_tokenizer_semantics_sha256": (
                bridge.donor_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                bridge.target_tokenizer_semantics_sha256
            ),
            "span_evidence_sha256": span_evidence["evidence_sha256"],
            "input_codebook_artifact_sha256": codebook.artifact_sha256,
            "output_refit_cache_key": output_result.cache_key,
            "output_refit_result_sha256": output_result.result_sha256,
            "prompt_labels_sha256": canonical_sha256(list(prompt_labels)),
        }
        request = build_teacher_forced_trajectory_audit_request(
            artifact_bindings=artifact_bindings,
            runtime_dtype_policy=runtime_policy,
            metric_dtype=metric_dtype,
            score_dtype=score_dtype,
            donor_row_count=donor_count,
            target_row_count=target_count,
            hidden_size=hidden_size,
            execution_contract={
                "maximum_embedding_rows_per_chunk": args.embedding_rows_per_chunk,
                "maximum_donor_rows_per_score_chunk": args.donor_score_rows_per_chunk,
                "progress_log": str(progress.path),
                "progress_logging_changes_math_identity": False,
                "chunking_changes_math_identity": False,
            },
        )
        cache_key = request["cache_key"]
        cache_directory = output_root / cache_key
        cache_directory.mkdir(exist_ok=True)
        if cache_directory.is_symlink():
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Trajectory cache directory cannot be a link."
            )
        cached = _cached_audit(cache_directory, cache_key)
        if cached is not None:
            progress(
                {
                    "event": "audit-cache-hit",
                    "stage": "complete",
                    "cache_key": cache_key,
                    "audit_sha256": cached["audit_sha256"],
                    "child_model_objects_constructed": 0,
                    "child_forward_calls": 0,
                }
            )
            return cached

        progress(
            {
                "event": "audit-cache-miss",
                "stage": "preflight",
                "cache_key": cache_key,
                "cache_directory": str(cache_directory),
            }
        )
        checkpoint_io = Lfm2DirectLocalCheckpointIO(
            tokenizer_bridge=bridge,
            device=torch.device(args.provenance_device),
            dtype=provenance_dtype,
        )
        child, portable_receipt = checkpoint_io.load_authenticated_portable(
            checkpoint_directory,
            runtime_device=torch.device(args.runtime_device),
            runtime_dtype=runtime_dtype,
        )
        portable = portable_receipt.as_dict()
        if portable["staged_state_sha256"] != artifact_bindings[
            "expected_checkpoint_staged_state_sha256"
        ]:
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Authenticated child state differs from expectation."
            )
        live_table = child.get_input_embeddings().weight
        if (
            child.get_output_embeddings().weight is not live_table
            or tuple(live_table.shape) != (target_count, hidden_size)
        ):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Child must expose one tied target table with expected geometry."
            )
        routing = bind_lfm2_separate_sparse_semantic_code_runtime(
            span_evidence,
            live_table,
            input_semantic_code_offsets=codebook.semantic_code_offsets,
            input_semantic_code_target_ids=codebook.semantic_code_target_ids,
            input_semantic_code_coefficients=codebook.semantic_code_coefficients,
            output_semantic_code_offsets=codebook.semantic_code_offsets,
            output_semantic_code_target_ids=codebook.semantic_code_target_ids,
            output_semantic_code_coefficients=(
                output_result.output_semantic_code_coefficients
            ),
            target_to_donor=codebook.target_to_donor,
        )
        body = getattr(child, "model", None)
        if not callable(body):
            raise Lfm2TeacherForcedTrajectoryRunnerError(
                "Direct-local child has no callable body."
            )
        snapshot = _model_snapshot(child)
        batch_checkpoint_directory = cache_directory / "batches"
        batch_checkpoint_directory.mkdir(exist_ok=True)
        prompt_results = []
        checkpoint_hashes = []
        with trace_cache.selective_authenticated_session() as trace_session:
            progress(
                {
                    "event": "trace-session-authenticated",
                    "stage": "trace-cache",
                    "trace_session_sha256": trace_session.evidence_sha256,
                    "batch_count": trace_session.batch_count,
                }
            )
            with routing.authenticated_live_table_session(live_table) as routing_session:
                for reference in trace_session.batch_references:
                    checkpoint_path = (
                        batch_checkpoint_directory
                        / f"batch-{reference.index:06d}.json"
                    )
                    if checkpoint_path.exists():
                        checkpoint = _authenticate_batch_checkpoint(
                            _read_json(checkpoint_path, "trajectory batch checkpoint"),
                            request_cache_key=cache_key,
                            reference=reference,
                        )
                        progress(
                            {
                                "event": "batch-cache-hit",
                                "stage": "trajectory",
                                "batch_index": reference.index,
                                "prompt_id": prompt_labels[reference.index],
                                "checkpoint_sha256": checkpoint[
                                    "checkpoint_sha256"
                                ],
                            }
                        )
                    else:
                        parent_batch = load_parent_trajectory_batch(
                            session=trace_session,
                            reference=reference,
                            prompt_id=prompt_labels[reference.index],
                            trace_parent_fingerprint=trace_cache.parent_fingerprint,
                            donor_row_count=donor_count,
                            hidden_size=hidden_size,
                        )
                        span_receipt = assert_text_span_roundtrip_only(
                            donor_tokenizer=donor_tokenizer,
                            target_tokenizer=target_tokenizer,
                            prompt_token_ids=parent_batch.prompt_token_ids,
                            forced_token_ids=parent_batch.forced_token_ids,
                        )
                        prompt_result = run_teacher_forced_trajectory_batch(
                            child_body=body,
                            routing=routing,
                            routing_session=routing_session,
                            live_target_table=live_table,
                            exact_parent_table=exact_table,
                            batch=parent_batch,
                            metric_dtype=metric_dtype,
                            score_dtype=score_dtype,
                            maximum_embedding_rows_per_chunk=(
                                args.embedding_rows_per_chunk
                            ),
                            maximum_donor_rows_per_score_chunk=(
                                args.donor_score_rows_per_chunk
                            ),
                            text_span_roundtrip_receipt=span_receipt,
                            progress=progress,
                        )
                        checkpoint = _batch_checkpoint(
                            request_cache_key=cache_key,
                            reference=reference,
                            prompt_result=prompt_result,
                        )
                        _atomic_write_json(checkpoint_path, checkpoint)
                        checkpoint = _authenticate_batch_checkpoint(
                            _read_json(checkpoint_path, "trajectory batch checkpoint"),
                            request_cache_key=cache_key,
                            reference=reference,
                        )
                        progress(
                            {
                                "event": "batch-checkpoint-published",
                                "stage": "trajectory",
                                "batch_index": reference.index,
                                "prompt_id": prompt_labels[reference.index],
                                "checkpoint_sha256": checkpoint[
                                    "checkpoint_sha256"
                                ],
                                "checkpoint_file_sha256": _file_sha256(
                                    checkpoint_path
                                ),
                            }
                        )
                    prompt_results.append(checkpoint["prompt_result"])
                    checkpoint_hashes.append(checkpoint["checkpoint_sha256"])
            trace_session.verify()
            trace_receipt = trace_session.as_dict()
        _verify_model_snapshot(child, snapshot)
        audit = aggregate_teacher_forced_trajectory_audit(
            request=request,
            prompt_results=prompt_results,
            trace_session_receipt=trace_receipt,
        )
        audit_body = dict(audit)
        audit_sha = audit_body.pop("audit_sha256")
        audit_body["runtime_authentication"] = {
            "direct_local_runtime_abi": runtime_abi,
            "portable_child_load_receipt": portable,
            "routing_live_table_session_receipt": routing_session.as_dict(),
            "child_parameter_snapshot_unchanged": True,
        }
        audit = {**audit_body, "audit_sha256": canonical_sha256(audit_body)}
        audit_path = cache_directory / "audit.json"
        _atomic_write_json(audit_path, audit)
        completion_body = {
            "schema": COMPLETION_SCHEMA,
            "request_cache_key": cache_key,
            "audit_sha256": audit["audit_sha256"],
            "audit_file_sha256": _file_sha256(audit_path),
            "batch_checkpoint_sha256s": checkpoint_hashes,
            "original_core_audit_sha256_before_runtime_receipts": audit_sha,
            "atomic_batch_checkpoints": True,
            "reusable_identical_math": True,
        }
        _atomic_write_json(
            cache_directory / "complete.json",
            {
                **completion_body,
                "completion_sha256": canonical_sha256(completion_body),
            },
        )
        loaded = _cached_audit(cache_directory, cache_key)
        assert loaded is not None
        progress(
            {
                "event": "audit-complete",
                "stage": "complete",
                "cache_key": cache_key,
                "audit_sha256": loaded["audit_sha256"],
                "prompt_count": loaded["prompt_count"],
                "lane_summaries": loaded["lane_summaries"],
            }
        )
        return loaded
    except BaseException as exc:
        progress(
            {
                "event": "run-failed",
                "stage": "failure",
                "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            }
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--parent-tensor-key", default=PARENT_TIED_KEY)
    parser.add_argument("--expected-parent-source-file-sha256", required=True)
    parser.add_argument("--expected-parent-table-sha256", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--expected-checkpoint-staged-state-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", type=Path, required=True)
    parser.add_argument("--target-tokenizer-directory", type=Path, required=True)
    parser.add_argument("--bridge-directory", type=Path, required=True)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--span-evidence-json", type=Path, required=True)
    parser.add_argument("--expected-span-evidence-sha256", required=True)
    parser.add_argument("--codebook-directory", type=Path, required=True)
    parser.add_argument("--expected-codebook-artifact-sha256", required=True)
    parser.add_argument("--output-coefficients-safetensors", type=Path, required=True)
    parser.add_argument("--output-coefficients-key", default=OUTPUT_COEFFICIENT_KEY)
    parser.add_argument("--output-refit-receipt-json", type=Path, required=True)
    parser.add_argument("--expected-output-refit-cache-key", required=True)
    parser.add_argument("--expected-output-refit-result-sha256", required=True)
    parser.add_argument("--trace-cache-directory", type=Path, required=True)
    parser.add_argument("--expected-trace-cache-key", required=True)
    parser.add_argument("--expected-trace-parent-fingerprint", required=True)
    parser.add_argument("--expected-trace-batch-count", type=int, required=True)
    parser.add_argument("--prompt-labels-json", type=Path)
    parser.add_argument("--donor-row-count", type=int, required=True)
    parser.add_argument("--target-row-count", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--provenance-device", required=True)
    parser.add_argument("--provenance-dtype", required=True)
    parser.add_argument("--runtime-device", required=True)
    parser.add_argument("--runtime-dtype", required=True)
    parser.add_argument("--metric-dtype", required=True)
    parser.add_argument("--score-dtype", required=True)
    parser.add_argument("--embedding-rows-per-chunk", type=int, default=1024)
    parser.add_argument("--donor-score-rows-per-chunk", type=int, default=4096)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (
        Lfm2TeacherForcedTrajectoryRunnerError,
        Lfm2TeacherForcedTrajectoryAuditError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"trajectory audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
