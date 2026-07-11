#!/usr/bin/env python3
"""Run sequential, zero-recovery compensated neuron folds on a Qwen3 parent.

Every installed fold is a reversible transaction until a cached-parent-relative
behavior, choice, and generation gate accepts it. Accepted transactions refresh
real post-SwiGLU activations before any further proposal. The final ragged-width
runtime is stored as a safetensors artifact with an explicit custom loader; it
is never represented as a stock Hugging Face checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.hf_qwen_compensated_neuron_fold import (
    CompensatedNeuronFoldWeights,
    Qwen3CompensatedFoldError,
    Qwen3CompensatedMLP,
    fit_compensated_neuron_pairs,
    install_qwen3_compensated_fold,
    synchronize_qwen3_intermediate_size,
)
from neural_forge.hf_qwen_multi_anchor_fold import (
    MultiAnchorNeuronFoldWeights,
    fit_multi_anchor_neuron_fold,
)
from neural_forge.hf_tools import evaluate_loaded_text_model


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import run_qwen_codebook_in_memory as codebook_runner  # noqa: E402


RUN_SCHEMA = "neural-forge-qwen3-compensated-neuron-fold-run-v2"
ACTIVATION_SCHEMA = "neural-forge-qwen3-ragged-activation-cache-v1"
CHECKPOINT_SCHEMA = "neural-forge-qwen3-ragged-safetensors-v1"
LOADER_CONTRACT = "run_qwen_compensated_neuron_fold.load_ragged_qwen_checkpoint"
IDENTITY_SCHEMA = "neural-forge-exact-parent-eval-identity-v1"
SPLIT_SCHEMA = "neural-forge-surgery-discovery-selection-certification-v1"
CERTIFICATION_SCHEMA = "neural-forge-zero-recovery-certification-v1"
GENERATION_HEALTH_CONTRACT = {
    "min_continuation_chars": 24,
    "min_unique_token_ratio": 0.25,
    "min_distinct_bigram_ratio": 0.50,
    "max_repeated_bigram_share": 0.35,
    "repeated_prompt_allowed": False,
    "empty_continuation_passes": False,
    "low_information_continuation_passes": False,
}


class QwenCompensatedRunnerError(RuntimeError):
    """The compensated-fold experiment cannot prove a required invariant."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenCompensatedRunnerError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenCompensatedRunnerError(
            f"{name} must be a positive integer."
        ) from exc
    if result <= 0 or result != value:
        raise QwenCompensatedRunnerError(f"{name} must be a positive integer.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenCompensatedRunnerError(
            f"{name} must be a non-negative integer."
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenCompensatedRunnerError(
            f"{name} must be a non-negative integer."
        ) from exc
    if result < 0 or result != value:
        raise QwenCompensatedRunnerError(
            f"{name} must be a non-negative integer."
        )
    return result


def _finite_fraction(value: Any, name: str, *, upper: float = 1.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenCompensatedRunnerError(f"{name} must be finite.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= upper:
        raise QwenCompensatedRunnerError(
            f"{name} must be in [0, {upper}]."
        )
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, chunk_bytes: int = 8 * 1024**2) -> str:
    tensor = value.detach()
    if tensor.device.type == "meta":
        raise QwenCompensatedRunnerError("Cannot fingerprint a meta tensor.")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    raw = tensor.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    step = max(1, int(chunk_bytes))
    for start in range(0, int(raw.numel()), step):
        piece = raw[start : start + step].to(device="cpu").contiguous()
        digest.update(piece.numpy().tobytes())
    return digest.hexdigest()


def _state_identity(model: nn.Module) -> dict[str, Any]:
    tensors = []
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, Tensor):
            raise QwenCompensatedRunnerError(f"State entry is not a tensor: {name}.")
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        )
    return {
        "tensor_count": len(tensors),
        "state_sha256": _canonical_sha256(tensors),
        "tensor_manifest_sha256": _canonical_sha256(tensors),
    }


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        raise QwenCompensatedRunnerError(
            "Exact parent binding requires tokenizer.get_vocab()."
        )
    vocabulary = getter()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise QwenCompensatedRunnerError("Tokenizer vocabulary is empty or invalid.")
    rows = sorted((str(token), int(index)) for token, index in vocabulary.items())
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": len(rows),
        "vocab_sha256": _canonical_sha256(rows),
        "special_tokens_map": dict(getattr(tokenizer, "special_tokens_map", {}) or {}),
        "added_vocab": dict(getattr(tokenizer, "get_added_vocab", lambda: {})() or {}),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
    }
    payload["tokenizer_sha256"] = _canonical_sha256(payload)
    return payload


def build_exact_parent_identity(
    model: nn.Module,
    tokenizer: Any,
    *,
    source_model: str,
    source_revision: str,
    source_tokenizer: str,
    eval_suite: Path,
) -> dict[str, Any]:
    revision = str(source_revision).strip()
    if not revision:
        raise QwenCompensatedRunnerError("An exact source revision is required.")
    config = model.config.to_dict()
    configured_commit = str(config.get("_commit_hash") or "").strip()
    if configured_commit and configured_commit != revision:
        raise QwenCompensatedRunnerError(
            "Loaded model commit differs from --source-revision: "
            f"{configured_commit}!={revision}."
        )
    source_path = Path(str(source_model)).expanduser()
    if source_path.exists() and source_path.parent.name == "snapshots":
        snapshot_revision = source_path.resolve().name
        if snapshot_revision != revision:
            raise QwenCompensatedRunnerError(
                "Snapshot directory differs from --source-revision: "
                f"{snapshot_revision}!={revision}."
            )
    import neural_forge.hf_tools as hf_tools_module
    from neural_forge.hf_tools import _load_eval_suite

    loaded_suite = _load_eval_suite(eval_suite)
    suite_payload = {
        name: loaded_suite[name]
        for name in (
            "text_evals",
            "choice_probes",
            "generation_prompts",
            "behavior_probes",
        )
    }
    state = _state_identity(model)
    tokenizer_report = _tokenizer_identity(tokenizer)
    identity = {
        "schema": IDENTITY_SCHEMA,
        "source_model": str(source_model),
        "source_revision": revision,
        "source_tokenizer": str(source_tokenizer),
        "state_sha256": state["state_sha256"],
        "state_tensor_count": state["tensor_count"],
        "config_sha256": _canonical_sha256(config),
        "tokenizer_sha256": tokenizer_report["tokenizer_sha256"],
        "eval_suite_file_sha256": _file_sha256(eval_suite),
        "eval_suite_content_sha256": _canonical_sha256(suite_payload),
        "evaluator_module_sha256": _file_sha256(Path(hf_tools_module.__file__)),
        "generation_health_contract": dict(GENERATION_HEALTH_CONTRACT),
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def derive_parent_identity_for_eval_suite(
    base_identity: Mapping[str, Any], eval_suite: Path
) -> dict[str, Any]:
    from neural_forge.hf_tools import _load_eval_suite

    loaded_suite = _load_eval_suite(eval_suite)
    suite_payload = {
        name: loaded_suite[name]
        for name in (
            "text_evals",
            "choice_probes",
            "generation_prompts",
            "behavior_probes",
        )
    }
    identity = dict(base_identity)
    identity["eval_suite_file_sha256"] = _file_sha256(eval_suite)
    identity["eval_suite_content_sha256"] = _canonical_sha256(suite_payload)
    identity.pop("identity_sha256", None)
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def validate_bound_parent_report(
    report: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    actual = report.get("identity_binding")
    if not isinstance(actual, Mapping):
        raise QwenCompensatedRunnerError(
            f"{label} lacks an exact identity binding; rebuild this cache once."
        )
    if dict(actual) != dict(expected_identity):
        differing = sorted(
            key
            for key in set(actual) | set(expected_identity)
            if actual.get(key) != expected_identity.get(key)
        )
        raise QwenCompensatedRunnerError(
            f"{label} identity differs from the loaded source: {differing}."
        )
    health = (report.get("generation_health") or {}).get("contract")
    if health != GENERATION_HEALTH_CONTRACT:
        raise QwenCompensatedRunnerError(
            f"{label} was not scored with the strict generation-health contract."
        )
    return {
        "identity_sha256": str(actual["identity_sha256"]),
        "state_sha256": str(actual["state_sha256"]),
        "tokenizer_sha256": str(actual["tokenizer_sha256"]),
        "eval_suite_file_sha256": str(actual["eval_suite_file_sha256"]),
        "evaluator_module_sha256": str(actual["evaluator_module_sha256"]),
        "cached_report_sha256": _canonical_sha256(report),
    }


def _current_process_rss_bytes() -> tuple[int, int]:
    status = Path("/proc/self/status")
    if status.is_file():
        values: dict[str, int] = {}
        for line in status.read_text(encoding="utf-8").splitlines():
            name, separator, trailing = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                values[name] = int(trailing.strip().split()[0]) * 1024
        return values.get("VmRSS", 0), values.get("VmHWM", values.get("VmRSS", 0))
    return 0, 0


@dataclass(slots=True)
class WorkflowResourceGuard:
    device: str
    max_host_rss_bytes: int
    max_cuda_allocated_bytes: int
    max_cuda_reserved_bytes: int
    max_wall_seconds: int
    phases: dict[str, dict[str, Any]]
    started: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_host_rss_bytes",
            "max_cuda_allocated_bytes",
            "max_cuda_reserved_bytes",
            "max_wall_seconds",
        ):
            _positive_int(getattr(self, name), name)
        self.started = time.monotonic()

    def check(self, phase: str) -> dict[str, Any]:
        elapsed = float(time.monotonic() - self.started)
        current_rss, peak_rss = _current_process_rss_bytes()
        values: dict[str, Any] = {
            "elapsed_seconds": elapsed,
            "host_rss_bytes": current_rss,
            "host_peak_rss_bytes": peak_rss,
            "max_host_rss_bytes": self.max_host_rss_bytes,
            "cuda_observed": False,
        }
        if elapsed > self.max_wall_seconds:
            raise QwenCompensatedRunnerError(
                f"Workflow wall-time cap exceeded during {phase}."
            )
        observed_host = max(current_rss, peak_rss)
        if observed_host and observed_host > self.max_host_rss_bytes:
            raise QwenCompensatedRunnerError(
                f"Host RSS cap exceeded during {phase}: "
                f"peak={observed_host}>{self.max_host_rss_bytes}."
            )
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            allocated = int(torch.cuda.max_memory_allocated(self.device))
            reserved = int(torch.cuda.max_memory_reserved(self.device))
            values.update(
                {
                    "cuda_observed": True,
                    "cuda_peak_allocated_bytes": allocated,
                    "cuda_peak_reserved_bytes": reserved,
                    "max_cuda_allocated_bytes": self.max_cuda_allocated_bytes,
                    "max_cuda_reserved_bytes": self.max_cuda_reserved_bytes,
                }
            )
            if allocated > self.max_cuda_allocated_bytes:
                raise QwenCompensatedRunnerError(
                    f"CUDA allocated cap exceeded during {phase}: "
                    f"{allocated}>{self.max_cuda_allocated_bytes}."
                )
            if reserved > self.max_cuda_reserved_bytes:
                raise QwenCompensatedRunnerError(
                    f"CUDA reserved cap exceeded during {phase}: "
                    f"{reserved}>{self.max_cuda_reserved_bytes}."
                )
        self.phases[phase] = values
        return values

    @contextlib.contextmanager
    def phase(self, name: str):
        self.check(f"{name}:start")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            yield
        finally:
            self.check(name)


def _assert_json_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise QwenCompensatedRunnerError(f"{path} contains a non-finite value.")


def _write_json(path: Path, value: Any) -> None:
    _assert_json_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_stage(output: Path, stage: str, status: str, **details: Any) -> None:
    _write_json(
        output / "run_state.json",
        {
            "schema": RUN_SCHEMA,
            "stage": stage,
            "status": status,
            "details": details,
            "final_report": str(output / "run_report.json"),
        },
    )


def _progress(event: str, **values: Any) -> None:
    fields = " ".join(f"{key}={values[key]}" for key in sorted(values))
    print(f"neural_forge_progress event={event} {fields}".rstrip(), flush=True)


def _qwen_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    try:
        return codebook_runner._qwen_layers(model)
    except Exception as exc:
        raise QwenCompensatedRunnerError(
            "Model is not a supported split-QKV Qwen3 causal LM."
        ) from exc


def _assert_model_safe(model: nn.Module, stage: str) -> None:
    try:
        codebook_runner._assert_zero_gradients(model, stage)
        codebook_runner._assert_model_finite(model, stage)
    except Exception as exc:
        raise QwenCompensatedRunnerError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ModelAccounting:
    unique_parameter_count: int
    unique_parameter_bytes: int
    unique_buffer_count: int
    unique_buffer_bytes: int
    parameter_tensor_count: int
    buffer_tensor_count: int
    layer_widths: tuple[int, ...]
    configured_intermediate_size: int
    ragged: bool

    @property
    def resident_tensor_bytes(self) -> int:
        return self.unique_parameter_bytes + self.unique_buffer_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "unique_parameter_count": self.unique_parameter_count,
            "unique_parameter_bytes": self.unique_parameter_bytes,
            "unique_buffer_count": self.unique_buffer_count,
            "unique_buffer_bytes": self.unique_buffer_bytes,
            "resident_tensor_bytes": self.resident_tensor_bytes,
            "parameter_tensor_count": self.parameter_tensor_count,
            "buffer_tensor_count": self.buffer_tensor_count,
            "layer_widths": list(self.layer_widths),
            "configured_intermediate_size": self.configured_intermediate_size,
            "ragged": self.ragged,
            "custom_ragged_layer_count": sum(
                width != self.configured_intermediate_size
                for width in self.layer_widths
            ),
        }


def model_accounting(model: nn.Module) -> ModelAccounting:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    widths = tuple(
        int(layer.mlp.gate_proj.weight.shape[0]) for layer in _qwen_layers(model)
    )
    if not widths or min(widths) <= 0:
        raise QwenCompensatedRunnerError("Qwen layer widths are invalid.")
    configured_intermediate_size = int(model.config.intermediate_size)
    return ModelAccounting(
        unique_parameter_count=sum(int(value.numel()) for value in parameters),
        unique_parameter_bytes=sum(
            int(value.numel()) * int(value.element_size()) for value in parameters
        ),
        unique_buffer_count=sum(int(value.numel()) for value in buffers),
        unique_buffer_bytes=sum(
            int(value.numel()) * int(value.element_size()) for value in buffers
        ),
        parameter_tensor_count=len(parameters),
        buffer_tensor_count=len(buffers),
        layer_widths=widths,
        configured_intermediate_size=configured_intermediate_size,
        ragged=len(set(widths)) > 1,
    )


def _same_storage_accounting(left: ModelAccounting, right: ModelAccounting) -> bool:
    return (
        left.unique_parameter_count == right.unique_parameter_count
        and left.unique_parameter_bytes == right.unique_parameter_bytes
        and left.unique_buffer_count == right.unique_buffer_count
        and left.unique_buffer_bytes == right.unique_buffer_bytes
        and left.parameter_tensor_count == right.parameter_tensor_count
        and left.buffer_tensor_count == right.buffer_tensor_count
        and left.layer_widths == right.layer_widths
        and left.ragged == right.ragged
    )


@dataclass(slots=True)
class RaggedActivationCache:
    neuron_activations: tuple[Tensor, ...]
    report: dict[str, Any]


def collect_bounded_ragged_qwen_activations(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    activation_rows: int,
    device: str,
    max_host_activation_bytes: int,
) -> RaggedActivationCache:
    """Capture bounded post-SwiGLU rows while permitting a width per layer."""

    layers = _qwen_layers(model)
    row_count = _positive_int(activation_rows, "activation_rows")
    host_cap = _positive_int(max_host_activation_bytes, "max_host_activation_bytes")
    encoded = codebook_runner._encode_calibration_prompts(
        tokenizer, prompts, _positive_int(max_length, "calibration_max_length")
    )
    local_positions, selected_global = codebook_runner._activation_row_plan(
        encoded, row_count
    )
    widths = tuple(int(layer.mlp.down_proj.in_features) for layer in layers)
    if min(widths) <= 0:
        raise QwenCompensatedRunnerError("Post-SwiGLU widths must be positive.")
    modules = [layer.mlp.down_proj for layer in layers]
    if len({id(module) for module in modules}) != len(modules):
        raise QwenCompensatedRunnerError("Ragged activation hooks overlap.")

    pieces: list[list[Tensor]] = [[] for _ in layers]
    calls = [0 for _ in layers]
    active_prompt = -1
    handles: list[Any] = []
    retained_piece_bytes = 0
    finalized_bytes = 0
    peak_bytes = 0
    backbone = getattr(model, "model", None)
    if not isinstance(backbone, nn.Module):
        raise QwenCompensatedRunnerError("Qwen model has no callable backbone.")

    def capture(layer_index: int) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def hook(_module: nn.Module, arguments: tuple[Any, ...]) -> None:
            nonlocal retained_piece_bytes, peak_bytes
            if not 0 <= active_prompt < len(encoded):
                raise QwenCompensatedRunnerError(
                    "Activation hook fired outside a calibration forward."
                )
            calls[layer_index] += 1
            if calls[layer_index] != active_prompt + 1:
                raise QwenCompensatedRunnerError(
                    f"Activation hook call mismatch at layer {layer_index}."
                )
            if len(arguments) != 1 or not isinstance(arguments[0], Tensor):
                raise QwenCompensatedRunnerError(
                    f"Invalid activation hook input at layer {layer_index}."
                )
            value = arguments[0]
            width = widths[layer_index]
            if value.ndim < 2 or int(value.shape[-1]) != width:
                raise QwenCompensatedRunnerError(
                    f"Activation width mismatch at layer {layer_index}."
                )
            flattened = value.detach().reshape(-1, width)
            expected = int(encoded[active_prompt].model_inputs["input_ids"].numel())
            if int(flattened.shape[0]) != expected:
                raise QwenCompensatedRunnerError(
                    f"Activation token-row mismatch at layer {layer_index}."
                )
            codebook_runner._assert_finite_tensor(
                flattened, f"ragged activation layer {layer_index}"
            )
            positions = local_positions[active_prompt]
            if int(positions.numel()):
                selected = flattened.index_select(0, positions.to(flattened.device))
                retained = selected.to("cpu", dtype=torch.float32).contiguous()
                pieces[layer_index].append(retained)
                retained_piece_bytes += int(retained.numel()) * retained.element_size()
                peak_bytes = max(
                    peak_bytes, finalized_bytes + retained_piece_bytes
                )

        return hook

    was_training = model.training
    model.eval()
    try:
        for layer_index, module in enumerate(modules):
            handles.append(module.register_forward_pre_hook(capture(layer_index)))
        with torch.inference_mode():
            for prompt_index, prompt in enumerate(encoded):
                active_prompt = prompt_index
                inputs = {
                    name: value.to(device) for name, value in prompt.model_inputs.items()
                }
                backbone(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                missing = [
                    layer_index
                    for layer_index, count in enumerate(calls)
                    if count != prompt_index + 1
                ]
                if missing:
                    raise QwenCompensatedRunnerError(
                        f"Activation hooks did not fire at layers {missing[:8]}."
                    )
                active_prompt = -1
    finally:
        active_prompt = -1
        for handle in handles:
            handle.remove()
        model.train(was_training)

    result: list[Tensor] = []
    for layer_index, layer_pieces in enumerate(pieces):
        if not layer_pieces:
            raise QwenCompensatedRunnerError(
                f"No activation rows were retained for layer {layer_index}."
            )
        rows = torch.cat(layer_pieces, dim=0)
        row_bytes = int(rows.numel()) * rows.element_size()
        peak_bytes = max(
            peak_bytes, finalized_bytes + retained_piece_bytes + row_bytes
        )
        if tuple(rows.shape) != (row_count, widths[layer_index]):
            raise QwenCompensatedRunnerError(
                f"Ragged activation shape mismatch at layer {layer_index}."
            )
        codebook_runner._assert_finite_tensor(
            rows, f"cached ragged activation layer {layer_index}"
        )
        piece_bytes = sum(
            int(piece.numel()) * piece.element_size() for piece in layer_pieces
        )
        retained_piece_bytes -= piece_bytes
        finalized_bytes += row_bytes
        layer_pieces.clear()
        result.append(rows)
    if retained_piece_bytes != 0:
        raise QwenCompensatedRunnerError("Activation memory accounting did not close.")
    if peak_bytes > host_cap:
        raise QwenCompensatedRunnerError(
            f"Peak host activation bytes exceed cap: {peak_bytes}>{host_cap}."
        )
    report = {
        "schema": ACTIVATION_SCHEMA,
        "method": "streaming-forward-hooks-global-even-token-rows",
        "logical_layers": len(layers),
        "layer_widths": list(widths),
        "ragged": len(set(widths)) > 1,
        "prompt_count": len(prompts),
        "requested_rows_per_layer": row_count,
        "selected_global_token_rows": selected_global,
        "selected_global_token_rows_sha256": _canonical_sha256(selected_global),
        "retained_host_activation_bytes": finalized_bytes,
        "peak_host_activation_bytes": peak_bytes,
        "max_host_activation_bytes": host_cap,
        "hook_calls_per_layer": calls,
        "post_swiglu_down_projection_inputs": True,
        "materialized_full_pairwise_matrix": False,
        "materialized_full_prompt_activation_cache": False,
        "bounded": True,
        "finite": True,
        "fitting_requires_grad": False,
    }
    return RaggedActivationCache(tuple(result), report)


@dataclass(slots=True)
class ActivationPair:
    calibration: tuple[Tensor, ...]
    validation: tuple[Tensor, ...]
    report: dict[str, Any]


def collect_activation_pair(
    model: nn.Module,
    tokenizer: Any,
    calibration_prompts: Sequence[str],
    validation_prompts: Sequence[str],
    *,
    max_length: int,
    activation_rows: int,
    device: str,
    max_host_activation_bytes: int,
) -> ActivationPair:
    calibration = collect_bounded_ragged_qwen_activations(
        model,
        tokenizer,
        calibration_prompts,
        max_length=max_length,
        activation_rows=activation_rows,
        device=device,
        max_host_activation_bytes=max_host_activation_bytes,
    )
    validation = collect_bounded_ragged_qwen_activations(
        model,
        tokenizer,
        validation_prompts,
        max_length=max_length,
        activation_rows=activation_rows,
        device=device,
        max_host_activation_bytes=max_host_activation_bytes,
    )
    combined_peak = (
        calibration.report["retained_host_activation_bytes"]
        + validation.report["peak_host_activation_bytes"]
    )
    if combined_peak > max_host_activation_bytes:
        raise QwenCompensatedRunnerError(
            "Combined calibration and held-out activation caches exceed the host cap: "
            f"{combined_peak}>{max_host_activation_bytes}."
        )
    return ActivationPair(
        calibration.neuron_activations,
        validation.neuron_activations,
        {
            "calibration": calibration.report,
            "validation": validation.report,
            "combined_retained_host_activation_bytes": (
                calibration.report["retained_host_activation_bytes"]
                + validation.report["retained_host_activation_bytes"]
            ),
            "combined_peak_host_activation_bytes": combined_peak,
            "max_host_activation_bytes": max_host_activation_bytes,
            "bounded": True,
            "split_roles": {
                "pair_discovery": "calibration-only",
                "pair_model_selection": "validation-only",
                "transaction_acceptance": "external-gate-not-present",
            },
        },
    )


def measure_high_damage_anchors(
    activations: Tensor,
    down_weight: Tensor,
    *,
    fraction: float,
    minimum: int,
) -> tuple[tuple[int, ...], dict[str, Any], Tensor]:
    """Rank exact local output-contribution energy and protect its upper tail."""

    if activations.ndim != 2 or down_weight.ndim != 2:
        raise QwenCompensatedRunnerError("Anchor tensors must be rank two.")
    width = int(activations.shape[1])
    if int(down_weight.shape[1]) != width or width < 3:
        raise QwenCompensatedRunnerError("Anchor width is invalid.")
    anchor_fraction = _finite_fraction(fraction, "anchor_fraction")
    minimum_count = _nonnegative_int(minimum, "minimum_anchor_count")
    count = max(minimum_count, math.ceil(width * anchor_fraction))
    count = min(count, width - 2)
    activation_rms = torch.sqrt(
        activations.detach().to("cpu", dtype=torch.float64).square().mean(dim=0)
    )
    consumer_norm = torch.linalg.vector_norm(
        down_weight.detach().to("cpu", dtype=torch.float64), dim=0
    )
    damage = activation_rms * consumer_norm
    if not bool(torch.isfinite(damage).all()):
        raise QwenCompensatedRunnerError("Anchor damage contains NaN or infinity.")
    if count:
        ranked = torch.argsort(damage, descending=True, stable=True)[:count].tolist()
    else:
        ranked = []
    anchors = tuple(sorted(int(index) for index in ranked))
    report = {
        "method": "post-swiglu-rms-times-down-column-l2",
        "width": width,
        "protected_count": len(anchors),
        "protected_indices": list(anchors),
        "protected_indices_sha256": _canonical_sha256(anchors),
        "damage_sha256": _canonical_sha256([float(value) for value in damage]),
        "top_damage": [
            {"index": int(index), "damage": float(damage[index])}
            for index in ranked[:64]
        ],
        "exact_local_contribution_measure": True,
    }
    return anchors, report, damage


def verify_exact_anchors(
    original_mlp: nn.Module,
    installed_mlp: nn.Module,
    fold: CompensatedNeuronFoldWeights,
    anchors: Sequence[int],
) -> dict[str, Any]:
    checks = 0
    for source_index in anchors:
        if source_index not in fold.kept_source_indices:
            raise QwenCompensatedRunnerError("A protected anchor was removed.")
        target_index = fold.kept_source_indices.index(source_index)
        for original, installed in (
            (original_mlp.gate_proj.weight[source_index], installed_mlp.gate_proj.weight[target_index]),
            (original_mlp.up_proj.weight[source_index], installed_mlp.up_proj.weight[target_index]),
            (original_mlp.down_proj.weight[:, source_index], installed_mlp.down_proj.weight[:, target_index]),
        ):
            if not torch.equal(original.detach(), installed.detach()):
                raise QwenCompensatedRunnerError(
                    f"Protected anchor {source_index} changed during installation."
                )
            checks += 1
    return {
        "protected_anchor_count": len(tuple(anchors)),
        "exact_tensor_checks": checks,
        "all_exact": True,
    }


@dataclass(frozen=True, slots=True)
class GateThresholds:
    max_behavior_losses: int = 0
    max_choice_losses: int = 0
    max_generation_losses: int = 0

    def __post_init__(self) -> None:
        for name in (
            "max_behavior_losses",
            "max_choice_losses",
            "max_generation_losses",
        ):
            _nonnegative_int(getattr(self, name), name)

    def as_dict(self) -> dict[str, int]:
        return {
            "max_behavior_losses": self.max_behavior_losses,
            "max_choice_losses": self.max_choice_losses,
            "max_generation_losses": self.max_generation_losses,
        }


def _item_map(items: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    return {
        str(item["id"]): dict(item)
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }


def _domain_gate(
    parent_items: Mapping[str, Mapping[str, Any]],
    candidate_items: Mapping[str, Mapping[str, Any]],
    expected_ids: Sequence[str],
    *,
    pass_key: str,
    max_losses: int,
) -> dict[str, Any]:
    expected = [str(value) for value in expected_ids]
    if len(expected) != len(set(expected)) or not expected:
        raise QwenCompensatedRunnerError("Gate IDs must be unique and non-empty.")
    if set(parent_items) != set(expected) or set(candidate_items) != set(expected):
        raise QwenCompensatedRunnerError("Gate report IDs differ from the locked slice.")
    losses: list[str] = []
    gains: list[str] = []
    parent_correct = 0
    candidate_correct = 0
    for item_id in expected:
        parent_passed = bool(parent_items[item_id].get(pass_key, False))
        candidate_passed = bool(candidate_items[item_id].get(pass_key, False))
        parent_correct += int(parent_passed)
        candidate_correct += int(candidate_passed)
        if parent_passed and not candidate_passed:
            losses.append(item_id)
        elif candidate_passed and not parent_passed:
            gains.append(item_id)
    return {
        "total": len(expected),
        "parent_correct": parent_correct,
        "candidate_correct": candidate_correct,
        "retained_parent_passes": parent_correct - len(losses),
        "loss_count": len(losses),
        "gain_count": len(gains),
        "loss_ids": losses[:128],
        "gain_ids": gains[:128],
        "max_allowed_losses": max_losses,
        "passed": len(losses) <= max_losses,
    }


def score_cached_parent_gate(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    behavior_ids: Sequence[str],
    choice_ids: Sequence[str],
    generation_ids: Sequence[str],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    parent_behavior = _item_map((parent.get("behavior_suite") or {}).get("items"))
    child_behavior = _item_map((candidate.get("behavior_suite") or {}).get("items"))
    parent_choice = _item_map((parent.get("choice_accuracy") or {}).get("items"))
    child_choice = _item_map((candidate.get("choice_accuracy") or {}).get("items"))
    parent_generation = _item_map(parent.get("generation_diagnostics"))
    child_generation = _item_map(candidate.get("generation_diagnostics"))
    domains = {
        "behavior": _domain_gate(
            {key: parent_behavior[key] for key in behavior_ids},
            child_behavior,
            behavior_ids,
            pass_key="passed",
            max_losses=thresholds.max_behavior_losses,
        ),
        "choice": _domain_gate(
            {key: parent_choice[key] for key in choice_ids},
            child_choice,
            choice_ids,
            pass_key="correct",
            max_losses=thresholds.max_choice_losses,
        ),
        "generation": _domain_gate(
            {key: parent_generation[key] for key in generation_ids},
            child_generation,
            generation_ids,
            pass_key="passed",
            max_losses=thresholds.max_generation_losses,
        ),
    }
    passed = all(domain["passed"] for domain in domains.values())
    return {
        "method": "cached-parent-relative-behavior-choice-generation-retention",
        "domains": domains,
        "thresholds": thresholds.as_dict(),
        "perplexity_used_for_acceptance": False,
        "parent_forward_passes": 0,
        "passed": passed,
    }


def _category(item: Mapping[str, Any]) -> str:
    explicit = item.get("category")
    if explicit:
        return str(explicit).strip().lower().replace("-", "_")
    text = f"{item.get('id', '')} {item.get('prompt', '')}".lower()
    if any(value in text for value in ("python", "code", "function")):
        return "code"
    if any(value in text for value in ("math", "addition", "multiply")):
        return "math"
    if any(value in text for value in ("tokenizer", "vocab")):
        return "tokenizer"
    if any(value in text for value in ("merge", "prun", "surgery", "compress")):
        return "model_surgery"
    return "other"


def _stratified_items(items: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    requested = _positive_int(count, "gate slice count")
    if requested > len(items):
        raise QwenCompensatedRunnerError(
            f"Gate requests {requested} rows from {len(items)} available rows."
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(_category(item), []).append(dict(item))
    selected: list[dict[str, Any]] = []
    cursor = 0
    keys = sorted(groups)
    while len(selected) < requested:
        progressed = False
        for key in keys:
            if cursor < len(groups[key]) and len(selected) < requested:
                selected.append(groups[key][cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    if len(selected) != requested:
        raise QwenCompensatedRunnerError("Could not fill the stratified gate slice.")
    return selected


def build_gate_suite(
    eval_suite: Path | str | Mapping[str, Any],
    cached_parent: Mapping[str, Any],
    *,
    text_count: int,
    behavior_count: int,
    choice_count: int,
    generation_count: int,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]], dict[str, Any]]:
    from neural_forge.hf_tools import _load_eval_suite

    suite = _load_eval_suite(eval_suite)
    selected = {
        "text_evals": list(suite["text_evals"][: _positive_int(text_count, "gate_text_count")]),
        "behavior_probes": _stratified_items(
            suite["behavior_probes"], behavior_count
        ),
        "choice_probes": _stratified_items(suite["choice_probes"], choice_count),
        "generation_prompts": _stratified_items(
            suite["generation_prompts"], generation_count
        ),
    }
    ids = {
        "behavior": tuple(str(item["id"]) for item in selected["behavior_probes"]),
        "choice": tuple(str(item["id"]) for item in selected["choice_probes"]),
        "generation": tuple(
            str(item["id"]) for item in selected["generation_prompts"]
        ),
    }
    parent_sets = {
        "behavior": set(_item_map((cached_parent.get("behavior_suite") or {}).get("items"))),
        "choice": set(_item_map((cached_parent.get("choice_accuracy") or {}).get("items"))),
        "generation": set(_item_map(cached_parent.get("generation_diagnostics"))),
    }
    for domain, expected in ids.items():
        if not set(expected).issubset(parent_sets[domain]):
            raise QwenCompensatedRunnerError(
                f"Cached parent is missing {domain} gate IDs."
            )
    inline = {
        "text_evals": selected["text_evals"],
        "choice_probes": selected["choice_probes"],
        "generation_prompts": selected["generation_prompts"],
        "behavior_probes": selected["behavior_probes"],
    }
    report = {
        "selection": "deterministic-category-round-robin",
        "counts": {name: len(values) for name, values in inline.items()},
        "ids": {name: list(values) for name, values in ids.items()},
        "suite_sha256": _canonical_sha256(inline),
        "cached_parent_forward_passes": 0,
    }
    return inline, ids, report


def build_three_split_provenance(
    tokenizer: Any,
    calibration_prompts: Sequence[str],
    validation_prompts: Sequence[str],
    selection_gate_suite: Mapping[str, Any],
    certification_suite: Mapping[str, Any],
    *,
    calibration_source: Path,
    selection_source: Path,
    cached_parent_selection_source: Path,
    certification_source: Path,
    cached_parent_certification_source: Path,
    max_length: int,
) -> dict[str, Any]:
    """Bind discovery, adaptive selection, and one-shot certification provenance."""

    calibration_tokens = {
        codebook_runner._token_ids(tokenizer, prompt, max_length=max_length)
        for prompt in calibration_prompts
    }
    validation_tokens = {
        codebook_runner._token_ids(tokenizer, prompt, max_length=max_length)
        for prompt in validation_prompts
    }
    def suite_texts(suite: Mapping[str, Any]) -> list[str]:
        values = [str(value) for value in suite["text_evals"]]
        for collection in (
            "choice_probes",
            "generation_prompts",
            "behavior_probes",
        ):
            values.extend(str(item["prompt"]) for item in suite[collection])
        return values

    selection_texts = suite_texts(selection_gate_suite)
    certification_texts = suite_texts(certification_suite)
    selection_tokens = {
        codebook_runner._token_ids(tokenizer, prompt, max_length=max_length)
        for prompt in selection_texts
    }
    certification_tokens = {
        codebook_runner._token_ids(tokenizer, prompt, max_length=max_length)
        for prompt in certification_texts
    }
    overlaps = {
        "calibration_validation": len(calibration_tokens & validation_tokens),
        "calibration_selection": len(calibration_tokens & selection_tokens),
        "validation_selection": len(validation_tokens & selection_tokens),
        "calibration_certification": len(calibration_tokens & certification_tokens),
        "validation_certification": len(validation_tokens & certification_tokens),
        "selection_certification": len(selection_tokens & certification_tokens),
    }
    if any(overlaps.values()):
        raise QwenCompensatedRunnerError(
            f"Discovery/selection/certification overlap is nonzero: {overlaps}."
        )
    payload: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "calibration": {
            "role": "pair-discovery-and-closed-form-fit-only",
            "source": str(calibration_source.resolve()),
            "source_sha256": _file_sha256(calibration_source),
            "prompt_count": len(calibration_prompts),
            "prompt_sha256": _canonical_sha256(list(calibration_prompts)),
            "token_path_count": len(calibration_tokens),
            "token_paths_sha256": _canonical_sha256(sorted(calibration_tokens)),
        },
        "validation": {
            "role": "candidate-model-selection-only",
            "source": str(calibration_source.resolve()),
            "source_sha256": _file_sha256(calibration_source),
            "prompt_count": len(validation_prompts),
            "prompt_sha256": _canonical_sha256(list(validation_prompts)),
            "token_path_count": len(validation_tokens),
            "token_paths_sha256": _canonical_sha256(sorted(validation_tokens)),
        },
        "adaptive_selection_gate": {
            "role": "cached-parent-relative-retry-and-transaction-selection",
            "source": str(selection_source.resolve()),
            "source_sha256": _file_sha256(selection_source),
            "cached_parent_source": str(cached_parent_selection_source.resolve()),
            "cached_parent_sha256": _file_sha256(cached_parent_selection_source),
            "suite_sha256": _canonical_sha256(selection_gate_suite),
            "prompt_count": len(selection_texts),
            "token_path_count": len(selection_tokens),
            "token_paths_sha256": _canonical_sha256(sorted(selection_tokens)),
            "adaptively_reused": True,
        },
        "untouched_certification": {
            "role": "one-shot-parent-relative-final-certification-only",
            "source": str(certification_source.resolve()),
            "source_sha256": _file_sha256(certification_source),
            "cached_parent_source": str(
                cached_parent_certification_source.resolve()
            ),
            "cached_parent_sha256": _file_sha256(
                cached_parent_certification_source
            ),
            "suite_sha256": _canonical_sha256(certification_suite),
            "prompt_count": len(certification_texts),
            "token_path_count": len(certification_tokens),
            "token_paths_sha256": _canonical_sha256(sorted(certification_tokens)),
            "candidate_evaluation_budget": 1,
            "visible_before_surgery_complete": False,
        },
        "pairwise_token_path_overlaps": overlaps,
        "tokenization_max_length": max_length,
        "fit_visible_splits": ["calibration", "validation"],
        "selection_gate_visible_to_fit": False,
        "certification_scores_or_parent_outputs_visible_to_fit_or_selection": False,
        "certification_prompt_text_used_for_decontamination_only": True,
        "pair_discovery_uses_validation": False,
        "pair_discovery_uses_selection_gate": False,
        "pair_discovery_uses_certification": False,
        "all_provenance_bound": True,
        "all_token_paths_disjoint": True,
    }
    payload["provenance_sha256"] = _canonical_sha256(payload)
    return payload


class CachedParentGateEvaluator:
    """Adaptively score candidates on selection data; never touch certification."""

    def __init__(
        self,
        *,
        parent_report: Mapping[str, Any],
        gate_suite: Mapping[str, Any],
        gate_ids: Mapping[str, Sequence[str]],
        thresholds: GateThresholds,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
        output_directory: Path,
        split_provenance_sha256: str,
        evaluator: Callable[..., dict[str, Any]] = evaluate_loaded_text_model,
    ) -> None:
        self.parent_report = parent_report
        self.gate_suite = gate_suite
        self.gate_ids = gate_ids
        self.thresholds = thresholds
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.output_directory = output_directory
        self.split_provenance_sha256 = str(split_provenance_sha256)
        if len(self.split_provenance_sha256) != 64:
            raise QwenCompensatedRunnerError(
                "External gate requires a bound three-split provenance hash."
            )
        self.evaluator = evaluator
        parent_health = (parent_report.get("generation_health") or {}).get("contract")
        if parent_health != GENERATION_HEALTH_CONTRACT:
            raise QwenCompensatedRunnerError(
                "Selection parent cache lacks strict generation-health evidence."
            )
        self.parent_evaluation_calls = 0
        self.candidate_evaluation_calls = 0

    def __call__(
        self, model: nn.Module, attempt: int, proposal: "LayerProposal"
    ) -> tuple[bool, dict[str, Any]]:
        self.candidate_evaluation_calls += 1
        work = self.output_directory / f".gate-{attempt:06d}.work.json"
        final = self.output_directory / f"gate-{attempt:06d}.json"
        try:
            candidate = self.evaluator(
                model,
                self.tokenizer,
                work,
                model_label=(
                    f"compensated-fold:attempt={attempt}:layer={proposal.layer_index}"
                ),
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                eval_suite=dict(self.gate_suite),
                max_text_evals=len(self.gate_suite["text_evals"]),
                max_choice_probes=len(self.gate_suite["choice_probes"]),
                max_generation_prompts=len(self.gate_suite["generation_prompts"]),
                max_behavior_probes=len(self.gate_suite["behavior_probes"]),
                min_continuation_chars=GENERATION_HEALTH_CONTRACT[
                    "min_continuation_chars"
                ],
                min_unique_token_ratio=GENERATION_HEALTH_CONTRACT[
                    "min_unique_token_ratio"
                ],
            )
            health = (candidate.get("generation_health") or {}).get("contract")
            if health != GENERATION_HEALTH_CONTRACT:
                raise QwenCompensatedRunnerError(
                    "Selection candidate lacks strict generation-health evidence."
                )
            verdict = score_cached_parent_gate(
                self.parent_report,
                candidate,
                behavior_ids=self.gate_ids["behavior"],
                choice_ids=self.gate_ids["choice"],
                generation_ids=self.gate_ids["generation"],
                thresholds=self.thresholds,
            )
            payload = {
                "attempt": attempt,
                "layer_index": proposal.layer_index,
                "candidate": candidate,
                "verdict": verdict,
                "parent_evaluation_calls": self.parent_evaluation_calls,
                "candidate_evaluation_calls": self.candidate_evaluation_calls,
                "split_provenance_sha256": self.split_provenance_sha256,
                "selection_data_adaptively_reused": True,
                "certification_candidate_evaluation_calls": 0,
            }
            _write_json(final, payload)
            compact = {
                "attempt": attempt,
                "layer_index": proposal.layer_index,
                "verdict": verdict,
                "artifact": str(final),
                "artifact_sha256": _file_sha256(final),
                "parent_evaluation_calls": 0,
                "candidate_evaluation_calls": self.candidate_evaluation_calls,
                "split_provenance_sha256": self.split_provenance_sha256,
                "selection_data_adaptively_reused": True,
                "certification_candidate_evaluation_calls": 0,
            }
            return bool(verdict["passed"]), compact
        finally:
            work.unlink(missing_ok=True)


@dataclass(slots=True)
class LayerProposal:
    layer_index: int
    fold: CompensatedNeuronFoldWeights | MultiAnchorNeuronFoldWeights
    anchors: tuple[int, ...]
    temporary_protected: tuple[int, ...]
    anchor_report: dict[str, Any]
    prefilter_score: float
    score: float
    mean_score: float
    strategy_evidence: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        diagnostics = self.fold.diagnostics
        pair_merges = all(
            hasattr(item, "left") and hasattr(item, "right")
            for item in self.fold.merges
        )
        return {
            "layer_index": self.layer_index,
            "source_width": self.fold.source_width,
            "target_width": self.fold.target_width,
            "parameter_savings": self.fold.parameter_savings,
            "calibration_prefilter_score": self.prefilter_score,
            "score": self.score,
            "mean_score": self.mean_score,
            "anchors": list(self.anchors),
            "temporary_protected": list(self.temporary_protected),
            "merges": [item.as_dict() for item in self.fold.merges],
            "fold_strategy": "pair" if pair_merges else "multi-anchor",
            "adjacent_pair_count": (
                sum(abs(item.left - item.right) <= 1 for item in self.fold.merges)
                if pair_merges
                else 0
            ),
            "nonadjacent_pair_count": (
                sum(abs(item.left - item.right) > 1 for item in self.fold.merges)
                if pair_merges
                else 0
            ),
            "multi_anchor_merge_count": 0 if pair_merges else len(self.fold.merges),
            "multi_anchor_edge_count": (
                0
                if pair_merges
                else sum(len(item.anchors) for item in self.fold.merges)
            ),
            "arbitrary_pairing_enabled": pair_merges,
            "arbitrary_nonlocal_folding_enabled": True,
            "fold_sha256": self.fold.fold_sha256,
            "anchor_report": self.anchor_report,
            "strategy_evidence": self.strategy_evidence,
            "internal_split_proof": {
                "calibration_activation_sha256": (
                    diagnostics.calibration_activation_sha256
                ),
                "validation_activation_sha256": (
                    diagnostics.validation_activation_sha256
                ),
                "activation_hashes_distinct": (
                    diagnostics.calibration_activation_sha256
                    != diagnostics.validation_activation_sha256
                ),
                "pair_discovery": "calibration-only",
                "model_selection": "validation-only",
                "external_transaction_gate_visible_to_fit": False,
            },
        }


def _initialize_persistent_anchors(
    model: nn.Module,
    activations: ActivationPair,
    config: "FoldLoopConfig",
) -> tuple[
    dict[int, tuple[int, ...]],
    dict[int, dict[str, Any]],
    dict[int, list[tuple[int, ...]]],
]:
    anchors: dict[int, tuple[int, ...]] = {}
    reports: dict[int, dict[str, Any]] = {}
    lineage: dict[int, list[tuple[int, ...]]] = {}
    for layer_index, layer in enumerate(_qwen_layers(model)):
        width = int(layer.mlp.gate_proj.weight.shape[0])
        selected, report, damage = measure_high_damage_anchors(
            activations.calibration[layer_index],
            layer.mlp.down_proj.weight,
            fraction=config.anchor_fraction,
            minimum=config.minimum_anchor_count,
        )
        report = {
            **report,
            "selection_epoch": 0,
            "immutable_after_selection": True,
            "source_neuron_ids": list(selected),
            "measurement": "exact-single-neuron-layer-output-ablation-frobenius",
            "measurement_split": "calibration-only",
        }
        anchors[layer_index] = selected
        reports[layer_index] = report
        lineage[layer_index] = [(index,) for index in range(width)]
        del damage
    return anchors, reports, lineage


def _current_anchor_indices(
    source_anchor_ids: Sequence[int], lineage: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    anchor_set = {int(value) for value in source_anchor_ids}
    current = tuple(
        index
        for index, source_ids in enumerate(lineage)
        if anchor_set.intersection(int(value) for value in source_ids)
    )
    if len(current) != len(anchor_set):
        raise QwenCompensatedRunnerError(
            "Persistent anchor lineage is missing or merged with another source neuron."
        )
    return current


def _lineage_after_fold(
    lineage: Sequence[Sequence[int]],
    fold: CompensatedNeuronFoldWeights | MultiAnchorNeuronFoldWeights,
) -> list[tuple[int, ...]]:
    if fold.merges and not hasattr(fold.merges[0], "representative"):
        result = [
            tuple(sorted(int(value) for value in lineage[source_index]))
            for source_index in fold.kept_source_indices
        ]
        if len(result) != fold.target_width:
            raise QwenCompensatedRunnerError(
                "Multi-anchor survivor lineage width is inconsistent."
            )
        return result
    merged_into: dict[int, int] = {
        int(item.representative): int(item.dropped) for item in fold.merges
    }
    result: list[tuple[int, ...]] = []
    for source_index in fold.kept_source_indices:
        values = list(int(value) for value in lineage[source_index])
        dropped = merged_into.get(int(source_index))
        if dropped is not None:
            values.extend(int(value) for value in lineage[dropped])
        result.append(tuple(sorted(set(values))))
    flattened = [value for group in result for value in group]
    expected = [value for group in lineage for value in group]
    if sorted(flattened) != sorted(expected) or len(flattened) != len(set(flattened)):
        raise QwenCompensatedRunnerError("Neuron lineage did not remain a partition.")
    return result


@dataclass(frozen=True, slots=True)
class FoldLoopConfig:
    device: str = "cpu"
    calibration_max_length: int = 160
    activation_rows: int = 64
    max_host_activation_bytes: int = 8 * 1024**3
    anchor_fraction: float = 0.02
    minimum_anchor_count: int = 8
    projection_count: int = 16
    neighbor_window: int = 8
    max_pair_candidates: int = 131_072
    max_generated_pairs: int = 524_288
    score_chunk_size: int = 1_024
    proposal_layer_limit: int = 6
    prefilter_columns: int = 256
    fold_strategy: str = "pair"
    calibration_provenance_id: str = ""
    validation_provenance_id: str = ""
    multi_anchor_candidate_pool_size: int = 512
    multi_anchor_max_anchors: int = 4
    multi_anchor_similarity_chunk_size: int = 64
    multi_anchor_max_similarity_products: int = 16_000_000
    multi_anchor_max_projection_fits: int = 131_072
    multi_anchor_ridge: float = 1e-6
    multi_anchor_max_coefficient_l1: float = 16.0
    merges_per_transaction: int = 1
    max_candidate_attempts: int = 32
    max_transactions: int = 10_000
    full_milestone_interval: int = 32
    max_history_records: int = 256
    max_history_json_bytes: int = 64 * 1024**2
    excluded_layers: tuple[int, ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "calibration_max_length",
            "activation_rows",
            "max_host_activation_bytes",
            "projection_count",
            "neighbor_window",
            "max_pair_candidates",
            "max_generated_pairs",
            "score_chunk_size",
            "proposal_layer_limit",
            "prefilter_columns",
            "multi_anchor_candidate_pool_size",
            "multi_anchor_max_anchors",
            "multi_anchor_similarity_chunk_size",
            "multi_anchor_max_similarity_products",
            "multi_anchor_max_projection_fits",
            "merges_per_transaction",
            "max_candidate_attempts",
            "max_transactions",
            "full_milestone_interval",
            "max_history_records",
            "max_history_json_bytes",
        ):
            _positive_int(getattr(self, name), name)
        _nonnegative_int(self.minimum_anchor_count, "minimum_anchor_count")
        _nonnegative_int(self.seed, "seed")
        _finite_fraction(self.anchor_fraction, "anchor_fraction")
        if self.fold_strategy not in {"pair", "multi-anchor"}:
            raise QwenCompensatedRunnerError(
                "fold_strategy must be 'pair' or 'multi-anchor'."
            )
        if self.fold_strategy == "multi-anchor" and (
            not self.calibration_provenance_id.strip()
            or not self.validation_provenance_id.strip()
            or self.calibration_provenance_id == self.validation_provenance_id
        ):
            raise QwenCompensatedRunnerError(
                "Multi-anchor folding requires distinct exact calibration and validation provenance IDs."
            )
        for name in ("multi_anchor_ridge", "multi_anchor_max_coefficient_l1"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise QwenCompensatedRunnerError(f"{name} must be finite and positive.")
        hard_limits = {
            "projection_count": 256,
            "neighbor_window": 256,
            "max_pair_candidates": 2_000_000,
            "max_generated_pairs": 4_000_000,
            "score_chunk_size": 65_536,
            "proposal_layer_limit": 256,
            "merges_per_transaction": 4_096,
            "max_transactions": 4_096,
            "max_history_records": 4_096,
            "max_history_json_bytes": 1024**3,
            "multi_anchor_candidate_pool_size": 65_536,
            "multi_anchor_max_anchors": 8,
            "multi_anchor_similarity_chunk_size": 1_024,
            "multi_anchor_max_similarity_products": 64_000_000,
            "multi_anchor_max_projection_fits": 1_000_000,
        }
        for name, maximum in hard_limits.items():
            if int(getattr(self, name)) > maximum:
                raise QwenCompensatedRunnerError(
                    f"{name} exceeds the immutable production ceiling {maximum}."
                )
        if self.max_pair_candidates > self.max_generated_pairs:
            raise QwenCompensatedRunnerError(
                "max_pair_candidates cannot exceed max_generated_pairs."
            )
        if len(set(self.excluded_layers)) != len(self.excluded_layers) or any(
            index < 0 for index in self.excluded_layers
        ):
            raise QwenCompensatedRunnerError("excluded_layers are invalid.")


def _target_reached(
    accounting: ModelAccounting,
    target_parameters: int | None,
    target_bytes: int | None,
    target_resident_bytes: int | None = None,
) -> bool:
    checks: list[bool] = []
    if target_parameters is not None:
        checks.append(accounting.unique_parameter_count <= target_parameters)
    if target_bytes is not None:
        checks.append(accounting.unique_parameter_bytes <= target_bytes)
    if target_resident_bytes is not None:
        checks.append(accounting.resident_tensor_bytes <= target_resident_bytes)
    if not checks:
        raise QwenCompensatedRunnerError("A parameter or byte target is required.")
    return all(checks)


def _needed_drops(
    accounting: ModelAccounting,
    *,
    hidden_size: int,
    element_size: int,
    target_parameters: int | None,
    target_bytes: int | None,
    target_resident_bytes: int | None,
    maximum: int,
) -> int:
    needed = 1
    per_drop_parameters = 3 * hidden_size
    per_drop_bytes = per_drop_parameters * element_size
    if target_parameters is not None:
        gap = max(0, accounting.unique_parameter_count - target_parameters)
        needed = max(needed, math.ceil(gap / per_drop_parameters))
    if target_bytes is not None:
        gap = max(0, accounting.unique_parameter_bytes - target_bytes)
        needed = max(needed, math.ceil(gap / per_drop_bytes))
    if target_resident_bytes is not None:
        gap = max(0, accounting.resident_tensor_bytes - target_resident_bytes)
        needed = max(needed, math.ceil(gap / per_drop_bytes))
    return min(needed, maximum)


def _fit_layer_proposal(
    model: nn.Module,
    layer_index: int,
    calibration: Tensor,
    validation: Tensor,
    *,
    drops: int,
    blocked_indices: Sequence[int],
    persistent_anchor_indices: Sequence[int],
    persistent_anchor_report: Mapping[str, Any],
    config: FoldLoopConfig,
    epoch: int,
    prefilter_score: float,
) -> LayerProposal:
    layer = _qwen_layers(model)[layer_index]
    mlp = layer.mlp
    width = int(mlp.gate_proj.weight.shape[0])
    requested_drops = min(_positive_int(drops, "drops"), width // 2)
    if requested_drops <= 0 or width - requested_drops <= 0:
        raise QwenCompensatedRunnerError("Layer cannot be reduced further.")
    if tuple(calibration.shape)[1:] != (width,) or tuple(validation.shape)[1:] != (width,):
        raise QwenCompensatedRunnerError("Proposal activation width is stale.")
    anchors = tuple(sorted(int(value) for value in persistent_anchor_indices))
    anchor_report = dict(persistent_anchor_report)
    permanent = set(anchors) | {int(value) for value in blocked_indices}
    protected = tuple(sorted(permanent))
    if width - len(protected) < 2 * requested_drops:
        raise QwenCompensatedRunnerError(
            f"Layer {layer_index} has too few unprotected neurons."
        )
    try:
        strategy_evidence: dict[str, Any] = {
            "strategy": config.fold_strategy,
            "certification_contract_unchanged": True,
        }
        if config.fold_strategy == "pair":
            fold = fit_compensated_neuron_pairs(
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
                mlp.down_proj.weight,
                calibration,
                target_width=width - requested_drops,
                validation_activations=validation,
                protected_indices=protected,
                projection_count=config.projection_count,
                neighbor_window=config.neighbor_window,
                max_pair_candidates=config.max_pair_candidates,
                max_generated_pairs=config.max_generated_pairs,
                score_chunk_size=config.score_chunk_size,
                seed=config.seed + epoch * 1_000_003 + layer_index * 1009,
            )
        else:
            fold = fit_multi_anchor_neuron_fold(
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
                mlp.down_proj.weight,
                calibration,
                target_width=width - requested_drops,
                validation_activations=validation,
                protected_indices=protected,
                calibration_provenance=config.calibration_provenance_id,
                validation_provenance=config.validation_provenance_id,
                candidate_pool_size=config.multi_anchor_candidate_pool_size,
                max_anchors=config.multi_anchor_max_anchors,
                similarity_chunk_size=config.multi_anchor_similarity_chunk_size,
                max_similarity_products=(
                    config.multi_anchor_max_similarity_products
                ),
                max_projection_fits=config.multi_anchor_max_projection_fits,
                ridge=config.multi_anchor_ridge,
                max_coefficient_l1=config.multi_anchor_max_coefficient_l1,
            )
            tiny = torch.finfo(torch.float64).tiny
            checked_edges = 0
            per_merge_metric_mismatches = 0
            calibration64 = calibration.to(device="cpu", dtype=torch.float64)
            validation64 = validation.to(device="cpu", dtype=torch.float64)
            for merge in fold.merges:
                index = torch.tensor(merge.anchors, dtype=torch.long)
                coefficients = torch.tensor(
                    merge.coefficients, dtype=torch.float64
                )
                calibration_target = calibration64[:, merge.dropped]
                validation_target = validation64[:, merge.dropped]
                calibration_prediction = (
                    calibration64.index_select(1, index) @ coefficients
                )
                validation_prediction = (
                    validation64.index_select(1, index) @ coefficients
                )
                calibration_error = float(
                    (
                        torch.linalg.vector_norm(
                            calibration_target - calibration_prediction
                        )
                        / torch.linalg.vector_norm(calibration_target).clamp_min(tiny)
                    ).item()
                )
                validation_error = float(
                    (
                        torch.linalg.vector_norm(
                            validation_target - validation_prediction
                        )
                        / torch.linalg.vector_norm(validation_target).clamp_min(tiny)
                    ).item()
                )
                tolerance = 1e-8 + 1e-6 * max(
                    1.0,
                    abs(calibration_error),
                    abs(validation_error),
                )
                if (
                    abs(calibration_error - merge.calibration_path_relative_l2)
                    > tolerance
                    or abs(validation_error - merge.validation_path_relative_l2)
                    > tolerance
                ):
                    per_merge_metric_mismatches += 1
                checked_edges += len(merge.anchors)
            if checked_edges > requested_drops * config.multi_anchor_max_anchors:
                raise QwenCompensatedRunnerError(
                    "Multi-anchor evidence exceeded its declared edge cap."
                )
            if (
                fold.diagnostics.calibration_provenance
                != config.calibration_provenance_id
                or fold.diagnostics.validation_provenance
                != config.validation_provenance_id
            ):
                raise QwenCompensatedRunnerError(
                    "Multi-anchor fold diagnostics lost exact split provenance."
                )
            source_down = mlp.down_proj.weight.detach().to(
                device="cpu", dtype=torch.float64
            )
            folded_down = fold.down_weight.detach().to(
                device="cpu", dtype=torch.float64
            )
            kept = torch.tensor(fold.kept_source_indices, dtype=torch.long)

            def aggregate_error(values: Tensor) -> float:
                expected = values @ source_down.T
                actual = values.index_select(1, kept) @ folded_down.T
                denominator = torch.linalg.vector_norm(expected).clamp_min(tiny)
                return float(
                    (torch.linalg.vector_norm(expected - actual) / denominator).item()
                )

            calibration_aggregate = aggregate_error(calibration64)
            validation_aggregate = aggregate_error(validation64)
            aggregate_tolerance = 1e-8 + 1e-6 * max(
                1.0, calibration_aggregate, validation_aggregate
            )
            if (
                abs(
                    calibration_aggregate
                    - fold.diagnostics.aggregate_calibration_relative_l2
                )
                > aggregate_tolerance
                or abs(
                    validation_aggregate
                    - fold.diagnostics.aggregate_validation_relative_l2
                )
                > aggregate_tolerance
            ):
                raise QwenCompensatedRunnerError(
                    "Multi-anchor aggregate evidence failed independent recomputation."
                )
            strategy_evidence.update(
                {
                    "calibration_provenance_id": config.calibration_provenance_id,
                    "validation_provenance_id": config.validation_provenance_id,
                    "provenance_ids_distinct": True,
                    "independent_aggregate_calibration_relative_l2": (
                        calibration_aggregate
                    ),
                    "independent_aggregate_validation_relative_l2": (
                        validation_aggregate
                    ),
                    "independent_aggregate_metrics_trusted": True,
                    "per_merge_metric_mismatch_count": per_merge_metric_mismatches,
                    "per_merge_metrics_used_for_runner_selection": False,
                    "checked_anchor_edge_count": checked_edges,
                    "work_caps": {
                        "candidate_pool_size": config.multi_anchor_candidate_pool_size,
                        "max_anchors": config.multi_anchor_max_anchors,
                        "similarity_chunk_size": (
                            config.multi_anchor_similarity_chunk_size
                        ),
                        "max_similarity_products": (
                            config.multi_anchor_max_similarity_products
                        ),
                        "max_projection_fits": (
                            config.multi_anchor_max_projection_fits
                        ),
                    },
                }
            )
    except Qwen3CompensatedFoldError as exc:
        raise QwenCompensatedRunnerError(
            f"Layer {layer_index} has no bounded arbitrary-pair proposal: {exc}"
        ) from exc
    scores = (
        [float(item.score) for item in fold.merges]
        if config.fold_strategy == "pair"
        else [float(fold.diagnostics.aggregate_validation_relative_l2)]
    )
    return LayerProposal(
        layer_index=layer_index,
        fold=fold,
        anchors=anchors,
        temporary_protected=(),
        anchor_report=anchor_report,
        prefilter_score=float(prefilter_score),
        score=max(scores),
        mean_score=sum(scores) / len(scores),
        strategy_evidence=strategy_evidence,
    )


def _layer_redundancy_prefilter(
    calibration: Tensor,
    *,
    max_columns: int,
) -> float:
    """Cheap calibration-only layer ranking without a width-squared matrix."""

    if calibration.ndim != 2 or int(calibration.shape[1]) < 2:
        raise QwenCompensatedRunnerError("Prefilter activations are invalid.")
    width = int(calibration.shape[1])
    count = min(_positive_int(max_columns, "prefilter_columns"), width)
    indices = torch.linspace(0, width - 1, count, dtype=torch.float64).round().long()
    values = calibration.detach().to(device="cpu", dtype=torch.float32).index_select(
        1, indices
    )
    values = values / torch.linalg.vector_norm(values, dim=0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    similarity = (values.T @ values).abs()
    similarity.fill_diagonal_(0.0)
    nearest = similarity.max(dim=1).values
    top_count = min(32, count)
    redundancy = torch.topk(nearest, top_count, largest=True).values.mean()
    return float((1.0 - redundancy).clamp_min(0.0).item())


def _select_best_proposal(
    model: nn.Module,
    activations: ActivationPair,
    *,
    drops: int,
    blocked_by_layer: Mapping[int, Sequence[int]],
    anchor_sources_by_layer: Mapping[int, Sequence[int]],
    anchor_reports_by_layer: Mapping[int, Mapping[str, Any]],
    lineage_by_layer: Mapping[int, Sequence[Sequence[int]]],
    config: FoldLoopConfig,
    epoch: int,
) -> tuple[LayerProposal | None, list[dict[str, Any]]]:
    best: LayerProposal | None = None
    skipped: list[dict[str, Any]] = []
    excluded = set(config.excluded_layers)
    ranked_layers: list[tuple[float, int]] = []
    for layer_index, _layer in enumerate(_qwen_layers(model)):
        if layer_index in excluded:
            skipped.append({"layer_index": layer_index, "reason": "explicitly-excluded"})
            continue
        ranked_layers.append(
            (
                _layer_redundancy_prefilter(
                    activations.calibration[layer_index],
                    max_columns=config.prefilter_columns,
                ),
                layer_index,
            )
        )
    ranked_layers.sort()
    selected_layers = ranked_layers[: config.proposal_layer_limit]
    for prefilter_score, layer_index in ranked_layers[config.proposal_layer_limit :]:
        skipped.append(
            {
                "layer_index": layer_index,
                "reason": "bounded-calibration-prefilter-not-selected",
                "prefilter_score": prefilter_score,
            }
        )
    for prefilter_score, layer_index in selected_layers:
        try:
            current_anchors = _current_anchor_indices(
                anchor_sources_by_layer[layer_index],
                lineage_by_layer[layer_index],
            )
            candidate = _fit_layer_proposal(
                model,
                layer_index,
                activations.calibration[layer_index],
                activations.validation[layer_index],
                drops=drops,
                blocked_indices=blocked_by_layer.get(layer_index, ()),
                persistent_anchor_indices=current_anchors,
                persistent_anchor_report=anchor_reports_by_layer[layer_index],
                config=config,
                epoch=epoch,
                prefilter_score=prefilter_score,
            )
        except QwenCompensatedRunnerError as exc:
            skipped.append({"layer_index": layer_index, "reason": str(exc)})
            continue
        if best is None or (
            candidate.score,
            candidate.mean_score,
            candidate.layer_index,
        ) < (best.score, best.mean_score, best.layer_index):
            if best is not None:
                del best
            best = candidate
        else:
            del candidate
    return best, skipped


SmallGate = Callable[[nn.Module, int, LayerProposal], tuple[bool, dict[str, Any]]]
MilestoneGate = Callable[[nn.Module, int, LayerProposal], tuple[bool, dict[str, Any]]]
ActivationProvider = Callable[[nn.Module], ActivationPair]


def _history_size(values: Sequence[Mapping[str, Any]]) -> int:
    return len(
        json.dumps(
            list(values),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )


def _append_bounded_history(
    history: list[dict[str, Any]],
    value: dict[str, Any],
    *,
    max_records: int,
    max_json_bytes: int,
) -> int:
    history.append(value)
    omitted = 0
    while len(history) > max_records or _history_size(history) > max_json_bytes:
        history.pop(0)
        omitted += 1
    if not history:
        raise QwenCompensatedRunnerError(
            "A single compact transaction record exceeds the history byte cap."
        )
    return omitted


def _append_transaction_ledger(path: Path | None, value: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _transaction_runtime_snapshot(model: nn.Module) -> dict[str, Any]:
    config_objects = [model.config, getattr(getattr(model, "model", None), "config", None)]
    return {
        "module_modes": [(module, bool(module.training)) for module in model.modules()],
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "configs": [
            (config, getattr(config, "intermediate_size", None))
            for config in config_objects
            if config is not None
        ],
    }


def _restore_transaction_runtime(snapshot: Mapping[str, Any]) -> None:
    for module, training in snapshot["module_modes"]:
        module.train(bool(training))
    random.setstate(snapshot["python_rng"])
    torch.set_rng_state(snapshot["torch_rng"])
    if snapshot["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
    for config, intermediate_size in snapshot["configs"]:
        if intermediate_size is not None:
            config.intermediate_size = intermediate_size


def run_fold_transactions(
    model: nn.Module,
    *,
    activation_provider: ActivationProvider,
    small_gate: SmallGate,
    target_parameters: int | None,
    target_bytes: int | None,
    target_resident_bytes: int | None = None,
    config: FoldLoopConfig,
    milestone_gate: MilestoneGate | None = None,
    transaction_ledger: Path | None = None,
    resource_guard: WorkflowResourceGuard | None = None,
) -> dict[str, Any]:
    """Sequentially install, gate, roll back or commit compensated folds."""

    initial = model_accounting(model)
    if target_parameters is not None:
        _positive_int(target_parameters, "target_unique_parameters")
    if target_bytes is not None:
        _positive_int(target_bytes, "target_unique_bytes")
    if target_resident_bytes is not None:
        _positive_int(target_resident_bytes, "target_resident_tensor_bytes")
    if (
        target_parameters is None
        and target_bytes is None
        and target_resident_bytes is None
    ):
        raise QwenCompensatedRunnerError("An explicit target is required.")
    _assert_model_safe(model, "transaction-loop start")
    activations = activation_provider(model)
    activation_refreshes = 1
    anchor_sources, anchor_reports, lineage = _initialize_persistent_anchors(
        model, activations, config
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    skipped_history: list[dict[str, Any]] = []
    accepted_omitted = 0
    rejected_omitted = 0
    skipped_omitted = 0
    blocked_by_layer: dict[int, set[int]] = {}
    attempts = 0
    accounting = initial
    stop_reason = "target-reached" if _target_reached(
        accounting, target_parameters, target_bytes, target_resident_bytes
    ) else "max-transactions"

    while not _target_reached(
        accounting, target_parameters, target_bytes, target_resident_bytes
    ):
        if resource_guard is not None:
            resource_guard.check(f"transaction-{attempts + 1:06d}:start")
        if attempts >= config.max_transactions:
            stop_reason = "max-transactions"
            break
        first_layer = _qwen_layers(model)[0]
        hidden_size = int(first_layer.mlp.gate_proj.weight.shape[1])
        drops = _needed_drops(
            accounting,
            hidden_size=hidden_size,
            element_size=int(first_layer.mlp.gate_proj.weight.element_size()),
            target_parameters=target_parameters,
            target_bytes=target_bytes,
            target_resident_bytes=target_resident_bytes,
            maximum=config.merges_per_transaction,
        )
        proposal, skipped = _select_best_proposal(
            model,
            activations,
            drops=drops,
            blocked_by_layer=blocked_by_layer,
            anchor_sources_by_layer=anchor_sources,
            anchor_reports_by_layer=anchor_reports,
            lineage_by_layer=lineage,
            config=config,
            epoch=len(accepted),
        )
        for item in skipped:
            skipped_omitted += _append_bounded_history(
                skipped_history,
                {"attempt": attempts + 1, **item},
                max_records=config.max_history_records,
                max_json_bytes=config.max_history_json_bytes,
            )
        if proposal is None:
            stop_reason = "no-eligible-arbitrary-pair-proposal"
            break
        attempts += 1
        before = accounting
        snapshot = _transaction_runtime_snapshot(model)
        handle = None
        appended_to_accepted = False
        try:
            handle = install_qwen3_compensated_fold(
                model, proposal.layer_index, proposal.fold
            )
            anchor_proof = verify_exact_anchors(
                handle.original_mlp,
                handle.installed_mlp,
                proposal.fold,
                proposal.anchors,
            )
            provisional = model_accounting(model)
            expected_parameters = (
                before.unique_parameter_count - proposal.fold.parameter_savings
            )
            if provisional.unique_parameter_count != expected_parameters:
                raise QwenCompensatedRunnerError(
                    "Installed transaction parameter accounting is not exact."
                )
            accepted_by_small, small_report = small_gate(model, attempts, proposal)
            milestone_report: dict[str, Any] | None = None
            accepted_by_milestone = True
            prospective_accepts = accepted_omitted + len(accepted) + 1
            if (
                accepted_by_small
                and milestone_gate is not None
                and prospective_accepts % config.full_milestone_interval == 0
            ):
                accepted_by_milestone, milestone_report = milestone_gate(
                    model, attempts, proposal
                )
            transaction_report = {
                "attempt": attempts,
                "proposal": proposal.summary(),
                "before": before.as_dict(),
                "provisional": provisional.as_dict(),
                "anchor_proof": anchor_proof,
                "small_gate": small_report,
                "milestone_gate": milestone_report,
                "zero_optimizer_steps": 0,
                "backward_called": False,
            }
            if not (accepted_by_small and accepted_by_milestone):
                handle.rollback()
                _restore_transaction_runtime(snapshot)
                after_rollback = model_accounting(model)
                if after_rollback != before:
                    raise QwenCompensatedRunnerError(
                        "Rejected transaction did not restore exact accounting."
                    )
                blocked = blocked_by_layer.setdefault(proposal.layer_index, set())
                for merge in proposal.fold.merges:
                    blocked.add(merge.dropped)
                transaction_report.update(
                    {
                        "decision": "rejected-rolled-back",
                        "after": after_rollback.as_dict(),
                        "rollback_exact": True,
                    }
                )
                _append_transaction_ledger(transaction_ledger, transaction_report)
                rejected_omitted += _append_bounded_history(
                    rejected,
                    transaction_report,
                    max_records=config.max_history_records,
                    max_json_bytes=config.max_history_json_bytes,
                )
                del handle
                del proposal
                continue

            _assert_model_safe(model, f"provisional transaction {attempts}")
            next_activations = activation_provider(model)
            activation_refreshes += 1
            next_lineage = _lineage_after_fold(
                lineage[proposal.layer_index], proposal.fold
            )
            synchronize_qwen3_intermediate_size(model)
            committed_accounting = model_accounting(model)
            if not _same_storage_accounting(committed_accounting, provisional):
                raise QwenCompensatedRunnerError(
                    "Commit-ready transaction changed storage accounting."
                )
            _assert_model_safe(model, f"commit-ready transaction {attempts}")
            transaction_report.update(
                {
                    "decision": "accepted-committed",
                    "after": committed_accounting.as_dict(),
                    "rollback_exact": None,
                    "refresh_completed_before_commit": True,
                    "persistent_anchor_lineage_sha256": _canonical_sha256(
                        next_lineage
                    ),
                }
            )
            _append_transaction_ledger(transaction_ledger, transaction_report)
            accepted_omitted += _append_bounded_history(
                accepted,
                transaction_report,
                max_records=config.max_history_records,
                max_json_bytes=config.max_history_json_bytes,
            )
            appended_to_accepted = True
            handle.commit()
        except BaseException:
            if appended_to_accepted and accepted and accepted[-1].get("attempt") == attempts:
                accepted.pop()
            if handle is not None and not handle.committed and not handle.reverted:
                handle.rollback()
            _restore_transaction_runtime(snapshot)
            if model_accounting(model) != before:
                raise QwenCompensatedRunnerError(
                    "Exceptional transaction rollback did not restore exact accounting."
                )
            raise

        accounting = committed_accounting
        lineage[proposal.layer_index] = next_lineage
        blocked_by_layer.clear()
        del handle
        del proposal
        del activations
        gc.collect()
        activations = next_activations
        if resource_guard is not None:
            resource_guard.check(f"transaction-{attempts:06d}:committed")
        if _target_reached(
            accounting, target_parameters, target_bytes, target_resident_bytes
        ):
            stop_reason = "target-reached"
            break

    del activations
    _assert_model_safe(model, "transaction-loop final")
    final = model_accounting(model)
    return {
        "initial": initial.as_dict(),
        "final": final.as_dict(),
        "target": {
            "unique_parameter_count": target_parameters,
            "unique_parameter_bytes": target_bytes,
            "resident_tensor_bytes": target_resident_bytes,
            "reached": _target_reached(
                final, target_parameters, target_bytes, target_resident_bytes
            ),
        },
        "attempt_count": attempts,
        "accepted_count": accepted_omitted + len(accepted),
        "rejected_count": rejected_omitted + len(rejected),
        "activation_collection_count": activation_refreshes,
        "activation_refresh_after_every_accept": activation_refreshes == len(accepted) + 1,
        "accepted_transactions": accepted,
        "rejected_transactions": rejected,
        "layer_skips": skipped_history,
        "history": {
            "max_records_per_collection": config.max_history_records,
            "max_json_bytes_per_collection": config.max_history_json_bytes,
            "accepted_omitted_from_inline": accepted_omitted,
            "rejected_omitted_from_inline": rejected_omitted,
            "layer_skips_omitted_from_inline": skipped_omitted,
            "ledger": str(transaction_ledger) if transaction_ledger else None,
            "ledger_sha256": (
                _file_sha256(transaction_ledger)
                if transaction_ledger is not None and transaction_ledger.is_file()
                else None
            ),
        },
        "persistent_anchors": {
            str(index): {
                **anchor_reports[index],
                "final_current_indices": list(
                    _current_anchor_indices(anchor_sources[index], lineage[index])
                ),
                "final_lineage_sha256": _canonical_sha256(lineage[index]),
            }
            for index in sorted(anchor_sources)
        },
        "exact_layers": [
            index
            for index, (before_width, after_width) in enumerate(
                zip(initial.layer_widths, final.layer_widths, strict=True)
            )
            if before_width == after_width
        ],
        "stop_reason": stop_reason,
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "recovery_examples": 0,
    }


def _state_structure(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
        for name, value in model.state_dict().items()
    ]


def _dtype_name(dtype: torch.dtype) -> str:
    values = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if dtype not in values:
        raise QwenCompensatedRunnerError(
            f"Unsupported checkpoint construction dtype: {dtype}."
        )
    return values[dtype]


def _dtype_from_name(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in values:
        raise QwenCompensatedRunnerError(
            f"Unsupported checkpoint dtype: {name}."
        )
    return values[name]


def _checkpoint_files(directory: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            result.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return result


def artifact_tree_report(directory: Path) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise QwenCompensatedRunnerError(f"Artifact directory does not exist: {root}.")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": _file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise QwenCompensatedRunnerError("Artifact directory is empty.")
    return {
        "directory": str(root),
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "tree_sha256": _canonical_sha256(files),
        "files": files,
    }


def _filesystem_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    filesystem_type = "unknown"
    mount_point = "unknown"
    if os.name == "posix":
        try:
            candidates: list[tuple[int, str, str]] = []
            for line in Path("/proc/self/mountinfo").read_text(
                encoding="utf-8"
            ).splitlines():
                before, separator, after = line.partition(" - ")
                if not separator:
                    continue
                fields = before.split()
                trailing = after.split()
                if len(fields) < 5 or not trailing:
                    continue
                candidate = fields[4].replace("\\040", " ")
                try:
                    resolved.relative_to(Path(candidate))
                except ValueError:
                    continue
                candidates.append((len(candidate), candidate, trailing[0]))
            if candidates:
                _length, mount_point, filesystem_type = max(candidates)
        except OSError:
            pass
    remote_types = {
        "9p",
        "cifs",
        "fuse.sshfs",
        "nfs",
        "nfs4",
        "smb2",
        "smb3",
    }
    usage = shutil.disk_usage(resolved)
    return {
        "path": str(resolved),
        "filesystem_type": filesystem_type,
        "mount_point": mount_point,
        "remote_filesystem": filesystem_type.lower() in remote_types,
        "free_bytes_before": int(usage.free),
        "total_bytes": int(usage.total),
    }


def prepare_local_checkpoint_staging(
    output: Path,
    configured_root: Path | None,
    *,
    required_bytes: int,
) -> tuple[Path, Path, dict[str, Any]]:
    root_value = configured_root or Path(
        os.environ.get(
            "NEURAL_FORGE_CHECKPOINT_STAGING_ROOT",
            str(Path(tempfile.gettempdir()) / "neural-forge-checkpoints"),
        )
    )
    root = root_value.expanduser().resolve()
    filesystem = _filesystem_report(root)
    if filesystem["remote_filesystem"]:
        raise QwenCompensatedRunnerError(
            "Checkpoint staging must be on a local filesystem, not "
            f"{filesystem['filesystem_type']}."
        )
    required = _positive_int(required_bytes, "required checkpoint staging bytes")
    safety_margin = max(64 * 1024**2, math.ceil(required * 0.05))
    if int(filesystem["free_bytes_before"]) < required + safety_margin:
        raise QwenCompensatedRunnerError(
            "Local checkpoint staging has insufficient free space: "
            f"free={filesystem['free_bytes_before']}, required={required + safety_margin}."
        )
    run_root = Path(
        tempfile.mkdtemp(
            prefix=f"{output.name or 'run'}-{os.getpid()}-",
            dir=root,
        )
    ).resolve()
    checkpoint = run_root / "ragged_child"
    report = {
        **filesystem,
        "run_root": str(run_root),
        "checkpoint_directory": str(checkpoint),
        "required_bytes": required,
        "safety_margin_bytes": safety_margin,
        "local_staging_required": True,
    }
    return run_root, checkpoint, report


def _fsync_directory(directory: Path) -> float:
    started = time.monotonic()
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return float(time.monotonic() - started)


def publish_staged_checkpoint(
    staged_checkpoint: Path,
    published_checkpoint: Path,
    *,
    lock_path: Path,
) -> dict[str, Any]:
    """Publish one large artifact under the cross-process CIFS writer lease."""

    source = staged_checkpoint.expanduser().resolve()
    destination = published_checkpoint.expanduser().resolve()
    lock = lock_path.expanduser().resolve()
    source_report = artifact_tree_report(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.parent / (
        f".{destination.name}.publish-{os.getpid()}-{_canonical_sha256(str(source))[:12]}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    _progress("compact_checkpoint_io_wait", lock=lock)
    publish_started = time.monotonic()
    with codebook_runner._checkpoint_io_lease(lock) as lease:
        _progress(
            "compact_checkpoint_io_acquired",
            wait_seconds=round(float(lease["wait_seconds"]), 6),
        )
        try:
            copy_started = time.monotonic()
            temporary.mkdir()
            copied_files: list[dict[str, Any]] = []
            fsync_seconds = 0.0
            for item in source_report["files"]:
                relative = Path(str(item["path"]))
                source_file = source / relative
                target_file = temporary / relative
                if source_file.is_symlink() or not source_file.is_file():
                    raise QwenCompensatedRunnerError(
                        "Checkpoint publication rejects links and non-files."
                    )
                target_file.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                copied_bytes = 0
                with source_file.open("rb") as reader, target_file.open("xb") as writer:
                    for chunk in iter(lambda: reader.read(8 * 1024**2), b""):
                        writer.write(chunk)
                        digest.update(chunk)
                        copied_bytes += len(chunk)
                    writer.flush()
                    fsync_started = time.monotonic()
                    os.fsync(writer.fileno())
                    fsync_seconds += time.monotonic() - fsync_started
                actual_hash = digest.hexdigest()
                if copied_bytes != int(item["bytes"]) or actual_hash != item["sha256"]:
                    raise QwenCompensatedRunnerError(
                        f"Published checkpoint copy differs for {relative.as_posix()}."
                    )
                copied_files.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": copied_bytes,
                        "sha256": actual_hash,
                    }
                )
            copy_seconds = float(time.monotonic() - copy_started)
            temporary_report = {
                "directory": str(temporary),
                "file_count": len(copied_files),
                "total_bytes": sum(int(item["bytes"]) for item in copied_files),
                "tree_sha256": _canonical_sha256(copied_files),
                "files": copied_files,
            }
            if (
                temporary_report["tree_sha256"] != source_report["tree_sha256"]
                or temporary_report["total_bytes"] != source_report["total_bytes"]
            ):
                raise QwenCompensatedRunnerError(
                    "Published checkpoint bytes differ before atomic promotion."
                )
            os.replace(temporary, destination)
            promotion_fsync_seconds = _fsync_directory(destination.parent)
            published_sizes = [
                {
                    "path": item["path"],
                    "bytes": int((destination / item["path"]).stat().st_size),
                }
                for item in copied_files
            ]
            if any(
                item["bytes"] != copied["bytes"]
                for item, copied in zip(
                    source_report["files"], published_sizes, strict=True
                )
            ):
                raise QwenCompensatedRunnerError(
                    "Published checkpoint size differs after atomic promotion."
                )
            published_report = {
                **temporary_report,
                "directory": str(destination),
            }
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {
        "lock_path": str(lock),
        "io_lease": dict(lease),
        "lease_wait_seconds": float(lease["wait_seconds"]),
        "copy_seconds": copy_seconds,
        "fsync_seconds": fsync_seconds,
        "promotion_fsync_seconds": promotion_fsync_seconds,
        "publish_wall_seconds_including_wait": float(
            time.monotonic() - publish_started
        ),
        "staging": source_report,
        "published": published_report,
        "hashes_match": True,
        "bytes_match": True,
        "hashes_verified_during_copy": True,
        "content_readback_after_publish": False,
        "single_writer_large_io": bool(lease["serialized"]),
    }


def remove_checkpoint_staging(run_root: Path, staging_root: Path) -> None:
    resolved_run = run_root.expanduser().resolve()
    resolved_root = staging_root.expanduser().resolve()
    try:
        relative = resolved_run.relative_to(resolved_root)
    except ValueError as exc:
        raise QwenCompensatedRunnerError(
            "Refusing to remove checkpoint staging outside its configured root."
        ) from exc
    if not relative.parts or resolved_run == resolved_root:
        raise QwenCompensatedRunnerError(
            "Refusing to remove the checkpoint staging root itself."
        )
    shutil.rmtree(resolved_run)


def _write_stable_manifest(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest_bytes = 0
    artifact_bytes = sum(int(item["bytes"]) for item in payload["files"])
    for _ in range(16):
        payload["artifact_accounting"]["manifest_bytes"] = manifest_bytes
        payload["artifact_accounting"]["artifact_total_bytes"] = (
            artifact_bytes + manifest_bytes
        )
        envelope = {
            "schema": CHECKPOINT_SCHEMA,
            "payload": payload,
            "payload_sha256": _canonical_sha256(payload),
        }
        encoded = (
            json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(encoded) == manifest_bytes:
            _write_json(manifest_path, envelope)
            if manifest_path.stat().st_size != manifest_bytes:
                raise QwenCompensatedRunnerError(
                    "Stable manifest byte accounting changed during write."
                )
            return envelope
        manifest_bytes = len(encoded)
    raise QwenCompensatedRunnerError("Manifest byte accounting did not converge.")


def save_ragged_qwen_checkpoint(
    model: nn.Module,
    tokenizer: Any | None,
    directory: Path,
    *,
    transaction_records: Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save a standalone ragged Qwen runtime without pickle."""

    try:
        from safetensors.torch import save_model
    except ImportError as exc:
        raise QwenCompensatedRunnerError("safetensors is required.") from exc
    destination = directory.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _assert_model_safe(model, "checkpoint save")
        accounting = model_accounting(model)
        config = model.config.to_dict()
        if str(config.get("model_type")) != "qwen3":
            raise QwenCompensatedRunnerError("Only model_type='qwen3' is supported.")
        _write_json(temporary / "ragged_config.json", config)
        save_model(
            model,
            str(temporary / "model.safetensors"),
            metadata={
                "format": "pt",
                "schema": CHECKPOINT_SCHEMA,
                "loader_contract": LOADER_CONTRACT,
            },
        )
        tokenizer_saved = False
        if tokenizer is not None:
            save = getattr(tokenizer, "save_pretrained", None)
            if not callable(save):
                raise QwenCompensatedRunnerError(
                    "Tokenizer cannot be saved into the standalone artifact."
                )
            save(temporary / "tokenizer")
            tokenizer_saved = True
        files = _checkpoint_files(temporary)
        parameter_dtypes = sorted({str(value.dtype) for value in model.parameters()})
        if len(parameter_dtypes) != 1:
            raise QwenCompensatedRunnerError(
                "Checkpoint construction requires one parameter dtype."
            )
        construction_dtype = _dtype_name(next(model.parameters()).dtype)
        tensor_file_bytes = (temporary / "model.safetensors").stat().st_size
        tensor_value_bytes = (
            accounting.unique_parameter_bytes + accounting.unique_buffer_bytes
        )
        payload: dict[str, Any] = {
            "loader_contract": {
                "required_loader": LOADER_CONTRACT,
                "safe_custom_reload": True,
                "pickle_allowed": False,
                "trust_remote_code": False,
                "stock_huggingface_reload_supported": False,
                "stock_huggingface_reload_attempted": False,
                "ragged_widths_require_custom_loader": accounting.ragged,
                "stock_reload_rejection_reason": (
                    "per-layer-ragged-intermediate-widths"
                    if accounting.ragged
                    else "explicit-custom-artifact-layout"
                ),
            },
            "model": {
                "model_type": "qwen3",
                "construction_dtype": construction_dtype,
                "layer_widths": list(accounting.layer_widths),
                "hidden_size": int(model.config.hidden_size),
                "logical_layers": len(accounting.layer_widths),
                "state_structure": _state_structure(model),
                "state_structure_sha256": _canonical_sha256(_state_structure(model)),
                "accounting": accounting.as_dict(),
            },
            "transactions": [dict(value) for value in transaction_records],
            "transactions_sha256": _canonical_sha256(
                [dict(value) for value in transaction_records]
            ),
            "source_provenance": dict(source_provenance),
            "tokenizer_saved": tokenizer_saved,
            "files": files,
            "artifact_accounting": {
                "model_tensor_value_bytes": tensor_value_bytes,
                "safetensors_file_bytes": tensor_file_bytes,
                "safetensors_header_and_alignment_bytes": (
                    tensor_file_bytes - tensor_value_bytes
                ),
                "non_manifest_file_bytes": sum(int(item["bytes"]) for item in files),
                "manifest_bytes": 0,
                "artifact_total_bytes": 0,
                "ragged_width_metadata_bytes": len(
                    json.dumps(list(accounting.layer_widths), separators=(",", ":")).encode(
                        "ascii"
                    )
                ),
                "all_artifact_files_hashed_and_counted": True,
                "all_metadata_counted": True,
            },
            "zero_optimizer_steps": 0,
            "backward_called": False,
            "recovery_examples": 0,
        }
        envelope = _write_stable_manifest(temporary, payload)
        os.replace(temporary, destination)
        return envelope
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_checkpoint_files(directory: Path, payload: Mapping[str, Any]) -> None:
    expected = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in payload.get("files", [])
    }
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != set(expected):
        raise QwenCompensatedRunnerError(
            "Checkpoint files differ from the closed manifest."
        )
    for relative, (size, digest) in expected.items():
        path = directory / relative
        if path.stat().st_size != size or _file_sha256(path) != digest:
            raise QwenCompensatedRunnerError(
                f"Checkpoint file integrity failed: {relative}."
            )


def load_ragged_qwen_checkpoint(
    directory: Path,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, Any | None, dict[str, Any]]:
    """Safely reconstruct a ragged Qwen runtime without accessing its parent."""

    try:
        import transformers
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:
        raise QwenCompensatedRunnerError(
            "transformers and safetensors are required."
        ) from exc
    source = directory.expanduser().resolve()
    try:
        envelope = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenCompensatedRunnerError("Cannot read checkpoint manifest.") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != CHECKPOINT_SCHEMA:
        raise QwenCompensatedRunnerError("Checkpoint schema is not supported.")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != _canonical_sha256(payload):
        raise QwenCompensatedRunnerError("Checkpoint manifest integrity failed.")
    contract = payload.get("loader_contract") or {}
    if (
        contract.get("required_loader") != LOADER_CONTRACT
        or contract.get("pickle_allowed") is not False
        or contract.get("safe_custom_reload") is not True
        or contract.get("stock_huggingface_reload_supported") is not False
    ):
        raise QwenCompensatedRunnerError("Checkpoint loader contract is unsafe.")
    _verify_checkpoint_files(source, payload)
    try:
        config_data = json.loads(
            (source / "ragged_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenCompensatedRunnerError("Cannot read ragged Qwen config.") from exc
    if not isinstance(config_data, dict) or config_data.get("model_type") != "qwen3":
        raise QwenCompensatedRunnerError("Ragged config is not Qwen3.")
    model_report = payload.get("model") or {}
    widths = tuple(int(value) for value in model_report.get("layer_widths", []))
    if len(widths) != int(config_data.get("num_hidden_layers", -1)) or min(widths) <= 0:
        raise QwenCompensatedRunnerError("Ragged layer-width contract is invalid.")
    model_type = config_data.pop("model_type")
    config_data.pop("_name_or_path", None)
    config = transformers.AutoConfig.for_model(model_type, **config_data)
    construction_dtype = _dtype_from_name(str(model_report["construction_dtype"]))
    try:
        with torch.device("meta"):
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(construction_dtype)
            try:
                model = transformers.AutoModelForCausalLM.from_config(
                    config, trust_remote_code=False
                )
            finally:
                torch.set_default_dtype(previous_dtype)
            layers = _qwen_layers(model)
            hidden_size = int(config.hidden_size)
            for layer, width in zip(layers, widths, strict=True):
                original = layer.mlp
                if int(original.gate_proj.out_features) != width:
                    layer.mlp = Qwen3CompensatedMLP(
                        hidden_size, width, original.act_fn
                    ).to(dtype=construction_dtype)
    except Exception as exc:
        raise QwenCompensatedRunnerError(
            "Could not construct the explicit ragged runtime on the meta device."
        ) from exc
    tensor_path = source / "model.safetensors"
    try:
        state = load_file(tensor_path, device="cpu", backend="mmap")
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            aliases = dict(handle.metadata() or {})
        model_keys = set(model.state_dict())
        for alias, target in aliases.items():
            if alias in model_keys and target in state:
                state[alias] = state[target]
        incompatible = model.load_state_dict(state, strict=False, assign=True)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        del state
    except Exception as exc:
        raise QwenCompensatedRunnerError(
            "Safetensors state does not match the explicit ragged runtime."
        ) from exc
    if missing or unexpected:
        raise QwenCompensatedRunnerError(
            f"Ragged state mismatch: missing={missing}, unexpected={unexpected}."
        )
    model.tie_weights()
    materialized_nonpersistent_buffers = 0
    for module in model.modules():
        meta_buffers = [
            name
            for name, value in module._buffers.items()
            if isinstance(value, Tensor) and value.device.type == "meta"
        ]
        if not meta_buffers:
            continue
        config_value = getattr(module, "config", None)
        try:
            replacement = type(module)(config_value, device="cpu")
        except Exception as exc:
            raise QwenCompensatedRunnerError(
                f"Cannot materialize nonpersistent buffers for {type(module).__name__}."
            ) from exc
        for name in meta_buffers:
            replacement_value = replacement._buffers.get(name)
            if not isinstance(replacement_value, Tensor):
                raise QwenCompensatedRunnerError(
                    f"Missing CPU reconstruction for nonpersistent buffer {name}."
                )
            module._buffers[name] = replacement_value
            materialized_nonpersistent_buffers += 1
    try:
        if any(value.device.type == "meta" for value in model.state_dict().values()):
            raise QwenCompensatedRunnerError(
                "Direct ragged loading left unresolved meta tensors."
            )
    except Exception:
        raise
    model.to(device)
    model.eval()
    _assert_model_safe(model, "parent-free reload")
    actual_structure = _state_structure(model)
    if _canonical_sha256(actual_structure) != model_report.get("state_structure_sha256"):
        raise QwenCompensatedRunnerError(
            "Reloaded state structure differs from the manifest."
        )
    expected_accounting = model_report.get("accounting")
    if model_accounting(model).as_dict() != expected_accounting:
        raise QwenCompensatedRunnerError(
            "Reloaded parameter or ragged-width accounting differs."
        )
    tokenizer = None
    if payload.get("tokenizer_saved"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            source / "tokenizer",
            local_files_only=True,
            trust_remote_code=False,
        )
    envelope["runtime_load"] = {
        "construction_device": "meta",
        "state_assignment": "mmap-load_file-plus-assign-true",
        "full_random_model_allocated": False,
        "parent_checkpoint_accesses": 0,
        "accounting_hard_asserted": True,
        "materialized_nonpersistent_buffer_count": materialized_nonpersistent_buffers,
    }
    return model, tokenizer, envelope


@dataclass(slots=True)
class ReloadProbe:
    prompt: str
    inputs: dict[str, Tensor]
    expected_logits_sha256: str
    expected_logits_shape: tuple[int, ...]
    expected_generation_ids: Tensor


@dataclass(slots=True)
class ReloadReference:
    probes: tuple[ReloadProbe, ...]
    accounting: ModelAccounting
    generation_tokens: int


def capture_reload_reference(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str] | str,
    *,
    device: str,
    max_length: int,
    generation_tokens: int = 8,
) -> ReloadReference:
    values = [str(prompts)] if isinstance(prompts, str) else [str(v) for v in prompts]
    if len(values) < 2 or len(values) != len(set(values)):
        raise QwenCompensatedRunnerError(
            "Reload equivalence requires at least two distinct prompts."
        )
    generated = _positive_int(generation_tokens, "reload generation tokens")
    probes: list[ReloadProbe] = []
    for prompt in values:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        inputs = {
            name: value.to(device)
            for name, value in encoded.items()
            if isinstance(value, Tensor)
        }
        with torch.inference_mode():
            expected = model(
                **inputs, use_cache=False, return_dict=True
            ).logits.detach()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=generated,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            ).detach().cpu()
        probes.append(
            ReloadProbe(
                prompt=prompt,
                inputs={name: value.detach().cpu() for name, value in inputs.items()},
                expected_logits_sha256=_tensor_sha256(expected),
                expected_logits_shape=tuple(int(v) for v in expected.shape),
                expected_generation_ids=output_ids,
            )
        )
    return ReloadReference(
        probes=tuple(probes),
        accounting=model_accounting(model),
        generation_tokens=generated,
    )


def verify_reload_equivalence(
    reference: ReloadReference,
    after: nn.Module,
    tokenizer: Any,
    *,
    device: str,
) -> dict[str, Any]:
    probe_reports: list[dict[str, Any]] = []
    for index, probe in enumerate(reference.probes):
        encoded = tokenizer(
            probe.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(int(v.shape[-1]) for v in probe.inputs.values()),
            add_special_tokens=True,
        )
        reencoded = {
            name: value.detach().cpu()
            for name, value in encoded.items()
            if isinstance(value, Tensor)
        }
        if set(reencoded) != set(probe.inputs) or any(
            not torch.equal(reencoded[name], probe.inputs[name]) for name in probe.inputs
        ):
            raise QwenCompensatedRunnerError(
                f"Reloaded tokenizer differs on equivalence probe {index}."
            )
        inputs = {name: value.to(device) for name, value in probe.inputs.items()}
        with torch.inference_mode():
            actual = after(
                **inputs, use_cache=False, return_dict=True
            ).logits.detach()
            generated = after.generate(
                **inputs,
                max_new_tokens=reference.generation_tokens,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            ).detach().cpu()
        actual_hash = _tensor_sha256(actual)
        if (
            tuple(actual.shape) != probe.expected_logits_shape
            or actual_hash != probe.expected_logits_sha256
        ):
            raise QwenCompensatedRunnerError(
                f"Parent-free ragged reload is not logit-exact on probe {index}."
            )
        if not torch.equal(generated, probe.expected_generation_ids):
            raise QwenCompensatedRunnerError(
                f"Parent-free deterministic generation differs on probe {index}."
            )
        probe_reports.append(
            {
                "index": index,
                "prompt_sha256": _canonical_sha256(probe.prompt),
                "logits_sha256": actual_hash,
                "logits_shape": list(actual.shape),
                "tokenization_bitwise_exact": True,
                "deterministic_generation_bitwise_exact": True,
            }
        )
    accounting_exact = reference.accounting == model_accounting(after)
    if not accounting_exact:
        raise QwenCompensatedRunnerError(
            "Parent-free reload accounting differs from the staged runtime."
        )
    return {
        "logits_bitwise_exact": True,
        "tokenization_bitwise_exact": True,
        "deterministic_generation_bitwise_exact": True,
        "probe_count": len(probe_reports),
        "probes": probe_reports,
        "accounting_exact": accounting_exact,
        "parent_checkpoint_accesses": 0,
    }


def _compact_checkpoint_transactions(
    transactions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep surgery provenance in the artifact without embedding eval payloads."""

    compact: list[dict[str, Any]] = []
    for transaction in transactions:
        small_gate = transaction.get("small_gate") or {}
        milestone_gate = transaction.get("milestone_gate") or {}
        compact.append(
            {
                "attempt": int(transaction["attempt"]),
                "proposal": dict(transaction["proposal"]),
                "before": dict(transaction["before"]),
                "after": dict(transaction["after"]),
                "anchor_proof": dict(transaction["anchor_proof"]),
                "decision": str(transaction["decision"]),
                "small_gate_verdict": dict(small_gate.get("verdict") or {}),
                "small_gate_sha256": _canonical_sha256(small_gate),
                "milestone_gate_verdict": dict(
                    milestone_gate.get("verdict") or {}
                ),
                "milestone_gate_sha256": _canonical_sha256(milestone_gate),
                "zero_optimizer_steps": 0,
                "backward_called": False,
            }
        )
    return compact


def _suite_contract_for(
    args: argparse.Namespace,
    *,
    eval_suite: Path,
    max_text_evals: int,
    max_choice_probes: int,
    max_generation_prompts: int,
    max_behavior_probes: int,
) -> dict[str, Any]:
    values = dict(vars(args))
    values.update(
        {
            "eval_suite": eval_suite,
            "max_text_evals": max_text_evals,
            "max_choice_probes": max_choice_probes,
            "max_generation_prompts": max_generation_prompts,
            "max_behavior_probes": max_behavior_probes,
        }
    )
    return codebook_runner._suite_contract(argparse.Namespace(**values))


def _selected_suite(
    path: Path,
    *,
    max_text_evals: int,
    max_choice_probes: int,
    max_generation_prompts: int,
    max_behavior_probes: int,
) -> dict[str, Any]:
    from neural_forge.hf_tools import _load_eval_suite

    suite = _load_eval_suite(path)
    return {
        "source": suite["source"],
        "text_evals": list(suite["text_evals"][:max_text_evals]),
        "choice_probes": list(suite["choice_probes"][:max_choice_probes]),
        "generation_prompts": list(
            suite["generation_prompts"][:max_generation_prompts]
        ),
        "behavior_probes": list(
            suite["behavior_probes"][:max_behavior_probes]
        ),
    }


def _assert_production_certification_contract(
    contract: Mapping[str, Any], *, manual_sample_count: int
) -> None:
    counts = contract.get("counts") or {}
    required = {
        "behavior_probes": 1000,
        "generation_prompts": 1000,
        "choice_probes": 250,
    }
    underfilled = {
        name: {"actual": int(counts.get(name, 0)), "required": minimum}
        for name, minimum in required.items()
        if int(counts.get(name, 0)) < minimum
    }
    if underfilled:
        raise QwenCompensatedRunnerError(
            f"Certification suite is underfilled: {underfilled}."
        )
    available_manual = int(counts["behavior_probes"]) + int(
        counts["generation_prompts"]
    )
    if _positive_int(manual_sample_count, "manual_sample_count") > available_manual:
        raise QwenCompensatedRunnerError(
            "Manual review count exceeds behavior plus generation evidence."
        )


def _assert_selection_contract(
    contract: Mapping[str, Any],
    *,
    gate_behavior_count: int,
    gate_choice_count: int,
    gate_generation_count: int,
) -> None:
    counts = contract.get("counts") or {}
    required = {
        "behavior_probes": gate_behavior_count,
        "choice_probes": gate_choice_count,
        "generation_prompts": gate_generation_count,
    }
    underfilled = {
        name: {"actual": int(counts.get(name, 0)), "required": int(minimum)}
        for name, minimum in required.items()
        if int(counts.get(name, 0)) < int(minimum)
    }
    if underfilled:
        raise QwenCompensatedRunnerError(
            f"Adaptive selection suite is underfilled: {underfilled}."
        )


def _manual_item_maps(report: Mapping[str, Any], collection: str) -> dict[str, dict[str, Any]]:
    if collection == "behavior":
        return _item_map((report.get("behavior_suite") or {}).get("items"))
    if collection == "generation":
        return _item_map(report.get("generations"))
    raise QwenCompensatedRunnerError(f"Unknown manual-review collection: {collection}.")


def build_manual_review_packet(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    required_count: int,
) -> dict[str, Any]:
    selected = comparison.get("manual_samples")
    if not isinstance(selected, Mapping):
        raise QwenCompensatedRunnerError("Comparison lacks manual samples.")
    samples: list[dict[str, Any]] = []
    parent_generation_diagnostics = _item_map(parent.get("generation_diagnostics"))
    child_generation_diagnostics = _item_map(candidate.get("generation_diagnostics"))
    for collection in ("behavior", "generation"):
        parent_items = _manual_item_maps(parent, collection)
        child_items = _manual_item_maps(candidate, collection)
        for selection in selected.get(collection, []):
            item_id = str(selection["id"])
            if item_id not in parent_items or item_id not in child_items:
                raise QwenCompensatedRunnerError(
                    f"Manual-review item is missing from {collection}: {item_id}."
                )
            parent_item = parent_items[item_id]
            child_item = child_items[item_id]
            parent_output = parent_item.get("output")
            child_output = child_item.get("output")
            if not isinstance(parent_output, str) or not isinstance(child_output, str):
                raise QwenCompensatedRunnerError(
                    f"Manual-review outputs are not verbatim strings: {item_id}."
                )
            parent_diagnostics = dict(parent_item.get("diagnostics") or {})
            child_diagnostics = dict(child_item.get("diagnostics") or {})
            if collection == "generation":
                parent_diagnostics.update(parent_generation_diagnostics.get(item_id, {}))
                child_diagnostics.update(child_generation_diagnostics.get(item_id, {}))
            samples.append(
                {
                    **dict(selection),
                    "collection": collection,
                    "parent_output": parent_output,
                    "child_output": child_output,
                    "parent_diagnostics": parent_diagnostics,
                    "child_diagnostics": child_diagnostics,
                    "verbatim_outputs_included": True,
                }
            )
    expected = _positive_int(required_count, "manual_sample_count")
    if len(samples) != expected or int(selected.get("selected_count", -1)) != expected:
        raise QwenCompensatedRunnerError(
            f"Manual-review cardinality is not exact: {len(samples)}!={expected}."
        )
    packet = {
        "schema": "neural-forge-qualitative-parent-child-review-v1",
        "status": "awaiting-explicit-review",
        "required_sample_count": expected,
        "selected_sample_count": len(samples),
        "samples": samples,
        "sample_ids_sha256": _canonical_sha256(
            [(item["collection"], item["id"]) for item in samples]
        ),
        "review_contract": {
            "human_or_explicit_agent_inspection_required": True,
            "reviewer_must_inspect_parent_and_child_text": True,
            "automated_metrics_are_not_a_review_verdict": True,
            "positive_verdict_required_before_publication": True,
        },
    }
    packet["packet_sha256"] = _canonical_sha256(packet)
    return packet


def validate_manual_review_verdict(
    verdict: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    if verdict.get("schema") != "neural-forge-qualitative-review-verdict-v1":
        raise QwenCompensatedRunnerError("Manual-review verdict schema is invalid.")
    if verdict.get("verdict") != "pass":
        raise QwenCompensatedRunnerError("Manual review did not pass.")
    reviewer = str(verdict.get("reviewer") or "").strip()
    if not reviewer:
        raise QwenCompensatedRunnerError("Manual-review verdict lacks a reviewer.")
    if verdict.get("packet_sha256") != packet.get("packet_sha256"):
        raise QwenCompensatedRunnerError(
            "Manual-review verdict is bound to a different packet."
        )
    if int(verdict.get("reviewed_sample_count", -1)) != int(
        packet["selected_sample_count"]
    ):
        raise QwenCompensatedRunnerError(
            "Manual-review verdict did not cover every selected sample."
        )
    if not isinstance(verdict.get("notes"), str) or not verdict["notes"].strip():
        raise QwenCompensatedRunnerError("Manual-review verdict requires notes.")
    return {
        "schema": str(verdict["schema"]),
        "verdict": "pass",
        "reviewer": reviewer,
        "packet_sha256": str(packet["packet_sha256"]),
        "reviewed_sample_count": int(verdict["reviewed_sample_count"]),
        "notes": str(verdict["notes"]),
        "validated": True,
    }


def wait_for_manual_review(
    path: Path,
    packet: Mapping[str, Any],
    *,
    timeout_seconds: int,
    poll_seconds: float = 2.0,
    resource_guard: WorkflowResourceGuard | None = None,
) -> dict[str, Any]:
    timeout = _nonnegative_int(timeout_seconds, "manual_review_timeout_seconds")
    started = time.monotonic()
    while True:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise QwenCompensatedRunnerError(
                    "Manual-review verdict file is unreadable."
                ) from exc
            if not isinstance(value, Mapping):
                raise QwenCompensatedRunnerError(
                    "Manual-review verdict must be a JSON object."
                )
            return validate_manual_review_verdict(value, packet)
        if time.monotonic() - started >= timeout:
            raise QwenCompensatedRunnerError(
                "Explicit manual review is required before checkpoint publication."
            )
        if resource_guard is not None:
            resource_guard.check("awaiting-manual-review")
        time.sleep(max(0.05, float(poll_seconds)))


def seal_checkpoint_certification(
    directory: Path, certificate: Mapping[str, Any]
) -> dict[str, Any]:
    source = directory.expanduser().resolve()
    manifest_path = source / "manifest.json"
    try:
        envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenCompensatedRunnerError("Cannot seal checkpoint certification.") from exc
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise QwenCompensatedRunnerError("Checkpoint payload is invalid during sealing.")
    _write_json(source / "certification.json", dict(certificate))
    files = _checkpoint_files(source)
    payload["files"] = files
    payload["artifact_accounting"]["non_manifest_file_bytes"] = sum(
        int(item["bytes"]) for item in files
    )
    payload["certification"] = {
        "required": True,
        "file": "certification.json",
        "sha256": _file_sha256(source / "certification.json"),
        "status": str(certificate.get("status")),
    }
    sealed = _write_stable_manifest(source, payload)
    _verify_checkpoint_files(source, sealed["payload"])
    tree = artifact_tree_report(source)
    if tree["total_bytes"] != sealed["payload"]["artifact_accounting"][
        "artifact_total_bytes"
    ]:
        raise QwenCompensatedRunnerError(
            "Certified artifact byte accounting is not exact."
        )
    return {"manifest": sealed, "tree": tree}


def publish_certified_checkpoint(
    staged_checkpoint: Path,
    published_checkpoint: Path,
    *,
    lock_path: Path,
) -> dict[str, Any]:
    certificate_path = staged_checkpoint / "certification.json"
    try:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        envelope = json.loads(
            (staged_checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenCompensatedRunnerError(
            "Checkpoint is not sealed for certified publication."
        ) from exc
    if (
        certificate.get("schema") != CERTIFICATION_SCHEMA
        or certificate.get("status") != "certified-zero-recovery"
        or certificate.get("certification_gate", {}).get("passed") is not True
        or certificate.get("manual_review", {}).get("verdict") != "pass"
    ):
        raise QwenCompensatedRunnerError(
            "Checkpoint certification is absent, incomplete, or failed."
        )
    payload = envelope.get("payload") or {}
    certification = payload.get("certification") or {}
    if (
        certification.get("status") != "certified-zero-recovery"
        or certification.get("sha256") != _file_sha256(certificate_path)
    ):
        raise QwenCompensatedRunnerError(
            "Checkpoint manifest does not bind its certification marker."
        )
    _verify_checkpoint_files(staged_checkpoint, payload)
    report = publish_staged_checkpoint(
        staged_checkpoint, published_checkpoint, lock_path=lock_path
    )
    report["certified_only"] = True
    report["certification_sha256"] = certification["sha256"]
    return report


def _full_ids(contract: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    ids = contract["prompt_ids"]
    return {
        "behavior": tuple(ids["behavior_probes"]),
        "choice": tuple(ids["choice_probes"]),
        "generation": tuple(ids["generation_prompts"]),
    }


class SelectionMilestoneGate:
    def __init__(
        self,
        *,
        parent_report: Mapping[str, Any],
        contract: Mapping[str, Any],
        thresholds: GateThresholds,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
        eval_suite: Path,
        max_text_evals: int,
        max_choice_probes: int,
        max_generation_prompts: int,
        max_behavior_probes: int,
        output_directory: Path,
        split_provenance_sha256: str,
        evaluator: Callable[..., dict[str, Any]] = evaluate_loaded_text_model,
    ) -> None:
        self.parent_report = parent_report
        self.contract = contract
        self.ids = _full_ids(contract)
        self.thresholds = thresholds
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.eval_suite = eval_suite
        self.max_text_evals = max_text_evals
        self.max_choice_probes = max_choice_probes
        self.max_generation_prompts = max_generation_prompts
        self.max_behavior_probes = max_behavior_probes
        self.output_directory = output_directory
        self.split_provenance_sha256 = str(split_provenance_sha256)
        if len(self.split_provenance_sha256) != 64:
            raise QwenCompensatedRunnerError(
                "Selection milestone requires bound split provenance."
            )
        if (parent_report.get("generation_health") or {}).get(
            "contract"
        ) != GENERATION_HEALTH_CONTRACT:
            raise QwenCompensatedRunnerError(
                "Selection milestone parent lacks strict generation-health evidence."
            )
        self.evaluator = evaluator
        self.parent_evaluation_calls = 0
        self.candidate_evaluation_calls = 0

    def __call__(
        self, model: nn.Module, attempt: int, proposal: LayerProposal
    ) -> tuple[bool, dict[str, Any]]:
        self.candidate_evaluation_calls += 1
        work = self.output_directory / f".milestone-{attempt:06d}.work.json"
        final = self.output_directory / f"milestone-{attempt:06d}-gate.json"
        try:
            candidate = self.evaluator(
                model,
                self.tokenizer,
                work,
                model_label=f"compensated-fold-selection-milestone:{attempt}",
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                eval_suite=self.eval_suite,
                max_text_evals=self.max_text_evals,
                max_choice_probes=self.max_choice_probes,
                max_generation_prompts=self.max_generation_prompts,
                max_behavior_probes=self.max_behavior_probes,
                min_continuation_chars=GENERATION_HEALTH_CONTRACT[
                    "min_continuation_chars"
                ],
                min_unique_token_ratio=GENERATION_HEALTH_CONTRACT[
                    "min_unique_token_ratio"
                ],
            )
            codebook_runner._validate_eval_report_contract(
                candidate, self.contract, "Selection milestone candidate evaluation"
            )
            if (candidate.get("generation_health") or {}).get(
                "contract"
            ) != GENERATION_HEALTH_CONTRACT:
                raise QwenCompensatedRunnerError(
                    "Selection milestone candidate lacks strict generation evidence."
                )
            verdict = score_cached_parent_gate(
                self.parent_report,
                candidate,
                behavior_ids=self.ids["behavior"],
                choice_ids=self.ids["choice"],
                generation_ids=self.ids["generation"],
                thresholds=self.thresholds,
            )
            payload = {
                "attempt": attempt,
                "proposal": proposal.summary(),
                "candidate": candidate,
                "verdict": verdict,
                "parent_evaluation_calls": 0,
                "candidate_evaluation_calls": self.candidate_evaluation_calls,
                "split_provenance_sha256": self.split_provenance_sha256,
                "selection_data_adaptively_reused": True,
                "certification_candidate_evaluation_calls": 0,
            }
            _write_json(final, payload)
            compact = {
                "attempt": attempt,
                "verdict": verdict,
                "artifact": str(final),
                "artifact_sha256": _file_sha256(final),
                "parent_evaluation_calls": 0,
                "candidate_evaluation_calls": self.candidate_evaluation_calls,
                "split_provenance_sha256": self.split_provenance_sha256,
                "selection_data_adaptively_reused": True,
                "certification_candidate_evaluation_calls": 0,
            }
            return bool(verdict["passed"]), compact
        finally:
            work.unlink(missing_ok=True)


def _select_disjoint_prompt_sets(
    tokenizer: Any,
    calibration_file: Path,
    selection_eval_suite: Path,
    certification_eval_suite: Path,
    *,
    calibration_count: int,
    validation_count: int,
    max_length: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    from neural_forge.hf_tools import _load_eval_suite

    selection_suite = _load_eval_suite(selection_eval_suite)
    certification_suite = _load_eval_suite(certification_eval_suite)
    combined_eval_suite = {
        "source": "selection-plus-certification-prompt-exclusion-only",
        "text_evals": [
            *selection_suite["text_evals"],
            *certification_suite["text_evals"],
        ],
        "choice_probes": [
            *selection_suite["choice_probes"],
            *certification_suite["choice_probes"],
        ],
        "generation_prompts": [
            *selection_suite["generation_prompts"],
            *certification_suite["generation_prompts"],
        ],
        "behavior_probes": [
            *selection_suite["behavior_probes"],
            *certification_suite["behavior_probes"],
        ],
    }
    all_prompts = codebook_runner.load_calibration_prompts(calibration_file)
    combined, report = codebook_runner.select_token_disjoint_calibration_prompts(
        all_prompts,
        calibration_count + validation_count,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        eval_suite=combined_eval_suite,
        max_length=max_length,
    )
    calibration = combined[:calibration_count]
    validation = combined[calibration_count:]
    cal_ids = {
        codebook_runner._token_ids(tokenizer, value, max_length=max_length)
        for value in calibration
    }
    validation_ids = {
        codebook_runner._token_ids(tokenizer, value, max_length=max_length)
        for value in validation
    }
    if cal_ids & validation_ids:
        raise QwenCompensatedRunnerError(
            "Calibration and validation activation prompts overlap."
        )
    codebook_runner.verify_token_split_disjoint(
        tokenizer, combined, combined_eval_suite, max_length=max_length
    )
    report.update(
        {
            "calibration_count": len(calibration),
            "validation_count": len(validation),
            "calibration_sha256": _canonical_sha256(calibration),
            "validation_sha256": _canonical_sha256(validation),
            "calibration_validation_token_overlap": 0,
            "evaluation_token_overlap": 0,
            "selection_and_certification_excluded": True,
            "all_discovery_selection_certification_splits_disjoint": True,
        }
    )
    return calibration, validation, report


def _validate_args(args: argparse.Namespace, *, require_target: bool = True) -> None:
    for name in (
        "calibration_count",
        "validation_count",
        "calibration_max_length",
        "activation_rows",
        "projection_count",
        "neighbor_window",
        "max_pair_candidates",
        "max_generated_pairs",
        "score_chunk_size",
        "proposal_layer_limit",
        "prefilter_columns",
        "merges_per_transaction",
        "max_candidate_attempts",
        "max_transactions",
        "full_milestone_interval",
        "max_new_tokens",
        "max_text_evals",
        "max_choice_probes",
        "max_generation_prompts",
        "max_behavior_probes",
        "manual_sample_count",
        "max_host_activation_bytes",
        "max_cuda_allocated_bytes",
        "max_cuda_reserved_bytes",
        "gate_text_count",
        "gate_behavior_count",
        "gate_choice_count",
        "gate_generation_count",
        "selection_max_text_evals",
        "selection_max_choice_probes",
        "selection_max_generation_prompts",
        "selection_max_behavior_probes",
        "max_host_rss_bytes",
        "max_wall_seconds",
        "max_history_records",
        "max_history_json_bytes",
        "max_artifact_bytes",
        "reload_equivalence_generation_tokens",
        "multi_anchor_candidate_pool_size",
        "multi_anchor_max_anchors",
        "multi_anchor_similarity_chunk_size",
        "multi_anchor_max_similarity_products",
        "multi_anchor_max_projection_fits",
    ):
        _positive_int(getattr(args, name), name)
    for name in (
        "minimum_anchor_count",
        "max_gate_behavior_losses",
        "max_gate_choice_losses",
        "max_gate_generation_losses",
        "max_full_behavior_losses",
        "max_full_choice_losses",
        "max_full_generation_losses",
        "manual_review_timeout_seconds",
    ):
        _nonnegative_int(getattr(args, name), name)
    _finite_fraction(args.anchor_fraction, "anchor_fraction")
    for name in ("multi_anchor_ridge", "multi_anchor_max_coefficient_l1"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise QwenCompensatedRunnerError(f"{name} must be finite and positive.")
    if require_target and all(
        value is None
        for value in (
            args.target_unique_parameters,
            args.target_unique_bytes,
            args.target_resident_tensor_bytes,
        )
    ):
        raise QwenCompensatedRunnerError(
            "A resident parameter or tensor target is required."
        )
    for name in (
        "target_unique_parameters",
        "target_unique_bytes",
        "target_resident_tensor_bytes",
        "target_artifact_bytes",
    ):
        value = getattr(args, name)
        if value is not None:
            _positive_int(value, name)
    if args.max_behavior_probes < 1000 or args.max_generation_prompts < 1000:
        raise QwenCompensatedRunnerError(
            "Production final evaluation requires at least 1000 behavior and 1000 generation prompts."
        )
    if args.max_choice_probes < 250:
        raise QwenCompensatedRunnerError(
            "Production final evaluation requires at least 250 choice probes."
        )
    if not str(args.source_revision).strip():
        raise QwenCompensatedRunnerError("--source-revision is required.")


def _load_source_runtime(args: argparse.Namespace) -> tuple[nn.Module, Any, str]:
    import transformers

    tokenizer_name = args.source_tokenizer or args.source_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=args.source_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.source_model,
        revision=args.source_revision,
        torch_dtype=codebook_runner._dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    model.eval()
    _assert_model_safe(model, "loaded exact source")
    return model, tokenizer, str(tokenizer_name)


def prepare_bound_parent_caches(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the exact parent once and seal reusable selection/certification caches."""

    _validate_args(args, require_target=False)
    certification_contract = _suite_contract_for(
        args,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )
    _assert_production_certification_contract(
        certification_contract, manual_sample_count=args.manual_sample_count
    )
    selection_contract = _suite_contract_for(
        args,
        eval_suite=args.selection_eval_suite,
        max_text_evals=args.selection_max_text_evals,
        max_choice_probes=args.selection_max_choice_probes,
        max_generation_prompts=args.selection_max_generation_prompts,
        max_behavior_probes=args.selection_max_behavior_probes,
    )
    _assert_selection_contract(
        selection_contract,
        gate_behavior_count=args.gate_behavior_count,
        gate_choice_count=args.gate_choice_count,
        gate_generation_count=args.gate_generation_count,
    )
    for path in (args.cached_parent_eval, args.cached_parent_selection_eval):
        if path.exists() and not args.overwrite_parent_caches:
            raise FileExistsError(
                f"Refusing to overwrite parent cache without explicit opt-in: {path}."
            )
    guard = WorkflowResourceGuard(
        args.device,
        args.max_host_rss_bytes,
        args.max_cuda_allocated_bytes,
        args.max_cuda_reserved_bytes,
        args.max_wall_seconds,
        {},
    )
    with guard.phase("parent-cache-load"):
        model, tokenizer, tokenizer_name = _load_source_runtime(args)
    with guard.phase("parent-cache-identity"):
        certification_identity = build_exact_parent_identity(
            model,
            tokenizer,
            source_model=str(args.source_model),
            source_revision=str(args.source_revision),
            source_tokenizer=tokenizer_name,
            eval_suite=args.eval_suite,
        )
        selection_identity = derive_parent_identity_for_eval_suite(
            certification_identity, args.selection_eval_suite
        )
    with guard.phase("parent-cache-certification-eval"):
        certification = evaluate_loaded_text_model(
            model,
            tokenizer,
            args.cached_parent_eval,
            model_label=str(args.source_model),
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            eval_suite=args.eval_suite,
            max_text_evals=args.max_text_evals,
            max_choice_probes=args.max_choice_probes,
            max_generation_prompts=args.max_generation_prompts,
            max_behavior_probes=args.max_behavior_probes,
            min_continuation_chars=GENERATION_HEALTH_CONTRACT[
                "min_continuation_chars"
            ],
            min_unique_token_ratio=GENERATION_HEALTH_CONTRACT[
                "min_unique_token_ratio"
            ],
            identity_binding=certification_identity,
        )
    with guard.phase("parent-cache-selection-eval"):
        selection = evaluate_loaded_text_model(
            model,
            tokenizer,
            args.cached_parent_selection_eval,
            model_label=str(args.source_model),
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            eval_suite=args.selection_eval_suite,
            max_text_evals=args.selection_max_text_evals,
            max_choice_probes=args.selection_max_choice_probes,
            max_generation_prompts=args.selection_max_generation_prompts,
            max_behavior_probes=args.selection_max_behavior_probes,
            min_continuation_chars=GENERATION_HEALTH_CONTRACT[
                "min_continuation_chars"
            ],
            min_unique_token_ratio=GENERATION_HEALTH_CONTRACT[
                "min_unique_token_ratio"
            ],
            identity_binding=selection_identity,
        )
    codebook_runner._validate_eval_report_contract(
        certification, certification_contract, "Prepared certification parent"
    )
    codebook_runner._validate_eval_report_contract(
        selection, selection_contract, "Prepared selection parent"
    )
    return {
        "schema": "neural-forge-bound-parent-cache-preparation-v1",
        "source_model": str(args.source_model),
        "source_revision": str(args.source_revision),
        "certification_cache": str(args.cached_parent_eval.resolve()),
        "certification_cache_sha256": _file_sha256(args.cached_parent_eval),
        "selection_cache": str(args.cached_parent_selection_eval.resolve()),
        "selection_cache_sha256": _file_sha256(args.cached_parent_selection_eval),
        "certification_identity_sha256": certification_identity["identity_sha256"],
        "selection_identity_sha256": selection_identity["identity_sha256"],
        "resource_phases": guard.phases,
        "parent_forward_cache_built_once": True,
        "zero_optimizer_steps": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output = args.output_dir.expanduser().resolve()
    if (output / "run_report.json").exists():
        raise FileExistsError(f"Output is already finalized: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_stage(output, "preflight", "running")
    certification_contract = _suite_contract_for(
        args,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )
    _assert_production_certification_contract(
        certification_contract, manual_sample_count=args.manual_sample_count
    )
    selection_contract = _suite_contract_for(
        args,
        eval_suite=args.selection_eval_suite,
        max_text_evals=args.selection_max_text_evals,
        max_choice_probes=args.selection_max_choice_probes,
        max_generation_prompts=args.selection_max_generation_prompts,
        max_behavior_probes=args.selection_max_behavior_probes,
    )
    _assert_selection_contract(
        selection_contract,
        gate_behavior_count=args.gate_behavior_count,
        gate_choice_count=args.gate_choice_count,
        gate_generation_count=args.gate_generation_count,
    )
    cached_parent_certification = codebook_runner._load_eval_report(
        args.cached_parent_eval, "cached parent evaluation"
    )
    cached_parent_selection = codebook_runner._load_eval_report(
        args.cached_parent_selection_eval, "cached parent selection evaluation"
    )
    parent_certification_contract = codebook_runner._validate_eval_report_contract(
        cached_parent_certification,
        certification_contract,
        "Cached parent certification evaluation",
    )
    parent_selection_contract = codebook_runner._validate_eval_report_contract(
        cached_parent_selection,
        selection_contract,
        "Cached parent selection evaluation",
    )
    guard = WorkflowResourceGuard(
        args.device,
        args.max_host_rss_bytes,
        args.max_cuda_allocated_bytes,
        args.max_cuda_reserved_bytes,
        args.max_wall_seconds,
        {},
    )

    _write_stage(output, "load_parent", "running", model=args.source_model)
    _progress("load_parent", model=args.source_model)
    with guard.phase("load-parent"):
        model, tokenizer, tokenizer_name = _load_source_runtime(args)
    initial = model_accounting(model)
    with guard.phase("bind-parent-caches"):
        certification_identity = build_exact_parent_identity(
            model,
            tokenizer,
            source_model=str(args.source_model),
            source_revision=str(args.source_revision),
            source_tokenizer=tokenizer_name,
            eval_suite=args.eval_suite,
        )
        selection_identity = derive_parent_identity_for_eval_suite(
            certification_identity, args.selection_eval_suite
        )
        parent_certification_identity = validate_bound_parent_report(
            cached_parent_certification,
            certification_identity,
            label="Cached parent certification evaluation",
        )
        parent_selection_identity = validate_bound_parent_report(
            cached_parent_selection,
            selection_identity,
            label="Cached parent selection evaluation",
        )
    _write_json(
        output / "certification_contract.json",
        {
            "certification": {
                "contract": certification_contract,
                "cached_parent": parent_certification_contract,
                "identity": parent_certification_identity,
                "candidate_evaluation_budget": 1,
            },
            "adaptive_selection": {
                "contract": selection_contract,
                "cached_parent": parent_selection_contract,
                "identity": parent_selection_identity,
            },
        },
    )

    with guard.phase("split-selection-and-provenance"):
        calibration_prompts, validation_prompts, prompt_report = (
            _select_disjoint_prompt_sets(
                tokenizer,
                args.calibration_jsonl,
                args.selection_eval_suite,
                args.eval_suite,
                calibration_count=args.calibration_count,
                validation_count=args.validation_count,
                max_length=args.calibration_max_length,
            )
        )
    _write_json(output / "prompt_selection.json", prompt_report)
    gate_suite, gate_ids, gate_report = build_gate_suite(
        args.selection_eval_suite,
        cached_parent_selection,
        text_count=args.gate_text_count,
        behavior_count=args.gate_behavior_count,
        choice_count=args.gate_choice_count,
        generation_count=args.gate_generation_count,
    )
    _write_json(output / "small_gate_contract.json", gate_report)
    selected_selection_suite = _selected_suite(
        args.selection_eval_suite,
        max_text_evals=args.selection_max_text_evals,
        max_choice_probes=args.selection_max_choice_probes,
        max_generation_prompts=args.selection_max_generation_prompts,
        max_behavior_probes=args.selection_max_behavior_probes,
    )
    selected_certification_suite = _selected_suite(
        args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )
    split_provenance = build_three_split_provenance(
        tokenizer,
        calibration_prompts,
        validation_prompts,
        selected_selection_suite,
        selected_certification_suite,
        calibration_source=args.calibration_jsonl,
        selection_source=args.selection_eval_suite,
        cached_parent_selection_source=args.cached_parent_selection_eval,
        certification_source=args.eval_suite,
        cached_parent_certification_source=args.cached_parent_eval,
        max_length=args.calibration_max_length,
    )
    _write_json(output / "three_split_provenance.json", split_provenance)
    activation_reports: list[dict[str, Any]] = []

    def activation_provider(current: nn.Module) -> ActivationPair:
        phase = f"activation-refresh-{len(activation_reports):06d}"
        with guard.phase(phase):
            pair = collect_activation_pair(
                current,
                tokenizer,
                calibration_prompts,
                validation_prompts,
                max_length=args.calibration_max_length,
                activation_rows=args.activation_rows,
                device=args.device,
                max_host_activation_bytes=args.max_host_activation_bytes,
            )
        activation_reports.append(pair.report)
        _write_json(
            output / "activation_refreshes" / f"refresh-{len(activation_reports) - 1:06d}.json",
            pair.report,
        )
        return pair

    small_gate = CachedParentGateEvaluator(
        parent_report=cached_parent_selection,
        gate_suite=gate_suite,
        gate_ids=gate_ids,
        thresholds=GateThresholds(
            args.max_gate_behavior_losses,
            args.max_gate_choice_losses,
            args.max_gate_generation_losses,
        ),
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        output_directory=output / "transaction_gates",
        split_provenance_sha256=split_provenance["provenance_sha256"],
    )
    milestone_gate = SelectionMilestoneGate(
        parent_report=cached_parent_selection,
        contract=selection_contract,
        thresholds=GateThresholds(
            args.max_full_behavior_losses,
            args.max_full_choice_losses,
            args.max_full_generation_losses,
        ),
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        eval_suite=args.selection_eval_suite,
        max_text_evals=args.selection_max_text_evals,
        max_choice_probes=args.selection_max_choice_probes,
        max_generation_prompts=args.selection_max_generation_prompts,
        max_behavior_probes=args.selection_max_behavior_probes,
        output_directory=output / "milestones",
        split_provenance_sha256=split_provenance["provenance_sha256"],
    )
    calibration_provenance_id = _canonical_sha256(
        {
            "split": split_provenance["calibration"],
            "activation_rows": args.activation_rows,
            "activation_row_selection": "global-even-token-rows",
            "max_length": args.calibration_max_length,
        }
    )
    validation_provenance_id = _canonical_sha256(
        {
            "split": split_provenance["validation"],
            "activation_rows": args.activation_rows,
            "activation_row_selection": "global-even-token-rows",
            "max_length": args.calibration_max_length,
        }
    )
    if calibration_provenance_id == validation_provenance_id:
        raise QwenCompensatedRunnerError(
            "Calibration and validation provenance IDs unexpectedly match."
        )
    loop_config = FoldLoopConfig(
        device=args.device,
        calibration_max_length=args.calibration_max_length,
        activation_rows=args.activation_rows,
        max_host_activation_bytes=args.max_host_activation_bytes,
        anchor_fraction=args.anchor_fraction,
        minimum_anchor_count=args.minimum_anchor_count,
        projection_count=args.projection_count,
        neighbor_window=args.neighbor_window,
        max_pair_candidates=args.max_pair_candidates,
        max_generated_pairs=args.max_generated_pairs,
        score_chunk_size=args.score_chunk_size,
        proposal_layer_limit=args.proposal_layer_limit,
        prefilter_columns=args.prefilter_columns,
        fold_strategy=args.fold_strategy,
        calibration_provenance_id=calibration_provenance_id,
        validation_provenance_id=validation_provenance_id,
        multi_anchor_candidate_pool_size=args.multi_anchor_candidate_pool_size,
        multi_anchor_max_anchors=args.multi_anchor_max_anchors,
        multi_anchor_similarity_chunk_size=args.multi_anchor_similarity_chunk_size,
        multi_anchor_max_similarity_products=(
            args.multi_anchor_max_similarity_products
        ),
        multi_anchor_max_projection_fits=args.multi_anchor_max_projection_fits,
        multi_anchor_ridge=args.multi_anchor_ridge,
        multi_anchor_max_coefficient_l1=args.multi_anchor_max_coefficient_l1,
        merges_per_transaction=args.merges_per_transaction,
        max_candidate_attempts=args.max_candidate_attempts,
        max_transactions=args.max_transactions,
        full_milestone_interval=args.full_milestone_interval,
        max_history_records=args.max_history_records,
        max_history_json_bytes=args.max_history_json_bytes,
        excluded_layers=tuple(args.exclude_layer),
        seed=args.seed,
    )
    _write_stage(output, "transactions", "running")
    transaction_ledger = output / "transaction_ledger.jsonl"
    if transaction_ledger.exists():
        raise FileExistsError(transaction_ledger)
    with guard.phase("transactions"):
        transaction_report = run_fold_transactions(
            model,
            activation_provider=activation_provider,
            small_gate=small_gate,
            milestone_gate=milestone_gate,
            target_parameters=args.target_unique_parameters,
            target_bytes=args.target_unique_bytes,
            target_resident_bytes=args.target_resident_tensor_bytes,
            config=loop_config,
            transaction_ledger=transaction_ledger,
            resource_guard=guard,
        )
    transaction_report["three_split_provenance_sha256"] = split_provenance[
        "provenance_sha256"
    ]
    transaction_report["certification_candidate_evaluation_calls"] = 0
    transaction_report["certification_candidate_evaluation_calls_before_surgery_complete"] = 0
    transaction_report["fold_strategy"] = args.fold_strategy
    transaction_report["activation_provenance"] = {
        "calibration_id": calibration_provenance_id,
        "validation_id": validation_provenance_id,
        "ids_distinct": calibration_provenance_id != validation_provenance_id,
        "work_caps": {
            "candidate_pool_size": args.multi_anchor_candidate_pool_size,
            "max_anchors": args.multi_anchor_max_anchors,
            "similarity_chunk_size": args.multi_anchor_similarity_chunk_size,
            "max_similarity_products": args.multi_anchor_max_similarity_products,
            "max_projection_fits": args.multi_anchor_max_projection_fits,
            "max_coefficient_l1": args.multi_anchor_max_coefficient_l1,
        },
    }
    _write_json(output / "transaction_report.json", transaction_report)
    if not transaction_report["target"]["reached"]:
        _write_stage(
            output,
            "transactions",
            "target-not-reached-no-publication",
            accepted=transaction_report["accepted_count"],
            rejected=transaction_report["rejected_count"],
        )
        raise QwenCompensatedRunnerError(
            "Resident surgery target was not reached; no checkpoint was published."
        )

    _write_stage(output, "standalone_checkpoint", "running")
    checkpoint_dir = output / "ragged_child"
    folded_accounting = model_accounting(model)
    with guard.phase("prepare-local-checkpoint-staging"):
        staging_run_root, staged_checkpoint_dir, staging_report = (
            prepare_local_checkpoint_staging(
                output,
                args.checkpoint_staging_root,
                required_bytes=(
                    folded_accounting.unique_parameter_bytes
                    + folded_accounting.unique_buffer_bytes
                    + 1024**2
                ),
            )
        )
    _write_json(output / "checkpoint_staging.json", staging_report)
    parent_runtime_ref = weakref.ref(model)
    with guard.phase("save-staged-checkpoint"):
        checkpoint_manifest = save_ragged_qwen_checkpoint(
            model,
            tokenizer,
            staged_checkpoint_dir,
            transaction_records=_compact_checkpoint_transactions(
                transaction_report["accepted_transactions"]
            ),
            source_provenance={
                "source_model": str(args.source_model),
                "source_revision": str(args.source_revision),
                "source_tokenizer": str(tokenizer_name),
                "cached_parent_certification_eval": str(args.cached_parent_eval),
                "cached_parent_certification_eval_sha256": _file_sha256(
                    args.cached_parent_eval
                ),
                "cached_parent_selection_eval": str(
                    args.cached_parent_selection_eval
                ),
                "cached_parent_selection_eval_sha256": _file_sha256(
                    args.cached_parent_selection_eval
                ),
                "split_provenance_sha256": split_provenance[
                    "provenance_sha256"
                ],
                "transaction_ledger_sha256": transaction_report["history"][
                    "ledger_sha256"
                ],
            },
        )
    pre_certification_tree = artifact_tree_report(staged_checkpoint_dir)
    if pre_certification_tree["total_bytes"] > args.max_artifact_bytes:
        raise QwenCompensatedRunnerError(
            "Staged artifact exceeds --max-artifact-bytes before certification."
        )
    reload_prompts: list[str] = []
    for value in [
        *calibration_prompts[:2],
        *validation_prompts[:2],
        str(selected_selection_suite["generation_prompts"][0]["prompt"]),
    ]:
        if value not in reload_prompts:
            reload_prompts.append(value)
        if len(reload_prompts) == 3:
            break
    with guard.phase("capture-reload-reference"):
        reload_reference = capture_reload_reference(
            model,
            tokenizer,
            reload_prompts,
            device=args.device,
            max_length=args.calibration_max_length,
            generation_tokens=args.reload_equivalence_generation_tokens,
        )
    del model
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    source_runtime_released = parent_runtime_ref() is None
    if not source_runtime_released:
        raise QwenCompensatedRunnerError(
            "The folded source runtime remains referenced before custom reload."
        )
    with guard.phase("parent-free-ragged-reload"):
        reloaded, reloaded_tokenizer, loaded_manifest = load_ragged_qwen_checkpoint(
            staged_checkpoint_dir, device=args.device
        )
    if reloaded_tokenizer is None:
        raise QwenCompensatedRunnerError(
            "Standalone production checkpoint did not reload its tokenizer."
        )
    reload_equivalence = verify_reload_equivalence(
        reload_reference,
        reloaded,
        reloaded_tokenizer,
        device=args.device,
    )

    _write_stage(output, "untouched_certification", "running")
    final_work = output / ".final_eval.work.json"
    with guard.phase("untouched-certification-evaluation"):
        final_eval = evaluate_loaded_text_model(
            reloaded,
            reloaded_tokenizer,
            final_work,
            model_label="qwen3-compensated-ragged-parent-free",
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            eval_suite=args.eval_suite,
            max_text_evals=args.max_text_evals,
            max_choice_probes=args.max_choice_probes,
            max_generation_prompts=args.max_generation_prompts,
            max_behavior_probes=args.max_behavior_probes,
            min_continuation_chars=GENERATION_HEALTH_CONTRACT[
                "min_continuation_chars"
            ],
            min_unique_token_ratio=GENERATION_HEALTH_CONTRACT[
                "min_unique_token_ratio"
            ],
        )
    final_contract = codebook_runner._validate_eval_report_contract(
        final_eval, certification_contract, "Final parent-free certification"
    )
    if (final_eval.get("generation_health") or {}).get(
        "contract"
    ) != GENERATION_HEALTH_CONTRACT:
        raise QwenCompensatedRunnerError(
            "Final candidate lacks strict generation-health evidence."
        )
    final_eval_path = output / "final_eval.json"
    _write_json(final_eval_path, final_eval)
    final_work.unlink(missing_ok=True)
    with guard.phase("parent-relative-comparison"):
        final_comparison = codebook_runner._compare_evals(
            args.cached_parent_eval,
            final_eval_path,
            manual_sample_count=args.manual_sample_count,
        )
        _write_json(output / "parent_final_comparison.json", final_comparison)
    certification_verdict = score_cached_parent_gate(
        cached_parent_certification,
        final_eval,
        behavior_ids=_full_ids(certification_contract)["behavior"],
        choice_ids=_full_ids(certification_contract)["choice"],
        generation_ids=_full_ids(certification_contract)["generation"],
        thresholds=GateThresholds(
            args.max_full_behavior_losses,
            args.max_full_choice_losses,
            args.max_full_generation_losses,
        ),
    )
    certification_verdict["candidate_evaluation_calls"] = 1
    certification_verdict["candidate_evaluation_budget"] = 1
    certification_verdict["selection_uses_certification"] = False
    _write_json(output / "certification_verdict.json", certification_verdict)
    if not certification_verdict["passed"]:
        _write_stage(output, "untouched_certification", "failed-no-publication")
        raise QwenCompensatedRunnerError(
            "Untouched parent-relative certification failed; no checkpoint was published."
        )

    with guard.phase("manual-review-packet"):
        manual_packet = build_manual_review_packet(
            cached_parent_certification,
            final_eval,
            final_comparison,
            required_count=args.manual_sample_count,
        )
    manual_packet_path = output / "manual_review_packet.json"
    _write_json(manual_packet_path, manual_packet)
    verdict_path = (
        args.manual_review_verdict_file
        if args.manual_review_verdict_file is not None
        else output / "manual_review_verdict.json"
    )
    _write_json(
        output / "manual_review_verdict.template.json",
        {
            "schema": "neural-forge-qualitative-review-verdict-v1",
            "verdict": "replace-with-pass-or-fail",
            "reviewer": "replace-with-reviewer-identity",
            "packet_sha256": manual_packet["packet_sha256"],
            "reviewed_sample_count": manual_packet["selected_sample_count"],
            "notes": "replace-with-concrete-observations-from-parent-and-child-text",
        },
    )
    _write_stage(
        output,
        "manual_review",
        "awaiting-explicit-verdict",
        packet=str(manual_packet_path),
        packet_sha256=manual_packet["packet_sha256"],
        verdict_file=str(verdict_path),
    )
    final_accounting = model_accounting(reloaded)
    del reloaded
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    manual_verdict = wait_for_manual_review(
        verdict_path,
        manual_packet,
        timeout_seconds=args.manual_review_timeout_seconds,
        resource_guard=guard,
    )
    _write_json(output / "validated_manual_review.json", manual_verdict)

    certificate: dict[str, Any] = {
        "schema": CERTIFICATION_SCHEMA,
        "status": "pending-exact-artifact-accounting",
        "zero_recovery": True,
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "resident_target": transaction_report["target"],
        "artifact_target_bytes": args.target_artifact_bytes,
        "artifact_total_bytes": 0,
        "certification_gate": certification_verdict,
        "certification_eval_sha256": _file_sha256(final_eval_path),
        "cached_parent_certification_sha256": _file_sha256(args.cached_parent_eval),
        "comparison_sha256": _file_sha256(output / "parent_final_comparison.json"),
        "manual_review": manual_verdict,
        "reload_equivalence": reload_equivalence,
        "pre_certification_tree_sha256": pre_certification_tree["tree_sha256"],
        "source_identity_sha256": certification_identity["identity_sha256"],
        "certification_candidate_evaluation_calls": 1,
        "certification_candidate_evaluation_calls_before_surgery_complete": 0,
    }
    sealed: dict[str, Any] | None = None
    with guard.phase("seal-pending-certification"):
        for _attempt in range(8):
            sealed = seal_checkpoint_certification(staged_checkpoint_dir, certificate)
            actual_artifact_bytes = int(sealed["tree"]["total_bytes"])
            if certificate["artifact_total_bytes"] == actual_artifact_bytes:
                break
            certificate["artifact_total_bytes"] = actual_artifact_bytes
        else:
            raise QwenCompensatedRunnerError(
                "Certified artifact accounting did not reach a stable fixed point."
            )
    assert sealed is not None
    artifact_total_bytes = int(sealed["tree"]["total_bytes"])
    if artifact_total_bytes > args.max_artifact_bytes:
        raise QwenCompensatedRunnerError(
            "Certified artifact exceeds --max-artifact-bytes; no publication occurred."
        )
    if (
        args.target_artifact_bytes is not None
        and artifact_total_bytes > args.target_artifact_bytes
    ):
        raise QwenCompensatedRunnerError(
            "Serialized artifact target was missed after exact sealing; no publication occurred."
        )
    certificate["status"] = "certified-zero-recovery"
    certificate["artifact_total_bytes"] = artifact_total_bytes
    with guard.phase("seal-final-certification"):
        for _attempt in range(8):
            sealed = seal_checkpoint_certification(staged_checkpoint_dir, certificate)
            actual_artifact_bytes = int(sealed["tree"]["total_bytes"])
            if certificate["artifact_total_bytes"] == actual_artifact_bytes:
                break
            certificate["artifact_total_bytes"] = actual_artifact_bytes
        else:
            raise QwenCompensatedRunnerError(
                "Final certified artifact accounting did not stabilize."
            )
    artifact_total_bytes = int(sealed["tree"]["total_bytes"])
    if artifact_total_bytes > args.max_artifact_bytes or (
        args.target_artifact_bytes is not None
        and artifact_total_bytes > args.target_artifact_bytes
    ):
        raise QwenCompensatedRunnerError(
            "Final certification metadata exceeded the artifact target."
        )

    checkpoint_lock = output.parent / ".compact_checkpoint_io.lock"
    _write_stage(output, "publication", "certified-publish-running")
    with guard.phase("certified-publication"):
        publish_report = publish_certified_checkpoint(
            staged_checkpoint_dir,
            checkpoint_dir,
            lock_path=checkpoint_lock,
        )
    staging_removed = False
    if not args.keep_checkpoint_staging:
        remove_checkpoint_staging(staging_run_root, Path(staging_report["path"]))
        staging_removed = True

    report = {
        "schema": RUN_SCHEMA,
        "status": "complete-target-reached-certified-published",
        "source": {
            "model": str(args.source_model),
            "revision": str(args.source_revision),
            "tokenizer": str(tokenizer_name),
            "initial_accounting": initial.as_dict(),
            "parent_evaluation_calls": 0,
            "cached_parent_certification": parent_certification_contract,
            "cached_parent_selection": parent_selection_contract,
            "certification_identity": parent_certification_identity,
            "selection_identity": parent_selection_identity,
        },
        "prompt_selection": prompt_report,
        "three_split_provenance": split_provenance,
        "small_gate_contract": gate_report,
        "transactions": transaction_report,
        "activation_refreshes": activation_reports,
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "saved_manifest": checkpoint_manifest,
            "loaded_manifest_sha256_before_certification": loaded_manifest[
                "payload_sha256"
            ],
            "loader_runtime": loaded_manifest["runtime_load"],
            "reload_equivalence": reload_equivalence,
            "certification": certificate,
            "certified_tree": sealed["tree"],
            "artifact_target_bytes": args.target_artifact_bytes,
            "artifact_target_reached": (
                args.target_artifact_bytes is None
                or artifact_total_bytes <= args.target_artifact_bytes
            ),
            "staging": {
                **staging_report,
                "retained": not staging_removed,
                "removed_after_verified_publish": staging_removed,
            },
            "publish": publish_report,
            "source_runtime_released": source_runtime_released,
            "standalone_parent_free": True,
            "stock_huggingface_reload_claimed": False,
        },
        "final": {
            "accounting": final_accounting.as_dict(),
            "eval": final_contract,
            "comparison": final_comparison,
            "manual_samples": manual_packet["samples"],
            "manual_review_packet": str(manual_packet_path),
            "manual_review": manual_verdict,
            "certification_verdict": certification_verdict,
            "certification_candidate_evaluation_calls": 1,
        },
        "runtime": {
            "device": args.device,
            "dtype": args.dtype,
            "resource_phases": guard.phases,
            "torch_version": torch.__version__,
        },
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "recovery_examples": 0,
        "parent_outputs_recomputed": False,
        "certification_suite_used_for_adaptive_selection": False,
        "published_only_after_certification": True,
    }
    _write_json(output / "run_report.json", report)
    _write_stage(
        output,
        "complete",
        "target-reached-certified-published",
        accepted=transaction_report["accepted_count"],
        rejected=transaction_report["rejected_count"],
    )
    return report


def _parse_layers(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Layer indices must be non-negative.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-tokenizer")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-eval", type=Path, required=True)
    parser.add_argument("--selection-eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-selection-eval", type=Path, required=True)
    parser.add_argument("--prepare-parent-caches", action="store_true")
    parser.add_argument("--overwrite-parent-caches", action="store_true")
    parser.add_argument("--target-unique-parameters", type=int)
    parser.add_argument("--target-unique-bytes", type=int)
    parser.add_argument("--target-resident-tensor-bytes", type=int)
    parser.add_argument("--target-artifact-bytes", type=int)
    parser.add_argument(
        "--checkpoint-staging-root",
        type=Path,
        help=(
            "Local filesystem used to save and reload the large artifact before "
            "serialized publication to the output filesystem. Defaults to "
            "NEURAL_FORGE_CHECKPOINT_STAGING_ROOT or the system temp directory."
        ),
    )
    parser.add_argument("--keep-checkpoint-staging", action="store_true")
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--calibration-max-length", type=int, default=160)
    parser.add_argument("--activation-rows", type=int, default=64)
    parser.add_argument("--anchor-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-anchor-count", type=int, default=8)
    parser.add_argument("--projection-count", type=int, default=16)
    parser.add_argument("--neighbor-window", type=int, default=8)
    parser.add_argument("--max-pair-candidates", type=int, default=131_072)
    parser.add_argument("--max-generated-pairs", type=int, default=524_288)
    parser.add_argument("--score-chunk-size", type=int, default=1_024)
    parser.add_argument("--proposal-layer-limit", type=int, default=6)
    parser.add_argument("--prefilter-columns", type=int, default=256)
    parser.add_argument(
        "--fold-strategy", choices=["pair", "multi-anchor"], default="pair"
    )
    parser.add_argument("--multi-anchor-candidate-pool-size", type=int, default=512)
    parser.add_argument("--multi-anchor-max-anchors", type=int, default=4)
    parser.add_argument("--multi-anchor-similarity-chunk-size", type=int, default=64)
    parser.add_argument(
        "--multi-anchor-max-similarity-products", type=int, default=16_000_000
    )
    parser.add_argument("--multi-anchor-max-projection-fits", type=int, default=131_072)
    parser.add_argument("--multi-anchor-ridge", type=float, default=1e-6)
    parser.add_argument("--multi-anchor-max-coefficient-l1", type=float, default=16.0)
    parser.add_argument("--merges-per-transaction", type=int, default=1)
    parser.add_argument("--max-candidate-attempts", type=int, default=32)
    parser.add_argument("--max-transactions", type=int, default=4_096)
    parser.add_argument("--full-milestone-interval", type=int, default=32)
    parser.add_argument("--max-history-records", type=int, default=256)
    parser.add_argument("--max-history-json-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--exclude-layer", action="append", type=_parse_layers, default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-text-count", type=int, default=8)
    parser.add_argument("--gate-behavior-count", type=int, default=32)
    parser.add_argument("--gate-choice-count", type=int, default=24)
    parser.add_argument("--gate-generation-count", type=int, default=32)
    parser.add_argument("--selection-max-text-evals", type=int, default=32)
    parser.add_argument("--selection-max-choice-probes", type=int, default=128)
    parser.add_argument("--selection-max-generation-prompts", type=int, default=256)
    parser.add_argument("--selection-max-behavior-probes", type=int, default=256)
    parser.add_argument("--max-gate-behavior-losses", type=int, default=0)
    parser.add_argument("--max-gate-choice-losses", type=int, default=0)
    parser.add_argument("--max-gate-generation-losses", type=int, default=0)
    parser.add_argument("--max-full-behavior-losses", type=int, default=0)
    parser.add_argument("--max-full-choice-losses", type=int, default=0)
    parser.add_argument("--max-full-generation-losses", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-text-evals", type=int, default=1000)
    parser.add_argument("--max-choice-probes", type=int, default=250)
    parser.add_argument("--max-generation-prompts", type=int, default=1000)
    parser.add_argument("--max-behavior-probes", type=int, default=1000)
    parser.add_argument("--manual-sample-count", type=int, default=128)
    parser.add_argument("--manual-review-verdict-file", type=Path)
    parser.add_argument("--manual-review-timeout-seconds", type=int, default=0)
    parser.add_argument("--reload-equivalence-generation-tokens", type=int, default=8)
    parser.add_argument("--max-host-activation-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--max-host-rss-bytes", type=int, default=256 * 1024**3)
    parser.add_argument("--max-cuda-allocated-bytes", type=int, default=80 * 1024**3)
    parser.add_argument("--max-cuda-reserved-bytes", type=int, default=80 * 1024**3)
    parser.add_argument("--max-wall-seconds", type=int, default=172_800)
    parser.add_argument("--max-artifact-bytes", type=int, default=32 * 1024**3)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepare_parent_caches:
        report = prepare_bound_parent_caches(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "parent_cache_preparation.json", report)
        return 0
    report = run(args)
    return 0 if report["status"] == "complete-target-reached-certified-published" else 2


if __name__ == "__main__":
    raise SystemExit(main())
