#!/usr/bin/env python3
"""Run the sealed-cache exact-4x LFM2 C/S/M causal diagnostic."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

import neural_forge.hf_lfm2_exact4x_causal_diagnostic as causal_module
import neural_forge.hf_lfm2_moving_gauge_residual_reclosure as moving_gauge_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
    PhaseAnchorAuthority,
    load_existing_lfm2_circuit_balanced_basis_phase,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_model
from neural_forge.hf_lfm2_exact4x_causal_diagnostic import (
    Lfm2CausalOperatorSpec,
    Lfm2Exact4xCausalDiagnosticError,
    canonical_intervention_subsets,
    canonical_operator_specs,
    default_lfm2_l0_csm_operator_specs,
)
from neural_forge.hf_lfm2_exact4x_first_divergence import (
    authoritative_split_by_batch,
    parent_selection_logical_bytes,
    phase_names,
    verify_complete_exact_budget_child,
)
from neural_forge.lfm2_carrier_bridge_publication import (
    fingerprint_lfm2_carrier_runtime_owned_tensor_state,
    relocate_and_bind_packaged_lfm2_carrier_routes,
)
from neural_forge.lfm2_carrier_child_package import (
    load_authenticated_lfm2_carrier_child_package,
)

try:
    from experiments import audit_lfm2_exact4x_first_divergence as v3_runner
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import audit_lfm2_exact4x_first_divergence as v3_runner


RUNNER_SCHEMA = "neural-forge-lfm2-exact4x-causal-diagnostic-runner-v4"
IMPLEMENTATION_SCHEMA = "neural-forge-lfm2-exact4x-causal-diagnostic-implementation-v1"
V3_IMPLEMENTATION_RELOCATION_SCHEMA = (
    "neural-forge-lfm2-exact4x-v3-implementation-relocation-v1"
)
FEATURE_RECEIPT_SCHEMA = "neural-forge-lfm2-exact4x-causal-feature-receipt-v1"
COMPLETION_SCHEMA = "neural-forge-lfm2-exact4x-causal-completion-v1"
PARTIAL_COMPLETION_SCHEMA = "neural-forge-lfm2-exact4x-causal-partial-completion-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-exact4x-causal-progress-v1"


class Lfm2Exact4xCausalRunnerError(RuntimeError):
    """The causal runner input, replay, or local artifact is invalid."""


class _ProgressJournal:
    def __init__(self, path: Path, request_sha256: str) -> None:
        self.path = path
        self.request_sha256 = _digest(request_sha256, "request_sha256")
        self.sequence = 0
        self.previous = "0" * 64
        if path.exists():
            if (
                path.is_symlink()
                or not path.is_file()
                or getattr(path.stat(), "st_nlink", 1) != 1
            ):
                raise Lfm2Exact4xCausalRunnerError(
                    "Causal progress journal is link-like or not a file."
                )
            payload = path.read_bytes()
            if payload and not payload.endswith(b"\n"):
                raise Lfm2Exact4xCausalRunnerError(
                    "Causal progress journal has an unterminated event."
                )
            for index, raw in enumerate(payload.splitlines()):
                try:
                    value = json.loads(raw.decode("ascii"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise Lfm2Exact4xCausalRunnerError(
                        "Causal progress journal is malformed."
                    ) from exc
                body = dict(value)
                claimed = body.pop("event_sha256", None)
                if (
                    value.get("schema") != PROGRESS_SCHEMA
                    or value.get("request_sha256") != self.request_sha256
                    or value.get("sequence") != index
                    or value.get("previous_event_sha256") != self.previous
                    or claimed != canonical_sha256(body)
                    or raw != _canonical_bytes(value)
                ):
                    raise Lfm2Exact4xCausalRunnerError(
                        "Causal progress journal hash chain does not verify."
                    )
                self.sequence = index + 1
                self.previous = str(claimed)

    def emit(self, event: str, **details: Any) -> None:
        body = {
            "schema": PROGRESS_SCHEMA,
            "request_sha256": self.request_sha256,
            "sequence": self.sequence,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "event": event,
            "previous_event_sha256": self.previous,
            "details": json.loads(_canonical_bytes(details).decode("ascii")),
        }
        row = {**body, "event_sha256": canonical_sha256(body)}
        payload = _canonical_bytes(row) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        sys.stdout.write(payload.decode("ascii"))
        sys.stdout.flush()
        self.sequence += 1
        self.previous = row["event_sha256"]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xCausalRunnerError(
            "Causal runner evidence must be finite canonical JSON."
        ) from exc


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact4xCausalRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a directory-entry transition on POSIX evidence hosts."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_safetensors(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
) -> None:
    """Publish one immutable ordinary safetensors file without replacement."""

    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xCausalRunnerError(
            f"Refusing to overwrite causal feature artifact {path}."
        )
    if not tensors:
        raise Lfm2Exact4xCausalRunnerError("A causal feature artifact cannot be empty.")
    canonical: dict[str, torch.Tensor] = {}
    for name, value in sorted(tensors.items()):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.is_sparse
            or value.is_complex()
            or not bool(torch.isfinite(value.detach()).all())
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"Causal feature tensor {name!r} is invalid."
            )
        canonical[name] = value.detach().to(device="cpu").contiguous().clone()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        safetensors.torch.save_file(canonical, str(temporary))
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise Lfm2Exact4xCausalRunnerError(
                f"Causal feature artifact appeared during publication: {path}."
            )
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError as exc:
            raise Lfm2Exact4xCausalRunnerError(
                f"Causal feature artifact appeared during publication: {path}."
            ) from exc
        except OSError as exc:
            raise Lfm2Exact4xCausalRunnerError(
                f"Could not publish causal feature artifact {path}."
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _build_feature_receipt(
    *,
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    request_sha256: str,
    batch_index: int,
    trace_input_sha256: str,
    trace_record_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": FEATURE_RECEIPT_SCHEMA,
        "request_sha256": _digest(request_sha256, "request_sha256"),
        "batch_index": int(batch_index),
        "trace_input_sha256": _digest(
            trace_input_sha256,
            "trace_input_sha256",
        ),
        "trace_record_sha256": _digest(
            trace_record_sha256,
            "trace_record_sha256",
        ),
        "relative_path": path.name,
        "file_size_bytes": int(path.stat().st_size),
        "file_sha256": _file_sha256(path),
        "tensor_descriptors": {
            name: _tensor_descriptor(value) for name, value in sorted(tensors.items())
        },
        "contains_parent_features": False,
        "contains_live_child_base_features": True,
        "contains_projected_parent_branch_targets": True,
        "contains_full_base_trajectory": True,
        "all_full_trajectory_rows_intervened": True,
        "semantic_positions_metrics_only": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _authenticate_feature_receipt(
    value: Mapping[str, Any],
    *,
    feature_directory: Path,
    expected_request_sha256: str,
    expected_batch_index: int,
    expected_trace_input_sha256: str,
    expected_trace_record_sha256: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    row = json.loads(_canonical_bytes(dict(value)).decode("ascii"))
    claimed = row.pop("receipt_sha256", None)
    expected_fields = {
        "batch_index",
        "all_full_trajectory_rows_intervened",
        "contains_full_base_trajectory",
        "contains_live_child_base_features",
        "contains_parent_features",
        "contains_projected_parent_branch_targets",
        "file_sha256",
        "file_size_bytes",
        "relative_path",
        "request_sha256",
        "schema",
        "semantic_positions_metrics_only",
        "tensor_descriptors",
        "trace_input_sha256",
        "trace_record_sha256",
    }
    if (
        set(row) != expected_fields
        or row.get("schema") != FEATURE_RECEIPT_SCHEMA
        or row.get("request_sha256") != expected_request_sha256
        or row.get("batch_index") != expected_batch_index
        or row.get("trace_input_sha256") != expected_trace_input_sha256
        or row.get("trace_record_sha256") != expected_trace_record_sha256
        or row.get("contains_parent_features") is not False
        or row.get("contains_live_child_base_features") is not True
        or row.get("contains_projected_parent_branch_targets") is not True
        or row.get("contains_full_base_trajectory") is not True
        or row.get("all_full_trajectory_rows_intervened") is not True
        or row.get("semantic_positions_metrics_only") is not True
        or claimed != canonical_sha256(row)
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Causal feature receipt does not authenticate its record."
        )
    relative = row.get("relative_path")
    if (
        not isinstance(relative, str)
        or Path(relative).name != relative
        or not relative.endswith(".safetensors")
    ):
        raise Lfm2Exact4xCausalRunnerError("Causal feature receipt has an unsafe path.")
    path = feature_directory / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or getattr(path.stat(), "st_nlink", 1) != 1
        or path.stat().st_size != row.get("file_size_bytes")
        or _file_sha256(path) != row.get("file_sha256")
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Causal feature file differs from its immutable receipt."
        )
    loaded = safetensors.torch.load_file(str(path), device="cpu")
    tensors = {
        name: value.detach().contiguous().clone() for name, value in loaded.items()
    }
    del loaded
    descriptors = row.get("tensor_descriptors")
    if (
        not isinstance(descriptors, Mapping)
        or set(tensors) != set(descriptors)
        or any(
            _tensor_descriptor(tensors[name]) != descriptors[name] for name in tensors
        )
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Causal feature tensor inventory does not authenticate."
        )
    return {**row, "receipt_sha256": claimed}, tensors


def _module_at_path(model: Any, path: str) -> Any:
    current = model
    for part in path.split("."):
        if not part:
            raise Lfm2Exact4xCausalRunnerError(f"Module path {path!r} is invalid.")
        if part.isdecimal():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError) as exc:
                raise Lfm2Exact4xCausalRunnerError(
                    f"Module path {path!r} cannot resolve index {part}."
                ) from exc
        else:
            try:
                current = getattr(current, part)
            except AttributeError as exc:
                raise Lfm2Exact4xCausalRunnerError(
                    f"Module path {path!r} cannot resolve {part!r}."
                ) from exc
    if not isinstance(current, torch.nn.Module):
        raise Lfm2Exact4xCausalRunnerError(
            f"Module path {path!r} did not resolve a torch module."
        )
    return current


def _semantic_positions(
    *,
    prompt_count: int,
    forced_count: int,
) -> torch.Tensor:
    if prompt_count <= 0 or forced_count < 0:
        raise Lfm2Exact4xCausalRunnerError(
            "Trajectory prompt/decode geometry is invalid."
        )
    return torch.arange(
        prompt_count - 1,
        prompt_count + forced_count,
        dtype=torch.long,
    )


def _replace_rows(
    value: torch.Tensor,
    positions: torch.Tensor,
    replacement: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or int(value.shape[0]) != 1
        or not torch.is_floating_point(value)
        or value.is_complex()
        or not isinstance(positions, torch.Tensor)
        or positions.ndim != 1
        or positions.dtype != torch.long
        or not isinstance(replacement, torch.Tensor)
        or replacement.ndim != 2
        or tuple(replacement.shape) != (int(positions.numel()), int(value.shape[2]))
        or int(positions.numel()) <= 0
        or int(positions[0]) < 0
        or int(positions[-1]) >= int(value.shape[1])
        or not bool(torch.isfinite(value.detach()).all())
        or not bool(torch.isfinite(replacement.detach()).all())
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Causal intervention rows have invalid geometry."
        )
    result = value.clone()
    result[0, positions.to(device=value.device)] = replacement.to(
        device=value.device,
        dtype=value.dtype,
    )
    return result


def _tensor_from_module_input(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[torch.Tensor, str | int]:
    hidden = kwargs.get("hidden_states")
    if isinstance(hidden, torch.Tensor):
        return hidden, "hidden_states"
    if args and isinstance(args[0], torch.Tensor):
        return args[0], 0
    raise Lfm2Exact4xCausalRunnerError(
        "Causal module-input hook saw no hidden-state tensor."
    )


def _replace_module_input(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    positions: torch.Tensor,
    replacement: torch.Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    hidden, location = _tensor_from_module_input(args, kwargs)
    replaced = _replace_rows(hidden, positions, replacement)
    updated_args = list(args)
    updated_kwargs = dict(kwargs)
    if location == "hidden_states":
        updated_kwargs[location] = replaced
    else:
        updated_args[0] = replaced
    return tuple(updated_args), updated_kwargs


def _unwrap_tensor_output(value: Any) -> tuple[torch.Tensor, str]:
    if isinstance(value, torch.Tensor):
        return value, "tensor"
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        return value[0], "tuple"
    raise Lfm2Exact4xCausalRunnerError(
        "Causal module-output hook saw no tensor output."
    )


def _replace_module_output(
    output: Any,
    positions: torch.Tensor,
    replacement: torch.Tensor,
) -> Any:
    hidden, kind = _unwrap_tensor_output(output)
    replaced = _replace_rows(hidden, positions, replacement)
    if kind == "tensor":
        return replaced
    return (replaced, *output[1:])


def _capture_module_input(
    destination: dict[str, torch.Tensor],
    name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> None:
    if name in destination:
        raise Lfm2Exact4xCausalRunnerError(
            f"Feature hook {name!r} executed more than once."
        )
    value, _location = _tensor_from_module_input(args, kwargs)
    destination[name] = value.detach().to(device="cpu").contiguous().clone()


def _capture_module_output(
    destination: dict[str, torch.Tensor],
    name: str,
    output: Any,
) -> None:
    if name in destination:
        raise Lfm2Exact4xCausalRunnerError(
            f"Feature hook {name!r} executed more than once."
        )
    value, _kind = _unwrap_tensor_output(output)
    destination[name] = value.detach().to(device="cpu").contiguous().clone()


def _run_causal_replay(
    *,
    child: Any,
    reconstructed_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    intervention_positions: torch.Tensor,
    semantic_positions: torch.Tensor,
    operator_specs: Sequence[Lfm2CausalOperatorSpec],
    intervention_members: Sequence[str],
    projected_parent_targets: Mapping[str, torch.Tensor],
    capture_full_features: bool,
) -> dict[str, torch.Tensor]:
    """Execute one temporary-hook replay and return detached observations."""

    specs = canonical_operator_specs(operator_specs)
    members = tuple(intervention_members)
    if len(set(members)) != len(members) or not set(members).issubset(
        {spec.group_id for spec in specs}
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Intervention members do not form a canonical operator subset."
        )
    observations: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    terminal_module = _module_at_path(child, "model.layers")[-1]
    lm_head = _module_at_path(child, "lm_head")
    try:
        for spec in specs:
            if capture_full_features and spec.consumer_module_path is not None:
                consumer = _module_at_path(child, spec.consumer_module_path)
                feature_name = f"feature.{spec.group_id}"
                if spec.consumer_hook_kind == "module_input":

                    def capture_input(
                        _module: Any,
                        args: tuple[Any, ...],
                        kwargs: Mapping[str, Any],
                        *,
                        name: str = feature_name,
                    ) -> None:
                        _capture_module_input(observations, name, args, kwargs)

                    handles.append(
                        consumer.register_forward_pre_hook(
                            capture_input,
                            with_kwargs=True,
                        )
                    )
                else:

                    def capture_output(
                        _module: Any,
                        _args: tuple[Any, ...],
                        _kwargs: Mapping[str, Any],
                        output: Any,
                        *,
                        name: str = feature_name,
                    ) -> None:
                        _capture_module_output(observations, name, output)

                    handles.append(
                        consumer.register_forward_hook(
                            capture_output,
                            with_kwargs=True,
                        )
                    )
            if spec.group_id not in members:
                continue
            target = projected_parent_targets.get(spec.group_id)
            if not isinstance(target, torch.Tensor):
                raise Lfm2Exact4xCausalRunnerError(
                    f"Projected target for {spec.group_id!r} is missing."
                )
            module = _module_at_path(child, spec.target_module_path)
            if spec.hook_kind == "module_input":

                def replace_input(
                    _module: Any,
                    args: tuple[Any, ...],
                    kwargs: Mapping[str, Any],
                    *,
                    rows: torch.Tensor = target,
                ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                    return _replace_module_input(
                        args,
                        kwargs,
                        intervention_positions,
                        rows,
                    )

                handles.append(
                    module.register_forward_pre_hook(
                        replace_input,
                        with_kwargs=True,
                    )
                )
            else:

                def replace_output(
                    _module: Any,
                    _args: tuple[Any, ...],
                    _kwargs: Mapping[str, Any],
                    output: Any,
                    *,
                    rows: torch.Tensor = target,
                ) -> Any:
                    return _replace_module_output(
                        output,
                        intervention_positions,
                        rows,
                    )

                handles.append(
                    module.register_forward_hook(
                        replace_output,
                        with_kwargs=True,
                    )
                )

        def capture_terminal(
            _module: Any,
            _args: tuple[Any, ...],
            _kwargs: Mapping[str, Any],
            output: Any,
        ) -> None:
            _capture_module_output(
                observations,
                "terminal_residual",
                output,
            )

        handles.append(
            terminal_module.register_forward_hook(
                capture_terminal,
                with_kwargs=True,
            )
        )

        def capture_lm_head_input(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            _capture_module_input(
                observations,
                "lm_head_input",
                args,
                kwargs,
            )

        handles.append(
            lm_head.register_forward_pre_hook(
                capture_lm_head_input,
                with_kwargs=True,
            )
        )
        child.eval()
        with torch.inference_mode():
            child(
                inputs_embeds=reconstructed_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in reversed(handles):
            handle.remove()
    required = {"terminal_residual", "lm_head_input"}
    if capture_full_features:
        required.update(
            f"feature.{spec.group_id}"
            for spec in specs
            if spec.consumer_module_path is not None
        )
    if set(observations) != required:
        raise Lfm2Exact4xCausalRunnerError(
            "Causal replay did not capture every required endpoint exactly once."
        )
    for name, value in tuple(observations.items()):
        if (
            value.ndim != 3
            or int(value.shape[0]) != 1
            or int(value.shape[1]) != int(reconstructed_embeddings.shape[1])
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"Causal replay observation {name!r} has invalid geometry."
            )
        if not capture_full_features:
            observations[name] = value[:, semantic_positions, :].contiguous()
    return observations


def _project_parent_rows(
    coordinate_map: Any,
    parent_rows: torch.Tensor,
) -> torch.Tensor:
    callback = getattr(coordinate_map, "encode", None)
    source_width = getattr(coordinate_map, "source_width", None)
    target_width = getattr(coordinate_map, "target_width", None)
    if (
        not callable(callback)
        or not isinstance(parent_rows, torch.Tensor)
        or parent_rows.ndim != 2
        or int(parent_rows.shape[1]) != source_width
        or isinstance(target_width, bool)
        or not isinstance(target_width, int)
        or target_width <= 0
        or not bool(torch.isfinite(parent_rows).all())
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Projected parent rows have invalid source geometry."
        )
    projected = callback(parent_rows.to(dtype=torch.float64))
    if (
        not isinstance(projected, torch.Tensor)
        or tuple(projected.shape) != (int(parent_rows.shape[0]), target_width)
        or not bool(torch.isfinite(projected).all())
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Balanced coordinate map returned invalid target rows."
        )
    return projected.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _parent_phase_matrix(
    parent_tensors: Mapping[str, torch.Tensor],
    *,
    phases: Sequence[str],
    layer_index: int,
    endpoint: str,
    source_width: int,
) -> torch.Tensor:
    rows = []
    for phase in phases:
        name = f"{phase}.layer.{layer_index:03d}.{endpoint}"
        value = parent_tensors.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or tuple(value.shape[:2]) != (1, 1)
            or int(value.shape[2]) != source_width
            or not torch.is_floating_point(value)
            or value.is_complex()
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"Cached parent endpoint {name!r} has invalid geometry."
            )
        rows.append(value[0, 0].to(dtype=torch.float64))
    return torch.stack(rows).to(device="cpu").contiguous()


def _parent_full_trajectory_matrix(
    parent_tensors: Mapping[str, torch.Tensor],
    *,
    phases: Sequence[str],
    layer_index: int,
    endpoint: str,
    source_width: int,
    prompt_count: int,
    forced_count: int,
) -> torch.Tensor:
    if len(phases) != forced_count + 1:
        raise Lfm2Exact4xCausalRunnerError(
            "Causal endpoint phases differ from the trajectory horizon."
        )
    prefix_name = f"{phases[0]}.layer.{layer_index:03d}.{endpoint}"
    prefix = parent_tensors.get(prefix_name)
    if (
        not isinstance(prefix, torch.Tensor)
        or prefix.ndim != 3
        or tuple(prefix.shape) != (1, prompt_count, source_width)
        or not torch.is_floating_point(prefix)
        or prefix.is_complex()
        or not bool(torch.isfinite(prefix).all())
    ):
        raise Lfm2Exact4xCausalRunnerError(
            f"Cached full-prefill endpoint {prefix_name!r} is invalid."
        )
    rows = [prefix[0].detach().to(device="cpu", dtype=torch.float64)]
    for phase in phases[1:]:
        name = f"{phase}.layer.{layer_index:03d}.{endpoint}"
        value = parent_tensors.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or int(value.shape[0]) != 1
            or int(value.shape[1]) <= 0
            or int(value.shape[2]) != source_width
            or not torch.is_floating_point(value)
            or value.is_complex()
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"Cached decode endpoint {name!r} is invalid."
            )
        rows.append(value[0, -1:].detach().to(device="cpu", dtype=torch.float64))
    result = torch.cat(rows, dim=0).contiguous()
    if tuple(result.shape) != (prompt_count + forced_count, source_width):
        raise Lfm2Exact4xCausalRunnerError(
            "Cached causal endpoint does not cover the complete trajectory."
        )
    return result


def _parent_logit_matrix(
    parent_tensors: Mapping[str, torch.Tensor],
    *,
    phases: Sequence[str],
    physical_row_count: int,
) -> torch.Tensor:
    rows = []
    for phase in phases:
        value = parent_tensors.get(f"{phase}.logits")
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or tuple(value.shape[:2]) != (1, 1)
            or int(value.shape[2]) != physical_row_count
            or not torch.is_floating_point(value)
            or value.is_complex()
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"Cached parent logits for {phase!r} have invalid geometry."
            )
        rows.append(value[0, 0].to(dtype=torch.float64))
    return torch.stack(rows).to(device="cpu").contiguous()


def _relative_l2_rows(
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if (
        candidate.ndim != 2
        or tuple(candidate.shape) != tuple(target.shape)
        or not bool(torch.isfinite(candidate).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Terminal candidate and target rows must have equal finite geometry."
        )
    left = candidate.to(dtype=torch.float64)
    right = target.to(dtype=torch.float64)
    numerator = torch.linalg.vector_norm(left - right, dim=1)
    denominator = torch.linalg.vector_norm(right, dim=1).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return numerator / denominator


def _parent_winner_margin_rows(
    parent_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    defined_semantic_row_count: int,
) -> torch.Tensor:
    count = int(defined_semantic_row_count)
    if (
        count <= 1
        or parent_logits.ndim != 2
        or candidate_logits.ndim != 2
        or tuple(parent_logits.shape[:-1]) != tuple(candidate_logits.shape[:-1])
        or int(parent_logits.shape[1]) < count
        or int(candidate_logits.shape[1]) != count
        or not bool(torch.isfinite(parent_logits).all())
        or not bool(torch.isfinite(candidate_logits).all())
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Parent-winner margin logits have invalid defined-vocabulary geometry."
        )
    parent = parent_logits[:, :count].to(dtype=torch.float64)
    candidate = candidate_logits.to(dtype=torch.float64)
    winners = parent.argmax(dim=1)
    rows = torch.arange(int(parent.shape[0]), dtype=torch.long)
    winner_scores = candidate[rows, winners]
    competitors = candidate.clone()
    competitors[rows, winners] = -torch.inf
    return winner_scores - competitors.max(dim=1).values


def _score_donor_logits(
    *,
    routing: Any,
    output_routing_session: Any,
    lm_head_rows: torch.Tensor,
    maximum_donor_rows_per_score_chunk: int,
) -> torch.Tensor:
    scores = []
    for row in lm_head_rows:
        target = routing.score_target_tokens(output_routing_session, row)
        donor = routing.compute_all_donor_logits_from_target_scores(
            output_routing_session,
            target,
            maximum_donor_rows_per_chunk=maximum_donor_rows_per_score_chunk,
        )
        if (
            not isinstance(donor, torch.Tensor)
            or donor.ndim != 1
            or int(donor.numel()) != int(routing.donor_row_count)
            or not bool(torch.isfinite(donor).all())
        ):
            raise Lfm2Exact4xCausalRunnerError(
                "C_out route returned invalid donor logits."
            )
        scores.append(donor.detach().to(device="cpu", dtype=torch.float64))
    return torch.stack(scores).contiguous()


def _causal_parent_tensor_names(
    reference: Any,
    *,
    layer_count: int,
    operator_specs: Sequence[Lfm2CausalOperatorSpec],
) -> tuple[str, ...]:
    specs = canonical_operator_specs(operator_specs)
    if layer_count <= 0 or any(spec.layer_index >= layer_count for spec in specs):
        raise Lfm2Exact4xCausalRunnerError(
            "Causal operator specs leave the authenticated layer stack."
        )
    names = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
    }
    capture = getattr(reference, "record", {}).get("capture", {})
    decode_steps = capture.get("decode_steps")
    if (
        isinstance(decode_steps, bool)
        or not isinstance(decode_steps, int)
        or decode_steps < 0
    ):
        raise Lfm2Exact4xCausalRunnerError("Trace record has no valid decode horizon.")
    inventory = set(getattr(reference, "ordered_tensor_names", ()))
    if "input.position_ids" in inventory:
        names.add("input.position_ids")
    for phase in phase_names(decode_steps):
        names.add(f"{phase}.logits")
        names.add(f"{phase}.layer.{layer_count - 1:03d}.residual.post_layer")
        for spec in specs:
            names.add(f"{phase}.layer.{spec.layer_index:03d}.{spec.parent_endpoint}")
    missing = tuple(sorted(names - inventory))
    if missing:
        raise Lfm2Exact4xCausalRunnerError(
            f"Trace record lacks causal parent endpoints: {missing!r}."
        )
    return tuple(sorted(names))


def _compact_causal_parent_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    operator_specs: Sequence[Lfm2CausalOperatorSpec],
) -> dict[str, torch.Tensor]:
    endpoint_suffixes = tuple(
        f".layer.{spec.layer_index:03d}.{spec.parent_endpoint}"
        for spec in canonical_operator_specs(operator_specs)
    )
    compact = {}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise Lfm2Exact4xCausalRunnerError(
                f"Selected parent tensor {name!r} is invalid."
            )
        keep_full = name.startswith("input.") or (
            name.startswith("prefill.") and name.endswith(endpoint_suffixes)
        )
        retained = value if keep_full else value[:, -1:, :]
        compact[name] = retained.detach().to(device="cpu").contiguous().clone()
    return compact


def _trajectory_geometry(
    parent_tensors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    prompt_ids = parent_tensors.get("input.input_ids")
    forced_ids = parent_tensors.get("input.forced_decode_ids")
    attention = parent_tensors.get("input.attention_mask")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not isinstance(prompt_ids, torch.Tensor)
        or not isinstance(forced_ids, torch.Tensor)
        or not isinstance(attention, torch.Tensor)
        or prompt_ids.ndim != 2
        or forced_ids.ndim != 2
        or attention.ndim != 2
        or prompt_ids.dtype not in integer_dtypes
        or forced_ids.dtype not in integer_dtypes
        or attention.dtype not in integer_dtypes
        or tuple(attention.shape) != tuple(prompt_ids.shape)
        or int(prompt_ids.shape[0]) != 1
        or int(prompt_ids.shape[1]) <= 0
        or int(forced_ids.shape[0]) != 1
        or bool((prompt_ids < 0).any())
        or bool((forced_ids < 0).any())
        or not bool((attention == 1).all())
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Cached donor trajectory IDs or attention mask are invalid."
        )
    prompt_count = int(prompt_ids.shape[1])
    forced_count = int(forced_ids.shape[1])
    trajectory_ids = torch.cat(
        (prompt_ids.to(torch.long), forced_ids.to(torch.long)),
        dim=1,
    )
    trajectory_mask = torch.cat(
        (
            attention.to(torch.long),
            torch.ones((1, forced_count), dtype=torch.long),
        ),
        dim=1,
    )
    position_ids = v3_runner._trajectory_position_ids(
        parent_tensors,
        forced_count,
    )
    return (
        trajectory_ids,
        trajectory_mask,
        position_ids,
        prompt_count,
        forced_count,
    )


def _projected_parent_targets(
    *,
    parent_tensors: Mapping[str, torch.Tensor],
    phases: Sequence[str],
    operator_specs: Sequence[Lfm2CausalOperatorSpec],
    coordinate_map: Any,
    source_width: int,
    prompt_count: int,
    forced_count: int,
    terminal_layer_index: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    targets = {}
    for spec in canonical_operator_specs(operator_specs):
        parent = _parent_full_trajectory_matrix(
            parent_tensors,
            phases=phases,
            layer_index=spec.layer_index,
            endpoint=spec.parent_endpoint,
            source_width=source_width,
            prompt_count=prompt_count,
            forced_count=forced_count,
        )
        targets[spec.group_id] = _project_parent_rows(coordinate_map, parent)
    terminal_parent = _parent_phase_matrix(
        parent_tensors,
        phases=phases,
        layer_index=terminal_layer_index,
        endpoint="residual.post_layer",
        source_width=source_width,
    )
    return targets, _project_parent_rows(coordinate_map, terminal_parent)


def _subset_metric_rows(
    *,
    replay: Mapping[str, torch.Tensor],
    semantic_positions: torch.Tensor,
    base_full_trajectory: bool,
    projected_parent_terminal: torch.Tensor,
    parent_logits: torch.Tensor,
    defined_semantic_row_count: int,
    routing: Any,
    output_routing_session: Any,
    maximum_donor_rows_per_score_chunk: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    terminal = replay["terminal_residual"]
    lm_head_input = replay["lm_head_input"]
    if base_full_trajectory:
        terminal = terminal[:, semantic_positions, :]
        lm_head_input = lm_head_input[:, semantic_positions, :]
    terminal_rows = terminal[0].to(dtype=torch.float64)
    lm_head_rows = lm_head_input[0]
    donor_logits = _score_donor_logits(
        routing=routing,
        output_routing_session=output_routing_session,
        lm_head_rows=lm_head_rows,
        maximum_donor_rows_per_score_chunk=maximum_donor_rows_per_score_chunk,
    )
    terminal_error = _relative_l2_rows(
        terminal_rows,
        projected_parent_terminal,
    )
    defined_vocab_kl = causal_module.defined_vocab_kl_rows(
        parent_logits,
        donor_logits,
        defined_semantic_row_count=defined_semantic_row_count,
    )
    parent_margin = _parent_winner_margin_rows(
        parent_logits,
        donor_logits,
        defined_semantic_row_count=defined_semantic_row_count,
    )
    return (
        {
            causal_module.TERMINAL_ERROR_METRIC: float(terminal_error.mean().item()),
            causal_module.DEFINED_VOCAB_KL_METRIC: float(
                defined_vocab_kl.mean().item()
            ),
            causal_module.PARENT_WINNER_MARGIN_METRIC: float(
                parent_margin.mean().item()
            ),
        },
        {
            "terminal_relative_l2_error": terminal_error,
            "defined_vocab_kl": defined_vocab_kl,
            "parent_winner_margin": parent_margin,
        },
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CausalBatchResume:
    results: tuple[dict[str, Any], ...]
    checkpoints: tuple[dict[str, Any], ...]
    feature_receipts: tuple[dict[str, Any], ...]
    feature_tensors: tuple[dict[str, torch.Tensor], ...]
    checkpoint_hit_count: int
    computed_batch_count: int
    historical_checkpoint_child_forward_calls: int
    current_run_child_forward_calls: int
    total_batch_count: int

    @property
    def completed_batch_count(self) -> int:
        return len(self.results)

    @property
    def remaining_batch_count(self) -> int:
        return self.total_batch_count - self.completed_batch_count

    @property
    def is_complete(self) -> bool:
        return self.remaining_batch_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_hit_count": self.checkpoint_hit_count,
            "computed_batch_count": self.computed_batch_count,
            "completed_batch_count": self.completed_batch_count,
            "remaining_batch_count": self.remaining_batch_count,
            "total_batch_count": self.total_batch_count,
            "is_complete": self.is_complete,
            "historical_checkpoint_child_forward_calls": (
                self.historical_checkpoint_child_forward_calls
            ),
            "current_run_child_forward_calls": self.current_run_child_forward_calls,
            "persisted_child_forward_calls": (
                self.historical_checkpoint_child_forward_calls
                + self.current_run_child_forward_calls
            ),
        }


def _feature_receipt_from_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = result.get("oracle_evidence")
    receipt = evidence.get("feature_receipt") if isinstance(evidence, Mapping) else None
    if not isinstance(receipt, Mapping):
        raise Lfm2Exact4xCausalRunnerError(
            "Causal batch checkpoint has no HMAC-bound feature receipt."
        )
    return receipt


def _quarantine_orphan_feature(path: Path) -> dict[str, Any] | None:
    """Move an uncheckpointed runner-owned feature file aside without overwrite."""

    if not path.exists() and not path.is_symlink():
        return None
    if (
        path.is_symlink()
        or not path.is_file()
        or getattr(path.stat(), "st_nlink", 1) != 1
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Uncheckpointed causal feature artifact is link-like or not ordinary."
        )
    file_sha256 = _file_sha256(path)
    quarantine = path.parent / "orphaned-incomplete-captures"
    quarantine.mkdir(parents=True, exist_ok=True)
    if quarantine.is_symlink():
        raise Lfm2Exact4xCausalRunnerError(
            "Causal orphan quarantine cannot be a symlink."
        )
    destination = None
    for ordinal in range(1_000_000):
        candidate = quarantine / (
            f"{path.stem}.{file_sha256}.orphaned-{ordinal:06d}.safetensors"
        )
        try:
            os.link(path, candidate)
            _fsync_directory(quarantine)
        except FileExistsError:
            continue
        destination = candidate
        break
    if destination is None:
        raise Lfm2Exact4xCausalRunnerError(
            "Causal orphan quarantine exhausted its collision-safe namespace."
        )
    path.unlink()
    _fsync_directory(path.parent)
    return {
        "source_name": path.name,
        "quarantine_relative_path": str(destination.relative_to(path.parent)),
        "file_sha256": file_sha256,
        "file_size_bytes": int(destination.stat().st_size),
        "reason": "feature-published-before-hmac-checkpoint-crash-window",
        "recomputation_required": True,
    }


def resume_causal_batches(
    *,
    request: Mapping[str, Any],
    references: Sequence[Any],
    checkpoint_directory: Path,
    feature_directory: Path,
    checkpoint_authority: Any,
    processor: Any,
    progress: _ProgressJournal | None = None,
    maximum_new_batches: int | None = None,
) -> CausalBatchResume:
    """Authenticate one canonical prefix and execute only missing batches."""

    authenticated_request = (
        causal_module.authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    )
    records = authenticated_request["record_bindings"]
    if len(records) != len(references):
        raise Lfm2Exact4xCausalRunnerError(
            "Trace references differ from the causal record manifest."
        )
    if maximum_new_batches is not None and (
        isinstance(maximum_new_batches, bool) or maximum_new_batches < 0
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "maximum_new_batches must be a nonnegative integer or None."
        )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    feature_directory.mkdir(parents=True, exist_ok=True)
    if checkpoint_directory.is_symlink() or feature_directory.is_symlink():
        raise Lfm2Exact4xCausalRunnerError(
            "Causal checkpoint and feature directories must be ordinary."
        )
    existing_values = []
    first_missing = len(references)
    for index in range(len(references)):
        path = checkpoint_directory / f"batch-{index:06d}.json"
        if path.exists() or path.is_symlink():
            if index >= first_missing:
                raise Lfm2Exact4xCausalRunnerError(
                    "Causal checkpoints must form one contiguous prefix."
                )
            existing_values.append(
                v3_runner._read_json(path, f"causal batch {index} checkpoint")
            )
        else:
            first_missing = min(first_missing, index)
    prefix = causal_module.authenticate_lfm2_exact4x_causal_resume(
        existing_values,
        request=authenticated_request,
        authority=checkpoint_authority,
    )
    for index, (record, reference) in enumerate(zip(records, references, strict=True)):
        if (
            int(reference.index) != index
            or str(reference.input_sha256) != record["trace_input_sha256"]
            or str(reference.record_sha256) != record["trace_record_sha256"]
        ):
            raise Lfm2Exact4xCausalRunnerError(
                "Trace reference differs from its v3-bound causal record."
            )
    execution_end = len(references)
    if maximum_new_batches is not None:
        execution_end = min(
            execution_end,
            len(prefix) + int(maximum_new_batches),
        )
    results: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    features: list[dict[str, torch.Tensor]] = []
    historical_forwards = 0
    current_forwards = 0
    for index, (record, reference) in enumerate(
        zip(records[:execution_end], references[:execution_end], strict=True)
    ):
        if index < len(prefix):
            checkpoint = prefix[index]
            result = checkpoint["result"]
            receipt, tensors = _authenticate_feature_receipt(
                _feature_receipt_from_result(result),
                feature_directory=feature_directory,
                expected_request_sha256=authenticated_request["request_sha256"],
                expected_batch_index=index,
                expected_trace_input_sha256=record["trace_input_sha256"],
                expected_trace_record_sha256=record["trace_record_sha256"],
            )
            historical_forwards += int(result["operation_audit"]["child_forward_calls"])
            if progress is not None:
                progress.emit(
                    "batch-checkpoint-hit",
                    batch_index=index,
                    split=record["split"],
                    checkpoint_sha256=checkpoint["checkpoint_sha256"],
                    feature_receipt_sha256=receipt["receipt_sha256"],
                    current_run_child_forward_calls=0,
                )
        else:
            orphan = _quarantine_orphan_feature(
                feature_directory / f"batch-{index:06d}.safetensors"
            )
            if orphan is not None and progress is not None:
                progress.emit(
                    "orphan-feature-quarantined",
                    batch_index=index,
                    **orphan,
                )
            if progress is not None:
                progress.emit(
                    "batch-start",
                    batch_index=index,
                    split=record["split"],
                    full_trajectory_interventions=True,
                    planned_child_forward_calls=8,
                )
            result, receipt, tensors = processor(reference, record)
            result = causal_module.authenticate_lfm2_exact4x_causal_batch_result(
                result,
                request=authenticated_request,
            )
            authenticated_receipt, authenticated_tensors = (
                _authenticate_feature_receipt(
                    _feature_receipt_from_result(result),
                    feature_directory=feature_directory,
                    expected_request_sha256=authenticated_request["request_sha256"],
                    expected_batch_index=index,
                    expected_trace_input_sha256=record["trace_input_sha256"],
                    expected_trace_record_sha256=record["trace_record_sha256"],
                )
            )
            if authenticated_receipt != receipt or set(authenticated_tensors) != set(
                tensors
            ):
                raise Lfm2Exact4xCausalRunnerError(
                    "Processor feature evidence changed before checkpointing."
                )
            receipt = authenticated_receipt
            tensors = authenticated_tensors
            checkpoint = causal_module.build_lfm2_exact4x_causal_batch_checkpoint(
                authenticated_request,
                result,
                authority=checkpoint_authority,
            )
            v3_runner._atomic_write_new_json(
                checkpoint_directory / f"batch-{index:06d}.json",
                checkpoint,
            )
            current_forwards += int(result["operation_audit"]["child_forward_calls"])
            if progress is not None:
                progress.emit(
                    "batch-complete",
                    batch_index=index,
                    split=record["split"],
                    checkpoint_sha256=checkpoint["checkpoint_sha256"],
                    batch_result_sha256=result["batch_result_sha256"],
                    feature_receipt_sha256=receipt["receipt_sha256"],
                    current_run_child_forward_calls=int(
                        result["operation_audit"]["child_forward_calls"]
                    ),
                )
        results.append(dict(result))
        checkpoints.append(dict(checkpoint))
        receipts.append(dict(receipt))
        features.append(dict(tensors))
    return CausalBatchResume(
        results=tuple(results),
        checkpoints=tuple(checkpoints),
        feature_receipts=tuple(receipts),
        feature_tensors=tuple(features),
        checkpoint_hit_count=len(prefix),
        computed_batch_count=execution_end - len(prefix),
        historical_checkpoint_child_forward_calls=historical_forwards,
        current_run_child_forward_calls=current_forwards,
        total_batch_count=len(references),
    )


def _build_partial_completion_receipt(
    *,
    request: Mapping[str, Any],
    resume: CausalBatchResume,
    checkpoint_authority: Any,
    child_package_sha256: str,
    child_state_sha256: str,
    route_package_sha256: str,
    route_state_sha256: str,
    implementation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        resume.is_complete
        or resume.total_batch_count != int(request["record_count"])
        or len(resume.checkpoints) != resume.completed_batch_count
        or len(resume.feature_receipts) != resume.completed_batch_count
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Partial completion requires one incomplete authenticated request prefix."
        )
    checkpoint_rows = [
        {
            "batch_index": checkpoint["batch_index"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_hmac_sha256": checkpoint["checkpoint_hmac_sha256"],
            "batch_result_sha256": checkpoint["batch_result_sha256"],
        }
        for checkpoint in resume.checkpoints
    ]
    body = {
        "schema": PARTIAL_COMPLETION_SCHEMA,
        "status": "partial-observation",
        "request_sha256": request["request_sha256"],
        "checkpoint_hmac_key_id": checkpoint_authority.key_id,
        "checkpoint_chain_authenticated": True,
        "ordered_batch_checkpoint_authentication": checkpoint_rows,
        "completed_batch_count": resume.completed_batch_count,
        "remaining_batch_count": resume.remaining_batch_count,
        "total_batch_count": resume.total_batch_count,
        "next_batch_index": resume.completed_batch_count,
        "full_request_was_built_and_authenticated": True,
        "execution_limit_changes_schedule_only": True,
        "aggregate_evaluated": False,
        "live_feature_oracles_evaluated": False,
        "child_package_sha256": child_package_sha256,
        "child_state_sha256": child_state_sha256,
        "route_package_sha256": route_package_sha256,
        "route_state_sha256": route_state_sha256,
        "implementation_identity": dict(implementation_identity),
        "operation_audit": {
            "parent_forward_calls": 0,
            "child_forward_calls": sum(
                int(result["operation_audit"]["child_forward_calls"])
                for result in resume.results
            ),
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "persistent_model_mutations": 0,
        },
    }
    sealed = {
        **body,
        "partial_completion_sha256": canonical_sha256(body),
    }
    return {
        **sealed,
        "partial_completion_hmac_sha256": checkpoint_authority.sign_ledger(sealed),
    }


def _process_causal_batch(
    *,
    reference: Any,
    record: Mapping[str, Any],
    request: Mapping[str, Any],
    trace_session: Any,
    child: Any,
    routing: Any,
    input_routing_session: Any,
    output_routing_session: Any,
    coordinate_map: Any,
    operator_specs: Sequence[Lfm2CausalOperatorSpec],
    layer_count: int,
    source_width: int,
    feature_directory: Path,
    maximum_selected_parent_bytes: int,
    maximum_donor_rows_per_score_chunk: int,
    progress: _ProgressJournal | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    names = _causal_parent_tensor_names(
        reference,
        layer_count=layer_count,
        operator_specs=operator_specs,
    )
    logical_bytes = parent_selection_logical_bytes(reference, names)
    if logical_bytes > maximum_selected_parent_bytes:
        raise Lfm2Exact4xCausalRunnerError(
            "Selected causal parent payload exceeds the configured read ceiling."
        )
    payload = trace_session.load_selected_tensors(reference.index, names)
    selected_parent_payload_sha256 = str(payload.selection_sha256)
    try:
        parent_tensors = _compact_causal_parent_tensors(
            payload.tensors,
            operator_specs=operator_specs,
        )
        (
            trajectory_ids,
            trajectory_mask,
            position_ids,
            prompt_count,
            forced_count,
        ) = _trajectory_geometry(parent_tensors)
        phases = phase_names(forced_count)
        semantic_positions = _semantic_positions(
            prompt_count=prompt_count,
            forced_count=forced_count,
        )
        intervention_positions = torch.arange(
            prompt_count + forced_count,
            dtype=torch.long,
        )
        reconstructed = routing.reconstruct_selected_donor_embeddings(
            input_routing_session,
            trajectory_ids.reshape(-1),
        ).reshape(1, prompt_count + forced_count, -1)
        if (
            reconstructed.ndim != 3
            or int(reconstructed.shape[2]) != int(coordinate_map.target_width)
            or not bool(torch.isfinite(reconstructed).all())
        ):
            raise Lfm2Exact4xCausalRunnerError(
                "Authenticated C_in route returned invalid full-trajectory embeddings."
            )
        projected_targets, projected_terminal = _projected_parent_targets(
            parent_tensors=parent_tensors,
            phases=phases,
            operator_specs=operator_specs,
            coordinate_map=coordinate_map,
            source_width=source_width,
            prompt_count=prompt_count,
            forced_count=forced_count,
            terminal_layer_index=layer_count - 1,
        )
        parent_logits = _parent_logit_matrix(
            parent_tensors,
            phases=phases,
            physical_row_count=int(
                request["metric_contract"]["parent_physical_row_capacity"]
            ),
        )
        subset_metrics: dict[tuple[str, ...], dict[str, float]] = {}
        feature_tensors: dict[str, torch.Tensor] = {
            "intervention_positions": intervention_positions,
            "semantic_positions": semantic_positions,
            "record_full_row_weights": torch.full(
                (int(intervention_positions.numel()),),
                1.0 / int(intervention_positions.numel()),
                dtype=torch.float64,
            ),
            "record_semantic_row_weights": torch.full(
                (int(semantic_positions.numel()),),
                1.0 / int(semantic_positions.numel()),
                dtype=torch.float64,
            ),
            "projected_target.terminal": projected_terminal,
        }
        for group, target in projected_targets.items():
            feature_tensors[f"projected_target.{group}"] = target
        for subset in canonical_intervention_subsets(request["group_ids"]):
            is_base = subset["mask"] == 0
            replay = _run_causal_replay(
                child=child,
                reconstructed_embeddings=reconstructed,
                attention_mask=trajectory_mask.to(device=reconstructed.device),
                position_ids=(
                    None
                    if position_ids is None
                    else position_ids.to(device=reconstructed.device)
                ),
                intervention_positions=intervention_positions,
                semantic_positions=semantic_positions,
                operator_specs=operator_specs,
                intervention_members=subset["members"],
                projected_parent_targets=projected_targets,
                capture_full_features=is_base,
            )
            metrics, scalar_rows = _subset_metric_rows(
                replay=replay,
                semantic_positions=semantic_positions,
                base_full_trajectory=is_base,
                projected_parent_terminal=projected_terminal,
                parent_logits=parent_logits,
                defined_semantic_row_count=int(
                    request["metric_contract"]["defined_semantic_row_count"]
                ),
                routing=routing,
                output_routing_session=output_routing_session,
                maximum_donor_rows_per_score_chunk=(maximum_donor_rows_per_score_chunk),
            )
            coalition = tuple(subset["members"])
            subset_metrics[coalition] = metrics
            for metric_name, rows in scalar_rows.items():
                feature_tensors[f"subset.{subset['subset_id']}.{metric_name}"] = rows
            if is_base:
                for name, tensor in replay.items():
                    feature_tensors[f"base.{name}"] = tensor
            del replay
        feature_path = (
            feature_directory / f"batch-{int(reference.index):06d}.safetensors"
        )
        _atomic_publish_safetensors(feature_path, feature_tensors)
        feature_receipt = _build_feature_receipt(
            path=feature_path,
            tensors=feature_tensors,
            request_sha256=request["request_sha256"],
            batch_index=int(reference.index),
            trace_input_sha256=str(reference.input_sha256),
            trace_record_sha256=str(reference.record_sha256),
        )
        oracle_evidence = {
            "feature_receipt": feature_receipt,
            "selected_parent_payload_sha256": selected_parent_payload_sha256,
            "selected_parent_logical_bytes": logical_bytes,
            "full_trajectory_intervention_row_count": int(
                intervention_positions.numel()
            ),
            "semantic_metric_row_count": int(semantic_positions.numel()),
            "all_prefill_and_decode_rows_intervened": True,
            "semantic_positions_used_only_for_metrics": True,
            "live_feature_consumer_oracle_capture_available": True,
            "transported_feature_oracle_evaluated": False,
            "producer_mismatch_classification": None,
        }
        result = causal_module.build_lfm2_exact4x_causal_batch_result(
            request,
            batch_index=int(reference.index),
            subset_metrics=subset_metrics,
            child_forward_calls=8,
            oracle_evidence=oracle_evidence,
        )
    finally:
        trace_session.release_selected_tensors(payload)
        if progress is not None:
            progress.emit(
                "parent-selected-payload-released",
                batch_index=int(reference.index),
                selected_parent_payload_sha256=selected_parent_payload_sha256,
                selected_parent_logical_bytes=logical_bytes,
                parent_forward_calls=0,
            )
    return result, feature_receipt, feature_tensors


def _mean_relative_l2_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    rows = _relative_l2_rows(prediction, target)
    return rows, float(rows.mean().item())


def _consumer_oracle_evidence(
    *,
    child: Any,
    request: Mapping[str, Any],
    operator_specs: Sequence[Lfm2CausalOperatorSpec],
    feature_tensors: Sequence[Mapping[str, torch.Tensor]],
    progress: _ProgressJournal | None = None,
) -> dict[str, Any]:
    records = request["record_bindings"]
    if len(records) != len(feature_tensors):
        raise Lfm2Exact4xCausalRunnerError(
            "Live-feature oracle inputs differ from the causal record manifest."
        )
    operator_rows = []
    for spec in canonical_operator_specs(operator_specs):
        if spec.consumer_module_path is None:
            continue
        all_features = []
        all_targets = []
        all_record_ids = []
        all_splits = []
        for record, tensors in zip(records, feature_tensors, strict=True):
            feature = tensors.get(f"base.feature.{spec.group_id}")
            target = tensors.get(f"projected_target.{spec.group_id}")
            if (
                not isinstance(feature, torch.Tensor)
                or feature.ndim != 3
                or int(feature.shape[0]) != 1
                or not isinstance(target, torch.Tensor)
                or target.ndim != 2
                or int(feature.shape[1]) != int(target.shape[0])
            ):
                raise Lfm2Exact4xCausalRunnerError(
                    f"Live feature or projected target for {spec.group_id} is invalid."
                )
            rows = feature[0].to(dtype=torch.float64)
            all_features.append(rows)
            all_targets.append(target.to(dtype=torch.float64))
            all_record_ids.extend([record["record_id"]] * int(rows.shape[0]))
            all_splits.extend([record["split"]] * int(rows.shape[0]))
        feature_matrix = torch.cat(all_features, dim=0)
        target_matrix = torch.cat(all_targets, dim=0)
        fit_mask = torch.tensor(
            [split == "fit" for split in all_splits],
            dtype=torch.bool,
        )
        heldout_mask = ~fit_mask
        fit_features = feature_matrix[fit_mask]
        fit_targets = target_matrix[fit_mask]
        fit_record_ids = [
            record_id
            for record_id, selected in zip(
                all_record_ids, fit_mask.tolist(), strict=True
            )
            if selected
        ]
        weights = causal_module.equal_record_mass_weights(fit_record_ids)
        consumer = _module_at_path(child, spec.consumer_module_path)
        weight = getattr(consumer, "weight", None)
        bias = getattr(consumer, "bias", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or tuple(weight.shape)
            != (int(target_matrix.shape[1]), int(feature_matrix.shape[1]))
            or (
                bias is not None and tuple(bias.shape) != (int(target_matrix.shape[1]),)
            )
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"Consumer weight for {spec.group_id} has invalid oracle geometry."
            )
        baseline = weight.detach().to(device="cpu", dtype=torch.float64).T
        bias_row = (
            torch.zeros((1, int(target_matrix.shape[1])), dtype=torch.float64)
            if bias is None
            else bias.detach().to(device="cpu", dtype=torch.float64).reshape(1, -1)
        )
        if progress is not None:
            progress.emit(
                "live-feature-oracle-operator-start",
                group_id=spec.group_id,
                consumer_module_path=spec.consumer_module_path,
                fit_row_count=int(fit_features.shape[0]),
                feature_width=int(fit_features.shape[1]),
                output_width=int(fit_targets.shape[1]),
                solve_dtype="float64",
                solution_installed=False,
            )
        solution, solve_receipt = (
            causal_module.solve_lfm2_trace_scaled_weighted_anchored_ridge(
                fit_features,
                fit_targets - bias_row,
                baseline,
                weights,
                output_dtype=(
                    torch.bfloat16 if weight.dtype == torch.bfloat16 else torch.float32
                ),
            )
        )
        diagnostic_solution = solution.detach().to(dtype=torch.float64)
        split_rows = {}
        for split, mask in (("fit", fit_mask), ("heldout", heldout_mask)):
            selected_features = feature_matrix[mask]
            selected_targets = target_matrix[mask]
            current_prediction = selected_features @ baseline + bias_row
            oracle_prediction = selected_features @ diagnostic_solution + bias_row
            current_rows, _current_plain_mean = _mean_relative_l2_rows(
                current_prediction,
                selected_targets,
            )
            oracle_rows, _oracle_plain_mean = _mean_relative_l2_rows(
                oracle_prediction,
                selected_targets,
            )
            improvement = current_rows - oracle_rows
            selected_record_ids = [
                record_id
                for record_id, selected in zip(
                    all_record_ids,
                    mask.tolist(),
                    strict=True,
                )
                if selected
            ]
            split_weights = causal_module.equal_record_mass_weights(selected_record_ids)
            current_mean = float(torch.dot(current_rows, split_weights).item())
            oracle_mean = float(torch.dot(oracle_rows, split_weights).item())
            weighted_improvement = float(torch.dot(improvement, split_weights).item())
            split_rows[split] = {
                "row_count": int(mask.sum().item()),
                "record_count": len(set(selected_record_ids)),
                "current_mean_relative_l2": current_mean,
                "oracle_mean_relative_l2": oracle_mean,
                "mean_relative_l2_improvement": weighted_improvement,
                "record_total_mass": "equal",
                "within_record_row_mass": "uniform",
                "record_clustered_improvement_bootstrap": (
                    causal_module.deterministic_record_clustered_bootstrap(
                        improvement,
                        selected_record_ids,
                        seed=(
                            f"{request['bootstrap_contract']['seed']}\0"
                            f"live-consumer\0{spec.group_id}\0{split}"
                        ),
                        resample_count=int(
                            request["bootstrap_contract"]["resample_count"]
                        ),
                        confidence_level=float(
                            request["bootstrap_contract"]["confidence_level"]
                        ),
                    )
                ),
            }
        operator_rows.append(
            {
                "group_id": spec.group_id,
                "consumer_module_path": spec.consumer_module_path,
                "feature_endpoint": spec.feature_endpoint,
                "solve_receipt": dict(solve_receipt),
                "splits": split_rows,
                "solution_not_installed": True,
                "persistent_model_mutations": 0,
            }
        )
        if progress is not None:
            progress.emit(
                "live-feature-oracle-operator-complete",
                group_id=spec.group_id,
                solve_report_sha256=solve_receipt["report_sha256"],
                fit_mean_relative_l2_improvement=split_rows["fit"][
                    "mean_relative_l2_improvement"
                ],
                heldout_mean_relative_l2_improvement=split_rows["heldout"][
                    "mean_relative_l2_improvement"
                ],
                solution_installed=False,
                persistent_model_mutations=0,
            )
    body = {
        "schema": "neural-forge-lfm2-live-feature-consumer-oracles-v1",
        "request_sha256": request["request_sha256"],
        "operators": operator_rows,
        "transported_feature_oracle_evaluated": False,
        "producer_mismatch_classification": None,
        "consumer_dominance_is_diagnostic_not_a_numeric_gate": True,
        "ridge_linear_solve_count": len(operator_rows),
        "gradient_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "solutions_installed": 0,
    }
    return {**body, "oracle_sha256": canonical_sha256(body)}


def _implementation_identity(
    *,
    v3_implementation_identity: Mapping[str, Any],
    v3_relocation_receipt: Mapping[str, Any],
    runtime_device: torch.device,
    runtime_dtype_name: str,
) -> dict[str, Any]:
    v3_sha256 = _digest(
        v3_implementation_identity.get("implementation_sha256"),
        "v3 implementation SHA-256",
    )
    files = {
        "causal_runner": Path(__file__).resolve(),
        "causal_core": Path(causal_module.__file__).resolve(),
        "trace_scaled_anchored_ridge": Path(moving_gauge_module.__file__).resolve(),
    }
    body = {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in sorted(files.items())
        },
        "v3_implementation_sha256": v3_sha256,
        "v3_implementation_relocation": dict(v3_relocation_receipt),
        "runtime_device": str(runtime_device),
        "runtime_dtype": runtime_dtype_name,
        "execution_contract": {
            "offline_direct_local_child": True,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
            "child_replays_per_record": 8,
            "base_full_trajectory_replays_per_record": 1,
            "intervention_replays_per_record": 7,
            "full_trajectory_interventions": True,
            "semantic_rows_are_metric_sampling_only": True,
            "gradient_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "persistent_model_mutations": 0,
        },
    }
    return {**body, "implementation_sha256": canonical_sha256(body)}


def _authenticate_relocated_v3_implementation_identity(
    *,
    sealed_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove v3 code/runtime identity while allowing absolute checkout relocation."""

    sealed = json.loads(_canonical_bytes(dict(sealed_identity)).decode("ascii"))
    current = json.loads(_canonical_bytes(dict(current_identity)).decode("ascii"))
    expected_fields = {
        "execution_contract",
        "files",
        "implementation_sha256",
        "numeric_contract",
        "packages",
        "schema",
    }
    for identity, name in ((sealed, "sealed"), (current, "current")):
        claimed = identity.get("implementation_sha256")
        body = {
            key: value
            for key, value in identity.items()
            if key != "implementation_sha256"
        }
        if (
            set(identity) != expected_fields
            or _digest(claimed, f"{name} v3 implementation_sha256") != claimed
            or canonical_sha256(body) != claimed
            or not isinstance(identity.get("files"), Mapping)
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"{name.capitalize()} v3 implementation identity is invalid."
            )
    for field in ("schema", "packages", "numeric_contract", "execution_contract"):
        if current[field] != sealed[field]:
            raise Lfm2Exact4xCausalRunnerError(
                "Current v3 runtime differs from the sealed non-path contract."
            )
    if set(current["files"]) != set(sealed["files"]):
        raise Lfm2Exact4xCausalRunnerError(
            "Current v3 implementation file inventory differs from the sealed audit."
        )
    file_rows = []
    for name in sorted(sealed["files"]):
        sealed_file = sealed["files"][name]
        current_file = current["files"][name]
        if (
            not isinstance(sealed_file, Mapping)
            or not isinstance(current_file, Mapping)
            or set(sealed_file) != {"path", "sha256"}
            or set(current_file) != {"path", "sha256"}
            or _digest(sealed_file.get("sha256"), f"sealed v3 {name} SHA-256")
            != sealed_file["sha256"]
            or current_file.get("sha256") != sealed_file["sha256"]
            or not isinstance(sealed_file.get("path"), str)
            or not sealed_file["path"]
            or not isinstance(current_file.get("path"), str)
            or not current_file["path"]
        ):
            raise Lfm2Exact4xCausalRunnerError(
                f"Current v3 implementation file {name!r} is not byte-identical."
            )
        file_rows.append(
            {
                "name": name,
                "sha256": sealed_file["sha256"],
                "sealed_path": sealed_file["path"],
                "current_path": current_file["path"],
                "path_relocated": current_file["path"] != sealed_file["path"],
            }
        )
    body = {
        "schema": V3_IMPLEMENTATION_RELOCATION_SCHEMA,
        "sealed_implementation_sha256": sealed["implementation_sha256"],
        "current_implementation_sha256": current["implementation_sha256"],
        "files": file_rows,
        "all_file_sha256s_equal": True,
        "non_path_runtime_contract_equal": True,
        "absolute_paths_are_diagnostic_not_code_identity": True,
    }
    return {**body, "relocation_sha256": canonical_sha256(body)}


def _operator_specs_for_layer(
    layer_index: int,
    operator_kind: str = "short_conv",
) -> tuple[Lfm2CausalOperatorSpec, ...]:
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
    ):
        raise Lfm2Exact4xCausalRunnerError("operator_layer_index must be nonnegative.")
    if operator_kind not in {"short_conv", "full_attention"}:
        raise Lfm2Exact4xCausalRunnerError(
            "operator_kind must be short_conv or full_attention."
        )
    if layer_index == 0 and operator_kind == "short_conv":
        return default_lfm2_l0_csm_operator_specs()

    def shifted(path: str | None) -> str | None:
        if path is None:
            return None
        marker = "model.layers.0"
        if not path.startswith(marker):
            raise Lfm2Exact4xCausalRunnerError(
                "Default causal operator path cannot be shifted generically."
            )
        return f"model.layers.{layer_index}{path[len(marker) :]}"

    short_conv_specs = tuple(
        dataclasses.replace(
            spec,
            layer_index=layer_index,
            target_module_path=str(shifted(spec.target_module_path)),
            consumer_module_path=shifted(spec.consumer_module_path),
        )
        for spec in default_lfm2_l0_csm_operator_specs()
    )
    if operator_kind == "short_conv":
        return short_conv_specs
    c_spec, _short_conv_spec, m_spec = short_conv_specs
    attention_spec = Lfm2CausalOperatorSpec(
        group_id="A",
        stage_order=1,
        layer_index=layer_index,
        component_kind="attention_write",
        parent_endpoint="operator.output",
        target_module_path=f"model.layers.{layer_index}.self_attn",
        hook_kind="module_output",
        feature_endpoint="attention.context",
        consumer_module_path=f"model.layers.{layer_index}.self_attn.out_proj",
        consumer_hook_kind="module_input",
    )
    return (c_spec, attention_spec, m_spec)


def _authenticated_operator_kind(
    *,
    topology_layers: Sequence[Mapping[str, Any]],
    child_layers: Sequence[Any],
    layer_index: int,
    requested_kind: str,
) -> str:
    if (
        layer_index < 0
        or layer_index >= len(topology_layers)
        or layer_index >= len(child_layers)
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Requested operator layer leaves the authenticated topology."
        )
    contract = topology_layers[layer_index].get("contract")
    topology_kind = contract.get("kind") if isinstance(contract, Mapping) else None
    child_attention = getattr(child_layers[layer_index], "is_attention_layer", None)
    child_kind = (
        "full_attention"
        if child_attention is True
        else "short_conv"
        if child_attention is False
        else None
    )
    if topology_kind not in {"short_conv", "full_attention"} or (
        child_kind != topology_kind
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Child operator kind differs from the authenticated trace topology."
        )
    if requested_kind not in {"auto", "short_conv", "full_attention"}:
        raise Lfm2Exact4xCausalRunnerError("Requested operator kind is invalid.")
    if requested_kind != "auto" and requested_kind != topology_kind:
        raise Lfm2Exact4xCausalRunnerError(
            "Requested operator kind differs from the authenticated trace topology."
        )
    return str(topology_kind)


def _record_bindings(
    *,
    references: Sequence[Any],
    split_by_batch: Mapping[int, str],
    v3_ledger: Mapping[str, Any],
) -> tuple[Any, ...]:
    batch_sha256s = v3_ledger["ordered_batch_audit_sha256s"]
    if len(batch_sha256s) != len(references):
        raise Lfm2Exact4xCausalRunnerError(
            "V3 ledger batches differ from the trace references."
        )
    return tuple(
        causal_module.Lfm2CausalRecordBinding(
            batch_index=index,
            trace_input_sha256=str(reference.input_sha256),
            trace_record_sha256=str(reference.record_sha256),
            split=str(split_by_batch[index]),
            v3_batch_audit_sha256=str(batch_sha256s[index]),
        )
        for index, reference in enumerate(references)
    )


def _authenticate_v3_batch_results(
    *,
    first_divergence_directory: Path,
    v3_request: Mapping[str, Any],
    v3_ledger: Mapping[str, Any],
    references: Sequence[Any],
    split_by_batch: Mapping[int, str],
    checkpoint_authority: Any,
) -> tuple[dict[str, Any], ...]:
    results = []
    expected_sha256s = v3_ledger["ordered_batch_audit_sha256s"]
    if len(expected_sha256s) != len(references):
        raise Lfm2Exact4xCausalRunnerError(
            "V3 batch digest ledger differs from the trace record count."
        )
    for index, reference in enumerate(references):
        split = split_by_batch.get(index)
        if split not in {"fit", "heldout"}:
            raise Lfm2Exact4xCausalRunnerError(
                "V3 trace record lacks an authenticated split."
            )
        checkpoint = v3_runner._authenticate_batch_checkpoint(
            v3_runner._read_json(
                first_divergence_directory / "batches" / f"batch-{index:06d}.json",
                f"v3 batch {index} checkpoint",
            ),
            request=v3_request,
            reference=reference,
            split=split,
            authority=checkpoint_authority,
        )
        result = checkpoint["result"]
        if result["batch_audit_sha256"] != expected_sha256s[index]:
            raise Lfm2Exact4xCausalRunnerError(
                "V3 HMAC checkpoint differs from the ordered ledger digest."
            )
        results.append(dict(result))
    return tuple(results)


def _publish_or_authenticate_json(
    path: Path,
    value: Mapping[str, Any],
    name: str,
) -> None:
    if path.exists() or path.is_symlink():
        if v3_runner._read_json(path, name) != value:
            raise Lfm2Exact4xCausalRunnerError(
                f"Existing {name} differs from the authenticated causal run."
            )
    else:
        v3_runner._atomic_write_new_json(path, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-divergence-directory", type=Path, required=True)
    parser.add_argument("--first-divergence-ledger-sha256", required=True)
    parser.add_argument("--child-directory", type=Path, required=True)
    parser.add_argument("--child-package-sha256", required=True)
    parser.add_argument("--trace-cache", type=Path, required=True)
    parser.add_argument("--basis-phase-cache-root", type=Path, required=True)
    parser.add_argument("--basis-phase-key-sha256", required=True)
    parser.add_argument("--basis-phase-hmac-key-file", type=Path, required=True)
    parser.add_argument("--basis-phase-hmac-key-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--runtime-device", default="cuda:0")
    parser.add_argument("--runtime-dtype", default="float32")
    parser.add_argument("--operator-layer-index", type=int, default=0)
    parser.add_argument(
        "--operator-kind",
        choices=("auto", "short_conv", "full_attention"),
        default="auto",
    )
    parser.add_argument("--transported-feature-lineage-json", type=Path)
    parser.add_argument(
        "--bootstrap-seed",
        default="lfm2-exact4x-causal-diagnostic-v4-preregistered",
    )
    parser.add_argument("--bootstrap-resample-count", type=int, default=4096)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--maximum-selected-parent-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--maximum-donor-rows-per-score-chunk",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--maximum-new-batches",
        type=int,
        help=(
            "Execution-only limit for a bounded partial run; the full request is "
            "still authenticated and evidence semantics are unchanged."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    if args.maximum_new_batches is not None and args.maximum_new_batches < 0:
        raise Lfm2Exact4xCausalRunnerError(
            "maximum_new_batches must be a nonnegative integer."
        )
    child_directory = v3_runner._ordinary_directory(
        args.child_directory,
        "child directory",
    )
    trace_directory = v3_runner._ordinary_directory(
        args.trace_cache,
        "trace cache",
    )
    phase_root = v3_runner._ordinary_directory(
        args.basis_phase_cache_root,
        "balanced-basis phase cache root",
    )
    first_divergence_directory = v3_runner._ordinary_directory(
        args.first_divergence_directory,
        "first-divergence directory",
    )
    output = Path(args.output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise Lfm2Exact4xCausalRunnerError(
            "Causal output directory cannot be a symlink."
        )
    v3_request = v3_runner._read_json(
        first_divergence_directory / "request.json",
        "v3 first-divergence request",
    )
    v3_ledger = causal_module.authenticate_v3_first_divergence_ledger(
        v3_runner._read_json(
            first_divergence_directory / "first-divergence-ledger.json",
            "v3 first-divergence ledger",
        ),
        expected_request=v3_request,
    )
    if v3_ledger["ledger_sha256"] != _digest(
        args.first_divergence_ledger_sha256,
        "first_divergence_ledger_sha256",
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "V3 first-divergence ledger differs from the requested seal."
        )
    if v3_ledger["request"] != v3_request:
        raise Lfm2Exact4xCausalRunnerError(
            "V3 request file differs from the request embedded in its ledger."
        )
    v3_request = v3_ledger["request"]
    expected_child_package = _digest(
        args.child_package_sha256,
        "child_package_sha256",
    )
    if expected_child_package != v3_request["child_package_sha256"]:
        raise Lfm2Exact4xCausalRunnerError(
            "Requested child package differs from the v3 ledger."
        )
    runtime_device = torch.device(args.runtime_device)
    runtime_dtype = v3_runner._runtime_dtype(args.runtime_dtype)
    if (
        str(runtime_device) != v3_request["runtime_device"]
        or args.runtime_dtype != v3_request["runtime_dtype"]
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Causal runtime device/dtype differ from the authenticated v3 audit."
        )
    current_v3_identity = v3_runner._implementation_identity(
        runtime_device=runtime_device,
        runtime_dtype_name=args.runtime_dtype,
    )
    sealed_v3_identity = v3_request["implementation_identity"]
    v3_relocation = _authenticate_relocated_v3_implementation_identity(
        sealed_identity=sealed_v3_identity,
        current_identity=current_v3_identity,
    )
    v3_identity = sealed_v3_identity
    implementation = _implementation_identity(
        v3_implementation_identity=v3_identity,
        v3_relocation_receipt=v3_relocation,
        runtime_device=runtime_device,
        runtime_dtype_name=args.runtime_dtype,
    )
    transported_lineage = (
        None
        if args.transported_feature_lineage_json is None
        else v3_runner._read_json(
            Path(args.transported_feature_lineage_json),
            "transported-feature lineage",
        )
    )

    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        expected_package_sha256=expected_child_package,
    )
    canonical_child = load_lfm2_direct_local_model(
        child_directory,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=canonical_child,
        expected_package_sha256=package.package_sha256,
    )
    child, routing, relocation = relocate_and_bind_packaged_lfm2_carrier_routes(
        canonical_child,
        package.routes,
        provenance_resource_fingerprint=v3_identity["implementation_sha256"],
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        expected_target_rows=int(package.receipt["target_rows"]),
        expected_hidden_size=int(package.receipt["hidden_size"]),
    )
    del canonical_child
    gc.collect()
    child.eval()
    child_before = fingerprint_lfm2_parent(child)
    route_before = fingerprint_lfm2_carrier_runtime_owned_tensor_state(routing)
    if (
        child_before.identity_sha256 != v3_request["child_state_sha256"]
        or package.routes.package_sha256 != v3_request["route_package_sha256"]
        or routing.as_dict()["runtime_sha256"] != v3_request["route_runtime_sha256"]
        or relocation.as_dict()["receipt_sha256"]
        != v3_request["portable_runtime_receipt_sha256"]
    ):
        raise Lfm2Exact4xCausalRunnerError(
            "Loaded child or carrier route differs from the v3 evidence."
        )
    try:
        child_layers = tuple(child.model.layers)
    except (AttributeError, TypeError) as exc:
        raise Lfm2Exact4xCausalRunnerError(
            "Complete child has no iterable LFM2 layer stack."
        ) from exc
    layer_count = len(child_layers)
    if layer_count != int(v3_request["layer_count"]):
        raise Lfm2Exact4xCausalRunnerError(
            "Complete child depth differs from the v3 request."
        )
    secret = v3_runner._read_raw_hmac_key(args.basis_phase_hmac_key_file)
    try:
        basis_authority = PhaseAnchorAuthority(
            key_id=args.basis_phase_hmac_key_id,
            secret=secret,
        )
        checkpoint_authority = causal_module.Lfm2CausalCheckpointAuthority(
            key_id=args.basis_phase_hmac_key_id,
            secret=secret,
        )
        v3_checkpoint_authority = v3_runner.FirstDivergenceCheckpointAuthority(
            key_id=args.basis_phase_hmac_key_id,
            secret=secret,
        )
    finally:
        del secret
    trace_cache = open_lfm2_circuit_trace_cache(trace_directory, verify=False)
    if trace_cache.cache_key != v3_request["trace_cache_key"]:
        raise Lfm2Exact4xCausalRunnerError("Trace cache differs from the v3 request.")
    topology = trace_cache.manifest.get("layer_topology")
    topology_layers = topology.get("layers") if isinstance(topology, Mapping) else None
    if not isinstance(topology_layers, list) or len(topology_layers) != layer_count:
        raise Lfm2Exact4xCausalRunnerError(
            "Trace topology differs from the complete child."
        )
    source_widths = {
        layer.get("contract", {}).get("hidden_size")
        for layer in topology_layers
        if isinstance(layer, Mapping)
    }
    if len(source_widths) != 1:
        raise Lfm2Exact4xCausalRunnerError(
            "Trace topology has no single residual source width."
        )
    source_width = next(iter(source_widths))
    if isinstance(source_width, bool) or not isinstance(source_width, int):
        raise Lfm2Exact4xCausalRunnerError("Trace residual source width is invalid.")
    operator_kind = _authenticated_operator_kind(
        topology_layers=topology_layers,
        child_layers=child_layers,
        layer_index=args.operator_layer_index,
        requested_kind=args.operator_kind,
    )
    operator_specs = _operator_specs_for_layer(
        args.operator_layer_index,
        operator_kind,
    )
    for spec in operator_specs:
        _module_at_path(child, spec.target_module_path)
        if spec.consumer_module_path is not None:
            _module_at_path(child, spec.consumer_module_path)

    with trace_cache.selective_authenticated_session() as trace_session:
        if (
            trace_session.evidence_sha256
            != v3_request["trace_selective_session_sha256"]
        ):
            raise Lfm2Exact4xCausalRunnerError(
                "Trace selective-session evidence differs from v3."
            )
        basis_phase = load_existing_lfm2_circuit_balanced_basis_phase(
            DynamicFoldPhaseCache(phase_root),
            args.basis_phase_key_sha256,
            anchor_authority=basis_authority,
            expected_parent_identity_sha256=trace_cache.parent_fingerprint,
            expected_trace_cache_key=trace_cache.cache_key,
            expected_trace_selective_session_sha256=trace_session.evidence_sha256,
            expected_source_width=source_width,
            expected_target_width=int(package.receipt["hidden_size"]),
        )
        if (
            basis_phase.phase_key.phase_key_sha256
            != v3_request["basis_phase_key_sha256"]
            or basis_phase.completion_sha256 != v3_request["basis_completion_sha256"]
            or basis_phase.map_sha256 != v3_request["basis_map_sha256"]
        ):
            raise Lfm2Exact4xCausalRunnerError(
                "Balanced basis differs from the v3 request."
            )
        budget = verify_complete_exact_budget_child(
            child,
            basis_phase=basis_phase,
            route_package=package.routes,
            portable_runtime_receipt=relocation,
        )
        if budget != v3_request["budget_observation"]:
            raise Lfm2Exact4xCausalRunnerError(
                "Exact-budget child observation differs from v3."
            )
        references = trace_session.batch_references
        split_by_batch = authoritative_split_by_batch(
            basis_phase.report,
            references,
        )
        records = _record_bindings(
            references=references,
            split_by_batch=split_by_batch,
            v3_ledger=v3_ledger,
        )
        v3_batch_results = _authenticate_v3_batch_results(
            first_divergence_directory=first_divergence_directory,
            v3_request=v3_request,
            v3_ledger=v3_ledger,
            references=references,
            split_by_batch=split_by_batch,
            checkpoint_authority=v3_checkpoint_authority,
        )
        request = causal_module.build_lfm2_exact4x_causal_diagnostic_request(
            first_divergence_request=v3_request,
            first_divergence_ledger=v3_ledger,
            first_divergence_batch_results=v3_batch_results,
            record_bindings=records,
            operator_specs=operator_specs,
            implementation_identity_sha256=implementation["implementation_sha256"],
            checkpoint_hmac_key_id=checkpoint_authority.key_id,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resample_count=args.bootstrap_resample_count,
            bootstrap_confidence_level=args.bootstrap_confidence_level,
            transported_feature_lineage=transported_lineage,
        )
        _publish_or_authenticate_json(
            output / "request.json",
            request,
            "causal request",
        )
        progress = _ProgressJournal(
            output / "progress.jsonl",
            request["request_sha256"],
        )
        progress.emit(
            "request-authenticated",
            v3_request_sha256=v3_request["request_sha256"],
            v3_ledger_sha256=v3_ledger["ledger_sha256"],
            authenticated_v3_batch_result_count=len(v3_batch_results),
            child_package_sha256=package.package_sha256,
            child_state_sha256=child_before.identity_sha256,
            route_package_sha256=package.routes.package_sha256,
            route_state_sha256=route_before["state_sha256"],
            trace_cache_key=trace_cache.cache_key,
            basis_map_sha256=basis_phase.map_sha256,
            operator_layer_index=args.operator_layer_index,
            operator_kind=operator_kind,
            group_ids=request["group_ids"],
            full_trajectory_interventions=True,
            semantic_positions_metrics_only=True,
            parent_forward_calls=0,
        )
        input_table = child.get_input_embeddings().weight
        output_table = child.get_output_embeddings().weight
        paired_sessions = routing.authenticated_live_table_sessions(
            input_table,
            output_table,
        )
        with paired_sessions as (input_session, output_session):

            def processor(reference: Any, record: Mapping[str, Any]):
                return _process_causal_batch(
                    reference=reference,
                    record=record,
                    request=request,
                    trace_session=trace_session,
                    child=child,
                    routing=routing,
                    input_routing_session=input_session,
                    output_routing_session=output_session,
                    coordinate_map=basis_phase,
                    operator_specs=operator_specs,
                    layer_count=layer_count,
                    source_width=source_width,
                    feature_directory=output / "features",
                    maximum_selected_parent_bytes=(args.maximum_selected_parent_bytes),
                    maximum_donor_rows_per_score_chunk=(
                        args.maximum_donor_rows_per_score_chunk
                    ),
                    progress=progress,
                )

            resume = resume_causal_batches(
                request=request,
                references=references,
                checkpoint_directory=output / "batches",
                feature_directory=output / "features",
                checkpoint_authority=checkpoint_authority,
                processor=processor,
                progress=progress,
                maximum_new_batches=args.maximum_new_batches,
            )
        trace_session.verify()

    child_after = fingerprint_lfm2_parent(child)
    route_after = fingerprint_lfm2_carrier_runtime_owned_tensor_state(routing)
    if child_after != child_before or route_after != route_before:
        raise Lfm2Exact4xCausalRunnerError(
            "Child or carrier route changed during causal observation."
        )
    if not resume.is_complete:
        partial = _build_partial_completion_receipt(
            request=request,
            resume=resume,
            checkpoint_authority=checkpoint_authority,
            child_package_sha256=package.package_sha256,
            child_state_sha256=child_after.identity_sha256,
            route_package_sha256=package.routes.package_sha256,
            route_state_sha256=route_after["state_sha256"],
            implementation_identity=implementation,
        )
        _publish_or_authenticate_json(
            output
            / "partial-completions"
            / f"prefix-{resume.completed_batch_count:06d}.json",
            partial,
            "causal partial completion",
        )
        progress.emit(
            "run-partial",
            partial_completion_sha256=partial["partial_completion_sha256"],
            completed_batch_count=resume.completed_batch_count,
            remaining_batch_count=resume.remaining_batch_count,
            maximum_new_batches=args.maximum_new_batches,
            batch_resume=resume.as_dict(),
            aggregate_evaluated=False,
            live_feature_oracles_evaluated=False,
            parent_forward_calls=0,
            training_steps=0,
            optimizer_steps=0,
            persistent_model_mutations=0,
        )
        return partial
    progress.emit(
        "live-feature-oracles-start",
        checkpoint_hit_count=resume.checkpoint_hit_count,
        computed_batch_count=resume.computed_batch_count,
        current_run_child_forward_calls=resume.current_run_child_forward_calls,
    )
    oracle = _consumer_oracle_evidence(
        child=child,
        request=request,
        operator_specs=operator_specs,
        feature_tensors=resume.feature_tensors,
        progress=progress,
    )
    child_final = fingerprint_lfm2_parent(child)
    route_final = fingerprint_lfm2_carrier_runtime_owned_tensor_state(routing)
    if child_final != child_before or route_final != route_before:
        raise Lfm2Exact4xCausalRunnerError(
            "Child or carrier route changed during live-feature oracle analysis."
        )
    _publish_or_authenticate_json(
        output / "live-feature-consumer-oracles.json",
        oracle,
        "live-feature consumer oracle evidence",
    )
    progress.emit(
        "live-feature-oracles-complete",
        oracle_sha256=oracle["oracle_sha256"],
        ridge_linear_solve_count=oracle["ridge_linear_solve_count"],
        analytic_solutions_installed=0,
    )
    ledger = causal_module.aggregate_lfm2_exact4x_causal_checkpoints(
        request,
        resume.checkpoints,
        authority=checkpoint_authority,
    )
    ledger = causal_module.authenticate_lfm2_exact4x_causal_ledger(
        ledger,
        request=request,
        checkpoints=resume.checkpoints,
        authority=checkpoint_authority,
    )
    _publish_or_authenticate_json(
        output / "causal-ledger.json",
        ledger,
        "causal ledger",
    )
    progress.emit(
        "causal-ledger-complete",
        ledger_sha256=ledger["ledger_sha256"],
        persisted_child_forward_calls=ledger["operation_audit"]["child_forward_calls"],
        parent_forward_calls=0,
    )
    completion_body = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete-observation",
        "request_sha256": request["request_sha256"],
        "causal_ledger_sha256": ledger["ledger_sha256"],
        "live_feature_oracle_sha256": oracle["oracle_sha256"],
        "child_package_sha256": package.package_sha256,
        "child_state_sha256": child_final.identity_sha256,
        "route_package_sha256": package.routes.package_sha256,
        "route_state_sha256": route_final["state_sha256"],
        "implementation_identity": implementation,
        "batch_count": len(resume.results),
        "parent_forward_calls": 0,
        "persisted_child_forward_calls": ledger["operation_audit"][
            "child_forward_calls"
        ],
        "completion_scope": "stable-persisted-evidence-across-invocations",
        "invocation_relative_resume_accounting_is_in_progress_journal": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "persistent_model_mutations": 0,
        "analytic_diagnostic_linear_solves": oracle["ridge_linear_solve_count"],
        "analytic_solutions_installed": 0,
        "transported_feature_oracle_available": request[
            "transported_feature_availability"
        ]["transported_feature_oracle_available"],
        "numeric_acceptance_decision": None,
        "manual_whole_response_intelligence_review_is_authoritative": True,
    }
    completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    _publish_or_authenticate_json(
        output / "complete.json",
        completion,
        "causal completion",
    )
    progress.emit(
        "run-complete",
        completion_sha256=completion["completion_sha256"],
        causal_ledger_sha256=ledger["ledger_sha256"],
        live_feature_oracle_sha256=oracle["oracle_sha256"],
        batch_resume=resume.as_dict(),
        parent_forward_calls=0,
        training_steps=0,
        optimizer_steps=0,
        persistent_model_mutations=0,
    )
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        completion = run(args)
    except (
        Lfm2Exact4xCausalDiagnosticError,
        Lfm2Exact4xCausalRunnerError,
        RuntimeError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    sys.stdout.write(_canonical_bytes(completion).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLETION_SCHEMA",
    "FEATURE_RECEIPT_SCHEMA",
    "IMPLEMENTATION_SCHEMA",
    "PARTIAL_COMPLETION_SCHEMA",
    "Lfm2Exact4xCausalRunnerError",
    "RUNNER_SCHEMA",
    "CausalBatchResume",
    "main",
    "resume_causal_batches",
    "run",
]
