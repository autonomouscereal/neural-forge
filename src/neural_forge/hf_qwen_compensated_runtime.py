"""Standalone ragged Qwen3 runtime for authenticated compensated folds."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .hf_qwen_compensated_neuron_fold import Qwen3CompensatedMLP
from .hf_qwen_direct_local import (
    Qwen3DirectLocalError,
    read_json_object,
    rebuild_qwen3_runtime_buffers,
    stream_local_safetensors_into_model,
    temporary_default_dtype,
)


RUNTIME_SCHEMA = "neural-forge-qwen3-compensated-runtime-v1"
LAYOUT_ATTRIBUTE = "neural_forge_compensated_layout"
REPORT_ATTRIBUTE = "neural_forge_compensated_report"


class Qwen3CompensatedRuntimeError(RuntimeError):
    """A compensated Qwen3 topology or standalone artifact is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen3CompensatedRuntimeError(f"{name} must be a positive integer.")
    return value


def validate_compute_indices(
    compute_width: int,
    stored_width: int,
    compute_indices: Sequence[int],
) -> tuple[int, ...]:
    source = _positive_int(compute_width, "compute_width")
    stored = _positive_int(stored_width, "stored_width")
    indices = tuple(int(value) for value in compute_indices)
    if (
        source < stored
        or len(indices) != stored
        or len(set(indices)) != stored
        or any(value < 0 or value >= source for value in indices)
        or tuple(sorted(indices)) != indices
    ):
        raise Qwen3CompensatedRuntimeError(
            "Source-padded indices must be a sorted unique stored-width "
            "subset of the compute width."
        )
    return indices


class Qwen3SourcePaddedDownProjection:
    """Factory namespace retained for a stable public runtime symbol."""

    @staticmethod
    def create(
        stored_width: int,
        hidden_size: int,
        *,
        compute_width: int,
        compute_indices: Sequence[int],
        max_transient_compute_bytes: int,
        device: Any = None,
        dtype: Any = None,
    ) -> Any:
        import torch
        from torch import nn

        class _Projection(nn.Linear):
            def __init__(self) -> None:
                super().__init__(
                    stored_width,
                    hidden_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                self.compute_width = _positive_int(
                    compute_width,
                    "compute_width",
                )
                self.compute_indices = validate_compute_indices(
                    self.compute_width,
                    stored_width,
                    compute_indices,
                )
                self.max_transient_compute_bytes = _positive_int(
                    max_transient_compute_bytes,
                    "max_transient_compute_bytes",
                )
                self.peak_transient_compute_bytes = 0

            def forward(self, compact_activations: Any) -> Any:
                from torch.nn import functional as functional

                if int(compact_activations.shape[-1]) != self.in_features:
                    raise Qwen3CompensatedRuntimeError(
                        "Source-padded down input differs from stored width."
                    )
                rows = int(compact_activations.numel()) // self.in_features
                transient = (
                    rows * self.compute_width + self.out_features * self.compute_width
                ) * int(compact_activations.element_size())
                self.peak_transient_compute_bytes = max(
                    self.peak_transient_compute_bytes,
                    transient,
                )
                if transient > self.max_transient_compute_bytes:
                    raise Qwen3CompensatedRuntimeError(
                        "Source-padded down projection exceeds its transient "
                        f"ceiling: {transient}>{self.max_transient_compute_bytes}."
                    )
                indices = torch.tensor(
                    self.compute_indices,
                    dtype=torch.long,
                    device=compact_activations.device,
                )
                full_activations = compact_activations.new_zeros(
                    (*compact_activations.shape[:-1], self.compute_width)
                )
                full_activations.index_copy_(
                    -1,
                    indices,
                    compact_activations,
                )
                full_weight = self.weight.new_zeros(
                    (self.out_features, self.compute_width)
                )
                full_weight.index_copy_(1, indices, self.weight)
                return functional.linear(full_activations, full_weight)

        return _Projection()


class Qwen3SourcePaddedCompensatedMLP(Qwen3CompensatedMLP):
    """Persist a smaller MLP while restoring source GEMM dimensions transiently."""

    def __init__(
        self,
        hidden_size: int,
        stored_width: int,
        act_fn: Any,
        *,
        compute_width: int,
        compute_indices: Sequence[int],
        max_transient_compute_bytes: int,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(hidden_size, stored_width, act_fn)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)
        self.compute_width = _positive_int(compute_width, "compute_width")
        self.compute_indices = validate_compute_indices(
            self.compute_width,
            stored_width,
            compute_indices,
        )
        self.max_transient_compute_bytes = _positive_int(
            max_transient_compute_bytes,
            "max_transient_compute_bytes",
        )
        old_down = self.down_proj
        self.down_proj = Qwen3SourcePaddedDownProjection.create(
            stored_width,
            hidden_size,
            compute_width=self.compute_width,
            compute_indices=self.compute_indices,
            max_transient_compute_bytes=self.max_transient_compute_bytes,
            device=old_down.weight.device,
            dtype=old_down.weight.dtype,
        )
        self.peak_transient_compute_bytes = 0

    def forward(self, hidden_states: Any) -> Any:
        import torch
        from torch.nn import functional as functional

        if int(hidden_states.shape[-1]) != self.gate_proj.in_features:
            raise Qwen3CompensatedRuntimeError(
                "Source-padded MLP hidden width differs from stored weights."
            )
        rows = int(hidden_states.numel()) // int(hidden_states.shape[-1])
        element_size = int(hidden_states.element_size())
        full_weight_elements = 3 * self.compute_width * self.gate_proj.in_features
        full_activation_elements = 4 * rows * self.compute_width
        transient = (full_weight_elements + full_activation_elements) * element_size
        self.peak_transient_compute_bytes = max(
            self.peak_transient_compute_bytes,
            transient,
        )
        if transient > self.max_transient_compute_bytes:
            raise Qwen3CompensatedRuntimeError(
                "Source-padded MLP exceeds its transient ceiling: "
                f"{transient}>{self.max_transient_compute_bytes}."
            )
        indices = torch.tensor(
            self.compute_indices,
            dtype=torch.long,
            device=hidden_states.device,
        )
        full_gate = self.gate_proj.weight.new_zeros(
            (self.compute_width, self.gate_proj.in_features)
        )
        full_gate.index_copy_(0, indices, self.gate_proj.weight)
        gate_values = self.act_fn(functional.linear(hidden_states, full_gate))
        del full_gate
        full_up = self.up_proj.weight.new_zeros(
            (self.compute_width, self.up_proj.in_features)
        )
        full_up.index_copy_(0, indices, self.up_proj.weight)
        up_values = functional.linear(hidden_states, full_up)
        del full_up
        compact = (gate_values * up_values).index_select(-1, indices)
        return self.down_proj(compact)

    def compute_shape_report(self) -> dict[str, Any]:
        return {
            "mode": "source-padded",
            "stored_width": int(self.gate_proj.out_features),
            "compute_width": self.compute_width,
            "compute_indices": list(self.compute_indices),
            "compute_indices_sha256": _canonical_sha256(self.compute_indices),
            "max_transient_compute_bytes": self.max_transient_compute_bytes,
            "peak_transient_compute_bytes": max(
                self.peak_transient_compute_bytes,
                self.down_proj.peak_transient_compute_bytes,
            ),
            "persistent_full_width_parameter_or_buffer": False,
            "transient_full_width_materialization": True,
            "stored_parameter_width_is_target_width": True,
        }


def qwen3_layers(model: Any) -> tuple[Any, ...]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise Qwen3CompensatedRuntimeError(
            "Qwen3 runtime has no decoder layer collection."
        )
    result = tuple(layers)
    if not result:
        raise Qwen3CompensatedRuntimeError("Qwen3 runtime has no decoder layers.")
    return result


def adjusted_qwen3_layout(
    dense_layout: Any,
    layer_widths: Sequence[int],
) -> Any:
    widths = tuple(_positive_int(int(value), "layer width") for value in layer_widths)
    blocks = tuple(dense_layout.blocks)
    if len(widths) != len(blocks):
        raise Qwen3CompensatedRuntimeError(
            "Layer-width metadata differs from the Qwen3 decoder depth."
        )
    adjusted = tuple(
        replace(
            block,
            mlp=replace(block.mlp, intermediate_size=widths[block.index]),
        )
        for block in blocks
    )
    return replace(dense_layout, blocks=adjusted)


def attach_compensated_runtime_metadata(
    model: Any,
    dense_layout: Any,
) -> dict[str, Any]:
    layers = qwen3_layers(model)
    widths = tuple(int(layer.mlp.gate_proj.out_features) for layer in layers)
    compact_layers = tuple(
        index
        for index, layer in enumerate(layers)
        if isinstance(layer.mlp, Qwen3SourcePaddedCompensatedMLP)
    )
    if not compact_layers:
        raise Qwen3CompensatedRuntimeError(
            "Compensated runtime has no source-padded compact layer."
        )
    layout = adjusted_qwen3_layout(dense_layout, widths)
    report = {
        "schema": RUNTIME_SCHEMA,
        "layer_widths": list(widths),
        "configured_intermediate_size": int(model.config.intermediate_size),
        "compact_layer_indices": list(compact_layers),
        "parameter_count": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "zero_optimizer_steps": 0,
        "backward_called": False,
    }
    setattr(model, LAYOUT_ATTRIBUTE, layout)
    setattr(model, REPORT_ATTRIBUTE, report)
    return report


def create_blank_compensated_qwen3(
    parent_model: Any,
    *,
    layer_index: int,
    target_width: int,
    compute_indices: Sequence[int],
    max_transient_compute_bytes: int,
) -> Any:
    """Materialize a deterministic zero blank with the final ragged topology."""

    import torch
    from transformers import Qwen3ForCausalLM

    from .hf_arch_layout import build_decoder_layout

    parent_layers = qwen3_layers(parent_model)
    layer = int(layer_index)
    if layer < 0 or layer >= len(parent_layers):
        raise Qwen3CompensatedRuntimeError("layer_index is out of range.")
    devices = {parameter.device for parameter in parent_model.parameters()}
    dtypes = {parameter.dtype for parameter in parent_model.parameters()}
    if len(devices) != 1 or len(dtypes) != 1:
        raise Qwen3CompensatedRuntimeError(
            "Compensated target synthesis requires one parent device and dtype."
        )
    device = next(iter(devices))
    dtype = next(iter(dtypes))
    config = copy.deepcopy(parent_model.config)
    with temporary_default_dtype(dtype), torch.device("meta"):
        child = Qwen3ForCausalLM(config)
        dense_layout = build_decoder_layout(child)
        original = qwen3_layers(child)[layer].mlp
        qwen3_layers(child)[layer].mlp = Qwen3SourcePaddedCompensatedMLP(
            int(config.hidden_size),
            _positive_int(target_width, "target_width"),
            original.act_fn,
            compute_width=int(config.intermediate_size),
            compute_indices=compute_indices,
            max_transient_compute_bytes=max_transient_compute_bytes,
        )
    child.to_empty(device=device)
    child.tie_weights()
    rebuild_qwen3_runtime_buffers(child, device)
    with torch.no_grad():
        seen: set[int] = set()
        for parameter in child.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameter.zero_()
    child.eval()
    attach_compensated_runtime_metadata(child, dense_layout)
    return child


def validate_compensated_qwen3_runtime(model: Any) -> dict[str, Any]:
    """Prove that attached layout metadata matches every live ragged MLP."""

    from .hf_arch_layout import DecoderLayout, resolve_module

    layout = getattr(model, LAYOUT_ATTRIBUTE, None)
    report = getattr(model, REPORT_ATTRIBUTE, None)
    if not isinstance(layout, DecoderLayout) or not isinstance(report, Mapping):
        raise Qwen3CompensatedRuntimeError(
            "Qwen3 compensated layout/report metadata is missing."
        )
    if report.get("schema") != RUNTIME_SCHEMA:
        raise Qwen3CompensatedRuntimeError(
            "Qwen3 compensated runtime schema is unsupported."
        )
    layers = qwen3_layers(model)
    widths = tuple(int(layer.mlp.gate_proj.out_features) for layer in layers)
    if list(widths) != report.get("layer_widths") or len(layout.blocks) != len(layers):
        raise Qwen3CompensatedRuntimeError(
            "Attached Qwen3 layer widths differ from the live runtime."
        )
    compact_indices = []
    hidden = int(model.config.hidden_size)
    for block, layer in zip(layout.blocks, layers, strict=True):
        mlp = resolve_module(model, block.mlp.module_path, family="qwen3")
        width = widths[block.index]
        if mlp is not layer.mlp or block.mlp.intermediate_size != width:
            raise Qwen3CompensatedRuntimeError(
                f"Compensated layout differs at layer {block.index}."
            )
        expected_shapes = {
            "gate": (width, hidden),
            "up": (width, hidden),
            "down": (hidden, width),
        }
        actual_shapes = {
            "gate": tuple(mlp.gate_proj.weight.shape),
            "up": tuple(mlp.up_proj.weight.shape),
            "down": tuple(mlp.down_proj.weight.shape),
        }
        if actual_shapes != expected_shapes:
            raise Qwen3CompensatedRuntimeError(
                f"Compensated MLP shapes differ at layer {block.index}."
            )
        if isinstance(mlp, Qwen3SourcePaddedCompensatedMLP):
            compact_indices.append(block.index)
            validate_compute_indices(
                mlp.compute_width,
                width,
                mlp.compute_indices,
            )
            if mlp.compute_width != int(model.config.intermediate_size):
                raise Qwen3CompensatedRuntimeError(
                    "Source-padded compute width differs from Qwen3 config."
                )
        elif width != int(model.config.intermediate_size):
            raise Qwen3CompensatedRuntimeError(
                "A ragged Qwen3 layer lacks the source-padded runtime."
            )
    if compact_indices != list(report.get("compact_layer_indices", [])):
        raise Qwen3CompensatedRuntimeError(
            "Compact layer inventory differs from attached metadata."
        )
    actual_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    if actual_parameters != int(report.get("parameter_count", -1)):
        raise Qwen3CompensatedRuntimeError(
            "Compensated runtime parameter accounting differs."
        )
    if str(next(model.parameters()).dtype) != report.get("parameter_dtype"):
        raise Qwen3CompensatedRuntimeError(
            "Compensated runtime parameter dtype differs."
        )
    return dict(report)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(dict(value)))
        handle.write(b"\n")


def save_compensated_qwen3(
    model: Any,
    directory: str | Path,
    *,
    tokenizer_source: str | Path | None,
    transaction_records: Sequence[Mapping[str, Any]],
    evidence_binding: Mapping[str, Any],
) -> None:
    """Save one safe, standalone compensated Qwen3 child."""

    from safetensors.torch import save_model

    report = validate_compensated_qwen3_runtime(model)
    destination = Path(directory).expanduser().absolute()
    if not destination.is_dir() or any(destination.iterdir()):
        raise Qwen3CompensatedRuntimeError(
            "Compensated saver requires an existing empty destination."
        )
    config = model.config.to_dict()
    _write_json(destination / "config.json", config)
    generation = getattr(model, "generation_config", None)
    if generation is not None and callable(getattr(generation, "to_dict", None)):
        _write_json(
            destination / "generation_config.json",
            generation.to_dict(),
        )
    transactions = tuple(dict(value) for value in transaction_records)
    manifest = {
        "schema": RUNTIME_SCHEMA,
        "report": report,
        "transactions": list(transactions),
        "transactions_sha256": _canonical_sha256(transactions),
        "evidence_binding": dict(evidence_binding),
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "pickle_allowed": False,
        "auto_loader_allowed": False,
    }
    _write_json(destination / "fold_manifest.json", manifest)
    save_model(
        model,
        str(destination / "model.safetensors"),
        metadata={
            "format": "pt",
            "schema": RUNTIME_SCHEMA,
            "loader": "neural_forge.hf_qwen_compensated_runtime",
        },
    )
    if tokenizer_source is not None:
        source = Path(tokenizer_source).expanduser().resolve(strict=True)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ):
            path = source / name
            if path.is_file():
                shutil.copy2(path, destination / name)


def load_compensated_qwen3(
    directory: str | Path,
    *,
    device: str | Any,
) -> Any:
    """Load the standalone child using concrete local Qwen3 classes only."""

    import torch
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

    from .hf_arch_layout import build_decoder_layout

    source = Path(directory).expanduser().resolve(strict=True)
    config_payload = read_json_object(source / "config.json", "Qwen3 config")
    manifest = read_json_object(
        source / "fold_manifest.json",
        "compensated fold manifest",
    )
    if (
        config_payload.get("model_type") != "qwen3"
        or manifest.get("schema") != RUNTIME_SCHEMA
        or manifest.get("pickle_allowed") is not False
        or manifest.get("auto_loader_allowed") is not False
    ):
        raise Qwen3CompensatedRuntimeError(
            "Standalone compensated Qwen3 contract is invalid."
        )
    report = manifest.get("report")
    if not isinstance(report, Mapping):
        raise Qwen3CompensatedRuntimeError(
            "Standalone compensated report is malformed."
        )
    widths = tuple(int(value) for value in report.get("layer_widths", ()))
    compact_indices = tuple(
        int(value) for value in report.get("compact_layer_indices", ())
    )
    if len(compact_indices) != 1:
        raise Qwen3CompensatedRuntimeError(
            "This production loader requires exactly one compact Qwen3 layer."
        )
    transactions = manifest.get("transactions")
    if not isinstance(transactions, list) or manifest.get(
        "transactions_sha256"
    ) != _canonical_sha256(tuple(transactions)):
        raise Qwen3CompensatedRuntimeError(
            "Standalone transaction manifest integrity failed."
        )
    config_payload.pop("model_type", None)
    config = Qwen3Config(**config_payload)
    target = torch.device(device)
    dtype_name = str(report.get("parameter_dtype") or "")
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise Qwen3CompensatedRuntimeError(
            f"Unsupported construction dtype: {dtype_name}."
        )
    layer_index = compact_indices[0]
    compute_indices = tuple(range(int(config.intermediate_size)))
    for transaction in transactions:
        proposal = transaction.get("proposal", transaction)
        merge_rows = proposal.get("merges")
        if (
            int(proposal.get("layer_index", -1)) != layer_index
            or not isinstance(merge_rows, list)
            or len(merge_rows) != 1
        ):
            raise Qwen3CompensatedRuntimeError(
                "Standalone compensated transaction is malformed."
            )
        dropped = int(merge_rows[0]["dropped"])
        compute_indices = tuple(
            value for index, value in enumerate(compute_indices) if index != dropped
        )
    with temporary_default_dtype(dtype), torch.device("meta"):
        model = Qwen3ForCausalLM(config)
        dense_layout = build_decoder_layout(model)
        original = qwen3_layers(model)[layer_index].mlp
        qwen3_layers(model)[layer_index].mlp = Qwen3SourcePaddedCompensatedMLP(
            int(config.hidden_size),
            widths[layer_index],
            original.act_fn,
            compute_width=int(config.intermediate_size),
            compute_indices=compute_indices,
            max_transient_compute_bytes=2 * 1024**3,
        )
    model.to_empty(device=target)
    model.tie_weights()
    rebuild_qwen3_runtime_buffers(model, target)
    try:
        stream_local_safetensors_into_model(model, source)
    except Qwen3DirectLocalError as exc:
        raise Qwen3CompensatedRuntimeError(
            "Standalone compensated tensor loading failed."
        ) from exc
    model.tie_weights()
    generation_path = source / "generation_config.json"
    if generation_path.is_file():
        model.generation_config = GenerationConfig(
            **read_json_object(generation_path, "generation config")
        )
    model.requires_grad_(False)
    model.eval()
    attach_compensated_runtime_metadata(model, dense_layout)
    validate_compensated_qwen3_runtime(model)
    return model


__all__ = [
    "LAYOUT_ATTRIBUTE",
    "REPORT_ATTRIBUTE",
    "RUNTIME_SCHEMA",
    "Qwen3CompensatedRuntimeError",
    "Qwen3SourcePaddedCompensatedMLP",
    "Qwen3SourcePaddedDownProjection",
    "adjusted_qwen3_layout",
    "attach_compensated_runtime_metadata",
    "create_blank_compensated_qwen3",
    "load_compensated_qwen3",
    "qwen3_layers",
    "save_compensated_qwen3",
    "validate_compensated_qwen3_runtime",
    "validate_compute_indices",
]
