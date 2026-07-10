"""Deterministic activation traces for supported Hugging Face decoders.

The collector in this module deliberately follows :mod:`hf_arch_layout`'s
validated operator paths.  It is a calibration-only operation: the model is
temporarily put in evaluation mode, the forward pass runs under
``torch.inference_mode()``, and every retained tensor is a detached CPU copy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_arch_layout import (
    DecoderBlockLayout,
    DecoderLayout,
    build_decoder_layout,
    resolve_module,
)


class TransportTraceError(RuntimeError):
    """Raised when a decoder cannot produce a complete, trustworthy trace."""


@dataclass(frozen=True, slots=True)
class DecoderBlockTrace:
    """Captured residual, attention, and MLP states for one decoder block."""

    index: int
    block_input: Any
    pre_attention_norm_output: Any
    fused_qkv_projection_output: Any | None
    query_projection_output: Any
    key_projection_output: Any
    value_projection_output: Any
    attention_output_projection_input: Any
    attention_output_projection_output: Any
    attention_delta: Any
    post_attention_residual: Any
    pre_mlp_norm_output: Any
    mlp_input_projection_output: Any | None
    mlp_gate_projection_output: Any | None
    mlp_up_projection_output: Any | None
    activation_output: Any
    down_projection_input: Any
    down_projection_output: Any
    mlp_delta: Any
    block_output: Any

    @property
    def q_projection_output(self) -> Any:
        """Alias for :attr:`query_projection_output`."""

        return self.query_projection_output

    @property
    def k_projection_output(self) -> Any:
        """Alias for :attr:`key_projection_output`."""

        return self.key_projection_output

    @property
    def v_projection_output(self) -> Any:
        """Alias for :attr:`value_projection_output`."""

        return self.value_projection_output

    @property
    def activated_intermediate(self) -> Any:
        """The dense or gated intermediate immediately consumed by down-proj."""

        return self.down_projection_input

    def flatten_valid(self, valid_token_mask: Any) -> DecoderBlockTrace:
        """Return this block with each tensor reduced to valid rows in B/S order."""

        values: dict[str, Any] = {}
        for name in _BLOCK_TENSOR_FIELDS:
            value = getattr(self, name)
            values[name] = (
                None
                if value is None
                else _flatten_valid_tensor(value, valid_token_mask, name)
            )
        return DecoderBlockTrace(index=self.index, **values)


@dataclass(frozen=True, slots=True)
class FlattenedTransportTrace:
    """A deterministic row-major view containing only valid token positions."""

    layout: DecoderLayout
    source_batch_sequence_shape: tuple[int, int]
    source_valid_token_mask: Any
    valid_token_count: int
    blocks: tuple[DecoderBlockTrace, ...]
    final_norm_output: Any
    logits: Any


@dataclass(frozen=True, slots=True)
class HFTransportTrace:
    """Complete calibration trace with batch/sequence-shaped captured tensors."""

    layout: DecoderLayout
    valid_token_mask: Any
    blocks: tuple[DecoderBlockTrace, ...]
    final_norm_output: Any
    logits: Any
    hook_counts: Mapping[str, int]

    @property
    def family(self) -> str:
        return self.layout.family

    @property
    def batch_sequence_shape(self) -> tuple[int, int]:
        return tuple(int(size) for size in self.valid_token_mask.shape)

    @property
    def valid_token_count(self) -> int:
        return int(self.valid_token_mask.sum().item())

    def flatten_valid_rows(self, tensor: Any) -> Any:
        """Flatten ``tensor`` in batch-major/sequence-major valid-token order."""

        return _flatten_valid_tensor(tensor, self.valid_token_mask, "tensor")

    # A short spelling is convenient for transport fitting call sites.
    valid_rows_for = flatten_valid_rows

    def flattened_valid_rows(self) -> FlattenedTransportTrace:
        """Return every capture flattened to deterministic valid-token rows."""

        return FlattenedTransportTrace(
            layout=self.layout,
            source_batch_sequence_shape=self.batch_sequence_shape,
            source_valid_token_mask=self.valid_token_mask.clone(),
            valid_token_count=self.valid_token_count,
            blocks=tuple(
                block.flatten_valid(self.valid_token_mask) for block in self.blocks
            ),
            final_norm_output=self.flatten_valid_rows(self.final_norm_output),
            logits=self.flatten_valid_rows(self.logits),
        )

    @property
    def valid_rows(self) -> FlattenedTransportTrace:
        """Property form of :meth:`flattened_valid_rows`."""

        return self.flattened_valid_rows()


_BLOCK_TENSOR_FIELDS = (
    "block_input",
    "pre_attention_norm_output",
    "fused_qkv_projection_output",
    "query_projection_output",
    "key_projection_output",
    "value_projection_output",
    "attention_output_projection_input",
    "attention_output_projection_output",
    "attention_delta",
    "post_attention_residual",
    "pre_mlp_norm_output",
    "mlp_input_projection_output",
    "mlp_gate_projection_output",
    "mlp_up_projection_output",
    "activation_output",
    "down_projection_input",
    "down_projection_output",
    "mlp_delta",
    "block_output",
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without HF extras
        raise RuntimeError(
            "Hugging Face transport tracing requires the 'hf' optional dependencies."
        ) from exc
    return torch


def _flatten_valid_tensor(tensor: Any, mask: Any, name: str) -> Any:
    torch = _torch()
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("valid_token_mask must be a rank-2 boolean tensor.")
    if tensor.ndim < 2 or tuple(tensor.shape[:2]) != tuple(mask.shape):
        raise ValueError(
            f"{name} must begin with batch/sequence shape {tuple(mask.shape)}."
        )
    trailing = tuple(int(size) for size in tensor.shape[2:])
    flat = tensor.reshape(int(mask.numel()), *trailing)
    return flat[mask.reshape(-1)].contiguous().clone()


def _tensor_output(torch: Any, output: Any, role: str, *, first: bool = False) -> Any:
    if torch.is_tensor(output):
        return output
    if first and isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    expectation = (
        "a tensor or a sequence beginning with a tensor" if first else "a tensor"
    )
    raise TransportTraceError(f"Malformed output from {role}: expected {expectation}.")


def _module_input(
    torch: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    role: str,
) -> Any:
    if args and torch.is_tensor(args[0]):
        return args[0]
    for name in ("hidden_states", "x", "input"):
        value = kwargs.get(name)
        if torch.is_tensor(value):
            return value
    raise TransportTraceError(
        f"Malformed input to {role}: expected a leading hidden-state tensor."
    )


def _capture_tensor(torch: Any, tensor: Any, name: str) -> Any:
    if not torch.is_tensor(tensor):
        raise TransportTraceError(f"Malformed capture {name}: expected a tensor.")
    return tensor.detach().to(device="cpu").clone()


class _CaptureStore:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.expected: list[str] = []
        self.calls: dict[str, int] = {}
        self.tensors: dict[str, Any] = {}

    def expect(self, role: str) -> None:
        if role in self.calls:
            raise TransportTraceError(f"Duplicate trace operator role {role!r}.")
        self.expected.append(role)
        self.calls[role] = 0

    def record(self, role: str, values: Mapping[str, Any]) -> None:
        count = self.calls.get(role)
        if count is None:
            raise TransportTraceError(f"Unexpected trace operator role {role!r}.")
        count += 1
        self.calls[role] = count
        if count != 1:
            raise TransportTraceError(
                f"Trace hook for {role!r} fired {count} times; expected exactly once."
            )
        for name, tensor in values.items():
            if name in self.tensors:
                raise TransportTraceError(f"Trace capture {name!r} was written twice.")
            self.tensors[name] = _capture_tensor(self.torch, tensor, name)

    def verify_complete(self) -> None:
        wrong = {
            role: self.calls[role] for role in self.expected if self.calls[role] != 1
        }
        if wrong:
            details = ", ".join(f"{role}={count}" for role, count in wrong.items())
            raise TransportTraceError(
                "Every contracted trace hook must fire exactly once; observed "
                + details
                + "."
            )

    def regular_tensors(self) -> dict[str, Any]:
        # Cloning after inference mode converts inference tensors into ordinary,
        # detached tensors that downstream calibration math can consume.
        return {name: tensor.clone() for name, tensor in self.tensors.items()}


class _HookInstaller:
    def __init__(self, store: _CaptureStore) -> None:
        self.store = store
        self.handles: list[Any] = []
        self.roles_by_module_id: dict[int, str] = {}

    def add(self, module: Any, role: str, callback: Any) -> None:
        prior = self.roles_by_module_id.get(id(module))
        if prior is not None:
            raise TransportTraceError(
                f"Trace roles {prior!r} and {role!r} resolve to the same module; "
                "shared operator instances are unsupported."
            )
        self.roles_by_module_id[id(module)] = role
        self.store.expect(role)

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            output: Any,
        ) -> None:
            callback(args, kwargs, output)

        self.handles.append(
            module.register_forward_hook(hook, with_kwargs=True, always_call=False)
        )

    def remove_all(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def _simple_io_callback(
    torch: Any,
    store: _CaptureStore,
    role: str,
    input_name: str,
    output_name: str,
    *,
    first_output: bool = False,
) -> Any:
    def callback(args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        store.record(
            role,
            {
                input_name: _module_input(torch, args, kwargs, role),
                output_name: _tensor_output(torch, output, role, first=first_output),
            },
        )

    return callback


def _output_callback(
    torch: Any,
    store: _CaptureStore,
    role: str,
    output_name: str,
    *,
    first_output: bool = False,
) -> Any:
    def callback(
        _args: tuple[Any, ...], _kwargs: Mapping[str, Any], output: Any
    ) -> None:
        store.record(
            role,
            {output_name: _tensor_output(torch, output, role, first=first_output)},
        )

    return callback


def _install_block_hooks(
    torch: Any,
    model: Any,
    family: str,
    block: DecoderBlockLayout,
    store: _CaptureStore,
    installer: _HookInstaller,
) -> None:
    index = block.index
    prefix = f"block.{index}"

    installer.add(
        resolve_module(model, block.path, family=family),
        prefix,
        _simple_io_callback(
            torch,
            store,
            prefix,
            f"{prefix}.input",
            f"{prefix}.output",
            first_output=True,
        ),
    )
    attention_norm_role = f"{prefix}.attention_norm"
    installer.add(
        resolve_module(model, block.attention_input_norm.path, family=family),
        attention_norm_role,
        _simple_io_callback(
            torch,
            store,
            attention_norm_role,
            f"{prefix}.attention_norm_input",
            f"{prefix}.attention_norm_output",
        ),
    )
    attention_role = f"{prefix}.attention"
    installer.add(
        resolve_module(model, block.attention.module_path, family=family),
        attention_role,
        _simple_io_callback(
            torch,
            store,
            attention_role,
            f"{prefix}.attention_input",
            f"{prefix}.attention_delta",
            first_output=True,
        ),
    )

    if block.attention.fused_qkv_path is not None:
        fused_role = f"{prefix}.attention.qkv"

        def fused_callback(
            _args: tuple[Any, ...], _kwargs: Mapping[str, Any], output: Any
        ) -> None:
            fused = _tensor_output(torch, output, fused_role)
            widths = (
                block.attention.query_width,
                block.attention.key_value_width,
                block.attention.key_value_width,
            )
            if fused.ndim != 3 or int(fused.shape[-1]) != sum(widths):
                raise TransportTraceError(
                    f"Malformed output from {fused_role}: expected [batch, sequence, "
                    f"{sum(widths)}], got {tuple(fused.shape)}."
                )
            query, key, value = torch.split(fused, widths, dim=-1)
            store.record(
                fused_role,
                {
                    f"{prefix}.fused_qkv": fused,
                    f"{prefix}.query": query,
                    f"{prefix}.key": key,
                    f"{prefix}.value": value,
                },
            )

        installer.add(
            resolve_module(
                model,
                block.attention.fused_qkv_path,
                family=family,
            ),
            fused_role,
            fused_callback,
        )
    else:
        projection_paths = (
            ("q", "query", block.attention.query_path),
            ("k", "key", block.attention.key_path),
            ("v", "value", block.attention.value_path),
        )
        for suffix, capture_name, path in projection_paths:
            if path is None:  # The layout contract makes this unreachable.
                raise TransportTraceError(
                    f"Split-QKV block {index} is missing its {suffix} projection path."
                )
            role = f"{prefix}.attention.{suffix}"
            installer.add(
                resolve_module(model, path, family=family),
                role,
                _output_callback(torch, store, role, f"{prefix}.{capture_name}"),
            )

    output_role = f"{prefix}.attention.output"
    installer.add(
        resolve_module(model, block.attention.output_path, family=family),
        output_role,
        _simple_io_callback(
            torch,
            store,
            output_role,
            f"{prefix}.attention_output_input",
            f"{prefix}.attention_output",
        ),
    )

    mlp_norm_role = f"{prefix}.mlp_norm"
    installer.add(
        resolve_module(model, block.mlp_input_norm.path, family=family),
        mlp_norm_role,
        _simple_io_callback(
            torch,
            store,
            mlp_norm_role,
            f"{prefix}.post_attention_residual",
            f"{prefix}.mlp_norm_output",
        ),
    )
    mlp_role = f"{prefix}.mlp"
    installer.add(
        resolve_module(model, block.mlp.module_path, family=family),
        mlp_role,
        _simple_io_callback(
            torch,
            store,
            mlp_role,
            f"{prefix}.mlp_input",
            f"{prefix}.mlp_delta",
        ),
    )

    if block.mlp.input_projection_path is not None:
        role = f"{prefix}.mlp.input"
        installer.add(
            resolve_module(model, block.mlp.input_projection_path, family=family),
            role,
            _output_callback(torch, store, role, f"{prefix}.mlp_input_projection"),
        )
    if block.mlp.gate_projection_path is not None:
        role = f"{prefix}.mlp.gate"
        installer.add(
            resolve_module(model, block.mlp.gate_projection_path, family=family),
            role,
            _output_callback(torch, store, role, f"{prefix}.mlp_gate_projection"),
        )
    if block.mlp.up_projection_path is not None:
        role = f"{prefix}.mlp.up"
        installer.add(
            resolve_module(model, block.mlp.up_projection_path, family=family),
            role,
            _output_callback(torch, store, role, f"{prefix}.mlp_up_projection"),
        )

    activation_role = f"{prefix}.mlp.activation"
    installer.add(
        resolve_module(model, block.mlp.activation_path, family=family),
        activation_role,
        _output_callback(torch, store, activation_role, f"{prefix}.activation"),
    )
    down_role = f"{prefix}.mlp.down"
    installer.add(
        resolve_module(model, block.mlp.down_projection_path, family=family),
        down_role,
        _simple_io_callback(
            torch,
            store,
            down_role,
            f"{prefix}.down_input",
            f"{prefix}.down_output",
        ),
    )


def _install_hooks(
    torch: Any,
    model: Any,
    layout: DecoderLayout,
    store: _CaptureStore,
    installer: _HookInstaller,
) -> None:
    for block in layout.blocks:
        _install_block_hooks(torch, model, layout.family, block, store, installer)

    installer.add(
        resolve_module(model, layout.final_norm.path, family=layout.family),
        "final_norm",
        _simple_io_callback(
            torch,
            store,
            "final_norm",
            "final_norm.input",
            "final_norm.output",
        ),
    )
    installer.add(
        resolve_module(model, layout.lm_head_path, family=layout.family),
        "lm_head",
        _simple_io_callback(torch, store, "lm_head", "lm_head.input", "lm_head.output"),
    )


def _normalize_inputs(
    torch: Any,
    model_inputs: Mapping[str, Any] | Any | None,
    attention_mask: Any | None,
    valid_token_mask: Any | None,
    extra_inputs: Mapping[str, Any],
    hidden_size: int,
) -> tuple[dict[str, Any], Any, tuple[int, int]]:
    if model_inputs is None:
        values: dict[str, Any] = {}
    elif isinstance(model_inputs, Mapping):
        values = dict(model_inputs)
    else:
        values = {"input_ids": model_inputs}

    duplicates = set(values).intersection(extra_inputs)
    if duplicates:
        raise ValueError(f"Duplicate model inputs: {sorted(duplicates)}.")
    values.update(extra_inputs)
    if attention_mask is not None:
        if "attention_mask" in values:
            raise ValueError("attention_mask was provided twice.")
        values["attention_mask"] = attention_mask

    _reject_non_calibration_inputs(values)
    has_ids = "input_ids" in values and values["input_ids"] is not None
    has_embeds = "inputs_embeds" in values and values["inputs_embeds"] is not None
    if has_ids == has_embeds:
        raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

    if has_ids:
        anchor = torch.as_tensor(values["input_ids"])
        if anchor.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if (
            anchor.dtype == torch.bool
            or anchor.is_floating_point()
            or anchor.is_complex()
        ):
            raise ValueError("input_ids must contain integer token indices.")
        values["input_ids"] = anchor
        batch_sequence_shape = tuple(int(size) for size in anchor.shape)
    else:
        anchor = torch.as_tensor(values["inputs_embeds"])
        if anchor.ndim != 3 or int(anchor.shape[-1]) != hidden_size:
            raise ValueError(
                "inputs_embeds must have shape [batch, sequence, hidden_size]."
            )
        values["inputs_embeds"] = anchor
        batch_sequence_shape = tuple(int(size) for size in anchor.shape[:2])

    if min(batch_sequence_shape) <= 0:
        raise ValueError("Calibration batches must contain at least one token.")

    raw_attention_mask = values.get("attention_mask")
    if raw_attention_mask is None:
        model_mask = torch.ones(
            batch_sequence_shape, dtype=torch.long, device=anchor.device
        )
    else:
        model_mask = torch.as_tensor(raw_attention_mask, device=anchor.device)
        _validate_binary_mask(model_mask, batch_sequence_shape, "attention_mask")
    values["attention_mask"] = model_mask
    attention_bool = model_mask.to(dtype=torch.bool)

    if valid_token_mask is None:
        valid = attention_bool
    else:
        valid = torch.as_tensor(valid_token_mask, device=anchor.device)
        _validate_binary_mask(valid, batch_sequence_shape, "valid_token_mask")
        valid = valid.to(dtype=torch.bool)
        if bool((valid & ~attention_bool).any().item()):
            raise ValueError("valid_token_mask cannot select attention-masked tokens.")
    if not bool(valid.any().item()):
        raise ValueError("valid_token_mask must select at least one token.")

    values["use_cache"] = False
    values["return_dict"] = True
    values["output_attentions"] = False
    values["output_hidden_states"] = False
    return (
        values,
        valid.detach().cpu().to(dtype=torch.bool).clone(),
        batch_sequence_shape,
    )


def _reject_non_calibration_inputs(values: dict[str, Any]) -> None:
    constrained = {
        "use_cache": False,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
    }
    for name, required in constrained.items():
        if name not in values:
            continue
        supplied = values.pop(name)
        if supplied is not None and supplied is not required:
            raise ValueError(f"{name} must be {required!r} during trace collection.")

    for name in ("past_key_values", "past_key_value", "cache_position"):
        if name in values:
            supplied = values.pop(name)
            if supplied is not None:
                raise ValueError(f"Cached decoding input {name!r} is unsupported.")
    for name in ("labels", "encoder_hidden_states", "encoder_attention_mask"):
        if name in values and values[name] is not None:
            raise ValueError(f"Training/cross-attention input {name!r} is unsupported.")
        values.pop(name, None)


def _validate_binary_mask(
    mask: Any, expected_shape: tuple[int, int], name: str
) -> None:
    if mask.ndim != 2 or tuple(int(size) for size in mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")
    if mask.is_complex() or not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError(f"{name} must contain only boolean or binary values.")


def _checkpointing_operator(model: Any) -> str | None:
    if bool(getattr(model, "is_gradient_checkpointing", False)):
        return type(model).__name__
    config = getattr(model, "config", None)
    if bool(getattr(config, "gradient_checkpointing", False)):
        return "config"
    for name, module in model.named_modules():
        if bool(getattr(module, "gradient_checkpointing", False)):
            return name or type(module).__name__
    return None


def _model_logits(torch: Any, output: Any) -> Any:
    cache = getattr(output, "past_key_values", None)
    if isinstance(output, Mapping):
        cache = output.get("past_key_values", cache)
    if cache is not None:
        raise TransportTraceError("Model returned a cache despite use_cache=False.")

    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if not torch.is_tensor(logits):
        raise TransportTraceError(
            "Malformed causal-decoder output: expected a tensor-valued logits field."
        )
    return logits


def _expect_shape(tensor: Any, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(int(size) for size in tensor.shape)
    if actual != expected:
        raise TransportTraceError(
            f"Malformed trace tensor {name!r}: expected shape {expected}, got {actual}."
        )
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise TransportTraceError(f"Trace tensor {name!r} retained a gradient graph.")
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
        tensor.isfinite().all().item()
    ):
        raise TransportTraceError(f"Trace tensor {name!r} contains non-finite values.")


def _assert_close(torch: Any, actual: Any, expected: Any, name: str) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise TransportTraceError(
            f"Trace identity {name!r} compares incompatible shapes "
            f"{tuple(actual.shape)} and {tuple(expected.shape)}."
        )
    dtypes = {actual.dtype, expected.dtype}
    if torch.float16 in dtypes or torch.bfloat16 in dtypes:
        tolerance = 5e-3
    elif dtypes == {torch.float64}:
        tolerance = 1e-10
    else:
        tolerance = 2e-5
    if not bool(torch.allclose(actual, expected, rtol=tolerance, atol=tolerance)):
        difference = float(
            (actual.to(torch.float64) - expected.to(torch.float64)).abs().max().item()
        )
        raise TransportTraceError(
            f"Trace identity {name!r} failed (max absolute error {difference:.6g})."
        )


def _build_trace(
    torch: Any,
    layout: DecoderLayout,
    valid_token_mask: Any,
    batch_sequence_shape: tuple[int, int],
    captures: Mapping[str, Any],
    returned_logits: Any,
    hook_counts: Mapping[str, int],
) -> HFTransportTrace:
    batch_size, sequence_length = batch_sequence_shape
    residual_shape = (batch_size, sequence_length, layout.hidden_size)
    intermediate_shape = (batch_size, sequence_length, layout.intermediate_size)
    blocks: list[DecoderBlockTrace] = []

    for block_layout in layout.blocks:
        index = block_layout.index
        prefix = f"block.{index}"
        attention = block_layout.attention
        query_shape = (batch_size, sequence_length, attention.query_width)
        kv_shape = (batch_size, sequence_length, attention.key_value_width)
        attention_input_shape = (
            batch_size,
            sequence_length,
            attention.query_width,
        )

        names_and_shapes = {
            f"{prefix}.input": residual_shape,
            f"{prefix}.output": residual_shape,
            f"{prefix}.attention_norm_input": residual_shape,
            f"{prefix}.attention_norm_output": residual_shape,
            f"{prefix}.attention_input": residual_shape,
            f"{prefix}.query": query_shape,
            f"{prefix}.key": kv_shape,
            f"{prefix}.value": kv_shape,
            f"{prefix}.attention_output_input": attention_input_shape,
            f"{prefix}.attention_output": residual_shape,
            f"{prefix}.attention_delta": residual_shape,
            f"{prefix}.post_attention_residual": residual_shape,
            f"{prefix}.mlp_norm_output": residual_shape,
            f"{prefix}.mlp_input": residual_shape,
            f"{prefix}.activation": intermediate_shape,
            f"{prefix}.down_input": intermediate_shape,
            f"{prefix}.down_output": residual_shape,
            f"{prefix}.mlp_delta": residual_shape,
        }
        if attention.fused_qkv_path is not None:
            names_and_shapes[f"{prefix}.fused_qkv"] = (
                batch_size,
                sequence_length,
                attention.query_width + 2 * attention.key_value_width,
            )
        if block_layout.mlp.input_projection_path is not None:
            names_and_shapes[f"{prefix}.mlp_input_projection"] = intermediate_shape
        if block_layout.mlp.gate_projection_path is not None:
            names_and_shapes[f"{prefix}.mlp_gate_projection"] = intermediate_shape
        if block_layout.mlp.up_projection_path is not None:
            names_and_shapes[f"{prefix}.mlp_up_projection"] = intermediate_shape
        for name, expected_shape in names_and_shapes.items():
            if name not in captures:
                raise TransportTraceError(
                    f"Required trace capture {name!r} is missing."
                )
            _expect_shape(captures[name], expected_shape, name)

        block_input = captures[f"{prefix}.input"]
        attention_norm = captures[f"{prefix}.attention_norm_output"]
        attention_delta = captures[f"{prefix}.attention_delta"]
        attention_output = captures[f"{prefix}.attention_output"]
        post_attention = captures[f"{prefix}.post_attention_residual"]
        mlp_norm = captures[f"{prefix}.mlp_norm_output"]
        mlp_delta = captures[f"{prefix}.mlp_delta"]
        down_output = captures[f"{prefix}.down_output"]
        block_output = captures[f"{prefix}.output"]
        activation = captures[f"{prefix}.activation"]
        down_input = captures[f"{prefix}.down_input"]

        _assert_close(
            torch,
            captures[f"{prefix}.attention_norm_input"],
            block_input,
            f"block {index} pre-attention norm input",
        )
        _assert_close(
            torch,
            captures[f"{prefix}.attention_input"],
            attention_norm,
            f"block {index} attention input",
        )
        _assert_close(
            torch,
            attention_delta,
            attention_output,
            f"block {index} attention projection delta",
        )
        _assert_close(
            torch,
            post_attention,
            block_input + attention_delta,
            f"block {index} post-attention residual",
        )
        _assert_close(
            torch,
            captures[f"{prefix}.mlp_input"],
            mlp_norm,
            f"block {index} MLP input",
        )
        _assert_close(
            torch,
            mlp_delta,
            down_output,
            f"block {index} MLP projection delta",
        )
        _assert_close(
            torch,
            block_output,
            post_attention + mlp_delta,
            f"block {index} output residual",
        )
        _assert_close(
            torch,
            block_output,
            block_input + attention_delta + mlp_delta,
            f"block {index} complete residual",
        )

        if block_layout.mlp.kind == "dense":
            _assert_close(
                torch,
                down_input,
                activation,
                f"block {index} activated dense intermediate",
            )
        else:
            up_output = captures[f"{prefix}.mlp_up_projection"]
            _assert_close(
                torch,
                down_input,
                activation * up_output,
                f"block {index} gated intermediate",
            )

        fused = captures.get(f"{prefix}.fused_qkv")
        if fused is not None:
            _assert_close(
                torch,
                fused,
                torch.cat(
                    (
                        captures[f"{prefix}.query"],
                        captures[f"{prefix}.key"],
                        captures[f"{prefix}.value"],
                    ),
                    dim=-1,
                ),
                f"block {index} fused QKV split",
            )

        blocks.append(
            DecoderBlockTrace(
                index=index,
                block_input=block_input,
                pre_attention_norm_output=attention_norm,
                fused_qkv_projection_output=fused,
                query_projection_output=captures[f"{prefix}.query"],
                key_projection_output=captures[f"{prefix}.key"],
                value_projection_output=captures[f"{prefix}.value"],
                attention_output_projection_input=captures[
                    f"{prefix}.attention_output_input"
                ],
                attention_output_projection_output=attention_output,
                attention_delta=attention_delta,
                post_attention_residual=post_attention,
                pre_mlp_norm_output=mlp_norm,
                mlp_input_projection_output=captures.get(
                    f"{prefix}.mlp_input_projection"
                ),
                mlp_gate_projection_output=captures.get(
                    f"{prefix}.mlp_gate_projection"
                ),
                mlp_up_projection_output=captures.get(f"{prefix}.mlp_up_projection"),
                activation_output=activation,
                down_projection_input=down_input,
                down_projection_output=down_output,
                mlp_delta=mlp_delta,
                block_output=block_output,
            )
        )

    for left, right in zip(blocks, blocks[1:]):
        _assert_close(
            torch,
            right.block_input,
            left.block_output,
            f"block boundary {left.index}->{right.index}",
        )

    final_norm_input = captures["final_norm.input"]
    final_norm_output = captures["final_norm.output"]
    lm_head_input = captures["lm_head.input"]
    lm_head_output = captures["lm_head.output"]
    _expect_shape(final_norm_input, residual_shape, "final_norm.input")
    _expect_shape(final_norm_output, residual_shape, "final_norm.output")
    logits_shape = (
        batch_size,
        sequence_length,
        layout.embeddings.vocab_size,
    )
    _expect_shape(lm_head_input, residual_shape, "lm_head.input")
    _expect_shape(lm_head_output, logits_shape, "lm_head.output")
    _expect_shape(returned_logits, logits_shape, "model.logits")
    _assert_close(
        torch, final_norm_input, blocks[-1].block_output, "final norm residual input"
    )
    _assert_close(torch, lm_head_input, final_norm_output, "language-head input")
    _assert_close(torch, returned_logits, lm_head_output, "returned logits")

    return HFTransportTrace(
        layout=layout,
        valid_token_mask=valid_token_mask,
        blocks=tuple(blocks),
        final_norm_output=final_norm_output,
        logits=returned_logits,
        hook_counts=MappingProxyType(dict(hook_counts)),
    )


def collect_hf_transport_trace(
    model: Any,
    model_inputs: Mapping[str, Any] | Any | None = None,
    attention_mask: Any | None = None,
    *,
    valid_token_mask: Any | None = None,
    **forward_inputs: Any,
) -> HFTransportTrace:
    """Collect one deterministic, no-gradient GPT-2 or Llama calibration trace.

    ``model_inputs`` may be a Hugging Face input mapping or an ``input_ids``
    tensor.  ``valid_token_mask`` can select a binary subset of the ordinary
    attention mask.  All returned tensors retain ``[batch, sequence, ...]``
    shape; :meth:`HFTransportTrace.flattened_valid_rows` provides the stable
    batch-major/sequence-major row view used by calibration solvers.
    """

    torch = _torch()
    layout = build_decoder_layout(model)
    checkpointing = _checkpointing_operator(model)
    if checkpointing is not None:
        raise TransportTraceError(
            "Gradient checkpointing must be disabled during trace collection; "
            f"it is enabled at {checkpointing!r}."
        )
    inputs, valid_mask, batch_sequence_shape = _normalize_inputs(
        torch,
        model_inputs,
        attention_mask,
        valid_token_mask,
        forward_inputs,
        layout.hidden_size,
    )

    module_modes = [(module, bool(module.training)) for module in model.modules()]
    store = _CaptureStore(torch)
    installer = _HookInstaller(store)
    try:
        _install_hooks(torch, model, layout, store, installer)
        model.eval()
        with torch.inference_mode():
            output = model(**inputs)
            returned_logits = _capture_tensor(
                torch, _model_logits(torch, output), "model.logits"
            )
        store.verify_complete()
        captures = store.regular_tensors()
        returned_logits = returned_logits.clone()
        return _build_trace(
            torch,
            layout,
            valid_mask,
            batch_sequence_shape,
            captures,
            returned_logits,
            store.calls,
        )
    finally:
        installer.remove_all()
        # Assign directly so heterogeneous child-module modes are restored
        # exactly instead of recursively overwritten by Module.train().
        for module, was_training in module_modes:
            module.training = was_training


# Stable short aliases for callers that are already scoped to Hugging Face.
collect_transport_trace = collect_hf_transport_trace
collect_calibration_trace = collect_hf_transport_trace
BlockTransportTrace = DecoderBlockTrace
CalibrationTrace = HFTransportTrace


__all__ = [
    "BlockTransportTrace",
    "CalibrationTrace",
    "DecoderBlockTrace",
    "FlattenedTransportTrace",
    "HFTransportTrace",
    "TransportTraceError",
    "collect_calibration_trace",
    "collect_hf_transport_trace",
    "collect_transport_trace",
]
