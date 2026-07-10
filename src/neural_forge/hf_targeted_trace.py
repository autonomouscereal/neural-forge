"""One-block deterministic activation traces for supported HF decoders.

Unlike :mod:`neural_forge.hf_transport_trace`, this collector runs the whole
decoder but hooks only one selected block.  Final norm and language-head
interfaces are independently optional.  Every hook capture is a detached CPU
copy and the reported byte count covers exactly those copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.hf_arch_layout import (
    DecoderBlockLayout,
    DecoderLayout,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_transport_trace import DecoderBlockTrace, TransportTraceError


TargetedTraceMode = Literal["sequence", "flatten_valid", "mean", "last_valid"]


@dataclass(frozen=True, slots=True)
class CapturedTensorInfo:
    """Immutable metadata for one tensor copied by the targeted collector."""

    shape: tuple[int, ...]
    dtype: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class FinalNormHeadTrace:
    """Optional final decoder interface, reduced using the requested mode."""

    final_norm_input: Any
    final_norm_output: Any
    lm_head_input: Any | None
    logits: Any | None


@dataclass(frozen=True, slots=True)
class HFTargetedTrace:
    """A validated trace for exactly one decoder block."""

    layout: DecoderLayout
    block_index: int
    block: DecoderBlockTrace
    mode: TargetedTraceMode
    source_batch_sequence_shape: tuple[int, int]
    valid_token_mask: Any
    valid_token_counts: Any
    final_interface: FinalNormHeadTrace | None
    hook_counts: Mapping[str, int]
    capture_inventory: Mapping[str, CapturedTensorInfo]
    captured_byte_count: int

    @property
    def family(self) -> str:
        return self.layout.family

    @property
    def blocks(self) -> tuple[DecoderBlockTrace, ...]:
        """Tuple-compatible view for consumers expecting transport blocks."""

        return (self.block,)

    @property
    def final_norm_output(self) -> Any | None:
        if self.final_interface is None:
            return None
        return self.final_interface.final_norm_output

    @property
    def logits(self) -> Any | None:
        if self.final_interface is None:
            return None
        return self.final_interface.logits

    @property
    def captured_tensor_count(self) -> int:
        return len(self.capture_inventory)

    @property
    def valid_token_count(self) -> int:
        return int(self.valid_token_mask.sum().item())


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Hugging Face targeted tracing requires the 'hf' optional dependencies."
        ) from exc
    return torch


def _tensor_output(torch: Any, output: Any, role: str, *, first: bool = False) -> Any:
    if torch.is_tensor(output):
        return output
    if first and isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    expected = "a tensor or a sequence beginning with a tensor" if first else "a tensor"
    raise TransportTraceError(f"Malformed output from {role}: expected {expected}.")


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


class _CaptureStore:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.expected: list[str] = []
        self.calls: dict[str, int] = {}
        self.tensors: dict[str, Any] = {}

    def expect(self, role: str) -> None:
        if role in self.calls:
            raise TransportTraceError(f"Duplicate targeted trace role {role!r}.")
        self.expected.append(role)
        self.calls[role] = 0

    def _copy(self, name: str, tensor: Any) -> None:
        if name in self.tensors:
            raise TransportTraceError(
                f"Targeted trace capture {name!r} was written twice."
            )
        if not self.torch.is_tensor(tensor):
            raise TransportTraceError(f"Malformed capture {name}: expected a tensor.")
        # Disable inference tensor creation for the retained copy.  This avoids
        # a second clone after the forward pass while still preventing autograd.
        with self.torch.inference_mode(False):
            copied = tensor.detach().to(device="cpu").clone()
        self.tensors[name] = copied

    def record(self, role: str, values: Mapping[str, Any]) -> None:
        count = self.calls.get(role)
        if count is None:
            raise TransportTraceError(f"Unexpected targeted trace role {role!r}.")
        count += 1
        self.calls[role] = count
        if count != 1:
            raise TransportTraceError(
                f"Targeted trace hook for {role!r} fired {count} times; expected once."
            )
        for name, tensor in values.items():
            self._copy(name, tensor)

    def copy_model_output(self, name: str, tensor: Any) -> None:
        self._copy(name, tensor)

    def verify_complete(self) -> None:
        wrong = {
            role: self.calls[role] for role in self.expected if self.calls[role] != 1
        }
        if wrong:
            details = ", ".join(f"{role}={count}" for role, count in wrong.items())
            raise TransportTraceError(
                "Every targeted trace hook must fire exactly once; observed "
                + details
                + "."
            )

    def inventory(self) -> Mapping[str, CapturedTensorInfo]:
        items = {
            name: CapturedTensorInfo(
                shape=tuple(int(size) for size in tensor.shape),
                dtype=str(tensor.dtype),
                byte_count=int(tensor.numel() * tensor.element_size()),
            )
            for name, tensor in self.tensors.items()
        }
        return MappingProxyType(items)


class _HookInstaller:
    def __init__(self, store: _CaptureStore) -> None:
        self.store = store
        self.handles: list[Any] = []
        self.roles_by_module_id: dict[int, str] = {}

    def add(self, module: Any, role: str, callback: Any) -> None:
        prior = self.roles_by_module_id.get(id(module))
        if prior is not None:
            raise TransportTraceError(
                f"Targeted trace roles {prior!r} and {role!r} resolve to the same "
                "module; shared operator instances are unsupported."
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
    torch: Any, store: _CaptureStore, role: str, output_name: str
) -> Any:
    def callback(
        _args: tuple[Any, ...], _kwargs: Mapping[str, Any], output: Any
    ) -> None:
        store.record(role, {output_name: _tensor_output(torch, output, role)})

    return callback


def _projection_callback(
    torch: Any,
    store: _CaptureStore,
    role: str,
    output_name: str,
    expected_width: int,
) -> Any:
    def callback(
        _args: tuple[Any, ...], _kwargs: Mapping[str, Any], output: Any
    ) -> None:
        tensor = _tensor_output(torch, output, role)
        if tensor.ndim != 3 or int(tensor.shape[-1]) != expected_width:
            raise TransportTraceError(
                f"Malformed output from {role}: expected [batch, sequence, "
                f"{expected_width}], got {tuple(tensor.shape)}."
            )
        store.record(role, {output_name: tensor})

    return callback


def _install_block_hooks(
    torch: Any,
    model: Any,
    layout: DecoderLayout,
    block: DecoderBlockLayout,
    store: _CaptureStore,
    installer: _HookInstaller,
) -> None:
    prefix = f"block.{block.index}"
    family = layout.family
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
    for role_suffix, path, input_name, output_name, first_output in (
        (
            "attention_norm",
            block.attention_input_norm.path,
            "attention_norm_input",
            "attention_norm_output",
            False,
        ),
        (
            "attention",
            block.attention.module_path,
            "attention_input",
            "attention_delta",
            True,
        ),
    ):
        role = f"{prefix}.{role_suffix}"
        installer.add(
            resolve_module(model, path, family=family),
            role,
            _simple_io_callback(
                torch,
                store,
                role,
                f"{prefix}.{input_name}",
                f"{prefix}.{output_name}",
                first_output=first_output,
            ),
        )

    attention = block.attention
    if attention.fused_qkv_path is not None:
        fused_role = f"{prefix}.attention.qkv"

        def fused_callback(
            _args: tuple[Any, ...], _kwargs: Mapping[str, Any], output: Any
        ) -> None:
            fused = _tensor_output(torch, output, fused_role)
            widths = (
                attention.query_width,
                attention.key_value_width,
                attention.key_value_width,
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
            resolve_module(model, attention.fused_qkv_path, family=family),
            fused_role,
            fused_callback,
        )
    else:
        projections = (
            ("q", "query", attention.query_path, attention.query_width),
            ("k", "key", attention.key_path, attention.key_value_width),
            ("v", "value", attention.value_path, attention.key_value_width),
        )
        for suffix, name, path, width in projections:
            if path is None:
                raise TransportTraceError(
                    f"Split-QKV block {block.index} is missing its {suffix} path."
                )
            role = f"{prefix}.attention.{suffix}"
            installer.add(
                resolve_module(model, path, family=family),
                role,
                _projection_callback(torch, store, role, f"{prefix}.{name}", width),
            )

    role = f"{prefix}.attention.output"
    installer.add(
        resolve_module(model, attention.output_path, family=family),
        role,
        _simple_io_callback(
            torch,
            store,
            role,
            f"{prefix}.attention_output_input",
            f"{prefix}.attention_output",
        ),
    )

    role = f"{prefix}.mlp_norm"
    installer.add(
        resolve_module(model, block.mlp_input_norm.path, family=family),
        role,
        _simple_io_callback(
            torch,
            store,
            role,
            f"{prefix}.post_attention_residual",
            f"{prefix}.mlp_norm_output",
        ),
    )
    role = f"{prefix}.mlp"
    installer.add(
        resolve_module(model, block.mlp.module_path, family=family),
        role,
        _simple_io_callback(
            torch,
            store,
            role,
            f"{prefix}.mlp_input",
            f"{prefix}.mlp_delta",
        ),
    )

    mlp = block.mlp
    for suffix, path, name in (
        ("input", mlp.input_projection_path, "mlp_input_projection"),
        ("gate", mlp.gate_projection_path, "mlp_gate_projection"),
        ("up", mlp.up_projection_path, "mlp_up_projection"),
    ):
        if path is None:
            continue
        role = f"{prefix}.mlp.{suffix}"
        installer.add(
            resolve_module(model, path, family=family),
            role,
            _output_callback(torch, store, role, f"{prefix}.{name}"),
        )
    role = f"{prefix}.mlp.activation"
    installer.add(
        resolve_module(model, mlp.activation_path, family=family),
        role,
        _output_callback(torch, store, role, f"{prefix}.activation"),
    )
    role = f"{prefix}.mlp.down"
    installer.add(
        resolve_module(model, mlp.down_projection_path, family=family),
        role,
        _simple_io_callback(
            torch,
            store,
            role,
            f"{prefix}.down_input",
            f"{prefix}.down_output",
        ),
    )


def _install_final_hooks(
    torch: Any,
    model: Any,
    layout: DecoderLayout,
    store: _CaptureStore,
    installer: _HookInstaller,
    *,
    capture_logits: bool,
) -> None:
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
    if capture_logits:
        installer.add(
            resolve_module(model, layout.lm_head_path, family=layout.family),
            "lm_head",
            _simple_io_callback(
                torch, store, "lm_head", "lm_head.input", "lm_head.output"
            ),
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


def _validate_binary_mask(mask: Any, shape: tuple[int, int], name: str) -> None:
    if mask.ndim != 2 or tuple(int(size) for size in mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if mask.is_complex() or not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError(f"{name} must contain only boolean or binary values.")


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
    has_ids = values.get("input_ids") is not None
    has_embeds = values.get("inputs_embeds") is not None
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
        shape = tuple(int(size) for size in anchor.shape)
    else:
        anchor = torch.as_tensor(values["inputs_embeds"])
        if anchor.ndim != 3 or int(anchor.shape[-1]) != hidden_size:
            raise ValueError(
                "inputs_embeds must have shape [batch, sequence, hidden_size]."
            )
        values["inputs_embeds"] = anchor
        shape = tuple(int(size) for size in anchor.shape[:2])
    if min(shape) <= 0:
        raise ValueError("Calibration batches must contain at least one token.")

    raw_mask = values.get("attention_mask")
    if raw_mask is None:
        model_mask = torch.ones(shape, dtype=torch.long, device=anchor.device)
    else:
        model_mask = torch.as_tensor(raw_mask, device=anchor.device)
        _validate_binary_mask(model_mask, shape, "attention_mask")
    values["attention_mask"] = model_mask
    attention_bool = model_mask.to(dtype=torch.bool)
    if valid_token_mask is None:
        valid = attention_bool
    else:
        valid = torch.as_tensor(valid_token_mask, device=anchor.device)
        _validate_binary_mask(valid, shape, "valid_token_mask")
        valid = valid.to(dtype=torch.bool)
        if bool((valid & ~attention_bool).any().item()):
            raise ValueError("valid_token_mask cannot select attention-masked tokens.")
    if not bool(valid.any().item()):
        raise ValueError("valid_token_mask must select at least one token.")
    values.update(
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    return values, valid.detach().cpu().to(dtype=torch.bool).clone(), shape


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
            "Malformed causal-decoder output: expected tensor-valued logits."
        )
    return logits


def _expect_shape(
    torch: Any, tensor: Any, expected: tuple[int, ...], name: str
) -> None:
    if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
        raise TransportTraceError(
            f"Malformed trace tensor {name!r}: expected dense tensor."
        )
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
    tolerance = 5e-3 if torch.float16 in dtypes or torch.bfloat16 in dtypes else 2e-5
    if dtypes == {torch.float64}:
        tolerance = 1e-10
    if not bool(torch.allclose(actual, expected, rtol=tolerance, atol=tolerance)):
        difference = float(
            (actual.to(torch.float64) - expected.to(torch.float64)).abs().max().item()
        )
        raise TransportTraceError(
            f"Trace identity {name!r} failed (max absolute error {difference:.6g})."
        )


def _build_block(
    torch: Any,
    layout: DecoderLayout,
    block_layout: DecoderBlockLayout,
    shape: tuple[int, int],
    captures: Mapping[str, Any],
) -> DecoderBlockTrace:
    batch, sequence = shape
    hidden_shape = (batch, sequence, layout.hidden_size)
    intermediate_shape = (batch, sequence, layout.intermediate_size)
    attention = block_layout.attention
    prefix = f"block.{block_layout.index}"
    expected = {
        f"{prefix}.input": hidden_shape,
        f"{prefix}.output": hidden_shape,
        f"{prefix}.attention_norm_input": hidden_shape,
        f"{prefix}.attention_norm_output": hidden_shape,
        f"{prefix}.attention_input": hidden_shape,
        f"{prefix}.query": (batch, sequence, attention.query_width),
        f"{prefix}.key": (batch, sequence, attention.key_value_width),
        f"{prefix}.value": (batch, sequence, attention.key_value_width),
        f"{prefix}.attention_output_input": (
            batch,
            sequence,
            attention.query_width,
        ),
        f"{prefix}.attention_output": hidden_shape,
        f"{prefix}.attention_delta": hidden_shape,
        f"{prefix}.post_attention_residual": hidden_shape,
        f"{prefix}.mlp_norm_output": hidden_shape,
        f"{prefix}.mlp_input": hidden_shape,
        f"{prefix}.activation": intermediate_shape,
        f"{prefix}.down_input": intermediate_shape,
        f"{prefix}.down_output": hidden_shape,
        f"{prefix}.mlp_delta": hidden_shape,
    }
    if attention.fused_qkv_path is not None:
        expected[f"{prefix}.fused_qkv"] = (
            batch,
            sequence,
            attention.query_width + 2 * attention.key_value_width,
        )
    for path in (
        block_layout.mlp.input_projection_path,
        block_layout.mlp.gate_projection_path,
        block_layout.mlp.up_projection_path,
    ):
        if path is not None:
            suffix = (
                "mlp_input_projection"
                if path == block_layout.mlp.input_projection_path
                else "mlp_gate_projection"
                if path == block_layout.mlp.gate_projection_path
                else "mlp_up_projection"
            )
            expected[f"{prefix}.{suffix}"] = intermediate_shape
    for name, tensor_shape in expected.items():
        if name not in captures:
            raise TransportTraceError(f"Required targeted capture {name!r} is missing.")
        _expect_shape(torch, captures[name], tensor_shape, name)

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
    identities = (
        (captures[f"{prefix}.attention_norm_input"], block_input, "norm input"),
        (captures[f"{prefix}.attention_input"], attention_norm, "attention input"),
        (attention_delta, attention_output, "attention projection delta"),
        (post_attention, block_input + attention_delta, "post-attention residual"),
        (captures[f"{prefix}.mlp_input"], mlp_norm, "MLP input"),
        (mlp_delta, down_output, "MLP projection delta"),
        (block_output, post_attention + mlp_delta, "output residual"),
        (
            block_output,
            block_input + attention_delta + mlp_delta,
            "complete residual",
        ),
    )
    for actual, wanted, label in identities:
        _assert_close(torch, actual, wanted, f"block {block_layout.index} {label}")

    if block_layout.mlp.kind == "dense":
        _assert_close(
            torch,
            down_input,
            activation,
            f"block {block_layout.index} activated dense intermediate",
        )
    else:
        up = captures[f"{prefix}.mlp_up_projection"]
        _assert_close(
            torch,
            down_input,
            activation * up,
            f"block {block_layout.index} gated intermediate",
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
            f"block {block_layout.index} fused QKV split",
        )

    return DecoderBlockTrace(
        index=block_layout.index,
        block_input=block_input,
        pre_attention_norm_output=attention_norm,
        fused_qkv_projection_output=fused,
        query_projection_output=captures[f"{prefix}.query"],
        key_projection_output=captures[f"{prefix}.key"],
        value_projection_output=captures[f"{prefix}.value"],
        attention_output_projection_input=captures[f"{prefix}.attention_output_input"],
        attention_output_projection_output=attention_output,
        attention_delta=attention_delta,
        post_attention_residual=post_attention,
        pre_mlp_norm_output=mlp_norm,
        mlp_input_projection_output=captures.get(f"{prefix}.mlp_input_projection"),
        mlp_gate_projection_output=captures.get(f"{prefix}.mlp_gate_projection"),
        mlp_up_projection_output=captures.get(f"{prefix}.mlp_up_projection"),
        activation_output=activation,
        down_projection_input=down_input,
        down_projection_output=down_output,
        mlp_delta=mlp_delta,
        block_output=block_output,
    )


def _reduce_tensor(torch: Any, value: Any, mask: Any, mode: TargetedTraceMode) -> Any:
    if mode == "sequence":
        return value
    if mode == "flatten_valid":
        trailing = tuple(int(size) for size in value.shape[2:])
        return (
            value.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)]
            .contiguous()
            .clone()
        )
    counts = mask.sum(dim=1)
    empty = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if empty:
        shown = ", ".join(str(int(index)) for index in empty)
        raise ValueError(
            f"Prompt pooling requires one valid token per sequence; empty rows: {shown}."
        )
    if mode == "last_valid":
        positions = torch.arange(value.shape[1]).unsqueeze(0).expand(value.shape[0], -1)
        indices = positions.masked_fill(~mask, -1).max(dim=1).values
        pooled = value[torch.arange(value.shape[0]), indices]
    else:
        pooled = torch.stack(
            [value[index, mask[index]].mean(dim=0) for index in range(value.shape[0])]
        )
    result = pooled.detach().contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise TransportTraceError(
            f"Targeted {mode} reduction produced non-finite values."
        )
    return result


def _reduce_block(
    torch: Any,
    block: DecoderBlockTrace,
    mask: Any,
    mode: TargetedTraceMode,
) -> DecoderBlockTrace:
    if mode == "sequence":
        return block
    values: dict[str, Any] = {}
    for field in fields(DecoderBlockTrace):
        if field.name == "index":
            continue
        value = getattr(block, field.name)
        values[field.name] = (
            None if value is None else _reduce_tensor(torch, value, mask, mode)
        )
    return DecoderBlockTrace(index=block.index, **values)


def _normalize_mode(mode: Any, pooling: Any | None) -> TargetedTraceMode:
    aliases = {
        "none": "sequence",
        "full": "sequence",
        "prompt_mean": "mean",
        "prompt_last_valid": "last_valid",
    }
    mode = aliases.get(mode, mode)
    if pooling is not None:
        pooling = aliases.get(pooling, pooling)
        if mode != "sequence" and mode != pooling:
            raise ValueError("mode and pooling request conflicting reductions.")
        mode = pooling
    if mode not in {"sequence", "flatten_valid", "mean", "last_valid"}:
        raise ValueError(
            "mode must be 'sequence', 'flatten_valid', 'mean', or 'last_valid'."
        )
    return mode


def collect_hf_targeted_trace(
    model: Any,
    model_inputs: Mapping[str, Any] | Any | None = None,
    attention_mask: Any | None = None,
    *,
    block_index: int,
    valid_token_mask: Any | None = None,
    mode: TargetedTraceMode = "sequence",
    pooling: Literal["mean", "last_valid"] | None = None,
    capture_final_norm: bool = False,
    capture_logits: bool = False,
    **forward_inputs: Any,
) -> HFTargetedTrace:
    """Run one full forward pass while retaining one selected block only.

    ``flatten_valid`` returns valid rows in batch/sequence order.  ``mean`` and
    ``last_valid`` return one prompt row per batch item.  Final norm tensors are
    captured only when ``capture_final_norm`` or ``capture_logits`` is true;
    head tensors and returned logits are captured only for ``capture_logits``.
    """

    torch = _torch()
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise TypeError("block_index must be an integer.")
    if not isinstance(capture_final_norm, bool) or not isinstance(capture_logits, bool):
        raise TypeError("capture_final_norm and capture_logits must be boolean.")
    selected_mode = _normalize_mode(mode, pooling)
    layout = build_decoder_layout(model)
    if block_index < 0 or block_index >= layout.layer_count:
        raise IndexError(
            f"block_index must be in [0, {layout.layer_count}), got {block_index}."
        )
    checkpointing = _checkpointing_operator(model)
    if checkpointing is not None:
        raise TransportTraceError(
            "Gradient checkpointing must be disabled during targeted trace "
            f"collection; it is enabled at {checkpointing!r}."
        )
    inputs, valid_mask, shape = _normalize_inputs(
        torch,
        model_inputs,
        attention_mask,
        valid_token_mask,
        forward_inputs,
        layout.hidden_size,
    )
    if selected_mode in {"mean", "last_valid"} and bool(
        (valid_mask.sum(dim=1) == 0).any().item()
    ):
        raise ValueError(
            "Prompt pooling requires at least one valid token per sequence."
        )

    store = _CaptureStore(torch)
    installer = _HookInstaller(store)
    module_modes = [(module, bool(module.training)) for module in model.modules()]
    needs_final = capture_final_norm or capture_logits
    try:
        block_layout = layout.blocks[block_index]
        _install_block_hooks(torch, model, layout, block_layout, store, installer)
        if needs_final:
            _install_final_hooks(
                torch,
                model,
                layout,
                store,
                installer,
                capture_logits=capture_logits,
            )
        model.eval()
        with torch.inference_mode():
            output = model(**inputs)
            returned_logits = _model_logits(torch, output)
            if capture_logits:
                store.copy_model_output("model.logits", returned_logits)
        store.verify_complete()
        block = _build_block(torch, layout, block_layout, shape, store.tensors)

        final_interface = None
        if needs_final:
            hidden_shape = (shape[0], shape[1], layout.hidden_size)
            final_input = store.tensors["final_norm.input"]
            final_output = store.tensors["final_norm.output"]
            _expect_shape(torch, final_input, hidden_shape, "final_norm.input")
            _expect_shape(torch, final_output, hidden_shape, "final_norm.output")
            if block_index == layout.layer_count - 1:
                _assert_close(
                    torch, final_input, block.block_output, "final norm residual input"
                )
            head_input = None
            logits = None
            if capture_logits:
                head_input = store.tensors["lm_head.input"]
                head_output = store.tensors["lm_head.output"]
                returned = store.tensors["model.logits"]
                logits_shape = (
                    shape[0],
                    shape[1],
                    layout.embeddings.vocab_size,
                )
                _expect_shape(torch, head_input, hidden_shape, "lm_head.input")
                _expect_shape(torch, head_output, logits_shape, "lm_head.output")
                _expect_shape(torch, returned, logits_shape, "model.logits")
                _assert_close(torch, head_input, final_output, "language-head input")
                _assert_close(torch, returned, head_output, "returned logits")
                logits = returned
            final_interface = FinalNormHeadTrace(
                final_norm_input=_reduce_tensor(
                    torch, final_input, valid_mask, selected_mode
                ),
                final_norm_output=_reduce_tensor(
                    torch, final_output, valid_mask, selected_mode
                ),
                lm_head_input=(
                    None
                    if head_input is None
                    else _reduce_tensor(torch, head_input, valid_mask, selected_mode)
                ),
                logits=(
                    None
                    if logits is None
                    else _reduce_tensor(torch, logits, valid_mask, selected_mode)
                ),
            )

        inventory = store.inventory()
        captured_bytes = sum(item.byte_count for item in inventory.values())
        return HFTargetedTrace(
            layout=layout,
            block_index=block_index,
            block=_reduce_block(torch, block, valid_mask, selected_mode),
            mode=selected_mode,
            source_batch_sequence_shape=shape,
            valid_token_mask=valid_mask,
            valid_token_counts=valid_mask.sum(dim=1).to(dtype=torch.int64),
            final_interface=final_interface,
            hook_counts=MappingProxyType(dict(store.calls)),
            capture_inventory=inventory,
            captured_byte_count=captured_bytes,
        )
    finally:
        installer.remove_all()
        for module, was_training in module_modes:
            module.training = was_training


collect_targeted_trace = collect_hf_targeted_trace
TargetedTransportTrace = HFTargetedTrace


__all__ = [
    "CapturedTensorInfo",
    "FinalNormHeadTrace",
    "HFTargetedTrace",
    "TargetedTraceMode",
    "TargetedTransportTrace",
    "collect_hf_targeted_trace",
    "collect_targeted_trace",
]
