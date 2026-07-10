"""Canonical read/write adapters for validated supported decoder operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neural_forge.hf_arch_layout import (
    AttentionLayout,
    DecoderBlockLayout,
    MLPLayout,
    resolve_module,
)


class OperatorAdapterError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LinearOperator:
    path: str
    weight: Any
    bias: Any | None


@dataclass(frozen=True, slots=True)
class AttentionOperators:
    query: LinearOperator
    key: LinearOperator
    value: LinearOperator
    output: LinearOperator


@dataclass(frozen=True, slots=True)
class MLPOperators:
    kind: str
    input: LinearOperator | None
    gate: LinearOperator | None
    up: LinearOperator | None
    down: LinearOperator


def read_attention_operators(
    model: Any,
    layout: AttentionLayout,
) -> AttentionOperators:
    torch = _require_torch()
    with torch.no_grad():
        if layout.projection_style == "fused_qkv":
            if layout.fused_qkv_path is None:
                raise OperatorAdapterError("Fused attention has no fused QKV path.")
            fused_module = resolve_module(model, layout.fused_qkv_path)
            fused = _read_standard_linear(
                torch,
                fused_module,
                layout.fused_qkv_path,
                layout.projection_weight_layout,
            )
            expected_width = layout.query_width + 2 * layout.key_value_width
            if tuple(fused.weight.shape) != (expected_width, layout.hidden_size):
                raise OperatorAdapterError(
                    "Fused QKV canonical weight has an unexpected shape."
                )
            q_end = layout.query_width
            k_end = q_end + layout.key_value_width
            query = _slice_linear(fused, 0, q_end, f"{fused.path}:query")
            key = _slice_linear(fused, q_end, k_end, f"{fused.path}:key")
            value = _slice_linear(
                fused,
                k_end,
                expected_width,
                f"{fused.path}:value",
            )
        elif layout.projection_style == "split_qkv":
            if (
                layout.query_path is None
                or layout.key_path is None
                or layout.value_path is None
            ):
                raise OperatorAdapterError("Split attention is missing a Q/K/V path.")
            query = _read_path(model, layout.query_path, layout)
            key = _read_path(model, layout.key_path, layout)
            value = _read_path(model, layout.value_path, layout)
        else:  # pragma: no cover - validated architecture literal
            raise OperatorAdapterError(
                f"Unsupported projection style: {layout.projection_style}"
            )
        output = _read_path(model, layout.output_path, layout)
        _expect_shape(query.weight, (layout.query_width, layout.hidden_size), "query")
        _expect_shape(
            key.weight,
            (layout.key_value_width, layout.hidden_size),
            "key",
        )
        _expect_shape(
            value.weight,
            (layout.key_value_width, layout.hidden_size),
            "value",
        )
        _expect_shape(
            output.weight,
            (layout.hidden_size, layout.query_width),
            "attention output",
        )
        return AttentionOperators(
            query=query,
            key=key,
            value=value,
            output=output,
        )


def write_attention_operators(
    model: Any,
    layout: AttentionLayout,
    operators: AttentionOperators,
) -> None:
    torch = _require_torch()
    if not isinstance(operators, AttentionOperators):
        raise OperatorAdapterError("operators must be AttentionOperators.")
    with torch.no_grad():
        _expect_shape(
            operators.query.weight,
            (layout.query_width, layout.hidden_size),
            "query",
        )
        _expect_shape(
            operators.key.weight,
            (layout.key_value_width, layout.hidden_size),
            "key",
        )
        _expect_shape(
            operators.value.weight,
            (layout.key_value_width, layout.hidden_size),
            "value",
        )
        _expect_shape(
            operators.output.weight,
            (layout.hidden_size, layout.query_width),
            "attention output",
        )
        if layout.projection_style == "fused_qkv":
            assert layout.fused_qkv_path is not None
            fused = LinearOperator(
                path=layout.fused_qkv_path,
                weight=torch.cat(
                    [
                        operators.query.weight,
                        operators.key.weight,
                        operators.value.weight,
                    ],
                    dim=0,
                ),
                bias=_cat_biases(
                    torch,
                    operators.query.bias,
                    operators.key.bias,
                    operators.value.bias,
                ),
            )
            _write_path(model, layout.fused_qkv_path, layout, fused)
        else:
            assert layout.query_path is not None
            assert layout.key_path is not None
            assert layout.value_path is not None
            _write_path(model, layout.query_path, layout, operators.query)
            _write_path(model, layout.key_path, layout, operators.key)
            _write_path(model, layout.value_path, layout, operators.value)
        _write_path(model, layout.output_path, layout, operators.output)


def read_mlp_operators(model: Any, block: DecoderBlockLayout) -> MLPOperators:
    torch = _require_torch()
    layout = block.mlp
    with torch.no_grad():
        input_operator = (
            None
            if layout.input_projection_path is None
            else _read_mlp_path(model, layout.input_projection_path, layout)
        )
        gate = (
            None
            if layout.gate_projection_path is None
            else _read_mlp_path(model, layout.gate_projection_path, layout)
        )
        up = (
            None
            if layout.up_projection_path is None
            else _read_mlp_path(model, layout.up_projection_path, layout)
        )
        down = _read_mlp_path(model, layout.down_projection_path, layout)
        if layout.kind == "dense":
            if input_operator is None or gate is not None or up is not None:
                raise OperatorAdapterError("Dense MLP operator contract is incomplete.")
            _expect_shape(
                input_operator.weight,
                (layout.intermediate_size, layout.hidden_size),
                "dense MLP input",
            )
        elif layout.kind == "gated":
            if input_operator is not None or gate is None or up is None:
                raise OperatorAdapterError("Gated MLP operator contract is incomplete.")
            _expect_shape(
                gate.weight,
                (layout.intermediate_size, layout.hidden_size),
                "MLP gate",
            )
            _expect_shape(
                up.weight,
                (layout.intermediate_size, layout.hidden_size),
                "MLP up",
            )
        else:  # pragma: no cover - validated architecture literal
            raise OperatorAdapterError(f"Unsupported MLP kind: {layout.kind}")
        _expect_shape(
            down.weight,
            (layout.hidden_size, layout.intermediate_size),
            "MLP down",
        )
        return MLPOperators(
            kind=layout.kind,
            input=input_operator,
            gate=gate,
            up=up,
            down=down,
        )


def write_mlp_operators(
    model: Any,
    block: DecoderBlockLayout,
    operators: MLPOperators,
) -> None:
    torch = _require_torch()
    layout = block.mlp
    if not isinstance(operators, MLPOperators) or operators.kind != layout.kind:
        raise OperatorAdapterError("MLP operator kind does not match target layout.")
    with torch.no_grad():
        if layout.kind == "dense":
            if (
                operators.input is None
                or operators.gate is not None
                or operators.up is not None
            ):
                raise OperatorAdapterError("Dense MLP write contract is incomplete.")
            assert layout.input_projection_path is not None
            _write_mlp_path(
                model,
                layout.input_projection_path,
                layout,
                operators.input,
            )
        else:
            if (
                operators.input is not None
                or operators.gate is None
                or operators.up is None
            ):
                raise OperatorAdapterError("Gated MLP write contract is incomplete.")
            assert layout.gate_projection_path is not None
            assert layout.up_projection_path is not None
            _write_mlp_path(model, layout.gate_projection_path, layout, operators.gate)
            _write_mlp_path(model, layout.up_projection_path, layout, operators.up)
        _write_mlp_path(model, layout.down_projection_path, layout, operators.down)


def replace_linear_operator(
    operator: LinearOperator,
    *,
    weight: Any,
    bias: Any | None,
) -> LinearOperator:
    return LinearOperator(path=operator.path, weight=weight, bias=bias)


def _read_path(model: Any, path: str, layout: AttentionLayout) -> LinearOperator:
    return _read_standard_linear(
        _require_torch(),
        resolve_module(model, path),
        path,
        layout.projection_weight_layout,
    )


def _write_path(
    model: Any,
    path: str,
    layout: AttentionLayout,
    operator: LinearOperator,
) -> None:
    _write_standard_linear(
        _require_torch(),
        resolve_module(model, path),
        path,
        layout.projection_weight_layout,
        operator,
    )


def _read_mlp_path(model: Any, path: str, layout: MLPLayout) -> LinearOperator:
    return _read_standard_linear(
        _require_torch(),
        resolve_module(model, path),
        path,
        layout.projection_weight_layout,
    )


def _write_mlp_path(
    model: Any,
    path: str,
    layout: MLPLayout,
    operator: LinearOperator,
) -> None:
    _write_standard_linear(
        _require_torch(),
        resolve_module(model, path),
        path,
        layout.projection_weight_layout,
        operator,
    )


def _read_standard_linear(
    torch: Any,
    module: Any,
    path: str,
    weight_layout: str,
) -> LinearOperator:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise OperatorAdapterError(f"Operator {path!r} has no matrix weight.")
    canonical = weight.detach().to(device="cpu", dtype=torch.float64)
    if weight_layout == "input_output":
        canonical = canonical.T
    elif weight_layout != "output_input":
        raise OperatorAdapterError(f"Unknown weight layout: {weight_layout}")
    bias = getattr(module, "bias", None)
    canonical_bias = (
        None
        if bias is None
        else bias.detach().to(device="cpu", dtype=torch.float64).contiguous()
    )
    if not bool(torch.isfinite(canonical).all()) or (
        canonical_bias is not None and not bool(torch.isfinite(canonical_bias).all())
    ):
        raise OperatorAdapterError(f"Operator {path!r} contains non-finite values.")
    if canonical_bias is not None and tuple(canonical_bias.shape) != (
        canonical.shape[0],
    ):
        raise OperatorAdapterError(f"Operator {path!r} bias width is inconsistent.")
    return LinearOperator(
        path=path,
        weight=canonical.contiguous(),
        bias=canonical_bias,
    )


def _write_standard_linear(
    torch: Any,
    module: Any,
    path: str,
    weight_layout: str,
    operator: LinearOperator,
) -> None:
    if not isinstance(operator, LinearOperator):
        raise OperatorAdapterError("operator must be LinearOperator.")
    target_weight = getattr(module, "weight", None)
    if not isinstance(target_weight, torch.Tensor):
        raise OperatorAdapterError(f"Operator {path!r} has no target weight.")
    canonical = _canonical_matrix(torch, operator.weight, f"{path}.weight")
    stored = canonical.T if weight_layout == "input_output" else canonical
    if weight_layout not in {"input_output", "output_input"}:
        raise OperatorAdapterError(f"Unknown weight layout: {weight_layout}")
    if tuple(stored.shape) != tuple(target_weight.shape):
        raise OperatorAdapterError(
            f"Operator {path!r} stored weight shape would be {tuple(stored.shape)}, "
            f"expected {tuple(target_weight.shape)}."
        )
    target_bias = getattr(module, "bias", None)
    if (operator.bias is None) != (target_bias is None):
        raise OperatorAdapterError(f"Operator {path!r} bias presence does not match.")
    target_weight.copy_(
        stored.to(device=target_weight.device, dtype=target_weight.dtype)
    )
    if target_bias is not None:
        bias = _canonical_vector(torch, operator.bias, f"{path}.bias")
        if tuple(bias.shape) != tuple(target_bias.shape):
            raise OperatorAdapterError(f"Operator {path!r} bias shape does not match.")
        target_bias.copy_(bias.to(device=target_bias.device, dtype=target_bias.dtype))


def _slice_linear(
    operator: LinearOperator,
    start: int,
    end: int,
    path: str,
) -> LinearOperator:
    return LinearOperator(
        path=path,
        weight=operator.weight[start:end].contiguous(),
        bias=(None if operator.bias is None else operator.bias[start:end].contiguous()),
    )


def _cat_biases(torch: Any, *biases: Any | None) -> Any | None:
    if all(bias is None for bias in biases):
        return None
    if any(bias is None for bias in biases):
        raise OperatorAdapterError("Q/K/V bias presence must be consistent.")
    return torch.cat(list(biases), dim=0).contiguous()


def _expect_shape(value: Any, expected: tuple[int, int], label: str) -> None:
    if tuple(value.shape) != expected:
        raise OperatorAdapterError(
            f"Canonical {label} weight must have shape {expected}, got "
            f"{tuple(value.shape)}."
        )


def _canonical_matrix(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise OperatorAdapterError(f"{name} must be a matrix tensor.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise OperatorAdapterError(f"{name} contains non-finite values.")
    return result


def _canonical_vector(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise OperatorAdapterError(f"{name} must be a vector tensor.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise OperatorAdapterError(f"{name} contains non-finite values.")
    return result


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for operator adapters.") from exc
    return torch


__all__ = [
    "AttentionOperators",
    "LinearOperator",
    "MLPOperators",
    "OperatorAdapterError",
    "read_attention_operators",
    "read_mlp_operators",
    "replace_linear_operator",
    "write_attention_operators",
    "write_mlp_operators",
]
