"""Closed-form terminal-action residual closure for an LFM2 CPCF child.

The sequential CPCF compiler already folds every layer on activations produced
by the installed child prefix.  What it does not currently compile is one
whole-child correction after all local residual errors have composed through
the final RMSNorm and tied output head.

This module implements that missing analytical primitive without integrating
it into the model compiler:

* FIT supplies actual final-layer child replay rows and cached parent final-norm
  rows.
* A damped, closed-form RMSNorm Gauss-Newton inverse produces a pre-norm target.
* One weighted ridge solve produces a delta for the existing final SwiGLU
  ``w2`` writer.
* A fixed authenticated frontier is materialized as private exact-BF16 states.
* FIT-CHECK may veto a state, SELECTION ranks admissible states, and VALIDATION
  evaluates only the already-selected state.

No model is executed or mutated.  There is no autograd graph, backward pass,
optimizer, trace capture, parent call, new parameter, pruning, or quantization.
Rejected validation returns an exact clone of a functional incumbent BF16
writer.  An exact-zero incumbent is instead an authenticated cold start: the
incumbent/noop is ineligible, the relative-to-incumbent weight cap is undefined,
and one finite fully dense writer is frozen before the other splits are
evaluated diagnostically.  FIT metrics order structurally admissible cold-start
writers and select the sole pre-review tensor; their proxy thresholds never
decide structural eligibility or veto that selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Literal


LFM2_CPCF_TERMINAL_ACTION_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-evidence-v1"
)
LFM2_CPCF_TERMINAL_ACTION_SEED_SCHEMA = "neural-forge-lfm2-cpcf-terminal-action-seed-v1"
LFM2_CPCF_TERMINAL_ACTION_POLICY_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-policy-v1"
)
LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-result-v1"
)
LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-exact-zero-eligibility-v1"
)
_LFM2_CPCF_TERMINAL_ACTION_COMPILER_BINDING_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-compiler-binding-v1"
)
_TERMINAL_RESULT_COMPILER_ISSUER = object()
_TERMINAL_ACTION_RESULT_METHOD = (
    "closed_form_rmsnorm_gauss_newton_then_weighted_ridge_"
    "exact_bfloat16_terminal_writer_frontier"
)

TerminalActionSplit = Literal["fit", "fit_check", "selection", "validation"]
_SPLITS: tuple[TerminalActionSplit, ...] = (
    "fit",
    "fit_check",
    "selection",
    "validation",
)

TERMINAL_ACTION_GAUSS_NEWTON_DAMPING_FRONTIER = (
    1e-6,
    1e-5,
    1e-4,
)
TERMINAL_ACTION_RIDGE_FRONTIER = (
    1e-6,
    1e-5,
    1e-4,
)
TERMINAL_ACTION_ALPHA_FRONTIER = (
    0.0,
    0.125,
    0.25,
    0.5,
    1.0,
)

_ZERO_OPERATION_AUDIT = {
    "parent_model_forward_calls": 0,
    "child_model_forward_calls": 0,
    "model_generate_calls": 0,
    "parent_cache_generation_calls": 0,
    "trace_recaptures": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "new_parameters": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2CpcfTerminalActionResidualError(ValueError):
    """Terminal-action evidence, math, or selection failed closed."""


class Lfm2CpcfTerminalActionAuthenticationError(Lfm2CpcfTerminalActionResidualError):
    """An authenticated policy, seed, evidence bank, or result changed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for CPCF terminal-action residual closure."
        ) from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal-action metadata must be finite canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str = "tensor") -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            f"{name} must be a materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "stride": [int(size) for size in tensor.stride()],
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(values: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {name: _tensor_sha256(value, name) for name, value in sorted(values.items())}
    )


def _finite_number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CpcfTerminalActionResidualError(f"{name} must be a finite number.")
    result = float(value)
    if (
        not math.isfinite(result)
        or (positive and result <= 0.0)
        or (nonnegative and result < 0.0)
    ):
        qualifier = "positive " if positive else "nonnegative " if nonnegative else ""
        raise Lfm2CpcfTerminalActionResidualError(
            f"{name} must be a finite {qualifier}number."
        )
    return result


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        suffix = "" if maximum is None else f" no greater than {maximum}"
        raise Lfm2CpcfTerminalActionResidualError(
            f"{name} must be a positive integer{suffix}."
        )
    return value


def _owned_float_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
    exact_dtype: Any | None = None,
    convert_dtype: Any | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or (exact_dtype is not None and value.dtype != exact_dtype)
    ):
        dtype_text = "" if exact_dtype is None else f" exact {exact_dtype} "
        raise Lfm2CpcfTerminalActionResidualError(
            f"{name} must be a nonempty dense{dtype_text}rank-{dimensions} "
            "real floating tensor."
        )
    result = (
        value.detach()
        .to(
            device="cpu",
            dtype=value.dtype if convert_dtype is None else convert_dtype,
        )
        .contiguous()
        .clone()
    )
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfTerminalActionResidualError(f"{name} contains NaN or infinity.")
    return result


def _owned_weights(value: Any, rows: int, name: str) -> Any:
    torch = _torch()
    weights = _owned_float_tensor(
        value,
        name,
        dimensions=1,
        convert_dtype=torch.float64,
    )
    if int(weights.shape[0]) != rows or not bool((weights > 0.0).all().item()):
        raise Lfm2CpcfTerminalActionResidualError(
            f"{name} must contain one strictly positive value per row."
        )
    total = weights.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
        raise Lfm2CpcfTerminalActionResidualError(
            f"{name} has no positive finite mass."
        )
    return (weights / total).contiguous()


def _row_ids(values: Sequence[str], rows: int) -> tuple[str, ...]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != rows
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "row_ids must contain one identity per evidence row."
        )
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result) or len(set(result)) != rows:
        raise Lfm2CpcfTerminalActionResidualError(
            "row_ids must be nonempty and unique within a split."
        )
    return result


def _rmsnorm(rows: Any, gain: Any, epsilon: float) -> Any:
    torch = _torch()
    source = rows.to(device="cpu", dtype=torch.float64)
    scale = (source.square().mean(dim=-1, keepdim=True) + float(epsilon)).sqrt()
    return (source / scale) * gain.to(device="cpu", dtype=torch.float64)


def rmsnorm_jvp(
    rows: Any,
    vectors: Any,
    gain: Any,
    *,
    epsilon: float,
) -> Any:
    """Apply the exact analytical RMSNorm Jacobian to rowwise vectors."""

    torch = _torch()
    source = _owned_float_tensor(
        rows,
        "rows",
        dimensions=2,
        convert_dtype=torch.float64,
    )
    direction = _owned_float_tensor(
        vectors,
        "vectors",
        dimensions=2,
        convert_dtype=torch.float64,
    )
    diagonal_gain = _owned_float_tensor(
        gain,
        "gain",
        dimensions=1,
        convert_dtype=torch.float64,
    )
    epsilon_value = _finite_number(epsilon, "epsilon", positive=True)
    if tuple(direction.shape) != tuple(source.shape) or int(
        diagonal_gain.shape[0]
    ) != int(source.shape[1]):
        raise Lfm2CpcfTerminalActionResidualError(
            "RMSNorm JVP rows, vectors, and gain have incompatible shapes."
        )
    hidden = int(source.shape[1])
    scale = (source.square().mean(dim=1, keepdim=True) + epsilon_value).sqrt()
    radial = (source * direction).sum(dim=1, keepdim=True)
    result = diagonal_gain * (
        direction / scale - source * radial / (hidden * scale.pow(3))
    )
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfTerminalActionResidualError(
            "RMSNorm JVP produced a nonfinite value."
        )
    return result.contiguous()


def damped_rmsnorm_gauss_newton_target(
    rows: Any,
    output_residual: Any,
    gain: Any,
    *,
    epsilon: float,
    damping: float,
) -> Any:
    """Return the exact damped rowwise Gauss-Newton pre-norm correction.

    For each row this solves

    ``(J.T @ J + damping * I) @ delta = J.T @ output_residual``

    without constructing an ``H x H`` Jacobian or using autograd.  The RMSNorm
    Jacobian is diagonal minus rank one, so its damped normal matrix is diagonal
    plus rank two.  A vectorized two-by-two Woodbury solve is therefore exact in
    FP64.
    """

    torch = _torch()
    source = _owned_float_tensor(
        rows,
        "rows",
        dimensions=2,
        convert_dtype=torch.float64,
    )
    residual = _owned_float_tensor(
        output_residual,
        "output_residual",
        dimensions=2,
        convert_dtype=torch.float64,
    )
    diagonal_gain = _owned_float_tensor(
        gain,
        "gain",
        dimensions=1,
        convert_dtype=torch.float64,
    )
    epsilon_value = _finite_number(epsilon, "epsilon", positive=True)
    damping_value = _finite_number(damping, "damping", positive=True)
    if tuple(residual.shape) != tuple(source.shape) or int(
        diagonal_gain.shape[0]
    ) != int(source.shape[1]):
        raise Lfm2CpcfTerminalActionResidualError(
            "Gauss-Newton rows, output residual, and gain have incompatible shapes."
        )

    hidden = int(source.shape[1])
    scale = (source.square().mean(dim=1, keepdim=True) + epsilon_value).sqrt()
    diagonal = diagonal_gain.reshape(1, hidden) / scale
    rank_vector = diagonal_gain.reshape(1, hidden) * source / (hidden * scale.pow(3))
    normal_cross = diagonal * rank_vector
    rank_energy = rank_vector.square().sum(dim=1, keepdim=True)
    inverse_base = 1.0 / (diagonal.square() + damping_value)
    right = diagonal * residual - source * (rank_vector * residual).sum(
        dim=1, keepdim=True
    )
    base_solution = inverse_base * right

    base_source = inverse_base * source
    base_cross = inverse_base * normal_cross
    matrix_00 = (source * base_source).sum(dim=1)
    matrix_01 = -1.0 + (source * base_cross).sum(dim=1)
    matrix_11 = -rank_energy.squeeze(1) + (normal_cross * base_cross).sum(dim=1)
    right_0 = (source * base_solution).sum(dim=1)
    right_1 = (normal_cross * base_solution).sum(dim=1)
    determinant = matrix_00 * matrix_11 - matrix_01.square()
    determinant_floor = (
        torch.finfo(torch.float64).eps
        * torch.stack(
            (
                matrix_00.abs(),
                matrix_01.abs(),
                matrix_11.abs(),
                torch.ones_like(matrix_00),
            ),
            dim=1,
        )
        .amax(dim=1)
        .square()
    )
    if bool(
        (~torch.isfinite(determinant) | (determinant.abs() <= determinant_floor))
        .any()
        .item()
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "Damped RMSNorm Gauss-Newton Woodbury system is singular."
        )
    coefficient_0 = (right_0 * matrix_11 - matrix_01 * right_1) / determinant
    coefficient_1 = (matrix_00 * right_1 - matrix_01 * right_0) / determinant
    correction = (
        base_solution
        - base_source * coefficient_0.unsqueeze(1)
        - base_cross * coefficient_1.unsqueeze(1)
    )
    if not bool(torch.isfinite(correction).all().item()):
        raise Lfm2CpcfTerminalActionResidualError(
            "Damped RMSNorm Gauss-Newton correction is nonfinite."
        )
    return correction.contiguous()


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTerminalActionEvidence:
    """One immutable split of actual-child terminal replay evidence."""

    split: TerminalActionSplit
    row_ids: tuple[str, ...]
    mlp_input: Any
    pre_mlp_residual: Any
    parent_final_norm_output: Any
    sample_weights: Any
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        torch = _torch()
        if self.split not in _SPLITS:
            raise Lfm2CpcfTerminalActionResidualError(
                "split must be fit, fit_check, selection, or validation."
            )
        mlp_input = _owned_float_tensor(
            self.mlp_input,
            "mlp_input",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        pre_mlp = _owned_float_tensor(
            self.pre_mlp_residual,
            "pre_mlp_residual",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        parent = _owned_float_tensor(
            self.parent_final_norm_output,
            "parent_final_norm_output",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        rows = int(mlp_input.shape[0])
        if (
            int(pre_mlp.shape[0]) != rows
            or int(parent.shape[0]) != rows
            or int(pre_mlp.shape[1]) != int(mlp_input.shape[1])
        ):
            raise Lfm2CpcfTerminalActionResidualError(
                "Terminal evidence row or target-hidden geometry differs."
            )
        row_ids = _row_ids(self.row_ids, rows)
        weights = _owned_weights(self.sample_weights, rows, "sample_weights")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "mlp_input", mlp_input)
        object.__setattr__(self, "pre_mlp_residual", pre_mlp)
        object.__setattr__(self, "parent_final_norm_output", parent)
        object.__setattr__(self, "sample_weights", weights)
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"{self.split} evidence authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    @property
    def rows(self) -> int:
        return int(self.mlp_input.shape[0])

    @property
    def target_hidden_size(self) -> int:
        return int(self.mlp_input.shape[1])

    @property
    def source_hidden_size(self) -> int:
        return int(self.parent_final_norm_output.shape[1])

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_TERMINAL_ACTION_EVIDENCE_SCHEMA,
            "split": self.split,
            "rows": self.rows,
            "target_hidden_size": self.target_hidden_size,
            "source_hidden_size": self.source_hidden_size,
            "row_ids": list(self.row_ids),
            "row_ids_sha256": _canonical_sha256(list(self.row_ids)),
            "mlp_input_sha256": _tensor_sha256(
                self.mlp_input,
                "mlp_input",
            ),
            "pre_mlp_residual_sha256": _tensor_sha256(
                self.pre_mlp_residual,
                "pre_mlp_residual",
            ),
            "parent_final_norm_output_sha256": _tensor_sha256(
                self.parent_final_norm_output,
                "parent_final_norm_output",
            ),
            "sample_weights_sha256": _tensor_sha256(
                self.sample_weights,
                "sample_weights",
            ),
            "sample_weight_sum": float(self.sample_weights.sum().item()),
            "exact_bfloat16_child_replay": True,
            "exact_bfloat16_parent_final_norm_evidence": True,
            "new_causal_suffix_rows_only": True,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _canonical_sha256(
            self.authentication_body()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTerminalActionSeed:
    """The incumbent BF16 final writer and fixed terminal interface."""

    gate_weight: Any
    up_weight: Any
    down_weight: Any
    final_norm_weight: Any
    producer_encoder: Any
    target_tied_weight: Any
    source_tied_table: Any
    norm_epsilon: float
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        torch = _torch()
        gate = _owned_float_tensor(
            self.gate_weight,
            "gate_weight",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        up = _owned_float_tensor(
            self.up_weight,
            "up_weight",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        down = _owned_float_tensor(
            self.down_weight,
            "down_weight",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        norm = _owned_float_tensor(
            self.final_norm_weight,
            "final_norm_weight",
            dimensions=1,
            exact_dtype=torch.bfloat16,
        )
        encoder = _owned_float_tensor(
            self.producer_encoder,
            "producer_encoder",
            dimensions=2,
            convert_dtype=torch.float64,
        )
        target_table = _owned_float_tensor(
            self.target_tied_weight,
            "target_tied_weight",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        source_table = _owned_float_tensor(
            self.source_tied_table,
            "source_tied_table",
            dimensions=2,
            exact_dtype=torch.bfloat16,
        )
        epsilon = _finite_number(
            self.norm_epsilon,
            "norm_epsilon",
            positive=True,
        )
        intermediate, hidden = map(int, gate.shape)
        source_hidden, encoded_hidden = map(int, encoder.shape)
        vocabulary, table_hidden = map(int, target_table.shape)
        if (
            tuple(up.shape) != tuple(gate.shape)
            or tuple(down.shape) != (hidden, intermediate)
            or tuple(norm.shape) != (hidden,)
            or encoded_hidden != hidden
            or table_hidden != hidden
            or tuple(source_table.shape) != (vocabulary, source_hidden)
            or vocabulary < 2
        ):
            raise Lfm2CpcfTerminalActionResidualError(
                "Terminal seed tensors do not form one compatible LFM2 "
                "SwiGLU/final-norm/tied-interface topology."
            )
        object.__setattr__(self, "gate_weight", gate)
        object.__setattr__(self, "up_weight", up)
        object.__setattr__(self, "down_weight", down)
        object.__setattr__(self, "final_norm_weight", norm)
        object.__setattr__(self, "producer_encoder", encoder)
        object.__setattr__(self, "target_tied_weight", target_table)
        object.__setattr__(self, "source_tied_table", source_table)
        object.__setattr__(self, "norm_epsilon", epsilon)
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal-action seed authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    @property
    def target_hidden_size(self) -> int:
        return int(self.gate_weight.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_weight.shape[0])

    @property
    def source_hidden_size(self) -> int:
        return int(self.producer_encoder.shape[0])

    @property
    def vocabulary_size(self) -> int:
        return int(self.target_tied_weight.shape[0])

    def authentication_body(self) -> dict[str, Any]:
        tensors = {
            "feed_forward.w1.weight": self.gate_weight,
            "feed_forward.w3.weight": self.up_weight,
            "feed_forward.w2.weight": self.down_weight,
            "embedding_norm.weight": self.final_norm_weight,
            "interface.producer_encoder": self.producer_encoder,
            "interface.target_tied_weight": self.target_tied_weight,
            "interface.source_tied_table": self.source_tied_table,
        }
        return {
            "schema": LFM2_CPCF_TERMINAL_ACTION_SEED_SCHEMA,
            "target_hidden_size": self.target_hidden_size,
            "intermediate_size": self.intermediate_size,
            "source_hidden_size": self.source_hidden_size,
            "vocabulary_size": self.vocabulary_size,
            "norm_epsilon": self.norm_epsilon,
            "tensor_sha256": {
                name: _tensor_sha256(value, name) for name, value in tensors.items()
            },
            "incumbent_state_sha256": _state_sha256(
                {"feed_forward.w2.weight": self.down_weight}
            ),
            "exact_bfloat16_runtime_parameters": True,
            "tied_output_interface_fixed": True,
            "parameter_count": int(self.down_weight.numel()),
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _canonical_sha256(
            self.authentication_body()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTerminalActionPolicy:
    """Authenticated fixed numerical frontier and promotion gates."""

    gauss_newton_damping_frontier: tuple[float, ...] = (
        TERMINAL_ACTION_GAUSS_NEWTON_DAMPING_FRONTIER
    )
    ridge_frontier: tuple[float, ...] = TERMINAL_ACTION_RIDGE_FRONTIER
    alpha_frontier: tuple[float, ...] = TERMINAL_ACTION_ALPHA_FRONTIER
    minimum_selection_centered_rmse_improvement_fraction: float = 1e-6
    minimum_selection_oracle_top1_gain: float = 0.0
    minimum_selection_parent_top1_gain: float = 0.0
    minimum_selection_parent_positive_margin_gain: float = 0.0
    metric_regression_tolerance: float = 1e-9
    maximum_post_layer_rms_ratio: float = 4.0
    maximum_weight_delta_rms_ratio: float = 4.0
    weight_reference_floor: float = 1e-6
    metric_chunk_rows: int = 64
    maximum_rows_per_split: int = 1_000_000
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if tuple(self.gauss_newton_damping_frontier) != (
            TERMINAL_ACTION_GAUSS_NEWTON_DAMPING_FRONTIER
        ):
            raise Lfm2CpcfTerminalActionResidualError(
                "The terminal Gauss-Newton damping frontier is fixed."
            )
        if tuple(self.ridge_frontier) != TERMINAL_ACTION_RIDGE_FRONTIER:
            raise Lfm2CpcfTerminalActionResidualError(
                "The terminal ridge frontier is fixed."
            )
        if tuple(self.alpha_frontier) != TERMINAL_ACTION_ALPHA_FRONTIER:
            raise Lfm2CpcfTerminalActionResidualError(
                "The terminal exact-BF16 alpha frontier is fixed."
            )
        for name in (
            "minimum_selection_centered_rmse_improvement_fraction",
            "minimum_selection_oracle_top1_gain",
            "minimum_selection_parent_top1_gain",
            "minimum_selection_parent_positive_margin_gain",
            "metric_regression_tolerance",
        ):
            value = _finite_number(
                getattr(self, name),
                name,
                nonnegative=True,
            )
            if value > 1.0:
                raise Lfm2CpcfTerminalActionResidualError(f"{name} cannot exceed one.")
            object.__setattr__(self, name, value)
        for name in (
            "maximum_post_layer_rms_ratio",
            "maximum_weight_delta_rms_ratio",
            "weight_reference_floor",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, positive=True),
            )
        object.__setattr__(
            self,
            "metric_chunk_rows",
            _positive_int(
                self.metric_chunk_rows,
                "metric_chunk_rows",
                maximum=8192,
            ),
        )
        object.__setattr__(
            self,
            "maximum_rows_per_split",
            _positive_int(
                self.maximum_rows_per_split,
                "maximum_rows_per_split",
                maximum=10_000_000,
            ),
        )
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal-action policy authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_TERMINAL_ACTION_POLICY_SCHEMA,
            "gauss_newton_damping_frontier": list(self.gauss_newton_damping_frontier),
            "ridge_frontier": list(self.ridge_frontier),
            "alpha_frontier": list(self.alpha_frontier),
            "minimum_selection_centered_rmse_improvement_fraction": (
                self.minimum_selection_centered_rmse_improvement_fraction
            ),
            "minimum_selection_oracle_top1_gain": (
                self.minimum_selection_oracle_top1_gain
            ),
            "minimum_selection_parent_top1_gain": (
                self.minimum_selection_parent_top1_gain
            ),
            "minimum_selection_parent_positive_margin_gain": (
                self.minimum_selection_parent_positive_margin_gain
            ),
            "metric_regression_tolerance": self.metric_regression_tolerance,
            "maximum_post_layer_rms_ratio": (self.maximum_post_layer_rms_ratio),
            "maximum_weight_delta_rms_ratio": (self.maximum_weight_delta_rms_ratio),
            "weight_reference_floor": self.weight_reference_floor,
            "metric_chunk_rows": self.metric_chunk_rows,
            "maximum_rows_per_split": self.maximum_rows_per_split,
            "frontier_is_fixed": True,
            "candidate_materialization_dtype": "bfloat16",
            "fit_check_can_refit": False,
            "fit_check_can_rank": False,
            "selection_can_fit": False,
            "validation_can_fit": False,
            "validation_can_rank": False,
            "validation_can_only_gate_frozen_selection": True,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _canonical_sha256(
            self.authentication_body()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTerminalActionLogitMetrics:
    rows: int
    centered_logit_relative_rmse: float
    top1_agreement_fraction: float
    reference_winner_mean_margin: float
    reference_winner_positive_margin_fraction: float

    def __post_init__(self) -> None:
        _positive_int(self.rows, "rows")
        rmse = _finite_number(
            self.centered_logit_relative_rmse,
            "centered_logit_relative_rmse",
            nonnegative=True,
        )
        top1 = _finite_number(
            self.top1_agreement_fraction,
            "top1_agreement_fraction",
            nonnegative=True,
        )
        mean_margin = _finite_number(
            self.reference_winner_mean_margin,
            "reference_winner_mean_margin",
        )
        positive = _finite_number(
            self.reference_winner_positive_margin_fraction,
            "reference_winner_positive_margin_fraction",
            nonnegative=True,
        )
        if top1 > 1.0 or positive > 1.0:
            raise Lfm2CpcfTerminalActionResidualError(
                "Terminal top-1 and positive-margin fractions must be in [0,1]."
            )
        object.__setattr__(self, "centered_logit_relative_rmse", rmse)
        object.__setattr__(self, "top1_agreement_fraction", top1)
        object.__setattr__(self, "reference_winner_mean_margin", mean_margin)
        object.__setattr__(
            self,
            "reference_winner_positive_margin_fraction",
            positive,
        )

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTerminalActionSplitMetrics:
    split: TerminalActionSplit
    oracle: Lfm2CpcfTerminalActionLogitMetrics
    mapped_parent: Lfm2CpcfTerminalActionLogitMetrics
    maximum_post_layer_rms_ratio_to_incumbent: float
    mean_post_layer_rms_ratio_to_incumbent: float
    finite: bool

    def __post_init__(self) -> None:
        if self.split not in _SPLITS:
            raise Lfm2CpcfTerminalActionResidualError(
                "Terminal split metrics use an invalid split."
            )
        for name in (
            "maximum_post_layer_rms_ratio_to_incumbent",
            "mean_post_layer_rms_ratio_to_incumbent",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(
                    getattr(self, name),
                    name,
                    nonnegative=True,
                ),
            )
        if not isinstance(self.finite, bool):
            raise TypeError("finite must be boolean.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "oracle": self.oracle.as_dict(),
            "mapped_parent": self.mapped_parent.as_dict(),
            "maximum_post_layer_rms_ratio_to_incumbent": (
                self.maximum_post_layer_rms_ratio_to_incumbent
            ),
            "mean_post_layer_rms_ratio_to_incumbent": (
                self.mean_post_layer_rms_ratio_to_incumbent
            ),
            "finite": self.finite,
        }


@dataclass(frozen=True, slots=True)
class _PreparedSplit:
    evidence: Lfm2CpcfTerminalActionEvidence
    features: Any
    pre_mlp_residual: Any
    parent_final_norm_output: Any
    oracle_hidden: Any
    incumbent_post_layer: Any
    incumbent_final_norm_output: Any


def _prepare_split(
    evidence: Lfm2CpcfTerminalActionEvidence,
    seed: Lfm2CpcfTerminalActionSeed,
) -> _PreparedSplit:
    torch = _torch()
    if (
        evidence.target_hidden_size != seed.target_hidden_size
        or evidence.source_hidden_size != seed.source_hidden_size
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            f"{evidence.split} evidence differs from the terminal seed geometry."
        )
    mlp_input = evidence.mlp_input.to(torch.float64)
    gate = seed.gate_weight.to(torch.float64)
    up = seed.up_weight.to(torch.float64)
    features = torch.nn.functional.silu(mlp_input @ gate.T) * (mlp_input @ up.T)
    pre_mlp = evidence.pre_mlp_residual.to(torch.float64)
    incumbent = pre_mlp + features @ seed.down_weight.to(torch.float64).T
    incumbent_norm = _rmsnorm(
        incumbent,
        seed.final_norm_weight,
        seed.norm_epsilon,
    )
    parent = evidence.parent_final_norm_output.to(torch.float64)
    oracle = parent @ seed.producer_encoder
    values = (features, pre_mlp, incumbent, incumbent_norm, parent, oracle)
    if not all(bool(torch.isfinite(value).all().item()) for value in values):
        raise Lfm2CpcfTerminalActionResidualError(
            f"{evidence.split} terminal preparation produced nonfinite rows."
        )
    return _PreparedSplit(
        evidence=evidence,
        features=features.contiguous(),
        pre_mlp_residual=pre_mlp.contiguous(),
        parent_final_norm_output=parent.contiguous(),
        oracle_hidden=oracle.contiguous(),
        incumbent_post_layer=incumbent.contiguous(),
        incumbent_final_norm_output=incumbent_norm.contiguous(),
    )


@dataclass(slots=True)
class _PairMetricAccumulator:
    error_square: float = 0.0
    reference_square: float = 0.0
    top1: float = 0.0
    margin: float = 0.0
    positive_margin: float = 0.0


def _accumulate_pair_metrics(
    accumulator: _PairMetricAccumulator,
    candidate_logits: Any,
    reference_logits: Any,
    weights: Any,
) -> None:
    torch = _torch()
    candidate_centered = candidate_logits - candidate_logits.mean(
        dim=1,
        keepdim=True,
    )
    reference_centered = reference_logits - reference_logits.mean(
        dim=1,
        keepdim=True,
    )
    difference = candidate_centered - reference_centered
    accumulator.error_square += float(
        (weights[:, None] * difference.square()).sum().item()
    )
    accumulator.reference_square += float(
        (weights[:, None] * reference_centered.square()).sum().item()
    )
    reference_winner = reference_logits.argmax(dim=1, keepdim=True)
    candidate_winner = candidate_logits.argmax(dim=1, keepdim=True)
    accumulator.top1 += float(
        (weights * (reference_winner == candidate_winner).squeeze(1).to(torch.float64))
        .sum()
        .item()
    )
    top_two = torch.topk(candidate_logits, k=2, dim=1)
    best_is_reference = (top_two.indices[:, :1] == reference_winner).squeeze(1)
    competitor = torch.where(
        best_is_reference,
        top_two.values[:, 1],
        top_two.values[:, 0],
    )
    winner_value = candidate_logits.gather(
        1,
        reference_winner,
    ).squeeze(1)
    margin = winner_value - competitor
    accumulator.margin += float((weights * margin).sum().item())
    accumulator.positive_margin += float(
        (weights * (margin > 0.0).to(torch.float64)).sum().item()
    )


def _finish_pair_metrics(
    accumulator: _PairMetricAccumulator,
    rows: int,
) -> Lfm2CpcfTerminalActionLogitMetrics:
    torch = _torch()
    fraction_tolerance = 64.0 * torch.finfo(torch.float64).eps
    for name, value in (
        ("top1", accumulator.top1),
        ("positive_margin", accumulator.positive_margin),
    ):
        if value < -fraction_tolerance or value > 1.0 + fraction_tolerance:
            raise Lfm2CpcfTerminalActionResidualError(
                f"Accumulated terminal {name} fraction escaped [0,1]."
            )

    def bounded_fraction(value: float) -> float:
        if abs(value) <= fraction_tolerance:
            return 0.0
        if abs(value - 1.0) <= fraction_tolerance:
            return 1.0
        return min(1.0, max(0.0, value))

    denominator = max(
        accumulator.reference_square,
        torch.finfo(torch.float64).tiny,
    )
    return Lfm2CpcfTerminalActionLogitMetrics(
        rows=rows,
        centered_logit_relative_rmse=math.sqrt(
            max(0.0, accumulator.error_square) / denominator
        ),
        top1_agreement_fraction=bounded_fraction(accumulator.top1),
        reference_winner_mean_margin=accumulator.margin,
        reference_winner_positive_margin_fraction=(
            bounded_fraction(accumulator.positive_margin)
        ),
    )


def _candidate_post_layer(
    prepared: _PreparedSplit,
    down_weight: Any,
) -> Any:
    return (
        prepared.pre_mlp_residual
        + prepared.features @ down_weight.to(dtype=_torch().float64).T
    ).contiguous()


def _evaluate_split(
    prepared: _PreparedSplit,
    seed: Lfm2CpcfTerminalActionSeed,
    down_weight: Any,
    *,
    metric_chunk_rows: int,
) -> Lfm2CpcfTerminalActionSplitMetrics:
    torch = _torch()
    if down_weight.dtype != torch.bfloat16:
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal candidate evaluation requires an exact BF16 w2 state."
        )
    candidate_post = _candidate_post_layer(prepared, down_weight)
    candidate_norm = _rmsnorm(
        candidate_post,
        seed.final_norm_weight,
        seed.norm_epsilon,
    )
    incumbent_rms = prepared.incumbent_post_layer.square().mean(dim=1).sqrt()
    candidate_rms = candidate_post.square().mean(dim=1).sqrt()
    ratio = candidate_rms / incumbent_rms.clamp_min(torch.finfo(torch.float64).tiny)
    finite = bool(
        torch.isfinite(candidate_post).all().item()
        and torch.isfinite(candidate_norm).all().item()
        and torch.isfinite(ratio).all().item()
    )
    if not finite:
        raise Lfm2CpcfTerminalActionResidualError(
            "An exact BF16 terminal candidate produced nonfinite replay."
        )

    target_table = seed.target_tied_weight.to(torch.float64)
    source_table = seed.source_tied_table.to(torch.float64)
    oracle_accumulator = _PairMetricAccumulator()
    parent_accumulator = _PairMetricAccumulator()
    rows = prepared.evidence.rows
    weights = prepared.evidence.sample_weights
    for start in range(0, rows, metric_chunk_rows):
        stop = min(rows, start + metric_chunk_rows)
        chunk_weights = weights[start:stop]
        candidate_logits = candidate_norm[start:stop] @ target_table.T
        oracle_logits = prepared.oracle_hidden[start:stop] @ target_table.T
        parent_logits = prepared.parent_final_norm_output[start:stop] @ source_table.T
        if not bool(
            torch.isfinite(candidate_logits).all().item()
            and torch.isfinite(oracle_logits).all().item()
            and torch.isfinite(parent_logits).all().item()
        ):
            raise Lfm2CpcfTerminalActionResidualError(
                "Terminal logit evaluation produced NaN or infinity."
            )
        _accumulate_pair_metrics(
            oracle_accumulator,
            candidate_logits,
            oracle_logits,
            chunk_weights,
        )
        _accumulate_pair_metrics(
            parent_accumulator,
            candidate_logits,
            parent_logits,
            chunk_weights,
        )
    return Lfm2CpcfTerminalActionSplitMetrics(
        split=prepared.evidence.split,
        oracle=_finish_pair_metrics(oracle_accumulator, rows),
        mapped_parent=_finish_pair_metrics(parent_accumulator, rows),
        maximum_post_layer_rms_ratio_to_incumbent=float(ratio.max().item()),
        mean_post_layer_rms_ratio_to_incumbent=float((weights * ratio).sum().item()),
        finite=finite,
    )


def _evaluate_oracle_ceiling(
    prepared: _PreparedSplit,
    seed: Lfm2CpcfTerminalActionSeed,
    *,
    metric_chunk_rows: int,
) -> Lfm2CpcfTerminalActionLogitMetrics:
    torch = _torch()
    target_table = seed.target_tied_weight.to(torch.float64)
    source_table = seed.source_tied_table.to(torch.float64)
    accumulator = _PairMetricAccumulator()
    rows = prepared.evidence.rows
    for start in range(0, rows, metric_chunk_rows):
        stop = min(rows, start + metric_chunk_rows)
        oracle_logits = prepared.oracle_hidden[start:stop] @ target_table.T
        parent_logits = prepared.parent_final_norm_output[start:stop] @ source_table.T
        _accumulate_pair_metrics(
            accumulator,
            oracle_logits,
            parent_logits,
            prepared.evidence.sample_weights[start:stop],
        )
    return _finish_pair_metrics(accumulator, rows)


def _weight_delta_rms_ratio(
    candidate: Any,
    incumbent: Any,
    floor: float,
) -> float:
    torch = _torch()
    delta = candidate.to(torch.float64) - incumbent.to(torch.float64)
    numerator = float(delta.square().mean().sqrt().item())
    denominator = max(
        float(incumbent.to(torch.float64).square().mean().sqrt().item()),
        floor,
    )
    return numerator / denominator


def _terminal_writer_structural_coverage(candidate: Any) -> Mapping[str, Any]:
    """Seal complete dense W2 coverage for an exact-zero cold start.

    A zero writer has no meaningful incumbent-relative scale.  Cold-start
    publication therefore uses an exact structural predicate: every BF16
    coefficient must be finite, the matrix must contain signal, and every
    target-output row and source-feature column must participate.  Individual
    zero coefficients remain valid dense topology.  This does not impose a
    magnitude threshold or compare parent and child weight values.
    """

    torch = _torch()
    if (
        not isinstance(candidate, torch.Tensor)
        or candidate.layout != torch.strided
        or candidate.device.type == "meta"
        or candidate.ndim != 2
        or candidate.dtype != torch.bfloat16
        or candidate.numel() <= 0
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "Cold-start terminal W2 must be a materialized dense BF16 matrix."
        )
    value = candidate.detach().to(device="cpu").contiguous()
    finite = torch.isfinite(value)
    nonzero = value.ne(0)
    finite_count = int(finite.sum().item())
    nonzero_count = int(nonzero.sum().item())
    output_width, source_feature_width = map(int, value.shape)
    covered_outputs = int(nonzero.any(dim=1).sum().item())
    covered_sources = int(nonzero.any(dim=0).sum().item())
    body = {
        "schema": "neural-forge-lfm2-terminal-w2-cold-start-coverage-v1",
        "tensor_sha256": _tensor_sha256(value, "cold_start_terminal_w2"),
        "shape": [output_width, source_feature_width],
        "dtype": "bfloat16",
        "element_count": int(value.numel()),
        "finite_element_count": finite_count,
        "nonzero_element_count": nonzero_count,
        "all_elements_finite": finite_count == int(value.numel()),
        "all_elements_nonzero": nonzero_count == int(value.numel()),
        "target_output_axis_size": output_width,
        "covered_target_output_count": covered_outputs,
        "all_target_outputs_covered": covered_outputs == output_width,
        "source_feature_axis_size": source_feature_width,
        "covered_source_feature_count": covered_sources,
        "all_source_features_covered": covered_sources == source_feature_width,
        "complete_dense_writer_coverage": bool(
            finite_count == int(value.numel())
            and nonzero_count > 0
            and covered_outputs == output_width
            and covered_sources == source_feature_width
        ),
        "numeric_parent_child_weight_equality_required": False,
    }
    return MappingProxyType({**body, "coverage_sha256": _canonical_sha256(body)})


def _metric_gates(
    candidate: Lfm2CpcfTerminalActionSplitMetrics,
    incumbent: Lfm2CpcfTerminalActionSplitMetrics,
    policy: Lfm2CpcfTerminalActionPolicy,
    *,
    require_selection_improvement: bool,
) -> dict[str, bool]:
    tolerance = policy.metric_regression_tolerance
    baseline_rmse = incumbent.oracle.centered_logit_relative_rmse
    candidate_rmse = candidate.oracle.centered_logit_relative_rmse
    improvement_fraction = (baseline_rmse - candidate_rmse) / max(
        baseline_rmse, _torch().finfo(_torch().float64).tiny
    )
    gates = {
        "finite": candidate.finite,
        "post_layer_norm": (
            candidate.maximum_post_layer_rms_ratio_to_incumbent
            <= policy.maximum_post_layer_rms_ratio
        ),
        "oracle_centered_rmse_nonregression": (
            candidate_rmse <= baseline_rmse + tolerance
        ),
        "oracle_top1_nonregression": (
            candidate.oracle.top1_agreement_fraction + tolerance
            >= incumbent.oracle.top1_agreement_fraction
        ),
        "mapped_parent_centered_rmse_nonregression": (
            candidate.mapped_parent.centered_logit_relative_rmse
            <= incumbent.mapped_parent.centered_logit_relative_rmse + tolerance
        ),
        "mapped_parent_top1_nonregression": (
            candidate.mapped_parent.top1_agreement_fraction + tolerance
            >= incumbent.mapped_parent.top1_agreement_fraction
        ),
        "mapped_parent_positive_margin_nonregression": (
            candidate.mapped_parent.reference_winner_positive_margin_fraction
            + tolerance
            >= incumbent.mapped_parent.reference_winner_positive_margin_fraction
        ),
    }
    if require_selection_improvement:
        gates.update(
            {
                "selection_centered_rmse_improvement": (
                    improvement_fraction + tolerance
                    >= policy.minimum_selection_centered_rmse_improvement_fraction
                ),
                "selection_oracle_top1_gain": (
                    candidate.oracle.top1_agreement_fraction
                    - incumbent.oracle.top1_agreement_fraction
                    + tolerance
                    >= policy.minimum_selection_oracle_top1_gain
                ),
                "selection_parent_top1_gain": (
                    candidate.mapped_parent.top1_agreement_fraction
                    - incumbent.mapped_parent.top1_agreement_fraction
                    + tolerance
                    >= policy.minimum_selection_parent_top1_gain
                ),
                "selection_parent_positive_margin_gain": (
                    candidate.mapped_parent.reference_winner_positive_margin_fraction
                    - incumbent.mapped_parent.reference_winner_positive_margin_fraction
                    + tolerance
                    >= policy.minimum_selection_parent_positive_margin_gain
                ),
            }
        )
    return gates


def _all_passed(gates: Mapping[str, bool]) -> bool:
    return bool(gates) and all(value is True for value in gates.values())


def _selection_key(
    metrics: Lfm2CpcfTerminalActionSplitMetrics,
    *,
    alpha: float,
    damping: float,
    ridge: float,
    state_sha256: str,
) -> tuple[Any, ...]:
    return (
        metrics.oracle.centered_logit_relative_rmse,
        -metrics.oracle.top1_agreement_fraction,
        metrics.mapped_parent.centered_logit_relative_rmse,
        -metrics.mapped_parent.top1_agreement_fraction,
        -metrics.mapped_parent.reference_winner_positive_margin_fraction,
        metrics.maximum_post_layer_rms_ratio_to_incumbent,
        alpha,
        damping,
        ridge,
        state_sha256,
    )


def _selection_key_report(key: tuple[Any, ...]) -> list[Any]:
    return [float(value) if isinstance(value, (float, int)) else value for value in key]


def _validate_evidence_set(
    seed: Lfm2CpcfTerminalActionSeed,
    evidence: Sequence[Lfm2CpcfTerminalActionEvidence],
    policy: Lfm2CpcfTerminalActionPolicy,
) -> tuple[Lfm2CpcfTerminalActionEvidence, ...]:
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or len(evidence) != len(_SPLITS)
        or not all(
            isinstance(value, Lfm2CpcfTerminalActionEvidence) for value in evidence
        )
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "evidence must contain one terminal bank for every required split."
        )
    by_split = {value.split: value for value in evidence}
    if set(by_split) != set(_SPLITS):
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal evidence must exactly cover FIT, FIT-CHECK, SELECTION, "
            "and VALIDATION."
        )
    ordered = tuple(by_split[split] for split in _SPLITS)
    seen: set[str] = set()
    evidence_digests: set[str] = set()
    for value in ordered:
        if not value.verify_authentication():
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"{value.split} evidence changed after authentication."
            )
        overlap = seen.intersection(value.row_ids)
        if overlap:
            raise Lfm2CpcfTerminalActionResidualError(
                "Terminal split row identities overlap: "
                + ", ".join(sorted(overlap)[:3])
                + "."
            )
        seen.update(value.row_ids)
        if value.authentication_sha256 in evidence_digests:
            raise Lfm2CpcfTerminalActionResidualError(
                "Two terminal splits have the same evidence seal."
            )
        evidence_digests.add(value.authentication_sha256)
        if (
            value.rows > policy.maximum_rows_per_split
            or value.target_hidden_size != seed.target_hidden_size
            or value.source_hidden_size != seed.source_hidden_size
        ):
            raise Lfm2CpcfTerminalActionResidualError(
                f"{value.split} evidence exceeds policy or seed geometry."
            )
    return ordered


def _ridge_system(prepared_fit: _PreparedSplit) -> tuple[Any, Any, Any]:
    torch = _torch()
    features = prepared_fit.features
    weights = prepared_fit.evidence.sample_weights
    gram = features.T @ (weights[:, None] * features)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    if not bool(
        torch.isfinite(gram).all().item()
        and torch.isfinite(eigenvalues).all().item()
        and torch.isfinite(eigenvectors).all().item()
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal weighted ridge Gram is nonfinite."
        )
    return gram.contiguous(), eigenvalues.contiguous(), eigenvectors.contiguous()


def _ridge_delta(
    prepared_fit: _PreparedSplit,
    target_delta: Any,
    eigenvalues: Any,
    eigenvectors: Any,
    ridge: float,
) -> Any:
    weights = prepared_fit.evidence.sample_weights
    right = prepared_fit.features.T @ (weights[:, None] * target_delta)
    eigen_right = eigenvectors.T @ right
    solution = eigenvectors @ (eigen_right / (eigenvalues + ridge).unsqueeze(1))
    if not bool(_torch().isfinite(solution).all().item()):
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal weighted ridge delta is nonfinite."
        )
    return solution.T.contiguous()


def _terminal_compiler_input_binding(
    *,
    seed: Lfm2CpcfTerminalActionSeed,
    evidence: Sequence[Lfm2CpcfTerminalActionEvidence],
    policy: Lfm2CpcfTerminalActionPolicy,
    incumbent: Any,
) -> Mapping[str, Any]:
    """Bind a result to the exact authenticated compiler inputs that issued it."""

    body = {
        "schema": _LFM2_CPCF_TERMINAL_ACTION_COMPILER_BINDING_SCHEMA,
        "seed_sha256": seed.authentication_sha256,
        "incumbent_seed_writer_sha256": _tensor_sha256(
            incumbent,
            "compiler_binding.incumbent",
        ),
        "policy_sha256": policy.authentication_sha256,
        "split_evidence_sha256": {
            value.split: value.authentication_sha256 for value in evidence
        },
        "split_rows": {value.split: value.rows for value in evidence},
        "split_row_ids_sha256": {
            value.split: _canonical_sha256(list(value.row_ids)) for value in evidence
        },
    }
    return MappingProxyType({**body, "binding_sha256": _canonical_sha256(body)})


def _reported_frontier_selection_key(record: Mapping[str, Any]) -> list[Any]:
    """Reconstruct one reported FIT key instead of trusting its serialized copy."""

    metrics = record.get("fit_metrics")
    oracle = metrics.get("oracle") if isinstance(metrics, Mapping) else None
    mapped_parent = (
        metrics.get("mapped_parent") if isinstance(metrics, Mapping) else None
    )
    if not isinstance(oracle, Mapping) or not isinstance(mapped_parent, Mapping):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Cold-start frontier omitted authenticated FIT metrics."
        )

    def number(container: Mapping[str, Any], name: str) -> float:
        value = container.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier metric {name!r} is not numeric."
            )
        result = float(value)
        if not math.isfinite(result):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier metric {name!r} is nonfinite."
            )
        return result

    state_sha256 = _required_sha256(
        record.get("state_sha256"),
        "cold-start frontier state_sha256",
    )
    alpha = number(record, "alpha")
    damping = number(record, "gauss_newton_damping")
    ridge = number(record, "ridge")
    return [
        number(oracle, "centered_logit_relative_rmse"),
        -number(oracle, "top1_agreement_fraction"),
        number(mapped_parent, "centered_logit_relative_rmse"),
        -number(mapped_parent, "top1_agreement_fraction"),
        -number(mapped_parent, "reference_winner_positive_margin_fraction"),
        number(metrics, "maximum_post_layer_rms_ratio_to_incumbent"),
        alpha,
        damping,
        ridge,
        state_sha256,
    ]


def _authenticate_exact_zero_compiled_result(
    *,
    body: Mapping[str, Any],
    incumbent: Any,
    selected: Any,
    published: Any,
    policy: Lfm2CpcfTerminalActionPolicy,
    compiler_input_binding: Mapping[str, Any] | None,
    compiler_issued: bool,
) -> Mapping[str, Any]:
    """Fail closed unless a cold result is the internally issued FIT frontier."""

    torch = _torch()
    if not compiler_issued or not isinstance(compiler_input_binding, Mapping):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Accepted exact-zero terminal results must be compiler-issued."
        )
    binding = dict(compiler_input_binding)
    binding_sha256 = binding.pop("binding_sha256", None)
    if (
        binding.get("schema") != _LFM2_CPCF_TERMINAL_ACTION_COMPILER_BINDING_SCHEMA
        or binding_sha256 != _canonical_sha256(binding)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero terminal compiler-input binding is invalid."
        )
    expected_output_hashes = {
        "incumbent": _tensor_sha256(incumbent, "incumbent"),
        "selected": _tensor_sha256(selected, "selected"),
        "published": _tensor_sha256(published, "published"),
    }
    if (
        body.get("schema") != LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA
        or body.get("method") != _TERMINAL_ACTION_RESULT_METHOD
        or body.get("policy_sha256") != policy.authentication_sha256
        or binding.get("policy_sha256") != policy.authentication_sha256
        or body.get("seed_sha256") != binding.get("seed_sha256")
        or body.get("incumbent_seed_writer_sha256")
        != binding.get("incumbent_seed_writer_sha256")
        or binding.get("incumbent_seed_writer_sha256")
        != expected_output_hashes["incumbent"]
        or body.get("split_evidence_sha256") != binding.get("split_evidence_sha256")
        or body.get("split_rows") != binding.get("split_rows")
        or body.get("split_row_ids_sha256") != binding.get("split_row_ids_sha256")
        or body.get("output_state_sha256") != expected_output_hashes
        or body.get("operation_audit") != dict(_ZERO_OPERATION_AUDIT)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero terminal result is not bound to its compiler inputs."
        )
    selected_id = body.get("selected_candidate_id")
    if (
        not isinstance(selected_id, str)
        or not selected_id
        or selected_id == "incumbent"
        or body.get("selected_nonincumbent") is not True
        or body.get("accepted") is not True
        or body.get("published_is_incumbent") is not False
        or body.get("rollback_performed") is not False
        or body.get("exact_zero_cold_start") is not True
        or body.get("incumbent_all_elements_zero") is not True
        or int(torch.count_nonzero(incumbent).item()) != 0
        or torch.equal(selected, incumbent)
        or not torch.equal(published, selected)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero terminal publication state is internally inconsistent."
        )

    frontier = body.get("frontier")
    expected_count = 1 + (
        len(policy.gauss_newton_damping_frontier)
        * len(policy.ridge_frontier)
        * (len(policy.alpha_frontier) - 1)
    )
    expected_candidate_ids = {"incumbent"}
    expected_candidate_ids.update(
        f"gn-{damping_index:02d}-ridge-{ridge_index:02d}-alpha-{alpha_index:02d}"
        for damping_index, _damping in enumerate(policy.gauss_newton_damping_frontier)
        for ridge_index, _ridge in enumerate(policy.ridge_frontier)
        for alpha_index, _alpha in enumerate(policy.alpha_frontier[1:], start=1)
    )
    if (
        not isinstance(frontier, list)
        or len(frontier) != expected_count
        or body.get("frontier_count") != expected_count
        or body.get("expected_frontier_count") != expected_count
        or not all(isinstance(value, Mapping) for value in frontier)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero terminal frontier cardinality is invalid."
        )
    records = {str(value.get("candidate_id")): value for value in frontier}
    if len(records) != expected_count or set(records) != expected_candidate_ids:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero terminal frontier identities are invalid."
        )
    incumbent_record = records["incumbent"]
    incumbent_key = _reported_frontier_selection_key(incumbent_record)
    if (
        incumbent_record.get("state_sha256") != expected_output_hashes["incumbent"]
        or incumbent_record.get("selection_key") != incumbent_key
        or incumbent_record.get("eligible") is not False
        or incumbent_record.get("exact_zero_incumbent_noop_forbidden") is not True
        or incumbent_record.get("candidate_ranking_split") != "fit"
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero incumbent frontier control is invalid."
        )

    eligible_records = []
    for candidate_id, record in records.items():
        reported_key = _reported_frontier_selection_key(record)
        if record.get("selection_key") != reported_key:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier key for {candidate_id!r} is invalid."
            )
        if candidate_id == "incumbent":
            continue
        coverage = record.get("structural_coverage")
        fit_gates = record.get("fit_gates")
        if not isinstance(coverage, Mapping) or not isinstance(fit_gates, Mapping):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier record {candidate_id!r} is incomplete."
            )
        coverage_body = dict(coverage)
        coverage_sha256 = coverage_body.pop("coverage_sha256", None)
        coverage_authenticated = bool(
            coverage_sha256 == _canonical_sha256(coverage_body)
            and record.get("structural_coverage_sha256") == coverage_sha256
            and coverage.get("tensor_sha256") == record.get("state_sha256")
        )
        if not coverage_authenticated:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier coverage for {candidate_id!r} is invalid."
            )
        coverage_complete = coverage.get("complete_dense_writer_coverage") is True
        if record.get("weight_norm_gate_passed") is not coverage_complete:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier coverage gate for {candidate_id!r} is invalid."
            )
        if not fit_gates or any(type(value) is not bool for value in fit_gates.values()):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier FIT diagnostics for {candidate_id!r} are "
                "invalid."
            )
        fit_passed = all(value is True for value in fit_gates.values())
        effectful_nonincumbent = (
            record.get("state_sha256") != expected_output_hashes["incumbent"]
        )
        derived_eligible = bool(
            record.get("weight_norm_gate_passed") is True
            and coverage_complete
            and effectful_nonincumbent
        )
        if (
            record.get("candidate_ranking_split") != "fit"
            or record.get("relative_to_zero_weight_delta_gate_applied") is not False
            or record.get("effectful_nonincumbent") is not effectful_nonincumbent
            or record.get("structurally_eligible") is not derived_eligible
            or record.get("candidate_eligibility_basis")
            != "finite_dense_coverage_and_nonincumbent_effect"
            or record.get("fit_metric_gates_passed") is not fit_passed
            or record.get(
                "fit_metric_thresholds_used_for_candidate_eligibility"
            )
            is not False
            or record.get("fit_metrics_used_for_candidate_ordering") is not True
            or record.get("fit_metrics_used_for_pre_review_candidate_selection")
            is not True
            or record.get("fit_metrics_used_for_structural_eligibility") is not False
            or record.get("fit_metric_thresholds_used_for_publication") is not False
            or record.get("fit_metrics_have_final_intelligence_authority") is not False
            or record.get("eligible") is not derived_eligible
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Cold-start frontier eligibility for {candidate_id!r} is invalid."
            )
        if derived_eligible:
            eligible_records.append((tuple(reported_key), candidate_id, record))
    if not eligible_records:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero terminal frontier has no structurally eligible candidate."
        )
    _winning_key, winning_id, selected_record = min(eligible_records)
    selected_metrics = body.get("selected_metrics_before_validation")
    if (
        winning_id != selected_id
        or selected_record.get("state_sha256") != expected_output_hashes["selected"]
        or body.get("selected_state_sha256") != expected_output_hashes["selected"]
        or body.get("selected_selection_key") != selected_record.get("selection_key")
        or not isinstance(selected_metrics, Mapping)
        or selected_record.get("fit_metrics") != selected_metrics.get("fit")
        or selected_record.get("fit_check_metrics") != selected_metrics.get("fit_check")
        or selected_record.get("selection_metrics") != selected_metrics.get("selection")
        or selected_record.get("fit_check_and_selection_metrics_diagnostic_only")
        is not True
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero terminal winner is not the authenticated FIT minimum."
        )
    unevaluated_marker = {"not_evaluated_before_fit_selection_freeze": True}
    if any(
        record.get("fit_check_metrics") is not None
        or record.get("selection_metrics") is not None
        or record.get("fit_check_gates") != unevaluated_marker
        or record.get("selection_gates") != unevaluated_marker
        for candidate_id, record in records.items()
        if candidate_id not in {"incumbent", selected_id}
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "A nonselected cold-start frontier state consumed held-out evidence."
        )
    split_roles = body.get("split_roles")
    structurally_eligible_fit_misses = sum(
        int(record.get("fit_metric_gates_passed") is False)
        for _key, _candidate_id, record in eligible_records
    )
    if (
        body.get("exact_zero_candidate_eligibility_schema")
        != LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA
        or body.get("candidate_selection_evidence_scope")
        != "finite-dense-structural-eligibility-plus-fit-ordering"
        or body.get("candidate_eligibility_basis")
        != "finite_dense_coverage_and_nonincumbent_effect"
        or body.get("fit_metrics_role")
        != "ordering-among-structurally-eligible-not-threshold-gating"
        or body.get("fit_metrics_used_for_candidate_ordering") is not True
        or body.get("fit_metrics_used_for_pre_review_candidate_selection") is not True
        or body.get("fit_metrics_used_for_structural_eligibility") is not False
        or body.get("fit_metrics_have_final_intelligence_authority") is not False
        or body.get("fit_metric_thresholds_used_for_candidate_eligibility") is not False
        or body.get("fit_metric_thresholds_used_for_publication") is not False
        or body.get("structurally_eligible_frontier_count")
        != len(eligible_records)
        or body.get("structurally_eligible_fit_metric_gate_miss_count")
        != structurally_eligible_fit_misses
        or body.get("fit_metric_gate_miss_can_exclude_structural_candidate")
        is not False
        or body.get("selected_before_any_heldout_publication_diagnostic") is not True
        or body.get("heldout_candidate_evaluation_count_before_fit_freeze") != 0
        or body.get("heldout_candidate_evaluation_count_after_fit_freeze") != 6
        or body.get("heldout_candidate_count_evaluated_after_fit_freeze") != 2
        or not isinstance(split_roles, Mapping)
        or split_roles.get("candidate_eligibility_uses_only_finite_dense_structure")
        is not True
        or split_roles.get("fit_used_for_candidate_eligibility") is not False
        or split_roles.get("only_fit_rows_used_for_candidate_ordering")
        is not True
        or split_roles.get("fit_metrics_used_for_pre_review_candidate_selection")
        is not True
        or split_roles.get("fit_metrics_have_final_intelligence_authority") is not False
        or split_roles.get("fit_metric_thresholds_used_for_publication") is not False
        or split_roles.get("fit_check_diagnostic_only") is not True
        or split_roles.get("selection_diagnostic_only") is not True
        or split_roles.get("validation_diagnostic_only") is not True
        or body.get("published_writer_is_finite_dense_nonzero_structural_plant")
        is not True
        or body.get("nonfinite_candidate_or_replay_is_hard_failure") is not True
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Exact-zero held-out diagnostic ordering is not authenticated."
        )
    return MappingProxyType({**binding, "binding_sha256": binding_sha256})


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTerminalActionResidualResult:
    """Authenticated private selection and rollback-safe publication state."""

    policy: Lfm2CpcfTerminalActionPolicy
    incumbent_down_weight: Any
    selected_down_weight: Any
    published_down_weight: Any
    selected_candidate_id: str
    accepted: bool
    report: Mapping[str, Any]
    result_sha256: str
    _compiler_input_binding: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _compiler_issuer: InitVar[Any] = None
    _compiler_issued: bool = field(
        init=False,
        default=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _compiler_issuer: Any) -> None:
        torch = _torch()
        if (
            not isinstance(self.policy, Lfm2CpcfTerminalActionPolicy)
            or not self.policy.verify_authentication()
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal result policy is unauthenticated."
            )
        shapes = set()
        owned = []
        for name, value in (
            ("incumbent_down_weight", self.incumbent_down_weight),
            ("selected_down_weight", self.selected_down_weight),
            ("published_down_weight", self.published_down_weight),
        ):
            tensor = _owned_float_tensor(
                value,
                name,
                dimensions=2,
                exact_dtype=torch.bfloat16,
            )
            shapes.add(tuple(tensor.shape))
            owned.append(tensor)
        if len(shapes) != 1:
            raise Lfm2CpcfTerminalActionResidualError(
                "Terminal result BF16 writer shapes differ."
            )
        if not isinstance(self.selected_candidate_id, str) or not (
            self.selected_candidate_id
        ):
            raise Lfm2CpcfTerminalActionResidualError(
                "selected_candidate_id must be nonempty text."
            )
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be boolean.")
        result_sha = _required_sha256(self.result_sha256, "result_sha256")
        body = dict(self.report)
        supplied = body.pop("result_sha256", None)
        if supplied != result_sha or _canonical_sha256(body) != result_sha:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal result report seal is invalid."
            )
        expected_hashes = {
            "incumbent": _tensor_sha256(owned[0], "incumbent"),
            "selected": _tensor_sha256(owned[1], "selected"),
            "published": _tensor_sha256(owned[2], "published"),
        }
        if (
            body.get("output_state_sha256") != expected_hashes
            or body.get("selected_candidate_id") != self.selected_candidate_id
            or body.get("accepted") is not self.accepted
            or body.get("policy_sha256") != self.policy.authentication_sha256
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal result tensors differ from their report."
            )
        compiler_issued = _compiler_issuer is _TERMINAL_RESULT_COMPILER_ISSUER
        compiler_binding = self._compiler_input_binding
        accepted_exact_zero = bool(
            self.accepted and int(torch.count_nonzero(owned[0]).item()) == 0
        )
        if accepted_exact_zero:
            compiler_binding = _authenticate_exact_zero_compiled_result(
                body=body,
                incumbent=owned[0],
                selected=owned[1],
                published=owned[2],
                policy=self.policy,
                compiler_input_binding=compiler_binding,
                compiler_issued=compiler_issued,
            )
        object.__setattr__(self, "incumbent_down_weight", owned[0])
        object.__setattr__(self, "selected_down_weight", owned[1])
        object.__setattr__(self, "published_down_weight", owned[2])
        object.__setattr__(self, "_compiler_input_binding", compiler_binding)
        object.__setattr__(self, "_compiler_issued", compiler_issued)
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "result_sha256": result_sha}),
        )
        object.__setattr__(self, "result_sha256", result_sha)

    @property
    def changed(self) -> bool:
        return bool(
            self.accepted
            and not _torch().equal(
                self.published_down_weight,
                self.incumbent_down_weight,
            )
        )

    def verify_authentication(self) -> bool:
        body = dict(self.report)
        supplied = body.pop("result_sha256", None)
        base_valid = bool(
            supplied == self.result_sha256
            and _canonical_sha256(body) == self.result_sha256
            and body.get("output_state_sha256")
            == {
                "incumbent": _tensor_sha256(
                    self.incumbent_down_weight,
                    "incumbent",
                ),
                "selected": _tensor_sha256(
                    self.selected_down_weight,
                    "selected",
                ),
                "published": _tensor_sha256(
                    self.published_down_weight,
                    "published",
                ),
            }
        )
        if not base_valid:
            return False
        if (
            self.accepted
            and int(_torch().count_nonzero(self.incumbent_down_weight).item()) == 0
        ):
            try:
                _authenticate_exact_zero_compiled_result(
                    body=body,
                    incumbent=self.incumbent_down_weight,
                    selected=self.selected_down_weight,
                    published=self.published_down_weight,
                    policy=self.policy,
                    compiler_input_binding=self._compiler_input_binding,
                    compiler_issued=self._compiler_issued,
                )
            except Lfm2CpcfTerminalActionResidualError:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def compile_lfm2_cpcf_terminal_action_residual(
    *,
    seed: Lfm2CpcfTerminalActionSeed,
    fit_evidence: Lfm2CpcfTerminalActionEvidence,
    fit_check_evidence: Lfm2CpcfTerminalActionEvidence,
    selection_evidence: Lfm2CpcfTerminalActionEvidence,
    validation_evidence: Lfm2CpcfTerminalActionEvidence,
    policy: Lfm2CpcfTerminalActionPolicy | None = None,
) -> Lfm2CpcfTerminalActionResidualResult:
    """Compile and select one private exact-BF16 final-writer correction."""

    torch = _torch()
    if not isinstance(seed, Lfm2CpcfTerminalActionSeed):
        raise TypeError("seed must be Lfm2CpcfTerminalActionSeed.")
    selected_policy = policy or Lfm2CpcfTerminalActionPolicy()
    if (
        not isinstance(selected_policy, Lfm2CpcfTerminalActionPolicy)
        or not selected_policy.verify_authentication()
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "policy must be an authenticated terminal-action policy."
        )
    if not seed.verify_authentication():
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal-action seed changed after authentication."
        )
    ordered_evidence = _validate_evidence_set(
        seed,
        (
            fit_evidence,
            fit_check_evidence,
            selection_evidence,
            validation_evidence,
        ),
        selected_policy,
    )
    incumbent_input_sha = _tensor_sha256(seed.down_weight, "seed.down_weight")
    incumbent = seed.down_weight.detach().clone()
    if incumbent.data_ptr() == seed.down_weight.data_ptr():
        raise Lfm2CpcfTerminalActionResidualError(
            "Incumbent private snapshot unexpectedly aliases the seed."
        )
    exact_zero_cold_start = bool(torch.count_nonzero(incumbent).item() == 0)
    compiler_input_binding = _terminal_compiler_input_binding(
        seed=seed,
        evidence=ordered_evidence,
        policy=selected_policy,
        incumbent=incumbent,
    )
    evidence_by_split = {value.split: value for value in ordered_evidence}
    frontier_metric_splits = ("fit",) if exact_zero_cold_start else _SPLITS[:-1]
    prepared = {
        split: _prepare_split(evidence_by_split[split], seed)
        for split in frontier_metric_splits
    }

    incumbent_metrics = {
        split: _evaluate_split(
            prepared[split],
            seed,
            incumbent,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        for split in frontier_metric_splits
    }
    oracle_ceilings = {
        split: _evaluate_oracle_ceiling(
            prepared[split],
            seed,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        for split in frontier_metric_splits
    }
    gram, eigenvalues, eigenvectors = _ridge_system(prepared["fit"])
    gram_sha = _tensor_sha256(gram, "fit.gram")
    eigenvalue_sha = _tensor_sha256(eigenvalues, "fit.eigenvalues")

    incumbent_state_sha = _tensor_sha256(incumbent, "incumbent")
    incumbent_key = _selection_key(
        incumbent_metrics["fit" if exact_zero_cold_start else "selection"],
        alpha=0.0,
        damping=selected_policy.gauss_newton_damping_frontier[0],
        ridge=selected_policy.ridge_frontier[0],
        state_sha256=incumbent_state_sha,
    )
    incumbent_frontier = {
        "candidate_id": "incumbent",
        "gauss_newton_damping": (selected_policy.gauss_newton_damping_frontier[0]),
        "ridge": selected_policy.ridge_frontier[0],
        "alpha": 0.0,
        "exact_bfloat16_materialization": True,
        "private_candidate_storage": True,
        "state_sha256": incumbent_state_sha,
        "weight_delta_rms_ratio": 0.0,
        "weight_norm_gate_passed": True,
        "fit_metrics": incumbent_metrics["fit"].as_dict(),
        "fit_check_metrics": (
            None if exact_zero_cold_start else incumbent_metrics["fit_check"].as_dict()
        ),
        "selection_metrics": (
            None if exact_zero_cold_start else incumbent_metrics["selection"].as_dict()
        ),
        "fit_gates": {"incumbent_control": True},
        "fit_check_gates": {"incumbent_control": True},
        "selection_gates": {"incumbent_control": True},
        "eligible": True,
        "selection_key": _selection_key_report(incumbent_key),
    }
    if exact_zero_cold_start:
        incumbent_frontier.update(
            {
                "weight_norm_gate_passed": False,
                "fit_gates": {"exact_zero_control_not_publishable": True},
                "fit_check_gates": {"not_evaluated_before_fit_selection_freeze": True},
                "selection_gates": {"not_evaluated_before_fit_selection_freeze": True},
                "eligible": False,
                "exact_zero_incumbent_noop_forbidden": True,
                "candidate_ranking_split": "fit",
            }
        )
    frontier: list[dict[str, Any]] = [incumbent_frontier]
    selected_id = "incumbent"
    selected_key = None if exact_zero_cold_start else incumbent_key
    selected_weight = incumbent.detach().clone()
    selected_hyperparameters = {
        "gauss_newton_damping": (selected_policy.gauss_newton_damping_frontier[0]),
        "ridge": selected_policy.ridge_frontier[0],
        "alpha": 0.0,
    }
    selected_split_metrics = {
        split: incumbent_metrics[split] for split in frontier_metric_splits
    }
    target_residual = (
        prepared["fit"].oracle_hidden - prepared["fit"].incumbent_final_norm_output
    )
    fit_target_reports = []

    with torch.no_grad():
        for damping_index, damping in enumerate(
            selected_policy.gauss_newton_damping_frontier
        ):
            target_delta = damped_rmsnorm_gauss_newton_target(
                prepared["fit"].incumbent_post_layer,
                target_residual,
                seed.final_norm_weight,
                epsilon=seed.norm_epsilon,
                damping=damping,
            )
            jvp_prediction = rmsnorm_jvp(
                prepared["fit"].incumbent_post_layer,
                target_delta,
                seed.final_norm_weight,
                epsilon=seed.norm_epsilon,
            )
            target_report = {
                "damping_index": damping_index,
                "damping": damping,
                "target_delta_sha256": _tensor_sha256(
                    target_delta,
                    f"fit.target_delta.{damping_index}",
                ),
                "linearized_output_relative_rmse": float(
                    (
                        (jvp_prediction - target_residual).square().sum()
                        / target_residual.square()
                        .sum()
                        .clamp_min(torch.finfo(torch.float64).tiny)
                    )
                    .sqrt()
                    .item()
                ),
                "analytical_rmsnorm_jacobian": "diagonal_minus_rank_one",
                "analytical_damped_normal_inverse": ("diagonal_plus_rank_two_woodbury"),
            }
            fit_target_reports.append(target_report)
            for ridge_index, ridge in enumerate(selected_policy.ridge_frontier):
                delta_weight = _ridge_delta(
                    prepared["fit"],
                    target_delta,
                    eigenvalues,
                    eigenvectors,
                    ridge,
                )
                for alpha_index, alpha in enumerate(
                    selected_policy.alpha_frontier[1:],
                    start=1,
                ):
                    candidate_id = (
                        f"gn-{damping_index:02d}-ridge-{ridge_index:02d}-"
                        f"alpha-{alpha_index:02d}"
                    )
                    candidate = (
                        (
                            seed.down_weight.to(torch.float32)
                            + alpha * delta_weight.to(torch.float32)
                        )
                        .to(torch.bfloat16)
                        .contiguous()
                        .clone()
                    )
                    if (
                        candidate.data_ptr() == seed.down_weight.data_ptr()
                        or candidate.data_ptr() == incumbent.data_ptr()
                        or candidate.requires_grad
                        or candidate.grad_fn is not None
                    ):
                        raise Lfm2CpcfTerminalActionResidualError(
                            "A terminal frontier state is not private detached BF16."
                        )
                    state_sha = _tensor_sha256(candidate, candidate_id)
                    weight_ratio = _weight_delta_rms_ratio(
                        candidate,
                        incumbent,
                        selected_policy.weight_reference_floor,
                    )
                    structural_coverage = (
                        _terminal_writer_structural_coverage(candidate)
                        if exact_zero_cold_start
                        else None
                    )
                    weight_gate = bool(torch.isfinite(candidate).all().item()) and (
                        structural_coverage.get("complete_dense_writer_coverage")
                        is True
                        if exact_zero_cold_start
                        else weight_ratio
                        <= selected_policy.maximum_weight_delta_rms_ratio
                    )
                    candidate_metrics = {
                        split: _evaluate_split(
                            prepared[split],
                            seed,
                            candidate,
                            metric_chunk_rows=(selected_policy.metric_chunk_rows),
                        )
                        for split in frontier_metric_splits
                    }
                    fit_gates = _metric_gates(
                        candidate_metrics["fit"],
                        incumbent_metrics["fit"],
                        selected_policy,
                        require_selection_improvement=False,
                    )
                    if exact_zero_cold_start:
                        fit_check_gates = {
                            "not_evaluated_before_fit_selection_freeze": True
                        }
                        selection_gates = {
                            "not_evaluated_before_fit_selection_freeze": True
                        }
                    else:
                        fit_check_gates = _metric_gates(
                            candidate_metrics["fit_check"],
                            incumbent_metrics["fit_check"],
                            selected_policy,
                            require_selection_improvement=False,
                        )
                        selection_gates = _metric_gates(
                            candidate_metrics["selection"],
                            incumbent_metrics["selection"],
                            selected_policy,
                            require_selection_improvement=True,
                        )
                    key = _selection_key(
                        candidate_metrics[
                            "fit" if exact_zero_cold_start else "selection"
                        ],
                        alpha=alpha,
                        damping=damping,
                        ridge=ridge,
                        state_sha256=state_sha,
                    )
                    effectful_nonincumbent = not torch.equal(candidate, incumbent)
                    fit_metric_gates_passed = _all_passed(fit_gates)
                    cold_start_structurally_eligible = bool(
                        exact_zero_cold_start
                        and weight_gate
                        and effectful_nonincumbent
                    )
                    eligible = bool(
                        weight_gate
                        and (
                            effectful_nonincumbent
                            if exact_zero_cold_start
                            else fit_metric_gates_passed
                            and _all_passed(fit_check_gates)
                            and _all_passed(selection_gates)
                        )
                    )
                    candidate_report = {
                        "candidate_id": candidate_id,
                        "gauss_newton_damping": damping,
                        "ridge": ridge,
                        "alpha": alpha,
                        "exact_bfloat16_materialization": True,
                        "private_candidate_storage": True,
                        "state_sha256": state_sha,
                        "weight_delta_rms_ratio": weight_ratio,
                        "weight_norm_gate_passed": weight_gate,
                        "fit_metrics": (candidate_metrics["fit"].as_dict()),
                        "fit_check_metrics": (
                            None
                            if exact_zero_cold_start
                            else candidate_metrics["fit_check"].as_dict()
                        ),
                        "selection_metrics": (
                            None
                            if exact_zero_cold_start
                            else candidate_metrics["selection"].as_dict()
                        ),
                        "fit_gates": fit_gates,
                        "fit_check_gates": fit_check_gates,
                        "selection_gates": selection_gates,
                        "eligible": eligible,
                        "selection_key": _selection_key_report(key),
                    }
                    if exact_zero_cold_start:
                        candidate_report.update(
                            {
                                "candidate_ranking_split": "fit",
                                "fit_check_and_selection_metrics_diagnostic_only": (
                                    True
                                ),
                                "relative_to_zero_weight_delta_gate_applied": False,
                                "legacy_relative_weight_delta_gate_would_pass": (
                                    weight_ratio
                                    <= selected_policy.maximum_weight_delta_rms_ratio
                                ),
                                "effectful_nonincumbent": effectful_nonincumbent,
                                "structurally_eligible": (
                                    cold_start_structurally_eligible
                                ),
                                "candidate_eligibility_basis": (
                                    "finite_dense_coverage_and_nonincumbent_effect"
                                ),
                                "fit_metric_gates_passed": (
                                    fit_metric_gates_passed
                                ),
                                "fit_metric_thresholds_used_for_candidate_eligibility": (
                                    False
                                ),
                                "fit_metrics_used_for_candidate_ordering": True,
                                "fit_metrics_used_for_pre_review_candidate_selection": (
                                    True
                                ),
                                "fit_metrics_used_for_structural_eligibility": False,
                                "fit_metric_thresholds_used_for_publication": False,
                                "fit_metrics_have_final_intelligence_authority": False,
                                "structural_coverage": dict(structural_coverage),
                                "structural_coverage_sha256": structural_coverage[
                                    "coverage_sha256"
                                ],
                            }
                        )
                    frontier.append(candidate_report)
                    if eligible and (selected_key is None or key < selected_key):
                        selected_id = candidate_id
                        selected_key = key
                        selected_weight = candidate.detach().clone()
                        selected_hyperparameters = {
                            "gauss_newton_damping": damping,
                            "ridge": ridge,
                            "alpha": alpha,
                        }
                        selected_split_metrics = dict(candidate_metrics)

    selected_structural_coverage = None
    if exact_zero_cold_start:
        if selected_id == "incumbent" or selected_key is None:
            raise Lfm2CpcfTerminalActionResidualError(
                "Exact-zero terminal cold start found no finite, fully dense "
                "effectful closed-form W2 candidate; incumbent publication is "
                "forbidden."
            )
        selected_structural_coverage = _terminal_writer_structural_coverage(
            selected_weight
        )
        if selected_structural_coverage.get(
            "complete_dense_writer_coverage"
        ) is not True or torch.equal(selected_weight, incumbent):
            raise Lfm2CpcfTerminalActionResidualError(
                "Exact-zero terminal cold start did not freeze a complete dense "
                "nonincumbent W2 before held-out diagnostics."
            )

    selected_before_heldout = selected_id
    selected_state_sha = _tensor_sha256(
        selected_weight,
        "selected_down_weight",
    )
    if exact_zero_cold_start:
        frozen_selected_metrics: dict[str, Lfm2CpcfTerminalActionSplitMetrics] = {}
        try:
            for split in _SPLITS[1:]:
                prepared[split] = _prepare_split(evidence_by_split[split], seed)
                incumbent_metrics[split] = _evaluate_split(
                    prepared[split],
                    seed,
                    incumbent,
                    metric_chunk_rows=selected_policy.metric_chunk_rows,
                )
                oracle_ceilings[split] = _evaluate_oracle_ceiling(
                    prepared[split],
                    seed,
                    metric_chunk_rows=selected_policy.metric_chunk_rows,
                )
                frozen_selected_metrics[split] = _evaluate_split(
                    prepared[split],
                    seed,
                    selected_weight,
                    metric_chunk_rows=selected_policy.metric_chunk_rows,
                )
        except Exception as exc:
            raise Lfm2CpcfTerminalActionResidualError(
                "A held-out terminal diagnostic failed after the structurally "
                "eligible, FIT-ordered W2 candidate was frozen; no result was "
                "issued."
            ) from exc
        selected_split_metrics.update(
            {
                split: frozen_selected_metrics[split]
                for split in ("fit_check", "selection")
            }
        )
        selected_frontier_record = next(
            value for value in frontier if value["candidate_id"] == selected_id
        )
        diagnostic_gates = {
            "fit_check": _metric_gates(
                frozen_selected_metrics["fit_check"],
                incumbent_metrics["fit_check"],
                selected_policy,
                require_selection_improvement=False,
            ),
            "selection": _metric_gates(
                frozen_selected_metrics["selection"],
                incumbent_metrics["selection"],
                selected_policy,
                require_selection_improvement=True,
            ),
        }
        incumbent_frontier.update(
            {
                "fit_check_metrics": incumbent_metrics["fit_check"].as_dict(),
                "selection_metrics": incumbent_metrics["selection"].as_dict(),
                "fit_check_gates": {"diagnostic_only_after_fit_freeze": True},
                "selection_gates": {"diagnostic_only_after_fit_freeze": True},
            }
        )
        selected_frontier_record.update(
            {
                "fit_check_metrics": frozen_selected_metrics["fit_check"].as_dict(),
                "selection_metrics": frozen_selected_metrics["selection"].as_dict(),
                "fit_check_gates": diagnostic_gates["fit_check"],
                "selection_gates": diagnostic_gates["selection"],
                "heldout_diagnostics_evaluated_after_fit_selection_freeze": True,
                "frozen_candidate_state_sha256": selected_state_sha,
            }
        )
        selected_validation_metrics = frozen_selected_metrics["validation"]
    else:
        prepared["validation"] = _prepare_split(
            evidence_by_split["validation"],
            seed,
        )
        incumbent_metrics["validation"] = _evaluate_split(
            prepared["validation"],
            seed,
            incumbent,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        oracle_ceilings["validation"] = _evaluate_oracle_ceiling(
            prepared["validation"],
            seed,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        selected_validation_metrics = _evaluate_split(
            prepared["validation"],
            seed,
            selected_weight,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
    validation_gates = _metric_gates(
        selected_validation_metrics,
        incumbent_metrics["validation"],
        selected_policy,
        require_selection_improvement=False,
    )
    validation_passed = _all_passed(validation_gates)
    selected_nonincumbent = selected_id != "incumbent"
    accepted = bool(
        selected_nonincumbent and (exact_zero_cold_start or validation_passed)
    )
    published = (
        selected_weight.detach().clone() if accepted else incumbent.detach().clone()
    )
    rollback_performed = bool(
        not exact_zero_cold_start and selected_nonincumbent and not validation_passed
    )
    rollback_exact = bool(not rollback_performed or torch.equal(published, incumbent))
    if not rollback_exact:
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal validation rejection did not restore the exact incumbent."
        )
    if _tensor_sha256(seed.down_weight, "seed.down_weight") != incumbent_input_sha:
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal compilation mutated the authenticated seed writer."
        )
    if exact_zero_cold_start and (
        not accepted
        or torch.equal(published, incumbent)
        or _terminal_writer_structural_coverage(published).get(
            "complete_dense_writer_coverage"
        )
        is not True
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "Exact-zero terminal cold start attempted to publish a noop or an "
            "incomplete W2."
        )

    expected_frontier_count = 1 + (
        len(selected_policy.gauss_newton_damping_frontier)
        * len(selected_policy.ridge_frontier)
        * (len(selected_policy.alpha_frontier) - 1)
    )
    if (
        len(frontier) != expected_frontier_count
        or selected_id != selected_before_heldout
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal frontier cardinality or frozen selection changed."
        )
    cold_start_structurally_eligible_count = 0
    cold_start_fit_metric_gate_miss_count = 0
    if exact_zero_cold_start:
        cold_start_candidates = tuple(
            value for value in frontier if value["candidate_id"] != "incumbent"
        )
        cold_start_structurally_eligible_count = sum(
            int(value["structurally_eligible"] is True)
            for value in cold_start_candidates
        )
        cold_start_fit_metric_gate_miss_count = sum(
            int(
                value["structurally_eligible"] is True
                and value["fit_metric_gates_passed"] is False
            )
            for value in cold_start_candidates
        )
        if cold_start_structurally_eligible_count <= 0:
            raise Lfm2CpcfTerminalActionResidualError(
                "Exact-zero terminal frontier lost every structurally eligible "
                "candidate."
            )
    output_hashes = {
        "incumbent": _tensor_sha256(incumbent, "incumbent"),
        "selected": _tensor_sha256(selected_weight, "selected"),
        "published": _tensor_sha256(published, "published"),
    }
    validation_report = {
        "evaluated_candidate_id": selected_id,
        "candidate_count": 1,
        "metrics": selected_validation_metrics.as_dict(),
        "gates": validation_gates,
        "passed": validation_passed,
        "used_for_fit": False,
        "used_for_fit_check_veto": False,
        "used_for_candidate_ranking": False,
        "used_to_change_selected_candidate_id": False,
        "used_only_for_frozen_candidate_acceptance": True,
        "accessed_only_after_selected_id_and_state_sha256_frozen": True,
    }
    split_roles = {
        "fit_used_for_gauss_newton_target": True,
        "fit_used_for_weighted_ridge": True,
        "fit_check_used_for_fit": False,
        "fit_check_used_for_candidate_veto": True,
        "fit_check_used_for_candidate_ranking": False,
        "selection_used_for_fit": False,
        "selection_used_for_candidate_ranking": True,
        "validation_used_for_fit": False,
        "validation_used_for_candidate_ranking": False,
        "validation_candidate_count": 1,
    }
    if exact_zero_cold_start:
        validation_report.update(
            {
                "used_only_for_frozen_candidate_acceptance": False,
                "heldout_diagnostic_only": True,
                "publication_gate_applied": False,
                "frozen_candidate_state_sha256": selected_state_sha,
            }
        )
        split_roles.update(
            {
                "fit_check_used_for_candidate_veto": False,
                "fit_check_diagnostic_only": True,
                "selection_used_for_candidate_ranking": False,
                "selection_diagnostic_only": True,
                "validation_diagnostic_only": True,
                "candidate_eligibility_uses_only_finite_dense_structure": True,
                "fit_used_for_candidate_eligibility": False,
                "only_fit_rows_used_for_candidate_ordering": True,
                "fit_metrics_used_for_pre_review_candidate_selection": True,
                "fit_metrics_have_final_intelligence_authority": False,
                "fit_metric_thresholds_used_for_publication": False,
            }
        )
    body = {
        "schema": LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA,
        "method": _TERMINAL_ACTION_RESULT_METHOD,
        "policy_sha256": selected_policy.authentication_sha256,
        "seed_sha256": seed.authentication_sha256,
        "split_evidence_sha256": {
            value.split: value.authentication_sha256 for value in ordered_evidence
        },
        "split_rows": {value.split: value.rows for value in ordered_evidence},
        "split_row_ids_sha256": {
            value.split: _canonical_sha256(list(value.row_ids))
            for value in ordered_evidence
        },
        "split_row_id_sets_disjoint": True,
        "fit_weighted_ridge": {
            "gram_sha256": gram_sha,
            "eigenvalues_sha256": eigenvalue_sha,
            "feature_width": int(gram.shape[0]),
            "output_width": seed.target_hidden_size,
            "sample_weights_sha256": _tensor_sha256(
                prepared["fit"].evidence.sample_weights,
                "fit.sample_weights",
            ),
            "fit_rows_only": True,
            "fit_check_rows_used": 0,
            "selection_rows_used": 0,
            "validation_rows_used": 0,
        },
        "gauss_newton_targets": fit_target_reports,
        "incumbent_metrics": {
            split: metrics.as_dict() for split, metrics in incumbent_metrics.items()
        },
        "oracle_to_mapped_parent_ceiling": {
            split: metrics.as_dict() for split, metrics in oracle_ceilings.items()
        },
        "frontier": frontier,
        "frontier_count": len(frontier),
        "expected_frontier_count": expected_frontier_count,
        "frontier_exact_bfloat16_materialization": True,
        "frontier_candidates_private_and_never_installed": True,
        "selected_candidate_id": selected_id,
        "selected_hyperparameters": selected_hyperparameters,
        "selected_state_sha256": selected_state_sha,
        "selected_selection_key": _selection_key_report(selected_key),
        "selected_metrics_before_validation": {
            split: metrics.as_dict()
            for split, metrics in selected_split_metrics.items()
        },
        "validation": validation_report,
        "split_roles": split_roles,
        "selected_nonincumbent": selected_nonincumbent,
        "accepted": accepted,
        "rollback_performed": rollback_performed,
        "rollback_exact": rollback_exact,
        "published_is_incumbent": not accepted,
        "source_seed_writer_unchanged": True,
        "parameter_identity_or_topology_mutation": False,
        "parameter_count_before": int(seed.down_weight.numel()),
        "parameter_count_after": int(published.numel()),
        "output_state_sha256": output_hashes,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    if exact_zero_cold_start:
        body.update(
            {
                "exact_zero_cold_start": True,
                "incumbent_seed_writer_sha256": incumbent_state_sha,
                "incumbent_all_elements_zero": True,
                "incumbent_nonzero_element_count": 0,
                "incumbent_noop_publication_forbidden": True,
                "exact_zero_candidate_eligibility_schema": (
                    LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA
                ),
                "candidate_selection_evidence_scope": (
                    "finite-dense-structural-eligibility-plus-fit-ordering"
                ),
                "candidate_eligibility_basis": (
                    "finite_dense_coverage_and_nonincumbent_effect"
                ),
                "fit_metrics_role": (
                    "ordering-among-structurally-eligible-not-threshold-gating"
                ),
                "fit_metrics_used_for_candidate_ordering": True,
                "fit_metrics_used_for_pre_review_candidate_selection": True,
                "fit_metrics_used_for_structural_eligibility": False,
                "fit_metrics_have_final_intelligence_authority": False,
                "fit_metric_thresholds_used_for_candidate_eligibility": False,
                "fit_metric_thresholds_used_for_publication": False,
                "structurally_eligible_frontier_count": (
                    cold_start_structurally_eligible_count
                ),
                "structurally_eligible_fit_metric_gate_miss_count": (
                    cold_start_fit_metric_gate_miss_count
                ),
                "fit_metric_gate_miss_can_exclude_structural_candidate": False,
                "relative_to_zero_weight_delta_gate_applied": False,
                "heldout_evidence_role": ("diagnostic-only-after-frozen-fit-selection"),
                "selected_before_any_heldout_publication_diagnostic": True,
                "heldout_candidate_evaluation_count_before_fit_freeze": 0,
                "heldout_candidate_evaluation_count_after_fit_freeze": 6,
                "heldout_candidate_count_evaluated_after_fit_freeze": 2,
                "selected_writer_structural_coverage": dict(
                    selected_structural_coverage
                ),
                "selected_writer_structural_coverage_sha256": (
                    selected_structural_coverage["coverage_sha256"]
                ),
                "selected_writer_all_elements_finite": True,
                "selected_writer_all_elements_nonzero": (
                    selected_structural_coverage["all_elements_nonzero"]
                ),
                "selected_writer_all_source_features_covered": True,
                "published_writer_is_finite_dense_nonzero_structural_plant": True,
                "nonfinite_candidate_or_replay_is_hard_failure": True,
                "zero_training_pruning_quantization": True,
            }
        )
    result_sha = _canonical_sha256(body)
    report = {**body, "result_sha256": result_sha}
    return Lfm2CpcfTerminalActionResidualResult(
        policy=selected_policy,
        incumbent_down_weight=incumbent,
        selected_down_weight=selected_weight,
        published_down_weight=published,
        selected_candidate_id=selected_id,
        accepted=accepted,
        report=report,
        result_sha256=result_sha,
        _compiler_input_binding=compiler_input_binding,
        _compiler_issuer=_TERMINAL_RESULT_COMPILER_ISSUER,
    )


__all__ = [
    "LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_EVIDENCE_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_POLICY_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_SEED_SCHEMA",
    "Lfm2CpcfTerminalActionAuthenticationError",
    "Lfm2CpcfTerminalActionEvidence",
    "Lfm2CpcfTerminalActionLogitMetrics",
    "Lfm2CpcfTerminalActionPolicy",
    "Lfm2CpcfTerminalActionResidualError",
    "Lfm2CpcfTerminalActionResidualResult",
    "Lfm2CpcfTerminalActionSeed",
    "Lfm2CpcfTerminalActionSplitMetrics",
    "TERMINAL_ACTION_ALPHA_FRONTIER",
    "TERMINAL_ACTION_GAUSS_NEWTON_DAMPING_FRONTIER",
    "TERMINAL_ACTION_RIDGE_FRONTIER",
    "compile_lfm2_cpcf_terminal_action_residual",
    "damped_rmsnorm_gauss_newton_target",
    "rmsnorm_jvp",
]
