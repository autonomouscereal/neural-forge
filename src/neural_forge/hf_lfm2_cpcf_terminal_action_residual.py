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
* FIT, FIT-CHECK, and SELECTION provide diagnostic/ranking evidence without
  excluding a structurally safe state; VALIDATION evaluates only a frozen
  diagnostic lead and cannot select, accept, publish, or reject a candidate.

No model is executed or mutated.  There is no autograd graph, backward pass,
optimizer, trace capture, parent call, new parameter, pruning, or quantization.
The incumbent/noop is not a candidate.  Candidate eligibility is only finite
dense structural safety plus an effectful nonincumbent change.  Numerical proxy
thresholds remain authenticated diagnostics, and numerical metrics may
deterministically order structurally eligible writers for later whole-response
review.  The compiler performs no final selection, acceptance, or publication;
it retains the incumbent until manual review.  This rule applies to exact-zero
and nonzero incumbents.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal
from weakref import WeakKeyDictionary


LFM2_CPCF_TERMINAL_ACTION_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-evidence-v1"
)
LFM2_CPCF_TERMINAL_ACTION_SEED_SCHEMA = "neural-forge-lfm2-cpcf-terminal-action-seed-v1"
LFM2_CPCF_TERMINAL_ACTION_DOWN_WEIGHT_PRIOR_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-down-weight-prior-v1"
)
LFM2_CPCF_TERMINAL_ACTION_POLICY_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-policy-v1"
)
LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-result-v1"
)
LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-exact-zero-eligibility-v1"
)
LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-structural-nonveto-v1"
)
LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-review-pending-v1"
)
_LFM2_CPCF_TERMINAL_ACTION_COMPILER_BINDING_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-compiler-binding-v1"
)
_LFM2_CPCF_TERMINAL_ACTION_COMPILER_LINEAGE_SCHEMA = (
    "neural-forge-lfm2-cpcf-terminal-action-compiler-lineage-v1"
)
_TERMINAL_RESULT_COMPILER_ISSUER = object()
_TERMINAL_ACTION_RESULT_METHOD = (
    "closed_form_rmsnorm_gauss_newton_then_weighted_ridge_"
    "exact_bfloat16_terminal_writer_frontier"
)
_QDEIM_PRIOR_RECIPE_LANE = MappingProxyType(
    {
        "nonlinear_qdeim_structural": "nonlinear_qdeim_structural_priority",
        "nonlinear_qdeim_fit_corrected": (
            "nonlinear_qdeim_fit_corrected_priority"
        ),
        "nonlinear_qdeim_coupled_attention_structural": (
            "nonlinear_qdeim_structural_priority"
        ),
        "nonlinear_qdeim_coupled_attention_fit_corrected": (
            "nonlinear_qdeim_fit_corrected_priority"
        ),
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer": (
            "nonlinear_qdeim_structural_priority"
        ),
    }
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


def lfm2_cpcf_terminal_action_tensor_sha256(value: Any) -> str:
    """Public hash contract used to bind deferred W2 priors across adapters."""

    return _tensor_sha256(value, "terminal_action_tensor")


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
class Lfm2CpcfTerminalActionDownWeightPrior:
    """Selected, uninstalled final-MLP W2 carried into the terminal frontier."""

    down_weight: Any
    body_recipe_id: str
    body_recipe_sha256: str
    mlp_lane: str
    selected_role: str
    selected_candidate_sha256: str
    preparation_sha256: str
    mlp_installation_core_sha256: str
    live_deferred_w2_sha256: str
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        torch = _torch()
        down = _owned_float_tensor(
            self.down_weight,
            "down_weight_prior",
            dimensions=2,
        )
        if down.dtype not in {torch.bfloat16, torch.float32}:
            raise Lfm2CpcfTerminalActionResidualError(
                "down_weight_prior must use one supported BF16 or FP32 target dtype."
            )
        if (
            not isinstance(self.body_recipe_id, str)
            or _QDEIM_PRIOR_RECIPE_LANE.get(self.body_recipe_id) != self.mlp_lane
            or self.selected_role != "full-w1-w3-w2"
        ):
            raise Lfm2CpcfTerminalActionResidualError(
                "A terminal W2 prior must identify the exact selected nonlinear "
                "Q-DEIM recipe/lane pair."
            )
        for value, name in (
            (self.body_recipe_sha256, "body_recipe_sha256"),
            (self.selected_candidate_sha256, "selected_candidate_sha256"),
            (self.preparation_sha256, "preparation_sha256"),
            (
                self.mlp_installation_core_sha256,
                "mlp_installation_core_sha256",
            ),
            (self.live_deferred_w2_sha256, "live_deferred_w2_sha256"),
        ):
            _required_sha256(value, name)
        object.__setattr__(self, "down_weight", down)
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal W2 prior authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_TERMINAL_ACTION_DOWN_WEIGHT_PRIOR_SCHEMA,
            "body_recipe_id": self.body_recipe_id,
            "body_recipe_sha256": self.body_recipe_sha256,
            "mlp_lane": self.mlp_lane,
            "selected_role": self.selected_role,
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "preparation_sha256": self.preparation_sha256,
            "mlp_installation_core_sha256": (
                self.mlp_installation_core_sha256
            ),
            "live_deferred_w2_sha256": self.live_deferred_w2_sha256,
            "down_weight_sha256": _tensor_sha256(
                self.down_weight,
                "down_weight_prior",
            ),
            "shape": [int(value) for value in self.down_weight.shape],
            "dtype": str(self.down_weight.dtype).removeprefix("torch."),
            "installed_before_terminal": False,
            "installation_count_before_terminal": 0,
            "terminal_frontier_role": "authenticated-prior-center",
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
    """Authenticated fixed frontier and diagnostic numerical thresholds."""

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
            "validation_can_only_gate_frozen_selection": False,
            "all_numeric_thresholds_are_diagnostic_only": True,
            "structural_safety_is_the_only_candidate_veto": True,
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
    """Seal complete dense W2 coverage and exact private storage safety.

    A zero writer has no meaningful incumbent-relative scale.  Cold-start
    publication uses an exact structural predicate: every BF16
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
    exact_storage_bytes = int(candidate.nelement() * candidate.element_size())
    owns_exact_storage = bool(
        candidate.device.type == "cpu"
        and candidate.is_contiguous()
        and int(candidate.storage_offset()) == 0
        and int(candidate.untyped_storage().nbytes()) == exact_storage_bytes
        and candidate.requires_grad is False
        and candidate.grad_fn is None
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
        "contiguous_cpu_storage": bool(
            candidate.device.type == "cpu" and candidate.is_contiguous()
        ),
        "storage_offset": int(candidate.storage_offset()),
        "storage_bytes": int(candidate.untyped_storage().nbytes()),
        "expected_storage_bytes": exact_storage_bytes,
        "detached_without_grad": bool(
            candidate.requires_grad is False and candidate.grad_fn is None
        ),
        "owns_exact_storage_allocation": owns_exact_storage,
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
            and owns_exact_storage
        ),
        "complete_dense_writer_structural_safety": bool(
            finite_count == int(value.numel())
            and nonzero_count > 0
            and covered_outputs == output_width
            and covered_sources == source_feature_width
            and owns_exact_storage
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
    down_weight_prior: Lfm2CpcfTerminalActionDownWeightPrior | None = None,
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
    if down_weight_prior is not None:
        body.update(
            {
                "down_weight_prior_authentication_sha256": (
                    down_weight_prior.authentication_sha256
                ),
                "down_weight_prior_tensor_sha256": _tensor_sha256(
                    down_weight_prior.down_weight,
                    "compiler_binding.down_weight_prior",
                ),
                "down_weight_prior_live_incumbent_sha256": (
                    down_weight_prior.live_deferred_w2_sha256
                ),
            }
        )
    return MappingProxyType({**body, "binding_sha256": _canonical_sha256(body)})


def _terminal_compiler_lineage_facts(
    value: Mapping[str, Any],
    *,
    context: str,
) -> tuple[tuple[str, Any], ...]:
    """Extract a deeply immutable compiler-lineage fact tuple."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfTerminalActionAuthenticationError(f"{context} must be a mapping.")

    def split_sha256_facts(name: str) -> tuple[tuple[str, str], ...]:
        raw = value.get(name)
        if not isinstance(raw, Mapping) or set(raw) != set(_SPLITS):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"{context}.{name} must contain exactly the terminal splits."
            )
        return tuple(
            (
                split,
                _required_sha256(
                    raw.get(split),
                    f"{context}.{name}.{split}",
                ),
            )
            for split in _SPLITS
        )

    raw_rows = value.get("split_rows")
    if not isinstance(raw_rows, Mapping) or set(raw_rows) != set(_SPLITS):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            f"{context}.split_rows must contain exactly the terminal splits."
        )
    row_facts = []
    for split in _SPLITS:
        rows = raw_rows.get(split)
        if type(rows) is not int or rows <= 0:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"{context}.split_rows.{split} must be a positive integer."
            )
        row_facts.append((split, rows))

    facts: tuple[tuple[str, Any], ...] = (
        (
            "seed_sha256",
            _required_sha256(value.get("seed_sha256"), f"{context}.seed_sha256"),
        ),
        (
            "incumbent_seed_writer_sha256",
            _required_sha256(
                value.get("incumbent_seed_writer_sha256"),
                f"{context}.incumbent_seed_writer_sha256",
            ),
        ),
        (
            "policy_sha256",
            _required_sha256(
                value.get("policy_sha256"),
                f"{context}.policy_sha256",
            ),
        ),
        ("split_evidence_sha256", split_sha256_facts("split_evidence_sha256")),
        ("split_rows", tuple(row_facts)),
        ("split_row_ids_sha256", split_sha256_facts("split_row_ids_sha256")),
    )
    prior_names = (
        "down_weight_prior_authentication_sha256",
        "down_weight_prior_tensor_sha256",
        "down_weight_prior_live_incumbent_sha256",
    )
    present_prior_names = tuple(name for name in prior_names if name in value)
    if present_prior_names and present_prior_names != prior_names:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            f"{context} has a partial terminal down-weight prior binding."
        )
    if present_prior_names:
        facts = facts + tuple(
            (
                name,
                _required_sha256(value.get(name), f"{context}.{name}"),
            )
            for name in prior_names
        )
    return facts


def _terminal_compiler_lineage_sha256(
    facts: tuple[tuple[str, Any], ...],
) -> str:
    return _canonical_sha256(
        {
            "schema": _LFM2_CPCF_TERMINAL_ACTION_COMPILER_LINEAGE_SCHEMA,
            "facts": facts,
        }
    )


def _deeply_immutable_lineage_value(value: Any) -> bool:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return True
    return isinstance(value, tuple) and all(
        _deeply_immutable_lineage_value(item) for item in value
    )


class _TerminalCompilerLineageProof:
    """Identity-only proof whose authentication state lives in a private registry."""

    __slots__ = ("__weakref__",)

    def __new__(cls):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal compiler lineage proofs are issued only by a real compile."
        )

    def __copy__(self):
        return object.__new__(type(self))

    def __deepcopy__(self, memo):
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        return copied


@dataclass(frozen=True, slots=True)
class _TerminalCompilerLineageRegistration:
    result_sha256: str
    binding_sha256: str
    frozen_input_facts: tuple[tuple[str, Any], ...]


def _build_terminal_compiler_lineage_registry():
    registry: WeakKeyDictionary[
        _TerminalCompilerLineageProof,
        _TerminalCompilerLineageRegistration,
    ] = WeakKeyDictionary()
    registry_lock = RLock()

    def issue(
        compiler_input_binding: Mapping[str, Any],
        *,
        result_sha256: str,
    ) -> _TerminalCompilerLineageProof:
        binding = dict(compiler_input_binding)
        binding_sha256 = binding.pop("binding_sha256", None)
        if (
            binding.get("schema") != _LFM2_CPCF_TERMINAL_ACTION_COMPILER_BINDING_SCHEMA
            or binding_sha256 != _canonical_sha256(binding)
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal compiler cannot register an invalid input binding."
            )
        facts = _terminal_compiler_lineage_facts(
            compiler_input_binding,
            context="compiler_input_binding",
        )
        if not _deeply_immutable_lineage_value(facts):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal compiler lineage facts are not deeply immutable."
            )
        proof = object.__new__(_TerminalCompilerLineageProof)
        registration = _TerminalCompilerLineageRegistration(
            result_sha256=_required_sha256(
                result_sha256,
                "compiler_lineage.result_sha256",
            ),
            binding_sha256=_required_sha256(
                binding_sha256,
                "compiler_lineage.binding_sha256",
            ),
            frozen_input_facts=facts,
        )
        with registry_lock:
            registry[proof] = registration
        return proof

    def authenticate(
        *,
        proof: Any,
        compiler_input_binding: Mapping[str, Any] | None,
        report_body: Mapping[str, Any],
        result_sha256: str,
    ) -> _TerminalCompilerLineageProof:
        if type(proof) is not _TerminalCompilerLineageProof:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal result omitted its private compiler-lineage proof."
            )
        with registry_lock:
            registration = registry.get(proof)
        if registration is None:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal result used an unregistered compiler-lineage proof."
            )
        if not isinstance(compiler_input_binding, Mapping):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal result omitted its compiler input binding."
            )
        binding = dict(compiler_input_binding)
        binding_sha256 = binding.pop("binding_sha256", None)
        if (
            binding.get("schema") != _LFM2_CPCF_TERMINAL_ACTION_COMPILER_BINDING_SCHEMA
            or binding_sha256 != _canonical_sha256(binding)
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal report or input binding diverges from compiler-issued lineage."
            )
        binding_facts = _terminal_compiler_lineage_facts(
            compiler_input_binding,
            context="compiler_input_binding",
        )
        report_facts = _terminal_compiler_lineage_facts(
            report_body,
            context="terminal_report",
        )
        observed = _TerminalCompilerLineageRegistration(
            result_sha256=_required_sha256(
                result_sha256,
                "terminal_result.result_sha256",
            ),
            binding_sha256=_required_sha256(
                binding_sha256,
                "terminal_result.binding_sha256",
            ),
            frozen_input_facts=binding_facts,
        )
        if (
            observed != registration
            or report_facts != registration.frozen_input_facts
            or report_body.get("compiler_lineage_sha256")
            != _terminal_compiler_lineage_sha256(registration.frozen_input_facts)
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Terminal report or input binding diverges from compiler-issued lineage."
            )
        return proof

    def size() -> int:
        with registry_lock:
            return len(registry)

    return issue, authenticate, size


(
    _issue_terminal_compiler_lineage_proof,
    _authenticate_terminal_compiler_lineage,
    _terminal_compiler_lineage_registry_size,
) = _build_terminal_compiler_lineage_registry()
del _build_terminal_compiler_lineage_registry


def _reported_frontier_selection_key(
    record: Mapping[str, Any],
    *,
    ranking_split: str = "fit",
) -> list[Any]:
    """Reconstruct one reported ranking key instead of trusting its copy."""

    if ranking_split not in {"fit", "selection"}:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal frontier ranking split is unsupported."
        )
    metrics = record.get(f"{ranking_split}_metrics")
    oracle = metrics.get("oracle") if isinstance(metrics, Mapping) else None
    mapped_parent = (
        metrics.get("mapped_parent") if isinstance(metrics, Mapping) else None
    )
    if not isinstance(oracle, Mapping) or not isinstance(mapped_parent, Mapping):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            f"Terminal frontier omitted authenticated {ranking_split} metrics."
        )

    def number(container: Mapping[str, Any], name: str) -> float:
        value = container.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Terminal frontier metric {name!r} is not numeric."
            )
        result = float(value)
        if not math.isfinite(result):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Terminal frontier metric {name!r} is nonfinite."
            )
        return result

    state_sha256 = _required_sha256(
        record.get("state_sha256"),
        "terminal frontier state_sha256",
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


def _authenticate_terminal_validation_diagnostic_contract(
    validation: Any,
    *,
    diagnostic_lead_candidate_id: str,
    diagnostic_lead_state_sha256: str,
) -> Mapping[str, Any]:
    """Require the exact validation schema and its diagnostic-only role."""

    required_fields = {
        "evaluated_candidate_id",
        "candidate_count",
        "metrics",
        "gates",
        "passed",
        "used_for_fit",
        "used_for_fit_check_veto",
        "used_for_diagnostic_ordering",
        "used_for_final_selection",
        "used_for_candidate_acceptance",
        "heldout_diagnostic_only",
        "publication_gate_applied",
        "numeric_thresholds_diagnostic_only",
        "accessed_only_after_diagnostic_lead_id_and_state_sha256_frozen",
        "frozen_diagnostic_lead_state_sha256",
    }
    if not isinstance(validation, Mapping) or set(validation) != required_fields:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal validation report does not have its exact diagnostic schema."
        )
    gates = validation.get("gates")
    metrics = validation.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(gates, Mapping)
        or not gates
        or any(type(value) is not bool for value in gates.values())
        or validation.get("evaluated_candidate_id") != diagnostic_lead_candidate_id
        or validation.get("candidate_count") != 1
        or validation.get("passed") is not _all_passed(gates)
        or validation.get("used_for_fit") is not False
        or validation.get("used_for_fit_check_veto") is not False
        or validation.get("used_for_diagnostic_ordering") is not False
        or validation.get("used_for_final_selection") is not False
        or validation.get("used_for_candidate_acceptance") is not False
        or validation.get("heldout_diagnostic_only") is not True
        or validation.get("publication_gate_applied") is not False
        or validation.get("numeric_thresholds_diagnostic_only") is not True
        or validation.get(
            "accessed_only_after_diagnostic_lead_id_and_state_sha256_frozen"
        )
        is not True
        or validation.get("frozen_diagnostic_lead_state_sha256")
        != diagnostic_lead_state_sha256
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal validation report is not diagnostic-only."
        )
    return validation


def _authenticate_deferred_structural_frontier_compiled_result(
    *,
    body: Mapping[str, Any],
    incumbent: Any,
    selected: Any,
    published: Any,
    policy: Lfm2CpcfTerminalActionPolicy,
    compiler_input_binding: Mapping[str, Any] | None,
    compiler_issued: bool,
) -> Mapping[str, Any]:
    """Authenticate a complete frontier with no pre-review final decision."""

    torch = _torch()
    if not compiler_issued or not isinstance(compiler_input_binding, Mapping):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "A deferred terminal frontier must be compiler-issued."
        )
    binding = dict(compiler_input_binding)
    binding_sha256 = binding.pop("binding_sha256", None)
    if (
        binding.get("schema") != _LFM2_CPCF_TERMINAL_ACTION_COMPILER_BINDING_SCHEMA
        or binding_sha256 != _canonical_sha256(binding)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal compiler-input binding is invalid."
        )
    output_hashes = {
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
        or output_hashes["incumbent"]
        != binding.get("incumbent_seed_writer_sha256")
        or body.get("split_evidence_sha256") != binding.get("split_evidence_sha256")
        or body.get("split_rows") != binding.get("split_rows")
        or body.get("split_row_ids_sha256") != binding.get("split_row_ids_sha256")
        or body.get("output_state_sha256") != output_hashes
        or body.get("operation_audit") != dict(_ZERO_OPERATION_AUDIT)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal result is not bound to its compiler inputs."
        )
    prior_binding_names = (
        "down_weight_prior_authentication_sha256",
        "down_weight_prior_tensor_sha256",
        "down_weight_prior_live_incumbent_sha256",
    )
    prior_bound = all(name in binding for name in prior_binding_names)
    prior_receipt_value = body.get("down_weight_prior")
    if prior_bound:
        if not isinstance(prior_receipt_value, Mapping):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Deferred terminal result omitted its authenticated W2 prior."
            )
        prior_receipt = dict(prior_receipt_value)
        prior_authentication_sha256 = prior_receipt.pop(
            "authentication_sha256",
            None,
        )
        if (
            prior_authentication_sha256 != _canonical_sha256(prior_receipt)
            or prior_receipt.get("schema")
            != LFM2_CPCF_TERMINAL_ACTION_DOWN_WEIGHT_PRIOR_SCHEMA
            or prior_receipt.get("down_weight_sha256")
            != binding.get("down_weight_prior_tensor_sha256")
            or prior_receipt.get("live_deferred_w2_sha256")
            != output_hashes["incumbent"]
            or body.get("down_weight_prior_authentication_sha256")
            != binding.get("down_weight_prior_authentication_sha256")
            or body.get("down_weight_prior_authentication_sha256")
            != prior_authentication_sha256
            or body.get("down_weight_prior_tensor_sha256")
            != binding.get("down_weight_prior_tensor_sha256")
            or body.get("down_weight_prior_live_incumbent_sha256")
            != binding.get("down_weight_prior_live_incumbent_sha256")
            or body.get("down_weight_prior_live_incumbent_sha256")
            != output_hashes["incumbent"]
            or body.get("frontier_centered_on_authenticated_down_weight_prior")
            is not True
            or body.get("live_seed_w2_retained_as_separate_incumbent_control")
            is not True
            or body.get("down_weight_prior_installed_before_terminal") is not False
            or body.get("down_weight_prior_installation_count_before_terminal") != 0
            or body.get("authenticated_pure_prior_frontier_candidate_count") != 1
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Deferred terminal W2 prior differs from its compiler binding."
            )
    elif prior_receipt_value is not None or any(
        name in body for name in prior_binding_names
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Legacy terminal result unexpectedly introduced a partial W2 prior."
        )
    if (
        body.get("selected_candidate_id") != "incumbent"
        or body.get("selected_nonincumbent") is not False
        or body.get("accepted") is not False
        or body.get("published_is_incumbent") is not True
        or body.get("rollback_performed") is not False
        or body.get("rollback_exact") is not True
        or body.get("final_selection_status")
        != "deferred-to-manual-whole-response-review"
        or body.get("review_pending_schema")
        != LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA
        or body.get("final_selected_candidate_id") is not None
        or body.get("final_selection_performed") is not False
        or body.get("automatic_candidate_selection_performed") is not False
        or body.get("automatic_candidate_acceptance_performed") is not False
        or body.get("candidate_published_before_manual_review") is not False
        or body.get("manual_whole_response_review_required") is not True
        or body.get("diagnostic_order_has_final_selection_authority") is not False
        or body.get("diagnostic_order_is_provisional") is not True
        or body.get("published_writer_is_incumbent_pending_manual_review") is not True
        or body.get("published_writer_is_structurally_safe_effectful_nonincumbent")
        is not False
        or not torch.equal(selected, incumbent)
        or not torch.equal(published, incumbent)
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal result assigned pre-review decision semantics."
        )

    expected_ids = (
        "incumbent",
        *_frontier_candidate_ids(
            policy,
            include_down_weight_prior=prior_bound,
        ),
    )
    frontier = body.get("frontier")
    if (
        not isinstance(frontier, list)
        or len(frontier) != len(expected_ids)
        or body.get("frontier_count") != len(expected_ids)
        or body.get("expected_frontier_count") != len(expected_ids)
        or not all(isinstance(value, Mapping) for value in frontier)
        or tuple(value.get("candidate_id") for value in frontier) != expected_ids
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal frontier identities are invalid."
        )
    records = {str(value["candidate_id"]): value for value in frontier}
    incumbent_record = records["incumbent"]
    ranking_split = "fit" if int(torch.count_nonzero(incumbent).item()) == 0 else "selection"
    incumbent_key = _reported_frontier_selection_key(
        incumbent_record,
        ranking_split=ranking_split,
    )
    if (
        incumbent_record.get("state_sha256") != output_hashes["incumbent"]
        or incumbent_record.get("diagnostic_ordering_key") != incumbent_key
        or incumbent_record.get("eligible") is not False
        or incumbent_record.get("structurally_eligible") is not False
        or incumbent_record.get("effectful_nonincumbent") is not False
        or incumbent_record.get("incumbent_control_not_a_candidate") is not True
        or incumbent_record.get("candidate_diagnostic_ordering_split")
        != ranking_split
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal incumbent control is invalid."
        )

    eligible: list[tuple[tuple[Any, ...], str, Mapping[str, Any]]] = []
    diagnostic_misses = 0
    expected_shape = [int(size) for size in incumbent.shape]
    for candidate_id in expected_ids[1:]:
        record = records[candidate_id]
        reported_key = _reported_frontier_selection_key(
            record,
            ranking_split=ranking_split,
        )
        coverage_value = record.get("structural_coverage")
        fit_gates = record.get("fit_gates")
        if (
            record.get("diagnostic_ordering_key") != reported_key
            or record.get("candidate_diagnostic_ordering_split") != ranking_split
            or not isinstance(coverage_value, Mapping)
            or not isinstance(fit_gates, Mapping)
            or not fit_gates
            or any(type(value) is not bool for value in fit_gates.values())
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Deferred frontier record {candidate_id!r} is incomplete."
            )
        if prior_bound and (
            record.get("candidate_centered_on_authenticated_down_weight_prior")
            is not True
            or record.get("down_weight_prior_authentication_sha256")
            != binding.get("down_weight_prior_authentication_sha256")
            or record.get("down_weight_prior_tensor_sha256")
            != binding.get("down_weight_prior_tensor_sha256")
            or record.get("live_incumbent_remains_separate_control") is not True
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Deferred frontier record {candidate_id!r} lost its W2 prior."
            )
        if candidate_id == "qdeim-prior-alpha-00" and (
            record.get("pure_authenticated_down_weight_prior") is not True
            or record.get("alpha") != 0.0
            or record.get("state_sha256")
            != binding.get("down_weight_prior_tensor_sha256")
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Deferred pure-prior candidate differs from the carried Q-DEIM W2."
            )
        coverage = dict(coverage_value)
        coverage_sha256 = coverage.pop("coverage_sha256", None)
        coverage_complete = bool(
            coverage_sha256 == _canonical_sha256(coverage)
            and record.get("structural_coverage_sha256") == coverage_sha256
            and coverage_value.get("tensor_sha256") == record.get("state_sha256")
            and coverage_value.get("shape") == expected_shape
            and coverage_value.get("dtype") == "bfloat16"
            and coverage_value.get("all_elements_finite") is True
            and coverage_value.get("owns_exact_storage_allocation") is True
            and coverage_value.get("complete_dense_writer_structural_safety") is True
        )
        effectful = record.get("state_sha256") != output_hashes["incumbent"]
        structurally_eligible = bool(coverage_complete and effectful)
        fit_passed = _all_passed(fit_gates)
        if ranking_split == "fit":
            diagnostic_miss = not fit_passed
        else:
            fit_check_gates = record.get("fit_check_gates")
            selection_gates = record.get("selection_gates")
            if (
                not isinstance(fit_check_gates, Mapping)
                or not isinstance(selection_gates, Mapping)
                or not fit_check_gates
                or not selection_gates
                or any(type(value) is not bool for value in fit_check_gates.values())
                or any(type(value) is not bool for value in selection_gates.values())
            ):
                raise Lfm2CpcfTerminalActionAuthenticationError(
                    f"Deferred frontier diagnostics for {candidate_id!r} are invalid."
                )
            diagnostic_miss = not (
                fit_passed
                and _all_passed(fit_check_gates)
                and _all_passed(selection_gates)
            )
        if (
            record.get("effectful_nonincumbent") is not effectful
            or record.get("structurally_eligible") is not structurally_eligible
            or record.get("structural_safety_passed") is not coverage_complete
            or record.get("weight_norm_gate_passed") is not coverage_complete
            or record.get("eligible") is not structurally_eligible
            or record.get("candidate_eligibility_basis")
            != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
            or record.get("numeric_diagnostic_gate_miss") is not diagnostic_miss
            or record.get("weight_delta_threshold_used_for_eligibility") is not False
            or record.get("fit_thresholds_used_for_eligibility") is not False
            or record.get("fit_check_thresholds_used_for_eligibility") is not False
            or record.get("selection_thresholds_used_for_eligibility") is not False
            or record.get("numeric_thresholds_used_for_publication") is not False
            or record.get("numeric_metrics_used_for_diagnostic_ordering") is not True
            or record.get("numeric_metrics_used_for_final_selection") is not False
            or record.get("numeric_metrics_have_final_intelligence_authority")
            is not False
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Deferred frontier eligibility for {candidate_id!r} is invalid."
            )
        if structurally_eligible:
            eligible.append((tuple(reported_key), candidate_id, record))
            diagnostic_misses += int(diagnostic_miss)
    if not eligible:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal frontier has no structurally eligible candidate."
        )
    ordered = sorted(eligible, key=lambda value: (value[0], value[1]))
    diagnostic_order = [candidate_id for _key, candidate_id, _record in ordered]
    _lead_key, lead_id, lead_record = ordered[0]
    lead_state_sha256 = _required_sha256(
        lead_record.get("state_sha256"),
        "diagnostic lead state_sha256",
    )
    lead_metrics = body.get("diagnostic_lead_metrics")
    if (
        body.get("diagnostic_order_candidate_ids") != diagnostic_order
        or body.get("diagnostic_order_candidate_count") != len(diagnostic_order)
        or body.get("diagnostic_lead_candidate_id") != lead_id
        or body.get("diagnostic_lead_state_sha256") != lead_state_sha256
        or body.get("diagnostic_lead_ordering_key")
        != lead_record.get("diagnostic_ordering_key")
        or body.get("diagnostic_lead_has_final_selection_authority") is not False
        or not isinstance(lead_metrics, Mapping)
        or any(
            lead_record.get(f"{split}_metrics") != lead_metrics.get(split)
            for split in ("fit", "fit_check", "selection")
        )
        or body.get("diagnostic_lead_writer_structural_coverage")
        != lead_record.get("structural_coverage")
        or body.get("diagnostic_lead_writer_structural_coverage_sha256")
        != lead_record.get("structural_coverage_sha256")
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal diagnostic order is not authenticated."
        )
    _authenticate_terminal_validation_diagnostic_contract(
        body.get("validation"),
        diagnostic_lead_candidate_id=lead_id,
        diagnostic_lead_state_sha256=lead_state_sha256,
    )
    split_roles = body.get("split_roles")
    if (
        body.get("structural_nonveto_schema")
        != LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        or body.get("candidate_eligibility_basis")
        != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        or body.get("structural_safety_is_only_candidate_veto") is not True
        or body.get("numeric_metrics_used_for_diagnostic_ordering") is not True
        or body.get("numeric_metrics_used_for_final_selection") is not False
        or body.get("numeric_metrics_used_for_candidate_eligibility") is not False
        or body.get("numeric_thresholds_used_for_candidate_eligibility") is not False
        or body.get("numeric_thresholds_used_for_publication") is not False
        or body.get("numeric_metrics_have_final_intelligence_authority") is not False
        or body.get("behavioral_intelligence_review_outside_terminal_compiler")
        is not True
        or body.get("structurally_eligible_frontier_count") != len(eligible)
        or body.get("structurally_eligible_numeric_diagnostic_miss_count")
        != diagnostic_misses
        or body.get("numeric_diagnostic_miss_can_exclude_structural_candidate")
        is not False
        or not isinstance(split_roles, Mapping)
        or split_roles.get("candidate_eligibility_uses_only_structural_safety")
        is not True
        or split_roles.get("numeric_metrics_used_for_diagnostic_ordering") is not True
        or split_roles.get("numeric_metrics_used_for_final_selection") is not False
        or split_roles.get("manual_whole_response_review_required") is not True
        or split_roles.get("validation_diagnostic_only") is not True
        or split_roles.get("validation_publication_gate_applied") is not False
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Deferred terminal decision contract is invalid."
        )
    return MappingProxyType({**binding, "binding_sha256": binding_sha256})


def _frontier_candidate_ids(
    policy: Lfm2CpcfTerminalActionPolicy,
    *,
    include_down_weight_prior: bool = False,
) -> tuple[str, ...]:
    generated = tuple(
        f"gn-{damping_index:02d}-ridge-{ridge_index:02d}-alpha-{alpha_index:02d}"
        for damping_index, _damping in enumerate(policy.gauss_newton_damping_frontier)
        for ridge_index, _ridge in enumerate(policy.ridge_frontier)
        for alpha_index, _alpha in enumerate(policy.alpha_frontier[1:], start=1)
    )
    return (
        ("qdeim-prior-alpha-00", *generated)
        if include_down_weight_prior
        else generated
    )


def _authenticate_frontier_down_weights(
    value: Any,
    *,
    policy: Lfm2CpcfTerminalActionPolicy,
    report_body: Mapping[str, Any],
    expected_shape: tuple[int, ...],
    selected_candidate_id: str,
    incumbent_down_weight: Any,
    selected_down_weight: Any,
    published_down_weight: Any,
    own_tensors: bool,
) -> Mapping[str, Any]:
    """Authenticate the complete private, nonincumbent BF16 frontier bank."""

    torch = _torch()
    expected_ids = _frontier_candidate_ids(
        policy,
        include_down_weight_prior=isinstance(
            report_body.get("down_weight_prior"),
            Mapping,
        ),
    )
    expected_id_set = set(expected_ids)
    if not isinstance(value, Mapping) or set(value) != expected_id_set:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal frontier writer bank must contain exactly the "
            "nonincumbent candidate IDs derived from its authenticated policy."
        )
    raw_frontier = report_body.get("frontier")
    if (
        not isinstance(raw_frontier, list)
        or report_body.get("frontier_count") != len(raw_frontier)
        or report_body.get("expected_frontier_count") != len(expected_ids) + 1
        or len(raw_frontier) != len(expected_ids) + 1
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal report frontier cardinality is invalid."
        )
    report_ids = [
        record.get("candidate_id") if isinstance(record, Mapping) else None
        for record in raw_frontier
    ]
    if report_ids != ["incumbent", *expected_ids]:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal report frontier order differs from the fixed policy frontier."
        )
    records = {
        str(record["candidate_id"]): record
        for record in raw_frontier[1:]
        if isinstance(record, Mapping)
    }
    if set(records) != expected_id_set:
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal report does not cover the exact private frontier bank."
        )
    selected_state_sha256 = _tensor_sha256(
        selected_down_weight,
        "selected_down_weight",
    )
    selected_record = (
        raw_frontier[0]
        if selected_candidate_id == "incumbent"
        else (records.get(selected_candidate_id))
    )
    if (
        not isinstance(selected_record, Mapping)
        or selected_record.get("state_sha256") != selected_state_sha256
        or report_body.get("selected_state_sha256") != selected_state_sha256
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Selected terminal writer hash differs from its selected frontier record."
        )
    authenticated: dict[str, Any] = {}
    storage_pointers: set[int] = {
        int(tensor.untyped_storage().data_ptr())
        for tensor in (
            incumbent_down_weight,
            selected_down_weight,
            published_down_weight,
        )
    }
    for candidate_id in expected_ids:
        source = value[candidate_id]
        if own_tensors:
            tensor = _owned_float_tensor(
                source,
                f"frontier_down_weights[{candidate_id!r}]",
                dimensions=2,
                exact_dtype=torch.bfloat16,
            )
        else:
            if (
                not isinstance(source, torch.Tensor)
                or source.layout != torch.strided
                or source.device.type != "cpu"
                or source.ndim != 2
                or source.numel() == 0
                or source.dtype != torch.bfloat16
                or source.requires_grad
                or source.grad_fn is not None
                or not source.is_contiguous()
                or not bool(torch.isfinite(source).all().item())
            ):
                raise Lfm2CpcfTerminalActionAuthenticationError(
                    f"Private terminal candidate {candidate_id!r} is no longer "
                    "a finite detached contiguous CPU BF16 writer."
                )
            tensor = source
        if tuple(tensor.shape) != expected_shape:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Private terminal candidate {candidate_id!r} has the wrong shape."
            )
        storage = tensor.untyped_storage()
        storage_pointer = int(storage.data_ptr())
        expected_storage_bytes = int(tensor.nelement() * tensor.element_size())
        if (
            int(tensor.storage_offset()) != 0
            or int(storage.nbytes()) != expected_storage_bytes
            or storage_pointer in storage_pointers
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Private terminal frontier candidates must each own one exact, "
                "unaliased storage allocation."
            )
        storage_pointers.add(storage_pointer)
        record = records[candidate_id]
        if (
            record.get("private_candidate_storage") is not True
            or record.get("exact_bfloat16_materialization") is not True
            or record.get("state_sha256")
            != _tensor_sha256(tensor, f"frontier_down_weights[{candidate_id!r}]")
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                f"Private terminal candidate {candidate_id!r} differs from its "
                "authenticated frontier record."
            )
        authenticated[candidate_id] = tensor
    if selected_candidate_id == "incumbent":
        if not torch.equal(selected_down_weight, incumbent_down_weight):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "An incumbent terminal selection must retain the exact incumbent "
                "writer as its selected state."
            )
    else:
        selected = authenticated.get(selected_candidate_id)
        if selected is None or not torch.equal(selected, selected_down_weight):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Selected terminal writer differs from its private frontier state."
            )
    return MappingProxyType(authenticated)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTerminalActionResidualResult:
    """Authenticated private frontier with a review-pending incumbent state."""

    policy: Lfm2CpcfTerminalActionPolicy
    incumbent_down_weight: Any
    selected_down_weight: Any
    published_down_weight: Any
    selected_candidate_id: str
    accepted: bool
    report: Mapping[str, Any]
    result_sha256: str
    frontier_down_weights: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _compiler_input_binding: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _compiler_issuer: InitVar[Any] = None
    _compiler_lineage_receipt: InitVar[Any] = None
    _compiler_issued: bool = field(
        init=False,
        default=False,
        repr=False,
        compare=False,
    )
    _compiler_lineage_token: _TerminalCompilerLineageProof | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
        _compiler_issuer: Any,
        _compiler_lineage_receipt: Any,
    ) -> None:
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
        if "frontier" in body:
            frontier_down_weights = _authenticate_frontier_down_weights(
                self.frontier_down_weights,
                policy=self.policy,
                report_body=body,
                expected_shape=tuple(owned[0].shape),
                selected_candidate_id=self.selected_candidate_id,
                incumbent_down_weight=owned[0],
                selected_down_weight=owned[1],
                published_down_weight=owned[2],
                own_tensors=True,
            )
        elif compiler_issued or (
            self.frontier_down_weights is not None
            and len(self.frontier_down_weights) != 0
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "A compiler-issued or populated terminal result cannot omit its "
                "authenticated frontier report."
            )
        else:
            # Backwards-compatible support for narrow hand-built test doubles.
            # Every compiler-issued result takes the strict branch above.
            frontier_down_weights = MappingProxyType({})
        compiler_binding = self._compiler_input_binding
        structural_nonveto = bool(
            body.get("structural_nonveto_schema")
            == LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        )
        requires_compiler_authentication = bool(
            compiler_issued
            or self.accepted
            or "frontier" in body
            or bool(frontier_down_weights)
            or body.get("method") == _TERMINAL_ACTION_RESULT_METHOD
        )
        if requires_compiler_authentication and not compiler_issued:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Every accepted or frontier-bearing terminal result must be "
                "compiler-issued."
            )
        if (
            requires_compiler_authentication
            and body.get("method") != _TERMINAL_ACTION_RESULT_METHOD
        ):
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "A compiler-issued terminal result cannot relabel its method."
            )
        if requires_compiler_authentication and not structural_nonveto:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "Compiler terminal result omitted its structural-nonveto schema."
            )
        compiler_lineage_token = None
        if requires_compiler_authentication:
            compiler_lineage_token = _authenticate_terminal_compiler_lineage(
                proof=_compiler_lineage_receipt,
                compiler_input_binding=compiler_binding,
                report_body=body,
                result_sha256=result_sha,
            )
        elif _compiler_lineage_receipt is not None:
            raise Lfm2CpcfTerminalActionAuthenticationError(
                "A legacy terminal test double cannot carry compiler lineage."
            )
        if structural_nonveto:
            compiler_binding = _authenticate_deferred_structural_frontier_compiled_result(
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
        object.__setattr__(self, "frontier_down_weights", frontier_down_weights)
        object.__setattr__(self, "_compiler_input_binding", compiler_binding)
        object.__setattr__(self, "_compiler_issued", compiler_issued)
        object.__setattr__(self, "_compiler_lineage_token", compiler_lineage_token)
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
        structural_nonveto = bool(
            body.get("structural_nonveto_schema")
            == LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        )
        requires_compiler_authentication = bool(
            self._compiler_issued
            or self.accepted
            or "frontier" in body
            or bool(self.frontier_down_weights)
            or body.get("method") == _TERMINAL_ACTION_RESULT_METHOD
        )
        if requires_compiler_authentication and (
            not self._compiler_issued
            or not isinstance(self._compiler_input_binding, Mapping)
            or body.get("method") != _TERMINAL_ACTION_RESULT_METHOD
            or not structural_nonveto
        ):
            return False
        if requires_compiler_authentication:
            try:
                _authenticate_terminal_compiler_lineage(
                    proof=self._compiler_lineage_token,
                    compiler_input_binding=self._compiler_input_binding,
                    report_body=body,
                    result_sha256=self.result_sha256,
                )
            except Lfm2CpcfTerminalActionResidualError:
                return False
        elif self._compiler_lineage_token is not None:
            return False
        if "frontier" in body:
            try:
                _authenticate_frontier_down_weights(
                    self.frontier_down_weights,
                    policy=self.policy,
                    report_body=body,
                    expected_shape=tuple(self.incumbent_down_weight.shape),
                    selected_candidate_id=self.selected_candidate_id,
                    incumbent_down_weight=self.incumbent_down_weight,
                    selected_down_weight=self.selected_down_weight,
                    published_down_weight=self.published_down_weight,
                    own_tensors=False,
                )
            except Lfm2CpcfTerminalActionResidualError:
                return False
        elif self.frontier_down_weights:
            return False
        if structural_nonveto:
            try:
                _authenticate_deferred_structural_frontier_compiled_result(
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
    down_weight_prior: Lfm2CpcfTerminalActionDownWeightPrior | None = None,
) -> Lfm2CpcfTerminalActionResidualResult:
    """Compile a private exact-BF16 frontier without making a final selection."""

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
    if down_weight_prior is not None and (
        not isinstance(
            down_weight_prior,
            Lfm2CpcfTerminalActionDownWeightPrior,
        )
        or not down_weight_prior.verify_authentication()
        or tuple(down_weight_prior.down_weight.shape)
        != tuple(seed.down_weight.shape)
        or down_weight_prior.down_weight.dtype != seed.down_weight.dtype
        or down_weight_prior.live_deferred_w2_sha256
        != _tensor_sha256(seed.down_weight, "seed.down_weight")
    ):
        raise Lfm2CpcfTerminalActionAuthenticationError(
            "Terminal down-weight prior is unauthenticated or differs from the "
            "live deferred-W2 lineage."
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
    frontier_prior = (
        incumbent.detach().clone()
        if down_weight_prior is None
        else down_weight_prior.down_weight.detach().clone()
    )
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
        down_weight_prior=down_weight_prior,
    )
    compiler_lineage_facts = _terminal_compiler_lineage_facts(
        compiler_input_binding,
        context="compiler_input_binding",
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
        "eligible": False,
        "diagnostic_ordering_key": _selection_key_report(incumbent_key),
        "candidate_diagnostic_ordering_split": (
            "fit" if exact_zero_cold_start else "selection"
        ),
        "effectful_nonincumbent": False,
        "structurally_eligible": False,
        "incumbent_control_not_a_candidate": True,
    }
    if exact_zero_cold_start:
        incumbent_frontier.update(
            {
                "weight_norm_gate_passed": False,
                "fit_gates": {"exact_zero_control_not_a_candidate": True},
                "fit_check_gates": {
                    "not_evaluated_before_diagnostic_lead_freeze": True
                },
                "selection_gates": {
                    "not_evaluated_before_diagnostic_lead_freeze": True
                },
                "eligible": False,
                "exact_zero_incumbent_control_only": True,
            }
        )
    frontier: list[dict[str, Any]] = [incumbent_frontier]
    frontier_down_weights: dict[str, Any] = {}
    selected_id = "incumbent"
    selected_weight = incumbent.detach().clone()
    diagnostic_order_inputs: list[tuple[Any, str, dict[str, float], dict[str, Any]]] = []
    frontier_prior_post_layer = _candidate_post_layer(
        prepared["fit"],
        frontier_prior,
    )
    frontier_prior_final_norm_output = _rmsnorm(
        frontier_prior_post_layer,
        seed.final_norm_weight,
        seed.norm_epsilon,
    )
    target_residual = (
        prepared["fit"].oracle_hidden - frontier_prior_final_norm_output
    )
    fit_target_reports = []

    if down_weight_prior is not None:
        prior_candidate_id = "qdeim-prior-alpha-00"
        prior_candidate = frontier_prior.detach().contiguous().clone()
        prior_state_sha = _tensor_sha256(prior_candidate, prior_candidate_id)
        prior_weight_ratio = _weight_delta_rms_ratio(
            prior_candidate,
            incumbent,
            selected_policy.weight_reference_floor,
        )
        prior_structural_coverage = _terminal_writer_structural_coverage(
            prior_candidate
        )
        prior_weight_gate = bool(
            prior_structural_coverage.get(
                "complete_dense_writer_structural_safety"
            )
            is True
        )
        prior_metrics = {
            split: _evaluate_split(
                prepared[split],
                seed,
                prior_candidate,
                metric_chunk_rows=selected_policy.metric_chunk_rows,
            )
            for split in frontier_metric_splits
        }
        prior_fit_gates = _metric_gates(
            prior_metrics["fit"],
            incumbent_metrics["fit"],
            selected_policy,
            require_selection_improvement=False,
        )
        if exact_zero_cold_start:
            prior_fit_check_gates = {
                "not_evaluated_before_diagnostic_lead_freeze": True
            }
            prior_selection_gates = {
                "not_evaluated_before_diagnostic_lead_freeze": True
            }
        else:
            prior_fit_check_gates = _metric_gates(
                prior_metrics["fit_check"],
                incumbent_metrics["fit_check"],
                selected_policy,
                require_selection_improvement=False,
            )
            prior_selection_gates = _metric_gates(
                prior_metrics["selection"],
                incumbent_metrics["selection"],
                selected_policy,
                require_selection_improvement=True,
            )
        prior_key = _selection_key(
            prior_metrics["fit" if exact_zero_cold_start else "selection"],
            alpha=0.0,
            damping=selected_policy.gauss_newton_damping_frontier[0],
            ridge=selected_policy.ridge_frontier[0],
            state_sha256=prior_state_sha,
        )
        prior_effectful = not torch.equal(prior_candidate, incumbent)
        prior_fit_passed = _all_passed(prior_fit_gates)
        prior_fit_check_passed = _all_passed(prior_fit_check_gates)
        prior_selection_passed = _all_passed(prior_selection_gates)
        prior_structurally_eligible = bool(prior_weight_gate and prior_effectful)
        prior_numeric_miss = not bool(
            prior_fit_passed
            and (
                True
                if exact_zero_cold_start
                else prior_fit_check_passed and prior_selection_passed
            )
        )
        prior_record = {
            "candidate_id": prior_candidate_id,
            "gauss_newton_damping": (
                selected_policy.gauss_newton_damping_frontier[0]
            ),
            "ridge": selected_policy.ridge_frontier[0],
            "alpha": 0.0,
            "exact_bfloat16_materialization": True,
            "private_candidate_storage": True,
            "state_sha256": prior_state_sha,
            "weight_delta_rms_ratio": prior_weight_ratio,
            "weight_norm_gate_passed": prior_weight_gate,
            "fit_metrics": prior_metrics["fit"].as_dict(),
            "fit_check_metrics": (
                None
                if exact_zero_cold_start
                else prior_metrics["fit_check"].as_dict()
            ),
            "selection_metrics": (
                None
                if exact_zero_cold_start
                else prior_metrics["selection"].as_dict()
            ),
            "fit_gates": prior_fit_gates,
            "fit_check_gates": prior_fit_check_gates,
            "selection_gates": prior_selection_gates,
            "eligible": prior_structurally_eligible,
            "diagnostic_ordering_key": _selection_key_report(prior_key),
            "candidate_diagnostic_ordering_split": (
                "fit" if exact_zero_cold_start else "selection"
            ),
            "effectful_nonincumbent": prior_effectful,
            "structurally_eligible": prior_structurally_eligible,
            "structural_safety_passed": prior_weight_gate,
            "candidate_eligibility_basis": (
                "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
            ),
            "fit_metric_gates_passed": prior_fit_passed,
            "fit_check_metric_gates_passed": prior_fit_check_passed,
            "selection_metric_gates_passed": prior_selection_passed,
            "numeric_diagnostic_gate_miss": prior_numeric_miss,
            "legacy_weight_delta_threshold_would_pass": (
                prior_weight_ratio
                <= selected_policy.maximum_weight_delta_rms_ratio
            ),
            "weight_delta_threshold_used_for_eligibility": False,
            "fit_thresholds_used_for_eligibility": False,
            "fit_check_thresholds_used_for_eligibility": False,
            "selection_thresholds_used_for_eligibility": False,
            "numeric_thresholds_used_for_publication": False,
            "numeric_metrics_used_for_diagnostic_ordering": True,
            "numeric_metrics_used_for_final_selection": False,
            "numeric_metrics_have_final_intelligence_authority": False,
            "structural_coverage": dict(prior_structural_coverage),
            "structural_coverage_sha256": prior_structural_coverage[
                "coverage_sha256"
            ],
            "candidate_centered_on_authenticated_down_weight_prior": True,
            "pure_authenticated_down_weight_prior": True,
            "down_weight_prior_authentication_sha256": (
                down_weight_prior.authentication_sha256
            ),
            "down_weight_prior_tensor_sha256": prior_state_sha,
            "live_incumbent_remains_separate_control": True,
        }
        if exact_zero_cold_start:
            prior_record.update(
                {
                    "fit_check_and_selection_metrics_diagnostic_only": True,
                    "relative_to_zero_weight_delta_gate_applied": False,
                    "legacy_relative_weight_delta_gate_would_pass": (
                        prior_weight_ratio
                        <= selected_policy.maximum_weight_delta_rms_ratio
                    ),
                    "fit_metric_thresholds_used_for_candidate_eligibility": False,
                    "fit_metrics_used_for_diagnostic_ordering": True,
                    "fit_metrics_used_for_pre_review_candidate_selection": False,
                    "fit_metrics_used_for_structural_eligibility": False,
                    "fit_metric_thresholds_used_for_publication": False,
                    "fit_metrics_have_final_intelligence_authority": False,
                }
            )
        frontier.append(prior_record)
        frontier_down_weights[prior_candidate_id] = prior_candidate.detach().clone()
        if prior_structurally_eligible:
            diagnostic_order_inputs.append(
                (
                    prior_key,
                    prior_candidate_id,
                    {
                        "gauss_newton_damping": (
                            selected_policy.gauss_newton_damping_frontier[0]
                        ),
                        "ridge": selected_policy.ridge_frontier[0],
                        "alpha": 0.0,
                    },
                    dict(prior_metrics),
                )
            )

    with torch.no_grad():
        for damping_index, damping in enumerate(
            selected_policy.gauss_newton_damping_frontier
        ):
            target_delta = damped_rmsnorm_gauss_newton_target(
                frontier_prior_post_layer,
                target_residual,
                seed.final_norm_weight,
                epsilon=seed.norm_epsilon,
                damping=damping,
            )
            jvp_prediction = rmsnorm_jvp(
                frontier_prior_post_layer,
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
                            frontier_prior.to(torch.float32)
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
                    structural_coverage = _terminal_writer_structural_coverage(
                        candidate
                    )
                    weight_gate = bool(
                        structural_coverage.get(
                            "complete_dense_writer_structural_safety"
                        )
                        is True
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
                            "not_evaluated_before_diagnostic_lead_freeze": True
                        }
                        selection_gates = {
                            "not_evaluated_before_diagnostic_lead_freeze": True
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
                    fit_check_metric_gates_passed = _all_passed(fit_check_gates)
                    selection_metric_gates_passed = _all_passed(selection_gates)
                    structurally_eligible = bool(weight_gate and effectful_nonincumbent)
                    numeric_diagnostic_gate_miss = not bool(
                        fit_metric_gates_passed
                        and (
                            True
                            if exact_zero_cold_start
                            else fit_check_metric_gates_passed
                            and selection_metric_gates_passed
                        )
                    )
                    eligible = structurally_eligible
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
                        "diagnostic_ordering_key": _selection_key_report(key),
                        "candidate_diagnostic_ordering_split": (
                            "fit" if exact_zero_cold_start else "selection"
                        ),
                        "effectful_nonincumbent": effectful_nonincumbent,
                        "structurally_eligible": structurally_eligible,
                        "structural_safety_passed": weight_gate,
                        "candidate_eligibility_basis": (
                            "finite_dense_shape_storage_coverage_and_"
                            "nonincumbent_effect"
                        ),
                        "fit_metric_gates_passed": fit_metric_gates_passed,
                        "fit_check_metric_gates_passed": (
                            fit_check_metric_gates_passed
                        ),
                        "selection_metric_gates_passed": (
                            selection_metric_gates_passed
                        ),
                        "numeric_diagnostic_gate_miss": (numeric_diagnostic_gate_miss),
                        "legacy_weight_delta_threshold_would_pass": (
                            weight_ratio
                            <= selected_policy.maximum_weight_delta_rms_ratio
                        ),
                        "weight_delta_threshold_used_for_eligibility": False,
                        "fit_thresholds_used_for_eligibility": False,
                        "fit_check_thresholds_used_for_eligibility": False,
                        "selection_thresholds_used_for_eligibility": False,
                        "numeric_thresholds_used_for_publication": False,
                        "numeric_metrics_used_for_diagnostic_ordering": True,
                        "numeric_metrics_used_for_final_selection": False,
                        "numeric_metrics_have_final_intelligence_authority": False,
                        "structural_coverage": dict(structural_coverage),
                        "structural_coverage_sha256": structural_coverage[
                            "coverage_sha256"
                        ],
                    }
                    if down_weight_prior is not None:
                        candidate_report.update(
                            {
                                "candidate_centered_on_authenticated_down_weight_prior": (
                                    True
                                ),
                                "down_weight_prior_authentication_sha256": (
                                    down_weight_prior.authentication_sha256
                                ),
                                "down_weight_prior_tensor_sha256": _tensor_sha256(
                                    frontier_prior,
                                    "frontier_prior",
                                ),
                                "live_incumbent_remains_separate_control": True,
                            }
                        )
                    if exact_zero_cold_start:
                        candidate_report.update(
                            {
                                "candidate_diagnostic_ordering_split": "fit",
                                "fit_check_and_selection_metrics_diagnostic_only": (
                                    True
                                ),
                                "relative_to_zero_weight_delta_gate_applied": False,
                                "legacy_relative_weight_delta_gate_would_pass": (
                                    weight_ratio
                                    <= selected_policy.maximum_weight_delta_rms_ratio
                                ),
                                "effectful_nonincumbent": effectful_nonincumbent,
                                "structurally_eligible": structurally_eligible,
                                "candidate_eligibility_basis": (
                                    "finite_dense_shape_storage_coverage_and_"
                                    "nonincumbent_effect"
                                ),
                                "fit_metric_gates_passed": (fit_metric_gates_passed),
                                "fit_metric_thresholds_used_for_candidate_eligibility": (
                                    False
                                ),
                                "fit_metrics_used_for_diagnostic_ordering": True,
                                "fit_metrics_used_for_pre_review_candidate_selection": (
                                    False
                                ),
                                "fit_metrics_used_for_structural_eligibility": False,
                                "fit_metric_thresholds_used_for_publication": False,
                                "fit_metrics_have_final_intelligence_authority": False,
                            }
                        )
                    frontier.append(candidate_report)
                    frontier_down_weights[candidate_id] = candidate.detach().clone()
                    if eligible:
                        diagnostic_order_inputs.append(
                            (
                                key,
                                candidate_id,
                                {
                                    "gauss_newton_damping": damping,
                                    "ridge": ridge,
                                    "alpha": alpha,
                                },
                                dict(candidate_metrics),
                            )
                        )

    diagnostic_order_inputs.sort(key=lambda value: (value[0], value[1]))
    if not diagnostic_order_inputs:
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal frontier found no finite, fully dense effectful closed-form "
            "W2 candidate for whole-response review."
        )
    (
        diagnostic_lead_key,
        diagnostic_lead_id,
        diagnostic_lead_hyperparameters,
        diagnostic_lead_split_metrics,
    ) = diagnostic_order_inputs[0]
    diagnostic_lead_weight = frontier_down_weights[diagnostic_lead_id].detach().clone()
    diagnostic_lead_structural_coverage = _terminal_writer_structural_coverage(
        diagnostic_lead_weight
    )
    if diagnostic_lead_structural_coverage.get(
        "complete_dense_writer_structural_safety"
    ) is not True or torch.equal(diagnostic_lead_weight, incumbent):
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal frontier did not freeze a structurally safe diagnostic lead."
        )

    diagnostic_lead_before_heldout = diagnostic_lead_id
    diagnostic_lead_state_sha = _tensor_sha256(
        diagnostic_lead_weight,
        "diagnostic_lead_down_weight",
    )
    if exact_zero_cold_start:
        frozen_diagnostic_lead_metrics: dict[
            str, Lfm2CpcfTerminalActionSplitMetrics
        ] = {}
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
                frozen_diagnostic_lead_metrics[split] = _evaluate_split(
                    prepared[split],
                    seed,
                    diagnostic_lead_weight,
                    metric_chunk_rows=selected_policy.metric_chunk_rows,
                )
        except Exception as exc:
            raise Lfm2CpcfTerminalActionResidualError(
                "A held-out terminal diagnostic failed after the structurally "
                "eligible diagnostic lead was frozen; no result was "
                "issued."
            ) from exc
        diagnostic_lead_split_metrics.update(
            {
                split: frozen_diagnostic_lead_metrics[split]
                for split in ("fit_check", "selection")
            }
        )
        diagnostic_lead_frontier_record = next(
            value
            for value in frontier
            if value["candidate_id"] == diagnostic_lead_id
        )
        diagnostic_gates = {
            "fit_check": _metric_gates(
                frozen_diagnostic_lead_metrics["fit_check"],
                incumbent_metrics["fit_check"],
                selected_policy,
                require_selection_improvement=False,
            ),
            "selection": _metric_gates(
                frozen_diagnostic_lead_metrics["selection"],
                incumbent_metrics["selection"],
                selected_policy,
                require_selection_improvement=True,
            ),
        }
        incumbent_frontier.update(
            {
                "fit_check_metrics": incumbent_metrics["fit_check"].as_dict(),
                "selection_metrics": incumbent_metrics["selection"].as_dict(),
                "fit_check_gates": {
                    "diagnostic_only_after_diagnostic_lead_freeze": True
                },
                "selection_gates": {
                    "diagnostic_only_after_diagnostic_lead_freeze": True
                },
            }
        )
        diagnostic_lead_frontier_record.update(
            {
                "fit_check_metrics": frozen_diagnostic_lead_metrics[
                    "fit_check"
                ].as_dict(),
                "selection_metrics": frozen_diagnostic_lead_metrics[
                    "selection"
                ].as_dict(),
                "fit_check_gates": diagnostic_gates["fit_check"],
                "selection_gates": diagnostic_gates["selection"],
                "heldout_diagnostics_evaluated_after_diagnostic_lead_freeze": True,
                "frozen_diagnostic_lead_state_sha256": diagnostic_lead_state_sha,
            }
        )
        diagnostic_lead_validation_metrics = frozen_diagnostic_lead_metrics[
            "validation"
        ]
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
        diagnostic_lead_validation_metrics = _evaluate_split(
            prepared["validation"],
            seed,
            diagnostic_lead_weight,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
    validation_gates = _metric_gates(
        diagnostic_lead_validation_metrics,
        incumbent_metrics["validation"],
        selected_policy,
        require_selection_improvement=False,
    )
    validation_passed = _all_passed(validation_gates)
    selected_nonincumbent = False
    accepted = False
    published = incumbent.detach().clone()
    rollback_performed = False
    rollback_exact = True
    if _tensor_sha256(seed.down_weight, "seed.down_weight") != incumbent_input_sha:
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal compilation mutated the authenticated seed writer."
        )
    if not torch.equal(selected_weight, incumbent) or not torch.equal(
        published, incumbent
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "Deferred terminal compilation changed the incumbent before review."
        )

    expected_frontier_count = 1 + (
        len(selected_policy.gauss_newton_damping_frontier)
        * len(selected_policy.ridge_frontier)
        * (len(selected_policy.alpha_frontier) - 1)
    ) + int(down_weight_prior is not None)
    if (
        len(frontier) != expected_frontier_count
        or diagnostic_lead_id != diagnostic_lead_before_heldout
    ):
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal frontier cardinality or frozen diagnostic lead changed."
        )
    nonincumbent_candidates = tuple(
        value for value in frontier if value["candidate_id"] != "incumbent"
    )
    structurally_eligible_count = sum(
        int(value["structurally_eligible"] is True) for value in nonincumbent_candidates
    )
    structurally_eligible_numeric_diagnostic_miss_count = sum(
        int(
            value["structurally_eligible"] is True
            and value["numeric_diagnostic_gate_miss"] is True
        )
        for value in nonincumbent_candidates
    )
    cold_start_fit_metric_gate_miss_count = sum(
        int(
            value["structurally_eligible"] is True
            and value["fit_metric_gates_passed"] is False
        )
        for value in nonincumbent_candidates
    )
    if structurally_eligible_count <= 0:
        raise Lfm2CpcfTerminalActionResidualError(
            "Terminal frontier lost every structurally eligible candidate."
        )
    output_hashes = {
        "incumbent": _tensor_sha256(incumbent, "incumbent"),
        "selected": _tensor_sha256(selected_weight, "selected"),
        "published": _tensor_sha256(published, "published"),
    }
    validation_report = {
        "evaluated_candidate_id": diagnostic_lead_id,
        "candidate_count": 1,
        "metrics": diagnostic_lead_validation_metrics.as_dict(),
        "gates": validation_gates,
        "passed": validation_passed,
        "used_for_fit": False,
        "used_for_fit_check_veto": False,
        "used_for_diagnostic_ordering": False,
        "used_for_final_selection": False,
        "used_for_candidate_acceptance": False,
        "heldout_diagnostic_only": True,
        "publication_gate_applied": False,
        "numeric_thresholds_diagnostic_only": True,
        "accessed_only_after_diagnostic_lead_id_and_state_sha256_frozen": True,
        "frozen_diagnostic_lead_state_sha256": diagnostic_lead_state_sha,
    }
    split_roles = {
        "fit_used_for_gauss_newton_target": True,
        "fit_used_for_weighted_ridge": True,
        "fit_check_used_for_fit": False,
        "fit_check_used_for_candidate_veto": False,
        "fit_check_used_for_diagnostic_ordering": False,
        "fit_check_diagnostic_only": True,
        "selection_used_for_fit": False,
        "selection_used_for_diagnostic_ordering": not exact_zero_cold_start,
        "selection_thresholds_diagnostic_only": True,
        "validation_used_for_fit": False,
        "validation_used_for_diagnostic_ordering": False,
        "validation_diagnostic_only": True,
        "validation_publication_gate_applied": False,
        "candidate_eligibility_uses_only_structural_safety": True,
        "numeric_metrics_used_for_diagnostic_ordering": True,
        "numeric_metrics_used_for_final_selection": False,
        "manual_whole_response_review_required": True,
        "validation_candidate_count": 1,
    }
    if exact_zero_cold_start:
        split_roles.update(
            {
                "fit_check_used_for_candidate_veto": False,
                "fit_check_diagnostic_only": True,
                "selection_used_for_diagnostic_ordering": False,
                "selection_diagnostic_only": True,
                "validation_diagnostic_only": True,
                "candidate_eligibility_uses_only_finite_dense_structure": True,
                "fit_used_for_candidate_eligibility": False,
                "only_fit_rows_used_for_diagnostic_ordering": True,
                "fit_metrics_used_for_pre_review_candidate_selection": False,
                "fit_metrics_have_final_intelligence_authority": False,
                "fit_metric_thresholds_used_for_publication": False,
            }
        )
    body = {
        "schema": LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA,
        "method": _TERMINAL_ACTION_RESULT_METHOD,
        "policy_sha256": selected_policy.authentication_sha256,
        "seed_sha256": seed.authentication_sha256,
        "incumbent_seed_writer_sha256": incumbent_state_sha,
        "compiler_lineage_sha256": _terminal_compiler_lineage_sha256(
            compiler_lineage_facts
        ),
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
        "selected_state_sha256": output_hashes["selected"],
        "review_pending_schema": LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA,
        "final_selection_status": "deferred-to-manual-whole-response-review",
        "final_selected_candidate_id": None,
        "final_selection_performed": False,
        "automatic_candidate_selection_performed": False,
        "automatic_candidate_acceptance_performed": False,
        "candidate_published_before_manual_review": False,
        "manual_whole_response_review_required": True,
        "diagnostic_order_candidate_ids": [
            value[1] for value in diagnostic_order_inputs
        ],
        "diagnostic_order_candidate_count": len(diagnostic_order_inputs),
        "diagnostic_order_has_final_selection_authority": False,
        "diagnostic_order_is_provisional": True,
        "diagnostic_lead_candidate_id": diagnostic_lead_id,
        "diagnostic_lead_hyperparameters": diagnostic_lead_hyperparameters,
        "diagnostic_lead_state_sha256": diagnostic_lead_state_sha,
        "diagnostic_lead_ordering_key": _selection_key_report(
            diagnostic_lead_key
        ),
        "diagnostic_lead_has_final_selection_authority": False,
        "diagnostic_lead_metrics": {
            split: metrics.as_dict()
            for split, metrics in diagnostic_lead_split_metrics.items()
        },
        "validation": validation_report,
        "split_roles": split_roles,
        "structural_nonveto_schema": (
            LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        ),
        "candidate_eligibility_basis": (
            "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        ),
        "structural_safety_is_only_candidate_veto": True,
        "numeric_metrics_used_for_diagnostic_ordering": True,
        "numeric_metrics_used_for_final_selection": False,
        "numeric_metrics_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_publication": False,
        "weight_delta_threshold_used_for_candidate_eligibility": False,
        "fit_thresholds_used_for_candidate_eligibility": False,
        "fit_check_thresholds_used_for_candidate_eligibility": False,
        "selection_thresholds_used_for_candidate_eligibility": False,
        "validation_thresholds_used_for_publication": False,
        "validation_publication_veto_applied": False,
        "rollback_authorized": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "behavioral_intelligence_review_outside_terminal_compiler": True,
        "structurally_eligible_frontier_count": structurally_eligible_count,
        "structurally_eligible_numeric_diagnostic_miss_count": (
            structurally_eligible_numeric_diagnostic_miss_count
        ),
        "numeric_diagnostic_miss_can_exclude_structural_candidate": False,
        "diagnostic_lead_writer_structural_coverage": dict(
            diagnostic_lead_structural_coverage
        ),
        "diagnostic_lead_writer_structural_coverage_sha256": (
            diagnostic_lead_structural_coverage["coverage_sha256"]
        ),
        "published_writer_is_structurally_safe_effectful_nonincumbent": False,
        "published_writer_is_incumbent_pending_manual_review": True,
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
    if down_weight_prior is not None:
        prior_receipt = down_weight_prior.as_dict()
        body.update(
            {
                "down_weight_prior": prior_receipt,
                "down_weight_prior_authentication_sha256": (
                    down_weight_prior.authentication_sha256
                ),
                "down_weight_prior_tensor_sha256": _tensor_sha256(
                    frontier_prior,
                    "frontier_prior",
                ),
                "down_weight_prior_live_incumbent_sha256": (
                    down_weight_prior.live_deferred_w2_sha256
                ),
                "frontier_centered_on_authenticated_down_weight_prior": True,
                "live_seed_w2_retained_as_separate_incumbent_control": True,
                "down_weight_prior_installed_before_terminal": False,
                "down_weight_prior_installation_count_before_terminal": 0,
                "authenticated_pure_prior_frontier_candidate_count": 1,
                "pure_prior_frontier_candidate_id": "qdeim-prior-alpha-00",
            }
        )
    if exact_zero_cold_start:
        body.update(
            {
                "exact_zero_cold_start": True,
                "incumbent_seed_writer_sha256": incumbent_state_sha,
                "incumbent_all_elements_zero": True,
                "incumbent_nonzero_element_count": 0,
                "incumbent_is_noncandidate_control": True,
                "candidate_publication_deferred_until_manual_review": True,
                "exact_zero_candidate_eligibility_schema": (
                    LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA
                ),
                "candidate_diagnostic_ordering_evidence_scope": (
                    "finite-dense-structural-eligibility-plus-fit-ordering"
                ),
                "candidate_eligibility_basis": (
                    "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
                ),
                "fit_metrics_role": (
                    "ordering-among-structurally-eligible-not-threshold-gating"
                ),
                "fit_metrics_used_for_diagnostic_ordering": True,
                "fit_metrics_used_for_pre_review_candidate_selection": False,
                "fit_metrics_used_for_structural_eligibility": False,
                "fit_metrics_have_final_intelligence_authority": False,
                "fit_metric_thresholds_used_for_candidate_eligibility": False,
                "fit_metric_thresholds_used_for_publication": False,
                "structurally_eligible_frontier_count": (structurally_eligible_count),
                "structurally_eligible_fit_metric_gate_miss_count": (
                    cold_start_fit_metric_gate_miss_count
                ),
                "fit_metric_gate_miss_can_exclude_structural_candidate": False,
                "relative_to_zero_weight_delta_gate_applied": False,
                "heldout_evidence_role": (
                    "diagnostic-only-after-frozen-diagnostic-lead"
                ),
                "diagnostic_lead_frozen_before_any_heldout_diagnostic": True,
                "heldout_candidate_evaluation_count_before_lead_freeze": 0,
                "heldout_candidate_evaluation_count_after_lead_freeze": 6,
                "heldout_candidate_count_evaluated_after_lead_freeze": 2,
                "diagnostic_lead_writer_structural_coverage": dict(
                    diagnostic_lead_structural_coverage
                ),
                "diagnostic_lead_writer_structural_coverage_sha256": (
                    diagnostic_lead_structural_coverage["coverage_sha256"]
                ),
                "diagnostic_lead_writer_all_elements_finite": True,
                "diagnostic_lead_writer_all_elements_nonzero": (
                    diagnostic_lead_structural_coverage["all_elements_nonzero"]
                ),
                "diagnostic_lead_writer_all_source_features_covered": True,
                "published_writer_is_finite_dense_nonzero_structural_plant": False,
                "nonfinite_candidate_or_replay_is_hard_failure": True,
                "zero_training_pruning_quantization": True,
            }
        )
    result_sha = _canonical_sha256(body)
    report = {**body, "result_sha256": result_sha}
    compiler_lineage_token = _issue_terminal_compiler_lineage_proof(
        compiler_input_binding,
        result_sha256=result_sha,
    )
    return Lfm2CpcfTerminalActionResidualResult(
        policy=selected_policy,
        incumbent_down_weight=incumbent,
        selected_down_weight=selected_weight,
        published_down_weight=published,
        frontier_down_weights=frontier_down_weights,
        selected_candidate_id=selected_id,
        accepted=accepted,
        report=report,
        result_sha256=result_sha,
        _compiler_input_binding=compiler_input_binding,
        _compiler_issuer=_TERMINAL_RESULT_COMPILER_ISSUER,
        _compiler_lineage_receipt=compiler_lineage_token,
    )


__all__ = [
    "LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_DOWN_WEIGHT_PRIOR_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_EVIDENCE_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_POLICY_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_SEED_SCHEMA",
    "LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA",
    "Lfm2CpcfTerminalActionAuthenticationError",
    "Lfm2CpcfTerminalActionEvidence",
    "Lfm2CpcfTerminalActionDownWeightPrior",
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
    "lfm2_cpcf_terminal_action_tensor_sha256",
]
