"""Fit-only fixed-Q/K sequence-effective V/O folding for LFM2 attention.

The production CPCF attention compiler already derives one authenticated,
target-native Q/K/V producer from complete causal fit trajectories.  This
module keeps its exact Q/K state and folds the remaining value/output path by
the sequence function it implements:

``z_h = softmax(Q_h K_h^T) X_K`` and ``delta = sum_h z_h V_g O_h``.

For one fixed production policy, the compiler:

* ridge-fits the complete causal attended-input operator to the parent
  residual write;
* extracts one shared value subspace per target KV group by SVD;
* performs a bounded, monotone sequence of closed-form block V/O solves; and
* applies a fixed producer-value subspace anchor before materializing ordinary
  dense target V/O tensors.

Only ``fit_cases`` are accepted.  Selection and validation have no interface
to this primitive.  It performs no model forward, backward, optimizer,
training, pruning, or quantization operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge import hf_lfm2_structural_residual_attention_fold as structural
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedAttentionFoldCase,
    Lfm2CompiledStructuralAttentionProducerFold,
)


LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA = (
    "neural-forge-lfm2-fixed-qk-sequence-effective-attention-fold-v2"
)
LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_INSTALL_SCHEMA = (
    "neural-forge-lfm2-fixed-qk-sequence-effective-attention-install-v1"
)
LFM2_SEQUENCE_EFFECTIVE_PCG_OBSERVATION_SCHEMA = (
    "neural-forge-lfm2-sequence-effective-pcg-observation-v1"
)
LFM2_SEQUENCE_EFFECTIVE_ROW_BLOCK_JACOBI_SCHEMA = (
    "neural-forge-lfm2-sequence-effective-row-block-jacobi-v1"
)
FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID = (
    "fixed_qk_sequence_effective_vo_als"
)
_IMPLEMENTATION = "neural_forge.hf_lfm2_sequence_effective_attention_fold"
_MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS = 8_000_000
_MAXIMUM_EXPLICIT_COEFFICIENT_ELEMENTS = 16_000_000
_MAXIMUM_STREAMED_ALS_ITERATIONS = 2
_STATISTICS_ROW_CHUNK_SIZE = 2048
_PCG_MAXIMUM_ITERATIONS = 64
_PCG_RESTART_BLOCK_ITERATIONS = 64
_PCG_HARD_MAXIMUM_ITERATIONS = 512
_PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS = 4096
_PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE = 1e-8
_ZERO_EXECUTION = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "target_layer_forward_calls": 0,
    "generation_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "autograd_graphs": 0,
}


class Lfm2SequenceEffectiveAttentionFoldError(ValueError):
    """Sequence-effective attention input, topology, or authentication error."""

    def __init__(
        self,
        message: str,
        *,
        failure_report: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_report = (
            None
            if failure_report is None
            else MappingProxyType(dict(failure_report))
        )


class Lfm2SequenceEffectiveAttentionInstallError(RuntimeError):
    """Exact dense installation or rollback failure."""


_PCG_OBSERVER: ContextVar[Callable[[Mapping[str, Any]], None] | None] = ContextVar(
    "lfm2_sequence_effective_pcg_observer",
    default=None,
)


@contextmanager
def observe_lfm2_sequence_effective_pcg(
    observer: Callable[[Mapping[str, Any]], None],
) -> Iterator[None]:
    """Observe sealed PCG reports without replacing or wrapping solver calls.

    The observer is scoped to the current context, receives a detached
    canonical copy only after a solve has produced its authenticated report,
    and cannot mutate the report returned by the solver.  Nested observers are
    rejected so a probe has one unambiguous telemetry owner.
    """

    if not callable(observer):
        raise TypeError("observer must be callable.")
    if _PCG_OBSERVER.get() is not None:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "A sequence-effective PCG observer is already active."
        )
    token = _PCG_OBSERVER.set(observer)
    try:
        yield
    finally:
        _PCG_OBSERVER.reset(token)


def _observe_pcg_report(
    solver_kind: str,
    report: Mapping[str, Any],
) -> None:
    observer = _PCG_OBSERVER.get()
    if observer is None:
        return
    detached_report = json.loads(_canonical_json(dict(report)).decode("ascii"))
    body = {
        "schema": LFM2_SEQUENCE_EFFECTIVE_PCG_OBSERVATION_SCHEMA,
        "solver_kind": solver_kind,
        "report": detached_report,
        "report_sha256": detached_report["report_sha256"],
        "solver_result_or_arguments_modified": False,
    }
    event = MappingProxyType(
        {**body, "observation_sha256": _sha256(body)}
    )
    try:
        observer(event)
    except BaseException as exc:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "Sequence-effective PCG observation failed closed."
        ) from exc


def _seal_observe_pcg_report(
    solver_kind: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    report = _seal(body)
    _observe_pcg_report(solver_kind, report)
    return report


def _torch() -> Any:
    return structural._torch()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "Sequence-effective metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["report_sha256"] = _sha256(value)
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise Lfm2SequenceEffectiveAttentionFoldError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{name} must be finite."
        ) from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{name} is outside its supported range."
        )
    return result


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{name} must be a supported positive integer."
        )
    return value


def _tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype
        not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{name} must be one finite dense floating rank-{dimensions} tensor."
        )
    return value.detach().to(device="cpu").contiguous()


def _tensor_sha256(value: Any) -> str:
    return structural._tensor_sha256(value)


def _state_sha256(values: Mapping[str, Any]) -> str:
    return structural._state_sha256(values)


@dataclass(frozen=True, slots=True)
class _PcgRestartRowBlockPreconditioner:
    """Caller-built exact operator row blocks, authenticated only on use."""

    row_blocks: Any
    construction_report: Mapping[str, Any]


def _copy_parameter(target: Any, source: Any) -> None:
    target.copy_(source)


def _relative_rmse(prediction: Any, target: Any) -> float:
    return structural._relative_rmse(prediction, target)


def _case_bundle_sha256(
    cases: Sequence[Lfm2CachedAttentionFoldCase],
) -> str:
    tensor_fields = (
        "target_query_input",
        "target_key_value_input",
        "parent_query_linear",
        "parent_key_linear",
        "parent_value",
        "parent_query",
        "parent_key",
        "parent_context",
        "parent_post_attention_residual_target_space",
        "target_pre_attention_residual",
        "query_cosine",
        "query_sine",
        "key_cosine",
        "key_sine",
        "key_mask",
        "parent_contribution",
        "query_row_sqrt_weights",
        "key_value_row_sqrt_weights",
    )
    body = []
    for case_index, case in enumerate(cases):
        values = {}
        for field in tensor_fields:
            value = getattr(case, field)
            values[field] = None if value is None else _tensor_sha256(value)
        body.append(
            {
                "case_index": case_index,
                "tensor_sha256": values,
            }
        )
    return _sha256(body)


@dataclass(frozen=True, slots=True)
class Lfm2FixedQkSequenceEffectiveVoAlsPolicy:
    """Authenticated singular production policy proven before integration."""

    ridge_scale: float = 1e-4
    absolute_ridge_floor: float = 1e-12
    value_anchor_fraction: float = 0.25
    maximum_als_iterations: int = 12
    fit_tolerance: float = 1e-12
    horizon_power: float = 0.0
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        ridge = _finite_float(
            self.ridge_scale,
            "ridge_scale",
            minimum=1e-12,
            maximum=1.0,
        )
        floor = _finite_float(
            self.absolute_ridge_floor,
            "absolute_ridge_floor",
            minimum=1e-16,
            maximum=1e-2,
        )
        anchor = _finite_float(
            self.value_anchor_fraction,
            "value_anchor_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        tolerance = _finite_float(
            self.fit_tolerance,
            "fit_tolerance",
            minimum=0.0,
            maximum=1e-4,
        )
        horizon = _finite_float(
            self.horizon_power,
            "horizon_power",
            minimum=0.0,
            maximum=0.0,
        )
        iterations = _positive_int(
            self.maximum_als_iterations,
            "maximum_als_iterations",
            maximum=64,
        )
        object.__setattr__(self, "ridge_scale", ridge)
        object.__setattr__(self, "absolute_ridge_floor", floor)
        object.__setattr__(self, "value_anchor_fraction", anchor)
        object.__setattr__(self, "fit_tolerance", tolerance)
        object.__setattr__(self, "horizon_power", horizon)
        object.__setattr__(self, "maximum_als_iterations", iterations)
        if (
            ridge != 1e-4
            or floor != 1e-12
            or anchor != 0.25
            or iterations != 12
            or tolerance != 1e-12
            or horizon != 0.0
        ):
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "The sequence-effective production policy is singular and fixed."
            )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Policy authentication_sha256 is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA,
            "kind": "fixed_qk_sequence_effective_vo_als_policy",
            "ridge_scale": self.ridge_scale,
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "value_anchor_fraction": self.value_anchor_fraction,
            "maximum_als_iterations": self.maximum_als_iterations,
            "fit_tolerance": self.fit_tolerance,
            "horizon_power": self.horizon_power,
            "fixed_query_key_required": True,
            "causal_attended_input_operator": True,
            "shared_value_subspace_svd": True,
            "monotone_closed_form_value_output_als": (
                "dense_small_topology_only"
            ),
            "monotone_pcg_normal_equation_value_output_als": True,
            "streamed_cross_head_sufficient_statistics": True,
            "maximum_dense_ridge_system_elements": (
                _MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS
            ),
            "maximum_explicit_coefficient_elements": (
                _MAXIMUM_EXPLICIT_COEFFICIENT_ELEMENTS
            ),
            "statistics_row_chunk_size": _STATISTICS_ROW_CHUNK_SIZE,
            "pcg_maximum_iterations": _PCG_MAXIMUM_ITERATIONS,
            "pcg_relative_normal_residual_tolerance": (
                _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
            ),
            "accumulation_dtype": "torch.float64",
            "selection_inputs_accessible": False,
            "validation_inputs_accessible": False,
            "execution": {
                **_ZERO_EXECUTION,
                "closed_form_linear_solves": True,
                "dense_target_value_output_materialization": True,
            },
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_POLICY_SHA256 = (
    Lfm2FixedQkSequenceEffectiveVoAlsPolicy().authentication_sha256
)


@dataclass(frozen=True, slots=True)
class Lfm2CompiledFixedQkSequenceEffectiveAttentionFold:
    """Authenticated exact-Q/K plus sequence-effective dense V/O state."""

    producer_artifact: Lfm2CompiledStructuralAttentionProducerFold
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy
    value_weight: Any
    output_weight: Any
    fit_case_count: int
    fit_case_bundle_sha256: str
    producer_state_sha256_before: str
    producer_state_sha256_after: str
    target_template_sha256: str
    compiled_state_sha256: str
    candidate_frontier_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(
            self.producer_artifact,
            Lfm2CompiledStructuralAttentionProducerFold,
        ):
            raise TypeError(
                "producer_artifact must be an LFM2 attention producer fold."
            )
        if not isinstance(
            self.policy,
            Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
        ) or not self.policy.verify_authentication():
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Sequence-effective policy authentication failed."
            )
        topology = self.producer_artifact.topology
        value = _tensor(self.value_weight, "value_weight", dimensions=2)
        output = _tensor(self.output_weight, "output_weight", dimensions=2)
        if tuple(value.shape) != (
            topology.target_key_value_heads * topology.head_dim,
            topology.target_hidden_size,
        ) or tuple(output.shape) != (
            topology.target_output_size,
            topology.target_query_heads * topology.head_dim,
        ):
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Compiled V/O tensors differ from the producer topology."
            )
        case_count = _positive_int(
            self.fit_case_count,
            "fit_case_count",
            maximum=1_000_000,
        )
        fit_sha = _digest(
            self.fit_case_bundle_sha256,
            "fit_case_bundle_sha256",
        )
        producer_before = _digest(
            self.producer_state_sha256_before,
            "producer_state_sha256_before",
        )
        producer_after = _digest(
            self.producer_state_sha256_after,
            "producer_state_sha256_after",
        )
        template_sha = _digest(
            self.target_template_sha256,
            "target_template_sha256",
        )
        state_sha = _digest(
            self.compiled_state_sha256,
            "compiled_state_sha256",
        )
        frontier_sha = _digest(
            self.candidate_frontier_sha256,
            "candidate_frontier_sha256",
        )
        if producer_before != producer_after:
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "The fixed Q/K producer changed during V/O compilation."
            )
        producer_state_sha = _state_sha256(self.producer_artifact.state_dict())
        if (
            producer_state_sha != producer_before
            or producer_state_sha
            != self.producer_artifact.compiled_state_sha256
        ):
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Producer state authentication differs from the artifact."
            )
        object.__setattr__(self, "value_weight", value.clone())
        object.__setattr__(self, "output_weight", output.clone())
        if _state_sha256(self.state_dict()) != state_sha:
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Compiled Q/K/V/O state authentication failed."
            )
        report = dict(self.report)
        supplied = report.pop("report_sha256", None)
        inventory = report.get("candidate_frontier")
        execution = report.get("execution")
        if (
            supplied != _sha256(report)
            or report.get("schema")
            != LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA
            or report.get("candidate_id")
            != FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID
            or report.get("policy") != self.policy.as_dict()
            or report.get("policy_sha256") != self.policy.authentication_sha256
            or report.get("producer_artifact")
            != self.producer_artifact.as_dict()
            or report.get("producer_artifact_sha256")
            != _sha256(self.producer_artifact.as_dict())
            or report.get("fixed_query_key_sha256")
            != self.producer_artifact.score_state_sha256
            or report.get("fit_case_count") != case_count
            or report.get("fit_case_bundle_sha256") != fit_sha
            or report.get("producer_state_sha256_before") != producer_before
            or report.get("producer_state_sha256_after") != producer_after
            or report.get("target_template_sha256") != template_sha
            or report.get("compiled_state_sha256") != state_sha
            or not isinstance(inventory, list)
            or not inventory
            or _sha256(inventory) != frontier_sha
            or report.get("candidate_frontier_sha256") != frontier_sha
            or report.get("fit_only_compilation") is not True
            or report.get("compiled_from_splits") != ["fit"]
            or report.get("selection_case_count_consumed") != 0
            or report.get("validation_case_count_consumed") != 0
            or report.get("dense_target_native_qkvo") is not True
            or not isinstance(execution, Mapping)
            or any(execution.get(name) != value for name, value in _ZERO_EXECUTION.items())
        ):
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Sequence-effective artifact seal is invalid."
            )
        object.__setattr__(self, "fit_case_count", case_count)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def topology(self) -> Any:
        return self.producer_artifact.topology

    @property
    def cluster_plan(self) -> Any:
        return self.producer_artifact.cluster_plan

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.producer_artifact.query_weight.detach()
                .cpu()
                .clone(),
                "k_proj.weight": self.producer_artifact.key_weight.detach()
                .cpu()
                .clone(),
                "v_proj.weight": self.value_weight.detach().cpu().clone(),
                "out_proj.weight": self.output_weight.detach().cpu().clone(),
            }
        )

    def verify_authentication(self) -> bool:
        try:
            report = dict(self.report)
            supplied = report.pop("report_sha256", None)
            return bool(
                self.policy.verify_authentication()
                and supplied == _sha256(report)
                and _state_sha256(self.state_dict()) == self.compiled_state_sha256
                and report.get("candidate_frontier_sha256")
                == _sha256(report.get("candidate_frontier"))
                and report.get("fixed_query_key_sha256")
                == self.producer_artifact.score_state_sha256
            )
        except (TypeError, ValueError, RuntimeError):
            return False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2SequenceEffectiveAttentionInstallReport:
    """Authenticated exact in-place dense Q/K/V/O installation receipt."""

    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        body = dict(self.report)
        supplied = body.pop("report_sha256", None)
        if (
            supplied != _sha256(body)
            or body.get("schema")
            != LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_INSTALL_SCHEMA
            or body.get("accepted") is not True
        ):
            raise Lfm2SequenceEffectiveAttentionInstallError(
                "Sequence-effective installation report is invalid."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def accepted(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _ridge_solve(
    design: Any,
    target: Any,
    *,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    if (
        design.ndim != 2
        or target.ndim != 2
        or int(design.shape[0]) != int(target.shape[0])
        or int(design.shape[0]) == 0
        or int(design.shape[1]) == 0
        or int(target.shape[1]) == 0
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} ridge system is empty or misaligned."
        )
    rows = int(design.shape[0])
    features = int(design.shape[1])
    trace_scale = float(design.square().sum().item()) / max(features, 1)
    ridge = max(
        policy.absolute_ridge_floor,
        policy.ridge_scale * max(trace_scale, 0.0),
    )
    # The sequence operator can have Hq*hidden columns.  Solve its ridge
    # system in row space when that is smaller, avoiding a quadratic feature
    # Gram while preserving the same closed-form ridge solution.
    use_dual = features > rows
    if use_dual:
        regularized = design @ design.T + ridge * torch.eye(
            rows,
            dtype=torch.float64,
        )
        right = target
    else:
        regularized = design.T @ design + ridge * torch.eye(
            features,
            dtype=torch.float64,
        )
        right = design.T @ target
    solver = "solve"
    try:
        solved = torch.linalg.solve(regularized, right)
    except RuntimeError:
        solver = "lstsq"
        solved = torch.linalg.lstsq(
            regularized,
            right,
            driver="gelsd",
        ).solution
    solution = design.T @ solved if use_dual else solved
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} ridge solution is non-finite."
        )
    prediction = design @ solution
    report = {
        "role": role,
        "rows": int(design.shape[0]),
        "features": int(design.shape[1]),
        "outputs": int(target.shape[1]),
        "system_domain": "row_dual" if use_dual else "feature_primal",
        "linear_system_dimension": rows if use_dual else features,
        "ridge_scale": policy.ridge_scale,
        "trace_scale": trace_scale,
        "ridge": ridge,
        "solver": solver,
        "relative_rmse": _relative_rmse(prediction, target),
        "solution_sha256": _tensor_sha256(solution),
    }
    return solution.contiguous(), _seal(report)


def _sequence_effective_solver_plan(
    *,
    fit_rows: int,
    target_query_heads: int,
    target_hidden_size: int,
    target_output_size: int,
    head_dim: int,
) -> dict[str, Any]:
    rows = _positive_int(fit_rows, "fit_rows", maximum=1_000_000_000)
    query_heads = _positive_int(
        target_query_heads,
        "target_query_heads",
        maximum=1_000_000,
    )
    hidden = _positive_int(
        target_hidden_size,
        "target_hidden_size",
        maximum=1_000_000,
    )
    output = _positive_int(
        target_output_size,
        "target_output_size",
        maximum=1_000_000,
    )
    dimension = _positive_int(head_dim, "head_dim", maximum=1_000_000)
    global_features = query_heads * hidden
    global_system_dimension = min(rows, global_features)
    global_system_elements = global_system_dimension**2
    value_feature_count = hidden * dimension
    forbidden_value_coefficient_elements = (
        rows * output * value_feature_count
    )
    dense_small_path = bool(
        global_system_elements <= _MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS
        and value_feature_count**2 <= _MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS
        and forbidden_value_coefficient_elements
        <= _MAXIMUM_EXPLICIT_COEFFICIENT_ELEMENTS
    )
    if dense_small_path:
        strategy = "complete_causal_operator_exact_ridge"
        largest_system_dimension = global_system_dimension
        estimated_statistics_bytes = global_system_elements * 8
    else:
        strategy = "streamed_cross_head_statistics_pcg"
        largest_system_dimension = max(
            hidden,
            query_heads * dimension,
        )
        if largest_system_dimension**2 > _MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS:
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Sequence-effective topology exceeds the authenticated dense "
                "headwise ridge budget."
            )
        estimated_statistics_bytes = (
            query_heads * query_heads * hidden * hidden
            + query_heads * hidden * output
        ) * 8
    body = {
        "strategy": strategy,
        "fit_rows": rows,
        "target_query_heads": query_heads,
        "target_hidden_size": hidden,
        "target_output_size": output,
        "head_dim": dimension,
        "complete_operator_feature_count": global_features,
        "complete_operator_dense_system_dimension": global_system_dimension,
        "complete_operator_dense_system_elements": global_system_elements,
        "complete_operator_dense_system_bytes": global_system_elements * 8,
        "maximum_dense_ridge_system_elements": (
            _MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS
        ),
        "maximum_explicit_coefficient_elements": (
            _MAXIMUM_EXPLICIT_COEFFICIENT_ELEMENTS
        ),
        "largest_executed_dense_system_dimension": largest_system_dimension,
        "largest_executed_dense_system_elements": largest_system_dimension**2,
        "estimated_streamed_statistics_bytes": estimated_statistics_bytes,
        "forbidden_value_coefficient_elements_per_kv_group": (
            forbidden_value_coefficient_elements
        ),
        "forbidden_value_coefficient_bytes_per_kv_group": (
            forbidden_value_coefficient_elements * 8
        ),
        "forbidden_value_dense_gram_elements": value_feature_count**2,
        "forbidden_value_dense_gram_bytes": value_feature_count**2 * 8,
        "forbidden_value_coefficient_materialized": False,
        "forbidden_value_dense_gram_materialized": False,
        "statistics_row_chunk_size": _STATISTICS_ROW_CHUNK_SIZE,
        "complete_operator_quadratic_gram_avoided": False,
        "monolithic_flat_feature_gram_materialized": (
            strategy == "complete_causal_operator_exact_ridge"
        ),
        "complete_operator_dense_factorization_avoided": (
            strategy == "streamed_cross_head_statistics_pcg"
        ),
        "matrix_materialization_bound_authenticated": True,
    }
    return _seal(body)


def _ridge_solve_statistics(
    gram: Any,
    right: Any,
    *,
    target_square_sum: float,
    rows: int,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    features = int(gram.shape[0])
    outputs = int(right.shape[1])
    if (
        gram.ndim != 2
        or right.ndim != 2
        or tuple(gram.shape) != (features, features)
        or int(right.shape[0]) != features
        or rows <= 0
        or not bool(torch.isfinite(gram).all().item())
        or not bool(torch.isfinite(right).all().item())
        or not math.isfinite(target_square_sum)
        or target_square_sum < 0.0
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} streamed ridge statistics are invalid."
        )
    trace_scale = float(torch.trace(gram).item()) / max(features, 1)
    ridge = max(
        policy.absolute_ridge_floor,
        policy.ridge_scale * max(trace_scale, 0.0),
    )
    regularized = gram + ridge * torch.eye(features, dtype=torch.float64)
    solver = "solve"
    try:
        solution = torch.linalg.solve(regularized, right)
    except RuntimeError:
        solver = "lstsq"
        solution = torch.linalg.lstsq(
            regularized,
            right,
            driver="gelsd",
        ).solution
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} streamed ridge solution is non-finite."
        )
    prediction_square = float((solution * (gram @ solution)).sum().item())
    cross = float((solution * right).sum().item())
    squared_error = max(
        prediction_square - 2.0 * cross + target_square_sum,
        0.0,
    )
    relative_rmse = math.sqrt(
        squared_error / max(target_square_sum, torch.finfo(torch.float64).eps)
    )
    report = {
        "role": role,
        "rows": rows,
        "features": features,
        "outputs": outputs,
        "system_domain": "streamed_feature_primal",
        "linear_system_dimension": features,
        "ridge_scale": policy.ridge_scale,
        "trace_scale": trace_scale,
        "ridge": ridge,
        "solver": solver,
        "relative_rmse": relative_rmse,
        "solution_sha256": _tensor_sha256(solution),
    }
    return solution.contiguous(), _seal(report)


def _pcg_multi_rhs(
    *,
    right: Any,
    matvec: Any,
    diagonal: Any,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    """Deterministic Jacobi-PCG with one independent Krylov scalar per output."""

    torch = _torch()
    solution = torch.zeros_like(right)
    residual = right.clone()
    right_norm = torch.sqrt(
        residual.square().sum(dim=tuple(range(residual.ndim - 1)))
    ).clamp_min(torch.finfo(torch.float64).eps)
    preconditioned = residual / diagonal.unsqueeze(-1)
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum(
        dim=tuple(range(residual.ndim - 1))
    )
    relative = torch.sqrt(
        residual.square().sum(dim=tuple(range(residual.ndim - 1)))
    ) / right_norm
    active = relative > _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
    if not bool(active.any().item()):
        body = {
            "role": role,
            "solver": "deterministic_jacobi_pcg_multi_rhs",
            "maximum_iterations": _PCG_MAXIMUM_ITERATIONS,
            "iterations_executed": 0,
            "required_relative_normal_residual": (
                _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
            ),
            "initial_relative_normal_residual": float(relative.max().item()),
            "final_maximum_relative_normal_residual": float(
                relative.max().item()
            ),
            "relative_normal_residual_history": [],
            "solution_sha256": _tensor_sha256(solution),
        }
        return solution.contiguous(), _seal_observe_pcg_report(
            "multi_rhs",
            body,
        )
    history = []
    iterations = 0
    for iteration in range(1, _PCG_MAXIMUM_ITERATIONS + 1):
        if not bool(active.any().item()):
            break
        applied = matvec(direction)
        denominator = (direction * applied).sum(
            dim=tuple(range(direction.ndim - 1))
        )
        if bool(
            (
                active
                & (denominator.abs() <= torch.finfo(torch.float64).eps)
            ).any().item()
        ):
            raise Lfm2SequenceEffectiveAttentionFoldError(
                f"{role} PCG encountered a singular search direction."
            )
        alpha = torch.where(
            active,
            rz / torch.where(
                active,
                denominator,
                torch.ones_like(denominator),
            ),
            torch.zeros_like(rz),
        )
        view = (1,) * (solution.ndim - 1) + (-1,)
        solution = solution + direction * alpha.reshape(view)
        residual = residual - applied * alpha.reshape(view)
        relative = torch.sqrt(
            residual.square().sum(dim=tuple(range(residual.ndim - 1)))
        ) / right_norm
        maximum_relative = float(relative.max().item())
        history.append(maximum_relative)
        iterations = iteration
        next_active = relative > _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
        if not bool(next_active.any().item()):
            break
        next_preconditioned = residual / diagonal.unsqueeze(-1)
        next_rz = (residual * next_preconditioned).sum(
            dim=tuple(range(residual.ndim - 1))
        )
        beta = torch.where(
            next_active,
            next_rz
            / torch.where(next_active, rz, torch.ones_like(rz)),
            torch.zeros_like(rz),
        )
        direction = (
            next_preconditioned + direction * beta.reshape(view)
        ) * next_active.reshape(view)
        rz = next_rz
        active = next_active
    final_relative = history[-1]
    if (
        not bool(torch.isfinite(solution).all().item())
        or final_relative > _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} PCG failed its authenticated residual tolerance "
            f"({final_relative:.6e})."
        )
    body = {
        "role": role,
        "solver": "deterministic_jacobi_pcg_multi_rhs",
        "maximum_iterations": _PCG_MAXIMUM_ITERATIONS,
        "iterations_executed": iterations,
        "required_relative_normal_residual": (
            _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
        ),
        "initial_relative_normal_residual": 1.0,
        "final_maximum_relative_normal_residual": final_relative,
        "relative_normal_residual_history": history,
        "solution_sha256": _tensor_sha256(solution),
    }
    return solution.contiguous(), _seal_observe_pcg_report(
        "multi_rhs",
        body,
    )


def _pcg_json_float(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


class _PcgRowBlockPreconditionerAuthenticationError(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = {} if details is None else dict(details)


def _row_block_numerical_tolerance(value: Any) -> float:
    torch = _torch()
    scale = max(1.0, float(value.abs().max().item()))
    return 128.0 * float(torch.finfo(value.dtype).eps) * scale


def _build_structured_value_row_block_jacobi_preconditioner(
    *,
    cross: Any,
    output_blocks: Any,
    group_heads: Sequence[int],
    diagonal: Any,
    ridge: float,
    role: str,
) -> _PcgRestartRowBlockPreconditioner:
    """Materialize only the exact head-dimension block for each hidden row."""

    torch = _torch()
    heads = tuple(int(head) for head in group_heads)
    ridge_value = _pcg_json_float(ridge)
    if (
        cross.ndim != 4
        or output_blocks.ndim != 3
        or diagonal.ndim != 2
        or not heads
        or any(
            head < 0 or head >= int(cross.shape[0])
            for head in heads
        )
        or int(cross.shape[0]) != int(cross.shape[1])
        or int(cross.shape[2]) != int(cross.shape[3])
        or int(cross.shape[2]) != int(diagonal.shape[0])
        or int(output_blocks.shape[0]) != int(cross.shape[0])
        or int(output_blocks.shape[1]) != int(diagonal.shape[1])
        or cross.dtype != diagonal.dtype
        or output_blocks.dtype != diagonal.dtype
        or cross.device != diagonal.device
        or output_blocks.device != diagonal.device
        or ridge_value is None
        or ridge_value <= 0.0
        or not role
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} row-block Jacobi construction inputs are invalid."
        )
    if (
        any(
            not bool(torch.isfinite(cross[head, other]).all().item())
            for head in heads
            for other in heads
        )
        or not bool(torch.isfinite(output_blocks).all().item())
        or not bool(torch.isfinite(diagonal).all().item())
        or not bool((diagonal > 0.0).all().item())
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} row-block Jacobi construction inputs must be finite "
            "with a positive scalar diagonal."
        )

    row_count, block_size = map(int, diagonal.shape)
    row_blocks = torch.zeros(
        row_count,
        block_size,
        block_size,
        dtype=diagonal.dtype,
        device=diagonal.device,
    )
    source_terms = []
    for head in heads:
        for other in heads:
            cross_diagonal = torch.diagonal(cross[head, other]).contiguous()
            coupling = (
                output_blocks[other] @ output_blocks[head].T
            ).contiguous()
            # For row-major vec(V), the row block maps a columnized row by
            # coupling.T: (C @ V @ coupling)[i, a] =
            # sum_b C[i, i] * coupling[b, a] * V[i, b].
            row_blocks.add_(
                cross_diagonal.reshape(row_count, 1, 1)
                * coupling.T.unsqueeze(0)
            )
            source_terms.append(
                {
                    "head": head,
                    "other_head": other,
                    "cross_diagonal_sha256": _tensor_sha256(cross_diagonal),
                    "coupling_sha256": _tensor_sha256(coupling),
                }
            )
    row_blocks.add_(
        ridge_value
        * torch.eye(
            block_size,
            dtype=row_blocks.dtype,
            device=row_blocks.device,
        ).unsqueeze(0)
    )
    # The complete least-squares normal operator is mathematically symmetric.
    # Pairwise averaging only removes floating summation skew between the
    # mirrored (head, other) terms; it does not invent a new approximation.
    row_blocks = (
        0.5 * (row_blocks + row_blocks.transpose(1, 2))
    ).contiguous()
    row_block_diagonal = torch.diagonal(
        row_blocks,
        dim1=1,
        dim2=2,
    ).contiguous()
    maximum_asymmetry = float(
        (row_blocks - row_blocks.transpose(1, 2)).abs().max().item()
    )
    symmetry_tolerance = _row_block_numerical_tolerance(row_blocks)
    scalar_diagonal_error = float(
        (row_block_diagonal - diagonal).abs().max().item()
    )
    diagonal_tolerance = _row_block_numerical_tolerance(row_block_diagonal)
    body = {
        "schema": LFM2_SEQUENCE_EFFECTIVE_ROW_BLOCK_JACOBI_SCHEMA,
        "role": role,
        "construction": "exact_structured_operator_row_blocks",
        "row_block_count": row_count,
        "row_block_size": block_size,
        "row_blocks_shape": list(row_blocks.shape),
        "row_blocks_dtype": str(row_blocks.dtype),
        "row_blocks_sha256": _tensor_sha256(row_blocks),
        "scalar_diagonal_sha256": _tensor_sha256(diagonal),
        "scalar_diagonal_maximum_absolute_error": scalar_diagonal_error,
        "scalar_diagonal_match_tolerance": diagonal_tolerance,
        "row_blocks_maximum_asymmetry": maximum_asymmetry,
        "row_blocks_symmetry_tolerance": symmetry_tolerance,
        "source_term_count": len(source_terms),
        "source_terms_sha256": _sha256(source_terms),
        "ridge": ridge_value,
        "full_operator_materialized": False,
    }
    return _PcgRestartRowBlockPreconditioner(
        row_blocks=row_blocks,
        construction_report=MappingProxyType(_seal(body)),
    )


def _authenticate_restart_row_block_preconditioner(
    *,
    supplied: Any,
    right: Any,
    diagonal: Any,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    """Authenticate and factor caller-built row blocks after scalar miss."""

    torch = _torch()

    def reject(
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        raise _PcgRowBlockPreconditionerAuthenticationError(
            reason,
            details=details,
        )

    if not isinstance(supplied, _PcgRestartRowBlockPreconditioner):
        reject("row_block_preconditioner_has_wrong_type")
    row_blocks = supplied.row_blocks
    if (
        not isinstance(row_blocks, torch.Tensor)
        or row_blocks.layout != torch.strided
        or row_blocks.device.type == "meta"
        or row_blocks.ndim != 3
        or right.ndim != 2
        or diagonal.ndim != 2
    ):
        reject("row_block_preconditioner_tensor_is_invalid")
    expected_shape = (
        int(right.shape[0]),
        int(right.shape[1]),
        int(right.shape[1]),
    )
    if (
        tuple(row_blocks.shape) != expected_shape
        or tuple(diagonal.shape) != tuple(right.shape)
        or row_blocks.dtype != right.dtype
        or diagonal.dtype != right.dtype
        or row_blocks.device != right.device
        or diagonal.device != right.device
    ):
        reject(
            "row_block_preconditioner_topology_mismatch",
            details={
                "expected_shape": list(expected_shape),
                "observed_shape": list(row_blocks.shape),
                "right_shape": list(right.shape),
                "diagonal_shape": list(diagonal.shape),
                "dtype_matches": row_blocks.dtype == right.dtype,
                "device_matches": row_blocks.device == right.device,
            },
        )
    if not bool(torch.isfinite(row_blocks).all().item()):
        reject("row_block_preconditioner_is_non_finite")

    try:
        report = dict(supplied.construction_report)
    except (TypeError, ValueError):
        reject("row_block_preconditioner_report_is_invalid")
    expected_report_keys = {
        "schema",
        "role",
        "construction",
        "row_block_count",
        "row_block_size",
        "row_blocks_shape",
        "row_blocks_dtype",
        "row_blocks_sha256",
        "scalar_diagonal_sha256",
        "scalar_diagonal_maximum_absolute_error",
        "scalar_diagonal_match_tolerance",
        "row_blocks_maximum_asymmetry",
        "row_blocks_symmetry_tolerance",
        "source_term_count",
        "source_terms_sha256",
        "ridge",
        "full_operator_materialized",
        "report_sha256",
    }
    if set(report) != expected_report_keys:
        reject("row_block_preconditioner_report_fields_are_not_exact")
    supplied_sha256 = report.pop("report_sha256")
    try:
        report_sha256 = _sha256(report)
    except Lfm2SequenceEffectiveAttentionFoldError:
        reject("row_block_preconditioner_report_is_not_canonical")
    if supplied_sha256 != report_sha256:
        reject("row_block_preconditioner_report_seal_mismatch")
    source_term_count = report.get("source_term_count")
    source_terms_sha256 = report.get("source_terms_sha256")
    ridge_value = _pcg_json_float(report.get("ridge"))
    if (
        report.get("schema")
        != LFM2_SEQUENCE_EFFECTIVE_ROW_BLOCK_JACOBI_SCHEMA
        or report.get("role") != role
        or report.get("construction")
        != "exact_structured_operator_row_blocks"
        or report.get("row_block_count") != expected_shape[0]
        or report.get("row_block_size") != expected_shape[1]
        or report.get("row_blocks_shape") != list(expected_shape)
        or report.get("row_blocks_dtype") != str(row_blocks.dtype)
        or report.get("full_operator_materialized") is not False
        or not isinstance(source_term_count, int)
        or isinstance(source_term_count, bool)
        or source_term_count <= 0
        or ridge_value is None
        or ridge_value <= 0.0
    ):
        reject("row_block_preconditioner_report_binding_mismatch")
    try:
        _digest(source_terms_sha256, "source_terms_sha256")
    except Lfm2SequenceEffectiveAttentionFoldError:
        reject("row_block_preconditioner_source_digest_is_invalid")
    if (
        report.get("row_blocks_sha256") != _tensor_sha256(row_blocks)
        or report.get("scalar_diagonal_sha256") != _tensor_sha256(diagonal)
    ):
        reject("row_block_preconditioner_tensor_hash_mismatch")

    row_block_diagonal = torch.diagonal(
        row_blocks,
        dim1=1,
        dim2=2,
    ).contiguous()
    diagonal_error = float(
        (row_block_diagonal - diagonal).abs().max().item()
    )
    diagonal_tolerance = _row_block_numerical_tolerance(row_block_diagonal)
    maximum_asymmetry = float(
        (row_blocks - row_blocks.transpose(1, 2)).abs().max().item()
    )
    symmetry_tolerance = _row_block_numerical_tolerance(row_blocks)
    if (
        report.get("scalar_diagonal_maximum_absolute_error")
        != diagonal_error
        or report.get("scalar_diagonal_match_tolerance")
        != diagonal_tolerance
        or diagonal_error > diagonal_tolerance
    ):
        reject(
            "row_block_preconditioner_scalar_diagonal_mismatch",
            details={
                "maximum_absolute_error": diagonal_error,
                "required_tolerance": diagonal_tolerance,
            },
        )
    if (
        report.get("row_blocks_maximum_asymmetry")
        != maximum_asymmetry
        or report.get("row_blocks_symmetry_tolerance")
        != symmetry_tolerance
        or maximum_asymmetry > symmetry_tolerance
    ):
        reject(
            "row_block_preconditioner_is_not_symmetric",
            details={
                "maximum_asymmetry": maximum_asymmetry,
                "required_tolerance": symmetry_tolerance,
            },
        )

    factor, factorization_info = torch.linalg.cholesky_ex(
        row_blocks,
        upper=False,
        check_errors=False,
    )
    failing = torch.nonzero(
        factorization_info != 0,
        as_tuple=False,
    ).flatten()
    if (
        int(failing.numel()) != 0
        or not bool(torch.isfinite(factor).all().item())
        or not bool(
            (
                torch.diagonal(factor, dim1=1, dim2=2) > 0.0
            ).all().item()
        )
    ):
        reject(
            "row_block_preconditioner_cholesky_failed",
            details={
                "failing_row_count": int(failing.numel()),
                "first_failing_row": (
                    int(failing[0].item())
                    if int(failing.numel()) != 0
                    else None
                ),
                "factor_finite": bool(torch.isfinite(factor).all().item()),
            },
        )
    construction_report = dict(supplied.construction_report)
    receipt = _seal(
        {
            "kind": "deterministic_exact_row_block_jacobi_cholesky",
            "construction_report": construction_report,
            "construction_report_sha256": construction_report[
                "report_sha256"
            ],
            "row_blocks_sha256": construction_report[
                "row_blocks_sha256"
            ],
            "factorization": "batched_lower_cholesky_ex",
            "factor_sha256": _tensor_sha256(factor),
            "row_block_count": expected_shape[0],
            "row_block_size": expected_shape[1],
            "all_row_blocks_spd": True,
            "factorization_materialized_only_after_original_lane_exact_miss": (
                True
            ),
            "full_operator_materialized": False,
        }
    )
    return factor.contiguous(), receipt


def _single_tensor_pcg_extended_body(
    *,
    role: str,
    solution: Any,
    initial_relative: float | None,
    original_lane_iterations: int,
    original_lane_final_relative: float | None,
    total_iterations: int,
    history: Sequence[float],
    exact_relative: float | None,
    exact_final_residual_authenticated: bool,
    exact_checkpoints: Sequence[Mapping[str, Any]],
    restart_schedule: Sequence[Mapping[str, Any]],
    total_matvec_calls: int,
    extension_triggered: bool,
) -> dict[str, Any]:
    torch = _torch()
    solution_finite = bool(torch.isfinite(solution).all().item())
    recurrence_history = list(history)
    checkpoints = [dict(value) for value in exact_checkpoints]
    schedule = [dict(value) for value in restart_schedule]
    final_recurrence = (
        recurrence_history[-1]
        if recurrence_history
        else _pcg_json_float(initial_relative)
    )
    return {
        "role": role,
        "solver": "deterministic_restarted_jacobi_pcg_single_tensor",
        "maximum_iterations": _PCG_HARD_MAXIMUM_ITERATIONS,
        "original_lane_maximum_iterations": _PCG_MAXIMUM_ITERATIONS,
        "original_lane_iterations_executed": original_lane_iterations,
        "restart_block_iterations": _PCG_RESTART_BLOCK_ITERATIONS,
        "iterations_executed": total_iterations,
        "required_relative_normal_residual": (
            _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
        ),
        "initial_relative_normal_residual": _pcg_json_float(initial_relative),
        "original_lane_final_recurrence_relative_normal_residual": (
            _pcg_json_float(original_lane_final_relative)
        ),
        "final_recurrence_relative_normal_residual": _pcg_json_float(
            final_recurrence
        ),
        "final_relative_normal_residual": (
            _pcg_json_float(exact_relative)
            if exact_final_residual_authenticated
            else None
        ),
        "relative_normal_residual_history": recurrence_history,
        "recurrence_history_sha256": _sha256(recurrence_history),
        "exact_relative_normal_residual_checkpoints": checkpoints,
        "exact_residual_checkpoints_sha256": _sha256(checkpoints),
        "restart_schedule": schedule,
        "restart_schedule_sha256": _sha256(schedule),
        "total_matvec_calls": total_matvec_calls,
        "exact_final_residual_authenticated": (
            exact_final_residual_authenticated
        ),
        "extended_after_original_lane_exact_authentication_miss": (
            extension_triggered
        ),
        "solution_finite": solution_finite,
        "solution_sha256": (
            _tensor_sha256(solution) if solution_finite else None
        ),
    }


def _single_tensor_pcg_continuous_row_block_body(
    *,
    role: str,
    solution: Any,
    initial_relative: float | None,
    original_lane_iterations: int,
    original_lane_final_relative: float | None,
    total_iterations: int,
    history: Sequence[float],
    exact_relative: float | None,
    exact_final_residual_authenticated: bool,
    exact_checkpoints: Sequence[Mapping[str, Any]],
    total_matvec_calls: int,
    lane_state: Mapping[str, Any],
    preconditioner_state: Mapping[str, Any],
) -> dict[str, Any]:
    torch = _torch()
    solution_finite = bool(torch.isfinite(solution).all().item())
    recurrence_history = list(history)
    checkpoints = [dict(value) for value in exact_checkpoints]
    final_recurrence = (
        recurrence_history[-1]
        if recurrence_history
        else _pcg_json_float(initial_relative)
    )
    return {
        "role": role,
        "solver": (
            "deterministic_continuous_row_block_jacobi_pcg_single_tensor"
        ),
        "maximum_iterations": _PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS,
        "hard_cap_includes_original_lane_iterations": True,
        "original_lane_maximum_iterations": _PCG_MAXIMUM_ITERATIONS,
        "original_lane_iterations_executed": original_lane_iterations,
        "single_restart_after_original_lane_exact_authentication_miss": True,
        "periodic_restart_count": 0,
        "iterations_executed": total_iterations,
        "continuous_row_block_lane_iterations_executed": (
            total_iterations - original_lane_iterations
        ),
        "required_relative_normal_residual": (
            _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
        ),
        "initial_relative_normal_residual": _pcg_json_float(initial_relative),
        "original_lane_final_recurrence_relative_normal_residual": (
            _pcg_json_float(original_lane_final_relative)
        ),
        "final_recurrence_relative_normal_residual": _pcg_json_float(
            final_recurrence
        ),
        "final_relative_normal_residual": (
            _pcg_json_float(exact_relative)
            if exact_final_residual_authenticated
            else None
        ),
        "relative_normal_residual_history": recurrence_history,
        "recurrence_history_sha256": _sha256(recurrence_history),
        "exact_relative_normal_residual_checkpoints": checkpoints,
        "exact_residual_checkpoints_sha256": _sha256(checkpoints),
        "total_matvec_calls": total_matvec_calls,
        "exact_final_residual_authenticated": (
            exact_final_residual_authenticated
        ),
        "extended_after_original_lane_exact_authentication_miss": True,
        "exact_residual_used_to_initialize_continuous_lane_once": True,
        "exact_diagnostic_matvecs_modify_recurrence_state_or_direction": False,
        "continuous_row_block_lane": dict(lane_state),
        "restart_preconditioner": dict(preconditioner_state),
        "solution_finite": solution_finite,
        "solution_sha256": (
            _tensor_sha256(solution) if solution_finite else None
        ),
    }


def _pcg_single_tensor(
    *,
    right: Any,
    matvec: Any,
    diagonal: Any,
    initial: Any,
    role: str,
    restart_row_block_preconditioner: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    solution = initial.clone()
    history: list[float] = []
    exact_checkpoints: list[dict[str, Any]] = []
    restart_schedule: list[dict[str, Any]] = []
    total_iterations = 0
    total_matvec_calls = 0
    initial_relative: float | None = None
    original_lane_iterations = 0
    original_lane_final_relative: float | None = None
    exact_residual = None
    exact_relative: float | None = None
    exact_iteration: int | None = None
    extension_triggered = False
    row_block_lane_started = False
    row_block_lane_state: dict[str, Any] = {}
    row_block_preconditioner_state: dict[str, Any] = {
        "kind": "deterministic_exact_row_block_jacobi_cholesky",
        "supplied": restart_row_block_preconditioner is not None,
        "authentication_attempted": False,
        "authenticated": False,
        "applied": False,
        "application_attempt_count": 0,
        "application_count": 0,
        "receipt": None,
        "authentication_failure_reason": None,
    }

    def extended_body() -> dict[str, Any]:
        if row_block_lane_started:
            return _single_tensor_pcg_continuous_row_block_body(
                role=role,
                solution=solution,
                initial_relative=initial_relative,
                original_lane_iterations=original_lane_iterations,
                original_lane_final_relative=original_lane_final_relative,
                total_iterations=total_iterations,
                history=history,
                exact_relative=exact_relative,
                exact_final_residual_authenticated=(
                    exact_iteration == total_iterations
                    and _pcg_json_float(exact_relative) is not None
                ),
                exact_checkpoints=exact_checkpoints,
                total_matvec_calls=total_matvec_calls,
                lane_state=row_block_lane_state,
                preconditioner_state=row_block_preconditioner_state,
            )
        return _single_tensor_pcg_extended_body(
            role=role,
            solution=solution,
            initial_relative=initial_relative,
            original_lane_iterations=original_lane_iterations,
            original_lane_final_relative=original_lane_final_relative,
            total_iterations=total_iterations,
            history=history,
            exact_relative=exact_relative,
            exact_final_residual_authenticated=(
                exact_iteration == total_iterations
                and _pcg_json_float(exact_relative) is not None
            ),
            exact_checkpoints=exact_checkpoints,
            restart_schedule=restart_schedule,
            total_matvec_calls=total_matvec_calls,
            extension_triggered=extension_triggered,
        )

    def fail_closed(
        reason: str,
        *,
        stage: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        failure_report = _seal(
            {
                **extended_body(),
                "status": "failed_closed",
                "failure_reason": reason,
                "failure_stage": stage,
                "failure_details": (
                    {} if details is None else dict(details)
                ),
            }
        )
        authenticated = failure_report[
            "exact_final_residual_authenticated"
        ]
        value = failure_report["final_relative_normal_residual"]
        residual_text = (
            f"{float(value):.6e}"
            if authenticated and value is not None
            else "unavailable"
        )
        raise Lfm2SequenceEffectiveAttentionFoldError(
            f"{role} PCG failed its authenticated exact residual tolerance "
            f"or solver invariant at {stage}; exact residual {residual_text}; "
            f"failure report "
            f"{failure_report['report_sha256']}.",
            failure_report=failure_report,
        )

    right_finite = bool(torch.isfinite(right).all().item())
    initial_finite = bool(torch.isfinite(solution).all().item())
    diagonal_finite = bool(torch.isfinite(diagonal).all().item())
    diagonal_positive = bool((diagonal > 0.0).all().item())
    if (
        not right_finite
        or not initial_finite
        or not diagonal_finite
        or not diagonal_positive
    ):
        fail_closed(
            "invalid_non_finite_or_non_positive_input",
            stage="input_validation",
            details={
                "right_finite": right_finite,
                "initial_finite": initial_finite,
                "diagonal_finite": diagonal_finite,
                "diagonal_strictly_positive": diagonal_positive,
            },
        )

    right_norm = torch.linalg.vector_norm(right).clamp_min(
        torch.finfo(torch.float64).eps
    )
    if not bool(torch.isfinite(right_norm).item()):
        fail_closed(
            "non_finite_right_norm",
            stage="input_validation",
        )
    applied_initial = matvec(solution)
    total_matvec_calls += 1
    residual = right - applied_initial
    if not bool(torch.isfinite(applied_initial).all().item()) or not bool(
        torch.isfinite(residual).all().item()
    ):
        fail_closed(
            "non_finite_initial_operator_application",
            stage="initial_exact_residual",
        )
    initial_relative = float(
        (torch.linalg.vector_norm(residual) / right_norm).item()
    )
    if not math.isfinite(initial_relative):
        fail_closed(
            "non_finite_initial_relative_residual",
            stage="initial_exact_residual",
        )
    exact_residual = residual
    exact_relative = initial_relative
    exact_iteration = 0
    if initial_relative <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE:
        body = {
            "role": role,
            "solver": "deterministic_jacobi_pcg_single_tensor",
            "maximum_iterations": _PCG_MAXIMUM_ITERATIONS,
            "iterations_executed": 0,
            "required_relative_normal_residual": (
                _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
            ),
            "initial_relative_normal_residual": initial_relative,
            "final_relative_normal_residual": initial_relative,
            "relative_normal_residual_history": [],
            "solution_sha256": _tensor_sha256(solution),
        }
        return solution.contiguous(), _seal_observe_pcg_report(
            "single_tensor",
            body,
        )

    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    rz = float((residual * preconditioned).sum().item())
    if not math.isfinite(rz) or rz <= 0.0:
        if not math.isfinite(rz):
            exact_iteration = None
        fail_closed(
            "non_positive_or_non_finite_preconditioned_residual",
            stage="original_lane_initialization",
            details={"residual_preconditioner_inner_product": _pcg_json_float(rz)},
        )

    original_lane_exact_checked = False
    for iteration in range(1, _PCG_MAXIMUM_ITERATIONS + 1):
        applied = matvec(direction)
        total_matvec_calls += 1
        denominator = float((direction * applied).sum().item())
        applied_finite = bool(torch.isfinite(applied).all().item())
        curvature_finite = math.isfinite(denominator)
        if (
            not applied_finite
            or not curvature_finite
            or denominator <= 0.0
        ):
            if not applied_finite or not curvature_finite:
                exact_iteration = None
            original_lane_iterations = total_iterations
            original_lane_final_relative = (
                history[-1] if history else initial_relative
            )
            fail_closed(
                "non_positive_or_non_finite_curvature",
                stage="original_lane_curvature",
                details={
                    "attempted_iteration": iteration,
                    "curvature": _pcg_json_float(denominator),
                    "curvature_finite_strictly_positive": False,
                    "curvature_matvec_call": total_matvec_calls,
                },
            )
        alpha = rz / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        total_iterations = iteration
        exact_iteration = None
        if (
            not bool(torch.isfinite(solution).all().item())
            or not bool(torch.isfinite(residual).all().item())
            or not math.isfinite(alpha)
        ):
            original_lane_iterations = total_iterations
            original_lane_final_relative = (
                history[-1] if history else initial_relative
            )
            fail_closed(
                "non_finite_original_lane_update",
                stage="original_lane_update",
                details={"attempted_iteration": iteration},
            )
        relative = float(
            (torch.linalg.vector_norm(residual) / right_norm).item()
        )
        if not math.isfinite(relative):
            original_lane_iterations = total_iterations
            original_lane_final_relative = (
                history[-1] if history else initial_relative
            )
            fail_closed(
                "non_finite_original_lane_recurrence_residual",
                stage="original_lane_update",
                details={"attempted_iteration": iteration},
            )
        history.append(relative)
        if relative <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE:
            original_lane_iterations = total_iterations
            original_lane_final_relative = relative
            applied_exact = matvec(solution)
            total_matvec_calls += 1
            exact_residual = right - applied_exact
            exact_finite = (
                bool(torch.isfinite(applied_exact).all().item())
                and bool(torch.isfinite(exact_residual).all().item())
            )
            exact_relative = (
                float(
                    (
                        torch.linalg.vector_norm(exact_residual)
                        / right_norm
                    ).item()
                )
                if exact_finite
                else None
            )
            exact_finite = (
                exact_finite
                and _pcg_json_float(exact_relative) is not None
            )
            exact_iteration = total_iterations if exact_finite else None
            exact_checkpoints.append(
                {
                    "iteration": total_iterations,
                    "relative_normal_residual": (
                        _pcg_json_float(exact_relative)
                    ),
                    "finite": exact_finite,
                }
            )
            original_lane_exact_checked = True
            if (
                exact_finite
                and exact_relative is not None
                and exact_relative
                <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
            ):
                body = {
                    "role": role,
                    "solver": "deterministic_jacobi_pcg_single_tensor",
                    "maximum_iterations": _PCG_MAXIMUM_ITERATIONS,
                    "iterations_executed": iteration,
                    "required_relative_normal_residual": (
                        _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
                    ),
                    "initial_relative_normal_residual": initial_relative,
                    "final_relative_normal_residual": relative,
                    "relative_normal_residual_history": history,
                    "solution_sha256": _tensor_sha256(solution),
                }
                return solution.contiguous(), _seal_observe_pcg_report(
                    "single_tensor",
                    body,
                )
            extension_triggered = True
            break
        next_preconditioned = residual / diagonal
        next_rz = float((residual * next_preconditioned).sum().item())
        if not math.isfinite(next_rz) or next_rz <= 0.0:
            original_lane_iterations = total_iterations
            original_lane_final_relative = relative
            fail_closed(
                "non_positive_or_non_finite_preconditioned_residual",
                stage="original_lane_direction_update",
                details={
                    "attempted_iteration": iteration,
                    "residual_preconditioner_inner_product": (
                        _pcg_json_float(next_rz)
                    ),
                },
            )
        direction = next_preconditioned + (next_rz / rz) * direction
        preconditioned = next_preconditioned
        rz = next_rz

    if not original_lane_exact_checked:
        original_lane_iterations = total_iterations
        original_lane_final_relative = history[-1]
        applied_exact = matvec(solution)
        total_matvec_calls += 1
        exact_residual = right - applied_exact
        exact_finite = (
            bool(torch.isfinite(applied_exact).all().item())
            and bool(torch.isfinite(exact_residual).all().item())
        )
        exact_relative = (
            float(
                (
                    torch.linalg.vector_norm(exact_residual)
                    / right_norm
                ).item()
            )
            if exact_finite
            else None
        )
        exact_finite = (
            exact_finite and _pcg_json_float(exact_relative) is not None
        )
        exact_iteration = total_iterations if exact_finite else None
        exact_checkpoints.append(
            {
                "iteration": total_iterations,
                "relative_normal_residual": _pcg_json_float(exact_relative),
                "finite": exact_finite,
            }
        )
        extension_triggered = True

    if exact_iteration != total_iterations or exact_relative is None:
        fail_closed(
            "non_finite_original_lane_exact_residual",
            stage="original_lane_exact_authentication",
        )
    if exact_relative <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE:
        return solution.contiguous(), _seal_observe_pcg_report(
            "single_tensor",
            extended_body(),
        )

    if restart_row_block_preconditioner is not None:
        row_block_lane_started = True
        lane_matvec_start = total_matvec_calls
        row_block_lane_state.update(
            {
                "start_iteration": total_iterations,
                "planned_stop_iteration": (
                    _PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
                ),
                "iterations_executed": 0,
                "attempted_matvec_calls": 0,
                "exact_diagnostic_matvec_calls": 0,
                "termination": "preconditioner_authentication",
                "ending_iteration": total_iterations,
                "ending_recurrence_relative_normal_residual": (
                    original_lane_final_relative
                ),
                "ending_exact_relative_normal_residual": exact_relative,
                "ending_exact_residual_authenticated": True,
            }
        )
        row_block_preconditioner_state[
            "authentication_attempted"
        ] = True
        try:
            row_block_factor, row_block_receipt = (
                _authenticate_restart_row_block_preconditioner(
                    supplied=restart_row_block_preconditioner,
                    right=right,
                    diagonal=diagonal,
                    role=role,
                )
            )
        except _PcgRowBlockPreconditionerAuthenticationError as exc:
            row_block_preconditioner_state[
                "authentication_failure_reason"
            ] = exc.reason
            row_block_lane_state["termination"] = (
                "preconditioner_authentication_failure"
            )
            fail_closed(
                exc.reason,
                stage="row_block_preconditioner_authentication",
                details=exc.details,
            )
        row_block_preconditioner_state.update(
            {
                "authenticated": True,
                "receipt": row_block_receipt,
            }
        )

        def apply_row_block_preconditioner(value: Any) -> Any:
            row_block_preconditioner_state[
                "application_attempt_count"
            ] += 1
            try:
                result = torch.cholesky_solve(
                    value.unsqueeze(-1),
                    row_block_factor,
                    upper=False,
                ).squeeze(-1)
            except RuntimeError as exc:
                row_block_lane_state["termination"] = (
                    "preconditioner_application_failure"
                )
                fail_closed(
                    "row_block_preconditioner_application_failed",
                    stage="row_block_preconditioner_application",
                    details={"exception_type": type(exc).__name__},
                )
            if not bool(torch.isfinite(result).all().item()):
                row_block_lane_state["termination"] = (
                    "preconditioner_application_failure"
                )
                fail_closed(
                    "row_block_preconditioner_application_is_non_finite",
                    stage="row_block_preconditioner_application",
                )
            row_block_preconditioner_state["applied"] = True
            row_block_preconditioner_state["application_count"] += 1
            return result

        # This is the sole restart.  The exact residual authenticated after
        # the historical scalar lane initializes one continuous block-PCG
        # recurrence; later diagnostic matvecs never replace this state.
        residual = exact_residual
        preconditioned = apply_row_block_preconditioner(residual)
        direction = preconditioned.clone()
        rz = float((residual * preconditioned).sum().item())
        if not math.isfinite(rz) or rz <= 0.0:
            if not math.isfinite(rz):
                exact_iteration = None
            row_block_lane_state["termination"] = (
                "preconditioned_residual_failure"
            )
            fail_closed(
                "non_positive_or_non_finite_preconditioned_residual",
                stage="continuous_row_block_lane_initialization",
                details={
                    "residual_preconditioner_inner_product": (
                        _pcg_json_float(rz)
                    ),
                },
            )
        row_block_lane_state["termination"] = "running"

        for attempted_iteration in range(
            total_iterations + 1,
            _PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS + 1,
        ):
            applied = matvec(direction)
            total_matvec_calls += 1
            row_block_lane_state["attempted_matvec_calls"] = (
                total_matvec_calls - lane_matvec_start
            )
            denominator = float((direction * applied).sum().item())
            applied_finite = bool(torch.isfinite(applied).all().item())
            curvature_finite = math.isfinite(denominator)
            if (
                not applied_finite
                or not curvature_finite
                or denominator <= 0.0
            ):
                if not applied_finite or not curvature_finite:
                    exact_iteration = None
                row_block_lane_state["termination"] = "curvature_failure"
                fail_closed(
                    "non_positive_or_non_finite_curvature",
                    stage="continuous_row_block_lane_curvature",
                    details={
                        "attempted_iteration": attempted_iteration,
                        "curvature": _pcg_json_float(denominator),
                        "curvature_finite_strictly_positive": False,
                        "curvature_matvec_call": total_matvec_calls,
                    },
                )
            alpha = rz / denominator
            solution = solution + alpha * direction
            residual = residual - alpha * applied
            total_iterations += 1
            exact_iteration = None
            row_block_lane_state["iterations_executed"] = (
                total_iterations - original_lane_iterations
            )
            row_block_lane_state["ending_iteration"] = total_iterations
            row_block_lane_state[
                "ending_exact_residual_authenticated"
            ] = False
            row_block_lane_state[
                "ending_exact_relative_normal_residual"
            ] = None
            if (
                not bool(torch.isfinite(solution).all().item())
                or not bool(torch.isfinite(residual).all().item())
                or not math.isfinite(alpha)
            ):
                row_block_lane_state["termination"] = "non_finite_update"
                fail_closed(
                    "non_finite_continuous_row_block_lane_update",
                    stage="continuous_row_block_lane_update",
                    details={"attempted_iteration": attempted_iteration},
                )
            relative = float(
                (torch.linalg.vector_norm(residual) / right_norm).item()
            )
            if not math.isfinite(relative):
                row_block_lane_state["termination"] = (
                    "non_finite_recurrence_residual"
                )
                fail_closed(
                    "non_finite_continuous_row_block_lane_recurrence_residual",
                    stage="continuous_row_block_lane_update",
                    details={"attempted_iteration": attempted_iteration},
                )
            history.append(relative)
            row_block_lane_state[
                "ending_recurrence_relative_normal_residual"
            ] = relative

            if relative <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE:
                applied_exact = matvec(solution)
                total_matvec_calls += 1
                row_block_lane_state["attempted_matvec_calls"] = (
                    total_matvec_calls - lane_matvec_start
                )
                row_block_lane_state[
                    "exact_diagnostic_matvec_calls"
                ] += 1
                exact_residual = right - applied_exact
                exact_finite = (
                    bool(torch.isfinite(applied_exact).all().item())
                    and bool(torch.isfinite(exact_residual).all().item())
                )
                exact_relative = (
                    float(
                        (
                            torch.linalg.vector_norm(exact_residual)
                            / right_norm
                        ).item()
                    )
                    if exact_finite
                    else None
                )
                exact_finite = (
                    exact_finite
                    and _pcg_json_float(exact_relative) is not None
                )
                exact_iteration = (
                    total_iterations if exact_finite else None
                )
                exact_checkpoints.append(
                    {
                        "iteration": total_iterations,
                        "relative_normal_residual": (
                            _pcg_json_float(exact_relative)
                        ),
                        "finite": exact_finite,
                    }
                )
                row_block_lane_state[
                    "ending_exact_relative_normal_residual"
                ] = _pcg_json_float(exact_relative)
                row_block_lane_state[
                    "ending_exact_residual_authenticated"
                ] = exact_finite
                if (
                    exact_finite
                    and exact_relative is not None
                    and exact_relative
                    <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
                ):
                    row_block_lane_state["termination"] = (
                        "exact_tolerance_reached"
                    )
                    return solution.contiguous(), _seal_observe_pcg_report(
                        "single_tensor",
                        extended_body(),
                    )
                if not exact_finite:
                    row_block_lane_state["termination"] = (
                        "non_finite_exact_diagnostic"
                    )
                    fail_closed(
                        "non_finite_continuous_row_block_lane_exact_residual",
                        stage="continuous_row_block_lane_exact_diagnostic",
                    )

            if (
                total_iterations
                >= _PCG_ROW_BLOCK_HARD_MAXIMUM_ITERATIONS
            ):
                break
            next_preconditioned = apply_row_block_preconditioner(residual)
            next_rz = float(
                (residual * next_preconditioned).sum().item()
            )
            if not math.isfinite(next_rz) or next_rz <= 0.0:
                row_block_lane_state["termination"] = (
                    "preconditioned_residual_failure"
                )
                fail_closed(
                    "non_positive_or_non_finite_preconditioned_residual",
                    stage="continuous_row_block_lane_direction_update",
                    details={
                        "attempted_iteration": attempted_iteration,
                        "residual_preconditioner_inner_product": (
                            _pcg_json_float(next_rz)
                        ),
                    },
                )
            direction = (
                next_preconditioned + (next_rz / rz) * direction
            )
            preconditioned = next_preconditioned
            rz = next_rz

        if exact_iteration != total_iterations:
            applied_exact = matvec(solution)
            total_matvec_calls += 1
            row_block_lane_state["attempted_matvec_calls"] = (
                total_matvec_calls - lane_matvec_start
            )
            row_block_lane_state["exact_diagnostic_matvec_calls"] += 1
            exact_residual = right - applied_exact
            exact_finite = (
                bool(torch.isfinite(applied_exact).all().item())
                and bool(torch.isfinite(exact_residual).all().item())
            )
            exact_relative = (
                float(
                    (
                        torch.linalg.vector_norm(exact_residual)
                        / right_norm
                    ).item()
                )
                if exact_finite
                else None
            )
            exact_finite = (
                exact_finite
                and _pcg_json_float(exact_relative) is not None
            )
            exact_iteration = total_iterations if exact_finite else None
            exact_checkpoints.append(
                {
                    "iteration": total_iterations,
                    "relative_normal_residual": (
                        _pcg_json_float(exact_relative)
                    ),
                    "finite": exact_finite,
                }
            )
            row_block_lane_state[
                "ending_exact_relative_normal_residual"
            ] = _pcg_json_float(exact_relative)
            row_block_lane_state[
                "ending_exact_residual_authenticated"
            ] = exact_finite
        if exact_iteration != total_iterations or exact_relative is None:
            row_block_lane_state["termination"] = (
                "non_finite_final_exact_residual"
            )
            fail_closed(
                "non_finite_continuous_row_block_lane_exact_residual",
                stage="continuous_row_block_lane_final_exact_authentication",
            )
        if exact_relative <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE:
            row_block_lane_state["termination"] = (
                "exact_tolerance_reached_at_hard_cap"
            )
            return solution.contiguous(), _seal_observe_pcg_report(
                "single_tensor",
                extended_body(),
            )
        row_block_lane_state["termination"] = "hard_iteration_cap_reached"
        fail_closed(
            "hard_iteration_cap_reached",
            stage="continuous_row_block_lane_hard_iteration_cap",
        )

    while total_iterations < _PCG_HARD_MAXIMUM_ITERATIONS:
        block_start = total_iterations
        block_stop = min(
            block_start + _PCG_RESTART_BLOCK_ITERATIONS,
            _PCG_HARD_MAXIMUM_ITERATIONS,
        )
        residual = exact_residual
        preconditioned = residual / diagonal
        direction = preconditioned.clone()
        rz = float((residual * preconditioned).sum().item())
        if not math.isfinite(rz) or rz <= 0.0:
            if not math.isfinite(rz):
                exact_iteration = None
            fail_closed(
                "non_positive_or_non_finite_preconditioned_residual",
                stage="restart_initialization",
                details={
                    "restart_iteration": block_start,
                    "residual_preconditioner_inner_product": (
                        _pcg_json_float(rz)
                    ),
                },
            )
        block_history_start = len(history)
        block_matvec_start = total_matvec_calls
        exact_checkpoint_taken = False

        for attempted_iteration in range(block_start + 1, block_stop + 1):
            applied = matvec(direction)
            total_matvec_calls += 1
            denominator = float((direction * applied).sum().item())
            applied_finite = bool(torch.isfinite(applied).all().item())
            curvature_finite = math.isfinite(denominator)
            if (
                not applied_finite
                or not curvature_finite
                or denominator <= 0.0
            ):
                if not applied_finite or not curvature_finite:
                    exact_iteration = None
                exact_is_current = exact_iteration == total_iterations
                restart_schedule.append(
                    {
                        "restart_iteration": block_start,
                        "planned_stop_iteration": block_stop,
                        "iterations_executed": (
                            len(history) - block_history_start
                        ),
                        "ending_iteration": total_iterations,
                        "attempted_matvec_calls": (
                            total_matvec_calls - block_matvec_start
                        ),
                        "termination": "curvature_failure",
                        "ending_recurrence_relative_normal_residual": (
                            history[-1]
                        ),
                        "ending_exact_relative_normal_residual": (
                            _pcg_json_float(exact_relative)
                            if exact_is_current
                            else None
                        ),
                        "ending_exact_residual_authenticated": exact_is_current,
                    }
                )
                fail_closed(
                    "non_positive_or_non_finite_curvature",
                    stage="restart_curvature",
                    details={
                        "attempted_iteration": attempted_iteration,
                        "curvature": _pcg_json_float(denominator),
                        "curvature_finite_strictly_positive": False,
                        "curvature_matvec_call": total_matvec_calls,
                    },
                )
            alpha = rz / denominator
            solution = solution + alpha * direction
            residual = residual - alpha * applied
            total_iterations += 1
            exact_iteration = None
            if (
                not bool(torch.isfinite(solution).all().item())
                or not bool(torch.isfinite(residual).all().item())
                or not math.isfinite(alpha)
            ):
                fail_closed(
                    "non_finite_restart_update",
                    stage="restart_update",
                    details={"attempted_iteration": attempted_iteration},
                )
            relative = float(
                (torch.linalg.vector_norm(residual) / right_norm).item()
            )
            if not math.isfinite(relative):
                fail_closed(
                    "non_finite_restart_recurrence_residual",
                    stage="restart_update",
                    details={"attempted_iteration": attempted_iteration},
                )
            history.append(relative)
            if relative <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE:
                applied_exact = matvec(solution)
                total_matvec_calls += 1
                exact_residual = right - applied_exact
                exact_finite = (
                    bool(torch.isfinite(applied_exact).all().item())
                    and bool(torch.isfinite(exact_residual).all().item())
                )
                exact_relative = (
                    float(
                        (
                            torch.linalg.vector_norm(exact_residual)
                            / right_norm
                        ).item()
                    )
                    if exact_finite
                    else None
                )
                exact_finite = (
                    exact_finite
                    and _pcg_json_float(exact_relative) is not None
                )
                exact_iteration = total_iterations if exact_finite else None
                exact_checkpoints.append(
                    {
                        "iteration": total_iterations,
                        "relative_normal_residual": (
                            _pcg_json_float(exact_relative)
                        ),
                        "finite": exact_finite,
                    }
                )
                exact_checkpoint_taken = True
                break
            next_preconditioned = residual / diagonal
            next_rz = float((residual * next_preconditioned).sum().item())
            if not math.isfinite(next_rz) or next_rz <= 0.0:
                fail_closed(
                    "non_positive_or_non_finite_preconditioned_residual",
                    stage="restart_direction_update",
                    details={
                        "attempted_iteration": attempted_iteration,
                        "residual_preconditioner_inner_product": (
                            _pcg_json_float(next_rz)
                        ),
                    },
                )
            direction = next_preconditioned + (next_rz / rz) * direction
            preconditioned = next_preconditioned
            rz = next_rz

        if not exact_checkpoint_taken:
            applied_exact = matvec(solution)
            total_matvec_calls += 1
            exact_residual = right - applied_exact
            exact_finite = (
                bool(torch.isfinite(applied_exact).all().item())
                and bool(torch.isfinite(exact_residual).all().item())
            )
            exact_relative = (
                float(
                    (
                        torch.linalg.vector_norm(exact_residual)
                        / right_norm
                    ).item()
                )
                if exact_finite
                else None
            )
            exact_finite = (
                exact_finite and _pcg_json_float(exact_relative) is not None
            )
            exact_iteration = total_iterations if exact_finite else None
            exact_checkpoints.append(
                {
                    "iteration": total_iterations,
                    "relative_normal_residual": _pcg_json_float(exact_relative),
                    "finite": exact_finite,
                }
            )

        exact_authenticated = exact_iteration == total_iterations
        restart_schedule.append(
            {
                "restart_iteration": block_start,
                "planned_stop_iteration": block_stop,
                "iterations_executed": len(history) - block_history_start,
                "ending_iteration": total_iterations,
                "attempted_matvec_calls": (
                    total_matvec_calls - block_matvec_start
                ),
                "termination": (
                    "exact_tolerance_reached"
                    if exact_authenticated
                    and exact_relative is not None
                    and exact_relative
                    <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
                    else "block_boundary"
                ),
                "ending_recurrence_relative_normal_residual": history[-1],
                "ending_exact_relative_normal_residual": (
                    _pcg_json_float(exact_relative)
                    if exact_authenticated
                    else None
                ),
                "ending_exact_residual_authenticated": exact_authenticated,
            }
        )
        if not exact_authenticated or exact_relative is None:
            fail_closed(
                "non_finite_restart_exact_residual",
                stage="restart_exact_authentication",
            )
        if exact_relative <= _PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE:
            return solution.contiguous(), _seal_observe_pcg_report(
                "single_tensor",
                extended_body(),
            )

    fail_closed(
        "hard_iteration_cap_reached",
        stage="hard_iteration_cap",
    )
    raise AssertionError("unreachable")


def _canonicalize_value_factor(
    value_factor: Any,
    output_blocks: Any,
) -> tuple[Any, Any]:
    factors = value_factor.clone()
    blocks = output_blocks.clone()
    groups, _hidden, head_dim = map(int, factors.shape)
    heads_per_group = int(blocks.shape[0]) // groups
    for group in range(groups):
        head_slice = slice(
            group * heads_per_group,
            (group + 1) * heads_per_group,
        )
        for coordinate in range(head_dim):
            pivot = int(factors[group, :, coordinate].abs().argmax().item())
            if float(factors[group, pivot, coordinate].item()) < 0.0:
                factors[group, :, coordinate] *= -1.0
                blocks[head_slice, coordinate, :] *= -1.0
    return factors.contiguous(), blocks.contiguous()


def _context_heads(
    attended_inputs: Any,
    value_factor: Any,
    *,
    heads_per_group: int,
) -> Any:
    torch = _torch()
    return torch.stack(
        [
            attended_inputs[:, head]
            @ value_factor[head // heads_per_group]
            for head in range(int(attended_inputs.shape[1]))
        ],
        dim=1,
    )


def _fit_output(
    attended_inputs: Any,
    target: Any,
    value_factor: Any,
    *,
    heads_per_group: int,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    role: str,
) -> tuple[Any, Any, dict[str, Any]]:
    contexts = _context_heads(
        attended_inputs,
        value_factor,
        heads_per_group=heads_per_group,
    )
    features = contexts.reshape(
        int(contexts.shape[0]),
        int(contexts.shape[1]) * int(contexts.shape[2]),
    )
    output_transpose, report = _ridge_solve(
        features,
        target,
        policy=policy,
        role=role,
    )
    output_blocks = output_transpose.reshape(
        int(contexts.shape[1]),
        int(contexts.shape[2]),
        int(target.shape[1]),
    )
    return output_blocks.contiguous(), features, report


def _prediction(
    attended_inputs: Any,
    value_factor: Any,
    output_blocks: Any,
    *,
    heads_per_group: int,
) -> Any:
    contexts = _context_heads(
        attended_inputs,
        value_factor,
        heads_per_group=heads_per_group,
    )
    return _torch().einsum("nhd,hdo->no", contexts, output_blocks)


def _initial_shared_value_factors(
    effective: Any,
    *,
    target_key_value_heads: int,
    heads_per_group: int,
    head_dim: int,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    torch = _torch()
    value_groups = []
    output_blocks = []
    reports = []
    for group in range(target_key_value_heads):
        head_slice = slice(
            group * heads_per_group,
            (group + 1) * heads_per_group,
        )
        concatenated = torch.cat(
            tuple(effective[head_slice]),
            dim=1,
        )
        left, singular, right = torch.linalg.svd(
            concatenated,
            full_matrices=False,
        )
        if int(singular.numel()) < head_dim:
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "The effective operator cannot support the target head dimension."
            )
        root = singular[:head_dim].clamp_min(0.0).sqrt()
        value_group = left[:, :head_dim] * root.unsqueeze(0)
        output_group = (
            root.unsqueeze(1) * right[:head_dim]
        ).reshape(
            head_dim,
            heads_per_group,
            int(effective.shape[-1]),
        ).permute(1, 0, 2)
        value_groups.append(value_group)
        output_blocks.append(output_group)
        reports.append(
            {
                "group": group,
                "singular_values": [
                    float(value) for value in singular[:head_dim].tolist()
                ],
                "retained_rank": head_dim,
                "concatenated_effective_operator_sha256": (
                    _tensor_sha256(concatenated)
                ),
            }
        )
    factors = torch.stack(value_groups, dim=0)
    blocks = torch.cat(output_blocks, dim=0)
    factors, blocks = _canonicalize_value_factor(factors, blocks)
    return factors, blocks, reports


def _block_value_sweep(
    attended_inputs: Any,
    target: Any,
    value_factor: Any,
    output_blocks: Any,
    *,
    heads_per_group: int,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    iteration: int,
) -> tuple[Any, list[dict[str, Any]]]:
    torch = _torch()
    updated = value_factor.clone()
    reports = []
    group_count = int(updated.shape[0])
    for group in range(group_count):
        prediction = _prediction(
            attended_inputs,
            updated,
            output_blocks,
            heads_per_group=heads_per_group,
        )
        start = group * heads_per_group
        stop = start + heads_per_group
        group_context = torch.stack(
            [
                attended_inputs[:, head] @ updated[group]
                for head in range(start, stop)
            ],
            dim=1,
        )
        group_prediction = torch.einsum(
            "nhd,hdo->no",
            group_context,
            output_blocks[start:stop],
        )
        residual = target - (prediction - group_prediction)
        coefficients = torch.einsum(
            "nhi,hdo->noid",
            attended_inputs[:, start:stop],
            output_blocks[start:stop],
        ).reshape(
            int(target.shape[0]) * int(target.shape[1]),
            int(attended_inputs.shape[2]) * int(updated.shape[2]),
        )
        solution, solve_report = _ridge_solve(
            coefficients,
            residual.reshape(-1, 1),
            policy=policy,
            role=f"sequence_effective_value_iteration_{iteration}_group_{group}",
        )
        updated[group] = solution.reshape(
            int(attended_inputs.shape[2]),
            int(updated.shape[2]),
        )
        reports.append(solve_report)
    return updated.contiguous(), reports


def _effective_operator_value_sweep(
    effective: Any,
    value_factor: Any,
    output_blocks: Any,
    *,
    heads_per_group: int,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    iteration: int,
) -> tuple[Any, list[dict[str, Any]]]:
    """Update shared V against headwise effective operators without row tensors."""

    updated = value_factor.clone()
    reports = []
    for group in range(int(updated.shape[0])):
        start = group * heads_per_group
        stop = start + heads_per_group
        effective_group = _torch().cat(
            tuple(effective[start:stop]),
            dim=1,
        )
        output_group = _torch().cat(
            tuple(output_blocks[start:stop]),
            dim=1,
        )
        value_transpose, solve_report = _ridge_solve(
            output_group.T,
            effective_group.T,
            policy=policy,
            role=(
                f"sequence_effective_streamed_value_iteration_{iteration}_"
                f"group_{group}"
            ),
        )
        updated[group] = value_transpose.T
        reports.append(solve_report)
    return updated.contiguous(), reports


def _producer_value_anchor(
    value_factor: Any,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    *,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
) -> tuple[Any, list[dict[str, Any]]]:
    torch = _torch()
    topology = producer.topology
    anchored = value_factor.clone()
    reports = []
    producer_value = producer.value_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    for group in range(topology.target_key_value_heads):
        value_slice = slice(
            group * topology.head_dim,
            (group + 1) * topology.head_dim,
        )
        anchor = producer_value[value_slice].T.contiguous()
        gram = anchor.T @ anchor + policy.absolute_ridge_floor * torch.eye(
            topology.head_dim,
            dtype=torch.float64,
        )
        alignment = torch.linalg.solve(
            gram,
            anchor.T @ anchored[group],
        )
        aligned_anchor = anchor @ alignment
        anchored[group] = (
            (1.0 - policy.value_anchor_fraction) * anchored[group]
            + policy.value_anchor_fraction * aligned_anchor
        )
        reports.append(
            {
                "group": group,
                "anchor_fraction": policy.value_anchor_fraction,
                "producer_value_group_sha256": _tensor_sha256(anchor),
                "alignment_sha256": _tensor_sha256(alignment),
                "anchored_value_group_sha256": _tensor_sha256(anchored[group]),
            }
        )
    return anchored.contiguous(), reports


def _fit_context_preservation(
    *,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    normalized_cases: Sequence[Mapping[str, Any]],
    fit_cases: Sequence[Lfm2CachedAttentionFoldCase],
    materialized_value_factor: Any,
) -> float:
    torch = _torch()
    topology = producer.topology
    predictions = []
    targets = []
    for normalized, case in zip(normalized_cases, fit_cases, strict=True):
        child_heads = _context_heads(
            normalized["weighted_inputs"],
            materialized_value_factor,
            heads_per_group=topology.target_query_heads_per_group,
        )
        parent_context = case.parent_context.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        target_heads = []
        for members, weights in zip(
            producer.query_groups,
            producer.query_barycentric_weights,
            strict=True,
        ):
            selected = parent_context.index_select(
                1,
                torch.tensor(members, dtype=torch.long),
            )
            coefficients = torch.tensor(
                weights,
                dtype=torch.float64,
            ).reshape(1, len(members), 1, 1)
            target_heads.append((selected * coefficients).sum(dim=1))
        target = torch.stack(target_heads, dim=1)
        predictions.append(child_heads.reshape(-1, topology.head_dim))
        targets.append(target.reshape(-1, topology.head_dim))
    return _relative_rmse(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


def _case_query_row_weights(
    case: Lfm2CachedAttentionFoldCase,
    *,
    case_index: int,
    batch: int,
    query_length: int,
) -> tuple[Any, bool]:
    torch = _torch()
    if case.query_row_sqrt_weights is None:
        return (
            torch.ones(batch, query_length, dtype=torch.float64),
            False,
        )
    weights = _tensor(
        case.query_row_sqrt_weights,
        f"fit_cases[{case_index}].query_row_sqrt_weights",
        dimensions=2,
    ).to(torch.float64)
    if tuple(weights.shape) != (batch, query_length) or bool(
        (weights <= 0.0).any().item()
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "Fit row weights must be positive and query-aligned."
        )
    return weights, True


def _normalize_sequence_case(
    case: Lfm2CachedAttentionFoldCase,
    *,
    case_index: int,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    query_norm_weight: Any | None,
    key_norm_weight: Any | None,
    norm_epsilon: float,
) -> Mapping[str, Any]:
    try:
        return structural._normalize_attention_value_output_case(
            case,
            case_index=case_index,
            producer=producer,
            query_norm_weight=query_norm_weight,
            key_norm_weight=key_norm_weight,
            normalized_query_norm_weight=query_norm_weight,
            normalized_key_norm_weight=key_norm_weight,
            norm_epsilon=norm_epsilon,
        )
    except structural.Lfm2StructuralResidualAttentionFoldError as exc:
        raise Lfm2SequenceEffectiveAttentionFoldError(str(exc)) from exc


def _accumulate_streamed_sequence_statistics(
    *,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    query_norm_weight: Any | None,
    key_norm_weight: Any | None,
    norm_epsilon: float,
) -> tuple[Mapping[str, Any], dict[str, Any], int]:
    torch = _torch()
    topology = producer.topology
    cross = torch.zeros(
        topology.target_query_heads,
        topology.target_query_heads,
        topology.target_hidden_size,
        topology.target_hidden_size,
        dtype=torch.float64,
    )
    right = torch.zeros(
        topology.target_query_heads,
        topology.target_hidden_size,
        topology.target_output_size,
        dtype=torch.float64,
    )
    target_square_sum = 0.0
    row_count = 0
    explicit_weight_count = 0
    chunk_count = 0
    for case_index, case in enumerate(cases):
        normalized = _normalize_sequence_case(
            case,
            case_index=case_index,
            producer=producer,
            query_norm_weight=query_norm_weight,
            key_norm_weight=key_norm_weight,
            norm_epsilon=norm_epsilon,
        )
        batch = int(normalized["batch"])
        query_length = int(normalized["query_length"])
        attended = normalized["weighted_inputs"].permute(
            0,
            2,
            1,
            3,
        ).reshape(
            batch * query_length,
            topology.target_query_heads,
            topology.target_hidden_size,
        )
        row_weights, explicit = _case_query_row_weights(
            case,
            case_index=case_index,
            batch=batch,
            query_length=query_length,
        )
        explicit_weight_count += int(explicit)
        weight = row_weights.reshape(-1, 1)
        weighted_attended = attended * weight.unsqueeze(1)
        weighted_target = normalized["endpoint"] * weight
        for start in range(
            0,
            int(weighted_attended.shape[0]),
            _STATISTICS_ROW_CHUNK_SIZE,
        ):
            stop = min(
                start + _STATISTICS_ROW_CHUNK_SIZE,
                int(weighted_attended.shape[0]),
            )
            design = weighted_attended[start:stop]
            target = weighted_target[start:stop]
            for head in range(topology.target_query_heads):
                right[head].add_(design[:, head].T @ target)
                for other in range(head, topology.target_query_heads):
                    block = design[:, head].T @ design[:, other]
                    cross[head, other].add_(block)
                    if other != head:
                        cross[other, head].add_(block.T)
            target_square_sum += float(target.square().sum().item())
            row_count += int(target.shape[0])
            chunk_count += 1
    statistics = MappingProxyType(
        {
            "cross_head_gram": cross,
            "endpoint_right": right,
            "endpoint_square_sum": target_square_sum,
            "row_count": row_count,
        }
    )
    body = {
        "role": "streamed_sequence_effective_sufficient_statistics",
        "row_count": row_count,
        "chunk_count": chunk_count,
        "row_chunk_size": _STATISTICS_ROW_CHUNK_SIZE,
        "cross_head_gram_shape": list(cross.shape),
        "endpoint_right_shape": list(right.shape),
        "cross_head_gram_sha256": _tensor_sha256(cross),
        "endpoint_right_sha256": _tensor_sha256(right),
        "endpoint_square_sum": target_square_sum,
        "accumulation_dtype": "torch.float64",
        "full_cross_head_symmetry_materialized": True,
        "row_design_released_after_each_case": True,
        "forbidden_value_coefficient_tensor_materialized": False,
        "forbidden_hidden_head_dim_gram_materialized": False,
    }
    return statistics, _seal(body), explicit_weight_count


def _fit_streamed_global_effective_operator(
    *,
    statistics: Mapping[str, Any],
    topology: Any,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    cross = statistics["cross_head_gram"]
    right = statistics["endpoint_right"]
    feature_count = topology.target_query_heads * topology.target_hidden_size
    trace_scale = (
        sum(
            float(torch.trace(cross[head, head]).item())
            for head in range(topology.target_query_heads)
        )
        / feature_count
    )
    ridge = max(
        policy.absolute_ridge_floor,
        policy.ridge_scale * max(trace_scale, 0.0),
    )
    diagonal = torch.stack(
        [
            torch.diagonal(cross[head, head]) + ridge
            for head in range(topology.target_query_heads)
        ],
        dim=0,
    ).clamp_min(policy.absolute_ridge_floor)

    def matvec(value: Any) -> Any:
        result = torch.zeros_like(value)
        for head in range(topology.target_query_heads):
            for other in range(topology.target_query_heads):
                result[head].add_(cross[head, other] @ value[other])
        return result + ridge * value

    solution, pcg_report = _pcg_multi_rhs(
        right=right,
        matvec=matvec,
        diagonal=diagonal,
        role="sequence_effective_complete_causal_operator_pcg",
    )
    applied = matvec(solution)
    normal_residual = float(
        (
            torch.linalg.vector_norm(applied - right)
            / torch.linalg.vector_norm(right).clamp_min(
                torch.finfo(torch.float64).eps
            )
        ).item()
    )
    body = {
        "role": "sequence_effective_complete_causal_operator",
        "rows": statistics["row_count"],
        "features": feature_count,
        "outputs": topology.target_output_size,
        "system_domain": "streamed_cross_head_statistics_pcg",
        "linear_system_dimension": feature_count,
        "ridge_scale": policy.ridge_scale,
        "trace_scale": trace_scale,
        "ridge": ridge,
        "relative_normal_residual": normal_residual,
        "pcg_report": pcg_report,
        "solution_sha256": _tensor_sha256(solution),
    }
    return solution.contiguous(), _seal(body)


def _fit_streamed_headwise_effective_operator(
    *,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    query_norm_weight: Any | None,
    key_norm_weight: Any | None,
    norm_epsilon: float,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
) -> tuple[Any, dict[str, Any], int]:
    """Fit target-head contribution operators from streamed sufficient stats."""

    torch = _torch()
    topology = producer.topology
    grams = [
        torch.zeros(
            topology.target_hidden_size,
            topology.target_hidden_size,
            dtype=torch.float64,
        )
        for _head in range(topology.target_query_heads)
    ]
    rights = [
        torch.zeros(
            topology.target_hidden_size,
            topology.target_output_size,
            dtype=torch.float64,
        )
        for _head in range(topology.target_query_heads)
    ]
    target_square_sums = [0.0] * topology.target_query_heads
    row_counts = [0] * topology.target_query_heads
    explicit_weight_count = 0
    for case_index, case in enumerate(cases):
        normalized = _normalize_sequence_case(
            case,
            case_index=case_index,
            producer=producer,
            query_norm_weight=query_norm_weight,
            key_norm_weight=key_norm_weight,
            norm_epsilon=norm_epsilon,
        )
        batch = int(normalized["batch"])
        query_length = int(normalized["query_length"])
        attended = normalized["weighted_inputs"].permute(
            0,
            2,
            1,
            3,
        ).reshape(
            batch * query_length,
            topology.target_query_heads,
            topology.target_hidden_size,
        )
        row_weights, explicit = _case_query_row_weights(
            case,
            case_index=case_index,
            batch=batch,
            query_length=query_length,
        )
        explicit_weight_count += int(explicit)
        contribution = _tensor(
            case.parent_contribution,
            f"fit_cases[{case_index}].parent_contribution",
            dimensions=4,
        ).to(torch.float64)
        if tuple(contribution.shape) != (
            batch,
            topology.source_query_heads,
            query_length,
            topology.target_output_size,
        ):
            raise Lfm2SequenceEffectiveAttentionFoldError(
                "Streamed headwise ridge requires complete target-space "
                "parent attention-head contributions."
            )
        weight = row_weights.reshape(-1, 1)
        for target_head, members in enumerate(producer.query_groups):
            index = torch.tensor(members, dtype=torch.long)
            target = contribution.index_select(1, index).sum(dim=1).reshape(
                batch * query_length,
                topology.target_output_size,
            )
            design = attended[:, target_head]
            weighted_design = design * weight
            weighted_target = target * weight
            grams[target_head].add_(weighted_design.T @ weighted_design)
            rights[target_head].add_(weighted_design.T @ weighted_target)
            target_square_sums[target_head] += float(
                weighted_target.square().sum().item()
            )
            row_counts[target_head] += int(weighted_design.shape[0])
    effective = []
    solve_reports = []
    for head in range(topology.target_query_heads):
        solution, report = _ridge_solve_statistics(
            grams[head],
            rights[head],
            target_square_sum=target_square_sums[head],
            rows=row_counts[head],
            policy=policy,
            role=f"sequence_effective_streamed_head_{head}",
        )
        effective.append(solution)
        solve_reports.append(report)
    body = {
        "role": "sequence_effective_streamed_headwise_contribution_operator",
        "head_count": topology.target_query_heads,
        "head_solve_report_sha256": [
            report["report_sha256"] for report in solve_reports
        ],
        "largest_linear_system_dimension": topology.target_hidden_size,
        "complete_parent_contribution_partition": True,
        "source_head_groups_partitioned_without_drop": (
            sorted(
                member
                for group in producer.query_groups
                for member in group
            )
            == list(range(topology.source_query_heads))
        ),
        "global_hq_times_hidden_gram_materialized": False,
        "row_design_materialized_across_cases": False,
    }
    return (
        torch.stack(effective, dim=0).contiguous(),
        _seal(body),
        explicit_weight_count,
    )


def _fit_output_streamed(
    *,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    query_norm_weight: Any | None,
    key_norm_weight: Any | None,
    norm_epsilon: float,
    value_factor: Any,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit O from streamed Hq*head_dim context sufficient statistics."""

    torch = _torch()
    topology = producer.topology
    feature_count = topology.target_query_heads * topology.head_dim
    gram = torch.zeros(feature_count, feature_count, dtype=torch.float64)
    right = torch.zeros(
        feature_count,
        topology.target_output_size,
        dtype=torch.float64,
    )
    target_square_sum = 0.0
    row_count = 0
    for case_index, case in enumerate(cases):
        normalized = _normalize_sequence_case(
            case,
            case_index=case_index,
            producer=producer,
            query_norm_weight=query_norm_weight,
            key_norm_weight=key_norm_weight,
            norm_epsilon=norm_epsilon,
        )
        batch = int(normalized["batch"])
        query_length = int(normalized["query_length"])
        attended = normalized["weighted_inputs"].permute(
            0,
            2,
            1,
            3,
        ).reshape(
            batch * query_length,
            topology.target_query_heads,
            topology.target_hidden_size,
        )
        features = _context_heads(
            attended,
            value_factor,
            heads_per_group=topology.target_query_heads_per_group,
        ).reshape(batch * query_length, feature_count)
        row_weights, _explicit = _case_query_row_weights(
            case,
            case_index=case_index,
            batch=batch,
            query_length=query_length,
        )
        weight = row_weights.reshape(-1, 1)
        weighted_features = features * weight
        weighted_target = normalized["endpoint"] * weight
        gram.add_(weighted_features.T @ weighted_features)
        right.add_(weighted_features.T @ weighted_target)
        target_square_sum += float(weighted_target.square().sum().item())
        row_count += int(weighted_features.shape[0])
    output_transpose, report = _ridge_solve_statistics(
        gram,
        right,
        target_square_sum=target_square_sum,
        rows=row_count,
        policy=policy,
        role=role,
    )
    return (
        output_transpose.reshape(
            topology.target_query_heads,
            topology.head_dim,
            topology.target_output_size,
        ).contiguous(),
        report,
    )


def _fit_output_from_sequence_statistics(
    *,
    statistics: Mapping[str, Any],
    topology: Any,
    value_factor: Any,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    cross = statistics["cross_head_gram"]
    right = statistics["endpoint_right"]
    feature_count = topology.target_query_heads * topology.head_dim
    gram = torch.zeros(feature_count, feature_count, dtype=torch.float64)
    output_right = torch.zeros(
        feature_count,
        topology.target_output_size,
        dtype=torch.float64,
    )
    for head in range(topology.target_query_heads):
        group = head // topology.target_query_heads_per_group
        head_slice = slice(
            head * topology.head_dim,
            (head + 1) * topology.head_dim,
        )
        output_right[head_slice] = value_factor[group].T @ right[head]
        for other in range(topology.target_query_heads):
            other_group = other // topology.target_query_heads_per_group
            other_slice = slice(
                other * topology.head_dim,
                (other + 1) * topology.head_dim,
            )
            gram[head_slice, other_slice] = (
                value_factor[group].T
                @ cross[head, other]
                @ value_factor[other_group]
            )
    output_transpose, report = _ridge_solve_statistics(
        gram,
        output_right,
        target_square_sum=float(statistics["endpoint_square_sum"]),
        rows=int(statistics["row_count"]),
        policy=policy,
        role=role,
    )
    return (
        output_transpose.reshape(
            topology.target_query_heads,
            topology.head_dim,
            topology.target_output_size,
        ).contiguous(),
        report,
    )


def _structured_value_pcg_sweep(
    *,
    statistics: Mapping[str, Any],
    topology: Any,
    value_factor: Any,
    output_blocks: Any,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    iteration: int,
) -> tuple[Any, list[dict[str, Any]]]:
    torch = _torch()
    cross = statistics["cross_head_gram"]
    endpoint_right = statistics["endpoint_right"]
    updated = value_factor.clone()
    reports = []
    for group in range(topology.target_key_value_heads):
        start = group * topology.target_query_heads_per_group
        stop = start + topology.target_query_heads_per_group
        group_heads = tuple(range(start, stop))
        other_heads = tuple(
            head
            for head in range(topology.target_query_heads)
            if head not in group_heads
        )
        right = torch.zeros(
            topology.target_hidden_size,
            topology.head_dim,
            dtype=torch.float64,
        )
        for head in group_heads:
            right.add_(endpoint_right[head] @ output_blocks[head].T)
            for other in other_heads:
                other_group = (
                    other // topology.target_query_heads_per_group
                )
                coupling = output_blocks[other] @ output_blocks[head].T
                right.sub_(
                    cross[head, other]
                    @ updated[other_group]
                    @ coupling
                )
        trace_numerator = 0.0
        diagonal = torch.zeros_like(updated[group])
        for head in group_heads:
            for other in group_heads:
                coupling = output_blocks[other] @ output_blocks[head].T
                trace_numerator += float(
                    torch.trace(cross[head, other]).item()
                ) * float((output_blocks[head] * output_blocks[other]).sum())
                diagonal.add_(
                    torch.diagonal(cross[head, other]).unsqueeze(1)
                    * torch.diagonal(coupling).unsqueeze(0)
                )
        trace_scale = trace_numerator / (
            topology.target_hidden_size * topology.head_dim
        )
        ridge = max(
            policy.absolute_ridge_floor,
            policy.ridge_scale * max(trace_scale, 0.0),
        )
        diagonal = (diagonal + ridge).clamp_min(
            policy.absolute_ridge_floor
        )

        def matvec(value: Any) -> Any:
            result = torch.zeros_like(value)
            for head in group_heads:
                for other in group_heads:
                    coupling = output_blocks[other] @ output_blocks[head].T
                    result.add_(
                        cross[head, other] @ value @ coupling
                    )
            return result + ridge * value

        role = (
            f"sequence_effective_structured_value_iteration_{iteration}_"
            f"group_{group}"
        )
        row_block_preconditioner = (
            _build_structured_value_row_block_jacobi_preconditioner(
                cross=cross,
                output_blocks=output_blocks,
                group_heads=group_heads,
                diagonal=diagonal,
                ridge=ridge,
                role=role,
            )
        )
        solution, pcg_report = _pcg_single_tensor(
            right=right,
            matvec=matvec,
            diagonal=diagonal,
            initial=updated[group],
            role=role,
            restart_row_block_preconditioner=row_block_preconditioner,
        )
        updated[group] = solution
        body = {
            "role": role,
            "group": group,
            "heads": list(group_heads),
            "features": topology.target_hidden_size * topology.head_dim,
            "ridge_scale": policy.ridge_scale,
            "trace_scale": trace_scale,
            "ridge": ridge,
            "pcg_report": pcg_report,
            "row_block_jacobi_construction_report": dict(
                row_block_preconditioner.construction_report
            ),
            "row_block_jacobi_supplied_to_restart_lane": True,
            "solution_sha256": _tensor_sha256(solution),
            "explicit_coefficient_tensor_materialized": False,
            "hidden_head_dim_gram_materialized": False,
        }
        reports.append(_seal(body))
    return updated.contiguous(), reports


def _streaming_materialized_metrics(
    *,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    query_norm_weight: Any | None,
    key_norm_weight: Any | None,
    norm_epsilon: float,
    value_factor: Any,
    output_blocks: Any,
) -> tuple[float, float, float]:
    torch = _torch()
    topology = producer.topology
    weighted_error = weighted_target_square = 0.0
    plain_error = plain_target_square = 0.0
    context_error = context_target_square = 0.0
    for case_index, case in enumerate(cases):
        normalized = _normalize_sequence_case(
            case,
            case_index=case_index,
            producer=producer,
            query_norm_weight=query_norm_weight,
            key_norm_weight=key_norm_weight,
            norm_epsilon=norm_epsilon,
        )
        batch = int(normalized["batch"])
        query_length = int(normalized["query_length"])
        attended = normalized["weighted_inputs"].permute(
            0,
            2,
            1,
            3,
        ).reshape(
            batch * query_length,
            topology.target_query_heads,
            topology.target_hidden_size,
        )
        prediction = _prediction(
            attended,
            value_factor,
            output_blocks,
            heads_per_group=topology.target_query_heads_per_group,
        )
        target = normalized["endpoint"]
        row_weights, _explicit = _case_query_row_weights(
            case,
            case_index=case_index,
            batch=batch,
            query_length=query_length,
        )
        weight = row_weights.reshape(-1, 1)
        weighted_error += float(((prediction - target) * weight).square().sum())
        weighted_target_square += float((target * weight).square().sum())
        plain_error += float((prediction - target).square().sum())
        plain_target_square += float(target.square().sum())

        child_heads = _context_heads(
            attended,
            value_factor,
            heads_per_group=topology.target_query_heads_per_group,
        ).reshape(
            batch,
            query_length,
            topology.target_query_heads,
            topology.head_dim,
        )
        parent_context = case.parent_context.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        target_heads = []
        for members, weights in zip(
            producer.query_groups,
            producer.query_barycentric_weights,
            strict=True,
        ):
            selected = parent_context.index_select(
                1,
                torch.tensor(members, dtype=torch.long),
            )
            coefficients = torch.tensor(
                weights,
                dtype=torch.float64,
            ).reshape(1, len(members), 1, 1)
            target_heads.append((selected * coefficients).sum(dim=1))
        context_target = torch.stack(target_heads, dim=2)
        context_error += float((child_heads - context_target).square().sum())
        context_target_square += float(context_target.square().sum())
    epsilon = torch.finfo(torch.float64).eps
    return (
        math.sqrt(weighted_error / max(weighted_target_square, epsilon)),
        math.sqrt(plain_error / max(plain_target_square, epsilon)),
        math.sqrt(context_error / max(context_target_square, epsilon)),
    )


def compile_lfm2_fixed_qk_sequence_effective_attention_fold(
    *,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    target_value_weight_template: Any,
    target_output_weight_template: Any,
    fit_cases: Sequence[Lfm2CachedAttentionFoldCase],
    query_norm_weight: Any | None = None,
    key_norm_weight: Any | None = None,
    norm_epsilon: float = 1e-6,
    policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy | None = None,
) -> Lfm2CompiledFixedQkSequenceEffectiveAttentionFold:
    """Compile one authenticated fit-only exact-Q/K sequence-effective arm."""

    torch = _torch()
    selected_policy = (
        Lfm2FixedQkSequenceEffectiveVoAlsPolicy()
        if policy is None
        else policy
    )
    if not isinstance(
        selected_policy,
        Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    ) or not selected_policy.verify_authentication():
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "policy must be an authenticated sequence-effective policy."
        )
    if not isinstance(
        producer,
        Lfm2CompiledStructuralAttentionProducerFold,
    ) or not producer.full_source_coverage:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "producer must be one full-coverage structural attention producer."
        )
    cases = tuple(fit_cases)
    if not cases or any(
        not isinstance(case, Lfm2CachedAttentionFoldCase) for case in cases
    ):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "fit_cases must contain at least one cached attention case."
        )
    epsilon = _finite_float(
        norm_epsilon,
        "norm_epsilon",
        minimum=1e-12,
        maximum=1.0,
    )
    topology = producer.topology
    try:
        value_dtype, value_shape = structural._dense_template(
            target_value_weight_template,
            "target_value_weight_template",
        )
        output_dtype, output_shape = structural._dense_template(
            target_output_weight_template,
            "target_output_weight_template",
        )
    except structural.Lfm2StructuralResidualAttentionFoldError as exc:
        raise Lfm2SequenceEffectiveAttentionFoldError(str(exc)) from exc
    expected_value_shape = (
        topology.target_key_value_heads * topology.head_dim,
        topology.target_hidden_size,
    )
    expected_output_shape = (
        topology.target_output_size,
        topology.target_query_heads * topology.head_dim,
    )
    if value_shape != expected_value_shape or output_shape != expected_output_shape:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "Target V/O templates differ from the fixed Q/K topology."
        )
    q_norm = (
        None
        if query_norm_weight is None
        else _tensor(query_norm_weight, "query_norm_weight", dimensions=1).to(
            torch.float64
        )
    )
    k_norm = (
        None
        if key_norm_weight is None
        else _tensor(key_norm_weight, "key_norm_weight", dimensions=1).to(
            torch.float64
        )
    )
    if q_norm is not None and tuple(q_norm.shape) != (topology.head_dim,):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "query_norm_weight has the wrong head dimension."
        )
    if k_norm is not None and tuple(k_norm.shape) != (topology.head_dim,):
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "key_norm_weight has the wrong head dimension."
        )

    producer_before = _state_sha256(producer.state_dict())
    case_bundle_before = _case_bundle_sha256(cases)
    fit_rows = sum(
        int(case.target_query_input.shape[0])
        * int(case.target_query_input.shape[1])
        for case in cases
    )
    solver_plan = _sequence_effective_solver_plan(
        fit_rows=fit_rows,
        target_query_heads=topology.target_query_heads,
        target_hidden_size=topology.target_hidden_size,
        target_output_size=topology.target_output_size,
        head_dim=topology.head_dim,
    )
    streamed_headwise = (
        solver_plan["strategy"] == "streamed_cross_head_statistics_pcg"
    )
    normalized_cases: list[Mapping[str, Any]] = []
    attended_inputs = endpoint_target = None
    plain_attended_inputs = plain_endpoint_target = None
    sequence_statistics: Mapping[str, Any] | None = None
    statistics_report: Mapping[str, Any] | None = None
    if streamed_headwise:
        (
            sequence_statistics,
            statistics_report,
            explicit_weight_count,
        ) = _accumulate_streamed_sequence_statistics(
            producer=producer,
            cases=cases,
            query_norm_weight=q_norm,
            key_norm_weight=k_norm,
            norm_epsilon=epsilon,
        )
        effective = None
        effective_solve = _seal(
            {
                "role": "sequence_effective_scalable_shared_value_seed",
                "seed": "fit_compiled_producer_shared_value_subspace",
                "producer_value_sha256": _tensor_sha256(
                    producer.value_weight
                ),
                "complete_effective_operator_solve_skipped": True,
                "reason": (
                    "avoid_qheads_times_hidden_multi_output_factorization"
                ),
                "selection_inputs_consumed": 0,
                "validation_inputs_consumed": 0,
            }
        )
    else:
        weighted_attended = []
        weighted_targets = []
        unweighted_attended = []
        unweighted_targets = []
        explicit_weight_count = 0
        for case_index, case in enumerate(cases):
            normalized = _normalize_sequence_case(
                case,
                case_index=case_index,
                producer=producer,
                query_norm_weight=q_norm,
                key_norm_weight=k_norm,
                norm_epsilon=epsilon,
            )
            normalized_cases.append(normalized)
            batch = int(normalized["batch"])
            query_length = int(normalized["query_length"])
            attended = normalized["weighted_inputs"].permute(
                0,
                2,
                1,
                3,
            ).reshape(
                batch * query_length,
                topology.target_query_heads,
                topology.target_hidden_size,
            )
            endpoint = normalized["endpoint"]
            row_weights, explicit = _case_query_row_weights(
                case,
                case_index=case_index,
                batch=batch,
                query_length=query_length,
            )
            explicit_weight_count += int(explicit)
            weight = row_weights.reshape(-1, 1)
            unweighted_attended.append(attended)
            unweighted_targets.append(endpoint)
            weighted_attended.append(attended * weight.unsqueeze(1))
            weighted_targets.append(endpoint * weight)
        attended_inputs = torch.cat(weighted_attended, dim=0).to(torch.float64)
        endpoint_target = torch.cat(weighted_targets, dim=0).to(torch.float64)
        plain_attended_inputs = torch.cat(unweighted_attended, dim=0).to(
            torch.float64
        )
        plain_endpoint_target = torch.cat(unweighted_targets, dim=0).to(
            torch.float64
        )
        flat_design = attended_inputs.reshape(
            int(attended_inputs.shape[0]),
            topology.target_query_heads * topology.target_hidden_size,
        )
        effective_rows, effective_solve = _ridge_solve(
            flat_design,
            endpoint_target,
            policy=selected_policy,
            role="sequence_effective_complete_causal_operator",
        )
        effective = effective_rows.reshape(
            topology.target_query_heads,
            topology.target_hidden_size,
            topology.target_output_size,
        )
    if streamed_headwise:
        value_factor = producer.value_weight.to(torch.float64).reshape(
            topology.target_key_value_heads,
            topology.head_dim,
            topology.target_hidden_size,
        ).permute(0, 2, 1).contiguous()
        output_blocks = torch.zeros(
            topology.target_query_heads,
            topology.head_dim,
            topology.target_output_size,
            dtype=torch.float64,
        )
        svd_reports = [
            {
                "group": group,
                "retained_rank": topology.head_dim,
                "seed": "producer_shared_value_subspace",
                "producer_value_group_sha256": _tensor_sha256(
                    value_factor[group]
                ),
                "large_topology_svd_materialization_avoided": True,
            }
            for group in range(topology.target_key_value_heads)
        ]
    else:
        value_factor, output_blocks, svd_reports = (
            _initial_shared_value_factors(
                effective,
                target_key_value_heads=topology.target_key_value_heads,
                heads_per_group=topology.target_query_heads_per_group,
                head_dim=topology.head_dim,
            )
        )
    if streamed_headwise:
        output_blocks, output_solve = _fit_output_from_sequence_statistics(
            statistics=sequence_statistics,
            topology=topology,
            value_factor=value_factor,
            policy=selected_policy,
            role="sequence_effective_streamed_svd_seed_output",
        )
    else:
        output_blocks, _features, output_solve = _fit_output(
            attended_inputs,
            endpoint_target,
            value_factor,
            heads_per_group=topology.target_query_heads_per_group,
            policy=selected_policy,
            role="sequence_effective_svd_seed_output",
        )
    value_factor, output_blocks = _canonicalize_value_factor(
        value_factor,
        output_blocks,
    )
    accepted_error = (
        float(output_solve["relative_rmse"])
        if streamed_headwise
        else _relative_rmse(
            _prediction(
                attended_inputs,
                value_factor,
                output_blocks,
                heads_per_group=topology.target_query_heads_per_group,
            ),
            endpoint_target,
        )
    )
    frontier = [
        {
            "proposal_index": 0,
            "stage": (
                "scalable_producer_shared_value_sequence_output_seed"
                if streamed_headwise
                else "effective_operator_shared_value_svd_seed"
            ),
            "accepted": True,
            "weighted_fit_relative_rmse": accepted_error,
            "value_factor_sha256": _tensor_sha256(value_factor),
            "output_blocks_sha256": _tensor_sha256(output_blocks),
            "solve_report_sha256": output_solve["report_sha256"],
        }
    ]
    accepted_iterations = 0
    solve_reports = [effective_solve, output_solve]
    als_iteration_limit = (
        min(
            selected_policy.maximum_als_iterations,
            _MAXIMUM_STREAMED_ALS_ITERATIONS,
        )
        if streamed_headwise
        else selected_policy.maximum_als_iterations
    )
    for iteration in range(1, als_iteration_limit + 1):
        if streamed_headwise:
            proposed_value, value_reports = _structured_value_pcg_sweep(
                statistics=sequence_statistics,
                topology=topology,
                value_factor=value_factor,
                output_blocks=output_blocks,
                policy=selected_policy,
                iteration=iteration,
            )
            proposed_output, proposed_output_report = (
                _fit_output_from_sequence_statistics(
                    statistics=sequence_statistics,
                    topology=topology,
                    value_factor=proposed_value,
                    policy=selected_policy,
                    role=(
                        f"sequence_effective_streamed_output_iteration_{iteration}"
                    ),
                )
            )
        else:
            proposed_value, value_reports = _block_value_sweep(
                attended_inputs,
                endpoint_target,
                value_factor,
                output_blocks,
                heads_per_group=topology.target_query_heads_per_group,
                policy=selected_policy,
                iteration=iteration,
            )
            proposed_output, _features, proposed_output_report = _fit_output(
                attended_inputs,
                endpoint_target,
                proposed_value,
                heads_per_group=topology.target_query_heads_per_group,
                policy=selected_policy,
                role=f"sequence_effective_output_iteration_{iteration}",
            )
        proposed_value, proposed_output = _canonicalize_value_factor(
            proposed_value,
            proposed_output,
        )
        proposed_error = (
            float(proposed_output_report["relative_rmse"])
            if streamed_headwise
            else _relative_rmse(
                _prediction(
                    attended_inputs,
                    proposed_value,
                    proposed_output,
                    heads_per_group=topology.target_query_heads_per_group,
                ),
                endpoint_target,
            )
        )
        accepted = proposed_error <= accepted_error + selected_policy.fit_tolerance
        solve_reports.extend(value_reports)
        solve_reports.append(proposed_output_report)
        frontier.append(
            {
                "proposal_index": iteration,
                "stage": (
                    "monotone_streamed_effective_space_value_output_als"
                    if streamed_headwise
                    else "monotone_closed_form_value_output_als"
                ),
                "accepted": accepted,
                "weighted_fit_relative_rmse": proposed_error,
                "value_factor_sha256": _tensor_sha256(proposed_value),
                "output_blocks_sha256": _tensor_sha256(proposed_output),
                "value_solve_report_sha256": [
                    report["report_sha256"] for report in value_reports
                ],
                "output_solve_report_sha256": (
                    proposed_output_report["report_sha256"]
                ),
            }
        )
        if not accepted:
            break
        value_factor = proposed_value
        output_blocks = proposed_output
        accepted_error = proposed_error
        accepted_iterations += 1

    anchored_value, anchor_reports = _producer_value_anchor(
        value_factor,
        producer,
        policy=selected_policy,
    )
    if streamed_headwise:
        anchored_output, anchor_output_report = (
            _fit_output_from_sequence_statistics(
                statistics=sequence_statistics,
                topology=topology,
                value_factor=anchored_value,
                policy=selected_policy,
                role="sequence_effective_streamed_anchored_output",
            )
        )
    else:
        anchored_output, _features, anchor_output_report = _fit_output(
            attended_inputs,
            endpoint_target,
            anchored_value,
            heads_per_group=topology.target_query_heads_per_group,
            policy=selected_policy,
            role="sequence_effective_anchored_output",
        )
    anchored_value, anchored_output = _canonicalize_value_factor(
        anchored_value,
        anchored_output,
    )
    anchored_error = (
        float(anchor_output_report["relative_rmse"])
        if streamed_headwise
        else _relative_rmse(
            _prediction(
                attended_inputs,
                anchored_value,
                anchored_output,
                heads_per_group=topology.target_query_heads_per_group,
            ),
            endpoint_target,
        )
    )
    solve_reports.append(anchor_output_report)
    frontier.append(
        {
            "proposal_index": len(frontier),
            "stage": "fixed_producer_value_subspace_anchor",
            "accepted": True,
            "selected_for_materialization": True,
            "weighted_fit_relative_rmse": anchored_error,
            "value_factor_sha256": _tensor_sha256(anchored_value),
            "output_blocks_sha256": _tensor_sha256(anchored_output),
            "anchor_report_sha256": [_sha256(report) for report in anchor_reports],
            "output_solve_report_sha256": anchor_output_report["report_sha256"],
        }
    )

    value_fp64 = anchored_value.permute(0, 2, 1).reshape(
        topology.target_key_value_heads * topology.head_dim,
        topology.target_hidden_size,
    )
    output_fp64 = anchored_output.reshape(
        topology.target_query_heads * topology.head_dim,
        topology.target_output_size,
    ).T
    compiled_value = value_fp64.to(dtype=value_dtype).contiguous().clone()
    compiled_output = output_fp64.to(dtype=output_dtype).contiguous().clone()
    materialized_factor = compiled_value.to(torch.float64).reshape(
        topology.target_key_value_heads,
        topology.head_dim,
        topology.target_hidden_size,
    ).permute(0, 2, 1)
    materialized_blocks = compiled_output.to(torch.float64).T.reshape(
        topology.target_query_heads,
        topology.head_dim,
        topology.target_output_size,
    )
    if streamed_headwise:
        (
            weighted_materialized_error,
            plain_materialized_error,
            fit_context_error,
        ) = _streaming_materialized_metrics(
            producer=producer,
            cases=cases,
            query_norm_weight=q_norm,
            key_norm_weight=k_norm,
            norm_epsilon=epsilon,
            value_factor=materialized_factor,
            output_blocks=materialized_blocks,
        )
    else:
        weighted_materialized_prediction = _prediction(
            attended_inputs,
            materialized_factor,
            materialized_blocks,
            heads_per_group=topology.target_query_heads_per_group,
        )
        plain_materialized_prediction = _prediction(
            plain_attended_inputs,
            materialized_factor,
            materialized_blocks,
            heads_per_group=topology.target_query_heads_per_group,
        )
        weighted_materialized_error = _relative_rmse(
            weighted_materialized_prediction,
            endpoint_target,
        )
        plain_materialized_error = _relative_rmse(
            plain_materialized_prediction,
            plain_endpoint_target,
        )
        fit_context_error = _fit_context_preservation(
            producer=producer,
            normalized_cases=normalized_cases,
            fit_cases=cases,
            materialized_value_factor=materialized_factor,
        )

    producer_after = _state_sha256(producer.state_dict())
    case_bundle_after = _case_bundle_sha256(cases)
    if producer_before != producer_after or case_bundle_before != case_bundle_after:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "Producer or fit evidence changed during sequence-effective folding."
        )
    template_sha = structural._template_sha256(
        {
            "v_proj.weight": (value_shape, str(value_dtype)),
            "out_proj.weight": (output_shape, str(output_dtype)),
        }
    )
    state = {
        "q_proj.weight": producer.query_weight,
        "k_proj.weight": producer.key_weight,
        "v_proj.weight": compiled_value,
        "out_proj.weight": compiled_output,
    }
    compiled_sha = _state_sha256(state)
    source_logical = (
        producer.source_logical_parameters
        + topology.source_hidden_size
        * topology.source_query_heads
        * topology.head_dim
    )
    target_logical = sum(int(value.numel()) for value in state.values())
    if target_logical >= source_logical:
        raise Lfm2SequenceEffectiveAttentionFoldError(
            "Sequence-effective target attention is not structurally smaller."
        )
    frontier_sha = _sha256(frontier)
    body = {
        "schema": LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA,
        "candidate_id": FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID,
        "implementation": _IMPLEMENTATION,
        "policy": selected_policy.as_dict(),
        "policy_sha256": selected_policy.authentication_sha256,
        "producer_artifact": producer.as_dict(),
        "producer_artifact_sha256": _sha256(producer.as_dict()),
        "fixed_query_key_sha256": producer.score_state_sha256,
        "fixed_query_key_state_reused_exactly": True,
        "topology": topology.as_dict(),
        "fit_case_count": len(cases),
        "fit_case_bundle_sha256": case_bundle_before,
        "fit_row_weighting_mode": (
            "explicit_authenticated_query_row_sqrt_weights"
            if explicit_weight_count == len(cases)
            else (
                "uniform_default"
                if explicit_weight_count == 0
                else "mixed_explicit_and_uniform"
            )
        ),
        "producer_state_sha256_before": producer_before,
        "producer_state_sha256_after": producer_after,
        "producer_state_unchanged": True,
        "target_template_sha256": template_sha,
        "compiled_state_sha256": compiled_sha,
        "solver_plan": solver_plan,
        "solver_strategy": solver_plan["strategy"],
        "streamed_sufficient_statistics": (
            None if statistics_report is None else dict(statistics_report)
        ),
        "complete_operator_quadratic_gram_avoided": solver_plan[
            "complete_operator_quadratic_gram_avoided"
        ],
        "complete_operator_dense_factorization_avoided": solver_plan[
            "complete_operator_dense_factorization_avoided"
        ],
        "forbidden_value_coefficient_tensor_materialized": False,
        "forbidden_hidden_head_dim_gram_materialized": False,
        "candidate_frontier": frontier,
        "candidate_frontier_sha256": frontier_sha,
        "bounded_proposal_count": len(frontier),
        "maximum_als_iterations": selected_policy.maximum_als_iterations,
        "executed_als_iteration_limit": als_iteration_limit,
        "accepted_als_iterations": accepted_iterations,
        "selected_proposal_index": len(frontier) - 1,
        "effective_operator_solve": effective_solve,
        "shared_value_svd": svd_reports,
        "value_subspace_anchor": anchor_reports,
        "solve_report_sha256": [
            report["report_sha256"] for report in solve_reports
        ],
        "weighted_fit_relative_rmse_fp64_before_anchor": accepted_error,
        "weighted_fit_relative_rmse_fp64_after_anchor": anchored_error,
        "weighted_fit_relative_rmse_target_dtype": weighted_materialized_error,
        "unweighted_fit_relative_rmse_target_dtype": plain_materialized_error,
        "fit_context_barycenter_relative_rmse_target_dtype": fit_context_error,
        "source_logical_parameters": source_logical,
        "target_logical_parameters": target_logical,
        "removed_logical_parameters": source_logical - target_logical,
        "parameter_reduction_fraction": (
            (source_logical - target_logical) / source_logical
        ),
        "fit_only_compilation": True,
        "compiled_from_splits": ["fit"],
        "selection_case_count_consumed": 0,
        "validation_case_count_consumed": 0,
        "selection_inputs_accessible": False,
        "validation_inputs_accessible": False,
        "causal_attended_input_operator": "z_h=softmax(Q_hK_h^T)X_K",
        "ordinary_dense_target_value_output": True,
        "dense_target_native_qkvo": True,
        "blank_target_values_read": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    return Lfm2CompiledFixedQkSequenceEffectiveAttentionFold(
        producer_artifact=producer,
        policy=selected_policy,
        value_weight=compiled_value,
        output_weight=compiled_output,
        fit_case_count=len(cases),
        fit_case_bundle_sha256=case_bundle_before,
        producer_state_sha256_before=producer_before,
        producer_state_sha256_after=producer_after,
        target_template_sha256=template_sha,
        compiled_state_sha256=compiled_sha,
        candidate_frontier_sha256=frontier_sha,
        report=_seal(body),
    )


def install_lfm2_fixed_qk_sequence_effective_attention_fold(
    target_attention: Any,
    compiled: Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
) -> Lfm2SequenceEffectiveAttentionInstallReport:
    """Install exact dense Q/K/V/O in place, restoring bytes on any failure."""

    torch = _torch()
    if not isinstance(
        compiled,
        Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
    ) or not compiled.verify_authentication():
        raise Lfm2SequenceEffectiveAttentionInstallError(
            "compiled must be an authenticated sequence-effective artifact."
        )
    modules = {
        "q_proj.weight": getattr(
            getattr(target_attention, "q_proj", None),
            "weight",
            None,
        ),
        "k_proj.weight": getattr(
            getattr(target_attention, "k_proj", None),
            "weight",
            None,
        ),
        "v_proj.weight": getattr(
            getattr(target_attention, "v_proj", None),
            "weight",
            None,
        ),
        "out_proj.weight": getattr(
            getattr(target_attention, "out_proj", None),
            "weight",
            None,
        ),
    }
    expected = compiled.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in modules.values()):
        raise Lfm2SequenceEffectiveAttentionInstallError(
            "target_attention must expose dense Q/K/V/O weights."
        )
    for name, parameter in modules.items():
        if (
            tuple(parameter.shape) != tuple(expected[name].shape)
            or parameter.dtype != expected[name].dtype
        ):
            raise Lfm2SequenceEffectiveAttentionInstallError(
                f"Target {name} differs from the compiled shape or dtype."
            )
    identities = {name: id(value) for name, value in modules.items()}
    shapes = {name: tuple(value.shape) for name, value in modules.items()}
    dtypes = {name: str(value.dtype) for name, value in modules.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in modules.items()
    }
    backup = {
        name: value.detach().to(device="cpu").clone()
        for name, value in modules.items()
    }
    try:
        with torch.no_grad():
            for name, parameter in modules.items():
                _copy_parameter(
                    parameter,
                    expected[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                )
        installed_sha = _state_sha256(
            {
                name: parameter.detach().to(device="cpu").contiguous()
                for name, parameter in modules.items()
            }
        )
        body = {
            "schema": LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_INSTALL_SCHEMA,
            "candidate_id": FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID,
            "compiled_artifact_report_sha256": (
                compiled.report["report_sha256"]
            ),
            "compiled_state_sha256": compiled.compiled_state_sha256,
            "installed_state_sha256": installed_sha,
            "parameter_identities_preserved": all(
                id(modules[name]) == identities[name] for name in modules
            ),
            "parameter_shapes_preserved": all(
                tuple(modules[name].shape) == shapes[name] for name in modules
            ),
            "parameter_dtypes_preserved": all(
                str(modules[name].dtype) == dtypes[name] for name in modules
            ),
            "gradient_state_preserved": all(
                (
                    bool(modules[name].requires_grad),
                    None if modules[name].grad is None else id(modules[name].grad),
                    None
                    if modules[name].grad is None
                    else _tensor_sha256(modules[name].grad),
                )
                == gradients[name]
                for name in modules
            ),
            "fixed_query_key_installed_exactly": (
                _state_sha256(
                    {
                        "q_proj.weight": modules["q_proj.weight"],
                        "k_proj.weight": modules["k_proj.weight"],
                    }
                )
                == compiled.producer_artifact.score_state_sha256
            ),
            "ordinary_dense_target_value_output": True,
            "accepted": False,
            "execution": dict(_ZERO_EXECUTION),
        }
        accepted = bool(
            body["parameter_identities_preserved"]
            and body["parameter_shapes_preserved"]
            and body["parameter_dtypes_preserved"]
            and body["gradient_state_preserved"]
            and body["fixed_query_key_installed_exactly"]
            and installed_sha == compiled.compiled_state_sha256
        )
        body["accepted"] = accepted
        if not accepted:
            raise Lfm2SequenceEffectiveAttentionInstallError(
                "Installed sequence-effective state failed exact authentication."
            )
        return Lfm2SequenceEffectiveAttentionInstallReport(_seal(body))
    except BaseException:
        try:
            with torch.no_grad():
                for name, parameter in modules.items():
                    _copy_parameter(
                        parameter,
                        backup[name].to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        ),
                    )
            if any(
                not torch.equal(
                    parameter.detach().to(device="cpu"),
                    backup[name],
                )
                for name, parameter in modules.items()
            ):
                raise Lfm2SequenceEffectiveAttentionInstallError(
                    "Sequence-effective installation rollback was not exact."
                )
        except BaseException as rollback_error:
            raise Lfm2SequenceEffectiveAttentionInstallError(
                "Sequence-effective installation and exact rollback both failed."
            ) from rollback_error
        raise


__all__ = [
    "FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID",
    "FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_POLICY_SHA256",
    "LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_INSTALL_SCHEMA",
    "LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA",
    "LFM2_SEQUENCE_EFFECTIVE_PCG_OBSERVATION_SCHEMA",
    "LFM2_SEQUENCE_EFFECTIVE_ROW_BLOCK_JACOBI_SCHEMA",
    "Lfm2CompiledFixedQkSequenceEffectiveAttentionFold",
    "Lfm2FixedQkSequenceEffectiveVoAlsPolicy",
    "Lfm2SequenceEffectiveAttentionFoldError",
    "Lfm2SequenceEffectiveAttentionInstallError",
    "Lfm2SequenceEffectiveAttentionInstallReport",
    "compile_lfm2_fixed_qk_sequence_effective_attention_fold",
    "install_lfm2_fixed_qk_sequence_effective_attention_fold",
    "observe_lfm2_sequence_effective_pcg",
]
