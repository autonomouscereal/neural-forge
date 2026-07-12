"""Output-Jacobian audit for the exact Qwen target-native QKVO transaction.

This module does not train or install another residual tranche.  It replays the
existing behavior-distillation loss twice, observes gradients at the live QKVO
projection boundaries, and asks a memory-bounded sketch for right directions
outside the inherited QKVO factor spaces.  Only a four-tensor, target-shaped
right-basis pack can leave the audit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neural_forge import hf_target_native_behavior_distillation as _trainer
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAGeometry,
    DenseGQAQKVOResidualProjection,
)
from neural_forge.hf_output_jacobian_sketch import (
    OutputJacobianBlockIdentity,
    OutputJacobianSketchReport,
    OutputJacobianSketchSpec,
    TwoPassOutputJacobianSketch,
)
from neural_forge.hf_progressive_target_bridge import (
    ExactTargetCompilerTransaction,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
    BehaviorDistillationSchedule,
    BehaviorDomainSuite,
)


QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA = "neural-forge-qkvo-output-jacobian-audit-v1"
GUIDED_QKVO_RIGHT_BASIS_SCHEMA = "neural-forge-guided-qkvo-right-basis-v1"
_HARD_MAX_BYTES = 256 * 1024 * 1024
_ROLES: tuple[Literal["q", "k", "v", "o"], ...] = ("q", "k", "v", "o")
_PACK_NAMES = {
    "q": "q_orthogonal_right",
    "k": "k_orthogonal_right",
    "v": "v_orthogonal_right",
    "o": "o_orthogonal_right",
}


class QKVOOutputJacobianAuditError(RuntimeError):
    """Base failure for an uncertifiable QKVO output-Jacobian audit."""


class QKVOOutputJacobianAuditTopologyError(QKVOOutputJacobianAuditError):
    """The live transaction is not the exact 120-site stage-one topology."""


class QKVOOutputJacobianAuditReplayError(QKVOOutputJacobianAuditError):
    """The two behavior-loss/sketch passes did not replay identically."""


class QKVOOutputJacobianAuditLimitError(QKVOOutputJacobianAuditError):
    """The aggregate sketch plan exceeds its hard memory ceiling."""


class QKVOOutputJacobianAuditGateError(QKVOOutputJacobianAuditError):
    """A rejected audit was asked to publish a guided basis pack."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QKVOOutputJacobianAuditError(
            "Audit evidence is not canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QKVOOutputJacobianAuditError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QKVOOutputJacobianAuditError(f"{name} must be positive.") from error
    if result <= 0:
        raise QKVOOutputJacobianAuditError(f"{name} must be positive.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QKVOOutputJacobianAuditError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QKVOOutputJacobianAuditError(
            f"{name} must be finite and nonnegative."
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise QKVOOutputJacobianAuditError(f"{name} must be finite and nonnegative.")
    return result


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise QKVOOutputJacobianAuditError("Only tensors can enter a tensor hash.")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    # Reshape before the byte view: rank-one non-contiguous slices cannot be
    # viewed as bytes safely until their element stride has been normalized.
    flat = tensor.reshape(-1).contiguous().view(torch.uint8)
    for start in range(0, int(flat.numel()), 1 << 20):
        digest.update(memoryview(flat[start : start + (1 << 20)].numpy()))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QKVOOutputJacobianAuditConfig:
    """Preregistered, exact stage-one audit bounds and aggregate gates."""

    prompt_count: int = 24
    horizon: int = 8
    lanes: tuple[str, str] = ("parent-greedy", "child-greedy")
    rank: int = 8
    sketch_width: int = 16
    seed: int = 0
    min_normal_fraction: float = 0.20
    min_guided_over_random_ratio: float = 1.50
    max_bytes: int = _HARD_MAX_BYTES

    def __post_init__(self) -> None:
        for name in ("prompt_count", "horizon", "rank", "sketch_width", "max_bytes"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if isinstance(self.seed, bool):
            raise QKVOOutputJacobianAuditError("seed must be an integer.")
        try:
            seed = operator.index(self.seed)
        except (TypeError, ValueError, OverflowError) as error:
            raise QKVOOutputJacobianAuditError("seed must be an integer.") from error
        object.__setattr__(self, "seed", seed)
        lanes = tuple(str(value).strip().lower() for value in self.lanes)
        object.__setattr__(self, "lanes", lanes)
        for name in ("min_normal_fraction", "min_guided_over_random_ratio"):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        if self.prompt_count != 24 or self.horizon != 8 or self.rank != 8:
            raise QKVOOutputJacobianAuditError(
                "The exact audit requires 24 prompts, horizon 8, and rank 8."
            )
        if lanes != ("parent-greedy", "child-greedy"):
            raise QKVOOutputJacobianAuditError(
                "The exact audit requires parent-greedy and child-greedy lanes."
            )
        if self.sketch_width < self.rank:
            raise QKVOOutputJacobianAuditError(
                "sketch_width cannot be smaller than rank."
            )
        if self.min_normal_fraction < 0.20:
            raise QKVOOutputJacobianAuditError(
                "The aggregate normal-fraction gate cannot be below 0.20."
            )
        if self.min_guided_over_random_ratio < 1.50:
            raise QKVOOutputJacobianAuditError(
                "The aggregate guided/random gate cannot be below 1.50."
            )
        if self.max_bytes > _HARD_MAX_BYTES:
            raise QKVOOutputJacobianAuditError(
                "max_bytes exceeds the hard 256 MiB audit ceiling."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lanes"] = list(self.lanes)
        return payload

    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class QKVOOutputJacobianAuditReport:
    """Canonical report plus the strict runner-facing attestations."""

    payload: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.payload["passed"])

    @property
    def report_sha256(self) -> str:
        return str(self.payload["report_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class QKVOOutputJacobianAuditResult:
    """Scaled guided rows and their immutable audit report."""

    guided_right_rows: Mapping[str, Tensor]
    report: QKVOOutputJacobianAuditReport

    def basis_pack_payload(self) -> dict[str, Any]:
        """Return a pack whose authorization bit is true only after both gates."""

        factor_tensors = {
            _PACK_NAMES[role]: self.guided_right_rows[role]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .clone()
            for role in _ROLES
        }
        factors = [
            _tensor_manifest_row(name, tensor)
            for name, tensor in factor_tensors.items()
        ]
        manifest_sha256 = _canonical_sha256(factors)
        if manifest_sha256 != self.report.to_dict()["basis_manifest_sha256"]:
            raise QKVOOutputJacobianAuditError(
                "Live guided rows disagree with the audited basis manifest."
            )
        return {
            "schema": GUIDED_QKVO_RIGHT_BASIS_SCHEMA,
            "factor_state_sha256": self.report.to_dict()["factor_state_sha256_before"],
            "basis_manifest_sha256": manifest_sha256,
            "tensor_count": len(factor_tensors),
            "total_numel": sum(int(value.numel()) for value in factor_tensors.values()),
            "total_nbytes": sum(
                int(value.numel() * value.element_size())
                for value in factor_tensors.values()
            ),
            "factors": factors,
            "factor_tensors": factor_tensors,
            "authorized_for_training": self.report.passed,
            "contains_model_state": False,
            "contains_parent_state": False,
            "contains_optimizer_state": False,
            "contains_gradients": False,
            "contains_activations": False,
            "contains_hidden_states": False,
        }


def _tensor_manifest_row(name: str, value: Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "tensor_sha256": _tensor_sha256(value),
        "numel": int(value.numel()),
        "nbytes": int(value.numel() * value.element_size()),
    }


@dataclass(frozen=True, slots=True)
class _ParameterState:
    name: str
    identity: int
    data_pointer: int
    version: int
    requires_grad: bool
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str

    def manifest_row(self) -> dict[str, Any]:
        return asdict(self)


def _parameter_state(
    model: nn.Module, *, excluded_ids: frozenset[int]
) -> tuple[_ParameterState, ...]:
    rows: list[_ParameterState] = []
    for name, value in model.named_parameters():
        if id(value) in excluded_ids:
            continue
        rows.append(
            _ParameterState(
                name=name,
                identity=id(value),
                data_pointer=int(value.data_ptr()),
                version=int(value._version),
                requires_grad=bool(value.requires_grad),
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                sha256=_tensor_sha256(value),
            )
        )
    return tuple(rows)


def _parameter_state_sha256(values: Sequence[_ParameterState]) -> str:
    return _canonical_sha256([value.manifest_row() for value in values])


def _version_state_sha256(
    entries: Sequence[tuple[str, nn.Parameter]],
) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "identity": id(value),
                "data_pointer": int(value.data_ptr()),
                "version": int(value._version),
            }
            for name, value in entries
        ]
    )


def _qkvo_factor_state_sha256(registry: nn.Module) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
            for name, value in registry.named_parameters(recurse=False)
        ]
    )


def _validate_inherited_qkvo_factors(registry: nn.Module) -> None:
    """Require every inherited site product to retain its declared rank."""

    rank = _positive_int(getattr(registry, "rank", None), "qkvo rank")
    for role in _ROLES:
        left = getattr(registry, f"{role}_left", None)
        right = getattr(registry, f"{role}_right", None)
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            raise QKVOOutputJacobianAuditTopologyError(
                "The inherited QKVO registry is missing a factor tensor."
            )
        if (
            not bool(torch.isfinite(left.detach()).all())
            or not bool(torch.isfinite(right.detach()).all())
            or int(torch.count_nonzero(left.detach()).item()) == 0
            or int(torch.count_nonzero(right.detach()).item()) == 0
        ):
            raise QKVOOutputJacobianAuditTopologyError(
                "All eight inherited QKVO factor tensors must be finite and nonzero."
            )
        if int(left.shape[0]) != 120 or int(right.shape[0]) != 120:
            raise QKVOOutputJacobianAuditTopologyError(
                "Inherited QKVO factors do not cover all 120 sites."
            )
        # Full rank of both small factor sides is equivalent to full rank of
        # their product.  Work on rank-by-rank Gram cores instead of forming a
        # 512/1536-wide dense correction.
        left32 = left.detach().to(torch.float32)
        right32 = right.detach().to(torch.float32)
        left_gram = left32.transpose(1, 2) @ left32
        right_gram = right32 @ right32.transpose(1, 2)
        left_ranks = torch.linalg.matrix_rank(left_gram).detach().cpu()
        right_ranks = torch.linalg.matrix_rank(right_gram).detach().cpu()
        invalid = torch.nonzero(
            (left_ranks != rank) | (right_ranks != rank), as_tuple=False
        ).reshape(-1)
        if int(invalid.numel()) > 0:
            site_row = int(invalid[0].item())
            raise QKVOOutputJacobianAuditTopologyError(
                f"Inherited {role.upper()} site {site_row} product rank is not "
                f"the declared rank {rank}."
            )


def _validate_domain_and_schedule(
    domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    config: QKVOOutputJacobianAuditConfig,
) -> None:
    if not isinstance(domain, BehaviorDomainSuite):
        raise QKVOOutputJacobianAuditError(
            "basis_domain must be a BehaviorDomainSuite."
        )
    if domain.split_name != "fit" or len(domain.prompts) != config.prompt_count:
        raise QKVOOutputJacobianAuditError(
            "The basis must expose exactly 24 source-candidate prompts as fit only."
        )
    if not isinstance(schedule, BehaviorDistillationSchedule):
        raise QKVOOutputJacobianAuditError(
            "schedule must be a BehaviorDistillationSchedule."
        )
    if schedule.fit_horizon != config.horizon:
        raise QKVOOutputJacobianAuditError(
            "schedule.fit_horizon must match the exact audit horizon."
        )
    if schedule.child_lane_loss_weight <= 0.0:
        raise QKVOOutputJacobianAuditError(
            "The child-greedy lane must have positive loss weight."
        )
    if any(
        len(prompt.token_ids) > schedule.max_prompt_tokens for prompt in domain.prompts
    ):
        raise QKVOOutputJacobianAuditError(
            "A basis prompt exceeds schedule.max_prompt_tokens."
        )


def _validate_exact_topology(
    transaction: ExactTargetCompilerTransaction,
) -> tuple[DenseGQAGeometry, tuple[tuple[int, int], ...]]:
    if not isinstance(transaction, ExactTargetCompilerTransaction):
        raise QKVOOutputJacobianAuditTopologyError(
            "transaction must be ExactTargetCompilerTransaction."
        )
    transaction.assert_live()
    registry = transaction.qkvo_handle.adapter_registry
    geometry = registry.geometry
    expected_geometry = {
        "logical_layers": 30,
        "hidden_size": 1536,
        "query_heads": 16,
        "key_value_heads": 4,
        "head_dim": 128,
        "query_group_width": 512,
    }
    actual_geometry = {
        "logical_layers": geometry.logical_layers,
        "hidden_size": geometry.hidden_size,
        "query_heads": geometry.query_heads,
        "key_value_heads": geometry.key_value_heads,
        "head_dim": geometry.head_dim,
        "query_group_width": geometry.query_group_width,
    }
    if actual_geometry != expected_geometry or registry.site_count != 120:
        raise QKVOOutputJacobianAuditTopologyError(
            "The audit requires exact h=1536,L=30,Hq=16,Hkv=4,d=128 topology."
        )
    sites = tuple(
        (int(layer), int(group))
        for layer, group in registry.site_indices.detach().cpu().tolist()
    )
    expected_sites = tuple((layer, group) for layer in range(30) for group in range(4))
    if sites != expected_sites:
        raise QKVOOutputJacobianAuditTopologyError(
            "The QKVO registry must cover all 120 physical sites in canonical order."
        )
    if transaction.lexical_handle is None:
        raise QKVOOutputJacobianAuditTopologyError(
            "The audit requires an authenticated v2 transaction with lexical factors."
        )
    return geometry, sites


def _sketch_dimensions(role: str, geometry: DenseGQAGeometry) -> tuple[int, int]:
    if role == "q":
        return geometry.query_group_width, geometry.hidden_size
    if role in {"k", "v"}:
        return geometry.head_dim, geometry.hidden_size
    if role == "o":
        return geometry.hidden_size, geometry.query_group_width
    raise QKVOOutputJacobianAuditTopologyError("Unknown QKVO role.")


def _sketch_memory_plan(
    *,
    geometry: DenseGQAGeometry,
    site_count: int,
    inherited_rank: int,
    config: QKVOOutputJacobianAuditConfig,
    max_rows: int,
) -> dict[str, int]:
    """Conservative simultaneous-persistent plus one-operation peak plan."""

    persistent_total = 0
    largest_transient = 0
    for role in _ROLES:
        out_features, in_features = _sketch_dimensions(role, geometry)
        occupied = inherited_rank
        width = config.sketch_width
        rank = config.rank
        range_width = min(width, in_features - occupied)
        occupied_bytes = 4 * occupied * (out_features + in_features)
        pass1_persistent = (
            occupied_bytes + 4 * out_features * width + 8 * in_features * width
        )
        pass2_persistent = (
            occupied_bytes
            + 4 * in_features * (range_width + rank)
            + 4 * out_features * (range_width + rank)
        )
        persistent_total += site_count * max(pass1_persistent, pass2_persistent)
        factor_analysis = (
            8 * inherited_rank * (out_features + in_features)
            + 32 * inherited_rank * inherited_rank
            + max(
                8
                * (
                    out_features * min(out_features, inherited_rank)
                    + inherited_rank * inherited_rank
                    + inherited_rank
                ),
                8
                * (
                    in_features * min(in_features, inherited_rank)
                    + inherited_rank * inherited_rank
                    + inherited_rank
                ),
            )
        )
        pass1_block = 4 * max_rows * (
            3 * out_features + 3 * in_features + 2 * occupied
        ) + 4 * (max_rows * width + in_features * width)
        transition = (
            24 * in_features * width + 16 * in_features * rank + 8 * width * width
        )
        pass2_block = 4 * max_rows * (
            3 * out_features + 3 * in_features + 2 * occupied
        ) + 4 * (
            max_rows * (range_width + rank) + out_features * max(range_width, rank)
        )
        finalize = (
            16 * out_features * range_width
            + 8 * range_width * range_width
            + 8 * range_width
            + 8 * in_features * (range_width + rank)
            + 12 * rank * in_features
        )
        largest_transient = max(
            largest_transient,
            factor_analysis,
            pass1_block,
            transition,
            pass2_block,
            finalize,
        )
    wrapper_transient = _wrapper_hook_transient_bytes(geometry, max_rows)
    peak = persistent_total + largest_transient + wrapper_transient
    return {
        "persistent_upper_bound_bytes": persistent_total,
        "largest_single_operation_bytes": largest_transient,
        "wrapper_hook_transient_bytes": wrapper_transient,
        "accounted_global_peak_bytes": peak,
        "max_bytes": config.max_bytes,
    }


def _wrapper_hook_transient_bytes(geometry: DenseGQAGeometry, rows: int) -> int:
    """Upper-bound full-wrapper casts and compact-fingerprint temporaries."""

    widths = {
        "q": (geometry.hidden_size, geometry.query_width),
        "k": (geometry.hidden_size, geometry.key_value_width),
        "v": (geometry.hidden_size, geometry.key_value_width),
        "o": (geometry.query_width, geometry.hidden_size),
    }
    largest = 0
    for input_width, output_width in widths.values():
        elements = rows * (input_width + output_width)
        fp32_casts = 4 * elements
        # Finiteness masks plus square/absolute statistic staging are
        # conservatively treated as if both full tensors coexist. Sampled
        # values and indices are tiny but included explicitly.
        fingerprint_staging = 5 * elements + 16 * (8 + 4) * 2
        largest = max(largest, fp32_casts + fingerprint_staging)
    return largest


def _require_memory_within_limit(
    memory: Mapping[str, int],
    config: QKVOOutputJacobianAuditConfig,
    *,
    phase: str,
) -> None:
    peak = int(memory["accounted_global_peak_bytes"])
    if peak > config.max_bytes:
        raise QKVOOutputJacobianAuditLimitError(
            f"Aggregate QKVO sketches exceed the {phase} byte ceiling before backward."
        )


@dataclass(frozen=True, slots=True)
class _ReplayPlan:
    prompt: Any
    parent_tokens: tuple[int, ...]
    parent_rows: tuple[Tensor, ...]
    child_tokens: tuple[int, ...]
    position_weights: tuple[float, ...]
    child_indices: tuple[int, ...]


def _build_replay_plans(
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    config: QKVOOutputJacobianAuditConfig,
) -> tuple[_ReplayPlan, ...]:
    plans: list[_ReplayPlan] = []
    for prompt in domain.prompts:
        parent_tokens, parent_rows = _trainer._greedy_path(
            parent_model, prompt.token_ids, config.horizon
        )
        child_tokens, _ = _trainer._greedy_path(
            transaction.model, prompt.token_ids, config.horizon
        )
        divergence = _trainer._first_divergence(parent_tokens, child_tokens)
        plans.append(
            _ReplayPlan(
                prompt=prompt,
                parent_tokens=parent_tokens,
                parent_rows=parent_rows,
                child_tokens=child_tokens,
                position_weights=_trainer._divergence_position_weights(
                    config.horizon, divergence, schedule
                ),
                child_indices=_trainer._child_lane_indices(
                    child_tokens, divergence, schedule
                ),
            )
        )
    return tuple(plans)


@dataclass(frozen=True, slots=True)
class _LossContext:
    lane: str
    prompt_id: str
    prefix_sha256: str
    loss_weight: float
    block_ordinal: int

    def semantic_identity(self, *, layer: int, role: str, group: int, site: int) -> str:
        return _canonical_json(
            {
                "lane": self.lane,
                "prompt_id": self.prompt_id,
                "prefix_token_sha256": self.prefix_sha256,
                "layer": layer,
                "role": role,
                "physical_group": group,
                "site_row": site,
                "loss_weight": self.loss_weight,
                "block_ordinal": self.block_ordinal,
            }
        ).decode("ascii")


def _token_sha256(values: Sequence[int]) -> str:
    return _canonical_sha256([int(value) for value in values])


def _compact_numeric_fingerprint(inputs: Tensor, delta: Tensor) -> str:
    """Hash compact full-wrapper FP32 statistics once per gradient callback."""

    def stats(value: Tensor) -> dict[str, Any]:
        if not bool(torch.isfinite(value).all()):
            raise QKVOOutputJacobianAuditReplayError(
                "A hooked QKVO input or output gradient is non-finite."
            )
        flat = value.reshape(-1)
        count = int(flat.numel())
        if count == 0:
            raise QKVOOutputJacobianAuditReplayError("A hooked QKVO block is empty.")
        sample_indices = sorted(
            {0, count - 1, *(int(index * (count - 1) / 15) for index in range(16))}
        )
        sampled = flat.index_select(
            0, torch.tensor(sample_indices, dtype=torch.long, device=flat.device)
        )
        return {
            "shape": list(value.shape),
            "sum": float(flat.sum(dtype=torch.float32).item()),
            "square_sum": float(flat.square().sum(dtype=torch.float32).item()),
            "abs_sum": float(flat.abs().sum(dtype=torch.float32).item()),
            "minimum": float(flat.min().item()),
            "maximum": float(flat.max().item()),
            "samples": [float(item) for item in sampled.detach().cpu().tolist()],
        }

    return _canonical_sha256({"inputs": stats(inputs), "delta": stats(delta)})


def _slice_qkvo_blocks(
    role: str,
    group: int,
    geometry: DenseGQAGeometry,
    flat_inputs: Tensor,
    flat_delta: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return D/X views for one physical group after one full-wrapper cast."""

    if role == "q":
        start = group * geometry.query_group_width
        return flat_delta[:, start : start + geometry.query_group_width], flat_inputs
    if role in {"k", "v"}:
        start = group * geometry.head_dim
        return flat_delta[:, start : start + geometry.head_dim], flat_inputs
    if role == "o":
        start = group * geometry.query_group_width
        return flat_delta, flat_inputs[:, start : start + geometry.query_group_width]
    raise QKVOOutputJacobianAuditTopologyError("Unknown hooked QKVO role.")


class _QKVOHookCollector:
    def __init__(
        self,
        *,
        transaction: ExactTargetCompilerTransaction,
        geometry: DenseGQAGeometry,
        sites: tuple[tuple[int, int], ...],
        sketches: Mapping[tuple[str, int], TwoPassOutputJacobianSketch],
    ) -> None:
        self.transaction = transaction
        self.geometry = geometry
        self.sites = sites
        self.sketches = sketches
        self.phase: Literal["pass1", "pass2"] = "pass1"
        self.current: _LossContext | None = None
        self.handles: list[Any] = []
        self.observations: dict[tuple[str, int], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row, (layer, group) in enumerate(sites):
            by_layer[layer].append((row, group))
        self.by_layer = dict(by_layer)

    def install(self) -> None:
        handle = self.transaction.qkvo_handle
        for layer in range(self.geometry.logical_layers):
            attention = handle.attention_modules[layer]
            for role in _ROLES:
                wrapper = getattr(attention, handle.report.projection_names[role], None)
                if not isinstance(wrapper, DenseGQAQKVOResidualProjection):
                    raise QKVOOutputJacobianAuditTopologyError(
                        "A required live QKVO projection wrapper is missing."
                    )
                self.handles.append(wrapper.register_forward_hook(self._forward_hook))
        if len(self.handles) != 120:
            self.remove()
            raise QKVOOutputJacobianAuditTopologyError(
                "The audit must register exactly 120 projection hooks."
            )

    def _forward_hook(
        self, module: nn.Module, arguments: tuple[Any, ...], output: Any
    ) -> None:
        context = self.current
        if context is None:
            raise QKVOOutputJacobianAuditReplayError(
                "A hooked child forward occurred outside an authenticated loss block."
            )
        if (
            not isinstance(module, DenseGQAQKVOResidualProjection)
            or len(arguments) != 1
            or not isinstance(arguments[0], Tensor)
            or not isinstance(output, Tensor)
            or not output.requires_grad
        ):
            raise QKVOOutputJacobianAuditTopologyError(
                "A QKVO wrapper did not expose one differentiable tensor boundary."
            )
        inputs = arguments[0].detach()
        layer = module.layer_index
        role = module.role

        def output_gradient_hook(delta: Tensor) -> Tensor:
            # Cast and flatten the full wrapper tensors exactly once.  Physical
            # groups below receive only FP32 views of these two values.
            flat_inputs = inputs.to(dtype=torch.float32).reshape(
                -1, int(inputs.shape[-1])
            )
            flat_delta = (
                delta.detach().to(dtype=torch.float32).reshape(-1, int(delta.shape[-1]))
            )
            fingerprint = _compact_numeric_fingerprint(flat_inputs, flat_delta)
            phase_index = 0 if self.phase == "pass1" else 1
            for site_row, group in self.by_layer[layer]:
                d_block, x_block = _slice_qkvo_blocks(
                    role, group, self.geometry, flat_inputs, flat_delta
                )
                identity = OutputJacobianBlockIdentity(
                    semantic_identity=context.semantic_identity(
                        layer=layer,
                        role=role,
                        group=group,
                        site=site_row,
                    ),
                    numeric_fingerprint_sha256=fingerprint,
                )
                sketch = self.sketches[(role, site_row)]
                if self.phase == "pass1":
                    sketch.add_pass1(d_block, x_block, block_identity=identity)
                else:
                    sketch.add_pass2(d_block, x_block, block_identity=identity)
                self.observations[(role, site_row)][phase_index] += 1
            return delta

        output.register_hook(output_gradient_hook)

    def remove(self) -> None:
        while self.handles:
            self.handles.pop().remove()

    @property
    def active_hook_count(self) -> int:
        return len(self.handles)


def _loss_divisor(
    plans: Sequence[_ReplayPlan], schedule: BehaviorDistillationSchedule
) -> float:
    divisor = 0.0
    for plan in plans:
        divisor += schedule.parent_lane_loss_weight * sum(plan.position_weights)
        divisor += schedule.child_lane_loss_weight * sum(
            plan.position_weights[index] for index in plan.child_indices
        )
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise QKVOOutputJacobianAuditReplayError(
            "The behavior replay produced an invalid loss divisor."
        )
    return divisor


def _replay_loss_pass(
    *,
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    plans: Sequence[_ReplayPlan],
    schedule: BehaviorDistillationSchedule,
    collector: _QKVOHookCollector,
    factors: Sequence[nn.Parameter],
) -> tuple[str, int]:
    divisor = _loss_divisor(plans, schedule)
    evidence: list[dict[str, Any]] = []
    ordinal = 0
    for plan in plans:
        lanes = (
            (
                "parent-greedy",
                plan.parent_tokens,
                tuple(range(len(plan.parent_tokens))),
                schedule.parent_lane_loss_weight,
            ),
            (
                "child-greedy",
                plan.child_tokens,
                plan.child_indices,
                schedule.child_lane_loss_weight,
            ),
        )
        for lane, lane_tokens, indices, lane_weight in lanes:
            active = set(indices)
            prefix = plan.prompt.token_ids
            for lane_step, token in enumerate(lane_tokens):
                if lane_step not in active:
                    prefix = (*prefix, int(token))
                    continue
                if lane == "parent-greedy":
                    parent_logits = plan.parent_rows[lane_step]
                else:
                    with torch.no_grad():
                        parent_logits = _trainer._last_causal_logits(
                            parent_model, prefix
                        )
                loss_weight = lane_weight * plan.position_weights[lane_step] / divisor
                context = _LossContext(
                    lane=lane,
                    prompt_id=plan.prompt.prompt_id,
                    prefix_sha256=_token_sha256(prefix),
                    loss_weight=loss_weight,
                    block_ordinal=ordinal,
                )
                collector.current = context
                try:
                    child_logits = _trainer._last_causal_logits(
                        transaction.model, prefix
                    )
                finally:
                    collector.current = None
                repetition_tokens: tuple[int, ...] = ()
                if lane == "child-greedy":
                    recent = lane_tokens[
                        max(0, lane_step - schedule.repetition_window) : lane_step
                    ]
                    repetition_tokens = tuple(
                        sorted(
                            {int(value) for value in recent if recent.count(value) >= 2}
                        )
                    )
                loss = (
                    _trainer._training_loss(
                        parent_logits,
                        child_logits,
                        schedule,
                        prefix=prefix,
                        repetition_token_ids=repetition_tokens,
                    )
                    * loss_weight
                )
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise QKVOOutputJacobianAuditReplayError(
                        "Behavior replay loss is non-finite."
                    )
                evidence.append(
                    {
                        "lane": lane,
                        "prompt_id": plan.prompt.prompt_id,
                        "prefix_sha256": context.prefix_sha256,
                        "block_ordinal": ordinal,
                        "loss_weight": loss_weight,
                        "loss": float(loss.detach().cpu().item()),
                        "parent_action": int(torch.argmax(parent_logits).item()),
                        "child_action": int(torch.argmax(child_logits).item()),
                        "child_logits_sha256": _tensor_sha256(child_logits),
                        "repetition_token_ids": list(repetition_tokens),
                    }
                )
                loss.backward()
                for factor in factors:
                    factor.grad = None
                ordinal += 1
                prefix = (*prefix, int(token))
    return _canonical_sha256(evidence), ordinal


def _aggregate_reports(
    reports: Sequence[tuple[str, OutputJacobianSketchReport]],
    *,
    config: QKVOOutputJacobianAuditConfig,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    def aggregate(values: Sequence[OutputJacobianSketchReport]) -> dict[str, Any]:
        raw = sum(value.raw_frobenius_sq_estimate for value in values)
        normal = sum(value.normal_frobenius_sq_estimate for value in values)
        guided = sum(value.guided_capture_energy for value in values)
        random = sum(value.random_capture_energy for value in values)
        tiny = torch.finfo(torch.float64).tiny
        normal_fraction = normal / raw if raw > 0.0 else 0.0
        guided_capture = guided / normal if normal > 0.0 else 0.0
        random_capture = random / normal if normal > 0.0 else 0.0
        ratio = guided / max(random, max(normal * 1e-30, tiny))
        metrics = {
            "sketch_count": len(values),
            "raw_energy": raw,
            "normal_energy": normal,
            "guided_energy": guided,
            "random_energy": random,
            "normal_fraction": normal_fraction,
            "guided_capture_fraction": guided_capture,
            "random_capture_fraction": random_capture,
            "guided_over_random_ratio": ratio,
            "normal_fraction_gate_passed": (
                normal_fraction >= config.min_normal_fraction
            ),
            "guided_over_random_gate_passed": (
                ratio >= config.min_guided_over_random_ratio
            ),
        }
        metrics["passed"] = bool(
            metrics["normal_fraction_gate_passed"]
            and metrics["guided_over_random_gate_passed"]
        )
        return metrics

    all_reports = [value for _role, value in reports]
    aggregate_all = aggregate(all_reports)
    per_role = {
        role: aggregate([value for candidate, value in reports if candidate == role])
        for role in _ROLES
    }
    return aggregate_all, per_role


def _source_helper_sha256() -> str:
    names = (
        "_greedy_path",
        "_last_causal_logits",
        "_training_loss",
        "_first_divergence",
        "_divergence_position_weights",
        "_child_lane_indices",
    )
    return _canonical_sha256(
        {name: inspect.getsource(getattr(_trainer, name)) for name in names}
    )


def run_qkvo_output_jacobian_audit(
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    basis_domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    config: QKVOOutputJacobianAuditConfig,
) -> QKVOOutputJacobianAuditResult:
    """Run the deterministic two-pass QKVO audit without an optimizer/update."""

    if not isinstance(config, QKVOOutputJacobianAuditConfig):
        raise QKVOOutputJacobianAuditError(
            "config must be QKVOOutputJacobianAuditConfig."
        )
    geometry, sites = _validate_exact_topology(transaction)
    if not isinstance(parent_model, nn.Module) or parent_model is transaction.model:
        raise QKVOOutputJacobianAuditError(
            "parent_model must be a distinct torch module."
        )
    _validate_domain_and_schedule(basis_domain, schedule, config)

    factor_state_before = _trainer.capture_target_native_factor_state(transaction)
    if factor_state_before.schema != BEHAVIOR_FACTOR_PACK_V2_SCHEMA:
        raise QKVOOutputJacobianAuditTopologyError(
            "The live transaction is not an authenticated v2 factor state."
        )
    factor_entries = _trainer._factor_parameters(transaction)
    _validate_inherited_qkvo_factors(transaction.qkvo_handle.adapter_registry)
    factor_ids = frozenset(id(value) for _name, value in factor_entries)
    factor_versions_before = _version_state_sha256(factor_entries)
    inherited_before = _qkvo_factor_state_sha256(
        transaction.qkvo_handle.adapter_registry
    )
    native_before = _parameter_state(transaction.model, excluded_ids=factor_ids)
    native_sha_before = _parameter_state_sha256(native_before)
    for _name, value in factor_entries:
        value.grad = None

    max_rows = (
        max(len(prompt.token_ids) for prompt in basis_domain.prompts)
        + config.horizon
        - 1
    )
    inherited_rank = transaction.qkvo_handle.adapter_registry.rank
    memory = _sketch_memory_plan(
        geometry=geometry,
        site_count=len(sites),
        inherited_rank=inherited_rank,
        config=config,
        max_rows=max_rows,
    )
    _require_memory_within_limit(memory, config, phase="preregistered global")

    registry = transaction.qkvo_handle.adapter_registry
    sketches: dict[tuple[str, int], TwoPassOutputJacobianSketch] = {}
    for role_index, role in enumerate(_ROLES):
        out_features, in_features = _sketch_dimensions(role, geometry)
        left = getattr(registry, f"{role}_left")
        right = getattr(registry, f"{role}_right")
        for site_row in range(len(sites)):
            sketches[(role, site_row)] = TwoPassOutputJacobianSketch(
                OutputJacobianSketchSpec(
                    out_features=out_features,
                    in_features=in_features,
                    rank=config.rank,
                    sketch_width=config.sketch_width,
                    seed=config.seed + role_index * len(sites) + site_row,
                    max_bytes=config.max_bytes,
                ),
                inherited_left=left[site_row],
                inherited_right=right[site_row],
            )

    phase1_persistent = sum(
        sketch.accounted_persistent_bytes for sketch in sketches.values()
    )
    phase1_transient = max(
        sketch.accounted_batch_transient_bytes(max_rows, 1)
        for sketch in sketches.values()
    )
    phase1_actual_peak = phase1_persistent + phase1_transient
    wrapper_hook_transient = _wrapper_hook_transient_bytes(geometry, max_rows)
    phase1_actual_peak += wrapper_hook_transient
    memory.update(
        {
            "phase1_actual_persistent_bytes": phase1_persistent,
            "phase1_actual_max_batch_transient_bytes": phase1_transient,
            "phase1_wrapper_hook_transient_bytes": wrapper_hook_transient,
            "phase1_actual_peak_bytes": phase1_actual_peak,
            "manual_plan_covers_phase1": int(
                memory["accounted_global_peak_bytes"] >= phase1_actual_peak
            ),
        }
    )
    memory["accounted_global_peak_bytes"] = max(
        memory["accounted_global_peak_bytes"], phase1_actual_peak
    )
    _require_memory_within_limit(memory, config, phase="actual pass-one")

    collector = _QKVOHookCollector(
        transaction=transaction,
        geometry=geometry,
        sites=sites,
        sketches=sketches,
    )
    parent_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    pass1_sha = ""
    pass2_sha = ""
    replay_blocks = 0
    hooks_registered = 0
    try:
        parent_model.eval()
        with transaction.output_training_scope() as trainable:
            if {id(value) for value in trainable} != factor_ids:
                raise QKVOOutputJacobianAuditTopologyError(
                    "Output-training scope disagrees with the authenticated v2 factors."
                )
            for value in trainable:
                value.grad = None
            plans = _build_replay_plans(
                transaction, parent_model, basis_domain, schedule, config
            )
            collector.install()
            hooks_registered = len(collector.handles)
            try:
                collector.phase = "pass1"
                pass1_sha, replay_blocks = _replay_loss_pass(
                    transaction=transaction,
                    parent_model=parent_model,
                    plans=plans,
                    schedule=schedule,
                    collector=collector,
                    factors=trainable,
                )
                for sketch in sketches.values():
                    sketch.finish_pass1()
                phase2_persistent = sum(
                    sketch.accounted_persistent_bytes for sketch in sketches.values()
                )
                phase2_transient = max(
                    sketch.accounted_batch_transient_bytes(max_rows, 2)
                    for sketch in sketches.values()
                )
                phase2_actual_peak = (
                    phase2_persistent + phase2_transient + wrapper_hook_transient
                )
                memory.update(
                    {
                        "phase2_actual_persistent_bytes": phase2_persistent,
                        "phase2_actual_max_batch_transient_bytes": phase2_transient,
                        "phase2_wrapper_hook_transient_bytes": wrapper_hook_transient,
                        "phase2_actual_peak_bytes": phase2_actual_peak,
                        "manual_plan_covers_phase2": int(
                            memory["accounted_global_peak_bytes"] >= phase2_actual_peak
                        ),
                    }
                )
                memory["accounted_global_peak_bytes"] = max(
                    memory["accounted_global_peak_bytes"], phase2_actual_peak
                )
                _require_memory_within_limit(memory, config, phase="actual pass-two")
                collector.phase = "pass2"
                pass2_sha, pass2_blocks = _replay_loss_pass(
                    transaction=transaction,
                    parent_model=parent_model,
                    plans=plans,
                    schedule=schedule,
                    collector=collector,
                    factors=trainable,
                )
                if pass2_blocks != replay_blocks or pass2_sha != pass1_sha:
                    raise QKVOOutputJacobianAuditReplayError(
                        "Behavior loss replay changed between sketch passes."
                    )
            finally:
                collector.remove()
                collector.current = None
                for value in trainable:
                    value.grad = None
    finally:
        collector.remove()
        collector.current = None
        for module, training in parent_modes:
            module.training = training
        for _name, value in factor_entries:
            value.grad = None

    expected_observations = {
        (role, row) for role in _ROLES for row in range(len(sites))
    }
    if set(collector.observations) != expected_observations or any(
        counts != [replay_blocks, replay_blocks]
        for counts in collector.observations.values()
    ):
        raise QKVOOutputJacobianAuditReplayError(
            "Not all 120 sites and four roles were observed identically in both passes."
        )

    sketch_results = {key: sketch.finalize() for key, sketch in sketches.items()}
    report_pairs = [
        (role, sketch_results[(role, row)].report)
        for role in _ROLES
        for row in range(len(sites))
    ]
    aggregate, per_role = _aggregate_reports(report_pairs, config=config)

    guided: dict[str, Tensor] = {}
    scale_records: dict[str, list[float]] = {role: [] for role in _ROLES}
    for role in _ROLES:
        inherited_right = (
            getattr(registry, f"{role}_right")
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        rows: list[Tensor] = []
        for site_row in range(len(sites)):
            norms = torch.linalg.vector_norm(inherited_right[site_row], dim=1)
            scale = float(torch.median(norms).item())
            if not math.isfinite(scale) or scale <= 0.0:
                raise QKVOOutputJacobianAuditReplayError(
                    "An inherited-right site has no finite positive median row norm."
                )
            rows.append(
                sketch_results[(role, site_row)]
                .guided_right_rows.detach()
                .to(device="cpu", dtype=torch.float32)
                .mul(scale)
                .contiguous()
            )
            scale_records[role].append(scale)
        guided[role] = torch.stack(rows, dim=0)

    expected_shapes = {
        "q": (120, 8, 1536),
        "k": (120, 8, 1536),
        "v": (120, 8, 1536),
        "o": (120, 8, 512),
    }
    if any(tuple(guided[role].shape) != expected_shapes[role] for role in _ROLES):
        raise QKVOOutputJacobianAuditTopologyError(
            "Scaled guided rows do not have the exact stage-one pack shapes."
        )
    factor_tensors = {_PACK_NAMES[role]: guided[role] for role in _ROLES}
    basis_manifest = [
        _tensor_manifest_row(name, value) for name, value in factor_tensors.items()
    ]
    basis_manifest_sha256 = _canonical_sha256(basis_manifest)
    basis_total_numel = sum(int(value.numel()) for value in guided.values())
    if basis_total_numel != 4_915_200:
        raise QKVOOutputJacobianAuditTopologyError(
            "The guided QKVO basis must contain exactly 4,915,200 elements."
        )

    transaction.assert_live()
    factor_state_after = _trainer.capture_target_native_factor_state(transaction)
    factor_versions_after = _version_state_sha256(factor_entries)
    inherited_after = _qkvo_factor_state_sha256(registry)
    native_after = _parameter_state(transaction.model, excluded_ids=factor_ids)
    native_sha_after = _parameter_state_sha256(native_after)
    gradients_cleared = all(value.grad is None for _name, value in factor_entries)
    replay_match = pass1_sha == pass2_sha
    mutation_safe = (
        factor_state_after.factor_state_sha256
        == factor_state_before.factor_state_sha256
        and factor_versions_after == factor_versions_before
        and inherited_after == inherited_before
        and native_sha_after == native_sha_before
        and native_after == native_before
    )
    passed = bool(
        aggregate["normal_fraction_gate_passed"]
        and aggregate["guided_over_random_gate_passed"]
        and replay_match
        and mutation_safe
        and gradients_cleared
        and collector.active_hook_count == 0
        and transaction.is_live
    )

    site_reports = []
    for role in _ROLES:
        for row, (layer, group) in enumerate(sites):
            site_reports.append(
                {
                    "role": role,
                    "site_row": row,
                    "layer": layer,
                    "physical_group": group,
                    "inherited_right_median_row_norm": scale_records[role][row],
                    "sketch_report": sketch_results[(role, row)].report.to_dict(),
                }
            )
    report_payload: dict[str, Any] = {
        "schema": QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
        "passed": passed,
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "site_count": 120,
        "role_count": 4,
        "sketch_count": 480,
        "basis_prompt_count": 24,
        "basis_domain_sha256": basis_domain.domain_sha256,
        "basis_artifact_sha256": basis_domain.artifact_sha256,
        "basis_source_role": "source-exposed-candidate",
        "source_exposed_candidate_as_fit": True,
        "new_candidate_accessed": False,
        "sealed_accessed": False,
        "optimizer_created": False,
        "optimizer_used": False,
        "optimizer_steps": 0,
        "hooks_registered": hooks_registered,
        "hooks_removed": collector.active_hook_count == 0,
        "active_hook_count_after": collector.active_hook_count,
        "transaction_live_after": transaction.is_live,
        "rollback_compatible_after": transaction.is_live,
        "native_state_sha256_before": native_sha_before,
        "native_state_sha256_after": native_sha_after,
        "factor_state_sha256_before": factor_state_before.factor_state_sha256,
        "factor_state_sha256_after": factor_state_after.factor_state_sha256,
        "inherited_qkvo_state_sha256_before": inherited_before,
        "inherited_qkvo_state_sha256_after": inherited_after,
        "factor_parameter_versions_sha256_before": factor_versions_before,
        "factor_parameter_versions_sha256_after": factor_versions_after,
        "factor_gradients_cleared": gradients_cleared,
        "pass1_replay_sha256": pass1_sha,
        "pass2_replay_sha256": pass2_sha,
        "replay_match": replay_match,
        "replay_loss_block_count": replay_blocks,
        "accounted_global_peak_bytes": memory["accounted_global_peak_bytes"],
        "max_bytes": config.max_bytes,
        "memory_accounting": memory,
        "aggregate": aggregate,
        "per_role": per_role,
        "basis_manifest": basis_manifest,
        "basis_manifest_sha256": basis_manifest_sha256,
        "basis_total_numel": basis_total_numel,
        "source_trainer_helper_sha256": _source_helper_sha256(),
        "site_reports": site_reports,
        "internal_weight_matching_used": False,
        "activation_targets_used": False,
        "new_tranche_installed": False,
    }
    report_payload["report_sha256"] = _canonical_sha256(report_payload)
    report = QKVOOutputJacobianAuditReport(MappingProxyType(dict(report_payload)))
    return QKVOOutputJacobianAuditResult(
        guided_right_rows=MappingProxyType(guided),
        report=report,
    )


__all__ = [
    "GUIDED_QKVO_RIGHT_BASIS_SCHEMA",
    "QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA",
    "QKVOOutputJacobianAuditConfig",
    "QKVOOutputJacobianAuditError",
    "QKVOOutputJacobianAuditGateError",
    "QKVOOutputJacobianAuditLimitError",
    "QKVOOutputJacobianAuditReplayError",
    "QKVOOutputJacobianAuditReport",
    "QKVOOutputJacobianAuditResult",
    "QKVOOutputJacobianAuditTopologyError",
    "run_qkvo_output_jacobian_audit",
]
