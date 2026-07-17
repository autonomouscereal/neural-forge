"""Target-native CPCF value/output folding across attention head dimensions.

The score path is compiled separately by
``hf_lfm2_cpcf_attention_dimension_fold``.  This module consumes its fixed
target probabilities and completes the D64 -> D32 value/output half without
executing either model:

* every source KV/value head is assigned exactly once to a positive
  barycentric target group;
* fit-only parent value evidence defines one target-dimensional feature basis
  per target KV head;
* actual child full-prefix residuals are ridge-fitted to dense target V
  projections;
* the supplied fixed target probabilities produce the actual target contexts;
* a dense target O projection is ridge-fitted to the post-operator boundary
  write; and
* disjoint validation cases gate value, context, and residual-endpoint
  behavior.

The compiler uses bounded CPU float64 eigendecompositions and ridge solves
under ``torch.no_grad()``.  It performs no forward, backward, optimizer,
training, pruning, selection, quantization, or fallback operation.  Target
templates contribute shape and dtype only and all evidence/templates are
authenticated before and after compilation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (
    Lfm2CpcfAttentionDimensionTopology,
    Lfm2CpcfBarycentricHeadGroup,
)


LFM2_CPCF_ATTENTION_VALUE_DIMENSION_FOLD_SCHEMA = (
    "neural-forge-lfm2-cpcf-attention-value-dimension-fold-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2CpcfAttentionValueDimensionFoldError(ValueError):
    """The fixed-score value/output evidence or topology is invalid."""


class Lfm2CpcfAttentionValueDimensionValidationError(
    Lfm2CpcfAttentionValueDimensionFoldError
):
    """A compiled V/O contraction failed enforced held-out behavior gates."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = MappingProxyType(dict(report))


class Lfm2CpcfAttentionValueDimensionInstallError(RuntimeError):
    """A V/O installation or exact rollback failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for CPCF attention value-dimension folding."
        ) from exc
    return torch


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
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "CPCF V/O metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be finite and nonnegative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be finite and nonnegative."
        )
    return result


def _fraction(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result > 1.0:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be within [0, 1]."
        )
    return result


def _bounded_float(value: Any, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be within [{lower}, {upper}]."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be within [{lower}, {upper}]."
        ) from exc
    if not math.isfinite(result) or result < lower or result > upper:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be within [{lower}, {upper}]."
        )
    return result


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be a materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    body = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256(_canonical_json(body))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bundle_sha256(values: Mapping[str, Any]) -> str:
    return _sha(
        {
            name: _tensor_sha256(value, name)
            for name, value in sorted(values.items())
        }
    )


def _float_tensor(value: Any, name: str, *, dimensions: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be a nonempty dense real rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} contains NaN or infinity."
        )
    return result


def _preserved_float_tensor(value: Any, name: str, *, dimensions: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be a nonempty dense real rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result.to(torch.float32)).all().item()):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} contains NaN or infinity."
        )
    return result


def _bool_tensor(
    value: Any | None,
    name: str,
    *,
    shape: tuple[int, ...],
) -> Any:
    torch = _torch()
    if value is None:
        return torch.ones(shape, dtype=torch.bool)
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.bool
        or tuple(int(size) for size in value.shape) != shape
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be bool with shape {shape}."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _template(value: Any, name: str) -> tuple[Any, tuple[int, ...]]:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} must be a materialized dense real rank-2 target template."
        )
    return value.dtype, tuple(int(size) for size in value.shape)


def _case_tensor_map(
    case: "Lfm2CpcfAttentionValueDimensionCase",
) -> dict[str, Any]:
    values = {
        "actual_child_key_value_residual_history": (
            case.actual_child_key_value_residual_history
        ),
        "target_query_key_probabilities": case.target_query_key_probabilities,
        "parent_value_history": case.parent_value_history,
        "parent_query_key_probabilities": case.parent_query_key_probabilities,
        "parent_context": case.parent_context,
        "actual_child_pre_operator_residual": (
            case.actual_child_pre_operator_residual
        ),
        "parent_post_operator_residual_target_space": (
            case.parent_post_operator_residual_target_space
        ),
    }
    if case.valid_key_mask is not None:
        values["valid_key_mask"] = case.valid_key_mask
    if case.valid_query_mask is not None:
        values["valid_query_mask"] = case.valid_query_mask
    return values


def _case_payload_sha256(
    case: "Lfm2CpcfAttentionValueDimensionCase",
) -> str:
    return _sha(
        {
            "fixed_query_key_sha256": case.fixed_query_key_sha256,
            "tensors": {
                name: _tensor_sha256(value, f"case.{name}")
                for name, value in sorted(_case_tensor_map(case).items())
            },
        }
    )


def _case_authentication_sha256(
    case: "Lfm2CpcfAttentionValueDimensionCase",
) -> str:
    return _sha(
        {
            "schema": LFM2_CPCF_ATTENTION_VALUE_DIMENSION_FOLD_SCHEMA,
            "scope": (
                "fixed_target_qk_probabilities_plus_full_parent_value_context_"
                "chain_and_target_residual_endpoint"
            ),
            "case_id": case.case_id,
            "split": case.split,
            "payload_sha256": _case_payload_sha256(case),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionValueDimensionCase:
    """One fixed-score causal route aligned to an actual child prefix."""

    case_id: str
    split: str
    fixed_query_key_sha256: str
    actual_child_key_value_residual_history: Any
    target_query_key_probabilities: Any
    parent_value_history: Any
    parent_query_key_probabilities: Any
    parent_context: Any
    actual_child_pre_operator_residual: Any
    parent_post_operator_residual_target_space: Any
    valid_key_mask: Any | None = None
    valid_query_mask: Any | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        split = str(self.split).strip().lower()
        if not case_id:
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "case_id must be nonempty text."
            )
        if split not in {"fit", "validation"}:
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "split must be 'fit' or 'validation'."
            )
        _digest(self.fixed_query_key_sha256, "fixed_query_key_sha256")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "split", split)
        digest = _case_authentication_sha256(self)
        if self.authentication_sha256 and self.authentication_sha256 != digest:
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                f"Case {case_id!r} authentication digest does not match."
            )
        object.__setattr__(self, "authentication_sha256", digest)

    @property
    def payload_sha256(self) -> str:
        return _case_payload_sha256(self)

    def verify_authentication(self) -> bool:
        return _case_authentication_sha256(self) == self.authentication_sha256


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionValueDimensionPolicy:
    """Fit-capacity and disjoint held-out behavior gates."""

    minimum_fit_value_basis_energy_fraction: float = 0.90
    maximum_fit_value_relative_rmse: float = 1.0
    maximum_validation_value_relative_rmse: float = 1.0
    maximum_fit_context_relative_rmse: float = 1.0
    maximum_validation_context_relative_rmse: float = 1.0
    minimum_validation_context_cosine_similarity: float = 0.0
    maximum_fit_endpoint_relative_rmse: float = 1.0
    maximum_validation_endpoint_relative_rmse: float = 1.0
    minimum_validation_endpoint_cosine_similarity: float = 0.0
    authentication_rtol: float = 2e-5
    authentication_atol: float = 2e-6
    enforce: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_fit_value_basis_energy_fraction",
            _fraction(
                self.minimum_fit_value_basis_energy_fraction,
                "minimum_fit_value_basis_energy_fraction",
            ),
        )
        for name in (
            "maximum_fit_value_relative_rmse",
            "maximum_validation_value_relative_rmse",
            "maximum_fit_context_relative_rmse",
            "maximum_validation_context_relative_rmse",
            "maximum_fit_endpoint_relative_rmse",
            "maximum_validation_endpoint_relative_rmse",
            "authentication_rtol",
            "authentication_atol",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name),
            )
        for name in (
            "minimum_validation_context_cosine_similarity",
            "minimum_validation_endpoint_cosine_similarity",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_float(
                    getattr(self, name),
                    name,
                    lower=-1.0,
                    upper=1.0,
                ),
            )
        if not isinstance(self.enforce, bool):
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "enforce must be boolean."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionValueDimensionLimits:
    """Hard ceilings for bounded analytical V/O compilation."""

    maximum_fit_cases: int = 64
    maximum_validation_cases: int = 64
    maximum_batch: int = 256
    maximum_query_tokens: int = 512
    maximum_key_tokens: int = 4096
    maximum_source_query_heads: int = 128
    maximum_source_key_value_heads: int = 128
    maximum_source_head_dim: int = 256
    maximum_target_hidden_width: int = 16384
    maximum_fit_value_rows: int = 1_048_576
    maximum_fit_endpoint_rows: int = 1_048_576

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(slots=True)
class _NormalizedCase:
    case_id: str
    split: str
    fixed_query_key_sha256: str
    child_key_value_input: Any
    target_probabilities: Any
    parent_values: Any
    parent_probabilities: Any
    parent_context: Any
    child_pre_residual: Any
    parent_endpoint: Any
    key_mask: Any
    query_mask: Any
    authentication_sha256: str
    payload_sha256: str


def _validate_groups(
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> None:
    if (
        len(query_groups) != topology.target_query_heads
        or len(key_value_groups) != topology.target_key_value_heads
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Group counts must equal target query and KV head counts."
        )

    def assign(
        groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
        source_count: int,
        role: str,
    ) -> tuple[int, ...]:
        destinations = [-1] * source_count
        for target, group in enumerate(groups):
            if not isinstance(group, Lfm2CpcfBarycentricHeadGroup):
                raise TypeError(
                    "All head groups must be Lfm2CpcfBarycentricHeadGroup."
                )
            for source in group.source_indices:
                if source >= source_count or destinations[source] >= 0:
                    raise Lfm2CpcfAttentionValueDimensionFoldError(
                        f"{role} groups must assign every source head exactly once."
                    )
                destinations[source] = target
        if any(value < 0 for value in destinations):
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                f"{role} groups dropped at least one source head."
            )
        return tuple(destinations)

    query_assignment = assign(
        query_groups,
        topology.source_query_heads,
        "Query",
    )
    key_assignment = assign(
        key_value_groups,
        topology.source_key_value_heads,
        "Key/value",
    )
    for source_query, target_query in enumerate(query_assignment):
        source_key_value = (
            source_query // topology.source_queries_per_key_value
        )
        target_key_value = (
            target_query // topology.target_queries_per_key_value
        )
        if key_assignment[source_key_value] != target_key_value:
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "Query and key/value groups violate source-to-target GQA coupling."
            )


def _authenticate_probabilities(
    probabilities: Any,
    *,
    key_mask: Any,
    query_mask: Any,
    name: str,
    rtol: float,
    atol: float,
) -> None:
    torch = _torch()
    if bool((probabilities < -atol).any().item()):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} contains a negative probability."
        )
    batch, heads, query_length, key_length = map(int, probabilities.shape)
    expanded_key = key_mask.reshape(batch, 1, 1, key_length).expand_as(
        probabilities
    )
    expanded_query = query_mask.reshape(
        batch, 1, query_length, 1
    ).expand_as(probabilities)
    valid = expanded_key & expanded_query
    if bool((probabilities.masked_select(~valid).abs() > atol).any().item()):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} leaks probability into an invalid query/key position."
        )
    offset = key_length - query_length
    query_positions = torch.arange(query_length).reshape(1, 1, query_length, 1)
    key_positions = torch.arange(key_length).reshape(1, 1, 1, key_length)
    future = key_positions > (offset + query_positions)
    if bool((probabilities.masked_select(future.expand_as(probabilities)).abs() > atol).any().item()):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} leaks probability into a future causal position."
        )
    sums = probabilities.sum(dim=-1)
    desired = query_mask.reshape(batch, 1, query_length).expand(
        batch, heads, query_length
    ).to(torch.float64)
    if not bool(torch.allclose(sums, desired, rtol=rtol, atol=atol)):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{name} does not sum to one on every valid query."
        )


def _normalize_case(
    case: Lfm2CpcfAttentionValueDimensionCase,
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    policy: Lfm2CpcfAttentionValueDimensionPolicy,
    limits: Lfm2CpcfAttentionValueDimensionLimits,
) -> _NormalizedCase:
    torch = _torch()
    if not isinstance(case, Lfm2CpcfAttentionValueDimensionCase):
        raise TypeError(
            "cases must contain Lfm2CpcfAttentionValueDimensionCase values."
        )
    if not case.verify_authentication():
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"Case {case.case_id!r} failed evidence authentication."
        )
    child_input = _float_tensor(
        case.actual_child_key_value_residual_history,
        f"{case.case_id}.actual_child_key_value_residual_history",
        dimensions=3,
    )
    target_probabilities = _float_tensor(
        case.target_query_key_probabilities,
        f"{case.case_id}.target_query_key_probabilities",
        dimensions=4,
    )
    parent_values = _float_tensor(
        case.parent_value_history,
        f"{case.case_id}.parent_value_history",
        dimensions=4,
    )
    parent_probabilities = _float_tensor(
        case.parent_query_key_probabilities,
        f"{case.case_id}.parent_query_key_probabilities",
        dimensions=4,
    )
    parent_context = _float_tensor(
        case.parent_context,
        f"{case.case_id}.parent_context",
        dimensions=4,
    )
    child_pre = _float_tensor(
        case.actual_child_pre_operator_residual,
        f"{case.case_id}.actual_child_pre_operator_residual",
        dimensions=3,
    )
    endpoint = _float_tensor(
        case.parent_post_operator_residual_target_space,
        f"{case.case_id}.parent_post_operator_residual_target_space",
        dimensions=3,
    )
    batch, key_length, target_hidden = map(int, child_input.shape)
    query_length = int(child_pre.shape[1])
    expected_target_probability = (
        batch,
        topology.target_query_heads,
        query_length,
        key_length,
    )
    expected_parent_value = (
        batch,
        topology.source_key_value_heads,
        key_length,
        topology.source_head_dim,
    )
    expected_parent_probability = (
        batch,
        topology.source_query_heads,
        query_length,
        key_length,
    )
    expected_parent_context = (
        batch,
        topology.source_query_heads,
        query_length,
        topology.source_head_dim,
    )
    if (
        query_length > key_length
        or tuple(target_probabilities.shape) != expected_target_probability
        or tuple(parent_values.shape) != expected_parent_value
        or tuple(parent_probabilities.shape) != expected_parent_probability
        or tuple(parent_context.shape) != expected_parent_context
        or tuple(child_pre.shape) != (batch, query_length, target_hidden)
        or tuple(endpoint.shape) != tuple(child_pre.shape)
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"Case {case.case_id!r} tensors do not match the attention topology."
        )
    if (
        batch > limits.maximum_batch
        or query_length > limits.maximum_query_tokens
        or key_length > limits.maximum_key_tokens
        or target_hidden > limits.maximum_target_hidden_width
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"Case {case.case_id!r} exceeds an analytical shape ceiling."
        )
    key_mask = _bool_tensor(
        case.valid_key_mask,
        f"{case.case_id}.valid_key_mask",
        shape=(batch, key_length),
    )
    query_mask = _bool_tensor(
        case.valid_query_mask,
        f"{case.case_id}.valid_query_mask",
        shape=(batch, query_length),
    )
    if bool((query_mask.sum(dim=-1) == 0).any().item()):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"Case {case.case_id!r} has a sequence with no valid query."
        )
    _authenticate_probabilities(
        target_probabilities,
        key_mask=key_mask,
        query_mask=query_mask,
        name=f"{case.case_id}.target_query_key_probabilities",
        rtol=policy.authentication_rtol,
        atol=policy.authentication_atol,
    )
    _authenticate_probabilities(
        parent_probabilities,
        key_mask=key_mask,
        query_mask=query_mask,
        name=f"{case.case_id}.parent_query_key_probabilities",
        rtol=policy.authentication_rtol,
        atol=policy.authentication_atol,
    )
    repeated_parent_values = parent_values.repeat_interleave(
        topology.source_queries_per_key_value,
        dim=1,
    )
    reconstructed_parent_context = torch.matmul(
        parent_probabilities,
        repeated_parent_values,
    )
    query_valid = query_mask.reshape(batch, 1, query_length, 1).expand_as(
        parent_context
    )
    if not bool(
        torch.allclose(
            reconstructed_parent_context.masked_select(query_valid),
            parent_context.masked_select(query_valid),
            rtol=policy.authentication_rtol,
            atol=policy.authentication_atol,
        )
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"Case {case.case_id!r} failed parent probability->value->context "
            "authentication."
        )
    return _NormalizedCase(
        case_id=case.case_id,
        split=case.split,
        fixed_query_key_sha256=case.fixed_query_key_sha256,
        child_key_value_input=child_input,
        target_probabilities=target_probabilities,
        parent_values=parent_values,
        parent_probabilities=parent_probabilities,
        parent_context=parent_context,
        child_pre_residual=child_pre,
        parent_endpoint=endpoint,
        key_mask=key_mask,
        query_mask=query_mask,
        authentication_sha256=case.authentication_sha256,
        payload_sha256=case.payload_sha256,
    )


def _weighted_head_group(value: Any, group: Lfm2CpcfBarycentricHeadGroup) -> Any:
    torch = _torch()
    indices = torch.tensor(group.source_indices, dtype=torch.long)
    weights = torch.tensor(group.weights, dtype=torch.float64)
    selected = value.index_select(1, indices)
    shape = [1, len(group.weights)] + [1] * (value.ndim - 2)
    return (selected * weights.reshape(shape)).sum(dim=1)


def _source_value_barycenters(
    case: _NormalizedCase,
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> Any:
    return _torch().stack(
        [
            _weighted_head_group(case.parent_values, group)
            for group in key_value_groups
        ],
        dim=1,
    )


def _source_context_barycenters(
    case: _NormalizedCase,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> Any:
    return _torch().stack(
        [
            _weighted_head_group(case.parent_context, group)
            for group in query_groups
        ],
        dim=1,
    )


def _canonicalize_basis_signs(basis: Any) -> Any:
    torch = _torch()
    result = basis.clone()
    for column in range(int(result.shape[1])):
        vector = result[:, column]
        anchor = int(torch.argmax(vector.abs()).item())
        if float(vector[anchor].item()) < 0.0:
            result[:, column].neg_()
    return result.contiguous()


def _fit_value_bases(
    fit_cases: Sequence[_NormalizedCase],
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> tuple[tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
    torch = _torch()
    barycenters = [
        _source_value_barycenters(case, key_value_groups)
        for case in fit_cases
    ]
    bases = []
    reports = []
    for target_head in range(topology.target_key_value_heads):
        rows = torch.cat(
            [
                values[:, target_head].reshape(
                    -1,
                    topology.source_head_dim,
                )[case.key_mask.reshape(-1)]
                for case, values in zip(fit_cases, barycenters, strict=True)
            ],
            dim=0,
        )
        covariance = rows.T @ rows
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        ordered_values = eigenvalues.index_select(0, order).clamp_min(0.0)
        basis = eigenvectors.index_select(1, order[: topology.target_head_dim])
        basis = _canonicalize_basis_signs(basis)
        total = ordered_values.sum()
        if float(total.item()) <= torch.finfo(torch.float64).tiny:
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                f"Target KV head {target_head} has zero fit-only value energy."
            )
        retained = ordered_values[: topology.target_head_dim].sum() / total
        report_body = {
            "target_key_value_head": target_head,
            "fit_rows": int(rows.shape[0]),
            "source_head_dim": topology.source_head_dim,
            "target_head_dim": topology.target_head_dim,
            "retained_fit_energy_fraction": float(retained.item()),
            "basis_sha256": _tensor_sha256(
                basis,
                f"value_basis.{target_head}",
            ),
            "basis_orthonormal": bool(
                torch.allclose(
                    basis.T @ basis,
                    torch.eye(
                        topology.target_head_dim,
                        dtype=torch.float64,
                    ),
                    rtol=1e-10,
                    atol=1e-10,
                )
            ),
            "validation_evidence_used": False,
            "solver": "fit_only_symmetric_eigendecomposition",
        }
        bases.append(basis)
        reports.append(
            MappingProxyType(
                {
                    **report_body,
                    "report_sha256": _sha(report_body),
                }
            )
        )
    return tuple(bases), tuple(reports)


def _project_values(
    case: _NormalizedCase,
    *,
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    bases: tuple[Any, ...],
) -> Any:
    barycenters = _source_value_barycenters(case, key_value_groups)
    return _torch().stack(
        [
            barycenters[:, target_head] @ bases[target_head]
            for target_head in range(len(bases))
        ],
        dim=1,
    )


def _project_contexts(
    case: _NormalizedCase,
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    bases: tuple[Any, ...],
) -> Any:
    barycenters = _source_context_barycenters(case, query_groups)
    return _torch().stack(
        [
            barycenters[:, target_query]
            @ bases[
                target_query // topology.target_queries_per_key_value
            ]
            for target_query in range(topology.target_query_heads)
        ],
        dim=1,
    )


def _ridge_fit(inputs: Any, targets: Any, *, ridge: float, role: str) -> Any:
    torch = _torch()
    if (
        inputs.ndim != 2
        or targets.ndim != 2
        or int(inputs.shape[0]) != int(targets.shape[0])
        or int(inputs.shape[0]) == 0
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{role} ridge rows are empty or misaligned."
        )
    gram = inputs.T @ inputs
    regularized = gram + ridge * torch.eye(
        int(gram.shape[0]),
        dtype=torch.float64,
    )
    try:
        solution = torch.linalg.solve(
            regularized,
            inputs.T @ targets,
        )
    except RuntimeError as exc:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{role} ridge solve failed; no fallback is permitted."
        ) from exc
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            f"{role} ridge solve produced NaN or infinity."
        )
    return solution.T.contiguous()


def _target_values(
    case: _NormalizedCase,
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    value_weight: Any,
) -> Any:
    batch, key_length, input_width = map(
        int,
        case.child_key_value_input.shape,
    )
    projected = case.child_key_value_input.reshape(-1, input_width) @ value_weight.T
    return (
        projected.reshape(
            batch,
            key_length,
            topology.target_key_value_heads,
            topology.target_head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _target_contexts(
    case: _NormalizedCase,
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    target_values: Any,
) -> Any:
    repeated_values = target_values.repeat_interleave(
        topology.target_queries_per_key_value,
        dim=1,
    )
    return _torch().matmul(case.target_probabilities, repeated_values)


def _context_rows(contexts: Any) -> Any:
    return contexts.permute(0, 2, 1, 3).reshape(
        int(contexts.shape[0]) * int(contexts.shape[2]),
        int(contexts.shape[1]) * int(contexts.shape[3]),
    )


def _masked_rows(value: Any, mask: Any, *, sequence_axis: int) -> Any:
    if sequence_axis == 2:
        rows = value.permute(0, 2, 1, 3).reshape(
            int(value.shape[0]) * int(value.shape[2]),
            int(value.shape[1]) * int(value.shape[3]),
        )
    elif sequence_axis == 1:
        rows = value.reshape(
            int(value.shape[0]) * int(value.shape[1]),
            -1,
        )
    else:  # pragma: no cover - internal invariant
        raise AssertionError("Unsupported sequence axis.")
    return rows[mask.reshape(-1)]


def _metrics(prediction: Any, target: Any) -> Mapping[str, Any]:
    torch = _torch()
    if tuple(prediction.shape) != tuple(target.shape) or prediction.numel() == 0:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Metric tensors are empty or misaligned."
        )
    difference = prediction - target
    error = torch.sqrt(torch.mean(difference.square()))
    scale = torch.sqrt(torch.mean(target.square())).clamp_min(
        torch.finfo(torch.float64).eps
    )
    left = prediction.reshape(-1)
    right = target.reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 and float(right_norm.item()) == 0.0:
        cosine = 1.0
    elif float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(left, right).div(left_norm * right_norm).item())
    return MappingProxyType(
        {
            "rows": int(prediction.shape[0]),
            "width": int(prediction.shape[1]),
            "relative_rmse": float((error / scale).item()),
            "cosine_similarity": cosine,
            "maximum_absolute_error": float(difference.abs().max().item()),
            "mean_absolute_error": float(difference.abs().mean().item()),
            "prediction_sha256": _tensor_sha256(
                prediction,
                "metric_prediction",
            ),
            "target_sha256": _tensor_sha256(target, "metric_target"),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionValueOutputReplay:
    """One detached replay of the compiled target-native V/O path."""

    target_values: Any
    desired_values: Any
    actual_target_contexts: Any
    desired_target_contexts: Any
    residual_write: Any
    post_operator_residual: Any

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                value.detach().to(device="cpu").contiguous().clone(),
            )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledCpcfAttentionValueDimensionFold:
    """Dense target V/O state plus fit-only bases and sealed behavior receipt."""

    topology: Lfm2CpcfAttentionDimensionTopology
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...]
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...]
    value_feature_bases: tuple[Any, ...]
    value_weight: Any
    output_weight: Any
    target_hidden_width: int
    fixed_query_key_sha256: str
    fit_metrics: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        _digest(self.fixed_query_key_sha256, "fixed_query_key_sha256")
        value = _preserved_float_tensor(
            self.value_weight,
            "value_weight",
            dimensions=2,
        )
        output = _preserved_float_tensor(
            self.output_weight,
            "output_weight",
            dimensions=2,
        )
        expected_value = (
            self.topology.target_key_value_heads
            * self.topology.target_head_dim,
            self.target_hidden_width,
        )
        expected_output = (
            self.target_hidden_width,
            self.topology.target_query_heads
            * self.topology.target_head_dim,
        )
        if tuple(value.shape) != expected_value or tuple(output.shape) != expected_output:
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "Compiled V/O state does not match the target topology."
            )
        bases = tuple(
            _float_tensor(
                basis,
                f"value_feature_bases[{index}]",
                dimensions=2,
            )
            for index, basis in enumerate(self.value_feature_bases)
        )
        if len(bases) != self.topology.target_key_value_heads or any(
            tuple(basis.shape)
            != (
                self.topology.source_head_dim,
                self.topology.target_head_dim,
            )
            for basis in bases
        ):
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "Value feature bases do not match the source/target head dimensions."
            )
        object.__setattr__(self, "value_feature_bases", bases)
        object.__setattr__(self, "value_weight", value)
        object.__setattr__(self, "output_weight", output)
        object.__setattr__(
            self,
            "fit_metrics",
            MappingProxyType(dict(self.fit_metrics)),
        )
        object.__setattr__(
            self,
            "validation_metrics",
            MappingProxyType(dict(self.validation_metrics)),
        )
        report = dict(self.report)
        report_sha = report.pop("report_sha256", None)
        if (
            not isinstance(report_sha, str)
            or _SHA256.fullmatch(report_sha) is None
            or _sha(report) != report_sha
            or report.get("fixed_query_key_sha256")
            != self.fixed_query_key_sha256
            or report.get("topology") != self.topology.as_dict()
            or report.get("compiled_state_sha256")
            != _tensor_bundle_sha256(
                {
                    "v_proj.weight": value,
                    "out_proj.weight": output,
                }
            )
        ):
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "Compiled V/O receipt seal or state binding is invalid."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "report_sha256": report_sha}),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "v_proj.weight": self.value_weight.detach().cpu().clone(),
                "out_proj.weight": self.output_weight.detach().cpu().clone(),
            }
        )

    def replay(
        self,
        case: Lfm2CpcfAttentionValueDimensionCase,
    ) -> Lfm2CpcfAttentionValueOutputReplay:
        normalized = _normalize_case(
            case,
            topology=self.topology,
            policy=Lfm2CpcfAttentionValueDimensionPolicy(
                minimum_fit_value_basis_energy_fraction=0.0,
                maximum_fit_value_relative_rmse=1e300,
                maximum_validation_value_relative_rmse=1e300,
                maximum_fit_context_relative_rmse=1e300,
                maximum_validation_context_relative_rmse=1e300,
                minimum_validation_context_cosine_similarity=-1.0,
                maximum_fit_endpoint_relative_rmse=1e300,
                maximum_validation_endpoint_relative_rmse=1e300,
                minimum_validation_endpoint_cosine_similarity=-1.0,
                enforce=False,
            ),
            limits=Lfm2CpcfAttentionValueDimensionLimits(),
        )
        if normalized.fixed_query_key_sha256 != self.fixed_query_key_sha256:
            raise Lfm2CpcfAttentionValueDimensionFoldError(
                "Replay case belongs to a different fixed Q/K compilation."
            )
        desired_values = _project_values(
            normalized,
            key_value_groups=self.key_value_groups,
            bases=self.value_feature_bases,
        )
        target_values = _target_values(
            normalized,
            topology=self.topology,
            value_weight=self.value_weight.to(_torch().float64),
        )
        contexts = _target_contexts(
            normalized,
            topology=self.topology,
            target_values=target_values,
        )
        desired_contexts = _project_contexts(
            normalized,
            topology=self.topology,
            query_groups=self.query_groups,
            bases=self.value_feature_bases,
        )
        write = (
            _context_rows(contexts)
            @ self.output_weight.to(_torch().float64).T
        ).reshape_as(normalized.child_pre_residual)
        return Lfm2CpcfAttentionValueOutputReplay(
            target_values=target_values,
            desired_values=desired_values,
            actual_target_contexts=contexts,
            desired_target_contexts=desired_contexts,
            residual_write=write,
            post_operator_residual=normalized.child_pre_residual + write,
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _split_metrics(
    cases: Sequence[_NormalizedCase],
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    bases: tuple[Any, ...],
    value_weight: Any,
    output_weight: Any,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    value_predictions = []
    value_targets = []
    context_predictions = []
    context_targets = []
    endpoint_predictions = []
    endpoint_targets = []
    case_reports = []
    for case in cases:
        desired_values = _project_values(
            case,
            key_value_groups=key_value_groups,
            bases=bases,
        )
        target_values = _target_values(
            case,
            topology=topology,
            value_weight=value_weight,
        )
        target_contexts = _target_contexts(
            case,
            topology=topology,
            target_values=target_values,
        )
        desired_contexts = _project_contexts(
            case,
            topology=topology,
            query_groups=query_groups,
            bases=bases,
        )
        context_rows = _context_rows(target_contexts)
        write = (context_rows @ output_weight.T).reshape_as(
            case.child_pre_residual
        )
        endpoint = case.child_pre_residual + write
        value_prediction = _masked_rows(
            target_values,
            case.key_mask,
            sequence_axis=2,
        )
        value_target = _masked_rows(
            desired_values,
            case.key_mask,
            sequence_axis=2,
        )
        context_prediction = _masked_rows(
            target_contexts,
            case.query_mask,
            sequence_axis=2,
        )
        context_target = _masked_rows(
            desired_contexts,
            case.query_mask,
            sequence_axis=2,
        )
        endpoint_prediction = _masked_rows(
            endpoint,
            case.query_mask,
            sequence_axis=1,
        )
        endpoint_target = _masked_rows(
            case.parent_endpoint,
            case.query_mask,
            sequence_axis=1,
        )
        value_predictions.append(value_prediction)
        value_targets.append(value_target)
        context_predictions.append(context_prediction)
        context_targets.append(context_target)
        endpoint_predictions.append(endpoint_prediction)
        endpoint_targets.append(endpoint_target)
        body = {
            "case_id": case.case_id,
            "split": case.split,
            "authentication_sha256": case.authentication_sha256,
            "payload_sha256": case.payload_sha256,
            "value": dict(_metrics(value_prediction, value_target)),
            "context": dict(_metrics(context_prediction, context_target)),
            "residual_endpoint": dict(
                _metrics(endpoint_prediction, endpoint_target)
            ),
        }
        case_reports.append(
            MappingProxyType({**body, "report_sha256": _sha(body)})
        )
    metrics_body = {
        "case_count": len(cases),
        "value": dict(
            _metrics(
                _torch().cat(value_predictions, dim=0),
                _torch().cat(value_targets, dim=0),
            )
        ),
        "context": dict(
            _metrics(
                _torch().cat(context_predictions, dim=0),
                _torch().cat(context_targets, dim=0),
            )
        ),
        "residual_endpoint": dict(
            _metrics(
                _torch().cat(endpoint_predictions, dim=0),
                _torch().cat(endpoint_targets, dim=0),
            )
        ),
    }
    return (
        MappingProxyType(
            {
                **metrics_body,
                "report_sha256": _sha(metrics_body),
            }
        ),
        tuple(case_reports),
    )


def compile_lfm2_cpcf_attention_value_dimension_fold(
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    cases: Sequence[Lfm2CpcfAttentionValueDimensionCase],
    query_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    key_value_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    target_value_weight_template: Any,
    target_output_weight_template: Any,
    policy: Lfm2CpcfAttentionValueDimensionPolicy | None = None,
    limits: Lfm2CpcfAttentionValueDimensionLimits | None = None,
    ridge: float = 1e-8,
) -> Lfm2CompiledCpcfAttentionValueDimensionFold:
    """Compile dense Dsource->Dtarget V/O under one fixed target Q/K route."""

    torch = _torch()
    if not isinstance(topology, Lfm2CpcfAttentionDimensionTopology):
        raise TypeError(
            "topology must be Lfm2CpcfAttentionDimensionTopology."
        )
    if topology.target_head_dim >= topology.source_head_dim:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "This compiler requires a strict value head-dimension contraction."
        )
    case_values = tuple(cases)
    if not case_values:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "At least one fit and one validation case are required."
        )
    policy_value = policy or Lfm2CpcfAttentionValueDimensionPolicy()
    limits_value = limits or Lfm2CpcfAttentionValueDimensionLimits()
    if not isinstance(policy_value, Lfm2CpcfAttentionValueDimensionPolicy):
        raise TypeError(
            "policy must be Lfm2CpcfAttentionValueDimensionPolicy."
        )
    if not isinstance(limits_value, Lfm2CpcfAttentionValueDimensionLimits):
        raise TypeError(
            "limits must be Lfm2CpcfAttentionValueDimensionLimits."
        )
    ridge_value = _nonnegative_float(ridge, "ridge")
    if ridge_value == 0.0:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "ridge must be positive for bounded deterministic solves."
        )
    query_group_values = tuple(query_groups)
    key_group_values = tuple(key_value_groups)
    _validate_groups(topology, query_group_values, key_group_values)
    if (
        topology.source_query_heads > limits_value.maximum_source_query_heads
        or topology.source_key_value_heads
        > limits_value.maximum_source_key_value_heads
        or topology.source_head_dim > limits_value.maximum_source_head_dim
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Attention topology exceeds an analytical ceiling."
        )

    value_dtype, value_shape = _template(
        target_value_weight_template,
        "target_value_weight_template",
    )
    output_dtype, output_shape = _template(
        target_output_weight_template,
        "target_output_weight_template",
    )
    target_hidden = int(value_shape[1])
    expected_value_shape = (
        topology.target_key_value_heads * topology.target_head_dim,
        target_hidden,
    )
    expected_output_shape = (
        target_hidden,
        topology.target_query_heads * topology.target_head_dim,
    )
    if value_shape != expected_value_shape or output_shape != expected_output_shape:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Target V/O templates do not match the target attention topology."
        )
    if target_hidden > limits_value.maximum_target_hidden_width:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Target hidden width exceeds the analytical ceiling."
        )
    template_inputs = {
        "v_proj.weight": target_value_weight_template,
        "out_proj.weight": target_output_weight_template,
    }
    evidence_inputs = {
        f"{case.split}.{index:04d}.{name}": value
        for index, case in enumerate(case_values)
        for name, value in _case_tensor_map(case).items()
    }
    templates_before = _tensor_bundle_sha256(template_inputs)
    evidence_before = _tensor_bundle_sha256(evidence_inputs)

    normalized = tuple(
        _normalize_case(
            case,
            topology=topology,
            policy=policy_value,
            limits=limits_value,
        )
        for case in case_values
    )
    fit_cases = tuple(case for case in normalized if case.split == "fit")
    validation_cases = tuple(
        case for case in normalized if case.split == "validation"
    )
    if not fit_cases or not validation_cases:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Disjoint nonempty fit and validation cases are required."
        )
    if (
        len(fit_cases) > limits_value.maximum_fit_cases
        or len(validation_cases) > limits_value.maximum_validation_cases
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Case count exceeds an analytical ceiling."
        )
    fit_payloads = {case.payload_sha256 for case in fit_cases}
    validation_payloads = {
        case.payload_sha256 for case in validation_cases
    }
    if fit_payloads & validation_payloads:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Fit and validation evidence payloads must be disjoint."
        )
    fixed_qk_values = {
        case.fixed_query_key_sha256 for case in normalized
    }
    if len(fixed_qk_values) != 1:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Every V/O case must bind the same fixed Q/K compilation."
        )
    fixed_qk_sha = next(iter(fixed_qk_values))
    widths = {
        int(case.child_key_value_input.shape[-1])
        for case in normalized
    }
    if widths != {target_hidden}:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Actual child KV inputs must all use the target hidden width."
        )

    fit_value_rows = sum(
        int(case.key_mask.sum().item())
        for case in fit_cases
    )
    fit_endpoint_rows = sum(
        int(case.query_mask.sum().item())
        for case in fit_cases
    )
    if (
        fit_value_rows > limits_value.maximum_fit_value_rows
        or fit_endpoint_rows > limits_value.maximum_fit_endpoint_rows
    ):
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Fit rows exceed a bounded analytical ceiling."
        )

    with torch.no_grad():
        bases, basis_reports = _fit_value_bases(
            fit_cases,
            topology=topology,
            key_value_groups=key_group_values,
        )
        minimum_energy = min(
            float(report["retained_fit_energy_fraction"])
            for report in basis_reports
        )
        fit_inputs = torch.cat(
            [
                case.child_key_value_input.reshape(-1, target_hidden)[
                    case.key_mask.reshape(-1)
                ]
                for case in fit_cases
            ],
            dim=0,
        )
        fit_value_targets = torch.cat(
            [
                _masked_rows(
                    _project_values(
                        case,
                        key_value_groups=key_group_values,
                        bases=bases,
                    ),
                    case.key_mask,
                    sequence_axis=2,
                )
                for case in fit_cases
            ],
            dim=0,
        )
        value_fp64 = _ridge_fit(
            fit_inputs,
            fit_value_targets,
            ridge=ridge_value,
            role="target_value_projection",
        )
        compiled_value = (
            value_fp64.to(device="cpu", dtype=value_dtype)
            .contiguous()
            .clone()
        )
        materialized_value = compiled_value.to(torch.float64)

        output_inputs = []
        output_targets = []
        for case in fit_cases:
            values = _target_values(
                case,
                topology=topology,
                value_weight=materialized_value,
            )
            contexts = _target_contexts(
                case,
                topology=topology,
                target_values=values,
            )
            output_inputs.append(
                _context_rows(contexts)[case.query_mask.reshape(-1)]
            )
            output_targets.append(
                (
                    case.parent_endpoint - case.child_pre_residual
                ).reshape(-1, target_hidden)[case.query_mask.reshape(-1)]
            )
        output_design = torch.cat(output_inputs, dim=0)
        output_target = torch.cat(output_targets, dim=0)
        output_fp64 = _ridge_fit(
            output_design,
            output_target,
            ridge=ridge_value,
            role="target_output_projection",
        )
        compiled_output = (
            output_fp64.to(device="cpu", dtype=output_dtype)
            .contiguous()
            .clone()
        )
        materialized_output = compiled_output.to(torch.float64)
        fit_metrics, fit_case_reports = _split_metrics(
            fit_cases,
            topology=topology,
            query_groups=query_group_values,
            key_value_groups=key_group_values,
            bases=bases,
            value_weight=materialized_value,
            output_weight=materialized_output,
        )
        validation_metrics, validation_case_reports = _split_metrics(
            validation_cases,
            topology=topology,
            query_groups=query_group_values,
            key_value_groups=key_group_values,
            bases=bases,
            value_weight=materialized_value,
            output_weight=materialized_output,
        )

    templates_after = _tensor_bundle_sha256(template_inputs)
    evidence_after = _tensor_bundle_sha256(evidence_inputs)
    templates_unchanged = templates_before == templates_after
    evidence_unchanged = evidence_before == evidence_after
    if not templates_unchanged or not evidence_unchanged:
        raise Lfm2CpcfAttentionValueDimensionFoldError(
            "Compiler evidence or target templates changed during folding."
        )
    compiled_state = {
        "v_proj.weight": compiled_value,
        "out_proj.weight": compiled_output,
    }
    compiled_state_sha = _tensor_bundle_sha256(compiled_state)
    checks = {
        "minimum_fit_value_basis_energy_fraction": (
            minimum_energy
            >= policy_value.minimum_fit_value_basis_energy_fraction
        ),
        "maximum_fit_value_relative_rmse": (
            float(fit_metrics["value"]["relative_rmse"])
            <= policy_value.maximum_fit_value_relative_rmse
        ),
        "maximum_validation_value_relative_rmse": (
            float(validation_metrics["value"]["relative_rmse"])
            <= policy_value.maximum_validation_value_relative_rmse
        ),
        "maximum_fit_context_relative_rmse": (
            float(fit_metrics["context"]["relative_rmse"])
            <= policy_value.maximum_fit_context_relative_rmse
        ),
        "maximum_validation_context_relative_rmse": (
            float(validation_metrics["context"]["relative_rmse"])
            <= policy_value.maximum_validation_context_relative_rmse
        ),
        "minimum_validation_context_cosine_similarity": (
            float(validation_metrics["context"]["cosine_similarity"])
            >= policy_value.minimum_validation_context_cosine_similarity
        ),
        "maximum_fit_endpoint_relative_rmse": (
            float(fit_metrics["residual_endpoint"]["relative_rmse"])
            <= policy_value.maximum_fit_endpoint_relative_rmse
        ),
        "maximum_validation_endpoint_relative_rmse": (
            float(validation_metrics["residual_endpoint"]["relative_rmse"])
            <= policy_value.maximum_validation_endpoint_relative_rmse
        ),
        "minimum_validation_endpoint_cosine_similarity": (
            float(
                validation_metrics["residual_endpoint"]["cosine_similarity"]
            )
            >= policy_value.minimum_validation_endpoint_cosine_similarity
        ),
    }
    gate_passed = all(checks.values())
    provenance_body = {
        "query_groups": [
            group.as_dict() for group in query_group_values
        ],
        "key_value_groups": [
            group.as_dict() for group in key_group_values
        ],
        "source_query_head_assignment_count": sum(
            len(group.source_indices) for group in query_group_values
        ),
        "source_key_value_head_assignment_count": sum(
            len(group.source_indices) for group in key_group_values
        ),
        "source_query_heads_dropped": 0,
        "source_key_value_heads_dropped": 0,
        "source_value_heads_pruned": 0,
        "full_source_query_coverage": True,
        "full_source_key_value_coverage": True,
        "all_barycentric_members_positive": True,
        "fit_only_value_bases": [dict(report) for report in basis_reports],
        "validation_evidence_used_for_basis_or_fit": False,
        "fit_case_authentication_sha256": [
            case.authentication_sha256 for case in fit_cases
        ],
        "validation_case_authentication_sha256": [
            case.authentication_sha256 for case in validation_cases
        ],
    }
    body = {
        "schema": LFM2_CPCF_ATTENTION_VALUE_DIMENSION_FOLD_SCHEMA,
        "method": (
            "fit_only_positive_barycentric_value_pca_then_dense_v_ridge_"
            "actual_fixed_qk_context_then_dense_o_endpoint_ridge"
        ),
        "accepted": gate_passed,
        "topology": topology.as_dict(),
        "target_hidden_width": target_hidden,
        "head_dimension_compression_ratio": (
            topology.source_head_dim / topology.target_head_dim
        ),
        "fixed_query_key_sha256": fixed_qk_sha,
        "fit_case_count": len(fit_cases),
        "validation_case_count": len(validation_cases),
        "fit_value_rows": fit_value_rows,
        "fit_endpoint_rows": fit_endpoint_rows,
        "minimum_fit_value_basis_energy_fraction": minimum_energy,
        "fit_metrics": dict(fit_metrics),
        "validation_metrics": dict(validation_metrics),
        "fit_case_reports": [
            dict(report) for report in fit_case_reports
        ],
        "validation_case_reports": [
            dict(report) for report in validation_case_reports
        ],
        "behavior_gate": {
            "policy": policy_value.as_dict(),
            "checks": checks,
            "passed": gate_passed,
            "heldout_validation_used_for_fitting": False,
        },
        "provenance": {
            **provenance_body,
            "provenance_sha256": _sha(provenance_body),
        },
        "ridge": ridge_value,
        "limits": limits_value.as_dict(),
        "compiled_state_sha256": compiled_state_sha,
        "target_logical_parameters": sum(
            int(value.numel()) for value in compiled_state.values()
        ),
        "audit": {
            "evidence_sha256_before": evidence_before,
            "evidence_sha256_after": evidence_after,
            "evidence_unchanged": evidence_unchanged,
            "target_templates_sha256_before": templates_before,
            "target_templates_sha256_after": templates_after,
            "target_templates_unchanged": templates_unchanged,
            "target_template_values_used": False,
            "target_templates_used_for_shape_and_dtype_only": True,
            "case_authentication_verified": True,
            "parent_probability_value_context_chain_authenticated": True,
            "returned_tensors_detached": all(
                not value.requires_grad for value in compiled_state.values()
            ),
        },
        "execution": {
            "fit_only_symmetric_eigendecomposition": True,
            "positive_ridge_normal_equation_solves": 2,
            "fallback_solves": 0,
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    report = MappingProxyType(
        {**body, "report_sha256": _sha(body)}
    )
    if policy_value.enforce and not gate_passed:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise Lfm2CpcfAttentionValueDimensionValidationError(
            "Compiled CPCF attention V/O failed enforced behavior gates: "
            + ", ".join(failed),
            report,
        )
    return Lfm2CompiledCpcfAttentionValueDimensionFold(
        topology=topology,
        query_groups=query_group_values,
        key_value_groups=key_group_values,
        value_feature_bases=bases,
        value_weight=compiled_value,
        output_weight=compiled_output,
        target_hidden_width=target_hidden,
        fixed_query_key_sha256=fixed_qk_sha,
        fit_metrics=fit_metrics,
        validation_metrics=validation_metrics,
        report=report,
    )


def _resolve_attention_parameters(target_attention: Any) -> dict[str, Any]:
    torch = _torch()
    value = getattr(getattr(target_attention, "v_proj", None), "weight", None)
    output_module = getattr(target_attention, "out_proj", None)
    if output_module is None:
        output_module = getattr(target_attention, "o_proj", None)
    output = getattr(output_module, "weight", None)
    if not isinstance(value, torch.nn.Parameter) or not isinstance(
        output,
        torch.nn.Parameter,
    ):
        raise Lfm2CpcfAttentionValueDimensionInstallError(
            "Target attention must expose v_proj.weight and out_proj/o_proj.weight "
            "Parameters."
        )
    return {
        "v_proj.weight": value,
        "out_proj.weight": output,
    }


def _copy_parameter(parameter: Any, value: Any) -> None:
    parameter.copy_(
        value.to(device=parameter.device, dtype=parameter.dtype)
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionValueDimensionInstallReport:
    phase: str
    parameter_identities_preserved: bool
    parameter_shapes_preserved: bool
    parameter_dtypes_preserved: bool
    gradient_objects_preserved: bool
    parameter_values_verified: bool
    expected_state_sha256: str
    observed_state_sha256: str
    target_logical_parameters: int

    @property
    def accepted(self) -> bool:
        return (
            self.parameter_identities_preserved
            and self.parameter_shapes_preserved
            and self.parameter_dtypes_preserved
            and self.gradient_objects_preserved
            and self.parameter_values_verified
            and self.expected_state_sha256 == self.observed_state_sha256
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        } | {
            "accepted": self.accepted,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


class Lfm2CpcfAttentionValueDimensionInstallTransaction:
    """Atomic in-place V/O install retaining an exact pre-install snapshot."""

    __slots__ = (
        "_active",
        "_committed",
        "_compiled_state",
        "_gradient_objects",
        "_gradient_snapshots",
        "_original_dtypes",
        "_original_ids",
        "_original_requires_grad",
        "_original_shapes",
        "_original_state",
        "_original_state_sha256",
        "_parameters",
        "_rolled_back",
        "_target",
        "install_report",
    )

    def __init__(
        self,
        target_attention: Any,
        compiled: Lfm2CompiledCpcfAttentionValueDimensionFold,
    ) -> None:
        if not isinstance(
            compiled,
            Lfm2CompiledCpcfAttentionValueDimensionFold,
        ):
            raise TypeError(
                "compiled must be Lfm2CompiledCpcfAttentionValueDimensionFold."
            )
        self._target = target_attention
        self._parameters = _resolve_attention_parameters(target_attention)
        self._compiled_state = dict(compiled.state_dict())
        for name, parameter in self._parameters.items():
            expected = self._compiled_state[name]
            if (
                tuple(parameter.shape) != tuple(expected.shape)
                or parameter.dtype != expected.dtype
            ):
                raise Lfm2CpcfAttentionValueDimensionInstallError(
                    f"Target {name} differs from compiled shape or dtype."
                )
        self._original_ids = {
            name: id(parameter)
            for name, parameter in self._parameters.items()
        }
        self._original_shapes = {
            name: tuple(parameter.shape)
            for name, parameter in self._parameters.items()
        }
        self._original_dtypes = {
            name: parameter.dtype
            for name, parameter in self._parameters.items()
        }
        self._original_requires_grad = {
            name: bool(parameter.requires_grad)
            for name, parameter in self._parameters.items()
        }
        self._gradient_objects = {
            name: parameter.grad
            for name, parameter in self._parameters.items()
        }
        self._gradient_snapshots = {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().contiguous().clone()
            )
            for name, parameter in self._parameters.items()
        }
        self._original_state = {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in self._parameters.items()
        }
        self._original_state_sha256 = _tensor_bundle_sha256(
            self._original_state
        )
        self._active = True
        self._committed = False
        self._rolled_back = False
        self.install_report = None
        try:
            with _torch().no_grad():
                for name, parameter in self._parameters.items():
                    _copy_parameter(parameter, self._compiled_state[name])
            self.install_report = self._report(
                phase="installed",
                expected=self._compiled_state,
            )
            if not self.install_report.accepted:
                raise Lfm2CpcfAttentionValueDimensionInstallError(
                    "Installed V/O state failed identity, gradient, or content audit."
                )
        except Exception as exc:
            try:
                self._restore_original()
            except Exception as rollback_exc:
                self._active = False
                raise Lfm2CpcfAttentionValueDimensionInstallError(
                    "V/O installation failed and exact rollback also failed."
                ) from rollback_exc
            self._active = False
            self._rolled_back = True
            if isinstance(
                exc,
                Lfm2CpcfAttentionValueDimensionInstallError,
            ):
                raise
            raise Lfm2CpcfAttentionValueDimensionInstallError(
                "V/O installation failed and was rolled back exactly."
            ) from exc

    @property
    def active(self) -> bool:
        return self._active

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def rolled_back(self) -> bool:
        return self._rolled_back

    @property
    def original_state_sha256(self) -> str:
        return self._original_state_sha256

    def _static_state(self) -> tuple[bool, bool, bool, bool]:
        identities = True
        shapes = True
        dtypes = True
        gradients = True
        current = _resolve_attention_parameters(self._target)
        for name, original in self._parameters.items():
            parameter = current[name]
            identities = identities and (
                parameter is original
                and id(parameter) == self._original_ids[name]
            )
            shapes = shapes and (
                tuple(parameter.shape) == self._original_shapes[name]
            )
            dtypes = dtypes and (
                parameter.dtype == self._original_dtypes[name]
            )
            gradient = self._gradient_objects[name]
            gradients = gradients and (
                bool(parameter.requires_grad)
                == self._original_requires_grad[name]
                and parameter.grad is gradient
            )
            if gradient is not None:
                gradients = gradients and (
                    _tensor_sha256(
                        gradient,
                        f"{name}.gradient",
                    )
                    == _tensor_sha256(
                        self._gradient_snapshots[name],
                        f"{name}.gradient_snapshot",
                    )
                )
        return identities, shapes, dtypes, gradients

    def _report(
        self,
        *,
        phase: str,
        expected: Mapping[str, Any],
    ) -> Lfm2CpcfAttentionValueDimensionInstallReport:
        observed = {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in self._parameters.items()
        }
        expected_sha = _tensor_bundle_sha256(expected)
        observed_sha = _tensor_bundle_sha256(observed)
        static = self._static_state()
        return Lfm2CpcfAttentionValueDimensionInstallReport(
            phase=phase,
            parameter_identities_preserved=static[0],
            parameter_shapes_preserved=static[1],
            parameter_dtypes_preserved=static[2],
            gradient_objects_preserved=static[3],
            parameter_values_verified=expected_sha == observed_sha,
            expected_state_sha256=expected_sha,
            observed_state_sha256=observed_sha,
            target_logical_parameters=sum(
                int(parameter.numel())
                for parameter in self._parameters.values()
            ),
        )

    def _restore_original(self) -> None:
        current = _resolve_attention_parameters(self._target)
        if any(
            current[name] is not parameter
            for name, parameter in self._parameters.items()
        ):
            raise Lfm2CpcfAttentionValueDimensionInstallError(
                "Cannot restore a target whose Parameter object was replaced."
            )
        with _torch().no_grad():
            for name, parameter in self._parameters.items():
                _copy_parameter(parameter, self._original_state[name])
                parameter.requires_grad_(
                    self._original_requires_grad[name]
                )
                gradient = self._gradient_objects[name]
                if gradient is None:
                    parameter.grad = None
                else:
                    _copy_parameter(
                        gradient,
                        self._gradient_snapshots[name],
                    )
                    parameter.grad = gradient

    def commit(self) -> Lfm2CpcfAttentionValueDimensionInstallReport:
        if not self._active:
            raise Lfm2CpcfAttentionValueDimensionInstallError(
                "Only an active V/O transaction can be committed."
            )
        report = self._report(
            phase="committed",
            expected=self._compiled_state,
        )
        if not report.accepted:
            self.rollback()
            raise Lfm2CpcfAttentionValueDimensionInstallError(
                "V/O commit audit failed; the target was rolled back."
            )
        self._active = False
        self._committed = True
        return report

    def rollback(self) -> Lfm2CpcfAttentionValueDimensionInstallReport:
        if not self._active:
            if self._rolled_back:
                return self._report(
                    phase="rolled-back",
                    expected=self._original_state,
                )
            raise Lfm2CpcfAttentionValueDimensionInstallError(
                "A committed V/O transaction cannot be rolled back."
            )
        self._restore_original()
        report = self._report(
            phase="rolled-back",
            expected=self._original_state,
        )
        if not report.accepted:
            raise Lfm2CpcfAttentionValueDimensionInstallError(
                "V/O rollback was not exact."
            )
        self._active = False
        self._rolled_back = True
        return report

    def __enter__(
        self,
    ) -> "Lfm2CpcfAttentionValueDimensionInstallTransaction":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._active:
            self.rollback()
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "committed": self._committed,
            "rolled_back": self._rolled_back,
            "original_state_sha256": self._original_state_sha256,
            "install_report": (
                None
                if self.install_report is None
                else self.install_report.as_dict()
            ),
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


def install_lfm2_cpcf_attention_value_dimension_fold(
    target_attention: Any,
    compiled: Lfm2CompiledCpcfAttentionValueDimensionFold,
) -> Lfm2CpcfAttentionValueDimensionInstallTransaction:
    """Atomically install dense V/O while retaining exact rollback state."""

    return Lfm2CpcfAttentionValueDimensionInstallTransaction(
        target_attention,
        compiled,
    )


__all__ = [
    "LFM2_CPCF_ATTENTION_VALUE_DIMENSION_FOLD_SCHEMA",
    "Lfm2CompiledCpcfAttentionValueDimensionFold",
    "Lfm2CpcfAttentionValueDimensionCase",
    "Lfm2CpcfAttentionValueDimensionFoldError",
    "Lfm2CpcfAttentionValueDimensionInstallError",
    "Lfm2CpcfAttentionValueDimensionInstallReport",
    "Lfm2CpcfAttentionValueDimensionInstallTransaction",
    "Lfm2CpcfAttentionValueDimensionLimits",
    "Lfm2CpcfAttentionValueDimensionPolicy",
    "Lfm2CpcfAttentionValueDimensionValidationError",
    "Lfm2CpcfAttentionValueOutputReplay",
    "compile_lfm2_cpcf_attention_value_dimension_fold",
    "install_lfm2_cpcf_attention_value_dimension_fold",
]
