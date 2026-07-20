"""Cached FIT-evidence adapter for production LFM2 Q/K folding.

All folding math lives in the public
``compile_lfm2_cpcf_attention_dimension_fold`` primitive.  This module only
seals a deterministic FIT/FIT-CHECK partition, adapts cached structural rows
to the primitive's public production case, and publishes the exact target
dtype dense Q/K state.  Macrocell selection and true validation are external.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (
    Lfm2CompiledCpcfAttentionDimensionFold,
    Lfm2CpcfAttentionCaseProvenance,
    Lfm2CpcfAttentionDimensionLimits,
    Lfm2CpcfAttentionDimensionPolicy,
    Lfm2CpcfAttentionDimensionProductionCase,
    Lfm2CpcfAttentionDimensionRuntime,
    Lfm2CpcfAttentionDimensionTopology,
    Lfm2CpcfAttentionRoutingMetrics,
    Lfm2CpcfBarycentricHeadGroup,
    build_lfm2_cpcf_attention_dimension_production_case,
    compile_lfm2_cpcf_attention_dimension_fold,
    lfm2_runtime_rms_norm,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedAttentionFoldCase,
)


CACHED_QK_FUNCTIONAL_REFIT_SCHEMA = "neural-forge-lfm2-cached-qk-functional-refit-v2"
CachedQkEvidencePartition = Literal["fit", "fit_check"]

_CACHED_QK_EVIDENCE_FIELDS = (
    "target_query_input",
    "target_key_value_input",
    "parent_query_linear",
    "parent_key_linear",
    "parent_query",
    "parent_key",
    "query_cosine",
    "query_sine",
    "key_cosine",
    "key_sine",
    "key_mask",
    "query_row_sqrt_weights",
    "key_value_row_sqrt_weights",
)


class Lfm2CachedQkFunctionalRefitError(ValueError):
    """Cached Q/K evidence, topology, or FIT partition is invalid."""


class Lfm2CachedQkArtifactAuthenticationError(Lfm2CachedQkFunctionalRefitError):
    """A cached evidence or dense artifact seal failed authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for cached Q/K functional refitting."
        ) from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            "Q/K evidence hashing requires a dense non-meta tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _required_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedQkArtifactAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(character not in allowed for character in value)
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            f"{name} must be a nonempty portable identifier."
        )
    return value


def _exact_float_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.dtype
        not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            f"{name} must be a finite dense FP rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CachedQkFunctionalRefitError(f"{name} contains NaN or infinity.")
    return result


def _cached_case_sha256(case: Lfm2CachedAttentionFoldCase) -> str:
    if not isinstance(case, Lfm2CachedAttentionFoldCase):
        raise TypeError("Cached Q/K banks require Lfm2CachedAttentionFoldCase values.")
    digest = hashlib.sha256()
    for name in _CACHED_QK_EVIDENCE_FIELDS:
        value = getattr(case, name)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            b"none" if value is None else bytes.fromhex(_tensor_sha256(value))
        )
    if case.parent_probabilities is not None:
        digest.update(b"parent_probabilities")
        digest.update(b"\0")
        digest.update(bytes.fromhex(_tensor_sha256(case.parent_probabilities)))
    return digest.hexdigest()


def _bank_sha256(
    partition: CachedQkEvidencePartition,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    trajectory_sha256s: Sequence[str],
    source_case_ordinals: Sequence[int],
    source_split_sha256: str,
    source_runtime_sha256: str,
) -> str:
    body = {
        "partition": partition,
        "upstream_split": "fit",
        "source_split_sha256": source_split_sha256,
        "source_runtime_sha256": source_runtime_sha256,
        "rows": [
            {
                "source_case_ordinal": ordinal,
                "trajectory_sha256": trajectory,
                "case_sha256": _cached_case_sha256(case),
            }
            for case, trajectory, ordinal in zip(
                cases,
                trajectory_sha256s,
                source_case_ordinals,
                strict=True,
            )
        ],
    }
    return _canonical_sha256(body)


def _cached_resource_usage(
    banks: Sequence["Lfm2CachedQkEvidenceBank"],
) -> dict[str, int]:
    torch = _torch()
    query_rows = 0
    key_rows = 0
    evidence_bytes = 0
    for bank in banks:
        for case in bank.cases:
            query_rows += int(case.target_query_input.shape[0]) * int(
                case.target_query_input.shape[1]
            )
            key_rows += int(case.target_key_value_input.shape[0]) * int(
                case.target_key_value_input.shape[1]
            )
            for name in _CACHED_QK_EVIDENCE_FIELDS:
                value = getattr(case, name)
                if isinstance(value, torch.Tensor):
                    evidence_bytes += int(value.numel()) * int(value.element_size())
            if isinstance(case.parent_probabilities, torch.Tensor):
                evidence_bytes += int(case.parent_probabilities.numel()) * int(
                    case.parent_probabilities.element_size()
                )
    return {
        "aggregate_query_rows": query_rows,
        "aggregate_key_rows": key_rows,
        "aggregate_evidence_bytes": evidence_bytes,
    }


@dataclass(frozen=True, slots=True)
class Lfm2CachedQkEvidenceBank:
    """One sealed local partition of the literal upstream FIT split."""

    partition: CachedQkEvidencePartition
    cases: tuple[Lfm2CachedAttentionFoldCase, ...]
    trajectory_sha256s: tuple[str, ...]
    source_case_ordinals: tuple[int, ...]
    source_split_sha256: str
    source_runtime_sha256: str
    upstream_split: Literal["fit"] = "fit"
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if self.partition not in {"fit", "fit_check"}:
            raise Lfm2CachedQkFunctionalRefitError(
                "Cached Q/K partition must be exactly 'fit' or 'fit_check'."
            )
        if self.upstream_split != "fit":
            raise Lfm2CachedQkFunctionalRefitError(
                "Cached Q/K evidence must retain literal upstream split 'fit'."
            )
        cases = tuple(self.cases)
        trajectories = tuple(
            _required_digest(value, "trajectory_sha256")
            for value in self.trajectory_sha256s
        )
        ordinals = tuple(self.source_case_ordinals)
        source_split = _required_digest(
            self.source_split_sha256,
            "source_split_sha256",
        )
        source_runtime = _required_digest(
            self.source_runtime_sha256,
            "source_runtime_sha256",
        )
        if (
            not cases
            or not all(
                isinstance(value, Lfm2CachedAttentionFoldCase) for value in cases
            )
            or len(trajectories) != len(cases)
            or len(ordinals) != len(cases)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in ordinals
            )
            or len(set(ordinals)) != len(ordinals)
        ):
            raise Lfm2CachedQkFunctionalRefitError(
                "A cached Q/K bank requires aligned nonempty cases, trajectory "
                "receipts, and unique nonnegative source ordinals."
            )
        computed = _bank_sha256(
            self.partition,
            cases,
            trajectories,
            ordinals,
            source_split,
            source_runtime,
        )
        if self.evidence_sha256 and self.evidence_sha256 != computed:
            raise Lfm2CachedQkArtifactAuthenticationError(
                "Cached Q/K evidence does not match its supplied seal."
            )
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "trajectory_sha256s", trajectories)
        object.__setattr__(self, "source_case_ordinals", ordinals)
        object.__setattr__(self, "source_split_sha256", source_split)
        object.__setattr__(self, "source_runtime_sha256", source_runtime)
        object.__setattr__(self, "evidence_sha256", computed)

    def verify(self) -> str:
        computed = _bank_sha256(
            self.partition,
            self.cases,
            self.trajectory_sha256s,
            self.source_case_ordinals,
            self.source_split_sha256,
            self.source_runtime_sha256,
        )
        if computed != self.evidence_sha256:
            raise Lfm2CachedQkArtifactAuthenticationError(
                "Cached Q/K evidence changed after sealing."
            )
        return computed


def seal_lfm2_cached_qk_fit_bank(
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    *,
    trajectory_sha256s: Sequence[str],
    source_split_sha256: str,
    source_runtime_sha256: str,
    source_case_ordinals: Sequence[int] | None = None,
) -> Lfm2CachedQkEvidenceBank:
    values = tuple(cases)
    return Lfm2CachedQkEvidenceBank(
        partition="fit",
        cases=values,
        trajectory_sha256s=tuple(trajectory_sha256s),
        source_case_ordinals=(
            tuple(range(len(values)))
            if source_case_ordinals is None
            else tuple(source_case_ordinals)
        ),
        source_split_sha256=source_split_sha256,
        source_runtime_sha256=source_runtime_sha256,
    )


def seal_lfm2_cached_qk_fit_check_bank(
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    *,
    trajectory_sha256s: Sequence[str],
    source_split_sha256: str,
    source_runtime_sha256: str,
    source_case_ordinals: Sequence[int] | None = None,
) -> Lfm2CachedQkEvidenceBank:
    values = tuple(cases)
    return Lfm2CachedQkEvidenceBank(
        partition="fit_check",
        cases=values,
        trajectory_sha256s=tuple(trajectory_sha256s),
        source_case_ordinals=(
            tuple(range(len(values)))
            if source_case_ordinals is None
            else tuple(source_case_ordinals)
        ),
        source_split_sha256=source_split_sha256,
        source_runtime_sha256=source_runtime_sha256,
    )


def partition_lfm2_cached_qk_fit_cases(
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    *,
    trajectory_sha256s: Sequence[str],
    source_split_sha256: str,
    source_runtime_sha256: str,
    fit_check_count: int | None = None,
) -> tuple[Lfm2CachedQkEvidenceBank, Lfm2CachedQkEvidenceBank]:
    """Deterministically split upstream FIT trajectories into FIT/FIT-CHECK."""

    values = tuple(cases)
    trajectories = tuple(trajectory_sha256s)
    if len(values) < 2 or len(trajectories) != len(values):
        raise Lfm2CachedQkFunctionalRefitError(
            "Deterministic Q/K partitioning requires at least two aligned "
            "upstream FIT cases and trajectory receipts."
        )
    for value in trajectories:
        _required_digest(value, "trajectory_sha256")
    if len(set(trajectories)) != len(trajectories):
        raise Lfm2CachedQkFunctionalRefitError(
            "Deterministic Q/K partitioning requires one packed case per "
            "unique source trajectory."
        )
    if fit_check_count is None:
        count = max(1, round(len(values) * 0.13))
    elif (
        isinstance(fit_check_count, bool)
        or not isinstance(fit_check_count, int)
        or fit_check_count <= 0
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            "fit_check_count must be a positive integer."
        )
    else:
        count = fit_check_count
    if count >= len(values):
        raise Lfm2CachedQkFunctionalRefitError(
            "FIT-CHECK must leave at least one case for Q/K fitting."
        )
    ranked = sorted(
        range(len(values)),
        key=lambda index: (
            _canonical_sha256(
                {
                    "trajectory_sha256": trajectories[index],
                    "case_sha256": _cached_case_sha256(values[index]),
                    "source_case_ordinal": index,
                    "partition_seed": "lfm2-qk-fit-check-v1",
                }
            ),
            index,
        ),
    )
    fit_check_ordinals = set(ranked[:count])
    fit_indices = tuple(
        index for index in range(len(values)) if index not in fit_check_ordinals
    )
    fit_check_indices = tuple(
        index for index in range(len(values)) if index in fit_check_ordinals
    )
    return (
        seal_lfm2_cached_qk_fit_bank(
            tuple(values[index] for index in fit_indices),
            trajectory_sha256s=tuple(trajectories[index] for index in fit_indices),
            source_split_sha256=source_split_sha256,
            source_runtime_sha256=source_runtime_sha256,
            source_case_ordinals=fit_indices,
        ),
        seal_lfm2_cached_qk_fit_check_bank(
            tuple(values[index] for index in fit_check_indices),
            trajectory_sha256s=tuple(
                trajectories[index] for index in fit_check_indices
            ),
            source_split_sha256=source_split_sha256,
            source_runtime_sha256=source_runtime_sha256,
            source_case_ordinals=fit_check_indices,
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CachedQkFunctionalRefitPolicy:
    """FIT/FIT-CHECK vocabulary facade over the legacy policy field names."""

    minimum_score_spectral_energy_fraction: float = 0.99
    maximum_fit_centered_score_relative_rmse: float | None = 0.10
    maximum_fit_check_centered_score_relative_rmse: float | None = 0.20
    maximum_fit_check_js_divergence: float | None = 0.05
    maximum_fit_check_route_probability_relative_rmse: float | None = 0.20
    minimum_fit_check_top1_route_agreement: float | None = 0.90
    authentication_rtol: float = 2e-5
    authentication_atol: float = 2e-6
    rank_relative_tolerance: float = 1e-10
    anchor_exact_relative_rmse: float = 1e-8

    def to_primitive(self) -> Lfm2CpcfAttentionDimensionPolicy:
        return Lfm2CpcfAttentionDimensionPolicy(
            minimum_score_spectral_energy_fraction=(
                self.minimum_score_spectral_energy_fraction
            ),
            maximum_fit_centered_score_relative_rmse=(
                self.maximum_fit_centered_score_relative_rmse
            ),
            maximum_validation_centered_score_relative_rmse=(
                self.maximum_fit_check_centered_score_relative_rmse
            ),
            maximum_validation_js_divergence=(self.maximum_fit_check_js_divergence),
            maximum_validation_route_probability_relative_rmse=(
                self.maximum_fit_check_route_probability_relative_rmse
            ),
            minimum_validation_top1_route_agreement=(
                self.minimum_fit_check_top1_route_agreement
            ),
            authentication_rtol=self.authentication_rtol,
            authentication_atol=self.authentication_atol,
            rank_relative_tolerance=self.rank_relative_tolerance,
            anchor_exact_relative_rmse=self.anchor_exact_relative_rmse,
        )

    def as_dict(self) -> dict[str, Any]:
        # Primitive construction performs all finite/range validation.
        self.to_primitive()
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def adapt_lfm2_cached_attention_case_for_qk_refit(
    case: Lfm2CachedAttentionFoldCase,
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    runtime: Lfm2CpcfAttentionDimensionRuntime,
    provenance: Lfm2CpcfAttentionCaseProvenance,
    case_name: str = "cached_case",
) -> Lfm2CpcfAttentionDimensionProductionCase:
    """Adapt one cached route through only public primitive APIs."""

    torch = _torch()
    if not isinstance(case, Lfm2CachedAttentionFoldCase):
        raise TypeError("case must be Lfm2CachedAttentionFoldCase.")
    if not isinstance(topology, Lfm2CpcfAttentionDimensionTopology):
        raise TypeError("topology must be Lfm2CpcfAttentionDimensionTopology.")
    if not isinstance(runtime, Lfm2CpcfAttentionDimensionRuntime):
        raise TypeError("runtime must be Lfm2CpcfAttentionDimensionRuntime.")
    query_linear = _exact_float_tensor(
        case.parent_query_linear,
        f"{case_name}.parent_query_linear",
        ndim=4,
    )
    key_linear = _exact_float_tensor(
        case.parent_key_linear,
        f"{case_name}.parent_key_linear",
        ndim=4,
    )
    parent_query = _exact_float_tensor(
        case.parent_query,
        f"{case_name}.parent_query",
        ndim=4,
    )
    parent_key = _exact_float_tensor(
        case.parent_key,
        f"{case_name}.parent_key",
        ndim=4,
    )
    batch, _heads, query_length, head_dim = map(int, parent_query.shape)
    key_length = int(parent_key.shape[2])
    rotary = (
        case.query_cosine,
        case.query_sine,
        case.key_cosine,
        case.key_sine,
    )
    if any(value is not None for value in rotary) and not all(
        value is not None for value in rotary
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            f"{case_name} must supply all four cached RoPE tensors together."
        )
    if all(value is None for value in rotary):
        normalized_query = lfm2_runtime_rms_norm(
            query_linear,
            runtime.source_query_norm_weight,
            runtime.norm_epsilon,
        )
        normalized_key = lfm2_runtime_rms_norm(
            key_linear,
            runtime.source_key_norm_weight,
            runtime.norm_epsilon,
        )
        if not torch.equal(normalized_query, parent_query) or not torch.equal(
            normalized_key,
            parent_key,
        ):
            raise Lfm2CachedQkFunctionalRefitError(
                f"{case_name} lacks RoPE evidence but is not an exact identity "
                "rotary route."
            )
        query_cos = torch.ones(
            (batch, query_length, head_dim),
            dtype=query_linear.dtype,
        )
        query_sin = torch.zeros_like(query_cos)
        key_cos = torch.ones(
            (batch, key_length, head_dim),
            dtype=key_linear.dtype,
        )
        key_sin = torch.zeros_like(key_cos)
    else:
        query_cos = case.query_cosine
        query_sin = case.query_sine
        key_cos = case.key_cosine
        key_sin = case.key_sine
    if case.key_mask is None:
        raise Lfm2CachedQkFunctionalRefitError(
            f"{case_name}.key_mask is required for authenticated causal replay."
        )
    return build_lfm2_cpcf_attention_dimension_production_case(
        topology=topology,
        runtime=runtime,
        target_query_residual_inputs=case.target_query_input,
        target_key_value_residual_inputs=case.target_key_value_input,
        parent_query_linear=query_linear,
        parent_key_linear=key_linear,
        parent_query_post_rope=parent_query,
        parent_key_post_rope=parent_key,
        query_rope_cos=query_cos,
        query_rope_sin=query_sin,
        key_rope_cos=key_cos,
        key_rope_sin=key_sin,
        causal_mask=case.key_mask,
        provenance=provenance,
        query_row_sqrt_weights=case.query_row_sqrt_weights,
        key_value_row_sqrt_weights=case.key_value_row_sqrt_weights,
    )


def _primitive_report_sha256(
    compiled: Lfm2CompiledCpcfAttentionDimensionFold,
) -> str:
    """Authenticate the real, non-skipped primitive report path once."""

    report = dict(compiled.report)
    sealed_value = report.pop("report_sha256", None)
    sealed = _required_digest(sealed_value, "report_sha256")
    if _canonical_sha256(report) != sealed:
        raise Lfm2CachedQkArtifactAuthenticationError(
            "The analytical primitive returned an invalid report seal."
        )
    return sealed


@dataclass(frozen=True, slots=True)
class Lfm2CompiledCachedQkFunctionalRefit:
    """Sealed exact-dtype dense Q/K state and FIT/FIT-CHECK metrics."""

    candidate_id: str
    topology: Lfm2CpcfAttentionDimensionTopology
    runtime: Lfm2CpcfAttentionDimensionRuntime
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...]
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...]
    selected_source_rope_frequencies: tuple[int, ...]
    query_weight: Any
    key_weight: Any
    fit_metrics: Lfm2CpcfAttentionRoutingMetrics
    fit_check_metrics: Lfm2CpcfAttentionRoutingMetrics
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        query = _exact_float_tensor(self.query_weight, "query_weight", ndim=2)
        key = _exact_float_tensor(self.key_weight, "key_weight", ndim=2)
        if (
            tuple(query.shape) != tuple(self.runtime.target_query_weight_template.shape)
            or tuple(key.shape) != tuple(self.runtime.target_key_weight_template.shape)
            or query.dtype != self.runtime.target_dtype
            or key.dtype != self.runtime.target_dtype
        ):
            raise Lfm2CachedQkFunctionalRefitError(
                "Published dense Q/K state differs from exact target templates."
            )
        report = json.loads(
            json.dumps(
                dict(self.report),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(self, "query_weight", query)
        object.__setattr__(self, "key_weight", key)
        object.__setattr__(self, "query_groups", tuple(self.query_groups))
        object.__setattr__(
            self,
            "key_value_groups",
            tuple(self.key_value_groups),
        )
        object.__setattr__(
            self,
            "selected_source_rope_frequencies",
            tuple(self.selected_source_rope_frequencies),
        )
        object.__setattr__(self, "report", MappingProxyType(report))
        self.verify()

    @property
    def artifact_sha256(self) -> str:
        return str(self.report["artifact_sha256"])

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.report)))

    def verify(self, expected_artifact_sha256: str | None = None) -> str:
        report = self.as_dict()
        sealed = _required_digest(
            report.pop("artifact_sha256", None),
            "artifact_sha256",
        )
        if _canonical_sha256(report) != sealed:
            raise Lfm2CachedQkArtifactAuthenticationError(
                "Cached Q/K artifact seal does not match its report."
            )
        if expected_artifact_sha256 is not None and sealed != _required_digest(
            expected_artifact_sha256,
            "expected_artifact_sha256",
        ):
            raise Lfm2CachedQkArtifactAuthenticationError(
                "Cached Q/K artifact differs from the externally expected seal."
            )
        bindings = {
            "candidate_id": self.candidate_id,
            "topology": self.topology.as_dict(),
            "runtime": self.runtime.as_dict(),
            "source_runtime_sha256": self.runtime.source_runtime_sha256,
            "query_groups": [value.as_dict() for value in self.query_groups],
            "key_value_groups": [value.as_dict() for value in self.key_value_groups],
            "selected_source_rope_frequencies": list(
                self.selected_source_rope_frequencies
            ),
            "query_weight_sha256": _tensor_sha256(self.query_weight),
            "key_weight_sha256": _tensor_sha256(self.key_weight),
            "fit_routing": self.fit_metrics.as_dict(),
            "fit_check_routing": self.fit_check_metrics.as_dict(),
        }
        if report.get("schema") != CACHED_QK_FUNCTIONAL_REFIT_SCHEMA or any(
            report.get(name) != value for name, value in bindings.items()
        ):
            raise Lfm2CachedQkArtifactAuthenticationError(
                "Cached Q/K report does not bind its typed dense state."
            )
        operation = report.get("operation_audit")
        zero_fields = (
            "model_forward_calls",
            "model_generate_calls",
            "backward_calls",
            "optimizer_steps",
            "training_steps",
            "pruned_parameters",
            "quantized_parameters",
            "target_parameter_mutations_during_fit_check",
            "selection_cases_consumed",
            "true_validation_cases_consumed",
            "trace_recaptures",
            "extra_published_parameters",
        )
        if (
            not isinstance(operation, dict)
            or not bool(operation.get("source_evidence_unchanged"))
            or not bool(operation.get("dense_target_projection_state"))
            or not bool(operation.get("published_state_cast_before_fit_check"))
            or any(int(operation.get(name, -1)) != 0 for name in zero_fields)
        ):
            raise Lfm2CachedQkArtifactAuthenticationError(
                "Cached Q/K artifact violates the analytical no-training or "
                "frozen-FIT-CHECK boundary."
            )
        return sealed


def compile_lfm2_cached_qk_functional_refit(
    *,
    candidate_id: str,
    topology: Lfm2CpcfAttentionDimensionTopology,
    runtime: Lfm2CpcfAttentionDimensionRuntime,
    fit_bank: Lfm2CachedQkEvidenceBank,
    fit_check_bank: Lfm2CachedQkEvidenceBank,
    query_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    key_value_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    policy: Lfm2CachedQkFunctionalRefitPolicy | None = None,
    limits: Lfm2CpcfAttentionDimensionLimits | None = None,
    ridge: float = 1e-8,
) -> Lfm2CompiledCachedQkFunctionalRefit:
    """Fold dense Q/K from FIT and gate the frozen exact-dtype state."""

    identifier = _identifier(candidate_id, "candidate_id")
    if not isinstance(topology, Lfm2CpcfAttentionDimensionTopology):
        raise TypeError("topology must be Lfm2CpcfAttentionDimensionTopology.")
    if not isinstance(runtime, Lfm2CpcfAttentionDimensionRuntime):
        raise TypeError("runtime must be Lfm2CpcfAttentionDimensionRuntime.")
    if {
        runtime.source_query_norm_weight.dtype,
        runtime.source_key_norm_weight.dtype,
        runtime.target_query_norm_weight.dtype,
        runtime.target_key_norm_weight.dtype,
    } != {runtime.target_dtype}:
        raise Lfm2CachedQkFunctionalRefitError(
            "Cached production Q/K currently requires one exact projection and "
            "RMSNorm dtype."
        )
    if (
        not isinstance(fit_bank, Lfm2CachedQkEvidenceBank)
        or fit_bank.partition != "fit"
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            "Q/K candidate generation requires a sealed FIT bank."
        )
    if (
        not isinstance(fit_check_bank, Lfm2CachedQkEvidenceBank)
        or fit_check_bank.partition != "fit_check"
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            "Q/K frozen replay requires a sealed FIT-CHECK bank."
        )
    if fit_bank.source_split_sha256 != fit_check_bank.source_split_sha256:
        raise Lfm2CachedQkFunctionalRefitError(
            "FIT and FIT-CHECK must derive from one upstream FIT split receipt."
        )
    if (
        fit_bank.source_runtime_sha256 != fit_check_bank.source_runtime_sha256
        or fit_bank.source_runtime_sha256 != runtime.source_runtime_sha256
    ):
        raise Lfm2CachedQkArtifactAuthenticationError(
            "FIT/FIT-CHECK evidence does not bind the supplied source RMSNorm "
            "runtime receipt."
        )
    if set(fit_bank.source_case_ordinals) & set(fit_check_bank.source_case_ordinals):
        raise Lfm2CachedQkFunctionalRefitError(
            "FIT and FIT-CHECK source case ordinals overlap."
        )
    if set(fit_bank.trajectory_sha256s) & set(fit_check_bank.trajectory_sha256s):
        raise Lfm2CachedQkFunctionalRefitError(
            "FIT and FIT-CHECK source trajectories overlap, including distinct "
            "phases or repacked subsets."
        )
    fit_sha_before = fit_bank.verify()
    fit_check_sha_before = fit_check_bank.verify()
    cached_resource_usage = _cached_resource_usage((fit_bank, fit_check_bank))
    if (
        cached_resource_usage["aggregate_query_rows"]
        > runtime.maximum_aggregate_query_rows
        or cached_resource_usage["aggregate_key_rows"]
        > runtime.maximum_aggregate_key_rows
        or cached_resource_usage["aggregate_evidence_bytes"]
        > runtime.maximum_aggregate_evidence_bytes
    ):
        raise Lfm2CachedQkFunctionalRefitError(
            "Cached Q/K evidence exceeds an aggregate row or byte ceiling."
        )
    policy_value = policy or Lfm2CachedQkFunctionalRefitPolicy()
    if not isinstance(policy_value, Lfm2CachedQkFunctionalRefitPolicy):
        raise TypeError("policy must be Lfm2CachedQkFunctionalRefitPolicy.")
    primitive_policy = policy_value.to_primitive()
    fit_cases = tuple(
        adapt_lfm2_cached_attention_case_for_qk_refit(
            case,
            topology=topology,
            runtime=runtime,
            provenance=Lfm2CpcfAttentionCaseProvenance(
                partition="fit",
                trajectory_sha256=trajectory,
                source_split_sha256=fit_bank.source_split_sha256,
                source_runtime_sha256=fit_bank.source_runtime_sha256,
                case_ordinal=ordinal,
            ),
            case_name=f"fit_bank.cases[{index}]",
        )
        for index, (case, trajectory, ordinal) in enumerate(
            zip(
                fit_bank.cases,
                fit_bank.trajectory_sha256s,
                fit_bank.source_case_ordinals,
                strict=True,
            )
        )
    )
    fit_check_cases = tuple(
        adapt_lfm2_cached_attention_case_for_qk_refit(
            case,
            topology=topology,
            runtime=runtime,
            provenance=Lfm2CpcfAttentionCaseProvenance(
                partition="fit_check",
                trajectory_sha256=trajectory,
                source_split_sha256=fit_check_bank.source_split_sha256,
                source_runtime_sha256=fit_check_bank.source_runtime_sha256,
                case_ordinal=ordinal,
            ),
            case_name=f"fit_check_bank.cases[{index}]",
        )
        for index, (case, trajectory, ordinal) in enumerate(
            zip(
                fit_check_bank.cases,
                fit_check_bank.trajectory_sha256s,
                fit_check_bank.source_case_ordinals,
                strict=True,
            )
        )
    )
    compiled = compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=fit_cases,
        validation_cases=(),
        fit_check_cases=fit_check_cases,
        query_groups=tuple(query_groups),
        key_value_groups=tuple(key_value_groups),
        policy=primitive_policy,
        limits=limits,
        ridge=ridge,
        runtime=runtime,
    )
    primitive_sha256 = _primitive_report_sha256(compiled)
    compiled.verify_authentication()
    if (
        fit_sha_before != fit_bank.verify()
        or fit_check_sha_before != fit_check_bank.verify()
    ):
        raise Lfm2CachedQkArtifactAuthenticationError(
            "Cached evidence changed during Q/K analytical compilation."
        )
    primitive_report = dict(compiled.report)
    query_groups_value = tuple(compiled.query_groups)
    key_groups_value = tuple(compiled.key_value_groups)
    selected = tuple(compiled.selected_source_rope_frequencies)
    dense_state_sha256 = _canonical_sha256(
        {
            "q_proj.weight": _tensor_sha256(compiled.query_weight),
            "k_proj.weight": _tensor_sha256(compiled.key_weight),
        }
    )
    operation_audit = {
        **dict(primitive_report["operation_audit"]),
        "raw_weight_matching_performed": False,
        "weight_transplant_searches": 0,
        "trace_recaptures": 0,
        "extra_published_parameters": 0,
        "parent_runtime_references": 0,
    }
    body = {
        "schema": CACHED_QK_FUNCTIONAL_REFIT_SCHEMA,
        "candidate_id": identifier,
        "method": "public_cpcf_production_cached_fit_partition_adapter",
        "accepted": True,
        "topology": topology.as_dict(),
        "runtime": runtime.as_dict(),
        "query_groups": [value.as_dict() for value in query_groups_value],
        "key_value_groups": [value.as_dict() for value in key_groups_value],
        "selected_source_rope_frequencies": list(selected),
        "query_weight_sha256": _tensor_sha256(compiled.query_weight),
        "key_weight_sha256": _tensor_sha256(compiled.key_weight),
        "query_weight_shape": list(compiled.query_weight.shape),
        "key_weight_shape": list(compiled.key_weight.shape),
        "query_weight_dtype": str(compiled.query_weight.dtype),
        "key_weight_dtype": str(compiled.key_weight.dtype),
        "dense_qk_state_sha256": dense_state_sha256,
        "ordinary_dense_target_qk_state": True,
        "published_extra_parameter_count": 0,
        "caller_head_assignment_retained": True,
        "full_source_assignment_fraction": primitive_report[
            "full_source_assignment_fraction"
        ],
        "fit_routing": compiled.fit_metrics.as_dict(),
        "fit_check_routing": compiled.fit_check_metrics.as_dict(),
        "fit_bank_sha256": fit_sha_before,
        "fit_check_bank_sha256": fit_check_sha_before,
        "upstream_fit_split_sha256": fit_bank.source_split_sha256,
        "source_runtime_sha256": fit_bank.source_runtime_sha256,
        "fit_case_count": len(fit_cases),
        "fit_check_case_count": len(fit_check_cases),
        "cached_resource_usage": cached_resource_usage,
        "macrocell_selection_case_count": 0,
        "true_validation_case_count": 0,
        "primitive_report_sha256": primitive_sha256,
        "primitive_schema": primitive_report["schema"],
        "primitive_method": primitive_report["method"],
        "primitive_capacity": primitive_report["capacity"],
        "primitive_query_projection_fit": primitive_report["query_projection_fit"],
        "primitive_key_projection_fit": primitive_report["key_projection_fit"],
        "primitive_analytical_solvers": primitive_report["analytical_solvers"],
        "primitive_solver_contract": primitive_report["solver_contract"],
        "primitive_solver_error_ladder": primitive_report["solver_error_ladder"],
        "primitive_factor_sufficient_state_count": primitive_report[
            "factor_sufficient_state_count"
        ],
        "primitive_factor_sufficient_state_bundle_sha256": primitive_report[
            "factor_sufficient_state_bundle_sha256"
        ],
        "fit_partition_contract": {
            "upstream_split": "fit",
            "fit_candidate_state_sources": "fit_partition_only",
            "fit_check_replays_frozen_candidate": True,
            "fit_check_used_for_frequency_selection": False,
            "fit_check_used_for_projection_fit": False,
            "fit_check_candidate_state_sha256": dense_state_sha256,
            "macrocell_selection_external": True,
            "true_validation_external": True,
        },
        "policy": policy_value.as_dict(),
        "limits": primitive_report["limits"],
        "ridge": float(ridge),
        "operation_audit": operation_audit,
    }
    report = {**body, "artifact_sha256": _canonical_sha256(body)}
    return Lfm2CompiledCachedQkFunctionalRefit(
        candidate_id=identifier,
        topology=topology,
        runtime=runtime,
        query_groups=query_groups_value,
        key_value_groups=key_groups_value,
        selected_source_rope_frequencies=selected,
        query_weight=compiled.query_weight,
        key_weight=compiled.key_weight,
        fit_metrics=compiled.fit_metrics,
        fit_check_metrics=compiled.fit_check_metrics,
        report=report,
    )


__all__ = [
    "CACHED_QK_FUNCTIONAL_REFIT_SCHEMA",
    "CachedQkEvidencePartition",
    "Lfm2CachedQkArtifactAuthenticationError",
    "Lfm2CachedQkEvidenceBank",
    "Lfm2CachedQkFunctionalRefitError",
    "Lfm2CachedQkFunctionalRefitPolicy",
    "Lfm2CompiledCachedQkFunctionalRefit",
    "adapt_lfm2_cached_attention_case_for_qk_refit",
    "compile_lfm2_cached_qk_functional_refit",
    "partition_lfm2_cached_qk_fit_cases",
    "seal_lfm2_cached_qk_fit_bank",
    "seal_lfm2_cached_qk_fit_check_bank",
]
